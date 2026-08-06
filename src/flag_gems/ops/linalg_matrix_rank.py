# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import warnings

import torch
import triton
import triton.language as tl

from flag_gems.ops.linalg_svdvals import linalg_svdvals
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _matrix_rank_from_values_kernel(
    values,
    atol,
    rtol,
    out,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offsets < K

    row_values = tl.load(values + row * K + offsets, mask=mask, other=0.0)
    max_value = tl.max(row_values, axis=0)
    absolute_tol = tl.load(atol + row)
    relative_tol = tl.load(rtol + row)
    threshold = tl.maximum(absolute_tol, relative_tol * max_value)

    rank = tl.sum((row_values > threshold).to(tl.int32), axis=0)
    tl.store(out + row, rank.to(tl.int64))


def _expand_tolerance(value, batch_shape, input, name):
    if isinstance(value, torch.Tensor):
        if value.is_complex():
            raise RuntimeError(
                f"torch.linalg.matrix_rank: {name} tensor of complex type is not "
                f"supported. Got {value.dtype}"
            )
        if value.device != input.device:
            raise RuntimeError(
                f"torch.linalg.matrix_rank: Expected {name} and input tensors to "
                f"be on the same device, but got {name} on {value.device} and "
                f"input on {input.device}"
            )
        try:
            value = value.expand(batch_shape)
        except RuntimeError as error:
            raise RuntimeError(
                f"torch.linalg.matrix_rank: {name} with shape {tuple(value.shape)} "
                f"is not broadcastable to batch shape {tuple(batch_shape)}"
            ) from error
        return value.to(dtype=torch.float32).contiguous()

    try:
        scalar = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"torch.linalg.matrix_rank: {name} must be a float or Tensor"
        ) from error
    return torch.full(batch_shape, scalar, dtype=torch.float32, device=input.device)


def _prepare_tolerances(input, atol, rtol):
    batch_shape = input.shape[:-2]
    atol_is_set = atol is not None
    atol_tensor = _expand_tolerance(
        0.0 if atol is None else atol, batch_shape, input, "atol"
    )

    if rtol is not None:
        rtol_tensor = _expand_tolerance(rtol, batch_shape, input, "rtol")
    else:
        default_rtol = max(input.shape[-2:]) * torch.finfo(input.dtype).eps
        if atol_is_set:
            rtol_tensor = torch.where(
                atol_tensor > 0,
                torch.zeros_like(atol_tensor),
                torch.full_like(atol_tensor, default_rtol),
            )
        else:
            rtol_tensor = torch.full_like(atol_tensor, default_rtol)

    return atol_tensor, rtol_tensor.contiguous()


def _check_input(input, hermitian):
    if input.ndim < 2:
        raise RuntimeError(
            "torch.linalg.matrix_rank: input must have at least 2 dimensions"
        )
    if input.dtype != torch.float32:
        raise NotImplementedError(
            "FlagGems linalg_matrix_rank currently supports float32 inputs only; "
            f"got {input.dtype}"
        )
    if hermitian and input.shape[-2] != input.shape[-1]:
        raise RuntimeError(
            "torch.linalg.matrix_rank: A must be batches of square matrices when "
            "hermitian=True"
        )


def linalg_matrix_rank(input, *, atol=None, rtol=None, hermitian=False):
    """Computes the numerical rank of a matrix or a batch of matrices.

    The initial FlagGems implementation supports float32 inputs. Matrix
    decomposition is delegated to linalg_svdvals or torch.linalg.eigvalsh,
    while a Triton kernel fuses the maximum, tolerance comparison, and rank
    reduction.
    """
    logger.debug("GEMS LINALG_MATRIX_RANK")
    _check_input(input, hermitian)

    output_shape = input.shape[:-2]
    if input.numel() == 0:
        return torch.zeros(output_shape, dtype=torch.int64, device=input.device)

    atol_tensor, rtol_tensor = _prepare_tolerances(input, atol, rtol)
    if hermitian:
        values = torch.linalg.eigvalsh(input).abs()
    else:
        values = linalg_svdvals(input)

    values = values.contiguous()
    K = values.shape[-1]
    batch_count = values.numel() // K
    out = torch.empty(output_shape, dtype=torch.int64, device=input.device)
    block_size = triton.next_power_of_2(K)
    num_warps = 4 if block_size <= 2048 else 8

    with torch_device_fn.device(input.device):
        _matrix_rank_from_values_kernel[(batch_count,)](
            values,
            atol_tensor,
            rtol_tensor,
            out,
            K,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    return out


def linalg_matrix_rank_tol(input, tol, hermitian=False):
    """NumPy-compatible legacy overload where tol is an absolute tolerance."""
    return linalg_matrix_rank(input, atol=tol, rtol=0.0, hermitian=hermitian)


def _copy_rank_to_out(input, result, out):
    if out is None:
        raise TypeError("torch.linalg.matrix_rank: out must be a Tensor")
    if out.device != input.device:
        raise RuntimeError(
            "torch.linalg.matrix_rank: Expected result and input tensors to be on "
            f"the same device, but got result on {out.device} and input on "
            f"{input.device}"
        )
    if not torch.can_cast(result.dtype, out.dtype):
        raise RuntimeError(
            "torch.linalg.matrix_rank: Expected result to be safely castable from "
            f"Long dtype, but got result with dtype {out.dtype}"
        )

    if out.numel() != 0 and out.shape != result.shape:
        warnings.warn(
            "An output with one or more elements was resized because it had shape "
            f"{tuple(out.shape)}, which does not match the required output shape "
            f"{tuple(result.shape)}.",
            UserWarning,
            stacklevel=3,
        )
    out.resize_(result.shape)
    out.copy_(result)
    return out


def linalg_matrix_rank_out(
    input, *, atol=None, rtol=None, hermitian=False, out=None
):
    result = linalg_matrix_rank(
        input, atol=atol, rtol=rtol, hermitian=hermitian
    )
    return _copy_rank_to_out(input, result, out)


def linalg_matrix_rank_tol_out(input, tol, hermitian=False, *, out=None):
    result = linalg_matrix_rank_tol(input, tol, hermitian)
    return _copy_rank_to_out(input, result, out)
