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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_SMALL_JACOBI_MAX_K = 16
_BLOCKED_JACOBI_MAX_K = 512
_JACOBI_MAX_ROWS = 1024


@libentry()
@triton.jit
def _matrix_rank_zero_kernel(out, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(out + offsets, 0, mask=offsets < N)


@libentry()
@triton.jit
def _matrix_rank_rank1_kernel(
    A,
    ATOL,
    RTOL,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    TALL: tl.constexpr,
    HERMITIAN: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    batch = tl.program_id(0)
    rows = tl.arange(0, BLOCK_R)
    row_mask = rows < ROWS
    a_base = A + batch * M * N

    if HERMITIAN:
        values = tl.load(a_base + rows * N, mask=row_mask, other=0.0)
    elif TALL:
        values = tl.load(a_base + rows * N, mask=row_mask, other=0.0)
    else:
        values = tl.load(a_base + rows, mask=row_mask, other=0.0)

    singular_value = tl.sqrt(tl.sum(values * values, axis=0))
    atol = tl.load(ATOL + batch)
    rtol = tl.load(RTOL + batch)
    threshold = tl.maximum(atol, rtol * singular_value)
    rank = (singular_value > threshold).to(tl.int64)
    tl.store(OUT + batch, rank)


@libentry()
@triton.jit
def _matrix_rank_rank2_kernel(
    A,
    ATOL,
    RTOL,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    TALL: tl.constexpr,
    HERMITIAN: tl.constexpr,
    BLOCK_R: tl.constexpr,
    REL_EPS: tl.constexpr,
    ABS_EPS: tl.constexpr,
):
    batch = tl.program_id(0)
    rows = tl.arange(0, BLOCK_R)
    row_mask = rows < ROWS
    a_base = A + batch * M * N

    if HERMITIAN:
        x = tl.load(a_base + rows * N, mask=row_mask, other=0.0)
        lower_rows = tl.maximum(rows, 1)
        lower_columns = tl.minimum(rows, 1)
        y = tl.load(
            a_base + lower_rows * N + lower_columns,
            mask=row_mask,
            other=0.0,
        )
    elif TALL:
        x = tl.load(a_base + rows * N, mask=row_mask, other=0.0)
        y = tl.load(a_base + rows * N + 1, mask=row_mask, other=0.0)
    else:
        x = tl.load(a_base + rows, mask=row_mask, other=0.0)
        y = tl.load(a_base + N + rows, mask=row_mask, other=0.0)

    alpha = tl.sum(x * x, axis=0)
    beta = tl.sum(y * y, axis=0)
    gamma = tl.sum(x * y, axis=0)
    active = tl.abs(gamma) > REL_EPS * tl.sqrt(alpha * beta + ABS_EPS)
    safe_gamma = tl.where(active, gamma, 1.0)
    tau = (beta - alpha) / (2.0 * safe_gamma)
    sign_tau = tl.where(tau >= 0.0, 1.0, -1.0)
    t = sign_tau / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
    c = 1.0 / tl.sqrt(1.0 + t * t)
    s = t * c
    c = tl.where(active, c, 1.0)
    s = tl.where(active, s, 0.0)

    rotated_x = c * x - s * y
    rotated_y = s * x + c * y
    singular_x = tl.sqrt(tl.sum(rotated_x * rotated_x, axis=0))
    singular_y = tl.sqrt(tl.sum(rotated_y * rotated_y, axis=0))
    max_value = tl.maximum(singular_x, singular_y)

    atol = tl.load(ATOL + batch)
    rtol = tl.load(RTOL + batch)
    threshold = tl.maximum(atol, rtol * max_value)
    rank = (singular_x > threshold).to(tl.int32)
    rank += (singular_y > threshold).to(tl.int32)
    tl.store(OUT + batch, rank.to(tl.int64))


@libentry()
@triton.jit
def _matrix_rank_small_jacobi_kernel(
    A,
    A_WORK,
    ATOL,
    RTOL,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    TALL: tl.constexpr,
    HERMITIAN: tl.constexpr,
    IS_FP64: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SWEEPS: tl.constexpr,
    REL_EPS: tl.constexpr,
    ABS_EPS: tl.constexpr,
):
    batch = tl.program_id(0)
    rows = tl.arange(0, BLOCK_R)
    singular_indices = tl.arange(0, BLOCK_K)
    row_mask = rows < ROWS
    a_base = A + batch * M * N
    work_base = A_WORK + batch * K * ROWS

    for column in tl.static_range(0, K):
        if HERMITIAN:
            source_rows = tl.maximum(rows, column)
            source_columns = tl.minimum(rows, column)
            values = tl.load(
                a_base + source_rows * N + source_columns,
                mask=row_mask,
                other=0.0,
            )
        elif TALL:
            values = tl.load(
                a_base + rows * N + column,
                mask=row_mask,
                other=0.0,
            )
        else:
            values = tl.load(
                a_base + column * N + rows,
                mask=row_mask,
                other=0.0,
            )
        tl.store(work_base + column * ROWS + rows, values, mask=row_mask)

    for _ in tl.static_range(0, SWEEPS):
        for p in tl.static_range(0, K):
            for q in tl.static_range(p + 1, K):
                ap = tl.load(
                    work_base + p * ROWS + rows,
                    mask=row_mask,
                    other=0.0,
                )
                aq = tl.load(
                    work_base + q * ROWS + rows,
                    mask=row_mask,
                    other=0.0,
                )
                alpha = tl.sum(ap * ap, axis=0)
                beta = tl.sum(aq * aq, axis=0)
                gamma = tl.sum(ap * aq, axis=0)
                active = (
                    tl.abs(gamma)
                    > REL_EPS * tl.sqrt(alpha * beta + ABS_EPS)
                )
                safe_gamma = tl.where(active, gamma, 1.0)
                tau = (beta - alpha) / (2.0 * safe_gamma)
                sign_tau = tl.where(tau >= 0.0, 1.0, -1.0)
                t = sign_tau / (
                    tl.abs(tau) + tl.sqrt(1.0 + tau * tau)
                )
                c = 1.0 / tl.sqrt(1.0 + t * t)
                s = t * c
                c = tl.where(active, c, 1.0)
                s = tl.where(active, s, 0.0)

                new_ap = c * ap - s * aq
                new_aq = s * ap + c * aq
                tl.store(
                    work_base + p * ROWS + rows,
                    new_ap,
                    mask=row_mask,
                )
                tl.store(
                    work_base + q * ROWS + rows,
                    new_aq,
                    mask=row_mask,
                )

    accumulator_dtype = tl.float64 if IS_FP64 else tl.float32
    singular_values = tl.full(
        (BLOCK_K,), 0.0, dtype=accumulator_dtype
    )
    for column in tl.static_range(0, K):
        values = tl.load(
            work_base + column * ROWS + rows,
            mask=row_mask,
            other=0.0,
        )
        norm = tl.sqrt(tl.sum(values * values, axis=0))
        singular_values = tl.where(
            singular_indices == column,
            norm,
            singular_values,
        )

    max_value = tl.max(singular_values, axis=0)
    atol = tl.load(ATOL + batch)
    rtol = tl.load(RTOL + batch)
    threshold = tl.maximum(atol, rtol * max_value)
    rank = tl.sum(
        (
            (singular_values > threshold)
            & (singular_indices < K)
        ).to(tl.int32),
        axis=0,
    )
    tl.store(OUT + batch, rank.to(tl.int64))


@libentry()
@triton.jit
def _matrix_rank_blocked_init_kernel(
    A,
    A_WORK,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    TALL: tl.constexpr,
    HERMITIAN: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    batch = tl.program_id(0)
    column = tl.program_id(1)
    rows = tl.arange(0, BLOCK_R)
    row_mask = rows < ROWS
    a_base = A + batch * M * N
    work_base = A_WORK + batch * K * ROWS

    if HERMITIAN:
        source_rows = tl.maximum(rows, column)
        source_columns = tl.minimum(rows, column)
        values = tl.load(
            a_base + source_rows * N + source_columns,
            mask=row_mask,
            other=0.0,
        )
    elif TALL:
        values = tl.load(
            a_base + rows * N + column,
            mask=row_mask,
            other=0.0,
        )
    else:
        values = tl.load(
            a_base + column * N + rows,
            mask=row_mask,
            other=0.0,
        )
    tl.store(work_base + column * ROWS + rows, values, mask=row_mask)


@libentry()
@triton.jit
def _matrix_rank_blocked_pair_kernel(
    A_WORK,
    STEP,
    K: tl.constexpr,
    ROUND: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    REL_EPS: tl.constexpr,
    ABS_EPS: tl.constexpr,
):
    batch = tl.program_id(0)
    pair = tl.program_id(1)
    rows = tl.arange(0, BLOCK_R)
    ring = ROUND - 1

    position_p = pair
    position_q = ROUND - 1 - pair
    p = tl.where(
        position_p == 0,
        0,
        ((position_p + ring - STEP - 1) % ring) + 1,
    )
    q = tl.where(
        position_q == 0,
        0,
        ((position_q + ring - STEP - 1) % ring) + 1,
    )
    valid_pair = (p < K) & (q < K)
    swap = p > q
    ordered_p = tl.where(swap, q, p)
    ordered_q = tl.where(swap, p, q)
    row_mask = (rows < ROWS) & valid_pair
    work_base = A_WORK + batch * K * ROWS

    ap = tl.load(
        work_base + ordered_p * ROWS + rows,
        mask=row_mask,
        other=0.0,
    )
    aq = tl.load(
        work_base + ordered_q * ROWS + rows,
        mask=row_mask,
        other=0.0,
    )
    alpha = tl.sum(ap * ap, axis=0)
    beta = tl.sum(aq * aq, axis=0)
    gamma = tl.sum(ap * aq, axis=0)
    active = (
        tl.abs(gamma)
        > REL_EPS * tl.sqrt(alpha * beta + ABS_EPS)
    )
    safe_gamma = tl.where(active, gamma, 1.0)
    tau = (beta - alpha) / (2.0 * safe_gamma)
    sign_tau = tl.where(tau >= 0.0, 1.0, -1.0)
    t = sign_tau / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
    c = 1.0 / tl.sqrt(1.0 + t * t)
    s = t * c
    c = tl.where(active & valid_pair, c, 1.0)
    s = tl.where(active & valid_pair, s, 0.0)

    new_ap = c * ap - s * aq
    new_aq = s * ap + c * aq
    tl.store(
        work_base + ordered_p * ROWS + rows,
        new_ap,
        mask=row_mask,
    )
    tl.store(
        work_base + ordered_q * ROWS + rows,
        new_aq,
        mask=row_mask,
    )


@libentry()
@triton.jit
def _matrix_rank_blocked_norm_kernel(
    A_WORK,
    S_WORK,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    batch = tl.program_id(0)
    column = tl.program_id(1)
    rows = tl.arange(0, BLOCK_R)
    row_mask = rows < ROWS
    work_base = A_WORK + batch * K * ROWS
    values = tl.load(
        work_base + column * ROWS + rows,
        mask=row_mask,
        other=0.0,
    )
    norm = tl.sqrt(tl.sum(values * values, axis=0))
    tl.store(S_WORK + batch * K + column, norm)


@libentry()
@triton.jit
def _matrix_rank_blocked_finalize_kernel(
    S_WORK,
    ATOL,
    RTOL,
    OUT,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch = tl.program_id(0)
    columns = tl.arange(0, BLOCK_K)
    column_mask = columns < K
    singular_values = tl.load(
        S_WORK + batch * K + columns,
        mask=column_mask,
        other=0.0,
    )
    max_value = tl.max(singular_values, axis=0)
    atol = tl.load(ATOL + batch)
    rtol = tl.load(RTOL + batch)
    threshold = tl.maximum(atol, rtol * max_value)
    rank = tl.sum(
        ((singular_values > threshold) & column_mask).to(tl.int32),
        axis=0,
    )
    tl.store(OUT + batch, rank.to(tl.int64))


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
        return value.to(dtype=input.dtype).contiguous()

    try:
        scalar = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"torch.linalg.matrix_rank: {name} must be a float or Tensor"
        ) from error
    return torch.full(batch_shape, scalar, dtype=input.dtype, device=input.device)


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
    if input.dtype not in (torch.float32, torch.float64):
        raise NotImplementedError(
            "FlagGems linalg_matrix_rank currently supports float32 and float64 "
            f"real inputs only; got {input.dtype}"
        )
    if hermitian and input.shape[-2] != input.shape[-1]:
        raise RuntimeError(
            "torch.linalg.matrix_rank: A must be batches of square matrices when "
            "hermitian=True"
        )


def _empty_matrix_rank(input, output_shape):
    out = torch.empty(output_shape, dtype=torch.int64, device=input.device)
    output_size = out.numel()
    if output_size:
        block_size = min(256, triton.next_power_of_2(output_size))
        with torch_device_fn.device(input.device):
            _matrix_rank_zero_kernel[(triton.cdiv(output_size, block_size),)](
                out,
                N=output_size,
                BLOCK_SIZE=block_size,
            )
    return out


def _launch_matrix_rank(
    input,
    atol_tensor,
    rtol_tensor,
    hermitian,
):
    output_shape = input.shape[:-2]
    m, n = input.shape[-2:]
    k = min(m, n)
    rows = max(m, n)
    if k > _BLOCKED_JACOBI_MAX_K or rows > _JACOBI_MAX_ROWS:
        raise NotImplementedError(
            "FlagGems linalg_matrix_rank Triton Jacobi path currently supports "
            f"min(m, n) <= {_BLOCKED_JACOBI_MAX_K} and max(m, n) <= "
            f"{_JACOBI_MAX_ROWS}; got ({m}, {n})"
        )

    batch_count = input.numel() // (m * n)
    matrix = input.contiguous().reshape(batch_count, m, n)
    out = torch.empty(output_shape, dtype=torch.int64, device=input.device)
    block_r = triton.next_power_of_2(rows)
    is_fp64 = input.dtype == torch.float64
    relative_epsilon = 1.0e-15 if is_fp64 else 1.0e-7
    absolute_epsilon = 1.0e-300 if is_fp64 else 1.0e-30
    num_warps = 1 if block_r <= 64 else 4

    with torch_device_fn.device(input.device):
        if k == 1:
            _matrix_rank_rank1_kernel[(batch_count,)](
                matrix,
                atol_tensor,
                rtol_tensor,
                out,
                M=m,
                N=n,
                ROWS=rows,
                TALL=m >= n,
                HERMITIAN=hermitian,
                BLOCK_R=block_r,
                num_warps=num_warps,
            )
        elif k == 2:
            _matrix_rank_rank2_kernel[(batch_count,)](
                matrix,
                atol_tensor,
                rtol_tensor,
                out,
                M=m,
                N=n,
                ROWS=rows,
                TALL=m >= n,
                HERMITIAN=hermitian,
                BLOCK_R=block_r,
                REL_EPS=relative_epsilon,
                ABS_EPS=absolute_epsilon,
                num_warps=num_warps,
            )
        elif k <= _SMALL_JACOBI_MAX_K:
            work = torch.empty(
                (batch_count, k, rows),
                dtype=input.dtype,
                device=input.device,
            )
            block_k = triton.next_power_of_2(k)
            # Numerical rank is more sensitive to residual column correlation
            # than returning approximate singular values. Keep a few more
            # sweeps than the general SVD path, especially for float64.
            sweeps = 12 if is_fp64 else 8
            _matrix_rank_small_jacobi_kernel[(batch_count,)](
                matrix,
                work,
                atol_tensor,
                rtol_tensor,
                out,
                M=m,
                N=n,
                K=k,
                ROWS=rows,
                TALL=m >= n,
                HERMITIAN=hermitian,
                IS_FP64=is_fp64,
                BLOCK_R=block_r,
                BLOCK_K=block_k,
                SWEEPS=sweeps,
                REL_EPS=relative_epsilon,
                ABS_EPS=absolute_epsilon,
                num_warps=num_warps,
            )
        else:
            work = torch.empty(
                (batch_count, k, rows),
                dtype=input.dtype,
                device=input.device,
            )
            singular_values = torch.empty(
                (batch_count, k),
                dtype=input.dtype,
                device=input.device,
            )
            _matrix_rank_blocked_init_kernel[(batch_count, k)](
                matrix,
                work,
                M=m,
                N=n,
                K=k,
                ROWS=rows,
                TALL=m >= n,
                HERMITIAN=hermitian,
                BLOCK_R=block_r,
                num_warps=num_warps,
            )

            round_size = k if k % 2 == 0 else k + 1
            half_round = round_size // 2
            if is_fp64:
                sweeps = 24 if k > 256 else 18
            else:
                sweeps = 18 if k > 256 else 14
            for _ in range(sweeps):
                for step in range(round_size - 1):
                    _matrix_rank_blocked_pair_kernel[
                        (batch_count, half_round)
                    ](
                        work,
                        step,
                        K=k,
                        ROUND=round_size,
                        ROWS=rows,
                        BLOCK_R=block_r,
                        REL_EPS=relative_epsilon,
                        ABS_EPS=absolute_epsilon,
                        num_warps=num_warps,
                    )

            _matrix_rank_blocked_norm_kernel[(batch_count, k)](
                work,
                singular_values,
                K=k,
                ROWS=rows,
                BLOCK_R=block_r,
                num_warps=num_warps,
            )
            block_k = triton.next_power_of_2(k)
            _matrix_rank_blocked_finalize_kernel[(batch_count,)](
                singular_values,
                atol_tensor,
                rtol_tensor,
                out,
                K=k,
                BLOCK_K=block_k,
                num_warps=4 if block_k >= 128 else 1,
            )
    return out


def linalg_matrix_rank(input, *, atol=None, rtol=None, hermitian=False):
    """Computes numerical matrix rank with shape-specialized Triton Jacobi."""
    logger.debug("GEMS LINALG_MATRIX_RANK")
    _check_input(input, hermitian)

    output_shape = input.shape[:-2]
    if input.numel() == 0:
        return _empty_matrix_rank(input, output_shape)

    atol_tensor, rtol_tensor = _prepare_tolerances(input, atol, rtol)
    return _launch_matrix_rank(
        input,
        atol_tensor,
        rtol_tensor,
        hermitian,
    )


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
