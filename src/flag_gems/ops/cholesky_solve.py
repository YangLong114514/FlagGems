# Copyright 2026, The FlagOS Contributors.
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

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils.triton_lang_extension import program_id

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def cholesky_solve_kernel(
    L_ptr, B_ptr, X_ptr, N, nrhs,
    batch_stride_L, batch_stride_B,
    stride_L, stride_B,
):
    """Cholesky solve kernel.

    Solves LL^T * X = B for X, given the lower-triangular Cholesky factor L
    and the right-hand side B. Each program computes one matrix in the batch.

    Algorithm:
      1. Forward substitution:  L * Y = B  (solve for Y, store in X)
      2. Backward substitution: L^T * X = Y (solve for X, in-place from Y)
    """
    pid = program_id(0)

    L_base = pid * batch_stride_L
    B_base = pid * batch_stride_B

    # Phase 1: Forward substitution: solve L * Y = B.
    for c in range(nrhs):
        for i in range(N):
            sum_val = tl.load(B_ptr + B_base + i * stride_B + c)
            for j in range(i):
                L_val = tl.load(L_ptr + L_base + i * stride_L + j)
                Y_val = tl.load(X_ptr + B_base + j * stride_B + c)
                sum_val = sum_val - L_val * Y_val
            diag = tl.load(L_ptr + L_base + i * stride_L + i)
            tl.store(X_ptr + B_base + i * stride_B + c, sum_val / diag)

    # Phase 2: Backward substitution: solve L^T * X = Y.
    for c in range(nrhs):
        for i in range(N - 1, -1, -1):
            sum_val = tl.load(X_ptr + B_base + i * stride_B + c)
            for j in range(i + 1, N):
                L_val = tl.load(L_ptr + L_base + j * stride_L + i)
                Xj_val = tl.load(X_ptr + B_base + j * stride_B + c)
                sum_val = sum_val - L_val * Xj_val
            diag = tl.load(L_ptr + L_base + i * stride_L + i)
            tl.store(X_ptr + B_base + i * stride_B + c, sum_val / diag)


def cholesky_solve(B, L, upper=False):
    """Solves a system of linear equations with a symmetric positive-definite
    matrix using the Cholesky factorization.

    Computes X such that A @ X = B, where A = L @ L^T (or A = U^T @ U if
    upper=True) and L (or U) is the Cholesky factor of A.

    Args:
        B: right-hand side tensor of shape (*, N, nrhs)
        L: Cholesky factor of shape (*, N, N), lower-triangular unless upper=True
        upper: if True, the Cholesky factor is upper-triangular

    Returns:
        X: solution tensor of shape (*, N, nrhs)
    """
    logger.debug("GEMS CHOLESKY_SOLVE")
    assert L.dtype in (
        torch.float32,
        torch.float64,
    ), "cholesky_solve only supports float32 and float64"
    assert B.dtype == L.dtype, "B and L must have the same dtype"
    if B.device != L.device:
        raise ValueError("B and L must be on the same device")

    if B.numel() == 0 or L.numel() == 0:
        return B

    L_shape = L.shape
    B_shape = B.shape

    if len(L_shape) < 2:
        raise ValueError("L must be at least 2D")
    if len(B_shape) < 2:
        raise ValueError("B must be at least 2D")

    N = L_shape[-1]
    if L_shape[-2] != N:
        raise ValueError("L must be a square matrix")
    if B_shape[-2] != N:
        raise ValueError(
            f"B's second-to-last dimension must equal L's last dimension, "
            f"got {B_shape[-2]} != {N}"
        )

    nrhs = B_shape[-1]

    try:
        batch_shape = torch.broadcast_shapes(B_shape[:-2], L_shape[:-2])
    except RuntimeError as exc:
        raise ValueError(
            f"B and L batch dimensions are not broadcastable: "
            f"{B_shape[:-2]} vs {L_shape[:-2]}"
        ) from exc

    L = L.expand(batch_shape + L_shape[-2:])
    B = B.expand(batch_shape + B_shape[-2:])

    if upper:
        L = L.transpose(-2, -1).contiguous()
    else:
        L = L.contiguous()

    B = B.contiguous()
    X = torch.empty_like(B)

    batch_size = 1
    for dim in batch_shape:
        batch_size *= dim

    L_kernel = L.reshape(-1, N, N)
    B_kernel = B.reshape(-1, N, nrhs)
    X_kernel = X.reshape(-1, N, nrhs)

    stride_L = L_kernel.stride(1)
    stride_B = B_kernel.stride(1)
    batch_stride_L = L_kernel.stride(0)
    batch_stride_B = B_kernel.stride(0)

    grid = (batch_size,)

    with torch.no_grad():
        cholesky_solve_kernel[grid](
            L_kernel,
            B_kernel,
            X_kernel,
            N,
            nrhs,
            batch_stride_L,
            batch_stride_B,
            stride_L,
            stride_B,
        )

    return X
