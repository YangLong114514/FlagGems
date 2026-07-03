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

    # Phase 1: Forward substitution — solve L * Y = B
    # L is lower-triangular, L[i,i] are the diagonal entries.
    # Y[i,c] = (B[i,c] - sum_{j=0}^{i-1} L[i,j] * Y[j,c]) / L[i,i]
    for c in range(nrhs):
        for i in range(N):
            sum_val = tl.load(B_ptr + B_base + i * stride_B + c)
            for j in range(i):
                L_val = tl.load(L_ptr + L_base + i * stride_L + j)
                Y_val = tl.load(X_ptr + B_base + j * stride_B + c)
                sum_val = sum_val - L_val * Y_val
            diag = tl.load(L_ptr + L_base + i * stride_L + i)
            tl.store(X_ptr + B_base + i * stride_B + c, sum_val / diag)

    # Phase 2: Backward substitution — solve L^T * X = Y
    # L^T is upper-triangular. We go from bottom to top.
    # X[i,c] = (Y[i,c] - sum_{j=i+1}^{N-1} L[j,i] * X[j,c]) / L[i,i]
    for c in range(nrhs):
        for i in range(N - 1, -1, -1):
            sum_val = tl.load(X_ptr + B_base + i * stride_B + c)
            for j in range(i + 1, N):
                # L[j,i] is the element at row j, column i of L
                # which corresponds to L^T[i,j]
                L_val = tl.load(L_ptr + L_base + j * stride_L + i)
                Xj_val = tl.load(X_ptr + B_base + j * stride_B + c)
                sum_val = sum_val - L_val * Xj_val
            diag = tl.load(L_ptr + L_base + i * stride_L + i)
            tl.store(X_ptr + B_base + i * stride_B + c, sum_val / diag)


def linalg_cholesky_solve(B, L, upper=False):
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
    logger.debug("GEMS LINALG_CHOLESKY_SOLVE")
    assert L.dtype in (
        torch.float32,
        torch.float64,
    ), "linalg_cholesky_solve only supports float32 and float64"

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

    # If upper=True, use L^T as the lower-triangular factor
    if upper:
        L = L.transpose(-2, -1).contiguous()
    else:
        L = L.contiguous()

    B = B.contiguous()
    X = torch.empty_like(B)

    # Flatten batch dimensions
    if len(L_shape) == 2:
        batch_size = 1
    else:
        batch_dims = L_shape[:-2]
        batch_size = 1
        for dim in batch_dims:
            batch_size *= dim

    # Reshape to (batch, N, N) and (batch, N, nrhs)
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
