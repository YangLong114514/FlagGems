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


CHOLESKY_SOLVE_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_RHS": block_rhs}, num_warps=1, num_stages=1)
    for block_rhs in (1, 2, 4, 8, 16, 32)
]


@libentry()
@triton.autotune(
    configs=CHOLESKY_SOLVE_AUTOTUNE_CONFIGS, key=["N", "nrhs", "dtype_flag", "upper"]
)
@triton.jit
def cholesky_solve_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    N: tl.constexpr,
    nrhs: tl.constexpr,
    batch_stride_L,
    batch_stride_B,
    stride_L,
    stride_B,
    BLOCK_RHS: tl.constexpr,
    dtype_flag: tl.constexpr,
    upper: tl.constexpr,
):
    """Cholesky solve kernel.

    Solves LL^T * X = B or U^T U * X = B for X, given the lower- or
    upper-triangular Cholesky factor and the right-hand side B. Each program
    computes one RHS tile for one matrix in the batch.

    Algorithm:
      lower=False path: L * Y = B, then L^T * X = Y.
      upper=True path: U^T * Y = B, then U * X = Y.
    """
    batch_pid = program_id(0)
    rhs_tile_pid = program_id(1)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    cols = rhs_tile_pid * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    cols_mask = cols < nrhs

    # Phase 1: Forward substitution: solve L * Y = B.
    for i in range(N):
        sum_val = tl.load(B_ptr + B_base + i * stride_B + cols, mask=cols_mask)
        for j in range(i):
            if upper:
                L_val = tl.load(L_ptr + L_base + j * stride_L + i)
            else:
                L_val = tl.load(L_ptr + L_base + i * stride_L + j)
            Y_val = tl.load(
                X_ptr + B_base + j * stride_B + cols, mask=cols_mask
            )
            sum_val = sum_val - L_val * Y_val
        diag = tl.load(L_ptr + L_base + i * stride_L + i)
        tl.store(
            X_ptr + B_base + i * stride_B + cols, sum_val / diag, mask=cols_mask
        )

    # Phase 2: Backward substitution: solve L^T * X = Y.
    for i in range(N - 1, -1, -1):
        sum_val = tl.load(X_ptr + B_base + i * stride_B + cols, mask=cols_mask)
        for j in range(i + 1, N):
            if upper:
                L_val = tl.load(L_ptr + L_base + i * stride_L + j)
            else:
                L_val = tl.load(L_ptr + L_base + j * stride_L + i)
            Xj_val = tl.load(
                X_ptr + B_base + j * stride_B + cols, mask=cols_mask
            )
            sum_val = sum_val - L_val * Xj_val
        diag = tl.load(L_ptr + L_base + i * stride_L + i)
        tl.store(
            X_ptr + B_base + i * stride_B + cols, sum_val / diag, mask=cols_mask
        )


@libentry()
@triton.jit
def cholesky_solve_blocked_lower_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    N: tl.constexpr,
    nrhs: tl.constexpr,
    batch_stride_L,
    batch_stride_B,
    stride_L,
    stride_B,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
):
    """Blocked lower-factor Cholesky solve prototype.

    This path solves L L^T X = B for one batch and one RHS tile. X is used as
    workspace: forward TRSM writes Y into X, then backward TRSM overwrites it
    with X. Trailing block updates use tl.dot.
    """
    batch_pid = program_id(0)
    rhs_tile_pid = program_id(1)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    rhs_cols = rhs_tile_pid * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    rhs_mask = rhs_cols < nrhs
    k_offsets = tl.arange(0, BLOCK_K)
    m_offsets = tl.arange(0, BLOCK_M)

    # Forward blocked TRSM: L * Y = B.
    for k in range(0, N, BLOCK_K):
        rows_k = k + k_offsets
        if k == 0:
            y_block = tl.load(
                B_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
                mask=(rows_k[:, None] < N) & rhs_mask[None, :],
                other=0.0,
            )
        else:
            y_block = tl.load(
                X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
                mask=(rows_k[:, None] < N) & rhs_mask[None, :],
                other=0.0,
            )

        for i in range(BLOCK_K):
            row_i = k + i
            if row_i < N:
                L_vals = tl.load(
                    L_ptr + L_base + row_i * stride_L + rows_k,
                    mask=rows_k < row_i,
                    other=0.0,
                )
                dot = tl.sum(L_vals[:, None] * y_block, axis=0)
                rhs_vals = tl.sum(
                    tl.where(rows_k[:, None] == row_i, y_block, 0.0), axis=0
                )
                diag = tl.load(L_ptr + L_base + row_i * stride_L + row_i)
                y_vals = (rhs_vals - dot) / diag
                y_block = tl.where(rows_k[:, None] == row_i, y_vals[None, :], y_block)

        tl.store(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            y_block,
            mask=(rows_k[:, None] < N) & rhs_mask[None, :],
        )

        for m in range(k + BLOCK_K, N, BLOCK_M):
            rows_m = m + m_offsets
            L_tile = tl.load(
                L_ptr + L_base + rows_m[:, None] * stride_L + rows_k[None, :],
                mask=(rows_m[:, None] < N) & (rows_k[None, :] < N),
                other=0.0,
            )
            if k == 0:
                tail = tl.load(
                    B_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                    mask=(rows_m[:, None] < N) & rhs_mask[None, :],
                    other=0.0,
                )
            else:
                tail = tl.load(
                    X_ptr
                    + B_base
                    + rows_m[:, None] * stride_B
                    + rhs_cols[None, :],
                    mask=(rows_m[:, None] < N) & rhs_mask[None, :],
                    other=0.0,
                )
            tail = tail - tl.dot(L_tile, y_block)
            tl.store(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                tail,
                mask=(rows_m[:, None] < N) & rhs_mask[None, :],
            )

    # Backward blocked TRSM: L^T * X = Y.
    for k in range(N - BLOCK_K, -1, -BLOCK_K):
        rows_k = k + k_offsets
        x_block = tl.load(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            mask=(rows_k[:, None] < N) & rhs_mask[None, :],
            other=0.0,
        )

        for ii in range(BLOCK_K - 1, -1, -1):
            row_i = k + ii
            if row_i < N:
                L_vals = tl.load(
                    L_ptr + L_base + rows_k * stride_L + row_i,
                    mask=(rows_k > row_i) & (rows_k < N),
                    other=0.0,
                )
                dot = tl.sum(L_vals[:, None] * x_block, axis=0)
                y_vals = tl.sum(
                    tl.where(rows_k[:, None] == row_i, x_block, 0.0), axis=0
                )
                diag = tl.load(L_ptr + L_base + row_i * stride_L + row_i)
                x_vals = (y_vals - dot) / diag
                x_block = tl.where(rows_k[:, None] == row_i, x_vals[None, :], x_block)

        tl.store(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            x_block,
            mask=(rows_k[:, None] < N) & rhs_mask[None, :],
        )

        for m in range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            L_tile = tl.load(
                L_ptr + L_base + rows_k[None, :] * stride_L + rows_m[:, None],
                mask=(rows_m[:, None] < N) & (rows_k[None, :] < N),
                other=0.0,
            )
            head = tl.load(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                mask=(rows_m[:, None] < N) & rhs_mask[None, :],
                other=0.0,
            )
            head = head - tl.dot(L_tile, x_block)
            tl.store(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                head,
                mask=(rows_m[:, None] < N) & rhs_mask[None, :],
            )


@libentry()
@triton.jit
def cholesky_solve_blocked_upper_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    N: tl.constexpr,
    nrhs: tl.constexpr,
    batch_stride_L,
    batch_stride_B,
    stride_L,
    stride_B,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
):
    """Blocked upper-factor Cholesky solve prototype.

    This path solves U^T U X = B for one batch and one RHS tile. It mirrors the
    lower blocked path while keeping the upper storage layout intact.
    """
    batch_pid = program_id(0)
    rhs_tile_pid = program_id(1)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    rhs_cols = rhs_tile_pid * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    rhs_mask = rhs_cols < nrhs
    k_offsets = tl.arange(0, BLOCK_K)
    m_offsets = tl.arange(0, BLOCK_M)

    # Forward blocked TRSM: U^T * Y = B.
    for k in range(0, N, BLOCK_K):
        rows_k = k + k_offsets
        if k == 0:
            y_block = tl.load(
                B_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
                mask=(rows_k[:, None] < N) & rhs_mask[None, :],
                other=0.0,
            )
        else:
            y_block = tl.load(
                X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
                mask=(rows_k[:, None] < N) & rhs_mask[None, :],
                other=0.0,
            )

        for i in range(BLOCK_K):
            row_i = k + i
            if row_i < N:
                U_vals = tl.load(
                    L_ptr + L_base + rows_k * stride_L + row_i,
                    mask=rows_k < row_i,
                    other=0.0,
                )
                dot = tl.sum(U_vals[:, None] * y_block, axis=0)
                rhs_vals = tl.sum(
                    tl.where(rows_k[:, None] == row_i, y_block, 0.0), axis=0
                )
                diag = tl.load(L_ptr + L_base + row_i * stride_L + row_i)
                y_vals = (rhs_vals - dot) / diag
                y_block = tl.where(rows_k[:, None] == row_i, y_vals[None, :], y_block)

        tl.store(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            y_block,
            mask=(rows_k[:, None] < N) & rhs_mask[None, :],
        )

        for m in range(k + BLOCK_K, N, BLOCK_M):
            rows_m = m + m_offsets
            U_tile_km = tl.load(
                L_ptr + L_base + rows_k[:, None] * stride_L + rows_m[None, :],
                mask=(rows_k[:, None] < N) & (rows_m[None, :] < N),
                other=0.0,
            )
            U_tile = tl.trans(U_tile_km)
            if k == 0:
                tail = tl.load(
                    B_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                    mask=(rows_m[:, None] < N) & rhs_mask[None, :],
                    other=0.0,
                )
            else:
                tail = tl.load(
                    X_ptr
                    + B_base
                    + rows_m[:, None] * stride_B
                    + rhs_cols[None, :],
                    mask=(rows_m[:, None] < N) & rhs_mask[None, :],
                    other=0.0,
                )
            tail = tail - tl.dot(U_tile, y_block)
            tl.store(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                tail,
                mask=(rows_m[:, None] < N) & rhs_mask[None, :],
            )

    # Backward blocked TRSM: U * X = Y.
    for k in range(N - BLOCK_K, -1, -BLOCK_K):
        rows_k = k + k_offsets
        x_block = tl.load(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            mask=(rows_k[:, None] < N) & rhs_mask[None, :],
            other=0.0,
        )

        for ii in range(BLOCK_K - 1, -1, -1):
            row_i = k + ii
            if row_i < N:
                U_vals = tl.load(
                    L_ptr + L_base + row_i * stride_L + rows_k,
                    mask=(rows_k > row_i) & (rows_k < N),
                    other=0.0,
                )
                dot = tl.sum(U_vals[:, None] * x_block, axis=0)
                y_vals = tl.sum(
                    tl.where(rows_k[:, None] == row_i, x_block, 0.0), axis=0
                )
                diag = tl.load(L_ptr + L_base + row_i * stride_L + row_i)
                x_vals = (y_vals - dot) / diag
                x_block = tl.where(rows_k[:, None] == row_i, x_vals[None, :], x_block)

        tl.store(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            x_block,
            mask=(rows_k[:, None] < N) & rhs_mask[None, :],
        )

        for m in range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            U_tile = tl.load(
                L_ptr + L_base + rows_m[:, None] * stride_L + rows_k[None, :],
                mask=(rows_m[:, None] < N) & (rows_k[None, :] < N),
                other=0.0,
            )
            head = tl.load(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                mask=(rows_m[:, None] < N) & rhs_mask[None, :],
                other=0.0,
            )
            head = head - tl.dot(U_tile, x_block)
            tl.store(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                head,
                mask=(rows_m[:, None] < N) & rhs_mask[None, :],
            )


def _can_use_blocked_lower_path(B, upper, N, nrhs):
    return (
        not upper
        and B.dtype == torch.float32
        and N >= 128
        and N % 32 == 0
        and nrhs >= 16
    )


def _can_use_blocked_upper_path(B, upper, N, nrhs):
    return (
        upper
        and B.dtype == torch.float32
        and N >= 128
        and N % 32 == 0
        and nrhs >= 16
    )



def _can_use_blocked_single_rhs_path(B, N, nrhs):
    return B.dtype == torch.float32 and nrhs == 1 and N >= 128 and N % 32 == 0


@libentry()
@triton.jit
def cholesky_solve_single_rhs_blocked_lower_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    N: tl.constexpr,
    batch_stride_L,
    batch_stride_B,
    stride_L,
    stride_B,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Blocked lower-factor single-RHS Cholesky solve."""
    batch_pid = program_id(0)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    k_offsets = tl.arange(0, BLOCK_K)
    m_offsets = tl.arange(0, BLOCK_M)

    # Forward blocked TRSV: L * Y = B.
    for k in range(0, N, BLOCK_K):
        rows_k = k + k_offsets
        if k == 0:
            y_block = tl.load(
                B_ptr + B_base + rows_k * stride_B,
                mask=rows_k < N,
                other=0.0,
            )
        else:
            y_block = tl.load(
                X_ptr + B_base + rows_k * stride_B,
                mask=rows_k < N,
                other=0.0,
            )

        for i in range(BLOCK_K):
            row_i = k + i
            if row_i < N:
                L_vals = tl.load(
                    L_ptr + L_base + row_i * stride_L + rows_k,
                    mask=rows_k < row_i,
                    other=0.0,
                )
                dot = tl.sum(L_vals * y_block, axis=0)
                rhs_val = tl.sum(tl.where(rows_k == row_i, y_block, 0.0), axis=0)
                diag = tl.load(L_ptr + L_base + row_i * stride_L + row_i)
                y_val = (rhs_val - dot) / diag
                y_block = tl.where(rows_k == row_i, y_val, y_block)

        tl.store(X_ptr + B_base + rows_k * stride_B, y_block, mask=rows_k < N)

        for m in range(k + BLOCK_K, N, BLOCK_M):
            rows_m = m + m_offsets
            L_tile = tl.load(
                L_ptr + L_base + rows_m[:, None] * stride_L + rows_k[None, :],
                mask=(rows_m[:, None] < N) & (rows_k[None, :] < N),
                other=0.0,
            )
            if k == 0:
                tail = tl.load(
                    B_ptr + B_base + rows_m * stride_B,
                    mask=rows_m < N,
                    other=0.0,
                )
            else:
                tail = tl.load(
                    X_ptr + B_base + rows_m * stride_B,
                    mask=rows_m < N,
                    other=0.0,
                )
            tail = tail - tl.sum(L_tile * y_block[None, :], axis=1)
            tl.store(X_ptr + B_base + rows_m * stride_B, tail, mask=rows_m < N)

    # Backward blocked TRSV: L^T * X = Y.
    for k in range(N - BLOCK_K, -1, -BLOCK_K):
        rows_k = k + k_offsets
        x_block = tl.load(
            X_ptr + B_base + rows_k * stride_B,
            mask=rows_k < N,
            other=0.0,
        )

        for ii in range(BLOCK_K - 1, -1, -1):
            row_i = k + ii
            if row_i < N:
                L_vals = tl.load(
                    L_ptr + L_base + rows_k * stride_L + row_i,
                    mask=(rows_k > row_i) & (rows_k < N),
                    other=0.0,
                )
                dot = tl.sum(L_vals * x_block, axis=0)
                y_val = tl.sum(tl.where(rows_k == row_i, x_block, 0.0), axis=0)
                diag = tl.load(L_ptr + L_base + row_i * stride_L + row_i)
                x_val = (y_val - dot) / diag
                x_block = tl.where(rows_k == row_i, x_val, x_block)

        tl.store(X_ptr + B_base + rows_k * stride_B, x_block, mask=rows_k < N)

        for m in range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            L_tile = tl.load(
                L_ptr + L_base + rows_k[None, :] * stride_L + rows_m[:, None],
                mask=(rows_m[:, None] < N) & (rows_k[None, :] < N),
                other=0.0,
            )
            head = tl.load(
                X_ptr + B_base + rows_m * stride_B,
                mask=rows_m < N,
                other=0.0,
            )
            head = head - tl.sum(L_tile * x_block[None, :], axis=1)
            tl.store(X_ptr + B_base + rows_m * stride_B, head, mask=rows_m < N)


@libentry()
@triton.jit
def cholesky_solve_single_rhs_blocked_upper_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    N: tl.constexpr,
    batch_stride_L,
    batch_stride_B,
    stride_L,
    stride_B,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Blocked upper-factor single-RHS Cholesky solve."""
    batch_pid = program_id(0)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    k_offsets = tl.arange(0, BLOCK_K)
    m_offsets = tl.arange(0, BLOCK_M)

    # Forward blocked TRSV: U^T * Y = B.
    for k in range(0, N, BLOCK_K):
        rows_k = k + k_offsets
        if k == 0:
            y_block = tl.load(
                B_ptr + B_base + rows_k * stride_B,
                mask=rows_k < N,
                other=0.0,
            )
        else:
            y_block = tl.load(
                X_ptr + B_base + rows_k * stride_B,
                mask=rows_k < N,
                other=0.0,
            )

        for i in range(BLOCK_K):
            row_i = k + i
            if row_i < N:
                U_vals = tl.load(
                    L_ptr + L_base + rows_k * stride_L + row_i,
                    mask=rows_k < row_i,
                    other=0.0,
                )
                dot = tl.sum(U_vals * y_block, axis=0)
                rhs_val = tl.sum(tl.where(rows_k == row_i, y_block, 0.0), axis=0)
                diag = tl.load(L_ptr + L_base + row_i * stride_L + row_i)
                y_val = (rhs_val - dot) / diag
                y_block = tl.where(rows_k == row_i, y_val, y_block)

        tl.store(X_ptr + B_base + rows_k * stride_B, y_block, mask=rows_k < N)

        for m in range(k + BLOCK_K, N, BLOCK_M):
            rows_m = m + m_offsets
            U_tile = tl.load(
                L_ptr + L_base + rows_k[:, None] * stride_L + rows_m[None, :],
                mask=(rows_k[:, None] < N) & (rows_m[None, :] < N),
                other=0.0,
            )
            if k == 0:
                tail = tl.load(
                    B_ptr + B_base + rows_m * stride_B,
                    mask=rows_m < N,
                    other=0.0,
                )
            else:
                tail = tl.load(
                    X_ptr + B_base + rows_m * stride_B,
                    mask=rows_m < N,
                    other=0.0,
                )
            tail = tail - tl.sum(U_tile * y_block[:, None], axis=0)
            tl.store(X_ptr + B_base + rows_m * stride_B, tail, mask=rows_m < N)

    # Backward blocked TRSV: U * X = Y.
    for k in range(N - BLOCK_K, -1, -BLOCK_K):
        rows_k = k + k_offsets
        x_block = tl.load(
            X_ptr + B_base + rows_k * stride_B,
            mask=rows_k < N,
            other=0.0,
        )

        for ii in range(BLOCK_K - 1, -1, -1):
            row_i = k + ii
            if row_i < N:
                U_vals = tl.load(
                    L_ptr + L_base + row_i * stride_L + rows_k,
                    mask=(rows_k > row_i) & (rows_k < N),
                    other=0.0,
                )
                dot = tl.sum(U_vals * x_block, axis=0)
                y_val = tl.sum(tl.where(rows_k == row_i, x_block, 0.0), axis=0)
                diag = tl.load(L_ptr + L_base + row_i * stride_L + row_i)
                x_val = (y_val - dot) / diag
                x_block = tl.where(rows_k == row_i, x_val, x_block)

        tl.store(X_ptr + B_base + rows_k * stride_B, x_block, mask=rows_k < N)

        for m in range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            U_tile = tl.load(
                L_ptr + L_base + rows_m[:, None] * stride_L + rows_k[None, :],
                mask=(rows_m[:, None] < N) & (rows_k[None, :] < N),
                other=0.0,
            )
            head = tl.load(
                X_ptr + B_base + rows_m * stride_B,
                mask=rows_m < N,
                other=0.0,
            )
            head = head - tl.sum(U_tile * x_block[None, :], axis=1)
            tl.store(X_ptr + B_base + rows_m * stride_B, head, mask=rows_m < N)

@libentry()
@triton.jit
def cholesky_solve_small_single_rhs_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    N: tl.constexpr,
    batch_stride_L,
    batch_stride_B,
    stride_L,
    stride_B,
    BLOCK_N: tl.constexpr,
    dtype_flag: tl.constexpr,
    upper: tl.constexpr,
):
    """Small-N single RHS kernel that keeps intermediate values in registers.

    This avoids the global-memory Y write/read round trip used by the generic
    single-RHS path. It is intended for small systems where BLOCK_N <= 64.
    """
    batch_pid = program_id(0)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    offsets = tl.arange(0, BLOCK_N)

    if dtype_flag == 0:
        y_vec = tl.zeros([BLOCK_N], dtype=tl.float32)
    else:
        y_vec = tl.zeros([BLOCK_N], dtype=tl.float64)

    # Phase 1: solve L * Y = B or U^T * Y = B.
    for i in range(N):
        if upper:
            L_vals = tl.load(
                L_ptr + L_base + offsets * stride_L + i,
                mask=offsets < i,
                other=0.0,
            )
        else:
            L_vals = tl.load(
                L_ptr + L_base + i * stride_L + offsets,
                mask=offsets < i,
                other=0.0,
            )
        dot = tl.sum(L_vals * y_vec, axis=0)
        rhs_val = tl.load(B_ptr + B_base + i * stride_B)
        diag = tl.load(L_ptr + L_base + i * stride_L + i)
        y_i = (rhs_val - dot) / diag
        y_vec = tl.where(offsets == i, y_i, y_vec)

    x_vec = y_vec

    # Phase 2: solve L^T * X = Y or U * X = Y.
    for i in range(N - 1, -1, -1):
        active = (offsets > i) & (offsets < N)
        if upper:
            L_vals = tl.load(
                L_ptr + L_base + i * stride_L + offsets,
                mask=active,
                other=0.0,
            )
        else:
            L_vals = tl.load(
                L_ptr + L_base + offsets * stride_L + i,
                mask=active,
                other=0.0,
            )
        dot = tl.sum(L_vals * x_vec, axis=0)
        y_i = tl.sum(tl.where(offsets == i, y_vec, 0.0), axis=0)
        diag = tl.load(L_ptr + L_base + i * stride_L + i)
        x_i = (y_i - dot) / diag
        x_vec = tl.where(offsets == i, x_i, x_vec)

    tl.store(
        X_ptr + B_base + offsets * stride_B,
        x_vec,
        mask=offsets < N,
    )


@libentry()
@triton.jit
def cholesky_solve_small_nrhs_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    N: tl.constexpr,
    nrhs: tl.constexpr,
    batch_stride_L,
    batch_stride_B,
    stride_L,
    stride_B,
    BLOCK_N: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
    dtype_flag: tl.constexpr,
    upper: tl.constexpr,
):
    """Small-N/small-RHS fused kernel with register-resident Y/X.

    This keeps the intermediate triangular-solve result as a [N, nrhs]
    register tile and writes the final X once. It is limited to small systems
    to avoid excessive register pressure.
    """
    batch_pid = program_id(0)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_RHS)
    cols_mask = cols < nrhs

    if dtype_flag == 0:
        y_tile = tl.zeros([BLOCK_N, BLOCK_RHS], dtype=tl.float32)
    else:
        y_tile = tl.zeros([BLOCK_N, BLOCK_RHS], dtype=tl.float64)

    # Phase 1: solve L * Y = B or U^T * Y = B for all small RHS columns.
    for i in range(N):
        if upper:
            L_vals = tl.load(
                L_ptr + L_base + rows * stride_L + i,
                mask=rows < i,
                other=0.0,
            )
        else:
            L_vals = tl.load(
                L_ptr + L_base + i * stride_L + rows,
                mask=rows < i,
                other=0.0,
            )
        dot = tl.sum(L_vals[:, None] * y_tile, axis=0)
        rhs_vals = tl.load(
            B_ptr + B_base + i * stride_B + cols,
            mask=cols_mask,
            other=0.0,
        )
        diag = tl.load(L_ptr + L_base + i * stride_L + i)
        y_vals = (rhs_vals - dot) / diag
        y_tile = tl.where(rows[:, None] == i, y_vals[None, :], y_tile)

    x_tile = y_tile

    # Phase 2: solve L^T * X = Y or U * X = Y.
    for i in range(N - 1, -1, -1):
        active = (rows > i) & (rows < N)
        if upper:
            L_vals = tl.load(
                L_ptr + L_base + i * stride_L + rows,
                mask=active,
                other=0.0,
            )
        else:
            L_vals = tl.load(
                L_ptr + L_base + rows * stride_L + i,
                mask=active,
                other=0.0,
            )
        dot = tl.sum(L_vals[:, None] * x_tile, axis=0)
        y_vals = tl.sum(tl.where(rows[:, None] == i, y_tile, 0.0), axis=0)
        diag = tl.load(L_ptr + L_base + i * stride_L + i)
        x_vals = (y_vals - dot) / diag
        x_tile = tl.where(rows[:, None] == i, x_vals[None, :], x_tile)

    tl.store(
        X_ptr + B_base + rows[:, None] * stride_B + cols[None, :],
        x_tile,
        mask=(rows[:, None] < N) & cols_mask[None, :],
    )


def _can_use_small_nrhs_path(N, nrhs):
    return N <= 32 and 1 < nrhs <= 4


@libentry()
@triton.jit
def cholesky_solve_single_rhs_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    N: tl.constexpr,
    batch_stride_L,
    batch_stride_B,
    stride_L,
    stride_B,
    dtype_flag: tl.constexpr,
    upper: tl.constexpr,
):
    """Specialized Cholesky solve kernel for nrhs == 1.

    This path avoids RHS tile vectors and tail masks used by the general
    multi-RHS kernel. Each program solves one single-RHS system for one batch.
    """
    batch_pid = program_id(0)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B

    # Phase 1: solve L * Y = B or U^T * Y = B.
    for i in range(N):
        sum_val = tl.load(B_ptr + B_base + i * stride_B)
        for j in range(i):
            if upper:
                L_val = tl.load(L_ptr + L_base + j * stride_L + i)
            else:
                L_val = tl.load(L_ptr + L_base + i * stride_L + j)
            Y_val = tl.load(X_ptr + B_base + j * stride_B)
            sum_val = sum_val - L_val * Y_val
        diag = tl.load(L_ptr + L_base + i * stride_L + i)
        tl.store(X_ptr + B_base + i * stride_B, sum_val / diag)

    # Phase 2: solve L^T * X = Y or U * X = Y.
    for i in range(N - 1, -1, -1):
        sum_val = tl.load(X_ptr + B_base + i * stride_B)
        for j in range(i + 1, N):
            if upper:
                L_val = tl.load(L_ptr + L_base + i * stride_L + j)
            else:
                L_val = tl.load(L_ptr + L_base + j * stride_L + i)
            Xj_val = tl.load(X_ptr + B_base + j * stride_B)
            sum_val = sum_val - L_val * Xj_val
        diag = tl.load(L_ptr + L_base + i * stride_L + i)
        tl.store(X_ptr + B_base + i * stride_B, sum_val / diag)


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

    dtype_flag = 0 if B.dtype == torch.float32 else 1

    with torch.no_grad():
        if _can_use_blocked_lower_path(B, upper, N, nrhs):
            grid = (batch_size, triton.cdiv(nrhs, 16))
            cholesky_solve_blocked_lower_kernel[grid](
                L_kernel,
                B_kernel,
                X_kernel,
                N,
                nrhs,
                batch_stride_L,
                batch_stride_B,
                stride_L,
                stride_B,
                BLOCK_K=32,
                BLOCK_M=32,
                BLOCK_RHS=16,
                num_warps=4,
                num_stages=3,
            )
        elif _can_use_blocked_upper_path(B, upper, N, nrhs):
            grid = (batch_size, triton.cdiv(nrhs, 16))
            cholesky_solve_blocked_upper_kernel[grid](
                L_kernel,
                B_kernel,
                X_kernel,
                N,
                nrhs,
                batch_stride_L,
                batch_stride_B,
                stride_L,
                stride_B,
                BLOCK_K=32,
                BLOCK_M=32,
                BLOCK_RHS=16,
                num_warps=4,
                num_stages=3,
            )
        elif _can_use_blocked_single_rhs_path(B, N, nrhs):
            if upper:
                cholesky_solve_single_rhs_blocked_upper_kernel[(batch_size,)](
                    L_kernel,
                    B_kernel,
                    X_kernel,
                    N,
                    batch_stride_L,
                    batch_stride_B,
                    stride_L,
                    stride_B,
                    BLOCK_K=32,
                    BLOCK_M=32,
                    num_warps=4,
                    num_stages=3,
                )
            else:
                cholesky_solve_single_rhs_blocked_lower_kernel[(batch_size,)](
                    L_kernel,
                    B_kernel,
                    X_kernel,
                    N,
                    batch_stride_L,
                    batch_stride_B,
                    stride_L,
                    stride_B,
                    BLOCK_K=32,
                    BLOCK_M=32,
                    num_warps=4,
                    num_stages=3,
                )
        elif nrhs == 1 and N <= 64:
            block_n = triton.next_power_of_2(N)
            cholesky_solve_small_single_rhs_kernel[(batch_size,)](
                L_kernel,
                B_kernel,
                X_kernel,
                N,
                batch_stride_L,
                batch_stride_B,
                stride_L,
                stride_B,
                BLOCK_N=block_n,
                dtype_flag=dtype_flag,
                upper=upper,
            )
        elif nrhs == 1:
            cholesky_solve_single_rhs_kernel[(batch_size,)](
                L_kernel,
                B_kernel,
                X_kernel,
                N,
                batch_stride_L,
                batch_stride_B,
                stride_L,
                stride_B,
                dtype_flag=dtype_flag,
                upper=upper,
            )
        elif _can_use_small_nrhs_path(N, nrhs):
            block_n = triton.next_power_of_2(N)
            block_rhs = triton.next_power_of_2(nrhs)
            cholesky_solve_small_nrhs_kernel[(batch_size,)](
                L_kernel,
                B_kernel,
                X_kernel,
                N,
                nrhs,
                batch_stride_L,
                batch_stride_B,
                stride_L,
                stride_B,
                BLOCK_N=block_n,
                BLOCK_RHS=block_rhs,
                dtype_flag=dtype_flag,
                upper=upper,
            )
        else:
            grid = lambda meta: (batch_size, triton.cdiv(nrhs, meta["BLOCK_RHS"]))
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
                dtype_flag=dtype_flag,
                upper=upper,
            )

    return X
