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

# Curated autotune configs for blocked Cholesky solve kernels.
# Following the group_gemm.py pattern: a hand-picked list rather than
# a Cartesian explosion. All configs use BLOCK_RHS >= 16 to satisfy
# the fp32 tl.dot MMA constraint (N >= 16).
BLOCKED_CHOLESKY_SOLVE_CONFIGS = [
    # Large tiles: for large N with high compute intensity
    triton.Config({"BLOCK_K": 32, "BLOCK_M": 64, "BLOCK_RHS": 16}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_K": 32, "BLOCK_M": 32, "BLOCK_RHS": 16}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_K": 32, "BLOCK_M": 64, "BLOCK_RHS": 16}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_K": 32, "BLOCK_M": 32, "BLOCK_RHS": 16}, num_warps=4, num_stages=3),
    # Medium tiles: balanced for mid-sized problems
    triton.Config({"BLOCK_K": 16, "BLOCK_M": 64, "BLOCK_RHS": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_K": 16, "BLOCK_M": 32, "BLOCK_RHS": 16}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_K": 16, "BLOCK_M": 32, "BLOCK_RHS": 16}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_K": 16, "BLOCK_M": 32, "BLOCK_RHS": 16}, num_warps=4, num_stages=4),
    # Small tiles: more outer-loop iterations, better for fp64 / small N
    triton.Config({"BLOCK_K": 16, "BLOCK_M": 16, "BLOCK_RHS": 16}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_K": 16, "BLOCK_M": 16, "BLOCK_RHS": 16}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_K": 16, "BLOCK_M": 16, "BLOCK_RHS": 16}, num_warps=2, num_stages=4),
]

# Deduplicate (some fp32/fp64 combos may overlap)
_seen = set()
_uniques = []
for _c in BLOCKED_CHOLESKY_SOLVE_CONFIGS:
    _key = (tuple(sorted(_c.kwargs.items())), _c.num_warps, _c.num_stages)
    if _key not in _seen:
        _seen.add(_key)
        _uniques.append(_c)
BLOCKED_CHOLESKY_SOLVE_CONFIGS = _uniques
del _seen, _uniques, _c, _key


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
        # Fast reciprocal with Newton refinement
        inv_diag = 1.0 / diag
        inv_diag = inv_diag * (2.0 - diag * inv_diag)
        if dtype_flag == 1:
            inv_diag = inv_diag * (2.0 - diag * inv_diag)
        tl.store(
            X_ptr + B_base + i * stride_B + cols, sum_val * inv_diag, mask=cols_mask
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
        inv_diag = 1.0 / diag
        inv_diag = inv_diag * (2.0 - diag * inv_diag)
        if dtype_flag == 1:
            inv_diag = inv_diag * (2.0 - diag * inv_diag)
        tl.store(
            X_ptr + B_base + i * stride_B + cols, sum_val * inv_diag, mask=cols_mask
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
    stride_L_col: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
    dtype_flag: tl.constexpr,
):
    """Blocked lower-factor Cholesky solve.

    Solves L L^T X = B for one batch and one RHS tile. Accepts both
    standard row-major lower (stride_L_col=1) and transposed upper
    factor as L' = U^T via swapped strides (stride_L_col=N, stride_L=1).
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
                mask=(rows_k[:, None] < N) & rhs_mask[None, :], other=0.0)
        else:
            y_block = tl.load(
                X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
                mask=(rows_k[:, None] < N) & rhs_mask[None, :], other=0.0)

        for i in range(BLOCK_K):
            row_i = k + i
            if row_i < N:
                row_mask = rows_k[:, None] == row_i
                y_cur = tl.sum(tl.where(row_mask, y_block, 0.0), axis=0)
                diag = tl.load(L_ptr + L_base + row_i * stride_L + row_i * stride_L_col)
                inv_diag = 1.0 / diag
                inv_diag = inv_diag * (2.0 - diag * inv_diag)
                if dtype_flag == 1:
                    inv_diag = inv_diag * (2.0 - diag * inv_diag)
                y_new = y_cur * inv_diag
                L_col = tl.load(
                    L_ptr + L_base + rows_k * stride_L + row_i * stride_L_col,
                    mask=(rows_k > row_i) & (rows_k < N), other=0.0)
                y_block = y_block - L_col[:, None] * y_new[None, :]
                y_block = tl.where(row_mask, y_new[None, :], y_block)

        tl.store(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            y_block, mask=(rows_k[:, None] < N) & rhs_mask[None, :])

        for m in range(k + BLOCK_K, N, BLOCK_M):
            rows_m = m + m_offsets
            L_tile = tl.load(
                L_ptr + L_base + rows_m[:, None] * stride_L + rows_k[None, :] * stride_L_col,
                mask=(rows_m[:, None] < N) & (rows_k[None, :] < N), other=0.0)
            if k == 0:
                tail = tl.load(
                    B_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                    mask=(rows_m[:, None] < N) & rhs_mask[None, :], other=0.0)
            else:
                tail = tl.load(
                    X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                    mask=(rows_m[:, None] < N) & rhs_mask[None, :], other=0.0)
            tail = tail - tl.dot(L_tile, y_block)
            tl.store(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                tail, mask=(rows_m[:, None] < N) & rhs_mask[None, :])

    # Backward blocked TRSM: L^T * X = Y.
    for k in range(N - BLOCK_K, -1, -BLOCK_K):
        rows_k = k + k_offsets
        x_block = tl.load(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            mask=(rows_k[:, None] < N) & rhs_mask[None, :], other=0.0)

        for ii in range(BLOCK_K - 1, -1, -1):
            row_i = k + ii
            if row_i < N:
                row_mask = rows_k[:, None] == row_i
                y_cur = tl.sum(tl.where(row_mask, x_block, 0.0), axis=0)
                diag = tl.load(L_ptr + L_base + row_i * stride_L + row_i * stride_L_col)
                inv_diag = 1.0 / diag
                inv_diag = inv_diag * (2.0 - diag * inv_diag)
                if dtype_flag == 1:
                    inv_diag = inv_diag * (2.0 - diag * inv_diag)
                x_new = y_cur * inv_diag
                L_row = tl.load(
                    L_ptr + L_base + row_i * stride_L + rows_k * stride_L_col,
                    mask=rows_k < row_i, other=0.0)
                x_block = x_block - L_row[:, None] * x_new[None, :]
                x_block = tl.where(row_mask, x_new[None, :], x_block)

        tl.store(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            x_block, mask=(rows_k[:, None] < N) & rhs_mask[None, :])

        for m in range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            L_tile = tl.load(
                L_ptr + L_base + rows_k[None, :] * stride_L + rows_m[:, None] * stride_L_col,
                mask=(rows_m[:, None] < N) & (rows_k[None, :] < N), other=0.0)
            head = tl.load(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                mask=(rows_m[:, None] < N) & rhs_mask[None, :], other=0.0)
            head = head - tl.dot(L_tile, x_block)
            tl.store(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                head, mask=(rows_m[:, None] < N) & rhs_mask[None, :])


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
    dtype_flag: tl.constexpr,
):
    """Blocked upper-factor Cholesky solve.

    Solves U^T U X = B for one batch and one RHS tile. Mirrors the lower
    blocked path while keeping the upper storage layout intact.

    Includes fast reciprocal (Newton refinement) for diagonal division,
    matching the algorithm used by cuSOLVER's potrs kernel.
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
                row_mask = rows_k[:, None] == row_i
                y_cur = tl.sum(tl.where(row_mask, y_block, 0.0), axis=0)
                diag = tl.load(L_ptr + L_base + row_i * stride_L + row_i)
                inv_diag = 1.0 / diag
                inv_diag = inv_diag * (2.0 - diag * inv_diag)
                if dtype_flag == 1:
                    inv_diag = inv_diag * (2.0 - diag * inv_diag)
                y_new = y_cur * inv_diag
                U_row = tl.load(
                    L_ptr + L_base + row_i * stride_L + rows_k,
                    mask=(rows_k > row_i) & (rows_k < N),
                    other=0.0,
                )
                y_block = y_block - U_row[:, None] * y_new[None, :]
                y_block = tl.where(row_mask, y_new[None, :], y_block)

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
                row_mask = rows_k[:, None] == row_i
                y_cur = tl.sum(tl.where(row_mask, x_block, 0.0), axis=0)
                diag = tl.load(L_ptr + L_base + row_i * stride_L + row_i)
                inv_diag = 1.0 / diag
                inv_diag = inv_diag * (2.0 - diag * inv_diag)
                if dtype_flag == 1:
                    inv_diag = inv_diag * (2.0 - diag * inv_diag)
                x_new = y_cur * inv_diag
                U_col = tl.load(
                    L_ptr + L_base + rows_k * stride_L + row_i,
                    mask=rows_k < row_i,
                    other=0.0,
                )
                x_block = x_block - U_col[:, None] * x_new[None, :]
                x_block = tl.where(row_mask, x_new[None, :], x_block)

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


@libentry()
@triton.jit
def cholesky_solve_blocked_lower_fp64_kernel(
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
    """fp64-dedicated blocked lower-factor Cholesky solve.

    Removes dtype_flag branch and fp32 logic. Uses standard row-major
    lower factor layout (diag access via row*stride_L+row).
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
                mask=(rows_k[:, None] < N) & rhs_mask[None, :], other=0.0)
        else:
            y_block = tl.load(
                X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
                mask=(rows_k[:, None] < N) & rhs_mask[None, :], other=0.0)

        for i in range(BLOCK_K):
            row_i = k + i
            if row_i < N:
                row_mask = rows_k[:, None] == row_i
                y_cur = tl.sum(tl.where(row_mask, y_block, 0.0), axis=0)
                diag = tl.load(L_ptr + L_base + row_i * stride_L + row_i)
                inv_diag = 1.0 / diag
                inv_diag = inv_diag * (2.0 - diag * inv_diag)
                inv_diag = inv_diag * (2.0 - diag * inv_diag)
                y_new = y_cur * inv_diag
                L_col = tl.load(
                    L_ptr + L_base + rows_k * stride_L + row_i,
                    mask=(rows_k > row_i) & (rows_k < N), other=0.0)
                y_block = y_block - L_col[:, None] * y_new[None, :]
                y_block = tl.where(row_mask, y_new[None, :], y_block)

        tl.store(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            y_block, mask=(rows_k[:, None] < N) & rhs_mask[None, :])

        for m in range(k + BLOCK_K, N, BLOCK_M):
            rows_m = m + m_offsets
            L_tile = tl.load(
                L_ptr + L_base + rows_m[:, None] * stride_L + rows_k[None, :],
                mask=(rows_m[:, None] < N) & (rows_k[None, :] < N), other=0.0)
            if k == 0:
                tail = tl.load(
                    B_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                    mask=(rows_m[:, None] < N) & rhs_mask[None, :], other=0.0)
            else:
                tail = tl.load(
                    X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                    mask=(rows_m[:, None] < N) & rhs_mask[None, :], other=0.0)
            tail = tail - tl.dot(L_tile, y_block)
            tl.store(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                tail, mask=(rows_m[:, None] < N) & rhs_mask[None, :])

    # Backward blocked TRSM: L^T * X = Y.
    for k in range(N - BLOCK_K, -1, -BLOCK_K):
        rows_k = k + k_offsets
        x_block = tl.load(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            mask=(rows_k[:, None] < N) & rhs_mask[None, :], other=0.0)

        for ii in range(BLOCK_K - 1, -1, -1):
            row_i = k + ii
            if row_i < N:
                row_mask = rows_k[:, None] == row_i
                y_cur = tl.sum(tl.where(row_mask, x_block, 0.0), axis=0)
                diag = tl.load(L_ptr + L_base + row_i * stride_L + row_i)
                inv_diag = 1.0 / diag
                inv_diag = inv_diag * (2.0 - diag * inv_diag)
                inv_diag = inv_diag * (2.0 - diag * inv_diag)
                x_new = y_cur * inv_diag
                L_row = tl.load(
                    L_ptr + L_base + row_i * stride_L + rows_k,
                    mask=rows_k < row_i, other=0.0)
                x_block = x_block - L_row[:, None] * x_new[None, :]
                x_block = tl.where(row_mask, x_new[None, :], x_block)

        tl.store(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            x_block, mask=(rows_k[:, None] < N) & rhs_mask[None, :])

        for m in range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            L_tile = tl.load(
                L_ptr + L_base + rows_k[None, :] * stride_L + rows_m[:, None],
                mask=(rows_m[:, None] < N) & (rows_k[None, :] < N), other=0.0)
            head = tl.load(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                mask=(rows_m[:, None] < N) & rhs_mask[None, :], other=0.0)
            head = head - tl.dot(L_tile, x_block)
            tl.store(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                head, mask=(rows_m[:, None] < N) & rhs_mask[None, :])


@libentry()
@triton.jit
def cholesky_solve_blocked_upper_fp64_kernel(
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
    """fp64-dedicated blocked upper-factor Cholesky solve.

    Removes dtype_flag branch and fp32 logic.
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
                mask=(rows_k[:, None] < N) & rhs_mask[None, :], other=0.0)
        else:
            y_block = tl.load(
                X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
                mask=(rows_k[:, None] < N) & rhs_mask[None, :], other=0.0)

        for i in range(BLOCK_K):
            row_i = k + i
            if row_i < N:
                row_mask = rows_k[:, None] == row_i
                y_cur = tl.sum(tl.where(row_mask, y_block, 0.0), axis=0)
                diag = tl.load(L_ptr + L_base + row_i * stride_L + row_i)
                inv_diag = 1.0 / diag
                inv_diag = inv_diag * (2.0 - diag * inv_diag)
                inv_diag = inv_diag * (2.0 - diag * inv_diag)
                y_new = y_cur * inv_diag
                U_row = tl.load(
                    L_ptr + L_base + row_i * stride_L + rows_k,
                    mask=(rows_k > row_i) & (rows_k < N), other=0.0)
                y_block = y_block - U_row[:, None] * y_new[None, :]
                y_block = tl.where(row_mask, y_new[None, :], y_block)

        tl.store(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            y_block, mask=(rows_k[:, None] < N) & rhs_mask[None, :])

        for m in range(k + BLOCK_K, N, BLOCK_M):
            rows_m = m + m_offsets
            U_tile_km = tl.load(
                L_ptr + L_base + rows_k[:, None] * stride_L + rows_m[None, :],
                mask=(rows_k[:, None] < N) & (rows_m[None, :] < N), other=0.0)
            U_tile = tl.trans(U_tile_km)
            if k == 0:
                tail = tl.load(
                    B_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                    mask=(rows_m[:, None] < N) & rhs_mask[None, :], other=0.0)
            else:
                tail = tl.load(
                    X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                    mask=(rows_m[:, None] < N) & rhs_mask[None, :], other=0.0)
            tail = tail - tl.dot(U_tile, y_block)
            tl.store(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                tail, mask=(rows_m[:, None] < N) & rhs_mask[None, :])

    # Backward blocked TRSM: U * X = Y.
    for k in range(N - BLOCK_K, -1, -BLOCK_K):
        rows_k = k + k_offsets
        x_block = tl.load(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            mask=(rows_k[:, None] < N) & rhs_mask[None, :], other=0.0)

        for ii in range(BLOCK_K - 1, -1, -1):
            row_i = k + ii
            if row_i < N:
                row_mask = rows_k[:, None] == row_i
                y_cur = tl.sum(tl.where(row_mask, x_block, 0.0), axis=0)
                diag = tl.load(L_ptr + L_base + row_i * stride_L + row_i)
                inv_diag = 1.0 / diag
                inv_diag = inv_diag * (2.0 - diag * inv_diag)
                inv_diag = inv_diag * (2.0 - diag * inv_diag)
                x_new = y_cur * inv_diag
                U_col = tl.load(
                    L_ptr + L_base + rows_k * stride_L + row_i,
                    mask=rows_k < row_i, other=0.0)
                x_block = x_block - U_col[:, None] * x_new[None, :]
                x_block = tl.where(row_mask, x_new[None, :], x_block)

        tl.store(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            x_block, mask=(rows_k[:, None] < N) & rhs_mask[None, :])

        for m in range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            U_tile = tl.load(
                L_ptr + L_base + rows_m[:, None] * stride_L + rows_k[None, :],
                mask=(rows_m[:, None] < N) & (rows_k[None, :] < N), other=0.0)
            head = tl.load(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                mask=(rows_m[:, None] < N) & rhs_mask[None, :], other=0.0)
            head = head - tl.dot(U_tile, x_block)
            tl.store(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                head, mask=(rows_m[:, None] < N) & rhs_mask[None, :])


def _can_use_blocked_lower_path(upper, N, nrhs):
    return (
        not upper
        and N >= 64
        and N % 32 == 0
        and nrhs >= 4
    )


def _can_use_blocked_upper_path(upper, N, nrhs):
    return (
        upper
        and N >= 64
        and N % 32 == 0
        and nrhs >= 4
    )


def _can_use_blocked_single_rhs_path(N, nrhs):
    return nrhs == 1 and N >= 128 and N % 32 == 0


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
    dtype_flag: tl.constexpr,
):
    """Blocked lower-factor single-RHS Cholesky solve.

    Includes fast reciprocal (Newton refinement) for diagonal division,
    matching the algorithm used by cuSOLVER's potrs kernel.
    """
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
                row_mask = rows_k == row_i
                y_cur = tl.sum(tl.where(row_mask, y_block, 0.0), axis=0)
                diag = tl.load(L_ptr + L_base + row_i * stride_L + row_i)
                inv_diag = 1.0 / diag
                inv_diag = inv_diag * (2.0 - diag * inv_diag)
                if dtype_flag == 1:
                    inv_diag = inv_diag * (2.0 - diag * inv_diag)
                y_new = y_cur * inv_diag
                L_col = tl.load(
                    L_ptr + L_base + rows_k * stride_L + row_i,
                    mask=(rows_k > row_i) & (rows_k < N),
                    other=0.0,
                )
                y_block = y_block - L_col * y_new
                y_block = tl.where(row_mask, y_new, y_block)

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
                row_mask = rows_k == row_i
                y_cur = tl.sum(tl.where(row_mask, x_block, 0.0), axis=0)
                diag = tl.load(L_ptr + L_base + row_i * stride_L + row_i)
                inv_diag = 1.0 / diag
                inv_diag = inv_diag * (2.0 - diag * inv_diag)
                if dtype_flag == 1:
                    inv_diag = inv_diag * (2.0 - diag * inv_diag)
                x_new = y_cur * inv_diag
                L_row = tl.load(
                    L_ptr + L_base + row_i * stride_L + rows_k,
                    mask=rows_k < row_i,
                    other=0.0,
                )
                x_block = x_block - L_row * x_new
                x_block = tl.where(row_mask, x_new, x_block)

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
    dtype_flag: tl.constexpr,
):
    """Blocked upper-factor single-RHS Cholesky solve.

    Includes fast reciprocal (Newton refinement) for diagonal division,
    matching the algorithm used by cuSOLVER's potrs kernel.
    """
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
                row_mask = rows_k == row_i
                y_cur = tl.sum(tl.where(row_mask, y_block, 0.0), axis=0)
                diag = tl.load(L_ptr + L_base + row_i * stride_L + row_i)
                inv_diag = 1.0 / diag
                inv_diag = inv_diag * (2.0 - diag * inv_diag)
                if dtype_flag == 1:
                    inv_diag = inv_diag * (2.0 - diag * inv_diag)
                y_new = y_cur * inv_diag
                U_row = tl.load(
                    L_ptr + L_base + row_i * stride_L + rows_k,
                    mask=(rows_k > row_i) & (rows_k < N),
                    other=0.0,
                )
                y_block = y_block - U_row * y_new
                y_block = tl.where(row_mask, y_new, y_block)

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
                row_mask = rows_k == row_i
                y_cur = tl.sum(tl.where(row_mask, x_block, 0.0), axis=0)
                diag = tl.load(L_ptr + L_base + row_i * stride_L + row_i)
                inv_diag = 1.0 / diag
                inv_diag = inv_diag * (2.0 - diag * inv_diag)
                if dtype_flag == 1:
                    inv_diag = inv_diag * (2.0 - diag * inv_diag)
                x_new = y_cur * inv_diag
                U_col = tl.load(
                    L_ptr + L_base + rows_k * stride_L + row_i,
                    mask=rows_k < row_i,
                    other=0.0,
                )
                x_block = x_block - U_col * x_new
                x_block = tl.where(row_mask, x_new, x_block)

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
                mask=offsets < i, other=0.0,
            )
        else:
            L_vals = tl.load(
                L_ptr + L_base + i * stride_L + offsets,
                mask=offsets < i, other=0.0,
            )
        dot = tl.sum(L_vals * y_vec, axis=0)
        rhs_val = tl.load(B_ptr + B_base + i * stride_B)
        diag = tl.load(L_ptr + L_base + i * stride_L + i)
        inv_diag = 1.0 / diag
        inv_diag = inv_diag * (2.0 - diag * inv_diag)
        if dtype_flag == 1:
            inv_diag = inv_diag * (2.0 - diag * inv_diag)
        y_i = (rhs_val - dot) * inv_diag
        y_vec = tl.where(offsets == i, y_i, y_vec)

    x_vec = y_vec

    # Phase 2: solve L^T * X = Y or U * X = Y.
    for i in range(N - 1, -1, -1):
        active = (offsets > i) & (offsets < N)
        if upper:
            L_vals = tl.load(
                L_ptr + L_base + i * stride_L + offsets,
                mask=active, other=0.0,
            )
        else:
            L_vals = tl.load(
                L_ptr + L_base + offsets * stride_L + i,
                mask=active, other=0.0,
            )
        dot = tl.sum(L_vals * x_vec, axis=0)
        y_i = tl.sum(tl.where(offsets == i, y_vec, 0.0), axis=0)
        diag = tl.load(L_ptr + L_base + i * stride_L + i)
        inv_diag = 1.0 / diag
        inv_diag = inv_diag * (2.0 - diag * inv_diag)
        if dtype_flag == 1:
            inv_diag = inv_diag * (2.0 - diag * inv_diag)
        x_i = (y_i - dot) * inv_diag
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
        inv_diag = 1.0 / diag
        inv_diag = inv_diag * (2.0 - diag * inv_diag)
        if dtype_flag == 1:
            inv_diag = inv_diag * (2.0 - diag * inv_diag)
        y_vals = (rhs_vals - dot) * inv_diag
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
        inv_diag = 1.0 / diag
        inv_diag = inv_diag * (2.0 - diag * inv_diag)
        if dtype_flag == 1:
            inv_diag = inv_diag * (2.0 - diag * inv_diag)
        x_vals = (y_vals - dot) * inv_diag
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
        inv_diag = 1.0 / diag
        inv_diag = inv_diag * (2.0 - diag * inv_diag)
        if dtype_flag == 1:
            inv_diag = inv_diag * (2.0 - diag * inv_diag)
        tl.store(X_ptr + B_base + i * stride_B, sum_val * inv_diag)

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
        inv_diag = 1.0 / diag
        inv_diag = inv_diag * (2.0 - diag * inv_diag)
        if dtype_flag == 1:
            inv_diag = inv_diag * (2.0 - diag * inv_diag)
        tl.store(X_ptr + B_base + i * stride_B, sum_val * inv_diag)


def _get_blocked_tile_configs(dtype, nrhs, N):
    """Return tile sizes for blocked kernels based on dtype, nrhs, and N.

    fp32: BLOCK_K=16 gives more outer-loop iterations for latency hiding;
          BLOCK_RHS pinned at 16 by tl.dot MMA constraint.
    fp64: BLOCK_K=16 for N<=64 (pipeline depth), BLOCK_K=32 for N>64
          (fewer iterations = less overhead).
    """
    if dtype == torch.float64:
        max_rhs, min_rhs = 8, 4
        blk_k, blk_m = 16, 16
    else:
        max_rhs, min_rhs = 16, 16
        blk_k, blk_m = 32, 32
    # Adaptive BLOCK_RHS: only reduce when cdiv(nrhs, max_rhs) < min_rhs_tiles
    # to avoid creating too many tiny tiles (launch overhead > compute)
    min_rhs_tiles = 2
    if triton.cdiv(nrhs, max_rhs) >= min_rhs_tiles:
        blk_rhs = max_rhs
    else:
        blk_rhs = max(min_rhs, triton.next_power_of_2(
            max(1, nrhs // min_rhs_tiles)))
    return {"BLOCK_K": blk_k, "BLOCK_M": blk_m, "BLOCK_RHS": blk_rhs}


def _get_blocked_warp_config(dtype):
    """Return warp/stage config for blocked kernels based on dtype."""
    if dtype == torch.float64:
        return {"num_warps": 4, "num_stages": 2}
    return {"num_warps": 4, "num_stages": 3}


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
    stride_L_col = L_kernel.stride(2)
    stride_B = B_kernel.stride(1)
    batch_stride_L = L_kernel.stride(0)
    batch_stride_B = B_kernel.stride(0)

    dtype_flag = 0 if B.dtype == torch.float32 else 1

    # For fp32 upper factor U stored row-major, treat it as L' = U^T by
    # swapping strides and routing through the blocked lower kernel.
    # Only applies to shapes that enter the blocked path.
    use_lower_for_upper = (
        upper and dtype_flag == 0 and N >= 128 and N % 32 == 0 and nrhs >= 4
    )
    if use_lower_for_upper:
        stride_L, stride_L_col = stride_L_col, stride_L

    with torch.no_grad():
        effective_upper = upper and not use_lower_for_upper
        if _can_use_blocked_lower_path(effective_upper, N, nrhs):
            tile = _get_blocked_tile_configs(B.dtype, nrhs, N)
            warp = _get_blocked_warp_config(B.dtype)
            grid = (batch_size, triton.cdiv(nrhs, tile["BLOCK_RHS"]))
            if use_lower_for_upper:
                cholesky_solve_blocked_lower_kernel[grid](
                    L_kernel, B_kernel, X_kernel, N, nrhs,
                    batch_stride_L, batch_stride_B, stride_L, stride_B,
                    stride_L_col=stride_L_col,
                    BLOCK_K=tile["BLOCK_K"], BLOCK_M=tile["BLOCK_M"],
                    BLOCK_RHS=tile["BLOCK_RHS"], dtype_flag=dtype_flag, **warp,
                )
            elif dtype_flag == 1:
                if effective_upper:
                    cholesky_solve_blocked_upper_fp64_kernel[grid](
                        L_kernel, B_kernel, X_kernel, N, nrhs,
                        batch_stride_L, batch_stride_B, stride_L, stride_B,
                        BLOCK_K=tile["BLOCK_K"], BLOCK_M=tile["BLOCK_M"],
                        BLOCK_RHS=tile["BLOCK_RHS"], **warp,
                    )
                else:
                    cholesky_solve_blocked_lower_fp64_kernel[grid](
                        L_kernel, B_kernel, X_kernel, N, nrhs,
                        batch_stride_L, batch_stride_B, stride_L, stride_B,
                        BLOCK_K=tile["BLOCK_K"], BLOCK_M=tile["BLOCK_M"],
                        BLOCK_RHS=tile["BLOCK_RHS"], **warp,
                    )
            else:
                cholesky_solve_blocked_lower_kernel[grid](
                    L_kernel, B_kernel, X_kernel, N, nrhs,
                    batch_stride_L, batch_stride_B, stride_L, stride_B,
                    stride_L_col=stride_L_col,
                    BLOCK_K=tile["BLOCK_K"], BLOCK_M=tile["BLOCK_M"],
                    BLOCK_RHS=tile["BLOCK_RHS"], dtype_flag=dtype_flag, **warp,
                )
        elif _can_use_blocked_upper_path(effective_upper, N, nrhs):
            tile = _get_blocked_tile_configs(B.dtype, nrhs, N)
            warp = _get_blocked_warp_config(B.dtype)
            grid = (batch_size, triton.cdiv(nrhs, tile["BLOCK_RHS"]))
            if dtype_flag == 1:
                cholesky_solve_blocked_upper_fp64_kernel[grid](
                    L_kernel, B_kernel, X_kernel, N, nrhs,
                    batch_stride_L, batch_stride_B, stride_L, stride_B,
                    BLOCK_K=tile["BLOCK_K"], BLOCK_M=tile["BLOCK_M"],
                    BLOCK_RHS=tile["BLOCK_RHS"], **warp,
                )
            else:
                cholesky_solve_blocked_upper_kernel[grid](
                    L_kernel, B_kernel, X_kernel, N, nrhs,
                    batch_stride_L, batch_stride_B, stride_L, stride_B,
                    BLOCK_K=tile["BLOCK_K"], BLOCK_M=tile["BLOCK_M"],
                    BLOCK_RHS=tile["BLOCK_RHS"], dtype_flag=dtype_flag, **warp,
                )
        elif _can_use_blocked_single_rhs_path(N, nrhs):
            tile = _get_blocked_tile_configs(B.dtype, nrhs, N)
            warp = _get_blocked_warp_config(B.dtype)
            if upper:
                cholesky_solve_single_rhs_blocked_upper_kernel[(batch_size,)](
                    L_kernel, B_kernel, X_kernel, N,
                    batch_stride_L, batch_stride_B, stride_L, stride_B,
                    BLOCK_K=tile["BLOCK_K"], BLOCK_M=tile["BLOCK_M"],
                    dtype_flag=dtype_flag, **warp,
                )
            else:
                cholesky_solve_single_rhs_blocked_lower_kernel[(batch_size,)](
                    L_kernel, B_kernel, X_kernel, N,
                    batch_stride_L, batch_stride_B, stride_L, stride_B,
                    BLOCK_K=tile["BLOCK_K"], BLOCK_M=tile["BLOCK_M"],
                    dtype_flag=dtype_flag, **warp,
                )
        elif nrhs == 1 and N <= 64:
            block_n = triton.next_power_of_2(N)
            cholesky_solve_small_single_rhs_kernel[(batch_size,)](
                L_kernel, B_kernel, X_kernel, N,
                batch_stride_L, batch_stride_B, stride_L, stride_B,
                BLOCK_N=block_n, dtype_flag=dtype_flag, upper=upper,
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
                L_kernel, B_kernel, X_kernel, N, nrhs,
                batch_stride_L, batch_stride_B, stride_L, stride_B,
                BLOCK_N=block_n, BLOCK_RHS=block_rhs,
                dtype_flag=dtype_flag, upper=upper,
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
