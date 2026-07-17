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
import triton.experimental.tle as tle

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Autotuning configuration for Ascend
# ---------------------------------------------------------------------------

def _get_cholesky_solve_tuned_configs():
    """Get tuned configs or fallback defaults for cholesky_solve on Ascend."""
    try:
        return runtime.get_tuned_config("cholesky_solve")
    except Exception:
        pass
    # Fallback: reasonable BLOCK_RHS values for Ascend 910B
    return [
        triton.Config({"BLOCK_RHS": br}, num_warps=1, num_stages=1)
        for br in (1, 2, 4, 8, 16)
    ]


# ---------------------------------------------------------------------------
# Scalar kernel for general cases
# ---------------------------------------------------------------------------

@libentry()
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
    """Cholesky solve kernel for Ascend NPU.

    Solves L L^T X = B or U^T U X = B for X, given the lower- or
    upper-triangular Cholesky factor and the right-hand side B.

    Each program computes one RHS tile for one matrix in the batch.
    """
    batch_pid = ext.program_id(0)
    rhs_tile_pid = ext.program_id(1)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    cols = rhs_tile_pid * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    cols_mask = cols < nrhs

    # Phase 1: Forward substitution
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
        # Fast reciprocal with Newton refinement (1 extra iter for fp32 on Ascend)
        inv_diag = 1.0 / diag
        inv_diag = inv_diag * (2.0 - diag * inv_diag)
        if dtype_flag == 1:
            inv_diag = inv_diag * (2.0 - diag * inv_diag)
        tl.store(
            X_ptr + B_base + i * stride_B + cols, sum_val * inv_diag, mask=cols_mask
        )

    # Phase 2: Backward substitution
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


# ---------------------------------------------------------------------------
# Blocked lower kernel (multi-RHS, N >= 64, N % 32 == 0)
# ---------------------------------------------------------------------------

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
    dtype_flag: tl.constexpr,
    PRELOAD_DIAG: tl.constexpr,
):
    """Blocked lower-factor Cholesky solve for Ascend NPU.

    Solves L L^T X = B with blocked TRSM (forward L, backward L^T).

    PRELOAD_DIAG=True:  load all block diagonals as a vector, compute
      inverses once, then use tl.where + tl.sum to apply per row.  Fewer
      scalar loads but more broadcasting.
    PRELOAD_DIAG=False: load each diagonal element individually inside the
      loop.  More scalar loads but avoids the expensive broadcast+mask
      pattern on Ascend VEC units.
    """
    batch_pid = ext.program_id(0)
    rhs_tile_pid = ext.program_id(1)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    rhs_cols = rhs_tile_pid * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    rhs_mask = rhs_cols < nrhs
    k_offsets = tl.arange(0, BLOCK_K)
    m_offsets = tl.arange(0, BLOCK_M)

    # Forward blocked TRSM: L * Y = B
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

        if PRELOAD_DIAG:
            diag_block = tl.load(
                L_ptr + L_base + rows_k * stride_L + rows_k,
                mask=rows_k < N, other=1.0,
            )
            inv_diag_block = 1.0 / diag_block
            inv_diag_block = inv_diag_block * (2.0 - diag_block * inv_diag_block)
            if dtype_flag == 1:
                inv_diag_block = inv_diag_block * (2.0 - diag_block * inv_diag_block)

        for i in range(BLOCK_K):
            row_i = k + i
            if row_i < N:
                row_mask = rows_k[:, None] == row_i
                if PRELOAD_DIAG:
                    # vector: y_block * inv_diag_block[:,None] → broadcast → where → sum
                    y_new = tl.sum(
                        tl.where(row_mask, y_block * inv_diag_block[:, None], 0.0),
                        axis=0,
                    )
                else:
                    # TLE DSA: O(1) extract row + scalar diag
                    y_row_2d = tle.dsa.extract_slice(y_block, (i, 0), (1, BLOCK_RHS), (1, 1))
                    tl.compile_hint(y_row_2d, "disable_bubble_up")
                    y_cur = tl.reshape(y_row_2d, (BLOCK_RHS,))
                    diag_i = tl.load(L_ptr + L_base + row_i * stride_L + row_i)
                    inv_diag = 1.0 / diag_i
                    inv_diag = inv_diag * (2.0 - diag_i * inv_diag)
                    if dtype_flag == 1:
                        inv_diag = inv_diag * (2.0 - diag_i * inv_diag)
                    y_new = y_cur * inv_diag

                L_col = tl.load(
                    L_ptr + L_base + rows_k * stride_L + row_i,
                    mask=(rows_k > row_i) & (rows_k < N),
                    other=0.0,
                )
                y_block = y_block - L_col[:, None] * y_new[None, :]
                if PRELOAD_DIAG:
                    y_block = tl.where(row_mask, y_new[None, :], y_block)
                else:
                    y_block = tle.dsa.insert_slice(
                        y_block, y_new[None, :], (i, 0), (1, BLOCK_RHS), (1, 1)
                    )

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
                    X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                    mask=(rows_m[:, None] < N) & rhs_mask[None, :],
                    other=0.0,
                )
            tail = tail - tl.dot(L_tile, y_block)
            tl.store(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                tail,
                mask=(rows_m[:, None] < N) & rhs_mask[None, :],
            )

    # Backward blocked TRSM: L^T * X = Y
    for k in range(N - BLOCK_K, -1, -BLOCK_K):
        rows_k = k + k_offsets
        x_block = tl.load(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            mask=(rows_k[:, None] < N) & rhs_mask[None, :],
            other=0.0,
        )

        if PRELOAD_DIAG:
            diag_block = tl.load(
                L_ptr + L_base + rows_k * stride_L + rows_k,
                mask=rows_k < N, other=1.0,
            )
            inv_diag_block = 1.0 / diag_block
            inv_diag_block = inv_diag_block * (2.0 - diag_block * inv_diag_block)
            if dtype_flag == 1:
                inv_diag_block = inv_diag_block * (2.0 - diag_block * inv_diag_block)

        for ii in range(BLOCK_K - 1, -1, -1):
            row_i = k + ii
            if row_i < N:
                row_mask = rows_k[:, None] == row_i
                if PRELOAD_DIAG:
                    x_new = tl.sum(
                        tl.where(row_mask, x_block * inv_diag_block[:, None], 0.0),
                        axis=0,
                    )
                else:
                    y_row_2d = tle.dsa.extract_slice(x_block, (ii, 0), (1, BLOCK_RHS), (1, 1))
                    tl.compile_hint(y_row_2d, "disable_bubble_up")
                    y_cur = tl.reshape(y_row_2d, (BLOCK_RHS,))
                    diag_i = tl.load(L_ptr + L_base + row_i * stride_L + row_i)
                    inv_diag = 1.0 / diag_i
                    inv_diag = inv_diag * (2.0 - diag_i * inv_diag)
                    if dtype_flag == 1:
                        inv_diag = inv_diag * (2.0 - diag_i * inv_diag)
                    x_new = y_cur * inv_diag

                L_row = tl.load(
                    L_ptr + L_base + row_i * stride_L + rows_k,
                    mask=rows_k < row_i,
                    other=0.0,
                )
                x_block = x_block - L_row[:, None] * x_new[None, :]
                if PRELOAD_DIAG:
                    x_block = tl.where(row_mask, x_new[None, :], x_block)
                else:
                    x_block = tle.dsa.insert_slice(
                        x_block, x_new[None, :], (ii, 0), (1, BLOCK_RHS), (1, 1)
                    )

        tl.store(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            x_block,
            mask=(rows_k[:, None] < N) & rhs_mask[None, :],
        )

        for m in range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            rows_m_mask = rows_m < k
            L_tile = tl.load(
                L_ptr + L_base + rows_k[None, :] * stride_L + rows_m[:, None],
                mask=rows_m_mask[:, None] & (rows_k[None, :] < N),
                other=0.0,
            )
            head = tl.load(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                mask=rows_m_mask[:, None] & rhs_mask[None, :],
                other=0.0,
            )
            head = head - tl.dot(L_tile, x_block)
            tl.store(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                head,
                mask=rows_m_mask[:, None] & rhs_mask[None, :],
            )


# ---------------------------------------------------------------------------
# Blocked upper kernel (multi-RHS, N >= 64, N % 32 == 0)
# ---------------------------------------------------------------------------

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
    """Blocked upper-factor Cholesky solve for Ascend NPU.

    Solves U^T U X = B with blocked TRSM (forward U^T, backward U).
    Uses per-row scalar diagonal loads (no PRELOAD_DIAG broadcast) to
    avoid expensive tl.where+vmul+tl.sum patterns on Ascend VEC units.
    """
    batch_pid = ext.program_id(0)
    rhs_tile_pid = ext.program_id(1)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    rhs_cols = rhs_tile_pid * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    rhs_mask = rhs_cols < nrhs
    k_offsets = tl.arange(0, BLOCK_K)
    m_offsets = tl.arange(0, BLOCK_M)

    # Forward blocked TRSM: U^T * Y = B
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
                # TLE DSA extract + insert
                y_row_2d = tle.dsa.extract_slice(y_block, (i, 0), (1, BLOCK_RHS), (1, 1))
                tl.compile_hint(y_row_2d, "disable_bubble_up")
                y_cur = tl.reshape(y_row_2d, (BLOCK_RHS,))
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
                y_block = tle.dsa.insert_slice(
                    y_block, y_new[None, :], (i, 0), (1, BLOCK_RHS), (1, 1)
                )

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
            if k == 0:
                tail = tl.load(
                    B_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                    mask=(rows_m[:, None] < N) & rhs_mask[None, :],
                    other=0.0,
                )
            else:
                tail = tl.load(
                    X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                    mask=(rows_m[:, None] < N) & rhs_mask[None, :],
                    other=0.0,
                )
            tail = tail - tl.dot(tl.trans(U_tile_km), y_block)
            tl.store(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                tail,
                mask=(rows_m[:, None] < N) & rhs_mask[None, :],
            )

    # Backward blocked TRSM: U * X = Y
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
                y_row_2d = tle.dsa.extract_slice(x_block, (ii, 0), (1, BLOCK_RHS), (1, 1))
                tl.compile_hint(y_row_2d, "disable_bubble_up")
                y_cur = tl.reshape(y_row_2d, (BLOCK_RHS,))
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
                x_block = tle.dsa.insert_slice(
                    x_block, x_new[None, :], (ii, 0), (1, BLOCK_RHS), (1, 1)
                )

        tl.store(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            x_block,
            mask=(rows_k[:, None] < N) & rhs_mask[None, :],
        )

        for m in range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            rows_m_mask = rows_m < k
            U_tile = tl.load(
                L_ptr + L_base + rows_m[:, None] * stride_L + rows_k[None, :],
                mask=rows_m_mask[:, None] & (rows_k[None, :] < N),
                other=0.0,
            )
            head = tl.load(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                mask=rows_m_mask[:, None] & rhs_mask[None, :],
                other=0.0,
            )
            head = head - tl.dot(U_tile, x_block)
            tl.store(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                head,
                mask=rows_m_mask[:, None] & rhs_mask[None, :],
            )


# ---------------------------------------------------------------------------
# Blocked single-RHS kernel (N >= 64, N % 32 == 0)
# ---------------------------------------------------------------------------

@libentry()
@triton.jit
def cholesky_solve_single_rhs_blocked_kernel(
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
    upper: tl.constexpr,
):
    """Blocked single-RHS Cholesky solve for Ascend NPU.

    The diagonal block stays pre-scaled by the reciprocal diagonal, so each
    dependent pivot is a DSA row extraction followed by one vector rank-1
    update. Off-diagonal panels use matvec reductions instead of padding the
    single RHS into a matrix-multiply tile. Dispatch guarantees that N is a
    multiple of both BLOCK_K and BLOCK_M, so panel loads need no tail masks.
    """
    batch_pid = ext.program_id(0)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    k_offsets = tl.arange(0, BLOCK_K)
    m_offsets = tl.arange(0, BLOCK_M)

    # Forward blocked TRSM: L * Y = B or U^T * Y = B.
    for k in range(0, N, BLOCK_K):
        rows_k = k + k_offsets
        if k == 0:
            y_block = tl.load(B_ptr + B_base + rows_k * stride_B)
        else:
            y_block = tl.load(X_ptr + B_base + rows_k * stride_B)

        diag_block = tl.load(L_ptr + L_base + rows_k * stride_L + rows_k)
        inv_diag_block = 1.0 / diag_block
        inv_diag_block = inv_diag_block * (
            2.0 - diag_block * inv_diag_block
        )
        if dtype_flag == 1:
            inv_diag_block = inv_diag_block * (
                2.0 - diag_block * inv_diag_block
            )

        w = y_block[:, None] * inv_diag_block[:, None]
        for i in range(BLOCK_K):
            if upper:
                factor_col = tl.load(
                    L_ptr + L_base + (k + i) * stride_L + rows_k,
                    mask=k_offsets > i,
                    other=0.0,
                )
            else:
                factor_col = tl.load(
                    L_ptr + L_base + rows_k * stride_L + (k + i),
                    mask=k_offsets > i,
                    other=0.0,
                )
            w_i_2d = tle.dsa.extract_slice(w, (i, 0), (1, 1), (1, 1))
            tl.compile_hint(w_i_2d, "disable_bubble_up")
            w_i = tl.reshape(w_i_2d, (1,))
            w = w - (
                factor_col * inv_diag_block
            )[:, None] * w_i[None, :]

        w_vec = tl.reshape(w, (BLOCK_K,))
        tl.store(X_ptr + B_base + rows_k * stride_B, w_vec)

        for m in range(k + BLOCK_K, N, BLOCK_M):
            rows_m = m + m_offsets
            if upper:
                factor_panel = tl.load(
                    L_ptr
                    + L_base
                    + rows_k[:, None] * stride_L
                    + rows_m[None, :]
                )
                update = tl.sum(
                    factor_panel * w_vec[:, None], axis=0
                )
            else:
                factor_panel = tl.load(
                    L_ptr
                    + L_base
                    + rows_m[:, None] * stride_L
                    + rows_k[None, :]
                )
                update = tl.sum(
                    factor_panel * w_vec[None, :], axis=1
                )
            if k == 0:
                tail = tl.load(B_ptr + B_base + rows_m * stride_B)
            else:
                tail = tl.load(X_ptr + B_base + rows_m * stride_B)
            tl.store(X_ptr + B_base + rows_m * stride_B, tail - update)

    # Backward blocked TRSM: L^T * X = Y or U * X = Y.
    for k in range(N - BLOCK_K, -1, -BLOCK_K):
        rows_k = k + k_offsets
        x_block = tl.load(X_ptr + B_base + rows_k * stride_B)

        diag_block = tl.load(L_ptr + L_base + rows_k * stride_L + rows_k)
        inv_diag_block = 1.0 / diag_block
        inv_diag_block = inv_diag_block * (
            2.0 - diag_block * inv_diag_block
        )
        if dtype_flag == 1:
            inv_diag_block = inv_diag_block * (
                2.0 - diag_block * inv_diag_block
            )

        w = x_block[:, None] * inv_diag_block[:, None]
        for ii in range(BLOCK_K - 1, -1, -1):
            if upper:
                factor_row = tl.load(
                    L_ptr + L_base + rows_k * stride_L + (k + ii),
                    mask=k_offsets < ii,
                    other=0.0,
                )
            else:
                factor_row = tl.load(
                    L_ptr + L_base + (k + ii) * stride_L + rows_k,
                    mask=k_offsets < ii,
                    other=0.0,
                )
            w_i_2d = tle.dsa.extract_slice(w, (ii, 0), (1, 1), (1, 1))
            tl.compile_hint(w_i_2d, "disable_bubble_up")
            w_i = tl.reshape(w_i_2d, (1,))
            w = w - (
                factor_row * inv_diag_block
            )[:, None] * w_i[None, :]

        w_vec = tl.reshape(w, (BLOCK_K,))
        tl.store(X_ptr + B_base + rows_k * stride_B, w_vec)

        for m in range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            if upper:
                factor_panel = tl.load(
                    L_ptr
                    + L_base
                    + rows_m[:, None] * stride_L
                    + rows_k[None, :]
                )
            else:
                factor_panel = tl.load(
                    L_ptr
                    + L_base
                    + rows_k[None, :] * stride_L
                    + rows_m[:, None]
                )
            head = tl.load(X_ptr + B_base + rows_m * stride_B)
            update = tl.sum(factor_panel * w_vec[None, :], axis=1)
            tl.store(X_ptr + B_base + rows_m * stride_B, head - update)


# ---------------------------------------------------------------------------
# Single-RHS scalar kernel
# ---------------------------------------------------------------------------

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
    """Scalar Cholesky solve kernel for nrhs == 1 on Ascend NPU."""
    batch_pid = ext.program_id(0)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B

    # Phase 1: Forward substitution
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

    # Phase 2: Backward substitution
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


# ---------------------------------------------------------------------------
# Small gather kernel (GPU v2: pre-scale + extract_slice, N<=32, nrhs<=8)
# ---------------------------------------------------------------------------

@libentry()
@triton.jit
def cholesky_solve_small_gather_kernel(
    L_ptr, B_ptr, X_ptr,
    N: tl.constexpr, nrhs: tl.constexpr,
    batch_stride_L, batch_stride_B, stride_L, stride_B,
    BLOCK_N: tl.constexpr, BLOCK_RHS: tl.constexpr,
    dtype_flag: tl.constexpr, upper: tl.constexpr,
):
    """Small-N register-resident solve: pre-scale + tle.dsa.extract_slice."""
    batch_pid = ext.program_id(0)
    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_RHS)
    cols_mask = cols < nrhs
    rows_mask = rows < N

    b = tl.load(
        B_ptr + B_base + rows[:, None] * stride_B + cols[None, :],
        mask=rows_mask[:, None] & cols_mask[None, :], other=0.0,
    )
    diag = tl.load(
        L_ptr + L_base + rows * stride_L + rows,
        mask=rows_mask, other=1.0,
    )
    inv_diag = 1.0 / diag
    inv_diag = inv_diag * (2.0 - diag * inv_diag)
    if dtype_flag == 1:
        inv_diag = inv_diag * (2.0 - diag * inv_diag)

    w = b * inv_diag[:, None]
    for i in range(N):
        if upper:
            col_vals = tl.load(
                L_ptr + L_base + i * stride_L + rows,
                mask=(rows > i) & rows_mask, other=0.0,
            )
        else:
            col_vals = tl.load(
                L_ptr + L_base + rows * stride_L + i,
                mask=(rows > i) & rows_mask, other=0.0,
            )
        w_i_2d = tle.dsa.extract_slice(w, (i, 0), (1, BLOCK_RHS), (1, 1))
        w_i = tl.reshape(w_i_2d, (BLOCK_RHS,))
        w = w - (col_vals * inv_diag)[:, None] * w_i

    w = w * inv_diag[:, None]
    for i in range(N - 1, -1, -1):
        if upper:
            col_vals = tl.load(
                L_ptr + L_base + rows * stride_L + i,
                mask=rows < i, other=0.0,
            )
        else:
            col_vals = tl.load(
                L_ptr + L_base + i * stride_L + rows,
                mask=rows < i, other=0.0,
            )
        w_i_2d = tle.dsa.extract_slice(w, (i, 0), (1, BLOCK_RHS), (1, 1))
        w_i = tl.reshape(w_i_2d, (BLOCK_RHS,))
        w = w - (col_vals * inv_diag)[:, None] * w_i

    tl.store(
        X_ptr + B_base + rows[:, None] * stride_B + cols[None, :],
        w, mask=rows_mask[:, None] & cols_mask[None, :],
    )


def _can_use_small_gather_path(N, nrhs):
    return N <= 32 and nrhs <= 8


def _can_use_blocked_single_rhs_path(N, nrhs):
    return nrhs == 1 and N >= 64 and N % 32 == 0


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------

def _can_use_blocked_lower_path(upper, N, nrhs):
    return not upper and N >= 64 and N % 32 == 0 and nrhs >= 4


def _can_use_blocked_upper_path(upper, N, nrhs):
    return upper and N >= 64 and N % 32 == 0 and nrhs >= 4


def _get_ascend_tile_config(dtype):
    """Return tile sizes optimized for Ascend 910B AI Cores.

    Ascend 910B has ~192KB UB per AI Core. Standard 32x32 tiles fit well.
    """
    if dtype == torch.float64:
        return 16, 32, 8  # BLOCK_K, BLOCK_M, BLOCK_RHS
    return 32, 32, 16


def _get_ascend_warp_config(dtype):
    """Return warp/stage config for Ascend 910B."""
    if dtype == torch.float64:
        return {"num_warps": 4, "num_stages": 2}
    return {"num_warps": 4, "num_stages": 3}


def _get_ascend_single_rhs_config():
    """Return the conservative 910B blocked single-RHS launch config."""
    return {
        "BLOCK_K": 32,
        "BLOCK_M": 32,
        "num_warps": 4,
        "num_stages": 1,
    }


# ---------------------------------------------------------------------------
# Main dispatch function for Ascend
# ---------------------------------------------------------------------------

def cholesky_solve(B, L, upper=False):
    """Solves a system of linear equations with a symmetric positive-definite
    matrix using the Cholesky factorization on Ascend NPU.

    Dispatch rules (based on what works on Ascend 910B):
      - N <= 32, nrhs <= 8          → register-resident small gather kernel
      - nrhs == 1, regular N >= 64  → blocked single-RHS matvec kernel
      - other nrhs == 1             → scalar single-RHS kernel
      - nrhs >= 4, N >= 64, N%32==0 → blocked kernel (tl.dot accelerated)
      - otherwise                   → scalar vectorized kernel

    Computes X such that A @ X = B, where A = L @ L^T (or A = U^T @ U if
    upper=True) and L (or U) is the Cholesky factor of A.

    Args:
        B: right-hand side tensor of shape (*, N, nrhs)
        L: Cholesky factor of shape (*, N, N), lower-triangular unless upper=True
        upper: if True, the Cholesky factor is upper-triangular

    Returns:
        X: solution tensor of shape (*, N, nrhs)
    """
    logger.debug("GEMS_ASCEND CHOLESKY_SOLVE")
    if L.dtype not in (torch.float32,):
        raise ValueError("cholesky_solve on Ascend only supports float32")
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

    batch_size = 1
    for dim in batch_shape:
        batch_size *= dim

    # torch.linalg.cholesky commonly returns a transpose-contiguous factor.
    # Reinterpret that storage through an mT view and flip the triangular
    # orientation instead of materializing an F-to-C layout conversion.
    # The large-batch small-N lower single-RHS kernel is an exception: on
    # Ascend, its row-contiguous lower specialization is faster even after
    # paying for the layout conversion. Small batches retain the zero-copy
    # effective-upper path because the conversion cost is not recovered.
    keep_small_batched_lower = (
        not upper
        and batch_size >= 64
        and nrhs == 1
        and _can_use_small_gather_path(N, nrhs)
    )
    effective_upper = upper
    if L.is_contiguous():
        pass
    elif L.mT.is_contiguous() and not keep_small_batched_lower:
        L = L.mT
        effective_upper = not upper
    else:
        L = L.contiguous()

    L = L.expand(batch_shape + L_shape[-2:])
    B = B.expand(batch_shape + B_shape[-2:])

    # Broadcasted batch dimensions may introduce zero strides and still need
    # materialization before batch flattening. The common non-broadcast path
    # remains a zero-copy view after the layout normalization above.
    if not L.is_contiguous():
        L = L.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    X = torch.empty_like(B)

    L_kernel = L.reshape(-1, N, N)
    B_kernel = B.reshape(-1, N, nrhs)
    X_kernel = X.reshape(-1, N, nrhs)

    stride_L = L_kernel.stride(1)
    stride_B = B_kernel.stride(1)
    batch_stride_L = L_kernel.stride(0)
    batch_stride_B = B_kernel.stride(0)

    dtype_flag = 0 if B.dtype == torch.float32 else 1
    device = B.device

    with torch_device_fn.device(device):
        # Path 1: small gather kernel (GPU v2, N<=32, nrhs<=8)
        if _can_use_small_gather_path(N, nrhs):
            block_n = triton.next_power_of_2(N)
            block_rhs = triton.next_power_of_2(nrhs)
            cholesky_solve_small_gather_kernel[(batch_size,)](
                L_kernel, B_kernel, X_kernel, N, nrhs,
                batch_stride_L, batch_stride_B, stride_L, stride_B,
                BLOCK_N=block_n, BLOCK_RHS=block_rhs,
                dtype_flag=0, upper=effective_upper,
                num_warps=2, num_stages=1,
            )
        # Path 2: regular medium/large single RHS → blocked matvec kernel
        elif _can_use_blocked_single_rhs_path(N, nrhs):
            single_rhs_config = _get_ascend_single_rhs_config()
            cholesky_solve_single_rhs_blocked_kernel[(batch_size,)](
                L_kernel, B_kernel, X_kernel, N,
                batch_stride_L, batch_stride_B, stride_L, stride_B,
                dtype_flag=dtype_flag, upper=effective_upper,
                **single_rhs_config,
            )
        # Path 3: irregular single RHS → scalar kernel
        elif nrhs == 1:
            cholesky_solve_single_rhs_kernel[(batch_size,)](
                L_kernel, B_kernel, X_kernel, N,
                batch_stride_L, batch_stride_B, stride_L, stride_B,
                dtype_flag=dtype_flag, upper=effective_upper,
            )
        # Path 4: blocked multi-RHS (tl.dot accelerated, nrhs>=4 required for alignment)
        elif not effective_upper and N >= 64 and N % 32 == 0 and nrhs >= 4:
            blk_k, blk_m, blk_rhs = _get_ascend_tile_config(B.dtype)
            warp = _get_ascend_warp_config(B.dtype)
            grid = (batch_size, triton.cdiv(nrhs, blk_rhs))
            cholesky_solve_blocked_lower_kernel[grid](
                L_kernel, B_kernel, X_kernel, N, nrhs,
                batch_stride_L, batch_stride_B, stride_L, stride_B,
                BLOCK_K=blk_k, BLOCK_M=blk_m, BLOCK_RHS=blk_rhs,
                dtype_flag=0, PRELOAD_DIAG=False, **warp,
            )
        elif effective_upper and N >= 64 and N % 32 == 0 and nrhs >= 4:
            blk_k, blk_m, blk_rhs = _get_ascend_tile_config(B.dtype)
            warp = _get_ascend_warp_config(B.dtype)
            grid = (batch_size, triton.cdiv(nrhs, blk_rhs))
            cholesky_solve_blocked_upper_kernel[grid](
                L_kernel, B_kernel, X_kernel, N, nrhs,
                batch_stride_L, batch_stride_B, stride_L, stride_B,
                BLOCK_K=blk_k, BLOCK_M=blk_m, BLOCK_RHS=blk_rhs,
                dtype_flag=dtype_flag, **warp,
            )
        # Path 5: general scalar vectorized kernel (any N, any nrhs)
        else:
            blk_rhs = min(nrhs, 16)
            grid = (batch_size, triton.cdiv(nrhs, blk_rhs))
            cholesky_solve_kernel[grid](
                L_kernel, B_kernel, X_kernel, N, nrhs,
                batch_stride_L, batch_stride_B, stride_L, stride_B,
                BLOCK_RHS=blk_rhs,
                dtype_flag=dtype_flag,
                upper=effective_upper,
            )

    return X
