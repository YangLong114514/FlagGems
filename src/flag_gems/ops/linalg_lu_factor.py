import logging
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

LinalgLUFactorResult = namedtuple("LinalgLUFactorResult", ["LU", "pivots"])

_LU_FACTOR_BLOCK_MAX = 64
_LU_FACTOR_PANEL = 16
_LU_FACTOR_TILE_B = 16
_LU_FACTOR_TILE_M = 64
_LU_FACTOR_TILE_N = 128
_LU_FACTOR_FUSED_TILE_N = 16
_LU_FACTOR_BLOCKED_M_MAX = 4096
_LU_FACTOR_BLOCKED_N_MAX = 4096


@libentry()
@triton.jit
def _linalg_lu_factor_kernel(
    A,
    LU,
    PIVOTS,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PIVOT: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    offsets = pid * M * N + rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    work = tl.load(A + offsets, mask=mask, other=0.0).to(tl.float32)

    for j_ind in tl.range(0, K):
        if PIVOT:
            col_j = tl.sum(tl.where(cols[:, None] == j_ind, tl.trans(work), 0.0), axis=0)
            abs_col = tl.abs(col_j)
            abs_col = tl.where(rows < j_ind, -1.0, abs_col)
            abs_col = tl.where(rows < M, abs_col, -1.0)
            pivot_val = tl.max(abs_col, axis=0)
            pivot_row = tl.min(tl.where(abs_col == pivot_val, rows, BLOCK_M), axis=0)

            row_j = tl.sum(tl.where(rows[:, None] == j_ind, work, 0.0), axis=0)
            row_p = tl.sum(tl.where(rows[:, None] == pivot_row, work, 0.0), axis=0)
            col_mask = cols[None, :] < N
            work = tl.where((rows[:, None] == j_ind) & col_mask, row_p, work)
            work = tl.where(
                (rows[:, None] == pivot_row) & col_mask, row_j, work
            )
            tl.store(PIVOTS + pid * K + j_ind, pivot_row + 1)
        else:
            tl.store(PIVOTS + pid * K + j_ind, j_ind + 1)

        pivot = tl.sum(
            tl.sum(
                tl.where((rows[:, None] == j_ind) & (cols[None, :] == j_ind), work, 0.0),
                axis=0,
            ),
            axis=0,
        )

        pivot_row_vals = tl.sum(tl.where(rows[:, None] == j_ind, work, 0.0), axis=0)
        active_cols = cols > j_ind
        work = tl.where(
            (rows[:, None] == j_ind) & active_cols[None, :], pivot_row_vals, work
        )

        col_vals = tl.sum(tl.where(cols[:, None] == j_ind, tl.trans(work), 0.0), axis=0)
        multipliers = tl.where(rows > j_ind, col_vals / pivot, col_vals)
        work = tl.where(
            (rows[:, None] > j_ind) & (cols[None, :] == j_ind), multipliers[:, None], work
        )

        l_col = tl.sum(tl.where(cols[:, None] == j_ind, tl.trans(work), 0.0), axis=0)
        u_row = tl.sum(tl.where(rows[:, None] == j_ind, work, 0.0), axis=0)
        update_mask = (rows[:, None] > j_ind) & (cols[None, :] > j_ind)
        work = tl.where(update_mask, work - l_col[:, None] * u_row[None, :], work)

    tl.store(LU + offsets, work, mask=mask)


@libentry()
@triton.jit
def _lu_factor_panel_no_pivot_kernel(
    LU,
    PIVOTS,
    K0: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    PANEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    bcols = tl.arange(0, BLOCK_B)
    cols = K0 + bcols

    offsets = pid * M * N + rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (bcols[None, :] < PANEL)
    panel = tl.load(LU + offsets, mask=mask, other=0.0).to(tl.float32)

    for jj in tl.static_range(0, BLOCK_B):
        if jj < PANEL:
            j = K0 + jj
            pivot = tl.sum(
                tl.sum(
                    tl.where(
                        (rows[:, None] == j) & (bcols[None, :] == jj), panel, 0.0
                    ),
                    axis=0,
                ),
                axis=0,
            )
            col_vals = tl.sum(
                tl.where(bcols[:, None] == jj, tl.trans(panel), 0.0), axis=0
            )
            col_vals = tl.where(rows > j, col_vals / pivot, col_vals)
            panel = tl.where(
                (rows[:, None] > j) & (bcols[None, :] == jj),
                col_vals[:, None],
                panel,
            )

            l_col = tl.sum(
                tl.where(bcols[:, None] == jj, tl.trans(panel), 0.0), axis=0
            )
            u_row = tl.sum(tl.where(rows[:, None] == j, panel, 0.0), axis=0)
            update_mask = (rows[:, None] > j) & (bcols[None, :] > jj)
            panel = tl.where(
                update_mask, panel - l_col[:, None] * u_row[None, :], panel
            )
            tl.store(PIVOTS + pid * K + j, j + 1)

    tl.store(LU + offsets, panel, mask=mask)


@libentry()
@triton.jit
def _lu_factor_panel_kernel(
    LU,
    PIVOTS,
    K0: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    PANEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_B: tl.constexpr,
    LEFT_BLOCK_N: tl.constexpr,
    APPLY_LEFT: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    bcols = tl.arange(0, BLOCK_B)
    cols = K0 + bcols
    left_cols = tl.arange(0, LEFT_BLOCK_N)

    offsets = pid * M * N + rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (bcols[None, :] < PANEL)
    panel = tl.load(LU + offsets, mask=mask, other=0.0).to(tl.float32)

    for jj in tl.static_range(0, BLOCK_B):
        if jj < PANEL:
            j = K0 + jj

            # pivot search
            col_vals = tl.sum(
                tl.where(bcols[:, None] == jj, tl.trans(panel), 0.0), axis=0
            )
            abs_col = tl.abs(col_vals)
            abs_col = tl.where(rows < j, -1.0, abs_col)
            abs_col = tl.where(rows < M, abs_col, -1.0)
            pivot_val = tl.max(abs_col, axis=0)
            pivot_row = tl.min(tl.where(abs_col == pivot_val, rows, BLOCK_M), axis=0)

            # swap rows in panel
            row_j = tl.sum(tl.where(rows[:, None] == j, panel, 0.0), axis=0)
            row_p = tl.sum(tl.where(rows[:, None] == pivot_row, panel, 0.0), axis=0)
            panel = tl.where((rows[:, None] == j) & mask, row_p[None, :], panel)
            panel = tl.where(
                (rows[:, None] == pivot_row) & mask, row_j[None, :], panel
            )
            tl.store(PIVOTS + pid * K + j, pivot_row + 1)

            if APPLY_LEFT:
                left_mask = left_cols < K0
                row_j_left_offsets = pid * M * N + j * N + left_cols
                row_p_left_offsets = pid * M * N + pivot_row * N + left_cols
                row_j_left = tl.load(
                    LU + row_j_left_offsets, mask=left_mask, other=0.0
                )
                row_p_left = tl.load(
                    LU + row_p_left_offsets, mask=left_mask, other=0.0
                )
                tl.store(LU + row_j_left_offsets, row_p_left, mask=left_mask)
                tl.store(LU + row_p_left_offsets, row_j_left, mask=left_mask)

            # pivot value after swap
            pivot = tl.sum(
                tl.sum(
                    tl.where(
                        (rows[:, None] == j) & (bcols[None, :] == jj), panel, 0.0
                    ),
                    axis=0,
                ),
                axis=0,
            )

            # scale column below diagonal
            col_vals = tl.sum(
                tl.where(bcols[:, None] == jj, tl.trans(panel), 0.0), axis=0
            )
            col_vals = tl.where(rows > j, col_vals / pivot, col_vals)
            panel = tl.where(
                (rows[:, None] > j) & (bcols[None, :] == jj),
                col_vals[:, None],
                panel,
            )

            # rank-1 update on trailing sub-panel
            l_col = tl.sum(
                tl.where(bcols[:, None] == jj, tl.trans(panel), 0.0), axis=0
            )
            u_row = tl.sum(tl.where(rows[:, None] == j, panel, 0.0), axis=0)
            update_mask = (rows[:, None] > j) & (bcols[None, :] > jj)
            panel = tl.where(
                update_mask, panel - l_col[:, None] * u_row[None, :], panel
            )

    tl.store(LU + offsets, panel, mask=mask)


@libentry()
@triton.jit
def _lu_factor_apply_panel_pivots_kernel(
    LU,
    PIVOTS,
    K0: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    PANEL: tl.constexpr,
    COL_START: tl.constexpr,
    NUM_COLS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    cols = COL_START + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < COL_START + NUM_COLS

    for jj in tl.static_range(0, BLOCK_B):
        if jj < PANEL:
            j = K0 + jj
            pivot_row = tl.load(PIVOTS + pid_b * K + j) - 1
            row_j_offsets = pid_b * M * N + j * N + cols
            row_p_offsets = pid_b * M * N + pivot_row * N + cols
            row_j = tl.load(LU + row_j_offsets, mask=col_mask, other=0.0)
            row_p = tl.load(LU + row_p_offsets, mask=col_mask, other=0.0)
            tl.store(LU + row_j_offsets, row_p, mask=col_mask)
            tl.store(LU + row_p_offsets, row_j, mask=col_mask)


@libentry()
@triton.jit
def _lu_factor_solve_block_row_no_pivot_kernel(
    LU,
    K0: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    PANEL: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Column-oriented forward substitution: solve L * X = B for X in-place.

    L is unit lower triangular (PANEL x PANEL), stored in LU[K0:K0+PANEL, K0:K0+PANEL].
    B is (PANEL x trailing_N), stored in LU[K0:K0+PANEL, K0+PANEL:N] (= vals).
    The result X overwrites B.

    Column-oriented algorithm (O(PANEL) iterations, no inner loop):
      For j = 0..PANEL-1:
        row_j = vals[j, :]           # already solved from prior rank-1 updates
        l_col = L[j+1:, j]          # column j of L below diagonal
        vals[j+1:, :] -= l_col * row_j  # rank-1 update
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    brows = tl.arange(0, BLOCK_B)
    cols = K0 + PANEL + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = K0 + brows

    offsets = pid_b * M * N + rows[:, None] * N + cols[None, :]
    mask = (brows[:, None] < PANEL) & (cols[None, :] < N)
    vals = tl.load(LU + offsets, mask=mask, other=0.0).to(tl.float32)

    for jj in tl.static_range(0, BLOCK_B):
        if jj < PANEL:
            # Extract row jj — already fully solved by prior rank-1 updates
            row_j = tl.sum(tl.where(brows[:, None] == jj, vals, 0.0), axis=0)

            # Load column jj of L from the panel (L is unit lower triangular,
            # stored in LU[K0:K0+PANEL, K0:K0+PANEL])
            l_col_offsets = pid_b * M * N + (K0 + brows) * N + (K0 + jj)
            l_col = tl.load(LU + l_col_offsets, mask=brows < PANEL, other=0.0)
            # Zero out L[jj, jj] (=1) and above-diagonal (we only need L[jj+1:, jj])
            l_col = tl.where(brows <= jj, 0.0, l_col)

            # Rank-1 update: vals[jj+1:, :] -= l_col[jj+1:] * row_j
            vals = tl.where(
                brows[:, None] > jj,
                vals - l_col[:, None] * row_j[None, :],
                vals,
            )

    tl.store(LU + offsets, vals, mask=mask)


@libentry()
@triton.jit
def _lu_factor_swap_right_and_solve_kernel(
    LU,
    PIVOTS,
    K0: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    PANEL: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Apply panel pivots to trailing columns and solve for U rows in one pass.

    Merges swap_right (apply panel pivots to trailing columns) and solve
    (forward substitution for U rows) into a single kernel to reduce kernel
    launch overhead.
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    brows = tl.arange(0, BLOCK_B)
    cols = K0 + PANEL + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = K0 + brows

    offsets = pid_b * M * N + rows[:, None] * N + cols[None, :]
    mask = (brows[:, None] < PANEL) & (cols[None, :] < N)
    vals = tl.load(LU + offsets, mask=mask, other=0.0).to(tl.float32)

    # Step 1: Apply row swaps from panel pivots.
    # For each panel pivot, swap row j with pivot_row in both vals (register copy)
    # and global memory. When pivot_row falls within the panel, vals must also
    # be updated so later swap/solve iterations see the correct data.
    col_mask = cols[None, :] < N
    for jj in tl.static_range(0, BLOCK_B):
        if jj < PANEL:
            j = K0 + jj
            pivot_row = tl.load(PIVOTS + pid_b * K + j) - 1
            row_j = tl.sum(tl.where(brows[:, None] == jj, vals, 0.0), axis=0)

            # Load pivot row from global memory
            row_p_offsets = pid_b * M * N + pivot_row * N + cols
            row_p = tl.load(LU + row_p_offsets, mask=cols < N, other=0.0)

            # Write pivot row into vals at row jj
            vals = tl.where(
                (brows[:, None] == jj) & col_mask, row_p[None, :], vals
            )

            # If pivot_row is within the loaded panel block, also update vals
            # so later iterations see the swapped data.
            rel_pivot = pivot_row - K0
            vals = tl.where(
                (brows[:, None] == rel_pivot) & col_mask, row_j[None, :], vals
            )

            # Write old row jj to pivot_row position in global memory
            tl.store(LU + row_p_offsets, row_j, mask=cols < N)

    # Step 2: Column-oriented forward substitution to solve for U rows.
    for jj in tl.static_range(0, BLOCK_B):
        if jj < PANEL:
            row_j = tl.sum(tl.where(brows[:, None] == jj, vals, 0.0), axis=0)

            l_col_offsets = pid_b * M * N + (K0 + brows) * N + (K0 + jj)
            l_col = tl.load(LU + l_col_offsets, mask=brows < PANEL, other=0.0)
            l_col = tl.where(brows <= jj, 0.0, l_col)

            vals = tl.where(
                brows[:, None] > jj,
                vals - l_col[:, None] * row_j[None, :],
                vals,
            )

    tl.store(LU + offsets, vals, mask=mask)


@libentry()
@triton.jit
def _lu_factor_panel_swap_right_and_solve_kernel(
    LU,
    PIVOTS,
    K0: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    PANEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_B: tl.constexpr,
    RIGHT_BLOCK_N: tl.constexpr,
    LEFT_BLOCK_N: tl.constexpr,
    APPLY_LEFT: tl.constexpr,
):
    """Factor a pivoted panel and solve its right block in one program.

    This specialization is used only when all trailing columns fit in one
    BLOCK_N tile. Wider matrices still use the multi-tile panel + swap/solve
    path because Triton programs cannot synchronize with each other inside a
    single kernel launch.
    """
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    bcols = tl.arange(0, BLOCK_B)
    panel_cols = K0 + bcols
    left_cols = tl.arange(0, LEFT_BLOCK_N)

    panel_offsets = pid * M * N + rows[:, None] * N + panel_cols[None, :]
    panel_mask = (rows[:, None] < M) & (bcols[None, :] < PANEL)
    panel_vals = tl.load(LU + panel_offsets, mask=panel_mask, other=0.0).to(
        tl.float32
    )

    brows = tl.arange(0, BLOCK_B)
    right_cols = K0 + PANEL + tl.arange(0, RIGHT_BLOCK_N)
    right_rows = K0 + brows
    right_offsets = pid * M * N + right_rows[:, None] * N + right_cols[None, :]
    right_mask = (brows[:, None] < PANEL) & (right_cols[None, :] < N)
    right_vals = tl.load(LU + right_offsets, mask=right_mask, other=0.0).to(
        tl.float32
    )

    for jj in tl.static_range(0, BLOCK_B):
        if jj < PANEL:
            j = K0 + jj

            col_vals = tl.sum(
                tl.where(bcols[:, None] == jj, tl.trans(panel_vals), 0.0), axis=0
            )
            abs_col = tl.abs(col_vals)
            abs_col = tl.where(rows < j, -1.0, abs_col)
            abs_col = tl.where(rows < M, abs_col, -1.0)
            pivot_val = tl.max(abs_col, axis=0)
            pivot_row = tl.min(tl.where(abs_col == pivot_val, rows, BLOCK_M), axis=0)

            row_j_panel = tl.sum(
                tl.where(rows[:, None] == j, panel_vals, 0.0), axis=0
            )
            row_p_panel = tl.sum(
                tl.where(rows[:, None] == pivot_row, panel_vals, 0.0), axis=0
            )
            panel_vals = tl.where(
                (rows[:, None] == j) & panel_mask, row_p_panel[None, :], panel_vals
            )
            panel_vals = tl.where(
                (rows[:, None] == pivot_row) & panel_mask,
                row_j_panel[None, :],
                panel_vals,
            )
            tl.store(PIVOTS + pid * K + j, pivot_row + 1)

            if APPLY_LEFT:
                left_mask = left_cols < K0
                row_j_left_offsets = pid * M * N + j * N + left_cols
                row_p_left_offsets = pid * M * N + pivot_row * N + left_cols
                row_j_left = tl.load(
                    LU + row_j_left_offsets, mask=left_mask, other=0.0
                )
                row_p_left = tl.load(
                    LU + row_p_left_offsets, mask=left_mask, other=0.0
                )
                tl.store(LU + row_j_left_offsets, row_p_left, mask=left_mask)
                tl.store(LU + row_p_left_offsets, row_j_left, mask=left_mask)

            row_j_right = tl.sum(
                tl.where(brows[:, None] == jj, right_vals, 0.0), axis=0
            )
            row_p_right_offsets = pid * M * N + pivot_row * N + right_cols
            row_p_right = tl.load(
                LU + row_p_right_offsets, mask=right_cols < N, other=0.0
            )
            right_vals = tl.where(
                (brows[:, None] == jj) & right_mask,
                row_p_right[None, :],
                right_vals,
            )
            rel_pivot = pivot_row - K0
            right_vals = tl.where(
                (brows[:, None] == rel_pivot) & right_mask,
                row_j_right[None, :],
                right_vals,
            )
            tl.store(LU + row_p_right_offsets, row_j_right, mask=right_cols < N)

            pivot = tl.sum(
                tl.sum(
                    tl.where(
                        (rows[:, None] == j) & (bcols[None, :] == jj),
                        panel_vals,
                        0.0,
                    ),
                    axis=0,
                ),
                axis=0,
            )

            col_vals = tl.sum(
                tl.where(bcols[:, None] == jj, tl.trans(panel_vals), 0.0), axis=0
            )
            col_vals = tl.where(rows > j, col_vals / pivot, col_vals)
            panel_vals = tl.where(
                (rows[:, None] > j) & (bcols[None, :] == jj),
                col_vals[:, None],
                panel_vals,
            )

            l_col = tl.sum(
                tl.where(bcols[:, None] == jj, tl.trans(panel_vals), 0.0), axis=0
            )
            u_row = tl.sum(tl.where(rows[:, None] == j, panel_vals, 0.0), axis=0)
            update_mask = (rows[:, None] > j) & (bcols[None, :] > jj)
            panel_vals = tl.where(
                update_mask, panel_vals - l_col[:, None] * u_row[None, :], panel_vals
            )

    for jj in tl.static_range(0, BLOCK_B):
        if jj < PANEL:
            row_j = tl.sum(tl.where(brows[:, None] == jj, right_vals, 0.0), axis=0)

            l_col_all = tl.sum(
                tl.where(bcols[:, None] == jj, tl.trans(panel_vals), 0.0), axis=0
            )
            l_col = tl.sum(
                tl.where(rows[:, None] == right_rows[None, :], l_col_all[:, None], 0.0),
                axis=0,
            )
            l_col = tl.where(brows <= jj, 0.0, l_col)
            right_vals = tl.where(
                brows[:, None] > jj,
                right_vals - l_col[:, None] * row_j[None, :],
                right_vals,
            )

    tl.store(LU + panel_offsets, panel_vals, mask=panel_mask)
    tl.store(LU + right_offsets, right_vals, mask=right_mask)


@libentry()
@triton.jit
def _lu_factor_trailing_update_no_pivot_kernel(
    LU,
    K0: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    PANEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    rows = K0 + PANEL + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = K0 + PANEL + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bidx = tl.arange(0, BLOCK_B)

    tile_offsets = pid_b * M * N + rows[:, None] * N + cols[None, :]
    tile_mask = (rows[:, None] < M) & (cols[None, :] < N)
    tile = tl.load(LU + tile_offsets, mask=tile_mask, other=0.0).to(tl.float32)

    l_offsets = pid_b * M * N + rows[:, None] * N + (K0 + bidx[None, :])
    u_offsets = pid_b * M * N + (K0 + bidx[:, None]) * N + cols[None, :]
    l_mask = (rows[:, None] < M) & (bidx[None, :] < PANEL)
    u_mask = (bidx[:, None] < PANEL) & (cols[None, :] < N)
    l_vals = tl.load(LU + l_offsets, mask=l_mask, other=0.0)
    u_vals = tl.load(LU + u_offsets, mask=u_mask, other=0.0)
    update = tl.dot(l_vals, u_vals, input_precision="tf32")

    tl.store(LU + tile_offsets, tile - update, mask=tile_mask)


def _linalg_lu_factor_check(input, pivot):
    if input.dim() < 2:
        raise RuntimeError(
            "torch.linalg.lu_factor: Expected input to have at least 2 dimensions, "
            f"got {input.dim()}"
        )
    if not input.is_cuda:
        raise NotImplementedError(
            "FlagGems linalg_lu_factor currently supports CUDA tensors only"
        )
    if input.dtype != torch.float32:
        raise NotImplementedError(
            "FlagGems linalg_lu_factor currently supports float32 only, "
            f"got {input.dtype}"
        )
    m, n = input.shape[-2], input.shape[-1]
    if m == 0 or n == 0:
        raise NotImplementedError(
            "FlagGems linalg_lu_factor currently does not support empty matrices"
        )
    if pivot not in (True, False):
        raise TypeError(f"pivot must be a bool, got {type(pivot)}")


def _can_use_triton(input):
    m, n = input.shape[-2], input.shape[-1]
    return m <= _LU_FACTOR_BLOCK_MAX and n <= _LU_FACTOR_BLOCK_MAX


def _can_use_blocked_triton(input):
    m, n = input.shape[-2], input.shape[-1]
    return (
        m <= _LU_FACTOR_BLOCKED_M_MAX
        and n <= _LU_FACTOR_BLOCKED_N_MAX
    )


def _blocked_lu_factor(input_contiguous, pivot):
    batch_shape = input_contiguous.shape[:-2]
    m, n = input_contiguous.shape[-2], input_contiguous.shape[-1]
    k = min(m, n)
    batch = input_contiguous.numel() // (m * n)

    lu = input_contiguous.clone()
    pivots = torch.empty(
        (*batch_shape, k), device=input_contiguous.device, dtype=torch.int32
    )

    block_m = triton.next_power_of_2(m)
    panel_block = triton.next_power_of_2(_LU_FACTOR_PANEL)
    apply_left_in_panel = pivot and k <= _LU_FACTOR_TILE_N

    with torch_device_fn.device(input_contiguous.device):
        for k0 in range(0, k, _LU_FACTOR_PANEL):
            panel = min(_LU_FACTOR_PANEL, k - k0)
            trailing_n = n - k0 - panel
            trailing_m = m - k0 - panel
            use_fused_panel_solve = (
                pivot and trailing_n > 0 and trailing_n <= _LU_FACTOR_FUSED_TILE_N
            )
            apply_left_this_panel = apply_left_in_panel and k0 > 0
            left_block = triton.next_power_of_2(k0) if apply_left_this_panel else 1

            if use_fused_panel_solve:
                _lu_factor_panel_swap_right_and_solve_kernel[(batch,)](
                    lu,
                    pivots,
                    k0,
                    m,
                    n,
                    k,
                    panel,
                    block_m,
                    panel_block,
                    _LU_FACTOR_FUSED_TILE_N,
                    left_block,
                    apply_left_this_panel,
                    num_warps=4,
                )
            elif pivot:
                _lu_factor_panel_kernel[(batch,)](
                    lu,
                    pivots,
                    k0,
                    m,
                    n,
                    k,
                    panel,
                    block_m,
                    panel_block,
                    left_block,
                    apply_left_this_panel,
                )

            else:
                _lu_factor_panel_no_pivot_kernel[(batch,)](
                    lu,
                    pivots,
                    k0,
                    m,
                    n,
                    k,
                    panel,
                    block_m,
                    panel_block,
                )

            if trailing_n > 0 and not use_fused_panel_solve:
                if pivot:
                    grid_combined = (
                        triton.cdiv(trailing_n, _LU_FACTOR_TILE_N),
                        batch,
                    )
                    _lu_factor_swap_right_and_solve_kernel[grid_combined](
                        lu,
                        pivots,
                        k0,
                        m,
                        n,
                        k,
                        panel,
                        panel_block,
                        _LU_FACTOR_TILE_N,
                        num_warps=4,
                    )
                else:
                    grid_solve = (triton.cdiv(trailing_n, _LU_FACTOR_TILE_N), batch)
                    _lu_factor_solve_block_row_no_pivot_kernel[grid_solve](
                        lu,
                        k0,
                        m,
                        n,
                        panel,
                        panel_block,
                        _LU_FACTOR_TILE_N,
                        num_warps=4,
                    )

            if trailing_m > 0 and trailing_n > 0:
                grid_update = (
                    triton.cdiv(trailing_m, _LU_FACTOR_TILE_M),
                    triton.cdiv(trailing_n, _LU_FACTOR_TILE_N),
                    batch,
                )
                _lu_factor_trailing_update_no_pivot_kernel[grid_update](
                    lu,
                    k0,
                    m,
                    n,
                    panel,
                    _LU_FACTOR_TILE_M,
                    _LU_FACTOR_TILE_N,
                    panel_block,
                    num_warps=4,
                )

        # Final pass: apply all pivots to the left columns (L factors)
        if pivot and not apply_left_in_panel:
            for k0 in range(_LU_FACTOR_PANEL, k, _LU_FACTOR_PANEL):
                panel = min(_LU_FACTOR_PANEL, k - k0)
                grid_swap_left = (triton.cdiv(k0, _LU_FACTOR_TILE_N), batch)
                _lu_factor_apply_panel_pivots_kernel[grid_swap_left](
                    lu,
                    pivots,
                    k0,
                    m,
                    n,
                    k,
                    panel,
                    0,
                    k0,
                    panel_block,
                    _LU_FACTOR_TILE_N,
                    num_warps=4,
                )

    return LinalgLUFactorResult(lu, pivots)


def linalg_lu_factor(input, *, pivot=True):
    logger.debug("GEMS LINALG_LU_FACTOR")
    _linalg_lu_factor_check(input, pivot)

    input_contiguous = input.contiguous()

    if not _can_use_triton(input_contiguous):
        if _can_use_blocked_triton(input_contiguous):
            logger.debug("GEMS LINALG_LU_FACTOR blocked Triton path")
            return _blocked_lu_factor(input_contiguous, pivot)
        raise NotImplementedError(
            "FlagGems linalg_lu_factor Triton large-shape path is not available "
            "for this input"
        )

    batch_shape = input_contiguous.shape[:-2]
    m, n = input_contiguous.shape[-2], input_contiguous.shape[-1]
    k = min(m, n)
    batch = input_contiguous.numel() // (m * n)

    lu = torch.empty_like(input_contiguous)
    pivots = torch.empty((*batch_shape, k), device=input.device, dtype=torch.int32)

    with torch_device_fn.device(input.device):
        _linalg_lu_factor_kernel[(batch,)](
            input_contiguous,
            lu,
            pivots,
            m,
            n,
            k,
            triton.next_power_of_2(m),
            triton.next_power_of_2(n),
            pivot,
        )
    return LinalgLUFactorResult(lu, pivots)


def _resolve_linalg_lu_factor_out_args(LU, pivots, out):
    if out is not None:
        if LU is not None or pivots is not None:
            raise TypeError("linalg_lu_factor(): out and LU/pivots cannot both be set")
        if len(out) != 2:
            raise TypeError(
                f"linalg_lu_factor(): out must be a tuple of 2 tensors, got {len(out)}"
            )
        return out
    if LU is None or pivots is None:
        raise TypeError(
            "linalg_lu_factor(): LU and pivots must both be provided for out variant"
        )
    return LU, pivots


def linalg_lu_factor_out(input, *, pivot=True, LU=None, pivots=None, out=None):
    logger.debug("GEMS LINALG_LU_FACTOR.OUT")
    lu_out, pivots_out = _resolve_linalg_lu_factor_out_args(LU, pivots, out)
    lu, piv = linalg_lu_factor(input, pivot=pivot)

    lu_out.resize_(lu.shape)
    pivots_out.resize_(piv.shape)
    lu_out.copy_(lu)
    pivots_out.copy_(piv)
    return LinalgLUFactorResult(lu_out, pivots_out)
