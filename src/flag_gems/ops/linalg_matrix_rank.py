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

_FUSED_JACOBI_MAX_K = 64
_FUSED_JACOBI_MAX_K_FP64 = 32
_FUSED_JACOBI_MAX_ROWS = 256
# Wider fused tiles (k past the narrow cap) hold both rotation tiles in
# registers at once, so they are only used when the row tile stays small
# enough to avoid register spills.
_FUSED_JACOBI_WIDE_MAX_ROWS = 128
_NATIVE_FP64_JACOBI_MAX_K = 64
_BLOCKED_JACOBI_MAX_K = 512
_JACOBI_MAX_ROWS = 1024
# Hermitian matrices at or above this order use the non-iterative
# tridiagonalization + Sturm-count path instead of Jacobi sweeps. The fp32
# threshold is one higher: at k == 32 the fused fp32 Jacobi kernel is still
# faster than the tridiagonalization path.
_HERM_TRIDIAG_MIN_K_FP64 = 32
_HERM_TRIDIAG_MIN_K_FP32 = 33
# Hermitian matrices at or above this order use the blocked (BLAS3)
# tridiagonalization kernel instead of the unblocked one.
_HERM_TRIDIAG_BLOCKED_MIN_K = 256
# Pairs of columns processed by one program of the multi-block sweep kernel.
_JACOBI_PAIRS_PER_BLOCK = 4
# Upper bound on the residency of the sweep kernel grid. Even at the
# architectural register ceiling (255 regs/thread, 128 threads/block) two
# blocks always fit on one SM, so a grid within ``2 * sm_count`` blocks is
# guaranteed to be co-resident and the software grid barrier cannot deadlock.
_RESIDENT_BLOCKS_PER_SM = 2

_SM_COUNT_CACHE = {}


def _sm_count(device):
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    count = _SM_COUNT_CACHE.get(index)
    if count is None:
        count = torch.cuda.get_device_properties(index).multi_processor_count
        _SM_COUNT_CACHE[index] = count
    return count


def _jacobi_sweeps(k, is_fp64):
    # Numerical rank is more sensitive to residual column correlation than
    # returning approximate singular values. Keep a few more sweeps than the
    # general SVD path, especially for float64. These are worst-case caps:
    # the kernels stop as soon as the Weyl bound on the residual off-diagonal
    # energy proves that no singular value can cross the rank threshold.
    if is_fp64:
        if k <= 16:
            return 12
        if k <= 32:
            return 16
        if k <= 256:
            return 18
        return 24
    if k <= 16:
        return 8
    if k <= 32:
        return 12
    if k <= 256:
        return 14
    return 18


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
def _matrix_rank_fused_jacobi_kernel(
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
    ROUND: tl.constexpr,
    PAIRS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
    SWEEPS,
    REL_EPS: tl.constexpr,
    ABS_EPS: tl.constexpr,
):
    # One program owns one matrix and runs the whole one-sided cyclic Jacobi
    # iteration. All pairs of a round-robin step are disjoint, so they are
    # processed together as a (BLOCK_P, BLOCK_C) tile instead of one kernel
    # launch (or one scalar loop iteration) per pair.
    batch = tl.program_id(0)
    rows = tl.arange(0, BLOCK_R)
    row_mask = rows < ROWS
    a_base = A + batch * M * N
    work_base = A_WORK + batch * K * ROWS

    column = 0
    while column < K:
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
        column += 1
    # The init stores are distributed across all warps of the block; the
    # sweep loop below reads the whole tile with a potentially different
    # lane mapping, so a block barrier is required for visibility.
    tl.debug_barrier()

    pair = tl.arange(0, BLOCK_P)
    ring: tl.constexpr = ROUND - 1
    accumulator_dtype = tl.float64 if IS_FP64 else tl.float32
    singular_indices = tl.arange(0, BLOCK_K)
    atol = tl.load(ATOL + batch)
    rtol = tl.load(RTOL + batch)
    sweep = 0
    e2_prev = tl.zeros((), dtype=accumulator_dtype)
    # Rebound by the per-sweep stability check; after the last sweep it
    # holds the column sums of the final work matrix.
    alphas = tl.zeros((BLOCK_K,), dtype=accumulator_dtype)
    keep_sweeping = 1
    while (sweep < SWEEPS) & (keep_sweeping != 0):
        rotations = 0
        e2_local = tl.zeros((), dtype=accumulator_dtype)
        step = 0
        while step < ROUND - 1:
            position_q = ROUND - 1 - pair
            p = tl.where(
                pair == 0,
                0,
                ((pair + ring - step - 1) % ring) + 1,
            )
            q = tl.where(
                position_q == 0,
                0,
                ((position_q + ring - step - 1) % ring) + 1,
            )
            valid_pair = (pair < PAIRS) & (p < K) & (q < K)
            swap = p > q
            ordered_p = tl.where(swap, q, p)
            ordered_q = tl.where(swap, p, q)
            pair_mask = valid_pair[:, None] & row_mask[None, :]

            ap = tl.load(
                work_base + ordered_p[:, None] * ROWS + rows[None, :],
                mask=pair_mask,
                other=0.0,
            )
            aq = tl.load(
                work_base + ordered_q[:, None] * ROWS + rows[None, :],
                mask=pair_mask,
                other=0.0,
            )
            alpha = tl.sum(ap * ap, axis=1)
            beta = tl.sum(aq * aq, axis=1)
            gamma = tl.sum(ap * aq, axis=1)
            e2_local += tl.sum(gamma * gamma, axis=0).to(accumulator_dtype)
            if IS_FP64:
                # Same double-single angle chain as the multi-block sweep
                # kernel: native float64 div/sqrt are software sequences on
                # devices with weak FP64 units and dominated the per-step
                # latency. Uses t = sign(diff) * 2*gamma /
                # (|diff| + sqrt(diff^2+4*gamma^2)) to avoid the
                # tau = diff/(2*gamma) overflow for tiny gamma.
                a_h = alpha.to(tl.float32)
                a_l = (alpha - a_h.to(tl.float64)).to(tl.float32)
                b_h = beta.to(tl.float32)
                b_l = (beta - b_h.to(tl.float64)).to(tl.float32)
                g_h = gamma.to(tl.float32)
                g_l = (gamma - g_h.to(tl.float64)).to(tl.float32)
                g2_h, g2_l = _df64_mul_ds(g_h, g_l, g_h, g_l)
                ab_h, ab_l = _df64_mul_ds(a_h, a_l, b_h, b_l)
                eps2 = REL_EPS * REL_EPS
                d_h, d_l = _df64_add(
                    g2_h,
                    g2_l,
                    -ab_h * eps2,
                    -(ab_l * eps2 + ABS_EPS * eps2),
                )
                active = valid_pair & (
                    (d_h > 0.0) | ((d_h == 0.0) & (d_l > 0.0))
                )
                diff_h, diff_l = _df64_add(b_h, b_l, -a_h, -a_l)
                d2_h, d2_l = _df64_mul_ds(diff_h, diff_l, diff_h, diff_l)
                u_h, u_l = _df64_add(d2_h, d2_l, 4.0 * g2_h, 4.0 * g2_l)
                # A subnormal u means |diff| and |gamma| are both below
                # ~1e-19: the pair is orthogonal far below any relevant
                # tolerance. This also keeps _df64_sqrt_ds away from
                # subnormal inputs, which flush to zero under FTZ.
                active = active & (u_h >= 1.1754944e-38)
                sq_h, sq_l = _df64_sqrt_ds(u_h, u_l)
                ad_h = tl.abs(diff_h)
                ad_l = tl.where(diff_h >= 0.0, diff_l, -diff_l)
                den_h, den_l = _df64_add(ad_h, ad_l, sq_h, sq_l)
                den_zero = den_h == 0.0
                den_h = tl.where(den_zero, 1.0, den_h)
                sign_diff = tl.where(diff_h >= 0.0, 1.0, -1.0)
                t_h, t_l = _df64_div_ds(
                    sign_diff * 2.0 * g_h,
                    sign_diff * 2.0 * g_l,
                    den_h,
                    den_l,
                )
                t_h = tl.where(den_zero, 1.0, t_h)
                t_l = tl.where(den_zero, 0.0, t_l)
                t2_h, t2_l = _df64_mul_ds(t_h, t_l, t_h, t_l)
                v_h, v_l = _df64_add(t2_h, t2_l, 1.0, 0.0)
                sq2_h, sq2_l = _df64_sqrt_ds(v_h, v_l)
                c_h, c_l = _df64_div_ds(
                    tl.zeros_like(v_h) + 1.0, tl.zeros_like(v_h), sq2_h, sq2_l
                )
                s_h, s_l = _df64_mul_ds(t_h, t_l, c_h, c_l)
                c = c_h.to(tl.float64) + c_l.to(tl.float64)
                s = s_h.to(tl.float64) + s_l.to(tl.float64)
            else:
                active = valid_pair & (
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
            rotations += tl.sum(active.to(tl.int32), axis=0)
            c = tl.where(active, c, 1.0)
            s = tl.where(active, s, 0.0)
            tl.store(
                work_base + ordered_p[:, None] * ROWS + rows[None, :],
                c[:, None] * ap - s[:, None] * aq,
                mask=pair_mask,
            )
            tl.store(
                work_base + ordered_q[:, None] * ROWS + rows[None, :],
                s[:, None] * ap + c[:, None] * aq,
                mask=pair_mask,
            )
            # Columns migrate across pair slots (and hence across warps)
            # between steps: the next step's tile loads must not race this
            # step's rotation stores.
            tl.debug_barrier()
            step += 1
        # --- rank-stability check (same criterion as the multi-block
        # sweep kernels) ---------------------------------------------
        # G = W^T W = diag(alpha) + E with ||E||_F = sqrt(sum gamma^2).
        # Weyl's theorem bounds every singular-value perturbation by
        # ||E||_F, so once ||E||_F stays below half the smallest
        # |alpha_i - tol^2| margin no singular value can cross tol.
        check_tile = tl.load(
            work_base + singular_indices[:, None] * ROWS + rows[None, :],
            mask=(singular_indices < K)[:, None] & row_mask[None, :],
            other=0.0,
        )
        # The next sweep's first rotation stores must not overtake this
        # read in a laggard warp.
        tl.debug_barrier()
        alphas = tl.sum(check_tile * check_tile, axis=1).to(accumulator_dtype)
        maxa = tl.max(alphas, axis=0)
        tol = tl.maximum(atol, rtol * tl.sqrt(maxa))
        tol2 = tol * tol
        margin = tl.min(
            tl.where(
                singular_indices < K,
                tl.abs(alphas - tol2),
                tl.full((BLOCK_K,), float("inf"), dtype=accumulator_dtype),
            ),
            axis=0,
        )
        # Two sufficient stop conditions: the Weyl bound proves every
        # singular value is separated from the threshold, or the residual
        # stalled at the arithmetic noise floor (equivalent to the classic
        # "no more rotations" Jacobi exit).
        stall_floor = 64.0 * REL_EPS * maxa
        stable = (e2_local <= 0.25 * margin * margin) | (
            (sweep > 0)
            & (e2_local >= 0.8 * e2_prev)
            & (e2_local <= stall_floor * stall_floor)
        )
        e2_prev = e2_local
        keep_sweeping = ((rotations != 0) & (stable == 0)).to(tl.int32)
        sweep += 1

    # The last stability check already read the final work matrix, so its
    # column sums give the singular values directly.
    singular_values = tl.sqrt(alphas)

    max_value = tl.max(singular_values, axis=0)
    threshold = tl.maximum(atol, rtol * max_value)
    rank = tl.sum(
        (
            (singular_values > threshold)
            & (singular_indices < K)
        ).to(tl.int32),
        axis=0,
    )
    tl.store(OUT + batch, rank.to(tl.int64))


@triton.jit
def _grid_barrier(BARRIER, base, generation, num_programs):
    # Software grid barrier. The launcher caps the grid at a co-residency
    # bound, so spinning here cannot deadlock. The counter is monotone
    # across launches: after ``generation`` barriers of this launch it
    # holds ``base + generation * num_programs``. The spin poll must be an
    # acquire-or-stronger atomic (not a volatile load): without acquire
    # semantics, subsequent buffer reads can be served stale data that
    # predates the barrier.
    #
    # Keep the poll directly in the while CONDITION with an empty body, and
    # use the DEFAULT memory order (acq_rel) -- the same form as
    # linalg_qr._barrier.  The read-modify-write must be re-evaluated by the
    # loop condition on every iteration: assigning the atomic to a
    # loop-carried scalar (``arrived = ...; while arrived < target:
    # arrived = ...``) miscompiles on some vendor backends (metax, hygon),
    # which then spin forever on a stale value.  acq_rel is a superset of
    # the acquire ordering the barrier-protected loads need.
    tl.atomic_add(BARRIER, 1)
    target = base + (generation + 1) * num_programs
    while tl.atomic_add(BARRIER, 0) < target:
        pass


@triton.jit
def _neighbor_sync(FLAGS, base, generation, program, pair_block, pair_blocks):
    # Neighbor barrier for the round-robin step schedule: a column migrates
    # exactly one pair slot per step, so a block's next step only reads
    # columns written by the two adjacent blocks of the same matrix.
    # Per-block monotone step flags avoid the global counter's RMW
    # serialization and the full-grid lockstep wait. Blocks must still
    # arrive every step even when their pair is inactive.
    tl.atomic_add(FLAGS + program, 1, sem="release", scope="gpu")
    target = base + generation + 1
    batch_base = program - pair_block
    prev = batch_base + (pair_block + pair_blocks - 1) % pair_blocks
    nxt = batch_base + (pair_block + 1) % pair_blocks
    # Same spin form as _grid_barrier: poll in the while condition with an
    # empty body and default (acq_rel) ordering; a loop-carried scalar
    # holding the poll result miscompiles on some vendor backends.
    while tl.atomic_add(FLAGS + prev, 0) < target:
        pass
    while tl.atomic_add(FLAGS + nxt, 0) < target:
        pass


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
def _matrix_rank_df64_init_kernel(
    A,
    A_WORK_HI,
    A_WORK_LO,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    TALL: tl.constexpr,
    HERMITIAN: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    # Splits every float64 entry into a float32 hi/lo pair (double-single).
    batch = tl.program_id(0)
    column = tl.program_id(1)
    rows = tl.arange(0, BLOCK_R)
    row_mask = rows < ROWS
    a_base = A + batch * M * N
    work_base_hi = A_WORK_HI + batch * K * ROWS
    work_base_lo = A_WORK_LO + batch * K * ROWS

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
    values_hi = values.to(tl.float32)
    values_lo = (values - values_hi.to(tl.float64)).to(tl.float32)
    tl.store(work_base_hi + column * ROWS + rows, values_hi, mask=row_mask)
    tl.store(work_base_lo + column * ROWS + rows, values_lo, mask=row_mask)


@triton.jit
def _sum3(a1, b1, c1, a2, b2, c2):
    # Triple dot products share one reduction tree (one shared-memory round
    # instead of three).
    return a1 + a2, b1 + b2, c1 + c2


@triton.jit
def _df64_add(h1, l1, h2, l2):
    # Error-free addition of two double-single numbers (Knuth TwoSum on the
    # hi parts, lo parts gathered afterwards, then one renormalization).
    s = h1 + h2
    z = s - h1
    e = (h1 - (s - z)) + (h2 - z)
    lo = l1 + l2 + e
    h = s + lo
    e2 = lo - (h - s)
    return h, e2


@triton.jit
def _df64_mul_ds(a_h, a_l, b_h, b_l):
    # Double-single product: TwoProd on the hi parts plus the cross terms.
    p = a_h * b_h
    e = tl.fma(a_h, b_h, -p) + a_h * b_l + a_l * b_h
    h = p + e
    l = e - (h - p)
    return h, l


@triton.jit
def _df64_div_ds(a_h, a_l, b_h, b_l):
    # Double-single division: fp32 quotient plus one df64 correction step.
    q1 = a_h / b_h
    p = q1 * b_h
    pe = tl.fma(q1, b_h, -p)
    r_h, r_l = _df64_add(a_h, a_l, -p, -(pe + q1 * b_l))
    q2 = r_h / b_h
    h = q1 + q2
    l = q2 - (h - q1)
    return h, l


@triton.jit
def _df64_sqrt_ds(a_h, a_l):
    # Double-single square root: fp32 root plus one Newton/df64 correction.
    x = tl.sqrt(a_h)
    p = x * x
    pe = tl.fma(x, x, -p)
    r_h, r_l = _df64_add(a_h, a_l, -p, -pe)
    corr = r_h / (2.0 * x)
    h = x + corr
    l = corr - (h - x)
    not_positive = a_h <= 0.0
    h = tl.where(not_positive, 0.0, h)
    l = tl.where(not_positive, 0.0, l)
    return h, l


@libentry()
@triton.jit
def _matrix_rank_jacobi_sweep_kernel(
    A_WORK,
    ATOL,
    RTOL,
    BARRIER,
    FLAG,
    MAXA,
    MINM,
    E2,
    ALPHA,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    ROUND: tl.constexpr,
    PAIRS: tl.constexpr,
    PPB: tl.constexpr,
    BLOCK_R: tl.constexpr,
    SWEEPS,
    SWEEP_BASE,
    TOTAL_SWEEPS,
    BARRIER_BASE,
    NFLAG,
    NEIGHBOR: tl.constexpr,
    IS_FP64: tl.constexpr,
    REL_EPS: tl.constexpr,
    ABS_EPS: tl.constexpr,
):
    # Persistent kernel: every program owns PPB pair slots of the round-robin
    # schedule for one matrix and programs synchronize between steps with a
    # software grid barrier instead of one kernel launch per step. The grid
    # size is capped by the launcher so that all programs are co-resident.
    # SWEEPS sweeps run per launch; the stability buffers are indexed by the
    # global sweep number SWEEP_BASE + sweep with stride TOTAL_SWEEPS so the
    # stall detector can compare against the previous launch's last sweep.
    program = tl.program_id(0)
    num_programs = tl.num_programs(0)
    pair_blocks = (PAIRS + PPB - 1) // PPB
    batch = program // pair_blocks
    pair_block = program % pair_blocks
    work_base = A_WORK + batch * K * ROWS

    pair = pair_block * PPB + tl.arange(0, PPB)
    rows = tl.arange(0, BLOCK_R)
    row_mask = rows < ROWS
    ring: tl.constexpr = ROUND - 1
    int_dtype = tl.int64 if IS_FP64 else tl.int32
    float_dtype = tl.float64 if IS_FP64 else tl.float32

    generation = 0
    check_gen = 0
    if NEIGHBOR:
        # Neighbor step flags are monotone across launches; all blocks of a
        # matrix advance them uniformly, so the value at entry is the base.
        nbase = tl.atomic_add(NFLAG + program, 0, sem="acquire", scope="gpu")
    else:
        nbase = 0
    sweep = 0
    # Batches proven rank-stable by an earlier launch start masked.
    unstable = 1 - tl.load(FLAG + batch)
    while sweep < SWEEPS:
        g_sweep = SWEEP_BASE + sweep
        e2_local = tl.zeros((), dtype=float_dtype)
        step = 0
        while step < ROUND - 1:
            position_q = ROUND - 1 - pair
            p = tl.where(
                pair == 0,
                0,
                ((pair + ring - step - 1) % ring) + 1,
            )
            q = tl.where(
                position_q == 0,
                0,
                ((position_q + ring - step - 1) % ring) + 1,
            )
            # Programs of a matrix that is already proven rank-stable skip
            # their memory traffic but still hit every barrier.
            valid_pair = (
                (pair < PAIRS) & (p < K) & (q < K) & (unstable != 0)
            )
            swap = p > q
            ordered_p = tl.where(swap, q, p)
            ordered_q = tl.where(swap, p, q)
            pair_mask = valid_pair[:, None] & row_mask[None, :]

            ap = tl.load(
                work_base + ordered_p[:, None] * ROWS + rows[None, :],
                mask=pair_mask,
                other=0.0,
                cache_modifier=".cg",
            )
            aq = tl.load(
                work_base + ordered_q[:, None] * ROWS + rows[None, :],
                mask=pair_mask,
                other=0.0,
                cache_modifier=".cg",
            )
            alpha, beta, gamma = tl.reduce(
                (ap * ap, aq * aq, ap * aq), 1, _sum3
            )
            e2_local += tl.sum(gamma * gamma, axis=0)
            if IS_FP64:
                # Native float64 div/sqrt are software sequences on devices
                # with weak FP64 units and dominated the per-step latency,
                # so the rotation angle is computed in double-single
                # arithmetic (float32 ops only, ~1e-14 relative accuracy,
                # far below what the rotation update needs). Uses
                # t = sign(diff) * 2*gamma / (|diff| + sqrt(diff^2+4*gamma^2))
                # to avoid the tau = diff/(2*gamma) overflow for tiny gamma.
                a_h = alpha.to(tl.float32)
                a_l = (alpha - a_h.to(tl.float64)).to(tl.float32)
                b_h = beta.to(tl.float32)
                b_l = (beta - b_h.to(tl.float64)).to(tl.float32)
                g_h = gamma.to(tl.float32)
                g_l = (gamma - g_h.to(tl.float64)).to(tl.float32)
                g2_h, g2_l = _df64_mul_ds(g_h, g_l, g_h, g_l)
                ab_h, ab_l = _df64_mul_ds(a_h, a_l, b_h, b_l)
                eps2 = REL_EPS * REL_EPS
                d_h, d_l = _df64_add(
                    g2_h,
                    g2_l,
                    -ab_h * eps2,
                    -(ab_l * eps2 + ABS_EPS * eps2),
                )
                active = valid_pair & (
                    (d_h > 0.0) | ((d_h == 0.0) & (d_l > 0.0))
                )
                diff_h, diff_l = _df64_add(b_h, b_l, -a_h, -a_l)
                d2_h, d2_l = _df64_mul_ds(diff_h, diff_l, diff_h, diff_l)
                u_h, u_l = _df64_add(d2_h, d2_l, 4.0 * g2_h, 4.0 * g2_l)
                # A subnormal u means |diff| and |gamma| are both below
                # ~1e-19: the pair is orthogonal far below any relevant
                # tolerance. This also keeps _df64_sqrt_ds away from
                # subnormal inputs, which flush to zero under FTZ.
                active = active & (u_h >= 1.1754944e-38)
                sq_h, sq_l = _df64_sqrt_ds(u_h, u_l)
                ad_h = tl.abs(diff_h)
                ad_l = tl.where(diff_h >= 0.0, diff_l, -diff_l)
                den_h, den_l = _df64_add(ad_h, ad_l, sq_h, sq_l)
                den_zero = den_h == 0.0
                den_h = tl.where(den_zero, 1.0, den_h)
                sign_diff = tl.where(diff_h >= 0.0, 1.0, -1.0)
                t_h, t_l = _df64_div_ds(
                    sign_diff * 2.0 * g_h,
                    sign_diff * 2.0 * g_l,
                    den_h,
                    den_l,
                )
                t_h = tl.where(den_zero, 1.0, t_h)
                t_l = tl.where(den_zero, 0.0, t_l)
                t2_h, t2_l = _df64_mul_ds(t_h, t_l, t_h, t_l)
                v_h, v_l = _df64_add(t2_h, t2_l, 1.0, 0.0)
                sq2_h, sq2_l = _df64_sqrt_ds(v_h, v_l)
                c_h, c_l = _df64_div_ds(
                    tl.zeros_like(v_h) + 1.0, tl.zeros_like(v_h), sq2_h, sq2_l
                )
                s_h, s_l = _df64_mul_ds(t_h, t_l, c_h, c_l)
                c = c_h.to(tl.float64) + c_l.to(tl.float64)
                s = s_h.to(tl.float64) + s_l.to(tl.float64)
                c = tl.where(active, c, 1.0)
                s = tl.where(active, s, 0.0)
            else:
                active = valid_pair & (
                    tl.abs(gamma)
                    > REL_EPS * tl.sqrt(alpha * beta + ABS_EPS)
                )
                safe_gamma = tl.where(active, gamma, 1.0)
                tau = (beta - alpha) / (2.0 * safe_gamma)
                sign_tau = tl.where(tau >= 0.0, 1.0, -1.0)
                t = sign_tau / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
                c = 1.0 / tl.sqrt(1.0 + t * t)
                s = t * c
                c = tl.where(active, c, 1.0)
                s = tl.where(active, s, 0.0)
            tl.store(
                work_base + ordered_p[:, None] * ROWS + rows[None, :],
                c[:, None] * ap - s[:, None] * aq,
                mask=pair_mask,
                cache_modifier=".cg",
            )
            tl.store(
                work_base + ordered_q[:, None] * ROWS + rows[None, :],
                s[:, None] * ap + c[:, None] * aq,
                mask=pair_mask,
                cache_modifier=".cg",
            )


            if NEIGHBOR:
                _neighbor_sync(
                    NFLAG, nbase, generation, program, pair_block, pair_blocks
                )
            else:
                _grid_barrier(
                    BARRIER, BARRIER_BASE, generation, num_programs
                )
            generation += 1
            step += 1

        # --- rank-stability check -------------------------------------
        # G = W^T W = diag(alpha) + E with ||E||_F = sqrt(sum gamma^2).
        # Weyl's theorem bounds every singular-value perturbation by
        # ||E||_F, so once ||E||_F stays below half the smallest
        # |alpha_i - tol^2| margin no singular value can cross tol.
        tl.atomic_add(E2 + batch * TOTAL_SWEEPS + g_sweep, e2_local)
        maxa_local = tl.zeros((), dtype=float_dtype)
        column = pair_block
        while column < K:
            col_mask = (column < K) & (rows < ROWS) & (unstable != 0)
            values = tl.load(
                work_base + column * ROWS + rows,
                mask=col_mask,
                other=0.0,
                cache_modifier=".cg",
            )
            alpha_i = tl.sum(values * values, axis=0)
            tl.store(
                ALPHA + batch * K + column,
                alpha_i,
                mask=(column < K) & (unstable != 0),
            )
            maxa_local = tl.maximum(maxa_local, alpha_i)
            column += pair_blocks
        tl.atomic_max(
            MAXA + batch * TOTAL_SWEEPS + g_sweep,
            maxa_local.to(int_dtype, bitcast=True),
            mask=(unstable != 0),
        )
        if NEIGHBOR:
            _grid_barrier(BARRIER, BARRIER_BASE, check_gen, num_programs)
            check_gen += 1
        else:
            _grid_barrier(BARRIER, BARRIER_BASE, generation, num_programs)
            generation += 1

        maxa = tl.atomic_add(MAXA + batch * TOTAL_SWEEPS + g_sweep, 0, sem="acquire", scope="gpu").to(
            float_dtype, bitcast=True
        )
        atol = tl.load(ATOL + batch)
        rtol = tl.load(RTOL + batch)
        tol = tl.maximum(atol, rtol * tl.sqrt(maxa))
        tol2 = tol * tol
        min_margin = tl.full((), float("inf"), dtype=float_dtype)
        column = pair_block
        while column < K:
            alpha_i = tl.atomic_add(ALPHA + batch * K + column, 0.0, sem="acquire", scope="gpu")
            min_margin = tl.minimum(min_margin, tl.abs(alpha_i - tol2))
            column += pair_blocks
        tl.atomic_min(
            MINM + batch * TOTAL_SWEEPS + g_sweep,
            min_margin.to(int_dtype, bitcast=True),
            mask=(unstable != 0),
        )
        if NEIGHBOR:
            _grid_barrier(BARRIER, BARRIER_BASE, check_gen, num_programs)
            check_gen += 1
        else:
            _grid_barrier(BARRIER, BARRIER_BASE, generation, num_programs)
            generation += 1

        e2 = tl.atomic_add(E2 + batch * TOTAL_SWEEPS + g_sweep, 0.0, sem="acquire", scope="gpu")
        margin = tl.atomic_add(MINM + batch * TOTAL_SWEEPS + g_sweep, 0, sem="acquire", scope="gpu").to(
            float_dtype, bitcast=True
        )
        e2_prev = tl.atomic_add(
            E2 + batch * TOTAL_SWEEPS + g_sweep - 1,
            0.0,
            sem="acquire",
            scope="gpu",
            mask=g_sweep > 0,
        )
        # Two sufficient stop conditions: the Weyl bound proves every
        # singular value is separated from the threshold, or the residual
        # stalled at the arithmetic noise floor (equivalent to the classic
        # "no more rotations" Jacobi exit).
        stall_floor = 64.0 * REL_EPS * maxa
        stable = (e2 <= 0.25 * margin * margin) | (
            (g_sweep > 0)
            & (e2 >= 0.8 * e2_prev)
            & (e2 <= stall_floor * stall_floor)
        )
        tl.atomic_max(
            FLAG + batch,
            (stable != 0).to(tl.int32),
            mask=(unstable != 0),
        )
        unstable = (stable == 0).to(tl.int32)
        sweep += 1


@libentry()
@triton.jit
def _matrix_rank_jacobi_sweep_df64_kernel(
    A_WORK_HI,
    A_WORK_LO,
    ATOL,
    RTOL,
    BARRIER,
    FLAG,
    MAXA,
    MINM,
    E2,
    ALPHA,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    ROUND: tl.constexpr,
    PAIRS: tl.constexpr,
    PPB: tl.constexpr,
    BLOCK_R: tl.constexpr,
    SWEEPS,
    SWEEP_BASE,
    TOTAL_SWEEPS,
    BARRIER_BASE,
    NFLAG,
    NEIGHBOR: tl.constexpr,
    REL_EPS: tl.constexpr,
    ABS_EPS: tl.constexpr,
):
    # float64 variant of the sweep kernel using double-single arithmetic:
    # values are stored as float32 hi/lo pairs and all vector math runs at
    # float32 throughput with error-free transformations, which is much
    # faster than native float64 on devices with weak FP64 units. Per-pair
    # scalars (dot results, rotation angles, stability bounds) are still
    # evaluated in native float64.
    program = tl.program_id(0)
    num_programs = tl.num_programs(0)
    pair_blocks = (PAIRS + PPB - 1) // PPB
    batch = program // pair_blocks
    pair_block = program % pair_blocks
    work_base_hi = A_WORK_HI + batch * K * ROWS
    work_base_lo = A_WORK_LO + batch * K * ROWS

    pair = pair_block * PPB + tl.arange(0, PPB)
    rows = tl.arange(0, BLOCK_R)
    row_mask = rows < ROWS
    ring: tl.constexpr = ROUND - 1

    generation = 0
    check_gen = 0
    if NEIGHBOR:
        # Neighbor step flags are monotone across launches; all blocks of a
        # matrix advance them uniformly, so the value at entry is the base.
        nbase = tl.atomic_add(NFLAG + program, 0, sem="acquire", scope="gpu")
    else:
        nbase = 0
    sweep = 0
    # Batches proven rank-stable by an earlier launch start masked.
    unstable = 1 - tl.load(FLAG + batch)
    while sweep < SWEEPS:
        g_sweep = SWEEP_BASE + sweep
        e2_local_h = tl.zeros((), dtype=tl.float32)
        e2_local_l = tl.zeros((), dtype=tl.float32)
        step = 0
        if unstable != 0:
            while step < ROUND - 1:
                position_q = ROUND - 1 - pair
                p = tl.where(
                    pair == 0,
                    0,
                    ((pair + ring - step - 1) % ring) + 1,
                )
                q = tl.where(
                    position_q == 0,
                    0,
                    ((position_q + ring - step - 1) % ring) + 1,
                )
                valid_pair = (
                    (pair < PAIRS) & (p < K) & (q < K) & (unstable != 0)
                )
                swap = p > q
                ordered_p = tl.where(swap, q, p)
                ordered_q = tl.where(swap, p, q)
                pair_mask = valid_pair[:, None] & row_mask[None, :]

                ap_hi = tl.load(
                    work_base_hi + ordered_p[:, None] * ROWS + rows[None, :],
                    mask=pair_mask,
                    other=0.0,
                    cache_modifier=".cg",
                )
                ap_lo = tl.load(
                    work_base_lo + ordered_p[:, None] * ROWS + rows[None, :],
                    mask=pair_mask,
                    other=0.0,
                    cache_modifier=".cg",
                )
                aq_hi = tl.load(
                    work_base_hi + ordered_q[:, None] * ROWS + rows[None, :],
                    mask=pair_mask,
                    other=0.0,
                    cache_modifier=".cg",
                )
                aq_lo = tl.load(
                    work_base_lo + ordered_q[:, None] * ROWS + rows[None, :],
                    mask=pair_mask,
                    other=0.0,
                    cache_modifier=".cg",
                )

                pp_hi = ap_hi * ap_hi
                pp_lo = (
                    tl.fma(ap_hi, ap_hi, -pp_hi)
                    + ap_hi * ap_lo
                    + ap_lo * ap_hi
                )
                alpha_hi, alpha_lo = tl.reduce(
                    (pp_hi, pp_lo), 1, _df64_add
                )
                qq_hi = aq_hi * aq_hi
                qq_lo = (
                    tl.fma(aq_hi, aq_hi, -qq_hi)
                    + aq_hi * aq_lo
                    + aq_lo * aq_hi
                )
                beta_hi, beta_lo = tl.reduce(
                    (qq_hi, qq_lo), 1, _df64_add
                )
                pq_hi = ap_hi * aq_hi
                pq_lo = (
                    tl.fma(ap_hi, aq_hi, -pq_hi)
                    + ap_hi * aq_lo
                    + ap_lo * aq_hi
                )
                gamma_hi, gamma_lo = tl.reduce(
                    (pq_hi, pq_lo), 1, _df64_add
                )

                # Rotation angle in double-single arithmetic (float32 ops only):
                # native float64 div/sqrt are extremely slow on devices with weak
                # FP64 units and dominated the per-step latency. Uses
                # t = sign(diff) * 2*gamma / (|diff| + sqrt(diff^2 + 4*gamma^2))
                # instead of the tau = diff/(2*gamma) form: tau^2 overflows the
                # float32 range of the hi plane when gamma is tiny.
                g2_h, g2_l = _df64_mul_ds(gamma_hi, gamma_lo, gamma_hi, gamma_lo)
                e2_local_h, e2_local_l = _df64_add(
                    e2_local_h, e2_local_l, tl.sum(g2_h, axis=0), tl.sum(g2_l, axis=0)
                )
                ab_h, ab_l = _df64_mul_ds(alpha_hi, alpha_lo, beta_hi, beta_lo)
                eps2 = REL_EPS * REL_EPS
                d_h, d_l = _df64_add(
                    g2_h, g2_l, -ab_h * eps2, -ab_l * eps2
                )
                active = valid_pair & ((d_h > 0.0) | ((d_h == 0.0) & (d_l > 0.0)))
                diff_h, diff_l = _df64_add(beta_hi, beta_lo, -alpha_hi, -alpha_lo)
                d2_h, d2_l = _df64_mul_ds(diff_h, diff_l, diff_h, diff_l)
                u_h, u_l = _df64_add(d2_h, d2_l, 4.0 * g2_h, 4.0 * g2_l)
                # A subnormal u means |diff| and |gamma| are both below ~1e-19,
                # so the pair is orthogonal far below any relevant tolerance and
                # the rotation can be skipped. This also keeps _df64_sqrt_ds away
                # from subnormal inputs, which flush to zero under FTZ and would
                # produce inf/NaN in the correction term.
                active = active & (u_h >= 1.1754944e-38)
                sq_h, sq_l = _df64_sqrt_ds(u_h, u_l)
                ad_h = tl.abs(diff_h)
                ad_l = tl.where(diff_h >= 0.0, diff_l, -diff_l)
                den_h, den_l = _df64_add(ad_h, ad_l, sq_h, sq_l)
                # den == 0 only when diff and gamma both underflowed; any
                # t with |t| == 1 orthogonalizes the pair in that case.
                den_zero = den_h == 0.0
                den_h = tl.where(den_zero, 1.0, den_h)
                sign_diff = tl.where(diff_h >= 0.0, 1.0, -1.0)
                t_h, t_l = _df64_div_ds(
                    sign_diff * 2.0 * gamma_hi,
                    sign_diff * 2.0 * gamma_lo,
                    den_h,
                    den_l,
                )
                t_h = tl.where(den_zero, 1.0, t_h)
                t_l = tl.where(den_zero, 0.0, t_l)
                t2_h, t2_l = _df64_mul_ds(t_h, t_l, t_h, t_l)
                v_h, v_l = _df64_add(t2_h, t2_l, 1.0, 0.0)
                sq2_h, sq2_l = _df64_sqrt_ds(v_h, v_l)
                c_hi, c_lo = _df64_div_ds(
                    tl.zeros_like(v_h) + 1.0, tl.zeros_like(v_h), sq2_h, sq2_l
                )
                s_hi, s_lo = _df64_mul_ds(t_h, t_l, c_hi, c_lo)
                c_hi = tl.where(active, c_hi, 1.0)
                c_lo = tl.where(active, c_lo, 0.0)
                s_hi = tl.where(active, s_hi, 0.0)
                s_lo = tl.where(active, s_lo, 0.0)
                c_h = c_hi[:, None]
                c_l = c_lo[:, None]
                s_h = s_hi[:, None]
                s_l = s_lo[:, None]

                # new_ap = c * ap - s * aq (double-single, then renormalize)
                ph = c_h * ap_hi
                pl = (
                    tl.fma(c_h, ap_hi, -ph)
                    + c_h * ap_lo
                    + c_l * ap_hi
                )
                qh = s_h * aq_hi
                ql = (
                    tl.fma(s_h, aq_hi, -qh)
                    + s_h * aq_lo
                    + s_l * aq_hi
                )
                rh = ph - qh
                rz = rh - ph
                rl = (pl - ql) + ((ph - (rh - rz)) - (qh + rz))
                new_ap_hi = rh + rl
                new_ap_lo = rl - (new_ap_hi - rh)

                # new_aq = s * ap + c * aq
                uh = s_h * ap_hi
                ul = (
                    tl.fma(s_h, ap_hi, -uh)
                    + s_h * ap_lo
                    + s_l * ap_hi
                )
                vh = c_h * aq_hi
                vl = (
                    tl.fma(c_h, aq_hi, -vh)
                    + c_h * aq_lo
                    + c_l * aq_hi
                )
                wh = uh + vh
                wz = wh - uh
                wl = (ul + vl) + ((uh - (wh - wz)) + (vh - wz))
                new_aq_hi = wh + wl
                new_aq_lo = wl - (new_aq_hi - wh)

                # Inactive pairs apply the identity rotation, so their columns
                # are left untouched; skip the stores entirely (late sweeps are
                # dominated by inactive pairs).
                store_mask = pair_mask & active[:, None]
                tl.store(
                    work_base_hi + ordered_p[:, None] * ROWS + rows[None, :],
                    new_ap_hi,
                    mask=store_mask,
                    cache_modifier=".cg",
                )
                tl.store(
                    work_base_lo + ordered_p[:, None] * ROWS + rows[None, :],
                    new_ap_lo,
                    mask=store_mask,
                    cache_modifier=".cg",
                )
                tl.store(
                    work_base_hi + ordered_q[:, None] * ROWS + rows[None, :],
                    new_aq_hi,
                    mask=store_mask,
                    cache_modifier=".cg",
                )
                tl.store(
                    work_base_lo + ordered_q[:, None] * ROWS + rows[None, :],
                    new_aq_lo,
                    mask=store_mask,
                    cache_modifier=".cg",
                )


                if NEIGHBOR:
                    _neighbor_sync(
                        NFLAG,
                        nbase,
                        generation,
                        program,
                        pair_block,
                        pair_blocks,
                    )
                else:
                    _grid_barrier(
                        BARRIER, BARRIER_BASE, generation, num_programs
                    )
                generation += 1
                step += 1
        else:
            # Rank-stable matrices only keep the grid in sync: every
            # program must still arrive at every barrier.
            while step < ROUND - 1:
                if NEIGHBOR:
                    _neighbor_sync(
                        NFLAG,
                        nbase,
                        generation,
                        program,
                        pair_block,
                        pair_blocks,
                    )
                else:
                    _grid_barrier(
                        BARRIER, BARRIER_BASE, generation, num_programs
                    )
                generation += 1
                step += 1

        # --- rank-stability check (see the native sweep kernel) --------
        tl.atomic_add(
            E2 + batch * TOTAL_SWEEPS + g_sweep,
            e2_local_h.to(tl.float64) + e2_local_l.to(tl.float64),
        )
        maxa_local = tl.zeros((), dtype=tl.float64)
        column = pair_block
        while column < K:
            col_mask = (column < K) & (rows < ROWS) & (unstable != 0)
            value_hi = tl.load(
                work_base_hi + column * ROWS + rows,
                mask=col_mask,
                other=0.0,
                cache_modifier=".cg",
            )
            value_lo = tl.load(
                work_base_lo + column * ROWS + rows,
                mask=col_mask,
                other=0.0,
                cache_modifier=".cg",
            )
            values = value_hi.to(tl.float64) + value_lo.to(tl.float64)
            alpha_i = tl.sum(values * values, axis=0)
            tl.store(
                ALPHA + batch * K + column,
                alpha_i,
                mask=(column < K) & (unstable != 0),
            )
            maxa_local = tl.maximum(maxa_local, alpha_i)
            column += pair_blocks
        tl.atomic_max(
            MAXA + batch * TOTAL_SWEEPS + g_sweep,
            maxa_local.to(tl.int64, bitcast=True),
            mask=(unstable != 0),
        )
        if NEIGHBOR:
            _grid_barrier(BARRIER, BARRIER_BASE, check_gen, num_programs)
            check_gen += 1
        else:
            _grid_barrier(BARRIER, BARRIER_BASE, generation, num_programs)
            generation += 1

        maxa = tl.atomic_add(MAXA + batch * TOTAL_SWEEPS + g_sweep, 0, sem="acquire", scope="gpu").to(
            tl.float64, bitcast=True
        )
        atol = tl.load(ATOL + batch)
        rtol = tl.load(RTOL + batch)
        tol = tl.maximum(atol, rtol * tl.sqrt(maxa))
        tol2 = tol * tol
        min_margin = tl.full((), float("inf"), dtype=tl.float64)
        column = pair_block
        while column < K:
            alpha_i = tl.atomic_add(ALPHA + batch * K + column, 0.0, sem="acquire", scope="gpu")
            min_margin = tl.minimum(min_margin, tl.abs(alpha_i - tol2))
            column += pair_blocks
        tl.atomic_min(
            MINM + batch * TOTAL_SWEEPS + g_sweep,
            min_margin.to(tl.int64, bitcast=True),
            mask=(unstable != 0),
        )
        if NEIGHBOR:
            _grid_barrier(BARRIER, BARRIER_BASE, check_gen, num_programs)
            check_gen += 1
        else:
            _grid_barrier(BARRIER, BARRIER_BASE, generation, num_programs)
            generation += 1

        e2 = tl.atomic_add(E2 + batch * TOTAL_SWEEPS + g_sweep, 0.0, sem="acquire", scope="gpu")
        margin = tl.atomic_add(MINM + batch * TOTAL_SWEEPS + g_sweep, 0, sem="acquire", scope="gpu").to(
            tl.float64, bitcast=True
        )
        e2_prev = tl.atomic_add(
            E2 + batch * TOTAL_SWEEPS + g_sweep - 1,
            0.0,
            sem="acquire",
            scope="gpu",
            mask=g_sweep > 0,
        )
        stall_floor = 64.0 * REL_EPS * maxa
        stable = (e2 <= 0.25 * margin * margin) | (
            (g_sweep > 0)
            & (e2 >= 0.8 * e2_prev)
            & (e2 <= stall_floor * stall_floor)
        )
        tl.atomic_max(
            FLAG + batch,
            (stable != 0).to(tl.int32),
            mask=(unstable != 0),
        )
        unstable = (stable == 0).to(tl.int32)
        sweep += 1



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
def _matrix_rank_df64_norm_kernel(
    A_WORK_HI,
    A_WORK_LO,
    S_WORK,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    batch = tl.program_id(0)
    column = tl.program_id(1)
    rows = tl.arange(0, BLOCK_R)
    row_mask = rows < ROWS
    work_base_hi = A_WORK_HI + batch * K * ROWS
    work_base_lo = A_WORK_LO + batch * K * ROWS
    value_hi = tl.load(
        work_base_hi + column * ROWS + rows,
        mask=row_mask,
        other=0.0,
    )
    value_lo = tl.load(
        work_base_lo + column * ROWS + rows,
        mask=row_mask,
        other=0.0,
    )
    values = value_hi.to(tl.float64) + value_lo.to(tl.float64)
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


# ===========================================================================
# Hermitian fast path for large matrices: Householder tridiagonalization
# followed by Sturm-sequence eigenvalue counting (Sylvester's law of
# inertia). Unlike the iterative Jacobi sweeps this is non-iterative:
# O(k^3) dense fp64 work spread over many blocks plus an O(k) counting
# pass, with no convergence sweeps and no per-step full-column exchange.
# ===========================================================================


@libentry()
@triton.jit
def _matrix_rank_herm_tridiag_init_kernel(
    A,
    S,
    K: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Build the symmetrized fp64 work matrix from the lower triangle of A
    # (matching torch's hermitian semantics): S[r, c] = A[max(r,c), min(r,c)].
    batch = tl.program_id(0)
    row_block = tl.program_id(1)
    rows = row_block * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_C)
    a_base = A + batch * K * K
    s_base = S + batch * K * K
    col = 0
    while col < K:
        cc = col + cols
        mask = (rows[:, None] < K) & (cc[None, :] < K)
        source_rows = tl.maximum(rows[:, None], cc[None, :])
        source_cols = tl.minimum(rows[:, None], cc[None, :])
        values = tl.load(a_base + source_rows * K + source_cols, mask=mask, other=0.0)
        tl.store(
            s_base + rows[:, None] * K + cc[None, :],
            values.to(S.dtype.element_ty),
            mask=mask,
        )
        col += BLOCK_C


@libentry()
@triton.jit
def _matrix_rank_herm_tridiag_kernel(
    S,
    W,
    D,
    E,
    BARRIER,
    K: tl.constexpr,
    NB: tl.constexpr,
    BJ: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Unblocked Householder tridiagonalization (LAPACK DSYTD2), parallelized
    # over NB blocks per matrix with a software grid barrier. Each block owns
    # a contiguous range of BJ rows; all phases access the work matrix by
    # rows so the main memory traffic is contiguous. The Householder scalars
    # (sigma, alpha, tau) of each column are recomputed redundantly and
    # bit-identically in every block, which removes any cross-block
    # reduction for them: v is never materialized because
    # v[c] = S[c, i] for c > i+1 and v[i+1] = x0 - alpha, so
    #   w[r] = tau * (sum_{c>i} S[r,c]*S[c,i] - alpha * S[r,i+1]).
    # Two barriers per column step: one after the w phase, one after the
    # symmetric rank-2 trailing update.
    pid = tl.program_id(0)
    batch = pid // NB
    jb = pid % NB
    num_programs = tl.num_programs(0)
    s_base = S + batch * K * K
    w_base = W + batch * K
    idx = tl.arange(0, BJ)
    cidx = tl.arange(0, BLOCK_C)
    rows = jb * BJ + idx
    generation = 0

    i = 0
    while i < K - 1:
        L0 = i + 1  # trailing start; v lives on indices [L0, K)
        # ---- Householder scalars for column i (redundant, identical) ----
        acc = tl.zeros((), dtype=S.dtype.element_ty)
        c = L0
        while c < K:
            cc = c + cidx
            cmask = cc < K
            x = tl.load(s_base + cc * K + i, mask=cmask, other=0.0)
            acc += tl.sum(x * x, axis=0)
            c += BLOCK_C
        sigma = tl.sqrt(acc)
        x0 = tl.load(s_base + L0 * K + i)
        alpha_r = tl.where(x0 >= 0.0, -sigma, sigma)
        vnorm2 = 2.0 * sigma * (sigma + tl.abs(x0))
        tau = tl.where(vnorm2 > 0.0, 2.0 / vnorm2, 0.0)
        if jb == 0:
            dval = tl.load(s_base + i * K + i)
            tl.store(D + batch * K + i, dval)
            tl.store(E + batch * K + i, alpha_r)

        # ---- w phase: w[r] = tau * S[r, :] . v ----
        rvalid = (rows >= L0) & (rows < K)
        w_acc = tl.zeros((BJ,), dtype=S.dtype.element_ty)
        if tau != 0.0:
            c = L0
            while c < K:
                cc = c + cidx
                cmask = cc < K
                tile = tl.load(
                    s_base + rows[:, None] * K + cc[None, :],
                    mask=rvalid[:, None] & cmask[None, :],
                    other=0.0,
                )
                vv = tl.load(s_base + cc * K + i, mask=cmask, other=0.0)
                vv = tl.where(cc == L0, x0 - alpha_r, vv)
                w_acc += tl.sum(tile * vv[None, :], axis=1)
                c += BLOCK_C
            w_acc *= tau
        tl.store(w_base + rows, w_acc, mask=rvalid)
        _grid_barrier(BARRIER, 0, generation, num_programs)
        generation += 1

        # ---- trailing update: A -= v w'^T + w' v^T, w' = w + beta*v ----
        if tau != 0.0:
            # beta = -tau/2 * (v . w); computed redundantly per block.
            dot = tl.zeros((), dtype=S.dtype.element_ty)
            c = L0
            while c < K:
                cc = c + cidx
                cmask = cc < K
                vv = tl.load(s_base + cc * K + i, mask=cmask, other=0.0)
                vv = tl.where(cc == L0, x0 - alpha_r, vv)
                ww = tl.load(w_base + cc, mask=cmask, other=0.0)
                dot += tl.sum(vv * ww, axis=0)
                c += BLOCK_C
            beta = -0.5 * tau * dot
            v_own = tl.load(s_base + rows * K + i, mask=rvalid, other=0.0)
            v_own = tl.where(rows == L0, x0 - alpha_r, v_own)
            w_own = tl.load(w_base + rows, mask=rvalid, other=0.0)
            w_own += beta * v_own
            c = L0
            while c < K:
                cc = c + cidx
                cmask = cc < K
                vc = tl.load(s_base + cc * K + i, mask=cmask, other=0.0)
                vc = tl.where(cc == L0, x0 - alpha_r, vc)
                wc = tl.load(w_base + cc, mask=cmask, other=0.0)
                wc += beta * vc
                tile = tl.load(
                    s_base + rows[:, None] * K + cc[None, :],
                    mask=rvalid[:, None] & cmask[None, :],
                    other=0.0,
                )
                tile -= v_own[:, None] * wc[None, :] + w_own[:, None] * vc[None, :]
                tl.store(
                    s_base + rows[:, None] * K + cc[None, :],
                    tile,
                    mask=rvalid[:, None] & cmask[None, :],
                )
                c += BLOCK_C
        _grid_barrier(BARRIER, 0, generation, num_programs)
        generation += 1
        i += 1

    if jb == 0:
        dval = tl.load(s_base + (K - 1) * K + (K - 1))
        tl.store(D + batch * K + (K - 1), dval)


@libentry()
@triton.jit
def _matrix_rank_herm_tridiag_blocked_kernel(
    S,
    V,
    W,
    D,
    E,
    SCRATCH,
    BARRIER,
    K: tl.constexpr,
    NB: tl.constexpr,
    NB_P: tl.constexpr,
    BJ: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_U: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    # Blocked Householder tridiagonalization (LAPACK DSYTRD), parallelized
    # over NB blocks per matrix with software grid barriers. Columns are
    # factored in panels of NB_P. Inside a panel the trailing matrix is NOT
    # updated; instead each step stores its Householder vector in V and its
    # corrected w in W, and the panel's reflection is applied afterwards as
    # one symmetric rank-2k update (S -= V W^T + W V^T) built with tl.dot.
    # Per column step:
    #   phase A  apply the deferred panel update to column i and accumulate
    #            the Householder sigma via atomic partial sums (this removes
    #            the latency-bound redundant full-column loops of the
    #            unblocked kernel),
    #   phase B  redundant Householder scalars, the S22.v GEMV, and atomic
    #            partials for the WY corrections and the beta dot product,
    #   phase C  apply corrections, scale by tau, finish w, store V/W/D/E.
    # Scratch layout per matrix: [sig | ... | dot | ... | w1 | ... | w2],
    # with every scalar/vector entry padded to its own 128-byte cache line
    # (stride LINE doubles). Thousands of fp64 atomics per column into a
    # handful of shared cache lines serialize in the L2 (~2.4 ns each);
    # padding spreads them over 2 + 2 * NB_P distinct lines.
    pid = tl.program_id(0)
    batch = pid // NB
    jb = pid % NB
    num_programs = tl.num_programs(0)
    s_base = S + batch * K * K
    v_base = V + batch * K * NB_P
    w_base = W + batch * K * NB_P
    scratch = SCRATCH + batch * (2 + 2 * NB_P) * 16
    sig_ptr = scratch
    dot_ptr = scratch + 16
    w1_ptr = scratch + 32
    w2_ptr = scratch + 32 + NB_P * 16
    idx = tl.arange(0, BJ)
    qidx = tl.arange(0, NB_P)
    qpad = qidx * 16
    cidx = tl.arange(0, BLOCK_C)
    uidx = tl.arange(0, BLOCK_U)
    vvidx = tl.arange(0, BLOCK_V)
    rows = jb * BJ + idx
    generation = 0

    j0 = 0
    while j0 < K - 1:
        p_end = tl.minimum(NB_P, K - 1 - j0)
        p = 0
        while p < p_end:
            i = j0 + p
            L0 = i + 1
            qm = qidx < p

            # ---- phase A: deferred column update + sigma partial ----
            # Tensor ops masked by runtime values (qm) are drastically slow
            # in Triton (~20 us/column measured), so every panel-column mask
            # is applied as tl.where on unmasked/static-masked loads; stale
            # entries for q >= p are finite (the launcher zero-initializes
            # the V/W buffers) and get zeroed by the where.
            rvalid_a = (rows >= i) & (rows < K)
            col = tl.load(s_base + rows * K + i, mask=rvalid_a, other=0.0)
            if p > 0:
                v_row_i = tl.where(qm, tl.load(v_base + i * NB_P + qidx), 0.0)
                w_row_i = tl.where(qm, tl.load(w_base + i * NB_P + qidx), 0.0)
                v_own_a = tl.load(
                    v_base + rows[:, None] * NB_P + qidx[None, :],
                    mask=(rows < K)[:, None],
                    other=0.0,
                )
                w_own_a = tl.load(
                    w_base + rows[:, None] * NB_P + qidx[None, :],
                    mask=(rows < K)[:, None],
                    other=0.0,
                )
                col -= tl.sum(v_own_a * w_row_i[None, :], axis=1) + tl.sum(
                    w_own_a * v_row_i[None, :], axis=1
                )
                tl.store(s_base + rows * K + i, col, mask=rvalid_a)
            sig_part = tl.sum(
                tl.where((rows >= L0) & rvalid_a, col * col, 0.0), axis=0
            )
            tl.atomic_add(sig_ptr, sig_part, sem="relaxed")
            if jb == 0:
                zero = tl.zeros((), dtype=S.dtype.element_ty)
                tl.store(dot_ptr, zero)
                tl.store(w1_ptr + qpad, tl.zeros((NB_P,), dtype=S.dtype.element_ty))
                tl.store(w2_ptr + qpad, tl.zeros((NB_P,), dtype=S.dtype.element_ty))
            _grid_barrier(BARRIER, 0, generation, num_programs)
            generation += 1

            # ---- phase B: scalars + GEMV + correction partials ----
            sigma = tl.sqrt(tl.load(sig_ptr))
            x0 = tl.load(s_base + L0 * K + i)
            alpha_r = tl.where(x0 >= 0.0, -sigma, sigma)
            vnorm2 = 2.0 * sigma * (sigma + tl.abs(x0))
            tau = tl.where(vnorm2 > 0.0, 2.0 / vnorm2, 0.0)
            if jb == 0:
                tl.store(D + batch * K + i, tl.load(s_base + i * K + i))
                tl.store(E + batch * K + i, alpha_r)
            rvalid = (rows >= L0) & (rows < K)
            v_own = tl.load(s_base + rows * K + i, mask=rvalid, other=0.0)
            v_own = tl.where(rows == L0, x0 - alpha_r, v_own)
            # Accumulate element-wise across tiles and reduce once at the
            # end: a tl.sum inside the loop forces a block-wide barrier per
            # tile, which fences the next tile's load and serializes the
            # whole GEMV on memory latency.
            acc = tl.zeros((BJ, BLOCK_C), dtype=S.dtype.element_ty)
            c = L0
            while c < K:
                cc = c + cidx
                cmask = cc < K
                tile = tl.load(
                    s_base + rows[:, None] * K + cc[None, :],
                    mask=rvalid[:, None] & cmask[None, :],
                    other=0.0,
                )
                vc = tl.load(s_base + cc * K + i, mask=cmask, other=0.0)
                vc = tl.where(cc == L0, x0 - alpha_r, vc)
                acc += tile * vc[None, :]
                c += BLOCK_C
            w_raw = tl.sum(acc, axis=1)
            v_own_q = tl.load(
                v_base + rows[:, None] * NB_P + qidx[None, :],
                mask=(rows < K)[:, None],
                other=0.0,
            )
            w_own_q = tl.load(
                w_base + rows[:, None] * NB_P + qidx[None, :],
                mask=(rows < K)[:, None],
                other=0.0,
            )
            if p > 0:
                # Unmasked atomics with zeroed payloads: a runtime mask on a
                # tensor atomic makes Triton emit serialized per-element
                # predicated updates (~20 us/column measured); zero payloads
                # are harmless because phase A re-zeroes the accumulators.
                tl.atomic_add(
                    w1_ptr + qpad,
                    tl.where(qm, tl.sum(w_own_q * v_own[:, None], axis=0), 0.0),
                    sem="relaxed",
                )
                tl.atomic_add(
                    w2_ptr + qpad,
                    tl.where(qm, tl.sum(v_own_q * v_own[:, None], axis=0), 0.0),
                    sem="relaxed",
                )
            tl.atomic_add(dot_ptr, tl.sum(v_own * w_raw, axis=0), sem="relaxed")
            _grid_barrier(BARRIER, 0, generation, num_programs)
            generation += 1

            # ---- phase C: corrections + final w + stores ----
            # beta = -(tau^2/2) * v^T A_p v, computed without a second
            # global reduction: v^T A_p v = v^T S v - 2 (V^T v).(W^T v),
            # where v^T S v is the atomic partial accumulated in phase B.
            # w1/w2 are loaded unmasked: phase A zeroed every entry, so the
            # q >= p slots are already zero. The V/W tiles are reloaded here
            # with a static bounds mask instead of being carried across the
            # barrier in registers.
            dot = tl.load(dot_ptr)
            if p > 0:
                w1 = tl.load(w1_ptr + qpad)
                w2 = tl.load(w2_ptr + qpad)
                v_own_qc = tl.load(
                    v_base + rows[:, None] * NB_P + qidx[None, :],
                    mask=(rows < K)[:, None],
                    other=0.0,
                )
                w_own_qc = tl.load(
                    w_base + rows[:, None] * NB_P + qidx[None, :],
                    mask=(rows < K)[:, None],
                    other=0.0,
                )
                w_raw -= tl.sum(v_own_qc * w1[None, :], axis=1) + tl.sum(
                    w_own_qc * w2[None, :], axis=1
                )
                dot -= 2.0 * tl.sum(w1 * w2, axis=0)
            w_fin = tau * w_raw
            w_fin += (-0.5 * tau * tau * dot) * v_own
            tl.store(w_base + rows * NB_P + p, w_fin, mask=rvalid)
            tl.store(v_base + rows * NB_P + p, v_own, mask=rvalid)
            if jb == 0:
                tl.store(sig_ptr, tl.zeros((), dtype=S.dtype.element_ty))
            _grid_barrier(BARRIER, 0, generation, num_programs)
            generation += 1
            p += 1

        # ---- trailing symmetric rank-2k update: S -= V W^T + W V^T ----
        t0 = j0 + p_end
        if t0 < K:
            qm2 = qidx < p_end
            rt = jb * BLOCK_U
            while rt < K:
                rows2 = rt + uidx
                rmask2 = (rows2 >= t0) & (rows2 < K)
                v_r = tl.where(
                    qm2[None, :],
                    tl.load(
                        v_base + rows2[:, None] * NB_P + qidx[None, :],
                        mask=rmask2[:, None],
                        other=0.0,
                    ),
                    0.0,
                )
                w_r = tl.where(
                    qm2[None, :],
                    tl.load(
                        w_base + rows2[:, None] * NB_P + qidx[None, :],
                        mask=rmask2[:, None],
                        other=0.0,
                    ),
                    0.0,
                )
                c = t0
                while c < K:
                    cc = c + vvidx
                    cmask = cc < K
                    v_c = tl.where(
                        qm2[None, :],
                        tl.load(
                            v_base + cc[:, None] * NB_P + qidx[None, :],
                            mask=cmask[:, None],
                            other=0.0,
                        ),
                        0.0,
                    )
                    w_c = tl.where(
                        qm2[None, :],
                        tl.load(
                            w_base + cc[:, None] * NB_P + qidx[None, :],
                            mask=cmask[:, None],
                            other=0.0,
                        ),
                        0.0,
                    )
                    if tl.constexpr(S.dtype.element_ty == tl.float64):
                        # fp64 tl.dot is miscompiled on some vendor backends
                        # (MetaX/Hygon: wrong results for every block shape
                        # and num_warps variant measured), so accumulate the
                        # rank-2k update as per-column outer products. The
                        # v_r/w_r/v_c/w_c tiles above are dead in this branch
                        # and get eliminated.
                        upd = tl.zeros((BLOCK_U, BLOCK_V), dtype=S.dtype.element_ty)
                        q = 0
                        while q < p_end:
                            v_rq = tl.load(
                                v_base + rows2 * NB_P + q, mask=rmask2, other=0.0
                            )
                            w_rq = tl.load(
                                w_base + rows2 * NB_P + q, mask=rmask2, other=0.0
                            )
                            v_cq = tl.load(
                                v_base + cc * NB_P + q, mask=cmask, other=0.0
                            )
                            w_cq = tl.load(
                                w_base + cc * NB_P + q, mask=cmask, other=0.0
                            )
                            upd += (
                                v_rq[:, None] * w_cq[None, :]
                                + w_rq[:, None] * v_cq[None, :]
                            )
                            q += 1
                    else:
                        upd = tl.dot(
                            v_r, tl.trans(w_c), input_precision="ieee"
                        ) + tl.dot(w_r, tl.trans(v_c), input_precision="ieee")
                    tile = tl.load(
                        s_base + rows2[:, None] * K + cc[None, :],
                        mask=rmask2[:, None] & cmask[None, :],
                        other=0.0,
                    )
                    tl.store(
                        s_base + rows2[:, None] * K + cc[None, :],
                        tile - upd,
                        mask=rmask2[:, None] & cmask[None, :],
                    )
                    c += BLOCK_V
                rt += NB * BLOCK_U
        _grid_barrier(BARRIER, 0, generation, num_programs)
        generation += 1
        j0 += p_end

    if jb == 0:
        tl.store(
            D + batch * K + (K - 1), tl.load(s_base + (K - 1) * K + (K - 1))
        )


@libentry()
@triton.jit
def _matrix_rank_bidiag_init_kernel(
    A,
    T,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    LDA: tl.constexpr,
    TALL: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Build the column-major tall work matrix T (K x ROWS, leading stride
    # ROWS) from the row-major input: tall inputs are copied, wide inputs
    # are transposed (singular values are invariant under transposition).
    batch = tl.program_id(0)
    row_block = tl.program_id(1)
    rows = row_block * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_C)
    a_base = A + batch * K * ROWS
    t_base = T + batch * K * ROWS
    col = 0
    while col < K:
        cc = col + cols
        mask = (rows[:, None] < ROWS) & (cc[None, :] < K)
        if TALL:
            values = tl.load(a_base + rows[:, None] * LDA + cc[None, :], mask=mask, other=0.0)
        else:
            values = tl.load(a_base + cc[None, :] * LDA + rows[:, None], mask=mask, other=0.0)
        tl.store(
            t_base + cc[None, :] * ROWS + rows[:, None],
            values.to(T.dtype.element_ty),
            mask=mask,
        )
        col += BLOCK_C


@libentry()
@triton.jit
def _matrix_rank_bidiag_kernel(
    T,
    WL,
    WR,
    D,
    E,
    BARRIER,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    NB: tl.constexpr,
    BJ: tl.constexpr,
    BJR: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Unblocked two-sided Householder bidiagonalization (LAPACK DGEBD2) of
    # the column-major tall work matrix T (K x ROWS, leading stride ROWS),
    # parallelized over NB blocks per matrix with software grid barriers.
    # Block jb owns columns [jb*BJ, ...) for the left-reflector phases and
    # rows [jb*BJR, ...) for the right-reflector phases, so reductions
    # always run along the contiguous row axis. Householder scalars are
    # recomputed redundantly and bit-identically in every block, which
    # removes any cross-block reduction for them: v is never materialized
    # because for the left reflector v[r] = T[r, i] for r > i and
    # v[i] = x0 - alpha, and for the right reflector v[c] = T[i, c] for
    # c > i + 1 and v[i + 1] = x0 - alpha. Four barriers per step: after
    # the left GEMV, the left trailing update, the right GEMV, and the
    # right trailing update.
    pid = tl.program_id(0)
    batch = pid // NB
    jb = pid % NB
    num_programs = tl.num_programs(0)
    t_base = T + batch * K * ROWS
    wl_base = WL + batch * K
    wr_base = WR + batch * ROWS
    idx = tl.arange(0, BJ)
    ridx = tl.arange(0, BJR)
    cidx = tl.arange(0, BLOCK_C)
    cols_own = jb * BJ + idx
    rows_own = jb * BJR + ridx
    generation = 0

    i = 0
    while i < K:
        # ---- left Householder scalars for column i (redundant) ----
        lacc = tl.zeros((BLOCK_C,), dtype=T.dtype.element_ty)
        r = i
        while r < ROWS:
            rr = r + cidx
            x = tl.load(t_base + i * ROWS + rr, mask=rr < ROWS, other=0.0)
            lacc += x * x
            r += BLOCK_C
        sigma = tl.sqrt(tl.sum(lacc, axis=0))
        x0 = tl.load(t_base + i * ROWS + i)
        alpha = tl.where(x0 >= 0.0, -sigma, sigma)
        vnorm2 = 2.0 * sigma * (sigma + tl.abs(x0))
        tau = tl.where(vnorm2 > 0.0, 2.0 / vnorm2, 0.0)
        if jb == 0:
            tl.store(D + batch * K + i, alpha)

        # ---- left GEMV: w[c] = sum_r T[c, r] * v[r], c > i ----
        # Element-wise accumulation across tiles, one reduction at the end:
        # a tl.sum inside the loop forces a block-wide barrier per tile,
        # which fences the next tile's load and serializes the whole GEMV.
        cvalid = (cols_own > i) & (cols_own < K)
        wl = tl.zeros((BJ,), dtype=T.dtype.element_ty)
        if tau != 0.0:
            gacc = tl.zeros((BJ, BLOCK_C), dtype=T.dtype.element_ty)
            r = i
            while r < ROWS:
                rr = r + cidx
                rmask = rr < ROWS
                vv = tl.load(t_base + i * ROWS + rr, mask=rmask, other=0.0)
                vv = tl.where(rr == i, x0 - alpha, vv)
                tile = tl.load(
                    t_base + cols_own[:, None] * ROWS + rr[None, :],
                    mask=cvalid[:, None] & rmask[None, :],
                    other=0.0,
                )
                gacc += tile * vv[None, :]
                r += BLOCK_C
            wl = tl.sum(gacc, axis=1)
        tl.store(wl_base + cols_own, wl, mask=cvalid)
        _grid_barrier(BARRIER, 0, generation, num_programs)
        generation += 1

        # ---- left trailing update: T[c, r] -= tau * v[r] * w[c] ----
        if tau != 0.0:
            wtc = tl.load(wl_base + cols_own, mask=cvalid, other=0.0) * tau
            r = i
            while r < ROWS:
                rr = r + cidx
                rmask = rr < ROWS
                vv = tl.load(t_base + i * ROWS + rr, mask=rmask, other=0.0)
                vv = tl.where(rr == i, x0 - alpha, vv)
                tile = tl.load(
                    t_base + cols_own[:, None] * ROWS + rr[None, :],
                    mask=cvalid[:, None] & rmask[None, :],
                    other=0.0,
                )
                tile -= wtc[:, None] * vv[None, :]
                tl.store(
                    t_base + cols_own[:, None] * ROWS + rr[None, :],
                    tile,
                    mask=cvalid[:, None] & rmask[None, :],
                )
                r += BLOCK_C
        _grid_barrier(BARRIER, 0, generation, num_programs)
        generation += 1

        if i < K - 1:
            # ---- right Householder scalars for row i (redundant) ----
            racc = tl.zeros((BLOCK_C,), dtype=T.dtype.element_ty)
            c = i + 1
            while c < K:
                cc = c + cidx
                x = tl.load(t_base + cc * ROWS + i, mask=cc < K, other=0.0)
                racc += x * x
                c += BLOCK_C
            sigma_r = tl.sqrt(tl.sum(racc, axis=0))
            x0r = tl.load(t_base + (i + 1) * ROWS + i)
            alpha_r = tl.where(x0r >= 0.0, -sigma_r, sigma_r)
            vnorm2r = 2.0 * sigma_r * (sigma_r + tl.abs(x0r))
            tau_r = tl.where(vnorm2r > 0.0, 2.0 / vnorm2r, 0.0)
            if jb == 0:
                tl.store(E + batch * K + i, alpha_r)

            # ---- right GEMV: w[r] = sum_c T[c, r] * v[c], r > i ----
            rvalid = (rows_own > i) & (rows_own < ROWS)
            wr = tl.zeros((BJR,), dtype=T.dtype.element_ty)
            if tau_r != 0.0:
                gacc2 = tl.zeros((BJR, BLOCK_C), dtype=T.dtype.element_ty)
                c = i + 1
                while c < K:
                    cc = c + cidx
                    cmask = cc < K
                    vr = tl.load(t_base + cc * ROWS + i, mask=cmask, other=0.0)
                    vr = tl.where(cc == i + 1, x0r - alpha_r, vr)
                    tile = tl.load(
                        t_base + cc[None, :] * ROWS + rows_own[:, None],
                        mask=rvalid[:, None] & cmask[None, :],
                        other=0.0,
                    )
                    gacc2 += tile * vr[None, :]
                    c += BLOCK_C
                wr = tl.sum(gacc2, axis=1)
            tl.store(wr_base + rows_own, wr, mask=rvalid)
            _grid_barrier(BARRIER, 0, generation, num_programs)
            generation += 1

            # ---- right trailing update: T[c, r] -= tau * w[r] * v[c] ----
            if tau_r != 0.0:
                wtr = tl.load(wr_base + rows_own, mask=rvalid, other=0.0) * tau_r
                c = i + 1
                while c < K:
                    cc = c + cidx
                    cmask = cc < K
                    vr = tl.load(t_base + cc * ROWS + i, mask=cmask, other=0.0)
                    vr = tl.where(cc == i + 1, x0r - alpha_r, vr)
                    tile = tl.load(
                        t_base + cc[None, :] * ROWS + rows_own[:, None],
                        mask=rvalid[:, None] & cmask[None, :],
                        other=0.0,
                    )
                    tile -= wtr[:, None] * vr[None, :]
                    tl.store(
                        t_base + cc[None, :] * ROWS + rows_own[:, None],
                        tile,
                        mask=rvalid[:, None] & cmask[None, :],
                    )
                    c += BLOCK_C
            _grid_barrier(BARRIER, 0, generation, num_programs)
            generation += 1

        i += 1


@libentry()
@triton.jit
def _matrix_rank_gk_init_kernel(
    D,
    E,
    GD,
    GE,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Golub-Kahan tridiagonal of order 2K for the bidiagonal (D, E): zero
    # diagonal, off-diagonal [d0, e0, d1, e1, ..., d_{K-1}]. Its eigenvalues
    # are exactly +/- sigma_i of the bidiagonal matrix.
    batch = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    tl.store(
        GD + batch * 2 * K + idx,
        tl.zeros((BLOCK,), dtype=GD.dtype.element_ty),
        mask=idx < 2 * K,
    )
    jj = idx // 2
    even = (idx % 2) == 0
    dv = tl.load(D + batch * K + jj, mask=even & (idx < 2 * K - 1), other=0.0)
    ev = tl.load(E + batch * K + jj, mask=(~even) & (idx < 2 * K - 1), other=0.0)
    tl.store(
        GE + batch * 2 * K + idx,
        tl.where(even, dv, ev),
        mask=idx < 2 * K - 1,
    )


@triton.jit
def _sturm_count_less(D, E2H, E2L, base, K: tl.constexpr, x):
    # Number of eigenvalues of the tridiagonal T = diag(d) + diag(e, +/-1)
    # that are <= x, via the qd recurrence (LAPACK DLANEG convention: a zero
    # pivot is replaced by a tiny negative value, keeping the count
    # consistent for clustered spectra). The recurrence runs in
    # double-single arithmetic: native fp64 division is a slow software
    # sequence on this target and would dominate the O(k) chain.
    xh = x.to(tl.float32)
    xl = (x - xh.to(tl.float64)).to(tl.float32)
    d0 = tl.load(D + base)
    dh = d0.to(tl.float32)
    dl = (d0 - dh.to(tl.float64)).to(tl.float32)
    qh, ql = _df64_add(dh, dl, -xh, -xl)
    zero_q = (qh == 0.0) & (ql == 0.0)
    qh = tl.where(zero_q, -1.1754944e-38, qh)
    ql = tl.where(zero_q, 0.0, ql)
    neg = tl.where(qh < 0.0, 1, 0)
    i = 1
    while i < K:
        di = tl.load(D + base + i)
        dh = di.to(tl.float32)
        dl = (di - dh.to(tl.float64)).to(tl.float32)
        th, t_l = _df64_add(dh, dl, -xh, -xl)
        e2h = tl.load(E2H + base + i - 1)
        e2l = tl.load(E2L + base + i - 1)
        rh, rl = _df64_div_ds(e2h, e2l, qh, ql)
        qh, ql = _df64_add(th, t_l, -rh, -rl)
        zero_q = (qh == 0.0) & (ql == 0.0)
        qh = tl.where(zero_q, -1.1754944e-38, qh)
        ql = tl.where(zero_q, 0.0, ql)
        neg += tl.where(qh < 0.0, 1, 0)
        i += 1
    return neg


@libentry()
@triton.jit
def _matrix_rank_sturm_rank_kernel(
    D,
    E,
    ATOL,
    RTOL,
    OUT,
    E2H,
    E2L,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BISECT_ITERS: tl.constexpr,
    GK: tl.constexpr,
):
    # Rank of a symmetric matrix from its tridiagonal form:
    #   rank = #{lambda > tol} + #{lambda < -tol},
    #   tol  = max(atol, rtol * sigma_max),
    # where sigma_max = max |lambda| comes from Gershgorin bounds, refined
    # by bisection on the Sturm count only when the rank actually depends
    # on the refinement (otherwise the cheap two-sided evaluation is
    # already exact).
    batch = tl.program_id(0)
    idx = tl.arange(0, BLOCK_K)
    base = batch * K

    d = tl.load(D + base + idx, mask=idx < K, other=0.0).to(tl.float64)
    e_cur = tl.load(E + base + idx, mask=idx < K - 1, other=0.0).to(tl.float64)
    e_prev = tl.load(
        E + base + idx - 1,
        mask=(idx >= 1) & (idx < K),
        other=0.0,
    )
    gershgorin = tl.abs(d) + tl.abs(e_cur) + tl.abs(e_prev)
    hi = tl.max(gershgorin, axis=0)
    dmax = tl.max(d, axis=0)
    dmin = tl.min(d, axis=0)

    # Precompute e^2 in double-single form (shared by every count).
    eh = e_cur.to(tl.float32)
    el = (e_cur - eh.to(tl.float64)).to(tl.float32)
    e2h, e2l = _df64_mul_ds(eh, el, eh, el)
    tl.store(E2H + base + idx, e2h, mask=idx < K - 1)
    tl.store(E2L + base + idx, e2l, mask=idx < K - 1)

    atol = tl.load(ATOL + batch).to(tl.float64)
    rtol = tl.load(RTOL + batch).to(tl.float64)

    if hi == 0.0:
        # The tridiagonal (and hence the matrix) is exactly zero.
        tl.store(OUT + batch, tl.zeros((), dtype=tl.int64))
    else:
        if GK:
            # Zero diagonal (Golub-Kahan form): the largest |e| is a lower
            # bound of sigma_max by 2x2 interlacing.
            sigma_lo = tl.max(tl.abs(e_cur), axis=0)
        else:
            sigma_lo = tl.maximum(tl.abs(dmax), tl.abs(dmin))
        tol_lo = tl.maximum(atol, rtol * sigma_lo)
        tol_hi = tl.maximum(atol, rtol * hi)
        cnt_lo = _sturm_count_less(D, E2H, E2L, base, K, tol_lo)
        cnt_hi = _sturm_count_less(D, E2H, E2L, base, K, tol_hi)
        if GK:
            # Eigenvalues come in +/- sigma pairs, so the positive side
            # alone gives #{sigma > tol} without parity issues.
            rank_lo = K - cnt_lo
            rank_hi = K - cnt_hi
        else:
            rank_lo = (K - cnt_lo) + _sturm_count_less(D, E2H, E2L, base, K, -tol_lo)
            rank_hi = (K - cnt_hi) + _sturm_count_less(D, E2H, E2L, base, K, -tol_hi)
        rank = rank_lo
        if rank_lo != rank_hi:
            # The rank depends on sigma_max: refine it by bisection.
            # lambda_max in [dmax, hi_pad] (count < K ... count == K).
            lo = dmax
            hi_p = hi * (1.0 + 1e-9) + 1e-292
            it = 0
            while it < BISECT_ITERS:
                mid = 0.5 * (lo + hi_p)
                cnt = _sturm_count_less(D, E2H, E2L, base, K, mid)
                if cnt >= K:
                    hi_p = mid
                else:
                    lo = mid
                it += 1
            lmax = 0.5 * (lo + hi_p)
            if GK:
                sigma_max = lmax
            else:
                # lambda_min in [-hi_pad, dmin] (count == 0 ... count > 0).
                lo = -(hi * (1.0 + 1e-9) + 1e-292)
                hi_p = dmin
                it = 0
                while it < BISECT_ITERS:
                    mid = 0.5 * (lo + hi_p)
                    cnt = _sturm_count_less(D, E2H, E2L, base, K, mid)
                    if cnt > 0:
                        hi_p = mid
                    else:
                        lo = mid
                    it += 1
                lmin = 0.5 * (lo + hi_p)
                sigma_max = tl.maximum(tl.abs(lmax), tl.abs(lmin))
            tol = tl.maximum(atol, rtol * sigma_max)
            cnt = _sturm_count_less(D, E2H, E2L, base, K, tol)
            if GK:
                rank = K - cnt
            else:
                rank = (K - cnt) + _sturm_count_less(D, E2H, E2L, base, K, -tol)
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


def _launch_herm_tridiag_rank(matrix, atol_tensor, rtol_tensor, out, k, batch_count, input):
    # Non-iterative hermitian path: symmetrize into an fp64 work matrix,
    # tridiagonalize with the multi-block Householder kernel, then count
    # eigenvalues outside [-tol, tol] with Sturm sequences.
    device = input.device
    work_dtype = input.dtype
    sym = torch.empty((batch_count, k, k), dtype=work_dtype, device=device)
    diag = torch.empty((batch_count, k), dtype=work_dtype, device=device)
    offdiag = torch.empty((batch_count, k), dtype=work_dtype, device=device)
    work_w = torch.empty((batch_count, k), dtype=work_dtype, device=device)
    e2_hi = torch.empty((batch_count, k), dtype=torch.float32, device=device)
    e2_lo = torch.empty((batch_count, k), dtype=torch.float32, device=device)
    _matrix_rank_herm_tridiag_init_kernel[(batch_count, triton.cdiv(k, 64))](
        matrix,
        sym,
        K=k,
        BLOCK_R=64,
        BLOCK_C=64,
        num_warps=4,
    )
    bj = 8
    nb = triton.cdiv(k, bj)
    barrier = torch.zeros(1, dtype=torch.int32, device=device)
    if k >= _HERM_TRIDIAG_BLOCKED_MIN_K:
        # Blocked WY path: panel factorization + BLAS3 trailing updates.
        nb_p = 32
        # Zero-init (not empty): the kernel relies on stale panel slots
        # q >= p being finite, because it multiplies unmasked V/W tile loads
        # against tl.where-zeroed values and Inf/NaN * 0 would poison the
        # sums.
        v_buf = torch.zeros((batch_count, k, nb_p), dtype=work_dtype, device=device)
        w_buf = torch.zeros((batch_count, k, nb_p), dtype=work_dtype, device=device)
        scratch = torch.zeros(
            (batch_count, (2 + 2 * nb_p) * 16), dtype=work_dtype, device=device
        )
        # The kernel synchronizes with a software grid barrier, so the whole
        # launch must stay co-resident; one block per SM is always safe.
        chunk = max(1, _sm_count(device) // nb)
        for batch_start in range(0, batch_count, chunk):
            current = min(chunk, batch_count - batch_start)
            barrier.zero_()
            scratch[batch_start : batch_start + current].zero_()
            _matrix_rank_herm_tridiag_blocked_kernel[(current * nb,)](
                sym[batch_start:],
                v_buf[batch_start:],
                w_buf[batch_start:],
                diag[batch_start:],
                offdiag[batch_start:],
                scratch[batch_start:],
                barrier,
                K=k,
                NB=nb,
                NB_P=nb_p,
                BJ=bj,
                BLOCK_C=128,
                BLOCK_U=16,
                BLOCK_V=64,
                num_warps=4,
            )
    else:
        # The tridiagonalization kernel synchronizes with a software grid
        # barrier, so the whole launch must stay co-resident.
        chunk = max(1, (_sm_count(device) * _RESIDENT_BLOCKS_PER_SM * 2) // nb)
        for batch_start in range(0, batch_count, chunk):
            current = min(chunk, batch_count - batch_start)
            barrier.zero_()
            _matrix_rank_herm_tridiag_kernel[(current * nb,)](
                sym[batch_start:],
                work_w[batch_start:],
                diag[batch_start:],
                offdiag[batch_start:],
                barrier,
                K=k,
                NB=nb,
                BJ=bj,
                BLOCK_C=64,
                num_warps=2,
            )
    _matrix_rank_sturm_rank_kernel[(batch_count,)](
        diag,
        offdiag,
        atol_tensor,
        rtol_tensor,
        out,
        e2_hi,
        e2_lo,
        K=k,
        BLOCK_K=triton.next_power_of_2(k),
        BISECT_ITERS=32,
        num_warps=1,
        GK=False,
        enable_fp_fusion=False,
    )


def _launch_bidiag_rank(matrix, atol_tensor, rtol_tensor, out, m, n, k, rows, batch_count, input):
    # Non-hermitian path past the Jacobi size limits: transpose-copy into a
    # column-major tall work matrix, reduce to bidiagonal form with
    # two-sided Householder reflections, then count singular values above
    # the tolerance with Sturm sequences on the Golub-Kahan tridiagonal.
    device = input.device
    work_dtype = input.dtype
    work = torch.empty((batch_count, k, rows), dtype=work_dtype, device=device)
    w_left = torch.empty((batch_count, k), dtype=work_dtype, device=device)
    w_right = torch.empty((batch_count, rows), dtype=work_dtype, device=device)
    diag = torch.empty((batch_count, k), dtype=work_dtype, device=device)
    offdiag = torch.empty((batch_count, k), dtype=work_dtype, device=device)
    gk_diag = torch.empty((batch_count, 2 * k), dtype=torch.float64, device=device)
    # Keep the per-batch stride at 2K (one slack entry): the Sturm kernel
    # indexes the off-diagonal with stride K == 2k.
    gk_off = torch.empty((batch_count, 2 * k), dtype=work_dtype, device=device)
    e2_hi = torch.empty((batch_count, 2 * k), dtype=torch.float32, device=device)
    e2_lo = torch.empty((batch_count, 2 * k), dtype=torch.float32, device=device)
    _matrix_rank_bidiag_init_kernel[(batch_count, triton.cdiv(rows, 64))](
        matrix,
        work,
        K=k,
        ROWS=rows,
        LDA=n,
        TALL=m >= n,
        BLOCK_R=64,
        BLOCK_C=64,
        num_warps=4,
    )
    sm = _sm_count(device)
    # The kernel synchronizes with a software grid barrier, so the whole
    # launch must stay co-resident; keep the block count within two blocks
    # per SM. Column and row partitions are sized independently so tall
    # work matrices (rows >> k) still fill the machine without blowing up
    # the per-block register tiles.
    bj = max(8, triton.next_power_of_2(-(-k // (2 * sm))))
    bjr = max(8, triton.next_power_of_2(-(-rows // (2 * sm))))
    nb_row = triton.cdiv(rows, bjr)
    # When the row partition alone fills the machine, shrink the column tile
    # so the left-reflector phases spread over more blocks; nb is unchanged,
    # so co-residency is preserved. Each column's GEMV reduction still runs
    # inside a single block, so results are bit-identical.
    while bj > 2 and triton.cdiv(k, bj // 2) <= nb_row:
        bj //= 2
    nb = max(triton.cdiv(k, bj), nb_row)
    barrier = torch.zeros(1, dtype=torch.int32, device=device)
    chunk = max(1, (sm * _RESIDENT_BLOCKS_PER_SM) // nb)
    for batch_start in range(0, batch_count, chunk):
        current = min(chunk, batch_count - batch_start)
        barrier.zero_()
        _matrix_rank_bidiag_kernel[(current * nb,)](
            work[batch_start:],
            w_left[batch_start:],
            w_right[batch_start:],
            diag[batch_start:],
            offdiag[batch_start:],
            barrier,
            K=k,
            ROWS=rows,
            NB=nb,
            BJ=bj,
            BJR=bjr,
            BLOCK_C=128,
            num_warps=4,
        )
    _matrix_rank_gk_init_kernel[(batch_count,)](
        diag,
        offdiag,
        gk_diag,
        gk_off,
        K=k,
        BLOCK=triton.next_power_of_2(2 * k),
        num_warps=4,
    )
    _matrix_rank_sturm_rank_kernel[(batch_count,)](
        gk_diag,
        gk_off,
        atol_tensor,
        rtol_tensor,
        out,
        e2_hi,
        e2_lo,
        K=2 * k,
        BLOCK_K=triton.next_power_of_2(2 * k),
        BISECT_ITERS=32,
        num_warps=1,
        GK=True,
        enable_fp_fusion=False,
    )


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
    is_fp64 = input.dtype == torch.float64
    herm_tridiag = hermitian and k >= (
        _HERM_TRIDIAG_MIN_K_FP64 if is_fp64 else _HERM_TRIDIAG_MIN_K_FP32
    )
    # Non-hermitian matrices past either Jacobi limit (k or rows) use the
    # bidiagonalization path. Hermitian inputs are square, so k == rows and
    # the tridiagonal path covers any size above its threshold.
    use_bidiag = (not hermitian) and (
        k > _BLOCKED_JACOBI_MAX_K or rows > _JACOBI_MAX_ROWS
    )

    batch_count = input.numel() // (m * n)
    matrix = input.contiguous().reshape(batch_count, m, n)
    out = torch.empty(output_shape, dtype=torch.int64, device=input.device)
    block_r = triton.next_power_of_2(rows)
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
        elif herm_tridiag:
            _launch_herm_tridiag_rank(
                matrix, atol_tensor, rtol_tensor, out, k, batch_count, input
            )
        elif use_bidiag:
            _launch_bidiag_rank(
                matrix,
                atol_tensor,
                rtol_tensor,
                out,
                m,
                n,
                k,
                rows,
                batch_count,
                input,
            )
        elif (
            rows <= _FUSED_JACOBI_MAX_ROWS
            and (
                (
                    is_fp64
                    and k <= _FUSED_JACOBI_MAX_K_FP64
                    and (k <= 16 or rows <= _FUSED_JACOBI_WIDE_MAX_ROWS)
                )
                or (
                    not is_fp64
                    and k <= _FUSED_JACOBI_MAX_K
                    and (k <= 32 or rows <= _FUSED_JACOBI_WIDE_MAX_ROWS)
                )
            )
        ):
            work = torch.empty(
                (batch_count, k, rows),
                dtype=input.dtype,
                device=input.device,
            )
            round_size = k if k % 2 == 0 else k + 1
            pairs = round_size // 2
            block_p = triton.next_power_of_2(pairs)
            block_k = triton.next_power_of_2(k)
            block_c = min(256, block_r)
            sweeps = _jacobi_sweeps(k, is_fp64)
            fused_warps = 8 if block_p * block_c >= 8192 else 4
            _matrix_rank_fused_jacobi_kernel[(batch_count,)](
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
                ROUND=round_size,
                PAIRS=pairs,
                BLOCK_R=block_r,
                BLOCK_P=block_p,
                BLOCK_K=block_k,
                BLOCK_C=block_c,
                SWEEPS=sweeps,
                REL_EPS=relative_epsilon,
                ABS_EPS=absolute_epsilon,
                num_warps=fused_warps,
                enable_fp_fusion=not is_fp64,
            )
        else:
            use_df64 = is_fp64 and k > _NATIVE_FP64_JACOBI_MAX_K
            if use_df64:
                work = torch.empty(
                    (2, batch_count, k, rows),
                    dtype=torch.float32,
                    device=input.device,
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
            if use_df64:
                _matrix_rank_df64_init_kernel[(batch_count, k)](
                    matrix,
                    work[0],
                    work[1],
                    M=m,
                    N=n,
                    K=k,
                    ROWS=rows,
                    TALL=m >= n,
                    HERMITIAN=hermitian,
                    BLOCK_R=block_r,
                    num_warps=num_warps,
                )
            else:
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
            pairs = round_size // 2
            if use_df64:
                # The df64 sweep kernel runs one pair per block with 2
                # warps: it uses ~107 registers per thread, so four
                # blocks per SM are co-resident with wide margin. One
                # pair per block keeps the per-step dependency chain
                # shortest and avoids two blocks timesharing an SM,
                # whose stretched chain is paid at every grid barrier.
                resident_per_sm = 4
                pairs_per_block = 1
                sweep_warps = 2
            else:
                # Pick PPB so the launch fills the machine: fewer blocks
                # (larger PPB) only once the pair count exceeds the
                # resident-block bound. Small column tiles keep most lanes
                # of a full 4-warp block idle, so scale the warps down.
                resident_per_sm = _RESIDENT_BLOCKS_PER_SM
                if block_r <= 32:
                    sweep_warps = 1
                elif block_r <= 64:
                    sweep_warps = 2
                else:
                    sweep_warps = 4
            max_resident = _sm_count(input.device) * resident_per_sm
            if not use_df64:
                pairs_per_block = max(
                    1,
                    min(_JACOBI_PAIRS_PER_BLOCK, -(-pairs // max_resident)),
                )
            pair_blocks = triton.cdiv(pairs, pairs_per_block)
            # With one pair slot per block the step schedule only couples
            # adjacent blocks, so a cheap neighbor barrier replaces the
            # global grid barrier between steps (the stability check still
            # uses the global barrier). It only pays off on large grids,
            # where the global counter's RMW serialization is significant.
            neighbor_sync = pairs_per_block == 1 and pair_blocks >= 128
            sweeps = _jacobi_sweeps(k, is_fp64)
            int_dtype = torch.int64 if is_fp64 else torch.int32
            # The sweep kernel synchronizes with a software grid barrier, so
            # every program of a launch must be co-resident. Cap each launch
            # at a conservative resident-block bound and chunk larger batches.
            batch_chunk = max(1, max_resident // pair_blocks)
            atol_flat = atol_tensor.reshape(batch_count)
            rtol_flat = rtol_tensor.reshape(batch_count)
            # Per-batch rank-stability flags, persistent across launches. A
            # batch proven stable is masked inside later launches, and once
            # every batch is stable the remaining sweeps are skipped. The
            # stop decision lives on the host (one sync per sweep block)
            # instead of inside the kernel, where the extra atomic and the
            # control-flow dependency disabled software pipelining of the
            # step loop (~2x slowdown).
            flags = torch.zeros(
                batch_count, dtype=torch.int32, device=input.device
            )
            # Fine-grained exit checks pay off once a sweep block costs
            # much more than the extra launch + sync (~50us).
            if k >= 512:
                sweep_chunk = 1
            elif k >= 128:
                sweep_chunk = 2
            else:
                sweep_chunk = 8
            batch_start = 0
            while batch_start < batch_count:
                chunk = min(batch_chunk, batch_count - batch_start)
                counters = torch.zeros(
                    1 + 2 * chunk * sweeps,
                    dtype=int_dtype,
                    device=input.device,
                )
                scratch = torch.zeros(
                    chunk * sweeps + chunk * k,
                    dtype=input.dtype,
                    device=input.device,
                )
                barrier = counters[0:1]
                max_alpha = counters[1 : 1 + chunk * sweeps]
                min_margin = counters[1 + chunk * sweeps :]
                # atomic_min over bit-cast non-negative floats must
                # start from the bit pattern of +inf, not zero.
                min_margin.fill_(
                    9218868437227405312 if is_fp64 else 2139095040
                )
                e2_energy = scratch[: chunk * sweeps]
                alpha_values = scratch[chunk * sweeps :]
                nflags = torch.zeros(
                    chunk * pair_blocks, dtype=torch.int32, device=input.device
                )
                sweep_base = 0
                barrier_base = 0
                while sweep_base < sweeps:
                    n_sweeps = min(sweep_chunk, sweeps - sweep_base)
                    if use_df64:
                        _matrix_rank_jacobi_sweep_df64_kernel[
                            (chunk * pair_blocks,)
                        ](
                            work[0, batch_start:],
                            work[1, batch_start:],
                            atol_flat[batch_start:],
                            rtol_flat[batch_start:],
                            barrier,
                            flags[batch_start:],
                            max_alpha,
                            min_margin,
                            e2_energy,
                            alpha_values,
                            K=k,
                            ROWS=rows,
                            ROUND=round_size,
                            PAIRS=pairs,
                            PPB=pairs_per_block,
                            BLOCK_R=block_r,
                            SWEEPS=n_sweeps,
                            SWEEP_BASE=sweep_base,
                            TOTAL_SWEEPS=sweeps,
                            BARRIER_BASE=barrier_base,
                            NFLAG=nflags,
                            NEIGHBOR=neighbor_sync,
                            REL_EPS=1.0e-12,
                            ABS_EPS=absolute_epsilon,
                            num_warps=sweep_warps,
                            # TwoSum/TwoProd error-free transformations break
                            # if mul+add pairs are contracted into fma.
                            enable_fp_fusion=False,
                        )
                    else:
                        _matrix_rank_jacobi_sweep_kernel[
                            (chunk * pair_blocks,)
                        ](
                            work[batch_start:],
                            atol_flat[batch_start:],
                            rtol_flat[batch_start:],
                            barrier,
                            flags[batch_start:],
                            max_alpha,
                            min_margin,
                            e2_energy,
                            alpha_values,
                            K=k,
                            ROWS=rows,
                            ROUND=round_size,
                            PAIRS=pairs,
                            PPB=pairs_per_block,
                            BLOCK_R=block_r,
                            SWEEPS=n_sweeps,
                            SWEEP_BASE=sweep_base,
                            TOTAL_SWEEPS=sweeps,
                            BARRIER_BASE=barrier_base,
                            NFLAG=nflags,
                            NEIGHBOR=neighbor_sync,
                            IS_FP64=is_fp64,
                            REL_EPS=relative_epsilon,
                            ABS_EPS=absolute_epsilon,
                            num_warps=sweep_warps,
                            # The float64 path computes rotation angles in
                            # double-single arithmetic; its TwoSum/TwoProd
                            # error-free transformations break if mul+add
                            # pairs are contracted into fma.
                            enable_fp_fusion=not is_fp64,
                        )
                    # The barrier counter is monotone across launches. In
                    # neighbor mode only the two check barriers of every
                    # sweep hit it; otherwise the (ROUND-1) step barriers
                    # count as well.
                    if neighbor_sync:
                        barrier_base += 2 * n_sweeps * chunk * pair_blocks
                    else:
                        barrier_base += (
                            n_sweeps * (round_size + 1) * chunk * pair_blocks
                        )
                    # One host sync per sweep block: stop once every batch
                    # of this chunk has proven rank-stable.
                    if bool(
                        (
                            flags[batch_start : batch_start + chunk] != 0
                        ).all()
                    ):
                        break
                    sweep_base += sweep_chunk
                batch_start += chunk

            if use_df64:
                _matrix_rank_df64_norm_kernel[(batch_count, k)](
                    work[0],
                    work[1],
                    singular_values,
                    K=k,
                    ROWS=rows,
                    BLOCK_R=block_r,
                    num_warps=num_warps,
                )
            else:
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
