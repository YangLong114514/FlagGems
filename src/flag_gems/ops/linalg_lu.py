import logging
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

LinalgLUResult = namedtuple("LinalgLUResult", ["P", "L", "U"])

_LU_BLOCK_MAX = 64


@libentry()
@triton.jit
def _linalg_lu_kernel(
    A,
    P,
    L,
    U,
    BATCH: tl.constexpr,
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

    a_offsets = pid * M * N + rows[:, None] * N + cols[None, :]
    a_mask = (rows[:, None] < M) & (cols[None, :] < N)
    work = tl.load(A + a_offsets, mask=a_mask, other=0.0).to(tl.float32)

    prow = tl.arange(0, BLOCK_M)
    pcol = tl.arange(0, BLOCK_M)
    perm = prow

    for j in tl.range(0, K, loop_unroll_factor=1):
        if PIVOT:
            col_j = tl.sum(tl.where(cols[:, None] == j, tl.trans(work), 0.0), axis=0)
            for t in tl.range(0, K, loop_unroll_factor=1):
                active = t < j
                l_it = tl.load(
                    L + pid * M * K + rows * K + t,
                    mask=active & (rows < M),
                    other=0.0,
                )
                u_tj = tl.load(U + pid * K * N + t * N + j, mask=active, other=0.0)
                col_j -= l_it * u_tj

            abs_col = tl.abs(col_j)
            abs_col = tl.where(rows < j, -1.0, abs_col)
            abs_col = tl.where(rows < M, abs_col, -1.0)
            pivot_val = tl.max(abs_col, axis=0)
            pivot_row = tl.min(tl.where(abs_col == pivot_val, rows, BLOCK_M), axis=0)

            row_j = tl.sum(tl.where(rows[:, None] == j, work, 0.0), axis=0)
            row_p = tl.sum(tl.where(rows[:, None] == pivot_row, work, 0.0), axis=0)
            work = tl.where((rows[:, None] == j) & (cols[None, :] < N), row_p, work)
            work = tl.where(
                (rows[:, None] == pivot_row) & (cols[None, :] < N), row_j, work
            )

            old_j = tl.load(
                L + pid * M * K + j * K + pcol, mask=pcol < j, other=0.0
            )
            old_p = tl.load(
                L + pid * M * K + pivot_row * K + pcol, mask=pcol < j, other=0.0
            )
            tl.store(L + pid * M * K + j * K + pcol, old_p, mask=pcol < j)
            tl.store(L + pid * M * K + pivot_row * K + pcol, old_j, mask=pcol < j)
            perm_j = tl.sum(tl.where(prow == j, perm, 0), axis=0)
            perm_p = tl.sum(tl.where(prow == pivot_row, perm, 0), axis=0)
            perm = tl.where(prow == j, perm_p, perm)
            perm = tl.where(prow == pivot_row, perm_j, perm)

        pivot = tl.sum(
            tl.sum(
                tl.where((rows[:, None] == j) & (cols[None, :] == j), work, 0.0),
                axis=0,
            ),
            axis=0,
        )
        for t in tl.range(0, K, loop_unroll_factor=1):
            active = t < j
            l_jt = tl.load(L + pid * M * K + j * K + t, mask=active, other=0.0)
            u_tj = tl.load(U + pid * K * N + t * N + j, mask=active, other=0.0)
            pivot -= l_jt * u_tj

        u_vals = tl.sum(tl.where(rows[:, None] == j, work, 0.0), axis=0)
        for t in tl.range(0, K, loop_unroll_factor=1):
            active = t < j
            l_jt = tl.load(L + pid * M * K + j * K + t, mask=active, other=0.0)
            u_t = tl.load(
                U + pid * K * N + t * N + cols,
                mask=active & (cols < N),
                other=0.0,
            )
            u_vals -= l_jt * u_t
        u_vals = tl.where(cols < j, 0.0, u_vals)
        tl.store(U + pid * K * N + j * N + cols, u_vals, mask=cols < N)

        l_vals = tl.sum(tl.where(cols[:, None] == j, tl.trans(work), 0.0), axis=0)
        for t in tl.range(0, K, loop_unroll_factor=1):
            active = t < j
            l_it = tl.load(
                L + pid * M * K + rows * K + t,
                mask=active & (rows < M),
                other=0.0,
            )
            u_tj = tl.load(U + pid * K * N + t * N + j, mask=active, other=0.0)
            l_vals -= l_it * u_tj
        l_vals = tl.where(rows == j, 1.0, l_vals / pivot)
        l_vals = tl.where(rows < j, 0.0, l_vals)
        tl.store(L + pid * M * K + rows * K + j, l_vals, mask=rows < M)

    if PIVOT:
        p_vals = tl.zeros((BLOCK_M, BLOCK_M), dtype=tl.float32)
        p_vals = tl.where(
            (prow[:, None] < M)
            & (pcol[None, :] < M)
            & (prow[:, None] == perm[None, :]),
            1.0,
            0.0,
        )
        tl.store(
            P + pid * M * M + prow[:, None] * M + pcol[None, :],
            p_vals,
            mask=(prow[:, None] < M) & (pcol[None, :] < M),
        )


def _linalg_lu_check(input, pivot):
    if input.dim() < 2:
        raise RuntimeError(
            f"torch.linalg.lu: Expected input to have at least 2 dimensions, got {input.dim()}"
        )
    if not input.is_cuda:
        raise NotImplementedError(
            "FlagGems linalg_lu currently supports CUDA tensors only"
        )
    if input.dtype != torch.float32:
        raise NotImplementedError(
            f"FlagGems linalg_lu currently supports float32 only, got {input.dtype}"
        )
    m, n = input.shape[-2], input.shape[-1]
    if m == 0 or n == 0:
        raise NotImplementedError(
            "FlagGems linalg_lu currently does not support empty matrices"
        )
    # if m > _LU_BLOCK_MAX or n > _LU_BLOCK_MAX:
    #     raise NotImplementedError(
    #         f"FlagGems linalg_lu currently supports matrices up to {_LU_BLOCK_MAX}x{_LU_BLOCK_MAX}, "
    #         f"got {m}x{n}"
    #     )
    if pivot not in (True, False):
        raise TypeError(f"pivot must be a bool, got {type(pivot)}")


def linalg_lu(input, *, pivot=True):
    logger.debug("GEMS LINALG_LU")
    print("GEMS LINALG_LU")
    _linalg_lu_check(input, pivot)

    input_contiguous = input.contiguous()
    batch_shape = input_contiguous.shape[:-2]
    m, n = input_contiguous.shape[-2], input_contiguous.shape[-1]
    k = min(m, n)
    batch = input_contiguous.numel() // (m * n)

    if pivot:
        p = torch.empty((*batch_shape, m, m), device=input.device, dtype=input.dtype)
    else:
        p = torch.empty((0,), device=input.device, dtype=input.dtype)
    l = torch.empty((*batch_shape, m, k), device=input.device, dtype=input.dtype)
    u = torch.empty((*batch_shape, k, n), device=input.device, dtype=input.dtype)

    kernel = _linalg_lu_kernel

    with torch_device_fn.device(input.device):
        # When pivot=False, the kernel does not access P, but we must pass a valid pointer.
        # Use a dummy tensor to avoid issues with empty tensor data_ptr.
        if pivot:
            p_arg = p
        else:
            p_arg = torch.empty(1, device=input.device, dtype=input.dtype)
        kernel[(batch,)](
            input_contiguous,
            p_arg,
            l,
            u,
            batch,
            m,
            n,
            k,
            triton.next_power_of_2(m),
            triton.next_power_of_2(n),
            pivot,
        )
    return LinalgLUResult(p, l, u)
