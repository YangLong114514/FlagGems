import logging
from collections import namedtuple

import torch
import triton

from flag_gems.ops.linalg_lu_factor import linalg_lu_factor
from flag_gems.ops.lu_unpack import (
    lu_unpack,
    lu_unpack_l_kernel,
    lu_unpack_p_kernel_large,
    lu_unpack_p_kernel_small,
    lu_unpack_u_kernel,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

LinalgLUResult = namedtuple("LinalgLUResult", ["P", "L", "U"])


# ---------------------------------------------------------------------------
# Input validation — adapted from linalg_lu_factor.py / linalg_lu_factor_ex.py
# ---------------------------------------------------------------------------


def _linalg_lu_check(input, pivot):
    if input.dim() < 2:
        raise RuntimeError(
            "torch.linalg.lu: Expected input to have at least 2 dimensions, "
            f"got {input.dim()}"
        )
    if input.dtype not in (torch.float32, torch.float64):
        raise NotImplementedError(
            "FlagGems linalg_lu currently supports float32 and float64 only, "
            f"got {input.dtype}"
        )
    m, n = input.shape[-2], input.shape[-1]
    if m == 0 or n == 0:
        raise NotImplementedError(
            "FlagGems linalg_lu currently does not support empty matrices"
        )
    if pivot not in (True, False):
        raise TypeError(f"pivot must be a bool, got {type(pivot)}")
    if not pivot and input.device.type != "cuda":
        raise NotImplementedError(
            "FlagGems linalg_lu: pivot=False is only supported on CUDA devices, "
            f"got device={input.device.type}"
        )


# ---------------------------------------------------------------------------
# Zero-copy unpack into pre-sized out tensors.
#
# Mirrors the kernel launches of flag_gems.ops.lu_unpack.lu_unpack, but
# writes directly into the provided P, L, U tensors, so the out variant
# skips the intermediate allocations and device-to-device copies.
# ---------------------------------------------------------------------------


def _lu_unpack_into(lu, pivots, P, L, U, pivot):
    batch_dims = lu.shape[:-2]
    m, n = lu.shape[-2], lu.shape[-1]
    k = min(m, n)

    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim

    pivots_shape = pivots.shape
    pivots_stride_b = pivots.stride(-2) if len(pivots_shape) > 1 else 0
    pivots_stride_k = pivots.stride(-1) if len(pivots_shape) > 0 else 0
    lu_stride_b = lu.stride(-3) if len(lu.shape) > 2 else 0
    l_stride_b = L.stride(-3) if len(batch_dims) > 0 else 0
    u_stride_b = U.stride(-3) if len(batch_dims) > 0 else 0

    with torch_device_fn.device(lu.device):
        if pivot:
            # P is pre-zeroed by the caller; only the 1s are scattered.
            p_stride_b = P.stride(-3) if len(batch_dims) > 0 else 0
            if m <= 512:
                lu_unpack_p_kernel_small[(batch_size,)](
                    pivots,
                    P,
                    m,
                    k,
                    pivots_stride_b,
                    pivots_stride_k,
                    p_stride_b,
                    P.stride(-2),
                    P.stride(-1),
                    triton.next_power_of_2(m),
                )
            else:
                lu_unpack_p_kernel_large[(batch_size * m,)](
                    pivots,
                    P,
                    m,
                    k,
                    pivots_stride_b,
                    pivots_stride_k,
                    p_stride_b,
                    P.stride(-2),
                    P.stride(-1),
                    1,
                )

        BLOCK_K = triton.next_power_of_2(k)
        if BLOCK_K > 1024:
            BLOCK_K = 1024
        lu_unpack_l_kernel[(batch_size * m,)](
            lu,
            L,
            m,
            n,
            k,
            lu_stride_b,
            lu.stride(-2),
            lu.stride(-1),
            l_stride_b,
            L.stride(-2),
            L.stride(-1),
            BLOCK_K,
        )

        BLOCK_N = triton.next_power_of_2(n)
        if BLOCK_N > 1024:
            BLOCK_N = 1024
        lu_unpack_u_kernel[(batch_size * k,)](
            lu,
            U,
            m,
            n,
            k,
            lu_stride_b,
            lu.stride(-2),
            lu.stride(-1),
            u_stride_b,
            U.stride(-2),
            U.stride(-1),
            BLOCK_N,
        )


# ---------------------------------------------------------------------------
# Internal implementation
#
# The whole computation runs through FlagGems Triton kernels, per the
# no-torch rule: the factorization is done by linalg_lu_factor and the
# packed LU factors are unpacked into (P, L, U) by lu_unpack.
#
# lu_unpack with unpack_pivots=pivot reproduces torch.linalg.lu semantics
# exactly: for pivot=True it returns P of shape (..., m, m) such that
# P @ L @ U = A, and for pivot=False it returns an empty (0,) P with
# A = L @ U.
# ---------------------------------------------------------------------------


def _linalg_lu_impl(input, *, pivot=True):
    lu, pivots = linalg_lu_factor(input, pivot=pivot)
    P, L, U = lu_unpack(lu, pivots, unpack_data=True, unpack_pivots=pivot)
    return LinalgLUResult(P, L, U)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def linalg_lu(input, *, pivot=True):
    logger.debug("GEMS LINALG_LU")
    _linalg_lu_check(input, pivot)
    return _linalg_lu_impl(input, pivot=pivot)


def _resolve_linalg_lu_out_args(P, L, U, out):
    if out is not None:
        if P is not None or L is not None or U is not None:
            raise TypeError("linalg_lu(): out and P/L/U cannot both be set")
        if len(out) != 3:
            raise TypeError(
                "linalg_lu(): out must be a tuple of 3 tensors, " f"got {len(out)}"
            )
        return out
    if P is None or L is None or U is None:
        raise TypeError("linalg_lu(): P, L and U must all be provided for out variant")
    return P, L, U


def linalg_lu_out(input, *, pivot=True, P=None, L=None, U=None, out=None):
    logger.debug("GEMS LINALG_LU_OUT")
    _linalg_lu_check(input, pivot)
    p_out, l_out, u_out = _resolve_linalg_lu_out_args(P, L, U, out)

    batch_shape = input.shape[:-2]
    m, n = input.shape[-2], input.shape[-1]
    k = min(m, n)

    # Resize the provided outputs to the expected shapes.  For pivot=False,
    # P is resized to the empty (0,) tensor, matching torch's out-variant
    # behavior.
    if pivot:
        p_out.resize_((*batch_shape, m, m))
        p_out.zero_()
    else:
        p_out.resize_((0,))
    l_out.resize_((*batch_shape, m, k))
    u_out.resize_((*batch_shape, k, n))

    lu, pivots = linalg_lu_factor(input, pivot=pivot)
    _lu_unpack_into(lu, pivots, p_out, l_out, u_out, pivot)

    return LinalgLUResult(p_out, l_out, u_out)
