from collections import namedtuple

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

DEVICE = flag_gems.device
VENDOR = flag_gems.vendor_name

if VENDOR == "nvidia":
    _TEST_DTYPES = [torch.float32, torch.float64]
else:
    _TEST_DTYPES = [torch.float32]

# pivot=False is only supported on CUDA
if utils.TO_CPU:
    _PIVOT_VALUES = [True]
elif DEVICE == "cuda":
    _PIVOT_VALUES = [True, False]
else:
    _PIVOT_VALUES = [True]


def _unpack_lu_no_pivot(lu):
    m, n = lu.shape[-2], lu.shape[-1]
    k = min(m, n)
    ll = lu[..., :, :k].tril()
    diag = torch.arange(k, device=lu.device)
    ll[..., diag, diag] = 1
    u = lu[..., :k, :].triu()
    return ll, u


def _make_input(shape, pivot, device, dtype):
    """Generate a test matrix suitable for the given pivot mode.

    For pivot=True, a random matrix is used (partial pivoting handles stability).
    For pivot=False, the matrix is constructed as L @ U where L has unit diagonal
    to guarantee a stable no-pivot LU factorization exists.
    """
    if pivot:
        return torch.randn(shape, dtype=dtype, device=device)

    # Construct A = L @ U where L is unit lower triangular and U is upper
    # triangular with a well-conditioned diagonal. Scale L's off-diagonal
    # elements to keep the triangular solve well-conditioned.
    *batch, m, n = shape
    k = min(m, n)
    scaling = k**-0.5
    L = (torch.randn(*batch, m, k, dtype=dtype, device=device) * scaling).tril()
    L.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    U = torch.randn(*batch, k, n, dtype=dtype, device=device).triu()
    # Make U diagonally dominant for numerical stability
    U.diagonal(dim1=-2, dim2=-1).abs_().add_(1.0)
    return L @ U


LinalgLUFactorResult = namedtuple("LinalgLUFactorResult", ["LU", "pivots"])


def _make_zero_pivot_input(shape, pos, device, dtype):
    """Diagonal matrix with an exactly-zero entry at ``pos`` along the diagonal.

    A zero pivot at the *last* elimination step has no trailing submatrix, so it
    is harmless.  At the first or a middle position the sub-diagonal column used
    to be scaled by that zero pivot, and since ``tl.where`` evaluates both of its
    branches, the kernel computed ``0/0 = NaN`` there; the rank-1 update then
    spread it across the whole trailing submatrix (``NaN * 0 = NaN``), leaving
    every column past the zero pivot -- and every later panel -- NaN.

    A diagonal matrix is the cheapest way to place the zero pivot exactly: the
    pivot search picks the diagonal entry itself, since the rest of the column is
    zero.
    """
    k = min(shape[-2], shape[-1])
    zero_at = {"first": 0, "middle": k // 2, "last": k - 1}[pos]

    diag = torch.arange(1, k + 1, dtype=dtype, device=device) + 1.0
    diag[zero_at] = 0.0
    mat = torch.zeros(shape, dtype=dtype, device=device)
    idx = torch.arange(k, device=device)
    mat[..., idx, idx] = diag
    return mat, zero_at


def _swap_rows(lu, i, pivot_row):
    *batch_shape, m, n = lu.shape
    device = lu.device

    rows = torch.arange(m, device=device).expand(*batch_shape, -1)

    mask_i = (rows == i).float().unsqueeze(-1)
    mask_p = (rows == pivot_row.unsqueeze(-1)).float().unsqueeze(-1)

    row_i_vals = (lu * mask_i).sum(dim=-2, keepdim=True)
    row_p_vals = (lu * mask_p).sum(dim=-2, keepdim=True)

    mask_i_full = mask_i.expand(*batch_shape, m, n)
    mask_p_full = mask_p.expand(*batch_shape, m, n)
    diff_ip = (row_p_vals - row_i_vals).expand(*batch_shape, m, n)
    diff_pi = (row_i_vals - row_p_vals).expand(*batch_shape, m, n)

    lu = lu + mask_i_full * diff_ip
    lu = lu + mask_p_full * diff_pi
    return lu


def _lu_factor_pivot(lu, m, n, k):
    *batch_shape, _, _ = lu.shape
    device = lu.device
    pivots = torch.empty((*batch_shape, k), dtype=torch.int32, device=device)

    for i in range(k):
        col = lu[..., i:, i].abs()
        pivot_rel = torch.argmax(col, dim=-1)
        pivot_row = pivot_rel + i
        pivots[..., i] = (pivot_row + 1).to(torch.int32)

        lu = _swap_rows(lu, i, pivot_row)

        pivot_val = lu[..., i, i]
        lu[..., i + 1 :, i] = lu[..., i + 1 :, i] / pivot_val.unsqueeze(-1)

        if i + 1 < m and i + 1 < n:
            l_col = lu[..., i + 1 :, i].unsqueeze(-1)
            u_row = lu[..., i : i + 1, i + 1 :]
            lu[..., i + 1 :, i + 1 :] = lu[..., i + 1 :, i + 1 :] - l_col @ u_row

    return lu, pivots


def _lu_factor_no_pivot(lu, m, n, k):
    *batch_shape, _, _ = lu.shape
    device = lu.device
    pivots = torch.empty((*batch_shape, k), dtype=torch.int32, device=device)

    for i in range(k):
        pivots[..., i] = i + 1
        pivot_val = lu[..., i, i]
        lu[..., i + 1 :, i] = lu[..., i + 1 :, i] / pivot_val.unsqueeze(-1)

        if i + 1 < m and i + 1 < n:
            l_col = lu[..., i + 1 :, i].unsqueeze(-1)
            u_row = lu[..., i : i + 1, i + 1 :]
            lu[..., i + 1 :, i + 1 :] = lu[..., i + 1 :, i + 1 :] - l_col @ u_row

    return lu, pivots


def ops_lu_factor(input, *, pivot=True):
    if input.dim() < 2:
        raise RuntimeError(
            "torch.linalg.lu_factor: Expected input to have at least 2 dimensions"
        )
    if input.dtype != torch.float32:
        raise NotImplementedError("Only float32 is supported")
    m, n = input.shape[-2], input.shape[-1]
    if m == 0 or n == 0:
        raise NotImplementedError("Empty matrices are not supported")
    if pivot not in (True, False):
        raise TypeError(f"pivot must be a bool, got {type(pivot)}")

    input_contiguous = input.contiguous()
    m, n = input_contiguous.shape[-2], input_contiguous.shape[-1]
    k = min(m, n)
    lu = input_contiguous.clone()

    if pivot:
        lu, pivots = _lu_factor_pivot(lu, m, n, k)
    else:
        lu, pivots = _lu_factor_no_pivot(lu, m, n, k)

    return LinalgLUFactorResult(lu, pivots)


def _run_torch_ops_path(inp, pivot):
    res = ops_lu_factor(inp, pivot=pivot)
    return res.LU, res.pivots


@pytest.mark.linalg_lu_factor
@pytest.mark.parametrize(
    "shape",
    [
        (4, 4),
        (32, 32),
        (16, 32),
        (64, 32),
        (128, 16, 16),
        (128, 128),
        (128, 64),
        (64, 128),
        (256, 256),
        (512, 512),
    ],
)
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
@pytest.mark.parametrize("pivot", _PIVOT_VALUES)
def test_linalg_lu_factor(shape, dtype, pivot):
    inp = _make_input(shape, pivot, flag_gems.device, dtype)
    ref_inp = utils.to_reference(inp)

    if flag_gems.vendor_name != "ascend":
        ref_lu, ref_pivots = torch.linalg.lu_factor(ref_inp, pivot=pivot)
    else:
        ref_lu, ref_pivots = _run_torch_ops_path(ref_inp, pivot=pivot)
    with flag_gems.use_gems():
        res_lu, res_pivots = torch.linalg.lu_factor(inp, pivot=pivot)
    batch_shape = inp.shape[:-2]
    m, n = inp.shape[-2], inp.shape[-1]
    k = min(m, n)

    assert res_lu.shape == inp.shape
    assert res_pivots.dtype == torch.int32
    assert res_pivots.shape == (*batch_shape, k)
    assert torch.all(res_pivots >= 1)
    assert torch.all(res_pivots <= m)

    torch.backends.cuda.matmul.allow_tf32 = False
    if pivot:
        res_p, res_l, res_u = torch.lu_unpack(res_lu, res_pivots)
        ref_p, ref_l, ref_u = torch.lu_unpack(ref_lu, ref_pivots)
        reconstructed = res_p @ res_l @ res_u
        ref_reconstructed = ref_p @ ref_l @ ref_u
    else:
        res_l, res_u = _unpack_lu_no_pivot(res_lu)
        ref_l, ref_u = _unpack_lu_no_pivot(ref_lu)
        reconstructed = res_l @ res_u
        ref_reconstructed = ref_l @ ref_u
    utils.gems_assert_close(reconstructed, ref_reconstructed, dtype, reduce_dim=k)


@pytest.mark.linalg_lu_factor_out
@pytest.mark.parametrize(
    "shape",
    [
        (4, 4),
        (32, 32),
        (16, 32),
        (64, 32),
        (128, 16, 16),
        (128, 128),
        (128, 64),
        (64, 128),
        (256, 256),
        (512, 512),
    ],
)
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
@pytest.mark.parametrize("pivot", _PIVOT_VALUES)
def test_linalg_lu_factor_out(shape, dtype, pivot):
    if not pivot and flag_gems.device != "cuda":
        pytest.skip("pivot=False only supported on CUDA")

    inp = _make_input(shape, pivot, flag_gems.device, dtype)
    ref_inp = utils.to_reference(inp)

    batch_shape = inp.shape[:-2]
    m, n = inp.shape[-2], inp.shape[-1]
    k = min(m, n)

    ref_LU_out = torch.empty_like(ref_inp)
    ref_pivots_out = torch.empty(
        (*batch_shape, k), dtype=torch.int32, device=ref_inp.device
    )
    if flag_gems.vendor_name != "ascend":
        ref_LU, ref_pivots = torch.linalg.lu_factor(
            ref_inp, pivot=pivot, out=(ref_LU_out, ref_pivots_out)
        )
    else:
        ref_LU, ref_pivots = _run_torch_ops_path(ref_inp, pivot=pivot)
        ref_LU_out.copy_(ref_LU)
        ref_pivots_out.copy_(ref_pivots)
        ref_LU, ref_pivots = ref_LU_out, ref_pivots_out

    res_LU_out = torch.empty_like(inp)
    res_pivots_out = torch.empty(
        (*batch_shape, k), dtype=torch.int32, device=inp.device
    )
    out = (res_LU_out, res_pivots_out)
    with flag_gems.use_gems():
        res_LU, res_pivots = torch.linalg.lu_factor(inp, pivot=pivot, out=out)

    assert res_LU is res_LU_out
    assert res_pivots is res_pivots_out

    assert res_LU.shape == inp.shape
    assert res_pivots.dtype == torch.int32
    assert res_pivots.shape == (*batch_shape, k)
    assert torch.all(res_pivots >= 1)
    assert torch.all(res_pivots <= m)

    torch.backends.cuda.matmul.allow_tf32 = False
    if pivot:
        res_p, res_l, res_u = torch.lu_unpack(res_LU, res_pivots)
        ref_p, ref_l, ref_u = torch.lu_unpack(ref_LU, ref_pivots)
        reconstructed = res_p @ res_l @ res_u
        ref_reconstructed = ref_p @ ref_l @ ref_u
    else:
        res_l, res_u = _unpack_lu_no_pivot(res_LU)
        ref_l, ref_u = _unpack_lu_no_pivot(ref_LU)
        reconstructed = res_l @ res_u
        ref_reconstructed = ref_l @ ref_u
    utils.gems_assert_close(reconstructed, ref_reconstructed, dtype, reduce_dim=k)


@pytest.mark.linalg_lu_factor
@pytest.mark.parametrize("shape", [(8, 8), (64, 32), (128, 128)])
@pytest.mark.parametrize("pos", ["first", "middle", "last"])
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
@pytest.mark.parametrize("pivot", _PIVOT_VALUES)
def test_linalg_lu_factor_zero_pivot(shape, pos, dtype, pivot):
    """A zero pivot must not contaminate the part of the factor past itself.

    Regression test for the exact-zero-pivot divide: the panel/single-tile
    kernels scaled the sub-diagonal column with ``col_vals / pivot`` inside a
    ``tl.where``, so an exactly-zero pivot produced ``0/0 = NaN`` in every lane
    and the following rank-1 update turned the entire trailing submatrix (and
    every later panel, through the L21 @ U12 update) into NaN.  ATen instead
    zeroes the multipliers, keeps the rest of the factor finite, and reports the
    zero pivot through ``info``.
    """
    inp, zero_at = _make_zero_pivot_input(shape, pos, flag_gems.device, dtype)
    ref_inp = utils.to_reference(inp)
    k = min(shape[-2], shape[-1])

    # ``torch.linalg.lu_factor`` raises on a singular factor, so ATen's
    # factorization is read through ``lu_factor_ex``.
    if flag_gems.vendor_name != "ascend":
        ref_lu = torch.linalg.lu_factor_ex(ref_inp, pivot=pivot, check_errors=False).LU

    with flag_gems.use_gems():
        res_lu, res_pivots = torch.linalg.lu_factor(inp, pivot=pivot)

    # The regression itself: nothing past the zero pivot may be NaN/Inf.
    assert not torch.isnan(res_lu).any(), "zero pivot contaminated the factor"
    assert not torch.isinf(res_lu).any()

    # The zero pivot stays where the input put it, and the columns after it are
    # intact -- the input reconstruction below fails loudly if they are not.
    diag_idx = torch.arange(k, device=res_lu.device)
    assert (res_lu[..., diag_idx, diag_idx][..., zero_at] == 0).all()
    if zero_at + 1 < k:
        assert (
            res_lu[..., diag_idx[zero_at + 1 :], diag_idx[zero_at + 1 :]] != 0
        ).any(), "entries past the zero pivot were wiped out"

    torch.backends.cuda.matmul.allow_tf32 = False
    if pivot:
        res_p, res_l, res_u = torch.lu_unpack(res_lu, res_pivots)
        reconstructed = res_p @ res_l @ res_u
    else:
        res_l, res_u = _unpack_lu_no_pivot(res_lu)
        reconstructed = res_l @ res_u
    # Compare against ``ref_inp``, not ``inp``: under ``--ref cpu`` the reference
    # must already live on the CPU (``accuracy_utils.to_cpu`` asserts it), and
    # ``to_reference`` is what moves it there.  In the default device-reference
    # mode it returns ``inp`` unchanged, so this is the same tensor.
    utils.gems_assert_close(reconstructed, ref_inp, dtype, reduce_dim=k)

    if flag_gems.vendor_name != "ascend":
        # ATen's answer is bit-identical here: a diagonal matrix needs no
        # rounding in the elimination, so the comparison can be exact.
        utils.gems_assert_close(res_lu, ref_lu, dtype)


@pytest.mark.linalg_lu_factor
@pytest.mark.skipif(
    utils.TO_CPU,
    reason="linalg.lu_factor: LU without pivoting is not implemented on the CPU",
)
@pytest.mark.parametrize(
    "inp_cpu",
    [
        # Zero pivot at (0, 0) with a non-zero column below it, so the two
        # candidate conventions differ: the multipliers must become 0 (ATen),
        # not Inf/NaN and not "leave the column alone".
        [[0.0, 1.0, 1.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0]],
        [[0.0, -2.0, 0.5], [1.0, 4.0, -3.0], [-6.0, 0.25, 2.0]],
    ],
)
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
def test_linalg_lu_factor_zero_pivot_no_pivot_dense(inp_cpu, dtype):
    """``pivot=False`` with an exactly-zero pivot: multipliers become zero.

    With ``pivot=False`` the column below a zero pivot is *not* necessarily zero
    (partial pivoting is what guarantees that), so this pins the convention
    rather than relying on it: an exactly-zero pivot yields zero multipliers and
    the rank-1 update still runs.  Reconstruction into the original matrix is
    intentionally not asserted -- as in ATen, dropping the multiplier loses that
    column of the factor -- only the agreement with ATen and the absence of
    NaN/Inf are.
    """
    inp = torch.tensor(inp_cpu, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    k = min(inp.shape[-2], inp.shape[-1])

    if flag_gems.vendor_name != "ascend":
        ref_lu = torch.linalg.lu_factor_ex(ref_inp, pivot=False, check_errors=False).LU

    with flag_gems.use_gems():
        res_lu, _ = torch.linalg.lu_factor(inp, pivot=False)

    assert not torch.isnan(res_lu).any(), "zero pivot produced NaN multipliers"
    assert not torch.isinf(res_lu).any()
    # Multipliers below the zero pivot are zeroed, not left unscaled.
    assert (res_lu[..., 1:, 0] == 0).all()

    if flag_gems.vendor_name != "ascend":
        utils.gems_assert_close(res_lu, ref_lu, dtype, reduce_dim=k)
