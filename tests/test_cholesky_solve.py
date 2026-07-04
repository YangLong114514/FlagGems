import math

import pytest
import torch

import flag_gems
from flag_gems.ops.cholesky_solve import cholesky_solve

from . import accuracy_utils as utils


CHOLESKY_SOLVE_BASIC_SHAPES = [
    (2, 1),
    (4, 2),
    (8, 4),
    (16, 8),
    (32, 16),
]
CHOLESKY_SOLVE_LARGE_SHAPES = [(64, 8), (128, 4)]
CHOLESKY_SOLVE_BATCH_SHAPES = [(2, 4, 1), (3, 8, 2), (2, 3, 16, 4)]


def _make_cholesky_solve_inputs(shape, dtype, matrix_scale=1.0, rhs_scale=1.0):
    *batch_dims, n, nrhs = shape
    B_mat = torch.randn(*batch_dims, n, n, dtype=dtype, device=flag_gems.device)
    eye = torch.eye(n, dtype=dtype, device=flag_gems.device)
    for _ in batch_dims:
        eye = eye.unsqueeze(0)
    A = matrix_scale * (B_mat @ B_mat.transpose(-2, -1)) + eye * 0.5
    L = torch.linalg.cholesky(A)
    rhs = rhs_scale * torch.randn(
        *batch_dims, n, nrhs, dtype=dtype, device=flag_gems.device
    )
    return A, L, rhs


def _make_conditioned_inputs(shape, dtype):
    *batch_dims, n, nrhs = shape
    condition = 1e3 if dtype == torch.float32 else 1e6
    Q_src = torch.randn(*batch_dims, n, n, dtype=dtype, device=flag_gems.device)
    Q, _ = torch.linalg.qr(Q_src)
    eigs = torch.logspace(
        0.0,
        math.log10(condition),
        n,
        dtype=dtype,
        device=flag_gems.device,
    )
    A = (Q * eigs) @ Q.transpose(-2, -1)
    L = torch.linalg.cholesky(A)
    rhs = torch.randn(*batch_dims, n, nrhs, dtype=dtype, device=flag_gems.device)
    return A, L, rhs


def _make_noncontiguous_last_dim(tensor):
    holder = torch.empty(
        *tensor.shape[:-1],
        tensor.shape[-1] * 2,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    holder[..., ::2] = tensor
    return holder[..., ::2]


def _solve_with_gems(rhs, L, upper=False):
    with flag_gems.use_gems(include=["cholesky_solve"]):
        assert "cholesky_solve" in flag_gems.current_work_registrar.get_all_keys()
        return torch.cholesky_solve(rhs, L, upper=upper)


def _assert_backward_error(A, X, rhs, dtype):
    residual = A @ X - rhs
    denom = A.norm() * X.norm() + rhs.norm()
    backward_error = residual.norm() / denom.clamp_min(torch.finfo(dtype).eps)
    threshold = 1e-3 if dtype == torch.float32 else 1e-10
    assert backward_error.item() < threshold, (
        f"Backward error too large: {backward_error.item()} >= {threshold}"
    )


@pytest.mark.cholesky_solve
@pytest.mark.parametrize("shape", CHOLESKY_SOLVE_BASIC_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cholesky_solve(shape, dtype):
    _, L, rhs = _make_cholesky_solve_inputs(shape, dtype)
    ref_out = torch.cholesky_solve(rhs, L, upper=False)
    res_out = _solve_with_gems(rhs, L, upper=False)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cholesky_solve
@pytest.mark.parametrize("shape", CHOLESKY_SOLVE_LARGE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cholesky_solve_larger_shapes(shape, dtype):
    _, L, rhs = _make_cholesky_solve_inputs(shape, dtype)
    ref_out = torch.cholesky_solve(rhs, L, upper=False)
    res_out = _solve_with_gems(rhs, L, upper=False)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cholesky_solve
@pytest.mark.parametrize("shape", [(2, 1), (4, 2), (8, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cholesky_solve_upper(shape, dtype):
    _, L, rhs = _make_cholesky_solve_inputs(shape, dtype)
    U = L.mT.contiguous()
    ref_out = torch.cholesky_solve(rhs, U, upper=True)
    res_out = _solve_with_gems(rhs, U, upper=True)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cholesky_solve
@pytest.mark.parametrize("shape", CHOLESKY_SOLVE_BATCH_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cholesky_solve_batch(shape, dtype):
    _, L, rhs = _make_cholesky_solve_inputs(shape, dtype)
    ref_out = torch.cholesky_solve(rhs, L, upper=False)
    res_out = _solve_with_gems(rhs, L, upper=False)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cholesky_solve
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cholesky_solve_broadcast_batch(dtype):
    _, L, rhs = _make_cholesky_solve_inputs((2, 8, 3), dtype)
    rhs = rhs[:1]

    ref_out = torch.cholesky_solve(rhs, L, upper=False)
    res_out = _solve_with_gems(rhs, L, upper=False)

    utils.gems_assert_close(res_out, ref_out, dtype)
    assert res_out.shape == ref_out.shape


@pytest.mark.cholesky_solve
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cholesky_solve_noncontiguous_inputs(dtype):
    _, L, rhs = _make_cholesky_solve_inputs((16, 4), dtype)
    L_nc = _make_noncontiguous_last_dim(L)
    rhs_nc = _make_noncontiguous_last_dim(rhs)

    assert not L_nc.is_contiguous()
    assert not rhs_nc.is_contiguous()

    ref_out = torch.cholesky_solve(rhs_nc, L_nc, upper=False)
    res_out = _solve_with_gems(rhs_nc, L_nc, upper=False)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cholesky_solve
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cholesky_solve_scaled_inputs(dtype):
    for matrix_scale, rhs_scale in [(1e-3, 1e3), (1e3, 1e-3)]:
        A, L, rhs = _make_cholesky_solve_inputs(
            (16, 4), dtype, matrix_scale=matrix_scale, rhs_scale=rhs_scale
        )
        ref_out = torch.cholesky_solve(rhs, L, upper=False)
        res_out = _solve_with_gems(rhs, L, upper=False)

        utils.gems_assert_close(res_out, ref_out, dtype)
        _assert_backward_error(A, res_out, rhs, dtype)


@pytest.mark.cholesky_solve
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cholesky_solve_conditioned_matrix(dtype):
    A, L, rhs = _make_conditioned_inputs((16, 4), dtype)
    ref_out = torch.cholesky_solve(rhs, L, upper=False)
    res_out = _solve_with_gems(rhs, L, upper=False)

    utils.gems_assert_close(res_out, ref_out, dtype)
    _assert_backward_error(A, res_out, rhs, dtype)


@pytest.mark.cholesky_solve
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cholesky_solve_accuracy(dtype):
    A, L, rhs = _make_cholesky_solve_inputs((4, 2), dtype)
    X = _solve_with_gems(rhs, L, upper=False)

    _assert_backward_error(A, X, rhs, dtype)


@pytest.mark.cholesky_solve
@pytest.mark.parametrize("shape", [(4, 2), (2, 4, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cholesky_solve_direct(shape, dtype):
    _, L, rhs = _make_cholesky_solve_inputs(shape, dtype)

    ref_out = torch.cholesky_solve(rhs, L, upper=False)
    res_out = cholesky_solve(rhs, L, upper=False)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cholesky_solve
def test_cholesky_solve_empty_input():
    B = torch.empty(0, 0, dtype=torch.float32, device=flag_gems.device)
    L = torch.empty(0, 0, dtype=torch.float32, device=flag_gems.device)

    assert cholesky_solve(B, L) is B


@pytest.mark.cholesky_solve
def test_cholesky_solve_invalid_inputs():
    B = torch.randn(2, 1, dtype=torch.float32, device=flag_gems.device)
    L = torch.randn(2, 3, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(ValueError, match="square matrix"):
        cholesky_solve(B, L)

    B_bad_n = torch.randn(3, 1, dtype=torch.float32, device=flag_gems.device)
    L_square = torch.eye(2, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(ValueError, match="second-to-last dimension"):
        cholesky_solve(B_bad_n, L_square)

    B_bad_batch = torch.randn(3, 2, 1, dtype=torch.float32, device=flag_gems.device)
    L_bad_batch = torch.eye(
        2, dtype=torch.float32, device=flag_gems.device
    ).expand(2, 2, 2)
    with pytest.raises(ValueError, match="not broadcastable"):
        cholesky_solve(B_bad_batch, L_bad_batch)

    B_bad_dtype = torch.randn(2, 1, dtype=torch.float64, device=flag_gems.device)
    with pytest.raises(AssertionError, match="same dtype"):
        cholesky_solve(B_bad_dtype, L_square)
