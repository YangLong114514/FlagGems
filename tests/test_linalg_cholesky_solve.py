import pytest
import torch

import flag_gems
from flag_gems.ops.linalg_cholesky_solve import linalg_cholesky_solve

from . import accuracy_utils as utils


def _make_cholesky_solve_inputs(shape, dtype):
    *batch_dims, n, nrhs = shape
    B_mat = torch.randn(*batch_dims, n, n, dtype=dtype, device=flag_gems.device)
    eye = torch.eye(n, dtype=dtype, device=flag_gems.device)
    for _ in batch_dims:
        eye = eye.unsqueeze(0)
    A = B_mat @ B_mat.transpose(-2, -1) + eye * 0.5
    L = torch.linalg.cholesky(A)
    rhs = torch.randn(*batch_dims, n, nrhs, dtype=dtype, device=flag_gems.device)
    return A, L, rhs


@pytest.mark.linalg_cholesky_solve
@pytest.mark.parametrize("shape", [(2, 1), (4, 2), (8, 4), (16, 8), (32, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_cholesky_solve(shape, dtype):
    _, L, rhs = _make_cholesky_solve_inputs(shape, dtype)
    ref_out = torch.cholesky_solve(rhs, L, upper=False)

    with flag_gems.use_gems(include=["linalg_cholesky_solve"]):
        res_out = torch.cholesky_solve(rhs, L, upper=False)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_cholesky_solve
@pytest.mark.parametrize("shape", [(2, 1), (4, 2), (8, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_cholesky_solve_upper(shape, dtype):
    _, L, rhs = _make_cholesky_solve_inputs(shape, dtype)
    U = L.mT.contiguous()
    ref_out = torch.cholesky_solve(rhs, U, upper=True)

    with flag_gems.use_gems(include=["linalg_cholesky_solve"]):
        res_out = torch.cholesky_solve(rhs, U, upper=True)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_cholesky_solve
@pytest.mark.parametrize("shape", [(2, 4, 1), (3, 8, 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_cholesky_solve_batch(shape, dtype):
    _, L, rhs = _make_cholesky_solve_inputs(shape, dtype)
    ref_out = torch.cholesky_solve(rhs, L, upper=False)

    with flag_gems.use_gems(include=["linalg_cholesky_solve"]):
        res_out = torch.cholesky_solve(rhs, L, upper=False)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_cholesky_solve
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_cholesky_solve_accuracy(dtype):
    """Verify numerical accuracy: check that A @ X is close to B."""
    A, L, rhs = _make_cholesky_solve_inputs((4, 2), dtype)

    with flag_gems.use_gems(include=["linalg_cholesky_solve"]):
        X = torch.cholesky_solve(rhs, L, upper=False)

    residual = A @ X - rhs
    max_residual = residual.abs().max().item()

    assert max_residual < 1e-2, f"Residual too large: {max_residual}"


@pytest.mark.linalg_cholesky_solve
@pytest.mark.parametrize("shape", [(4, 2), (2, 4, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_cholesky_solve_direct(shape, dtype):
    _, L, rhs = _make_cholesky_solve_inputs(shape, dtype)

    ref_out = torch.cholesky_solve(rhs, L, upper=False)
    res_out = linalg_cholesky_solve(rhs, L, upper=False)

    utils.gems_assert_close(res_out, ref_out, dtype)
