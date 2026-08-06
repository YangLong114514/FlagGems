import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


RANK_CASES = [
    ((3, 5), 3),
    ((5, 3), 2),
    ((2, 4, 4), 3),
    ((2, 3, 5, 3), 2),
]


def _make_matrix_with_rank(shape, rank):
    matrix = torch.zeros(shape, dtype=torch.float32, device=flag_gems.device)
    diagonal = torch.arange(rank, device=matrix.device)
    values = torch.arange(
        1, rank + 1, dtype=torch.float32, device=matrix.device
    )
    matrix[..., diagonal, diagonal] = values
    return matrix


@pytest.mark.linalg_matrix_rank
def test_linalg_matrix_rank_default_full_rank():
    matrix = torch.eye(8, dtype=torch.float32, device=flag_gems.device)
    expected = torch.tensor(8, dtype=torch.int64, device=matrix.device)

    result = flag_gems.linalg_matrix_rank(matrix)
    utils.gems_assert_equal(result, expected)

    with flag_gems.use_gems():
        dispatch_result = torch.linalg.matrix_rank(matrix)
    utils.gems_assert_equal(dispatch_result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("shape,expected_rank", RANK_CASES)
def test_linalg_matrix_rank(shape, expected_rank):
    matrix = _make_matrix_with_rank(shape, expected_rank)
    ref_matrix = utils.to_reference(matrix, True)

    ref_out = torch.linalg.matrix_rank(ref_matrix, atol=5e-2)
    res_out = flag_gems.linalg_matrix_rank(matrix, atol=5e-2)

    assert res_out.shape == matrix.shape[:-2]
    assert res_out.dtype == torch.int64
    utils.gems_assert_equal(res_out, ref_out)

    with flag_gems.use_gems():
        dispatch_out = torch.linalg.matrix_rank(matrix, atol=5e-2)
    utils.gems_assert_equal(dispatch_out, ref_out)


@pytest.mark.linalg_matrix_rank
def test_linalg_matrix_rank_tolerances():
    spectrum = torch.tensor(
        [1.5, 1.25, 0.8, 0.1],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    matrix = torch.diag(spectrum)

    cases = [
        ({"rtol": 0.75}, 2),
        ({"atol": 0.75}, 3),
        ({"atol": 0.75, "rtol": 0.75}, 2),
        ({"rtol": torch.tensor(0.75, device=matrix.device)}, 2),
    ]
    for kwargs, expected in cases:
        result = flag_gems.linalg_matrix_rank(matrix, **kwargs)
        utils.gems_assert_equal(
            result,
            torch.tensor(expected, dtype=torch.int64, device=matrix.device),
        )

    legacy_result = flag_gems.linalg_matrix_rank_tol(matrix, 0.75)
    utils.gems_assert_equal(
        legacy_result,
        torch.tensor(3, dtype=torch.int64, device=matrix.device),
    )

    with flag_gems.use_gems():
        dispatch_result = torch.linalg.matrix_rank(matrix, 0.75)
    utils.gems_assert_equal(dispatch_result, legacy_result)


@pytest.mark.linalg_matrix_rank
def test_linalg_matrix_rank_batch_tolerance():
    spectrum = torch.tensor(
        [1.5, 1.25, 0.8, 0.1],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    matrix = torch.stack((torch.diag(spectrum), torch.diag(spectrum)))
    atol = torch.tensor([0.75, 1.3], device=matrix.device)
    expected = torch.tensor([3, 1], dtype=torch.int64, device=matrix.device)

    result = flag_gems.linalg_matrix_rank(matrix, atol=atol)
    utils.gems_assert_equal(result, expected)

    with flag_gems.use_gems():
        dispatch_result = torch.linalg.matrix_rank(matrix, atol=atol)
    utils.gems_assert_equal(dispatch_result, expected)


@pytest.mark.linalg_matrix_rank
def test_linalg_matrix_rank_hermitian():
    matrix = torch.diag(
        torch.tensor(
            [2.0, 1.0, 0.0], dtype=torch.float32, device=flag_gems.device
        )
    )
    expected = torch.tensor(2, dtype=torch.int64, device=matrix.device)

    result = flag_gems.linalg_matrix_rank(
        matrix, atol=5e-2, hermitian=True
    )
    utils.gems_assert_equal(result, expected)

    with flag_gems.use_gems():
        dispatch_result = torch.linalg.matrix_rank(
            matrix, atol=5e-2, hermitian=True
        )
    utils.gems_assert_equal(dispatch_result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize(
    "shape",
    [
        (0, 0),
        (0, 3),
        (3, 0),
        (2, 0, 3),
        (2, 3, 0),
        (0, 3, 3),
    ],
)
def test_linalg_matrix_rank_empty(shape):
    matrix = torch.empty(shape, dtype=torch.float32, device=flag_gems.device)
    expected = torch.zeros(
        shape[:-2], dtype=torch.int64, device=flag_gems.device
    )

    result = flag_gems.linalg_matrix_rank(matrix)
    utils.gems_assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
def test_linalg_matrix_rank_non_contiguous_and_out():
    matrix = _make_matrix_with_rank((3, 5), 2).mT
    assert not matrix.is_contiguous()
    expected = torch.tensor(2, dtype=torch.int64, device=matrix.device)
    out = torch.empty((), dtype=torch.int64, device=matrix.device)

    result = flag_gems.linalg_matrix_rank_out(
        matrix, atol=5e-2, out=out
    )
    assert result.data_ptr() == out.data_ptr()
    utils.gems_assert_equal(result, expected)

    dispatch_out = torch.empty((), dtype=torch.int64, device=matrix.device)
    with flag_gems.use_gems():
        dispatch_result = torch.linalg.matrix_rank(
            matrix, atol=5e-2, out=dispatch_out
        )
    assert dispatch_result.data_ptr() == dispatch_out.data_ptr()
    utils.gems_assert_equal(dispatch_result, expected)


@pytest.mark.linalg_matrix_rank
def test_linalg_matrix_rank_initial_limitations():
    matrix = torch.eye(3, dtype=torch.float64, device=flag_gems.device)
    with pytest.raises(NotImplementedError, match="float32"):
        flag_gems.linalg_matrix_rank(matrix)

    matrix = matrix.float()
    complex_tol = torch.tensor(1 + 0j, device=matrix.device)
    with pytest.raises(RuntimeError, match="complex type"):
        flag_gems.linalg_matrix_rank(matrix, atol=complex_tol)
