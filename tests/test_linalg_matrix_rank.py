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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


# torch.linalg.matrix_rank officially accepts these four dtypes. FlagGems
# supports both real dtypes; complex inputs are deliberately rejected instead
# of being silently skipped.
SUPPORTED_DTYPE_CASES = [
    pytest.param(torch.float32, id="float32"),
    pytest.param(torch.float64, id="float64"),
]

OFFICIAL_DTYPE_CASES = [
    pytest.param(torch.float32, True, id="float32-supported"),
    pytest.param(torch.float64, True, id="float64-supported"),
    pytest.param(torch.complex64, False, id="complex64-unsupported"),
    pytest.param(torch.complex128, False, id="complex128-unsupported"),
]

RANK_CASES = [
    pytest.param((1, 7), 1, id="rank1-wide"),
    pytest.param((7, 2), 2, id="rank2-tall"),
    pytest.param((3, 5), 3, id="single-wide"),
    pytest.param((5, 3), 2, id="single-tall"),
    pytest.param((4, 4), 3, id="single-square"),
    pytest.param((16, 16), 15, id="small-jacobi-boundary"),
    pytest.param((17, 17), 16, id="serial-medium-square"),
    pytest.param((33, 33), 32, id="blocked-square"),
    pytest.param((2, 4, 4), 3, id="one-batch-dimension"),
    pytest.param((2, 3, 5, 3), 2, id="multiple-batch-dimensions"),
]

EMPTY_SHAPES = [
    pytest.param((0, 0), id="zero-by-zero"),
    pytest.param((0, 3), id="zero-by-n"),
    pytest.param((3, 0), id="m-by-zero"),
    pytest.param((2, 0, 3), id="batched-zero-by-n"),
    pytest.param((2, 3, 0), id="batched-m-by-zero"),
    pytest.param((0, 3, 3), id="empty-batch"),
]


def _make_matrix_with_rank(shape, rank, dtype=torch.float32):
    matrix = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    diagonal = torch.arange(rank, device=matrix.device)
    values = torch.arange(
        1, rank + 1, dtype=dtype, device=matrix.device
    )
    matrix[..., diagonal, diagonal] = values
    return matrix


def _to_reference_value(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    return value


def _native_matrix_rank(matrix, **kwargs):
    ref_matrix = utils.to_reference(matrix, True)
    ref_kwargs = {
        name: _to_reference_value(value, ref_matrix.device)
        for name, value in kwargs.items()
    }
    return torch.linalg.matrix_rank(ref_matrix, **ref_kwargs)


def _assert_output_metadata(result, matrix):
    assert result.shape == matrix.shape[:-2]
    assert result.dtype == torch.int64
    assert result.device == matrix.device


def _assert_direct_and_dispatch_match_native(
    matrix, *, check_dispatch=False, **kwargs
):
    native = _native_matrix_rank(matrix, **kwargs)

    direct = flag_gems.linalg_matrix_rank(matrix, **kwargs)
    _assert_output_metadata(direct, matrix)
    utils.gems_assert_equal(direct, native)

    if check_dispatch:
        with flag_gems.use_gems():
            dispatched = torch.linalg.matrix_rank(matrix, **kwargs)
        _assert_output_metadata(dispatched, matrix)
        utils.gems_assert_equal(dispatched, native)
    return direct


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_linalg_matrix_rank_default_identity(dtype):
    matrix = torch.eye(8, dtype=dtype, device=flag_gems.device)
    expected = torch.tensor(8, dtype=torch.int64, device=matrix.device)

    result = _assert_direct_and_dispatch_match_native(
        matrix, check_dispatch=True
    )
    utils.gems_assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
def test_linalg_matrix_rank_float64_preserves_small_singular_value():
    matrix = torch.tensor(
        [[1.0, 1.0], [1.0, 1.0 + 1e-10]],
        dtype=torch.float64,
        device=flag_gems.device,
    )
    expected = torch.tensor(2, dtype=torch.int64, device=matrix.device)

    result = _assert_direct_and_dispatch_match_native(matrix)
    utils.gems_assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
def test_linalg_matrix_rank_float64_tolerance_precision():
    matrix = torch.diag(
        torch.tensor(
            [1.0, 0.50000000000001],
            dtype=torch.float64,
            device=flag_gems.device,
        )
    )
    atol = torch.tensor(
        0.50000000000005, dtype=torch.float64, device=matrix.device
    )
    expected = torch.tensor(1, dtype=torch.int64, device=matrix.device)

    result = _assert_direct_and_dispatch_match_native(matrix, atol=atol)
    utils.gems_assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize(
    "dtype,k,tiny,atol",
    [
        pytest.param(torch.float32, 16, 1e-6, 1e-3, id="float32-small"),
        pytest.param(torch.float64, 17, 1e-12, 1e-9, id="float64-serial"),
    ],
)
def test_linalg_matrix_rank_well_separated_spectrum(dtype, k, tiny, atol):
    generator = torch.Generator(device=flag_gems.device).manual_seed(20260807)
    orthogonal = torch.linalg.qr(
        torch.randn(
            (k, k),
            dtype=dtype,
            device=flag_gems.device,
            generator=generator,
        )
    ).Q
    spectrum = torch.ones(k, dtype=dtype, device=flag_gems.device)
    spectrum[-1] = tiny
    matrix = orthogonal @ torch.diag(spectrum) @ orthogonal.mT
    expected = torch.tensor(k - 1, dtype=torch.int64, device=matrix.device)

    result = _assert_direct_and_dispatch_match_native(matrix, atol=atol)
    utils.gems_assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_linalg_matrix_rank_does_not_call_torch_decomposition(
    dtype, monkeypatch
):
    matrix = _make_matrix_with_rank((4, 4), 3, dtype)
    expected = _native_matrix_rank(matrix, atol=5e-2)

    def forbidden_decomposition(*args, **kwargs):
        raise AssertionError("FlagGems matrix_rank called a Torch decomposition")

    for name in ("svd", "svdvals", "eigh", "eigvalsh"):
        monkeypatch.setattr(
            torch.linalg,
            name,
            forbidden_decomposition,
        )

    result = flag_gems.linalg_matrix_rank(matrix, atol=5e-2)
    hermitian_result = flag_gems.linalg_matrix_rank(
        matrix,
        atol=5e-2,
        hermitian=True,
    )
    utils.gems_assert_equal(result, expected)
    utils.gems_assert_equal(hermitian_result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_linalg_matrix_rank_rank_deficient(dtype):
    matrix = _make_matrix_with_rank((5, 5), 3, dtype)
    expected = torch.tensor(3, dtype=torch.int64, device=matrix.device)

    result = _assert_direct_and_dispatch_match_native(
        matrix, atol=5e-2, hermitian=False
    )
    utils.gems_assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_linalg_matrix_rank_nonempty_zero_matrix(dtype):
    matrix = torch.zeros(
        (2, 4, 6), dtype=dtype, device=flag_gems.device
    )
    expected = torch.zeros((2,), dtype=torch.int64, device=matrix.device)

    result = _assert_direct_and_dispatch_match_native(
        matrix, hermitian=False
    )
    utils.gems_assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
@pytest.mark.parametrize("shape,expected_rank", RANK_CASES)
def test_linalg_matrix_rank_shapes(dtype, shape, expected_rank):
    matrix = _make_matrix_with_rank(shape, expected_rank, dtype)
    expected = torch.full(
        matrix.shape[:-2],
        expected_rank,
        dtype=torch.int64,
        device=matrix.device,
    )

    result = _assert_direct_and_dispatch_match_native(
        matrix, atol=5e-2, hermitian=False
    )
    utils.gems_assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
@pytest.mark.parametrize(
    "shape,expected_rank",
    [
        pytest.param((3, 5), 2, id="wide"),
        pytest.param((5, 3), 2, id="tall"),
        pytest.param((2, 3, 5, 3), 2, id="multi-batch"),
    ],
)
def test_linalg_matrix_rank_matches_adjoint(dtype, shape, expected_rank):
    matrix = _make_matrix_with_rank(shape, expected_rank, dtype)

    rank = _assert_direct_and_dispatch_match_native(
        matrix, atol=5e-2, hermitian=False
    )
    adjoint_rank = _assert_direct_and_dispatch_match_native(
        matrix.mH, atol=5e-2, hermitian=False
    )
    utils.gems_assert_equal(rank, adjoint_rank)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_linalg_matrix_rank_aah_svd_matches_hermitian(dtype):
    matrix = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=dtype,
        device=flag_gems.device,
    )
    matrix = torch.stack((matrix, matrix.roll(1, dims=0)))
    aah = matrix @ matrix.mH
    expected = torch.full((2,), 3, dtype=torch.int64, device=matrix.device)

    svd_rank = _assert_direct_and_dispatch_match_native(
        aah, atol=5e-2, hermitian=False
    )
    hermitian_rank = _assert_direct_and_dispatch_match_native(
        aah, atol=5e-2, hermitian=True
    )
    utils.gems_assert_equal(svd_rank, hermitian_rank)
    utils.gems_assert_equal(svd_rank, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
@pytest.mark.parametrize(
    "kwargs,expected_rank",
    [
        pytest.param({}, 4, id="default"),
        pytest.param({"rtol": 0.75}, 2, id="rtol-only"),
        pytest.param({"atol": 0.75}, 3, id="atol-only"),
        pytest.param(
            {"atol": 0.75, "rtol": 0.75}, 2, id="atol-and-rtol"
        ),
    ],
)
def test_linalg_matrix_rank_tolerance_combinations(
    dtype, kwargs, expected_rank
):
    spectrum = torch.tensor(
        [1.5, 1.25, 0.8, 0.1],
        dtype=dtype,
        device=flag_gems.device,
    )
    matrix = torch.diag(spectrum)

    result = _assert_direct_and_dispatch_match_native(matrix, **kwargs)
    expected = torch.tensor(
        expected_rank, dtype=torch.int64, device=matrix.device
    )
    utils.gems_assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
@pytest.mark.parametrize(
    "kwargs,expected_rank",
    [
        pytest.param({"atol": 0.75}, 3, id="python-float"),
        pytest.param(
            {"atol": torch.tensor(0.75)}, 3, id="zero-dim-atol-tensor"
        ),
        pytest.param(
            {"rtol": torch.tensor(0.75)}, 2, id="zero-dim-rtol-tensor"
        ),
    ],
)
def test_linalg_matrix_rank_scalar_tolerance_types(
    dtype, kwargs, expected_rank
):
    spectrum = torch.tensor(
        [1.5, 1.25, 0.8, 0.1],
        dtype=dtype,
        device=flag_gems.device,
    )
    matrix = torch.diag(spectrum)
    kwargs = {
        name: value.to(device=matrix.device, dtype=dtype)
        if isinstance(value, torch.Tensor)
        else value
        for name, value in kwargs.items()
    }

    result = _assert_direct_and_dispatch_match_native(matrix, **kwargs)
    expected = torch.tensor(
        expected_rank, dtype=torch.int64, device=matrix.device
    )
    utils.gems_assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_linalg_matrix_rank_legacy_float_tolerance(dtype):
    spectrum = torch.tensor(
        [1.5, 1.25, 0.8, 0.1],
        dtype=dtype,
        device=flag_gems.device,
    )
    matrix = torch.diag(spectrum)
    ref_matrix = utils.to_reference(matrix, True)
    native = torch.linalg.matrix_rank(ref_matrix, 0.75)

    direct = flag_gems.linalg_matrix_rank_tol(matrix, 0.75)
    utils.gems_assert_equal(direct, native)

    with flag_gems.use_gems():
        dispatched = torch.linalg.matrix_rank(matrix, 0.75)
    utils.gems_assert_equal(dispatched, native)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_linalg_matrix_rank_per_batch_tolerance(dtype):
    spectrum = torch.tensor(
        [1.5, 1.25, 0.8, 0.1],
        dtype=dtype,
        device=flag_gems.device,
    )
    matrix = torch.stack((torch.diag(spectrum), torch.diag(spectrum)))
    atol = torch.tensor([0.75, 1.3], dtype=dtype, device=matrix.device)
    expected = torch.tensor([3, 1], dtype=torch.int64, device=matrix.device)

    result = _assert_direct_and_dispatch_match_native(matrix, atol=atol)
    utils.gems_assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_linalg_matrix_rank_broadcast_tolerance(dtype):
    spectrum = torch.tensor(
        [1.5, 1.25, 0.8, 0.1],
        dtype=dtype,
        device=flag_gems.device,
    )
    base = torch.diag(spectrum)
    matrix = base.expand(2, 3, 4, 4).clone()
    atol = torch.tensor(
        [[0.75], [1.3]], dtype=dtype, device=matrix.device
    )
    expected = torch.tensor(
        [[3, 3, 3], [1, 1, 1]], dtype=torch.int64, device=matrix.device
    )

    result = _assert_direct_and_dispatch_match_native(matrix, atol=atol)
    utils.gems_assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_linalg_matrix_rank_hermitian_false(dtype):
    matrix = torch.tensor(
        [[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=dtype,
        device=flag_gems.device,
    )
    expected = torch.tensor(2, dtype=torch.int64, device=matrix.device)

    result = _assert_direct_and_dispatch_match_native(
        matrix, atol=5e-2, hermitian=False
    )
    utils.gems_assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_linalg_matrix_rank_hermitian_true(dtype):
    matrix = torch.tensor(
        [[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=dtype,
        device=flag_gems.device,
    )
    expected = torch.tensor(2, dtype=torch.int64, device=matrix.device)

    result = _assert_direct_and_dispatch_match_native(
        matrix, atol=5e-2, hermitian=True
    )
    utils.gems_assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_linalg_matrix_rank_hermitian_uses_lower_triangle(dtype):
    matrix = torch.tensor(
        [[4.0, 99.0], [2.0, 1.0]],
        dtype=dtype,
        device=flag_gems.device,
    )
    expected = torch.tensor(1, dtype=torch.int64, device=matrix.device)

    result = _assert_direct_and_dispatch_match_native(
        matrix, atol=5e-2, hermitian=True
    )
    utils.gems_assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_linalg_matrix_rank_hermitian_blocked(dtype):
    matrix = _make_matrix_with_rank((33, 33), 32, dtype)
    expected = torch.tensor(32, dtype=torch.int64, device=matrix.device)

    result = _assert_direct_and_dispatch_match_native(
        matrix, atol=5e-2, hermitian=True
    )
    utils.gems_assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
@pytest.mark.parametrize("shape", EMPTY_SHAPES)
def test_linalg_matrix_rank_empty(dtype, shape):
    matrix = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    expected = torch.zeros(
        shape[:-2], dtype=torch.int64, device=flag_gems.device
    )

    result = _assert_direct_and_dispatch_match_native(
        matrix, hermitian=False
    )
    utils.gems_assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_linalg_matrix_rank_out(dtype):
    matrix = _make_matrix_with_rank((3, 5), 2, dtype).mT
    assert not matrix.is_contiguous()
    expected = torch.tensor(2, dtype=torch.int64, device=matrix.device)
    out = torch.empty((), dtype=torch.int64, device=matrix.device)

    result = flag_gems.linalg_matrix_rank_out(
        matrix, atol=5e-2, hermitian=False, out=out
    )
    assert result.data_ptr() == out.data_ptr()
    _assert_output_metadata(result, matrix)
    utils.gems_assert_equal(result, expected)

    dispatch_out = torch.empty((), dtype=torch.int64, device=matrix.device)
    with flag_gems.use_gems():
        dispatch_result = torch.linalg.matrix_rank(
            matrix, atol=5e-2, hermitian=False, out=dispatch_out
        )
    assert dispatch_result.data_ptr() == dispatch_out.data_ptr()
    _assert_output_metadata(dispatch_result, matrix)
    utils.gems_assert_equal(dispatch_result, expected)


@pytest.mark.linalg_matrix_rank
def test_linalg_matrix_rank_out_wrong_dtype():
    matrix = torch.eye(3, dtype=torch.float32, device=flag_gems.device)
    out = torch.empty(0, dtype=torch.bool, device=matrix.device)

    with pytest.raises(RuntimeError, match="safely castable"):
        flag_gems.linalg_matrix_rank_out(matrix, out=out)


@pytest.mark.linalg_matrix_rank
def test_linalg_matrix_rank_out_wrong_device():
    matrix = torch.eye(3, dtype=torch.float32, device=flag_gems.device)
    if matrix.device.type == "cpu":
        pytest.skip("wrong-device out test requires an accelerator input")
    out = torch.empty(0, dtype=torch.int64, device="cpu")

    with pytest.raises(RuntimeError, match="same device"):
        flag_gems.linalg_matrix_rank_out(matrix, out=out)


@pytest.mark.linalg_matrix_rank
def test_linalg_matrix_rank_out_wrong_shape_warns_and_resizes():
    matrix = torch.eye(3, dtype=torch.float32, device=flag_gems.device)
    out = torch.empty((3,), dtype=torch.int64, device=matrix.device)
    expected = torch.tensor(3, dtype=torch.int64, device=matrix.device)

    with pytest.warns(UserWarning, match="output.*was resized"):
        result = flag_gems.linalg_matrix_rank_out(matrix, out=out)

    assert result.data_ptr() == out.data_ptr()
    assert out.shape == torch.Size([])
    utils.gems_assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype,is_supported", OFFICIAL_DTYPE_CASES)
def test_linalg_matrix_rank_official_dtype_contract(dtype, is_supported):
    matrix = torch.eye(3, dtype=dtype, device=flag_gems.device)

    if is_supported:
        result = _assert_direct_and_dispatch_match_native(matrix)
        expected = torch.tensor(3, dtype=torch.int64, device=matrix.device)
        utils.gems_assert_equal(result, expected)
    else:
        with pytest.raises(
            NotImplementedError, match="float32 and float64"
        ):
            flag_gems.linalg_matrix_rank(matrix)


@pytest.mark.linalg_matrix_rank
def test_linalg_matrix_rank_rejects_complex_tolerance():
    matrix = torch.eye(3, dtype=torch.float32, device=flag_gems.device)
    complex_tol = torch.tensor(1 + 0j, device=matrix.device)

    with pytest.raises(RuntimeError, match="complex type"):
        flag_gems.linalg_matrix_rank(matrix, atol=complex_tol)
