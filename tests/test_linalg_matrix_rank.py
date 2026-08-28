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

import importlib

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

VENDOR_NAME = getattr(flag_gems, "vendor_name", "")
IS_ASCEND = VENDOR_NAME == "ascend"
IS_THEAD = VENDOR_NAME == "thead"
SUPPORT_FP64 = flag_gems.runtime.device.support_fp64

# torch.linalg.matrix_rank officially accepts these four dtypes. FlagGems
# supports both real dtypes; complex inputs are deliberately rejected instead
# of being silently skipped. float64 cases only run where the device backend
# actually supports fp64 (Ascend does not).
SUPPORTED_DTYPE_CASES = [
    pytest.param(torch.float32, id="float32"),
] + (
    [pytest.param(torch.float64, id="float64")] if SUPPORT_FP64 else []
)

OFFICIAL_DTYPE_CASES = [
    pytest.param(torch.float32, True, id="float32-supported"),
    pytest.param(
        torch.float64,
        True,
        id="float64-supported",
        marks=pytest.mark.skipif(
            not SUPPORT_FP64, reason="float64 not supported on this device"
        ),
    ),
    # On Ascend complex tensors cannot even be constructed (aclnnEye has no
    # complex support), so the rejection contract is not exercisable there.
    pytest.param(
        torch.complex64,
        False,
        id="complex64-unsupported",
        marks=pytest.mark.skipif(
            IS_ASCEND, reason="complex tensors not constructible on Ascend"
        ),
    ),
    pytest.param(
        torch.complex128,
        False,
        id="complex128-unsupported",
        marks=pytest.mark.skipif(
            IS_ASCEND, reason="complex tensors not constructible on Ascend"
        ),
    ),
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

    # HGGC does not provide cusolverDnXsyevBatched_bufferSize.  Keep native
    # PyTorch as the oracle on THead, but compute batched Hermitian references
    # on CPU so the FlagGems device implementation can still be exercised.
    if (
        IS_THEAD
        and kwargs.get("hermitian", False)
        and ref_matrix.ndim > 2
        and ref_matrix.device.type != "cpu"
    ):
        cpu_kwargs = {
            name: _to_reference_value(value, torch.device("cpu"))
            for name, value in ref_kwargs.items()
        }
        return torch.linalg.matrix_rank(ref_matrix.cpu(), **cpu_kwargs).to(
            ref_matrix.device
        )
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
@pytest.mark.skipif(not SUPPORT_FP64, reason="float64 not supported on this device")
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
@pytest.mark.skipif(not SUPPORT_FP64, reason="float64 not supported on this device")
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
        pytest.param(
            torch.float64,
            17,
            1e-12,
            1e-9,
            id="float64-serial",
            marks=pytest.mark.skipif(
                not SUPPORT_FP64, reason="float64 not supported on this device"
            ),
        ),
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
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((2, 4, 6), id="batched-small"),
        pytest.param((513, 513), id="bidiag-k513"),
        pytest.param((1024, 1024), id="bidiag-k1024"),
        pytest.param((2, 513, 513), id="bidiag-batched"),
    ],
)
def test_linalg_matrix_rank_nonempty_zero_matrix(dtype, shape):
    # The k >= 513 shapes exercise the unblocked bidiagonalization path,
    # whose zero-matrix shortcut must still hand defined state to the
    # (unconditionally launched) final Sturm kernel.
    matrix = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    expected = torch.zeros(
        shape[:-2], dtype=torch.int64, device=matrix.device
    )

    if flag_gems.vendor_name in ("metax", "hygon"):
        # The MetaX and Hygon torch native references (matrix_rank via SVD)
        # do not converge on large all-zero matrices, so compare against the
        # analytic expectation while still checking direct and dispatch paths.
        result = flag_gems.linalg_matrix_rank(matrix, hermitian=False)
        _assert_output_metadata(result, matrix)
        with flag_gems.use_gems():
            dispatched = torch.linalg.matrix_rank(matrix, hermitian=False)
        _assert_output_metadata(dispatched, matrix)
        utils.gems_assert_equal(dispatched, result)
    else:
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
@pytest.mark.parametrize(
    "order,rank",
    [
        pytest.param(3, 2, id="fused-k3"),
        pytest.param(32, 28, id="fused-k32"),
        pytest.param(33, 29, id="padded-k33"),
        pytest.param(64, 60, id="padded-k64"),
    ],
)
def test_linalg_matrix_rank_hermitian_ignores_strict_upper(dtype, order, rank):
    # torch hermitian semantics: only the LOWER triangle of the input is
    # read.  Filling the strict upper triangle with huge garbage must not
    # change the rank.  Covers the fused (k <= 32) and padded (33..64)
    # tridiagonalization paths; the 2x2 closed form is covered above.
    generator = torch.Generator(device=flag_gems.device).manual_seed(7)
    basis = torch.randn(
        order, rank, dtype=dtype, device=flag_gems.device, generator=generator
    )
    matrix = basis @ basis.mT
    upper_rows, upper_cols = torch.triu_indices(
        order, order, offset=1, device=matrix.device
    )
    matrix[upper_rows, upper_cols] = 1.0e6
    expected = torch.tensor(rank, dtype=torch.int64, device=matrix.device)

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
@pytest.mark.parametrize(
    "shape,expected_rank",
    [
        pytest.param((256, 256), 200, id="tridiag-square"),
        pytest.param((257, 257), 250, id="tridiag-odd-order"),
        pytest.param((2, 300, 300), 250, id="tridiag-batched"),
        pytest.param((32, 32), 30, id="tridiag-k32"),
        pytest.param((33, 33), 30, id="tridiag-k33"),
        pytest.param((64, 64), 60, id="tridiag-k64"),
        pytest.param((128, 128), 120, id="tridiag-k128"),
        pytest.param((4, 32, 32), 30, id="tridiag-batched-small"),
        pytest.param((1024, 1024), 1000, id="tridiag-k1024"),
    ],
)
def test_linalg_matrix_rank_hermitian_tridiag(dtype, shape, expected_rank):
    matrix = _make_matrix_with_rank(shape, expected_rank, dtype)
    expected = torch.full(
        matrix.shape[:-2],
        expected_rank,
        dtype=torch.int64,
        device=matrix.device,
    )

    result = _assert_direct_and_dispatch_match_native(
        matrix, atol=5e-2, hermitian=True
    )
    utils.gems_assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
@pytest.mark.parametrize(
    "shape,expected_rank",
    [
        pytest.param((513, 513), 500, id="bidiag-k513"),
        pytest.param((1024, 1024), 1000, id="bidiag-k1024"),
        pytest.param((600, 700), 550, id="bidiag-wide"),
        pytest.param((700, 600), 550, id="bidiag-tall"),
        pytest.param((2, 513, 513), 500, id="bidiag-batched"),
        pytest.param((129, 2048), 100, id="bidiag-longrows-wide"),
        pytest.param((2048, 129), 100, id="bidiag-longrows-tall"),
    ],
)
def test_linalg_matrix_rank_bidiag(dtype, shape, expected_rank):
    matrix = _make_matrix_with_rank(shape, expected_rank, dtype)
    expected = torch.full(
        matrix.shape[:-2],
        expected_rank,
        dtype=torch.int64,
        device=matrix.device,
    )

    result = _assert_direct_and_dispatch_match_native(matrix, atol=5e-2)
    utils.gems_assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_linalg_matrix_rank_bidiag_dense(dtype):
    # Dense non-hermitian low-rank matrices exercise the two-sided
    # Householder bidiagonalization + Golub-Kahan Sturm-count path
    # (min(m, n) > 512) with a clear spectral gap at the tolerance.
    generator = torch.Generator(device=flag_gems.device).manual_seed(4321)
    n, rank = 1024, 1000
    left, _ = torch.linalg.qr(
        torch.randn(
            n, n, dtype=dtype, device=flag_gems.device, generator=generator
        )
    )
    right, _ = torch.linalg.qr(
        torch.randn(
            n, n, dtype=dtype, device=flag_gems.device, generator=generator
        )
    )
    values = torch.zeros(n, dtype=dtype, device=flag_gems.device)
    values[:rank] = torch.linspace(
        rank, 1, rank, dtype=dtype, device=flag_gems.device
    )
    matrix = left @ torch.diag(values) @ right.mT
    expected = torch.tensor(rank, dtype=torch.int64, device=matrix.device)

    result = _assert_direct_and_dispatch_match_native(matrix, atol=5e-2)
    utils.gems_assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
def test_linalg_matrix_rank_hermitian_tridiag_dense(dtype):
    # Dense symmetric low-rank matrices exercise the Householder
    # tridiagonalization + Sturm-count path (k >= 256) with a clear
    # spectral gap at the tolerance.
    generator = torch.Generator(device=flag_gems.device).manual_seed(1234)
    n, rank = 300, 250
    basis = torch.randn(
        n, rank, dtype=dtype, device=flag_gems.device, generator=generator
    )
    weights = torch.linspace(
        2.0, 1.0, rank, dtype=dtype, device=flag_gems.device
    )
    matrix = basis @ torch.diag(weights) @ basis.mT
    expected = torch.tensor(rank, dtype=torch.int64, device=matrix.device)

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
@pytest.mark.skipif(
    not IS_ASCEND, reason="fp64 rejection is specific to the Ascend backend"
)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((5, 1), id="k1"),
        pytest.param((5, 2), id="k2"),
        pytest.param((5, 5), id="fused"),
        pytest.param((40, 40), id="padded"),
        pytest.param((600, 600), id="bidiag"),
    ],
)
def test_linalg_matrix_rank_fp64_rejected(shape):
    # fp64 must fail fast with a clear error for EVERY shape class, before
    # any shape dispatch (k=1/2 used to slip past the check and die inside
    # the Triton compiler with MLIRCompilationError).
    matrix = torch.randn(shape, dtype=torch.float64, device=flag_gems.device)

    with pytest.raises(NotImplementedError, match="float64"):
        flag_gems.linalg_matrix_rank(matrix)


@pytest.mark.linalg_matrix_rank
def test_linalg_matrix_rank_rejects_complex_tolerance():
    matrix = torch.eye(3, dtype=torch.float32, device=flag_gems.device)
    complex_tol = torch.tensor(1 + 0j, device=matrix.device)

    with pytest.raises(RuntimeError, match="complex type"):
        flag_gems.linalg_matrix_rank(matrix, atol=complex_tol)


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(
    not IS_ASCEND,
    reason="FLAGGEMS_MR_EXACT_PATH is specific to the Ascend backend",
)
@pytest.mark.parametrize(
    "shape,rank,hermitian",
    [
        # Gram band: non-hermitian 33..64 and long-dimension k <= 64
        pytest.param((33, 33), 16, False, id="gram-band-k33"),
        pytest.param((256, 64), 32, False, id="gram-band-tall"),
        pytest.param((64, 512), 32, False, id="gram-band-wide"),
        pytest.param((1024, 8), 4, False, id="gram-band-long-dim-k8"),
        # QR band: 64 < k <= 512
        pytest.param((128, 128), 60, False, id="qr-band-k128"),
        pytest.param((256, 512), 100, False, id="qr-band-wide"),
        pytest.param((2, 100, 100), 40, False, id="qr-band-batched"),
        pytest.param((200, 200), 80, True, id="qr-band-hermitian"),
    ],
)
def test_linalg_matrix_rank_exact_path(shape, rank, hermitian, monkeypatch):
    # Opt-in exact reference path (FLAGGEMS_MR_EXACT_PATH=1): routes the
    # Gram/RRQR bands through the SVD-accurate Golub-Kahan bidiagonalization
    # + df64 Sturm count.  Slowly-decaying low-rank spectra (singular values
    # from 1 geometrically down to 1e-4) are where the Gram path
    # overestimates rank (sigma^2 domain) and the unpivoted QR miscounts
    # near the tolerance (|R_ii| != sigma_i); the exact path must match an
    # fp64 reference with fp32-semantics tolerance exactly.
    monkeypatch.setenv("FLAGGEMS_MR_EXACT_PATH", "1")
    generator = torch.Generator().manual_seed(2026)
    *batch, m, n = shape
    if hermitian:
        basis = torch.linalg.qr(
            torch.randn(m, m, generator=generator, dtype=torch.float64)
        )[0]
        values = torch.cat(
            [
                torch.logspace(0, -4, rank, dtype=torch.float64),
                torch.zeros(m - rank, dtype=torch.float64),
            ]
        )
        matrix = ((basis * values) @ basis.mT).to(torch.float32)
    else:
        left = torch.linalg.qr(
            torch.randn(*batch, m, rank, generator=generator, dtype=torch.float64)
        )[0]
        right = torch.linalg.qr(
            torch.randn(*batch, n, rank, generator=generator, dtype=torch.float64)
        )[0]
        values = torch.logspace(0, -4, rank, dtype=torch.float64)
        matrix = ((left * values) @ right.mT).to(torch.float32)

    matrix = matrix.to(device=flag_gems.device)
    rtol = max(m, n) * torch.finfo(torch.float32).eps
    reference = torch.linalg.matrix_rank(
        matrix.to(torch.float64).cpu(), atol=0.0, rtol=rtol, hermitian=hermitian
    )

    result = flag_gems.linalg_matrix_rank(matrix, hermitian=hermitian)
    _assert_output_metadata(result, matrix)
    utils.gems_assert_equal(result, reference.to(device=matrix.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
@pytest.mark.parametrize(
    "shape,fill_row,fill_col",
    [
        pytest.param((16, 5), 15, 4, id="tall-fused-band"),
        pytest.param((5, 16), 4, 15, id="wide-fused-band"),
        pytest.param((50, 40), 49, 39, id="tall-bidiag64-band"),
        pytest.param((40, 50), 39, 49, id="wide-bidiag64-band"),
    ],
)
def test_linalg_matrix_rank_nonsquare_tail_energy(dtype, shape, fill_row, fill_col):
    # GK bidiagonalization of a tall matrix needs K left reflections (the
    # last one folds the column K-1 tail into d[K-1]); a wide matrix is
    # handled by transposing to the tall form.  Putting the last column's
    # (resp. row's) energy beyond the diagonal band exposes a missing final
    # reflection: the rank comes out one short.  The fused (k <= 32) kernel
    # had exactly this latent bug -- random full-rank inputs mask it
    # because the lost energy stays below the tolerance.
    m, n = shape
    matrix = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    diagonal = torch.arange(min(m, n) - 1, device=matrix.device)
    matrix[diagonal, diagonal] = 1.0
    matrix[fill_row, fill_col] = 5.0
    expected = torch.tensor(min(m, n), dtype=torch.int64, device=matrix.device)

    result = _assert_direct_and_dispatch_match_native(matrix)
    utils.gems_assert_equal(result, expected)


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPE_CASES)
@pytest.mark.parametrize(
    "shape,rank",
    [
        pytest.param((16, 5), 3, id="tall-fused-band"),
        pytest.param((5, 16), 3, id="wide-fused-band"),
        pytest.param((32, 8), 4, id="tall-fused-band-k8"),
        pytest.param((8, 32), 4, id="wide-fused-band-k8"),
        pytest.param((33, 64), 16, id="wide-bidiag64-band"),
        pytest.param((64, 33), 16, id="tall-bidiag64-band"),
        pytest.param((48, 60), 24, id="wide-bidiag64-band-2"),
    ],
)
def test_linalg_matrix_rank_nonsquare_lowrank(dtype, shape, rank):
    # Random non-square low-rank matrices (slowly-decaying spectrum, sigma
    # from 1 down to 1e-4) across the fused and bidiag64 bands.  Constructed
    # in fp64 and rounded once, so the fp64 reference with fp32-semantics
    # tolerance is exact.
    generator = torch.Generator().manual_seed(17)
    m, n = shape
    left = torch.linalg.qr(
        torch.randn(m, rank, generator=generator, dtype=torch.float64)
    )[0]
    right = torch.linalg.qr(
        torch.randn(n, rank, generator=generator, dtype=torch.float64)
    )[0]
    values = torch.logspace(0, -4, rank, dtype=torch.float64)
    matrix = ((left * values) @ right.mT).to(dtype).to(flag_gems.device)

    rtol = max(m, n) * torch.finfo(torch.float32).eps
    reference = torch.linalg.matrix_rank(
        matrix.to(torch.float64).cpu(), atol=0.0, rtol=rtol
    )

    result = flag_gems.linalg_matrix_rank(matrix)
    _assert_output_metadata(result, matrix)
    utils.gems_assert_equal(result, reference.to(device=matrix.device))


@pytest.mark.linalg_matrix_rank
# k <= 32 is only exercised on Ascend: there the fused kernel counts with a
# Sturm qd chain (whose tie convention this test targets), while the generic
# path uses one-sided Jacobi whose column-norm comparison is directly strict
# (and its fp32 sum-of-squares cannot represent the subnormal-tie case).
@pytest.mark.parametrize("k", [3, 33, 65, 128, 257] if IS_ASCEND else [33, 65, 128, 257])
def test_linalg_matrix_rank_hermitian_strict_threshold(k, monkeypatch):
    # Pin the exact herm paths: k > 64 defaults to unpivoted QR (whose
    # |R_ii| count has the documented slow-decay/tie limitations); the
    # strict-threshold semantics live in the fused/padded/tridiag Sturm
    # counters exercised here.
    monkeypatch.setenv("FLAGGEMS_MR_EXACT_PATH", "1")
    # torch's hermitian semantics are STRICT: rank = #{|lambda| > tol}
    # = #{lambda > tol} + #{lambda < -tol}.  The Sturm qd zero-pivot guard
    # counts #{lambda <= x}, so the positive side K - #{<= tol} is already
    # strict, while the negative side uses the mirrored tie convention
    # (zero pivot -> tiny POSITIVE) which counts #{lambda < -tol} exactly
    # (_sturm_count_posneg2 / _sturm_count_strict*); otherwise an
    # eigenvalue exactly equal to -tol is wrongly counted, and with
    # atol == rtol == 0 a nonzero rank-deficient spectrum reports full rank.
    # Diagonal inputs keep the factorization exact, so these ties are
    # deterministic.  k covers every herm path: fused (<=32), padded
    # tridiag (33..64), and the large one-sided tridiagonalization.
    device = flag_gems.device

    def diag_case(values, atol, rtol):
        matrix = torch.diag(values).to(torch.float32).to(device)
        reference = torch.linalg.matrix_rank(
            matrix.double().cpu(), hermitian=True, atol=atol, rtol=rtol
        )
        result = flag_gems.linalg_matrix_rank(
            matrix, hermitian=True, atol=atol, rtol=rtol
        )
        utils.gems_assert_equal(result, reference.to(device))

    # negative tie: lambda == -tol must NOT be counted
    diag_case(torch.tensor([1.0, -0.5] + [0.0] * (k - 2)), 0.5, 0.0)
    # one ULP below the tie: pred(-tol) MUST be counted.  An arithmetic
    # threshold shift (-tol*(1+2eps)) lands 2-3 ULP below -tol depending on
    # tol's mantissa (2 ULP for tol=0.5, 3 ULP for tol=0.75) and would
    # wrongly skip this eigenvalue; only the mirrored zero-pivot tie
    # convention (zero pivot -> tiny positive) counts it exactly.
    pred_half = torch.nextafter(
        torch.tensor(-0.5, dtype=torch.float32),
        torch.tensor(float("-inf"), dtype=torch.float32),
    ).item()
    diag_case(torch.tensor([1.0, pred_half] + [0.0] * (k - 2)), 0.5, 0.0)
    pred_3q = torch.nextafter(
        torch.tensor(-0.75, dtype=torch.float32),
        torch.tensor(float("-inf"), dtype=torch.float32),
    ).item()
    diag_case(torch.tensor([1.0, pred_3q] + [0.0] * (k - 2)), 0.75, 0.0)
    # smallest-subnormal tolerance: an arithmetic shift rounds back onto
    # -tol itself (0 ULP), wrongly skipping lambda = -2*minsub.
    minsub = 1.401298464324817e-45  # smallest positive fp32 subnormal
    diag_case(torch.tensor([1.0, -2.0 * minsub] + [0.0] * (k - 2)), minsub, 0.0)
    # positive tie: lambda == +tol must NOT be counted
    diag_case(torch.tensor([0.5, -1.0] + [0.0] * (k - 2)), 0.5, 0.0)
    # atol == rtol == 0 on a nonzero rank-deficient spectrum: #{|lam| > 0}
    diag_case(torch.tensor([1.0, -2.0] + [0.0] * (k - 2)), 0.0, 0.0)
    # all-zero spectrum with atol == rtol == 0
    diag_case(torch.zeros(k), 0.0, 0.0)

    # dense (rotated) ties with margins above the fp32 noise floor
    generator = torch.Generator().manual_seed(k)
    basis = torch.linalg.qr(
        torch.randn(k, k, generator=generator, dtype=torch.float64)
    )[0]
    values = torch.zeros(k, dtype=torch.float64)
    values[:3] = torch.tensor([1.0, -0.5, 0.5])
    matrix = ((basis * values) @ basis.mT).float().to(device)
    for atol, expected_rank in [(0.49, 3), (0.51, 1)]:
        reference = torch.linalg.matrix_rank(
            matrix.double().cpu(), hermitian=True, atol=atol, rtol=0.0
        )
        assert reference.item() == expected_rank  # construction sanity
        result = flag_gems.linalg_matrix_rank(
            matrix, hermitian=True, atol=atol, rtol=0.0
        )
        utils.gems_assert_equal(result, reference.to(device))

    # batch + per-batch tensor tolerance
    matrix = torch.stack(
        [
            torch.diag(torch.tensor([1.0, -0.5] + [0.0] * (k - 2))),
            torch.diag(torch.tensor([1.0, -0.5] + [0.0] * (k - 2))),
        ]
    ).float()
    atol = torch.tensor([0.5, 0.6], device=device)
    rtol = torch.zeros(2, device=device)
    reference = torch.linalg.matrix_rank(
        matrix.double(), hermitian=True, atol=atol.cpu(), rtol=rtol.cpu()
    )
    result = flag_gems.linalg_matrix_rank(
        matrix.to(device), hermitian=True, atol=atol, rtol=rtol
    )
    utils.gems_assert_equal(result, reference.to(device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("k", [3, 33, 65, 257])
@pytest.mark.parametrize("hermitian", [False, True])
def test_linalg_matrix_rank_negative_tolerances(k, hermitian):
    # torch does not clamp the tolerance: tol = max(atol, rtol*sigma_max).
    # tol < 0 is reachable only when BOTH atol < 0 and rtol < 0, and then
    # every singular value (>= 0) exceeds tol -> rank == k for a nonzero
    # matrix; a zero matrix still gives 0 because rtol*0 == 0 lifts tol to
    # max(atol, 0) == 0.  A negative atol alone is harmless (rtol*sigma_max
    # >= 0 dominates the max).  The backend fixes the both-negative corner
    # up host-side; without it the hermitian split #{|lam|>tol} =
    # #{lam>tol} + #{lam<-tol} double-counts the overlap (rank > k) and the
    # sigma^2-domain paths square tol.
    device = flag_gems.device
    values = torch.zeros(k)
    values[:2] = torch.tensor([1.0, -0.5])
    matrix = torch.diag(values).to(torch.float32).to(device)
    zero = torch.zeros(k, k, dtype=torch.float32, device=device)

    def check(mat, atol, rtol):
        reference = torch.linalg.matrix_rank(
            mat.double().cpu(), hermitian=hermitian, atol=atol, rtol=rtol
        )
        result = flag_gems.linalg_matrix_rank(
            mat, hermitian=hermitian, atol=atol, rtol=rtol
        )
        utils.gems_assert_equal(result, reference.to(device))

    # negative atol alone: behaves as atol = 0
    check(matrix, -1.0, 0.0)
    # negative rtol alone: behaves as rtol = 0
    check(matrix, 0.0, -1.0)
    # both negative: tol < 0 -> every singular value counts -> full rank
    check(matrix, -1.0, -1.0)
    # both negative on a zero matrix: tol == 0 -> rank 0
    check(zero, -1.0, -1.0)

    if hermitian:
        # hermitian reads ONLY the lower triangle: strict-upper garbage is
        # invisible, so the both-negative fixup must test the lower
        # triangle for "nonzero" -- torch returns 0 here, not k.
        upper_only = torch.zeros(k, k, dtype=torch.float32, device=device)
        upper_only[0, k - 1] = 1.0
        reference = torch.linalg.matrix_rank(
            upper_only.double().cpu(), hermitian=True, atol=-1.0, rtol=-1.0
        )
        assert reference.item() == 0  # construction sanity
        result = flag_gems.linalg_matrix_rank(
            upper_only, hermitian=True, atol=-1.0, rtol=-1.0
        )
        utils.gems_assert_equal(result, reference.to(device))
        # ... and a lower-triangle-only nonzero DOES give full rank under
        # tol < 0 (eigenvalues +1/-1 of the symmetrized matrix).
        lower_only = torch.zeros(k, k, dtype=torch.float32, device=device)
        lower_only[k - 1, 0] = 1.0
        reference = torch.linalg.matrix_rank(
            lower_only.double().cpu(), hermitian=True, atol=-1.0, rtol=-1.0
        )
        assert reference.item() == k  # construction sanity
        result = flag_gems.linalg_matrix_rank(
            lower_only, hermitian=True, atol=-1.0, rtol=-1.0
        )
        utils.gems_assert_equal(result, reference.to(device))

        # Same three regimes through the TENSOR-tolerance branch (the async
        # early-exit fixup kernel): strict-upper garbage -> 0, lower-only
        # nonzero -> k, true zero -> 0.
        mixed = torch.stack([upper_only, lower_only, zero])
        atol_t = torch.full((3,), -1.0, device=device)
        rtol_t = torch.full((3,), -1.0, device=device)
        reference = torch.linalg.matrix_rank(
            mixed.double().cpu(), hermitian=True, atol=atol_t.cpu(), rtol=rtol_t.cpu()
        )
        assert reference.tolist() == [0, k, 0]  # construction sanity
        result = flag_gems.linalg_matrix_rank(
            mixed, hermitian=True, atol=atol_t, rtol=rtol_t
        )
        utils.gems_assert_equal(result, reference.to(device))

    # batch + per-batch tensor tolerances mixing all three regimes
    batch = torch.stack([matrix, zero, matrix])
    atol_t = torch.tensor([-1.0, -1.0, 0.0], device=device)
    rtol_t = torch.tensor([-1.0, -1.0, 0.0], device=device)
    reference = torch.linalg.matrix_rank(
        batch.double().cpu(), hermitian=hermitian, atol=atol_t.cpu(), rtol=rtol_t.cpu()
    )
    result = flag_gems.linalg_matrix_rank(
        batch, hermitian=hermitian, atol=atol_t, rtol=rtol_t
    )
    utils.gems_assert_equal(result, reference.to(device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(not IS_ASCEND, reason="Ascend-specific path coverage")
@pytest.mark.parametrize("shape", [(129, 64), (64, 129), (192, 64), (64, 192)])
def test_linalg_matrix_rank_longdim_exact_power2_nb(shape, monkeypatch):
    # Long-dimension k <= 64 under FLAGGEMS_MR_EXACT_PATH=1 QR-compresses to
    # the k x k R factor with the register panel kernel; for these shapes
    # rs = round_up(max(m, n), 64) = 192, and a raw NB = rs // 64 = 3
    # specialization is a marginal UB allocation that flip-flops between
    # fitting and "ub overflow" across compiles.  The launcher must clamp NB
    # to {1, 2, 4} (same as the main QR launcher).
    monkeypatch.setenv("FLAGGEMS_MR_EXACT_PATH", "1")
    m, n = shape
    rank = 17
    generator = torch.Generator().manual_seed(2026)
    left = torch.linalg.qr(
        torch.randn(m, rank, generator=generator, dtype=torch.float64)
    )[0]
    right = torch.linalg.qr(
        torch.randn(n, rank, generator=generator, dtype=torch.float64)
    )[0]
    values = torch.logspace(0, -4, rank, dtype=torch.float64)
    matrix = ((left * values) @ right.mT).to(torch.float32).to(flag_gems.device)

    rtol = max(m, n) * torch.finfo(torch.float32).eps
    reference = torch.linalg.matrix_rank(
        matrix.to(torch.float64).cpu(), atol=0.0, rtol=rtol
    )
    result = flag_gems.linalg_matrix_rank(matrix)
    _assert_output_metadata(result, matrix)
    utils.gems_assert_equal(result, reference.to(device=matrix.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.parametrize("k,expect_rank", [(65, 1), (128, 1), (257, 1), (513, 1)])
def test_linalg_matrix_rank_hermitian_deflated_spectrum(
    k, expect_rank, monkeypatch
):
    monkeypatch.setenv("FLAGGEMS_MR_EXACT_PATH", "1")
    # Strongly deflated spectra (a few significant eigenvalues, the rest at
    # the fp32 noise floor) drive the trailing subdiagonal to ~1e-10, where
    # tau = 2/vnorm2 ~ 1e20 and a naively grouped (tau*tau)*(v'w)/2
    # coefficient overflows fp32 in the rank-2 update -- verified to corrupt
    # the trailing diagonal to -inf and then NaN via 0*inf in the next
    # float mask.  The apply kernel regroups the coefficient as
    # tau*(tau*cs); this test guards that regression on the large
    # one-sided tridiagonalization path.
    generator = torch.Generator().manual_seed(k)
    basis = torch.linalg.qr(
        torch.randn(k, k, generator=generator, dtype=torch.float64)
    )[0]
    values = torch.zeros(k, dtype=torch.float64)
    values[:4] = torch.tensor([1.0, -0.5, 0.5, -0.25])
    matrix = ((basis * values) @ basis.mT).float().to(flag_gems.device)

    reference = torch.linalg.matrix_rank(
        matrix.double().cpu(), hermitian=True, atol=0.51, rtol=0.0
    )
    assert reference.item() == expect_rank  # 1.0 only; +/-0.5/-0.25 excluded
    result = flag_gems.linalg_matrix_rank(
        matrix, hermitian=True, atol=0.51, rtol=0.0
    )
    assert torch.isfinite(result.double()).all()
    utils.gems_assert_equal(result, reference.to(flag_gems.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(not IS_ASCEND, reason="Ascend-specific path coverage")
@pytest.mark.parametrize(
    "shape,rank,hermitian,kind",
    [
        # general: RRQR (65..128) vs default bidiag (>128) boundary.
        # The RRQR case uses an exactly-gapped spectrum: unpivoted QR's
        # |R_ii| undercounts slow-decay spectra even at 6x tolerance margin
        # (verified: sigma=1e-4 vs tol=1.5e-5 reports 59/60) -- that is the
        # documented exception-2 limitation, not a dispatch bug.
        pytest.param((128, 128), 60, False, "gapped", id="general-k128-rrqr"),
        pytest.param((255, 255), 120, False, "gapped", id="general-k255-rrqr"),
        pytest.param((256, 256), 120, False, "slowdecay", id="general-k256-bidiag"),
        pytest.param((256, 512), 120, False, "slowdecay", id="general-k256-wide"),
        pytest.param((512, 256), 120, False, "slowdecay", id="general-k256-tall"),
        pytest.param((2, 256, 256), 120, False, "slowdecay", id="general-k256-batched"),
        # hermitian: QR (65..255) vs one-sided big tridiag (>=256)
        pytest.param((64, 64), 30, True, "slowdecay", id="herm-k64-padded"),
        pytest.param((65, 65), 30, True, "gapped", id="herm-k65-rrqr"),
        pytest.param((256, 256), 120, True, "slowdecay", id="herm-k256-tridiag"),
    ],
)
def test_linalg_matrix_rank_dispatch_boundary(shape, rank, hermitian, kind):
    # Default-dispatch boundaries.  slowdecay = singular values 1 .. 1e-4
    # (exposes the Gram sigma^2 floor and QR near-tolerance miscount where
    # those paths are NOT expected); gapped = sigma in {1, 0} with an exact
    # gap (valid on every dispatch).  Reference: fp64 with fp32-semantics
    # tolerance.
    generator = torch.Generator().manual_seed(2026 + rank)
    *batch, m, n = shape
    if hermitian:
        basis = torch.linalg.qr(
            torch.randn(m, m, generator=generator, dtype=torch.float64)
        )[0]
        if kind == "gapped":
            nonzero = torch.ones(rank, dtype=torch.float64)
        else:
            nonzero = torch.logspace(0, -4, rank, dtype=torch.float64)
        full = torch.cat(
            [nonzero, torch.zeros(m - rank, dtype=torch.float64)]
        )
        matrix = ((basis * full) @ basis.mT).to(torch.float32)
    else:
        left = torch.linalg.qr(
            torch.randn(*batch, m, rank, generator=generator, dtype=torch.float64)
        )[0]
        right = torch.linalg.qr(
            torch.randn(*batch, n, rank, generator=generator, dtype=torch.float64)
        )[0]
        if kind == "gapped":
            values = torch.ones(rank, dtype=torch.float64)
        else:
            values = torch.logspace(0, -4, rank, dtype=torch.float64)
        matrix = ((left * values) @ right.mT).to(torch.float32)

    matrix = matrix.to(flag_gems.device)
    rtol = max(m, n) * torch.finfo(torch.float32).eps
    reference = torch.linalg.matrix_rank(
        matrix.to(torch.float64).cpu(), atol=0.0, rtol=rtol, hermitian=hermitian
    )
    result = flag_gems.linalg_matrix_rank(matrix, hermitian=hermitian)
    _assert_output_metadata(result, matrix)
    utils.gems_assert_equal(result, reference.to(flag_gems.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(not IS_ASCEND, reason="Ascend-specific path coverage")
@pytest.mark.parametrize("k,hermitian", [(256, False), (513, False), (256, True),
                                          (513, True)])
def test_linalg_matrix_rank_graph_vs_nograph(k, hermitian, monkeypatch):
    # The NPUGraph-replayed launch sequence must produce bit-identical
    # results to direct launches (FLAGGEMS_MR_NO_GRAPH=1).
    generator = torch.Generator().manual_seed(k)
    if hermitian:
        basis = torch.linalg.qr(
            torch.randn(k, k, generator=generator, dtype=torch.float64)
        )[0]
        values = torch.cat(
            [
                torch.logspace(0, -4, k // 3, dtype=torch.float64),
                torch.zeros(k - k // 3, dtype=torch.float64),
            ]
        )
        matrix = ((basis * values) @ basis.mT).float().to(flag_gems.device)
    else:
        matrix = torch.randn(k, k, generator=generator).float().to(flag_gems.device)

    monkeypatch.setenv("FLAGGEMS_MR_NO_GRAPH", "1")
    direct = flag_gems.linalg_matrix_rank(matrix, hermitian=hermitian)
    monkeypatch.delenv("FLAGGEMS_MR_NO_GRAPH")
    graphed = flag_gems.linalg_matrix_rank(matrix, hermitian=hermitian)  # captures
    replayed = flag_gems.linalg_matrix_rank(matrix, hermitian=hermitian)  # replays
    utils.gems_assert_equal(direct, graphed)
    utils.gems_assert_equal(direct, replayed)


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own graph test above")
@pytest.mark.parametrize(
    "shape,hermitian",
    [
        pytest.param((65, 65), True, id="herm-k65"),
        pytest.param((257, 257), True, id="herm-k257"),
        pytest.param((513, 513), False, id="bidiag-k513"),
    ],
)
def test_linalg_matrix_rank_generic_graph_vs_nograph(shape, hermitian, monkeypatch):
    # The CUDA-graph-replayed kernel sequence must produce the same rank as
    # direct launches (FLAGGEMS_MR_NO_GRAPH=1), and replays must refresh the
    # staging buffers: fresh input data and changed tolerances both have to
    # be honored by a replayed graph.
    generator = torch.Generator().manual_seed(sum(shape))
    matrix = torch.randn(*shape, generator=generator).float()
    if hermitian:
        matrix = matrix + matrix.mT
    matrix = matrix.to(flag_gems.device)
    reference = torch.linalg.matrix_rank(matrix.cpu(), hermitian=hermitian)

    monkeypatch.setenv("FLAGGEMS_MR_NO_GRAPH", "1")
    direct = flag_gems.linalg_matrix_rank(matrix, hermitian=hermitian)
    utils.gems_assert_equal(direct, reference.to(flag_gems.device))

    monkeypatch.delenv("FLAGGEMS_MR_NO_GRAPH")
    captured = flag_gems.linalg_matrix_rank(matrix, hermitian=hermitian)  # captures
    replayed = flag_gems.linalg_matrix_rank(matrix, hermitian=hermitian)  # replays
    utils.gems_assert_equal(captured, reference.to(flag_gems.device))
    utils.gems_assert_equal(replayed, reference.to(flag_gems.device))

    # Replay with fresh data: the graph must re-read the staging buffers.
    fresh = torch.randn(*shape, generator=generator).float()
    if hermitian:
        fresh = fresh + fresh.mT
    fresh = fresh.to(flag_gems.device)
    fresh_ref = torch.linalg.matrix_rank(fresh.cpu(), hermitian=hermitian)
    utils.gems_assert_equal(
        flag_gems.linalg_matrix_rank(fresh, hermitian=hermitian),
        fresh_ref.to(flag_gems.device),
    )
    # Replay with a changed tolerance: tolerances are staging inputs too.
    tol_ref = torch.linalg.matrix_rank(fresh.cpu(), hermitian=hermitian, atol=0.5)
    utils.gems_assert_equal(
        flag_gems.linalg_matrix_rank(fresh, hermitian=hermitian, atol=0.5),
        tol_ref.to(flag_gems.device),
    )


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(not IS_ASCEND, reason="Ascend-specific path coverage")
@pytest.mark.parametrize("k", [300, 513])
def test_linalg_matrix_rank_hermitian_ignores_strict_upper_large(k):
    # torch hermitian semantics read only the lower triangle; the large
    # one-sided tridiagonalization (init kernel's max/min addressing) must
    # not let garbage in the strict upper triangle into the spectrum.
    generator = torch.Generator().manual_seed(k)
    lower = torch.tril(torch.randn(k, k, generator=generator))
    matrix = lower.clone()
    matrix.masked_fill_(
        torch.triu(torch.ones(k, k, dtype=torch.bool), 1), 1e6
    )
    matrix = matrix.to(flag_gems.device)
    reference = torch.linalg.matrix_rank(
        (torch.tril(matrix.double().cpu()) + torch.tril(matrix.double().cpu(), -1).mT),
        hermitian=True,
    )
    result = flag_gems.linalg_matrix_rank(matrix, hermitian=True)
    utils.gems_assert_equal(result, reference.to(flag_gems.device))


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own implementation")
@pytest.mark.parametrize(
    "shape,hermitian",
    [
        pytest.param((33, 33), True, id="herm-small-tridiag"),
        pytest.param((65, 65), True, id="herm-padded-tridiag"),
        pytest.param((257, 257), True, id="herm-large-tridiag"),
        pytest.param((4, 65, 65), True, id="herm-batched"),
        pytest.param((65, 80), False, id="bidiag-medium"),
        pytest.param((513, 513), False, id="bidiag-k513"),
        pytest.param((600, 513), False, id="bidiag-tall"),
    ],
)
def test_linalg_matrix_rank_ds32_fallback(shape, hermitian, monkeypatch):
    # Force the pure-FP32 double-single Sturm tail (the path selected when
    # runtime device support_fp64 is False) on a device that natively has
    # fp64, and check every rank-relevant case against torch.  The fallback
    # only changes the Sturm count, so diagonal spectra stay exact and the
    # dense cases are built with margins far above the fp32 noise floor.
    module = importlib.import_module("flag_gems.ops.linalg_matrix_rank")
    monkeypatch.setattr(module.runtime_device, "support_fp64", False)

    device = flag_gems.device
    generator = torch.Generator().manual_seed(0)
    k = min(shape[-2:])
    rank = 7

    if hermitian:
        dense = torch.randn(shape, generator=generator)
        dense = dense + dense.mT
        x = torch.randn(shape[:-2] + (k, rank), generator=generator)
        low_rank = x @ x.mT
        values = torch.zeros(k)
        values[:6] = torch.tensor([1.0, -0.5, 1e-3, -1e-3, 1e-8, -1e-8])
        spectrum = torch.diag(values).expand(shape).contiguous()
    else:
        dense = torch.randn(shape, generator=generator)
        x = torch.randn(shape[:-2] + (shape[-2], rank), generator=generator)
        y = torch.randn(shape[:-2] + (shape[-1], rank), generator=generator)
        low_rank = x @ y.mT
        values = torch.zeros(k)
        values[:4] = torch.tensor([1.0, 2.0, 1e-3, 1e-8])
        spectrum = torch.zeros(shape)
        spectrum[..., torch.arange(k), torch.arange(k)] = values
    zero = torch.zeros(shape)

    def check(matrix, **kwargs):
        matrix = matrix.float().to(device)
        reference = torch.linalg.matrix_rank(matrix.cpu(), **kwargs)
        result = flag_gems.linalg_matrix_rank(matrix, **kwargs)
        utils.gems_assert_equal(result, reference.to(device))

    # full-rank dense
    check(dense, hermitian=hermitian)
    # exactly rank-`rank`, kept spectrum well above the default tolerance
    check(low_rank, hermitian=hermitian)
    # zero matrix -> rank 0 (the bracket must hand a defined zero tolerance
    # to the decisive count)
    check(zero, hermitian=hermitian)
    # near-threshold spectrum: 1e-3 above the default tolerance
    # (k*eps*sigma_max ~ 1e-5), 1e-8 below it
    check(spectrum, hermitian=hermitian)
    # explicit atol: only the O(1) part of the spectrum survives
    check(spectrum, hermitian=hermitian, atol=1e-2)


@pytest.mark.linalg_matrix_rank
@pytest.mark.skipif(IS_ASCEND, reason="Ascend backend has its own implementation")
def test_linalg_matrix_rank_fp64_input_requires_native_fp64(monkeypatch):
    # On a device without native FP64 the entry point must reject float64
    # input with NotImplementedError before any shape dispatch, instead of
    # silently computing in demoted precision.
    module = importlib.import_module("flag_gems.ops.linalg_matrix_rank")
    monkeypatch.setattr(module.runtime_device, "support_fp64", False)

    matrix = torch.randn(8, 8, dtype=torch.float64, device=flag_gems.device)
    with pytest.raises(NotImplementedError, match="native FP64"):
        flag_gems.linalg_matrix_rank(matrix)
    with pytest.raises(NotImplementedError, match="native FP64"):
        flag_gems.linalg_matrix_rank(matrix, hermitian=True)
