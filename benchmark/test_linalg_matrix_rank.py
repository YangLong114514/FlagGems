import pytest
import torch

import flag_gems

from . import base, consts

VENDOR_NAME = getattr(flag_gems, "vendor_name", "")
IS_ASCEND = VENDOR_NAME == "ascend"

# torch.linalg.matrix_rank on Ascend is backed by aclnn svd_npu, which is
# float32-only ("svd_npu only supported Float, but get double"). Restrict the
# performance comparison to float32 on Ascend; float64 accuracy is still
# exercised in tests/ against the CPU reference.
MATRIX_RANK_DTYPES = (
    [torch.float32] if IS_ASCEND else [torch.float32, torch.float64]
)


MATRIX_RANK_CORE_SHAPES = [
    (1, 256),
    (256, 1),
    (2, 256),
    (256, 2),
    (8, 8),
    (16, 16),
    (17, 17),
    (32, 32),
    (33, 33),
    (64, 64),
]

MATRIX_RANK_COMPREHENSIVE_SHAPES = [
    # Tall and wide matrices exercise both workspace orientations.
    (8, 256),
    (256, 8),
    (16, 512),
    (512, 16),
    (32, 1024),
    (1024, 32),
    (64, 512),
    (512, 64),
    # Default-dispatch boundary band (general RRQR -> bidiag at k = 129):
    # 65..127 stay on unpivoted QR, 129+ on the exact bidiagonalization.
    (65, 65),
    (80, 80),
    (96, 96),
    (112, 112),
    (120, 120),
    (127, 127),
    # Medium, large, and current native-support boundaries.
    (128, 128),
    (129, 129),
    (160, 160),
    (192, 192),
    (256, 256),
    (512, 512),
    (512, 1024),
    (1024, 512),
    (1024, 1024),
    # Non-square shapes across the k = 129 dispatch boundary.
    (129, 512),
    (512, 129),
    (129, 2048),
    (2048, 129),
    # Single- and multi-dimensional batches of small/medium matrices.
    (32, 8, 8),
    (8, 16, 16),
    (4, 32, 32),
    (2, 64, 64),
    (8, 64, 16),
    (8, 16, 64),
    (2, 4, 16, 16),
    (2, 129, 129),
    (8, 129, 129),
    (2, 256, 256),
]

MATRIX_RANK_HERMITIAN_CORE_SHAPES = [
    (1, 1),
    (2, 2),
    (8, 8),
    (16, 16),
    (17, 17),
    (32, 32),
    (33, 33),
    (64, 64),
]

MATRIX_RANK_HERMITIAN_COMPREHENSIVE_SHAPES = [
    # herm 65+ uses the one-sided tridiagonalization (default since the
    # stage-4 dispatch switch); 65/129/257 sample its coverage.
    (65, 65),
    (128, 128),
    (129, 129),
    (256, 256),
    (257, 257),
    (512, 512),
    (1024, 1024),
    (32, 8, 8),
    (8, 16, 16),
    (4, 32, 32),
    (2, 64, 64),
    (2, 4, 16, 16),
]


def _select_shapes(core_shapes, comprehensive_shapes):
    shapes = core_shapes.copy()
    if (
        not base.Config.query
        and base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE
    ):
        shapes.extend(comprehensive_shapes)
    return shapes


class MatrixRankBenchmark(base.GenericBenchmark):
    """Benchmark for torch.linalg.matrix_rank."""

    DEFAULT_SHAPE_DESC = "*, M, N"

    def set_shapes(self, shape_file_path=None):
        self.shapes = _select_shapes(
            MATRIX_RANK_CORE_SHAPES,
            MATRIX_RANK_COMPREHENSIVE_SHAPES,
        )


class MatrixRankHermitianBenchmark(base.GenericBenchmark):
    """Benchmark for torch.linalg.matrix_rank with hermitian=True."""

    DEFAULT_SHAPE_DESC = "*, N, N"

    def set_shapes(self, shape_file_path=None):
        self.shapes = _select_shapes(
            MATRIX_RANK_HERMITIAN_CORE_SHAPES,
            MATRIX_RANK_HERMITIAN_COMPREHENSIVE_SHAPES,
        )


def _composed_matrix_rank(matrix, atol=None, rtol=None, hermitian=False):
    """matrix_rank composed from native NPU ops (svdvals + reduction).

    torch.linalg.matrix_rank(hermitian=True) dispatches to
    aten::_linalg_eigh.eigenvalues, which has no NPU implementation and falls
    back to the CPU. This composed version keeps the reference latency on the
    NPU: singular values of a Hermitian matrix are the absolute eigenvalues,
    so svdvals (native aclnn) plus a threshold count reproduces the semantics.
    """
    if hermitian:
        # eigh reads only the lower triangle; mirror that before the SVD.
        matrix = torch.tril(matrix) + torch.tril(matrix, -1).mT
    svals = torch.linalg.svdvals(matrix)
    if atol is None:
        atol = 0.0
    if rtol is None:
        rtol = max(matrix.shape[-2], matrix.shape[-1]) * torch.finfo(matrix.dtype).eps
    smax = svals.amax(dim=-1, keepdim=True)
    tol = torch.clamp_min(smax * rtol, atol)
    return (svals > tol).sum(dim=-1)


@pytest.mark.linalg_matrix_rank
def test_linalg_matrix_rank():
    def matrix_rank_input_fn(shape, cur_dtype, device):
        matrix = torch.randn(shape, dtype=cur_dtype, device=device)
        yield (matrix,)

    bench = MatrixRankBenchmark(
        input_fn=matrix_rank_input_fn,
        op_name="linalg_matrix_rank",
        torch_op=torch.linalg.matrix_rank,
        dtypes=MATRIX_RANK_DTYPES,
    )
    bench.set_gems(flag_gems.linalg_matrix_rank)
    bench.run()


@pytest.mark.linalg_matrix_rank
def test_linalg_matrix_rank_hermitian():
    def matrix_rank_hermitian_input_fn(shape, cur_dtype, device):
        matrix = torch.randn(shape, dtype=cur_dtype, device=device)
        matrix = matrix + matrix.mT
        yield matrix, {"hermitian": True}

    if IS_ASCEND:
        # torch.linalg.matrix_rank(hermitian=True) falls back to the CPU on
        # Ascend (aten::_linalg_eigh.eigenvalues has no NPU kernel). Use a
        # baseline composed of native NPU ops instead, mirroring the
        # cholesky_solve benchmark.
        torch_op = _composed_matrix_rank
    else:
        torch_op = torch.linalg.matrix_rank

    bench = MatrixRankHermitianBenchmark(
        input_fn=matrix_rank_hermitian_input_fn,
        op_name="linalg_matrix_rank_hermitian",
        torch_op=torch_op,
        dtypes=MATRIX_RANK_DTYPES,
    )
    bench.set_gems(flag_gems.linalg_matrix_rank)
    bench.run()
