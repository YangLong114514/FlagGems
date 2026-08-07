import pytest
import torch

import flag_gems

from . import base


MATRIX_RANK_BENCHMARK_SHAPES = [
    (1, 256),
    (2, 256),
    (8, 8),
    (16, 16),
    (17, 17),
    (32, 32),
    (64, 64),
]

MATRIX_RANK_HERMITIAN_BENCHMARK_SHAPES = [
    (1, 1),
    (2, 2),
    (8, 8),
    (16, 16),
    (17, 17),
    (32, 32),
    (64, 64),
]


class MatrixRankBenchmark(base.GenericBenchmark2DOnly):
    """Benchmark for torch.linalg.matrix_rank."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = MATRIX_RANK_BENCHMARK_SHAPES.copy()


class MatrixRankHermitianBenchmark(base.GenericBenchmark2DOnly):
    """Benchmark for torch.linalg.matrix_rank with hermitian=True."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = MATRIX_RANK_HERMITIAN_BENCHMARK_SHAPES.copy()


@pytest.mark.linalg_matrix_rank
def test_linalg_matrix_rank():
    def matrix_rank_input_fn(shape, cur_dtype, device):
        matrix = torch.randn(shape, dtype=cur_dtype, device=device)
        yield (matrix,)

    bench = MatrixRankBenchmark(
        input_fn=matrix_rank_input_fn,
        op_name="linalg_matrix_rank",
        torch_op=torch.linalg.matrix_rank,
        dtypes=[torch.float32, torch.float64],
    )
    bench.set_gems(flag_gems.linalg_matrix_rank)
    bench.run()


@pytest.mark.linalg_matrix_rank
def test_linalg_matrix_rank_hermitian():
    def matrix_rank_hermitian_input_fn(shape, cur_dtype, device):
        matrix = torch.randn(shape, dtype=cur_dtype, device=device)
        matrix = matrix + matrix.mT
        yield matrix, {"hermitian": True}

    bench = MatrixRankHermitianBenchmark(
        input_fn=matrix_rank_hermitian_input_fn,
        op_name="linalg_matrix_rank_hermitian",
        torch_op=torch.linalg.matrix_rank,
        dtypes=[torch.float32, torch.float64],
    )
    bench.set_gems(flag_gems.linalg_matrix_rank)
    bench.run()
