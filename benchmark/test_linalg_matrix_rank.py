import pytest
import torch

import flag_gems

from . import base


MATRIX_RANK_BENCHMARK_SHAPES = [
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
]


class MatrixRankBenchmark(base.GenericBenchmark2DOnly):
    """Benchmark for torch.linalg.matrix_rank."""

    def set_more_shapes(self):
        return MATRIX_RANK_BENCHMARK_SHAPES


@pytest.mark.linalg_matrix_rank
def test_linalg_matrix_rank():
    def matrix_rank_input_fn(shape, cur_dtype, device):
        matrix = torch.randn(shape, dtype=cur_dtype, device=device)
        yield (matrix,)

    bench = MatrixRankBenchmark(
        input_fn=matrix_rank_input_fn,
        op_name="linalg_matrix_rank",
        torch_op=torch.linalg.matrix_rank,
        dtypes=[torch.float32],
    )
    bench.set_gems(flag_gems.linalg_matrix_rank)
    bench.run()
