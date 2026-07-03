import pytest
import torch

from . import base

# Cholesky solve benchmark shapes
# (N, nrhs) pairs covering small to medium-large use cases
CHOLESKY_SOLVE_SHAPES = [
    (2, 1),
    (4, 2),
    (8, 4),
    (16, 8),
    (32, 16),
    (64, 32),
    (128, 64),
    (256, 128),
]


class CholeskySolveBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CHOLESKY_SOLVE_SHAPES

    def get_input_iter(self, cur_dtype):
        for n, nrhs in self.shapes:
            # Create SPD matrix A = B @ B^T + I*eps
            B_mat = torch.randn(
                n, n, dtype=cur_dtype, device=self.device
            )
            A = (
                B_mat @ B_mat.transpose(-2, -1)
                + torch.eye(n, dtype=cur_dtype, device=self.device) * 0.1
            )
            L = torch.linalg.cholesky(A)
            rhs = torch.randn(n, nrhs, dtype=cur_dtype, device=self.device)
            yield (rhs, L)


@pytest.mark.linalg_cholesky_solve
def test_linalg_cholesky_solve():
    bench = CholeskySolveBenchmark(
        op_name="cholesky_solve",
        torch_op=torch.ops.aten.cholesky_solve,
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
