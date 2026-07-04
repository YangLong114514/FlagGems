import pytest
import torch

from . import base

# Shape format: (*batch_dims, N, nrhs). Two-dimensional entries are single
# systems; longer entries benchmark batched solves.
CHOLESKY_SOLVE_SHAPES = [
    # Single RHS: common when solving one vector per SPD system.
    (16, 1),
    (32, 1),
    (64, 1),
    (128, 1),
    # Moderate RHS counts: representative for probes, adapters, and small heads.
    (32, 4),
    (64, 8),
    (128, 16),
    # Many RHS counts: stress throughput and memory traffic.
    (64, 32),
    (128, 64),
    (256, 128),
    # Batched systems: important for GPU occupancy with this one-program-per-system
    # kernel structure.
    (8, 16, 4),
    (16, 32, 8),
    (32, 64, 16),
    (8, 128, 16),
]


class CholeskySolveBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CHOLESKY_SOLVE_SHAPES
        self.shape_desc = "(*batch, N, nrhs)"

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            *batch_dims, n, nrhs = shape
            B_mat = torch.randn(
                *batch_dims, n, n, dtype=cur_dtype, device=self.device
            )
            eye = torch.eye(n, dtype=cur_dtype, device=self.device)
            for _ in batch_dims:
                eye = eye.unsqueeze(0)
            A = B_mat @ B_mat.transpose(-2, -1) + eye * 0.1
            L = torch.linalg.cholesky(A)
            rhs = torch.randn(
                *batch_dims, n, nrhs, dtype=cur_dtype, device=self.device
            )
            yield (rhs, L)


@pytest.mark.cholesky_solve
def test_cholesky_solve():
    bench = CholeskySolveBenchmark(
        op_name="cholesky_solve",
        torch_op=torch.ops.aten.cholesky_solve,
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
