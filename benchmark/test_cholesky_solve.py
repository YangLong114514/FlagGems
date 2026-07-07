import pytest
import torch

from . import base

# Case format: ((*batch_dims, N, nrhs), upper). Two-dimensional shape entries
# are single systems; longer entries benchmark batched solves. The cases cover
# the main performance axes of potrs/cholesky_solve: matrix order N, number of
# right-hand sides, batch occupancy, and lower/upper Cholesky factors.
CHOLESKY_SOLVE_CASES = [
    # Single RHS: exposes the low-parallelism triangular-solve path.
    ((8, 1), False),
    ((16, 1), False),
    ((32, 1), False),
    ((64, 1), False),
    ((128, 1), False),
    ((256, 1), False),
    # RHS sweep around BLOCK_RHS boundaries and tail cases.
    ((64, 4), False),
    ((64, 16), False),
    ((64, 31), False),
    ((64, 32), False),
    ((64, 33), False),
    ((64, 64), False),
    ((64, 128), False),
    # Throughput-oriented larger systems.
    ((128, 16), False),
    ((128, 64), False),
    ((256, 128), False),
    # Batched systems: important for occupancy with one batch per program tile.
    ((16, 16, 1), False),
    ((64, 16, 1), False),
    ((256, 16, 1), False),
    ((16, 32, 8), False),
    ((32, 64, 16), False),
    ((8, 128, 16), False),
    # Upper-factor cases exercise the no-transpose upper=True path.
    ((16, 1), True),
    ((64, 8), True),
    ((128, 16), True),
    ((8, 128, 16), True),
]


class CholeskySolveBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CHOLESKY_SOLVE_CASES
        self.shape_desc = "((*batch, N, nrhs), upper)"

    def get_input_iter(self, cur_dtype):
        for shape, upper in self.shapes:
            *batch_dims, n, nrhs = shape
            B_mat = torch.randn(
                *batch_dims, n, n, dtype=cur_dtype, device=self.device
            )
            eye = torch.eye(n, dtype=cur_dtype, device=self.device)
            for _ in batch_dims:
                eye = eye.unsqueeze(0)
            A = B_mat @ B_mat.transpose(-2, -1) + eye * 0.1
            L = torch.linalg.cholesky(A)
            factor = L.mT.contiguous() if upper else L
            rhs = torch.randn(
                *batch_dims, n, nrhs, dtype=cur_dtype, device=self.device
            )
            yield (rhs, factor, upper)


@pytest.mark.cholesky_solve
def test_cholesky_solve():
    bench = CholeskySolveBenchmark(
        op_name="cholesky_solve",
        torch_op=torch.ops.aten.cholesky_solve,
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
