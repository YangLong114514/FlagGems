import pytest
import torch

import flag_gems

from . import base

IS_ASCEND = flag_gems.vendor_name == "ascend"

# Import cholesky_solve from the correct backend
if IS_ASCEND:
    from flag_gems.runtime.backend._ascend.ops.cholesky_solve import cholesky_solve
else:
    from flag_gems.ops.cholesky_solve import cholesky_solve


# ---------------------------------------------------------------------------
# Shape definitions — same as GPU branch
# ---------------------------------------------------------------------------

CHOLESKY_SOLVE_SHAPES = [
    # Single RHS
    (8, 1), (16, 1), (32, 1), (64, 1), (128, 1), (256, 1),
    # Small-N small-RHS
    (16, 2), (16, 4), (32, 4),
    # RHS sweep around BLOCK_RHS boundaries
    (64, 4), (64, 8), (64, 16), (64, 31), (64, 32), (64, 33),
    (64, 64), (64, 128),
    # Throughput-oriented larger systems
    (128, 16), (128, 64), (256, 16), (256, 128),
    # Batched
    (16, 16, 1), (64, 16, 1), (256, 16, 1),
    (16, 16, 4), (16, 32, 4), (16, 32, 8), (32, 64, 16), (8, 128, 16),
]

CHOLESKY_SOLVE_CASES = [
    (shape, upper)
    for upper in (False, True)
    for shape in CHOLESKY_SOLVE_SHAPES
]


# ---------------------------------------------------------------------------
# GPU benchmark (uses existing Benchmark framework)
# ---------------------------------------------------------------------------

class CholeskySolveBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CHOLESKY_SOLVE_CASES
        self.shape_desc = "((*batch, N, nrhs), upper)"

    def get_input_iter(self, cur_dtype):
        for shape, upper in self.shapes:
            *batch_dims, n, nrhs = shape
            B_mat = torch.randn(*batch_dims, n, n, dtype=cur_dtype, device=self.device)
            eye = torch.eye(n, dtype=cur_dtype, device=self.device)
            for _ in batch_dims:
                eye = eye.unsqueeze(0)
            A = B_mat @ B_mat.transpose(-2, -1) + eye * 0.1
            L = torch.linalg.cholesky(A)
            factor = L.mT.contiguous() if upper else L
            rhs = torch.randn(*batch_dims, n, nrhs, dtype=cur_dtype, device=self.device)
            yield (rhs, factor, upper)


@pytest.mark.cholesky_solve
def test_cholesky_solve():
    """GPU backend: aten dispatch benchmark. Ascend: skipped (no aten registration)."""
    if IS_ASCEND:
        pytest.skip("Ascend uses direct call, see test_cholesky_solve_ascend")
    bench = CholeskySolveBenchmark(
        op_name="cholesky_solve",
        torch_op=torch.ops.aten.cholesky_solve,
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()


# ---------------------------------------------------------------------------
# Ascend benchmark: direct kernel call vs composed solve_triangular×2
# ---------------------------------------------------------------------------

@pytest.mark.cholesky_solve
def test_cholesky_solve_ascend():
    """Ascend backend: bench kernel vs composed solve_triangular×2 (fp32 only)."""
    if not IS_ASCEND:
        pytest.skip("Ascend-only benchmark")

    import time
    from flag_gems.runtime import torch_device_fn

    device = flag_gems.device
    dtype = torch.float32
    warmup, iters = 30, 100

    def composed(rhs, L, upper):
        if upper:
            Y = torch.linalg.solve_triangular(L.transpose(-2, -1), rhs, upper=False)
            return torch.linalg.solve_triangular(L, Y, upper=True)
        else:
            Y = torch.linalg.solve_triangular(L, rhs, upper=False)
            return torch.linalg.solve_triangular(L.transpose(-2, -1), Y, upper=True)

    def build(shape, upper):
        *batch_dims, n, nrhs = shape
        B_mat = torch.randn(*batch_dims, n, n, dtype=dtype, device=device)
        eye = torch.eye(n, dtype=dtype, device=device)
        for _ in batch_dims:
            eye = eye.unsqueeze(0)
        A = B_mat @ B_mat.transpose(-2, -1) + eye * 0.1
        L_cpu = torch.linalg.cholesky(A)
        L = L_cpu.mT.contiguous() if upper else L_cpu
        L = L.to(device).contiguous()
        rhs = torch.randn(*batch_dims, n, nrhs, dtype=dtype, device=device).contiguous()
        return L, rhs

    # Flatten to test lower+upper per shape
    print(f"\n{'N':>6s} {'nrhs':>5s} {'upper':>6s} {'kernel(us)':>10s} {'compose(us)':>10s} {'ratio':>7s}")
    print("-" * 53)

    for shape, upper in CHOLESKY_SOLVE_CASES:
        *batch_dims, n, nrhs = shape
        L, rhs = build(shape, upper)

        # warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = cholesky_solve(rhs.clone(), L, upper=upper)
                _ = composed(rhs.clone(), L, upper=upper)
        torch_device_fn.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.no_grad():
                _ = cholesky_solve(rhs.clone(), L, upper=upper)
        torch_device_fn.synchronize()
        tk = (time.perf_counter() - t0) / iters * 1e6

        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.no_grad():
                _ = composed(rhs.clone(), L, upper=upper)
        torch_device_fn.synchronize()
        tc = (time.perf_counter() - t0) / iters * 1e6

        print(f"{n:6d} {nrhs:5d} {str(upper):>6s} {tk:9.0f}us {tc:9.0f}us {tk/tc:6.2f}x")
