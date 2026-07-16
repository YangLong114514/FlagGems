import os

import pytest
import torch
import triton

import flag_gems
from flag_gems.ops.cholesky_solve import (
    cholesky_solve_complex_single_rhs_blocked_kernel,
)

from . import base

# Two-dimensional entries are single systems; longer entries benchmark batched
# solves. Keep one shape list and derive both factor orientations from it so
# every lower/upper latency comparison has an exact counterpart.
CHOLESKY_SOLVE_SHAPES = [
    # Single RHS: exposes the low-parallelism triangular-solve path.
    (8, 1),
    (16, 1),
    (32, 1),
    (64, 1),
    (128, 1),
    (256, 1),
    # Small-N small-RHS fused path coverage.
    (16, 2),
    (16, 4),
    (32, 4),
    # RHS sweep around BLOCK_RHS boundaries and tail cases.
    (64, 4),
    (64, 8),
    (64, 16),
    (64, 31),
    (64, 32),
    (64, 33),
    (64, 64),
    (64, 128),
    # Throughput-oriented larger systems.
    (128, 16),
    (128, 64),
    (256, 16),
    (256, 128),
    # Batched systems: important for occupancy with one batch per program tile.
    (16, 16, 1),
    (64, 16, 1),
    (256, 16, 1),
    (16, 16, 4),
    (16, 32, 4),
    (16, 32, 8),
    (32, 64, 16),
    (8, 128, 16),
]

# Case format: ((*batch_dims, N, nrhs), upper). Group lower then upper while
# preserving identical shape order to make benchmark tables easy to compare.
CHOLESKY_SOLVE_CASES = [
    (shape, upper)
    for upper in (False, True)
    for shape in CHOLESKY_SOLVE_SHAPES
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
            A = B_mat @ B_mat.mH + eye * 0.1
            L = torch.linalg.cholesky(A)
            factor = L.mH.contiguous() if upper else L
            rhs = torch.randn(
                *batch_dims, n, nrhs, dtype=cur_dtype, device=self.device
            )
            yield (rhs, factor, upper)


@pytest.mark.cholesky_solve
def test_cholesky_solve():
    bench = CholeskySolveBenchmark(
        op_name="cholesky_solve",
        torch_op=torch.ops.aten.cholesky_solve,
        dtypes=[
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        ],
    )
    bench.run()


COMPLEX64_SINGLE_RHS_RAW_JIT_CONFIGS = [
    # Current production baseline.
    (32, 128, 4, 1),
    # Reduce CTA size without changing the panel decomposition.
    (32, 128, 2, 1),
    # Smaller panels trade more update groups for fewer inactive lanes.
    (32, 64, 1, 1),
    (32, 64, 2, 1),
    (32, 64, 4, 1),
    (32, 32, 1, 1),
    (32, 32, 2, 1),
    (32, 32, 4, 1),
    # Shorter diagonal blocks reduce the serial pivot chain.
    (16, 64, 1, 1),
    (16, 64, 2, 1),
    (16, 64, 4, 1),
]


def _normalize_complex_factor_for_raw_jit(factor, upper):
    storage_conj = factor.is_conj()
    if storage_conj:
        factor = factor.conj()

    if factor.is_contiguous():
        effective_upper = upper
    elif factor.mT.is_contiguous():
        factor = factor.mT
        effective_upper = not upper
        storage_conj = not storage_conj
    else:
        factor = factor.contiguous()
        effective_upper = upper
    return factor, effective_upper, storage_conj


@pytest.mark.skipif(
    os.getenv("FLAG_GEMS_CHOL_SOLVE_RAW_JIT_TUNE") != "1",
    reason="set FLAG_GEMS_CHOL_SOLVE_RAW_JIT_TUNE=1 to run the manual sweep",
)
def test_cholesky_solve_complex64_single_rhs_raw_jit_tuning():
    """Sweep launch configs without LibEntry's compiled-kernel cache."""
    if not torch.cuda.is_available():
        pytest.skip("Raw-JIT tuning requires a CUDA device")

    dtype = torch.complex64
    n = 256
    nrhs = 1
    device = flag_gems.device
    generator = torch.Generator(device=device)
    generator.manual_seed(20260716)

    matrix = torch.randn(n, n, dtype=dtype, device=device, generator=generator)
    identity = torch.eye(n, dtype=dtype, device=device)
    A = matrix @ matrix.mH + identity * 0.5
    L = torch.linalg.cholesky(A)
    rhs = torch.randn(n, nrhs, dtype=dtype, device=device, generator=generator)

    raw_kernel = cholesky_solve_complex_single_rhs_blocked_kernel.fn
    all_results = []
    for upper in (False, True):
        factor = L.mH.contiguous() if upper else L
        factor, effective_upper, storage_conj = _normalize_complex_factor_for_raw_jit(
            factor, upper
        )
        rhs_work = rhs.contiguous()
        output = torch.empty_like(rhs_work)
        factor_real = torch.view_as_real(factor).reshape(1, n, n, 2)
        rhs_real = torch.view_as_real(rhs_work).reshape(1, n, nrhs, 2)
        output_real = torch.view_as_real(output).reshape(1, n, nrhs, 2)

        for (
            block_k,
            block_m,
            num_warps,
            num_stages,
        ) in COMPLEX64_SINGLE_RHS_RAW_JIT_CONFIGS:
            config = (block_k, block_m, num_warps, num_stages)

            def launch():
                raw_kernel[(1,)](
                    factor_real,
                    rhs_real,
                    output_real,
                    n,
                    factor_real.stride(0),
                    rhs_real.stride(0),
                    factor_real.stride(1),
                    factor_real.stride(2),
                    rhs_real.stride(1),
                    BLOCK_K=block_k,
                    BLOCK_M=block_m,
                    upper=effective_upper,
                    storage_conj=storage_conj,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )

            try:
                launch()
                torch.cuda.synchronize()
                backward_error = ((A @ output - rhs).norm() / rhs.norm()).item()
                if backward_error >= 1e-3:
                    print(
                        f"INVALID upper={upper} config={config} "
                        f"backward_error={backward_error:.3e}"
                    )
                    continue
                latency_ms = triton.testing.do_bench(
                    launch,
                    warmup=25,
                    rep=100,
                    return_mode="median",
                )
            except Exception as error:
                print(
                    f"ERROR upper={upper} config={config}: "
                    f"{type(error).__name__}: {error}"
                )
                continue

            latency_us = latency_ms * 1000.0
            all_results.append((upper, latency_us, config, backward_error))

    assert all_results, "No Raw-JIT configuration compiled and passed correctness"
    for upper in (False, True):
        orientation_results = sorted(
            result for result in all_results if result[0] == upper
        )
        print(f"\nRaw-JIT results upper={upper}:")
        for _, latency_us, config, backward_error in orientation_results:
            print(
                f"  {latency_us:9.3f} us  "
                f"BK={config[0]:2d} BM={config[1]:3d} "
                f"warps={config[2]} stages={config[3]}  "
                f"error={backward_error:.3e}"
            )
