import pytest
import torch
import flag_gems

from . import base


class LinalgLuFactorBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "input shape, pivot"
    DEFAULT_DTYPES = [torch.float32]

    def get_input_iter(self, dtype):
        for inp_shape, pivot in self.shapes:
            inp = torch.randn(inp_shape, dtype=dtype, device=self.device)
            yield inp, {"pivot": pivot}


@pytest.mark.linalg_lu_factor
def test_linalg_lu_factor():
    bench = LinalgLuFactorBenchmark(
        op_name="linalg_lu_factor",
        torch_op=torch.linalg.lu_factor,
        dtypes=[torch.float32],
    )
    bench.set_gems(flag_gems.linalg_lu_factor)
    bench.run()
