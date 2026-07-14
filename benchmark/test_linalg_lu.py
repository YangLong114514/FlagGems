import pytest
import torch

from . import base


class LinalgLuBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "input shape, pivot"
    DEFAULT_DTYPES = [torch.float32]

    def get_input_iter(self, dtype):
        for inp_shape, pivot in self.shapes:
            inp = torch.randn(inp_shape, dtype=dtype, device=self.device)
            yield inp, {"pivot": pivot}


@pytest.mark.linalg_lu
def test_linalg_lu():
    bench = LinalgLuBenchmark(
        op_name="linalg_lu",
        torch_op=torch.linalg.lu,
        dtypes=[torch.float32],
    )
    bench.run()
