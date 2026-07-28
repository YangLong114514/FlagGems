# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

import torch


def check_cholesky_solve_out(B: torch.Tensor, out: torch.Tensor) -> None:
    """Match the device and safe-cast checks of aten::cholesky_solve.out."""
    if out.device != B.device:
        raise RuntimeError(
            "cholesky_solve: Expected result and input tensors to be on the "
            f"same device, but got result on {out.device} and input on {B.device}"
        )
    if not torch.can_cast(B.dtype, out.dtype):
        raise RuntimeError(
            "cholesky_solve: Expected result to be safely castable from "
            f"{B.dtype} dtype, but got result with dtype {out.dtype}"
        )


def copy_cholesky_solve_out(result: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Resize and copy a temporary solve result into an out tensor."""
    if tuple(out.shape) != tuple(result.shape):
        if out.numel() != 0:
            warnings.warn(
                "An output with one or more elements was resized since it had "
                f"shape {list(out.shape)}, which does not match the required "
                f"output shape {list(result.shape)}. This behavior is deprecated, "
                "and in a future PyTorch release outputs will not be resized "
                "unless they have zero elements. You can explicitly reuse an out "
                "tensor t by resizing it, inplace, to zero elements with "
                "t.resize_(0).",
                UserWarning,
                stacklevel=3,
            )
        out.resize_(result.shape)
    out.copy_(result)
    return out
