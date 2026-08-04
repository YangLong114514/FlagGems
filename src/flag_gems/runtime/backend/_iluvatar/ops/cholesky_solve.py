# Copyright 2026 FlagOS Contributors
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

import logging

from flag_gems.ops.cholesky_solve import (
    _can_write_cholesky_solve_out_direct,
    _check_cholesky_solve_out,
    _copy_cholesky_solve_out,
    cholesky_solve as _generic_cholesky_solve,
)

logger = logging.getLogger(__name__)


def cholesky_solve(B, L, upper=False, *, _out=None):
    """Use the no-gather implementation supported by CoreX Triton."""
    logger.debug("GEMS_ILUVATAR CHOLESKY_SOLVE")
    return _generic_cholesky_solve(
        B,
        L,
        upper=upper,
        _out=_out,
        _use_portable_kernels=True,
        _portable_min_dot_rhs=16,
    )


def cholesky_solve_out(B, L, upper=False, *, out):
    """Iluvatar out variant paired with the portable solve implementation."""
    logger.debug("GEMS_ILUVATAR CHOLESKY_SOLVE_OUT")
    _check_cholesky_solve_out(B, out)
    if _can_write_cholesky_solve_out_direct(B, L, out):
        return cholesky_solve(B, L, upper=upper, _out=out)
    result = cholesky_solve(B, L, upper=upper)
    return _copy_cholesky_solve_out(result, out)


__all__ = ["cholesky_solve", "cholesky_solve_out"]
