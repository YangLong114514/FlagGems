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
import os
import threading

import torch
import triton
import triton.language as tl
from triton.runtime import driver

from flag_gems.ops.rrelu_with_noise import (
    _check_rrelu_with_noise_args,
    _rrelu_with_noise_impl,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._ascend import heuristics_config_utils as _hcu
from flag_gems.utils.random_utils import philox_backend_seed_offset

logger = logging.getLogger(__name__)

DEFAULT_LOWER = 0.125
DEFAULT_UPPER = 0.3333333333333333

# Element offsets use int32 below this numel; larger tensors take the WIDE
# variant with int64 addressing.
_WIDE_NUMEL_THRESHOLD = 1 << 30


def _pop_all_blocks_parallel():
    # TRITON_ALL_BLOCKS_PARALLEL (set globally by the fused sparse-attention
    # module at import) is read at BOTH kernel compile and launch time by this
    # triton-ascend build, and the two must agree: with it set, kernels that
    # read and write the same buffer (the in-place variants here) misbehave.
    # Pop it around compile+launch and restore afterwards (same pattern as
    # linalg_matrix_rank). Plain function pair instead of a contextmanager to
    # keep the per-launch overhead minimal.
    return os.environ.pop("TRITON_ALL_BLOCKS_PARALLEL", None)


def _restore_all_blocks_parallel(saved):
    if saved is not None:
        os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = saved


@triton.jit
def _rrelu_noise_u32(h):
    # lowbias32 integer finalizer: a few vectorized uint32 mul/shift/xor ops.
    # tl.philox is not used on purpose: it compiles to a prohibitively slow
    # sequence on the current triton-ascend toolchain.
    h ^= h >> 16
    h = h * 0x21F0AAAD
    h ^= h >> 15
    h = h * 0x735A2D97
    h ^= h >> 15
    return h


@triton.jit(do_not_specialize=["seed", "offset", "N"])
def fused_rrelu_with_noise_train_kernel(
    x_ptr,
    out_ptr,
    noise_ptr,
    N,
    lower,
    span,
    seed,
    offset,
    BLOCK: tl.constexpr,
    WIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    if WIDE:
        off = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        ctr32 = off.to(tl.int32)
    else:
        off = pid * BLOCK + tl.arange(0, BLOCK)
        ctr32 = off
    mask = off < N
    x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
    # Keep the counter arithmetic in 32 bits: int64 vector ops are emulated
    # on the vector cores. Truncating seed/offset is fine for a hash input.
    ctr = (ctr32 + offset.to(tl.int32)).to(tl.uint32) ^ seed.to(tl.uint32)
    u = (_rrelu_noise_u32(ctr) & 0x007FFFFF).to(tl.int32, bitcast=True).to(
        tl.float32
    ) * (1.0 / 8388608.0)
    n = lower + span * u
    # torch_npu samples only strictly-negative inputs; signed zero and NaN
    # take the unit-slope path.
    neg = x < 0.0
    tl.store(
        out_ptr + off,
        tl.where(neg, x * n, x).to(out_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        noise_ptr + off,
        tl.where(neg, n, 1.0).to(noise_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit(do_not_specialize=["N"])
def rrelu_with_noise_eval_kernel(
    x_ptr,
    out_ptr,
    N,
    slope,
    BLOCK: tl.constexpr,
    WIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    if WIDE:
        off = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < N
    x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
    y = tl.where(x > 0.0, x, x * slope)
    tl.store(out_ptr + off, y.to(out_ptr.dtype.element_ty), mask=mask)


# Direct CompiledKernel launches bypass JITFunction.run, whose per-call
# overhead on this triton-ascend build (~0.4 ms) dwarfs the kernel time of a
# pointwise op. Same approach as linalg_matrix_rank._fast_launch: the cache
# key covers every property the compiled binary specializes on (constexprs,
# grid, tensor alignment, pointer dtype); N/seed/offset are marked
# do_not_specialize so they stay plain runtime arguments. Everything slow
# (compile lock, the TRITON_ALL_BLOCKS_PARALLEL pop, the device context)
# happens only on the cache-miss branch; a cached launch is just a stream
# query plus the C launcher call (~15 us).
_FAST_LAUNCH_CACHE = {}
_COMPILE_LOCK = threading.Lock()
_DRV = None  # driver.active is a LazyProxy; resolve it once


def _compile_entry(kernel, grid, key, device, args, kwargs):
    global _DRV
    saved = _pop_all_blocks_parallel()
    try:
        with torch_device_fn.device(device):
            compiled = kernel.warmup(*args, grid=grid, **kwargs)
            compiled._init_handles()
    finally:
        _restore_all_blocks_parallel(saved)
    if hasattr(kernel, "non_constexpr_indices"):
        suffix = ()
    else:
        # Some frontends expect the constexpr values appended to the launch
        # arguments; resolve that once at compile time.
        suffix = tuple(kwargs[name] for name in kernel.arg_names[len(args) :])
    entry = (
        compiled.run,
        compiled.function,
        compiled.packed_metadata,
        compiled.launch_metadata,
        suffix,
        compiled,
    )
    _FAST_LAUNCH_CACHE[key] = entry
    _DRV = driver.active
    return entry


def _fast_launch(kernel, grid, key_extra, device, *args, **kwargs):
    key = (
        id(kernel),
        grid[0],
        key_extra,
        tuple(kwargs.values()),
        tuple(a.data_ptr() % 16 == 0 if torch.is_tensor(a) else None for a in args),
    )
    entry = _FAST_LAUNCH_CACHE.get(key)
    if entry is None:
        with _COMPILE_LOCK:
            entry = _FAST_LAUNCH_CACHE.get(key)
            if entry is None:
                entry = _compile_entry(kernel, grid, key, device, args, kwargs)
    run, function, md, launch_metadata, suffix, _ = entry
    launch_args = args + suffix
    saved = _pop_all_blocks_parallel()
    try:
        stream = _DRV.get_current_stream(_DRV.get_current_device())
        lm = launch_metadata(grid, stream, *launch_args)
        run(grid[0], 1, 1, stream, function, md, lm, None, None, *launch_args)
    finally:
        _restore_all_blocks_parallel(saved)


def _heur(name, N):
    cfg = _hcu.HEURISTICS_CONFIGS[name]
    args = {"N": N}
    return cfg["BLOCK"](args), cfg["num_warps"](args)


def _fused_rrelu_with_noise_train(self, noise, out, lower, upper, generator):
    N = self.numel()
    lower = float(lower)
    span = float(upper) - lower

    # One hash counter per element; advancing the offset by N keeps successive
    # calls on disjoint counter ranges, matching philox increment semantics.
    seed, offset = philox_backend_seed_offset(N, generator)

    wide = N > _WIDE_NUMEL_THRESHOLD
    BLOCK, num_warps = _heur("rrelu_with_noise_train", N)
    grid = (triton.cdiv(N, BLOCK),)
    _fast_launch(
        fused_rrelu_with_noise_train_kernel,
        grid,
        (self.dtype, wide),
        self.device,
        self,
        out,
        noise,
        N,
        lower,
        span,
        seed,
        offset,
        BLOCK=BLOCK,
        WIDE=wide,
        num_warps=num_warps,
    )
    return out


def _rrelu_with_noise_eval_ascend(self, out, slope):
    N = self.numel()
    wide = N > _WIDE_NUMEL_THRESHOLD
    BLOCK, num_warps = _heur("rrelu_with_noise_eval", N)
    grid = (triton.cdiv(N, BLOCK),)
    _fast_launch(
        rrelu_with_noise_eval_kernel,
        grid,
        (self.dtype, wide),
        self.device,
        self,
        out,
        N,
        float(slope),
        BLOCK=BLOCK,
        WIDE=wide,
        num_warps=num_warps,
    )
    return out


def _rrelu_with_noise_ascend_impl(
    self,
    noise,
    lower=DEFAULT_LOWER,
    upper=DEFAULT_UPPER,
    training=False,
    generator=None,
    out=None,
):
    _check_rrelu_with_noise_args(self, noise, lower, upper)

    if self.numel() == 0:
        return torch.empty_like(self) if out is None else out

    supported_dtype = self.dtype in (torch.float16, torch.bfloat16, torch.float32)
    if not supported_dtype:
        return _rrelu_with_noise_impl(
            self, noise, lower, upper, training, generator, out
        )

    contiguous = (
        self.is_contiguous()
        and noise.is_contiguous()
        and (out is None or out.is_contiguous())
    )
    if not contiguous:
        # Bounce through contiguous buffers: for training this also keeps the
        # torch_npu-aligned sampling semantics (x < 0) on strided layouts,
        # and for eval it avoids the generic pointwise_dynamic path, whose
        # aliased strided out0 writes are miscompiled on triton-ascend.
        self_c = self.contiguous()
        if training:
            noise_c = torch.empty_like(self_c)
            out_c = torch.empty_like(self_c)
            _fused_rrelu_with_noise_train(
                self_c, noise_c, out_c, lower, upper, generator
            )
            noise.copy_(noise_c)
        else:
            slope = (float(lower) + float(upper)) * 0.5
            out_c = torch.empty_like(self_c)
            _rrelu_with_noise_eval_ascend(self_c, out_c, slope)
        if out is None:
            return out_c
        out.copy_(out_c)
        return out

    if training:
        if out is None:
            out = torch.empty_like(self)
        return _fused_rrelu_with_noise_train(self, noise, out, lower, upper, generator)

    slope = (float(lower) + float(upper)) * 0.5
    if out is None:
        out = torch.empty_like(self)
    return _rrelu_with_noise_eval_ascend(self, out, slope)


def rrelu_with_noise(
    self,
    noise,
    lower=DEFAULT_LOWER,
    upper=DEFAULT_UPPER,
    training=False,
    generator=None,
):
    """Ascend implementation of aten.rrelu_with_noise.

    Like the generic implementation, no Python autograd wrapper is built here:
    once registered on the device dispatch key, ``aten::rrelu_with_noise``
    keeps the autograd kernel PyTorch generates from its derivative formula,
    which calls the separately registered ``aten::rrelu_with_noise_backward``.
    """
    logger.debug("GEMS_ASCEND RRELU_WITH_NOISE")
    return _rrelu_with_noise_ascend_impl(self, noise, lower, upper, training, generator)


def rrelu_with_noise_(
    self,
    noise,
    lower=DEFAULT_LOWER,
    upper=DEFAULT_UPPER,
    training=False,
    generator=None,
):
    """Ascend implementation of aten.rrelu_with_noise_."""
    logger.debug("GEMS_ASCEND RRELU_WITH_NOISE_")
    _rrelu_with_noise_ascend_impl(self, noise, lower, upper, training, generator, self)
    return self


__all__ = ["rrelu_with_noise", "rrelu_with_noise_"]
