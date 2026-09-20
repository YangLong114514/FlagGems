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

"""Unit tests for ``flag_gems.rrelu_with_noise`` and ``flag_gems.rrelu_with_noise_``.

Both operators implement::

    aten::rrelu_with_noise(Tensor self, Tensor(a!) noise, Scalar lower,
                           Scalar upper, bool training, Generator? generator)

whose contract is:

* ``training=False``: ``out = self > 0 ? self : self * (lower + upper) / 2`` and
  the caller's ``noise`` buffer is left untouched;
* ``training=True``: every ``self <= 0`` element gets a slope drawn uniformly
  from ``[lower, upper]`` and stored in ``noise``, every other element stores
  ``1``, and ``out = self * noise``.

Every case is written once per operator and calls that operator by name: the
implementation under test through the public FlagGems API
(``flag_gems.rrelu_with_noise``), the comparison point through the native ATen
kernel (``torch.ops.aten.rrelu_with_noise``).  The FlagGems API is always called
explicitly and without ``flag_gems.use_gems``, as required by
``docs/content/zh-cn/contribution/overview.md``.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

# Defaults of torch.nn.functional.rrelu.
DEFAULT_LOWER = 0.125
DEFAULT_UPPER = 1.0 / 3.0

# A training call draws its slopes from the global generator, which cannot be
# replayed on the reference side.  Equal bounds remove that randomness -- every
# sampled element then holds exactly this slope -- so that the applied noise and
# the output stay comparable element-wise with ATen.  The sampling itself is
# covered by the *_train_sampling tests.
EQUAL_BOUNDS = (0.25, 0.25)


def _skip_half_cpu_reference(dtype, training):
    """Skip the cases the CPU reference cannot serve.

    ``torch.ops.aten.rrelu_with_noise`` has no Half kernel for the training
    path on CPU: it raises "rrelu_with_noise_out_cpu not implemented for
    'Half'".  fp16 keeps being covered by the reference-free tests.
    """
    if cfg.TO_CPU and training and dtype == torch.float16:
        pytest.skip("the CPU reference does not implement the training path for Half")


def _pair(shape, dtype, bounds, fill_noise=False):
    """Build matched (FlagGems, reference) inputs.

    ``fill_noise`` pre-fills the caller's noise buffer with a random slope, so
    that the eval path is tested against a buffer that differs from the unit
    slope rather than against zeros.
    """
    lower, upper = bounds
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    noise = torch.zeros_like(inp)
    if fill_noise:
        noise.uniform_(lower, upper)
    return (
        inp,
        noise,
        utils.to_reference(inp.clone()),
        utils.to_reference(noise.clone()),
    )


@pytest.mark.rrelu_with_noise
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_rrelu_with_noise_eval(shape, dtype):
    """Eval mode matches ATen and leaves the caller's noise buffer untouched."""
    lower, upper = DEFAULT_LOWER, DEFAULT_UPPER
    inp, noise, ref_inp, ref_noise = _pair(
        shape, dtype, (lower, upper), fill_noise=True
    )

    ref_out = torch.ops.aten.rrelu_with_noise(ref_inp, ref_noise, lower, upper, False)
    result = flag_gems.rrelu_with_noise(inp, noise, lower, upper, False)

    utils.gems_assert_close(result, ref_out, dtype)
    utils.gems_assert_close(noise, ref_noise, dtype)


@pytest.mark.rrelu_with_noise_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_rrelu_with_noise_inplace_eval(shape, dtype):
    """Eval mode matches ATen and leaves the caller's noise buffer untouched."""
    lower, upper = DEFAULT_LOWER, DEFAULT_UPPER
    inp, noise, ref_inp, ref_noise = _pair(
        shape, dtype, (lower, upper), fill_noise=True
    )

    ref_out = torch.ops.aten.rrelu_with_noise_(ref_inp, ref_noise, lower, upper, False)
    result = flag_gems.rrelu_with_noise_(inp, noise, lower, upper, False)

    utils.gems_assert_close(result, ref_out, dtype)
    utils.gems_assert_close(noise, ref_noise, dtype)


@pytest.mark.rrelu_with_noise
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_rrelu_with_noise_train(shape, dtype):
    """Training mode matches ATen: the same effective noise and output.

    The bounds are equal so that the sampled slope is deterministic; see
    ``EQUAL_BOUNDS``.
    """
    _skip_half_cpu_reference(dtype, True)
    lower, upper = EQUAL_BOUNDS
    inp, noise, ref_inp, ref_noise = _pair(shape, dtype, (lower, upper))

    ref_out = torch.ops.aten.rrelu_with_noise(ref_inp, ref_noise, lower, upper, True)
    result = flag_gems.rrelu_with_noise(inp, noise, lower, upper, True)

    utils.gems_assert_close(result, ref_out, dtype)
    # `noise` reports the effective slope: the sampled one for `self <= 0` and
    # a unit slope everywhere else.
    utils.gems_assert_close(noise, ref_noise, dtype)


@pytest.mark.rrelu_with_noise_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_rrelu_with_noise_inplace_train(shape, dtype):
    """Training mode matches ATen: the same effective noise and output.

    The bounds are equal so that the sampled slope is deterministic; see
    ``EQUAL_BOUNDS``.
    """
    _skip_half_cpu_reference(dtype, True)
    lower, upper = EQUAL_BOUNDS
    inp, noise, ref_inp, ref_noise = _pair(shape, dtype, (lower, upper))

    ref_out = torch.ops.aten.rrelu_with_noise_(ref_inp, ref_noise, lower, upper, True)
    result = flag_gems.rrelu_with_noise_(inp, noise, lower, upper, True)

    utils.gems_assert_close(result, ref_out, dtype)
    # `noise` reports the effective slope: the sampled one for `self <= 0` and
    # a unit slope everywhere else.
    utils.gems_assert_close(noise, ref_noise, dtype)


def _assert_train_sampling(result, noise, original, dtype):
    """Shared body of the *_train_sampling cases.

    The slope must be uniform over the whole ``[lower, upper]`` range and be
    drawn for exactly the ``self <= 0`` elements.
    """
    lower, upper = DEFAULT_LOWER, DEFAULT_UPPER
    span = upper - lower

    sampled = original <= 0
    assert 0 < int(sampled.sum()) < sampled.numel()
    drawn = noise[sampled]

    assert torch.all(drawn >= lower)
    assert torch.all(drawn <= upper)
    # ~2000 draws must cover the interval rather than a sub-range of it: their
    # extreme values sit close to the bounds and their mean close to the middle.
    assert drawn.min() <= lower + span * 1e-2
    assert drawn.max() >= upper - span * 1e-2
    assert abs(float(drawn.mean()) - (lower + upper) / 2) <= span * 1e-1
    assert torch.any(drawn != drawn[0])

    # Every other element stores a unit slope, and the output is the input
    # scaled by the effective noise.
    utils.gems_assert_equal(
        noise[~sampled], utils.to_reference(torch.ones_like(noise[~sampled]))
    )
    expected = utils.to_reference(torch.where(sampled, original * noise, original))
    utils.gems_assert_close(result, expected, dtype)


@pytest.mark.rrelu_with_noise
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_rrelu_with_noise_train_sampling(dtype):
    """The drawn slopes cover ``[lower, upper]`` and the input is preserved."""
    lower, upper = DEFAULT_LOWER, DEFAULT_UPPER
    original = torch.linspace(-2.0, 2.0, 4097, dtype=dtype, device=flag_gems.device)
    inp = original.clone()
    noise = torch.zeros_like(inp)

    result = flag_gems.rrelu_with_noise(inp, noise, lower, upper, True)

    _assert_train_sampling(result, noise, original, dtype)
    utils.gems_assert_equal(inp, utils.to_reference(original))


@pytest.mark.rrelu_with_noise_
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_rrelu_with_noise_inplace_train_sampling(dtype):
    """The drawn slopes cover ``[lower, upper]``."""
    lower, upper = DEFAULT_LOWER, DEFAULT_UPPER
    original = torch.linspace(-2.0, 2.0, 4097, dtype=dtype, device=flag_gems.device)
    inp = original.clone()
    noise = torch.zeros_like(inp)

    result = flag_gems.rrelu_with_noise_(inp, noise, lower, upper, True)

    _assert_train_sampling(result, noise, original, dtype)


@pytest.mark.rrelu_with_noise
def test_rrelu_with_noise_generator():
    """The generator argument drives the training slope: equal seeds reproduce
    it exactly, different seeds do not, and the stream keeps advancing."""
    lower, upper = DEFAULT_LOWER, DEFAULT_UPPER
    original = -torch.ones((4096,), dtype=torch.float32, device=flag_gems.device)

    def draw(seed):
        generator = torch.Generator(device=flag_gems.device)
        generator.manual_seed(seed)
        inp = original.clone()
        noise = torch.zeros_like(inp)
        out = flag_gems.rrelu_with_noise(inp, noise, lower, upper, True, generator)
        return generator, out, noise

    _, out_a, noise_a = draw(2026)
    generator_b, out_b, noise_b = draw(2026)
    _, _, noise_c = draw(2027)

    assert torch.equal(out_a, out_b)
    assert torch.equal(noise_a, noise_b)
    assert not torch.equal(noise_a, noise_c)

    inp = original.clone()
    noise = torch.zeros_like(inp)
    flag_gems.rrelu_with_noise(inp, noise, lower, upper, True, generator_b)
    assert not torch.equal(noise, noise_a)


@pytest.mark.rrelu_with_noise_
def test_rrelu_with_noise_inplace_generator():
    """The generator argument drives the training slope: equal seeds reproduce
    it exactly, different seeds do not, and the stream keeps advancing."""
    lower, upper = DEFAULT_LOWER, DEFAULT_UPPER
    original = -torch.ones((4096,), dtype=torch.float32, device=flag_gems.device)

    def draw(seed):
        generator = torch.Generator(device=flag_gems.device)
        generator.manual_seed(seed)
        inp = original.clone()
        noise = torch.zeros_like(inp)
        out = flag_gems.rrelu_with_noise_(inp, noise, lower, upper, True, generator)
        return generator, out, noise

    _, out_a, noise_a = draw(2026)
    generator_b, out_b, noise_b = draw(2026)
    _, _, noise_c = draw(2027)

    assert torch.equal(out_a, out_b)
    assert torch.equal(noise_a, noise_b)
    assert not torch.equal(noise_a, noise_c)

    inp = original.clone()
    noise = torch.zeros_like(inp)
    flag_gems.rrelu_with_noise_(inp, noise, lower, upper, True, generator_b)
    assert not torch.equal(noise, noise_a)


def _ascend_graph_entry(training, inp, noise, lower, upper, out):
    """The Ascend backend's cached NPUGraph for this configuration, if any.

    Eval keys store the slope in the ``lower`` slot and ``0.0`` for ``upper``
    (mirroring the backend's key layout).
    """
    if flag_gems.device != "npu":
        return None
    import sys

    # The backend module is loaded under its own top-level package name, so
    # resolve it from the registered function: importing by the flag_gems
    # path would create a second module instance with empty graph caches.
    backend = sys.modules[flag_gems.rrelu_with_noise_.__module__]
    return backend._debug_graph_entry(training, inp, noise, lower, upper, out)


@pytest.mark.rrelu_with_noise_
def test_rrelu_with_noise_inplace_train_stream_advances():
    """Pointer-stable repeated training calls consume a continuing RNG stream.

    Backends that cache or graph-capture a recurring configuration (the Ascend
    backend captures after several identical calls) must keep the per-call
    semantics intact: every call draws fresh slopes, writes them through the
    input alias, and a reseed reproduces the first draw exactly.
    """
    lower, upper = DEFAULT_LOWER, DEFAULT_UPPER
    generator = torch.Generator(device=flag_gems.device)
    generator.manual_seed(2026)
    inp = torch.full((4096,), -1.0, device=flag_gems.device)
    noise = torch.zeros_like(inp)

    snapshots = []
    for _ in range(12):
        # Reset the value without changing the pointer: an input left to
        # decay (x *= slope <= 1/3 each round) would soon underflow the
        # comparison tolerance and stop detecting wrong outputs.
        inp.fill_(-1.0)
        result = flag_gems.rrelu_with_noise_(inp, noise, lower, upper, True, generator)
        assert result is inp
        # Every element is negative, so every one of them is sampled.
        assert torch.all(noise >= lower) and torch.all(noise <= upper)
        utils.gems_assert_close(result, utils.to_reference(-noise), torch.float32)
        snapshots.append(noise.clone())

    for earlier, later in zip(snapshots, snapshots[1:]):
        assert not torch.equal(earlier, later)

    # Twelve identical calls cross the Ascend capture threshold: the cached
    # graph must exist, and the reseed below exercises its replay+resync.
    if flag_gems.device == "npu":
        assert _ascend_graph_entry(True, inp, noise, lower, upper, inp) is not None

    generator.manual_seed(2026)
    inp.fill_(-1.0)
    flag_gems.rrelu_with_noise_(inp, noise, lower, upper, True, generator)
    utils.gems_assert_equal(noise, utils.to_reference(snapshots[0]))
    utils.gems_assert_close(inp, utils.to_reference(-noise), torch.float32)


@pytest.mark.rrelu_with_noise_
def test_rrelu_with_noise_inplace_train_high_seed():
    """Seeds up to 2**64 - 1 are legal, reproducible, and not truncated.

    The high halves of the seed must participate in the draw: two seeds that
    share their low 32 bits must not produce the same slopes.
    """
    lower, upper = DEFAULT_LOWER, DEFAULT_UPPER
    high_seed = 2**64 - 1

    def draw(seed, calls):
        generator = torch.Generator(device=flag_gems.device)
        generator.manual_seed(seed)
        inp = torch.full((4096,), -1.0, device=flag_gems.device)
        noise = torch.zeros_like(inp)
        drawn = []
        for _ in range(calls):
            inp.fill_(-1.0)
            result = flag_gems.rrelu_with_noise_(
                inp, noise, lower, upper, True, generator
            )
            utils.gems_assert_close(result, utils.to_reference(-noise), torch.float32)
            drawn.append(noise.clone())
        return drawn

    # Enough calls to cross backend-side caching/graph-capture thresholds.
    first = draw(high_seed, 12)
    utils.gems_assert_equal(draw(high_seed, 1)[0], utils.to_reference(first[0]))
    for earlier, later in zip(first, first[1:]):
        assert not torch.equal(earlier, later)

    same_low32 = 0x5A5A0000_00000000 | (high_seed & 0xFFFFFFFF)
    assert not torch.equal(draw(same_low32, 1)[0], first[0])


@pytest.mark.rrelu_with_noise_
def test_rrelu_with_noise_inplace_eval_replay_consistent():
    """Repeated in-place eval calls always recompute from the live buffer.

    Pointer-stable repeated calls are replayed from a captured graph on
    backends that cache one; every replay must read the buffer's current
    contents rather than reproducing a stale result.
    """
    lower, upper = DEFAULT_LOWER, DEFAULT_UPPER
    slope = (lower + upper) / 2
    inp = torch.empty((4096,), dtype=torch.float32, device=flag_gems.device)
    noise = torch.zeros_like(inp)
    noise_before = noise.clone()

    for round_ in range(12):
        value = float(round_ + 1) * (1.0 if round_ % 2 == 0 else -1.0)
        inp.fill_(value)
        result = flag_gems.rrelu_with_noise_(inp, noise, lower, upper, False)
        assert result is inp
        expected = torch.full_like(inp, value if value > 0 else value * slope)
        utils.gems_assert_close(result, utils.to_reference(expected), torch.float32)

    # Eval never touches the caller's noise buffer.
    utils.gems_assert_equal(noise, utils.to_reference(noise_before))
    if flag_gems.device == "npu":
        assert _ascend_graph_entry(False, inp, noise, slope, 0.0, inp) is not None


@pytest.mark.rrelu_with_noise_
def test_rrelu_with_noise_inplace_capture_failure_fallback(monkeypatch):
    """A failed graph capture must fall back to direct launches.

    Forces the capture context to raise, crosses the capture threshold while
    it is broken, and checks the results stay correct throughout; once
    restored, the same configuration captures successfully.
    """
    if flag_gems.device != "npu":
        pytest.skip("NPUGraph capture is an Ascend-backend path")

    lower, upper = DEFAULT_LOWER, DEFAULT_UPPER
    generator = torch.Generator(device=flag_gems.device)
    generator.manual_seed(2026)
    inp = torch.full((4096,), -1.0, device=flag_gems.device)
    noise = torch.zeros_like(inp)

    class _FailingGraph:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            raise RuntimeError("forced capture failure")

        def __exit__(self, *args):
            return False

    def run_call():
        inp.fill_(-1.0)
        flag_gems.rrelu_with_noise_(inp, noise, lower, upper, True, generator)
        assert torch.all(noise >= lower) and torch.all(noise <= upper)
        utils.gems_assert_close(inp, utils.to_reference(-noise), torch.float32)

    with monkeypatch.context() as m:
        m.setattr(torch.npu, "graph", _FailingGraph)
        for _ in range(12):
            run_call()
        assert _ascend_graph_entry(True, inp, noise, lower, upper, inp) is None

    run_call()
    assert _ascend_graph_entry(True, inp, noise, lower, upper, inp) is not None


@pytest.mark.rrelu_with_noise_
def test_rrelu_with_noise_inplace_train_multi_device():
    """Graph capture/replay stays correct when several devices are in use.

    Each device's capture must bind a capture stream on that device: warming
    two devices past the capture threshold and replaying on both must keep
    producing advancing, reproducible slope streams.
    """
    if flag_gems.device != "npu" or torch.npu.device_count() < 2:
        pytest.skip("requires at least two visible NPU devices")
    lower, upper = DEFAULT_LOWER, DEFAULT_UPPER

    for dev in (0, 1):
        device = f"{flag_gems.device}:{dev}"
        generator = torch.Generator(device=device)
        generator.manual_seed(2026)
        inp = torch.full((4096,), -1.0, device=device)
        noise = torch.zeros_like(inp)
        snapshots = []
        for _ in range(12):
            inp.fill_(-1.0)
            flag_gems.rrelu_with_noise_(inp, noise, lower, upper, True, generator)
            assert torch.all(noise >= lower) and torch.all(noise <= upper)
            utils.gems_assert_close(inp, utils.to_reference(-noise), torch.float32)
            snapshots.append(noise.clone())
        for earlier, later in zip(snapshots, snapshots[1:]):
            assert not torch.equal(earlier, later)
        assert _ascend_graph_entry(True, inp, noise, lower, upper, inp) is not None
        generator.manual_seed(2026)
        inp.fill_(-1.0)
        flag_gems.rrelu_with_noise_(inp, noise, lower, upper, True, generator)
        utils.gems_assert_equal(noise, utils.to_reference(snapshots[0]))


@pytest.mark.rrelu_with_noise
def test_rrelu_with_noise_eval_is_side_effect_free():
    """Eval mode consumes no randomness and writes to neither input tensor."""
    generator = torch.Generator(device=flag_gems.device)
    generator.manual_seed(2026)
    state_before = generator.get_state().clone()
    inp = torch.randn((257,), device=flag_gems.device)
    noise = torch.randn_like(inp)
    noise_before = noise.clone()
    input_before = inp.clone()

    flag_gems.rrelu_with_noise(
        inp, noise, DEFAULT_LOWER, DEFAULT_UPPER, False, generator
    )

    assert torch.equal(generator.get_state(), state_before)
    assert torch.equal(noise, noise_before)
    assert torch.equal(inp, input_before)


@pytest.mark.rrelu_with_noise_
def test_rrelu_with_noise_inplace_eval_is_side_effect_free():
    """Eval mode consumes no randomness and writes to the noise buffer only."""
    generator = torch.Generator(device=flag_gems.device)
    generator.manual_seed(2026)
    state_before = generator.get_state().clone()
    inp = torch.randn((257,), device=flag_gems.device)
    noise = torch.randn_like(inp)
    noise_before = noise.clone()

    flag_gems.rrelu_with_noise_(
        inp, noise, DEFAULT_LOWER, DEFAULT_UPPER, False, generator
    )

    assert torch.equal(generator.get_state(), state_before)
    assert torch.equal(noise, noise_before)


@pytest.mark.rrelu_with_noise
@pytest.mark.parametrize("training", [False, True])
def test_rrelu_with_noise_aliasing(training):
    """The out-of-place variant returns a buffer that aliases neither ``self``
    nor ``noise``."""
    lower, upper = EQUAL_BOUNDS if training else (DEFAULT_LOWER, DEFAULT_UPPER)
    inp = torch.randn((37, 11), dtype=torch.float32, device=flag_gems.device)
    noise = torch.zeros_like(inp) if training else torch.rand_like(inp)
    input_ptr = inp.data_ptr()
    noise_ptr = noise.data_ptr()
    input_before = inp.clone()

    result = flag_gems.rrelu_with_noise(inp, noise, lower, upper, training)

    assert result.data_ptr() != input_ptr
    assert result.data_ptr() != noise_ptr
    assert torch.equal(inp, input_before)


@pytest.mark.rrelu_with_noise_
@pytest.mark.parametrize("training", [False, True])
def test_rrelu_with_noise_inplace_aliasing(training):
    """The in-place variant writes into and returns ``self``."""
    lower, upper = EQUAL_BOUNDS if training else (DEFAULT_LOWER, DEFAULT_UPPER)
    inp = torch.randn((37, 11), dtype=torch.float32, device=flag_gems.device)
    noise = torch.zeros_like(inp) if training else torch.rand_like(inp)
    input_ptr = inp.data_ptr()

    result = flag_gems.rrelu_with_noise_(inp, noise, lower, upper, training)

    assert result.data_ptr() == input_ptr


@pytest.mark.rrelu_with_noise
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("shape", [(0,), (0, 7), (2, 0, 3)])
def test_rrelu_with_noise_empty(training, shape):
    """Empty inputs keep the shape and the dtype."""
    inp = torch.empty(shape, device=flag_gems.device)
    noise = torch.empty_like(inp)

    result = flag_gems.rrelu_with_noise(
        inp, noise, DEFAULT_LOWER, DEFAULT_UPPER, training
    )

    assert result.shape == inp.shape
    assert result.dtype == inp.dtype
    # The out-of-place aliasing contract is not asserted here: zero-sized
    # buffers all share one address, so neither the pointer nor the object
    # identity tells an allocated result from the input.  See
    # test_rrelu_with_noise_aliasing for that contract.


@pytest.mark.rrelu_with_noise_
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("shape", [(0,), (0, 7), (2, 0, 3)])
def test_rrelu_with_noise_inplace_empty(training, shape):
    """Empty inputs keep the shape, the dtype and the in-place contract."""
    inp = torch.empty(shape, device=flag_gems.device)
    noise = torch.empty_like(inp)
    input_ptr = inp.data_ptr()

    result = flag_gems.rrelu_with_noise_(
        inp, noise, DEFAULT_LOWER, DEFAULT_UPPER, training
    )

    assert result.shape == inp.shape
    assert result.dtype == inp.dtype
    assert result.data_ptr() == input_ptr


@pytest.mark.rrelu_with_noise
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("dtype", utils.PRIMARY_FLOAT_DTYPES)
def test_rrelu_with_noise_non_contiguous(training, dtype):
    """Strided views match ATen and leave the memory outside the view alone."""
    _skip_half_cpu_reference(dtype, training)
    lower, upper = EQUAL_BOUNDS if training else (DEFAULT_LOWER, DEFAULT_UPPER)
    input_base = torch.linspace(
        -2.0, 2.0, 17 * 22, dtype=dtype, device=flag_gems.device
    ).reshape(17, 22)
    noise_base = torch.zeros_like(input_base)
    untouched_input = input_base[:, 1::2].clone()
    untouched_noise = noise_base[:, 1::2].clone()

    inp = input_base[:, ::2]
    noise = noise_base[:, ::2]
    assert not inp.is_contiguous()
    assert not noise.is_contiguous()

    ref_inp = utils.to_reference(inp.clone())
    ref_noise = utils.to_reference(noise.clone())
    ref_out = torch.ops.aten.rrelu_with_noise(
        ref_inp, ref_noise, lower, upper, training
    )

    result = flag_gems.rrelu_with_noise(inp, noise, lower, upper, training)

    utils.gems_assert_close(result, ref_out, dtype)
    utils.gems_assert_close(noise, ref_noise, dtype)
    assert torch.equal(input_base[:, 1::2], untouched_input)
    assert torch.equal(noise_base[:, 1::2], untouched_noise)


@pytest.mark.rrelu_with_noise_
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("dtype", utils.PRIMARY_FLOAT_DTYPES)
def test_rrelu_with_noise_inplace_non_contiguous(training, dtype):
    """Strided views match ATen and leave the memory outside the view alone."""
    _skip_half_cpu_reference(dtype, training)
    lower, upper = EQUAL_BOUNDS if training else (DEFAULT_LOWER, DEFAULT_UPPER)
    input_base = torch.linspace(
        -2.0, 2.0, 17 * 22, dtype=dtype, device=flag_gems.device
    ).reshape(17, 22)
    noise_base = torch.zeros_like(input_base)
    untouched_input = input_base[:, 1::2].clone()
    untouched_noise = noise_base[:, 1::2].clone()

    inp = input_base[:, ::2]
    noise = noise_base[:, ::2]
    assert not inp.is_contiguous()
    assert not noise.is_contiguous()

    ref_inp = utils.to_reference(inp.clone())
    ref_noise = utils.to_reference(noise.clone())
    ref_out = torch.ops.aten.rrelu_with_noise_(
        ref_inp, ref_noise, lower, upper, training
    )

    result = flag_gems.rrelu_with_noise_(inp, noise, lower, upper, training)

    utils.gems_assert_close(result, ref_out, dtype)
    utils.gems_assert_close(noise, ref_noise, dtype)
    assert torch.equal(input_base[:, 1::2], untouched_input)
    assert torch.equal(noise_base[:, 1::2], untouched_noise)


def _layout_inputs(dtype):
    """Dense inputs whose layout differs from the result ATen returns.

    Both are non-overlapping and dense, which is what makes them worth
    checking: an allocation that preserves the input's layout (as
    ``torch.empty_like`` does by default) keeps the layout here, while ATen
    hands back a legacy contiguous result.
    """
    layouts = []
    channels_last_base = torch.linspace(
        -2.0, 2.0, 2 * 8 * 4 * 4, dtype=dtype, device=flag_gems.device
    ).reshape(2, 8, 4, 4)
    try:
        layouts.append(
            ("channels_last", channels_last_base.to(memory_format=torch.channels_last))
        )
    except RuntimeError as exc:
        # Some backends (e.g. torch_npu: "Only contiguous_format or
        # preserve_format is supported.") lack the channels_last memory
        # format entirely; only the transposed case runs there.
        if "preserve_format" not in str(exc):
            raise
    transposed = (
        torch.linspace(-2.0, 2.0, 8 * 16, dtype=dtype, device=flag_gems.device)
        .reshape(8, 16)
        .t()
    )
    layouts.append(("transposed", transposed))
    return layouts


@pytest.mark.rrelu_with_noise
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("dtype", utils.PRIMARY_FLOAT_DTYPES)
def test_rrelu_with_noise_output_strides(training, dtype):
    """The out-of-place result comes back contiguous, as ATen returns it.

    ``aten::rrelu_with_noise`` allocates its result in legacy contiguous
    layout whatever layout the input has, so a channels-last or strided input
    must not carry its layout into the result.  The values are checked as well
    as the strides, because with the two tensors in different layouts it is
    the indexing that can go wrong.
    """
    _skip_half_cpu_reference(dtype, training)
    lower, upper = EQUAL_BOUNDS if training else (DEFAULT_LOWER, DEFAULT_UPPER)

    for label, inp in _layout_inputs(dtype):
        assert not inp.is_contiguous(), label
        noise = torch.zeros_like(inp)
        # The reference gets a contiguous noise buffer on purpose.  ATen's CUDA
        # training kernel does ``noise_.contiguous()`` and samples into that copy,
        # so with the input's layout the reference buffer would come back
        # untouched.  ours keeps the input's layout, which is the interesting
        # case; EQUAL_BOUNDS samples one slope value for every element, so the
        # two buffers are still comparable elementwise.
        ref_noise = utils.to_reference(
            torch.zeros_like(inp, memory_format=torch.contiguous_format)
        )

        ref_out = torch.ops.aten.rrelu_with_noise(
            utils.to_reference(inp.clone()), ref_noise, lower, upper, training
        )

        result = flag_gems.rrelu_with_noise(inp, noise, lower, upper, training)

        assert result.is_contiguous(), label
        assert result.stride() == ref_out.stride(), label
        utils.gems_assert_close(result, ref_out, dtype)
        utils.gems_assert_close(noise, ref_noise, dtype)


@pytest.mark.rrelu_with_noise_
@pytest.mark.parametrize("training", [False, True])
def test_rrelu_with_noise_inplace_output_strides(training):
    """The in-place result stays in the caller's layout, as ATen's does."""
    lower, upper = EQUAL_BOUNDS if training else (DEFAULT_LOWER, DEFAULT_UPPER)

    for label, inp in _layout_inputs(torch.float32):
        assert not inp.is_contiguous(), label
        noise = torch.zeros_like(inp)
        # Contiguous for the reference, see test_rrelu_with_noise_output_strides.
        ref_noise = utils.to_reference(
            torch.zeros_like(inp, memory_format=torch.contiguous_format)
        )

        ref_inp = inp.clone()
        if ref_inp.stride() != inp.stride():
            # The reference input must carry the caller's layout; on backends
            # whose clone() materializes a contiguous tensor (e.g. torch_npu)
            # rebuild it with an explicit strided allocation instead.
            ref_inp = torch.empty_strided(
                inp.shape, inp.stride(), dtype=inp.dtype, device=inp.device
            ).copy_(inp)
        ref_inp = utils.to_reference(ref_inp)
        if ref_inp.stride() != inp.stride():
            # The reference transfer itself normalized the layout (e.g.
            # torch_npu's .to("cpu")), so the layout comparison cannot run.
            pytest.skip(
                f"{label}: the reference path does not preserve the input layout"
            )

        ref_out = torch.ops.aten.rrelu_with_noise_(
            ref_inp, ref_noise, lower, upper, training
        )

        result = flag_gems.rrelu_with_noise_(inp, noise, lower, upper, training)

        assert result.stride() == ref_out.stride(), label
        utils.gems_assert_close(result, ref_out, torch.float32)
        utils.gems_assert_close(noise, ref_noise, torch.float32)


INVALID_ARG_CASES = [
    ("noise_shape", "same shape"),
    ("noise_dtype", "same dtype"),
    ("self_dtype", "not implemented"),
    ("noise_device", "same device"),
    ("lower_infinite", "must be finite"),
    ("upper_nan", "must be finite"),
    ("bounds_swapped", "less than or equal"),
]


def _invalid_args_case(case):
    """Build the (self, noise, lower, upper) tuple of one rejection case."""
    inp = torch.randn((5,), dtype=torch.float32, device=flag_gems.device)
    noise = torch.zeros_like(inp)
    lower, upper = DEFAULT_LOWER, DEFAULT_UPPER

    if case == "noise_shape":
        noise = torch.zeros((3,), dtype=torch.float32, device=flag_gems.device)
    elif case == "noise_dtype":
        noise = torch.zeros((5,), dtype=torch.int32, device=flag_gems.device)
    elif case == "self_dtype":
        inp = torch.ones((5,), dtype=torch.int32, device=flag_gems.device)
        noise = torch.zeros((5,), dtype=torch.int32, device=flag_gems.device)
    elif case == "noise_device":
        if flag_gems.device == "cpu":
            pytest.skip("needs a second device to pair with the CPU reference")
        noise = torch.zeros((5,), dtype=torch.float32, device="cpu")
    elif case == "lower_infinite":
        lower = float("inf")
    elif case == "upper_nan":
        upper = float("nan")
    elif case == "bounds_swapped":
        lower, upper = DEFAULT_UPPER, DEFAULT_LOWER

    return inp, noise, lower, upper


@pytest.mark.rrelu_with_noise
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("case,match", INVALID_ARG_CASES)
def test_rrelu_with_noise_invalid_args(training, case, match):
    """Invalid arguments are rejected before either tensor is touched."""
    inp, noise, lower, upper = _invalid_args_case(case)
    input_before = inp.clone()
    noise_before = noise.clone()

    with pytest.raises(RuntimeError, match=match):
        flag_gems.rrelu_with_noise(inp, noise, lower, upper, training)

    assert torch.equal(inp, input_before)
    assert torch.equal(noise, noise_before)


@pytest.mark.rrelu_with_noise_
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("case,match", INVALID_ARG_CASES)
def test_rrelu_with_noise_inplace_invalid_args(training, case, match):
    """Invalid arguments are rejected before either tensor is touched."""
    inp, noise, lower, upper = _invalid_args_case(case)
    input_before = inp.clone()
    noise_before = noise.clone()

    with pytest.raises(RuntimeError, match=match):
        flag_gems.rrelu_with_noise_(inp, noise, lower, upper, training)

    assert torch.equal(inp, input_before)
    assert torch.equal(noise, noise_before)
