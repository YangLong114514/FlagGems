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

"""Extended stress sweeps for linalg_matrix_rank (Ascend).

These are the reproducible versions of the ad-hoc scans quoted in
matrix_rank_昇腾算子实现报告.md (366-case all-path sweep, 34-case hermitian
stress, 22-case QR-band adversarial).  They are SLOW (tens of minutes,
including Triton JIT compiles) and therefore skipped unless
FLAGGEMS_MR_SWEEP=1 is set:

    FLAGGEMS_MR_SWEEP=1 pytest -s tests/test_linalg_matrix_rank_sweeps.py

The exact paths are the default dispatch; the fast Gram/unpivoted-QR
bands are opt-in via FLAGGEMS_MR_FAST_PATH=1.  The sweeps therefore clear
any FLAGGEMS_MR_FAST_PATH leftover from the environment
(monkeypatch.delenv) to pin the exact dispatch; the 366-case all-paths
sweep is run twice -- once in the exact default and once in fast mode --
with separate allow-lists.
"""

import os

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils  # noqa: F401  (kept for parity)

VENDOR_NAME = getattr(flag_gems, "vendor_name", "")
IS_ASCEND = VENDOR_NAME == "ascend"

pytestmark = [
    pytest.mark.linalg_matrix_rank,
    pytest.mark.skipif(not IS_ASCEND, reason="Ascend-specific stress sweeps"),
    pytest.mark.skipif(
        os.environ.get("FLAGGEMS_MR_SWEEP") != "1",
        reason="slow sweep: set FLAGGEMS_MR_SWEEP=1 to run",
    ),
]

EPS32 = 1.1920929e-7
DEV = flag_gems.device


def _ref_rank(a, herm=False, atol=None, rtol=None):
    a64 = a.to(torch.float64).cpu()
    if herm:
        a64 = torch.tril(a64) + torch.tril(a64, -1).mT
    if rtol is None:
        rtol = max(a.shape[-2], a.shape[-1]) * EPS32
    if atol is None:
        atol = 0.0
    return torch.linalg.matrix_rank(a64, atol=atol, rtol=rtol, hermitian=herm)


def _gen(shape, kind, herm):
    if kind == "rand":
        mt = torch.randn(shape)
    elif kind == "diag":
        mt = torch.zeros(shape)
        r = min(shape[-2], shape[-1]) - 1
        dg = torch.arange(r)
        mt[..., dg, dg] = torch.arange(1, r + 1, dtype=torch.float32)
    else:  # lowrank
        r = max(min(shape[-2], shape[-1]) // 2, 1)
        mt = (
            torch.randn(*shape[:-2], shape[-2], r)
            @ torch.randn(*shape[:-2], r, shape[-1])
        )
    if herm:
        mt = mt + mt.mT
    return mt.float()


def _mk_lowrank(shape, r, seed=0):
    g = torch.Generator().manual_seed(seed)
    *batch, m, n = shape
    u = torch.linalg.qr(torch.randn(*batch, m, r, generator=g, dtype=torch.float64))[0]
    v = torch.linalg.qr(torch.randn(*batch, n, r, generator=g, dtype=torch.float64))[0]
    s = torch.logspace(0, -4, r, dtype=torch.float64)
    return ((u * s) @ v.mT).float()


def _mk_herm_lowrank(k, r, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.linalg.qr(torch.randn(k, k, generator=g, dtype=torch.float64))[0]
    lam = torch.cat(
        [
            torch.logspace(0, -4, r, dtype=torch.float64),
            torch.zeros(k - r, dtype=torch.float64),
        ]
    )
    return ((q * lam) @ q.mT).float()


def _mk_herm_eigs(eigs, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = eigs.numel()
    q = torch.linalg.qr(torch.randn(k, k, generator=g, dtype=torch.float64))[0]
    return ((q * eigs[None, :]) @ q.mT).float()


def _run_all_paths_sweep():
    """Full-dispatch sweep (366 cases): square k=3..64 x {rand,diag,lowrank},
    hermitian small, long-dimension tall/wide/batch, batched small.
    Reference: torch CPU fp32 matrix_rank (same-dtype semantics).  Returns
    the mismatch list; the two wrappers below assert on it per mode."""
    torch.manual_seed(11)
    cases = []
    for k in list(range(3, 65)):
        cases.append(((k, k), False))
    for k in [3, 8, 16, 17, 31, 32, 33, 48, 64]:
        cases.append(((k, k), True))
    for rows in [128, 256, 1024]:
        for k in [3, 8, 33, 64]:
            cases.append(((rows, k), False))
            cases.append(((k, rows), False))
            cases.append(((2, rows, k), False))
            cases.append(((2, k, rows), False))
    cases.append(((2, 33, 33), False))
    cases.append(((4, 16, 16), False))
    cases.append(((2, 17, 17), True))

    mismatches = []
    total = 0
    for shape, herm in cases:
        for kind in ["rand", "diag", "lowrank"]:
            mt = _gen(shape, kind, herm)
            ref = torch.linalg.matrix_rank(mt, hermitian=herm)
            got = flag_gems.linalg_matrix_rank(mt.to(DEV), hermitian=herm).cpu()
            total += 1
            if not torch.equal(got, ref):
                mismatches.append(
                    f"shape={shape} herm={herm} kind={kind}: "
                    f"got={got.flatten().tolist()[:4]} ref={ref.flatten().tolist()[:4]}"
                )
    assert total == 366
    return mismatches


def _mismatch_shape(entry):
    shape_str = entry.split(" herm")[0].replace("shape=", "")
    return eval(shape_str)  # noqa: S307 (test data only)


# fp32 noise-region boundary: the fp32 CPU reference itself disagrees with
# fp64 on these tiny slowly-decaying spectra, so a lowrank mismatch there is
# not a backend defect (independent of the dispatch mode).
_NOISE_REGION_SHAPES = {(3, 3), (7, 7)}

# Long-dimension k <= 64 shapes whose lowrank mismatches are the documented
# Gram sigma^2-domain floor -- only reachable under FLAGGEMS_MR_FAST_PATH=1.
_GRAM_FLOOR_SHAPES = {
    (128, 3), (3, 128), (2, 128, 3), (2, 3, 128),
    (128, 8), (8, 128), (2, 128, 8), (2, 8, 128),
    (128, 33), (33, 128), (2, 128, 33), (2, 33, 128),
    (128, 64), (64, 128), (2, 128, 64), (2, 64, 128),
    (256, 3), (3, 256), (2, 256, 3), (2, 3, 256),
    (256, 8), (8, 256), (2, 256, 8), (2, 8, 256),
    (256, 33), (33, 256), (2, 256, 33), (2, 33, 256),
    (256, 64), (64, 256), (2, 256, 64), (2, 64, 256),
    (1024, 3), (3, 1024), (2, 1024, 3), (2, 3, 1024),
    (1024, 8), (8, 1024), (2, 1024, 8), (2, 8, 1024),
    (1024, 33), (33, 1024), (2, 1024, 33), (2, 33, 1024),
    (1024, 64), (64, 1024), (2, 1024, 64), (2, 64, 1024),
}


def test_sweep_all_paths_366_exact_default(monkeypatch):
    # Exact default dispatch: only the fp32 noise-region boundary cases may
    # mismatch.  In particular EVERY long-dimension lowrank case must pass
    # -- the Gram sigma^2-domain floor is gone from the default dispatch,
    # and this test locks that in (a regression that routes the default
    # back to Gram fails here).
    monkeypatch.delenv("FLAGGEMS_MR_FAST_PATH", raising=False)
    mismatches = _run_all_paths_sweep()
    unexpected = [
        m
        for m in mismatches
        if "lowrank" not in m
        or "herm=True" in m
        or _mismatch_shape(m) not in _NOISE_REGION_SHAPES
    ]
    assert not unexpected, "\n".join(unexpected)


def test_sweep_all_paths_366_fast_mode(monkeypatch):
    # Fast mode (FLAGGEMS_MR_FAST_PATH=1): the long-dimension Gram band
    # overestimates rank on slowly-decaying low-rank spectra (sigma^2
    # domain floor, documented); those mismatches are allow-listed per
    # shape, everything else must match exactly.
    monkeypatch.setenv("FLAGGEMS_MR_FAST_PATH", "1")
    mismatches = _run_all_paths_sweep()
    unexpected = [m for m in mismatches if "lowrank" not in m or "herm=True" in m]
    assert not unexpected, "\n".join(unexpected)
    allowed = _GRAM_FLOOR_SHAPES | _NOISE_REGION_SHAPES
    gram_floor = [m for m in mismatches if "lowrank" in m]
    for m in gram_floor:
        assert _mismatch_shape(m) in allowed, f"unexpected mismatch: {m}"


def test_sweep_hermitian_34(monkeypatch):
    # Hermitian stress (34 cases): rand/lowrank/near-default-tol (both
    # signs)/atol cluster/slow-decay through the sqrt(eps) floor/zero/
    # garbage strict upper/batch.  Reference: fp64 with fp32-semantics tol.
    # This is the acceptance scan for the exact herm path, which is the
    # default dispatch; clear any FAST_PATH leftover so the exact path is
    # really exercised.  (Under FLAGGEMS_MR_FAST_PATH=1 the 65..255 QR
    # band has the documented |R_ii| limitations on these spectra.)
    monkeypatch.delenv("FLAGGEMS_MR_FAST_PATH", raising=False)
    torch.manual_seed(7)
    mismatches = []
    total = 0

    def run(name, a, atol=None, rtol=None):
        nonlocal total
        total += 1
        got = flag_gems.linalg_matrix_rank(
            a.to(DEV), hermitian=True,
            **({"atol": atol} if atol is not None else {}),
            **({"rtol": rtol} if rtol is not None else {}),
        ).cpu()
        want = _ref_rank(a, herm=True, atol=atol, rtol=rtol)
        if not torch.equal(got.reshape(-1), want.reshape(-1)):
            mismatches.append(
                f"{name}: got={got.reshape(-1).tolist()} want={want.reshape(-1).tolist()}"
            )

    for k in [65, 100, 128, 256, 513, 1024]:
        run(f"rand sym ({k},{k})", _mk_herm_eigs(torch.randn(k, dtype=torch.float64), seed=k))
        e = torch.zeros(k, dtype=torch.float64)
        e[: k // 3] = torch.randn(k // 3, dtype=torch.float64) * 2
        run(f"lowrank ({k},{k})", _mk_herm_eigs(e, seed=k + 1))
        smax = 3.0
        tol = k * EPS32 * smax
        e = torch.full((k,), smax, dtype=torch.float64)
        e[k // 2] = tol * 1.5
        e[k // 2 + 1] = -tol * 1.5
        e[k // 2 + 2] = tol * 0.5
        e[k // 2 + 3] = -tol * 0.5
        run(f"near-default-tol ({k},{k})", _mk_herm_eigs(e, seed=k + 2))
        e = torch.full((k,), 2.0, dtype=torch.float64)
        e[-4:] = torch.tensor([0.11, -0.11, 0.09, -0.09], dtype=torch.float64)
        run(f"atol cluster ({k},{k})", _mk_herm_eigs(e, seed=k + 3), atol=0.1)
        e = torch.logspace(0, -6, k, dtype=torch.float64)
        e[k // 2 :] *= -1
        run(f"slowdecay ({k},{k})", _mk_herm_eigs(e, seed=k + 4))

    for k in [65, 513]:
        run(f"zero ({k},{k})", torch.zeros(k, k))

    k = 300
    g = torch.Generator().manual_seed(k)
    lower = torch.tril(torch.randn(k, k, generator=g))
    a = lower.clone()
    a.masked_fill_(torch.triu(torch.ones(k, k, dtype=torch.bool), 1), 1e6)
    run("garbage upper (300,300)", a)

    e = torch.zeros(100, dtype=torch.float64)
    e[:20] = 1.0
    b = torch.stack(
        [_mk_herm_eigs(e, seed=1), _mk_herm_eigs(torch.full((100,), 1.0, dtype=torch.float64), seed=2)]
    )
    total += 1
    got = flag_gems.linalg_matrix_rank(b.to(DEV), hermitian=True).cpu()
    want = torch.linalg.matrix_rank(
        (torch.tril(b.double()) + torch.tril(b.double(), -1).mT),
        hermitian=True, rtol=100 * EPS32,
    )
    if not torch.equal(got, want):
        mismatches.append(f"batch herm: got={got.tolist()} want={want.tolist()}")

    assert not mismatches, "\n".join(mismatches)


def test_sweep_qr_band_adversarial_22(monkeypatch):
    # QR-band adversarial (22 cases): slow-decay low-rank at 65..512,
    # near-threshold atol on 256^2/512^2 random matrices (5 seeds each),
    # per-batch tensor atol.  This sweep is the acceptance scan for the
    # exact path, which is the default dispatch; clear any FAST_PATH
    # leftover so the exact path is really exercised.
    monkeypatch.delenv("FLAGGEMS_MR_FAST_PATH", raising=False)
    mismatches = []
    total = 0

    def run(name, a, herm=False, atol=None, rtol=None):
        nonlocal total
        total += 1
        got = flag_gems.linalg_matrix_rank(
            a.to(DEV).float().contiguous(), atol=atol, rtol=rtol, hermitian=herm
        ).cpu()
        want = _ref_rank(a, herm, atol, rtol)
        if not torch.equal(got.reshape(-1), want.reshape(-1)):
            mismatches.append(
                f"{name}: got={got.reshape(-1).tolist()} want={want.reshape(-1).tolist()}"
            )

    for shape, r in [((65, 65), 32), ((128, 128), 64), ((256, 256), 100),
                     ((512, 512), 200), ((256, 512), 100), ((512, 256), 100),
                     ((2, 128, 128), 50), ((100, 300), 40), ((300, 100), 40)]:
        run(f"lowrank {shape} r={r}", _mk_lowrank(shape, r))
    run("herm lowrank (256,256) r=100", _mk_herm_lowrank(256, 100), herm=True)
    run("herm lowrank (128,128) r=60", _mk_herm_lowrank(128, 60), herm=True)

    for seed in range(5):
        a = torch.randn(256, 256, dtype=torch.float64,
                        generator=torch.Generator().manual_seed(100 + seed)).float()
        s = torch.linalg.svdvals(a.to(torch.float64))
        j = 128
        atol = ((s[j] * s[j + 1]).sqrt() * 0.999).item()
        run(f"threshold (256,256) seed{seed}", a, atol=atol)

    for seed in range(5):
        a = torch.randn(512, 512, dtype=torch.float64,
                        generator=torch.Generator().manual_seed(200 + seed)).float()
        s = torch.linalg.svdvals(a.to(torch.float64))
        j = 256
        atol = ((s[j] * s[j + 1]).sqrt() * 0.999).item()
        run(f"threshold (512,512) seed{seed}", a, atol=atol)

    a = _mk_lowrank((2, 100, 100), 40)
    atol_t = torch.tensor([0.0, 1e-3]).to(DEV)
    total += 1
    got = flag_gems.linalg_matrix_rank(a.to(DEV), atol=atol_t).cpu()
    want = torch.stack([_ref_rank(a[0], atol=0.0), _ref_rank(a[1], atol=1e-3)])
    if not torch.equal(got.reshape(-1), want.reshape(-1)):
        mismatches.append(
            f"batch tensor-atol: got={got.tolist()} want={want.tolist()}"
        )

    assert not mismatches, "\n".join(mismatches)
