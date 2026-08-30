"""Test 10 of NIFTY50_PLAN.md sec.8: the multi-layer extraction pass must not move A0.

`hidden.py --layers ... --pool ...` reads several states out of one forward pass. The
whole representation ladder (arms A1-A5) is only interpretable if the pass that produces
it still produces the record's cell -- final post-`norm` state, last position -- exactly
as the 2026-08-28 run did. So the regression is pinned against the artifact on disk,
`bench/out/nse/ml/hT_pretrained_pilot256.npy`, not against a re-derivation.

The two "differs" assertions are the other half of the same claim: if L12-pre-norm were
equal to the post-norm state, or mean10 equal to last, the new flags would be dumping
duplicates of A0 under new names and the ladder would be measuring nothing.

The last two tests cover arm A5 on its own 128-cell pilot: that the tokenizer's
pre-quantisation z is stored float32 (it overflows fp16, which is how the arm was first
found broken) and that reading it does not disturb the trunk state, because --tokenizer-z
reimplements tok.encode inline to get at the intermediate.

Marked slow because a cold run has to load Kronos-base and push 256 + 128 windows through
MPS (~15 s each including the checkpoint load). When the pilot artifacts are already on
disk the tests are file comparisons.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

ML = os.path.join(ROOT, "bench", "out", "nse", "ml")
N50 = os.path.join(ROOT, "bench", "out", "nse", "n50")
CACHED_A0 = os.path.join(ML, "hT_pretrained_pilot256.npy")
PILOT = {
    "final_last": os.path.join(N50, "hT_base_pretrained_final_last_pilot256.npy"),
    "final_mean10": os.path.join(N50, "hT_base_pretrained_final_mean10_pilot256.npy"),
    "L12_last": os.path.join(N50, "hT_base_pretrained_L12_last_pilot256.npy"),
    "L12_mean10": os.path.join(N50, "hT_base_pretrained_L12_mean10_pilot256.npy"),
}
PILOT_ARGV = ["bench/nse/hidden.py", "--arm", "pretrained", "--layers", "12",
              "--pool", "last,mean10", "--limit", "256",
              "--out-dir", "bench/out/nse/n50"]
# Arm A5 on a SHORTER pilot on purpose: 128 cells means its final-state array is
# _pilot128, so it cannot overwrite the _pilot256 artifacts the regression above compares,
# and rows 0..127 of the cached A0 are still the same 128 cells (the cell order is the
# windows_index row order and --limit takes a prefix of it).
TOKZ = {"final_last": os.path.join(N50, "hT_base_pretrained_final_last_pilot128.npy"),
        "z_last": os.path.join(N50, "hT_tokz_last_pilot128.npy"),
        "z_mean10": os.path.join(N50, "hT_tokz_mean10_pilot128.npy")}
TOKZ_ARGV = ["bench/nse/hidden.py", "--arm", "pretrained", "--tokenizer-z",
             "--limit", "128", "--out-dir", "bench/out/nse/n50"]
FP16_MAX = 65504.0

pytestmark = pytest.mark.slow


def _interpreter() -> str:
    """A Python that can import huggingface_hub, since hidden.py has to fetch weights.

    The repo venv is the environment the GPU queue runs under; a bare pyenv python is
    fine for reading .npy but cannot load a checkpoint.
    """
    for cand in (sys.executable, os.path.join(ROOT, ".venv", "bin", "python")):
        if cand and os.path.exists(cand) and subprocess.run(
                [cand, "-c", "import huggingface_hub, torch"],
                capture_output=True).returncode == 0:
            return cand
    return ""


@pytest.fixture(scope="module")
def pilot() -> dict:
    """The 256-cell multi-layer pass, run once if its artifacts are not already there."""
    if not os.path.exists(CACHED_A0):
        pytest.skip(f"{CACHED_A0} missing: run cache_windows.py + hidden.py first")
    if not all(os.path.exists(p) for p in PILOT.values()):
        exe = _interpreter()
        if not exe:
            pytest.skip("no interpreter with huggingface_hub+torch to run the pilot")
        env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")
        r = subprocess.run([exe] + PILOT_ARGV, cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, f"pilot run failed:\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}"
    return {k: np.load(p).astype(np.float32) for k, p in PILOT.items()}


@pytest.fixture(scope="module")
def tokz() -> dict:
    """The 128-cell arm-A5 pass, run once if its artifacts are not already there."""
    if not os.path.exists(CACHED_A0):
        pytest.skip(f"{CACHED_A0} missing: run cache_windows.py + hidden.py first")
    if not all(os.path.exists(p) for p in TOKZ.values()):
        exe = _interpreter()
        if not exe:
            pytest.skip("no interpreter with huggingface_hub+torch to run the pilot")
        env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")
        r = subprocess.run([exe] + TOKZ_ARGV, cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, f"A5 run failed:\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}"
    return {k: np.load(p) for k, p in TOKZ.items()}


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def test_final_last_reproduces_cached_a0(pilot):
    """A0 out of the multi-layer pass == the cached A0, to fp16 tolerance.

    Tolerance is expressed per feature dimension: the columns of h_T differ in scale by
    orders of magnitude, so a single absolute bound would be vacuous on the small ones and
    unreachable on the large ones.
    """
    a = np.load(CACHED_A0).astype(np.float32)
    b = pilot["final_last"]
    assert a.shape == b.shape == (256, 832)
    r = _pearson(a, b)
    diff = np.abs(a - b)
    scale = a.std(axis=0) + 1e-12
    rel = float((diff.max(axis=0) / scale).max())
    print(f"\n  pearson r = {r:.12f}  max|diff| = {diff.max():.6g}  "
          f"max per-dim |diff|/sd = {rel:.6g}  (bit-exact: {np.array_equal(a, b)})")
    assert r > 0.9999, f"A0 moved: r = {r}"
    assert rel < 5e-3, f"A0 moved: max per-dim |diff|/sd = {rel}"


def test_layer12_pre_norm_differs_from_final_post_norm(pilot):
    """L12 is the residual stream BEFORE mdl.norm; the final state is after it.

    RMSNorm rescales each position onto a fixed radius, so the two are not the same vector
    and must not correlate like one. If this ever passes trivially, the hook is reading
    the post-norm tensor and arm A1 is a relabelled A0.
    """
    pre, post = pilot["L12_last"], pilot["final_last"]
    r = _pearson(pre, post)
    ratio = float(np.linalg.norm(pre) / np.linalg.norm(post))
    print(f"\n  corr(L12_last, final_last) = {r:.6f}  |L12|/|final| = {ratio:.2f}")
    assert not np.allclose(pre, post, atol=1e-3)
    assert r < 0.99, f"pre-norm state indistinguishable from post-norm: r = {r}"
    assert ratio > 2.0 or ratio < 0.5, f"norm layer did not rescale: ratio = {ratio}"


def test_mean10_differs_from_last(pilot):
    """Pooling over the last 10 positions is arm A2, not a copy of the last position."""
    for layer in ("final", "L12"):
        last, mean10 = pilot[f"{layer}_last"], pilot[f"{layer}_mean10"]
        r = _pearson(last, mean10)
        rel = float(np.linalg.norm(mean10 - last) / np.linalg.norm(last))
        print(f"\n  {layer}: corr(last, mean10) = {r:.6f}  "
              f"|mean10-last|/|last| = {rel:.4f}")
        assert not np.allclose(last, mean10, atol=1e-3)
        assert r < 0.99, f"{layer} mean10 is the last position: r = {r}"
        assert rel > 0.02, f"{layer} mean10 barely moved: rel = {rel}"


def test_pilot_metas_carry_provenance(pilot):
    """Every array gets a sidecar naming the state, the rank diagnostic and the argv."""
    import json
    for key, path in PILOT.items():
        meta = json.load(open(path.replace(".npy", "_meta.json")))
        assert meta["provenance"]["model_repo"] == "NeoQuasar/Kronos-base"
        assert meta["provenance"]["model_revision"]
        assert meta["provenance"]["tokenizer_repo"] == "NeoQuasar/Kronos-Tokenizer-base"
        assert meta["effective_rank"] > 1.0
        assert 1 <= meta["n_pcs_for_95pct_var"] <= 832
        assert meta["ms_per_series"] > 0
        assert "hidden.py" in " ".join(meta["argv"])
        assert meta["layer"] == (12 if key.startswith("L12") else "final")
        assert meta["pool"] == ("mean10" if key.endswith("mean10") else "last")


def test_tokenizer_z_is_float32_and_finite(tokz):
    """Arm A5 must be stored float32: the pre-quantisation z overflows fp16 on this panel.

    quant_embed's output feeds a BSQuantizer that normalises it onto the unit sphere before
    taking signs, so nothing downstream of the tokenizer constrains its scale -- and it runs
    past the fp16 ceiling of 65504 into inf, which then took out the rank diagnostic with
    "SVD did not converge". The |z| assertion is the point of the test: if a future change
    makes z small enough to fit fp16, this arm is no longer reading the state it claims to.
    """
    import json
    for key in ("z_last", "z_mean10"):
        z = tokz[key]
        peak = float(np.abs(z).max())
        print(f"\n  {key}: dtype {z.dtype}  shape {z.shape}  max|z| = {peak:.5g} "
              f"(fp16 ceiling {FP16_MAX:.0f})")
        assert z.dtype == np.float32, f"{key} stored as {z.dtype}, will overflow"
        assert np.isfinite(z).all(), f"{key} holds non-finite values"
        assert z.shape[1] == 20, "codebook_dim is s1_bits + s2_bits = 20"
        meta = json.load(open(TOKZ[key].replace(".npy", "_meta.json")))
        assert meta["dtype"] == "float32"
        assert meta["n_nonfinite_rows"] == 0
        assert meta["state"].startswith("tokenizer quant_embed")
    assert float(np.abs(tokz["z_last"]).max()) > FP16_MAX, (
        "z now fits in fp16; check quant_embed is still being read pre-BSQ")


def test_tokenizer_z_branch_does_not_move_the_trunk_state(tokz):
    """--tokenizer-z reimplements tok.encode inline; the trunk state must not notice.

    encode_windows takes a second path (embed -> encoder -> quant_embed -> tokenizer) to
    get at z, instead of calling tok.encode. If that copy ever drifts from the library, the
    s1/s2 ids change and every arm extracted with --tokenizer-z is silently a different
    experiment. Bit-identity against the cached A0's first 128 rows is the check.
    """
    a = np.load(CACHED_A0)[:128]
    b = tokz["final_last"]
    assert a.shape == b.shape == (128, 832)
    print(f"\n  A5 pass vs cached A0 (first 128 cells) bit-identical: "
          f"{np.array_equal(a, b)}")
    assert np.array_equal(a, b), "the inline encode copy has drifted from tok.encode"
