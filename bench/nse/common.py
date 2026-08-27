"""Shared paths, provenance sidecars and small utilities for the NSE ranking build."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

DATA_NSE = os.path.join(ROOT, "data", "nse")
OUT_NSE = os.path.join(ROOT, "bench", "out", "nse")
PROCESSED = os.path.join(DATA_NSE, "processed")

# Kronos HF checkpoints, revision-pinned the way tests/test_kronos_regression.py does.
# NOTE: tests/test_kronos_regression.py's MODEL_REVISION (901c26c1...) is the SHA of
# Kronos-*small*, which is what that test loads. Reusing it for Kronos-base 404s.
# These SHAs were resolved from the HF API on 2026-08-27.
MODEL_REPO = "NeoQuasar/Kronos-base"
TOKENIZER_REPO = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_REVISION = "2b554741eca47781b64468546e77fef3e85130e6"
TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"
MODEL_REVISIONS = {
    "NeoQuasar/Kronos-base": "2b554741eca47781b64468546e77fef3e85130e6",
    "NeoQuasar/Kronos-small": "901c26c1332695a2a8f243eb2f37243a37bea320",
    "NeoQuasar/Kronos-mini": "f4e68697d9d5aed55cef5c96aabc3376bcad9f81",
}

FEATURE_LIST = ["open", "high", "low", "close", "vol", "amt"]
TIME_FEATURE_LIST = ["minute", "hour", "weekday", "day", "month"]
CLOSE_IDX = FEATURE_LIST.index("close")  # 3

SIGNAL_DEFS = ["last", "mean", "max", "min"]
SCALINGS = ["z", "ret", "recon"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def pkg_versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "pandas", "torch", "yfinance", "scipy", "pyarrow"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:  # noqa: BLE001
            out[mod] = "missing"
    return out


def sidecar(path: str, args: dict, extra: dict | None = None) -> None:
    """Write a JSON provenance record next to any produced artifact."""
    rec = {
        "written_at": utc_now(),
        "git_sha": git_sha(),
        "argv": sys.argv,
        "args": {k: (str(v) if not isinstance(v, (int, float, str, bool, type(None), list, dict)) else v)
                 for k, v in args.items()},
        "packages": pkg_versions(),
    }
    # Stamp what was ACTUALLY loaded. Stamping the module constants unconditionally made
    # every finetuned run's sidecar claim it was Kronos-base at the pinned revision --
    # recoverable from args.model_path, but a provenance record that needs cross-checking
    # against the thing it is supposed to certify is not doing its job.
    mp, tp = args.get("model_path"), args.get("tokenizer_path")
    mr = args.get("model_repo") or MODEL_REPO
    if mp:
        rec["model_repo"] = None
        rec["model_path"] = str(mp)
        rec["model_revision"] = None      # a local checkpoint has no hub revision
    else:
        rec["model_repo"] = mr
        rec["model_revision"] = MODEL_REVISIONS.get(mr, MODEL_REVISION)
    if tp:
        rec["tokenizer_repo"] = None
        rec["tokenizer_path"] = str(tp)
        rec["tokenizer_revision"] = None
    else:
        rec["tokenizer_repo"] = TOKENIZER_REPO
        rec["tokenizer_revision"] = TOKENIZER_REVISION
    if extra:
        rec.update(extra)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(rec, f, indent=2, default=str)


def load_kronos(device: str | None = None, max_context: int = 512,
                model_repo: str | None = None, model_path: str | None = None,
                tokenizer_path: str | None = None):
    """Load revision-pinned Kronos-base + tokenizer onto the best available device."""
    import torch
    sys.path.insert(0, ROOT)
    from model import Kronos, KronosTokenizer, KronosPredictor

    if device is None:
        if torch.cuda.is_available():
            device = "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if tokenizer_path:
        tok = KronosTokenizer.from_pretrained(tokenizer_path)
    else:
        tok = KronosTokenizer.from_pretrained(TOKENIZER_REPO, revision=TOKENIZER_REVISION)
    if model_path:
        mdl = Kronos.from_pretrained(model_path)
    else:
        repo = model_repo or MODEL_REPO
        mdl = Kronos.from_pretrained(repo, revision=MODEL_REVISIONS.get(repo))
    predictor = KronosPredictor(mdl, tok, device=device, max_context=max_context)
    predictor.tokenizer.eval()
    predictor.model.eval()
    return predictor


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "mps"):
        try:
            torch.mps.manual_seed(seed)
        except Exception:  # noqa: BLE001
            pass
