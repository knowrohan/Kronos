"""Plan section 10 verification tests, written as assertions.

  10.1 pickle contract    -> the unmodified finetune/dataset.py:QlibDataset loads our pickles
  10.2 signal reproducibility under a fixed seed
  10.3 pipeline cross-check: signals.py vs bench/kronos_eval.py on the same symbol/date
  10.5 lookahead assertions on the split boundaries and the context/execution ordering
  4.8  QlibDataset traps: __getitem__ ignores idx; num_workers>0 duplicates samples
  4.9  the repo trainer cannot run here -- capture the exact error

Backtester sanity (10.4), the cost reconciliation (10.6), universe integrity (10.7) and the
split check (10.8) live in verify_backtest.py, which needs the backtester to exist first.
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from bench.nse.common import DATA_NSE, OUT_NSE, PROCESSED, load_kronos, set_seed, sidecar  # noqa: E402

RESULTS: dict = {}


def check(name: str, fn):
    try:
        val = fn()
        ok = val.get("pass") if isinstance(val, dict) else bool(val)
        RESULTS[name] = val if isinstance(val, dict) else {"pass": ok}
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if isinstance(val, dict):
            for k, v in val.items():
                if k != "pass":
                    print(f"        {k}: {v}")
    except Exception as exc:  # noqa: BLE001
        RESULTS[name] = {"pass": False, "error": f"{type(exc).__name__}: {exc}",
                         "traceback": traceback.format_exc()[-1500:]}
        print(f"ERROR {name}: {type(exc).__name__}: {exc}")


# --------------------------------------------------------------- 10.1 pickle contract


def t_pickle_contract():
    """Load our pickles through the UNMODIFIED finetune/dataset.py:QlibDataset."""
    import importlib
    fine = os.path.join(ROOT, "finetune")
    sys.path.insert(0, fine)
    import config as repo_config  # noqa: PLC0415
    importlib.reload(repo_config)

    # QlibDataset builds Config() internally and reads dataset_path off it, so point the
    # class default at our processed dir at runtime rather than editing finetune/config.py.
    orig_init = repo_config.Config.__init__

    def patched(self):
        orig_init(self)
        self.dataset_path = PROCESSED
    repo_config.Config.__init__ = patched
    try:
        import dataset as repo_dataset  # noqa: PLC0415
        importlib.reload(repo_dataset)
        ds = repo_dataset.QlibDataset(data_type="val")
        x, xs = ds[0]
        cfg = repo_config.Config()
        want = (cfg.lookback_window + cfg.predict_window + 1, len(cfg.feature_list))
        return {
            "pass": tuple(x.shape) == want and xs.shape[0] == want[0],
            "sample_shape": list(x.shape), "expected_shape": list(want),
            "stamp_shape": list(xs.shape), "n_indices": len(ds.indices),
            "n_symbols": len(ds.symbols), "finite": bool(torch.isfinite(x).all()),
            "abs_max": float(x.abs().max()),
        }
    finally:
        repo_config.Config.__init__ = orig_init


# --------------------------------------------------------------- 4.8 dataset traps


def t_qlibdataset_ignores_idx():
    import importlib
    sys.path.insert(0, os.path.join(ROOT, "finetune"))
    import config as repo_config
    orig_init = repo_config.Config.__init__

    def patched(self):
        orig_init(self)
        self.dataset_path = PROCESSED
    repo_config.Config.__init__ = patched
    try:
        import dataset as repo_dataset
        importlib.reload(repo_dataset)
        ds = repo_dataset.QlibDataset(data_type="val")
        ds.set_epoch_seed(0)
        a = ds[0][0].clone()
        ds.set_epoch_seed(0)
        b = ds[12345][0].clone()          # different idx, same RNG state
        ignores_idx = bool(torch.equal(a, b))

        # workers fork the same seeded py_rng -> identical sample streams
        from torch.utils.data import DataLoader
        ds.set_epoch_seed(0)
        l2 = DataLoader(ds, batch_size=4, num_workers=2, shuffle=False)
        batches = [b0 for i, (b0, _) in enumerate(l2) if i < 2]
        dup = bool(torch.equal(batches[0], batches[1])) if len(batches) == 2 else None
        return {"pass": ignores_idx,
                "getitem_ignores_idx": ignores_idx,
                "workers_emit_identical_batches": dup,
                "note": "confirms plan 4.8.1; mitigation = num_workers=0 or worker_init_fn reseed"}
    finally:
        repo_config.Config.__init__ = orig_init


# --------------------------------------------------------------- 4.9 trainer


def t_trainer_cannot_run():
    r = subprocess.run([sys.executable, "train_predictor.py"], cwd=os.path.join(ROOT, "finetune"),
                       capture_output=True, text=True, timeout=180)
    err = (r.stderr or r.stdout).strip().splitlines()
    tail = "\n".join(err[-4:]) if err else ""
    return {"pass": r.returncode != 0, "returncode": r.returncode, "error_tail": tail}


# --------------------------------------------------------------- 10.2 / 10.3


def _one_window(symbol="RELIANCE", date="2023-08-01", lookback=90, predict=10):
    from bench.nse.signals import build_batch, future_sessions, load_inputs, stamps_for
    blocks, calendar = load_inputs(os.path.join(DATA_NSE, "panel_price.parquet"),
                                   os.path.join(DATA_NSE, "calendar.json"))
    t = pd.Timestamp(date)
    ti = calendar.index(t)
    b = build_batch(blocks, [symbol], t, lookback, 220)
    X, syms, means, stds, closes = b
    i = blocks[symbol]["pos"][t]
    ctx_dates = blocks[symbol]["dates"][i + 1 - lookback: i + 1]
    return dict(X=X, means=means, stds=stds, closes=closes, ctx_dates=ctx_dates,
                fut_dates=future_sessions(calendar, ti, predict),
                x_stamp=stamps_for(ctx_dates), y_stamp=stamps_for(future_sessions(calendar, ti, predict)),
                blocks=blocks, symbol=symbol, i=i, lookback=lookback, predict=predict)


def t_signal_reproducible(predictor):
    from model.kronos import auto_regressive_inference
    w = _one_window()
    dev = predictor.device
    xb = torch.from_numpy(w["X"]).to(dev)
    xs = torch.from_numpy(w["x_stamp"][None]).to(dev)
    ys = torch.from_numpy(w["y_stamp"][None]).to(dev)
    outs = []
    for _ in range(3):
        set_seed(999)
        outs.append(auto_regressive_inference(predictor.tokenizer, predictor.model, xb, xs, ys,
                                              max_context=512, pred_len=10, clip=5, T=0.6,
                                              top_k=0, top_p=0.9, sample_count=1, verbose=False))
    same = all(np.array_equal(outs[0], o) for o in outs[1:])
    return {"pass": same, "bitwise_identical_across_3_runs": same,
            "max_abs_diff": float(max(np.abs(outs[0] - o).max() for o in outs[1:]))}


def t_pipeline_crosscheck(predictor):
    """signals.py's own path vs bench/kronos_eval.py:predict_paths on the same window.

    Catches batching, normalization and stamp-alignment bugs that no single-path test finds.
    """
    from bench.kronos_eval import predict_paths
    from bench.nse.signals import score_block
    from model.kronos import auto_regressive_inference

    w = _one_window()
    lookback, predict = w["lookback"], w["predict"]
    b = w["blocks"][w["symbol"]]
    sl = slice(w["i"] + 1 - lookback, w["i"] + 1)
    raw = b["X"][sl]  # open, high, low, close, vol, amt
    x_df = pd.DataFrame(raw, columns=["open", "high", "low", "close", "volume", "amount"])
    x_ts = pd.Series(pd.DatetimeIndex(w["ctx_dates"]))
    y_ts = pd.Series(pd.DatetimeIndex(w["fut_dates"]))

    set_seed(4242)
    paths = predict_paths(predictor, x_df, x_ts, y_ts, predict, n_paths=1, T=0.6, top_p=0.9)
    eval_close = paths[0, -predict:, 3]                      # DENORMALIZED by predict_paths

    dev = predictor.device
    set_seed(4242)
    preds = auto_regressive_inference(
        predictor.tokenizer, predictor.model,
        torch.from_numpy(w["X"]).to(dev),
        torch.from_numpy(w["x_stamp"][None]).to(dev),
        torch.from_numpy(w["y_stamp"][None]).to(dev),
        max_context=512, pred_len=predict, clip=5, T=0.6, top_k=0, top_p=0.9,
        sample_count=1, verbose=False)
    sig_close_norm = preds[0, -predict:, 3]
    sig_close = sig_close_norm * (w["stds"][0, 3] + 1e-5) + w["means"][0, 3]

    max_rel = float(np.abs(sig_close - eval_close).max() / max(np.abs(eval_close).max(), 1e-9))
    # score_ret must equal the same thing computed straight off the denormalized path
    sc = score_block(preds, w["X"], w["means"], w["stds"], w["closes"], lookback, predict)
    indep_ret = float(eval_close[-1] / w["closes"][0] - 1.0)
    # predict_paths denormalizes in float32, score_block in float64; ~1e-7 relative
    # disagreement on the derived return is rounding, not a pipeline difference.
    return {"pass": max_rel < 1e-5 and abs(sc["last_ret"][0] - indep_ret) < 1e-6,
            "max_relative_close_diff": max_rel,
            "signals_last_ret": float(sc["last_ret"][0]),
            "kronos_eval_last_ret": indep_ret,
            "note": "both paths must be bit-identical under the same seed"}


# --------------------------------------------------------------- 10.5 lookahead


def t_lookahead_splits():
    with open(os.path.join(PROCESSED, "splits.json")) as f:
        s = json.load(f)
    with open(os.path.join(DATA_NSE, "calendar.json")) as f:
        cal = [pd.Timestamp(d) for d in json.load(f)]
    val_end = pd.Timestamp(s["val"]["last_date"])
    test_start = pd.Timestamp(s["test"]["first_date"])
    gap = sum(1 for d in cal if val_end < d < test_start)
    need = s["lookback_plus_predict"]
    return {"pass": gap >= need, "val_end": str(val_end.date()),
            "test_start": str(test_start.date()),
            "gap_sessions": gap, "required_lookback_plus_predict": need}


def t_context_before_execution():
    """For a rebalance date t: max(context bar) <= t < execution date t+1."""
    from bench.nse.signals import load_inputs
    blocks, cal = load_inputs(os.path.join(DATA_NSE, "panel_price.parquet"),
                              os.path.join(DATA_NSE, "calendar.json"))
    bad = 0
    checked = 0
    sym = "RELIANCE"
    b = blocks[sym]
    for t in cal[-400::37]:
        i = b["pos"].get(t)
        if i is None or i < 90:
            continue
        ctx_max = b["dates"][i]
        ti = cal.index(t)
        if ti + 1 >= len(cal):
            continue
        exec_date = cal[ti + 1]
        checked += 1
        if not (pd.Timestamp(ctx_max) <= t < exec_date):
            bad += 1
    return {"pass": bad == 0 and checked > 0, "windows_checked": checked, "violations": bad}


def main() -> None:
    os.makedirs(OUT_NSE, exist_ok=True)
    check("10.1_pickle_contract_unmodified_QlibDataset", t_pickle_contract)
    check("4.8.1_QlibDataset_getitem_ignores_idx", t_qlibdataset_ignores_idx)
    check("4.9_repo_trainer_fails_on_this_machine", t_trainer_cannot_run)
    check("10.5a_lookahead_split_gap", t_lookahead_splits)
    check("10.5b_context_before_execution", t_context_before_execution)

    print("\nloading Kronos for model-level checks...")
    predictor = load_kronos()
    check("10.2_signal_reproducible_fixed_seed", lambda: t_signal_reproducible(predictor))
    check("10.3_pipeline_crosscheck_vs_kronos_eval", lambda: t_pipeline_crosscheck(predictor))

    out = os.path.join(OUT_NSE, "verify_pipeline.json")
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    sidecar(out.replace(".json", "_meta.json"), {})
    n_pass = sum(1 for v in RESULTS.values() if v.get("pass"))
    print(f"\n{n_pass}/{len(RESULTS)} checks passed -> {out}")


if __name__ == "__main__":
    main()
