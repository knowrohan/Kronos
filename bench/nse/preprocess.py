"""Panel -> the repo's exact pickle contract, so finetune/dataset.py:QlibDataset loads it unchanged.

Contract details that silently break things if missed (plan 8.2):
  * columns are ``vol`` and ``amt``, not ``volume``/``amount``  (finetune/config.py:feature_list)
  * ``amt = (open + high + low + close) / 4 * vol``             (finetune/qlib_data_preprocess.py)
  * value is ``{symbol: DataFrame}`` indexed by a ``datetime`` index

Dividend convention is a required, explicit choice (plan 4.6). ``--dividends price`` (default)
uses the split-adjusted-but-not-dividend-adjusted close so the strategy leg matches the
^CNX200 / ^NSEI PRICE indices. ``--dividends total`` rescales OHLC by adj_close/close.

Also emits ``calendar.json`` (the NSE trading calendar implied by the benchmark series) and
``splits.json`` (exact date boundaries + per-split symbol/row counts).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from bench.nse.common import DATA_NSE, FEATURE_LIST, PROCESSED, sidecar  # noqa: E402

# Plan section 6, CORRECTED. The plan set val_end = 2023-06-30 and called the ~1-month
# gap to test_start "deliberate ... it exceeds lookback + predict". It does not:
# lookback+predict = 100 TRADING SESSIONS, and one calendar month is ~21. Verified on the
# real NSE calendar, that gap is 21 sessions -- 5x too small, so a finetuned model would
# see test-window context bars it had trained on.
# Pulling val_end back to 2023-02-15 gives a 111-session gap while keeping the full
# 756-session test window; val still spans 280 sessions.
SPLITS = {
    "train": ("2015-01-01", "2021-12-31"),
    "val": ("2022-01-01", "2023-02-15"),
    "test": ("2023-08-01", "2026-12-31"),
}


def apply_dividend_convention(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Return a frame whose open/high/low/close follow one consistent convention."""
    out = df.copy()
    if mode == "total":
        # yfinance's adj_close carries both split and dividend adjustment; close carries
        # only splits. Their ratio is the dividend-reinvestment factor for that bar.
        ratio = (out["adj_close"] / out["close"]).astype("float64")
        for c in ("open", "high", "low", "close"):
            out[c] = out[c] * ratio
    elif mode != "price":
        raise ValueError(f"unknown dividend mode {mode!r}")
    return out


def to_kronos_frame(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV -> the repo's feature_list, indexed by datetime."""
    out = pd.DataFrame(index=pd.DatetimeIndex(df["date"], name="datetime"))
    out["open"] = df["open"].to_numpy()
    out["high"] = df["high"].to_numpy()
    out["low"] = df["low"].to_numpy()
    out["close"] = df["close"].to_numpy()
    out["vol"] = df["volume"].to_numpy()
    out["amt"] = (out["open"] + out["high"] + out["low"] + out["close"]) / 4 * out["vol"]
    return out[FEATURE_LIST]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=os.path.join(DATA_NSE, "panel.parquet"))
    ap.add_argument("--bench", default=os.path.join(DATA_NSE, "bench.parquet"))
    ap.add_argument("--out-dir", default=PROCESSED)
    ap.add_argument("--dividends", choices=["price", "total"], default="price",
                    help="price = split-adj only (matches the price-index benchmarks); "
                         "total = dividend-reinvested")
    ap.add_argument("--lookback", type=int, default=90)
    ap.add_argument("--predict", type=int, default=10)
    ap.add_argument("--min-universe", type=int, default=100,
                    help="a date joins the trading calendar only if this many symbols traded")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    panel = pd.read_parquet(args.panel)
    bench = pd.read_parquet(args.bench)
    min_rows = args.lookback + args.predict + 1

    # ---- trading calendar ------------------------------------------------
    # Benchmark index dates are the authoritative NSE session list; intersect with dates
    # where the cross-section is actually populated to drop half-days / bad index rows.
    counts = panel.groupby("date")["symbol"].size()
    populated = set(counts[counts >= args.min_universe].index)
    bench_dates = set(pd.to_datetime(bench["date"]))
    calendar = sorted(bench_dates & populated)
    print(f"calendar: {len(calendar)} sessions {calendar[0].date()} -> {calendar[-1].date()} "
          f"(bench {len(bench_dates)}, populated {len(populated)})")

    # ---- per-symbol frames ----------------------------------------------
    panel = apply_dividend_convention(panel, args.dividends)
    cal_set = set(calendar)
    frames: dict[str, pd.DataFrame] = {}
    dropped: dict[str, str] = {}
    for sym, g in panel.groupby("symbol", sort=True):
        g = g[g["date"].isin(cal_set)].sort_values("date")
        kf = to_kronos_frame(g).dropna()
        kf = kf[(kf[["open", "high", "low", "close"]] > 0).all(axis=1)]
        if len(kf) < min_rows:
            dropped[sym] = f"only {len(kf)} rows (< {min_rows})"
            continue
        frames[sym] = kf
    print(f"symbols: {len(frames)} kept, {len(dropped)} dropped")
    for s, why in dropped.items():
        print(f"  DROP {s}: {why}")

    # ---- splits ----------------------------------------------------------
    split_meta = {}
    for name, (lo, hi) in SPLITS.items():
        lo_ts, hi_ts = pd.Timestamp(lo), pd.Timestamp(hi)
        part = {}
        for sym, kf in frames.items():
            sl = kf[(kf.index >= lo_ts) & (kf.index <= hi_ts)]
            if len(sl) >= min_rows:
                part[sym] = sl
        path = os.path.join(args.out_dir, f"{name}_data.pkl")
        with open(path, "wb") as f:
            pickle.dump(part, f)
        rows = int(sum(len(v) for v in part.values()))
        dates = sorted({d for v in part.values() for d in v.index})
        split_meta[name] = {
            "range": [lo, hi], "symbols": len(part), "rows": rows,
            "first_date": str(dates[0].date()) if dates else None,
            "last_date": str(dates[-1].date()) if dates else None,
            "n_sessions": len(dates), "path": path,
        }
        print(f"{name:>5s}: {len(part):3d} symbols  {rows:7d} rows  "
              f"{split_meta[name]['first_date']} -> {split_meta[name]['last_date']}")

    # Lookahead assertion (plan 10.5): the model-selection data must be far enough behind
    # the test window that no test context bar was ever seen in train or val.
    gap_sessions = sum(1 for d in calendar
                       if pd.Timestamp(SPLITS["val"][1]) < d < pd.Timestamp(SPLITS["test"][0]))
    assert gap_sessions >= 0
    split_meta["gap_sessions_val_to_test"] = gap_sessions
    split_meta["lookback_plus_predict"] = args.lookback + args.predict
    split_meta["dividend_convention"] = args.dividends
    split_meta["dropped_symbols"] = dropped
    split_meta["calendar_sessions"] = len(calendar)
    with open(os.path.join(args.out_dir, "splits.json"), "w") as f:
        json.dump(split_meta, f, indent=2)
    with open(os.path.join(DATA_NSE, "calendar.json"), "w") as f:
        json.dump([str(d.date()) for d in calendar], f)

    # A tidy, dividend-convention-applied panel is what signals/backtest actually read.
    tidy = panel[panel["date"].isin(cal_set) & panel["symbol"].isin(frames)].copy()
    tidy = tidy[["date", "symbol", "open", "high", "low", "close", "volume"]]
    tidy.to_parquet(os.path.join(DATA_NSE, f"panel_{args.dividends}.parquet"), index=False)
    print(f"tidy panel ({args.dividends}) -> panel_{args.dividends}.parquet  {len(tidy)} rows")

    sidecar(os.path.join(args.out_dir, "preprocess_meta.json"), vars(args), extra=split_meta)


if __name__ == "__main__":
    main()
