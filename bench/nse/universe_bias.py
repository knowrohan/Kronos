"""Decompose the equal-weight-universe return into a SIZE TILT and a SURVIVORSHIP component.

This matters more than any Kronos number in the study. Over the test window, equal-weighting
the *current* NIFTY 200 membership beats the cap-weighted NIFTY 200 by a very large margin with
no signal whatsoever. Any strategy that equal-weights a top-k slice of that list inherits the
whole gap, so "excess return vs ^CNX200" is not a measure of skill.

The gap has two distinguishable causes, and published NSE size indices let us separate them:

  1. SIZE TILT. Equal-weighting a 200-name list moves weight from the mega-caps into the
     smaller 150. That tilt is real and investable -- it is what a midcap index earns. It is
     estimated here from the *actual* published NIFTY Next 50 / Midcap 150 returns, which are
     survivorship-free because they are real indices with real historical membership.

  2. SURVIVORSHIP. The constituent list is today's membership projected backwards, so it
     contains names that were promoted INTO the NIFTY 200 during the window precisely because
     they went up, and excludes names that were demoted out of it. This part is not investable
     and is pure look-ahead.

Estimator for the counterfactual, survivorship-free equal-weight NIFTY 200:

    r_fair  =  (50/200) * r_NIFTY50  +  (150/200) * r_ranks_51_200

with r_ranks_51_200 proxied by the mean of NIFTY Next 50 (ranks 51-100) and
NIFTY Midcap 150 (ranks 101-250). Then

    size_tilt    = r_fair       - r_cap_weighted_200
    survivorship = r_ew_actual  - r_fair

Both legs use the same price-return convention and the same session window.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from bench.nse.backtest import load_market  # noqa: E402
from bench.nse.common import DATA_NSE, OUT_NSE, sidecar  # noqa: E402

# ticker -> (label, approximate market-cap rank band)
SIZE_INDICES = {
    "^NSEI": ("NIFTY 50", "ranks 1-50"),
    "^CNX200": ("NIFTY 200", "ranks 1-200"),
    "^CRSLDX": ("NIFTY 500", "ranks 1-500"),
    "^NSMIDCP": ("NIFTY Next 50", "ranks 51-100"),
    "NIFTYMIDCAP150.NS": ("NIFTY Midcap 150", "ranks 101-250"),
    "NIFTY_MIDCAP_100.NS": ("NIFTY Midcap 100", "ranks 101-200"),
    "^NSEMDCP50": ("NIFTY Midcap 50", "ranks 101-150"),
}


def ann_return(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 2:
        return float("nan")
    return float((s.iloc[-1] / s.iloc[0]) ** (252 / len(s)) - 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2023-08-01")
    ap.add_argument("--end", default="2026-08-25")
    ap.add_argument("--out", default=os.path.join(OUT_NSE, "universe_bias.json"))
    args = ap.parse_args()

    import yfinance as yf
    lo, hi = pd.Timestamp(args.start), pd.Timestamp(args.end)

    idx = {}
    for tk, (label, band) in SIZE_INDICES.items():
        d = yf.download(tk, start=lo - pd.Timedelta(days=10), end=hi + pd.Timedelta(days=2),
                        interval="1d", auto_adjust=False, progress=False, threads=False)
        if d is None or len(d) < 100:
            print(f"MISS {label} ({tk}) -- {0 if d is None else len(d)} rows")
            continue
        c = d["Close"]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        c = c.dropna()
        c = c[(c.index >= lo) & (c.index <= hi)]
        idx[label] = {"ticker": tk, "band": band, "sessions": int(len(c)),
                      "ann_return": ann_return(c)}
        print(f"{label:20s} {band:14s} {len(c):4d} sessions  ann {idx[label]['ann_return']:+.4f}")

    # equal-weighted return of OUR universe (current NIFTY 200 membership, back-projected)
    market = load_market()
    ew_daily = market.oo_ret.where(market.tradable).mean(axis=1)
    ew_daily = ew_daily[(ew_daily.index >= lo) & (ew_daily.index <= hi)].dropna()
    ew_ann = float((1 + ew_daily).prod() ** (252 / len(ew_daily)) - 1)
    print(f"{'EW universe (ours)':20s} {'current 200':14s} {len(ew_daily):4d} sessions  ann {ew_ann:+.4f}")

    r50 = idx["NIFTY 50"]["ann_return"]
    r200 = idx["NIFTY 200"]["ann_return"]
    mid = [idx[k]["ann_return"] for k in ("NIFTY Next 50", "NIFTY Midcap 150") if k in idx]
    r_51_200 = float(np.mean(mid))
    r_fair = 50 / 200 * r50 + 150 / 200 * r_51_200
    size_tilt = r_fair - r200
    surv = ew_ann - r_fair

    out = {
        "window": [args.start, args.end], "sessions": int(len(ew_daily)),
        "indices": idx,
        "equal_weight_current_membership_ann": ew_ann,
        "proxy_ranks_51_200_ann": r_51_200,
        "proxy_components": {k: idx[k]["ann_return"] for k in
                             ("NIFTY Next 50", "NIFTY Midcap 150") if k in idx},
        "counterfactual_fair_equal_weight_200_ann": r_fair,
        "decomposition": {
            "cap_weighted_nifty200": r200,
            "plus_size_tilt_from_equal_weighting": size_tilt,
            "plus_survivorship_from_current_membership": surv,
            "equals_equal_weight_current_membership": ew_ann,
            "total_gap_vs_cap_weighted_index": ew_ann - r200,
        },
        "note": ("survivorship is the residual after charging the equal-weight tilt at the "
                 "return real, survivorship-free midcap indices actually delivered; it is an "
                 "estimate, not an identity, and it is sensitive to the rank-band proxy"),
    }
    print("\n--- decomposition (annualised) ---")
    print(f"  cap-weighted NIFTY 200                       {r200:+.4f}")
    print(f"  + size tilt from equal-weighting             {size_tilt:+.4f}")
    print(f"  + survivorship (current membership)          {surv:+.4f}")
    print(f"  = equal-weight of our universe               {ew_ann:+.4f}")
    print(f"  total gap vs the index                       {ew_ann - r200:+.4f}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    sidecar(args.out.replace(".json", "_meta.json"), vars(args))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
