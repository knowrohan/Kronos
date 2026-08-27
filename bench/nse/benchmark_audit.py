"""Is the equal-weight benchmark a fair bar, and is any of it actually reachable?

Every headline in this study is quoted as excess over `ew200`, the equal-weighted eligible
universe. That series is computed as the cross-sectional MEAN of daily open-to-open returns,
which quietly assumes the book is rebalanced back to equal weight every single session at zero
cost. Two things follow and both need a number rather than a hand-wave:

  1. Daily rebalancing is itself a short-horizon reversal bet -- it sells whatever went up
     yesterday. In a market where 1-session reversal has a t of +7, that is not a neutral
     accounting choice; part of `ew200`'s return IS the very effect the strategies trade.
  2. Nobody can run it. Rebalancing 200 names daily costs roughly 27 bps round trip on real
     turnover, and the benchmark charges none.

So price the benchmark: rebalance the same universe daily / weekly / monthly / quarterly / never,
with and without costs, on both the biased and the point-in-time universe. Whatever excess a
strategy shows over the free daily version is the CONSERVATIVE number; this quantifies by how
much.
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
from bench.nse.backtest import annualise, bench_series, load_market, max_drawdown  # noqa: E402
from bench.nse.common import DATA_NSE, OUT_NSE, sidecar  # noqa: E402
from bench.nse.controls import eligible_mask_daily  # noqa: E402
from bench.nse.costs import PRESETS  # noqa: E402
from bench.nse.robustness import build_universe  # noqa: E402

TRADING_DAYS = 252
CADENCES = {"daily": 1, "weekly": 5, "monthly": 21, "quarterly": 63, "never": None}


def equal_weight_path(market, elig_daily, cadence: int | None, one_way: float) -> dict:
    """Equal-weight the eligible universe, rebalancing every `cadence` sessions.

    cadence=None means buy once and never rebalance -- weights drift with price, which turns the
    portfolio into a slowly cap-weighting one. Entries and exits from the universe are always
    honoured (a delisted or removed name has to be sold whatever the cadence).
    """
    cal = market.calendar
    idx = [i for i, d in enumerate(cal) if d in elig_daily.index]
    rows = []
    w = pd.Series(dtype=float)
    for k, s in enumerate(idx[:-1]):
        day = cal[s]
        # `s` positions into the full calendar (for market.oo_ret); elig_daily is date-indexed
        names = elig_daily.loc[day]
        names = names[names].index
        due = (cadence is not None and k % cadence == 0) or len(w) == 0
        # names that left the universe must be sold regardless of cadence
        stale = [n for n in w.index if n not in set(names)]
        turnover = 0.0
        if due:
            new_w = pd.Series(1.0 / max(len(names), 1), index=names)
        elif stale:
            new_w = w.drop(stale)
            new_w = new_w / new_w.sum() if new_w.sum() > 0 else new_w
        else:
            new_w = w
        if due or stale:
            allsym = w.index.union(new_w.index)
            turnover = float((new_w.reindex(allsym).fillna(0.0)
                              - w.reindex(allsym).fillna(0.0)).abs().sum())
        w = new_w
        r = market.oo_ret.iloc[s].reindex(w.index).fillna(0.0)
        gross = float((w * r).sum())
        cost = turnover * one_way
        rows.append({"date": day, "gross_return": gross, "cost": cost,
                     "net_return": gross - cost, "turnover": turnover, "n_held": len(w)})
        grown = w * (1.0 + r)
        tot = grown.sum()
        w = grown / tot if tot > 0 else grown
    df = pd.DataFrame(rows).set_index("date")
    n = len(df)
    return {"daily": df,
            "ann_return_gross": annualise(df["gross_return"]),
            "ann_return_net": annualise(df["net_return"]),
            "turnover_ann": float(df["turnover"].sum()) * TRADING_DAYS / max(n, 1),
            "cost_drag_ann_pct": float(df["cost"].sum()) * TRADING_DAYS / max(n, 1) * 100,
            "vol_ann": float(df["net_return"].std() * np.sqrt(TRADING_DAYS)),
            "max_drawdown": max_drawdown(df["net_return"]),
            "n_days": n}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calendar", default=os.path.join(DATA_NSE, "calendar.json"))
    ap.add_argument("--start", default="2023-08-01")
    ap.add_argument("--end", default="2026-08-25")
    ap.add_argument("--tag", default="benchmark_audit")
    ap.add_argument("--lookback", type=int, default=90)
    ap.add_argument("--capital", type=float, default=1e7)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--cost-preset", default="zerodha")
    args = ap.parse_args()

    cm = PRESETS[args.cost_preset]
    # a 200-name equal-weight book puts capital/200 in each line, not capital/topk
    one_way = cm.one_way_bps(args.capital / 200) / 1e4
    out = {"config": vars(args), "one_way_bps": one_way * 1e4, "universes": {}}
    print(f"cost: {one_way * 1e4:.2f} bps one-way at Rs {args.capital / 200:,.0f} per line",
          flush=True)

    for uname, panel, mem in (
            ("biased", os.path.join(DATA_NSE, "panel_price.parquet"), None),
            ("point_in_time", os.path.join(DATA_NSE, "panel_price_pit.parquet"),
             os.path.join(DATA_NSE, "membership_pit.parquet"))):
        market = load_market(panel, args.calendar)
        dates = [d for d in market.calendar
                 if pd.Timestamp(args.start) <= d <= pd.Timestamp(args.end)]
        elig = build_universe(market, dates, mem, args.lookback)
        elig_daily = eligible_mask_daily(market, elig).loc[dates[0]:dates[-1]]
        cap = bench_series(market, "cnx200").reindex(elig_daily.index).fillna(0.0)
        print(f"\n=== {uname}: {elig.sum(axis=1).mean():.1f} names/session ===", flush=True)
        u = {"cap_weighted_cnx200_ann": annualise(cap),
             "universe_mean_names": float(elig.sum(axis=1).mean()), "cadences": {}}
        for cname, cad in CADENCES.items():
            free = equal_weight_path(market, elig_daily, cad, 0.0)
            paid = equal_weight_path(market, elig_daily, cad, one_way)
            u["cadences"][cname] = {
                "ann_return_free": free["ann_return_gross"],
                "ann_return_net": paid["ann_return_net"],
                "turnover_ann": paid["turnover_ann"],
                "cost_drag_ann_pct": paid["cost_drag_ann_pct"],
                "vol_ann": paid["vol_ann"], "max_drawdown": paid["max_drawdown"],
                "excess_vs_cap_free": free["ann_return_gross"] - annualise(cap),
                "excess_vs_cap_net": paid["ann_return_net"] - annualise(cap)}
            c = u["cadences"][cname]
            print(f"  EW {cname:9s} free {c['ann_return_free']:+.4f} | "
                  f"net {c['ann_return_net']:+.4f} | turnover {c['turnover_ann']:6.2f}x | "
                  f"drag {c['cost_drag_ann_pct']:.2f}%", flush=True)
        d0 = u["cadences"]["daily"]
        m0 = u["cadences"]["monthly"]
        n0 = u["cadences"]["never"]
        u["rebalancing_premium_daily_vs_never"] = (d0["ann_return_free"]
                                                   - n0["ann_return_free"])
        u["benchmark_overstatement"] = d0["ann_return_free"] - m0["ann_return_net"]
        print(f"  cap-weighted CNX200 {u['cap_weighted_cnx200_ann']:+.4f}")
        print(f"  free-daily-EW minus never-rebalance-EW  = "
              f"{u['rebalancing_premium_daily_vs_never']:+.4f}  (the rebalancing premium)")
        print(f"  free-daily-EW minus net-monthly-EW      = "
              f"{u['benchmark_overstatement']:+.4f}  (what the benchmark gives itself)")
        out["universes"][uname] = u

    path = f"{OUT_NSE}/{args.tag}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    sidecar(path.replace(".json", "_meta.json"), vars(args))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
