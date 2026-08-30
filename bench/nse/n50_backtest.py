"""The cash-book money test on the point-in-time NIFTY 50 (NIFTY50_PLAN section 6.1).

A thin, opinionated wrapper over ``backtest.run_backtest`` / ``backtest.metrics``. Everything
it adds is a constraint the plan states and the generic backtester does not know about:

  * **Universe.** Scores are set to NaN for every name that is not an *eligible PIT NIFTY 50
    member on that signal date*, so the strategy cannot hold a name the index did not contain.
    Eligibility (the 90-bar lookback rule) and membership are both taken from
    ``dataset_n50.parquet``'s ``is_n50`` column, i.e. from the same file the probe grades on.
  * **Grid.** topk in {10, 15} x n_drop in {1, 2}, hold_thresh 5, weekly rebalance, drift
    weighting, ``zerodha`` statutory costs at Rs 10 lakh. k = 10, n_drop = 1 is the primary.
  * **Benchmarks.** Three price legs plus two book legs (NIFTY50_PLAN sec.11):
    ``niftybees_adj`` **close-to-close** -- the purchasable total-return tracker. Its OPEN feed is
    unusable: measured on the 799 sessions where all four price columns are present, the ETF's
    open-to-open return tracks ^NSEI open-to-open with a daily sd of 112 bps and misses by more
    than 2 pp on 68 sessions (worst +7.85 pp on 2025-04-04, -6.30 pp on 2025-04-07; the two
    ~3 pp days NIFTY50_PLAN sec.11 cites in 2024-06 are +2.27 and -3.42 and are not the worst
    of them), and the open series repeats the previous print exactly 25 times. Close-to-close on
    the same sessions tracks at 21 bps sd and misses by more than 2 pp twice, and repeats 7
    times. So open-to-open on that feed measures the ETF's quote staleness, not the index; half a
    session of clock offset against the book is the smaller error by a factor of five. This leg
    is the DESCRIPTION.
    ``nsei_tr_proxy`` = ^NSEI **open-to-open** plus a flat per-session accrual that compounds to
    +1.17 pp/yr -- the measured NIFTYBEES-minus-^NSEI gap (+0.52 / +1.28 / +1.14 / +1.23 pp over
    calendar 2023-26), i.e. the index dividend yield net of the fund's TER. Same open-to-open
    clock as the book, total-return level. This leg is the M2 GATE.
    ``nsei`` open-to-open price index, context only -- it is short the dividend and any book
    beats it by ~1.2 pp/yr for free.
    Plus an equal-weight PIT NIFTY 50 book and a constant-score book, both run through the same
    accounting and the same cost model on the same sessions, which is what separates stock
    selection from the equal-weight-vs-cap-weight tilt and from the engine's own hold rules.
  * **Null.** ``--null-seeds N`` runs N random-score books at identical settings and reports
    where the real book's annualised net excess falls in that distribution. A backtest without
    its own null is a story about one draw.

The one_way cost is recomputed from ``costs.CostModel`` and ASSERTED against what the engine
charged before anything is written (REBUILD_PLAN Appendix B, F11): a cost figure that is
merely printed is a cost figure nobody checked.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from bench.nse.backtest import (TRADING_DAYS, Market, annualise,  # noqa: E402
                                assert_execution_lag, metrics, run_backtest)
from bench.nse.common import sidecar  # noqa: E402
from bench.nse.costs import PRESETS, CostModel  # noqa: E402
from bench.nse.mlcore import DATA_NSE, OUT_NSE  # noqa: E402

N50 = os.path.join(OUT_NSE, "n50")
DATASET_N50 = os.path.join(N50, "dataset_n50.parquet")
BENCH_N50 = os.path.join(DATA_NSE, "bench_n50.parquet")

#: Annualised NIFTYBEES-adj minus ^NSEI over calendar 2023-26 (+0.52/+1.28/+1.14/+1.23 pp),
#: ratified in NIFTY50_PLAN sec.11 as the total-return accrual the price index is missing.
#: Re-measured here off ``bench_n50.parquet`` and the plan's figures reproduce on the study
#: window (close-to-close, from 2023-08-01: +0.47/+1.30/+1.07/+1.23 pp). It is the one number in
#: the benchmark set by hand, and the convention moves it: the whole-sample annualised gap is
#: +1.44 pp close-to-close and +0.97 pp open-to-open (the latter measured on the stale ETF open
#: feed, hence not preferred). 1.17 sits inside that band. ``n50_money.py`` reports what the M2
#: verdict would be across the band rather than asking the reader to trust the midpoint.
TR_ACCRUAL_ANNUAL = 0.0117
#: Per-session accrual that COMPOUNDS to TR_ACCRUAL_ANNUAL over a 252-session year. Compounded,
#: not divided: 0.0117/252 applied per session compounds to 1.1768 pp/yr, i.e. it would over-credit
#: the benchmark by 0.0068 pp/yr (verified: (1 + 0.0117/252)**252 - 1 = 0.011768). That is 0.2% of
#: the M2 bar -- small, but this is the one place in the file where the benchmark is being handed a
#: number by hand.
TR_ACCRUAL_PER_SESSION = (1.0 + TR_ACCRUAL_ANNUAL) ** (1.0 / TRADING_DAYS) - 1.0

#: name -> (parquet column, price convention, per-session additive accrual).
#: "open" means r[s] = px(s+1)/px(s) - 1, the return of a position opened at open(s) -- the book's
#: own clock. "close" is the same shift on the close series: it measures close(s) -> close(s+1),
#: half a session offset from the book, which is the price of not using a stale open feed.
BENCH_SPECS: dict[str, dict] = {
    "niftybees_adj": {"column": "niftybees_adj", "price": "close", "accrual": 0.0,
                      "role": "description: the purchasable total-return tracker"},
    "nsei": {"column": "nsei_open", "price": "open", "accrual": 0.0,
             "role": "context: price index, short the dividend"},
    "nsei_tr_proxy": {"column": "nsei_open", "price": "open",
                      "accrual": TR_ACCRUAL_PER_SESSION,
                      "role": "M2 GATE: price index on the book's clock + 1.17 pp/yr accrual"},
}
#: Back-compat view used by the tests and by anything that only needs the column names.
BENCH_COLS = {k: v["column"] for k, v in BENCH_SPECS.items()}


# ------------------------------------------------------------------ universe / scores


def load_universe(path: str = DATASET_N50) -> pd.DataFrame:
    """The eligible PIT NIFTY 50 cells as (date, symbol) -- the only names that may be held."""
    ds = pd.read_parquet(path, columns=["date", "symbol", "is_n50"])
    ds["date"] = pd.to_datetime(ds["date"])
    return ds.loc[ds["is_n50"], ["date", "symbol"]].reset_index(drop=True)


def eligible_frame(universe: pd.DataFrame) -> pd.DataFrame:
    """Wide boolean (signal date x symbol): True where the name is tradable-by-the-rules."""
    u = universe.assign(ok=True)
    return (u.pivot(index="date", columns="symbol", values="ok")
              .sort_index().fillna(False).astype(bool))


def masked_scores(scores_long: pd.DataFrame, column: str, elig: pd.DataFrame) -> pd.DataFrame:
    """Wide score frame, NaN wherever the name is not an eligible PIT member that session.

    NaN is the engine's own "ineligible" sentinel (``scores_wide.loc[d].dropna()``), so the
    universe constraint is enforced by construction rather than by a filter someone must
    remember to apply.
    """
    s = scores_long[["date", "symbol", column]].copy()
    s["date"] = pd.to_datetime(s["date"])
    wide = s.pivot(index="date", columns="symbol", values=column).sort_index()
    wide = wide.reindex(index=elig.index, columns=elig.columns)
    return wide.where(elig)


def constant_frame(elig: pd.DataFrame, value: float = 0.0) -> pd.DataFrame:
    """Every eligible name scored identically -- the placebo the comb guard acts on.

    ``topk_dropout_targets`` ranks ties in the order pandas' sort leaves them, so this book is
    "buy k names, then hold" plus whatever the sell-high/buy-low guard permits. It is the
    zero-information book run through the exact engine the strategy uses, which is a different
    and stricter control than an index return: it pays the same costs and obeys the same
    hold_thresh.
    """
    return elig.astype(float).where(elig) * 0.0 + value


def random_frame(elig: pd.DataFrame, seed: int) -> pd.DataFrame:
    """i.i.d. U(0,1) scores on the eligible cells only -- one draw of the N3 backtest null."""
    rng = np.random.default_rng(seed)
    a = rng.random(elig.shape)
    return pd.DataFrame(a, index=elig.index, columns=elig.columns).where(elig)


# ------------------------------------------------------------------ benchmarks


def bench_series(market: Market, col: str = "niftybees_adj", path: str = BENCH_N50,
                 ffill_limit: int = 2) -> tuple[pd.Series, int]:
    """Benchmark daily return on the market calendar, plus the number of forward-filled prices.

    ``r[s] = px(s+1)/px(s) - 1 + accrual``, where ``px`` and the accrual come from
    ``BENCH_SPECS[col]``: the open series for the two ^NSEI legs (the book's own clock), the
    adjusted CLOSE series for NIFTYBEES (whose open feed is stale -- NIFTY50_PLAN sec.11).

    yfinance leaves a handful of holes in both feeds. A hole left as NaN becomes a 0% day inside
    ``metrics``, which silently invents a flat session for the benchmark; forward-filling the
    PRICE instead makes that session's return 0 and hands the true two-day move to the next
    session, which is what actually happened. The count is returned so the number can be
    reported rather than assumed harmless.

    The accrual is added to every session in the index INCLUDING ones the price series cannot
    price (a filled hole), because a dividend accrues whether or not the vendor printed a quote;
    it is not added past the end of the series, where the return is NaN and ``metrics`` will
    treat the session as flat for both legs.
    """
    spec = BENCH_SPECS[col]
    b = pd.read_parquet(path)
    b["date"] = pd.to_datetime(b["date"])
    px = b.set_index("date")[spec["column"]].reindex(market.calendar)
    filled = int(px.isna().sum())
    px = px.ffill(limit=ffill_limit)
    filled -= int(px.isna().sum())
    r = px.shift(-1) / px - 1.0
    return r + float(spec["accrual"]) * r.notna(), filled


def equal_weight_book(market: Market, universe: pd.DataFrame, one_way: float, *,
                      rebalance: int = 5, weighting: str = "rebalance",
                      start=None, end=None, dates=None) -> dict:
    """Weekly equal-weight book over ALL eligible PIT NIFTY 50 names, same costs, same clock.

    Written out rather than routed through ``run_backtest`` because ``topk_dropout_targets``
    cannot express a whole-universe book: with ``n_drop = 0`` it never sells, so a name deleted
    from the index would be held forever (six deletions over this window), and with
    ``n_drop > 0`` the bottom of ``comb`` is a held name whenever no new member is available,
    so the book churns or shrinks every week. Everything else is identical to the engine --
    execution at open(t+1) with ``assert_execution_lag``, turnover = sum |dw| charged at the
    same one-way rate, drift of the weights by realised returns between rebalances -- so the
    output dict is the shape ``backtest.metrics`` consumes.

    ``dates`` restricts the signal dates to an explicit list -- the sessions the strategy book
    actually traded. Without it the equal-weight leg would run on every membership date and be
    compared, cell by cell, against a book measured on a shorter window: two annualised returns
    over two different windows differenced as if they were an excess.
    """
    members = {d: sorted(g) for d, g in universe.groupby("date")["symbol"]}
    cal = market.calendar
    sig_dates = [d for d in sorted(members) if d in market.idx]
    if dates is not None:
        keep = set(pd.to_datetime(pd.Index(list(dates))))
        sig_dates = [d for d in sig_dates if d in keep]
    if start:
        sig_dates = [d for d in sig_dates if d >= pd.Timestamp(start)]
    if end:
        sig_dates = [d for d in sig_dates if d <= pd.Timestamp(end)]
    sig_dates = sig_dates[::rebalance]
    exec_idxs = []
    for d in sig_dates:
        si = market.idx[d]
        if si + 1 >= len(cal):
            continue
        assert_execution_lag(si, si + 1, cal)
        exec_idxs.append((si, si + 1))
    if not exec_idxs:
        raise ValueError("no executable signal dates")
    next_reb = {ei: si for si, ei in exec_idxs}
    first_exec, last_exec = exec_idxs[0][1], exec_idxs[-1][1]

    weights = pd.Series(dtype=float)
    rows, rlog = [], []
    for s in range(first_exec, min(last_exec + 1, len(cal) - 1)):
        day = cal[s]
        turnover = 0.0
        if s in next_reb:
            tradable = market.tradable.iloc[s]
            target = [x for x in members[cal[next_reb[s]]]
                      if x in tradable.index and bool(tradable[x])]
            if target:
                if weighting == "rebalance":
                    new_w = pd.Series(1.0 / len(target), index=target)
                else:
                    keep = weights.reindex([t for t in target if t in weights.index]).fillna(0.0)
                    add_names = [t for t in target if t not in weights.index]
                    freed = 1.0 - keep.sum()
                    new_w = (pd.concat([keep, pd.Series(freed / len(add_names), index=add_names)])
                             if add_names else keep)
                tot = float(new_w.sum())
                new_w = new_w / tot if tot > 0 else pd.Series(1.0 / len(target), index=target)
            else:
                new_w = pd.Series(dtype=float)
            allsym = weights.index.union(new_w.index)
            delta = new_w.reindex(allsym).fillna(0.0) - weights.reindex(allsym).fillna(0.0)
            turnover = float(delta.abs().sum())
            weights = new_w
            rlog.append({"signal_date": cal[next_reb[s]], "exec_date": day,
                         "n_held": len(new_w), "n_buy": 0, "n_sell": 0, "turnover": turnover})

        r = market.oo_ret.iloc[s].reindex(weights.index).fillna(0.0)
        gross = float((weights * r).sum())
        cost = turnover * one_way
        rows.append({"date": day, "gross_return": gross, "cost": cost,
                     "net_return": gross - cost, "turnover": turnover,
                     "n_held": len(weights), "n_buy": 0, "n_sell": 0})
        if len(weights):
            grown = weights * (1.0 + r)
            tot = grown.sum()
            weights = grown / tot if tot > 0 else grown

    return {"daily": pd.DataFrame(rows).set_index("date"),
            "rebalances": pd.DataFrame(rlog), "one_way_cost": one_way}


# ------------------------------------------------------------------ driver


def run_grid(scores_wide: pd.DataFrame, market: Market, cm: CostModel, args) -> dict:
    """One book per (topk, n_drop) cell, with the cost the engine charged asserted."""
    books = {}
    for k in args.topk:
        for nd in args.n_drop:
            res = run_backtest(scores_wide, market, topk=k, n_drop=nd,
                               hold_thresh=args.hold_thresh, rebalance=args.rebalance,
                               cost_model=cm, capital=args.capital, weighting=args.weighting,
                               start=args.start, end=args.end, log_trades=True)
            want = cm.one_way_bps(args.capital / k) / 1e4
            got = res["one_way_cost"]
            assert abs(got - want) < 1e-12, (
                f"engine charged {got * 1e4:.4f} bps one-way at k={k}, "
                f"cost model says {want * 1e4:.4f} bps")
            books[f"k{k}_d{nd}"] = res
    return books


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default=DATASET_N50,
                    help="long parquet with date, symbol and the score column")
    ap.add_argument("--column", default="reversal_1d")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--dataset", default=DATASET_N50, help="source of the eligible PIT N50 cells")
    ap.add_argument("--panel", default=os.path.join(DATA_NSE, "panel_price_pit.parquet"))
    ap.add_argument("--calendar", default=os.path.join(DATA_NSE, "calendar.json"))
    ap.add_argument("--bench", default=BENCH_N50)
    ap.add_argument("--out-dir", default=N50)
    ap.add_argument("--topk", default="10,15", type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--n-drop", default="1,2", type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--hold-thresh", type=int, default=5)
    ap.add_argument("--rebalance", type=int, default=5, help="1 = daily, 5 = weekly")
    ap.add_argument("--weighting", choices=["drift", "rebalance"], default="drift")
    ap.add_argument("--cost-preset", default="zerodha", choices=list(PRESETS))
    ap.add_argument("--capital", type=float, default=1e6)
    ap.add_argument("--null-seeds", type=int, default=0,
                    help="N random-score books at the primary settings (N3 null)")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag or args.column
    cm = PRESETS[args.cost_preset]

    panel = pd.read_parquet(args.panel)
    with open(args.calendar) as f:
        cal = [pd.Timestamp(d) for d in json.load(f)]
    market = Market(panel, cal)

    universe = load_universe(args.dataset)
    elig = eligible_frame(universe)
    scores_long = pd.read_parquet(args.scores)
    wide = masked_scores(scores_long, args.column, elig)

    # Every h_T arm's score file starts later than the universe (the estimator needs a burn-in),
    # and ``masked_scores`` reindexes onto the full membership calendar, so the front of the
    # frame is all-NaN. Left alone, ``run_backtest`` starts its clock on the first membership
    # date and books flat sessions -- holding nothing, earning nothing -- while the benchmark
    # compounds through them. That is not a strategy result, it is a hole, and it would show up
    # as several points of annualised underperformance that no trade caused. Trim to the
    # sessions that actually carry a score and run EVERY leg (book, equal-weight, constant,
    # null) on exactly those sessions, so every annualised number in the file is over one window.
    scored = wide.notna().any(axis=1)
    if not bool(scored.any()):
        raise ValueError(f"no eligible PIT N50 cell carries a {args.column!r} score")
    wide = wide.loc[scored]
    if args.start:
        wide = wide.loc[wide.index >= pd.Timestamp(args.start)]
    if args.end:
        wide = wide.loc[wide.index <= pd.Timestamp(args.end)]
    if not len(wide):
        raise ValueError("no scored signal dates in [--start, --end]")
    args.start, args.end = wide.index[0], wide.index[-1]
    elig_sig = elig.loc[wide.index]                      # the shared clock for every leg
    n_per_date = elig_sig.sum(axis=1)
    print(f"universe: {len(elig)} membership dates, {len(wide)} scored signal dates "
          f"{args.start.date()} -> {args.end.date()}, eligible names per date "
          f"min {int(n_per_date.min())} median {n_per_date.median():.0f} "
          f"max {int(n_per_date.max())}", flush=True)
    print(f"scores: {args.column} from {os.path.relpath(args.scores, ROOT)}  "
          f"non-NaN cells {int(wide.notna().sum().sum())}", flush=True)

    # ---- benchmarks
    bench_daily, bench_fill = {}, {}
    for c in BENCH_SPECS:
        bench_daily[c], bench_fill[c] = bench_series(market, c, args.bench)
    ew_legs = args.capital / float(n_per_date.median())
    ew_one_way = cm.one_way_bps(ew_legs) / 1e4
    for w, name in (("rebalance", "ew_n50_rebal"), ("drift", "ew_n50_drift")):
        ew = equal_weight_book(market, universe, ew_one_way, rebalance=args.rebalance,
                               weighting=w, dates=wide.index)
        bench_daily[name] = ew["daily"]["net_return"]
        bench_fill[name] = 0
        ew["daily"].to_parquet(os.path.join(args.out_dir, f"bt_{tag}_{name}.parquet"))

    k0, d0 = args.topk[0], args.n_drop[0]
    const = run_backtest(constant_frame(elig_sig), market, topk=k0, n_drop=d0,
                         hold_thresh=args.hold_thresh, rebalance=args.rebalance,
                         cost_model=cm, capital=args.capital, weighting=args.weighting)
    bench_daily["const_score"] = const["daily"]["net_return"]
    bench_fill["const_score"] = 0
    const["daily"].to_parquet(
        os.path.join(args.out_dir, f"bt_{tag}_const_score_k{k0}_d{d0}.parquet"))

    # ---- the grid
    books = run_grid(wide, market, cm, args)
    prim = f"k{k0}_d{d0}"
    book_index = books[prim]["daily"].index
    pd.DataFrame({b: s.reindex(book_index).fillna(0.0) for b, s in bench_daily.items()},
                 index=book_index).to_parquet(
        os.path.join(args.out_dir, f"bt_{tag}_bench_daily.parquet"))

    out_metrics = {}
    for name, res in books.items():
        res["daily"].to_parquet(os.path.join(args.out_dir, f"bt_{tag}_{name}.parquet"))
        if "trades" in res and len(res["trades"]):
            res["trades"].to_parquet(
                os.path.join(args.out_dir, f"bt_{tag}_{name}_trades.parquet"), index=False)
        out_metrics[name] = {b: metrics(res, bench_daily[b], seed=args.seed)
                             for b in bench_daily}
        m = out_metrics[name]["niftybees_adj"]
        g = out_metrics[name]["nsei_tr_proxy"]
        print(f"  {name}: ann net {m['ann_return_net'] * 100:+.2f}%  "
              f"excess vs NIFTYBEES(cc) {m['ann_excess_net'] * 100:+.2f} pp  "
              f"vs nsei_tr_proxy {g['ann_excess_net'] * 100:+.2f} pp  "
              f"[CI {g['excess_ci95_annualised'][0] * 100:+.1f}, "
              f"{g['excess_ci95_annualised'][1] * 100:+.1f}]  "
              f"IR {g['information_ratio']:+.2f}  turnover/reb {m['turnover_per_rebalance']:.3f}  "
              f"MDD {m['max_drawdown'] * 100:.1f}%  "
              f"cost drag {m['cost_drag_ann_pct']:.2f} pp  one-way {m['one_way_cost_bps']:.2f} bps",
              flush=True)

    # The book legs are also scored against BOTH tracker legs, so "the strategy beat the
    # equal-weight book" and "the equal-weight book beat the tracker" are separable claims.
    bench_only = {}
    for b, s in bench_daily.items():
        if b in ("ew_n50_rebal", "ew_n50_drift", "const_score"):
            fake = {"daily": pd.DataFrame({"gross_return": s, "cost": 0.0, "net_return": s,
                                           "turnover": 0.0}),
                    "rebalances": pd.DataFrame(), "one_way_cost": ew_one_way}
            bench_only[b] = {ref: metrics(fake, bench_daily[ref], seed=args.seed)
                             for ref in ("niftybees_adj", "nsei_tr_proxy")}
    for b, mm in bench_only.items():
        print(f"  benchmark {b}: ann {mm['niftybees_adj']['ann_return_net'] * 100:+.2f}%  "
              f"vs NIFTYBEES(cc) {mm['niftybees_adj']['ann_excess_net'] * 100:+.2f} pp  "
              f"vs nsei_tr_proxy {mm['nsei_tr_proxy']['ann_excess_net'] * 100:+.2f} pp", flush=True)

    # ---- N3 random null at the primary cell
    null = {}
    if args.null_seeds > 0:
        # Annualised excess only: a full ``metrics`` call per draw would spend 10,000 bootstrap
        # resamples per benchmark per seed on a number nobody reads. annualise() is the same
        # arithmetic ``metrics`` uses for ann_excess_net, so the null and the observed statistic
        # are the same functional -- which is the whole point of F1.
        ann_b = {b: annualise(s.reindex(book_index).fillna(0.0)) for b, s in bench_daily.items()}
        obs = {b: out_metrics[prim][b]["ann_excess_net"] for b in bench_daily}
        draws: dict[str, list] = {b: [] for b in bench_daily}
        ann_nets = []
        for s in range(args.null_seeds):
            r = run_backtest(random_frame(elig_sig, 1000 + s), market, topk=k0, n_drop=d0,
                             hold_thresh=args.hold_thresh, rebalance=args.rebalance,
                             cost_model=cm, capital=args.capital, weighting=args.weighting)
            assert r["daily"].index.equals(book_index), "null draw is on a different clock"
            a_net = annualise(r["daily"]["net_return"])
            ann_nets.append(a_net)
            for b in bench_daily:
                draws[b].append(a_net - ann_b[b])
        null = {"cell": prim, "n_seeds": int(args.null_seeds),
                "null_ann_net": {"mean": float(np.mean(ann_nets)),
                                 "sd": float(np.std(ann_nets, ddof=1))},
                "observed_ann_net": float(out_metrics[prim]["niftybees_adj"]["ann_return_net"]),
                "by_benchmark": {}}
        for b, v in draws.items():
            a = np.asarray(v, dtype=float)
            sd = float(a.std(ddof=1))
            null["by_benchmark"][b] = {
                "observed_ann_excess_net": float(obs[b]),
                "null_mean": float(a.mean()), "null_sd": sd,
                "null_min": float(a.min()), "null_max": float(a.max()),
                "null_p05": float(np.percentile(a, 5)), "null_p50": float(np.percentile(a, 50)),
                "null_p95": float(np.percentile(a, 95)),
                "percentile_of_observed": float((a < obs[b]).mean() * 100.0),
                "p_value_one_sided": float((a >= obs[b]).mean()),
                "z_vs_null": float((obs[b] - a.mean()) / sd) if sd > 0 else float("nan"),
                "draws": [float(x) for x in a],
            }
        for b in ("niftybees_adj", "nsei_tr_proxy"):
            n = null["by_benchmark"][b]
            print(f"  N3 null ({args.null_seeds} seeds) vs {b}: observed "
                  f"{n['observed_ann_excess_net'] * 100:+.2f} pp  null mean "
                  f"{n['null_mean'] * 100:+.2f} pp  sd {n['null_sd'] * 100:.2f} pp  "
                  f"pctile {n['percentile_of_observed']:.1f}  p {n['p_value_one_sided']:.3f}  "
                  f"z {n['z_vs_null']:+.2f}", flush=True)

    payload = {
        "tag": tag, "column": args.column, "scores_path": os.path.abspath(args.scores),
        "universe": {"n_membership_dates": int(len(elig)),
                     "n_signal_dates": int(len(wide)),
                     "first_signal_date": str(args.start.date()),
                     "last_signal_date": str(args.end.date()),
                     "first_book_session": str(book_index[0].date()),
                     "last_book_session": str(book_index[-1].date()),
                     "n_book_sessions": int(len(book_index)),
                     "names_per_date": {"min": int(n_per_date.min()),
                                        "median": float(n_per_date.median()),
                                        "max": int(n_per_date.max())}},
        "bench_specs": {k: {kk: vv for kk, vv in v.items()} for k, v in BENCH_SPECS.items()},
        "tr_accrual_annual": TR_ACCRUAL_ANNUAL,
        "tr_accrual_per_session": TR_ACCRUAL_PER_SESSION,
        "books": out_metrics, "benchmark_books": bench_only,
        "benchmark_prices_forward_filled": bench_fill,
        "null_random_scores": null,
    }
    mpath = os.path.join(args.out_dir, f"bt_{tag}_metrics.json")
    with open(mpath, "w") as f:
        json.dump(payload, f, indent=2, default=float)

    # The cost figure is the one number in a money test that must never be taken on trust.
    cost_assert = {
        "cost_preset": args.cost_preset, "capital": args.capital,
        "one_way_bps_by_topk": {str(k): cm.one_way_bps(args.capital / k) for k in args.topk},
        "round_trip_bps_by_topk": {str(k): cm.round_trip_bps(args.capital / k) for k in args.topk},
        "engine_one_way_bps": {n: books[n]["one_way_cost"] * 1e4 for n in books},
        "equal_weight_one_way_bps": ew_one_way * 1e4,
        "plan_section_6_1_claim_bps": 11.2,
        "plan_claim_note": ("NIFTY50_PLAN section 6.1 says legs of Rs 1 lakh at k=10 cost ~11.2 bps "
                            "one-way. 11.2 bps is the Rs 10 LAKH-per-leg figure from costs.py's "
                            "docstring; at Rs 1 lakh legs the flat DP charge and the capped "
                            "brokerage have not yet amortised and the real one-way cost is "
                            f"{cm.one_way_bps(args.capital / 10):.2f} bps. The engine is charged the "
                            "measured number, not the plan's."),
    }
    sidecar(os.path.join(args.out_dir, f"bt_{tag}_meta.json"),
            {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in vars(args).items()},
            extra={"cost_assertion": cost_assert,
                   "n_books": len(books), "elapsed_s": round(time.time() - t0, 1)})
    print(f"\nwrote {mpath}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
