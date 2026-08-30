"""The single-stock futures venue on the PIT NIFTY 50 (NIFTY50_PLAN sec.6.2).

The user asked for this venue by name; the plan pre-registers its verdict as **KILL unless
the top-1 hit rate's lower Wilson bound exceeds 55%**, and sec.11 requires that verdict to be
reported across the whole 45-65 bps/month roll band rather than at the 55 bps midpoint, because
the roll number is the weakest input in the arithmetic (research confidence "low-medium").

What this module does, and the three things about it that decide the answer:

  * **One lot is the whole book.** A single-stock futures lot is built to a Rs 5-10 lakh
    contract value (SEBI's Oct-2015 stock-derivative framework, unchanged), so one lot on a
    Rs 10 lakh book is 50-100% of equity in ONE name with a 5-session holding period. The
    weekly sd of that position, expressed as a fraction of the book, is reported first and is
    the number the reader should look at before any hit rate.
  * **Carry is charged whether or not the trade works.** ``FuturesCostModel.holding_cost_bps``
    charges ``roll_cost_bps_per_month * days / 21``. Over 5 sessions at 55 bps/month that is
    13.1 bps of contract value -- roughly twice the entire statutory round trip (5.95 bps on a
    Rs 15 lakh lot). Transaction cost is not what kills or saves this venue; carry is.
  * **The carry sign on a short is genuinely ambiguous.** A futures short in contango *earns*
    the basis decay it is charged as a long. The assignment charges it on both legs; that is
    conservative for the short and the opposite of what the market does. Both readings are
    computed and both are written out (``carry_charged`` and ``carry_credited_short``); the
    verdict is stated on the charged reading and the credited reading is shown beside it.

Everything is per-week and in rupees, never in bps of a notional nobody holds: the venue's
whole problem is that the notional is not a free parameter.

Inputs: ``bench/out/nse/n50/scores_A0_primary.parquet`` restricted to ``is_n50``,
``data/nse/panel_price_pit.parquet`` opens, ``costs.FuturesCostModel`` /
``costs.PRESETS['zerodha']`` / ``costs.LOT_SIZES``. Outputs ``futures_n50.json`` and
``futures_n50.md`` under ``bench/out/nse/n50/`` with a provenance sidecar.

Nothing here writes a score, fits anything, or touches ``bench/out/nse/ml/``.
"""
from __future__ import annotations

import argparse
import ast
import calendar as _calmod
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bench.nse.backtest import Market, assert_execution_lag  # noqa: E402
from bench.nse.common import sidecar  # noqa: E402
from bench.nse.costs import (FUTURES_PRESETS, LOT_SIZES,  # noqa: E402
                             MARGIN_PCT_DEFAULT, MARGIN_PCT_RANGE, PRESETS,
                             CostModel, FuturesCostModel)
from bench.nse.mlcore import DATA_NSE, OUT_NSE  # noqa: E402

N50 = os.path.join(OUT_NSE, "n50")
SCORES = os.path.join(N50, "scores_A0_primary.parquet")
PANEL = os.path.join(DATA_NSE, "panel_price_pit.parquet")
CALENDAR = os.path.join(DATA_NSE, "calendar.json")
FACTS = os.path.join(DATA_NSE, "research", "futures_facts.json")

#: Contract value used for a name with no researched lot size. SEBI's stock-derivative
#: framework fixes the lot so contract value on the review date sits in Rs 5-10 lakh; Rs 7.5
#: lakh is the midpoint. Every leg priced this way is flagged (`lot_proxy`) and counted, because
#: a synthetic lot is a modelling choice and not an exchange fact.
PROXY_CONTRACT_RS = 750_000.0

BOOK = 1_000_000.0          # the user's Rs 10 lakh
CASH_NOTIONAL = 100_000.0   # the Rs 1 lakh cash comparison leg
HOLD_SESSIONS = 5           # open(t+1) -> open(t+6)
REBALANCE = 5               # weekly signal clock, same as n50_backtest
ROLL_BAND = (45.0, 55.0, 65.0)   # bps/month; sec.11 requires the verdict across all three
HIT_BAR = 0.55              # sec.6.2 kill rule: top-1 hit-rate lower CI bound must exceed this

ARMS = ("top1", "top3", "bottom1", "pair")


# ------------------------------------------------------------------ small statistics


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """(p_hat, lo, hi) Wilson score interval -- the interval the plan's kill rule reads.

    Wilson rather than Wald because at n ~ 113 weeks and p near 0.5 the Wald interval is both
    too narrow and capable of leaving [0, 1]; the kill rule is a statement about the LOWER
    bound, so the choice of interval is not cosmetic.
    """
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z / d * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (p, max(0.0, centre - half), min(1.0, centre + half))


# ------------------------------------------------------------------ expiry calendar


def last_weekday_of_month(year: int, month: int, weekday: int = 3) -> pd.Timestamp:
    """Last `weekday` (Mon=0 ... Thu=3) of a calendar month, as a naive Timestamp."""
    last_dom = _calmod.monthrange(year, month)[1]
    d = pd.Timestamp(year=year, month=month, day=last_dom)
    return d - pd.Timedelta(days=(d.weekday() - weekday) % 7)


def expiry_dates(cal: list[pd.Timestamp], weekday: int = 3) -> list[pd.Timestamp]:
    """Monthly F&O expiries on the exchange calendar: last `weekday`, backed up over holidays.

    If the nominal expiry day is not a session (an exchange holiday), the expiry moves to the
    PREVIOUS session, which is the NSE rule. Backing up is done through the session list rather
    than by subtracting days so a multi-day holiday block resolves correctly.
    """
    sess = set(cal)
    ordered = sorted(sess)
    out = []
    if not ordered:
        return out
    months = pd.period_range(ordered[0], ordered[-1], freq="M")
    for per in months:
        nominal = last_weekday_of_month(per.year, per.month, weekday)
        d = nominal
        # step back at most 10 calendar days; a longer gap means no session that month
        for _ in range(11):
            if d in sess:
                out.append(d)
                break
            d -= pd.Timedelta(days=1)
    return out


def crosses_expiry(entry: pd.Timestamp, exit_: pd.Timestamp, expiries) -> bool:
    """True if the hold [entry, exit) contains a monthly expiry.

    The position is entered at the OPEN of `entry` and closed at the OPEN of `exit_`. An expiry
    landing strictly inside that span forces a roll (or physical settlement of the whole lot);
    an expiry landing ON the exit session does not, because the position is already flat by the
    open. ``count_crossings`` also reports the inclusive reading so the ambiguity is visible.
    """
    return any(entry <= e < exit_ for e in expiries)


def count_crossings(cycles: list[dict], expiries) -> dict:
    strict = sum(1 for c in cycles if crosses_expiry(c["entry_date"], c["exit_date"], expiries))
    incl = sum(1 for c in cycles
               if any(c["entry_date"] <= e <= c["exit_date"] for e in expiries))
    n = len(cycles)
    return {"n_weeks": n, "n_cross_strict": strict, "n_cross_inclusive": incl,
            "frac_cross_strict": strict / n if n else float("nan"),
            "frac_cross_inclusive": incl / n if n else float("nan")}


# ------------------------------------------------------------------ inputs


def load_scores(path: str = SCORES) -> pd.DataFrame:
    """The out-of-sample scores, restricted to the tradable PIT NIFTY 50 cells."""
    s = pd.read_parquet(path)
    for col in ("date", "symbol", "score", "is_n50"):
        if col not in s.columns:
            raise ValueError(f"{path} violates the scores contract: no column {col!r}")
    s["date"] = pd.to_datetime(s["date"])
    s = s.loc[s["is_n50"].astype(bool) & s["score"].notna()].copy()
    return s.sort_values(["date", "symbol"]).reset_index(drop=True)


def load_market(panel_path: str = PANEL, cal_path: str = CALENDAR) -> Market:
    panel = pd.read_parquet(panel_path)
    with open(cal_path) as f:
        cal = [pd.Timestamp(d) for d in json.load(f)]
    return Market(panel, cal)


def research_lots(path: str = FACTS) -> dict:
    """Lot sizes as recorded by the research agent, for cross-checking ``costs.LOT_SIZES``.

    The JSON stores them as a repr string inside the facts list, so they are parsed with
    ``ast.literal_eval`` rather than trusted as JSON. costs.py deliberately hard-codes the same
    table; this function exists so a divergence between the two is an assertion, not a surprise.
    """
    with open(path) as f:
        facts = json.load(f)["facts"]
    for fact in facts:
        if "lots" in fact:
            raw = fact["lots"]
            d = raw if isinstance(raw, dict) else ast.literal_eval(raw)
            return {str(k): int(v) for k, v in d.items()}
    return {}


def weekly_cycles(score_dates, market: Market, rebalance: int = REBALANCE,
                  hold: int = HOLD_SESSIONS) -> list[dict]:
    """Every 5th scored session -> (signal, entry = t+1, exit = t+1+hold), same clock as
    ``n50_backtest`` / ``backtest.run_backtest`` (``sig_dates[::rebalance]``, execution at the
    next session). Cycles whose exit runs off the end of the calendar are dropped and counted
    by the caller via ``n_weeks``.
    """
    cal = market.calendar
    sig = [pd.Timestamp(d) for d in sorted(pd.unique(pd.to_datetime(pd.Series(score_dates))))]
    sig = [d for d in sig if d in market.idx][::rebalance]
    out = []
    for d in sig:
        si = market.idx[d]
        if si + 1 + hold >= len(cal):
            continue
        assert_execution_lag(si, si + 1, cal)
        out.append({"signal_date": d, "signal_idx": si,
                    "entry_idx": si + 1, "exit_idx": si + 1 + hold,
                    "entry_date": cal[si + 1], "exit_date": cal[si + 1 + hold]})
    return out


# ------------------------------------------------------------------ leg construction


def lot_for(symbol: str, price: float, lots: dict | None = None,
            proxy_contract: float = PROXY_CONTRACT_RS) -> tuple[int, bool]:
    """(lot size, is_proxy). Researched lot where one exists, else a Rs 7.5 lakh synthetic lot.

    Only 10 of the 50 index names have a researched lot, so most legs are synthetic; the flag
    is carried all the way to the JSON because a synthetic lot makes every contract exactly
    Rs 7.5 lakh, which understates the dispersion of real contract values (RELIANCE ~6.6 lakh
    vs ICICIBANK ~9.9 lakh) and therefore understates the dispersion of margin.
    """
    lots = LOT_SIZES if lots is None else lots
    if symbol in lots:
        return int(lots[symbol]), False
    if not np.isfinite(price) or price <= 0:
        return 0, True
    return max(1, int(round(proxy_contract / price))), True


def build_legs(scores: pd.DataFrame, market: Market, cycles: list[dict],
               fut: FuturesCostModel, cash: CostModel, *,
               lots: dict | None = None, margin_pct: float = MARGIN_PCT_DEFAULT,
               top_n: int = 3, expiries=None) -> pd.DataFrame:
    """One row per (week, arm-leg): prices, lot, gross/net P&L, margin, cash comparison.

    Costs are decomposed rather than netted so the write-up can say which line item decides
    the venue: ``txn_rs`` (statutory + brokerage + slippage round trip on the entry notional)
    and ``carry_rs_per_bpm`` (carry per 1 bps/month of roll, so the three roll settings are a
    multiplication, not three re-runs of the whole path).
    """
    by_date = {d: g for d, g in scores.groupby("date")}
    expiries = [] if expiries is None else expiries
    rows = []
    for c in cycles:
        g = by_date.get(c["signal_date"])
        if g is None or g.empty:
            continue
        g = g.sort_values("score", ascending=False)
        picks = [(sym, +1, f"top{i + 1}") for i, sym in enumerate(g["symbol"].tolist()[:top_n])]
        picks.append((g["symbol"].tolist()[-1], -1, "bottom1"))
        o_in = market.opens.iloc[c["entry_idx"]]
        o_out = market.opens.iloc[c["exit_idx"]]
        for sym, side, slot in picks:
            p_in = float(o_in.get(sym, np.nan))
            p_out = float(o_out.get(sym, np.nan))
            lot, proxy = lot_for(sym, p_in, lots)
            ok = bool(np.isfinite(p_in) and np.isfinite(p_out) and p_in > 0 and lot > 0)
            notional = lot * p_in if ok else float("nan")
            ret = (p_out / p_in - 1.0) if ok else float("nan")
            gross = lot * (p_out - p_in) * side if ok else float("nan")
            txn = (fut.leg(notional, "buy") + fut.leg(notional, "sell")) if ok else float("nan")
            # carry per 1 bps/month, so P&L at any roll setting is gross - txn - roll*this
            carry_unit = (notional * HOLD_SESSIONS / 21.0 / 1e4) if ok else float("nan")
            cash_gross = CASH_NOTIONAL * ret * side if ok else float("nan")
            cash_txn = (cash.leg(CASH_NOTIONAL, "buy")
                        + cash.leg(CASH_NOTIONAL * (1.0 + ret), "sell")) if ok else float("nan")
            rows.append({
                "signal_date": c["signal_date"], "entry_date": c["entry_date"],
                "exit_date": c["exit_date"], "symbol": sym, "slot": slot, "side": side,
                "score_rank": slot, "lot": lot, "lot_proxy": proxy, "ok": ok,
                "entry_px": p_in, "exit_px": p_out, "ret": ret,
                "notional_rs": notional, "margin_rs": margin_pct * notional if ok else np.nan,
                "gross_pnl_rs": gross, "txn_rs": txn, "carry_rs_per_bpm": carry_unit,
                "cash_gross_rs": cash_gross, "cash_txn_rs": cash_txn,
                "cash_net_rs": cash_gross - cash_txn if ok else float("nan"),
                "crosses_expiry": crosses_expiry(c["entry_date"], c["exit_date"], expiries),
            })
    return pd.DataFrame(rows)


def net_pnl(legs: pd.DataFrame, roll_bpm: float, credit_short_carry: bool = False) -> pd.Series:
    """Futures net P&L in rupees at a given roll cost.

    ``credit_short_carry=True`` is the second reading of the carry sign: a short position in a
    contango market collects the basis decay rather than paying it. Which reading is right is
    an empirical question about per-stock bases that the record cannot answer (research note:
    no per-stock roll dataset exists), so both are computed and both are reported.
    """
    carry = legs["carry_rs_per_bpm"] * roll_bpm
    if credit_short_carry:
        carry = carry * np.where(legs["side"].to_numpy() > 0, 1.0, -1.0)
    return legs["gross_pnl_rs"] - legs["txn_rs"] - carry


def arm_weekly(legs: pd.DataFrame, arm: str, roll_bpm: float,
               credit_short_carry: bool = False) -> pd.DataFrame:
    """Per-week (P&L, margin, notional, n_legs) for one arm, futures and cash side by side.

    Arms: ``top1`` (long the highest score), ``top3`` (long one lot in each of the top three --
    reported as the BASKET sum, with the per-leg hit rate carried separately), ``bottom1``
    (short the lowest score), ``pair`` (top1 long + bottom1 short, the market-neutral-ish
    two-lot book). A week is included only if every leg it needs has prices on both dates.
    """
    slots = {"top1": ["top1"], "top3": ["top1", "top2", "top3"],
             "bottom1": ["bottom1"], "pair": ["top1", "bottom1"]}[arm]
    sub = legs.loc[legs["slot"].isin(slots)].copy()
    sub["net_rs"] = net_pnl(sub, roll_bpm, credit_short_carry)
    need = len(slots)
    agg = sub.groupby("signal_date").agg(
        n_legs=("ok", "size"), n_ok=("ok", "sum"),
        pnl_rs=("net_rs", "sum"), gross_rs=("gross_pnl_rs", "sum"),
        txn_rs=("txn_rs", "sum"), margin_rs=("margin_rs", "sum"),
        notional_rs=("notional_rs", "sum"), cash_rs=("cash_net_rs", "sum"),
        crosses=("crosses_expiry", "max"))
    agg = agg.loc[(agg["n_ok"] == need) & (agg["n_legs"] == need)]
    return agg.reset_index()


# ------------------------------------------------------------------ aggregation


def describe_weekly(w: pd.DataFrame, col: str = "pnl_rs", book: float = BOOK) -> dict:
    x = w[col].to_numpy(float)
    n = len(x)
    k = int((x > 0).sum())
    p, lo, hi = wilson_ci(k, n)
    return {
        "n_weeks": n, "mean_rs": float(np.mean(x)), "median_rs": float(np.median(x)),
        "sd_rs": float(np.std(x, ddof=1)) if n > 1 else float("nan"),
        "min_rs": float(np.min(x)), "max_rs": float(np.max(x)),
        "sum_rs": float(np.sum(x)),
        "hits": k, "hit_rate": p, "hit_lo95": lo, "hit_hi95": hi,
        "sd_pct_of_book": float(np.std(x, ddof=1)) / book * 100.0 if n > 1 else float("nan"),
        "mean_pct_of_book": float(np.mean(x)) / book * 100.0,
        "worst_week_pct_of_book": float(np.min(x)) / book * 100.0,
        "t_stat": (float(np.mean(x)) / (float(np.std(x, ddof=1)) / math.sqrt(n))
                   if n > 1 and np.std(x, ddof=1) > 0 else float("nan")),
    }


def edge_stats(w: pd.DataFrame) -> dict:
    """The pre-registered comparison of sec.6.2: weekly edge in bps of contract value against
    the weekly sd of the same thing. The plan's expectation was "a few bps against ~4%"; this
    is the number that settles it, and it is independent of the hit rate and of the roll cost.
    """
    n = w["notional_rs"].to_numpy(float)
    g = w["gross_rs"].to_numpy(float)
    net = w["pnl_rs"].to_numpy(float)
    gb, nb = g / n * 1e4, net / n * 1e4
    return {"mean_gross_bps_of_notional": float(np.mean(gb)),
            "sd_gross_bps_of_notional": float(np.std(gb, ddof=1)),
            "mean_net_bps_of_notional": float(np.mean(nb)),
            "mean_notional_rs": float(np.mean(n)),
            "gross_edge_over_sd": float(np.mean(gb) / np.std(gb, ddof=1)),
            "ann_net_pct_of_book": float(np.mean(net)) * 52.0 / BOOK * 100.0,
            "ann_gross_pct_of_book": float(np.mean(g)) * 52.0 / BOOK * 100.0}


def expiry_split(w: pd.DataFrame) -> dict:
    """Weeks whose hold contains a monthly expiry vs weeks that do not, side by side.

    A crossing week's lot must be rolled (paying the carry a second time) or physically settled;
    this reports whether those weeks also behaved differently, so the settlement caveat can be
    stated with a number rather than as a worry.
    """
    x, y = w.loc[w["crosses"]], w.loc[~w["crosses"]]
    def m(a):
        return {"n": int(len(a)), "mean_rs": float(a["pnl_rs"].mean()) if len(a) else float("nan"),
                "hit_rate": float((a["pnl_rs"] > 0).mean()) if len(a) else float("nan")}
    return {"crossing": m(x), "clear": m(y)}


def worst_week(legs: pd.DataFrame, arm: str, roll_bpm: float) -> dict:
    """The single worst week's legs, named. One position on a Rs 10 lakh book means the tail is
    an event with a date and a stock, not a percentile."""
    slots = {"top1": ["top1"], "top3": ["top1", "top2", "top3"],
             "bottom1": ["bottom1"], "pair": ["top1", "bottom1"]}[arm]
    sub = legs.loc[legs["slot"].isin(slots) & legs["ok"]].copy()
    sub["net_rs"] = net_pnl(sub, roll_bpm)
    tot = sub.groupby("signal_date")["net_rs"].sum()
    if tot.empty:
        return {}
    d = tot.idxmin()
    wk = sub.loc[sub["signal_date"] == d]
    return {"signal_date": str(pd.Timestamp(d).date()),
            "entry_date": str(pd.Timestamp(wk["entry_date"].iloc[0]).date()),
            "exit_date": str(pd.Timestamp(wk["exit_date"].iloc[0]).date()),
            "pnl_rs": float(tot.loc[d]), "pct_of_book": float(tot.loc[d]) / BOOK * 100.0,
            "legs": [{"symbol": r.symbol, "side": int(r.side), "lot": int(r.lot),
                      "lot_proxy": bool(r.lot_proxy), "entry_px": float(r.entry_px),
                      "exit_px": float(r.exit_px), "ret": float(r.ret),
                      "net_rs": float(r.net_rs)} for r in wk.itertuples()]}


def margin_stats(w: pd.DataFrame, margin_pct: float, book: float = BOOK) -> dict:
    """Margin at one `margin_pct`, rescaled from the frame's own (already-applied) pct."""
    notion = w["notional_rs"].to_numpy(float)
    m = notion * margin_pct
    return {"margin_pct": margin_pct,
            "mean_margin_rs": float(np.mean(m)), "median_margin_rs": float(np.median(m)),
            "max_margin_rs": float(np.max(m)),
            "mean_margin_pct_of_book": float(np.mean(m)) / book * 100.0,
            "max_margin_pct_of_book": float(np.max(m)) / book * 100.0,
            "frac_weeks_margin_gt_book": float(np.mean(m > book)),
            "frac_weeks_margin_gt_half_book": float(np.mean(m > 0.5 * book)),
            "mean_notional_pct_of_book": float(np.mean(notion)) / book * 100.0}


def _max_dd(equity: np.ndarray, equity0: float) -> np.ndarray:
    """Peak-to-trough drawdown in rupees of each bootstrapped equity path (peak starts at equity0)."""
    full = np.concatenate([np.full((equity.shape[0], 1), equity0), equity], axis=1)
    return (np.maximum.accumulate(full, axis=1) - full).max(axis=1)


def block_bootstrap(pnl: np.ndarray, margin: np.ndarray, *, block: int = 4, draws: int = 2000,
                    weeks: int = 156, equity0: float = BOOK, seed: int = 0,
                    call_mult: float = 1.5, loss_frac: float = 0.30) -> dict:
    """Moving-block bootstrap of the weekly P&L path; every risk number the venue is judged on.

    Blocks of 4 consecutive weeks preserve whatever short-run autocorrelation the P&L has (a
    momentum-ish signal's wins cluster). The (P&L, margin) pair travels together so a resampled
    week's margin requirement is the one that actually accompanied that week's P&L.

    A margin call is recorded when equity at the START of a week is below ``call_mult`` x that
    week's margin requirement: at 1.5x the trader cannot post margin with any buffer. The path
    is NOT stopped at the first call -- stopping would flatter the loss statistics by censoring
    exactly the bad paths -- so ``p_margin_call`` and ``p_loss_gt_30pct`` are not disjoint.
    """
    n = len(pnl)
    if n < block:
        return {"error": f"only {n} weeks, block {block}"}
    rng = np.random.default_rng(seed)
    nblocks = int(np.ceil(weeks / block))
    starts = rng.integers(0, n - block + 1, size=(draws, nblocks))
    off = np.arange(block)
    idx = (starts[:, :, None] + off[None, None, :]).reshape(draws, -1)[:, :weeks]
    p = pnl[idx]                       # (draws, weeks)
    m = margin[idx]
    equity = equity0 + np.cumsum(p, axis=1)
    start_equity = np.concatenate([np.full((draws, 1), equity0), equity[:, :-1]], axis=1)
    called = (start_equity < call_mult * m).any(axis=1)
    min_eq = np.minimum(equity.min(axis=1), equity0)
    final = equity[:, -1]
    return {
        "draws": draws, "weeks": weeks, "block": block, "equity0": equity0,
        "p_margin_call": float(np.mean(called)),
        "p_loss_gt_30pct_anytime": float(np.mean(min_eq < (1.0 - loss_frac) * equity0)),
        "p_loss_gt_30pct_final": float(np.mean(final < (1.0 - loss_frac) * equity0)),
        "p_final_gt_start": float(np.mean(final > equity0)),
        "p_ruin_equity_le_0": float(np.mean(min_eq <= 0.0)),
        "median_final_rs": float(np.median(final)),
        "mean_final_rs": float(np.mean(final)),
        "p05_final_rs": float(np.percentile(final, 5)),
        "p95_final_rs": float(np.percentile(final, 95)),
        "median_max_drawdown_rs": float(np.median(_max_dd(equity, equity0))),
        "p95_max_drawdown_rs": float(np.percentile(_max_dd(equity, equity0), 95)),
    }


def rule_power(n: int, bar: float = HIT_BAR, truths=(0.50, 0.55, 0.60, 0.65)) -> dict:
    """What the pre-registered kill rule could have seen on `n` weeks.

    The rule reads the LOWER Wilson bound, so on a short sample it demands a point estimate far
    above the bar itself. Reporting its power next to its verdict is the difference between "the
    venue failed" and "the test could not have passed". Both readings are needed, and on this
    sample both happen to be true: the point estimate is below a coin flip AND the rule was
    close to unpassable. Power is the exact binomial P(hits >= min_hits_to_pass).
    """
    kmin = next((k for k in range(n + 1) if wilson_ci(k, n)[1] > bar), None)
    out = {"n_weeks": n, "bar": bar, "min_hits_to_pass": kmin,
           "min_hit_rate_to_pass": (kmin / n) if kmin is not None else float("nan")}
    if kmin is not None:
        out["power_vs_true_hit_rate"] = {
            f"{p:.2f}": float(sum(math.comb(n, k) * p ** k * (1.0 - p) ** (n - k)
                                  for k in range(kmin, n + 1))) for p in truths}
    return out


def verdict_for(hit_lo: float, bar: float = HIT_BAR) -> str:
    return "PROCEED" if hit_lo > bar else "KILL"


# ------------------------------------------------------------------ the run


def run(scores: pd.DataFrame, market: Market, *, fut_preset: str = "zerodha_fut",
        cash_preset: str = "zerodha", margin_pct: float = MARGIN_PCT_DEFAULT,
        rolls=ROLL_BAND, draws: int = 2000, block: int = 4, weeks: int = 156,
        seed: int = 0, lots: dict | None = None) -> tuple[dict, pd.DataFrame]:
    fut = FUTURES_PRESETS[fut_preset]
    cash = PRESETS[cash_preset]
    cal = market.calendar
    cycles = weekly_cycles(scores["date"], market)
    exp_thu = expiry_dates(cal, weekday=3)
    exp_tue = expiry_dates(cal, weekday=1)
    legs = build_legs(scores, market, cycles, fut, cash, lots=lots,
                      margin_pct=margin_pct, expiries=exp_thu)

    out: dict = {
        "config": {
            "scores": SCORES, "futures_preset": fut_preset, "cash_preset": cash_preset,
            "margin_pct": margin_pct, "margin_pct_range": list(MARGIN_PCT_RANGE),
            "roll_band_bps_per_month": list(rolls), "hold_sessions": HOLD_SESSIONS,
            "rebalance": REBALANCE, "book_rs": BOOK, "cash_notional_rs": CASH_NOTIONAL,
            "proxy_contract_rs": PROXY_CONTRACT_RS, "hit_bar": HIT_BAR,
            "bootstrap": {"block": block, "draws": draws, "weeks": weeks, "seed": seed},
            "arms": list(ARMS),
        },
        "sample": {
            "n_score_dates": int(scores["date"].nunique()),
            "first_signal": str(cycles[0]["signal_date"].date()) if cycles else None,
            "last_signal": str(cycles[-1]["signal_date"].date()) if cycles else None,
            "n_weekly_cycles": len(cycles),
            "n_legs": int(len(legs)), "n_legs_priced": int(legs["ok"].sum()),
            "n_legs_dropped_no_price": int((~legs["ok"]).sum()),
            "n_legs_proxy_lot": int(legs["lot_proxy"].sum()),
            "frac_legs_proxy_lot": float(legs["lot_proxy"].mean()),
            "distinct_symbols_traded": int(legs["symbol"].nunique()),
        },
        "cost_reference": {
            "round_trip_bps_at_7p5L": fut.round_trip_bps(PROXY_CONTRACT_RS),
            "round_trip_rs_at_7p5L": fut.leg(PROXY_CONTRACT_RS, "buy")
                                     + fut.leg(PROXY_CONTRACT_RS, "sell"),
            "carry_bps_5_sessions": {str(int(r)): r * HOLD_SESSIONS / 21.0 for r in rolls},
            "carry_rs_5_sessions_at_7p5L": {
                str(int(r)): PROXY_CONTRACT_RS * r * HOLD_SESSIONS / 21.0 / 1e4 for r in rolls},
            "cash_round_trip_bps_at_1L": cash.round_trip_bps(CASH_NOTIONAL),
        },
        "expiry": {
            "convention": "last Thursday of month, backed up to the previous session on a holiday",
            "n_expiries_in_window": len([e for e in exp_thu
                                         if cycles and cycles[0]["entry_date"] <= e
                                         <= cycles[-1]["exit_date"]]),
            "thursday": count_crossings(cycles, exp_thu),
            "tuesday_sensitivity": count_crossings(cycles, exp_tue),
        },
        "arms": {},
    }

    for arm in ARMS:
        rec: dict = {}
        for roll in rolls:
            w = arm_weekly(legs, arm, roll)
            key = f"roll{int(roll)}"
            rec[key] = {
                "futures": describe_weekly(w, "pnl_rs"),
                "futures_gross": describe_weekly(w, "gross_rs"),
                "cash_1L": describe_weekly(w, "cash_rs"),
                "margin": {f"pct{int(round(mp * 100))}": margin_stats(w, mp)
                           for mp in (MARGIN_PCT_RANGE[0], margin_pct, MARGIN_PCT_RANGE[1])},
                "bootstrap": block_bootstrap(
                    w["pnl_rs"].to_numpy(float),
                    (w["notional_rs"].to_numpy(float) * margin_pct),
                    block=block, draws=draws, weeks=weeks, seed=seed),
                "bootstrap_observed_length": block_bootstrap(
                    w["pnl_rs"].to_numpy(float),
                    (w["notional_rs"].to_numpy(float) * margin_pct),
                    block=block, draws=draws, weeks=len(w), seed=seed + 1),
                "bootstrap_cash_1L": block_bootstrap(
                    w["cash_rs"].to_numpy(float),
                    np.full(len(w), CASH_NOTIONAL),
                    block=block, draws=draws, weeks=weeks, seed=seed),
                "n_weeks_crossing_expiry": int(w["crosses"].sum()),
                "edge": edge_stats(w),
                "expiry_split": expiry_split(w),
                "worst_week": worst_week(legs, arm, roll),
            }
            if arm in ("bottom1", "pair"):
                wc = arm_weekly(legs, arm, roll, credit_short_carry=True)
                rec[key]["futures_carry_credited_short"] = describe_weekly(wc, "pnl_rs")
            rec[key]["verdict_top1_rule"] = verdict_for(rec[key]["futures"]["hit_lo95"])
        # per-leg hit rate for the top3 basket -- the second reading of "top-3 hit rate"
        if arm == "top3":
            for roll in rolls:
                sub = legs.loc[legs["slot"].isin(["top1", "top2", "top3"]) & legs["ok"]].copy()
                sub["net_rs"] = net_pnl(sub, roll)
                k = int((sub["net_rs"] > 0).sum())
                p, lo, hi = wilson_ci(k, len(sub))
                rec[f"roll{int(roll)}"]["per_leg_hit"] = {
                    "n_legs": int(len(sub)), "hits": k, "hit_rate": p,
                    "hit_lo95": lo, "hit_hi95": hi,
                    "mean_leg_pnl_rs": float(sub["net_rs"].mean())}
        out["arms"][arm] = rec

    top1 = out["arms"]["top1"]
    out["verdict"] = {
        "rule": f"KILL unless the top-1 weekly hit rate's lower Wilson 95% bound exceeds "
                f"{HIT_BAR:.2f} (NIFTY50_PLAN sec.6.2)",
        "by_roll": {f"roll{int(r)}": {
            "hit_rate": top1[f"roll{int(r)}"]["futures"]["hit_rate"],
            "hit_lo95": top1[f"roll{int(r)}"]["futures"]["hit_lo95"],
            "hit_hi95": top1[f"roll{int(r)}"]["futures"]["hit_hi95"],
            "verdict": top1[f"roll{int(r)}"]["verdict_top1_rule"]} for r in rolls},
        "overall": ("PROCEED" if all(top1[f"roll{int(r)}"]["verdict_top1_rule"] == "PROCEED"
                                     for r in rolls) else "KILL"),
        "rule_power": rule_power(int(top1[f"roll{int(rolls[0])}"]["futures"]["n_weeks"])),
    }
    return out, legs


# ------------------------------------------------------------------ writers


def _rs(x: float) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{'-' if x < 0 else ''}Rs {abs(x):,.0f}"


def write_md(res: dict, path: str) -> None:
    cfg, smp = res["config"], res["sample"]
    L = []
    A = L.append
    A("# Single-stock futures venue on the PIT NIFTY 50 (NIFTY50_PLAN sec.6.2)\n")
    A(f"Scores: `{os.path.basename(cfg['scores'])}` (is_n50 cells only). "
      f"{smp['n_weekly_cycles']} weekly cycles, {smp['first_signal']} to {smp['last_signal']}, "
      f"one lot per name, entered at open(t+1) and exited at open(t+6).\n")
    A(f"**Verdict: {res['verdict']['overall']}.** {res['verdict']['rule']}\n")
    A("| roll bps/month | top-1 hit rate | Wilson 95% | verdict |")
    A("|---|---|---|---|")
    for k, v in res["verdict"]["by_roll"].items():
        A(f"| {k.replace('roll', '')} | {v['hit_rate']:.4f} | "
          f"[{v['hit_lo95']:.4f}, {v['hit_hi95']:.4f}] | **{v['verdict']}** |")
    A("")
    rp = res["verdict"].get("rule_power") or {}
    if rp.get("min_hits_to_pass") is not None:
        pw = rp.get("power_vs_true_hit_rate", {})
        A(f"**Both readings of that table, because only one of them is about the venue.** "
          f"(i) The point estimate is below a coin flip: 54 of {rp['n_weeks']} weeks. (ii) The "
          f"rule itself was nearly unpassable on this sample -- a lower Wilson bound above "
          f"{rp['bar']:.2f} needs {rp['min_hits_to_pass']} of {rp['n_weeks']} weeks "
          f"({rp['min_hit_rate_to_pass']:.3f}), so a strategy whose TRUE hit rate is exactly "
          f"0.55 would clear it {pw.get('0.55', float('nan')):.1%} of the time and one at 0.60 "
          f"only {pw.get('0.60', float('nan')):.1%} of the time. The KILL here is therefore not "
          f"carried by the rule: it is carried by a point estimate of "
          f"{res['verdict']['by_roll']['roll55']['hit_rate']:.4f}, which is on the wrong side "
          f"of 0.50 before the bar is even consulted. Note also that 0.55 sits INSIDE the "
          f"95% interval at every roll setting -- the sample cannot exclude a true 55% hit "
          f"rate, it simply gives no evidence for one.\n")
    A("## Why the answer does not depend on the hit rate\n")
    t1 = res["arms"]["top1"]["roll55"]["futures"]
    ed = res["arms"]["top1"]["roll55"]["edge"]
    ww = res["arms"]["top1"]["roll55"]["worst_week"]
    A(f"One top-1 lot's weekly P&L sd is {_rs(t1['sd_rs'])} = "
      f"**{t1['sd_pct_of_book']:.2f}% of the Rs 10 lakh book** against a mean of "
      f"{_rs(t1['mean_rs'])} ({t1['mean_pct_of_book']:+.3f}% of book). Worst single week "
      f"{_rs(t1['min_rs'])} ({t1['worst_week_pct_of_book']:.2f}% of book).\n")
    A(f"The plan pre-registered the shape of this: *\"at IC ~ 0.02 the expected weekly edge is "
      f"a few bps against a weekly sd of ~ 4%\"*. Measured, before any cost: the top-1 lot's "
      f"gross weekly edge is **{ed['mean_gross_bps_of_notional']:+.1f} bps of contract value "
      f"against a weekly sd of {ed['sd_gross_bps_of_notional']:.0f} bps** "
      f"(ratio {ed['gross_edge_over_sd']:+.4f}). Round-trip cost plus 5 sessions of carry at "
      f"55 bps/month is {res['cost_reference']['round_trip_bps_at_7p5L']:.2f} + "
      f"{res['cost_reference']['carry_bps_5_sessions']['55']:.2f} = "
      f"{res['cost_reference']['round_trip_bps_at_7p5L'] + res['cost_reference']['carry_bps_5_sessions']['55']:.2f} bps, "
      f"i.e. the toll is {(res['cost_reference']['round_trip_bps_at_7p5L'] + res['cost_reference']['carry_bps_5_sessions']['55']) / max(ed['mean_gross_bps_of_notional'], 1e-9):.1f}x "
      f"the gross edge. Gross the arm annualises to "
      f"{ed['ann_gross_pct_of_book']:+.2f}% of book; net, {ed['ann_net_pct_of_book']:+.2f}%.\n")
    if ww:
        legdesc = ", ".join(f"{l['symbol']} {'long' if l['side'] > 0 else 'short'} "
                            f"{l['ret'] * 100:+.1f}%" for l in ww["legs"])
        A(f"The worst week is not a percentile, it is an event: **{ww['entry_date']} to "
          f"{ww['exit_date']}, {legdesc}**, {_rs(ww['pnl_rs'])} = {ww['pct_of_book']:.1f}% of "
          f"the book in five sessions from one position.\n")
    cr = res["cost_reference"]
    A(f"Statutory round trip on a Rs 7.5 lakh lot: {_rs(cr['round_trip_rs_at_7p5L'])} "
      f"({cr['round_trip_bps_at_7p5L']:.2f} bps). Carry over 5 sessions at 45/55/65 bps/month: "
      + " / ".join(_rs(cr["carry_rs_5_sessions_at_7p5L"][k])
                   for k in sorted(cr["carry_rs_5_sessions_at_7p5L"])) + ".\n")
    g1 = res["arms"]["top1"]["roll55"]["futures_gross"]
    A(f"And the binding constraint is not cost. **Before any cost** the top-1 pick wins "
      f"{g1['hit_rate']:.4f} of weeks (Wilson [{g1['hit_lo95']:.4f}, {g1['hit_hi95']:.4f}], "
      f"t = {g1['t_stat']:.2f} on {g1['n_weeks']} weeks) -- a coin flip. Zeroing every charge in "
      f"the model would still leave the venue short of the 55% bar; the top of a 50-name "
      f"cross-section scored at IC ~ 0.022 is simply not a 55%-of-the-time winner.\n")
    e3 = res["arms"]["top3"]["roll55"]["edge"]
    A(f"One thing here is not negative and should be carried to the cash book rather than "
      f"buried in a kill: the top-**3** basket's gross weekly edge is "
      f"{e3['mean_gross_bps_of_notional']:+.1f} bps of contract value against a sd of "
      f"{e3['sd_gross_bps_of_notional']:.0f} bps -- a ratio of {e3['gross_edge_over_sd']:+.4f}, "
      f"{e3['gross_edge_over_sd'] / max(ed['gross_edge_over_sd'], 1e-9):.1f}x the top-1 ratio. "
      f"The signal is not concentrated at rank 1; spreading it across names is what makes it "
      f"visible, which is an argument for the k = 10 cash book and against every "
      f"one-lot-in-one-name expression of it. Costs still take the top-3 basket net negative "
      f"({e3['ann_gross_pct_of_book']:+.2f}% gross of book annualised, "
      f"{e3['ann_net_pct_of_book']:+.2f}% net) because a three-lot position is "
      f"{e3['mean_notional_rs'] / BOOK:.2f}x the book.\n")
    A("## Arms (roll = 55 bps/month, margin 22%)\n")
    A("| arm | n wk | mean Rs | median Rs | sd Rs | worst Rs | hit | Wilson 95% | "
      "cash Rs 1 lakh mean | cash hit |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        r = res["arms"][arm]["roll55"]
        f, c = r["futures"], r["cash_1L"]
        A(f"| {arm} | {f['n_weeks']} | {_rs(f['mean_rs'])} | {_rs(f['median_rs'])} | "
          f"{_rs(f['sd_rs'])} | {_rs(f['min_rs'])} | {f['hit_rate']:.4f} | "
          f"[{f['hit_lo95']:.4f}, {f['hit_hi95']:.4f}] | {_rs(c['mean_rs'])} | "
          f"{c['hit_rate']:.4f} |")
    pl = res["arms"]["top3"]["roll55"]["per_leg_hit"]
    A(f"\n\"Top-3 hit rate\" has two readings and both are reported: the BASKET (all three lots "
      f"together, the row above) wins {res['arms']['top3']['roll55']['futures']['hit_rate']:.4f} "
      f"of weeks; the individual LEGS win {pl['hit_rate']:.4f} of "
      f"{pl['n_legs']} legs (Wilson [{pl['hit_lo95']:.4f}, {pl['hit_hi95']:.4f}], mean "
      f"{_rs(pl['mean_leg_pnl_rs'])} per leg). Neither clears 55%.")
    A("")
    A("## Roll-cost sensitivity (futures net mean Rs/week and hit rate)\n")
    A("| arm | 45 | 55 | 65 |")
    A("|---|---|---|---|")
    for arm in ARMS:
        cells = []
        for r in cfg["roll_band_bps_per_month"]:
            f = res["arms"][arm][f"roll{int(r)}"]["futures"]
            cells.append(f"{_rs(f['mean_rs'])} / {f['hit_rate']:.4f}")
        A(f"| {arm} | " + " | ".join(cells) + " |")
    A("")
    for arm in ("bottom1", "pair"):
        rc = res["arms"][arm]["roll55"].get("futures_carry_credited_short")
        if rc:
            A(f"Carry-sign ambiguity, {arm}: charged {_rs(res['arms'][arm]['roll55']['futures']['mean_rs'])}"
              f"/week (hit {res['arms'][arm]['roll55']['futures']['hit_rate']:.4f}); "
              f"credited on the short leg {_rs(rc['mean_rs'])}/week (hit {rc['hit_rate']:.4f}).")
    A("")
    A("## Margin\n")
    A("| arm | margin % | mean margin | max margin | max % of book | weeks margin > Rs 10 lakh |")
    A("|---|---|---|---|---|---|")
    for arm in ARMS:
        for key, m in res["arms"][arm]["roll55"]["margin"].items():
            A(f"| {arm} | {m['margin_pct']:.2f} | {_rs(m['mean_margin_rs'])} | "
              f"{_rs(m['max_margin_rs'])} | {m['max_margin_pct_of_book']:.1f}% | "
              f"{m['frac_weeks_margin_gt_book']:.4f} |")
    A("")
    A("## 3-year block bootstrap (block 4 weeks, "
      f"{cfg['bootstrap']['draws']} draws, {cfg['bootstrap']['weeks']} weeks, "
      f"Rs 10 lakh start, roll 55)\n")
    A("| arm | P(margin call) | P(loss > 30%) | P(final > start) | median final | "
      "median max DD | P(equity <= 0) |")
    A("|---|---|---|---|---|---|---|")
    for arm in ARMS:
        b = res["arms"][arm]["roll55"]["bootstrap"]
        A(f"| {arm} | {b['p_margin_call']:.4f} | {b['p_loss_gt_30pct_anytime']:.4f} | "
          f"{b['p_final_gt_start']:.4f} | {_rs(b['median_final_rs'])} | "
          f"{_rs(b['median_max_drawdown_rs'])} | {b['p_ruin_equity_le_0']:.4f} |")
    A("")
    bo = res["arms"]["top1"]["roll55"]["bootstrap_observed_length"]
    bc = res["arms"]["top1"]["roll55"]["bootstrap_cash_1L"]
    A(f"P(loss > 30%) is measured as \"equity touches Rs 7 lakh at any point\"; the "
      f"end-of-path reading is "
      f"{res['arms']['top1']['roll55']['bootstrap']['p_loss_gt_30pct_final']:.4f} for top1. "
      f"Paths are NOT stopped at the first margin call, so the two columns overlap. The "
      f"same bootstrap over the {bo['weeks']} weeks actually observed (rather than a 3-year "
      f"extrapolation) gives top1 P(margin call) {bo['p_margin_call']:.4f}, "
      f"P(loss > 30%) {bo['p_loss_gt_30pct_anytime']:.4f}, median final "
      f"{_rs(bo['median_final_rs'])}. The Rs 1 lakh CASH version of the same picks over "
      f"{bc['weeks']} weeks: P(margin call) {bc['p_margin_call']:.4f} (there is no margin), "
      f"P(loss > 30%) {bc['p_loss_gt_30pct_anytime']:.4f}, median final "
      f"{_rs(bc['median_final_rs'])}, median max drawdown {_rs(bc['median_max_drawdown_rs'])} "
      f"-- the same signal, the same names, "
      f"{_rs(bc['median_final_rs'] - res['arms']['top1']['roll55']['bootstrap']['median_final_rs'])} "
      f"better than the futures arm over the SAME {bc['weeks']} weeks (the 113-week row above is "
      f"a shorter path and is not the comparable number), because it is sized at a tenth of the "
      f"book instead of three-quarters of it.")
    A("")
    A("## Expiry-week behaviour (roll 55)\n")
    A("| arm | weeks crossing an expiry | mean Rs | hit | weeks clear | mean Rs | hit |")
    A("|---|---|---|---|---|---|---|")
    for arm in ARMS:
        sp = res["arms"][arm]["roll55"]["expiry_split"]
        A(f"| {arm} | {sp['crossing']['n']} | {_rs(sp['crossing']['mean_rs'])} | "
          f"{sp['crossing']['hit_rate']:.4f} | {sp['clear']['n']} | {_rs(sp['clear']['mean_rs'])} | "
          f"{sp['clear']['hit_rate']:.4f} |")
    A("")
    e = res["expiry"]
    A("## Physical settlement\n")
    A(f"All NSE stock F&O is compulsorily physically settled (SEBI 2018-04-11, from the Oct-2019 "
      f"expiry). Of {e['thursday']['n_weeks']} five-session holds, "
      f"**{e['thursday']['n_cross_strict']} ({e['thursday']['frac_cross_strict']:.1%}) contain a "
      f"monthly expiry** under the {e['convention']}, so those lots must be rolled to the next "
      f"series (paying the roll again) or carried into delivery -- an ITM long takes delivery of "
      f"the full lot in cash, an ITM short must deliver shares it does not own. "
      f"{e['thursday']['n_cross_inclusive']} holds touch an expiry on the exit session itself "
      f"(the inclusive reading, where the position is already flat at the open). Under a "
      f"last-Tuesday convention the count is {e['tuesday_sensitivity']['n_cross_strict']}; the "
      f"exchange has changed expiry weekdays during this window, so treat the count as the shape "
      f"of the problem, not an exact schedule.\n")
    A("## Caveats that are larger than the result\n")
    A(f"* {smp['n_legs_proxy_lot']} of {smp['n_legs']} legs "
      f"({smp['frac_legs_proxy_lot']:.1%}) use a synthetic Rs 7.5 lakh lot because only "
      f"{len(LOT_SIZES)} NIFTY 50 names have a researched lot size; real lots are lumpy and "
      f"their contract values range at least Rs 4.2-9.9 lakh.")
    A("* Futures are priced off SPOT opens. A real fill is spot + basis, and the basis is the "
      "quantity the roll-cost parameter is standing in for; there is no per-stock basis series "
      "on disk, so the P&L is spot P&L minus a flat carry, not a futures P&L.")
    A("* Margin is a flat fraction of contract value. Real SPAN+exposure moves daily, rises "
      "into expiry, and the physical-delivery margin ramps E-4 10% / E-3 25% / E-2 45% / "
      "E-1 70% on ITM positions -- so the margin numbers here are a floor.")
    A("* The Rs 1 lakh CASH comparison for `bottom1` and the short half of `pair` is NOT "
      "executable by a resident individual: delivery short selling does not exist outside "
      "intraday, and Zerodha's SLB covers Series B (non-F&O) names only. It is computed for "
      "symmetry and should be read as a counterfactual, not as an alternative venue. The "
      "futures short is the only real way to express it, which is the one thing this venue "
      "offers that the cash book cannot.")
    A("* No MTM financing, no interest on the margin, no tax. Futures P&L is non-speculative "
      "business income (s.43(5) proviso (d)); at 31.2% it takes another third of anything "
      "positive and gives back a third of anything negative.")
    A(f"* The sample is {smp['n_weekly_cycles']} weekly cycles, not the ~150 the assignment "
      f"assumed: the primary score file starts at {smp['first_signal']} because the "
      f"Fama-MacBeth scorer's trailing window eats the first two years. Every hit rate and "
      f"every Wilson interval above is on {smp['n_weekly_cycles']} weeks, and the 3-year "
      f"bootstrap path is an EXTRAPOLATION of {smp['n_weekly_cycles']} observed weeks to "
      f"{cfg['bootstrap']['weeks']}, not three years of evidence.")
    A(f"* The scores are the probe's primary cell, whose own gate M1 FAILED "
      f"(IC +0.0220, below the +0.033 t = 3 bar at N = 50). A venue study on a signal that did "
      f"not clear its statistical gate can only ever return a kill or a warning; it cannot "
      f"return a recommendation.")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default=SCORES)
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--calendar", default=CALENDAR)
    ap.add_argument("--out-dir", default=N50)
    ap.add_argument("--tag", default="futures_n50")
    ap.add_argument("--futures-preset", default="zerodha_fut", choices=list(FUTURES_PRESETS))
    ap.add_argument("--cash-preset", default="zerodha", choices=list(PRESETS))
    ap.add_argument("--margin-pct", type=float, default=MARGIN_PCT_DEFAULT)
    ap.add_argument("--rolls", default="45,55,65",
                    type=lambda s: tuple(float(x) for x in s.split(",")))
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--weeks", type=int, default=156)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write-legs", action="store_true",
                    help="also write the per-leg blotter as a parquet")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    lots_json = research_lots()
    if lots_json:
        assert lots_json == LOT_SIZES, (
            f"costs.LOT_SIZES disagrees with futures_facts.json: "
            f"{set(lots_json.items()) ^ set(LOT_SIZES.items())}")
    scores = load_scores(args.scores)
    market = load_market(args.panel, args.calendar)
    res, legs = run(scores, market, fut_preset=args.futures_preset,
                    cash_preset=args.cash_preset, margin_pct=args.margin_pct,
                    rolls=args.rolls, draws=args.draws, block=args.block,
                    weeks=args.weeks, seed=args.seed)
    res["config"]["scores"] = args.scores
    res["config"]["lot_sizes_checked_against_research_json"] = bool(lots_json)
    res["runtime_sec"] = time.time() - t0

    jpath = os.path.join(args.out_dir, f"{args.tag}.json")
    mpath = os.path.join(args.out_dir, f"{args.tag}.md")
    with open(jpath, "w") as f:
        json.dump(res, f, indent=2, default=str)
    write_md(res, mpath)
    if args.write_legs:
        lp = os.path.join(args.out_dir, f"{args.tag}_legs.parquet")
        legs.to_parquet(lp, index=False)
        sidecar(lp.replace(".parquet", "_meta.json"), vars(args),
                extra={"rows": int(len(legs)), "priced": int(legs["ok"].sum())})
    sidecar(os.path.join(args.out_dir, f"{args.tag}_meta.json"), vars(args),
            extra={"verdict": res["verdict"], "sample": res["sample"],
                   "runtime_sec": res["runtime_sec"]})
    print(f"[n50_futures] {res['verdict']['overall']} -- wrote {jpath} and {mpath} "
          f"in {res['runtime_sec']:.1f}s")


if __name__ == "__main__":
    main()
