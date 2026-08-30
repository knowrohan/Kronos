"""Gates M2 and M3: does the NIFTY 50 book beat the tracker after cost AND after tax?

NIFTY50_PLAN sec.6.3. ``n50_backtest.py`` stops at net-of-cost; this is the rest of the money
question, and it is where most of a weekly book's apparent edge goes:

  * **The tax asymmetry is the point.** A weekly top-k book realises essentially its whole gain
    inside every financial year at the short-term rate. A tracker holder realises nothing until
    they sell, and then pays 12.5% + cess above the Rs 1.25 lakh s.112A exemption. Comparing the
    two before tax hands the book several points a year it does not have -- the same order as
    the entire edge being tested. So the book is taxed under ``stcg`` (20.8%, STT added back
    because s.48's proviso disallows it), ``business_31`` (31.2%) and ``business_21`` (20.8%),
    and the tracker is taxed LTCG-on-exit; the "unrealised" tracker variant (never sells, never
    pays) is reported next to it as the other bracket of the honest comparison.
  * **Two tracker legs, ratified in sec.11.** ``nsei_tr_proxy`` (^NSEI open-to-open + 1.17 pp/yr
    accrual) is the GATE -- same open-to-open clock as the book, total-return level.
    ``niftybees_adj`` close-to-close is the DESCRIPTION -- the actually purchasable ETF, on a
    clock half a session away from the book's. Every number is reported against both, and where
    they disagree in sign the disagreement is printed rather than resolved.
  * **What this file does not do.** It does not re-simulate the strategy on the post-tax book
    (``tax.after_tax_curve`` withdraws the year's tax at FY end and lets the smaller book
    compound), it does not model advance-tax timing, and it is not tax advice. See the
    ``bench/nse/tax.py`` docstring for the full list of what is deliberately not modelled.

Inputs are whatever ``n50_backtest.py`` has already written into ``bench/out/nse/n50/``:
``bt_<tag>_metrics.json`` (which tags exist at all), ``bt_<tag>_<cell>.parquet`` (the daily
net-return series), ``bt_<tag>_<cell>_trades.parquet`` (the blotter the FIFO matcher needs) and
``bt_<tag>_bench_daily.parquet`` (every benchmark leg on the book's own sessions, written by the
backtest so the two files can never drift apart). The M3 clauses that live in the probe --
contamination split and fingerprint -- are read from ``probe_n50.json`` rather than recomputed.

Outputs: ``money_n50.json``, ``money_n50.md``, ``money_n50_meta.json``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from bench.nse import tax  # noqa: E402
from bench.nse.backtest import TRADING_DAYS, annualise  # noqa: E402
from bench.nse.common import sidecar  # noqa: E402
from bench.nse.costs import PRESETS  # noqa: E402
from bench.nse.mlcore import OUT_NSE  # noqa: E402
from bench.nse.n50_backtest import (TR_ACCRUAL_ANNUAL,  # noqa: E402
                                    TR_ACCRUAL_PER_SESSION)

N50 = os.path.join(OUT_NSE, "n50")

#: The plan's primary book (sec.6.1) and the two variants M3 asks about, plus the k=15/n_drop=2
#: corner, which is not a gate but is the cell that tells you whether a failure is a k effect or
#: an n_drop effect.
PRIMARY_CELL = "k10_d1"
SECONDARY_CELL = "k15_d1"
M3_VARIANT_CELLS = ("k15_d1", "k10_d2")
ALL_CELLS = ("k10_d1", "k10_d2", "k15_d1", "k15_d2")

GATE_BENCH = "nsei_tr_proxy"        # M2's bar (sec.11)
DESC_BENCH = "niftybees_adj"        # the purchasable tracker, description
TRACKERS = (GATE_BENCH, DESC_BENCH)
EW_BENCH = "ew_n50_drift"           # the equal-weight PIT N50 book, M3's last clause

REGIMES = ("stcg", "business_31", "business_21")
GATE_REGIMES = ("stcg", "business_31")     # the two M2 names
M2_EXCESS_BAR = 0.041                      # +4.1 pp, the record's money bar
FINGERPRINT_BAR = 0.30
#: The NIFTYBEES-minus-^NSEI gap measured on the whole bench_n50 sample, open-to-open (low, and
#: read off the ETF's stale open feed) and close-to-close (high, the clean feed). The hand-set
#: TR_ACCRUAL_ANNUAL of 1.17 pp/yr sits inside this; a verdict that changes across it is a
#: verdict about the constant rather than about the book.
MEASURED_BAND = (0.0097, 0.0144)
CHECKPOINT = "2025-07-01"                  # post-checkpoint = on or after this session


# ------------------------------------------------------------------ loading


def discover_tags(out_dir: str = N50) -> list[str]:
    """Every tag that has a finished ``bt_<tag>_metrics.json``, in a stable order."""
    tags = []
    for p in sorted(glob.glob(os.path.join(out_dir, "bt_*_metrics.json"))):
        tags.append(os.path.basename(p)[len("bt_"):-len("_metrics.json")])
    return tags


def load_tag(out_dir: str, tag: str) -> dict:
    """The backtest artefacts for one tag: metrics json, benchmark frame, per-cell book + blotter.

    A cell whose daily parquet is missing is skipped rather than faked; a cell with no trades
    parquet gets an empty blotter (which the tax layer handles: no realisation, no tax).
    """
    with open(os.path.join(out_dir, f"bt_{tag}_metrics.json")) as f:
        met = json.load(f)
    bpath = os.path.join(out_dir, f"bt_{tag}_bench_daily.parquet")
    bench = pd.read_parquet(bpath) if os.path.exists(bpath) else None
    if bench is not None:
        bench.index = pd.to_datetime(bench.index)
    cells = {}
    for cell in ALL_CELLS:
        dpath = os.path.join(out_dir, f"bt_{tag}_{cell}.parquet")
        if not os.path.exists(dpath):
            continue
        daily = pd.read_parquet(dpath)
        daily.index = pd.to_datetime(daily.index)
        tpath = os.path.join(out_dir, f"bt_{tag}_{cell}_trades.parquet")
        trades = (pd.read_parquet(tpath) if os.path.exists(tpath)
                  else pd.DataFrame(columns=["exec_date", "symbol", "weight_delta"]))
        cells[cell] = {"daily": daily, "trades": trades}
    return {"metrics": met, "bench": bench, "cells": cells}


# ------------------------------------------------------------------ after tax


def tracker_block(bench: pd.Series, capital: float, n_days: int) -> dict:
    """Pre-tax and after-tax terminal value / annualised return for one buy-and-hold leg."""
    realised = tax.tracker_after_tax(bench, capital, realise=True)
    unrealised = tax.tracker_after_tax(bench, capital, realise=False)
    held_days = int((bench.index[-1] - bench.index[0]).days) if len(bench) > 1 else 0
    return {
        "pre_tax_final_rs": float(unrealised),
        "pre_tax_ann_return": tax.annualised(unrealised, capital, n_days),
        "after_tax_final_rs_on_exit": float(realised),
        "after_tax_ann_return_on_exit": tax.annualised(realised, capital, n_days),
        "after_tax_final_rs_unrealised": float(unrealised),
        "after_tax_ann_return_unrealised": tax.annualised(unrealised, capital, n_days),
        "tax_paid_on_exit_rs": float(unrealised - realised),
        "held_days": held_days,
        "term": "long" if held_days > tax.LONG_TERM_DAYS else "short",
    }


def after_tax_block(daily: pd.DataFrame, trades: pd.DataFrame, bench: pd.DataFrame,
                    capital: float, cost_model, regimes=REGIMES) -> dict:
    """After-tax annualised return of the book, of each tracker, and the excess between them.

    The excess is a difference of ANNUALISED after-tax returns, not of terminal values: the two
    legs run over the same sessions so the exponent is common, and a difference of annualised
    returns is what the gate is written in.
    """
    net = daily["net_return"].astype(float)
    n = len(net)
    pre_final = float(capital * (1.0 + net).prod())
    out = {
        "n_days": int(n),
        "first_session": str(net.index[0].date()), "last_session": str(net.index[-1].date()),
        "capital_rs": float(capital),
        "pre_tax_final_rs": pre_final,
        "pre_tax_ann_return": tax.annualised(pre_final, capital, n),
        "regimes": {}, "trackers": {}, "after_tax_excess": {},
    }
    for reg in regimes:
        curve, parts = tax.after_tax_curve(net, trades, capital, reg, cost_model,
                                           return_parts=True)
        final = float(curve.iloc[-1])
        led = parts["tax"]
        out["regimes"][reg] = {
            "label": tax.REGIMES[reg]["label"], "rate": tax.REGIMES[reg]["rate"],
            "after_tax_final_rs": final,
            "after_tax_ann_return": tax.annualised(final, capital, n),
            "total_tax_rs": float(led["tax_rs"].sum()) if len(led) else 0.0,
            "tax_by_fy": ({str(r.fy): float(r.tax_rs) for r in led.itertuples()}
                          if len(led) else {}),
            "n_fills": int(len(parts["fills"])),
            "n_sell_matches": int(len(parts["matches"])),
            "n_partial_sells": (int(parts["matches"]["partial"].sum())
                                if len(parts["matches"]) else 0),
            "realised_gain_rs": (float(parts["matches"]["gain_rs"].sum())
                                 if len(parts["matches"]) else 0.0),
            "stt_paid_rs": (float(parts["fills"]["stt_rs"].sum())
                            if len(parts["fills"]) else 0.0),
            "transaction_cost_rs": (float(parts["fills"]["cost_rs"].sum())
                                    if len(parts["fills"]) else 0.0),
        }
    for t in TRACKERS:
        b = bench[t].reindex(net.index).fillna(0.0)
        out["trackers"][t] = tracker_block(b, capital, n)
    for reg in regimes:
        ab = out["regimes"][reg]["after_tax_ann_return"]
        out["after_tax_excess"][reg] = {
            t: {"on_exit": ab - out["trackers"][t]["after_tax_ann_return_on_exit"],
                "unrealised": ab - out["trackers"][t]["after_tax_ann_return_unrealised"]}
            for t in TRACKERS}
    return out


# ------------------------------------------------------------------ calendar years


def calendar_year_table(daily: pd.DataFrame, bench: pd.DataFrame,
                        benchmarks=TRACKERS) -> list[dict]:
    """Per calendar year: the book's net return, each tracker's, and the excess.

    Two readings, both reported, because they can differ in a partial year: ``excess_cum`` is
    the difference of the two cumulative within-year returns (what a holder actually experienced
    that year) and ``excess_ann`` is the difference after annualising both to 252 sessions
    (comparable across years of different length, and the convention the headline uses). The
    gate uses ``excess_cum``; if the two disagree in sign the row carries ``sign_disagrees``.
    """
    net = daily["net_return"].astype(float)
    rows = []
    for y, idx in net.groupby(net.index.year).groups.items():
        seg = net.loc[idx]
        n = len(seg)
        cum = float((1.0 + seg).prod() - 1.0)
        row = {"year": int(y), "n_days": int(n),
               "book_return_cum": cum,
               "book_return_ann": float((1.0 + cum) ** (TRADING_DAYS / max(n, 1)) - 1.0),
               "partial": bool(n < 200)}
        for b in benchmarks:
            bs = bench[b].reindex(net.index).fillna(0.0).loc[idx]
            bcum = float((1.0 + bs).prod() - 1.0)
            bann = float((1.0 + bcum) ** (TRADING_DAYS / max(n, 1)) - 1.0)
            ex_c, ex_a = cum - bcum, row["book_return_ann"] - bann
            row[f"bench_{b}_cum"] = bcum
            row[f"bench_{b}_ann"] = bann
            row[f"excess_{b}_cum"] = ex_c
            row[f"excess_{b}_ann"] = ex_a
            row[f"excess_{b}_sign_disagrees"] = bool((ex_c > 0) != (ex_a > 0))
        rows.append(row)
    return sorted(rows, key=lambda r: r["year"])


# ------------------------------------------------------------------ gates


def _clause(name: str, ok: bool | None, value, clause: str, **extra) -> dict:
    d = {"name": name, "pass": (None if ok is None else bool(ok)),
         "value": value, "clause": clause}
    d.update(extra)
    return d


def accrual_sensitivity(daily: pd.DataFrame, bench: pd.DataFrame,
                        grid=(0.0, 0.0065, 0.0097, TR_ACCRUAL_ANNUAL, 0.0144, 0.020)) -> dict:
    """What M2's headline excess would be at other values of the hand-set TR accrual.

    ``nsei_tr_proxy`` is a price index plus a constant nobody measured on this book: the
    +1.17 pp/yr NIFTYBEES-minus-^NSEI gap of NIFTY50_PLAN sec.11. Every pp of it comes straight
    off the reported excess, and the plausible band is wide (whole-sample +0.97 pp measured
    open-to-open, +1.44 pp close-to-close), so a gate decided within a pp of the bar could be an
    artefact of the constant. This strips the accrual back out of the benchmark leg, re-adds each
    candidate, and also solves for the accrual at which the verdict would flip -- so the reader
    can see whether the verdict survives the whole band or only the midpoint.
    """
    book = annualise(daily["net_return"].astype(float))
    leg = bench[GATE_BENCH].reindex(daily.index).fillna(0.0)
    per0 = TR_ACCRUAL_PER_SESSION
    live = leg != 0.0                       # sessions the price series could actually price
    raw = leg - per0 * live                 # the bare ^NSEI open-to-open leg

    def excess_at(a: float) -> float:
        per = (1.0 + a) ** (1.0 / TRADING_DAYS) - 1.0
        return float(book - annualise(raw + per * live))

    rows = [{"accrual_annual": float(a), "excess_ann": excess_at(a),
             "passes_m2a": bool(excess_at(a) >= M2_EXCESS_BAR)} for a in grid]
    # excess_at is monotone decreasing in a, so a plain bisection finds the flip point.
    lo, hi = -0.05, 0.20
    be = None
    if (excess_at(lo) - M2_EXCESS_BAR) * (excess_at(hi) - M2_EXCESS_BAR) < 0:
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if excess_at(mid) >= M2_EXCESS_BAR:
                lo = mid
            else:
                hi = mid
        be = 0.5 * (lo + hi)
    # The flag is over the MEASURED band only. The grid deliberately runs wider (0 pp is a bare
    # price index, which is not the user's total-return bar at all), and a flag computed over the
    # whole grid would report "does not survive" for a book whose verdict is in fact identical at
    # every accrual anyone has measured.
    lo_v, hi_v = excess_at(MEASURED_BAND[0]), excess_at(MEASURED_BAND[1])
    return {"grid": rows, "breakeven_accrual_annual": be,
            "measured_band_annual": list(MEASURED_BAND),
            "excess_at_measured_band": [lo_v, hi_v],
            "verdict_survives_measured_band":
                bool((lo_v >= M2_EXCESS_BAR) == (hi_v >= M2_EXCESS_BAR)),
            "note": ("excess is monotone decreasing in the accrual; breakeven is the accrual at "
                     "which the k10_d1 net excess equals the +4.1 pp bar, or null if the bar is "
                     "unreachable on this book over any accrual in [-5, +20] pp/yr. "
                     "verdict_survives_measured_band asks only whether the +0.97 pp (open-to-open) "
                     "and +1.44 pp (close-to-close) whole-sample readings agree on the verdict")}


def m2_gates(met: dict, atx: dict, tag: str, sens: dict | None = None) -> dict:
    """M2: +4.1 pp net-of-cost excess over nsei_tr_proxy, and after-tax excess > 0 twice over."""
    cells = met["books"]
    prim = cells.get(PRIMARY_CELL, {})
    ex_gate = prim.get(GATE_BENCH, {}).get("ann_excess_net")
    ex_desc = prim.get(DESC_BENCH, {}).get("ann_excess_net")
    cl = [_clause(
        "a_net_excess_vs_nsei_tr_proxy", (ex_gate is not None and ex_gate >= M2_EXCESS_BAR),
        ex_gate, f"k=10 n_drop=1 net-of-cost annualised excess over {GATE_BENCH} >= "
                 f"{M2_EXCESS_BAR * 100:.1f} pp",
        also_vs_niftybees_adj=ex_desc,
        ci95_annualised=prim.get(GATE_BENCH, {}).get("excess_ci95_annualised"),
        accrual_sensitivity=sens)]
    for reg in GATE_REGIMES:
        v = atx["after_tax_excess"][reg][GATE_BENCH]["on_exit"]
        cl.append(_clause(
            f"b_after_tax_excess_{reg}", v > 0, v,
            f"after-tax annualised excess over {GATE_BENCH} (tracker taxed LTCG-on-exit) > 0 "
            f"under {reg} ({tax.REGIMES[reg]['rate'] * 100:.1f}%)",
            vs_niftybees_adj=atx["after_tax_excess"][reg][DESC_BENCH]["on_exit"],
            vs_unrealised_tracker=atx["after_tax_excess"][reg][GATE_BENCH]["unrealised"]))
    return {"tag": tag, "cell": PRIMARY_CELL, "clauses": cl,
            "verdict": all(c["pass"] for c in cl),
            "note": ("the +-6-13 pp CI on the excess is description, not a gate: it cannot close "
                     "on this many sessions (NIFTY50_PLAN sec.6.3)")}


def m3_gates(met: dict, years: list[dict], probe: dict | None, tag: str) -> dict:
    """M3: robustness -- years, contamination, k/n_drop variants, fingerprint, equal-weight."""
    cells = met["books"]
    counted = [r for r in years if r["year"] in (2024, 2025, 2026)]
    pos = [r["year"] for r in counted if r[f"excess_{GATE_BENCH}_cum"] > 0]
    disagree = [r["year"] for r in counted if r[f"excess_{GATE_BENCH}_sign_disagrees"]]
    cl = [_clause(
        "a_positive_years", len(pos) >= 2, len(pos),
        f"positive net excess over {GATE_BENCH} in >= 2 of the 3 counted calendar years "
        f"(2024, 2025, 2026-to-date; 2023 is reported but not counted)",
        positive_years=pos, counted_years=[r["year"] for r in counted],
        per_year={str(r["year"]): r[f"excess_{GATE_BENCH}_cum"] for r in years},
        annualised_reading_disagrees_in=disagree)]

    pre = post = None
    if probe is not None:
        pre = probe["contamination"]["pre"]["mean_ic"]
        post = probe["contamination"]["post"]["mean_ic"]
    cl.append(_clause(
        "b_post_ge_pre_ic", None if pre is None else (post >= pre),
        None if pre is None else post - pre,
        f"post-checkpoint (>= {CHECKPOINT}) IC >= pre-checkpoint IC, from probe_n50.json",
        pre_ic=pre, post_ic=post,
        checkpoint=(probe["contamination"]["checkpoint"] if probe else None)))

    var = {c: cells.get(c, {}).get(GATE_BENCH, {}).get("ann_excess_net") for c in ALL_CELLS}
    need = [var[c] for c in M3_VARIANT_CELLS]
    cl.append(_clause(
        "c_variants_positive", all(v is not None and v > 0 for v in need),
        {c: var[c] for c in M3_VARIANT_CELLS},
        f"k=15 (k15_d1) and n_drop=2 (k10_d2) variants both have positive net excess over "
        f"{GATE_BENCH}", all_cells=var))

    fp = None
    if probe is not None:
        fp = max(abs(v) for v in probe["fingerprint"].values())
    cl.append(_clause(
        "d_fingerprint", None if fp is None else (fp < FINGERPRINT_BAR), fp,
        f"fingerprint max |rho| vs the free factors < {FINGERPRINT_BAR}, from probe_n50.json",
        argmax=(max(probe["fingerprint"], key=lambda k: abs(probe["fingerprint"][k]))
                if probe else None)))

    ew = cells.get(PRIMARY_CELL, {}).get(EW_BENCH, {}).get("ann_excess_net")
    cl.append(_clause(
        "e_beats_equal_weight", ew is not None and ew > 0, ew,
        "annualised net excess over the equal-weight PIT N50 book (same sessions, same costs) "
        "> 0 -- i.e. the book is not just an equal-weight tilt",
        ew_rebal=cells.get(PRIMARY_CELL, {}).get("ew_n50_rebal", {}).get("ann_excess_net"),
        const_score=cells.get(PRIMARY_CELL, {}).get("const_score", {}).get("ann_excess_net")))

    known = [c["pass"] for c in cl if c["pass"] is not None]
    return {"tag": tag, "cell": PRIMARY_CELL, "clauses": cl,
            "verdict": (all(known) if len(known) == len(cl) else None),
            "n_unknown_clauses": len(cl) - len(known)}


# ------------------------------------------------------------------ report


def _pp(x, nd: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) \
        else f"{x * 100:+.{nd}f}"


def _verdict(v) -> str:
    return "PASS" if v is True else ("FAIL" if v is False else "UNDECIDED")


def render_md(payload: dict) -> str:
    L: list[str] = []
    A = L.append
    A("# NIFTY 50 money test -- gates M2 and M3\n")
    A(f"_Written {payload['meta']['written_at']} by `bench/nse/n50_money.py`. "
      f"Capital Rs {payload['meta']['capital_rs']:,.0f}, cost preset "
      f"`{payload['meta']['cost_preset']}`. Working model, not tax advice._\n")
    A(f"Gate benchmark: **`{GATE_BENCH}`** = ^NSEI open-to-open + "
      f"{payload['meta']['tr_accrual_annual'] * 100:.2f} pp/yr accrual "
      f"({payload['meta']['tr_accrual_per_session']:.8f} per session, compounded). "
      f"Description benchmark: **`{DESC_BENCH}`** close-to-close.\n")

    dec = payload["decision_tag"]
    A(f"## Verdict on `{dec}` (the plan's primary book, k=10 n_drop=1)\n")
    g = payload["tags"][dec]
    A(f"* **M2 {_verdict(g['m2']['verdict'])}**  |  **M3 {_verdict(g['m3']['verdict'])}**")
    prim = g["books"][PRIMARY_CELL][GATE_BENCH]
    nb = g["null"].get("by_benchmark", {}).get(GATE_BENCH, {})
    ci = prim["excess_ci95_annualised"]
    A(f"* Net-of-cost annualised excess over `{GATE_BENCH}`: **{_pp(prim['ann_excess_net'])} pp** "
      f"against a **{M2_EXCESS_BAR * 100:.1f} pp** bar, block-bootstrap CI95 "
      f"[{_pp(ci[0])}, {_pp(ci[1])}] pp, IR {prim['information_ratio']:+.2f}, "
      f"max drawdown {_pp(prim['max_drawdown'])} pp.")
    if nb:
        # The closing sentence quotes the null's own arithmetic rather than an adjective fixed at
        # writing time: p is the share of random-score books that matched or beat the real one,
        # so "1 in N" is that share inverted, and it moves when the number moves.
        pv = float(nb["p_value_one_sided"])
        beat = int(round(pv * float(g["null"].get("n_seeds") or 0)))
        share = (f"1 in {1.0 / pv:.0f}" if pv > 0 else
                 f"0 of {g['null'].get('n_seeds')}")
        A(f"* N3 random-score null, {g['null'].get('n_seeds')} seeds at identical settings: the "
          f"observed excess sits at the **{nb['percentile_of_observed']:.1f}th percentile** "
          f"(one-sided p = {pv:.3f}, z = {nb['z_vs_null']:+.2f}; null mean "
          f"{_pp(nb['null_mean'])} pp, sd {_pp(nb['null_sd'])} pp). {beat} of the random "
          f"score vectors ({share}) matched or beat the real book"
          + (", which is not a demonstrated edge." if pv >= 0.05 else
             ", so the book is outside the random-score distribution at the 5% level."))
    sens = g["m2"]["clauses"][0].get("accrual_sensitivity")
    if sens:
        be = sens["breakeven_accrual_annual"]
        band = ", ".join(f"{r['accrual_annual'] * 100:.2f} pp -> {_pp(r['excess_ann'])} pp "
                         f"({_verdict(r['passes_m2a'])})" for r in sens["grid"])
        A(f"* The gate benchmark's total-return accrual is the one constant set by hand "
          f"({payload['meta']['tr_accrual_annual'] * 100:.2f} pp/yr, band roughly +0.97 to "
          f"+1.44 pp depending on the clock it is measured on). Excess at each candidate: "
          f"{band}. "
          + (f"M2 clause (a) would flip at an accrual of **{be * 100:+.2f} pp/yr**, "
             + (("outside the measured band, so the verdict is not an artefact of the constant."
                 if sens["verdict_survives_measured_band"] else
                 "INSIDE the measured +0.97 to +1.44 pp band, so the verdict turns on which "
                 "clock the accrual is measured on and must be read as undecided."))
             if be is not None else
             "The bar is unreachable at any accrual in [-5, +20] pp/yr, so the verdict is not "
             "an artefact of the constant."))
    A("")
    for gate in ("m2", "m3"):
        A(f"\n### {gate.upper()} clauses\n")
        A("| clause | PASS/FAIL | number | statement |")
        A("|---|---|---|---|")
        for c in g[gate]["clauses"]:
            v = c["value"]
            if isinstance(v, float):
                s = f"{v * 100:+.2f} pp" if abs(v) < 1 else f"{v:+.4f}"
                if c["name"].startswith(("b_post", "d_finger")):
                    s = f"{v:+.4f}"
            elif isinstance(v, dict):
                s = ", ".join(f"{k} {_pp(x)} pp" for k, x in v.items())
            else:
                s = str(v)
            A(f"| `{c['name']}` | **{_verdict(c['pass'])}** | {s} | {c['clause']} |")
        A("")

    A("\n## Windows -- why the tags are NOT comparable to each other\n")
    A("Each arm's estimator has its own burn-in, so each tag's book runs over its own sessions "
      "and each is measured against the tracker OVER THOSE SESSIONS. The tracker's own return "
      "moves by ten points across these windows, so a tag with a later start is not competing "
      "on the same course. Read down a column only within a tag.\n")
    A("| tag | first book session | last | sessions | tracker (`" + GATE_BENCH + "`) ann |")
    A("|---|---|---|---:|---:|")
    for tag, t in payload["tags"].items():
        atx = t["after_tax"].get(PRIMARY_CELL, {})
        tr = atx.get("trackers", {}).get(GATE_BENCH, {})
        A(f"| {tag} | {atx.get('first_session', '')} | {atx.get('last_session', '')} | "
          f"{atx.get('n_days', '')} | {_pp(tr.get('pre_tax_ann_return'))} |")

    A("\n## Net-of-cost books (every tag, every cell)\n")
    A("| tag | cell | sessions | ann net | ann gross | ex vs nsei_tr_proxy | ex vs NIFTYBEES(cc) "
      "| CI95 ann (proxy) | IR | MDD | turnover/yr | cost drag/yr | one-way bps | null pctile |")
    A("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tag, t in payload["tags"].items():
        nullp = t["null_percentile"]
        for cell in ALL_CELLS:
            m = t["books"].get(cell)
            if not m:
                continue
            g_ = m[GATE_BENCH]
            d_ = m[DESC_BENCH]
            ci = g_["excess_ci95_annualised"]
            A(f"| {tag} | {cell} | {g_['n_days']} | {_pp(g_['ann_return_net'])} | "
              f"{_pp(g_['ann_return_gross'])} | **{_pp(g_['ann_excess_net'])}** | "
              f"{_pp(d_['ann_excess_net'])} | [{_pp(ci[0], 1)}, {_pp(ci[1], 1)}] | "
              f"{g_['information_ratio']:+.2f} | {_pp(g_['max_drawdown'], 1)} | "
              f"{g_['turnover_ann']:.2f} | {g_['cost_drag_ann_pct']:.2f} | "
              f"{g_['one_way_cost_bps']:.2f} | "
              f"{(f'{nullp:.1f}' if cell == PRIMARY_CELL and nullp is not None else '')} |")
    A("\n_Excesses are in percentage points, annualised on 252 sessions. The null percentile is "
      "the rank of the observed k10_d1 annualised net excess among the random-score books at "
      "identical settings (N3), against nsei_tr_proxy._\n")

    A("\n## Benchmark books on the same sessions (equal-weight PIT N50, constant score)\n")
    A("| tag | leg | ann net | vs nsei_tr_proxy | vs NIFTYBEES(cc) |")
    A("|---|---|---:|---:|---:|")
    for tag, t in payload["tags"].items():
        for leg, mm in t["benchmark_books"].items():
            A(f"| {tag} | {leg} | {_pp(mm[DESC_BENCH]['ann_return_net'])} | "
              f"{_pp(mm[GATE_BENCH]['ann_excess_net'])} | "
              f"{_pp(mm[DESC_BENCH]['ann_excess_net'])} |")

    A("\n## After tax\n")
    A("Book taxed under each regime with the year's liability withdrawn at FY end; tracker taxed "
      "LTCG-on-exit (12.5% + cess above the Rs 1.25 lakh s.112A exemption) and, in the last two "
      "columns, not taxed at all because it is never sold.\n")
    A("| tag | cell | regime | book after-tax ann | tax paid Rs | vs proxy (LTCG-on-exit) | "
      "vs NIFTYBEES (LTCG-on-exit) | vs proxy (unrealised) | vs NIFTYBEES (unrealised) |")
    A("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for tag, t in payload["tags"].items():
        for cell, atx in t["after_tax"].items():
            for reg in REGIMES:
                r = atx["regimes"][reg]
                e = atx["after_tax_excess"][reg]
                A(f"| {tag} | {cell} | {reg} | {_pp(r['after_tax_ann_return'])} | "
                  f"{r['total_tax_rs']:,.0f} | **{_pp(e[GATE_BENCH]['on_exit'])}** | "
                  f"{_pp(e[DESC_BENCH]['on_exit'])} | {_pp(e[GATE_BENCH]['unrealised'])} | "
                  f"{_pp(e[DESC_BENCH]['unrealised'])} |")
    A("")
    for tag, t in payload["tags"].items():
        atx = t["after_tax"].get(PRIMARY_CELL)
        if not atx:
            continue
        tr = atx["trackers"][GATE_BENCH]
        A(f"* `{tag}` {PRIMARY_CELL}: tracker `{GATE_BENCH}` pre-tax ann "
          f"{_pp(tr['pre_tax_ann_return'])} pp, after-tax-on-exit "
          f"{_pp(tr['after_tax_ann_return_on_exit'])} pp (tax on exit Rs "
          f"{tr['tax_paid_on_exit_rs']:,.0f}, held {tr['held_days']} days = {tr['term']}-term); "
          f"book pre-tax ann {_pp(atx['pre_tax_ann_return'])} pp.")

    A("\n## Calendar years (net excess, percentage points)\n")
    A("| tag | cell | year | sessions | book | proxy | excess (cum) | excess (ann) | NIFTYBEES "
      "excess (cum) |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for tag, t in payload["tags"].items():
        for cell, rows in t["years"].items():
            for r in rows:
                mark = " *" if r["partial"] else ""
                A(f"| {tag} | {cell} | {r['year']}{mark} | {r['n_days']} | "
                  f"{_pp(r['book_return_cum'])} | {_pp(r[f'bench_{GATE_BENCH}_cum'])} | "
                  f"**{_pp(r[f'excess_{GATE_BENCH}_cum'])}** | "
                  f"{_pp(r[f'excess_{GATE_BENCH}_ann'])} | "
                  f"{_pp(r[f'excess_{DESC_BENCH}_cum'])} |")
    A("\n_`*` = fewer than 200 sessions, i.e. a partial year. 2023 (where present) is reported "
      "and NOT counted by M3; the counted years are 2024, 2025 and 2026-to-date. The `excess "
      "(ann)` column annualises a partial year and is meaningless on a 36-session stub (`free` "
      "2023 reads +64 pp): the gate uses `excess (cum)`, which is what the year actually paid._"
      "\n")

    A("\n## Gates for every other tag\n")
    A("| tag | M2 | M3 | net excess vs proxy (k10_d1) | after-tax excess stcg | "
      "after-tax excess business_31 |")
    A("|---|---|---|---:|---:|---:|")
    for tag, t in payload["tags"].items():
        m2, m3 = t["m2"], t["m3"]
        ex = m2["clauses"][0]["value"]
        s = m2["clauses"][1]["value"]
        b = m2["clauses"][2]["value"]
        A(f"| {tag} | {_verdict(m2['verdict'])} | {_verdict(m3['verdict'])} | {_pp(ex)} | "
          f"{_pp(s)} | {_pp(b)} |")
    # The "different courses" warning names the two tags actually furthest apart in this run's
    # tracker return, computed here, instead of repeating a pair that was true when the prose was
    # written. If a later run's tags all share a window the sentence says so rather than inventing
    # a spread.
    trk = {tg: t["after_tax"].get(PRIMARY_CELL, {}).get("trackers", {})
                 .get(GATE_BENCH, {}).get("pre_tax_ann_return")
           for tg, t in payload["tags"].items()}
    trk = {k: v for k, v in trk.items() if v is not None}
    if len(trk) >= 2:
        lo = min(trk, key=trk.get)
        hi = max(trk, key=trk.get)
        w_lo = payload["tags"][lo]["after_tax"][PRIMARY_CELL]
        w_hi = payload["tags"][hi]["after_tax"][PRIMARY_CELL]
        A(f"\n_Cross-tag rows are NOT a horse race: see the Windows table -- `{lo}` runs on "
          f"{w_lo['n_days']} sessions from {w_lo['first_session']} against a tracker that "
          f"returned {_pp(trk[lo])} pp/yr over them, while `{hi}` runs on {w_hi['n_days']} "
          f"sessions from {w_hi['first_session']} against a tracker that returned "
          f"{_pp(trk[hi])} pp/yr -- a spread of {_pp(trk[hi] - trk[lo])} pp/yr in the course "
          f"itself. Much of the gap between two tags' excesses is that course, not the horse._")
    else:
        A("\n_Only one tag has a usable primary book, so there is no cross-tag comparison to "
          "warn about._")
    A("\n_M2 and M3 are pre-registered for the primary arm only; the same clauses are evaluated "
      "for every tag so the reader can see whether the primary was the lucky cell or the modal "
      "one. Clauses `b_post_ge_pre_ic` and `d_fingerprint` come from the probe's primary cell "
      "and are therefore identical across tags._\n")
    return "\n".join(L)


# ------------------------------------------------------------------ driver


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=N50)
    ap.add_argument("--tags", default=None,
                    help="comma list; default = every bt_<tag>_metrics.json in --out-dir")
    ap.add_argument("--decision-tag", default="A0_primary",
                    help="the tag whose M2/M3 verdict is the plan's decision")
    ap.add_argument("--probe", default=None, help="probe_n50.json (M3 clauses b and d)")
    ap.add_argument("--capital", type=float, default=1e6)
    ap.add_argument("--cost-preset", default="zerodha", choices=list(PRESETS))
    ap.add_argument("--cells", default=",".join((PRIMARY_CELL, SECONDARY_CELL)),
                    help="cells to run the tax layer on (the net-of-cost table covers all four)")
    args = ap.parse_args()

    t0 = time.time()
    cm = PRESETS[args.cost_preset]
    probe_path = args.probe or os.path.join(args.out_dir, "probe_n50.json")
    probe = None
    if os.path.exists(probe_path):
        with open(probe_path) as f:
            probe = json.load(f)
    else:
        print(f"WARNING: {probe_path} not found -- M3 clauses b and d will be UNDECIDED",
              flush=True)

    tags = ([t for t in args.tags.split(",") if t] if args.tags
            else discover_tags(args.out_dir))
    tax_cells = [c for c in args.cells.split(",") if c]
    out_tags: dict[str, dict] = {}
    for tag in tags:
        d = load_tag(args.out_dir, tag)
        if d["bench"] is None:
            print(f"skipping {tag}: no bt_{tag}_bench_daily.parquet (re-run n50_backtest.py)",
                  flush=True)
            continue
        met = d["metrics"]
        atx, years = {}, {}
        for cell in tax_cells:
            if cell not in d["cells"]:
                continue
            c = d["cells"][cell]
            atx[cell] = after_tax_block(c["daily"], c["trades"], d["bench"], args.capital, cm)
            years[cell] = calendar_year_table(c["daily"], d["bench"])
        if PRIMARY_CELL not in atx:
            print(f"skipping {tag}: no {PRIMARY_CELL} book", flush=True)
            continue
        nullp = (met.get("null_random_scores", {}).get("by_benchmark", {})
                 .get(GATE_BENCH, {}).get("percentile_of_observed"))
        out_tags[tag] = {
            "window": met["universe"],
            "books": met["books"],
            "benchmark_books": met.get("benchmark_books", {}),
            "null_percentile": nullp,
            "null": met.get("null_random_scores", {}),
            "after_tax": atx,
            "years": years,
            "m2": m2_gates(met, atx[PRIMARY_CELL], tag,
                           accrual_sensitivity(d["cells"][PRIMARY_CELL]["daily"], d["bench"])),
            "m3": m3_gates(met, years[PRIMARY_CELL], probe, tag),
        }
        m2, m3 = out_tags[tag]["m2"], out_tags[tag]["m3"]
        print(f"{tag:16s} M2 {_verdict(m2['verdict']):9s} M3 {_verdict(m3['verdict']):9s} "
              f"net excess vs {GATE_BENCH} {_pp(m2['clauses'][0]['value'])} pp   "
              f"after-tax stcg {_pp(m2['clauses'][1]['value'])} pp   "
              f"business_31 {_pp(m2['clauses'][2]['value'])} pp", flush=True)

    if not out_tags:
        raise SystemExit("no tag had a usable backtest; run n50_backtest.py first")
    decision = args.decision_tag if args.decision_tag in out_tags else sorted(out_tags)[0]
    payload = {
        "meta": {
            "written_at": pd.Timestamp.now("UTC").isoformat(),
            "capital_rs": float(args.capital), "cost_preset": args.cost_preset,
            "gate_benchmark": GATE_BENCH, "description_benchmark": DESC_BENCH,
            "tr_accrual_annual": TR_ACCRUAL_ANNUAL,
            "tr_accrual_per_session": TR_ACCRUAL_PER_SESSION,
            "primary_cell": PRIMARY_CELL, "secondary_cell": SECONDARY_CELL,
            "tax_cells": tax_cells, "regimes": list(REGIMES),
            "m2_excess_bar": M2_EXCESS_BAR, "fingerprint_bar": FINGERPRINT_BAR,
            "probe": probe_path if probe else None,
            "n_tags": len(out_tags),
        },
        "decision_tag": decision,
        "tags": out_tags,
    }
    jpath = os.path.join(args.out_dir, "money_n50.json")
    with open(jpath, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    mpath = os.path.join(args.out_dir, "money_n50.md")
    with open(mpath, "w") as f:
        f.write(render_md(payload))
    sidecar(os.path.join(args.out_dir, "money_n50_meta.json"), vars(args),
            extra={"n_tags": len(out_tags), "decision_tag": decision,
                   "m2": _verdict(out_tags[decision]["m2"]["verdict"]),
                   "m3": _verdict(out_tags[decision]["m3"]["verdict"]),
                   "elapsed_s": round(time.time() - t0, 1)})
    print(f"\nwrote {jpath}\n      {mpath}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
