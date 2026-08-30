"""Stage A build for the NIFTY 50 money test: the N50 dataset, the TRI benchmark, the gates.

Three products, in the order the plan needs them (NIFTY50_PLAN sections 3 and F8):

  1. ``bench/out/nse/n50/dataset_n50.parquet`` -- ``bench/out/nse/ml/dataset.parquet`` with one
     extra column ``is_n50``. Same rows, same order; nothing is dropped, because the head is
     fitted on the whole PIT NIFTY 200 cross-section and only GRADED on the 50 (F3).
  2. ``data/nse/bench_n50.parquet`` -- the benchmark the user's bar is actually stated on.
     ``data/nse/bench.parquet`` carries ``^NSEI``, a PRICE index; the bar is Total Return, so
     the tracker is NIFTYBEES (dividends reinvested, net of TER -- the buyable instrument).
  3. ``bench/out/nse/n50/n50_summary.json`` -- names per date, the member-dates the 90-bar
     lookback rule makes ineligible, the benchmark returns, and the validation gate.

The gate refuses to hand the N50 dataset onward unless it reproduces the record's
``reversal_1d`` IC at h = 10 on the 200 (+0.0157, tolerance 0.0015), exactly as
``build_dataset.py`` does. A subset that cannot reproduce its own superset is not a subset.
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
from bench.nse.common import sidecar, utc_now  # noqa: E402
from bench.nse.mlcore import DATA_NSE, OUT_NSE, daily_spearman, summarise  # noqa: E402
from bench.nse.n50 import (evaluate, load_n50_membership, n50_mask,  # noqa: E402
                           power_statement, summary_json)

ML = os.path.join(OUT_NSE, "ml")
N50 = os.path.join(OUT_NSE, "n50")
RESEARCH = os.path.join(DATA_NSE, "research")
BENCH_N50 = os.path.join(DATA_NSE, "bench_n50.parquet")

#: The record's reversal_1d IC at h=10 on the PIT NIFTY 200 (controls_pit.json baseline_ic),
#: reproduced by build_dataset.py's own gate. Same number, same tolerance, so a drift in the
#: label pipeline fails here too instead of only upstream.
REV1D_H10_200 = 0.0157
REV1D_TOL = 0.0015


# --------------------------------------------------------------------- benchmark (F8)


def fetch_bench_n50(start="2023-06-01", end=None, out=BENCH_N50,
                    log_dir=RESEARCH) -> pd.DataFrame:
    """Fetch ^NSEI and NIFTYBEES.NS and write the dividend-adjusted tracker series.

    ``auto_adjust=False, actions=True`` so the raw Close and the dividend-adjusted Adj Close
    both survive -- with auto_adjust=True yfinance overwrites Close and the price/total-return
    distinction the user's bar rests on is silently lost.

    The book trades open(t+1) -> open(t+2), so the benchmark leg must be open-to-open too.
    yfinance publishes no adjusted OPEN, so it is approximated as ``open * adj_close/close``:
    the same day's split/dividend factor applied to the same day's open. That is exact for
    splits and correct to within one session's intraday move for a dividend, which on a ~1.2%
    annual yield is worth well under a basis point a year.
    """
    import yfinance as yf

    end = end or (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    t0 = time.time()
    raw = yf.download(["NIFTYBEES.NS", "^NSEI"], start=start, end=end, auto_adjust=False,
                      actions=True, progress=False, group_by="ticker")
    if raw is None or not len(raw):
        raise RuntimeError("yfinance returned nothing for NIFTYBEES.NS / ^NSEI")

    nb = raw["NIFTYBEES.NS"].copy()
    ns = raw["^NSEI"].copy()
    df = pd.DataFrame(index=raw.index)
    df["nsei"] = ns["Close"]
    df["nsei_open"] = ns["Open"]
    df["niftybees_close"] = nb["Close"]
    df["niftybees_open"] = nb["Open"]
    factor = nb["Adj Close"] / nb["Close"]
    df["niftybees_adj"] = nb["Adj Close"]
    df["niftybees_adj_open"] = nb["Open"] * factor
    df["niftybees_div"] = nb["Dividends"]
    df = df[df["niftybees_close"].notna() | df["nsei"].notna()]
    df = df.reset_index().rename(columns={"Date": "date", "index": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).astype("datetime64[ns]")
    df = df.sort_values("date").reset_index(drop=True)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_parquet(out, index=False)

    os.makedirs(log_dir, exist_ok=True)
    n_div = int((df["niftybees_div"].fillna(0) > 0).sum())
    with open(os.path.join(log_dir, "bench_n50_fetch.log"), "a") as f:
        f.write(f"{utc_now()}  yfinance NIFTYBEES.NS,^NSEI  start={start} end={end}  "
                f"rows={len(df)}  {df['date'].min().date()}->{df['date'].max().date()}  "
                f"dividend_days={n_div}  nsei_nan={int(df['nsei'].isna().sum())}  "
                f"niftybees_nan={int(df['niftybees_close'].isna().sum())}  "
                f"{time.time() - t0:.1f}s -> {out}\n")
    return df


def _window_return(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.iloc[-1] / s.iloc[0] - 1.0) if len(s) > 1 else float("nan")


def benchmark_summary(df: pd.DataFrame, tri_path: str | None = None) -> dict:
    """Calendar-year and full-window returns of the tracker and the price index, + TRI gap.

    The tracker-minus-price-index difference IS the dividend the price index throws away, so
    printing both is the cheapest possible check that the total-return series is doing its job:
    NIFTYBEES adj must beat ^NSEI by roughly the index yield (~1.1-1.5%/yr) minus the 0.04% TER.
    """
    d = df.set_index(pd.DatetimeIndex(df["date"]))
    out = {"start": str(d["date"].min().date()), "end": str(d["date"].max().date()),
           "n_rows": int(len(d)), "by_year": {}, "full_window": {}}
    cols = ["niftybees_adj", "niftybees_close", "nsei"]
    for c in cols:
        out["full_window"][c] = _window_return(d[c])
    out["full_window"]["dividend_carry_pct"] = (
        (out["full_window"]["niftybees_adj"] - out["full_window"]["niftybees_close"]) * 100)
    # Nippon India Nifty 50 BeES has made no distribution since 2023-06, so yfinance's Adj
    # Close is byte-identical to Close and the "adjustment" is a no-op. That is not a fetch
    # bug: the ETF accumulates its constituents' dividends into NAV, so its PRICE series is
    # already total return net of the 0.04% TER. The evidence is tracker_minus_price_pp
    # below -- ~1.2 pp/yr over ^NSEI, which is the index dividend yield. Recorded explicitly
    # because a reader who sees adj == close would otherwise conclude the TRI leg is missing.
    good = d["niftybees_close"].notna() & d["niftybees_adj"].notna()
    out["tracker_is_accumulating"] = {
        "dividend_days": int((d.get("niftybees_div", pd.Series(dtype=float)).fillna(0) > 0).sum()),
        "adj_equals_close_share": float((d.loc[good, "niftybees_adj"]
                                         == d.loc[good, "niftybees_close"]).mean()),
        "nan_rows": {c: int(d[c].isna().sum()) for c in
                     ["nsei", "nsei_open", "niftybees_close", "niftybees_adj", "niftybees_adj_open"]},
    }
    for y in sorted(d.index.year.unique()):
        sub = d[d.index.year == y]
        row = {c: _window_return(sub[c]) for c in cols}
        row["n_rows"] = int(len(sub))
        row["tracker_minus_price_pp"] = (row["niftybees_adj"] - row["nsei"]) * 100
        out["by_year"][str(int(y))] = row

    tri_path = tri_path or os.path.join(RESEARCH, "nifty50_tri.json")
    if os.path.exists(tri_path):
        tri = json.load(open(tri_path))
        levels = tri.get("tri_levels", [])
        comp = []
        for a, b in zip(levels, levels[1:]):
            da, db = pd.Timestamp(a["date"]), pd.Timestamp(b["date"])
            seg = d[(d.index >= da) & (d.index <= db)]
            if len(seg) < 2:
                continue
            yrs = max((db - da).days / 365.25, 1e-9)
            tri_r = b["nifty50_tri"] / a["nifty50_tri"] - 1.0
            trk = _window_return(seg["niftybees_adj"])
            pri_r = b["nifty50_pri"] / a["nifty50_pri"] - 1.0
            comp.append({
                "from": a["date"], "to": b["date"], "years": yrs,
                "tri_return": tri_r, "tracker_return": trk, "research_pri_return": pri_r,
                "gap_pp": (trk - tri_r) * 100, "gap_pp_per_year": (trk - tri_r) * 100 / yrs,
                "tri_is_estimate": bool(a.get("tri_is_estimate") or b.get("tri_is_estimate")),
            })
        exact = [c for c in comp if not c["tri_is_estimate"]]
        out["tri_cross_check"] = {
            "source": os.path.relpath(tri_path, ROOT), "segments": comp,
            "note": ("Segments whose endpoints are flagged tri_is_estimate carry the research "
                     "agent's back-discounted levels, not published TRI closes; a gap there is "
                     "evidence about the estimate, not about the tracker."),
            "max_abs_gap_pp_per_year": (max(abs(c["gap_pp_per_year"]) for c in comp)
                                        if comp else float("nan")),
            # Stage A's gate is "within 0.5 pp/yr of the researched TRI levels". It can only be
            # ADJUDICATED on a segment whose endpoints are published closes; on an estimated
            # level the gap measures the estimate. The independent check that does not need the
            # research file at all is tracker_minus_price_pp above: NIFTYBEES must beat ^NSEI by
            # about the index dividend yield (1.1-1.5%/yr) net of the 0.04% TER, and it does.
            "gate_0p5pp_per_year": {
                "n_segments": len(comp), "n_segments_on_published_levels": len(exact),
                "verdict": ("not adjudicable -- every segment has at least one estimated TRI "
                            "endpoint" if not exact else
                            ("pass" if max(abs(c["gap_pp_per_year"]) for c in exact) <= 0.5
                             else "fail")),
                "max_abs_gap_pp_per_year_published_only": (
                    max(abs(c["gap_pp_per_year"]) for c in exact) if exact else None),
            },
        }
    return out


# --------------------------------------------------------------------- dataset


def build_dataset_n50(ds: pd.DataFrame, mem: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add ``is_n50`` and count what the 90-bar lookback rule costs the index.

    A PIT member with no row in the dataset on one of its member dates is not missing data:
    it is a name that has not yet accumulated the 90 sessions the eligibility rule demands
    (recent listings such as JIOFIN and ETERNAL). The plan says to expect this and log it
    rather than paper over it, so the symbols and their date counts are reported.
    """
    out = ds.copy()
    out["is_n50"] = n50_mask(out, mem)

    lo, hi = out["date"].min(), out["date"].max()
    mem_w = mem[(mem["date"] >= lo) & (mem["date"] <= hi)]
    have = pd.MultiIndex.from_arrays([out["date"], out["symbol"].astype(str)])
    want = pd.MultiIndex.from_arrays([mem_w["date"], mem_w["symbol"]])
    miss = mem_w[~want.isin(have)]
    by_sym = miss.groupby("symbol")["date"].agg(["size", "min", "max"])

    per_date = out.loc[out["is_n50"]].groupby("date").size()
    mem_per_date = mem_w.groupby("date").size()
    info = {
        "n_rows": int(len(out)), "n_dates": int(out["date"].nunique()),
        "n_n50_rows": int(out["is_n50"].sum()),
        "n_n50_dates": int(per_date.size),
        "eligible_n50_per_date": {
            "min": int(per_date.min()), "median": float(per_date.median()),
            "max": int(per_date.max()), "mean": float(per_date.mean()),
            "dates_below_44": int((per_date < 44).sum()),
        },
        "members_per_date": {"min": int(mem_per_date.min()), "max": int(mem_per_date.max()),
                             "median": float(mem_per_date.median())},
        "ineligible_member_dates": {
            "count": int(len(miss)),
            "by_symbol": {s: {"n_dates": int(r["size"]), "first": str(r["min"].date()),
                              "last": str(r["max"].date())}
                          for s, r in by_sym.iterrows()},
            "rule": "90-bar lookback / max-stale eligibility, identical to signals.build_batch",
        },
    }
    return out, info


def validate(ds: pd.DataFrame, mask: np.ndarray) -> dict:
    """Reproduce reversal_1d @ h=10 on the 200, then report the same statistic on the 50.

    The 200 number is a gate (it must land within 0.0015 of +0.0157). The 50 number is NOT a
    gate -- it is the first honest look at how much precision the subset costs, reported at
    NW lags 10, 12 and 20 because the lag choice is worth more than the point estimate here.
    """
    d = ds["date"].to_numpy()
    y = ds["y_h10"].to_numpy(dtype=float)
    sc = ds["reversal_1d"].to_numpy(dtype=float)

    ics200 = daily_spearman(sc, y, d, min_names=20)
    got = float(ics200.mean())
    st200 = summarise(ics200, 10, lags=(None, 12, 20))
    ok = abs(got - REV1D_H10_200) < REV1D_TOL

    ev = evaluate(sc, ds, 10, mask, ks=(10, 15))
    return {
        "reversal_1d_h10_n200": {
            "got": got, "want": REV1D_H10_200, "delta": abs(got - REV1D_H10_200),
            "tolerance": REV1D_TOL, "pass": bool(ok), **st200},
        "reversal_1d_h10_n50": summary_json(ev),
        "validated": bool(ok),
    }


# --------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.path.join(ML, "dataset.parquet"))
    ap.add_argument("--membership", default=None, help="default: data/nse/membership_pit_n50.parquet")
    ap.add_argument("--out-dir", default=N50)
    ap.add_argument("--bench-out", default=BENCH_N50)
    ap.add_argument("--bench-start", default="2023-06-01")
    ap.add_argument("--bench-end", default=None, help="default: tomorrow (yfinance end is exclusive)")
    ap.add_argument("--fetch-bench", choices=["auto", "always", "never"], default="auto",
                    help="auto = fetch only if bench_n50.parquet is missing")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    print(f"reading {args.dataset} ...", flush=True)
    ds = pd.read_parquet(args.dataset)
    ds["date"] = pd.to_datetime(ds["date"])
    mem = load_n50_membership(args.membership)
    print(f"  dataset {ds.shape}  membership {mem.shape}", flush=True)

    out, info = build_dataset_n50(ds, mem)
    dst = os.path.join(args.out_dir, "dataset_n50.parquet")
    out.to_parquet(dst, index=False)
    e = info["eligible_n50_per_date"]
    print(f"wrote {dst}  shape={out.shape}  is_n50 rows={info['n_n50_rows']}", flush=True)
    print(f"  eligible N50 per date: min {e['min']}  median {e['median']}  max {e['max']}"
          f"  (dates < 44: {e['dates_below_44']})", flush=True)
    print(f"  member-dates ineligible under the 90-bar rule: "
          f"{info['ineligible_member_dates']['count']}", flush=True)
    for s, r in info["ineligible_member_dates"]["by_symbol"].items():
        print(f"    {s:12s} {r['n_dates']:4d} dates  {r['first']} -> {r['last']}", flush=True)

    # ---- benchmark (F8)
    need = (args.fetch_bench == "always"
            or (args.fetch_bench == "auto" and not os.path.exists(args.bench_out)))
    if need:
        print("fetching NIFTYBEES.NS + ^NSEI ...", flush=True)
        bdf = fetch_bench_n50(args.bench_start, args.bench_end, args.bench_out)
        print(f"  wrote {args.bench_out}  rows={len(bdf)}", flush=True)
    bench = {}
    if os.path.exists(args.bench_out):
        bdf = pd.read_parquet(args.bench_out)
        bench = benchmark_summary(bdf)
        fw = bench["full_window"]
        print(f"benchmark {bench['start']} -> {bench['end']}  "
              f"NIFTYBEES adj {fw['niftybees_adj'] * 100:+.2f}%  "
              f"price {fw['niftybees_close'] * 100:+.2f}%  ^NSEI {fw['nsei'] * 100:+.2f}%",
              flush=True)
        for y, r in bench["by_year"].items():
            print(f"  {y}: NIFTYBEES adj {r['niftybees_adj'] * 100:+.2f}%  "
                  f"^NSEI {r['nsei'] * 100:+.2f}%  diff {r['tracker_minus_price_pp']:+.2f} pp",
                  flush=True)
        for c in bench.get("tri_cross_check", {}).get("segments", []):
            print(f"  TRI {c['from']} -> {c['to']}: tracker {c['tracker_return'] * 100:+.2f}%  "
                  f"TRI {c['tri_return'] * 100:+.2f}%  gap {c['gap_pp_per_year']:+.2f} pp/yr"
                  f"{'  (TRI estimated)' if c['tri_is_estimate'] else ''}", flush=True)
    else:
        print(f"NOTE no benchmark at {args.bench_out}; run with --fetch-bench always", flush=True)

    # ---- power, stated before the validation number
    pw = power_statement(int(e["median"]), info["n_n50_dates"])
    print(f"\npower at N={pw['n_names']} T={pw['n_dates']}: naive SE {pw['se_naive']:.4f}  "
          f"lag-20 NW SE {pw['se_nw_lag20']:.4f} (x{pw['ar1_inflation']})  "
          f"t=3 bar {pw['bar_t3_nw_lag20']:+.4f}", flush=True)

    print("\n=== VALIDATION: reversal_1d IC @ h=10 ===", flush=True)
    val = validate(out, out["is_n50"].to_numpy())
    v2, v5 = val["reversal_1d_h10_n200"], val["reversal_1d_h10_n50"]["ic_summary"]
    print(f"  n200  got {v2['got']:+.4f}  want {v2['want']:+.4f}  delta {v2['delta']:.4f}  "
          f"{'PASS' if v2['pass'] else 'FAIL'}", flush=True)
    print(f"        NW lag10 SE {v2['se_nw_lag10']:.4f} t {v2['t_nw_lag10']:+.2f} | "
          f"lag12 {v2['se_nw_lag12']:.4f} t {v2['t_nw_lag12']:+.2f} | "
          f"lag20 {v2['se_nw_lag20']:.4f} t {v2['t_nw_lag20']:+.2f}", flush=True)
    print(f"  n50   IC {v5['mean_ic']:+.4f}  n_dates {v5['n_dates']}", flush=True)
    print(f"        NW lag10 SE {v5['se_nw_lag10']:.4f} t {v5['t_nw_lag10']:+.2f} | "
          f"lag12 {v5['se_nw_lag12']:.4f} t {v5['t_nw_lag12']:+.2f} | "
          f"lag20 {v5['se_nw_lag20']:.4f} t {v5['t_nw_lag20']:+.2f}", flush=True)
    sp = val["reversal_1d_h10_n50"]["spread_summary"]["10"]
    print(f"        top-10 spread {sp['mean_ic']:+.5f}/session  t_nw20 {sp['t_nw_lag20']:+.2f}",
          flush=True)

    summary = {"dataset": info, "power": pw, "validation": val, "benchmark": bench,
               "written_at": utc_now(), "elapsed_s": round(time.time() - t0, 1)}
    sp_path = os.path.join(args.out_dir, "n50_summary.json")
    with open(sp_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    sidecar(os.path.join(args.out_dir, "n50_summary_meta.json"), vars(args),
            extra={"n_rows": info["n_rows"], "n_n50_rows": info["n_n50_rows"],
                   "validated": val["validated"]})
    print(f"\nwrote {sp_path}  ({time.time() - t0:.0f}s)", flush=True)

    if not val["validated"]:
        print("VALIDATION FAILED -- the N50 subset does not reproduce its own superset.",
              flush=True)
        sys.exit(1)
    print("VALIDATION PASSED.", flush=True)


if __name__ == "__main__":
    main()
