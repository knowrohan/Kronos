"""Point-in-time NIFTY 50 evaluation layer: membership mask, per-date metrics, power bar.

The head is FITTED on the wide PIT NIFTY 200 cross-section and GRADED on the PIT NIFTY 50
subset of the same dates (NIFTY50_PLAN F3): a 50-name cross-section cannot support a 54-
regressor per-date ridge, but a 50-name *evaluation* is exactly what the tradable book sees.
So every function here masks scores that already exist; nothing here ever touches a fit.

Everything is reported at Newey-West lags {h, 12, 20} with a block-20 bootstrap because a
fitted score is a trailing mean of past coefficients and its per-date IC series is
autocorrelated far beyond h lags (NIFTY50_PLAN F4); the gate lag is 20.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from bench.nse.mlcore import (DATA_NSE, daily_spearman, daily_topk_spread,  # noqa: F401
                             subset_ic, summarise)

N50_PIT = os.path.join(DATA_NSE, "membership_pit_n50.parquet")

#: Evaluation cross-sections are ~50 names, so the 200-panel's min_names=20 would admit
#: dates where a third of the index is missing. 30 is the plan's floor (section 4).
MIN_NAMES = 30

#: The two book sizes the money test runs (NIFTY50_PLAN section 6.1): k=10 primary, k=15
#: secondary. The spread series is reported for both so the gate is not read off one k.
SPREAD_KS = (10, 15)


def load_n50_membership(path: str | None = None) -> pd.DataFrame:
    """PIT NIFTY 50 membership as (date, symbol), one row per session per member.

    Read from the artifact `reconstruct_membership.py --index nifty50` writes rather than
    rebuilt here, so the backtest, the probe and the tests all grade against the same file.
    """
    df = pd.read_parquet(path or N50_PIT)
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    return df[["date", "symbol"]].sort_values(["date", "symbol"]).reset_index(drop=True)


def membership_sets(mem: pd.DataFrame | None = None) -> dict:
    """{session -> frozenset(symbols)}. The form the backtester's per-date mask needs."""
    mem = load_n50_membership() if mem is None else mem
    return {d: frozenset(g) for d, g in mem.groupby("date")["symbol"]}


def n50_mask(ds: pd.DataFrame, mem: pd.DataFrame | None = None) -> np.ndarray:
    """Boolean row mask: this (date, symbol) is a PIT NIFTY 50 member on this session.

    `ds` rows are already eligibility-filtered (the 90-bar lookback rule), so the mask is
    the intersection "eligible AND in the index that day" -- the tradable universe. A member
    that is not in `ds` on a date is ineligible, not absent; n50_build.py counts those.
    """
    mem = load_n50_membership() if mem is None else mem
    key = pd.MultiIndex.from_arrays([pd.to_datetime(ds["date"]), ds["symbol"].astype(str)])
    ref = pd.MultiIndex.from_arrays([mem["date"], mem["symbol"]])
    return key.isin(ref)


# ------------------------------------------------------------------ power, stated first


def power_statement(n_names: int, n_dates: int, ic_sd_per_date: float | None = None,
                    ar1_inflation: float = 1.7) -> dict:
    """What a t = 3 result would have to be at this N and T, computed BEFORE the result.

    The per-date Spearman IC of a null score over N names has sd ~ 1/sqrt(N-1); at N = 50
    that is ~0.14, so the naive SE over T dates is ~0.14/sqrt(T). The Newey-West lag-20 SE
    of a persistent fitted score runs 1.5-2x the naive one, and `ar1_inflation` (default 1.7)
    is that multiplier stated as an argument rather than buried in prose -- pass the realised
    ratio once a series exists. The point of printing this first is that the honest t = 3 bar
    on 50 names is ABOVE the IC the record measured on 200 names (NIFTY50_PLAN section 4).
    """
    sd = float(1.0 / np.sqrt(max(n_names - 1, 1))) if ic_sd_per_date is None else float(ic_sd_per_date)
    se_naive = sd / np.sqrt(max(n_dates, 1))
    se_nw = se_naive * float(ar1_inflation)
    return {
        "n_names": int(n_names), "n_dates": int(n_dates),
        "ic_sd_per_date": sd, "ic_sd_source": "1/sqrt(N-1)" if ic_sd_per_date is None else "measured",
        "ar1_inflation": float(ar1_inflation),
        "se_naive": float(se_naive), "se_nw_lag20": float(se_nw),
        "bar_t3_naive": float(3.0 * se_naive), "bar_t3_nw_lag20": float(3.0 * se_nw),
        "bar_t2_nw_lag20": float(2.0 * se_nw),
    }


# ------------------------------------------------------------------ per-date metrics


def _by_year(s: pd.Series, horizon: int, seed: int = 0) -> dict:
    """summarise() per calendar year. A three-year window is three ~250-date samples and
    M3 gates on 2 of 3 being positive, so the split is a gate input, not decoration."""
    if not len(s):
        return {}
    idx = pd.DatetimeIndex(s.index)
    out = {}
    for y in sorted(idx.year.unique()):
        sub = s[idx.year == y]
        if len(sub) < 2:
            continue
        out[str(int(y))] = summarise(sub, horizon, seed=seed)
    return out


def evaluate(scores: np.ndarray, ds: pd.DataFrame, h: int, mask: np.ndarray | None = None,
             *, min_names: int = MIN_NAMES, ks=SPREAD_KS, seed: int = 0,
             ar1_inflation: float = 1.7) -> dict:
    """Grade a score vector on the PIT NIFTY 50 subset of `ds` at horizon h.

    Returns the per-date IC series and the per-date top-k spread series (mean forward return
    of the k best-scored names minus the mean of every eligible member that day) plus their
    lag ladders and calendar-year splits. Both are needed: IC is a whole-cross-section rank
    correlation and can be positive while the top of the book is not, and the book only ever
    captures the spread (NIFTY50_PLAN section 4).

    `mask` defaults to the PIT membership mask; pass one to grade a different subset (the
    200-panel control passes an all-True mask).
    """
    scores = np.asarray(scores, dtype=float)
    y = ds[f"y_h{h}"].to_numpy(dtype=float)
    d = ds["date"].to_numpy()
    m = np.ones(len(ds), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if len(scores) != len(ds):
        raise ValueError(f"scores has {len(scores)} rows, ds has {len(ds)}")

    ic = subset_ic(scores, y, d, m, min_names=min_names)
    spread = {int(k): daily_topk_spread(scores[m], y[m], d[m], int(k), min_names=min_names)
              for k in ks}
    n_per_date = pd.Series(d[m]).value_counts().sort_index()
    ic_sum = summarise(ic, h, seed=seed)
    n_med = int(n_per_date.median()) if len(n_per_date) else 0
    ic_sd = float(ic.std(ddof=1)) if len(ic) > 1 else None
    # `power` and `power_asymptotic` both price the lag-20 SE at the ASSUMED `ar1_inflation`,
    # which is a prior, not a measurement. Once the series exists the ratio is observable --
    # se_nw_lag20 / se_naive off this very series -- and quoting an assumed bar next to a
    # measured SE in the same JSON invites a reader to compare two numbers that were made
    # differently. `power_realised` is the same statement at the ratio the data actually shows.
    se_naive, se20 = ic_sum.get("se_naive"), ic_sum.get("se_nw_lag20")
    realised = (float(se20 / se_naive)
                if se_naive and se20 and np.isfinite(se20) and se_naive > 0 else float("nan"))

    return {
        "h": int(h), "min_names": int(min_names),
        "n_eval_rows": int(m.sum()), "n_eval_dates": int(len(np.unique(d[m]))),
        "names_per_date": {"min": int(n_per_date.min()) if len(n_per_date) else 0,
                           "median": float(n_per_date.median()) if len(n_per_date) else float("nan"),
                           "max": int(n_per_date.max()) if len(n_per_date) else 0},
        "ic": ic,
        "ic_summary": ic_sum,
        "ic_by_year": _by_year(ic, h, seed=seed),
        "spread": spread,
        "spread_summary": {k: summarise(v, h, seed=seed) for k, v in spread.items()},
        "spread_by_year": {k: _by_year(v, h, seed=seed) for k, v in spread.items()},
        "power": power_statement(n_med, int(len(ic)), ic_sd_per_date=ic_sd,
                                 ar1_inflation=ar1_inflation),
        "power_asymptotic": power_statement(n_med, int(len(ic)),
                                            ar1_inflation=ar1_inflation),
        "power_realised": (power_statement(n_med, int(len(ic)), ic_sd_per_date=ic_sd,
                                           ar1_inflation=realised)
                           if np.isfinite(realised) else {}),
    }


def summary_json(res: dict) -> dict:
    """`evaluate` output with the pandas Series dropped -- what goes in a *_summary.json."""
    out = {k: v for k, v in res.items() if k not in ("ic", "spread")}
    out["spread_summary"] = {str(k): v for k, v in res["spread_summary"].items()}
    out["spread_by_year"] = {str(k): v for k, v in res["spread_by_year"].items()}
    return out


def year_table(res: dict, k: int = 10) -> pd.DataFrame:
    """Calendar-year table: IC and top-k spread side by side, both at the lag-20 t."""
    rows = []
    for y in sorted(set(res["ic_by_year"]) | set(res["spread_by_year"].get(k, {}))):
        a = res["ic_by_year"].get(y, {})
        b = res["spread_by_year"].get(k, {}).get(y, {})
        rows.append({
            "year": y, "n_dates": a.get("n_dates", 0),
            "ic_mean": a.get("mean_ic", float("nan")),
            "ic_t_nw20": a.get("t_nw_lag20", float("nan")),
            f"spread{k}_mean": b.get("mean_ic", float("nan")),
            f"spread{k}_t_nw20": b.get("t_nw_lag20", float("nan")),
        })
    return pd.DataFrame(rows)
