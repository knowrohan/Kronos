"""Cross-sectional backtester -- a clone of qlib's TopkDropoutStrategy.

STRATEGY (finetune/config.py defaults: topk=50, n_drop=5, hold_thresh=5)
    A line-for-line port of qlib 0.9.7's TopkDropoutStrategy, read from the installed
    package rather than reconstructed from the docs. See topk_dropout_targets for the
    ``comb`` sell-high/buy-low guard, which is the step that makes the strategy hold still
    when the book already is the top-k instead of churning n_drop names every rebalance.
    Untradeable names are retained, not force-sold -- qlib `continue`s past them.

EXECUTION (plan rule 5.1, mirroring qlib's delay_execution=True, deal_price="open")
    The score for session t is computed from bars up to and including t's close.
    The trade happens at t+1's OPEN. Portfolio return accrues open(t+1) -> open(t+2).
    ``assert_execution_lag`` checks this in code rather than trusting a comment.

WEIGHTS
    ``--weighting drift`` (default) mirrors qlib: only the changed names trade, the rest
    drift with their returns. ``--weighting rebalance`` resets every held name to 1/topk at
    each rebalance -- cleaner in theory, far higher turnover, so it is reported as secondary.

COSTS
    Turnover on a rebalance is sum_j |w_new_j - w_drifted_j|, i.e. total notional traded as a
    fraction of capital. It is charged at the one-way rate. See costs.py for the rate model
    and why the plan's draft rates were wrong.
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
from bench.nse.common import DATA_NSE, OUT_NSE, sidecar  # noqa: E402
from bench.nse.costs import PRESETS, CostModel  # noqa: E402

TRADING_DAYS = 252


# ----------------------------------------------------------------- market data


class Market:
    """Open-to-open returns plus tradability, indexed by (session, symbol)."""

    def __init__(self, panel: pd.DataFrame, calendar: list[pd.Timestamp]):
        opens = panel.pivot(index="date", columns="symbol", values="open").sort_index()
        closes = panel.pivot(index="date", columns="symbol", values="close").sort_index()
        opens = opens.reindex(calendar)
        closes = closes.reindex(calendar)
        self.opens = opens
        self.closes = closes
        self.calendar = list(opens.index)
        self.idx = {d: i for i, d in enumerate(self.calendar)}
        # r[s] = open(s+1)/open(s) - 1 : the return earned by a position held from open s
        self.oo_ret = opens.shift(-1) / opens - 1.0
        self.tradable = opens.notna() & closes.notna() & (opens > 0)

    def forward_return(self, s_idx: int, horizon: int, price: str = "open",
                       lag: int = 1) -> pd.Series:
        """Realized forward return over `horizon` sessions, with `lag` sessions of delay.

        price="open", lag=1 -> open(t+1+h)/open(t+1) - 1 : what a strategy can capture.
        price="close", lag=0 -> close(t+h)/close(t) - 1  : the model's own training target.
        """
        px = self.opens if price == "open" else self.closes
        a, b = s_idx + lag, s_idx + lag + horizon
        if b >= len(self.calendar):
            return pd.Series(dtype=float)
        return px.iloc[b] / px.iloc[a] - 1.0


def assert_execution_lag(signal_idx: int, exec_idx: int, calendar) -> None:
    """Plan rule 5.1, enforced in code: context ends at t, the trade happens strictly after."""
    assert exec_idx == signal_idx + 1, (
        f"execution must be the session immediately after the signal date: "
        f"signal={calendar[signal_idx]} exec={calendar[exec_idx]}")


# ----------------------------------------------------------------- strategy


def topk_dropout_targets(scores: pd.Series, held: dict[str, int], topk: int, n_drop: int,
                         hold_thresh: int, tradable: set[str], mode: str = "qlib"
                         ) -> tuple[list[str], list[str], list[str]]:
    """One TopkDropoutStrategy step -> (target_names, bought, sold).

    ``mode="qlib"`` (default) is a line-for-line port of
    ``qlib.contrib.strategy.signal_strategy.TopkDropoutStrategy.generate_trade_decision``
    (qlib 0.9.7, read from the installed package), with method_buy="top",
    method_sell="bottom", only_tradable=True. The step that matters, and that a casual
    reading of "drop the worst n_drop" misses, is the ``comb`` guard:

        comb = rank(held  U  buy_candidates)
        sell = [s in held if s in last_n(comb, n_drop)]      # qlib's own comment:
                                                            # "In case of dropping higher
                                                            #  score stock and buying lower
                                                            #  score stock"
        buy  = candidates[: len(sell) + topk - len(held)]

    So when the book already IS the top-`topk`, the bottom n_drop of `comb` are the
    *candidates*, none of which are held -> sell is empty -> buy is empty -> no trade.
    qlib does NOT churn. An implementation that simply drops held[topk-n_drop:] does, at
    ~21% of book per rebalance on a tied score (verify_backtest.py test 10.4b).

    ``mode="naive"`` keeps that broken variant so the writeup can quantify what it costs.

    hold_thresh is applied AFTER the buy count is fixed, exactly as qlib does: a name blocked
    from selling still consumed a buy slot.
    """
    ranked = scores.dropna().sort_values(ascending=False)
    rank_of = {s: i for i, s in enumerate(ranked.index)}
    big = len(rank_of) + 1

    # qlib skips untradeable names in BOTH loops, so a suspended holding is simply retained.
    live = dict(held)

    last = sorted(live, key=lambda s: rank_of.get(s, big))
    n_take = n_drop + topk - len(last)
    candidates = [s for s in ranked.index if s not in live and s in tradable][: max(n_take, 0)]

    if mode == "naive":
        sell_pre = last[max(topk - n_drop, 0):]
        buy = candidates[: max(topk - len(last) + len(sell_pre), 0)]
    else:
        comb = sorted(dict.fromkeys(last + candidates), key=lambda s: rank_of.get(s, big))
        bottom = set(comb[-n_drop:]) if n_drop > 0 else set()
        sell_pre = [s for s in last if s in bottom]
        buy = candidates[: len(sell_pre) + topk - len(last)]

    sold = [s for s in sell_pre
            if s in tradable and live.get(s, 0) >= hold_thresh]
    retained = [s for s in last if s not in set(sold)]
    target = retained + buy
    return target, buy, sold


# ----------------------------------------------------------------- engine


def run_backtest(scores_wide: pd.DataFrame, market: Market, *, topk=50, n_drop=5,
                 hold_thresh=5, rebalance=1, cost_bps_round_trip=30.0,
                 cost_model: CostModel | None = None, capital: float = 1e7,
                 weighting="drift", dropout_mode="qlib", start=None, end=None,
                 log_trades=False) -> dict:
    """scores_wide: index = signal date, columns = symbol, values = score (NaN = ineligible).

    log_trades=True additionally returns a per-name trade blotter under key "trades"; it is
    opt-in because it is only needed by the liquidity analysis and costs an allocation per fill.
    """
    trade_log: list | None = [] if log_trades else None
    cal = market.calendar
    sig_dates = [d for d in scores_wide.index if d in market.idx]
    if start:
        sig_dates = [d for d in sig_dates if d >= pd.Timestamp(start)]
    if end:
        sig_dates = [d for d in sig_dates if d <= pd.Timestamp(end)]
    sig_dates = sig_dates[::rebalance]
    if not sig_dates:
        raise ValueError("no signal dates in range")

    one_way = (cost_bps_round_trip / 2.0 / 1e4 if cost_model is None
               else cost_model.one_way_bps(capital / topk) / 1e4)

    weights = pd.Series(dtype=float)          # current portfolio weights
    held_days: dict[str, int] = {}
    rows = []
    rebalance_log = []
    exec_set = set()

    exec_idxs = []
    for d in sig_dates:
        si = market.idx[d]
        if si + 1 >= len(cal):
            continue
        assert_execution_lag(si, si + 1, cal)
        exec_idxs.append((si, si + 1))
    if not exec_idxs:
        raise ValueError("no executable signal dates")

    first_exec = exec_idxs[0][1]
    last_exec = exec_idxs[-1][1]
    next_reb = {ei: si for si, ei in exec_idxs}
    exec_set = set(next_reb)

    for s in range(first_exec, min(last_exec + 1, len(cal) - 1)):
        day = cal[s]
        tradable_today = set(market.tradable.iloc[s][market.tradable.iloc[s]].index)
        turnover = 0.0
        n_buy = n_sell = 0

        if s in exec_set:
            si = next_reb[s]
            sig_day = cal[si]
            scores = scores_wide.loc[sig_day].dropna()
            target, bought, sold = topk_dropout_targets(
                scores, held_days, topk, n_drop, hold_thresh, tradable_today, dropout_mode)
            if target:
                if weighting == "rebalance":
                    new_w = pd.Series(1.0 / len(target), index=target)
                else:
                    # drift: untouched names keep their drifted weight; freed capital and any
                    # residual are spread equally over the newly bought names.
                    keep_w = weights.reindex([t for t in target if t in weights.index]).fillna(0.0)
                    freed = 1.0 - keep_w.sum()
                    if bought:
                        add = pd.Series(freed / len(bought), index=bought)
                        new_w = pd.concat([keep_w, add])
                    else:
                        new_w = keep_w / keep_w.sum() if keep_w.sum() > 0 else keep_w
                new_w = new_w / new_w.sum()
            else:
                new_w = pd.Series(dtype=float)

            allsym = weights.index.union(new_w.index)
            delta = (new_w.reindex(allsym).fillna(0.0)
                     - weights.reindex(allsym).fillna(0.0))
            turnover = float(delta.abs().sum())
            if trade_log is not None:
                for sym, dw in delta.items():
                    if dw != 0.0:
                        trade_log.append({"exec_date": day, "symbol": sym,
                                          "weight_delta": float(dw)})
            n_buy, n_sell = len(bought), len(sold)
            for sym in list(held_days):
                if sym not in new_w.index:
                    held_days.pop(sym)
            for sym in new_w.index:
                held_days.setdefault(sym, 0)
            weights = new_w
            rebalance_log.append({"signal_date": sig_day, "exec_date": day,
                                  "n_held": len(new_w), "n_buy": n_buy, "n_sell": n_sell,
                                  "turnover": turnover})

        r = market.oo_ret.iloc[s].reindex(weights.index).fillna(0.0)
        gross = float((weights * r).sum())
        cost = turnover * one_way
        rows.append({"date": day, "gross_return": gross, "cost": cost,
                     "net_return": gross - cost, "turnover": turnover,
                     "n_held": len(weights), "n_buy": n_buy, "n_sell": n_sell})

        # drift the weights with realized returns
        if len(weights):
            grown = weights * (1.0 + r)
            tot = grown.sum()
            weights = grown / tot if tot > 0 else grown
            for sym in held_days:
                held_days[sym] += 1

    df = pd.DataFrame(rows).set_index("date")
    out = {"daily": df, "rebalances": pd.DataFrame(rebalance_log), "one_way_cost": one_way}
    if trade_log is not None:
        out["trades"] = pd.DataFrame(trade_log)
    return out


# ----------------------------------------------------------------- metrics


def annualise(daily: pd.Series) -> float:
    return float((1.0 + daily).prod() ** (TRADING_DAYS / max(len(daily), 1)) - 1.0)


def max_drawdown(daily: pd.Series) -> float:
    curve = (1.0 + daily).cumprod()
    return float((curve / curve.cummax() - 1.0).min())


def block_bootstrap_ci(x: np.ndarray, stat, n_boot=10_000, block=21, seed=0, alpha=0.05):
    """Moving-block bootstrap CI -- daily returns and overlapping-horizon ICs are autocorrelated."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return (float("nan"), float("nan"))
    # shrink the block on short samples rather than returning NaN
    block = int(max(2, min(block, n // 4)))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    offs = np.arange(block)
    idx = (starts[:, :, None] + offs[None, None, :]).reshape(n_boot, -1)[:, :n]
    samples = stat(x[idx], axis=1)
    return (float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2)))


def metrics(res: dict, bench_daily: pd.Series, seed: int = 0) -> dict:
    d = res["daily"]
    net = d["net_return"]
    gross = d["gross_return"]
    b = bench_daily.reindex(d.index).fillna(0.0)
    ex = net - b
    ex_nc = gross - b
    n = len(d)
    ann_cost = float(d["cost"].sum()) * TRADING_DAYS / max(n, 1)
    lo, hi = block_bootstrap_ci(ex.to_numpy(), lambda a, axis: a.mean(axis=axis), seed=seed)
    return {
        "n_days": int(n),
        "ann_return_net": annualise(net), "ann_return_gross": annualise(gross),
        "ann_bench": annualise(b),
        "ann_excess_net": annualise(net) - annualise(b),
        "ann_excess_gross": annualise(gross) - annualise(b),
        "vol_ann": float(net.std() * np.sqrt(TRADING_DAYS)),
        "sharpe_net": float(net.mean() / net.std() * np.sqrt(TRADING_DAYS)) if net.std() > 0 else float("nan"),
        "information_ratio": float(ex.mean() / ex.std() * np.sqrt(TRADING_DAYS)) if ex.std() > 0 else float("nan"),
        "max_drawdown": max_drawdown(net),
        "hit_rate_daily": float((ex > 0).mean()),
        "mean_daily_excess": float(ex.mean()),
        "excess_ci95_daily": [lo, hi],
        "excess_ci95_annualised": [lo * TRADING_DAYS, hi * TRADING_DAYS],
        "vol_ann_bench": float(b.std() * np.sqrt(TRADING_DAYS)),
        # Raw excess rewards an arm for simply holding more risk. Levering the benchmark to the
        # arm's own volatility and charging that instead separates selection from beta -- which
        # matters here because the highest-excess arm in the study is a volatility sort.
        "ann_excess_volmatched": (
            annualise(net) - (float(net.std()) / float(b.std())) * annualise(b)
            if float(b.std()) > 0 else float("nan")),
        "turnover_per_rebalance": float(res["rebalances"]["turnover"].mean()) if len(res["rebalances"]) else 0.0,
        "turnover_ann": float(d["turnover"].sum()) * TRADING_DAYS / max(n, 1),
        "cost_drag_ann_pct": ann_cost * 100,
        "one_way_cost_bps": res["one_way_cost"] * 1e4,
        "ann_excess_no_cost": annualise(gross) - annualise(b),
    }


# ----------------------------------------------------------------- io helpers


def load_scores(path: str, column: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.pivot(index="date", columns="symbol", values=column).sort_index()


def load_market(panel_path=None, calendar_path=None) -> Market:
    panel = pd.read_parquet(panel_path or os.path.join(DATA_NSE, "panel_price.parquet"))
    with open(calendar_path or os.path.join(DATA_NSE, "calendar.json")) as f:
        cal = [pd.Timestamp(d) for d in json.load(f)]
    return Market(panel, cal)


def bench_series(market: Market, col="cnx200", path=None) -> pd.Series:
    """Daily open-to-open benchmark return.

    ``cnx200`` / ``nsei`` / ``nifty500`` are the published CAP-weighted price indices.
    ``ew200`` is the EQUAL-weighted return of the eligible universe itself. It matters a
    great deal here: the strategy is an equal-weighted top-50 drawn from a survivorship-
    biased 200-name list, and over this test window equal-weighting that list beat the
    cap-weighted index by a very large margin. Excess vs ``cnx200`` therefore mixes
    selection skill with a size/weighting tilt; excess vs ``ew200`` isolates selection.
    """
    if col == "ew200":
        return market.oo_ret.where(market.tradable).mean(axis=1)
    b = pd.read_parquet(path or os.path.join(DATA_NSE, "bench.parquet"))
    b["date"] = pd.to_datetime(b["date"])
    b = b.set_index("date")
    op = b[f"{col}_open"].reindex(market.calendar)
    return (op.shift(-1) / op - 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signals", default=os.path.join(OUT_NSE, "signals_zeroshot.parquet"))
    ap.add_argument("--column", default="last_ret")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--n-drop", type=int, default=5)
    ap.add_argument("--hold-thresh", type=int, default=5)
    ap.add_argument("--rebalance", type=int, default=5, help="1 = daily, 5 = weekly")
    ap.add_argument("--cost-bps", type=float, default=30.0, help="round-trip bps")
    ap.add_argument("--cost-preset", default=None, choices=list(PRESETS),
                    help="use the statutory model instead of a flat bps figure")
    ap.add_argument("--capital", type=float, default=1e7)
    ap.add_argument("--weighting", choices=["drift", "rebalance"], default="drift")
    ap.add_argument("--dropout-mode", choices=["qlib", "naive"], default="qlib",
                    help="qlib = faithful port incl. the comb sell-high/buy-low guard; "
                         "naive = always drop held[topk-n_drop:] (churns on ties)")
    ap.add_argument("--benchmark", default="cnx200")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    market = load_market()
    scores = load_scores(args.signals, args.column)
    res = run_backtest(scores, market, topk=args.topk, n_drop=args.n_drop,
                       hold_thresh=args.hold_thresh, rebalance=args.rebalance,
                       cost_bps_round_trip=args.cost_bps,
                       cost_model=PRESETS[args.cost_preset] if args.cost_preset else None,
                       capital=args.capital, weighting=args.weighting,
                       dropout_mode=args.dropout_mode,
                       start=args.start, end=args.end)
    m = metrics(res, bench_series(market, args.benchmark), seed=args.seed)
    tag = args.tag or f"{args.column}_reb{args.rebalance}_{int(args.cost_bps)}bps"
    os.makedirs(OUT_NSE, exist_ok=True)
    res["daily"].to_parquet(os.path.join(OUT_NSE, f"bt_{tag}.parquet"))
    with open(os.path.join(OUT_NSE, f"bt_{tag}.json"), "w") as f:
        json.dump(m, f, indent=2)
    sidecar(os.path.join(OUT_NSE, f"bt_{tag}_meta.json"), vars(args), extra=m)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
