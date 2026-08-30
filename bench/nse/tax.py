"""Indian personal tax on a systematic cash-equity book -- a WORKING MODEL, NOT TAX ADVICE.

Every number here is a reconstruction from a backtest blotter under stated assumptions. It is
built to answer one question: does the strategy's excess over NIFTYBEES survive tax, when the
tracker itself is taxed the way a buy-and-hold investor is taxed? Nothing in this file is a
filing position and none of it has been checked by anyone qualified to check it.

WHY IT MATTERS ENOUGH TO MODEL. A weekly book realises its entire gain every few weeks at the
short-term rate; the tracker realises nothing until it is sold and then pays LTCG on the excess
over Rs 1.25 lakh. Comparing the two before tax flatters the book by roughly the short-term rate
times its turnover-driven realisation. On a 3-year window that is several percentage points a
year -- the same order as the whole edge being tested (NIFTY50_PLAN sec.6.3, gate M2).

WHAT IS RECONSTRUCTED, AND FROM WHAT
    ``backtest.run_backtest(log_trades=True)`` returns a blotter of (exec_date, symbol,
    weight_delta) -- portfolio-weight changes, not rupees, not shares. Rupees come from the
    equity curve: a fill of weight w on session d moves ``w * V_open(d)`` rupees, where
    ``V_open(d) = capital * prod_{d' < d} (1 + net_return(d'))`` is the book's value at the OPEN
    of d, which is when the backtester executes. This is exact for the drift-weighted book:
    ``run_backtest`` renormalises weights daily by the portfolio total, so a name's rupee value
    is always its weight times the book value, and a full exit's ``weight_delta`` is exactly the
    drifted weight -- proceeds and cost basis are both recovered without any price data.

FIFO AND THE ONE APPROXIMATION
    Lots are matched first-in-first-out per symbol, which is what s.45 read with the demat rules
    requires for listed equity. A sell fill that liquidates at least ``full_exit_frac`` (0.5) of
    the symbol's recorded open weight is treated as a FULL exit. The reason: topk-dropout with
    drift weighting only ever exits a name in full, but the recorded weight at entry and the
    drifted weight at exit differ by the name's relative return since entry -- a few percent over
    a 1-4 week hold, nowhere near the 2x that would confuse the rule. Genuinely partial sells
    (which only ``weighting="rebalance"`` produces) consume lots FIFO pro-rata by cost basis.
    That allocation is the single approximation in the reconstruction and it is flagged in the
    output as ``n_partial_sells``.

COSTS: WHY STT IS ADDED BACK RATHER THAN DEDUCTED
    The equity curve is already net of every transaction cost -- ``run_backtest`` charges
    turnover x one-way rate as a daily return drag -- so the realised gains reconstructed from it
    are net of brokerage, exchange charges, GST, stamp duty, slippage AND STT. Under the business
    regimes that is the right base: all of it, STT included, is a deductible business expense.
    Under ``stcg`` it is not: s.48 allows expenditure wholly and exclusively in connection with
    the transfer but the proviso expressly disallows STT. So the model ADDS STT BACK to the
    taxable base under ``stcg`` instead of deducting it under the business regimes. Same
    difference, one subtraction, no double counting. STT per fill is taken from
    ``costs.CostModel.breakdown(notional)[side]["stt"]`` at the fill's own notional.

REGIMES
    ``stcg``        20% + 4% cess = 20.8% on net realised short-term capital gains (the rate in
                    force for listed equity from 23 Jul 2024). Long-term gains, if the book ever
                    holds past 365 days, at 12.5% + cess above the Rs 1.25 lakh exemption.
    ``business_31`` 31.2% (30% slab + 4% cess) on net profit -- the top-slab bound if the user
                    elects to report trading as non-speculative business income.
    ``business_21`` 20.8% on the same base -- the lower bound at the user's stated Rs 10 lakh
                    scale, where slab rates are well below 30%.
    All three set off losses within the financial year and carry net losses forward up to 8
    financial years (s.74 for capital losses, s.72 for business losses); the window here is 3
    years so nothing expires, but the ageing is implemented rather than assumed away.

WHAT IS DELIBERATELY NOT MODELLED
    Advance-tax timing and s.234B/C interest; surcharge (nil below Rs 50 lakh of total income);
    the rest of the user's income, so slab interaction is bounded by the two business rates
    rather than computed; s.44AD/44AB audit thresholds; securities-transaction-tax rebate under
    s.36(1)(xv) which the business regimes already capture as a plain expense; set-off against
    heads other than capital gains / business income. The carried-forward loss pool is one bucket,
    not two: s.74 lets a short-term capital loss shelter both short- and long-term gains but a
    long-term loss shelter only long-term gains, and the pool here would let a brought-forward
    long-term loss shelter a short-term gain. On a weekly book that holds nothing past 365 days
    the distinction never arises -- ``fy_ledger`` reports ``realised_long_rs`` so it is visible
    if it ever does. Tax is charged at financial-year end out of the book itself, and the
    strategy is NOT re-run on the smaller book -- the after-tax curve is the pre-tax curve with
    the tax withdrawn, not a re-simulation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252          # same convention as backtest.annualise

STCG_RATE = 0.208           # 20% + 4% health & education cess, listed equity, from 23 Jul 2024
BUSINESS_RATE_31 = 0.312    # 30% slab + 4% cess
BUSINESS_RATE_21 = 0.208    # 20% slab + 4% cess
LTCG_RATE = 0.125 * 1.04    # 12.5% + 4% cess = 0.13
LTCG_EXEMPTION = 125_000.0  # s.112A, per financial year
LONG_TERM_DAYS = 365        # listed equity: >12 months is long-term
CARRY_FORWARD_YEARS = 8     # s.74 / s.72

REGIMES: dict[str, dict] = {
    "stcg": {
        "rate": STCG_RATE, "deduct_stt": False, "capital_gains": True,
        "label": "short-term capital gains, 20% + 4% cess; STT not deductible (s.48 proviso)",
    },
    "business_31": {
        "rate": BUSINESS_RATE_31, "deduct_stt": True, "capital_gains": False,
        "label": "non-speculative business income at 30% slab + 4% cess; STT deductible",
    },
    "business_21": {
        "rate": BUSINESS_RATE_21, "deduct_stt": True, "capital_gains": False,
        "label": "non-speculative business income at 20% slab + 4% cess; STT deductible",
    },
}


# ------------------------------------------------------------------ financial years


def fy_start_year(ts) -> int:
    """Calendar year in which the Indian financial year containing `ts` began (1 Apr - 31 Mar)."""
    ts = pd.Timestamp(ts)
    return int(ts.year if ts.month >= 4 else ts.year - 1)


def fy_label(ts) -> str:
    """'FY2023-24' for any date from 1 Apr 2023 to 31 Mar 2024 -- the way ITR forms label it."""
    y = fy_start_year(ts)
    return f"FY{y}-{str(y + 1)[-2:]}"


# ------------------------------------------------------------------ equity curve


def _as_net_series(daily_net) -> pd.Series:
    """Accept either the net-return Series or the whole ``run_backtest(...)['daily']`` frame."""
    if isinstance(daily_net, pd.DataFrame):
        daily_net = daily_net["net_return"]
    s = pd.Series(daily_net).astype(float).sort_index()
    s.index = pd.to_datetime(s.index)
    return s


def equity_curve(daily_net, capital: float = 1e6) -> pd.Series:
    """Book value at the CLOSE of each session (i.e. after that session's return)."""
    net = _as_net_series(daily_net)
    return capital * (1.0 + net).cumprod()


def opening_values(daily_net, capital: float = 1e6) -> pd.Series:
    """Book value at the OPEN of each session -- the price at which the backtester fills.

    ``run_backtest`` executes at open(t+1) and books that session's return from open(t+1) to
    open(t+2), so the value a fill on session d transacts against is the value BEFORE d's own
    return: capital on the first session, the previous close thereafter.
    """
    net = _as_net_series(daily_net)
    v = capital * (1.0 + net).cumprod()
    v_open = v.shift(1)
    if len(v_open):
        v_open.iloc[0] = float(capital)
    return v_open


# ------------------------------------------------------------------ fills


def build_fills(blotter, daily_net, capital: float = 1e6,
                cost_model=None, min_weight: float = 1e-9) -> pd.DataFrame:
    """Blotter of weight deltas -> rupee fills, with per-fill STT and total transaction cost.

    Columns out: exec_date, symbol, weight_delta, side, notional_rs, stt_rs, cost_rs, fy.
    STT and cost come from ``cost_model.breakdown(notional)[side]``; STT is ad valorem so
    evaluating it at the fill's own notional is exact, while the backtester charges its flat
    one-way rate at the average leg size -- a few bps of difference on the expense side only,
    never on the gain.

    ``min_weight`` drops float dust. ``run_backtest``'s drift branch builds the new weight
    vector as ``concat(kept, added) / sum``, and that sum is 1.0 only to machine precision, so
    a name that did not actually trade can be logged with a delta of order 1e-17 -- a fill of
    Rs 1e-11 on a Rs 10 lakh book. Observed on real runs (one lone dust sell on an otherwise
    quiet rebalance). Kept, such a row would add a zero-cost lot and, on the sell side, set the
    ``partial`` flag that is supposed to report the ONE approximation in this reconstruction.
    1e-9 of book weight is Rs 0.001 at Rs 10 lakh: far below anything an exchange can fill.
    """
    if cost_model is None:
        from bench.nse.costs import PRESETS
        cost_model = PRESETS["zerodha"]
    bl = pd.DataFrame(blotter)
    cols = ["exec_date", "symbol", "weight_delta", "side", "notional_rs",
            "stt_rs", "cost_rs", "fy"]
    if len(bl) == 0:
        return pd.DataFrame(columns=cols)
    bl = bl.copy()
    bl["exec_date"] = pd.to_datetime(bl["exec_date"])
    bl["weight_delta"] = bl["weight_delta"].astype(float)
    bl = bl[bl["weight_delta"].abs() > float(min_weight)]
    if len(bl) == 0:
        return pd.DataFrame(columns=cols)

    v_open = opening_values(daily_net, capital)
    missing = set(bl["exec_date"]) - set(v_open.index)
    if missing:
        raise ValueError(
            f"{len(missing)} blotter exec_date(s) are not in the daily return index "
            f"(first: {sorted(missing)[0]}); the blotter and the curve must come from the "
            f"same run_backtest call")
    bl["side"] = np.where(bl["weight_delta"] > 0, "buy", "sell")
    bl["notional_rs"] = bl["weight_delta"].abs().to_numpy() * v_open.reindex(
        bl["exec_date"]).to_numpy()

    stt, tot = [], []
    for notional, side in zip(bl["notional_rs"], bl["side"]):
        bd = cost_model.breakdown(float(notional))[side]
        stt.append(bd["stt"])
        tot.append(bd["total"])
    bl["stt_rs"] = stt
    bl["cost_rs"] = tot
    bl["fy"] = [fy_label(d) for d in bl["exec_date"]]
    bl = bl.sort_values(["symbol", "exec_date", "side"], kind="stable").reset_index(drop=True)
    return bl[cols]


# ------------------------------------------------------------------ FIFO matching


@dataclass
class _Lot:
    buy_date: pd.Timestamp
    weight: float
    cost_rs: float


def fifo_match(fills: pd.DataFrame, full_exit_frac: float = 0.5) -> pd.DataFrame:
    """FIFO-match sell fills to buy lots; one row per matched piece.

    Columns: symbol, buy_date, sell_date, cost_rs, proceeds_rs, gain_rs, holding_days, term
    ('short' / 'long'), fy (of the SELL -- the year of realisation), partial (bool).

    FIFO is not a choice: for listed shares held in demat the Act treats the earliest acquired
    as the earliest transferred. ``full_exit_frac`` decides when a sell is read as closing the
    position outright rather than trimming it -- see the module docstring for why 0.5 is safe
    for a topk-dropout book and why partial sells are the only approximation here.
    """
    cols = ["symbol", "buy_date", "sell_date", "cost_rs", "proceeds_rs", "gain_rs",
            "holding_days", "term", "fy", "partial"]
    if len(fills) == 0:
        return pd.DataFrame(columns=cols)
    out: list[dict] = []
    for sym, grp in fills.groupby("symbol", sort=True):
        lots: list[_Lot] = []
        for row in grp.sort_values("exec_date", kind="stable").itertuples():
            if row.side == "buy":
                lots.append(_Lot(row.exec_date, abs(row.weight_delta), row.notional_rs))
                continue
            proceeds = float(row.notional_rs)
            open_w = sum(l.weight for l in lots)
            open_cost = sum(l.cost_rs for l in lots)
            if open_cost <= 0:
                # A sell with nothing open cannot happen in a long-only book; if it ever does,
                # refuse to guess a cost basis rather than book a fictitious 100% gain.
                raise ValueError(f"sell fill for {sym} on {row.exec_date} has no open lot")
            frac = abs(row.weight_delta) / open_w if open_w > 0 else 1.0
            full = frac >= full_exit_frac
            target_cost = open_cost if full else min(frac, 1.0) * open_cost
            consumed, taken = 0.0, []
            while lots and consumed < target_cost - 1e-9:
                lot = lots[0]
                take = min(lot.cost_rs, target_cost - consumed)
                share = take / lot.cost_rs if lot.cost_rs > 0 else 1.0
                taken.append((lot.buy_date, take))
                consumed += take
                if share >= 1.0 - 1e-12:
                    lots.pop(0)
                else:
                    lot.cost_rs -= take
                    lot.weight -= lot.weight * share
            if full:
                lots.clear()
            for buy_date, cost in taken:
                pro = proceeds * (cost / consumed) if consumed > 0 else 0.0
                days = int((row.exec_date - buy_date).days)
                out.append({
                    "symbol": sym, "buy_date": buy_date, "sell_date": row.exec_date,
                    "cost_rs": cost, "proceeds_rs": pro, "gain_rs": pro - cost,
                    "holding_days": days,
                    "term": "long" if days > LONG_TERM_DAYS else "short",
                    "fy": fy_label(row.exec_date), "partial": not full,
                })
    df = pd.DataFrame(out, columns=cols)
    if len(df):
        df = df.sort_values(["sell_date", "symbol", "buy_date"],
                            kind="stable").reset_index(drop=True)
    return df


def realised_by_fill(matches: pd.DataFrame) -> pd.DataFrame:
    """Collapse matched pieces back to one row per sell fill -- the blotter-level realised P&L."""
    cols = ["sell_date", "symbol", "cost_rs", "proceeds_rs", "gain_rs", "fy"]
    if len(matches) == 0:
        return pd.DataFrame(columns=cols)
    g = (matches.groupby(["sell_date", "symbol"], sort=True)
         .agg(cost_rs=("cost_rs", "sum"), proceeds_rs=("proceeds_rs", "sum"),
              gain_rs=("gain_rs", "sum"), fy=("fy", "first"))
         .reset_index())
    return g[cols]


# ------------------------------------------------------------------ per-FY ledger


def fy_ledger(fills: pd.DataFrame, matches: pd.DataFrame,
              all_fys: list[str] | None = None) -> pd.DataFrame:
    """One row per financial year: realised short/long gains, STT paid, total cost, fill counts.

    STT and total cost are booked in the FY of the FILL that paid them (both legs), not the FY
    of realisation, because that is when the expense is incurred.
    """
    fys = list(all_fys) if all_fys else []
    for src in (fills, matches):
        if len(src):
            fys += list(src["fy"].unique())
    fys = sorted(set(fys))
    rows = []
    for fy in fys:
        m = matches[matches["fy"] == fy] if len(matches) else matches
        f = fills[fills["fy"] == fy] if len(fills) else fills
        rows.append({
            "fy": fy, "fy_start_year": int(fy[2:6]),
            "realised_short_rs": float(m[m["term"] == "short"]["gain_rs"].sum()) if len(m) else 0.0,
            "realised_long_rs": float(m[m["term"] == "long"]["gain_rs"].sum()) if len(m) else 0.0,
            "stt_rs": float(f["stt_rs"].sum()) if len(f) else 0.0,
            "cost_rs": float(f["cost_rs"].sum()) if len(f) else 0.0,
            "n_fills": int(len(f)), "n_sells": int(len(m)),
        })
    return pd.DataFrame(rows)


def _age_pool(pool: list[dict], year: int, carry_years: int) -> float:
    """Drop brought-forward losses older than `carry_years` financial years; return what lapsed."""
    lapsed = sum(e["remaining"] for e in pool if year - e["year"] > carry_years)
    pool[:] = [e for e in pool if year - e["year"] <= carry_years and e["remaining"] > 0]
    return float(lapsed)


def _consume_pool(pool: list[dict], gain: float) -> tuple[float, float]:
    """Set off `gain` against brought-forward losses, oldest first. Returns (gain_left, used)."""
    used = 0.0
    for e in pool:
        if gain <= 0:
            break
        take = min(e["remaining"], gain)
        e["remaining"] -= take
        gain -= take
        used += take
    pool[:] = [e for e in pool if e["remaining"] > 1e-9]
    return float(gain), float(used)


def tax_by_fy(ledger: pd.DataFrame, regime: str = "stcg",
              carry_years: int = CARRY_FORWARD_YEARS,
              ltcg_rate: float = LTCG_RATE,
              ltcg_exemption: float = LTCG_EXEMPTION) -> pd.DataFrame:
    """Tax due per financial year under one regime, with set-off and carry-forward.

    Order of operations, which is the order the Act prescribes: current-year loss set-off first
    (s.70), then brought-forward loss set-off oldest-first (s.74/s.72), then the exemption, then
    the rate. Under the business regimes the short/long distinction does not exist -- trading
    profit is one number -- so the two buckets are added before anything else.
    """
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; expected one of {sorted(REGIMES)}")
    spec = REGIMES[regime]
    pool: list[dict] = []
    rows = []
    for r in ledger.sort_values("fy_start_year").itertuples():
        year = int(r.fy_start_year)
        lapsed = _age_pool(pool, year, carry_years)
        brought_in = sum(e["remaining"] for e in pool)
        st = float(r.realised_short_rs)
        lt = float(r.realised_long_rs)
        if not spec["deduct_stt"]:
            # s.48 proviso: STT is not an allowable deduction against capital gains, but the
            # equity curve already paid it, so add it back to the taxable base.
            st += float(r.stt_rs)
        exempt_used = 0.0
        if spec["capital_gains"]:
            # s.70(2): a short-term capital loss may be set off against long-term capital gain;
            # a long-term loss may not be set off against short-term gain.
            if st < 0 and lt > 0:
                use = min(-st, lt)
                st += use
                lt -= use
            st, used_st = _consume_pool(pool, st) if st > 0 else (st, 0.0)
            lt, used_lt = _consume_pool(pool, lt) if lt > 0 else (lt, 0.0)
            used = used_st + used_lt
            net_loss = -(min(st, 0.0) + min(lt, 0.0))
            taxable_st = max(st, 0.0)
            exempt_used = min(max(lt, 0.0), ltcg_exemption)
            taxable_lt = max(max(lt, 0.0) - ltcg_exemption, 0.0)
            tax = spec["rate"] * taxable_st + ltcg_rate * taxable_lt
        else:
            profit = st + lt
            profit, used = _consume_pool(pool, profit) if profit > 0 else (profit, 0.0)
            net_loss = -min(profit, 0.0)
            taxable_st, taxable_lt = max(profit, 0.0), 0.0
            tax = spec["rate"] * taxable_st
        if net_loss > 0:
            pool.append({"year": year, "remaining": net_loss})
        rows.append({
            "fy": r.fy, "fy_start_year": year, "regime": regime,
            "realised_short_rs": float(r.realised_short_rs),
            "realised_long_rs": float(r.realised_long_rs),
            "stt_rs": float(r.stt_rs), "cost_rs": float(r.cost_rs),
            "loss_brought_forward_rs": float(brought_in),
            "loss_setoff_rs": float(used), "loss_lapsed_rs": float(lapsed),
            "ltcg_exemption_used_rs": float(exempt_used),
            "taxable_short_rs": float(taxable_st), "taxable_long_rs": float(taxable_lt),
            "loss_carried_forward_rs": float(sum(e["remaining"] for e in pool)),
            "tax_rs": float(tax),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ after-tax curves


def after_tax_curve(daily_net, blotter, capital: float = 1e6, regime: str = "stcg",
                    cost_model=None, full_exit_frac: float = 0.5,
                    return_parts: bool = False):
    """After-tax equity curve: the pre-tax curve with each year's tax withdrawn at FY end.

    Tax is charged on the last session of each financial year present in the index -- including
    the last session overall, so an incomplete final FY is accrued rather than silently forgiven.
    The strategy is not re-simulated on the smaller book: the withdrawal reduces the level, and
    subsequent returns compound on the reduced level, but the weights and fills are those of the
    pre-tax run. Real tax is paid months later in advance-tax instalments, so this is the
    conservative end of the timing question.

    return_parts=True additionally returns the fills, matches and per-FY tax frames, which is
    what summary() and the report need; the default stays a bare Series so the common call is
    ``after_tax_curve(res['daily'], res['trades'], capital, 'stcg', cm)``.
    """
    net = _as_net_series(daily_net)
    fills = build_fills(blotter, net, capital, cost_model)
    matches = fifo_match(fills, full_exit_frac=full_exit_frac)
    all_fys = sorted({fy_label(d) for d in net.index})
    ledger = fy_ledger(fills, matches, all_fys=all_fys)
    tax = tax_by_fy(ledger, regime=regime)
    tax_map = dict(zip(tax["fy"], tax["tax_rs"])) if len(tax) else {}

    fy_of_day = [fy_label(d) for d in net.index]
    last_of_fy = {}
    for d, fy in zip(net.index, fy_of_day):
        last_of_fy[fy] = d

    eq, vals, paid = float(capital), [], 0.0
    for d, r, fy in zip(net.index, net.to_numpy(), fy_of_day):
        eq *= (1.0 + float(r))
        if last_of_fy.get(fy) == d:
            due = float(tax_map.get(fy, 0.0))
            eq -= due
            paid += due
        vals.append(eq)
    curve = pd.Series(vals, index=net.index, name=f"after_tax_{regime}")
    if return_parts:
        return curve, {"fills": fills, "matches": matches, "ledger": ledger,
                       "tax": tax, "tax_paid_rs": paid}
    return curve


def tracker_after_tax(daily_bench, capital: float = 1e6, exemption: float = LTCG_EXEMPTION,
                      ltcg: float = LTCG_RATE, stcg: float = STCG_RATE,
                      realise: bool = True) -> float:
    """Final value of buy-and-hold in the tracker, taxed once on exit at the end of the window.

    This is the bar the book has to beat, and it is a low bar to tax: one purchase, one sale, no
    intervening realisation, so the whole gain is long-term at 12.5% + cess above the Rs 1.25
    lakh s.112A exemption if the window exceeds 12 months, and short-term at 20.8% if it does
    not. The exemption is applied once, on exit -- not once per financial year -- because there
    is only one realisation event.

    realise=False is the unrealised variant: the investor never sells, so no tax is ever paid and
    the answer is simply the terminal value. Reporting both brackets the honest comparison, since
    a real holder would sell only part of the position and only when they needed the money.
    """
    b = _as_net_series(daily_bench)
    final = float(capital) * float((1.0 + b).prod())
    if not realise:
        return final
    gain = final - float(capital)
    if gain <= 0:
        return final
    held_days = int((b.index[-1] - b.index[0]).days) if len(b) > 1 else 0
    if held_days > LONG_TERM_DAYS:
        return final - ltcg * max(gain - exemption, 0.0)
    return final - stcg * gain


def annualised(final: float, capital: float, n_days: int) -> float:
    """Annualised return from a terminal value, on the 252-session year backtest.py uses."""
    if n_days <= 0 or capital <= 0:
        return float("nan")
    return float((final / capital) ** (TRADING_DAYS / n_days) - 1.0)


def summary(daily_net, blotter, capital: float = 1e6, cost_model=None,
            daily_bench=None, regimes: tuple[str, ...] = ("stcg", "business_31", "business_21"),
            full_exit_frac: float = 0.5) -> dict:
    """Everything gate M2 needs: after-tax annualised return per regime, and tax paid per FY.

    When `daily_bench` is given, the tracker's after-tax terminal value and annualised return are
    computed on the same sessions (realised on exit and unrealised), plus the after-tax excess
    per regime -- which is the number the gate is actually stated in.
    """
    net = _as_net_series(daily_net)
    n = len(net)
    pre = equity_curve(net, capital)
    out: dict = {
        "n_days": int(n),
        "capital_rs": float(capital),
        "first_session": str(net.index[0].date()) if n else None,
        "last_session": str(net.index[-1].date()) if n else None,
        "pre_tax_final_rs": float(pre.iloc[-1]) if n else float(capital),
        "pre_tax_ann_return": annualised(float(pre.iloc[-1]) if n else capital, capital, n),
        "regimes": {},
        "disclaimer": ("working model, not tax advice; see bench/nse/tax.py docstring for the "
                       "assumptions and for what is deliberately not modelled"),
    }
    parts = None
    for reg in regimes:
        curve, parts = after_tax_curve(net, blotter, capital, reg, cost_model,
                                       full_exit_frac=full_exit_frac, return_parts=True)
        final = float(curve.iloc[-1]) if n else float(capital)
        tax = parts["tax"]
        out["regimes"][reg] = {
            "label": REGIMES[reg]["label"],
            "rate": REGIMES[reg]["rate"],
            "after_tax_final_rs": final,
            "after_tax_ann_return": annualised(final, capital, n),
            "total_tax_rs": float(tax["tax_rs"].sum()) if len(tax) else 0.0,
            "tax_by_fy": ({str(r.fy): float(r.tax_rs) for r in tax.itertuples()}
                          if len(tax) else {}),
            "taxable_by_fy": ({str(r.fy): float(r.taxable_short_rs + r.taxable_long_rs)
                               for r in tax.itertuples()} if len(tax) else {}),
            "loss_carried_forward_rs": (float(tax["loss_carried_forward_rs"].iloc[-1])
                                        if len(tax) else 0.0),
        }
    if parts is not None:
        m = parts["matches"]
        out["realised_gain_rs"] = float(m["gain_rs"].sum()) if len(m) else 0.0
        out["unrealised_gain_rs"] = (out["pre_tax_final_rs"] - float(capital)
                                     - out["realised_gain_rs"])
        out["n_fills"] = int(len(parts["fills"]))
        out["n_sell_matches"] = int(len(m))
        out["n_partial_sells"] = int(m["partial"].sum()) if len(m) else 0
        out["stt_paid_rs"] = float(parts["fills"]["stt_rs"].sum()) if len(parts["fills"]) else 0.0
        out["transaction_cost_rs"] = (float(parts["fills"]["cost_rs"].sum())
                                      if len(parts["fills"]) else 0.0)
    if daily_bench is not None:
        b = _as_net_series(daily_bench).reindex(net.index).fillna(0.0)
        realised = tracker_after_tax(b, capital, realise=True)
        unrealised = tracker_after_tax(b, capital, realise=False)
        out["tracker"] = {
            "pre_tax_final_rs": unrealised,
            "pre_tax_ann_return": annualised(unrealised, capital, n),
            "after_tax_final_rs_on_exit": realised,
            "after_tax_ann_return_on_exit": annualised(realised, capital, n),
            "after_tax_final_rs_unrealised": unrealised,
            "ltcg_exemption_rs": float(LTCG_EXEMPTION),
            "ltcg_rate": float(LTCG_RATE),
        }
        for reg, d in out["regimes"].items():
            d["ann_excess_vs_tracker_after_tax"] = (
                d["after_tax_ann_return"] - out["tracker"]["after_tax_ann_return_on_exit"])
            d["ann_excess_vs_tracker_unrealised"] = (
                d["after_tax_ann_return"] - out["tracker"]["pre_tax_ann_return"])
    return out
