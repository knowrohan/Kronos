"""Indian equity delivery transaction costs. Every rate is a named parameter, never a literal.

Rates verified 2026-08-27 against primary sources; see bench/nse/VERIFIED.md section 4.1.
The plan's draft model was wrong in three ways that matter:

  * ``brokerage`` was 0.03% ad valorem. Real discount-broker delivery brokerage is ZERO
    (Zerodha) or a FLAT Rs 20/order (Groww / Upstox / Angel One). At a Rs 10 lakh trade the
    ad valorem figure is 15x too high.
  * ``exchange_txn`` was 0.00297%. NSE circular NSE/FA/73061 (27 Feb 2026, effective
    1 Mar 2026) sets transaction + IPFT to Rs 307/crore/side = 0.00307%.
  * The DP (depository) charge on the sell leg was missing entirely: ~Rs 13-20 per scrip
    per day, plus GST.

The consequence is structural, not just numerical: real Indian delivery cost is NOT
scale-invariant. Brokerage is capped and DP is flat, so cost in bps FALLS with trade size:

    cost_per_leg(V) = ad_valorem * V + flat      ->   bps(V) = 1e4*ad_valorem + 1e4*flat/V

At Rs 10 lakh/leg the round trip is ~22.4 bps (zero brokerage). The floor as V grows is
22.25 bps and is essentially all STT. Below ~Rs 89,000 per leg the plan's flat 29.3 bps
model UNDERSTATES cost, which is the dangerous direction.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class CostModel:
    """Statutory + broker cost model for NSE delivery (CNC) equity."""

    stt_buy: float = 0.001            # 0.10%, delivery equity, unchanged through Budget 2026
    stt_sell: float = 0.001           # 0.10%
    stamp_duty_buy: float = 0.00015   # 0.015%, BUY SIDE ONLY, uniform since 1 Jul 2020
    exchange_txn: float = 0.0000307   # NSE Rs 306.99 + IPFT Rs 0.01 per crore, per side
    sebi_fee: float = 0.000001        # Rs 10 per crore, per side
    brokerage_rate: float = 0.0       # Zerodha delivery = 0
    brokerage_flat: float = 0.0       # Rs per order; 20.0 for Groww / Upstox / Angel One
    brokerage_cap: float = 20.0       # Rs; applies only when brokerage_rate > 0
    dp_charge_sell: float = 13.0      # Rs per scrip on the SELL leg (CDSL 3.50 + broker markup)
    gst: float = 0.18                 # on brokerage + exchange_txn + sebi_fee + dp, NOT on STT/stamp
    slippage_bps_per_side: float = 2.5  # 5 bps round trip

    def leg(self, notional: float, side: str) -> float:
        """Rupee cost of one leg of `notional` rupees."""
        assert side in ("buy", "sell")
        stt = (self.stt_buy if side == "buy" else self.stt_sell) * notional
        stamp = self.stamp_duty_buy * notional if side == "buy" else 0.0
        exch = self.exchange_txn * notional
        sebi = self.sebi_fee * notional
        if self.brokerage_rate > 0:
            brok = min(self.brokerage_rate * notional, self.brokerage_cap)
        else:
            brok = self.brokerage_flat
        dp = self.dp_charge_sell if side == "sell" else 0.0
        gst = self.gst * (brok + exch + sebi + dp)
        slip = self.slippage_bps_per_side / 1e4 * notional
        return stt + stamp + exch + sebi + brok + dp + gst + slip

    def round_trip_bps(self, notional: float) -> float:
        """Round-trip cost in bps of one leg's notional."""
        return (self.leg(notional, "buy") + self.leg(notional, "sell")) / notional * 1e4

    def one_way_bps(self, notional: float) -> float:
        """Average one-way cost in bps -- what the backtester charges per unit of turnover."""
        return self.round_trip_bps(notional) / 2.0

    def breakdown(self, notional: float) -> dict:
        out = {}
        for side in ("buy", "sell"):
            brok = (min(self.brokerage_rate * notional, self.brokerage_cap)
                    if self.brokerage_rate > 0 else self.brokerage_flat)
            dp = self.dp_charge_sell if side == "sell" else 0.0
            exch = self.exchange_txn * notional
            sebi = self.sebi_fee * notional
            out[side] = {
                "stt": (self.stt_buy if side == "buy" else self.stt_sell) * notional,
                "stamp_duty": self.stamp_duty_buy * notional if side == "buy" else 0.0,
                "exchange_txn": exch, "sebi_fee": sebi, "brokerage": brok, "dp_charge": dp,
                "gst": self.gst * (brok + exch + sebi + dp),
                "slippage": self.slippage_bps_per_side / 1e4 * notional,
            }
            out[side]["total"] = sum(out[side].values())
        out["round_trip_rs"] = out["buy"]["total"] + out["sell"]["total"]
        out["round_trip_bps"] = out["round_trip_rs"] / notional * 1e4
        out["rates"] = asdict(self)
        return out


PRESETS = {
    # Zerodha: zero delivery brokerage. The realistic default for a retail systematic book.
    "zerodha": CostModel(),
    # Groww / Upstox / Angel One: flat Rs 20 per executed order.
    "flat20": CostModel(brokerage_flat=20.0),
    # The plan's draft model, kept so the writeup can show what it gets wrong.
    "plan_draft": CostModel(exchange_txn=0.0000297, brokerage_rate=0.0003, brokerage_cap=1e18,
                            dp_charge_sell=0.0, slippage_bps_per_side=2.5),
    "zero": CostModel(stt_buy=0, stt_sell=0, stamp_duty_buy=0, exchange_txn=0, sebi_fee=0,
                      brokerage_flat=0, dp_charge_sell=0, slippage_bps_per_side=0),
}


# --------------------------------------------------------------- single-stock futures

# Lot sizes as researched on 2026-08-29 (data/nse/research/futures_facts.json, "lots").
# They are here rather than read from the JSON at import time so that costs.py stays a pure,
# dependency-free rate table; n50_futures.py re-reads the JSON and asserts agreement.
# Lot sizes are revised periodically by the exchange -- these are an Aug-2026 snapshot, and the
# research note rates them "medium" confidence (aggregated broker pages, not an NSE master file).
LOT_SIZES = {
    "RELIANCE": 500,
    "HDFCBANK": 650,
    "ICICIBANK": 700,
    "INFY": 400,
    "TCS": 225,
    "ITC": 1725,
    "SBIN": 750,
    "BHARTIARTL": 475,
    "LT": 175,
    "TMPV": 1600,   # ex-TATAMOTORS passenger-vehicle entity, F&O relaunched 2025-10-14
}

# Initial margin (SPAN + exposure) as a fraction of contract value. 0.22 is the middle of the
# broker-quoted NRML band for large caps; the range is wide and it is the single number most
# likely to be wrong, so every margin-sensitive result must be reported across the range.
MARGIN_PCT_DEFAULT = 0.22
MARGIN_PCT_RANGE = (0.17, 0.41)


@dataclass
class FuturesCostModel:
    """Statutory + broker cost model for NSE single-stock futures (NRML), Aug 2026 rates.

    Rates from data/nse/research/futures_facts.json. Three things differ from the cash model
    and each changes the arithmetic in a direction that matters:

      * STT is SELL SIDE ONLY and was raised from 0.02% to 0.05% by Union Budget 2026,
        effective 2026-04-01. That single line is 84% of the statutory round trip.
      * The exchange transaction charge is 0.00173%, an order of magnitude below the cash
        0.00307%, and there is no DP charge -- nothing is delivered until expiry.
      * Brokerage is min(0.03%, Rs 20) per order, so it is capped at Rs 20 for any lot above
        Rs 66,667 of notional, i.e. always in practice.

    The statutory round trip is therefore ~5.95 bps on a Rs 15 lakh lot -- about a quarter of
    the cash book's 22.4 bps. That is NOT the reason to prefer futures: a lot is 1.5x the whole
    Rs 10 lakh book, so the position is levered ~1.5x before any of it is financed, and the
    carry (``roll_cost_bps_per_month``) is charged whether or not the trade works. Holding cost,
    not transaction cost, is what decides this venue.
    """

    stt_sell: float = 0.0005          # 0.05% sell side, effective 2026-04-01 (was 0.02%)
    stt_buy: float = 0.0              # no STT on the futures buy leg
    exchange_txn: float = 0.0000173   # NSE equity futures 0.00173% of turnover, per side
    sebi_fee: float = 0.000001        # Rs 10 per crore, per side
    stamp_duty_buy: float = 0.00002   # 0.002%, BUY SIDE ONLY
    brokerage_rate: float = 0.0003    # Zerodha F&O: min(0.03%, Rs 20) per executed order
    brokerage_cap: float = 20.0       # Rs
    gst: float = 0.18                 # on brokerage + exchange_txn + sebi_fee, NOT on STT/stamp
    slippage_bps_per_side: float = 2.0  # 4 bps round trip; single-stock futures books are
                                        # thinner than the cash book but the lot is one ticket
    roll_cost_bps_per_month: float = 55.0  # midpoint of the researched 45-65 bps rollover band
    margin_pct: float = MARGIN_PCT_DEFAULT  # see MARGIN_PCT_RANGE; report across it, never on it

    def leg(self, notional: float, side: str) -> float:
        """Rupee cost of one leg of `notional` rupees of contract value."""
        assert side in ("buy", "sell")
        stt = (self.stt_buy if side == "buy" else self.stt_sell) * notional
        stamp = self.stamp_duty_buy * notional if side == "buy" else 0.0
        exch = self.exchange_txn * notional
        sebi = self.sebi_fee * notional
        brok = min(self.brokerage_rate * notional, self.brokerage_cap)
        gst = self.gst * (brok + exch + sebi)
        slip = self.slippage_bps_per_side / 1e4 * notional
        return stt + stamp + exch + sebi + brok + gst + slip

    def round_trip_bps(self, notional: float) -> float:
        """Round-trip cost in bps of one leg's notional (buy then sell the same lot)."""
        return (self.leg(notional, "buy") + self.leg(notional, "sell")) / notional * 1e4

    def one_way_bps(self, notional: float) -> float:
        """Average one-way cost in bps -- what a turnover-based backtester charges."""
        return self.round_trip_bps(notional) / 2.0

    def holding_cost_bps(self, days: float) -> float:
        """Carry charged for holding `days` sessions, at roll_cost_bps_per_month / 21 per session.

        21 sessions is the NSE month used throughout this repo. This is a linear proxy for
        basis decay plus the calendar roll: the research note found only an index-level
        rollover band (45-65 bps/month, "low-medium" confidence) and no per-stock dataset, so
        the number is the weakest input in the futures arithmetic and is a parameter, not a
        constant, for exactly that reason.
        """
        return self.roll_cost_bps_per_month * days / 21.0

    def breakdown(self, notional: float) -> dict:
        out = {}
        for side in ("buy", "sell"):
            brok = min(self.brokerage_rate * notional, self.brokerage_cap)
            exch = self.exchange_txn * notional
            sebi = self.sebi_fee * notional
            out[side] = {
                "stt": (self.stt_buy if side == "buy" else self.stt_sell) * notional,
                "stamp_duty": self.stamp_duty_buy * notional if side == "buy" else 0.0,
                "exchange_txn": exch, "sebi_fee": sebi, "brokerage": brok,
                "gst": self.gst * (brok + exch + sebi),
                "slippage": self.slippage_bps_per_side / 1e4 * notional,
            }
            out[side]["total"] = sum(out[side].values())
        out["round_trip_rs"] = out["buy"]["total"] + out["sell"]["total"]
        out["round_trip_bps"] = out["round_trip_rs"] / notional * 1e4
        out["margin_rs"] = self.margin_pct * notional
        out["rates"] = asdict(self)
        return out


FUTURES_PRESETS = {
    # Zerodha NRML single-stock futures, post-2026-04-01 STT.
    "zerodha_fut": FuturesCostModel(),
    # The research agent's worked example excludes slippage, margin financing and roll; this
    # preset reproduces its Rs 891.98 / 5.95 bps figure exactly so the test can cite it.
    "worked_example": FuturesCostModel(slippage_bps_per_side=0.0, roll_cost_bps_per_month=0.0),
    # Pessimistic carry: top of the researched roll band and double the slippage.
    "zerodha_fut_high_carry": FuturesCostModel(slippage_bps_per_side=4.0,
                                               roll_cost_bps_per_month=65.0),
    "zero": FuturesCostModel(stt_sell=0, stt_buy=0, exchange_txn=0, sebi_fee=0, stamp_duty_buy=0,
                             brokerage_rate=0, brokerage_cap=0, slippage_bps_per_side=0,
                             roll_cost_bps_per_month=0),
}
