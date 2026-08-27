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
