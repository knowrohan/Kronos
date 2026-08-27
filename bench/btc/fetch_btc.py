"""Paginated Binance kline fetch.

bench/fetch_data.py caps at limit=1000, which buys 41 days of hourly bars -- enough for
the single-window demo it was written for, not enough to measure anything. This walks
startTime forward until it reaches the present, so the daily series goes back to
Binance's own launch and the hourly series covers several regimes rather than one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data", "crypto")
URL = "https://api.binance.com/api/v3/klines"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "amount", "trades", "tb_base", "tb_quote", "ignore"]


def fetch(symbol: str, interval: str, start: str, end: str | None = None,
          pause: float = 0.25) -> pd.DataFrame:
    def _ms(v) -> int:
        ts = pd.Timestamp(v)
        # pandas >= 2 returns tz-aware from utcnow(); localize only what is naive
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        return int(ts.timestamp() * 1000)

    t0 = _ms(start)
    t_end = _ms(end) if end else _ms(pd.Timestamp.utcnow())
    rows, seen = [], 0
    while t0 < t_end:
        r = requests.get(URL, params={"symbol": symbol, "interval": interval,
                                      "startTime": t0, "limit": 1000}, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        # advance past the last open_time; Binance returns [start, start+limit) inclusive
        nxt = batch[-1][0] + 1
        if nxt <= t0:
            break
        t0 = nxt
        seen += len(batch)
        print(f"  {symbol} {interval}: {seen} bars -> {pd.to_datetime(batch[-1][0], unit='ms')}",
              flush=True)
        if len(batch) < 1000:
            break
        time.sleep(pause)

    df = pd.DataFrame(rows, columns=COLS)
    df["timestamps"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ("open", "high", "low", "close", "volume", "amount"):
        df[c] = df[c].astype(float)
    df = (df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]
          .drop_duplicates(subset="timestamps").sort_values("timestamps")
          .reset_index(drop=True))
    # a live bar is still forming and would be a partial candle; drop it
    return df.iloc[:-1] if len(df) else df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="BTCUSDT:1d:2017-08-17,BTCUSDT:1h:2020-01-01,"
                                         "ETHUSDT:1d:2017-08-17")
    ap.add_argument("--end", default=None)
    a = ap.parse_args()

    os.makedirs(DATA, exist_ok=True)
    manifest = {}
    for spec in a.targets.split(","):
        sym, iv, start = spec.split(":")
        name = f"{sym.replace('USDT','').lower()}_{iv}"
        print(f"fetching {name} from {start} ...", flush=True)
        df = fetch(sym, iv, start, a.end)
        path = os.path.join(DATA, f"{name}.csv")
        df.to_csv(path, index=False)
        gaps = df["timestamps"].diff().value_counts().head(3).to_dict()
        manifest[name] = {
            "source": "binance", "symbol": sym, "interval": iv,
            "rows": len(df), "path": path,
            "first": str(df["timestamps"].iloc[0]), "last": str(df["timestamps"].iloc[-1]),
            "zero_volume_pct": float((df["volume"] <= 0).mean() * 100),
            "modal_gaps": {str(k): int(v) for k, v in gaps.items()},
        }
        print(f"  -> {len(df)} rows {manifest[name]['first']} .. {manifest[name]['last']}",
              flush=True)

    with open(os.path.join(DATA, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
