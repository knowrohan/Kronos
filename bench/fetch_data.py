"""Fetch original OHLCV data for the Kronos end-to-end test.

Sources:
  - Binance public REST (BTC/USDT)  -> real quote-asset volume maps to Kronos 'amount'
  - Yahoo Finance via yfinance (NSE) -> 'amount' derived as volume * typical price

Output: data/<name>.csv with Kronos schema:
  timestamps, open, high, low, close, volume, amount
"""
import os, sys, time, json
import pandas as pd
import requests

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(OUT, exist_ok=True)
COLS = ["timestamps", "open", "high", "low", "close", "volume", "amount"]


def fetch_binance(symbol="BTCUSDT", interval="1h", limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=30)
    r.raise_for_status()
    raw = r.json()
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore"])
    df["timestamps"] = pd.to_datetime(df["open_time"], unit="ms")          # UTC, tz-naive
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = df[c].astype("float64")
    df["amount"] = df["quote_volume"]                                       # genuine notional traded
    # drop the still-forming final bar
    df = df.iloc[:-1]
    return df[COLS].reset_index(drop=True)


def fetch_yahoo(symbol, interval, period):
    import yfinance as yf
    df = yf.download(symbol, period=period, interval=interval, progress=False,
                     auto_adjust=False, threads=False)
    if df is None or len(df) == 0:
        raise RuntimeError(f"yfinance returned no rows for {symbol} {interval} {period}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    tcol = "Datetime" if "Datetime" in df.columns else "Date"
    ts = pd.to_datetime(df[tcol])
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)          # local market clock
    df["timestamps"] = ts
    out = pd.DataFrame({
        "timestamps": df["timestamps"],
        "open": df["Open"].astype("float64"),
        "high": df["High"].astype("float64"),
        "low": df["Low"].astype("float64"),
        "close": df["Close"].astype("float64"),
        "volume": df["Volume"].astype("float64"),
    })
    typical = out[["open", "high", "low", "close"]].mean(axis=1)
    out["amount"] = out["volume"] * typical                                  # derived, not exchange-reported
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return out[COLS]


SPECS = [
    ("btc_1h",       "binance", dict(symbol="BTCUSDT", interval="1h", limit=1000)),
    ("btc_1d",       "binance", dict(symbol="BTCUSDT", interval="1d", limit=1000)),
    ("nifty50_1h",   "yahoo",   dict(symbol="^NSEI", interval="1h", period="730d")),
    ("nifty50_1d",   "yahoo",   dict(symbol="^NSEI", interval="1d", period="5y")),
    ("reliance_1d",  "yahoo",   dict(symbol="RELIANCE.NS", interval="1d", period="5y")),
    ("reliance_1h",  "yahoo",   dict(symbol="RELIANCE.NS", interval="1h", period="730d")),
]

if __name__ == "__main__":
    manifest = {}
    for name, src, kw in SPECS:
        try:
            df = fetch_binance(**kw) if src == "binance" else fetch_yahoo(**kw)
            path = os.path.join(OUT, f"{name}.csv")
            df.to_csv(path, index=False)
            manifest[name] = {
                "source": src, "params": kw, "rows": len(df), "path": path,
                "first": str(df["timestamps"].iloc[0]), "last": str(df["timestamps"].iloc[-1]),
                "zero_volume_pct": round(float((df["volume"] == 0).mean() * 100), 2),
            }
            print(f"OK   {name:14s} {len(df):5d} rows  {df['timestamps'].iloc[0]} -> {df['timestamps'].iloc[-1]}"
                  f"  zero-vol {manifest[name]['zero_volume_pct']}%")
        except Exception as e:
            manifest[name] = {"source": src, "params": kw, "error": f"{type(e).__name__}: {e}"}
            print(f"FAIL {name:14s} {type(e).__name__}: {str(e)[:160]}", file=sys.stderr)
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nmanifest -> {os.path.join(OUT, 'manifest.json')}")
