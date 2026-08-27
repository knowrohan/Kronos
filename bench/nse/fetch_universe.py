"""Fetch NIFTY 200 constituents + daily OHLCV panel + benchmarks.

Dividend convention (plan 4.6): we download with ``auto_adjust=False, actions=True`` so we
get BOTH conventions from one pass:

  * ``close``      -- split-adjusted, NOT dividend-adjusted  -> price return, consistent
                      with the ^CNX200 / ^NSEI price indices.
  * ``adj_close``  -- split- and dividend-adjusted           -> total return.

The panel stores both.  Downstream code selects one via ``--dividends {price,total}``
and the same choice is applied to strategy, benchmark and every control.
The default is ``price``: mixing a total-return stock series against a price-index
benchmark hands the strategy the whole dividend yield as fake alpha.

Outputs (under data/nse/):
    panel.parquet     date, symbol, open, high, low, close, adj_close, volume, dividends, splits
    bench.parquet     date, cnx200, nsei
    universe.json     per-symbol metadata
    fetch_log.txt     every failure / short history, nothing dropped silently
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from bench.nse.common import DATA_NSE, sidecar, utc_now  # noqa: E402

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}
CONSTITUENT_URL = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"

# yfinance symbol overrides for names whose NSE ticker does not map 1:1.
SYMBOL_OVERRIDES: dict[str, str] = {}


def fetch_constituents(cache_path: str) -> pd.DataFrame:
    """Download the current NIFTY 200 membership; snapshot it so the run is reproducible."""
    try:
        r = requests.get(CONSTITUENT_URL, headers=UA, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.to_csv(cache_path, index=False)
        print(f"constituents: {len(df)} rows from NSE archives -> {cache_path}")
    except Exception as exc:  # noqa: BLE001
        if not os.path.exists(cache_path):
            raise RuntimeError(f"constituent fetch failed and no snapshot at {cache_path}") from exc
        df = pd.read_csv(cache_path)
        print(f"constituents: NSE fetch failed ({exc}); using snapshot {cache_path} ({len(df)} rows)")
    df.columns = [c.strip() for c in df.columns]
    return df


def yf_symbol(nse_symbol: str) -> str:
    return SYMBOL_OVERRIDES.get(nse_symbol, f"{nse_symbol}.NS")


def _tidy(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Pull one ticker out of a (possibly multi-index) yfinance frame into tidy form."""
    if isinstance(raw.columns, pd.MultiIndex):
        # yfinance orients the MultiIndex as (ticker, field) with group_by="ticker"
        # and as (field, ticker) otherwise. Handle both.
        if ticker in raw.columns.get_level_values(0):
            sub = raw[ticker].copy()
        elif ticker in raw.columns.get_level_values(1):
            sub = raw.xs(ticker, axis=1, level=1).copy()
        else:
            return None
    else:
        sub = raw.copy()
    ren = {
        "Open": "open", "High": "high", "Low": "low", "Close": "close",
        "Adj Close": "adj_close", "Volume": "volume",
        "Dividends": "dividends", "Stock Splits": "splits",
    }
    sub = sub.rename(columns=ren)
    need = ["open", "high", "low", "close", "volume"]
    if not all(c in sub.columns for c in need):
        return None
    if "adj_close" not in sub.columns:
        sub["adj_close"] = sub["close"]
    for c in ("dividends", "splits"):
        if c not in sub.columns:
            sub[c] = 0.0
    sub = sub.reset_index().rename(columns={"Date": "date", "index": "date"})
    sub["date"] = pd.to_datetime(sub["date"])
    if getattr(sub["date"].dt, "tz", None) is not None:
        sub["date"] = sub["date"].dt.tz_localize(None)
    sub = sub[["date", "open", "high", "low", "close", "adj_close", "volume", "dividends", "splits"]]
    sub = sub.dropna(subset=["open", "high", "low", "close"])
    sub = sub[sub["close"] > 0]
    return sub.reset_index(drop=True) if len(sub) else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--end", default=None, help="exclusive end; default = today")
    ap.add_argument("--out-dir", default=DATA_NSE)
    ap.add_argument("--chunk", type=int, default=40, help="tickers per yfinance call")
    ap.add_argument("--min-rows", type=int, default=101,
                    help="drop symbols with fewer rows (lookback+predict+1 = 101)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    cons = fetch_constituents(os.path.join(args.out_dir, "nifty200_constituents.csv"))
    symbols = cons["Symbol"].astype(str).str.strip().tolist()
    industry = dict(zip(cons["Symbol"].astype(str).str.strip(), cons.get("Industry", pd.Series([""] * len(cons)))))

    tickers = [yf_symbol(s) for s in symbols]
    tick2sym = dict(zip(tickers, symbols))

    frames: list[pd.DataFrame] = []
    meta: dict[str, dict] = {}
    for i in range(0, len(tickers), args.chunk):
        batch = tickers[i:i + args.chunk]
        import yfinance as yf
        raw = yf.download(batch, start=args.start, end=args.end, interval="1d",
                          auto_adjust=False, actions=True, progress=False,
                          threads=True, group_by="ticker")
        for tk in batch:
            sym = tick2sym[tk]
            try:
                tdf = _tidy(raw, tk)
            except Exception as exc:  # noqa: BLE001
                tdf = None
                log(f"FAIL  {sym:<14s} ({tk}) tidy error: {type(exc).__name__}: {exc}")
            if tdf is None or len(tdf) == 0:
                log(f"MISS  {sym:<14s} ({tk}) yfinance returned no usable rows")
                meta[sym] = {"ticker": tk, "ok": False, "reason": "no_rows",
                             "industry": str(industry.get(sym, ""))}
                continue
            tdf.insert(1, "symbol", sym)
            nan_pct = float(tdf[["open", "high", "low", "close"]].isna().mean().mean() * 100)
            meta[sym] = {
                "ticker": tk, "ok": True, "industry": str(industry.get(sym, "")),
                "rows": int(len(tdf)),
                "first": str(tdf["date"].iloc[0].date()), "last": str(tdf["date"].iloc[-1].date()),
                "nan_pct": round(nan_pct, 4),
                "n_dividends": int((tdf["dividends"] > 0).sum()),
                "n_splits": int((tdf["splits"] > 0).sum()),
                "zero_volume_pct": round(float((tdf["volume"] == 0).mean() * 100), 3),
            }
            if len(tdf) < args.min_rows:
                log(f"SHORT {sym:<14s} ({tk}) only {len(tdf)} rows (< {args.min_rows}); dropped")
                meta[sym]["ok"] = False
                meta[sym]["reason"] = "short_history"
                continue
            frames.append(tdf)
        print(f"  chunk {i // args.chunk + 1}: {len(frames)} symbols kept so far", flush=True)

    panel = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    panel.to_parquet(os.path.join(args.out_dir, "panel.parquet"), index=False)
    log(f"panel: {panel['symbol'].nunique()} symbols, {len(panel)} rows, "
        f"{panel['date'].min().date()} -> {panel['date'].max().date()}")

    # --- benchmarks -------------------------------------------------------
    import yfinance as yf
    bench_frames = {}
    for name, tk in (("cnx200", "^CNX200"), ("nsei", "^NSEI")):
        braw = yf.download(tk, start=args.start, end=args.end, interval="1d",
                           auto_adjust=False, actions=True, progress=False, threads=False)
        bdf = _tidy(braw, tk)
        if bdf is None:
            log(f"FAIL  benchmark {name} ({tk}) returned nothing")
            continue
        bench_frames[name] = bdf.set_index("date")["close"]
        log(f"bench {name:<7s} ({tk}) {len(bdf)} rows {bdf['date'].iloc[0].date()} -> {bdf['date'].iloc[-1].date()}")
    bench = pd.DataFrame(bench_frames).reset_index()
    bench.to_parquet(os.path.join(args.out_dir, "bench.parquet"), index=False)

    with open(os.path.join(args.out_dir, "universe.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(args.out_dir, "fetch_log.txt"), "w") as f:
        f.write("\n".join(log_lines) + "\n")

    n_ok = sum(1 for v in meta.values() if v["ok"])
    sidecar(os.path.join(args.out_dir, "fetch_universe_meta.json"), vars(args), extra={
        "constituents": len(symbols), "symbols_ok": n_ok, "symbols_failed": len(symbols) - n_ok,
        "panel_rows": int(len(panel)), "wall_clock_sec": round(time.time() - t0, 1),
        "dividend_note": "panel carries both close (price-return) and adj_close (total-return)",
    })
    print(f"\ndone in {time.time() - t0:.0f}s: {n_ok}/{len(symbols)} symbols -> {args.out_dir}")


if __name__ == "__main__":
    main()
