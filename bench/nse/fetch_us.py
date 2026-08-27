"""Build the same panel/calendar artifacts for a US large-cap universe.

EXTENSION BEYOND THE PLAN. The NSE result on its own cannot distinguish
"Kronos has no cross-sectional skill" from "Kronos has no cross-sectional skill *in India*".
A second, independent market answers that for ~15 minutes of GPU, because every downstream
script (signals / controls / grid / backtest) is already parameterised by --panel and
--calendar.

Universe: S&P 100 constituents -- ~100 liquid US large caps, fetched from Wikipedia with a
checked-in snapshot fallback so the run stays reproducible. Same price-return dividend
convention, same min-history filter. This universe is CURRENT membership and therefore carries
the same survivorship bias as the uncorrected NSE panel; that is acceptable here because the
diagnostic is an information coefficient, and the bias applies identically to Kronos and to
every baseline it is compared against.

Outputs (data/us/): panel_price.parquet, calendar.json, bench.parquet (^GSPC, ^NDX), universe.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from bench.nse.common import sidecar  # noqa: E402
from bench.nse.fetch_universe import _tidy  # noqa: E402

DATA_US = os.path.join(ROOT, "data", "us")
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126 Safari/537.36")

# The Nasdaq-100 and DJIA articles no longer carry a components table; the S&P 100 page
# does, and is the right size for a cheap cross-sectional diagnostic (~100 liquid names).
WIKI = {
    "sp100": ("https://en.wikipedia.org/wiki/S%26P_100", "Symbol"),
}


def constituents(cache: str) -> list[str]:
    syms: set[str] = set()
    for name, (url, col) in WIKI.items():
        try:
            import io as _io
            import requests
            r = requests.get(url, headers={"User-Agent": UA}, timeout=60)
            r.raise_for_status()
            tables = pd.read_html(_io.StringIO(r.text))
            for t in tables:
                cols = [str(c) for c in t.columns]
                if col in cols and len(t) > 20:
                    syms |= {str(s).strip().replace(".", "-") for s in t[col].dropna()}
                    print(f"  {name}: +{len(t)} rows")
                    break
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: FAILED {type(exc).__name__}: {exc}")
    syms = {s for s in syms if s and s.isascii() and 1 <= len(s) <= 6 and s.replace("-", "").isalpha()}
    if syms:
        with open(cache, "w") as f:
            json.dump(sorted(syms), f, indent=1)
        return sorted(syms)
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    raise RuntimeError("no constituents and no snapshot")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--out-dir", default=DATA_US)
    ap.add_argument("--min-rows", type=int, default=101)
    ap.add_argument("--min-universe", type=int, default=60)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import yfinance as yf
    syms = constituents(os.path.join(args.out_dir, "constituents.json"))
    print(f"universe: {len(syms)} tickers")

    frames, meta = [], {}
    for i in range(0, len(syms), 40):
        batch = syms[i:i + 40]
        raw = yf.download(batch, start=args.start, interval="1d", auto_adjust=False,
                          actions=True, progress=False, threads=True, group_by="ticker")
        for tk in batch:
            try:
                d = _tidy(raw, tk)
            except Exception:  # noqa: BLE001
                d = None
            if d is None or len(d) < args.min_rows:
                meta[tk] = {"ok": False, "rows": 0 if d is None else len(d)}
                print(f"  MISS {tk}")
                continue
            d.insert(1, "symbol", tk)
            frames.append(d)
            meta[tk] = {"ok": True, "rows": int(len(d)),
                        "first": str(d["date"].iloc[0].date()), "last": str(d["date"].iloc[-1].date())}

    panel = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    counts = panel.groupby("date")["symbol"].size()
    populated = set(counts[counts >= args.min_universe].index)

    bench_cols = {}
    for name, tk in (("spx", "^GSPC"), ("ndx", "^NDX")):
        braw = yf.download(tk, start=args.start, interval="1d", auto_adjust=False,
                           actions=True, progress=False, threads=False)
        b = _tidy(braw, tk)
        if b is None:
            print(f"  MISS bench {name}")
            continue
        bench_cols[name] = b.set_index("date")["close"]
        bench_cols[f"{name}_open"] = b.set_index("date")["open"]
        print(f"  bench {name} ({tk}) {len(b)} rows")
    bench = pd.DataFrame(bench_cols)
    calendar = sorted(set(bench.index) & populated)
    bench.reset_index().rename(columns={"index": "date"}).to_parquet(
        os.path.join(args.out_dir, "bench.parquet"), index=False)

    tidy = panel[panel["date"].isin(set(calendar))][
        ["date", "symbol", "open", "high", "low", "close", "volume"]]
    tidy.to_parquet(os.path.join(args.out_dir, "panel_price.parquet"), index=False)
    with open(os.path.join(args.out_dir, "calendar.json"), "w") as f:
        json.dump([str(pd.Timestamp(d).date()) for d in calendar], f)
    with open(os.path.join(args.out_dir, "universe.json"), "w") as f:
        json.dump(meta, f, indent=2)
    n_ok = sum(1 for v in meta.values() if v["ok"])
    sidecar(os.path.join(args.out_dir, "fetch_us_meta.json"), vars(args),
            extra={"symbols_ok": n_ok, "symbols": len(syms), "rows": int(len(tidy)),
                   "sessions": len(calendar)})
    print(f"\npanel: {tidy['symbol'].nunique()} symbols, {len(tidy)} rows, "
          f"{len(calendar)} sessions {calendar[0]} -> {calendar[-1]}")


if __name__ == "__main__":
    main()
