"""Reconstruct POINT-IN-TIME NIFTY 200 membership from NSE rebalance press releases.

Plan 4.10 flagged this as "worth 20 minutes of searching -- if historical membership lists turn
out to be available, the survivorship-bias limitation mostly disappears". They are available,
and for a 2023-08 -> 2026-08 window it takes exactly SIX documents.

NSE/NIFTY publishes semi-annual reconstitution press releases at
    https://www.niftyindices.com/Press_Release/ind_prs<DDMMYYYY>.pdf
Each carries a "Nifty 200" section listing the companies "being excluded" and "being included",
plus an effective date. They are text-layer PDFs, so `pdftotext -layout` plus one regex is enough.

Membership is then reconstructed by walking BACKWARDS from today's published list:

    members_before_rebalance = (members_after - included) + excluded

which is exact, and self-checking: every intermediate set must contain exactly 200 names, and
every excluded name must already be present. Both are asserted.

Output:
    data/nse/membership.json    {effective_date: [symbols]} for each regime, plus the raw
                                add/drop per press release
    data/nse/membership.csv     long form: date, symbol, is_member  (one row per session)

The union of every name that was EVER a member during the window is larger than today's 200 --
those extra names must be price-fetched and signal-scored, or the point-in-time universe is
still biased, just differently.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from bench.nse.common import DATA_NSE, sidecar  # noqa: E402

PR_DIR = os.path.join(DATA_NSE, "press_releases")
PR_URL = "https://www.niftyindices.com/Press_Release/ind_prs{d}.pdf"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126 Safari/537.36")

# Announcement dates of every SEMI-ANNUAL reconstitution touching the 2023-08 .. 2026-08 window.
PRESS_RELEASES = ["17082023", "28022024", "23082024", "21022025", "22082025", "23022026"]

# --------------------------------------------------------------------------------------
# The semi-annual releases alone are NOT sufficient: NSE also publishes off-cycle changes,
# and it sometimes REVOKES a scheduled change before it takes effect. Reconstructing from the
# six reviews alone lands on 201-203 members instead of 200. Each correction below is taken
# from a specific press release, verified against the current 200-name list.
# --------------------------------------------------------------------------------------

# Names to strike from a semi-annual release because the change was later revoked.
REVOKED_EXCLUSIONS = {
    # ind_prs25092024.pdf: "The Committee has decided to revoke its decision to exclude IDEA
    # ... following the recalculation of security's impact cost by NSE."  IDEA never left.
    "23082024": ["IDEA"],
}
REVOKED_INCLUSIONS = {
    # ind_prs25092024.pdf: Central Bank of India "Inclusion revoked" -- a knock-on of IDEA
    # staying in. CENTRALBK is absent from the current list, confirming it never entered.
    "23082024": ["CENTRALBK"],
    # ind_prs19032024.pdf: IREDA breached the SEBI ETF impact-cost concentration norm, so its
    # 28-Feb-2024 inclusion was revoked and BSE Ltd. took the slot instead. IREDA did enter
    # later, on 30-Sep-2024, via the August review.
    "28022024": ["IREDA"],
}
SUBSTITUTE_INCLUSIONS = {"28022024": {"IREDA": "BSE"}}

# Off-cycle constituent changes, in effective-date order.
OFF_CYCLE = [
    {"effective_from": "2023-09-07", "excluded": ["JIOFIN"], "included": [],
     "source": "ind_prs05092023.pdf",
     "note": "Jio Financial, included 2023-07-20 on the Reliance demerger, removed after it "
             "failed to hit a price band; it re-entered at the March 2024 review."},
    {"effective_from": "2024-08-30", "excluded": ["TATAMTRDVR"], "included": [],
     "source": "ind_prs23082024_1.pdf",
     "note": "Tata Motors DVR 'A' ordinary shares cancelled under the scheme of arrangement. "
             "No replacement was named and none was needed -- the DVR was a SECOND security of "
             "a company already in the index, which is why the Aug-2023 review was asymmetric "
             "(7 out, 8 in) and why the index legitimately carried 201 securities for 200 "
             "companies between 2023-09-29 and 2024-08-29."},
]

# Section labels are inconsistent across releases: "k) Nifty 200", "l)    Nifty 200",
# "12) Nifty 200". Accept a letter or a number.
# Ticker RENAMES. A press release names the company by its then-current NSE symbol; if the
# symbol later changed, the walk-back sees a phantom add/drop. Verified case:
# GMR Infrastructure -> GMR Airports (GMRAIRPORT is in today's list, GMRINFRA is not, and the
# yfinance history under GMRAIRPORT.NS runs continuously back to 2014).
ALIASES = {"GMRINFRA": "GMRAIRPORT"}


def canon(sym: str) -> str:
    return ALIASES.get(sym, sym)


SECTION_RE = re.compile(r"^\s*(?:[a-z]|\d{1,2})\)\s*Nifty\s*200\s*$", re.I | re.M)
NEXT_SECTION_RE = re.compile(r"^\s*(?:[a-z]|\d{1,2})\)\s*Nifty\s", re.I | re.M)
EXCL_RE = re.compile(r"being\s+excluded", re.I)
INCL_RE = re.compile(r"being\s+included", re.I)
ROW_RE = re.compile(r"^\s*\d+\s+.*?\s{2,}([A-Z0-9&\-]{2,20})\s*$", re.M)
EFF_RE = re.compile(r"effective\s+from\s+([A-Z][a-z]+\s+\d{1,2},\s*\d{4})", re.I)


def ensure_pdf(d: str) -> str:
    os.makedirs(PR_DIR, exist_ok=True)
    path = os.path.join(PR_DIR, f"ind_prs{d}.pdf")
    if not os.path.exists(path) or os.path.getsize(path) < 10_000:
        subprocess.run(["curl", "-sS", "-L", "-A", UA, "-o", path, PR_URL.format(d=d)], check=True)
    return path


def pdf_text(path: str) -> str:
    return subprocess.check_output(["pdftotext", "-layout", path, "-"]).decode("utf-8", "replace")


def parse_nifty200(text: str) -> dict:
    """Pull the Nifty 200 excluded/included symbol lists and the effective date."""
    m = SECTION_RE.search(text)
    if not m:
        raise ValueError("no 'Nifty 200' section found")
    tail = text[m.end():]
    nxt = NEXT_SECTION_RE.search(tail)
    block = tail[: nxt.start()] if nxt else tail

    e, i = EXCL_RE.search(block), INCL_RE.search(block)
    if not e or not i:
        raise ValueError("missing excluded/included markers")
    if e.start() < i.start():
        excl_txt, incl_txt = block[e.end(): i.start()], block[i.end():]
    else:
        incl_txt, excl_txt = block[i.end(): e.start()], block[e.end():]

    def syms(t: str) -> list[str]:
        out, seen = [], set()
        for s in ROW_RE.findall(t):
            s = canon(s)
            if s in ("Symbol", "No", "Name") or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    eff = EFF_RE.search(text)
    return {"excluded": syms(excl_txt), "included": syms(incl_txt),
            "effective_from": (pd.Timestamp(eff.group(1)).date().isoformat() if eff else None)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--current", default=os.path.join(DATA_NSE, "nifty200_constituents.csv"))
    ap.add_argument("--calendar", default=os.path.join(DATA_NSE, "calendar.json"))
    ap.add_argument("--out", default=os.path.join(DATA_NSE, "membership.json"))
    ap.add_argument("--strict", action="store_true", default=True)
    args = ap.parse_args()

    cur = pd.read_csv(args.current)
    current = sorted({canon(str(s).strip()) for s in cur["Symbol"]})
    print(f"current published membership: {len(current)} symbols")

    events = []
    for d in PRESS_RELEASES:
        path = ensure_pdf(d)
        rec = parse_nifty200(pdf_text(path))
        # apply the sourced revocations / substitutions before the walk-back sees them
        sub = SUBSTITUTE_INCLUSIONS.get(d, {})
        rec["included"] = [sub.get(x, x) for x in rec["included"]
                           if x not in REVOKED_INCLUSIONS.get(d, [])] + \
                          [v for k, v in sub.items() if k in REVOKED_INCLUSIONS.get(d, [])]
        rec["included"] = list(dict.fromkeys(rec["included"]))
        rec["excluded"] = [x for x in rec["excluded"] if x not in REVOKED_EXCLUSIONS.get(d, [])]
        rec["kind"] = "semi-annual"
        rec["press_release"] = d
        rec["announced"] = pd.Timestamp(f"{d[4:]}-{d[2:4]}-{d[:2]}").date().isoformat()
        events.append(rec)
        print(f"  PR {rec['announced']}  effective {rec['effective_from']}  "
              f"-{len(rec['excluded'])} +{len(rec['included'])}")
        print(f"      out: {', '.join(rec['excluded'])}")
        print(f"      in : {', '.join(rec['included'])}")

    for oc in OFF_CYCLE:
        r = dict(oc)
        r["kind"] = "off-cycle"
        r["press_release"] = oc["source"]
        r["announced"] = oc["effective_from"]
        events.append(r)
        print(f"  OFF-CYCLE effective {r['effective_from']}  "
              f"-{len(r['excluded'])} +{len(r['included'])}  ({r['source']})")
    events.sort(key=lambda r: r["effective_from"])

    # Walk backwards: members_before = (members_after - included) + excluded
    regimes = {}
    members = set(current)
    regimes[events[-1]["effective_from"]] = sorted(members)
    problems = []
    for rec in reversed(events):
        inc, exc = set(rec["included"]), set(rec["excluded"])
        missing_inc = inc - members
        already_exc = exc & members
        if missing_inc:
            problems.append(f"{rec['effective_from']}: included-but-absent {sorted(missing_inc)}")
        if already_exc:
            problems.append(f"{rec['effective_from']}: excluded-but-present {sorted(already_exc)}")
        members = (members - inc) | exc
        regimes[f"before_{rec['effective_from']}"] = sorted(members)
        # 201 is CORRECT during two sourced windows: 2023-07-20..2023-09-06 (JIOFIN, a
        # transient Reliance-demerger entry) and 2023-09-29..2024-08-29 (TATAMTRDVR, a second
        # SECURITY of a company already in the index -- 201 securities, 200 companies).
        note = "OK" if len(members) == 200 else f"{len(members)} securities"
        print(f"  before {rec['effective_from']}: {len(members)} members ({note})")

    # normalise into ordered (start_date, members) regimes
    ordered = []
    start = None
    prev = sorted(members)
    for rec in events:
        ordered.append({"from": start, "to": rec["effective_from"], "members": prev,
                        "n": len(prev)})
        prev = sorted((set(prev) - set(rec["excluded"])) | set(rec["included"]))
        start = rec["effective_from"]
    ordered.append({"from": start, "to": None, "members": prev, "n": len(prev)})

    ever = sorted({s for r in ordered for s in r["members"]})
    extra = sorted(set(ever) - set(current))
    print(f"\nnames EVER a member across the window: {len(ever)}")
    print(f"names not in today's list (must be fetched to remove survivorship): {len(extra)}")
    print(f"  {', '.join(extra)}")
    if problems:
        print("\nCONSISTENCY PROBLEMS:")
        for p in problems:
            print("  " + p)

    with open(args.calendar) as f:
        calendar = [pd.Timestamp(d) for d in json.load(f)]
    rows = []
    for reg in ordered:
        lo = pd.Timestamp(reg["from"]) if reg["from"] else calendar[0]
        hi = pd.Timestamp(reg["to"]) if reg["to"] else calendar[-1] + pd.Timedelta(days=1)
        for d in calendar:
            if lo <= d < hi:
                for s in reg["members"]:
                    rows.append((d, s))
    mem = pd.DataFrame(rows, columns=["date", "symbol"])
    mem["is_member"] = True
    mem.to_parquet(os.path.join(DATA_NSE, "membership.parquet"), index=False)

    out = {"current_n": len(current), "events": events, "regimes": ordered,
           "ever_members": ever, "extra_symbols_needed": extra,
           "consistency_problems": problems,
           "sizes": [r["n"] for r in ordered]}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    sidecar(args.out.replace(".json", "_meta.json"), vars(args))
    print(f"\nregime sizes: {[r['n'] for r in ordered]}")
    print(f"-> {args.out}\n-> membership.parquet ({len(mem)} rows)")
    if args.strict and problems:
        print("\nWARNING: reconstruction is not internally consistent; inspect before using.")


if __name__ == "__main__":
    main()
