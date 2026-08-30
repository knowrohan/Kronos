"""Reconstruct POINT-IN-TIME NIFTY 200 / NIFTY 50 membership from NSE rebalance press releases.

Plan 4.10 flagged this as "worth 20 minutes of searching -- if historical membership lists turn
out to be available, the survivorship-bias limitation mostly disappears". They are available,
and for a 2023-08 -> 2026-08 window it takes exactly SIX documents for the 200 and eleven for
the 50 (the 50 turns over less often but attracts more off-cycle corporate-action traffic).

NSE/NIFTY publishes semi-annual reconstitution press releases at
    https://www.niftyindices.com/Press_Release/ind_prs<DDMMYYYY>.pdf
Each carries a "Nifty 200" and (when the 50 changed) a "Nifty 50" section listing the companies
"being excluded" and "being included", plus an effective date. They are text-layer PDFs, so
`pdftotext -layout` plus one regex is enough.

Membership is then reconstructed by walking BACKWARDS from today's published list:

    members_before_rebalance = (members_after - included) + excluded

which is exact, and self-checking: every intermediate set must contain the index's nominal size
in REAL names, and every excluded name must already be present. Both are asserted.

Outputs (NIFTY 200, unchanged):
    data/nse/membership.json    {effective_date: [symbols]} for each regime, plus the raw
                                add/drop per press release
    data/nse/membership.parquet long form: date, symbol, is_member  (one row per session)

Outputs (NIFTY 50, `--index nifty50`):
    data/nse/membership_pit_n50.parquet    date, symbol -- one row per session per member,
                                restricted to Window A and to TRADABLE names (see PLACEHOLDERS)
    data/nse/n50_membership_meta.json      regimes, events, sources, current-list as-of date,
                                ticker canonicalisation, missing-price exceptions, provenance
    data/nse/n50_membership_conflicts.json disagreements against the web-derived research JSON

The union of every name that was EVER a member during the window is larger than today's list --
those extra names must be price-fetched and signal-scored, or the point-in-time universe is
still biased, just differently.
"""
from __future__ import annotations

import argparse
import io
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
# nsearchives mirrors the releases that niftyindices.com drops from its own directory.
PR_URL_ARCHIVE = "https://nsearchives.nseindia.com/web/sites/default/files/{ym}/ind_prs{d}.pdf"
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

# --------------------------------------------------------------------------------------
# NIFTY 50 (NIFTY50_PLAN.md sec.2.1)
# --------------------------------------------------------------------------------------

# Every press release that carries a parseable "Nifty 50" section (or states, in that section,
# that the index is unchanged) over 2023-07 .. 2026-08. 04072023 is off-cycle but is laid out
# exactly like a review, so it is parsed rather than hand-encoded.
PRESS_RELEASES_N50 = ["04072023", "17082023", "15092023", "28022024", "23082024",
                      "21022025", "22082025", "23022026"]
# Which of those are periodic reviews. 04072023 is the HDFC/LTIM amalgamation replacement --
# same layout, but off-cycle, and the distinction is carried into the meta so a reader can see
# that the 50 turned over twice on schedule and four times on corporate actions.
SEMI_ANNUAL_N50 = {"17082023", "15092023", "28022024", "23082024", "21022025", "22082025",
                   "23022026"}

# Releases where the change is announced as a TABLE OF AFFECTED INDICES rather than an
# excluded/included section, so there is nothing for the section parser to read. Each is
# verified against its own PDF by `verify_off_cycle`: effective date, index table row, symbol.
OFF_CYCLE_N50 = [
    {"effective_from": "2023-07-20", "excluded": [], "included": ["JIOFIN"],
     "source": "ind_prs17072023.pdf", "symbol_in_pdf": "Jio Financial Services",
     "note": "Reliance demerger: the spun-off entity enters as a TEMPORARY 51st constituent at "
             "a constant price (close of 2023-07-19). NSE ran the index at 51 securities until "
             "the exclusion below, which is why 2023-08-01..2023-09-06 is a sourced exception "
             "to the 50-name assertion."},
    {"effective_from": "2023-09-07", "excluded": ["JIOFIN"], "included": [],
     "source": "ind_prs05092023.pdf", "symbol_in_pdf": "JIOFIN",
     "note": "JIOFIN listed 2023-08-21 and did not hit the price band on 2023-09-04/05, so it "
             "was excluded from close of 2023-09-06. It re-entered permanently on 2025-03-28."},
    {"effective_from": "2025-10-14", "excluded": [], "included": ["DUMMYTATAM"],
     "source": "ind_prs07102025.pdf", "symbol_in_pdf": "DUMMYTATAM",
     "note": "Tata Motors CV demerger: the demerged entity enters AT ZERO PRICE WITHOUT DIVISOR "
             "ADJUSTMENT under the dummy symbol DUMMYTATAM (close of 2025-10-13). It is not a "
             "tradable security, so it is stripped from the output panel (see PLACEHOLDERS)."},
    {"effective_from": "2025-11-17", "excluded": ["TMCV"], "included": [],
     "source": "ind_prs13112025.pdf", "symbol_in_pdf": "TMCV",
     "note": "The DUMMYTATAM placeholder listed as TMCV on 2025-11-12 and did not hit the price "
             "band on 2025-11-12/13, so it was excluded from close of 2025-11-14. Same security "
             "as DUMMYTATAM, so TMCV is canonicalised onto it and the episode nets to zero."},
]

# The panel (`data/nse/panel_price_pit.parquet`) is keyed on the symbol yfinance serves TODAY,
# which is the symbol NSE uses today. A press release names the symbol of the day, so a later
# rename makes the walk-back invent a phantom add/drop. Each mapping below is anchored on an
# identifier that survives the rename:
#   ZOMATO -> ETERNAL       renamed 2025-04-09 (NSE circular CML67420); ETERNAL's panel history
#                           runs back to the 2021-07-23 Zomato IPO.
#   TATAMOTORS -> TMPV      renamed 2025-10-24 on the CV demerger; TMPV's panel history runs the
#                           full calendar (2014-01-01), i.e. it IS the old Tata Motors line.
#   LTIM -> LTM             LTIMindtree; `nifty200_constituents.csv` carries ISIN INE214T01019
#                           under "LTM Ltd.", and the panel's LTM starts 2016-07-21 (the L&T
#                           Infotech IPO). The research JSON still says "LTIM", so this rename
#                           is written to the conflicts file as an informational disagreement.
#   TMCV -> DUMMYTATAM      not a rename but an identity: ind_prs13112025.pdf excludes "Tata
#                           Motors Ltd. (formerly known as TML Commercial Vehicles Ltd.)", the
#                           security that entered as DUMMYTATAM. Collapsing them keeps the
#                           placeholder episode a single security that nets to zero.
ALIASES_N50 = {"ZOMATO": "ETERNAL", "TATAMOTORS": "TMPV", "LTIM": "LTM", "TMCV": "DUMMYTATAM"}

# Index members that are not securities anyone can buy. They count towards the published
# constituent count but are excluded from the tradable universe the book is scored on.
PLACEHOLDERS_N50 = {"DUMMYTATAM"}

# Windows where the number of REAL (non-placeholder) names is legitimately not 50, each with the
# primary source that makes it so. Anything outside this list is a reconstruction bug.
SIZE_EXCEPTIONS_N50 = [
    {"from": "2023-07-20", "to": "2023-09-07", "n_real": 51,
     "source": "ind_prs17072023.pdf + ind_prs05092023.pdf",
     "reason": "JIOFIN was carried as a temporary 51st constituent from the Reliance demerger "
               "until it failed the price-band test; NSE published the index with 51 names."},
]

INDEX_SPECS = {
    "nifty200": {
        "index_name": "Nifty 200",
        "press_releases": PRESS_RELEASES,
        "off_cycle": OFF_CYCLE,
        "aliases": ALIASES,
        "placeholders": frozenset(),
        "nominal_n": 200,
        "current_csv": os.path.join(DATA_NSE, "nifty200_constituents.csv"),
        "out_prefix": os.path.join(DATA_NSE, "membership"),
    },
    "nifty50": {
        "index_name": "Nifty 50",
        "press_releases": PRESS_RELEASES_N50,
        "off_cycle": OFF_CYCLE_N50,
        "aliases": ALIASES_N50,
        "placeholders": PLACEHOLDERS_N50,
        "nominal_n": 50,
        "current_csv": os.path.join(DATA_NSE, "nifty50_constituents.csv"),
        "out_prefix": os.path.join(DATA_NSE, "n50_membership"),
    },
}

# Live constituent lists. niftyindices.com is the publisher; archives.nseindia.com mirrors the
# same bytes and is the fallback when the publisher 403s a scripted client.
CONSTITUENT_URLS = {
    "nifty200": ["https://niftyindices.com/IndexConstituent/ind_nifty200list.csv",
                 "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"],
    "nifty50": ["https://niftyindices.com/IndexConstituent/ind_nifty50list.csv",
                "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"],
}


def canon(sym: str, aliases: dict | None = None) -> str:
    """Map a press-release symbol onto the symbol the price panel uses today.

    Default is the NIFTY 200 alias table so the 2026-08-27 reconstruction is reproduced by
    calling `canon(s)` with no second argument, exactly as before.
    """
    return (ALIASES if aliases is None else aliases).get(sym, sym)


SECTION_TMPL = r"^\s*(?:[a-z]|\d{{1,2}})\)\s*{name}\s*$"
NEXT_SECTION_RE = re.compile(r"^\s*(?:[a-z]|\d{1,2})\)\s*Nifty\s", re.I | re.M)
EXCL_RE = re.compile(r"being\s+excluded", re.I)
INCL_RE = re.compile(r"being\s+included", re.I)
ROW_RE = re.compile(r"^\s*\d+\s+.*?\s{2,}([A-Z0-9&\-]{2,20})\s*$", re.M)
EFF_RE = re.compile(r"effective\s+from\s+([A-Z][a-z]+\s+\d{1,2},\s*\d{4})", re.I)
NO_CHANGE_RE = re.compile(r"no\s+changes?\s+(?:are|is)\s+being\s+made", re.I)


def section_re(index_name: str) -> re.Pattern:
    """Heading matcher for one index, tolerant of 'a)', 'l)   ' and '12)' numbering.

    Anchored at end of line so "Nifty 50" never swallows "Nifty 500", "Nifty50 Equal Weight"
    or "Nifty Midcap 50" -- the three ways this parser could silently read the wrong index.
    """
    return re.compile(SECTION_TMPL.format(name=re.escape(index_name).replace(r"\ ", r"\s*")),
                      re.I | re.M)


def ensure_pdf(d: str) -> str:
    """Download a press release once and keep it; a cached PDF is the primary source of record.

    niftyindices.com drops older releases from its own directory, so a 404 there is retried
    against the nsearchives mirror under the release's own year-month before giving up. The
    size floor is what makes the retry fire at all: a 403 or an error page is a small HTML
    body that `curl -o` writes to the destination just as happily as a PDF.
    """
    os.makedirs(PR_DIR, exist_ok=True)
    path = os.path.join(PR_DIR, f"ind_prs{d}.pdf")

    def ok() -> bool:
        return os.path.exists(path) and os.path.getsize(path) >= 10_000

    if not ok():
        subprocess.run(["curl", "-sS", "-L", "-A", UA, "-o", path, PR_URL.format(d=d)],
                       check=False)
    if not ok():
        ym = f"{d[4:]}-{d[2:4]}"
        subprocess.run(["curl", "-sS", "-L", "-A", UA, "-o", path,
                        PR_URL_ARCHIVE.format(ym=ym, d=d)], check=False)
    if not ok():
        raise FileNotFoundError(f"could not fetch ind_prs{d}.pdf from niftyindices or nsearchives")
    return path


def pdf_text(path: str) -> str:
    return subprocess.check_output(["pdftotext", "-layout", path, "-"]).decode("utf-8", "replace")


def parse_index_section(text: str, index_name: str = "Nifty 200",
                        aliases: dict | None = None, missing_ok: bool = False) -> dict | None:
    """Pull one index's excluded/included symbol lists and the release's effective date.

    `missing_ok` exists because "no section" and "an empty section" are both real, and they mean
    the same thing: the index did not change at that review. The Sep-2023 releases carry no
    Nifty 50 heading at all; ind_prs23022026.pdf carries the heading and says "No changes are
    being made". Treating a missing heading as an error would have forced a hand-coded
    exception per zero-churn review, which is exactly the sort of unsourced constant this
    reconstruction exists to avoid.
    """
    m = section_re(index_name).search(text)
    if not m:
        if missing_ok:
            eff = EFF_RE.search(text)
            return {"excluded": [], "included": [], "no_section": True,
                    "effective_from": (pd.Timestamp(eff.group(1)).date().isoformat()
                                       if eff else None)}
        raise ValueError(f"no '{index_name}' section found")
    tail = text[m.end():]
    nxt = NEXT_SECTION_RE.search(tail)
    block = tail[: nxt.start()] if nxt else tail

    e, i = EXCL_RE.search(block), INCL_RE.search(block)
    if not e or not i:
        if NO_CHANGE_RE.search(block):
            eff = EFF_RE.search(text)
            return {"excluded": [], "included": [], "no_change": True,
                    "effective_from": (pd.Timestamp(eff.group(1)).date().isoformat()
                                       if eff else None)}
        if missing_ok:
            return None
        raise ValueError("missing excluded/included markers")
    if e.start() < i.start():
        excl_txt, incl_txt = block[e.end(): i.start()], block[i.end():]
    else:
        incl_txt, excl_txt = block[i.end(): e.start()], block[e.end():]

    def syms(t: str) -> list[str]:
        out, seen = [], set()
        for s in ROW_RE.findall(t):
            s = canon(s, aliases)
            if s in ("Symbol", "No", "Name") or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    eff = EFF_RE.search(text)
    return {"excluded": syms(excl_txt), "included": syms(incl_txt),
            "effective_from": (pd.Timestamp(eff.group(1)).date().isoformat() if eff else None)}


def parse_nifty200(text: str) -> dict:
    """Back-compatible alias: the NIFTY 200 section with the NIFTY 200 alias table."""
    return parse_index_section(text, "Nifty 200", ALIASES)


def verify_off_cycle(rec: dict, index_name: str) -> list[str]:
    """Check a hand-encoded off-cycle event against its own PDF; return the failures.

    Hand-encoded events are the weak link in a reconstruction that is otherwise machine-read
    from primary documents, so each one is re-derived from the PDF it cites: the effective date
    must be the one the release states, the release must list this index among the affected
    indices, and the security must be named. A silent typo here would move a regime boundary.
    """
    path = os.path.join(PR_DIR, rec["source"])
    if not os.path.exists(path):
        return [f"{rec['effective_from']}: source {rec['source']} not on disk"]
    text = pdf_text(path)
    bad = []
    # An exclusion release recaps the inclusion it reverses, so the FIRST "effective from" in
    # the text is often the older date. Accept the declared date if the release states it
    # anywhere, and print every date it does state when it does not.
    got = sorted({pd.Timestamp(x).date().isoformat() for x in EFF_RE.findall(text)})
    if rec["effective_from"] not in got:
        bad.append(f"{rec['effective_from']}: {rec['source']} states effective_from={got}")
    if not re.search(rf"^\s*\d+\s+{re.escape(index_name)}\s*$", text, re.I | re.M):
        bad.append(f"{rec['effective_from']}: {rec['source']} does not list '{index_name}'")
    needle = rec.get("symbol_in_pdf")
    if needle and needle.lower() not in text.lower():
        bad.append(f"{rec['effective_from']}: {rec['source']} does not name '{needle}'")
    return bad


def walk_back(events: list[dict], current: list[str]) -> tuple[list[dict], list[str], dict]:
    """members_before = (members_after - included) + excluded, from today back to the start.

    Exact and self-checking: an included name that is absent from the later set, or an excluded
    name already absent, means a press release was missed or a symbol was renamed under us.
    Returns the ordered forward regimes, the consistency problems, and the per-step sizes.
    """
    members = set(current)
    problems, sizes = [], {}
    for rec in reversed(events):
        inc, exc = set(rec["included"]), set(rec["excluded"])
        missing_inc = inc - members
        already_exc = exc & members
        if missing_inc:
            problems.append(f"{rec['effective_from']}: included-but-absent {sorted(missing_inc)}")
        if already_exc:
            problems.append(f"{rec['effective_from']}: excluded-but-present {sorted(already_exc)}")
        members = (members - inc) | exc
        sizes[f"before_{rec['effective_from']}"] = len(members)

    ordered = []
    start = None
    prev = sorted(members)
    for rec in events:
        ordered.append({"from": start, "to": rec["effective_from"], "members": prev,
                        "n": len(prev)})
        prev = sorted((set(prev) - set(rec["excluded"])) | set(rec["included"]))
        start = rec["effective_from"]
    ordered.append({"from": start, "to": None, "members": prev, "n": len(prev)})
    return ordered, problems, sizes


def expand_calendar(ordered: list[dict], calendar: list, start=None, end=None) -> pd.DataFrame:
    """One (date, symbol) row per exchange session per member, over [start, end]."""
    lo_lim = pd.Timestamp(start) if start else None
    hi_lim = pd.Timestamp(end) if end else None
    rows = []
    for reg in ordered:
        lo = pd.Timestamp(reg["from"]) if reg["from"] else calendar[0]
        hi = pd.Timestamp(reg["to"]) if reg["to"] else calendar[-1] + pd.Timedelta(days=1)
        for d in calendar:
            if lo_lim is not None and d < lo_lim:
                continue
            if hi_lim is not None and d > hi_lim:
                continue
            if lo <= d < hi:
                for s in reg["members"]:
                    rows.append((d, s))
    out = pd.DataFrame(rows, columns=["date", "symbol"])
    # Pin nanoseconds. pandas >= 2.2 infers datetime64[us] from a list of Timestamps, but
    # every other artifact in data/nse (panel_price_pit, membership_pit) is datetime64[ns],
    # and a us/ns merge key is the kind of mismatch that silently selects zero rows.
    out["date"] = out["date"].astype("datetime64[ns]")
    return out


# --------------------------------------------------------------------------------------


def run_nifty200(args, spec) -> None:
    """The 2026-08-27 reconstruction, unchanged: same parse, same walk-back, same artifacts."""
    cur = pd.read_csv(args.current)
    current = sorted({canon(str(s).strip()) for s in cur["Symbol"]})
    print(f"current published membership: {len(current)} symbols")

    events = []
    for d in spec["press_releases"]:
        path = ensure_pdf(d)
        rec = parse_index_section(pdf_text(path), spec["index_name"], spec["aliases"])
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

    for oc in spec["off_cycle"]:
        r = dict(oc)
        r["kind"] = "off-cycle"
        r["press_release"] = oc["source"]
        r["announced"] = oc["effective_from"]
        events.append(r)
        print(f"  OFF-CYCLE effective {r['effective_from']}  "
              f"-{len(r['excluded'])} +{len(r['included'])}  ({r['source']})")
    events.sort(key=lambda r: r["effective_from"])

    ordered, problems, sizes = walk_back(events, current)
    # 201 is CORRECT during two sourced windows: 2023-07-20..2023-09-06 (JIOFIN, a
    # transient Reliance-demerger entry) and 2023-09-29..2024-08-29 (TATAMTRDVR, a second
    # SECURITY of a company already in the index -- 201 securities, 200 companies).
    for rec in reversed(events):
        n = sizes[f"before_{rec['effective_from']}"]
        note = "OK" if n == spec["nominal_n"] else f"{n} securities"
        print(f"  before {rec['effective_from']}: {n} members ({note})")

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
    mem = expand_calendar(ordered, calendar, args.start, args.end)
    mem["is_member"] = True
    mem.to_parquet(args.pit_out, index=False)

    out = {"current_n": len(current), "events": events, "regimes": ordered,
           "ever_members": ever, "extra_symbols_needed": extra,
           "consistency_problems": problems,
           "sizes": [r["n"] for r in ordered]}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    sidecar(args.meta, vars(args))
    print(f"\nregime sizes: {[r['n'] for r in ordered]}")
    print(f"-> {args.out}\n-> {args.pit_out} ({len(mem)} rows)")
    if args.strict and problems:
        print("\nWARNING: reconstruction is not internally consistent; inspect before using.")


def fetch_current_list(index: str, path: str) -> dict:
    """Download today's published constituent list; fall back, then snapshot, then research JSON.

    The as-of date matters more than the list: the walk-back is anchored on it, so a list
    fetched a day after an effective date reconstructs a different history. Whatever source
    ends up being used is recorded, because "we could not reach the publisher" is a fact the
    reader of the membership file needs.
    """
    import urllib.request
    for url in CONSTITUENT_URLS[index]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read().decode("utf-8-sig", "replace")
            df = pd.read_csv(io.StringIO(body))
            df.columns = [c.strip() for c in df.columns]
            df.to_csv(path, index=False)
            print(f"current list: {len(df)} rows from {url} -> {path}")
            return {"source": url, "kind": "primary", "n": len(df)}
        except Exception as exc:  # noqa: BLE001
            print(f"  constituent fetch failed ({url}): {exc}")
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"current list: fetch failed; using snapshot {path} ({len(df)} rows)")
        return {"source": path, "kind": "snapshot", "n": len(df)}
    return {"source": None, "kind": "research-json", "n": 0}


def run_nifty50(args, spec) -> None:
    """Walk the NIFTY 50 back from today's published list and cross-validate it (plan sec.2.1)."""
    aliases = spec["aliases"]
    holes = spec["placeholders"]

    fetch = {"source": args.current, "kind": "snapshot"}
    if args.fetch_current:
        fetch = fetch_current_list("nifty50", args.current)
    with open(args.research) as f:
        research = json.load(f)
    if fetch["kind"] == "research-json":
        current_raw = list(research["current_constituents"]["symbols"])
        current_asof = research["current_constituents"]["as_of"]
        print(f"current list: unreachable; using the research JSON ({len(current_raw)} symbols)")
    else:
        cur = pd.read_csv(args.current)
        current_raw = [str(s).strip() for s in cur["Symbol"]]
        current_asof = pd.Timestamp(os.path.getmtime(args.current), unit="s").date().isoformat()
    current = sorted({canon(s, aliases) for s in current_raw})
    print(f"current published membership: {len(current)} symbols (as of {current_asof})")

    events, zero_churn = [], []
    for d in spec["press_releases"]:
        path = ensure_pdf(d)
        rec = parse_index_section(pdf_text(path), spec["index_name"], aliases, missing_ok=True)
        if rec is None:
            raise ValueError(f"ind_prs{d}.pdf: '{spec['index_name']}' section is unreadable")
        rec["kind"] = "semi-annual" if d in SEMI_ANNUAL_N50 else "off-cycle"
        rec["press_release"] = d
        rec["source"] = f"ind_prs{d}.pdf"
        rec["announced"] = pd.Timestamp(f"{d[4:]}-{d[2:4]}-{d[:2]}").date().isoformat()
        if rec["excluded"] or rec["included"]:
            events.append(rec)
            print(f"  PR {rec['announced']}  effective {rec['effective_from']}  "
                  f"-{len(rec['excluded'])} +{len(rec['included'])}  ({rec['source']})")
            print(f"      out: {', '.join(rec['excluded']) or '-'}")
            print(f"      in : {', '.join(rec['included']) or '-'}")
        else:
            zero_churn.append(rec)
            why = "no Nifty 50 section" if rec.get("no_section") else "states 'no changes'"
            print(f"  PR {rec['announced']}  effective {rec['effective_from']}  "
                  f"ZERO CHURN ({why}, {rec['source']})")

    verify_problems = []
    for oc in spec["off_cycle"]:
        verify_problems += verify_off_cycle(oc, spec["index_name"])
        r = dict(oc)
        r["excluded"] = [canon(x, aliases) for x in oc["excluded"]]
        r["included"] = [canon(x, aliases) for x in oc["included"]]
        r["kind"] = "off-cycle"
        r["press_release"] = oc["source"]
        r["announced"] = oc["effective_from"]
        events.append(r)
        print(f"  OFF-CYCLE effective {r['effective_from']}  "
              f"-{len(r['excluded'])} +{len(r['included'])}  ({r['source']}, PDF-verified)")
    events.sort(key=lambda r: r["effective_from"])
    zero_churn.sort(key=lambda r: r["effective_from"] or "")
    for p in verify_problems:
        print(f"  OFF-CYCLE VERIFY FAILED: {p}")

    ordered, problems, sizes = walk_back(events, current)
    exc_lut = {(e["from"], e["to"]): e for e in SIZE_EXCEPTIONS_N50}
    for reg in ordered:
        reg["n_real"] = len(set(reg["members"]) - holes)
        reg["placeholders"] = sorted(set(reg["members"]) & holes)
        allowed = exc_lut.get((reg["from"], reg["to"]))
        expect = allowed["n_real"] if allowed else spec["nominal_n"]
        reg["expected_n_real"] = expect
        if reg["n_real"] != expect:
            problems.append(f"{reg['from']}..{reg['to']}: {reg['n_real']} real names, "
                            f"expected {expect}")
        flag = "OK" if reg["n_real"] == expect else "MISMATCH"
        print(f"  regime {str(reg['from']):>10} .. {str(reg['to']):>10}  "
              f"{reg['n_real']} real +{len(reg['placeholders'])} placeholder  [{flag}]")
    problems += verify_problems
    if problems:
        print("\nCONSISTENCY PROBLEMS:")
        for p in problems:
            print("  " + p)
        if args.strict:
            # Fail BEFORE anything is written. Every problem above is known from the walk-back
            # alone, and a membership file that is wrong by one name is worse on disk than
            # absent: `hidden.py --cells n50` would read it and score a universe nobody chose.
            raise SystemExit("reconstruction is not internally consistent; "
                             "re-run with --no-strict to write it anyway and inspect")

    conflicts = cross_validate(events, zero_churn, current, current_asof, research, aliases)

    with open(args.calendar) as f:
        calendar = [pd.Timestamp(d) for d in json.load(f)]
    mem = expand_calendar(ordered, calendar, args.start, args.end)
    mem = mem[~mem["symbol"].isin(holes)].sort_values(["date", "symbol"]).reset_index(drop=True)
    mem.to_parquet(args.pit_out, index=False)

    coverage = price_coverage(mem, args.panel)
    per_date = mem.groupby("date")["symbol"].nunique()
    print(f"\nsessions {per_date.index.min().date()} .. {per_date.index.max().date()} "
          f"({len(per_date)}), members/date min={per_date.min()} max={per_date.max()}")
    print(f"missing price rows: {coverage['n_missing_cells']} cells over "
          f"{len(coverage['missing_by_symbol'])} names")
    print(f"member-dates inside the 90-session lookback: {coverage['n_ineligible_lookback']}")
    if conflicts:
        print(f"\n{len(conflicts)} disagreement(s) vs the research JSON -> {args.conflicts}")
        for c in conflicts:
            print(f"  [{c['severity']}] {c['kind']}: {c['detail']}")

    meta = {
        "index": "nifty50",
        "window": {"start": args.start, "end": args.end, "n_sessions": int(len(per_date))},
        "current_list": {
            "path": args.current, "as_of": current_asof, "n": len(current),
            "fetch": fetch, "symbols": current,
            "matches_research": current == sorted({canon(s, aliases) for s in
                                                   research["current_constituents"]["symbols"]}),
        },
        "regimes": ordered,
        "walk_back_sizes": sizes,
        "events": events,
        "zero_churn_reviews": zero_churn,
        "size_exceptions": SIZE_EXCEPTIONS_N50,
        "placeholders": {"symbols": sorted(holes),
                         "reason": "zero-price demerger placeholders; index constituents but "
                                   "not tradable securities, so they are excluded from "
                                   "membership_pit_n50.parquet"},
        "canonicalisation": {
            "aliases": aliases,
            "panel": args.panel,
            "note": "press-release symbols are mapped onto the symbol data/nse/"
                    "panel_price_pit.parquet uses; see ALIASES_N50 for the evidence per name",
        },
        "price_coverage": coverage,
        "consistency_problems": problems,
        "conflicts_path": args.conflicts,
        "n_conflicts": len(conflicts),
        "regime_sizes": [r["n_real"] for r in ordered],
        "members_per_date": {"min": int(per_date.min()), "max": int(per_date.max()),
                             "median": float(per_date.median())},
    }
    with open(args.conflicts, "w") as f:
        json.dump(conflicts, f, indent=2)
    sidecar(args.meta, vars(args), extra=meta)
    print(f"\nregime sizes (real names): {[r['n_real'] for r in ordered]}")
    print(f"-> {args.pit_out} ({len(mem)} rows)\n-> {args.meta}\n-> {args.conflicts}")


def cross_validate(events, zero_churn, current, current_asof, research, aliases) -> list[dict]:
    """Diff the PDF-derived change set against the web-derived research JSON.

    The PDFs are primary and always win; the JSON is a second pair of eyes whose only job is to
    catch a press release we never found. Every disagreement is written out rather than
    reconciled in silence, and severity separates "the history is different" (error) from
    "the citation is different" (info).
    """
    out = []
    res_by_date, res_src = {}, {}
    for ch in research["changes"]:
        d = ch["effective_date"]
        res_by_date[d] = (frozenset(canon(x, aliases) for x in ch["removed"]),
                          frozenset(canon(x, aliases) for x in ch["added"]))
        res_src[d] = ch.get("source", "")
    pdf_by_date, pdf_src = {}, {}
    for e in events:
        d = e["effective_from"]
        exc, inc = pdf_by_date.get(d, (frozenset(), frozenset()))
        pdf_by_date[d] = (exc | frozenset(e["excluded"]), inc | frozenset(e["included"]))
        pdf_src.setdefault(d, []).append(e["source"])
    for z in zero_churn:
        d = z["effective_from"]
        pdf_by_date.setdefault(d, (frozenset(), frozenset()))
        pdf_src.setdefault(d, []).append(z["source"])

    for d in sorted(set(res_by_date) | set(pdf_by_date)):
        r, p = res_by_date.get(d), pdf_by_date.get(d)
        if r is None:
            out.append({"kind": "date_only_in_press_release", "severity": "error",
                        "effective_date": d, "press_release": pdf_src.get(d),
                        "detail": f"{d}: the PDFs carry -{sorted(p[0])} +{sorted(p[1])} but the "
                                  "research JSON has no event on that date"})
            continue
        if p is None:
            out.append({"kind": "date_only_in_research", "severity": "error",
                        "effective_date": d, "research_source": res_src.get(d),
                        "detail": f"{d}: the research JSON carries -{sorted(r[0])} "
                                  f"+{sorted(r[1])} but no press release on disk says so"})
            continue
        if r != p:
            out.append({"kind": "change_set_mismatch", "severity": "error",
                        "effective_date": d, "press_release": pdf_src.get(d),
                        "research_source": res_src.get(d),
                        "detail": f"{d}: PDFs say -{sorted(p[0])} +{sorted(p[1])}, research says "
                                  f"-{sorted(r[0])} +{sorted(r[1])}"})
            continue
        cited = os.path.basename(res_src.get(d, "").rstrip("/"))
        if cited and cited not in pdf_src.get(d, []):
            checked = ""
            local = os.path.join(PR_DIR, cited)
            if os.path.exists(local):
                has = bool(section_re("Nifty 50").search(pdf_text(local)))
                checked = (f"{cited} is on disk and does "
                           f"{'' if has else 'NOT '}carry a Nifty 50 section")
            out.append({"kind": "source_url_mismatch", "severity": "info",
                        "effective_date": d, "press_release": pdf_src.get(d),
                        "research_source": res_src.get(d), "checked": checked,
                        "detail": f"{d}: same change set, but the research JSON cites {cited} "
                                  f"while the primary evidence is {pdf_src.get(d)}"})

    res_cur = sorted({canon(s, aliases) for s in research["current_constituents"]["symbols"]})
    if res_cur != current:
        out.append({"kind": "current_list_mismatch", "severity": "error",
                    "as_of_fetched": current_asof,
                    "as_of_research": research["current_constituents"]["as_of"],
                    "detail": f"fetched-only {sorted(set(current) - set(res_cur))}, "
                              f"research-only {sorted(set(res_cur) - set(current))}"})
    known_renames = {r["old"]: r["new"] for r in research.get("renames", [])}
    for old, new in aliases.items():
        if old in known_renames or old in PLACEHOLDERS_N50 or new in PLACEHOLDERS_N50:
            continue
        out.append({"kind": "rename_missing_from_research", "severity": "info",
                    "detail": f"{old} -> {new} is applied here but the research JSON does not "
                              "list it; see ALIASES_N50 for the identifier the mapping is "
                              "anchored on"})
    return out


def price_coverage(mem: pd.DataFrame, panel_path: str) -> dict:
    """Which member-dates have no price row, and which sit inside the 90-session lookback.

    A member with no price is a hole in the point-in-time universe and has to be named; a
    member with a price but fewer than 90 prior sessions is merely INELIGIBLE under the
    record's lookback rule, which is expected for a recent listing and is counted, not flagged.
    """
    if not os.path.exists(panel_path):
        return {"panel": panel_path, "available": False}
    panel = pd.read_parquet(panel_path, columns=["date", "symbol"])
    panel["date"] = pd.to_datetime(panel["date"])
    have = panel.assign(_has=True)
    j = mem.merge(have, on=["date", "symbol"], how="left")
    missing = j[j["_has"].isna()]
    by_sym = {}
    for s, g in missing.groupby("symbol"):
        by_sym[str(s)] = {"n": int(len(g)), "first": g["date"].min().date().isoformat(),
                          "last": g["date"].max().date().isoformat()}
    rank = panel.sort_values(["symbol", "date"]).copy()
    rank["n_prior"] = rank.groupby("symbol").cumcount()
    j2 = mem.merge(rank, on=["date", "symbol"], how="left")
    inel = j2[j2["n_prior"] < 90]
    by_sym_inel = {str(s): int(len(g)) for s, g in inel.groupby("symbol")}
    return {"panel": panel_path, "available": True,
            "n_member_cells": int(len(mem)),
            "n_missing_cells": int(len(missing)),
            "missing_by_symbol": by_sym,
            "n_ineligible_lookback": int(len(inel)),
            "ineligible_by_symbol": by_sym_inel,
            "panel_symbols": int(panel["symbol"].nunique()),
            "members_not_in_panel": sorted(set(mem["symbol"]) - set(panel["symbol"]))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", choices=sorted(INDEX_SPECS), default="nifty200")
    ap.add_argument("--current", default=None, help="published constituent CSV (index default)")
    ap.add_argument("--calendar", default=os.path.join(DATA_NSE, "calendar.json"))
    ap.add_argument("--out-prefix", dest="out_prefix", default=None,
                    help="artifact stem; <p>.json, <p>.parquet, <p>_meta.json")
    ap.add_argument("--out", default=None, help="regime JSON path (overrides --out-prefix)")
    ap.add_argument("--pit-out", dest="pit_out", default=None, help="membership parquet path")
    ap.add_argument("--conflicts", default=None, help="nifty50: cross-validation report path")
    ap.add_argument("--research", default=None, help="nifty50: research change-set JSON")
    ap.add_argument("--panel", default=os.path.join(DATA_NSE, "panel_price_pit.parquet"))
    ap.add_argument("--start", default=None, help="first session to emit (default: calendar)")
    ap.add_argument("--end", default=None, help="last session to emit (default: calendar)")
    ap.add_argument("--fetch-current", dest="fetch_current", action="store_true",
                    help="nifty50: re-download the published constituent list first")
    ap.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True,
                    help="refuse to write an internally inconsistent reconstruction")
    args = ap.parse_args()

    spec = INDEX_SPECS[args.index]
    args.current = args.current or spec["current_csv"]
    prefix = args.out_prefix or spec["out_prefix"]
    args.meta = f"{prefix}_meta.json"
    if args.index == "nifty50":
        # Window A and the tradable-panel name are fixed by NIFTY50_PLAN.md sec.2.1, so they
        # are defaults rather than something a caller has to remember.
        args.pit_out = args.pit_out or os.path.join(DATA_NSE, "membership_pit_n50.parquet")
        args.conflicts = args.conflicts or os.path.join(DATA_NSE,
                                                        "n50_membership_conflicts.json")
        args.research = args.research or os.path.join(DATA_NSE, "research",
                                                      "n50_changes_2023_2026.json")
        args.start = args.start or "2023-08-01"
        args.end = args.end or "2026-08-25"
        run_nifty50(args, spec)
    else:
        args.out = args.out or f"{prefix}.json"
        args.pit_out = args.pit_out or f"{prefix}.parquet"
        run_nifty200(args, spec)


if __name__ == "__main__":
    main()
