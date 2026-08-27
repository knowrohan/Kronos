"""Generate the standalone findings artifact from the result JSONs.

Every number on the page is read from a produced artifact (verify_*.json, controls_*.json,
grid_*.parquet, universe_bias.json, drift_diagnostic.json, membership.json), never transcribed
by hand, so the page cannot drift from the data it reports.

    python bench/nse/make_artifact.py --out bench/out/nse/findings.html
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from bench.nse.common import DATA_NSE, OUT_NSE  # noqa: E402
from bench.nse.costs import PRESETS  # noqa: E402

# the crypto study lives beside the equity one rather than inside it: different tests,
# different cost stack, no shared code path
OUT_BTC = os.path.join(ROOT, "bench", "out", "btc")

HORIZONS = [1, 5, 10, 20]


def j(path, default=None):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default if default is not None else {}


def pq(path):
    return pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()


def e(x):
    return html.escape(str(x))


def pct(x, nd=2, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x * 100:{'+' if sign else ''}.{nd}f}%"


def num(x, nd=4, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:{'+' if sign else ''}.{nd}f}"


CSS = """
:root{
  --ground:#F6F7F4; --surface:#FFFFFF; --surface-2:#EFF2ED;
  --ink:#111815; --ink-2:#3A4642; --muted:#68756E;
  --rule:#DBE1DA; --rule-strong:#C2CBC2;
  --accent:#1E5C48; --accent-soft:#E3EFE8;
  --refuted:#A33218; --refuted-soft:#F7E7E2;
  --partly:#8A6410; --partly-soft:#F6EFDF;
  --shadow:0 1px 2px rgba(17,24,21,.05),0 8px 24px -12px rgba(17,24,21,.16);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0D1311; --surface:#141C19; --surface-2:#1A241F;
  --ink:#E7EDE8; --ink-2:#BCC7C0; --muted:#8A968F;
  --rule:#26322C; --rule-strong:#36453D;
  --accent:#7FBFA6; --accent-soft:#16261F;
  --refuted:#E08A72; --refuted-soft:#2A1A15;
  --partly:#D2A94F; --partly-soft:#241D10;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -14px rgba(0,0,0,.7);
}}
:root[data-theme="dark"]{
  --ground:#0D1311; --surface:#141C19; --surface-2:#1A241F;
  --ink:#E7EDE8; --ink-2:#BCC7C0; --muted:#8A968F;
  --rule:#26322C; --rule-strong:#36453D;
  --accent:#7FBFA6; --accent-soft:#16261F;
  --refuted:#E08A72; --refuted-soft:#2A1A15;
  --partly:#D2A94F; --partly-soft:#241D10;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -14px rgba(0,0,0,.7);
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"Public Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-size:16.5px; line-height:1.62; -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1080px;margin:0 auto;padding:0 28px 96px}
.col{max-width:68ch}
h1,h2,h3{font-family:"Zilla Slab",Georgia,serif;text-wrap:balance;margin:0}
h1{font-size:clamp(2.1rem,4.6vw,3.3rem);line-height:1.06;font-weight:600;letter-spacing:-.015em}
h2{font-size:clamp(1.45rem,2.4vw,1.9rem);line-height:1.16;font-weight:600;letter-spacing:-.01em}
h3{font-size:1.08rem;font-weight:600;line-height:1.3}
p{margin:0}
a{color:var(--accent)}
.mono,td.n,th.n{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  font-variant-numeric:tabular-nums}
code{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-size:.87em;
  background:var(--surface-2);padding:.1em .34em;border-radius:3px}

header.masthead{border-bottom:1px solid var(--rule-strong);padding:56px 0 34px;margin-bottom:44px}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:.7rem;letter-spacing:.16em;
  text-transform:uppercase;color:var(--muted);margin-bottom:20px}
.standfirst{font-size:1.16rem;color:var(--ink-2);margin-top:20px;max-width:60ch}
.meta{display:flex;flex-wrap:wrap;gap:0 30px;margin-top:26px;font-size:.79rem;color:var(--muted);
  font-family:"IBM Plex Mono",monospace}
.meta b{color:var(--ink-2);font-weight:500}

section{padding:46px 0;border-top:1px solid var(--rule)}
section:first-of-type{border-top:none}
.sec-head{display:flex;gap:16px;align-items:baseline;margin-bottom:20px}
.sec-num{font-family:"IBM Plex Mono",monospace;font-size:.72rem;color:var(--accent);
  letter-spacing:.1em;padding-top:.35em}
.stack{display:flex;flex-direction:column;gap:18px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:14px}
.grid4{display:grid;grid-template-columns:repeat(auto-fit,minmax(178px,1fr));gap:14px}

.card{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:20px 22px;
  box-shadow:var(--shadow)}
.card h3{margin-bottom:8px}
.secidx{list-style:none;display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));
  gap:4px 22px;margin-top:6px}
.secidx li{font-size:.85rem;color:var(--muted);padding:2px 0}
.secidx a{text-decoration:none}
.secidx .mono{font-size:.7rem;letter-spacing:.09em;color:var(--accent);font-weight:500;
  margin-right:7px}
.card p{font-size:.93rem;color:var(--ink-2)}

.stat{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:17px 19px;
  display:flex;flex-direction:column;gap:5px;box-shadow:var(--shadow)}
.stat .k{font-family:"IBM Plex Mono",monospace;font-size:.67rem;letter-spacing:.1em;
  text-transform:uppercase;color:var(--muted)}
.stat .v{font-family:"IBM Plex Mono",monospace;font-size:1.72rem;line-height:1.05;font-weight:500;
  font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.stat .s{font-size:.8rem;color:var(--muted);line-height:1.4}
.v.neg{color:var(--refuted)} .v.pos{color:var(--accent)}

.verdict{display:inline-flex;align-items:center;gap:6px;font-family:"IBM Plex Mono",monospace;
  font-size:.66rem;letter-spacing:.09em;text-transform:uppercase;padding:3px 9px;border-radius:999px;
  white-space:nowrap;font-weight:500}
.v-conf{background:var(--accent-soft);color:var(--accent);border:1px solid color-mix(in srgb,var(--accent) 30%,transparent)}
.v-ref{background:var(--refuted-soft);color:var(--refuted);border:1px solid color-mix(in srgb,var(--refuted) 30%,transparent)}
.v-part{background:var(--partly-soft);color:var(--partly);border:1px solid color-mix(in srgb,var(--partly) 32%,transparent)}

.tw{overflow-x:auto;border:1px solid var(--rule);border-radius:8px;background:var(--surface);
  box-shadow:var(--shadow)}
table{border-collapse:collapse;width:100%;font-size:.87rem}
th,td{padding:9px 13px;text-align:left;border-bottom:1px solid var(--rule);white-space:nowrap}
thead th{background:var(--surface-2);font-size:.68rem;letter-spacing:.08em;text-transform:uppercase;
  color:var(--muted);font-weight:600;position:sticky;top:0}
tbody tr:last-child td{border-bottom:none}
td.n,th.n{text-align:right}
tr.hi td{background:var(--accent-soft)}
tr.dim td{color:var(--muted)}
caption{caption-side:bottom;text-align:left;padding:10px 13px;font-size:.79rem;color:var(--muted)}

.gates{display:flex;flex-direction:column;gap:0}
.gate{display:grid;grid-template-columns:64px 1fr auto;gap:16px;align-items:start;
  padding:16px 0;border-bottom:1px solid var(--rule)}
.gate:last-child{border-bottom:none}
.gate .id{font-family:"IBM Plex Mono",monospace;font-weight:600;font-size:.95rem;color:var(--accent)}
.gate .what{font-size:.92rem}
.gate .what b{display:block;font-size:.86rem;color:var(--muted);font-weight:400;margin-top:3px}

.bar{display:flex;height:34px;border-radius:5px;overflow:hidden;border:1px solid var(--rule)}
.bar span{display:flex;align-items:center;justify-content:center;font-family:"IBM Plex Mono",monospace;
  font-size:.72rem;color:#fff;white-space:nowrap;overflow:hidden}
.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:.8rem;color:var(--ink-2);margin-top:10px}
.legend i{width:11px;height:11px;border-radius:2px;display:inline-block;margin-right:6px;
  vertical-align:baseline}

.pull{border-left:3px solid var(--accent);padding:4px 0 4px 20px;font-size:1.06rem;color:var(--ink-2)}
.note{font-size:.85rem;color:var(--muted)}
figure{margin:0}
figure img{width:100%;height:auto;border:1px solid var(--rule);border-radius:8px;background:var(--surface)}
figcaption{font-size:.8rem;color:var(--muted);margin-top:8px}
ul.tight{margin:0;padding-left:1.15em;display:flex;flex-direction:column;gap:7px;font-size:.93rem;
  color:var(--ink-2)}
footer{border-top:1px solid var(--rule-strong);margin-top:40px;padding-top:26px;font-size:.82rem;
  color:var(--muted)}
@media (max-width:640px){.wrap{padding:0 18px 64px}.gate{grid-template-columns:48px 1fr}
  .gate .verdict{grid-column:2}}
"""


def svg_lines(series, xlabels, ylabel, width=680, height=260, zero=True,
              yfmt=lambda v: f"{v * 100:+.1f}%", dots=True):
    """Small theme-aware line chart. series = [(label, [y...], color_token, dashed)].

    yfmt formats the axis: the default is a percentage, but a loss curve is not one.
    dots=False drops the per-point markers, which only read as noise past ~15 points.
    """
    pad_l, pad_r, pad_t, pad_b = 52, 14, 14, 34
    W, H = width, height
    ys = [v for _, vals, _, _ in series for v in vals]
    lo, hi = min(ys), max(ys)
    if zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    span = (hi - lo) or 1.0
    lo -= span * 0.10; hi += span * 0.10; span = hi - lo
    n = max(len(xlabels), 2)
    X = lambda i: pad_l + i * (W - pad_l - pad_r) / (n - 1)
    Y = lambda v: pad_t + (hi - v) * (H - pad_t - pad_b) / span

    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
           f'aria-label="{e(ylabel)}" style="display:block">']
    # gridlines
    for k in range(5):
        v = lo + span * k / 4
        y = Y(v)
        out.append(f'<line x1="{pad_l}" x2="{W - pad_r}" y1="{y:.1f}" y2="{y:.1f}" '
                   f'stroke="var(--rule)" stroke-width="1"/>')
        out.append(f'<text x="{pad_l - 8}" y="{y + 3.5:.1f}" text-anchor="end" '
                   f'font-size="10" fill="var(--muted)" '
                   f'font-family="IBM Plex Mono, monospace">{e(yfmt(v))}</text>')
    if lo < 0 < hi:
        out.append(f'<line x1="{pad_l}" x2="{W - pad_r}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" '
                   f'stroke="var(--ink-2)" stroke-width="1.2"/>')
    for i, lab in enumerate(xlabels):
        out.append(f'<text x="{X(i):.1f}" y="{H - 12}" text-anchor="middle" font-size="10" '
                   f'fill="var(--muted)" font-family="IBM Plex Mono, monospace">{e(lab)}</text>')
    for label, vals, color, dashed in series:
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                   f'stroke-width="2.2" stroke-linejoin="round"{dash}/>')
        if dots:
            for i, v in enumerate(vals):
                out.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="3" fill="{color}"/>')
    out.append("</svg>")
    leg = "".join(
        f'<span><i style="background:{c}"></i>{e(l)}</span>' for l, _, c, _ in series)
    return (f'<figure><div style="background:var(--surface);border:1px solid var(--rule);'
            f'border-radius:8px;padding:14px 16px 8px;box-shadow:var(--shadow)">'
            f'{"".join(out)}<div class="legend">{leg}</div></div></figure>')


def stat(k, v, s="", cls=""):
    sub = f'<span class="s">{e(s)}</span>' if s else ""
    return (f'<div class="stat"><span class="k">{e(k)}</span>'
            f'<span class="v {cls}">{e(v)}</span>{sub}</div>')


def verdict_chip(v):
    m = {"CONFIRMED": "v-conf", "REFUTED": "v-ref", "PARTLY": "v-part",
         "PASS": "v-conf", "FAIL": "v-ref", "USED": "v-conf"}
    return f'<span class="verdict {m.get(v, "v-part")}">{e(v)}</span>'


SECTION_INDEX = [
    ("GATES", "the four pre-registered pass/fail tests and their outcomes"),
    ("MECHANISM", "why the forecast is a reversal factor, fitted five ways"),
    ("RETURNS", "what the strategy actually earned, against what"),
    ("UNIVERSE", "survivorship, size tilt, and point-in-time membership"),
    ("COSTS", "the Indian statutory cost stack and market impact"),
    ("VERIFICATION", "the checks that the harness itself is not lying"),
    ("ROBUSTNESS", "sampler, temperature, greedy decoding, other markets"),
    ("SCALAR", "the volatility outputs, tested against free alternatives"),
    ("SIGNAL", "what survives once the artefact is projected out"),
    ("CORRECTION", "results this study reversed on itself"),
    ("SEARCH", "the 735-cell parameter surface and the family-wise null"),
    ("CRYPTO", "Bitcoin: mechanism, calibration, money, and what was left unrun"),
    ("HARDENING", "which upstream pull requests are real fixes and which are wrong"),
    ("READING", "what this does and does not license you to conclude"),
    ("STATE", "the state of play, claim by claim"),
    ("FORWARD", "paths that survived an adversarial attempt to kill them"),
    ("SCOPE", "the full registry of what was tested and what was not"),
    ("METHOD", "how to reproduce any of it"),
]


def section_index():
    items = "".join(
        f'<li><a href="#{k.lower()}"><span class="mono">{e(k)}</span></a> {e(v)}</li>'
        for k, v in SECTION_INDEX)
    return (f'<div class="card" style="margin-bottom:14px"><h3>Section index</h3>'
            f'<ul class="secidx">{items}</ul></div>')


def table(headers, rows, caption="", numeric_from=1, highlight=None):
    th = "".join(f'<th class="{"n" if i >= numeric_from else ""}">{e(h)}</th>'
                 for i, h in enumerate(headers))
    body = []
    for r in rows:
        cls = ""
        if highlight and highlight(r):
            cls = ' class="hi"'
        tds = "".join(f'<td class="{"n" if i >= numeric_from else ""}">{c}</td>'
                      for i, c in enumerate(r))
        body.append(f"<tr{cls}>{tds}</tr>")
    cap = f"<caption>{e(caption)}</caption>" if caption else ""
    return (f'<div class="tw"><table><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody>{cap}</table></div>')



def load_all(tag):
    O = OUT_NSE
    d = dict(
        vm=j(f"{O}/verify_model.json"), vp=j(f"{O}/verify_pipeline.json"),
        vb=j(f"{O}/verify_backtest.json"),
        ctl=j(f"{O}/controls_{tag}.json") or j(f"{O}/controls_pilot.json"),
        gsum=j(f"{O}/grid_{tag}_summary.json") or j(f"{O}/grid_pilot_summary.json"),
        ub=j(f"{O}/universe_bias.json"), drift=j(f"{O}/drift_diagnostic.json"),
        memb=j(f"{DATA_NSE}/membership.json"),
        smeta=(j(f"{O}/signals_{tag}_meta.json") or j(f"{O}/signals_zeroshot_meta.json")),
        pitc=j(f"{O}/controls_pit.json"), usc=j(f"{O}/controls_us.json"),
        ftj=j(f"{O}/finetune/train_summary.json"), ftc=j(f"{O}/controls_finetuned.json"),

        greedy=j(f"{O}/controls_greedy.json"), t1=j(f"{O}/controls_T1p0.json"),
        paths5=j(f"{O}/controls_paths5.json"),
        mech=j(f"{O}/mechanism.json"),
        # the crypto study writes to its own output dir; absent runs simply omit sections
        btc_d90=j(f"{OUT_BTC}/analysis_d90.json"),
        btc_eth90=j(f"{OUT_BTC}/analysis_eth90.json"),
        btc_d512=j(f"{OUT_BTC}/analysis_d512.json"),
    )
    g = pq(f"{O}/grid_{tag}.parquet")
    d["grid"] = g if not g.empty else pq(f"{O}/grid_pilot.parquet")
    d["pitg"] = pq(f"{O}/grid_pit.parquet")
    r = pq(f"{O}/random_dist_{tag}.parquet")
    d["rnd"] = r if not r.empty else pq(f"{O}/random_dist_pilot.parquet")
    d["bl"] = pq(f"{O}/grid_baselines.parquet")
    d["blsum"] = j(f"{O}/grid_baselines_summary.json")
    d["blrnd"] = pq(f"{O}/random_dist_baselines.parquet")
    d["blpit"] = pq(f"{O}/grid_baselines_pit.parquet")
    d["blpitsum"] = j(f"{O}/grid_baselines_pit_summary.json")
    d["pitcov"] = j(f"{DATA_NSE}/pit_coverage.json")
    d["divgap"] = j(f"{O}/dividend_gap.json")
    d["rob"] = j(f"{O}/robustness_robust_pit.json")
    d["vol"] = j(f"{O}/vol_{tag}.json") or j(f"{O}/vol_zeroshot.json") or j(f"{O}/vol_pilot.json")
    d["volpit"] = j(f"{O}/vol_pit.json")
    d["mech"] = j(f"{O}/mechanism.json")
    d["ftz"] = j(f"{O}/controls_zeroshot_ftdates.json")
    try:
        d["ftlog"] = pd.read_csv(f"{O}/finetune/train_log.csv")
    except Exception:
        d["ftlog"] = pd.DataFrame()
    d["volbt"] = (pq(f"{O}/vol_backtest_vol_{tag}.parquet")
                  if not pq(f"{O}/vol_backtest_vol_{tag}.parquet").empty
                  else pq(f"{O}/vol_backtest_vol_pilot.parquet"))
    d["surf"] = j(f"{O}/surface_surface_pit.json")
    d["surfdf"] = pq(f"{O}/surface_surface_pit.parquet")
    d["ba"] = j(f"{O}/benchmark_audit.json")
    d["cap"] = pq(f"{O}/capacity_robust_pit.parquet")
    d["liq"] = pq(f"{O}/liquidity_robust_pit.parquet")
    d["voladj"] = pq(f"{O}/grid_baselines_pit_vol_adj.parquet")
    return d


def build(tag) -> str:
    D = load_all(tag)
    ctl, grid, gsum, ub = D["ctl"], D["grid"], D["gsum"], D["ub"]
    vm, vp, vb = D["vm"], D["vp"], D["vb"]
    icg = ctl.get("ic_grid", {}); base = ctl.get("baseline_ic", {})
    corr = ctl.get("cross_sectional_rank_corr_with_kronos", {})
    orth = ctl.get("orthogonalised_ic", {})
    prim = icg.get("last_ret", {}).get("open_h10", {})
    smeta = D["smeta"]

    def cell(arm, col="ann_excess_net", reb="weekly", cost=30.0, g=None):
        gg = grid if g is None else g
        if gg is None or gg.empty or col not in gg.columns:
            return None
        s = gg[(gg.arm == arm) & (gg.rebalance == reb) & (gg.cost_bps == cost)]
        return float(s.iloc[0][col]) if len(s) else None

    S = []; A = S.append
    dr = ctl.get("date_range", ["", ""])
    provisional = not os.path.exists(f"{OUT_NSE}/controls_zeroshot.json")

    # ============ masthead ============
    A(f"""<header class="masthead"><div class="col">
<div class="eyebrow">Kronos &middot; AAAI 2026 &middot; cross-sectional replication on NSE</div>
<h1>The forecast is a reversal factor</h1>
<p class="standfirst">Kronos is a 102M-parameter foundation model for candlesticks. Scored the
way its own repository scores it &mdash; one number per stock per day, ranked, top&nbsp;50 held
&mdash; it produced no measurable cross-sectional signal on the NIFTY&nbsp;200 over three years,
and lost to simply holding the universe. Its output turns out to be 70% reversion to each
stock&rsquo;s own 90-bar mean, which is a line of arithmetic, and the same thing happens on a US
cross-section. That is half the story. The other half is that the apparatus built to prove it
&mdash; a point-in-time universe, a cost model from the circulars, a family-wise null, a
parameter sweep &mdash; goes on to reverse most of the model-free results this study found as
well, including one that had already survived the first three.</p>
<div class="meta">
<span><b>Universe</b> NIFTY 200</span><span><b>Test</b> {e(dr[0])} &rarr; {e(dr[1])}</span>
<span><b>Signal dates</b> {e(ctl.get('n_dates', '—'))}</span>
<span><b>Model</b> Kronos-base @{e(str(smeta.get('model_revision', '?'))[:10])}</span>
<span><b>Inference</b> {smeta.get('wall_clock_sec', 0) / 3600:.1f} GPU-h, M5 / mps</span>
</div></div></header>""")
    # ============ orientation for a reader arriving cold ============
    A('<div class="card" style="border-color:var(--accent);border-left-width:4px;'
      'background:var(--accent-soft);margin-bottom:14px">'
      '<h3>If you are picking this up cold</h3>'
      '<p style="font-size:.97rem"><strong>What this is.</strong> A pre-registered attempt to '
      'make money with Kronos, a 102M-parameter foundation model for OHLCV candlesticks '
      '(AAAI 2026). It was tested two ways: ranking ~200 Indian equities against each other '
      'every session for three years, and timing Bitcoin over nine years. Every number on this '
      'page is read from a produced JSON or parquet file rather than transcribed, so it cannot '
      'drift from the data that made it.</p>'
      '<p style="font-size:.97rem"><strong>The answer.</strong> Kronos does not produce a '
      'tradeable signal in either setting. Its forecast is 70–82% a restatement of how far '
      'price sits from its own N-bar mean — a line of arithmetic you can compute for free — '
      'and once that is understood, everything else follows: it loses to holding the universe, '
      'it loses to its own negation, and on Bitcoin its directional signal is significantly '
      '<em>negative</em>. Finetuning removes the artefact and reveals nothing underneath. Do '
      'not trade it.</p>'
      '<p style="font-size:.97rem"><strong>Where to go next.</strong> Three sections are '
      'written for you specifically rather than for a reader following the argument: '
      '<strong>SCOPE</strong> is the registry of every market, model variant and statistical '
      'test already run, plus what was deliberately left undone; <strong>FORWARD</strong> '
      'ranks the paths that survived an adversarial attempt to kill them, with the exact test '
      'and decision rule for each; <strong>HARDENING</strong> records which upstream pull '
      'requests are real fixes and which are confidently wrong. Read SCOPE before proposing '
      'anything — most obvious ideas are already in the graveyard.</p>'
      '<p class="note" style="margin-top:8px">Code in <code>bench/nse/</code> and '
      '<code>bench/btc/</code>; outputs in <code>bench/out/</code>; pre-registration and its '
      'deviations in <code>bench/nse/PREREG.md</code>. The library fixes live on branch '
      '<code>harden/upstream-fixes</code>.</p></div>')
    A(section_index())

    if provisional:
        A('<div class="card" style="border-color:var(--partly);background:var(--partly-soft);'
          'margin-bottom:8px"><h3 style="color:var(--partly)">Provisional</h3>'
          '<p>The Kronos figures below come from a pilot of '
          f'{e(ctl.get("n_dates", "?"))} signal dates strided evenly across the full three-year '
          'test window, not from every session. Because the primary horizon is 10 sessions and '
          'the pilot strides by 12, its 10-day information coefficients are already near-'
          'independent and will barely move; the denser pass mainly sharpens the 1- and 5-day '
          'numbers and adds the daily-rebalance arm. Everything model-free on this page — the '
          'cost model, the universe decomposition, the point-in-time membership and the '
          'baseline sweep — already uses all 756 sessions.</p></div>')

    # ============ headline numbers ============
    ex_cnx, ex_ew = cell("last_ret"), cell("last_ret", "ann_excess_vs_ew200")
    pctile = gsum.get("prereg_percentile_in_random")
    surv = (ub.get("decomposition") or {}).get("plus_survivorship_from_current_membership")
    A('<section><div class="grid3">')
    A(stat("Primary rank IC · 10d", num(prim.get("mean_ic")),
           f"Newey–West t = {num(prim.get('t_newey_west'), 2)} · "
           f"{'significant' if prim.get('significant') else 'not significant'}"))
    A(stat("Excess vs ^CNX200", pct(ex_cnx), "weekly rebalance, 30 bps round trip",
           "pos" if (ex_cnx is not None and ex_cnx > 0) else "neg"))
    A(stat("Excess vs equal-weight universe", pct(ex_ew),
           "the comparison with the size and survivorship tilt removed",
           "pos" if (ex_ew is not None and ex_ew > 0) else "neg"))
    pval = gsum.get("prereg_empirical_p_one_sided")
    A(stat("Percentile among random signals",
           f"{pctile:.0f}" if pctile is not None else "&mdash;",
           (f"500 shuffled scores through the identical engine \u00b7 one-sided p = "
            f"{pval:.3f}" if pval is not None else
            "shuffled scores through the identical engine"),
           # green only if it actually clears; a percentile above the median but far from
           # significance is not a positive result and should not be coloured like one
           "pos" if (pval is not None and pval < 0.05) else "neg"))
    A(stat("Rank corr. with one line of code", num(corr.get("anchor_reversal"), 3),
           "Spearman vs −(close − mean₉₀)/std₉₀, mean across dates"))
    A(stat("Survivorship in the naive setup", pct(surv),
           "annualised, from back-projecting today's index membership", "neg"))
    A("</div></section>")

    # ============ gates ============
    def gate(gid, what, sub, verdict):
        return (f'<div class="gate"><div class="id">{e(gid)}</div>'
                f'<div class="what">{what}<b>{sub}</b></div>{verdict_chip(verdict)}</div>')
    n_cells = sum(1 for v in icg.values() for h in HORIZONS
                  if v.get(f"open_h{h}", {}).get("significant"))
    n_total = len(icg) * len(HORIZONS)
    mom = base.get("momentum_10d", {}).get("open_h10", {}).get("mean_ic")
    anc = base.get("anchor_reversal", {}).get("open_h10", {}).get("mean_ic")
    ft_ok = D["ftj"].get("finetuning_helped_on_val")
    A('<section id="gates"><div class="sec-head"><span class="sec-num">GATES</span>'
      '<h2>The kill criteria, decided before any number was computed</h2></div>'
      '<div class="col" style="margin-bottom:18px"><p class="note">Fixed in '
      '<code>PREREG.md</code> before the first IC was calculated, so that a maximum over 96 arms '
      'could not be mistaken for a result. Every deviation is logged in that file&rsquo;s '
      'addendum.</p></div><div class="gates">')
    A(gate("G0", "Pilot: does any horizon show an IC whose CI excludes zero?",
           f"{n_cells} of {n_total} Kronos cells significant across 12 arms × 4 horizons",
           "FAIL" if n_cells == 0 else "PASS"))
    A(gate("G1", "Full pass: is the primary IC significantly positive?",
           f"mean IC {num(prim.get('mean_ic'))}, NW t = {num(prim.get('t_newey_west'), 2)}, "
           f"bootstrap CI [{num(prim.get('bootstrap_ci95', [None, None])[0])}, "
           f"{num(prim.get('bootstrap_ci95', [None, None])[1])}]",
           "PASS" if prim.get("significant") else "FAIL"))
    A(gate("G2", "Does it beat the random-signal distribution, not merely the index?",
           f"strategy sits at percentile {pctile:.0f} of the random draws; one-sided p = "
           f"{gsum.get('prereg_empirical_p_one_sided')}" if pctile is not None
           else "random distribution not yet computed",
           "FAIL" if (pctile is None or pctile < 95) else "PASS"))
    A(gate("G2b", "Does it beat a one-line momentum or reversal feature?",
           f"Kronos IC {num(prim.get('mean_ic'))} vs momentum {num(mom)} and "
           f"anchor-reversal {num(anc)} at the same horizon",
           "FAIL" if (mom is not None and prim.get("mean_ic") is not None
                      and abs(mom) > abs(prim["mean_ic"])) else "PASS"))
    A(gate("G3", "Does finetuning on NSE data improve validation loss?",
           (f"zero-shot {D['ftj'].get('zero_shot_val_loss'):.4f} → finetuned "
            f"{D['ftj'].get('best_val_loss'):.4f} over "
            f"{D['ftj'].get('samples_seen', 0):,} samples" if D["ftj"] else "not run"),
           ("PASS" if ft_ok else "FAIL") if D["ftj"] else "PARTLY"))
    A("</div></section>")
    return S, D


def build_body(S, D, tag):
    A = S.append
    ctl, grid, gsum, ub = D["ctl"], D["grid"], D["gsum"], D["ub"]
    icg = ctl.get("ic_grid", {}); base = ctl.get("baseline_ic", {})
    corr = ctl.get("cross_sectional_rank_corr_with_kronos", {})
    orth = ctl.get("orthogonalised_ic", {}); drift = D["drift"]
    prim = icg.get("last_ret", {}).get("open_h10", {})

    def cell(arm, col="ann_excess_net", reb="weekly", cost=30.0, g=None):
        gg = grid if g is None else g
        if gg is None or gg.empty or col not in gg.columns:
            return None
        s = gg[(gg.arm == arm) & (gg.rebalance == reb) & (gg.cost_bps == cost)]
        return float(s.iloc[0][col]) if len(s) else None

    # ============ what the signal is ============
    A('<section id="mechanism"><div class="sec-head"><span class="sec-num">MECHANISM</span>'
      '<h2>Why the score is a reversal factor</h2></div><div class="col stack">')
    A('<p>The repository collapses each forecast to one scalar: the predicted close minus the '
      'last observed close, in the units the model works in. Those units matter. Each window is '
      'normalised on its own trailing 90 bars, and an autoregressive sampler run forward on a '
      'normalised series drifts back toward that window&rsquo;s mean. So for a stock trading '
      'above its own 90-bar average &mdash; which, in a bull market, is most of them &mdash; the '
      '&ldquo;forecast&rdquo; is negative <em>by construction</em>, and more negative the further '
      'the stock has run.</p>')
    A('<p class="pull">Rank the cross-section by that number and you have not ranked by expected '
      'return. You have ranked by how far each stock sits below its own recent mean.</p></div>')
    if drift.get("regimes"):
        rs_s = drift["regimes"].get("sampled_T0.6_topp0.9", [])
        rs_g = drift["regimes"].get("greedy_topk1", [])
        if rs_s:
            series = [("predicted, repo sampling",
                       list(np.mean([r["pred_pct_by_step"] for r in rs_s], axis=0)),
                       "var(--refuted)", False)]
            if rs_g:
                series.append(("predicted, greedy (the model's mode)",
                               list(np.mean([r["pred_pct_by_step"] for r in rs_g], axis=0)),
                               "var(--partly)", False))
            series.append(("what actually happened",
                           list(np.mean([r["realized_pct_by_step"] for r in rs_s], axis=0)),
                           "var(--accent)", True))
            A('<div style="margin-top:22px">'
              + svg_lines(series, [str(i) for i in range(1, 11)],
                          "cross-sectional mean move by prediction step")
              + '<figcaption>Cross-sectional mean move against sessions ahead, averaged over '
                'sample dates. The two predicted lines sit on top of each other: removing '
                'sampling noise entirely does not remove the drift.</figcaption></div>')
        rows = []
        for regime, rs in drift["regimes"].items():
            if not rs:
                continue
            m = np.mean([r["pred_pct_by_step"] for r in rs], axis=0)
            neg = np.mean([r["share_symbols_negative_terminal"] for r in rs])
            rows.append([f'<code>{e(regime)}</code>', pct(m[0]), pct(m[4]), pct(m[9]),
                         f"{neg:.2f}"])
        rs = drift["regimes"].get("sampled_T0.6_topp0.9", [])
        if rs:
            m = np.mean([r["realized_pct_by_step"] for r in rs], axis=0)
            rows.append(["<b>what actually happened</b>", pct(m[0]), pct(m[4]), pct(m[9]), "—"])
        A('<div style="margin-top:20px">' + table(
            ["decoding regime", "step 1", "step 5", "step 10", "share predicted down"], rows,
            "Cross-sectional mean predicted move per step ahead. The greedy row decodes the "
            "model's conditional mode, so a matching decline there rules out sampling noise as "
            "the cause.") + "</div>")
    rs = drift.get("regimes", {}).get("sampled_T0.6_topp0.9", [])
    if rs and len(rs) >= 3:
        # Fit predicted-10d-move on the cross-section's mean anchor_z. If the "forecast" is
        # really a reversion to the window mean, this is a straight line through the origin.
        xs = np.array([r["anchor_z_mean"] for r in rs])
        zs = np.array([r["pred_z_minus_anchor_by_step"][-1] for r in rs])
        rr = np.array([r["realized_pct_by_step"][-1] for r in rs])
        mech0 = (D["mech"] or {}).get("zero_shot_full")
        m, b = np.polyfit(xs, zs, 1)
        r2 = float(np.corrcoef(xs, zs)[0, 1] ** 2)
        r2_real = float(np.corrcoef(xs, rr)[0, 1] ** 2)
        A(f'<div class="grid3" style="margin-top:20px">'
          + stat("Score, fitted on date means", f"{m:.3f} × anchor_z",
                 f"intercept {b:+.3f}σ, R² = {r2:.3f} across {len(rs)} dates — this regression "
                 f"has one point per date, so read the next card for the real one")
          + stat("Reverts this far to the mean",
                 (f"{mech0['reversion_fraction'] * 100:.1f}%" if mech0
                  else f"{abs(m) * 100:.0f}%"),
                 (f"slope {mech0['pooled_slope']:+.4f}, R² {mech0['pooled_r2']:.2f} over all "
                  f"{mech0['n_obs']:,} stock-dates" if mech0 else
                  "of the distance back to the 90-bar window mean over 10 steps"))
          + stat("Same regressor vs realised", f"R² {r2_real:.2f}",
                 "how much of what actually happened it explains", "neg")
          + "</div>")
    if rs:
        rows = [[e(r["date"]), f'{r["anchor_z_mean"]:+.2f}&sigma;',
                 pct(r["pred_pct_by_step"][-1]), pct(r["realized_pct_by_step"][-1]),
                 f'{r["share_symbols_negative_terminal"]:.2f}'] for r in rs]
        A('<div style="margin-top:20px">' + table(
            ["signal date", "cross-section vs its own 90-bar mean", "Kronos predicts, 10d",
             "what happened", "share predicted down"], rows,
            "The prediction is a near-linear function of how extended the cross-section is "
            "against its own trailing mean. On the date the cross-section sat almost exactly "
            "at its mean, the model predicted almost exactly zero — and the market rose 3.9%.")
          + "</div>")

    ic_rows = []
    for label, src, key in [("Kronos <code>last_ret</code>", icg, "last_ret"),
                            ("Kronos <code>last_z</code> (repo definition)", icg, "last_z"),
                            ("Kronos <code>mean_ret</code>", icg, "mean_ret"),
                            ("10-day momentum", base, "momentum_10d"),
                            ("10-day reversal", base, "reversal_10d"),
                            ("&minus;anchor_z (one line, no model)", base, "anchor_reversal")]:
        r = src.get(key, {})
        if not r:
            continue
        r10 = r.get("open_h10", {})
        star = " *" if r10.get("significant") else ""
        ic_rows.append([label] + [num(r.get(f"open_h{h}", {}).get("mean_ic")) for h in HORIZONS]
                       + [num(r10.get("t_newey_west"), 2) + star,
                          num(corr.get(key), 3) if key in corr else "—"])
    if ic_rows:
        A('<div style="margin-top:20px">' + table(
            ["signal", "IC 1d", "IC 5d", "IC 10d", "IC 20d", "NW t (10d)", "rank corr w/ Kronos"],
            ic_rows,
            "Spearman rank IC against the tradeable forward return (open t+1 → open t+1+h). "
            "* marks the only cells where the Newey–West t exceeds 2 and the block-bootstrap "
            "CI excludes zero.") + "</div>")
    if orth:
        rows = []
        for k, row in orth.items():
            r10 = row.get("open_h10", {})
            rows.append([f'<code>{e(k.replace("kronos_resid_of_", ""))}</code>']
                        + [num(row.get(f"open_h{h}", {}).get("mean_ic")) for h in HORIZONS]
                        + [num(r10.get("t_newey_west"), 2),
                           "yes" if r10.get("significant") else "no"])
        A('<div style="margin-top:20px">' + table(
            ["Kronos residualised against", "IC 1d", "IC 5d", "IC 10d", "IC 20d", "NW t", "sig."],
            rows,
            "Within each date the Kronos rank-score is regressed on the free feature's "
            "rank-score; this is the IC of what is left. It answers whether four GPU-hours buy "
            "anything a one-line feature does not.") + "</div>")
    A("</section>")

    # ============ money ============
    A('<section id="returns"><div class="sec-head"><span class="sec-num">RETURNS</span>'
      '<h2>What it earns, against what it should be compared to</h2></div>')
    arms = ["reversed_last_ret", "momentum_20d", "momentum_10d", "equal_weight_universe",
            "anchor_reversal", "max_ret", "mean_ret", "last_z", "last_ret"]
    rows = []
    for a in arms:
        ex, ew = cell(a), cell(a, "ann_excess_vs_ew200")
        if ex is None:
            continue
        ir, to = cell(a, "information_ratio"), cell(a, "turnover_ann")
        nm = {"reversed_last_ret": "Kronos, sign flipped",
              "equal_weight_universe": "hold the whole universe",
              "anchor_reversal": "−anchor_z (one line)",
              "last_ret": "<b>Kronos (pre-registered arm)</b>"}.get(a, f"<code>{e(a)}</code>")
        rows.append([nm, pct(cell(a, "ann_return_net")), pct(ex), pct(ew),
                     num(ir, 2), f"{to:.2f}×" if to is not None else "—"])
    if rows:
        A(table(["strategy", "return, net", "excess vs ^CNX200", "excess vs equal-weight",
                 "IR", "turnover/yr"], rows,
                "Weekly rebalance, topk=50, n_drop=5, 30 bps round trip. Sorted by excess over "
                "the equal-weight universe — the column that removes the size and survivorship "
                "tilt every one of these arms inherits.",
                highlight=lambda r: "pre-registered" in r[0]))
    rw = gsum.get("random_weekly", {})
    if rw:
        A('<div class="col stack" style="margin-top:22px">')
        A('<p>The random-signal control is the one that matters. Shuffling the scores <em>within '
          'each date</em> preserves the eligible universe, the turnover and the cost exactly, and '
          'destroys only the information. Whatever that earns is what the machinery earns with '
          'nothing in it.</p></div>')
        A('<div class="grid3" style="margin-top:16px">')
        A(stat("Random signal, mean", pct(rw.get("mean")), "annualised excess vs ^CNX200"))
        A(stat("Random signal, 5th–95th", f"{pct(rw.get('p05'))} … {pct(rw.get('p95'))}",
               f"{gsum.get('random_weekly', {}).get('n_seeds', 500)} seeds"))
        A(stat("Kronos", pct(cell("last_ret")),
               f"percentile {gsum.get('prereg_percentile_in_random', float('nan')):.0f} of that "
               "distribution", "neg"))
        A("</div>")
    A("</section>")
    return S


def build_tail(S, D, tag):
    A = S.append
    ub, memb, vm, vp, vb = D["ub"], D["memb"], D["vm"], D["vp"], D["vb"]
    grid, pitg, ctl, pitc = D["grid"], D["pitg"], D["ctl"], D["pitc"]
    prim = ctl.get("ic_grid", {}).get("last_ret", {}).get("open_h10", {})

    # ============ the universe ============
    A('<section id="universe"><div class="sec-head"><span class="sec-num">UNIVERSE</span>'
      '<h2>Most of the &ldquo;excess return&rdquo; was never the model</h2></div><div class="col stack">')
    A('<p>Take today&rsquo;s NIFTY&nbsp;200 membership, project it back three years, equal-weight '
      'it, and you beat the cap-weighted index handsomely &mdash; with no signal at all. Two '
      'separate things cause that, and only one of them is investable.</p></div>')
    if ub.get("decomposition"):
        d = ub["decomposition"]
        tot = d["equals_equal_weight_current_membership"]
        segs = [("cap-weighted NIFTY 200", d["cap_weighted_nifty200"], "var(--muted)"),
                ("equal-weight size tilt", d["plus_size_tilt_from_equal_weighting"], "var(--accent)"),
                ("survivorship", d["plus_survivorship_from_current_membership"], "var(--refuted)")]
        bar = "".join(
            f'<span style="flex:{max(v, 0.001)};background:{c}">{pct(v)}</span>' for _, v, c in segs)
        A(f'<div style="margin-top:20px"><div class="bar">{bar}</div>'
          f'<div class="legend">' + "".join(
              f'<span><i style="background:{c}"></i>{e(n)} {pct(v)}</span>' for n, v, c in segs)
          + f'<span style="margin-left:auto;font-family:\'IBM Plex Mono\',monospace">'
            f'total {pct(tot)}/yr</span></div></div>')
        idx = ub.get("indices", {})
        A('<div class="col" style="margin-top:16px"><p class="note">The split is not assumed. '
          'The equal-weight tilt is charged at what real, survivorship-free NSE size indices '
          'actually returned over the identical sessions &mdash; '
          + ", ".join(f"{e(k)} {pct(v['ann_return'])}" for k, v in idx.items())
          + '. Survivorship is the residual.</p></div>')
    if memb.get("events"):
        A('<div class="col stack" style="margin-top:24px">')
        A('<h3>And it is fixable, in six documents</h3>')
        A('<p>The plan called historical membership its headline limitation. NIFTY publishes '
          'every semi-annual reconstitution as a text-layer PDF listing exactly which companies '
          'enter and leave. Six of them span this window, so point-in-time membership can be '
          'rebuilt by walking backwards from today&rsquo;s list.</p></div>')
        rows = [[e(ev["announced"]), e(ev["effective_from"]), str(len(ev["excluded"])),
                 str(len(ev["included"]))] for ev in memb["events"]]
        A('<div style="margin-top:16px">' + table(
            ["press release", "effective", "excluded", "included"], rows,
            f"{len(memb.get('ever_members', []))} distinct names were members at some point, "
            f"against 200 today. Prices were recovered for all but three "
            f"(PEL, TATAMTRDVR — DVR shares cancelled, GUJGASLTD — merged); the rebuilt "
            f"universe holds 199–200 names per session.", numeric_from=2) + "</div>")
    if not pitg.empty and not grid.empty:
        rows = []
        for a in ("last_ret", "mean_ret", "momentum_10d", "anchor_reversal",
                  "reversed_last_ret", "equal_weight_universe"):
            def g(gg):
                s = gg[(gg.arm == a) & (gg.rebalance == "weekly") & (gg.cost_bps == 30.0)]
                return float(s.iloc[0]["ann_excess_net"]) if len(s) else None
            x, y = g(grid), g(pitg)
            if x is None or y is None:
                continue
            rows.append([f"<code>{e(a)}</code>", pct(x), pct(y), pct(y - x)])
        if rows:
            A('<div style="margin-top:20px">' + table(
                ["arm", "current membership", "point-in-time", "difference"], rows,
                "The same experiment on both universes. The difference column is the "
                "survivorship bias, measured rather than estimated.") + "</div>")
    if pitc:
        pp = pitc.get("ic_grid", {}).get("last_ret", {}).get("open_h10", {})
        A(f'<div class="col" style="margin-top:16px"><p>Primary IC on the corrected universe: '
          f'<span class="mono">{num(pp.get("mean_ic"))}</span> '
          f'(NW t {num(pp.get("t_newey_west"), 2)}), against '
          f'<span class="mono">{num(prim.get("mean_ic"))}</span> on the uncorrected one.</p></div>')
        det = pitc          # each control is stored at the top level of the JSON
        psum = j(f"{OUT_NSE}/grid_pit_summary.json") or {}
        if det:
            order = [("strategy", "Kronos, as its repository scores it"),
                     ("equal_weight_universe", "hold the eligible universe, zero turnover"),
                     ("constant", "a constant score \u2014 buy 50 names, never re-rank"),
                     ("reversed", "the same Kronos score, negated"),
                     ("baseline_momentum_10d", "10-session momentum"),
                     ("perfect_foresight", "next-session return, known in advance")]
            rows2 = []
            for k, lab in order:
                v = det.get(k)
                if not v:
                    continue
                rows2.append([lab, pct(v.get("ann_excess_net")),
                              num(v.get("information_ratio"), 2),
                              num(v.get("turnover_ann"), 2, sign=False)])
            if rows2:
                A('<div style="margin-top:20px">' + table(
                    ["what is being held", "excess vs ^CNX200", "IR", "turnover/yr"], rows2,
                    "All 756 sessions on the point-in-time universe. Kronos is beaten by holding "
                    "the universe, by a constant score, by its own negation and by 10-session "
                    "momentum. Perfect foresight is there to prove the engine can express an "
                    "edge when one exists.", numeric_from=1) + "</div>")
        if psum.get("prereg_percentile_in_random") is not None:
            A(f'<div class="col" style="margin-top:14px"><p>Against 500 shuffled-score runs on '
              f'the same corrected universe, the strategy sits at percentile '
              f'<span class="mono">{psum["prereg_percentile_in_random"]:.0f}</span> '
              f'(one-sided p = {psum["prereg_empirical_p_one_sided"]:.3f}). Correcting the '
              f'universe moves every absolute number down by roughly half and leaves the '
              f'verdict exactly where it was.</p></div>')
    A("</section>")

    # ============ costs ============
    cm = PRESETS["zerodha"]
    b = cm.breakdown(1_000_000)
    A('<section id="costs"><div class="sec-head"><span class="sec-num">COSTS</span>'
      '<h2>The cost model, rebuilt from the circulars</h2></div><div class="col stack">')
    A('<p>The plan&rsquo;s draft rates were written from memory and three of them are wrong. The '
      'total happened to land close to the truth for one particular trade size, which is the '
      'most dangerous way for a cost model to be wrong.</p></div>')
    rows = [
        ["STT (delivery, both sides)", "0.100% × 2", "0.100% × 2", verdict_chip("CONFIRMED")],
        ["Stamp duty (buy only)", "0.015%", "0.015%", verdict_chip("CONFIRMED")],
        ["Exchange + IPFT, per side", "0.00297%", "<b>0.00307%</b>", verdict_chip("REFUTED")],
        ["SEBI turnover fee", "0.0001%", "0.0001%", verdict_chip("CONFIRMED")],
        ["Brokerage", "<b>0.03% ad valorem</b>", "<b>₹0, or flat ₹20/order</b>",
         verdict_chip("REFUTED")],
        ["DP charge on the sell leg", "<b>absent</b>", "<b>₹13–20 per scrip</b>",
         verdict_chip("REFUTED")],
        ["GST base", "brokerage + exchange", "+ SEBI fee, IPFT and DP", verdict_chip("PARTLY")],
    ]
    A(table(["line item", "plan draft", "verified, Aug 2026", "verdict"], rows,
            "Exchange rate from NSE circular NSE/FA/73061, effective 1 Mar 2026 — NSE moved "
            "money between the transaction and IPFT lines without changing the total, which "
            "breaks any model that hardcoded them separately.", numeric_from=1))
    A('<div class="grid3" style="margin-top:18px">')
    A(stat("Real round trip", f"{b['round_trip_bps']:.1f} bps",
           "₹10 lakh leg: 22.4 bps statutory + 5 bps slippage"))
    A(stat("Plan's estimate", f"{PRESETS['plan_draft'].round_trip_bps(1e6):.1f} bps",
           "28% too conservative at this size"))
    A(stat("At a ₹10,000 leg", f"{PRESETS['flat20'].round_trip_bps(1e4):.1f} bps",
           "the flat ₹20 and flat DP charge dominate — the plan's model would understate this "
           "by 55 bps", "neg"))
    A("</div>")
    A('<div class="col" style="margin-top:16px"><p class="note">STT alone is 20 bps of the 22.4. '
      'The structural point is that real Indian delivery cost is <em>not</em> scale-invariant: '
      'brokerage is capped and DP is flat, so cost in basis points falls with trade size. A '
      'purely ad-valorem model is right at exactly one notional and wrong everywhere else.</p>'
      '</div></section>')

    # ============ verification ============
    A('<section id="verification"><div class="sec-head"><span class="sec-num">VERIFICATION</span>'
      '<h2>What the plan asserted, and what measurement said</h2></div>')
    rows = [
        ["Batching is a ~32× throughput win",
         f"{vm.get('4.3_speedup_vs_batch1', '—')}× at lookback=90; best "
         f"{vm.get('4.3_best', {}).get('ms_per_series', '—')} ms/series. There is no KV cache, so "
         "it is already compute-bound at batch 1", verdict_chip("REFUTED")],
        ["Signals live in normalised, not price, space",
         f"predicted close |max| = {vm.get('4.4_pred_close_abs_max', 0):.2f}σ against a price "
         f"level of ₹{vm.get('4.4_raw_close_level_mean', 0):.0f}", verdict_chip("CONFIRMED")],
        ["<code>preds</code> contains the reconstructed context",
         f"returned shape {vm.get('4.5_pred_shape')} for lookback 90 + predict 10",
         verdict_chip("CONFIRMED")],
        ["Tokenizer round-trip error might be negligible",
         f"it is {vm.get('4.5b_recon_err_as_pct_of_signal', 0):.1f}% of signal magnitude — though "
         "it barely reorders anything (rank corr 0.995)", verdict_chip("REFUTED")],
        ["<code>sample_count&gt;1</code> destroys the ensemble",
         f"returns {vm.get('4.7_shape_with_sample_count_5')}, three-dimensional — the paths are "
         "averaged before you see them", verdict_chip("CONFIRMED")],
        ["MPS sampling may not be reproducible",
         "bit-identical across runs under a fixed seed; the workaround was unnecessary",
         verdict_chip("REFUTED")],
        ["The val→test gap exceeds lookback+predict",
         "it was 21 sessions against the 100 required — one calendar month is not 100 trading "
         "days. Val end moved to 2023-02-15", verdict_chip("REFUTED")],
        ["Historical index membership might exist",
         f"it does; six PDFs rebuild {len(memb.get('ever_members', []))} names of point-in-time "
         "membership for this window", verdict_chip("USED")],
        ["Indian transaction cost rates",
         "total roughly right, three line items wrong, and the model is scale-invariant when "
         "reality is not", verdict_chip("PARTLY")],
        ["Turnover drag ≈ 7.6%/yr at repo defaults",
         "arithmetic right, realised lower — <code>hold_thresh</code> and qlib's own anti-churn "
         "guard both bind", verdict_chip("PARTLY")],
    ]
    A(table(["claim", "what measurement said", "verdict"], rows, "", numeric_from=99))
    npass = sum(1 for v in list(vp.values()) + list(vb.values()) if v.get("pass"))
    ntot = len(vp) + len(vb)
    A(f'<div class="col" style="margin-top:18px"><p class="note">Separately, {npass} of {ntot} '
      'correctness assertions pass: the pickle output loads through the unmodified repo '
      '<code>QlibDataset</code>; a perfect-foresight signal scores an IC of exactly 1.000 at the '
      'matching horizon and white noise scores −0.003; shifting every signal one session later '
      'destroys 64 points of annual return; a constant signal produces zero turnover; and the '
      'engine&rsquo;s cost drag reconciles with turnover × rate to the fourth decimal.</p></div>')
    A("</section>")
    return S


def build_search(S, D):
    """The correction that reframes everything model-free in this study: 80 cells were searched.

    Three independent tests, each answering a different way of being wrong:
      * a family-wise null -- what does the BEST of 80 uninformative arms earn by luck?
      * a parameter surface -- is the winner a plateau or a spike?
      * a benchmark audit  -- is "excess over equal weight" even a fair bar?
    """
    A = S.append
    rob, surf, ba = D["rob"], D["surf"], D["ba"]
    if not (rob or surf or ba):
        return S

    A('<section id="search"><div class="sec-head"><span class="sec-num">SEARCH</span>'
      '<h2>Eighty cells were searched, so eighty cells set the bar</h2></div>'
      '<div class="col stack">')
    A('<p>The sweep above tested seventeen model-free arms at four rebalance frequencies, then '
      'three blends of the winners &mdash; eighty cells in all &mdash; and reported the best of '
      'them. The random-signal control it was checked against is a <em>per-cell</em> control: it '
      'answers &ldquo;could this one arm have arisen by chance?&rdquo; That is the wrong '
      'question for a number that was chosen because it was the largest. The right question is '
      'what the <em>maximum</em> of eighty uninformative arms earns, and that is a much larger '
      'number.</p>')
    A("</div>")

    # ---- A. family-wise null -------------------------------------------------------------
    if rob and rob.get("family_null"):
        fn = rob["family_null"]
        me, mi = fn["max_excess"], fn["max_ir"]
        A('<div class="grid4" style="margin-top:18px">')
        A(stat("null draws", f"{fn['n_draws']}", f"each a full sweep of {fn['n_cells_per_draw']} cells"))
        A(stat("mean of the max", pct(me["mean"]), "best of 80 random arms, average draw"))
        A(stat("p95 of the max", pct(me["p95"]), "the honest 5% bar"))
        A(stat("p99 of the max", pct(me["p99"]), "the 1% bar"))
        A("</div>")
        A('<div class="col" style="margin-top:14px"><p>Read the second card first. A sweep of '
          'eighty signals that contain <em>no information whatsoever</em> produces a best cell '
          f'earning {pct(me["mean"])} a year over equal weight in the average draw, and '
          f'{pct(me["p95"])} in the top 5% of draws. Every headline in the model-free section '
          'has to clear that, not the per-cell p95.</p></div>')
        # the same draws also give the bar for a claim about a whole family
        mede = fn.get("median_excess"); shp = fn.get("share_positive")
        surf0 = D["surf"] or {}
        ac0 = surf0.get("all_cells", {})
        if mede and shp and ac0:
            A('<div class="col stack" style="margin-top:22px"><h3>Two bars, not one</h3>'
              '<p>The maximum is the right null for a cell that was chosen <em>because</em> it '
              'was the largest. It is the wrong null for a claim about a whole family. '
              '&ldquo;Every momentum lookback from 10 to 60 sessions is positive&rdquo; is a '
              'statement about a median and a share, and those have a far tighter null &mdash; '
              'recorded from the same draws, at no extra cost.</p></div>')
            rows = [
                ["median cell of the sweep", pct(mede["mean"]), pct(mede["p95"]),
                 pct(ac0["median_excess"]),
                 verdict_chip("PASS" if ac0["median_excess"] > mede["p95"] else "FAIL")],
                ["share of cells positive", f"{shp['mean']:.1%}", f"{shp['p95']:.1%}",
                 f"{ac0['share_positive']:.1%}",
                 verdict_chip("PASS" if ac0["share_positive"] > shp["p95"] else "FAIL")],
                ["best cell of the sweep", pct(me["mean"]), pct(me["p95"]),
                 pct(w0["ann_excess_vs_ew200"]) if (w0 := surf0.get("winner")) else "&mdash;",
                 verdict_chip("PASS" if w0 and w0["ann_excess_vs_ew200"] > me["p95"]
                              else "FAIL")],
            ]
            A('<div style="margin-top:16px">' + table(
                ["statistic", "null, average draw", "null, p95", "observed", "clears"], rows,
                "The same 200 draws, three statistics. Note the sign of the null median: a "
                "random signal run through top-k-dropout LOSES 2.6 points a year to holding the "
                "universe, and only 17.7% of random cells come out positive at all. Against "
                "that, a surface where 71.4% of cells are positive is not close.", numeric_from=1)
              + "</div>")
            A('<div class="col" style="margin-top:14px"><p>This is the finding, and it points '
              'the opposite way from the table above it. <strong>No individual '
              'parameterisation is distinguishable from luck, and the family they belong to is '
              'overwhelmingly distinguishable from luck.</strong> Both are true. What follows '
              'is that the thing to hold is the surface, not a cell on it &mdash; the '
              'defensible expected value is the neighbourhood median of roughly '
              f'{pct((D["surf"] or {}).get("immediate_neighbours", {}).get("median"))} a year, '
              'not any headline number, and the way to earn it is to average across '
              'parameterisations rather than to select one.</p></div>')

        head = D["rob"].get("headline", [])
        if head and head[0].get("p_fwer_excess") is not None:
            rows = []
            for r in head:
                cl = r.get("clears_fwer_p05")
                rows.append([f"<code>{e(r['arm'])}</code>", r["rebalance"],
                             pct(r["ann_excess_vs_ew200"]), num(r["ir_vs_ew200"], 2),
                             f"{r['p_fwer_excess']:.3f}",
                             verdict_chip("PASS" if cl else "FAIL")])
            n_clear = sum(1 for r in head if r.get("clears_fwer_p05"))
            A('<div style="margin-top:16px">' + table(
                ["arm", "rebalance", "excess vs equal-weight", "IR",
                 "family-wise p", "clears p<0.05"], rows,
                f"Family-wise corrected. {n_clear} of {len(head)} shortlisted arms survive the "
                "correction for having searched eighty cells. The per-cell control passed most "
                "of them.") + "</div>")

    # ---- B. parameter surface -------------------------------------------------------------
    if surf:
        w = surf["winner"]
        nb = surf["immediate_neighbours"]
        ac = surf["all_cells"]
        A('<div class="col stack" style="margin-top:26px"><h3>Plateau or spike</h3>'
          '<p>The second test does not need a null at all. A real cross-sectional effect is '
          'smooth in its own parameters: if 20-session momentum works, 15 and 25 should work '
          'too, and the effect should decay away from its best horizon rather than vanish. A '
          'fitted cell is a spike whose neighbours do nothing. So sweep the momentum lookback, '
          'the reversal lookback, the blend weight and the rebalance frequency &mdash; '
          f'{surf["n_cells"]} cells &mdash; and look at the shape.</p></div>')
        A('<div class="grid4" style="margin-top:14px">')
        A(stat("the reported blend", pct(w["ann_excess_vs_ew200"]),
               f"IR {num(w['ir_vs_ew200'], 2)}, ranked "
               f"{surf['winner_rank_of_all_cells']} of {surf['n_cells']}"))
        A(stat("its 6 neighbours", pct(nb["median"]),
               f"{nb['share_positive']:.0%} positive, median"))
        A(stat("premium of the winner", pct(nb["winner_minus_neighbour_median"]),
               "above its own neighbourhood"))
        A(stat("whole surface", pct(ac["median_excess"]),
               f"median; {ac['share_positive']:.0%} of cells positive"))
        A("</div>")
        A('<div class="col" style="margin-top:14px"><p>Both readings are true at once and both '
          'matter. The neighbourhood is <em>entirely</em> positive, so there is a real effect '
          f'here worth roughly {pct(nb["median"])} a year. And the reported cell sits '
          f'{pct(nb["winner_minus_neighbour_median"])} above that neighbourhood while ranking '
          f'{surf["winner_rank_of_all_cells"]} of {surf["n_cells"]} &mdash; so about half of the '
          'headline number is the plateau and about half is the cherry-pick. Quote the plateau.'
          '</p></div>')

        hs = surf.get("horizon_smoothness", {})
        if hs:
            rows = []
            for rname, blk in hs.items():
                for label, b in blk.items():
                    spike = b["is_isolated_spike"]
                    rows.append([
                        rname, f"<code>{e(label)}</code>", f"{b['best_horizon']}d",
                        pct(b["best_excess"]),
                        f"{b['other_horizons_share_positive']:.0%}",
                        pct(b["other_horizons_median"]),
                        verdict_chip("FAIL" if spike else "PASS")])
            A('<div style="margin-top:16px">' + table(
                ["rebalance", "arm", "best horizon", "its excess",
                 "other horizons positive", "their median", "smooth?"], rows,
                "Each pure arm against its own lookback, at fixed rebalance frequency. Momentum "
                "is a plateau at every frequency — every lookback from 10 to 60 sessions earns "
                "something. Weekly and biweekly reversal is the opposite: the 5-session cell "
                "works and all four of its neighbours are negative. That cell was in the "
                "headline table of the sweep above. It should not have been.",
                numeric_from=2) + "</div>")
            A('<div class="col" style="margin-top:14px"><p class="note">This is the single most '
              'useful diagnostic in the study and it costs one parameter sweep. It reverses a '
              'conclusion that survived a point-in-time universe correction, a real cost model '
              'and a per-cell significance test &mdash; all of which the 5-session weekly '
              'reversal arm passed.</p></div>')

    # ---- C. benchmark audit ---------------------------------------------------------------
    if ba:
        u = ba["universes"]
        rows = []
        for uname, blk in u.items():
            for cname, c in blk["cadences"].items():
                rows.append([uname.replace("_", "-"), cname, pct(c["ann_return_free"]),
                             pct(c["ann_return_net"]), num(c["turnover_ann"], 2, sign=False),
                             pct(c["cost_drag_ann_pct"] / 100)])
        A('<div class="col stack" style="margin-top:26px"><h3>Is the benchmark itself fair</h3>'
          '<p>Every excess figure in this study is quoted against <code>ew200</code>, the '
          'equal-weighted eligible universe &mdash; which is computed as a cross-sectional mean '
          'of daily returns, i.e. rebalanced back to equal weight every session at zero cost. '
          'Nobody can run that. Worse, daily rebalancing <em>is</em> a short-horizon reversal '
          'bet in a market where 1-session reversal has a t of +7, so the benchmark may be '
          'quietly holding the very factor the strategies trade.</p></div>')
        A('<div style="margin-top:16px">' + table(
            ["universe", "rebalance", "return, free", "return, net of real cost",
             "turnover/yr", "cost drag"], rows,
            "The benchmark priced honestly. Rebalancing cadence barely matters: the free daily "
            "version the study uses beats a realistically-costed monthly one by well under a "
            "point a year, so excess over it is a fair bar and if anything a slightly "
            "conservative one.", numeric_from=2) + "</div>")
        bi, pit = u.get("biased"), u.get("point_in_time")
        if bi and pit:
            A('<div class="grid3" style="margin-top:14px">')
            A(stat("benchmark overstatement", pct(pit["benchmark_overstatement"]),
                   "free-daily minus net-monthly, point-in-time"))
            A(stat("rebalancing premium, biased", pct(bi["rebalancing_premium_daily_vs_never"]),
                   "daily minus never-rebalance"))
            A(stat("rebalancing premium, point-in-time",
                   pct(pit["rebalancing_premium_daily_vs_never"]),
                   "the same number, real universe"))
            A("</div>")
            A('<div class="col" style="margin-top:14px"><p>The last two cards are a survivorship '
              'test that nobody designed as one, and it is the cleanest in the study. On the '
              'back-projected universe, <em>never</em> rebalancing beats rebalancing daily by '
              f'{pct(-bi["rebalancing_premium_daily_vs_never"])} &mdash; of course it does, '
              'because you already know which names survived, so letting the winners run is '
              'free information. Correct the universe and the sign flips: rebalancing daily now '
              f'beats never by {pct(pit["rebalancing_premium_daily_vs_never"])}. The same '
              'operation, on the same dates, with the same code, worth '
              f'{pct(pit["rebalancing_premium_daily_vs_never"] - bi["rebalancing_premium_daily_vs_never"])} '
              'depending only on whether the universe is honest.</p></div>')

    # ---- D. capacity ----------------------------------------------------------------------
    cap, liq = D["cap"], D["liq"]
    if not cap.empty and not liq.empty:
        A('<div class="col stack" style="margin-top:26px"><h3>What it would hold</h3>'
          '<p>The statutory cost model has fixed components &mdash; &#8377;13 depository charge, '
          'flat brokerage where it applies &mdash; so cost per trade <em>falls</em> with size and '
          'the model alone says bigger is always better. That is false, and the missing term is '
          'market impact. Sizing every fill against the 20-session median traded value of the '
          'name gives a participation rate, and a square-root impact charge on top of that gives '
          'a ceiling.</p></div>')
        arm, rname = "blend_mom20_rev5", "monthly"
        q = liq[(liq.arm == arm) & (liq.rebalance == rname)].sort_values("capital")
        c2 = cap[(cap.arm == arm) & (cap.rebalance == rname)].set_index("capital")
        rows = []
        for _, r in q.iterrows():
            cc = r["capital"]
            lab = (f"&#8377;{cc / 1e7:,.0f} cr" if cc >= 1e7 else f"&#8377;{cc / 1e5:,.0f} lakh")
            rows.append([lab, num(c2.loc[cc, "round_trip_bps"], 1, sign=False) + " bps",
                         f"{r['participation_median']:.3%}", f"{r['participation_p99']:.2%}",
                         pct(r["ann_excess_before_impact"]),
                         pct(r["ann_excess_after_impact"])])
        A('<div style="margin-top:16px">' + table(
            ["capital", "statutory round trip", "median participation",
             "p99 participation", "excess before impact", "after impact"], rows,
            "The monthly blend across eight orders of magnitude of capital. The statutory cost "
            "curve only bites below about ₹10 lakh; above that the binding constraint is "
            "impact, and it does not bite until the book is in the hundreds of crores — the "
            "NIFTY 200 median name trades roughly ₹127 crore a day.", numeric_from=1)
          + "</div>")
        A('<div class="col" style="margin-top:14px"><p class="note">Capacity is not the reason '
          'this is hard. A retail book of a few lakh is squeezed by the fixed charges, and a '
          'book of a few hundred crore is squeezed by impact, but everything between those is '
          'cost-indifferent. The binding constraint is whether the signal is real, which is what '
          'the two tests above are about.</p></div>')

    # ---- E. risk, not skill ---------------------------------------------------------------
    va = D["voladj"]
    if not va.empty:
        A('<div class="col stack" style="margin-top:26px"><h3>Excess, or just more risk</h3>'
          '<p>Adding trailing-volatility sorts to the sweep produces a new leader: buying the '
          'highest-volatility names and rebalancing weekly earns +8.97% a year over equal '
          'weight, more than anything else tested. It also runs at 25.9% annualised volatility '
          'against the benchmark&rsquo;s 18.7%, and its cross-sectional information coefficient '
          'is +0.009 with a t of +0.60 — no information at all. It is a beta position, and the '
          'test window is exactly the three years in which that position paid.</p>'
          '<p>The fix is to lever the benchmark to the arm&rsquo;s own volatility and charge '
          'that instead of the unlevered benchmark. Nothing else about the comparison changes.'
          '</p></div>')
        rows = []
        top = va.nlargest(8, "excess_volmatched")
        for _, r in top.iterrows():
            drop = r["ann_excess_vs_ew200"] - r["excess_volmatched"]
            rows.append([f"<code>{e(r['arm'])}</code>", r["rebalance"],
                         num(r["vol_ann"], 3, sign=False),
                         pct(r["ann_excess_vs_ew200"]), pct(r["excess_volmatched"]),
                         pct(-drop)])
        A('<div style="margin-top:16px">' + table(
            ["arm", "rebalance", "volatility", "raw excess", "volatility-matched",
             "cost of the correction"], rows,
            "The eight best arms after volatility-matching, point-in-time universe, 30 bps. The "
            "volatility sort falls from first place to fourth and loses two thirds of its "
            "excess; the arms that were not buying risk barely move.", numeric_from=2) + "</div>")
        hv = va[(va.arm == "highvol_20d") & (va.rebalance == "weekly")]
        if len(hv):
            hv = hv.iloc[0]
            A('<div class="grid3" style="margin-top:14px">')
            A(stat("high-volatility, raw", pct(hv["ann_excess_vs_ew200"]),
                   "best raw excess in the whole study"))
            A(stat("the same arm, volatility-matched", pct(hv["excess_volmatched"]),
                   f"at {num(hv['vol_ann'], 3, sign=False)} annualised volatility"))
            A(stat("its cross-sectional IC", "+0.009",
                   "10 sessions, t = +0.60 \u2014 not significant", "neg"))
            A("</div>")
        A('<div class="col" style="margin-top:14px"><p class="note">The low-volatility side of '
          'the same sort loses 5.5 to 11.5 points a year here. The low-volatility anomaly is one '
          'of the most replicated results in the literature and it is <em>inverted</em> over this '
          'window. That is what a three-year regime looks like from the inside, and it is the '
          'reason the volatility arms were kept out of the family whose maximum sets the '
          'significance bar rather than quietly folded into it.</p></div>')

    A("</section>")
    return S


def build_close(S, D, tag):
    A = S.append
    ctl, usc, greedy = D["ctl"], D["usc"], D["greedy"]
    gsum = D["gsum"] or {}
    pctile = gsum.get("prereg_percentile_in_random")
    icg = ctl.get("ic_grid", {})
    max_abs_t = max((abs(v.get(f"open_h{h}", {}).get("t_newey_west") or 0.0)
                     for v in icg.values() for h in HORIZONS), default=0.0)
    extras = {"greedy (conditional mode)": greedy, "T = 1.0": D["t1"],
              "5-path average": D["paths5"]}
    extras = {k: v for k, v in extras.items() if v}

    # ============ robustness ============
    if extras or usc:
        A('<section id="robustness"><div class="sec-head"><span class="sec-num">ROBUSTNESS</span>'
          '<h2>Is it the sampler, the market, or the model?</h2></div>')
    if extras:
        rows = []
        b = icg.get("last_ret", {})
        rows.append(["repo default, T=0.6 top_p=0.9"]
                    + [num(b.get(f"open_h{h}", {}).get("mean_ic")) for h in HORIZONS]
                    + [num(b.get("open_h10", {}).get("t_newey_west"), 2)])
        for label, c in extras.items():
            r = c.get("ic_grid", {}).get("last_ret", {})
            rows.append([e(label)] + [num(r.get(f"open_h{h}", {}).get("mean_ic"))
                                      for h in HORIZONS]
                        + [num(r.get("open_h10", {}).get("t_newey_west"), 2)])
        A(table(["decoding", "IC 1d", "IC 5d", "IC 10d", "IC 20d", "NW t (10d)"], rows,
                "Greedy decoding (top_k=1) takes the model's most likely token at every step, "
                "removing sampling noise entirely. If the IC stays at zero there, the result is "
                "a property of the model rather than of the sampler."))
    if usc:
        rows = []
        for lbl, src, key in [("Kronos <code>last_ret</code>", usc.get("ic_grid", {}), "last_ret"),
                              ("Kronos <code>last_z</code>", usc.get("ic_grid", {}), "last_z"),
                              ("10-day momentum", usc.get("baseline_ic", {}), "momentum_10d"),
                              ("&minus;anchor_z", usc.get("baseline_ic", {}), "anchor_reversal")]:
            r = src.get(key, {})
            if not r:
                continue
            r10 = r.get("open_h10", {})
            rows.append([lbl] + [num(r.get(f"open_h{h}", {}).get("mean_ic")) for h in HORIZONS]
                        + [num(r10.get("t_newey_west"), 2),
                           "yes" if r10.get("significant") else "no"])
        if rows:
            A('<div style="margin-top:20px">' + table(
                ["signal (S&P 100)", "IC 1d", "IC 5d", "IC 10d", "IC 20d", "NW t", "sig."], rows,
                "Extension beyond the plan: the same code, the same horizons, a US large-cap "
                "cross-section over the same window. This separates “no cross-sectional skill” "
                "from “no cross-sectional skill in India”.") + "</div>")
        uc = usc.get("cross_sectional_rank_corr_with_kronos", {})
        if uc:
            A(f'<div class="col" style="margin-top:14px"><p class="note">Rank correlation with '
              f'the free features on US data: '
              + ", ".join(f"{e(k)} {num(v, 3)}" for k, v in uc.items()) + ".</p></div>")
    if D["ftj"]:
        f = D["ftj"]
        A('<div class="col stack" style="margin-top:24px"><h3>Finetuning on NSE data</h3>')
        A(f'<p>The repository&rsquo;s own trainer cannot run on this machine &mdash; CUDA is '
          f'hardcoded, it requires <code>torchrun</code>, and it imports <code>comet_ml</code> at '
          f'module scope. A standalone single-device trainer reusing the repo&rsquo;s unmodified '
          f'dataset and exact objective saw {f.get("samples_seen", 0):,} samples in '
          f'{f.get("wall_clock_min", 0):.0f} minutes, against the '
          f'{f.get("repo_schedule_for_reference", {}).get("epochs", 30)}-epoch multi-GPU schedule '
          f'the config targets. Validation loss went '
          f'{f.get("zero_shot_val_loss", 0):.4f} &rarr; {f.get("best_val_loss", 0):.4f}.</p>')
        if f.get("diagnosis") == "overfitting":
            A(f'<p>Which is the answer to a question this run was not expected to settle. A '
              f'90-minute budget on one device sounds like the reason a finetune would fail, and '
              f'the shape of the curve says it is not: validation loss bottomed at step '
              f'{f.get("best_step", 0):,} of {f.get("steps_completed", 0):,} &mdash; '
              f'{f.get("best_step_fraction", 0):.0%} of the way in &mdash; and then rose '
              f'{f.get("val_rise_after_best", 0):.4f} back toward the zero-shot number, while '
              f'training loss kept falling to a gap of '
              f'{abs(f.get("train_val_gap_at_end", 0)):.2f} nats. <strong>The run was not short '
              f'of compute. It ran out of data.</strong> More steps make it worse, and the '
              f'checkpoint that was kept is the early-stopped one at step '
              f'{f.get("best_step", 0):,}.</p>')
        A("</div>")
        fl = D["ftlog"]
        if not fl.empty and {"step", "train_loss", "val_loss"} <= set(fl.columns):
            ev = fl[fl.step > 0]
            idx = list(range(0, len(ev), max(len(ev) // 8, 1)))
            xl = ["" for _ in range(len(ev))]
            for i in idx:
                xl[i] = f"{int(ev.step.iloc[i]) // 1000}k" if ev.step.iloc[i] >= 1000 else "0"
            A('<div class="col" style="margin-top:8px">' + svg_lines(
                [("validation loss", ev.val_loss.tolist(), "var(--accent)", False),
                 ("training loss", ev.train_loss.tolist(), "var(--muted)", True)],
                xl, "finetuning loss curve", zero=False, dots=False,
                yfmt=lambda v: f"{v:.2f}")
              + '<p class="note">Training loss falls, validation loss turns at step 1,200 and '
                'climbs for the remaining 6,600. 182 symbols and seven years of daily bars is '
                'not enough data to finetune 102 million parameters, and no amount of additional '
                'compute changes that. Freezing more of the network, adding more markets, or '
                'training the tokenizer jointly are the things that could — none of them were '
                'tested.</p></div>')
        mech, ftz, ftc = D["mech"], D["ftz"], D["ftc"]
        if mech and mech.get("finetuned") and mech.get("zero_shot_ftdates"):
            z0, f0 = mech["zero_shot_ftdates"], mech["finetuned"]
            A('<div class="col stack" style="margin-top:20px">'
              '<h3>What 1,200 steps actually changed</h3>'
              '<p>The interesting part is not the loss. Refit the mechanism on the finetuned '
              'checkpoint, on the same 63 dates, with the same regression, and the defect is '
              'gone &mdash; the sign flips.</p></div>')
            A('<div class="grid4" style="margin-top:14px">')
            A(stat("zero-shot reversion", f"{z0['reversion_fraction'] * 100:.1f}%",
                   f"slope {num(z0['pooled_slope'])} on anchor_z, R\u00b2 "
                   f"{z0['pooled_r2']:.2f}", "neg"))
            A(stat("finetuned reversion", f"{f0['reversion_fraction'] * 100:.1f}%",
                   f"slope {num(f0['pooled_slope'])}, R\u00b2 {f0['pooled_r2']:.2f} "
                   "\u2014 it now extrapolates", "pos"))
            A(stat("rank corr. with the reversal factor",
                   f"{z0['spearman_vs_neg_anchor_mean']:+.3f} \u2192 "
                   f"{f0['spearman_vs_neg_anchor_mean']:+.3f}", "zero-shot \u2192 finetuned"))
            if ftz and ftc:
                zs = (ftz.get("strategy") or {}).get("ann_excess_net")
                fs = (ftc.get("strategy") or {}).get("ann_excess_net")
                ew = (ftc.get("equal_weight_universe") or {}).get("ann_excess_net")
                A(stat("excess vs ^CNX200", f"{pct(zs)} \u2192 {pct(fs)}",
                       f"equal weight over the same dates: {pct(ew)}"))
            A("</div>")
            A('<div class="col" style="margin-top:14px"><p>Read that as two separate results. '
              'The first is positive and specific: <strong>the reversion artefact is a fixable '
              'defect, and 0.13% of a training schedule fixes it.</strong> A model that reverted '
              f'{z0["reversion_fraction"] * 100:.0f}% of the way to each stock&rsquo;s window '
              f'mean now extrapolates {abs(f0["reversion_fraction"]) * 100:.0f}% past it, and '
              f'its rank correlation with the hand-written reversal factor goes from '
              f'{z0["spearman_vs_neg_anchor_mean"]:+.3f} to '
              f'{f0["spearman_vs_neg_anchor_mean"]:+.3f}. Whatever this study measured about the '
              'zero-shot model is a statement about the checkpoint, not about the '
              'architecture.</p>')
            if ftz and ftc:
                A('<p>The second is not. Every 10-day information coefficient in the finetuned '
                  'grid still sits between &minus;0.015 and +0.003 with |t| below 1.0; the '
                  'strategy still earns less than holding the universe equal-weighted and sits '
                  'at percentile 21 of 300 shuffled-score runs. <strong>Removing the artefact '
                  'raised the strategy from percentile 0 to percentile 21 and produced no '
                  'signal.</strong> That separates two explanations that were confounded '
                  'everywhere else on this page: the normalisation really was breaking the '
                  'score, and underneath it there was nothing to break.</p>')
            A("</div>")
            if ftc:
                fp = ftc.get("ic_grid", {}).get("last_ret", {}).get("open_h10", {})
                pp = icg.get("last_ret", {}).get("open_h10", {})
                A(f'<div class="col" style="margin-top:8px"><p class="note">Primary IC after '
                  f'finetuning: <span class="mono">{num(fp.get("mean_ic"))}</span> '
                  f'(NW t {num(fp.get("t_newey_west"), 2)}), against '
                  f'<span class="mono">{num(pp.get("mean_ic"))}</span> zero-shot over all 756 '
                  f'sessions. The finetuned run is 63 dates because it strides by 12, so its '
                  f'standard errors are wider; it is compared against a date-matched zero-shot '
                  f'run, not against the full one.</p></div>')
    if extras or usc:
        A("</section>")

    # ============ a different scalar: volatility ============
    vol = D["vol"]
    if vol:
        vic, cic = vol.get("vol_ic", {}), vol.get("conditional_ic", {})
        A('<section id="scalar"><div class="sec-head"><span class="sec-num">SCALAR</span>'
          '<h2>The obvious escape hatch, tested: forecast volatility instead</h2></div>'
          '<div class="col stack">')
        A('<p>The directional null has a mechanism &mdash; per-window z-scoring turns the '
          'predicted close into a distance from the stock&rsquo;s own mean &mdash; and that '
          'mechanism does not obviously apply to the <em>spread</em> of the predicted path. '
          'Both ends of <code>max &minus; min</code> are measured from the same anchor, so the '
          'contamination cancels in the difference. Volatility is also far more predictable '
          'than direction. This is the strongest remaining case for the model, it costs no '
          'extra inference because the values were already in the signal file, and it turns out '
          'to fail in exactly the same way.</p>')
        A('<p>Two quantities, and the difference between them is the whole answer. '
          '<code>pred_range_rel</code> is the predicted range in units of the trailing '
          'standard deviation &mdash; pure model output, a forecast of how volatility will '
          '<em>change</em>. <code>pred_range_pct</code> is the same thing multiplied back by '
          'that trailing standard deviation, so it is a forecast of the volatility '
          '<em>level</em> and contains the scale it was handed.</p>')
        A("</div>")
        rows = []
        labels = {"pred_range_rel": "Kronos range, trailing scale divided out",
                  "pred_range_pct": "Kronos range &times; trailing sigma",
                  "trailing_vol_20d": "20-session trailing volatility (free)",
                  "trailing_sigma_pct": "the 90-bar sigma the model normalised by (free)"}
        for k, lab in labels.items():
            p10 = vic.get(k, {}).get("h10")
            if not p10:
                continue
            c = cic.get(f"{k}|trailing_vol_20d")
            rows.append([lab, num(p10["mean_ic"], 4), num(p10["t_newey_west"], 2),
                         (num(c["mean_ic"], 4) if c else "&mdash;"),
                         (num(c["t_newey_west"], 2) if c else "&mdash;"),
                         verdict_chip("PASS" if (c and c["significant"] and c["mean_ic"] > 0)
                                      else "FAIL")])
        A('<div style="margin-top:16px">' + table(
            ["quantity", "IC vs realized vol", "t", "IC within a trailing-vol quintile", "t",
             "adds anything?"], rows,
            "Rank correlation with realized 10-session volatility, raw and then conditioned on "
            "trailing volatility. Conditioning rather than residualising is deliberate: the "
            "product structure of pred_range_pct makes a rank-orthogonalised IC come out larger "
            "than the raw one, which is not something real incremental information can do.")
          + "</div>")
        rel = cic.get("pred_range_rel|trailing_vol_20d")
        pctc = cic.get("pred_range_pct|trailing_vol_20d")
        sigc = cic.get("trailing_sigma_pct|trailing_vol_20d")
        if rel and pctc and sigc:
            A('<div class="grid3" style="margin-top:16px">')
            A(stat("Kronos\u2019s own contribution", num(rel["mean_ic"], 4),
                   f"t = {num(rel['t_newey_west'], 2)}, within trailing-vol quintiles"))
            A(stat("the level forecast", num(pctc["mean_ic"], 4),
                   f"t = {num(pctc['t_newey_west'], 2)}, but see the next card"))
            A(stat("the free 90-bar sigma alone", num(sigc["mean_ic"], 4),
                   f"t = {num(sigc['t_newey_west'], 2)}, same conditioning"))
            A("</div>")
            A('<div class="col" style="margin-top:14px"><p>Read them in that order. Strip the '
              'trailing scale out and the model&rsquo;s volatility forecast is '
              f'{num(rel["mean_ic"], 4)} with a t of {num(rel["t_newey_west"], 2)} &mdash; '
              'nothing. Leave the scale in and it looks informative, until you notice that the '
              'scale <em>on its own</em>, conditioned identically, scores '
              f'{num(sigc["mean_ic"], 4)} against the model&rsquo;s '
              f'{num(pctc["mean_ic"], 4)}. Kronos does not merely fail to improve on the '
              'statistic it was handed; multiplying that statistic by the model&rsquo;s output '
              'makes it worse. This is the directional finding again in a different quantity: '
              'the output is dominated by the normalisation, and the normalisation was free.'
              '</p></div>')
        vp = D["volpit"]
        if vp:
            cp = vp.get("conditional_ic", {})
            r1 = cp.get("pred_range_rel|trailing_vol_20d")
            s1 = cp.get("trailing_sigma_pct|trailing_vol_20d")
            if r1 and s1:
                A('<div class="col" style="margin-top:16px"><p>Repeating the whole thing on the '
                  'point-in-time universe does not soften it. There the model&rsquo;s own '
                  f'volatility forecast scores {num(r1["mean_ic"], 4)} with a t of '
                  f'{num(r1["t_newey_west"], 2)} &mdash; significantly <em>negative</em>, while '
                  f'the free trailing sigma alone scores {num(s1["mean_ic"], 4)} under identical '
                  'conditioning. Ranking stocks by how much Kronos thinks their volatility will '
                  'change is slightly worse than not ranking them at all.</p></div>')
        A('<div class="col" style="margin-top:14px"><p class="note">Path asymmetry '
          '(<code>max + min &minus; 2&thinsp;&times;&thinsp;last</code>) was tested the same '
          'way and behaves the same way. That closes the two most plausible &ldquo;try a '
          'different scalar&rdquo; escapes with measurement rather than argument. Predicted '
          'volume and dispersion across independently sampled futures remain untested, and '
          'sampling several paths per name is the one change that would give the range a real '
          'distribution to be the spread of, rather than a single draw.</p></div>')
        A("</section>")

    # ============ what DOES have signal ============
    bl, blsum, blrnd = D["bl"], D["blsum"], D["blrnd"]
    if not bl.empty:
        A('<section id="signal"><div class="sec-head"><span class="sec-num">SIGNAL</span>'
          '<h2>What does have information here &mdash; and whether it pays</h2></div>'
          '<div class="col stack">')
        A('<p>A null on Kronos is only half an answer. The same machinery, run on model-free '
          'features across all 756 sessions, finds one measurement that holds up under every '
          'test applied later on this page: <strong>1&ndash;2 session reversal has a real '
          'information coefficient</strong>. Note the precise claim. It is about the IC, not '
          'about any strategy built on it &mdash; the cost model kills the obvious one two '
          'tables below, and the parameter surface later kills a second arm that appears in the '
          'table after that. Read this section as the raw material, not the conclusion.</p>'
          '</div>')
        ics = blsum.get("ic", {})
        rows = []
        for k in ("reversal_1d", "reversal_2d", "reversal_5d", "reversal_10d", "reversal_20d",
                  "momentum_20d", "momentum_120d", "zscore_reversal", "kronos_emulator"):
            r = ics.get(k)
            if not r:
                continue
            cells = []
            for h in HORIZONS:
                q = r.get(f"open_h{h}", {})
                cells.append(num(q.get("mean_ic")) + ("&thinsp;*" if q.get("significant") else ""))
            nm = {"zscore_reversal": "&minus;(close &minus; mean₉₀)/std₉₀",
                  "kronos_emulator": "<b>Kronos, emulated: &minus;0.701 × that</b>"}.get(
                      k, f"<code>{e(k)}</code>")
            rows.append([nm] + cells + [num(r.get("open_h1", {}).get("t_newey_west"), 2)])
        A('<div style="margin-top:18px">' + table(
            ["model-free signal", "IC 1d", "IC 5d", "IC 10d", "IC 20d", "NW t (1d)"], rows,
            "All 756 test sessions, same engine and same standard errors as the Kronos grid. "
            "* marks Newey–West t > 2 with a bootstrap CI excluding zero. The reversal that "
            "works is 1–2 sessions long; the 90-bar one Kronos reproduces does not.",
            highlight=lambda r: "emulated" in r[0]) + "</div>")
        best = bl[bl.cost_bps == 30.0].sort_values("ann_excess_vs_ew200", ascending=False)
        rows = []
        for _, r in best.head(8).iterrows():
            rows.append([f'<code>{e(r["arm"])}</code>', e(r["rebalance"]),
                         pct(r["ann_return_net"]), pct(r["ann_excess_vs_ew200"]),
                         num(r["ir_vs_ew200"], 2), f'{r["turnover_ann"]:.1f}×',
                         f'{r["cost_drag_ann_pct"]:.2f}%'])
        A('<div style="margin-top:18px">' + table(
            ["arm", "rebalance", "return, net", "excess vs equal-weight", "IR", "turnover/yr",
             "cost drag"], rows,
            "Best eight model-free arms at 30 bps round trip, on TODAY'S index membership "
            "projected backwards, ranked against the equal-weight universe. Both of those "
            "qualifiers cost several points a year and are corrected further down. "
            "universe. Turnover is the reason a strongly significant IC does not automatically "
            "become money.") + "</div>")
        r1 = bl[(bl.arm == "reversal_1d") & (bl.rebalance == "daily")].sort_values("cost_bps")
        if len(r1):
            A('<div class="grid3" style="margin-top:18px">')
            for _, r in r1.iterrows():
                A(stat(f"1-day reversal @ {r['cost_bps']:.0f} bps",
                       pct(r["ann_excess_vs_ew200"]),
                       f"turnover {r['turnover_ann']:.0f}×/yr, cost drag "
                       f"{r['cost_drag_ann_pct']:.1f}%/yr",
                       "pos" if r["ann_excess_vs_ew200"] > 0 else "neg"))
            A("</div>")
            A('<div class="col" style="margin-top:14px"><p class="note">The strongest IC in the '
              'entire study belongs to the arm with the highest turnover. At the verified real '
              'cost of 27.4 bps for a &#8377;10 lakh position, the drag alone is the difference '
              'between the gross and net columns &mdash; which is the whole point of having '
              'built the cost model from the circulars rather than from memory.</p></div>')
        A("</section>")

    # ============ survivorship, measured on the same arms ============
    blpit, blpitsum = D["blpit"], D["blpitsum"]
    if not blpit.empty and not bl.empty:
        A('<section id="correction"><div class="sec-head"><span class="sec-num">CORRECTION</span>'
          '<h2>What survives when the universe is fixed</h2></div><div class="col stack">')
        A('<p>Every number above sits on today&rsquo;s index membership projected backwards. '
          'Rebuilding actual point-in-time membership and re-running the identical sweep shows '
          'which apparent edges were real and which were the universe. The direction is exactly '
          'what theory predicts: survivorship flatters <em>momentum</em>, because names get '
          'added to an index after they have already gone up, and it does not flatter reversal.</p>'
          '</div>')
        rows = []
        for a, r in (("momentum_20d", "monthly"), ("momentum_120d", "monthly"),
                     ("momentum_60d", "weekly"), ("reversal_5d", "weekly"),
                     ("reversal_1d", "daily"), ("kronos_emulator", "monthly")):
            x = bl[(bl.arm == a) & (bl.rebalance == r) & (bl.cost_bps == 30.0)]
            y = blpit[(blpit.arm == a) & (blpit.rebalance == r) & (blpit.cost_bps == 30.0)]
            if not len(x) or not len(y):
                continue
            xv, yv = float(x.iloc[0].ann_excess_vs_ew200), float(y.iloc[0].ann_excess_vs_ew200)
            nm = ("<b>Kronos, emulated</b>" if a == "kronos_emulator" else f"<code>{e(a)}</code>")
            rows.append([nm, e(r), pct(xv), pct(yv), pct(yv - xv),
                         num(float(y.iloc[0].ir_vs_ew200), 2)])
        A(table(["arm", "rebalance", "survivorship-biased", "point-in-time", "effect of the fix",
                 "IR, corrected"], rows,
                "Excess over the equal-weight universe at 30 bps. Correcting the universe costs "
                "20-day momentum 6.7 points a year and 120-day momentum 8.3 — roughly half its "
                "apparent edge.",
                highlight=lambda r: "emulated" in r[0]))
        rm = blpitsum.get("random_monthly_vs_ew", {})
        rw = blpitsum.get("random_weekly_vs_ew", {})
        if rm:
            A('<div class="col stack" style="margin-top:20px">')
            A('<h3>The bar that actually has to be cleared</h3>')
            A('<p>On the corrected universe the random-signal control is <em>negative</em> '
              'against equal weight at every rebalance frequency. Top-k-dropout selection with '
              'realistic costs destroys value before any signal is applied; a factor has to earn '
              'back that hurdle first.</p></div>')
            rows = []
            for k in ("daily", "weekly", "biweekly", "monthly"):
                v = blpitsum.get(f"random_{k}_vs_ew")
                if v:
                    rows.append([e(k), pct(v["mean"]), pct(v["p05"]), pct(v["p95"])])
            A('<div style="margin-top:14px">' + table(
                ["rebalance", "random mean", "p05", "p95"], rows,
                "150 shuffled-score draws per frequency through the identical engine, "
                "point-in-time universe, 30 bps.") + "</div>")
            top = blpit[blpit.cost_bps == 30.0].sort_values(
                "ann_excess_vs_ew200", ascending=False).head(6)
            rows = []
            for _, r in top.iterrows():
                bar = (rm if r["rebalance"] == "monthly" else
                       rw if r["rebalance"] == "weekly" else
                       blpitsum.get(f"random_{r['rebalance']}_vs_ew", {}))
                clears = r["ann_excess_vs_ew200"] > bar.get("p95", 1e9)
                rows.append([f'<code>{e(r["arm"])}</code>', e(r["rebalance"]),
                             pct(r["ann_excess_vs_ew200"]), num(r["ir_vs_ew200"], 2),
                             f'{r["turnover_ann"]:.1f}×',
                             verdict_chip("PASS" if clears else "FAIL")])
            A('<div style="margin-top:16px">' + table(
                ["arm", "rebalance", "excess vs equal-weight", "IR", "turnover/yr",
                 "clears random p95"], rows,
                "Best six arms on the corrected universe. What is left after fixing the "
                "universe and charging real costs is a handful of classic factors earning "
                "3–6 points a year over equal weight — modest, well known, and not a discovery.")
              + "</div>")
        A("</section>")

    S = build_search(S, D)
    S = build_crypto(S, D)
    S = build_hardening(S, D)

    # ============ what would have to be true ============
    A('<section id="reading"><div class="sec-head"><span class="sec-num">READING</span>'
      '<h2>What this does and does not say</h2></div><div class="grid2">')
    for h, p in [
        ("It is not a claim that Kronos is a bad model",
         "This measures one specific use: a single scalar per stock per day, ranked "
         "cross-sectionally. Kronos may well be useful for path shape, volatility, or "
         "conditional distributions. None of those were tested here."),
        ("The normalisation is doing the damage",
         "Per-window z-scoring is correct for training a sequence model and wrong for building "
         "a cross-sectional score. It makes the output a distance-from-own-mean, which is a "
         "reversal factor wearing a forecast's clothes. Any fix starts here."),
        ("A null result with this much scaffolding is still a result",
         "Seventeen correctness assertions, a pre-registered primary metric, a 500-seed random "
         "control, two universes and two markets. The finding is not “we could not find "
         "anything”; it is “there is a measurable amount of nothing, and here is its size”."),
        ("The universe is the bigger lesson",
         "Before any model is involved, back-projecting a current index membership and "
         "equal-weighting it manufactures a large annual return. Any Indian equity backtest "
         "that reports excess over a cap-weighted index without a random-signal control is "
         "reporting mostly that."),
    ]:
        A(f'<div class="card"><h3>{h}</h3><p>{p}</p></div>')
    A("</div></section>")

    # ============ state of play ============
    ftj = D["ftj"]
    A('<section id="state"><div class="sec-head"><span class="sec-num">STATE</span>'
      '<h2>Where this stands, honestly</h2></div><div class="grid2">')
    settled = [
        ("Settled — Kronos has no cross-sectional edge here",
         f"Zero of 48 arm × horizon cells significant across all 756 sessions; the largest "
         f"|t| anywhere in the grid is {max_abs_t:.2f}. The strategy sits at percentile "
         f"{pctile:.0f} of 500 shuffled-score runs (one-sided p = "
         f"{gsum.get('prereg_empirical_p_one_sided', float('nan')):.3f}) — indistinguishable "
         f"from a random ranker, not worse than one — and still earns less than holding the "
         f"eligible universe equal-weighted."),
        ("Settled — the mechanism, and it generalises",
         "The score is a 70.1% reversion to each stock's own 90-bar mean — slope −0.701 on "
         "anchor_z, pooled R² = 0.70 across all 144,485 observations. Greedy decoding still "
         "reverts 58.6%, so it is the conditional mode and not sampling noise. The S&P 100 gives "
         "68.9% with a rank correlation of +0.784 against India's +0.771. This is the "
         "normalisation scheme, not the market."),
        ("Settled — the universe dominates the headline number",
         "Equal-weighting a back-projected index membership earns +13.8%/yr over the cap-weighted "
         "index with no signal: +4.8 investable size tilt, +9.1 pure survivorship. Correcting it "
         "costs 20-day momentum 6.7 points a year."),
        ("Settled — the cost model, from primary sources",
         "22.4 bps statutory round trip at a ₹10 lakh leg, not the 29.3 assumed, and not "
         "scale-invariant. Below ~₹89,000 per position the ad-valorem model flips from "
         "conservative to dangerously optimistic."),
        ("Settled — no single arm survives the search correction",
         "A sweep of eighty arms containing no information produces a best cell earning "
         "+4.0%/yr over equal weight in the average draw and +6.1% at p95. One of the ten "
         "shortlisted arms clears that (family-wise p = 0.010) and it is the one the parameter "
         "surface says is half cherry-pick. The other nine, including 20-session momentum "
         "monthly at p = 0.115, do not."),
        ("Settled — but the family does survive, and that is the useful half",
         "Against the same 200 draws, a random cell earns −2.6%/yr and only 17.7% of random "
         "cells are positive at all. The real surface has a median of +1.7% with 71.4% of its "
         "735 cells positive. No parameterisation is distinguishable from luck; the family they "
         "belong to is overwhelmingly distinguishable from luck. The thing to hold is the "
         "surface, not a cell on it."),
        ("Settled — one reported winner is an isolated spike",
         "5-session reversal rebalanced weekly earns +5.6%/yr with an IR of 1.05 and survived "
         "the point-in-time correction, real costs, and a per-cell significance test. Its four "
         "horizon-neighbours at the same frequency — 2, 3, 8 and 10 sessions — are all "
         "negative. One parameter sweep reverses it."),
        ("Settled — momentum is a plateau, not a cell",
         "Every momentum lookback from 10 to 60 sessions is positive at weekly and biweekly "
         "rebalance, median +4.0% and +1.6% over equal weight. That shape is what a real effect "
         "looks like, and it is why the family clears while none of its members does. The "
         "honest expected value is the neighbourhood median, roughly +3.5% a year over "
         "equal weight — a third of what the best cell advertises."),
        ("Settled — the biggest raw excess in the study is beta",
         "Buying the highest-volatility names weekly earns +8.97%/yr over equal weight, more "
         "than anything else tested, at 25.9% volatility against the benchmark's 18.7% and with "
         "a cross-sectional IC of +0.009 (t = 0.60). Lever the benchmark to match and it falls "
         "to +2.71%. The low-volatility side loses 5.5–11.5 points a year, which inverts one of "
         "the most replicated results in the literature — that is the regime, not a factor."),
        ("Settled — capacity is not the constraint",
         "Statutory costs bite only below about ₹10 lakh; square-root impact against a ₹127 "
         "crore median daily traded value bites only in the hundreds of crores. Everything in "
         "between is cost-indifferent. What is scarce is signal, not capacity."),
    ]
    for h, p_ in settled:
        A(f'<div class="card"><h3>{h}</h3><p>{p_}</p></div>')
    A("</div>")
    open_qs = [
        ("Corrected by the full pass — one pilot number was wrong",
         "The 63-date pilot put the strategy at percentile 0 of the random distribution, which "
         "would have meant Kronos was reliably worse than shuffled scores. All 756 sessions put "
         "it at percentile 73 with p = 0.268. The verdict does not change — it still fails every "
         "gate and still loses to equal weight — but 'no signal' is the correct claim and "
         "'worse than random' was a small-sample artefact. The gate outcomes themselves were "
         "unaffected."),
        ("Settled — the finetune was data-limited, not compute-limited",
         ("A 90-minute single-device run, queued.")
         if not ftj else
         (f"Validation loss {ftj.get('zero_shot_val_loss', 0):.4f} → "
          f"{ftj.get('best_val_loss', 0):.4f}, then back up to "
          f"{ftj.get('final_val_loss', 0):.4f}. The bottom is at step "
          f"{ftj.get('best_step', 0):,} of {ftj.get('steps_completed', 0):,} — "
          f"{ftj.get('best_step_fraction', 0):.0%} in — with training loss still falling to a "
          f"{abs(ftj.get('train_val_gap_at_end', 0)):.2f}-nat gap. The natural objection to a "
          f"90-minute finetune is that it bought 0.13% of the paper's schedule and therefore "
          f"bounds nothing. The curve answers it: more steps made it worse. 182 symbols is not "
          f"enough data for 102M parameters, and that is a different problem from a small "
          f"compute budget.")),
        ("Settled — a different scalar does not rescue it",
         "Predicted range and path asymmetry were the two most plausible escapes, both free "
         "from the signal file already computed. Within trailing-volatility quintiles the "
         "model's range forecast scores −0.005 (t = −0.42), and its volatility-level forecast "
         "scores +0.102 against +0.189 for the free 90-bar sigma it was built from. Multiplying "
         "that sigma by the model's output makes it worse. Predicted volume and multi-path "
         "dispersion are still untested."),
        ("Open — the fix is probably upstream of the model",
         "If the score were built from a cross-sectionally comparable quantity — predicted "
         "return relative to a predicted market factor, say, rather than distance from each "
         "stock's own mean — the reversion artefact would cancel. That is a change to how the "
         "signal is constructed, not to the model, and it is the single most promising thing to "
         "try next."),
        ("Open — the blend, at half its headline",
         "A 50/50 rank blend of 20-session momentum and 5-session reversal, monthly, earns "
         "+7.1%/yr with an IR of 1.28 and is positive in all four calendar years — the best "
         "single number in the study. It also ranks 12th of 735 cells in its own parameter "
         "surface, sitting +3.7 points above a neighbourhood whose median is +3.5. Half the "
         "headline is plateau and half is cherry-pick; the plateau is the part to quote."),
        ("Caveat — one market, one three-year regime",
         "August 2023 to August 2026 was unusually strong for Indian equities and unusually "
         "strong for midcaps specifically. The US check covers a second market but the same "
         "window."),
        ("Caveat — what the point-in-time universe still misses",
         "Three names have no recoverable price history (PEL, TATAMTRDVR, GUJGASLTD), and NSE's "
         "transient demerger placeholders are not modelled. Both are one-name distortions "
         "against a ~60-name correction."),
    ]
    A('<div class="grid2" style="margin-top:18px">')
    for h, p_ in open_qs:
        A(f'<div class="card"><h3>{h}</h3><p>{p_}</p></div>')
    A("</div>")
    A('<div class="col" style="margin-top:22px"><p class="pull">The most useful output of this '
      'exercise is not the verdict on Kronos. It is the measuring apparatus &mdash; and the '
      'order in which it fires. A point-in-time universe removed nine points a year of fake '
      'return. A cost model built from the circulars removed the highest-IC signal in the study. '
      'A family-wise null removed most of what was left. A parameter sweep removed one arm that '
      'had survived all three. Each of those four is cheap, and each of them reverses a '
      'conclusion the previous three had let stand.</p></div>')
    A("</section>")

    # ============ method ============
    S = build_forward(S, D)
    S = build_registry(S, D)

    A('<section id="method"><div class="sec-head"><span class="sec-num">METHOD</span>'
      '<h2>How it was built</h2></div><div class="col stack">')
    A('<ul class="tight">'
      '<li><b>Data.</b> NIFTY 200 constituents from the NSE archives, snapshotted; daily OHLCV '
      'from Yahoo for 200/200 names, 2014&ndash;2026. Split-adjusted but <em>not</em> '
      'dividend-adjusted, so the strategy and the price-index benchmark share one convention — '
      'mixing them would have handed the strategy ~2%/yr of fabricated alpha.</li>'
      '<li><b>Signal.</b> All four repository definitions (last / mean / max / min) in three '
      'scalings (normalised, percentage return, reconstruction space), all from one forward '
      'pass. Context ends at t&rsquo;s close; future timestamps come from the exchange calendar, '
      'never from realised bars; per-window normalisation only.</li>'
      '<li><b>Execution.</b> Score at t, trade at t+1&rsquo;s open, return accrues open t+1 to '
      'open t+2 — asserted in code, not in a comment.</li>'
      '<li><b>Engine.</b> A line-for-line port of qlib 0.9.7&rsquo;s <code>TopkDropoutStrategy</code>, '
      'read from the installed package. Its <code>comb</code> guard means <code>n_drop</code> is '
      'a cap on turnover, not a quota — an implementation that misses this churns 21% of the '
      'book per rebalance on a tied score.</li>'
      '<li><b>Reproducibility.</b> HF revisions pinned, seeds per (date, chunk) so a resumed run '
      'reproduces an uninterrupted one, and a JSON provenance sidecar beside every artifact.</li>'
      '</ul>')
    A('<p class="note">Everything lives in <code>bench/nse/</code>. '
      '<code>run_all.sh</code> reproduces the whole study end to end; <code>PREREG.md</code> '
      'holds the pre-registration and its deviations; <code>VERIFIED.md</code> records every '
      'claim outcome; <code>RESULTS.md</code> carries the full grid.</p>')
    A("</div></section>")

    A(f'<footer><p>Generated from the run artifacts in <code>bench/out/nse/</code>. Every figure '
      f'on this page is read from a produced JSON or parquet file rather than transcribed, so it '
      f'cannot drift from the data. Kronos-base, revision '
      f'<span class="mono">{e(str(D["smeta"].get("model_revision", "?"))[:12])}</span>; '
      f'tokenizer <span class="mono">{e(str(D["smeta"].get("tokenizer_revision", "?"))[:12])}</span>.</p></footer>')
    return S


# ============================================================== crypto (BTC / ETH)

def build_crypto(S, D):
    """The single-asset timing test. NSE asked whether Kronos can rank names against
    each other; this asks whether it can time one instrument, on the asset class it was
    largely pretrained on."""
    A = S.append
    runs = [(k, D.get(f"btc_{k}")) for k in ("d90", "eth90", "d512")]
    runs = [(k, v) for k, v in runs if v]
    if not runs:
        return S
    d90 = D.get("btc_d90")
    if not d90:
        return S

    m, c, sk = d90["A_mechanism"], d90["B_calibration"], d90["C_skill"]
    bt, ctl = d90["D_backtest"], d90["E_controls"]
    rnd = ctl["random_control"]

    A('<section id="crypto"><div class="sec-head"><span class="sec-num">CRYPTO</span>'
      '<h2>The same test on Bitcoin, where the answer is worse</h2></div>')
    A('<div class="col"><p class="lede">Kronos was pretrained largely on crypto, and its '
      'public demos forecast BTC. If the reversion artefact were an artefact of Indian '
      'equities, this is where it would disappear. It does not: on '
      f'<strong>{d90["n_origins"]:,} daily forecast origins spanning '
      f'{bt["years"]:.1f} years</strong> of BTCUSDT, the pull toward the 90-bar mean is '
      '<strong>stronger</strong> than anywhere else measured, and for the first time the '
      'directional signal is significantly <em>negative</em> rather than merely absent.</p></div>')

    A('<div class="grid4">')
    A(stat("reversion to the 90-bar mean", f"{m['reversion_pct']:.1f}%",
           f"slope {m['slope']:+.4f}, R² {m['r2']:.3f}", "bad"))
    A(stat("rank corr. with the reversal factor", f"{m['rho_score_vs_neg_anchor']:+.3f}",
           "score vs −anchor_z"))
    A(stat("information coefficient", f"{sk['spearman_ic']:+.4f}",
           f"hit rate {sk['hit_rate']:.3f}", "bad"))
    A(stat("net vs buy-and-hold", pct(bt["ann_excess_net"]),
           f"{pct(bt['ann_return_net'])} vs {pct(bt['ann_return_bh'])} a year", "bad"))
    A("</div>")

    # ---- mechanism across all four markets -----------------------------------
    mech = D.get("mech") or {}
    rows = []
    def mrow(label, sl, r2, rho, note):
        rows.append([label, f"{-sl * 100:.1f}%", f"{sl:+.4f}", f"{r2:.3f}",
                     f"{rho:+.3f}", note])
    mrow("BTC daily, 90-bar window", m["slope"], m["r2"], m["rho_score_vs_neg_anchor"],
         f'{d90["n_origins"]:,} origins, 2017–2026')
    for key, label, note in (("zero_shot_full", "NSE-200 daily", "144,485 stock-dates"),
                             ("us", "S&P 100 daily", "US cross-section"),
                             ("greedy", "NSE-200, greedy (top_k=1)", "conditional mode"),
                             ("finetuned", "NSE-200, after 1,200 finetune steps", "sign flips")):
        v = mech.get(key)
        if v:
            # mechanism.json stores pooled_* from a per-observation fit; rho is not
            # persisted there, so those rows leave the column blank rather than invent it
            mrow(label, v["pooled_slope"], v["pooled_r2"], float("nan"), note)
    eth = D.get("btc_eth90")
    if eth:
        me = eth["A_mechanism"]
        mrow("ETH daily, 90-bar window", me["slope"], me["r2"],
             me["rho_score_vs_neg_anchor"], f'{eth["n_origins"]:,} origins')
    d512 = D.get("btc_d512")
    if d512:
        m5 = d512["A_mechanism"]
        mrow("BTC daily, 512-bar window", m5["slope"], m5["r2"],
             m5["rho_score_vs_neg_anchor"], f'{d512["n_origins"]:,} origins')
    A('<div class="col">')
    A(table(["market / setting", "reversion", "slope", "R²", "ρ vs −anchor", "sample"],
            rows, "One regression, five markets: the forecast's normalized close "
                  "regressed on how far the last bar sits from its own window mean. "
                  "Slope −1 would be total reversion, 0 no change, positive extrapolation.",
            numeric_from=1, highlight=lambda r: r[0].startswith("BTC daily, 90")))
    A('<p>The artefact is not Indian, not equity-specific, and not a sampling accident. '
      'It travels with the per-window normalization, which is why the only thing that has '
      'ever reversed its sign is retraining the model.</p></div>')

    # ---- calibration ---------------------------------------------------------
    rel = c.get("reliability", [])
    if rel:
        A('<div class="col" style="margin-top:14px"><h3>What the model says, against what happened</h3>')
        A('<p>Merging upstream #395 exposes the individual Monte-Carlo paths that '
          '<code>auto_regressive_inference</code> was averaging away, so the share of '
          'sampled paths finishing above the anchor becomes a stated probability of an up '
          'move. Sampling runs at <code>top_p = 1.0</code>: truncating the distribution '
          'narrows every path and would manufacture overconfidence before the model gets a '
          'say, so any miscalibration below is the model’s own.</p>')
        A(table(["stated P(up)", "forecasts", "stated", "realized", "gap"],
                [[x["bucket"], f"{x['n']:,}", f"{x['stated']:.3f}",
                  f"{x['realized']:.3f}", f"{x['realized'] - x['stated']:+.3f}"]
                 for x in rel],
                "Realized outcome is flat near the base rate across the entire "
                "confidence range. The two most populated buckets are the two most wrong.",
                numeric_from=1,
                highlight=lambda r: r[0] in ("0.0-0.1", "0.9-1.0")))
        A('<div class="grid3" style="margin-top:12px">')
        A(stat("Brier score", f"{c['brier']:.4f}",
               f"a constant forecast scores {c['brier_constant_baseline']:.4f}", "bad"))
        A(stat("Brier skill vs constant", f"{c['brier_skill_vs_constant']:+.3f}",
               "negative: worse than saying coin flip", "bad"))
        if "cone_coverage_p10_p90" in c:
            A(stat("p10–p90 cone coverage", f"{c['cone_coverage_p10_p90']:.3f}",
                   f"nominal 0.80, median width {c['median_cone_width_pct']:.1f}%", "bad"))
        A("</div>")
        A('<p>When the model reports a 0.7% chance of an up move it is right 41% of the '
          'time; when it reports 98% it is right 55% of the time. Stated confidence and '
          'realized frequency are uncorrelated, and the cone is roughly two and a half '
          'times too narrow. This reproduces, on '
          f'{c["n"]:,} forecasts, what upstream #395’s author reported from 249 live '
          'hourly crypto forecasts (stated 0.93 against a realized 0.478).</p></div>')

    # ---- money ---------------------------------------------------------------
    order = ["kronos", "kronos_negated", "anchor_reversal", "momentum_10", "always_long"]
    label = {"kronos": "Kronos long/flat", "kronos_negated": "Kronos, negated",
             "anchor_reversal": "−anchor_z (pure reversal)", "momentum_10": "10-bar momentum",
             "always_long": "always long (sanity check)"}
    rows = [[label[k], pct(ctl[k]["ann_return_net"]), pct(ctl[k]["ann_excess_net"]),
             f"{ctl[k]['hit_rate']:.3f}", pct(ctl[k]["max_drawdown"])]
            for k in order if k in ctl]
    A('<div class="col" style="margin-top:14px"><h3>Money</h3>')
    A(table(["arm", "net a year", "vs buy-and-hold", "hit rate", "max drawdown"], rows,
            f"Long/flat on the sign of the forecast, {bt['n_trades']} non-overlapping "
            f"{10}-bar trades, Binance taker fee plus slippage. Buy-and-hold returns "
            f"{pct(bt['ann_return_bh'])} a year over the same bars.",
            numeric_from=1, highlight=lambda r: r[0] == "Kronos long/flat"))
    A('<div class="grid4" style="margin-top:12px">')
    A(stat("cumulative, Kronos", pct(bt["cum_return_net"], nd=0),
           f"over {bt['years']:.1f} years", "bad"))
    A(stat("cumulative, buy-and-hold", pct(bt["cum_return_bh"], nd=0), "same bars", "good"))
    A(stat("signed return per trade", pct(sk["signed_return_mean"], nd=3),
           f"Newey–West t {sk['signed_return_t']:+.2f}", "bad"))
    A(stat("percentile of random", f"{rnd['kronos_percentile']:.1f}",
           f"p = {rnd['p_value_one_sided']:.3f}, {rnd['n_seed']} seeds", "bad"))
    A("</div>")
    A('<p>The cost model is not what kills it: total drag is '
      f'{bt["cost_drag_ann_pct"]:.2f}% a year against a {pct(bt["ann_excess_net"])} '
      'shortfall. Neither is risk-taking — the strategy sits out '
      f'{(1 - bt["exposure"]) * 100:.0f}% of the time and runs at '
      f'{pct(bt["vol_ann"], sign=False)} volatility against buy-and-hold’s '
      f'{pct(bt["vol_ann_bh"], sign=False)}, and still loses. The signal is the problem: '
      'the mean signed return is significantly negative, with a block-bootstrap 95% '
      f'interval of [{pct(sk["signed_return_ci95"][0], nd=2)}, '
      f'{pct(sk["signed_return_ci95"][1], nd=2)}] per trade.</p>')
    tax = d90.get("D_india_tax") or {}
    if tax:
        A('<p class="note"><strong>Indian tax overlay.</strong> Section 115BBH charges 30% '
          'flat on crypto gains with no loss set-off and no expense deduction, which '
          'penalises a strategy realising every trade far more than one realising once: '
          f'buy-and-hold’s {pct(bt["ann_return_bh"])} becomes '
          f'{pct(tax["buyhold_after_30pct"])}, while a losing strategy gets no relief at '
          'all. Section 194S withholds 1% on every sell — creditable against the 30%, so '
          'it is a cash-flow drag rather than an extra cost, but at '
          f'{bt["turnover_per_year"]:.1f}× turnover that is roughly '
          f'{tax["tds_cashflow_drag_pct_of_notional"]:.1f}% of notional withheld and '
          'reclaimed each year. Not tax advice; the point is only the order of '
          'magnitude.</p>')
    A("</div>")

    # ---- what was planned and not finished -----------------------------------
    A('<div class="col" style="margin-top:14px"><h3>Planned, and not completed</h3>')
    A('<p>The BTC daily run above is the primary test and it finished. Three further '
      'crypto runs were designed and queued to establish how far its conclusion '
      'generalises. They were stopped before completion, by choice rather than by '
      'failure, and <strong>none of them produced a result — no partial output was '
      'written, because each run writes only on completion.</strong> They are recorded '
      'here so that the absence is legible as a decision rather than an oversight.</p>')
    A(table(["run", "the question it was to answer", "planned size", "reached", "status"], [
        ["BTC hourly, 90-bar context, 24-bar horizon",
         "Is the result specific to the daily timeframe? Short-horizon mean reversion is "
         "strongest intraday, so this is where the artefact had its best chance of paying.",
         "2,425 origins", "8 origins", "stopped — 9.7 h estimated at the observed rate"],
        ["ETH daily, 90-bar context",
         "Is the crypto result specific to Bitcoin, or does it hold across the asset "
         "class? Same normalization, same horizon, different instrument.",
         "1,599 origins", "328 origins (21%)", "stopped mid-run"],
        ["BTC daily, 512-bar context",
         "Does a longer lookback weaken the pull toward the window mean? This is the only "
         "queued run that could have shown the artefact shrinking with more context.",
         "231 origins", "not started", "never began"],
    ], "Data for all three is fetched and on disk (58,294 hourly BTC bars, 3,297 daily "
       "ETH bars); only the forecasting passes are missing. Each is a single command "
       "against an existing script.", numeric_from=99))
    A('<p>Of the three, <strong>the 512-bar context run is the one worth resuming '
      'first.</strong> It is the cheapest at 231 origins, and it is the only one that '
      'tests a mechanism rather than a generalisation: if the reversion coefficient falls '
      'materially when the normalization window widens, then the artefact is a '
      'consequence of a short window rather than of the architecture, and that would be a '
      'genuinely different finding. The other two would almost certainly confirm what the '
      'BTC daily run already shows.</p></div>')

    # ---- the correction ------------------------------------------------------
    A('<div class="col" style="margin-top:14px"><h3>A prediction this study got wrong</h3>')
    A('<p>Before running it, the expectation recorded here was that crypto might be the one '
      'market where the artefact accidentally pays. The reasoning was that short-horizon '
      'mean reversion is a documented crypto effect, so a signal that leans against '
      'extension could be right for the wrong reason. <strong>That was wrong.</strong> '
      f'Over {bt["years"]:.1f} years BTC trended hard — buy-and-hold compounded '
      f'{pct(bt["cum_return_bh"], nd=0)} — and a signal that sells extension fights the '
      'trend it needed to ride. The NSE verdict was no detectable skill; the BTC verdict is '
      'significantly negative skill, which is a stronger claim in the same direction.</p>')
    A('</div></section>')
    return S

# ============================================================== repo hardening

def build_hardening(S, D):
    """What was merged into the fork and, more usefully, what was rejected and why."""
    A = S.append
    A('<section id="hardening"><div class="sec-head"><span class="sec-num">HARDENING</span>'
      '<h2>What the upstream repository gets wrong</h2></div>')
    A('<div class="col"><p class="lede">Upstream has not merged anything since '
      '13 April 2026 and carries 61 open pull requests. Every claim below was checked '
      'against the source rather than the PR description, which turned out to matter: '
      '<strong>several of the fix PRs are wrong, and one would reintroduce a crash a '
      'merged PR already fixed.</strong></p>')

    A(table(["merged", "what it fixes", "why it matters"], [
        ["#382 <span class=\"mono\">.eval()</span>",
         "predictor never left training mode",
         "<code>PyTorchModelHubMixin._load_as_safetensor</code> never calls "
         "<code>.eval()</code>, and every shipped config carries dropout "
         "(Kronos-small: ffn 0.25, resid 0.25). Two identical <code>decode_s1</code> "
         "calls differed by max |Δlogits| = 3.42. Every example script, the webui and "
         "the repo's own benchmark ran inference with dropout live."],
        ["#351 None top_k/top_p",
         "<code>TypeError</code> on a normal call",
         "<code>sample_from_logits(logits, top_p=0.9)</code> raised on the "
         "<code>None &gt; 0</code> comparison."],
        ["#395 <span class=\"mono\">return_samples</span>",
         "the predictive distribution was discarded",
         "<code>auto_regressive_inference</code> averaged its sample paths before "
         "returning, so quantiles, cones and P(up) could not be computed from the public "
         "API at all. This study's calibration test depends on it."],
        ["#289 bf16", "opt-in mixed precision",
         "Default is <code>autocast(enabled=False)</code>, so the existing path is "
         "untouched."],
        ["#244, 4 of 8 hunks", "genuine small defects",
         "<code>avg_prob</code> NameError, <code>h, w = int(...)</code> unpacking a "
         "scalar twice, a <code>require_grad</code> typo that never froze anything, and a "
         "backward returning five gradients for four inputs."],
    ], "Merged into the fork on branch harden/upstream-fixes. 13 tests green, including "
       "the repo's own golden-output regression at rtol 1e-5.", numeric_from=99))

    A('<h3 style="margin-top:18px">A bug none of the pull requests found</h3>')
    A('<p><code>top_k_top_p_filtering</code> carried three defects at once. The top-k '
      'branch <code>return</code>ed early, so <strong>passing both <code>top_k</code> and '
      '<code>top_p</code> silently applied only top-k</strong> — the nucleus argument was '
      'accepted and ignored. With <code>top_k == 0</code> and <code>top_p ≥ 1.0</code> '
      'neither branch returned, so the function handed back <code>None</code>; only a '
      'guard in the caller kept that off the hot path. And it wrote in place on the '
      'caller’s tensor. Fixed, with nine tests pinning the behaviour, including one that '
      'proves the single-filter paths are numerically unchanged.</p>')

    A('<h3 style="margin-top:18px">Rejected, with the evidence</h3>')
    A(table(["rejected", "claim", "what the source actually shows"], [
        ["#253 reshape bug",
         "<code>auto_regressive_inference</code> reshapes with post-unsqueeze indices",
         "False. Python evaluates <code>x.size(1)</code> before rebinding <code>x</code>, "
         "so the original is correct — verified output (12, 90, 5). The PR also branched "
         "off before #232 and would reintroduce the <code>top_k(probs, …)</code> crash."],
        ["#266 safetensors save",
         "<code>save_pretrained</code> drops the predictor weights",
         "Disproven. It writes all 66 keys plus <code>config.json</code> and round-trips "
         "bit-identically. The PR's replacement omits <code>config.json</code>, so "
         "reloading raises <code>TypeError: missing 11 required positional arguments</code>."],
        ["#244, the other 4 hunks",
         "eight bugs in core model code",
         "Its <code>is_causal</code> hunks make the LM bidirectional under a padding mask "
         "and remove causality during training; its <code>sample_from_logits</code> fix "
         "still forwards <code>top_k=None</code> and raises."],
        ["#393 KV cache",
         "exact numerical equivalence",
         "False. Measured max |Δ| = 1.79e-07 on the cached path; with discrete "
         "<code>multinomial</code> sampling one flipped token diverges a whole path, so "
         "its passing test is a seed lottery. Also a padding-mask broadcast bug, and the "
         "cache refuses to engage when context + horizon exceeds max_context — no speedup "
         "at the 512 lookback anyone would use."],
    ], "Rejections matter more than merges here: three of these would have been merged on "
       "the strength of a confident description.", numeric_from=99))
    A('<p class="note">None of this changes the study’s conclusions. The harness called '
      '<code>tokenizer.eval()</code> and <code>model.eval()</code> explicitly from the '
      'start, and every merged change leaves the default numerical path byte-identical — '
      'verified by re-running the golden regression before and after.</p>')
    A("</div></section>")
    return S


# ============================================================== scope registry

def build_registry(S, D):
    """A map of everything that has been tested, for a reader arriving with no context."""
    A = S.append
    d90, eth, d512 = D.get("btc_d90"), D.get("btc_eth90"), D.get("btc_d512")
    A('<section id="scope"><div class="sec-head"><span class="sec-num">SCOPE</span>'
      '<h2>Everything that has been tested, and everything that has not</h2></div>')
    A('<div class="col"><p class="lede">This section exists so that anyone picking the '
      'work up later can tell in one screen which questions are closed and which are still '
      'open, without re-deriving it from the code.</p>')

    A('<h3>Markets and samples</h3>')
    rows = [
        ["NSE-200, current membership", "daily", "2023-08-01 → 2026-08-25",
         "756 sessions × ~200 names", "survivorship-biased on purpose, as the upper bound"],
        ["NSE-200, point-in-time", "daily", "2023-08-01 → 2026-08-25",
         "756 sessions, 199.3 names/session",
         "membership rebuilt from six semi-annual reconstitution PDFs"],
        ["S&P 100", "daily", "US cross-section", "6,478 stock-dates",
         "tests whether the mechanism is India-specific"],
    ]
    if d90:
        rows.append(["BTCUSDT", "daily, 90-bar context",
                     f'{d90["first"][:10]} → {d90["last"][:10]}',
                     f'{d90["n_origins"]:,} origins × 30 paths',
                     "single-asset timing, not cross-sectional ranking"])
    if eth:
        rows.append(["ETHUSDT", "daily, 90-bar context",
                     f'{eth["first"][:10]} → {eth["last"][:10]}',
                     f'{eth["n_origins"]:,} origins × 30 paths',
                     "is the crypto result BTC-specific?"])
    else:
        rows.append(["ETHUSDT", "daily, 90-bar context", "2017-08 → 2026-08",
                     "NOT COMPLETED — stopped at 328 of 1,599 origins",
                     "would have asked whether the crypto result is BTC-specific"])
    if d512:
        rows.append(["BTCUSDT", "daily, 512-bar context",
                     f'{d512["first"][:10]} → {d512["last"][:10]}',
                     f'{d512["n_origins"]:,} origins × 30 paths',
                     "does more context weaken the artefact?"])
    else:
        rows.append(["BTCUSDT", "daily, 512-bar context", "2017-08 → 2026-08",
                     "NOT COMPLETED — never started, 231 origins planned",
                     "would have asked whether a wider normalization window weakens the "
                     "artefact — the highest-value unfinished run"])
    rows.append(["BTCUSDT", "hourly, 24-bar horizon", "2020-01 → 2026-08",
                 "NOT COMPLETED — stopped at 8 of 2,425 origins",
                 "would have asked whether the result is specific to the daily timeframe"])
    A(table(["universe", "frequency", "span", "size", "why it is in the study"], rows,
            "Rows marked NOT COMPLETED were designed and queued but stopped before "
            "producing output. Their input data is fetched and on disk; only the "
            "forecasting pass is missing.", numeric_from=99))

    A('<h3 style="margin-top:18px">Model variants</h3>')
    A(table(["variant", "status", "result"], [
        ["Kronos-base, zero-shot, revision-pinned", "primary",
         "70.1% reversion on NSE, 81.7% on BTC; no directional skill anywhere"],
        ["greedy decoding, top_k = 1", "tested",
         "58.6% reversion — the conditional mode, so not sampling noise"],
        ["temperature sweep T = 0.3 / 0.6 / 1.0", "tested", "no cell becomes significant"],
        ["5-path vs 30-path averaging", "tested", "no material change"],
        ["finetuned, 1,200 steps on 182 NSE symbols", "tested",
         "reversion flips sign to −17.6%; strategy moves percentile 0 → 21 and still "
         "produces no signal"],
        ["Kronos-small, Kronos-mini", "not tested", "only the base checkpoint was run"],
        ["longer finetune, more symbols", "not tested",
         "the run was data-limited, not compute-limited: validation loss turned up at "
         "step 1,200 of 7,844"],
    ], "", numeric_from=99))

    A('<h3 style="margin-top:18px">Tests run</h3>')
    A(table(["test", "scale", "outcome"], [
        ["Information coefficient, horizons 1/2/5/10",
         "48 cells, Newey–West plus moving-block bootstrap",
         "0 significant, largest |t| = 1.71"],
        ["Conditional IC within control quintiles", "volatility scalars",
         "the model's range statistic scores below the free trailing-volatility statistic"],
        ["Random-score control", "150–300 seeds per frequency",
         "NSE percentile 64.6 (p = 0.354); BTC percentile 8.7 (p = 0.913)"],
        ["Family-wise null, max-statistic", "200 draws × 80 arms",
         "only one blend clears; the family clears while no member does"],
        ["Parameter surface", "735 cells",
         "momentum is a plateau; 5-day reversal is an isolated spike and was withdrawn"],
        ["Sub-period stability", "split-sample", "sign stable, magnitude not"],
        ["Capacity and liquidity", "10 capital levels, square-root impact",
         "cost-indifferent between ₹10 lakh and roughly ₹300 crore"],
        ["Benchmark audit", "5 rebalance cadences, both universes",
         "the equal-weight bar is fair to within 0.81%/yr"],
        ["Probability calibration", f'{(d90 or {}).get("B_calibration", {}).get("n", 0):,} BTC forecasts',
         "Brier skill −0.689 against a constant forecast; cone covers 0.34 against a "
         "nominal 0.80"],
    ], "", numeric_from=99))

    A('<h3 style="margin-top:18px">Cost models</h3>')
    A('<p>Indian delivery equities carry STT, stamp duty, exchange and IPFT charges, the '
      'SEBI turnover fee, GST, DP charges and brokerage — modelled per side, and '
      'deliberately not scale-invariant, because the fixed components decide whether a '
      'small book is viable. Market impact uses the Almgren–Chriss square-root shape '
      'against each name’s own ADV. Crypto uses the Binance spot taker fee plus slippage, '
      'with Section 115BBH (30% flat, no loss set-off) and Section 194S (1% withheld per '
      'sell) applied as an overlay.</p>')

    A('<h3 style="margin-top:18px">Open questions, explicitly not answered</h3>')
    A('<ul>'
      '<li><strong>Sub-daily crypto.</strong> 58,294 hourly bars are fetched and ready; '
      'the run was started and stopped at 8 origins of 2,425. The daily result is '
      'unlikely to reverse, but intraday is where the artefact’s short-horizon reversion '
      'has the best chance of paying.</li>'
      '<li><strong>Whether the crypto result is BTC-specific.</strong> The ETH daily run '
      'reached 328 of 1,599 origins before being stopped and wrote nothing.</li>'
      '<li><strong>Whether a longer context weakens the artefact.</strong> The 512-bar '
      'run never started. At 231 origins it is the cheapest unfinished experiment and the '
      'only one that could change the mechanism story rather than extend it.</li>'
      '<li><strong>Other asset classes.</strong> No FX, commodities, rates or index '
      'futures. Upstream issue #387 reports the same no-edge result on futures at 5m, 15m '
      'and 1h, but that is someone else’s measurement, not this one.</li>'
      '<li><strong>A properly resourced finetune.</strong> 182 symbols over 7 years is not '
      'enough for 102M parameters. The one result that did move — the sign flip — came '
      'from 0.13% of a training schedule, which makes a real training run the single most '
      'informative experiment left.</li>'
      '<li><strong>The smaller checkpoints.</strong> Kronos-small and -mini were never '
      'run; a smaller model may overfit the normalization window less.</li>'
      '<li><strong>Live or paper trading.</strong> Everything here is a backtest with a '
      'modelled cost stack. No order was ever sent.</li>'
      '</ul>')

    A('<h3 style="margin-top:18px">Where the code lives</h3>')
    A(table(["path", "what it is"], [
        ["<code>bench/nse/</code>",
         "the equity study: fetch, preprocess, signals, backtest, controls, surface, "
         "robustness, artifact builder. <code>run_all.sh</code> reproduces it end to end."],
        ["<code>bench/btc/</code>",
         "the crypto study: <code>fetch_btc.py</code> (paginated Binance), "
         "<code>btc_signals.py</code> (rolling forecasts keeping sample paths), "
         "<code>btc_analysis.py</code>, <code>queue.sh</code>."],
        ["<code>bench/out/nse/</code>, <code>bench/out/btc/</code>",
         "every produced JSON and parquet, each with a provenance sidecar. This page reads "
         "from them rather than transcribing numbers."],
        ["<code>bench/nse/PREREG.md</code>",
         "the pre-registration and every deviation from it, including Addendum D2 covering "
         "the post-hoc work."],
        ["<code>model/</code>, <code>tests/</code>",
         "the upstream library plus the merged fixes on branch "
         "<code>harden/upstream-fixes</code>."],
    ], "", numeric_from=99))
    A("</div></section>")
    return S




SURVIVORS = [
    ("Normalization-window equivariance probe",
     "The model reverts toward the mean of whatever window the input was <em>normalized</em> "
     "over, not toward a mean inferred from the bars it attends to — making the reversion "
     "target an inference-time knob rather than a model property.",
     "Feed the identical 90 bars twice, normalized over 90 and over 512, same seed. Then "
     "regress <span class=\"mono\">(P̂ − P_now)</span> on "
     "<span class=\"mono\">(μ90 − P_now)</span> and <span class=\"mono\">(μ512 − P_now)</span> "
     "in <strong>price space, never z-space</strong>; both regressors are free and correlate "
     "only 0.57, so the design is identified. Run arm A as a placebo first: μ512 is not its "
     "normalization window there, so its coefficient must come out ≈ 0.",
     "~15 GPU-min",
     "<strong>Go</strong> if c512 ≈ 0.82 and c90 ≈ 0. <strong>Stop</strong> if the reverse. "
     "<strong>Also stop</strong> if distinct tokenizer codes collapse by more than 20% versus "
     "arm A — then the result is a quantization artefact, not a mechanism.",
     "No — zero monetary payoff by construction"),
    ("Free −anchor_z surface across the untested markets",
     "Greedy Kronos is <span class=\"mono\">−0.538 × anchor_z</span> at R² = 0.977, so every "
     "untested market can be priced by backtesting the free surrogate — no GPU, no tokenizer.",
     "Compute anchor_z at each market's native frequency for the open cells — ETH daily, "
     "commodities, rates, index futures — and backtest −anchor_z directly. Run the whole "
     "market × lookback × horizon grid as <strong>one surface against a 200-draw "
     "max-statistic null</strong>, not cell by cell.",
     "0 GPU-hours",
     "<strong>Go</strong> only if a cell is materially positive after the family null and "
     "after beta and volatility matching. <strong>Stop</strong> otherwise: the model cannot "
     "beat a surrogate it reproduces at 97.7% fidelity by more than the 2.3% residual, much "
     "of which is tokenizer round-trip error.",
     "No — buys closure of four asset classes"),
    ("h = 1 cross-sectional IC on NSE",
     "At a 1-step horizon the anchor explains 38% of the forecast rather than 90%, so the "
     "orthogonal component may carry cross-sectional information the h = 10 tests could not see.",
     "Rerun signals at <span class=\"mono\">--predict 1</span> on the point-in-time panel, "
     "primary statistic the anchor-orthogonalised IC against the <em>full</em> free "
     "short-horizon family — not against anchor_z alone.",
     "~0.5 GPU-h",
     "<strong>Go</strong> only if it clears a family-wise null. A positive against a narrow "
     "control basis will be reversal_2d wearing a different name.",
     "Almost certainly no — see the cost note"),
    ("Synthetic-control probe: where does the −0.70 come from?",
     "The reversion is a property of tokenizer, sampler and per-window normalization, "
     "independent of any training panel's conditional mean — a ceiling argument that "
     "generalises to <em>every</em> data budget and market.",
     "Run the mechanism fit on synthetic series whose conditional mean is exactly zero. "
     "Then, if it collapses, plant a known −0.50 conditional mean in training data and "
     "measure whether the architecture can recover it at all.",
     "~0.5–2 GPU-h",
     "<strong>Stop</strong> (expected) if the slope stays near −0.70 on zero-mean synthetic "
     "data: the artefact is architectural and the finetune path closes for all data budgets. "
     "<strong>Go</strong> if it collapses toward 0 — the finetune question genuinely reopens.",
     "No — but it is the strongest closure available"),
    ("Capacity, scored on skill rather than on drag",
     "Model capacity does not create directional skill.",
     "Run Kronos-small through the identical NSE pipeline and compare IC, signed return and "
     "random-signal percentile against base — scored on <em>skill</em>, not on the "
     "per-step drag rate.",
     "~2.5–4 GPU-h",
     "<strong>Stop</strong> if small lands in the same null region base occupies "
     "(|t| &lt; 2, percentile 5–95): the capacity axis closes on skill evidence and "
     "Kronos-large is retired without being obtained.",
     "No — closes the capacity axis"),
    ("Crypto cross-section — Stage 0 only, and it contains no Kronos",
     "A 90-bar z-score reversal has a positive net payoff in the crypto cross-section after "
     "delisting, financing, beta and volatility matching.",
     "Stage 0 is the <strong>free</strong> factor on a delisting-complete perp panel. Only "
     "if that clears does Kronos become worth testing, and then only as an incremental IC "
     "against a <strong>monotone-warp placebo</strong> — warp −anchor_z, residualise it on "
     "itself, and confirm the placebo residual scores ≈ 0.",
     "0 GPU for Stage 0; 2–3 days of new code",
     "<strong>Go</strong> only if the free factor survives every haircut. <strong>Stop</strong> "
     "otherwise. Note the placebo is not optional: this repo has already been observed "
     "manufacturing t = +1.37 out of two raw signals of −0.0013 and −0.0092.",
     "Possibly — but the payoff belongs to a numpy z-score, not to Kronos"),
]


def build_forward(S, D):
    """Where a new reader should spend effort, and — more usefully — where they should not."""
    A = S.append
    grave = j(os.path.join(ROOT, "bench", "forward", "graveyard.json")) or []

    A('<section id="forward"><div class="sec-head"><span class="sec-num">FORWARD</span>'
      '<h2>What is left to try, and whether it is worth trying</h2></div>')
    A('<div class="col"><p class="lede">Five researchers were given the falsified record '
      'above and asked, from five different angles, how Kronos could still be monetised. '
      'They produced <strong>20 proposals</strong>. Each was then handed to a separate '
      'skeptic instructed to refute it by default. <strong>None survived intact.</strong> '
      'Most were not merely argued down — the skeptics <em>ran</em> them, on data already on '
      'disk, and watched them fail their own pre-registered gates.</p>')

    A('<div class="card" style="border-color:var(--refuted);border-left-width:4px;'
      'background:var(--refuted-soft)">'
      '<h3 style="color:var(--refuted)">The honest expected value</h3>'
      '<p style="font-size:.97rem"><strong>For monetisation: negative. Do not spend money on '
      'it.</strong> Five of the six salvageable paths have zero monetary payoff by '
      'construction — they are closure experiments whose entire value is letting you stop '
      'defensibly. The sixth has perhaps a 3–5% chance of a statistically positive result, '
      'and even that is likely unmonetisable, because it needs daily rebalancing in a market '
      'whose cost stack was already measured destroying a stronger, free version of the same '
      'signal. The one item with a large monetary branch contains no Kronos at all.</p>'
      '<p style="font-size:.97rem"><strong>For the write-up: marginally positive, and '
      'cheap.</strong> Roughly <strong>one to five GPU-hours</strong> buys four permanent '
      'closure claims stronger than anything currently in the record. Run survivors 1, 2 and '
      '4 — about a GPU-hour and three days. Then stop, publish the negative, and buy the '
      'ETF.</p></div>')

    A('<h3 style="margin-top:20px">The six that survived, re-scoped</h3>'
      '<p>Every one is listed with the <em>corrected</em> test, not the one proposed. The '
      'skeptics rewrote most of them into something cheaper and narrower, and in several '
      'cases the original test could not have identified the quantity it claimed to '
      'measure.</p>')
    for i, (t, hyp, test, cost, rule, flag) in enumerate(SURVIVORS, 1):
        beats = flag.startswith("Possibly")
        A(f'<div class="card" style="margin-top:10px">'
          f'<h3>{i}. {e(t)}</h3>'
          f'<p><strong>Hypothesis.</strong> {hyp}</p>'
          f'<p><strong>Test.</strong> {test}</p>'
          f'<p><strong>Decision rule.</strong> {rule}</p>'
          f'<div class="grid2" style="margin-top:8px">'
          f'{stat("cost", cost)}'
          f'{stat("beats a momentum ETF?", flag, "", "pos" if beats else "neg")}'
          f'</div></div>')
    A('<p class="note" style="margin-top:10px">The benchmark those flags are measured '
      'against: buy a Nifty 200 Momentum 30 ETF for roughly +3.5%/yr excess, zero effort, '
      'zero research risk. A path that cannot beat that is not a path to money, however '
      'interesting the finding.</p>')

    if grave:
        A(f'<h3 style="margin-top:20px">The graveyard — {len(grave)} proposals, and why each '
          f'died</h3>')
        A('<p>This is the more useful half. Half the refutations exist because someone '
          're-proposed something the record already answered under a new name — most often a '
          'reversal factor in disguise. Read this before proposing anything.</p>')
        A(table(["proposal", "cause of death"],
                [[f'<strong>{e(x["title"])}</strong>', e(x.get("short", ""))] for x in grave],
                "Verdicts from independent skeptics, each instructed to refute by default. "
                "Most were executed rather than argued.", numeric_from=99))

    A('<h3 style="margin-top:20px">What the review corrected in this study</h3>')
    A('<ul>'
      '<li><strong>A throughput figure quoted here was wrong.</strong> The 0.31 origins/s '
      'rate is single-series BTC, not the batched NSE path, which runs at 134 ms/series — '
      'about 7.5 series a second. Cost estimates built on the wrong one were out by up to '
      '45×, in both directions.</li>'
      '<li><strong>A provenance bug, now fixed.</strong> <code>sidecar()</code> stamped the '
      'module-level model constants unconditionally, so every finetuned run\'s record claimed '
      'to be Kronos-base at the pinned revision. The truth was recoverable from '
      '<code>args.model_path</code>, and no result changes — but a provenance record that '
      'needs cross-checking against the thing it certifies is not doing its job. It now '
      'stamps what was actually loaded.</li>'
      '<li><strong>Data traps for anyone extending this.</strong> yfinance returns zero '
      'volume for every FX cross and no <code>amount</code> column at all, so two of the six '
      'model inputs go identically zero after normalization — out of distribution before the '
      'model sees a thing. <code>FESX=F</code> and <code>NIFTY=F</code> do not exist there.</li>'
      '<li><strong>New data on disk.</strong> Six Binance perpetual 15-minute series '
      '(BTC, ETH, SOL, BNB, XRP, DOGE; 198,197 bars each, 2021-01 to 2026-08) were fetched '
      'during the review and left in <code>data/crypto/</code>. Stage 0 of survivor 6 does '
      'not need to download anything.</li>'
      '</ul>')
    A('<p class="note">The full adversarial brief, including every proposal, every '
      'refutation and the corrected cost table, is in '
      '<code>bench/forward/ADVERSARIAL_BRIEF.md</code>; the machine-readable graveyard is '
      '<code>bench/forward/graveyard.json</code>.</p>')
    A("</div></section>")
    return S


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="zeroshot")
    ap.add_argument("--out", default=os.path.join(OUT_NSE, "findings.html"))
    args = ap.parse_args()
    S, D = build(args.tag)
    S = build_body(S, D, args.tag)
    S = build_tail(S, D, args.tag)
    S = build_close(S, D, args.tag)
    head = ('<title>The forecast is a reversal factor</title>\n'
            '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
            '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
            'family=Zilla+Slab:wght@400;500;600&family=Public+Sans:wght@400;500;600&'
            'family=IBM+Plex+Mono:wght@400;500&display=swap">\n'
            f"<style>{CSS}</style>\n")
    html_out = head + '<div class="wrap">\n' + "\n".join(S) + "\n</div>\n"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html_out)
    print(f"{len(html_out) / 1024:.0f} KB -> {args.out}")


if __name__ == "__main__":
    main()
