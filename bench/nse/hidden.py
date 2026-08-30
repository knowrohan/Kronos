"""Stage 2 extraction: the trunk's final hidden state, instead of a decoded forecast.

The MVM. Three lines of model call, zero edits under model/:

    t0, t1 = tokenizer.encode(x, half=True)
    _, h   = model.decode_s1(t0, t1, stamp)
    hT     = h[:, -1, :]

No auto_regressive_inference, no decode_s2, no sample_from_logits, no tokenizer.decode.
That removes F2 (objective), F4 (decoding regime) and F5 (sampled distribution)
mechanically rather than by argument -- there is nothing left to sample from.

Two arms, selected with --arm:
    pretrained  the public Kronos-base checkpoint
    reservoir   the identical architecture with RANDOM weights and the same frozen
                public tokenizer. This is the control that cannot have memorised
                anything, and the gate: if pretrained does not beat it, pretraining
                bought nothing on this task.

Windows come from the cache built by cache_windows.py, which copies build_batch's
normalisation verbatim -- so the tensor fed here is bit-identical to the one behind
every number in bench/out/nse/.

Representation ladder (NIFTY50_PLAN.md sec.5, arms A1-A6)
---------------------------------------------------------
The default invocation above reads ONE vector per cell: post-`norm`, final layer, last
position (A0). Its effective rank is 8.6 on 150,527 cells because `norm` + `head` shape
that vector for the next-token logits, not for a cross-sectional ranking. The flags below
read other states out of the SAME forward pass, at no extra model cost:

    --layers 4,8,12   residual stream after transformer block i, BEFORE mdl.norm (A1).
                      Captured with forward hooks on mdl.transformer[i-1]; 1-based i so
                      the CLI matches the plan's "layers 4, 8, 12" wording.
    --pool last,mean10 last position (the record's pooling) and/or the mean of the last
                      10 positions (A2), which averages out single-token noise.
    --model small     capacity ladder (A3/A4): Kronos-small (512-d, 8 layers, same
                      tokenizer) and Kronos-mini (256-d, 4 layers, Kronos-Tokenizer-2k).
    --tokenizer-z     the tokenizer's continuous pre-quantisation embedding (A5): the
                      output of quant_embed(encoder(embed(x))) BEFORE the BSQuantizer.
                      Answers whether the signal is in the learned compression or in the
                      trunk. Note this vector is codebook_dim (= s1_bits + s2_bits = 20)
                      wide, not 256: quant_embed projects the 256-d encoder state down to
                      the codebook before quantisation, and the plan's "256-d" refers to
                      the encoder width one linear layer earlier. Stored float32, not fp16
                      like the trunk states: its scale is unconstrained (the quantiser
                      normalises it away) and it overflows fp16 on this panel.
    --cells n50       restrict to point-in-time NIFTY 50 (date, symbol) rows, for the
                      cheap reservoir seeds of A6.

Backwards compatibility is load-bearing: with none of the new flags and no --out-dir this
script writes bench/out/nse/ml/hT_{arm}.npy and its meta with exactly the old content, so
the 2026-08-28 numbers stay reproducible. Any new flag (or an explicit --out-dir) switches
to bench/out/nse/n50/ and the hT_{model}_{arm}_{layer}_{pool}.npy naming, which never
collides with the old files. A reservoir arm carries its seed in {arm} (`reservoir_s8`) and
an --cells n50 run carries `_n50` after the pool, so no two runs of the ladder land on the
same filename.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from bench.nse.common import DATA_NSE, MODEL_REPO, MODEL_REVISIONS  # noqa: E402
from bench.nse.common import TOKENIZER_REPO, TOKENIZER_REVISION  # noqa: E402
from bench.nse.common import git_sha, pkg_versions, set_seed, utc_now  # noqa: E402
from bench.nse.mlcore import OUT_NSE  # noqa: E402

ML = os.path.join(OUT_NSE, "ml")
N50 = os.path.join(OUT_NSE, "n50")
MEMBERSHIP_N50 = os.path.join(DATA_NSE, "membership_pit_n50.parquet")

# The capacity ladder. Kronos-mini was trained against a DIFFERENT tokenizer
# (Kronos-Tokenizer-2k, group_size 5 rather than 4); feeding it base-tokenizer ids would
# be a silent mismatch, so the tokenizer is a property of the model, not a constant.
MODEL_REPOS = {"base": MODEL_REPO,  # the record's checkpoint, from the shared constant
               "small": "NeoQuasar/Kronos-small",
               "mini": "NeoQuasar/Kronos-mini"}
# repo, pinned revision. None means "resolve from the HF API at run time": the 2k
# tokenizer was not in the 2026-08-27 pin set, and a hard-coded SHA nobody has verified is
# worse than a resolved one recorded in the sidecar.
TOKENIZER_REPOS = {"base": (TOKENIZER_REPO, TOKENIZER_REVISION),
                   "small": (TOKENIZER_REPO, TOKENIZER_REVISION),
                   "mini": ("NeoQuasar/Kronos-Tokenizer-2k", None)}
# Last-resort pin if the HF API is unreachable, so an offline queue run still produces a
# revision-stamped artifact. Resolved from the HF API on 2026-08-29.
TOKENIZER_REVISION_FALLBACK = {
    "NeoQuasar/Kronos-Tokenizer-2k": "26966d0035065a0cae0ebad7af8ece35bc1fb51c",
}
POOLS = ("last", "mean10")


def stamps_for(dates) -> np.ndarray:
    idx = pd.DatetimeIndex(dates)
    return np.stack([idx.minute, idx.hour, idx.weekday, idx.day, idx.month],
                    axis=1).astype(np.float32)


def resolve_tokenizer_revision(model: str) -> tuple[str, str | None, str]:
    """Tokenizer repo + revision for a model, resolving unpinned ones from the HF API.

    Returns (repo, revision, source). The source string goes into the sidecar so a reader
    can tell a pinned revision from one that was looked up on the day of the run.
    """
    repo, rev = TOKENIZER_REPOS[model]
    if rev is not None:
        return repo, rev, "pinned_in_common_py"
    try:
        from huggingface_hub import HfApi
        rev = HfApi().model_info(repo).sha
        return repo, rev, "hf_api"
    except Exception as exc:  # noqa: BLE001 -- offline is a legitimate state, not a crash
        rev = TOKENIZER_REVISION_FALLBACK.get(repo)
        if rev is None:
            raise SystemExit(f"cannot resolve a revision for {repo}: {exc}") from exc
        print(f"  HF API unreachable ({type(exc).__name__}); using the recorded pin "
              f"{rev[:8]} for {repo}", flush=True)
        return repo, rev, "fallback_pin_2026-08-29"


def build_model(arm: str, device: str, seed: int, model: str = "base"):
    """Pretrained checkpoint, or the same architecture with random weights."""
    from model import Kronos, KronosTokenizer

    repo = MODEL_REPOS[model]
    tok_repo, tok_rev, tok_src = resolve_tokenizer_revision(model)
    tok = KronosTokenizer.from_pretrained(tok_repo, revision=tok_rev)
    tok_prov = {"tokenizer_repo": tok_repo, "tokenizer_revision": tok_rev,
                "tokenizer_revision_source": tok_src}
    if arm == "pretrained":
        mdl = Kronos.from_pretrained(repo, revision=MODEL_REVISIONS.get(repo))
        provenance = {"model_repo": repo,
                      "model_revision": MODEL_REVISIONS.get(repo)}
    elif arm == "reservoir":
        # same config, fresh weights. common.py:79 would stamp this as Kronos-base,
        # which is the defect Appendix B flags -- so provenance is written by hand here.
        ref = Kronos.from_pretrained(repo, revision=MODEL_REVISIONS.get(repo))
        cfg = {k: getattr(ref, k) for k in
               ["s1_bits", "s2_bits", "n_layers", "d_model", "n_heads", "ff_dim",
                "ffn_dropout_p", "attn_dropout_p", "resid_dropout_p", "token_dropout_p",
                "learn_te"]}
        set_seed(seed)
        mdl = Kronos(**cfg)
        del ref
        provenance = {"model_repo": None, "model_revision": None,
                      "untrained_random_init": True, "init_seed": seed, "config": cfg}
    else:
        raise ValueError(arm)

    provenance.update(tok_prov)
    tok.eval().to(device)
    mdl.eval().to(device)
    return tok, mdl, provenance


def parse_layers(spec: str, n_layers: int) -> list[int]:
    """1-based transformer block indices, validated against the loaded trunk's depth."""
    if not spec.strip():
        return []
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        i = int(part)
        if not 1 <= i <= n_layers:
            raise SystemExit(f"--layers {i} out of range: this trunk has {n_layers} "
                             f"blocks (1-based indices 1..{n_layers})")
        if i not in out:
            out.append(i)
    return out


def parse_pools(spec: str) -> list[str]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if part not in POOLS:
            raise SystemExit(f"--pool {part!r} unknown; choose from {','.join(POOLS)}")
        if part not in out:
            out.append(part)
    if not out:
        raise SystemExit("--pool selected nothing")
    return out


def select_rows(idx: pd.DataFrame, cells: str) -> np.ndarray:
    """Row indices into windows_index.parquet for the requested cell set.

    'n50' is an inner join on point-in-time NIFTY 50 membership. It fails loudly when the
    membership file is absent: silently falling back to the 200 would put a NIFTY 200
    number under a NIFTY 50 heading, which is the one error this whole build is designed
    not to make.
    """
    if cells == "all":
        return np.arange(len(idx), dtype=np.int64)
    if not os.path.exists(MEMBERSHIP_N50):
        raise SystemExit(
            f"--cells n50 needs {MEMBERSHIP_N50}, which does not exist yet. "
            "Build it first (NIFTY50_PLAN.md sec.2.1: reconstruct_membership.py "
            "--index nifty50), then re-run.")
    mem = pd.read_parquet(MEMBERSHIP_N50, columns=["date", "symbol"]).drop_duplicates()
    mem["date"] = pd.to_datetime(mem["date"])
    left = pd.DataFrame({"date": pd.to_datetime(idx["date"].to_numpy()),
                         "symbol": idx["symbol"].to_numpy(),
                         "row": np.arange(len(idx), dtype=np.int64)})
    keep = left.merge(mem, on=["date", "symbol"], how="inner").sort_values("row")
    sel = keep["row"].to_numpy()
    if sel.size == 0:
        raise SystemExit("--cells n50 selected 0 rows: membership and windows_index "
                         "do not overlap. Check the date dtype and the symbol spelling.")
    return sel


def spectrum(H: np.ndarray) -> tuple[float, int, int]:
    """Effective rank, #PCs for 95% of variance, and how many rows were not finite.

    An untrained transformer can lose rank doubly exponentially with depth, so this is the
    diagnostic that makes the pretrained-vs-reservoir gate interpretable. Subsampling to
    <= ~20k rows is what the 2026-08-28 metas did; kept verbatim so the numbers compare --
    on an all-finite array the stride and therefore the spectrum are unchanged.

    Non-finite rows are DROPPED rather than left to raise "SVD did not converge": that
    exception lands after the forward pass is already paid for and costs a 40-minute
    extraction its sidecar. The honest record of an overflow is a count in the meta, which
    the reader can act on; it should be 0 for every arm on the ladder.
    """
    Hf = np.asarray(H, dtype=np.float32)
    ok = np.isfinite(Hf).all(axis=1)
    n_bad = int((~ok).sum())
    if n_bad:
        Hf = Hf[ok]  # only copy when it is needed: these arrays are ~0.5 GB in float32
    n = len(Hf)
    if n < 2:
        return float("nan"), -1, n_bad
    Hc = Hf - Hf.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(Hc[:: max(1, n // 20000)], compute_uv=False)
    p = sv ** 2 / (sv ** 2).sum()
    eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    var95 = int(np.searchsorted(np.cumsum(p), 0.95) + 1)
    return eff_rank, var95, n_bad


def pool_state(x: torch.Tensor, pool: str) -> torch.Tensor:
    """(B, T, d) -> (B, d). 'last' is the record's pooling; 'mean10' is arm A2."""
    if pool == "last":
        return x[:, -1, :]
    if pool == "mean10":
        return x[:, -10:, :].mean(dim=1)
    raise ValueError(pool)


def encode_windows(tok, xb: torch.Tensor, want_z: bool):
    """tok.encode(x, half=True), optionally also returning the pre-quantisation z.

    The two branches are the same arithmetic -- KronosTokenizer.encode IS
    embed -> encoder -> quant_embed -> BSQuantizer -- but the default path calls
    tok.encode verbatim so nothing about the reproducible run depends on this function
    being a faithful copy.
    """
    if not want_z:
        s1, s2 = tok.encode(xb, half=True)
        return s1, s2, None
    z = tok.embed(xb)
    for layer in tok.encoder:
        z = layer(z)
    z = tok.quant_embed(z)
    _loss, _q, z_indices = tok.tokenizer(z, half=True, collect_metrics=False)
    s1, s2 = z_indices
    return s1, s2, z


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["pretrained", "reservoir"], required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="pilot: first N cells only")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default=None)
    ap.add_argument("--model", choices=["base", "small", "mini"], default="base",
                    help="capacity ladder (A3/A4); mini uses Kronos-Tokenizer-2k")
    ap.add_argument("--layers", default="",
                    help="comma list of 1-based transformer blocks to dump, e.g. 4,8,12 "
                         "(residual stream after the block, BEFORE mdl.norm). "
                         "Empty = final post-norm state only, the old behaviour.")
    ap.add_argument("--pool", default="last",
                    help=f"comma list from {{{','.join(POOLS)}}}")
    ap.add_argument("--tokenizer-z", action="store_true",
                    help="also dump quant_embed(encoder(embed(x))), pre-BSQ (arm A5)")
    ap.add_argument("--cells", choices=["all", "n50"], default="all",
                    help="n50 restricts to point-in-time NIFTY 50 (date, symbol) rows")
    ap.add_argument("--out-dir", default=None,
                    help=f"default {N50}; omit it AND every new flag to reproduce the "
                         f"old {ML}/hT_{{arm}}.npy exactly")
    args = ap.parse_args()

    # "Old behaviour" is not a mode the user asks for, it is the absence of every request
    # for new behaviour -- so it cannot be reached by accident from a new-flag invocation.
    legacy = (args.out_dir is None and not args.layers.strip() and args.pool == "last"
              and args.model == "base" and not args.tokenizer_z and args.cells == "all")
    out_dir = ML if legacy else (args.out_dir or N50)
    os.makedirs(out_dir, exist_ok=True)

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    idx = pd.read_parquet(os.path.join(ML, "windows_index.parquet"))
    W = np.load(os.path.join(ML, "windows.npy"), mmap_mode="r")

    sel = select_rows(idx, args.cells)
    if args.limit:
        sel = sel[: args.limit]
    n = len(sel)
    pools = parse_pools(args.pool)

    tok, mdl, provenance = build_model(args.arm, device, args.seed, args.model)
    d_model = int(mdl.d_model)
    layers = parse_layers(args.layers, int(mdl.n_layers))
    print(f"arm={args.arm} model={args.model} device={device} n={n} d_model={d_model} "
          f"layers={layers or 'final only'} pool={pools} cells={args.cells}", flush=True)

    # Every axis that changes WHAT is in the array has to be in its name, or one run
    # silently overwrites another. Two reservoir seeds are two different arms (A6 exists
    # precisely to measure the spread between them), and the n50 cell set is a different
    # row set under the same (model, arm, layer, pool) -- so both are in the stem.
    arm_tag = args.arm if args.arm == "pretrained" else f"{args.arm}_s{args.seed}"
    pilot_tag = f"_pilot{args.limit}" if args.limit else ""
    suffix = ("_n50" if args.cells == "n50" else "") + pilot_tag
    if args.cells == "n50":
        rows_path = os.path.join(out_dir, f"rows_n50{pilot_tag}.parquet")
        pd.DataFrame({"row": sel,
                      "date": pd.to_datetime(idx["date"].to_numpy())[sel],
                      "symbol": idx["symbol"].to_numpy()[sel]}
                     ).to_parquet(rows_path, index=False)
        print(f"  cells=n50: {n}/{len(idx)} rows -> {os.path.basename(rows_path)}",
              flush=True)
    else:
        rows_path = None

    # ---- output arrays -------------------------------------------------------------
    # open_memmap writes a byte-identical .npy to np.save while keeping peak RSS flat:
    # 3 layers x 2 pools + final x 2 pools is 2 GB of fp16 at 150,527 cells.
    tok_dim = int(tok.s1_bits + tok.s2_bits)
    specs = []  # (name, width, source_key, pool, dtype)
    if legacy:
        specs.append((f"hT_{args.arm}{suffix}", d_model, "final", "last", np.float16))
    else:
        stem = f"hT_{args.model}_{arm_tag}"
        for i in layers:
            for p in pools:
                specs.append((f"{stem}_L{i}_{p}{suffix}", d_model, f"L{i}", p,
                              np.float16))
        for p in pools:
            specs.append((f"{stem}_final_{p}{suffix}", d_model, "final", p, np.float16))
        if args.tokenizer_z:
            # the z block belongs to the TOKENIZER, not the trunk, so it is named for the
            # tokenizer; the 2k variant gets its own name rather than overwriting the base
            # one when --model mini is extracted.
            #
            # float32, unlike every trunk state. quant_embed's output is the input to a
            # BSQuantizer that normalises it onto the unit sphere before taking signs, so
            # nothing downstream of the tokenizer constrains its scale -- and measured on
            # this panel it runs to +/-6.5e4, i.e. straight through the fp16 ceiling of
            # 65504 into inf. 20 columns x 150,527 cells is 12 MB in float32; the fp16
            # economy that matters for an 832-wide trunk state does not apply here.
            ztag = "tokz" if TOKENIZER_REPOS[args.model][0] == TOKENIZER_REPO else "tokz2k"
            for p in POOLS:  # A5 is specified as last AND mean10, whatever --pool says
                specs.append((f"hT_{ztag}_{p}{suffix}", tok_dim, "z", p, np.float32))
    outs = {name: np.lib.format.open_memmap(os.path.join(out_dir, f"{name}.npy"),
                                            mode="w+", dtype=dt, shape=(n, width))
            for name, width, _src, _p, dt in specs}

    # ---- layer capture --------------------------------------------------------------
    # The hook fires on mdl.transformer[i-1] and hands back that block's OUTPUT, i.e. the
    # residual stream after block i and before mdl.norm. decode_s1 returns the post-norm
    # state separately, so both sides of the norm come out of one forward pass.
    grabbed: dict[str, torch.Tensor] = {}
    handles = []
    for i in layers:
        def _hook(_mod, _inp, out, key=f"L{i}"):
            grabbed[key] = out[0] if isinstance(out, tuple) else out
        handles.append(mdl.transformer[i - 1].register_forward_hook(_hook))

    # stamps: the 90 context session dates per cell. Built per row because on the PIT
    # panel names that listed mid-window do not share a calendar (signals.py:249-256).
    from bench.nse.mlcore import LOOKBACK, load_panel, symbol_blocks
    panel, _cal, _m = load_panel(pit=True)
    blocks = symbol_blocks(panel)
    dates, syms = idx["date"].to_numpy()[sel], idx["symbol"].to_numpy()[sel]

    t0 = time.time()
    for c0 in range(0, n, args.batch_size):
        c1 = min(c0 + args.batch_size, n)
        rows = sel[c0:c1]
        xb = torch.from_numpy(np.asarray(W[rows], dtype=np.float32)).to(device)
        st = []
        for i in range(c0, c1):
            b = blocks[syms[i]]
            p = b["pos"][pd.Timestamp(dates[i])]
            st.append(stamps_for(b["dates"][p + 1 - LOOKBACK: p + 1]))
        sb = torch.from_numpy(np.stack(st)).to(device)

        s1, s2, z = encode_windows(tok, xb, args.tokenizer_z)
        _logits, h = mdl.decode_s1(s1, s2, sb)
        states = {"final": h, "z": z, **grabbed}
        for name, _w, src, p, dt in specs:
            outs[name][c0:c1] = pool_state(states[src], p).float().cpu().numpy(
            ).astype(dt)
        grabbed.clear()

        if (c0 // args.batch_size) % 100 == 0:
            el = time.time() - t0
            done = c1
            print(f"  {done}/{n}  {el:.0f}s  {1000 * el / max(done, 1):.1f}ms/series  "
                  f"ETA {(n - done) * el / max(done, 1) / 60:.1f}min", flush=True)
    for hd in handles:
        hd.remove()
    elapsed = time.time() - t0

    for name, _w, src, p, dt in specs:
        H = outs[name]
        H.flush()
        eff_rank, var95, n_bad = spectrum(H)
        if n_bad:
            print(f"  WARNING {name}: {n_bad}/{n} rows are not finite and were dropped "
                  f"from the rank diagnostic", flush=True)
        if legacy:
            # byte-for-byte the 2026-08-28 sidecar: same keys, same order, no additions.
            meta = {"arm": args.arm, "n": int(n), "d_model": d_model, "device": device,
                    "batch_size": args.batch_size, "seed": args.seed,
                    "elapsed_s": elapsed,
                    "ms_per_series": 1000 * elapsed / max(n, 1),
                    "effective_rank": eff_rank, "n_pcs_for_95pct_var": var95,
                    "provenance": {k: v for k, v in provenance.items()
                                   if not k.startswith("tokenizer_")},
                    "decoding_path_used": "decode_s1 only; no decode_s2, no sampling, "
                                          "no decode",
                    "torch": torch.__version__}
        else:
            meta = {"array": name, "arm": args.arm, "model": args.model,
                    "state": ("residual stream after block " + src[1:] + ", pre-norm")
                             if src.startswith("L") else
                             ("final post-norm trunk state" if src == "final" else
                              "tokenizer quant_embed(encoder(embed(x))), pre-BSQ"),
                    "layer": int(src[1:]) if src.startswith("L") else src,
                    "pool": p, "cells": args.cells, "limit": args.limit,
                    "dtype": np.dtype(dt).name, "n_nonfinite_rows": n_bad,
                    "rows_sidecar": os.path.basename(rows_path) if rows_path else None,
                    "n": int(n), "d_model": int(H.shape[1]), "trunk_d_model": d_model,
                    "n_layers": int(mdl.n_layers), "device": device,
                    "batch_size": args.batch_size, "seed": args.seed,
                    "elapsed_s": elapsed, "ms_per_series": 1000 * elapsed / max(n, 1),
                    "effective_rank": eff_rank, "n_pcs_for_95pct_var": var95,
                    "provenance": provenance,
                    "decoding_path_used": "decode_s1 only; no decode_s2, no sampling, "
                                          "no decode",
                    "torch": torch.__version__, "written_at": utc_now(),
                    "git_sha": git_sha(), "argv": sys.argv, "packages": pkg_versions()}
        json.dump(meta, open(os.path.join(out_dir, f"{name}_meta.json"), "w"),
                  indent=2, default=str)
        print(f"wrote {name}.npy {tuple(H.shape)}  eff_rank {eff_rank:.1f}  "
              f"95% var in {var95} PCs", flush=True)
    print(f"{len(specs)} array(s) in {elapsed:.0f}s "
          f"({1000 * elapsed / max(n, 1):.1f}ms/series) -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
