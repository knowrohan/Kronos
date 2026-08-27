"""Single-device (MPS) finetuner for the Kronos predictor. Standalone -- no DDP, no comet.

Why this file exists: finetune/train_predictor.py cannot run on this machine. Verified in
bench/out/nse/verify_pipeline.json:
    * ``import comet_ml`` at module scope (line 12) -- hard ImportError before anything runs
    * ``device = torch.device(f"cuda:{local_rank}")`` -- CUDA hardcoded
    * ``if "WORLD_SIZE" not in os.environ: raise`` -- must be launched by torchrun
    * ``torch.cuda.set_device`` + ``dist.all_reduce`` / ``dist.barrier`` throughout

This reuses finetune/dataset.py:QlibDataset UNCHANGED, working around its two traps
(plan 4.8, both confirmed empirically):
    1. ``__getitem__`` ignores ``idx`` and draws from a seeded ``self.py_rng``. With
       ``num_workers>0`` every worker forks the same RNG and emits the SAME sample stream --
       verified: two workers produced bit-identical batches. Mitigation here: num_workers=0.
    2. It constructs ``Config()`` internally, so the dataset path is injected by patching
       ``Config.__init__`` at runtime rather than editing finetune/config.py.

Only the PREDICTOR is finetuned; the tokenizer is frozen. Finetuning both would change the
token vocabulary the predictor was pretrained against, and the plan's like-for-like
comparison is against zero-shot Kronos-base with its own tokenizer.

Optimiser settings mirror finetune/config.py: predictor_learning_rate 4e-5,
AdamW betas (0.9, 0.95), weight decay 0.1.
The schedule is sized from MEASURED steps/sec against a stated wall-clock budget, not from
the repo's epochs=30 / n_train_iter=2000*batch_size, which target a multi-GPU box.
What was cut is recorded in the sidecar.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from bench.nse.common import OUT_NSE, PROCESSED, load_kronos, set_seed, sidecar  # noqa: E402


def make_dataset(data_type: str, dataset_path: str, n_iter: int | None):
    """Instantiate the UNMODIFIED repo QlibDataset against our pickles."""
    sys.path.insert(0, os.path.join(ROOT, "finetune"))
    import config as repo_config
    importlib.reload(repo_config)
    orig = repo_config.Config.__init__

    def patched(self):
        orig(self)
        self.dataset_path = dataset_path
        if n_iter:
            self.n_train_iter = n_iter
            self.n_val_iter = max(n_iter // 5, 1)
    repo_config.Config.__init__ = patched
    try:
        import dataset as repo_dataset
        importlib.reload(repo_dataset)
        return repo_dataset.QlibDataset(data_type=data_type), repo_config.Config()
    finally:
        repo_config.Config.__init__ = orig


def batch_loss(tokenizer, model, x, x_stamp, device):
    """The repo's exact objective, transcribed from finetune/train_predictor.py:101-109.

        token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
        token_in  = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
        token_out = [token_seq_0[:, 1:],  token_seq_1[:, 1:]]
        logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
        loss, s1_loss, s2_loss = model.head.compute_loss(logits[0], logits[1],
                                                        token_out[0], token_out[1])

    Note `use_teacher_forcing` defaults to **False**, so s2 is conditioned on an s1 token
    SAMPLED from the model's own distribution, not on the target. That matches inference,
    where decode_s2 is fed the freshly sampled s1 -- so this is deliberate, and teacher
    forcing here would train against a distribution the model never sees at generation time.
    Loss is (ce_s1 + ce_s2) / 2, per model/module.py:compute_loss.
    """
    x = x.to(device)
    x_stamp = x_stamp.to(device)
    with torch.no_grad():
        t0, t1 = tokenizer.encode(x, half=True)
    token_in = [t0[:, :-1], t1[:, :-1]]
    token_out = [t0[:, 1:], t1[:, 1:]]
    logits = model(token_in[0], token_in[1], x_stamp[:, :-1, :])
    loss, s1_loss, s2_loss = model.head.compute_loss(
        logits[0], logits[1], token_out[0], token_out[1])
    return loss, float(s1_loss.detach()), float(s2_loss.detach())


@torch.no_grad()
def evaluate(tokenizer, model, loader, device, max_batches=None):
    model.eval()
    tot, n = 0.0, 0
    for i, (x, xs) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        loss, _, _ = batch_loss(tokenizer, model, x, xs, device)
        tot += float(loss)
        n += 1
    model.train()
    return tot / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-path", default=PROCESSED)
    ap.add_argument("--out-dir", default=os.path.join(OUT_NSE, "finetune"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=4e-5, help="finetune/config.py predictor_learning_rate")
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--budget-min", type=float, default=90.0, help="wall-clock training budget")
    ap.add_argument("--max-steps", type=int, default=100_000)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    predictor = load_kronos(device=args.device)
    device = predictor.device
    tokenizer, model = predictor.tokenizer, predictor.model
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)
    model.train()

    train_ds, cfg = make_dataset("train", args.dataset_path, n_iter=args.batch_size * 200_000)
    val_ds, _ = make_dataset("val", args.dataset_path, n_iter=args.batch_size * 2_000)
    # num_workers=0 is REQUIRED: QlibDataset.__getitem__ draws from a seeded RNG that every
    # forked worker would share, silently duplicating training data (plan 4.8.1, verified).
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0, shuffle=False)
    print(f"device={device} train_windows={len(train_ds.indices)} val_windows={len(val_ds.indices)} "
          f"batch={args.batch_size}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                            betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

    best = float("inf")
    best_step = -1
    log_rows = []
    t0 = time.time()
    step = 0
    deadline = t0 + args.budget_min * 60
    base_val = evaluate(tokenizer, model, val_dl, device, args.eval_batches)
    print(f"step 0  val_loss {base_val:.5f}  (zero-shot baseline)", flush=True)
    log_rows.append({"step": 0, "train_loss": float("nan"), "val_loss": base_val,
                     "elapsed_sec": 0.0, "lr": 0.0})

    stop = False
    while not stop:
        train_ds.set_epoch_seed(step)
        for x, xs in train_dl:
            if step >= args.max_steps or time.time() > deadline:
                stop = True
                break
            lr = args.lr * min(1.0, (step + 1) / max(args.warmup, 1))
            for g in opt.param_groups:
                g["lr"] = lr
            loss, l1, l2 = batch_loss(tokenizer, model, x, xs, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            step += 1

            if step % args.eval_every == 0:
                v = evaluate(tokenizer, model, val_dl, device, args.eval_batches)
                el = time.time() - t0
                log_rows.append({"step": step, "train_loss": float(loss), "val_loss": v,
                                 "elapsed_sec": round(el, 1), "lr": lr})
                flag = ""
                if v < best:
                    best, best_step = v, step
                    model.save_pretrained(os.path.join(args.out_dir, "best_predictor"))
                    flag = " *best*"
                print(f"step {step:6d}  train {float(loss):.5f}  val {v:.5f}  "
                      f"{el / 60:5.1f}min  {step / el:.2f} steps/s{flag}", flush=True)
                pd.DataFrame(log_rows).to_csv(os.path.join(args.out_dir, "train_log.csv"), index=False)

    el = time.time() - t0
    pd.DataFrame(log_rows).to_csv(os.path.join(args.out_dir, "train_log.csv"), index=False)
    summary = {
        "steps_completed": step, "wall_clock_min": round(el / 60, 2),
        "steps_per_sec": round(step / max(el, 1e-9), 3),
        "samples_seen": step * args.batch_size,
        "zero_shot_val_loss": base_val, "best_val_loss": best, "best_step": best_step,
        "val_loss_improvement": base_val - best,
        "finetuning_helped_on_val": bool(best < base_val),
        "repo_schedule_for_reference": {
            "epochs": 30, "n_train_iter": cfg.n_train_iter,
            "samples_per_repo_run": 30 * cfg.n_train_iter,
        },
        "what_was_cut": (
            f"repo schedule is 30 epochs x {cfg.n_train_iter} samples = "
            f"{30 * cfg.n_train_iter} samples on a multi-GPU box; this run saw "
            f"{step * args.batch_size} samples "
            f"({step * args.batch_size / max(30 * cfg.n_train_iter, 1) * 100:.2f}%) "
            f"in {el / 60:.0f} min on one MPS device. Tokenizer frozen."),
        "best_checkpoint": os.path.join(args.out_dir, "best_predictor"),
    }
    # Was this run short of compute, or short of data? Those need different responses and the
    # curve tells them apart: a compute-limited run's validation loss is still falling when the
    # clock stops, an overfitting one turned upward long before it.
    ev = [r for r in log_rows if r["step"] > 0]
    if ev:
        final_val = ev[-1]["val_loss"]
        overfit = (final_val > best + 0.02) and (best_step < 0.5 * step)
        summary["final_val_loss"] = final_val
        summary["val_rise_after_best"] = final_val - best
        summary["best_step_fraction"] = best_step / max(step, 1)
        summary["train_val_gap_at_end"] = float(ev[-1]["train_loss"]) - final_val
        summary["diagnosis"] = "overfitting" if overfit else "still improving at the deadline"
        summary["diagnosis_note"] = (
            "validation loss bottomed at %.1f%% of the run and then rose %.4f while training "
            "loss kept falling -- more steps would not have helped; this is a data limit, not a "
            "compute limit" % (100 * summary["best_step_fraction"],
                               summary["val_rise_after_best"])
            if overfit else
            "validation loss had not turned upward when the budget ran out, so the run was "
            "compute-limited and a longer schedule is untested")
    with open(os.path.join(args.out_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    sidecar(os.path.join(args.out_dir, "train_mps_meta.json"), vars(args), extra=summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
