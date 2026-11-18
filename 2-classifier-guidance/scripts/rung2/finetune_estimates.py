#!/usr/bin/env python
"""Can the fine-tuned CLIP checker read SDXL's product estimate of the finished picture early enough to push?

The fully fine-tuned CLIP ViT-L/14 checker reads finished cat x dog pictures at 0.900 held-out balanced accuracy
(finetune/clip/best.pt), but a push acts on the sampler's estimates of the finished picture, which are blurry
early: the frozen CLIP linear checker reads them at 0.540 over steps 0 to 19 (probe/estimate.json). This takes
the estimates make_estimates.py drew and
    1. scores the finished-picture checker on the 220 held-out pictures' estimates at steps 0 to 40, as it is;
    2. fine-tunes it further on the training pictures' estimates (steps 10 to 30) together with their finished
       pictures, the epoch chosen on the same 10% validation split of training the finished-picture run held back
       (seed 0), so no validation picture was seen by either run;
    3. scores it again per step.
The per-step number says from which step a push through this checker could see what it is pushing toward.

Writes, under MOUNT_ROOT/outputs/classifier_guidance_ladder/rung2/:
    probe/finetune_estimates_<tag>.json   per-step balanced accuracy before training and after each epoch
    finetune/estimates-<tag>/best.pt      the chosen epoch's weights
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
import augment_encode  # noqa: E402
import probe_frozen_features as pff  # noqa: E402
from finetune_checker import SPECS, Checker, Pictures, balanced, load, logits_of  # noqa: E402
from poe_repair import paths  # noqa: E402

RUNG2 = pff.OUT.parent


def estimates():
    rows = {}
    for f in sorted((RUNG2 / "estimates").glob("index-*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[(r["id"], r["step"])] = r
    return list(rows.values())


def per_step(model, groups, device, kw, tf):
    out = {}
    for s, (ims, ys) in sorted(groups.items(), key=lambda kv: (isinstance(kv[0], str), str(kv[0]).zfill(3))):
        dl = torch.utils.data.DataLoader(Pictures(ims, ys, tf), **kw)
        out[s] = balanced(logits_of(model, dl, device) > 0, np.array(ys))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="clip", choices=sorted(SPECS))
    ap.add_argument("--init", type=Path, default=RUNG2 / "finetune/clip/best.pt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--tag", default="clip")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    device = torch.device("cuda")
    torch.manual_seed(0)
    pff.picture_path = augment_encode.picture_path

    train_all = json.loads((HERE / "training_set_catdog_more23.json").read_text())["tiles"]
    evals = json.loads((HERE / "eval_set_catdog_v2.json").read_text())["tiles"]
    rng = random.Random(0)  # the finished-picture run's split (its --seed 0)
    val_idx = set()
    for lab in (0, 1):
        idx = [i for i, t in enumerate(train_all) if t["label"] == lab]
        rng.shuffle(idx)
        val_idx |= set(idx[: round(0.1 * len(idx))])
    val_ids = {train_all[i]["id"] for i in val_idx}
    train = [t for t in train_all if t["id"] not in val_ids]
    val = [t for t in train_all if t["id"] in val_ids]

    est = estimates()
    eval_ids = {t["id"] for t in evals}
    tr_est = [r for r in est if r["split"] == "train" and r["id"] not in val_ids and r["id"] not in eval_ids]
    va_est = [r for r in est if r["split"] == "train" and r["id"] in val_ids]
    te_est = [r for r in est if r["split"] == "eval"]
    print(f"estimates: train {len(tr_est)}, validation {len(va_est)}, held out {len(te_est)} "
          f"(steps {sorted({r['step'] for r in te_est})})", flush=True)

    open_est = lambda rows: [Image.open(paths.MOUNT_ROOT / r["path"]).convert("RGB") for r in rows]
    spec = SPECS[args.model]
    norm = T.Normalize(spec["mean"], spec["std"])
    tf_train = T.Compose([T.RandomResizedCrop(spec["res"], scale=(0.6, 1.0), ratio=(0.85, 1.15)),
                          T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2, 0.02), T.ToTensor(), norm])
    tf_eval = T.Compose([T.Resize((spec["res"], spec["res"])), T.ToTensor(), norm])

    ims_tr = open_est(tr_est) + load(train)
    y_tr = [r["label"] for r in tr_est] + [t["label"] for t in train]
    ims_va, y_va = open_est(va_est), np.array([r["label"] for r in va_est])
    groups = collections.defaultdict(lambda: ([], []))
    for r, im in zip(te_est, open_est(te_est)):
        groups[r["step"]][0].append(im)
        groups[r["step"]][1].append(r["label"])
    groups["finished"] = (load(evals), [t["label"] for t in evals])
    kw = {"batch_size": args.batch, "num_workers": 6}
    dl_tr = torch.utils.data.DataLoader(Pictures(ims_tr, y_tr, tf_train), shuffle=True, drop_last=True, **kw)
    dl_va = torch.utils.data.DataLoader(Pictures(ims_va, y_va, tf_eval), **kw)

    model = Checker(args.model).to(device)
    model.load_state_dict(torch.load(args.init, map_location=device)["state"])
    show = lambda d: "  ".join(f"{k}:{v:.3f}" for k, v in d.items())
    before = per_step(model, groups, device, kw, tf_eval)
    print("before (the finished-picture checker as it is):", show(before), flush=True)

    opt = torch.optim.AdamW([{"params": model.net.parameters(), "lr": args.lr},
                             {"params": model.head.parameters(), "lr": args.lr * 10}], weight_decay=0.05)
    total, warm = args.epochs * len(dl_tr), len(dl_tr) // 2
    base = [g["lr"] for g in opt.param_groups]
    pos_weight = torch.tensor((len(y_tr) - sum(y_tr)) / max(1, sum(y_tr)), device=device)
    out_dir = RUNG2 / "finetune" / f"estimates-{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    va0 = balanced(logits_of(model, dl_va, device) > 0, y_va)
    curve = [{"epoch": 0, "validation": va0, "held_out": before}]
    best, step, t0 = {"epoch": 0, "validation": va0}, 0, time.time()
    torch.save({"model": args.model, "state": model.state_dict(), "epoch": 0}, out_dir / "best.pt")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in dl_tr:
            scale = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (total - warm)))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * scale
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(x.to(device))
            loss = nn.functional.binary_cross_entropy_with_logits(o.float(), y.float().to(device), pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            step += 1
        va = balanced(logits_of(model, dl_va, device) > 0, y_va)
        te = per_step(model, groups, device, kw, tf_eval)
        curve.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": va, "held_out": te})
        print(f"epoch {epoch}  loss {np.mean(losses):.3f}  validation {va:.3f}  held out {show(te)}  "
              f"({time.time() - t0:.0f} s)", flush=True)
        if va > best["validation"]:
            best = {"epoch": epoch, "validation": va}
            torch.save({"model": args.model, "state": model.state_dict(), "epoch": epoch}, out_dir / "best.pt")

    chosen = curve[best["epoch"]]["held_out"]
    mid = [v for k, v in chosen.items() if k != "finished" and 10 <= k <= 30]
    report = {"init": str(args.init), "estimates_train": len(tr_est), "estimates_validation": len(va_est),
              "before": before, "curve": curve, "chosen_epoch": best["epoch"], "after": chosen,
              "after_mean_steps_10_to_30": float(np.mean(mid)),
              "references": {"frozen_clip_linear_on_estimates_steps_0_to_19": 0.540,
                             "readout_iteration_4_steps_0_to_19": 0.755}}
    (pff.OUT / f"finetune_estimates_{args.tag}.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"chosen epoch {best['epoch']} (validation {best['validation']:.3f}); after: {show(chosen)}", flush=True)
    print(f"mean over steps 10 to 30: {report['after_mean_steps_10_to_30']:.3f}", flush=True)
    print("ESTIMATES FINETUNE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
