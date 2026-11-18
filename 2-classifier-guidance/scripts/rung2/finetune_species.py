#!/usr/bin/env python
"""Plan 26's species checker: "is there a clear cat?" and "is there a clear dog?", read on half-drawn guesses.

The readout checker's push makes two bodies early but leaves which animal each becomes to PoE, which mostly
draws two cats or a cat and a kitten. Plan 26 adds a second push from step 10 on that raises whichever animal
is missing. This trains the checker for it: the CLIP ViT-L/14 body fine-tuned on finished cat x dog pictures
(finetune/clip/best.pt) with its last layer replaced by two outputs, a clear cat present and a clear dog present.

Labels are presence_catdog.json's per-picture answers ({tile id: [cat, dog]}, -1 where unknown, left out of
the loss). Training pictures are training_set_catdog_more23.json's finished pictures and their half-drawn
guesses from make_estimates.py (steps 10 to 30), minus the same 10% validation split the clean-pair runs held
back (seed 0). The epoch is chosen on the validation guesses (mean of the two outputs' balanced accuracy).
Scored on eval_set_catdog_v2.json's 220 pictures' guesses at steps 0 to 40 and finished.

Writes, under MOUNT_ROOT/outputs/classifier_guidance_ladder/rung2/:
    probe/finetune_species.json   per-step balanced accuracy of each output, before training and per epoch
    finetune/species-clip/best.pt the chosen epoch's weights ({"model", "state", "epoch", "n_out": 2})
"""
from __future__ import annotations

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
from finetune_checker import SPECS, Checker, load  # noqa: E402
from finetune_estimates import estimates  # noqa: E402
from poe_repair import paths  # noqa: E402

RUNG2 = pff.OUT.parent
INIT = RUNG2 / "finetune/clip/best.pt"
EPOCHS, BATCH, LR = 8, 32, 1e-5


class Items(torch.utils.data.Dataset):
    def __init__(self, ims, ys, tf):
        self.ims, self.ys, self.tf = ims, ys, tf

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, i):
        return self.tf(self.ims[i]), torch.tensor(self.ys[i], dtype=torch.float32)


def balanced(pred, truth):
    pos = [p for p, t in zip(pred, truth) if t == 1]
    neg = [not p for p, t in zip(pred, truth) if t == 0]
    return (sum(pos) / len(pos) + sum(neg) / len(neg)) / 2 if pos and neg else float("nan")


@torch.no_grad()
def read(model, ims, tf, device):
    model.eval()
    out = []
    for i in range(0, len(ims), BATCH):
        x = torch.stack([tf(im) for im in ims[i:i + BATCH]]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(x).float().cpu())
    return torch.cat(out)


def score(model, groups, tf, device):
    res = {}
    for s, (ims, ys) in groups.items():
        o = read(model, ims, tf, device)
        res[s] = {"cat": balanced((o[:, 0] > 0).tolist(), [y[0] for y in ys]),
                  "dog": balanced((o[:, 1] > 0).tolist(), [y[1] for y in ys])}
    return res


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    device = torch.device("cuda")
    torch.manual_seed(0)
    pff.picture_path = augment_encode.picture_path
    presence = json.loads((HERE / "presence_catdog.json").read_text())
    train_all = json.loads((HERE / "training_set_catdog_more23.json").read_text())["tiles"]
    evals = json.loads((HERE / "eval_set_catdog_v2.json").read_text())["tiles"]
    rng = random.Random(0)  # the clean-pair runs' split (their --seed 0)
    val_idx = set()
    for lab in (0, 1):
        idx = [i for i, t in enumerate(train_all) if t["label"] == lab]
        rng.shuffle(idx)
        val_idx |= set(idx[: round(0.1 * len(idx))])
    val_ids = {train_all[i]["id"] for i in val_idx}
    known = lambda tid: presence.get(tid, [-1, -1])[0] >= 0
    train = [t for t in train_all if t["id"] not in val_ids and known(t["id"])]
    est = estimates()
    tr_est = [r for r in est if r["split"] == "train" and r["id"] not in val_ids and known(r["id"])]
    va_est = [r for r in est if r["split"] == "train" and r["id"] in val_ids and known(r["id"])]
    te_est = [r for r in est if r["split"] == "eval" and known(r["id"])]
    evals = [t for t in evals if known(t["id"])]
    print(f"with cat and dog answers: {len(train)} finished training pictures, {len(tr_est)} training guesses, "
          f"{len(va_est)} validation guesses, {len(te_est)} held-out guesses, {len(evals)} held-out finished", flush=True)

    open_est = lambda rows: [Image.open(paths.MOUNT_ROOT / r["path"]).convert("RGB") for r in rows]
    spec = SPECS["clip"]
    norm = T.Normalize(spec["mean"], spec["std"])
    tf_train = T.Compose([T.RandomResizedCrop(spec["res"], scale=(0.6, 1.0), ratio=(0.85, 1.15)),
                          T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2, 0.02), T.ToTensor(), norm])
    tf_eval = T.Compose([T.Resize((spec["res"], spec["res"])), T.ToTensor(), norm])
    ims_tr = open_est(tr_est) + load(train)
    y_tr = [presence[r["id"]] for r in tr_est] + [presence[t["id"]] for t in train]
    groups = collections.defaultdict(lambda: ([], []))
    for r, im in zip(te_est, open_est(te_est)):
        groups[r["step"]][0].append(im)
        groups[r["step"]][1].append(presence[r["id"]])
    groups = dict(sorted(groups.items()))
    groups["finished"] = (load(evals), [presence[t["id"]] for t in evals])
    ims_va, y_va = open_est(va_est), [presence[r["id"]] for r in va_est]
    count = collections.Counter(tuple(y) for y in y_tr)
    print("training answers [cat, dog]:", dict(count), flush=True)

    model = Checker("clip", n_out=2).to(device)
    init = torch.load(INIT, map_location=device)["state"]
    missing = model.load_state_dict({k: v for k, v in init.items() if not k.startswith("head.2.")}, strict=False)
    print("started from the clean-pair checker; new:", missing.missing_keys, flush=True)
    show = lambda d: "  ".join(f"{k}: cat {v['cat']:.3f} dog {v['dog']:.3f}" for k, v in d.items())
    before = score(model, groups, tf_eval, device)
    print("before (new last layer, untrained):", show(before), flush=True)

    dl = torch.utils.data.DataLoader(Items(ims_tr, y_tr, tf_train), batch_size=BATCH, shuffle=True, drop_last=True,
                                     num_workers=6)
    opt = torch.optim.AdamW([{"params": model.net.parameters(), "lr": LR},
                             {"params": model.head.parameters(), "lr": LR * 100}], weight_decay=0.05)
    total, warm = EPOCHS * len(dl), len(dl) // 2
    base = [g["lr"] for g in opt.param_groups]
    y = torch.tensor(y_tr, dtype=torch.float32)
    pos_w = torch.tensor([((y[:, k] == 0).sum() / (y[:, k] == 1).sum().clamp(min=1)).item() for k in range(2)],
                         device=device)
    out_dir = RUNG2 / "finetune/species-clip"
    out_dir.mkdir(parents=True, exist_ok=True)
    curve, best, step, t0 = [], None, 0, time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for x, yb in dl:
            sc = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (total - warm)))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * sc
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(x.to(device)).float()
            yb = yb.to(device)
            mask = (yb >= 0).float()
            loss = (nn.functional.binary_cross_entropy_with_logits(o, yb.clamp(min=0), pos_weight=pos_w,
                                                                   reduction="none") * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            step += 1
        vo = read(model, ims_va, tf_eval, device)
        va = (balanced((vo[:, 0] > 0).tolist(), [v[0] for v in y_va]) +
              balanced((vo[:, 1] > 0).tolist(), [v[1] for v in y_va])) / 2
        te = score(model, groups, tf_eval, device)
        curve.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation": va, "held_out": te})
        print(f"epoch {epoch}  loss {np.mean(losses):.3f}  validation {va:.3f}  held out {show(te)}  "
              f"({time.time() - t0:.0f} s)", flush=True)
        if best is None or va > best["validation"]:
            best = {"epoch": epoch, "validation": va, "held_out": te}
            torch.save({"model": "clip", "state": model.state_dict(), "epoch": epoch, "n_out": 2}, out_dir / "best.pt")
    report = {"init": str(INIT), "before": before, "curve": curve, "chosen_epoch": best["epoch"], "after": best["held_out"],
              "bar": "both outputs at 0.80 or more at step 15 (plans/review/26-...)"}
    (pff.OUT / "finetune_species.json").write_text(json.dumps(report, indent=1, default=str))
    s15 = best["held_out"].get(15, {})
    print(f"chosen epoch {best['epoch']}; step 15: cat {s15.get('cat', float('nan')):.3f}, dog {s15.get('dog', float('nan')):.3f}; "
          f"bar 0.80 on both: {'PASSED' if s15 and min(s15.values()) >= 0.80 else 'MISSED'}", flush=True)
    print("SPECIES DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
