#!/usr/bin/env python
"""How well can a fully fine-tuned pretrained network tell a clean cat and dog from a reject, on finished pictures?

The noisy-latent readout checkers stop near 0.755 held-out balanced accuracy (averaged over sampler steps 0 to
19), and CLIP ViT-L/14 features with only a linear layer on top reach 0.877 on finished pictures. This
measures the finished-picture ceiling with every weight of a pretrained network trained on our labels. If it
reaches about 0.95, the labels are learnable and the noise of the early steps is what holds the readout back;
if it stays near 0.88, the labels themselves are the limit.

Models (both cached on the cluster): DINOv2 ViT-B/14 with registers at 448 x 448 (torch hub, local), and the
CLIP ViT-L/14 vision tower at 224 x 224. The head reads the class token (and, for DINOv2, the mean patch token).

Data: training_set_catdog_more23.json (1,603 pictures, 576 clean); 10% of it, stratified by label, is held
back to choose the epoch. eval_set_catdog_v2.json's 220 pictures (109 clean, 111 rejects) are scored every
epoch for the curve, but the reported number is the epoch the validation split chose.

Writes, under MOUNT_ROOT/outputs/classifier_guidance_ladder/rung2/:
    probe/finetune_<model>.json      per-epoch curve, chosen epoch, held-out balanced accuracy, per reject kind
    finetune/<model>/best.pt         the chosen epoch's weights
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

SPECS = {
    "dinov2": {"res": 448, "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    "clip": {"res": 224, "mean": (0.48145466, 0.4578275, 0.40821073), "std": (0.26862954, 0.26130258, 0.27577711)},
}


class Checker(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        if name == "dinov2":
            hub = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
            self.net = torch.hub.load(str(hub), "dinov2_vitb14_reg", source="local")
            width = 2 * self.net.embed_dim
        else:
            from transformers import CLIPVisionModel
            self.net = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
            width = self.net.config.hidden_size
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Dropout(0.1), nn.Linear(width, 1))

    def forward(self, x):
        if self.name == "dinov2":
            f = self.net.forward_features(x)
            z = torch.cat([f["x_norm_clstoken"], f["x_norm_patchtokens"].mean(1)], -1)
        else:
            z = self.net(pixel_values=x).pooler_output
        return self.head(z.float()).squeeze(-1)


class Pictures(torch.utils.data.Dataset):
    def __init__(self, ims, labels, tf):
        self.ims, self.labels, self.tf = ims, labels, tf

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, i):
        return self.tf(self.ims[i]), float(self.labels[i])


def load(tiles):
    out = []
    for i, t in enumerate(tiles):
        out.append(augment_encode.square(Image.open(augment_encode.picture_path(t)).convert("RGB"), 512))
        if i % 400 == 0:
            print(f"  loaded {i} of {len(tiles)}", flush=True)
    return out


def balanced(pred, y):
    pred, y = np.asarray(pred, bool), np.asarray(y, bool)
    return float((pred[y].mean() + (~pred[~y]).mean()) / 2)


@torch.no_grad()
def logits_of(model, loader, device):
    model.eval()
    out = []
    for x, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(x.to(device)).float().cpu())
    return torch.cat(out).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(SPECS), required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5, help="backbone learning rate; the head uses 100 times this")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    device = torch.device("cuda")
    print("card:", torch.cuda.get_device_name(0), flush=True)
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    pff.picture_path = augment_encode.picture_path

    train_all = json.loads((HERE / "training_set_catdog_more23.json").read_text())["tiles"]
    evals = json.loads((HERE / "eval_set_catdog_v2.json").read_text())["tiles"]
    assert not {t["id"] for t in train_all} & {t["id"] for t in evals}, "training and evaluation overlap"
    rng = random.Random(args.seed)
    val_idx = set()
    for lab in (0, 1):
        idx = [i for i, t in enumerate(train_all) if t["label"] == lab]
        rng.shuffle(idx)
        val_idx |= set(idx[: round(0.1 * len(idx))])
    train = [t for i, t in enumerate(train_all) if i not in val_idx]
    val = [t for i, t in enumerate(train_all) if i in val_idx]
    print(f"train {len(train)} ({sum(t['label'] for t in train)} clean), validation {len(val)} "
          f"({sum(t['label'] for t in val)} clean), held out {len(evals)} ({sum(t['label'] for t in evals)} clean)",
          flush=True)

    spec = SPECS[args.model]
    norm = T.Normalize(spec["mean"], spec["std"])
    tf_train = T.Compose([T.RandomResizedCrop(spec["res"], scale=(0.6, 1.0), ratio=(0.85, 1.15)),
                          T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2, 0.02), T.ToTensor(), norm])
    tf_eval = T.Compose([T.Resize((spec["res"], spec["res"])), T.ToTensor(), norm])
    y_tr = [t["label"] for t in train]
    y_va = np.array([t["label"] for t in val])
    y_te = np.array([t["label"] for t in evals])
    ims_tr, ims_va, ims_te = load(train), load(val), load(evals)
    kw = {"batch_size": args.batch, "num_workers": 6}
    dl_tr = torch.utils.data.DataLoader(Pictures(ims_tr, y_tr, tf_train), shuffle=True, drop_last=True, **kw)
    dl_va = torch.utils.data.DataLoader(Pictures(ims_va, y_va, tf_eval), **kw)
    dl_te = torch.utils.data.DataLoader(Pictures(ims_te, y_te, tf_eval), **kw)

    model = Checker(args.model).to(device)
    opt = torch.optim.AdamW([{"params": model.net.parameters(), "lr": args.lr},
                             {"params": model.head.parameters(), "lr": args.lr * 100}], weight_decay=0.05)
    total = args.epochs * len(dl_tr)
    warm = len(dl_tr)
    base = [g["lr"] for g in opt.param_groups]
    pos_weight = torch.tensor((len(y_tr) - sum(y_tr)) / max(1, sum(y_tr)), device=device)
    out_dir = pff.OUT.parent / "finetune" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    curve, best, step, t0 = [], None, 0, time.time()
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
        te = balanced(logits_of(model, dl_te, device) > 0, y_te)
        curve.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": va, "held_out": te})
        print(f"epoch {epoch:2d}  loss {np.mean(losses):.3f}  validation {va:.3f}  held out {te:.3f}  "
              f"({time.time() - t0:.0f} s)", flush=True)
        if best is None or va > best["validation"]:
            best = dict(curve[-1])
            torch.save({"model": args.model, "state": model.state_dict(), "epoch": epoch}, out_dir / "best.pt")

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device)["state"])
    lg = logits_of(model, dl_te, device)
    pred = lg > 0
    per_kind = collections.defaultdict(lambda: [0, 0])
    for t, p in zip(evals, pred):
        right = bool(p) == bool(t["label"])
        per_kind[t["kind"]][0] += int(right)
        per_kind[t["kind"]][1] += 1
    report = {"model": args.model, "resolution": spec["res"], "train": len(train), "validation": len(val),
              "held_out": len(evals), "curve": curve, "chosen_epoch": best["epoch"],
              "held_out_balanced_accuracy": balanced(pred, y_te),
              "clean_called_clean": float(pred[y_te == 1].mean()), "rejects_called_reject": float((~pred[y_te == 0]).mean()),
              "per_kind_right": {k: {"right": r, "of": n} for k, (r, n) in sorted(per_kind.items())},
              "logits": {t["id"]: float(v) for t, v in zip(evals, lg)},
              "references": {"clip_linear_finished": 0.877, "readout_iteration_4_steps_0_to_19": 0.755}}
    (pff.OUT / f"finetune_{args.model}.json").write_text(json.dumps(report, indent=2))
    print(f"chosen epoch {best['epoch']} (validation {best['validation']:.3f}): held out "
          f"{report['held_out_balanced_accuracy']:.3f} (clean called clean {report['clean_called_clean']:.2f}, "
          f"rejects called reject {report['rejects_called_reject']:.2f})", flush=True)
    print("per kind:", {k: f"{r}/{n}" for k, (r, n) in sorted(per_kind.items())}, flush=True)
    print("FINETUNE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
