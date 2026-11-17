#!/usr/bin/env python
"""Can a frozen image model tell two separate things from a hybrid, on pairs it never saw?

The best case for a CLIP- or DINOv2-based discriminator: clean finished pictures, no noise. Every
called picture of ``training_set_v3.json`` (plan 22: 81 positives, 362 negatives) is embedded at
224 px by frozen CLIP ViT-L/14 and frozen DINOv2-with-registers base; a class-balanced logistic
regression is fitted on the training tiles and scored on the held-out tiles (7 whole pairs plus
cat x dog seeds 7 and 8), the same split plan 22's discriminators failed on. A second estimate
leaves out one training pair at a time.

The bar is plan 22's, read on clean pictures: held-out balanced accuracy at least 0.75. Below it,
no guidance built on that frozen model is worth trying.

Writes, under MOUNT_ROOT/outputs/classifier_guidance_ladder/rung2/probe/: probe.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from poe_repair import paths  # noqa: E402

TILES = Path(__file__).resolve().parent / "training_set_v3.json"
OUT = paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2/probe"
BAR = 0.75


def picture_path(t) -> Path:
    if t["source"] == "local":
        return paths.MOUNT_ROOT / t["picture"]
    return paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2/training_set/tiles" / f"{t['id']}.png"


@torch.no_grad()
def embed(tiles, device):
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    clip_p = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    dino = AutoModel.from_pretrained("facebook/dinov2-with-registers-base").to(device).eval()
    dino_p = AutoImageProcessor.from_pretrained("facebook/dinov2-with-registers-base")
    feats = defaultdict(list)
    for i in range(0, len(tiles), 16):
        ims = [Image.open(picture_path(t)).convert("RGB") for t in tiles[i:i + 16]]
        c = clip.get_image_features(**clip_p(images=ims, return_tensors="pt").to(device))
        feats["clip"].append(torch.nn.functional.normalize(c, dim=-1).cpu())
        h = dino(**dino_p(images=ims, return_tensors="pt").to(device)).last_hidden_state
        cls, patches = h[:, 0], h[:, 5:]                       # 4 register tokens after CLS
        feats["dino_cls"].append(torch.nn.functional.normalize(cls, dim=-1).cpu())
        feats["dino_cls_and_patch_mean"].append(torch.cat(
            [torch.nn.functional.normalize(cls, dim=-1), torch.nn.functional.normalize(patches.mean(1), dim=-1)], -1).cpu())
    return {k: torch.cat(v).float() for k, v in feats.items()}


def fit_and_score(x_tr, y_tr, x_te, y_te, weight_decay=1e-3, steps=2000):
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    a, b = (x_tr - mu) / sd, (x_te - mu) / sd
    w = torch.zeros(a.shape[1], requires_grad=True)
    c = torch.zeros(1, requires_grad=True)
    pos_w = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())
    opt = torch.optim.LBFGS([w, c], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            a @ w + c, y_tr, pos_weight=pos_w) + weight_decay * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        pred = ((b @ w + c) > 0).float()
    tpr = (pred[y_te == 1] == 1).float().mean().item() if (y_te == 1).any() else float("nan")
    tnr = (pred[y_te == 0] == 0).float().mean().item() if (y_te == 0).any() else float("nan")
    return (tpr + tnr) / 2, tpr, tnr, pred


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = json.loads(TILES.read_text())
    tiles = data["tiles"]
    y = torch.tensor([float(t["label"]) for t in tiles])
    check = torch.tensor([bool(t["check"]) for t in tiles])
    feats = embed(tiles, device)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"held_out_pairs": data["held_out_pairs"], "n_train": int((~check).sum()), "n_check": int(check.sum()),
              "bar": BAR, "features": {}}
    kinds = [t["kind"] for t in tiles]
    for name, x in feats.items():
        bal, tpr, tnr, pred = fit_and_score(x[~check], y[~check], x[check], y[check])
        idx = check.nonzero().flatten().tolist()
        per_kind = defaultdict(lambda: [0, 0])
        for j, i in enumerate(idx):
            right = int(pred[j].item() == y[i].item())
            per_kind[kinds[i]][0] += right
            per_kind[kinds[i]][1] += 1
        train_idx = (~check).nonzero().flatten().tolist()
        pairs = sorted({tiles[i]["pair"] for i in train_idx})
        lopo = []
        for p in pairs:
            te = [i for i in train_idx if tiles[i]["pair"] == p]
            if len({tiles[i]["label"] for i in te}) < 2:
                continue
            tr = [i for i in train_idx if tiles[i]["pair"] != p]
            lopo.append(fit_and_score(x[tr], y[tr], x[te], y[te])[0])
        report["features"][name] = {
            "held_out_balanced_accuracy": bal, "held_out_true_positive_rate": tpr, "held_out_true_negative_rate": tnr,
            "held_out_right_by_kind": {k: f"{r} of {n}" for k, (r, n) in per_kind.items()},
            "leave_one_pair_out_balanced_accuracy_median": float(np.median(lopo)) if lopo else None,
            "leave_one_pair_out_pairs": len(lopo), "passes_bar": bal >= BAR}
        print(f"{name:26s} held-out balanced accuracy {bal:.3f} (yes {tpr:.2f}, no {tnr:.2f}); "
              f"leave-one-pair-out median {np.median(lopo) if lopo else float('nan'):.3f} over {len(lopo)} pairs; "
              f"by kind {dict((k, f'{r}/{n}') for k, (r, n) in per_kind.items())}", flush=True)
    (OUT / "probe.json").write_text(json.dumps(report, indent=2))
    print("PROBE DONE", OUT / "probe.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
