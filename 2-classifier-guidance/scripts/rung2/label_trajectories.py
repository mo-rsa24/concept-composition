#!/usr/bin/env python
"""Plan 25's labels: how each saved cat x dog run finished, judged by the fine-tuned CLIP checker.

Every run render_trajectories.py saved has a finished picture. The CLIP ViT-L/14 checker fully fine-tuned on
finished pictures (finetune/clip/best.pt, 0.900 held-out balanced accuracy, 46 of 46 wrong pairs caught)
gives each a logit; every saved state of that run inherits the verdict. Only confident verdicts are kept,
to cut the checker's roughly 10% error: clean when the logit is above +MARGIN, reject when below -MARGIN,
left out in between. Seeds up to 3000 are for training, 3001 and above are held out for the reading check.

A spot-check sheet of 16 runs called clean and 16 called reject (training seeds, drawn at random with a fixed
seed) is written for the eye check every new labelled source gets.

Writes, under MOUNT_ROOT/outputs/classifier_guidance_ladder/rung2/trajectories/:
    labels.json   {stem: {"seed", "mode", "split", "logit", "label"}} with label 1, 0 or null
    spot-check.png
"""
from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
from finetune_checker import SPECS, Checker  # noqa: E402
from poe_repair import paths  # noqa: E402

RUNG2 = paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2"
OUT = RUNG2 / "trajectories"
CHECKER = RUNG2 / "finetune/clip/best.pt"
MARGIN = 2.0
HELD_OUT_FROM = 3001


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    device = torch.device("cuda")
    state = torch.load(CHECKER, map_location="cpu")
    model = Checker(state["model"])
    model.load_state_dict(state["state"])
    model = model.to(device).eval()
    spec = SPECS[state["model"]]
    tf = T.Compose([T.Resize((spec["res"], spec["res"])), T.ToTensor(), T.Normalize(spec["mean"], spec["std"])])
    pics = sorted((OUT / "pictures").glob("seed*__*.png"))
    labels = {}
    with torch.no_grad():
        for i in range(0, len(pics), 32):
            chunk = pics[i:i + 32]
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in chunk]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(x).float().cpu().tolist()
            for p, v in zip(chunk, lg):
                seed, mode = int(p.stem.split("__")[0][4:]), p.stem.split("__")[1]
                labels[p.stem] = {"seed": seed, "mode": mode, "split": "held_out" if seed >= HELD_OUT_FROM else "train",
                                  "logit": v, "label": 1 if v > MARGIN else 0 if v < -MARGIN else None}
    (OUT / "labels.json").write_text(json.dumps(labels, indent=0))
    count = collections.Counter((r["split"], r["mode"], r["label"]) for r in labels.values())
    for key in sorted(count, key=str):
        print(f"{key[0]:9s} {key[1]:8s} label {key[2]}: {count[key]}", flush=True)

    rng = random.Random(0)
    train = [s for s, r in labels.items() if r["split"] == "train"]
    pos = rng.sample([s for s in train if labels[s]["label"] == 1], min(16, sum(labels[s]["label"] == 1 for s in train)))
    neg = rng.sample([s for s in train if labels[s]["label"] == 0], min(16, sum(labels[s]["label"] == 0 for s in train)))
    W, font = 256, ImageFont.load_default(size=18)
    sheet = Image.new("RGB", (8 * W, 4 * (W + 26)), "white")
    d = ImageDraw.Draw(sheet)
    for k, s in enumerate(pos + neg):
        x, y = (k % 8) * W, (k // 8) * (W + 26)
        sheet.paste(Image.open(OUT / "pictures" / f"{s}.png").convert("RGB").resize((W, W)), (x, y))
        d.text((x + 4, y + W + 2), f"{'clean' if labels[s]['label'] else 'reject'} {labels[s]['logit']:+.1f} {s}",
               fill="black", font=font)
    sheet.save(OUT / "spot-check.png")
    print("spot check: rows 1 and 2 called clean, rows 3 and 4 called reject:", OUT / "spot-check.png", flush=True)
    print("LABELS DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
