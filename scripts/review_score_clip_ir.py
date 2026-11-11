"""CLIP score and ImageReward for the paper-review renders (route step 9).

CLIP score: openai/clip-vit-large-patch14, 100 * max(cosine(image, full prompt), 0),
the torchmetrics CLIPScore definition. ImageReward: ImageReward-v1.0 through
ImageReward.load("ImageReward-v1.0").score(prompt, image), the two calls
composition/focus/eval_metrics.py's ImageRewardScorer makes (that file does not import
under co3's transformers 4.44.1, which lacks Qwen2-VL).

Usage:
  python scripts/review_score_clip_ir.py --images a.png b.png --prompts "p" "p" --out probe.csv
  python scripts/review_score_clip_ir.py --in rows.csv --out scores.csv   (columns image,prompt;
      rows already in --out are skipped and new rows appended, so a stopped run resumes)
"""
import os
from __future__ import annotations

import argparse
import csv

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import ImageReward as RM

CLIP_ID = "openai/clip-vit-large-patch14"


class ClipScore:
    def __init__(self, device):
        self.device = device
        self.model = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_ID)

    @torch.no_grad()
    def score(self, images, texts):
        out = []
        for img, txt in zip(images, texts):
            inp = self.proc(text=[txt], images=img, return_tensors="pt", padding=True,
                            truncation=True, max_length=77).to(self.device)
            i = self.model.get_image_features(pixel_values=inp["pixel_values"])
            t = self.model.get_text_features(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
            cos = torch.nn.functional.cosine_similarity(i, t).item()
            out.append(100.0 * max(cos, 0.0))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+")
    ap.add_argument("--prompts", nargs="+")
    ap.add_argument("--in", dest="inp")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.inp:
        pairs = [(r["image"], r["prompt"]) for r in csv.DictReader(open(a.inp))]
    else:
        assert a.images and len(a.images) == len(a.prompts)
        pairs = list(zip(a.images, a.prompts))
    done = set()
    if os.path.exists(a.out):
        done = {r["image"] for r in csv.DictReader(open(a.out))}
    todo = [p for p in pairs if p[0] not in done]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_m = ClipScore(dev)
    rm = RM.load("ImageReward-v1.0", device=dev)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["image", "prompt", "clip_score", "image_reward"])
        for n, (path, prompt) in enumerate(todo):
            img = Image.open(path).convert("RGB")
            c = clip_m.score([img], [prompt])[0]
            with torch.no_grad():
                r = float(rm.score(prompt, img))
            w.writerow([path, prompt, f"{c:.4f}", f"{r:.4f}"])
            if n % 100 == 0:
                f.flush(); print(f"{n + 1}/{len(todo)}", flush=True)
    print(f"done: {len(todo)} scored, {len(done)} already in {a.out}")


if __name__ == "__main__":
    main()
