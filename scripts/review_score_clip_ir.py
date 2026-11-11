"""CLIP score and ImageReward for the paper-review renders (route step 9).

CLIP score: openai/clip-vit-large-patch14, 100 * max(cosine(image, full prompt), 0),
the torchmetrics CLIPScore definition. ImageReward: ImageReward-v1.0 through
ImageReward.load("ImageReward-v1.0").score(prompt, image), the two calls
composition/focus/eval_metrics.py's ImageRewardScorer makes (that file does not import
under co3's transformers 4.44.1, which lacks Qwen2-VL).

Usage:
  python scripts/review_score_clip_ir.py --images a.png b.png --prompts "p" "p" --out probe.csv
"""
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
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert len(a.images) == len(a.prompts)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imgs = [Image.open(p).convert("RGB") for p in a.images]
    clip = ClipScore(dev).score(imgs, a.prompts)
    rm = RM.load("ImageReward-v1.0", device=dev)
    with torch.no_grad():
        ir = [float(rm.score(t, i)) for t, i in zip(a.prompts, imgs)]
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "prompt", "clip_score", "image_reward"])
        for row in zip(a.images, a.prompts, clip, ir):
            w.writerow([row[0], row[1], f"{row[2]:.4f}", f"{row[3]:.4f}"])
    print(open(a.out).read())


if __name__ == "__main__":
    main()
