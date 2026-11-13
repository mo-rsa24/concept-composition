#!/usr/bin/env python
"""Embed every image and the three captions with CLIP, and record the cosine similarities.

Runs on the cluster, where both the photographs and a torch environment exist. Reads the
manifest push_renders.py wrote, writes similarities.json beside it.

Per image: its cosine similarity to "a photo of a cat", to "a photo of a dog" and to
"a photo of a cat and a dog", plus the smaller of the two single-animal ones. Cosine similarity
here runs from -1 to 1; CLIP's image and text embeddings sit in a narrow band well above 0, so
the spread between sets matters more than any single value.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XFORMERS_DISABLED", "1")

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

OUT = Path(sys.argv[1] if len(sys.argv) > 1
           else "/datasets/mmolefe/poe_repair_min/outputs/hybrid_caption_fit")
MODEL = "openai/clip-vit-base-patch32"
BATCH = 32


def main() -> None:
    manifest = json.loads((OUT / "manifest.json").read_text())
    captions = manifest["captions"]
    rows = manifest["images"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL).to(device).eval()
    proc = CLIPProcessor.from_pretrained(MODEL)

    keys = ["cat", "dog", "both"]
    with torch.no_grad():
        text = proc(text=[captions[k] for k in keys], return_tensors="pt", padding=True)
        text = {k: v.to(device) for k, v in text.items()}
        temb = model.get_text_features(**text)
        temb = temb / temb.norm(dim=-1, keepdim=True)

    out = []
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        images = [Image.open(r["path"]).convert("RGB") for r in chunk]
        with torch.no_grad():
            batch = proc(images=images, return_tensors="pt")
            iemb = model.get_image_features(pixel_values=batch["pixel_values"].to(device))
            iemb = iemb / iemb.norm(dim=-1, keepdim=True)
            sims = (iemb @ temb.T).cpu().tolist()
        for row, sim in zip(chunk, sims):
            rec = dict(row)
            rec.update({f"sim_{k}": float(s) for k, s in zip(keys, sim)})
            rec["sim_weaker"] = min(rec["sim_cat"], rec["sim_dog"])
            out.append(rec)
        print(f"{len(out)}/{len(rows)}", flush=True)

    (OUT / "similarities.json").write_text(json.dumps(
        {"model": MODEL, "captions": captions, "device": device, "records": out}, indent=1))
    print("wrote", OUT / "similarities.json")


if __name__ == "__main__":
    main()
