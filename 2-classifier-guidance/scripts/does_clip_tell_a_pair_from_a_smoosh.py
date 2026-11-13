#!/usr/bin/env python
"""Does CLIP already tell a cat-and-dog pair from one smooshed cat-dog animal?

This decides how classifier guidance toward co-occurrence has to be built. If CLIP scores real
pairs clearly above smooshes against "a cat and a dog", CLIP can steer as it is. If it scores
them about the same, CLIP cannot see the difference, and any guidance built on it has to be
taught the difference first.

Two captions, each read as a raw cosine similarity with no softmax, because a softmax between
captions forces a choice and hid a smooshed animal behind a confident single-species call before:
    pair     "a photo of a cat and a dog"
    smoosh   "a photo of one animal that is half cat and half dog"
The number that would let guidance push toward co-occurrence is pair minus smoosh: positive on a
real pair, negative on a smooshed animal.

Groups, whole image each:
    smooshed renders   the 8 product renders a person confirmed are single fused animals
    joint renders      "a cat and a dog" rendered directly, 8 held-out seeds, same model
    pair photos        20 audited COCO photos holding a real cat and a real dog
    cat photos, dog photos   20 each, single animals, for reference
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from poe_repair import paths  # noqa: E402

CLIP_ID = "openai/clip-vit-base-patch32"
TEMPLATES = ("{}", "a photograph showing {}", "an image of {}")
PAIR_TEXT = "a cat and a dog"
SMOOSH_TEXT = "one animal that is half cat and half dog"
OUT = paths.MOUNT_ROOT / "outputs/presence_head/clip_pair_check"


def text_direction(model, proc, phrase, device):
    prompts = [t.format(phrase) for t in TEMPLATES]
    with torch.no_grad():
        ins = proc(text=prompts, return_tensors="pt", padding=True)
        f = model.get_text_features(input_ids=ins["input_ids"].to(device),
                                    attention_mask=ins["attention_mask"].to(device))
    f = f / f.norm(dim=-1, keepdim=True)
    m = f.mean(dim=0)
    return m / m.norm()


def image_feature(model, proc, path, device):
    with Image.open(path) as im:
        with torch.no_grad():
            px = proc(images=[im.convert("RGB")], return_tensors="pt")["pixel_values"].to(device)
            f = model.get_image_features(pixel_values=px)
    return (f / f.norm(dim=-1, keepdim=True))[0]


def main() -> None:
    from transformers import CLIPModel, CLIPProcessor
    if not torch.cuda.is_available():
        raise SystemExit("guard: no CUDA")
    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    model = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_ID)
    pair_dir = text_direction(model, proc, PAIR_TEXT, device)
    smoosh_dir = text_direction(model, proc, SMOOSH_TEXT, device)

    hc = json.loads((paths.MOUNT_ROOT / "outputs/presence_head/hybrid_check/per_render.json").read_text())
    man = json.loads((paths.MOUNT_ROOT / "outputs/inclusive_captions_pool/manifest.json").read_text())
    tiles = paths.MOUNT_ROOT / "outputs/presence_head/sweep/tiles"
    groups = {
        "smooshed renders": [r["path"] for r in hc if r["cat_pass"] and r["dog_pass"]],
        "joint renders": [str(tiles / f"seed{s:02d}__joint_prompt.png") for s in range(9, 17)],
        "pair photos": [e["file"] for e in man["sets"]["both"][:20]],
        "cat photos": [e["file"] for e in man["sets"]["cat_only"][:20]],
        "dog photos": [e["file"] for e in man["sets"]["dog_only"][:20]],
    }

    rows = []
    for g, ps in groups.items():
        for p in ps:
            f = image_feature(model, proc, p, device)
            pair = float(f @ pair_dir)
            smoosh = float(f @ smoosh_dir)
            rows.append({"group": g, "path": p, "pair": round(pair, 4),
                         "smoosh": round(smoosh, 4), "pair_minus_smoosh": round(pair - smoosh, 4)})

    summary = {}
    print(f"{'group':18s} {'n':>3s}  {'to pair':>8s}  {'to smoosh':>9s}  {'pair - smoosh':>14s}  range of pair - smoosh")
    for g in groups:
        rs = [r for r in rows if r["group"] == g]
        d = [r["pair_minus_smoosh"] for r in rs]
        summary[g] = {"n": len(rs),
                      "median_to_pair": statistics.median(r["pair"] for r in rs),
                      "median_to_smoosh": statistics.median(r["smoosh"] for r in rs),
                      "median_pair_minus_smoosh": statistics.median(d),
                      "min": min(d), "max": max(d)}
        s = summary[g]
        print(f"{g:18s} {s['n']:3d}  {s['median_to_pair']:8.4f}  {s['median_to_smoosh']:9.4f}  "
              f"{s['median_pair_minus_smoosh']:+14.4f}  {s['min']:+.4f} to {s['max']:+.4f}")

    (OUT / "scores.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    # The pictures that matter, ranked by the number guidance would follow.
    show = [r for r in rows if r["group"] in ("smooshed renders", "joint renders")]
    show.sort(key=lambda r: r["pair_minus_smoosh"])
    colour = {"smooshed renders": "crimson", "joint renders": "darkgreen"}
    n = len(show)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(((n + 1) // 2) * 2.4, 6.0))
    axes = axes.ravel()
    for ax, r in zip(axes, show):
        with Image.open(r["path"]) as im:
            ax.imshow(im.convert("RGB").resize((300, 300)))
        tag = "SMOOSH" if r["group"] == "smooshed renders" else "JOINT"
        ax.set_title(f"{tag}\n{r['pair_minus_smoosh']:+.3f}", fontsize=10, color=colour[r["group"]])
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Ranked left to right by how much more CLIP thinks each picture is\n"
                 "\"a cat and a dog\" than \"one animal that is half cat and half dog\". "
                 "Red = smooshed, green = asked for both directly.", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(OUT / "ranked.png", dpi=100)
    print("wrote", OUT / "ranked.png")


if __name__ == "__main__":
    main()
