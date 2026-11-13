#!/usr/bin/env python
"""Clean the inclusive piles in the two ways a person's look at the sample sheets asked for.

1. Pairs where one word contains the other are dropped. An ostrich is a bird and a turtle is
   nearly a tortoise, so one animal that is both is the right picture there, not a smooshed
   failure, and labelling it neither would teach the head that a correct picture is wrong. The
   generic nouns and the near-synonym pairs are constants below, so the list is in the diff.

2. Every two-animal picture from the cache must hold the two animals it was asked for. On the
   sheet about 7 of 24 did not: "hornbill and rhino" drew two rhinos, "meerkat and wildebeest" two
   meerkats. Under inclusive labels that picture would teach the head a hornbill is present. Each
   animal box is cut out and CLIP is asked which of the two prompted animals it looks more like,
   as raw similarities with no softmax. The picture is kept only if its boxes pick both animals.
   Judging finished pictures is the job CLIP was measured to do well here; steering is the job it
   failed at, and this step never steers.

The single clean animals found in the neither pile are left in and counted as a known error,
since they cannot be told from a smoosh automatically.

Writes, under MOUNT_ROOT/outputs/cooccurrence_head/piles/:
    piles_clean.json       the piles after both clean-ups
    clean_summary.json     what was dropped, by which rule, and the counts left
    sheet_removed_both.png 24 random pictures the species check threw out
"""
from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from poe_repair import paths  # noqa: E402
from cooccurrence_label_cache import noun  # noqa: E402

PILES = paths.MOUNT_ROOT / "outputs/cooccurrence_head/piles"
CLIP_ID = "openai/clip-vit-base-patch32"
TEMPLATES = ("a photo of a {}", "a {}", "a close-up photo of a {}")
BOX_PAD = 0.08

# A pair holding one of these is dropped: the word names a whole group the other animal belongs to.
GENERIC_NOUNS = {"bird", "fish", "fowl", "eater"}
# Pairs whose two animals are near enough that one animal can honestly be both.
NEAR_SYNONYMS = [{"turtle", "tortoise"}, {"frog", "toad"}, {"rabbit", "hare"},
                 {"alligator", "crocodile"}, {"dolphin", "porpoise"}, {"donkey", "mule"},
                 {"horse", "pony"}, {"llama", "alpaca"}, {"llama", "guanaco"},
                 {"alpaca", "vicuna"}, {"guanaco", "vicuna"}, {"puppy", "dog"},
                 {"kitten", "cat"}, {"hound", "dog"}, {"lamb", "sheep"}, {"pig", "boar"},
                 {"rhino", "rhinoceros"}, {"hippo", "hippopotamus"}, {"goose", "duck"}]


def pair_nouns(slug):
    a, b = slug.split("__x__")
    return noun(a), noun(b)


def drop_reason(slug):
    a, b = pair_nouns(slug)
    if a in GENERIC_NOUNS or b in GENERIC_NOUNS:
        return "a generic noun"
    if {a, b} in NEAR_SYNONYMS or a == b:
        return "near synonyms"
    return None


def crop(im, box):
    x0, y0, x1, y1 = box
    pad = BOX_PAD * max(x1 - x0, y1 - y0)
    w, h = im.size
    return im.crop((max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)))


def main() -> int:
    from transformers import CLIPModel, CLIPProcessor
    if not torch.cuda.is_available():
        raise SystemExit("guard: no CUDA")
    device = torch.device("cuda")
    model = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_ID)
    text_cache = {}

    def text_dir(word):
        if word not in text_cache:
            with torch.no_grad():
                ins = proc(text=[t.format(word) for t in TEMPLATES], return_tensors="pt",
                           padding=True).to(device)
                f = F.normalize(model.get_text_features(**ins), dim=-1).mean(0)
            text_cache[word] = F.normalize(f, dim=-1)
        return text_cache[word]

    rows = json.loads((PILES / "piles.json").read_text())
    kept, dropped, removed_both = [], collections.Counter(), []
    for r in rows:
        if r["source"] != "cache":
            kept.append(r)
            continue
        why = drop_reason(r["pair"])
        if why:
            dropped[f"{r['pile']}: {why}"] += 1
            continue
        if r["pile"] == "both":
            a, b = pair_nouns(r["pair"])
            ta, tb = text_dir(a), text_dir(b)
            picks = set()
            with Image.open(r["path"]) as im:
                im = im.convert("RGB")
                for box in r["boxes"]:
                    with torch.no_grad():
                        px = proc(images=[crop(im, box)], return_tensors="pt")["pixel_values"].to(device)
                        f = F.normalize(model.get_image_features(pixel_values=px), dim=-1)[0]
                    picks.add(a if float(f @ ta) >= float(f @ tb) else b)
            if picks != {a, b}:
                dropped["both: boxes did not pick both asked-for animals"] += 1
                removed_both.append({**r, "boxes_picked": sorted(picks)})
                continue
        kept.append(r)

    counts = collections.Counter((r["source"], r["pile"]) for r in kept)
    summary = {
        "dropped": dict(dropped),
        "kept": {f"{s}:{p}": v for (s, p), v in sorted(counts.items())},
        "n_kept": len(kept),
        "known_error_left_in": "some product pictures in the neither pile are one clean animal "
                               "rather than a smoosh; on a 24-picture sample about 5 were",
        "generic_nouns": sorted(GENERIC_NOUNS),
        "near_synonyms": [sorted(p) for p in NEAR_SYNONYMS],
    }
    (PILES / "piles_clean.json").write_text(json.dumps(kept))
    (PILES / "clean_summary.json").write_text(json.dumps(summary, indent=2))

    rng = random.Random(0)
    show = rng.sample(removed_both, min(24, len(removed_both)))
    cols = 6
    fig, axes = plt.subplots((len(show) + cols - 1) // cols, cols, figsize=(cols * 2.6, 4 * 2.9))
    axes = axes.ravel()
    for ax, r in zip(axes, show):
        with Image.open(r["path"]) as im:
            ax.imshow(im.convert("RGB").resize((280, 280)))
        a, b = pair_nouns(r["pair"])
        ax.set_title(f"asked: {a} + {b}\nboxes look like: {' + '.join(r['boxes_picked'])}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(show):]:
        ax.axis("off")
    fig.suptitle("REMOVED from the both pile: two animals, but not the two that were asked for",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PILES / "sheet_removed_both.png", dpi=100)
    plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("near_synonyms", "generic_nouns")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
