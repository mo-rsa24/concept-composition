#!/usr/bin/env python
"""Sort the counted pictures into the piles the inclusive presence head learns from.

The rule is the user's inclusive labelling plus the one line that makes it mean two animals. A
picture holding a bear and a fox is labelled bear present and fox present, both yes. A smooshed
animal is labelled neither, because a fused bear-fox looks like a bear and like a fox at once (on
cat x dog, CLIP rated the smooshes 0.262 cat-like and 0.268 dog-like against 0.265 and 0.262 for
real photos of each), so a head that called it both would already be satisfied by the failure.

Piles, from the validated instance count:
    both      a joint or product picture of an animal pair with two or more animal boxes:
              yes to both of the pair's species
    neither   a product picture of an animal pair with exactly one box: a smooshed animal,
              no to both. Some may be one clean animal of a single species rather than a smoosh;
              the neither sheet exists so a person can see how many.
    skipped   a joint picture with one box (the joint prompt failed and drew one thing, of a
              species the counter cannot name), and any picture with no animal box
Single-species pictures come from the audited COCO pool, cat-only and dog-only, so the two
species this experiment steers on are seen alone as well as together.

Writes, under MOUNT_ROOT/outputs/cooccurrence_head/piles/:
    piles.json          one row per picture: path, species present, pile, source
    summary.json        counts per pile and source, the species list
    sheet_both.png, sheet_neither.png   24 random pictures from each, labelled with their pair
"""
from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from poe_repair import paths  # noqa: E402
from cooccurrence_label_cache import noun  # noqa: E402

LABELS = paths.MOUNT_ROOT / "outputs/cooccurrence_head/labels/rows.json"
MANIFEST = paths.MOUNT_ROOT / "outputs/inclusive_captions_pool/manifest.json"
OUT = paths.MOUNT_ROOT / "outputs/cooccurrence_head/piles"
SHEET_N = 24
SEED = 0


def sheet(rows, title, path):
    rng = random.Random(SEED)
    picked = rng.sample(rows, min(SHEET_N, len(rows)))
    cols = 6
    nr = (len(picked) + cols - 1) // cols
    fig, axes = plt.subplots(nr, cols, figsize=(cols * 2.6, nr * 2.9))
    axes = axes.ravel()
    for ax, r in zip(axes, picked):
        with Image.open(r["path"]) as im:
            ax.imshow(im.convert("RGB").resize((280, 280)))
        a, b = r["pair"].split("__x__")
        ax.set_title(f"{noun(a)} x {noun(b)}\n{r['sampler']}, seed {r['seed']}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(picked):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    counted = json.loads(LABELS.read_text())
    piles, skipped = [], collections.Counter()
    for r in counted:
        a, b = (noun(s) for s in r["pair"].split("__x__"))
        if r["count"] >= 2:
            pile, present = "both", sorted({a, b})
        elif r["count"] == 1 and r["sampler"] == "product":
            pile, present = "neither", []
        else:
            skipped[f"{r['sampler']} with {r['count']} box"] += 1
            continue
        piles.append({**r, "pile": pile, "present": present, "source": "cache"})

    manifest = json.loads(MANIFEST.read_text())
    for set_name, present in (("cat_only", ["cat"]), ("dog_only", ["dog"]),
                              ("both", ["cat", "dog"]), ("neither", [])):
        for e in manifest["sets"][set_name]:
            piles.append({"path": e["file"], "pair": None, "seed": None, "sampler": "photo",
                          "pile": {"cat_only": "single", "dog_only": "single"}.get(set_name, set_name),
                          "present": present, "source": "coco"})

    species = sorted({s for r in piles for s in r["present"]})
    counts = collections.Counter((r["source"], r["pile"]) for r in piles)
    summary = {
        "rule": "two or more animal boxes on an animal pair is yes to both of its species; a "
                "product picture with one box is a smooshed animal and no to both; audited COCO "
                "photos supply cat alone, dog alone, both, and neither",
        "n_rows": len(piles),
        "counts": {f"{s}:{p}": v for (s, p), v in sorted(counts.items())},
        "skipped": dict(skipped),
        "n_species": len(species), "species": species,
    }
    (OUT / "piles.json").write_text(json.dumps(piles))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    cache_rows = [r for r in piles if r["source"] == "cache"]
    sheet([r for r in cache_rows if r["pile"] == "both"],
          "BOTH pile: 24 random pictures the counter found two animals in.\n"
          "Each should hold two separate animals.", OUT / "sheet_both.png")
    sheet([r for r in cache_rows if r["pile"] == "neither"],
          "NEITHER pile: 24 random product pictures with one animal box.\n"
          "Each should be ONE smooshed animal. A clean single animal here is a mislabel.",
          OUT / "sheet_neither.png")
    print(json.dumps({k: v for k, v in summary.items() if k != "species"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
