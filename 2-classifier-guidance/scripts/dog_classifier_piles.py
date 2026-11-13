#!/usr/bin/env python
"""The two piles a dog-only classifier learns from: a dog is in the picture, or it is not.

The frozen head steered toward scenes and people because it was taught that one animal in frame
means no: about 3,000 single smooshed animals were labelled no against about 500 single animals
labelled yes. Here a single dog is the main positive and no smooshed animal involving a dog appears
at all; those return only when "both" is taught.

    dog       COCO photos with a dog box at least MIN_BOX_FRACTION of the longer side and no cat;
              the 91 audited COCO photos holding a cat and a dog; cache pictures the species check
              confirmed hold a dog among two animals
    no dog    SDXL's own no-prompt renders, which is where guidance starts; COCO photos with a cat
              and no dog; COCO photos with neither; cache pictures of two animals neither a dog;
              cache smooshes of pairs with no dog in them

Writes, under MOUNT_ROOT/outputs/dog_classifier/piles/:
    rows.json      one row per picture: path, dog (0 or 1), source
    summary.json   the count from each source
    sheet_dog.png, sheet_no_dog.png   24 random pictures from each pile
"""
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

COCO_ANN = Path("/datasets/mmolefe/coco2017/annotations_trainval2017/annotations/instances_train2017.json")
COCO_IMG = Path("/datasets/mmolefe/coco2017/train2017/train2017")
MANIFEST = paths.MOUNT_ROOT / "outputs/inclusive_captions_pool/manifest.json"
PILES = paths.MOUNT_ROOT / "outputs/cooccurrence_head/piles/piles_clean.json"
NO_PROMPT = paths.MOUNT_ROOT / "outputs/dog_classifier/no_prompt"

MIN_BOX_FRACTION = 0.25
CAT_ID, DOG_ID = 17, 18
N_COCO_CAT_ONLY = 1000
N_COCO_NEITHER = 1200
N_CACHE_TWO_NOT_DOG = 800
N_CACHE_SMOOSH_NOT_DOG = 500
SEED = 0


def sheet(rows, title, path):
    pick = random.Random(SEED).sample(rows, min(24, len(rows)))
    fig, axes = plt.subplots(4, 6, figsize=(6 * 2.6, 4 * 2.8))
    for ax, r in zip(axes.ravel(), pick):
        with Image.open(r["path"]) as im:
            ax.imshow(im.convert("RGB").resize((260, 260)))
        ax.set_title(r["source"], fontsize=8)
        ax.axis("off")
    for ax in axes.ravel()[len(pick):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--species", choices=("dog", "cat"), default="dog",
                    help="the animal the classifier looks for; with cat every pile mirrors")
    sp = ap.parse_args().species
    other = "cat" if sp == "dog" else "dog"
    T_ID, O_ID = (DOG_ID, CAT_ID) if sp == "dog" else (CAT_ID, DOG_ID)
    OUT = paths.MOUNT_ROOT / f"outputs/{sp}_classifier/piles"
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    coco = json.loads(COCO_ANN.read_text())
    size = {im["id"]: (im["width"], im["height"]) for im in coco["images"]}
    biggest = collections.defaultdict(lambda: {CAT_ID: 0.0, DOG_ID: 0.0})
    has = collections.defaultdict(set)
    for a in coco["annotations"]:
        has[a["image_id"]].add(a["category_id"])
        if a["category_id"] in (CAT_ID, DOG_ID):
            w, h = size[a["image_id"]]
            frac = max(a["bbox"][2], a["bbox"][3]) / max(w, h)
            b = biggest[a["image_id"]]
            b[a["category_id"]] = max(b[a["category_id"]], frac)

    def photo(i):
        return str(COCO_IMG / f"{i:012d}.jpg")

    target_only = [i for i in size if T_ID in has[i] and O_ID not in has[i]
                   and biggest[i][T_ID] >= MIN_BOX_FRACTION]
    other_only = [i for i in size if O_ID in has[i] and T_ID not in has[i]
                  and biggest[i][O_ID] >= MIN_BOX_FRACTION]
    neither = [i for i in size if CAT_ID not in has[i] and DOG_ID not in has[i]]

    rows = [{"path": photo(i), sp: 1, "source": f"coco {sp}"} for i in target_only]
    manifest = json.loads(MANIFEST.read_text())
    rows += [{"path": e["file"], sp: 1, "source": "coco cat and dog"} for e in manifest["sets"]["both"]]
    rows += [{"path": photo(i), sp: 0, "source": f"coco {other}"} for i in rng.sample(other_only, N_COCO_CAT_ONLY)]
    rows += [{"path": photo(i), sp: 0, "source": "coco neither"} for i in rng.sample(neither, N_COCO_NEITHER)]
    rows += [{"path": str(p), sp: 0, "source": "sdxl no prompt"} for p in sorted(NO_PROMPT.glob("seed_*.png"))]

    cache = [r for r in json.loads(PILES.read_text()) if r["source"] == "cache"]
    def pair_has_target(r):
        a, b = r["pair"].split("__x__")
        return sp in (noun(a), noun(b))
    rows += [{"path": r["path"], sp: 1, "source": f"cache two animals, one a {sp}"}
             for r in cache if r["pile"] == "both" and sp in r["present"]]
    two_without = [r for r in cache if r["pile"] == "both" and not pair_has_target(r)]
    smoosh_without = [r for r in cache if r["pile"] == "neither" and not pair_has_target(r)]
    rows += [{"path": r["path"], sp: 0, "source": f"cache two animals, no {sp}"}
             for r in rng.sample(two_without, min(N_CACHE_TWO_NOT_DOG, len(two_without)))]
    rows += [{"path": r["path"], sp: 0, "source": f"cache smoosh, no {sp}"}
             for r in rng.sample(smoosh_without, min(N_CACHE_SMOOSH_NOT_DOG, len(smoosh_without)))]

    counts = collections.Counter((r[sp], r["source"]) for r in rows)
    summary = {"n_rows": len(rows), f"n_{sp}": sum(r[sp] for r in rows),
               f"n_no_{sp}": sum(1 - r[sp] for r in rows),
               "by_source": {f"{sp if d else 'no ' + sp}: {s}": v for (d, s), v in sorted(counts.items())},
               "rule": f"a single {sp} is the main positive; no smooshed animal involving a {sp} appears"}
    (OUT / "rows.json").write_text(json.dumps(rows))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    sheet([r for r in rows if r[sp] == 1], f"{sp.upper()} pile: 24 random pictures. Each should hold a {sp}.",
          OUT / f"sheet_{sp}.png")
    sheet([r for r in rows if r[sp] == 0], f"NO {sp.upper()} pile: 24 random pictures. None should hold a {sp}.",
          OUT / f"sheet_no_{sp}.png")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
