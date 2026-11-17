#!/usr/bin/env python
"""Build training_set_v4.json: clean two-thing pictures against rejects, with every source on both sides.

v1 learned which prompt made a picture instead of what it showed. v4 guards against that by taking
both sides from each source, so the source cannot separate them and only the content can. Every
picture is labelled by what it shows (the Claude API's verdict, or the judge's), never by the method
that drew it:

| Source | Real (a clean pair) | Fake |
|---|---|---|
| plan 23 drawings (kim) | API "faithful pair" | every other API verdict |
| judge renders (present-revision-to-supervisor) | "two", photo | "hybrid", "one", "count error", "broken", or "two" drawn as an illustration |
| inclusive-captions photos (COCO, SD 3.5) | "both" photos the API passes on the square crop | "both" rejects, every cat-only and dog-only photo |
| v3 hand calls (training tiles only) | hand positive | hand negative |

Left out: the judge's "intrusion" (both drawn plus an extra thing, neither side), every picture of a
v3 held-out pair, v3's check tiles, and cat x dog seeds 9 to 16.

Non-square photos are centre-cropped to a square here, the crop the encoder would make, and the API
labels for COCO "both" photos must come from those crops (--crop-labels).

Writes training_set_v4.json beside this script and the pictures to --stage (rsync to
MOUNT_ROOT/outputs/classifier_guidance_ladder/rung2/dataset-v4/pictures/).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RES = REPO / "2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog"
JUDGE = REPO / "artifacts/drips/present-revision-to-supervisor/judge"
PHOTOS = REPO / "4-inclusive-captions/data/training-set"
V4 = "outputs/classifier_guidance_ladder/rung2/dataset-v4"
KIM = "outputs/classifier_guidance_ladder/rung2/kim/states"


def jsonl(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", type=Path, required=True, help="local folder the square pictures are written to")
    ap.add_argument("--crop-labels", type=Path, required=True, help="API labels of the cropped 'both' photos")
    args = ap.parse_args()
    v3 = json.loads((HERE / "training_set_v3.json").read_text())
    held = set(v3["held_out_pairs"])
    tiles = []

    def add(**t):
        if t["pair"] in held or (t["pair"] == "a_cat__x__a_dog" and 9 <= t["seed"] <= 16):
            return
        tiles.append(t | {"check": False})

    def stage(src: Path, tid: str) -> str:
        out = args.stage / f"{tid}.jpg"
        if not out.exists():
            with Image.open(src) as im:
                im = im.convert("RGB")
                w, h = im.size
                s = min(w, h)
                im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).save(out, quality=95)
        return f"{V4}/pictures/{tid}.jpg"

    args.stage.mkdir(parents=True, exist_ok=True)
    for r in jsonl(RES / "api-labels/drawings.jsonl"):
        if "verdict" not in r:
            continue
        real = r["verdict"] == "faithful pair"
        add(id=f"kim-{r['id']}", latent=f"{KIM}/{r['id']}.pt", state_index=-1, pair=r["pair"], seed=r["seed"],
            method=r["method"], label=int(real), kind="positive" if real else r["verdict"], source="kim")

    key = {r["id"]: r for r in csv.DictReader(open(JUDGE / "key.csv"))}
    fake_judge = {"hybrid", "one", "count error", "broken"}
    for r in csv.DictReader(open(JUDGE / "judge_labels.csv")):
        k, lab = key.get(r["id"]), r["label"]
        if k is None or lab == "intrusion" or not lab:
            continue
        real = lab == "two" and r["style_loss"] == "no"
        if not real and lab not in fake_judge and lab != "two":
            continue
        tid = f"judge-{r['id']}"
        add(id=tid, picture=stage(JUDGE / "images" / f"{r['id']}.jpg", tid), latent=f"{V4}/latents/{tid}.pt",
            pair=k["pair"], seed=int(k["seed"]), method=k["setting"], label=int(real),
            kind="positive" if real else ("illustration" if lab == "two" else lab), source="judge")

    crop = {r["id"]: r for r in jsonl(args.crop_labels) if "verdict" in r}
    for f in sorted((PHOTOS / "both").glob("*")):
        i, src = int(f.name.split("_")[0]), f.name.split("_")[1]
        lid = f"a_cat__x__a_dog__seed{(1000 if src == 'coco' else 2000) + i}__photo"
        if lid not in crop:
            continue
        real = crop[lid]["verdict"] == "faithful pair"
        tid = f"photo-both-{f.stem}"
        add(id=tid, picture=stage(f, tid), latent=f"{V4}/latents/{tid}.pt", pair="a_cat__x__a_dog", seed=-1,
            method=src, label=int(real), kind="positive" if real else crop[lid]["verdict"], source="photos")
    for sub in ("cat_only", "dog_only"):
        for f in sorted((PHOTOS / sub).glob("*")):
            tid = f"photo-{sub}-{f.stem}"
            add(id=tid, picture=stage(f, tid), latent=f"{V4}/latents/{tid}.pt", pair="a_cat__x__a_dog", seed=-1,
                method=f.name.split("_")[1], label=0, kind="single", source="photos")

    for t in v3["tiles"]:
        if not t["check"]:
            tiles.append(t | {"source": "v3"})

    out = {"rule": __doc__.split("\n\n")[1], "held_out_pairs": sorted(held), "tiles": tiles}
    (HERE / "training_set_v4.json").write_text(json.dumps(out, indent=0))
    c = collections.Counter((t["source"], t["label"]) for t in tiles)
    for k in sorted(c):
        print(f"{k[0]:7s} {'real' if k[1] else 'fake'} {c[k]}")
    print(f"total {len(tiles)}; pairs with a real example {len({t['pair'] for t in tiles if t['label'] == 1})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
