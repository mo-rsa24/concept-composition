#!/usr/bin/env python
"""Build training_set_v4.json: clean two-thing pictures against rejects, with every source on both sides.

v1 learned which prompt made a picture instead of what it showed. v4 guards against that by taking
both sides from each source, so the source cannot separate them and only the content can. Every
picture is labelled by what it shows (the Claude API's verdict, one rule for every source), never by the method
that drew it:

| Source | Real (a clean pair) | Fake |
|---|---|---|
| plan 23 drawings (kim) | API "faithful pair" | every other API verdict |
| judge renders (present-revision-to-supervisor) | API "faithful pair" (api-labels/judge.jsonl) | every other API verdict |
| inclusive-captions photos (COCO, SD 3.5) | "both" photos the API passes on the square crop | "both" rejects, every cat-only and dog-only photo |
| v3 hand calls (training tiles only) | hand positive | hand negative |
| Attend-and-Excite benchmark tiles (paper review, 30 pairs, 5 methods) | strict API "faithful pair" | every other strict verdict |

With --strict-labels, every clean pair is checked again under a stricter rule that also calls
borrowed body shape, build, size or proportions a hybrid (a tiger with a dog's long legs and upright
stature), and a clean pair that fails it moves to the fake side. Evaluation tiles keep their
pre-registered labels and carry the strict verdict beside them as strict_label.

Left out: every pair whose second item is a setting rather than a second body (SCENES below: a
mailbox in a snowfield is one thing in a place, which would teach "an object in a scene" as the good
case), every picture of a
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
SCENES = {"brick_wall", "cactus_garden", "desert_dune", "desert_landscape", "flower_meadow", "forest_clearing",
          "glacier", "grassy_field", "harbor", "iceberg", "lagoon", "ocean_wave", "ocean_with_stormy_waves",
          "pond", "sand_dune", "snowfield", "snowstorm", "snowy_mountain", "tropical_beach", "waterfall"}


# Dropped by the by-eye spot check: a pillow drawn inside the pan like bread, called a clean pair.
DROP = {"judge-" + x for x in ("j59746", "j81739")}


def is_scene(pair: str) -> bool:
    return pair.split("__x__")[1].split("_", 1)[1] in SCENES


def jsonl(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", type=Path, required=True, help="local folder the square pictures are written to")
    ap.add_argument("--crop-labels", type=Path, required=True, help="API labels of the cropped 'both' photos")
    ap.add_argument("--strict-labels", type=Path, help="api_label.py --strict verdicts on every clean pair and the "
                    "Attend-and-Excite tiles; a clean pair that fails them becomes a fake")
    ap.add_argument("--strict-index", type=Path, help="staged number -> tile id map for --strict-labels")
    args = ap.parse_args()
    strict, ae = {}, []
    if args.strict_labels:
        index = json.loads(args.strict_index.read_text())
        for r in jsonl(args.strict_labels):
            if "verdict" not in r:
                continue
            v = index[str(r["seed"])]
            if v["set"] == "ae":
                ae.append(v | {"verdict": r["verdict"]})
            else:
                strict[v["id"]] = r["verdict"]
    v3 = json.loads((HERE / "training_set_v3.json").read_text())
    held = set(v3["held_out_pairs"])
    tiles, evals = [], []

    def add(**t):
        if t["id"] in DROP or t["pair"] in held or is_scene(t["pair"]) or (t["pair"] == "a_cat__x__a_dog" and 9 <= t["seed"] <= 16):
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
    for r in jsonl(RES / "api-labels/judge.jsonl"):
        if "verdict" not in r:
            continue
        jid = "j" + r["id"].split("__seed")[1].split("__")[0]
        k, real = key[jid], r["verdict"] == "faithful pair"
        tid = f"judge-{jid}"
        if k["pair"] in held and not is_scene(k["pair"]):
            evals.append({"id": tid, "picture": stage(JUDGE / "images" / f"{jid}.jpg", tid),
                          "latent": f"{V4}/latents/{tid}.pt", "pair": k["pair"], "seed": int(k["seed"]),
                          "method": k["setting"], "label": int(real), "kind": "positive" if real else r["verdict"],
                          "source": "judge", "check": True})
            continue
        if is_scene(k["pair"]):
            continue
        add(id=tid, picture=stage(JUDGE / "images" / f"{jid}.jpg", tid), latent=f"{V4}/latents/{tid}.pt",
            pair=k["pair"], seed=int(k["seed"]), method=k["setting"], label=int(real),
            kind="positive" if real else r["verdict"], source="judge")

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
        if not t["check"] and not is_scene(t["pair"]):
            tiles.append(t | {"source": "v3"})

    ae_verdicts = collections.defaultdict(set)
    for v in ae:
        ae_verdicts[v["id"]].add(v["verdict"] == "faithful pair")
    seen_ae = set()
    for v in ae:
        # The same file name sits in two paper-review folders: keep one, drop it if its two labels disagree.
        if v["id"] in seen_ae or len(ae_verdicts[v["id"]]) > 1:
            continue
        seen_ae.add(v["id"])
        real = v["verdict"] == "faithful pair"
        add(id=v["id"], picture=stage(REPO / v["path"], v["id"]), latent=f"{V4}/latents/{v['id']}.pt", pair=v["pair"],
            seed=v["seed"], method=v["method"], label=int(real), kind="positive" if real else v["verdict"], source="ae")

    demoted = collections.Counter()
    for t in tiles:
        if t["label"] == 1 and strict.get(t["id"], "faithful pair") != "faithful pair":
            t.update(label=0, kind=f"{strict[t['id']]} (stature rule)")
            demoted[t["source"]] += 1
    if strict:
        print("clean pairs the stature rule moved to the fake side:", dict(demoted))

    # More joint-prompt drawings (render_more.py), labelled under the stricter rule: cat x dog for
    # training, the 7 held-out pairs for evaluation. Latents are the sampler's own finished latents.
    MORE = "outputs/classifier_guidance_ladder/rung2/more"
    for name, use in (("more_catdog", "training"), ("more_heldout", "evaluation")):
        for r in jsonl(RES / f"api-labels/{name}.jsonl"):
            if "verdict" not in r:
                continue
            real = r["verdict"] == "faithful pair"
            t = dict(id=f"{name}-{r['id']}", latent=f"{MORE}/{name}/final/{r['id']}.pt", pair=r["pair"], seed=r["seed"],
                     method="joint", label=int(real), kind="positive" if real else r["verdict"], source=name,
                     finals=f"{MORE}/{name}/finals/{r['id']}.jpg")
            if use == "training":
                add(**t)
            else:
                evals.append(t | {"check": True, "strict_label": int(real)})

    evals += [t | {"source": "v3"} for t in v3["tiles"] if t["check"]]
    for t in evals:
        t["strict_label"] = int(strict.get(t["id"], "faithful pair" if t["label"] else "no") == "faithful pair") \
            if t["label"] == 1 else 0
    import hashlib
    def md5(t):
        if "picture" in t and (args.stage / Path(t["picture"]).name).exists():
            return hashlib.md5((args.stage / Path(t["picture"]).name).read_bytes()).hexdigest()
        return t["id"] if t.get("source") == "v3" else t["latent"]
    shared = {md5(t) for t in tiles} & {md5(t) for t in evals}
    if shared or {t["pair"] for t in tiles} & held:
        raise SystemExit(f"guard: evaluation and training overlap ({len(shared)} pictures)")
    (HERE / "eval_set_v4.json").write_text(json.dumps({"rule": "v3 hand-called check tiles plus the judge's "
        "API-labelled pictures of the 7 held-out pairs; md5-disjoint from training_set_v4.json",
        "held_out_pairs": sorted(held), "tiles": evals}, indent=0))
    ce = collections.Counter((t["source"], t["label"], t["pair"] == "a_cat__x__a_dog") for t in evals)
    print("evaluation clean / reject by source:", dict(collections.Counter((t["source"], t["label"]) for t in evals)))
    print("evaluation:", {f"{s}/{'real' if l else 'fake'}/{'catdog' if c else 'heldout'}": n for (s, l, c), n in sorted(ce.items())})
    out = {"rule": __doc__.split("\n\n")[1], "held_out_pairs": sorted(held), "tiles": tiles}
    (HERE / "training_set_v4.json").write_text(json.dumps(out, indent=0))
    c = collections.Counter((t["source"], t["label"]) for t in tiles)
    for k in sorted(c):
        print(f"{k[0]:7s} {'real' if k[1] else 'fake'} {c[k]}")
    print(f"total {len(tiles)}; pairs with a real example {len({t['pair'] for t in tiles if t['label'] == 1})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
