#!/usr/bin/env python
"""Cat x dog only: the training and evaluation lists for a checker that answers rung 2's own question.

Rung 2 asks whether a push fixes cat x dog. The v4 checkers were held to 0.75 on 7 pairs they never saw,
which is the later, general claim. This keeps only cat x dog tiles and holds out by seed.

The evaluation list is fixed so every training list is scored on the same pictures:
    eval_set_catdog_v2.json   v3's hand-called cat x dog check tiles (seeds 7 and 8), joint drawings on seeds
                              201 to 250 and 601 to 750 (render_more.py, labelled under the stricter rule):
                              about 109 clean pairs, so a balanced accuracy reads to about +-0.05
    (eval_set_catdog.json, the first 70 of those pictures, is kept unchanged for the earlier runs)

A training list is the base plus the sources named with --sources, so each loop iteration adds one:
    base         every cat x dog tile of training_set_v4.json except joint drawings on seeds 201 to 250
    coco         COCO 2017 photos (coco_catdog.py): a one-cat-one-dog candidate is clean when the Claude API
                 passes it under the strict and real-only rules; other candidates, two-or-more of one
                 animal, and single animals are rejects
    more2        joint drawings on seeds 251 to 600 (render_more.py, more_catdog2.json), strict rule
    openimages   Open Images V7 photos (openimages_catdog.py), the same rules as coco
    laion        Re-LAION-2B photos whose caption names a cat and a dog (laion_catdog.py): clean when the API
                 passes them; rejects only where the API saw at least one real cat or dog, so pet products
                 with no animal are left out
Every source sits on both sides. Seeds 9 to 16 stay out of both lists: they are the push test's.
Writes training_set_catdog_<tag>.json and eval_set_catdog_v2.json beside this script.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAIR = "a_cat__x__a_dog"
HOLD = range(201, 251)
EVAL_SEEDS = range(601, 751)
TRAIN_SEEDS = range(251, 601)
ROOT = HERE.parents[2]
RESULTS = ROOT / "2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog"
RUNG2 = "outputs/classifier_guidance_ladder/rung2"
MORE2 = f"{RUNG2}/more/more_catdog2"


def jsonl(name):
    return [json.loads(l) for l in (RESULTS / "api-labels" / name).read_text().splitlines() if l.strip()]


def tile(tid, picture, source, real, kind, seed=-1, latent=None):
    return {"id": tid, "picture": picture, "latent": latent or f"{RUNG2}/{source}-catdog/latents/{tid}.pt",
            "pair": PAIR, "seed": seed, "method": source, "label": int(real),
            "kind": "positive" if real else kind, "source": source, "check": False}


def photo_source(name, folder, labels, stage_index=None):
    """COCO or Open Images: clean candidates by API verdict, the rest rejects by box count."""
    data = ROOT / f"data/{folder}/catdog"
    index = json.loads((data / "stage-index.json").read_text()) if stage_index else None
    verdict = {}
    for r in jsonl(labels):
        seed = r["id"].split("__seed")[1].split("__")[0]
        verdict[index[seed] if index else seed] = r.get("verdict")
    out = []
    for kind in ("clean", "wrong_pair", "single"):
        for f in sorted((data / kind).glob("*.jpg")):
            if kind == "clean" and f.stem not in verdict:
                continue
            real = kind == "clean" and verdict[f.stem] == "faithful pair"
            out.append(tile(f"{name}-{f.stem}", f"{RUNG2}/{name}-catdog/{kind}/{f.name}", name, real,
                            verdict[f.stem] if kind == "clean" else kind.replace("_", " ")))
    return out


def drawings(rows, seeds):
    out = []
    for r in rows:
        if "verdict" not in r or r["seed"] not in seeds:
            continue
        real = r["verdict"] == "faithful pair"
        out.append(tile(f"more2-{r['id']}", None, "more2", real, r["verdict"], seed=r["seed"],
                        latent=f"{MORE2}/final/{r['id']}.pt") | {"finals": f"{MORE2}/finals/{r['id']}.jpg"})
    for t in out:
        del t["picture"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", nargs="*", default=[], choices=["coco", "more2", "openimages", "laion"])
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    train_v4 = json.loads((HERE / "training_set_v4.json").read_text())
    eval_v4 = json.loads((HERE / "eval_set_v4.json").read_text())
    cd = [t for t in train_v4["tiles"] if t["pair"] == PAIR]
    held = [t for t in cd if t["source"] == "more_catdog" and t["seed"] in HOLD]
    train = [t for t in cd if t not in held]
    more2 = jsonl("more_catdog2.jsonl")
    if "coco" in args.sources:
        train += photo_source("coco", "coco-2017", "coco_clean_real.jsonl")
    if "more2" in args.sources:
        train += drawings(more2, TRAIN_SEEDS)
    if "openimages" in args.sources:
        train += photo_source("openimages", "open-images-v7", "openimages_clean.jsonl", stage_index=True)
    if "laion" in args.sources:
        index = json.loads((ROOT / "data/relaion-catdog/stage-index.json").read_text())
        for r in jsonl("laion.jsonl"):
            if "verdict" not in r:
                continue
            real = r["verdict"] == "faithful pair"
            if not real and not (r.get("a_present") or r.get("b_present")):
                continue
            key = index[str(r["seed"])]
            train.append(tile(f"laion-{key}", f"{RUNG2}/laion-catdog/{key}.jpg", "laion", real, r["verdict"]))
    evals = [t for t in eval_v4["tiles"] if t["pair"] == PAIR] + \
            [t | {"check": True, "strict_label": t["label"]} for t in held] + \
            [t | {"check": True, "strict_label": t["label"]} for t in drawings(more2, EVAL_SEEDS)]
    if any(9 <= t.get("seed", -1) <= 16 for t in train + evals):
        raise SystemExit("guard: seeds 9 to 16 are the push test's")
    if {t["id"] for t in train} & {t["id"] for t in evals}:
        raise SystemExit("guard: a tile is in both lists")
    rule = __doc__.split("\n\n")[1]
    (HERE / f"training_set_catdog_{args.tag}.json").write_text(
        json.dumps({"rule": rule, "sources": args.sources, "held_out_pairs": [], "tiles": train}, indent=0))
    (HERE / "eval_set_catdog_v2.json").write_text(json.dumps({"rule": rule, "held_out_pairs": [], "tiles": evals}, indent=0))
    for name, tiles in (("training", train), ("evaluation", evals)):
        c = collections.Counter((t["source"], "clean" if t["label"] else "reject") for t in tiles)
        print(name, len(tiles), dict(sorted(c.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
