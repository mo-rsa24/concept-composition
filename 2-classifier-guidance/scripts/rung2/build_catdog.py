#!/usr/bin/env python
"""Cat x dog only: the training and evaluation lists for a checker that answers rung 2's own question.

Rung 2 asks whether a push fixes cat x dog. The v4 checkers were held to 0.75 on 7 pairs they never saw,
which is the later, general claim. This keeps only cat x dog tiles from training_set_v4.json and
eval_set_v4.json and holds out the newest joint drawings by seed:

    training     every cat x dog tile of training_set_v4.json except joint drawings on seeds 201 to 250
    evaluation   v3's hand-called cat x dog check tiles (seeds 7 and 8) plus the joint drawings on seeds
                 201 to 250 (render_more.py, labelled under the stricter rule)

COCO 2017 photos join training only, as source "coco" on both sides (coco_catdog.py picks them by
COCO's instance counts): a candidate with exactly one cat and one dog is clean when the Claude API passes it
under the strict and real-only rules (an animal on a screen, poster or van is not present); every other
candidate, every photo of two or more cats or two or more dogs, and every photo of one animal is a reject.

Seeds 9 to 16 stay out of both: they are the push test's.
Writes training_set_catdog.json and eval_set_catdog.json beside this script.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAIR = "a_cat__x__a_dog"
HOLD = range(201, 251)
ROOT = HERE.parents[2]
RESULTS = ROOT / "2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog"
COCO_MOUNT = "outputs/classifier_guidance_ladder/rung2/coco-catdog"


def main() -> int:
    train_v4 = json.loads((HERE / "training_set_v4.json").read_text())
    eval_v4 = json.loads((HERE / "eval_set_v4.json").read_text())
    cd = [t for t in train_v4["tiles"] if t["pair"] == PAIR]
    held = [t for t in cd if t["source"] == "more_catdog" and t["seed"] in HOLD]
    train = [t for t in cd if t not in held]
    coco = ROOT / "data/coco-2017/catdog"
    verdict = {r["id"].split("__seed")[1].split("__")[0]: r.get("verdict") for r in
               map(json.loads, (RESULTS / "api-labels/coco_clean_real.jsonl").read_text().splitlines()) if r}
    for kind in ("clean", "wrong_pair", "single"):
        for f in sorted((coco / kind).glob("*.jpg")):
            real = kind == "clean" and verdict.get(f.stem) == "faithful pair"
            if kind == "clean" and f.stem not in verdict:
                continue
            tid = f"coco-{f.stem}"
            train.append({"id": tid, "picture": f"{COCO_MOUNT}/{kind}/{f.name}", "latent": f"{COCO_MOUNT}/latents/{tid}.pt",
                          "pair": PAIR, "seed": -1, "method": "coco", "label": int(real),
                          "kind": "positive" if real else (verdict[f.stem] if kind == "clean" else kind.replace("_", " ")),
                          "source": "coco", "check": False})
    evals = [t for t in eval_v4["tiles"] if t["pair"] == PAIR] + \
            [t | {"check": True, "strict_label": t["label"]} for t in held]
    if any(9 <= t.get("seed", -1) <= 16 for t in train + evals):
        raise SystemExit("guard: seeds 9 to 16 are the push test's")
    if {t["id"] for t in train} & {t["id"] for t in evals}:
        raise SystemExit("guard: a tile is in both lists")
    rule = __doc__.split("\n\n")[1]
    (HERE / "training_set_catdog.json").write_text(json.dumps({"rule": rule, "held_out_pairs": [], "tiles": train}, indent=0))
    (HERE / "eval_set_catdog.json").write_text(json.dumps({"rule": rule, "held_out_pairs": [], "tiles": evals}, indent=0))
    for name, tiles in (("training", train), ("evaluation", evals)):
        c = collections.Counter((t["source"], "clean" if t["label"] else "reject") for t in tiles)
        print(name, len(tiles), dict(sorted(c.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
