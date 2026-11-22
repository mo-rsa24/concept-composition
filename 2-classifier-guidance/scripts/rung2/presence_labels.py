#!/usr/bin/env python
"""Per-picture "a cat present?" and "a dog present?" answers, for the identity push's two extra outputs.

The pair checker's push mostly makes two cats or two dogs: "clean pair?" says nothing about which animals.
Two extra outputs, trained on whether a clear cat and a clear dog are present, give the push a direction
toward the missing animal. The answers already exist in the Claude API labels (each row's a_present and
b_present: a body that is clearly and only a cat, or a dog):
    kim          api-labels/drawings.jsonl            more2   api-labels/more_catdog2.jsonl
    more3        api-labels/more_catdog3.jsonl        more_catdog   api-labels/more_catdog.jsonl
    photos both  api-labels/both-photos-cropped.jsonl photos cat_only / dog_only   by folder (cat, dog)
    v3           api-labels/validate.jsonl (the API run against the hand calls)
A clean pair is (1, 1) by definition. A picture with no answer gets -1 for both and is left out of the two
presence outputs' loss (it still trains "clean pair?").

Writes presence_catdog.json beside this script: {tile id: [cat, dog]} for every tile of
training_set_catdog_more23.json and eval_set_catdog_v2.json.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
A = Path("/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/by-run") / "api-labels"


def rows(name):
    p = A / name
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def main() -> int:
    by = {}
    for r in rows("drawings.jsonl"):
        by[f"kim-{r['id']}"] = r
    for name, prefix in (("more_catdog2.jsonl", "more2-"), ("more_catdog3.jsonl", "more3-"),
                         ("more_catdog.jsonl", "more_catdog-")):
        for r in rows(name):
            by[prefix + r["id"]] = r
    crop = {r["id"]: r for r in rows("both-photos-cropped.jsonl")}
    for r in rows("validate.jsonl"):
        by[r["id"]] = r
    tiles = json.loads((HERE / "training_set_catdog_more23.json").read_text())["tiles"] + \
        json.loads((HERE / "eval_set_catdog_v2.json").read_text())["tiles"]
    out, where = {}, collections.Counter()
    for t in tiles:
        tid = t["id"]
        r = by.get(tid)
        if r is None and tid.startswith("photo-both-"):
            stem = tid[len("photo-both-"):]
            i, src = int(stem.split("_")[0]), stem.split("_")[1]
            r = crop.get(f"a_cat__x__a_dog__seed{(1000 if src == 'coco' else 2000) + i}__photo")
        if t["label"] == 1:
            out[tid] = [1, 1]
            where["clean pair"] += 1
        elif tid.startswith("photo-cat_only-"):
            out[tid] = [1, 0]
            where["cat-only photo"] += 1
        elif tid.startswith("photo-dog_only-"):
            out[tid] = [0, 1]
            where["dog-only photo"] += 1
        elif r is not None and "a_present" in r:
            out[tid] = [int(bool(r["a_present"])), int(bool(r["b_present"]))]
            where["API answer"] += 1
        else:
            out[tid] = [-1, -1]
            where["no answer"] += 1
    (HERE / "presence_catdog.json").write_text(json.dumps(out))
    known = [v for v in out.values() if v[0] >= 0]
    print(dict(where))
    print("among answered pictures: cat present", sum(v[0] for v in known), "of", len(known),
          "; dog present", sum(v[1] for v in known), "of", len(known),
          "; two-of-a-kind or single (one of the two missing)", sum(1 for v in known if v[0] != v[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
