#!/usr/bin/env python
"""The multi-pair training list: the cat x dog list plus every labelled picture of a pool, as new sources.

The readout checker learnt "two separate things" from cat x dog alone and carried it to typewriter x cactus (14 of 16)
but not to "which body is which" on lion x meerkat (four seeds of two meerkats). This adds labelled pictures of many
pairs so it can learn one of each kind across pairs. Each pool picture becomes a tile (id "<pool>-<picture id>", its
pair, seed, label and verdict as kind, picture path under MOUNT_ROOT); its source is "<pool>-<method>", so the trainer,
which samples sources evenly, gives every method a share of each batch. The test pairs are written as held out, and
the trainer refuses any held-out pair or cat x dog seed 9 to 16 in the list.

Writes training_set_multipair.json beside this script.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEST_PAIRS = ["a_lion__x__a_meerkat", "a_typewriter__x__a_cactus"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", action="append", required=True,
                    help="name=path: a pool folder name under rung2/ and its local labelled.json")
    ap.add_argument("--out", type=Path, default=HERE / "training_set_multipair.json")
    args = ap.parse_args()
    base = json.loads((HERE / "training_set_catdog_more23.json").read_text())
    tiles = list(base["tiles"])
    added = collections.Counter()
    for spec in args.pool:
        name, path = spec.split("=", 1)
        labelled = json.loads(Path(path).read_text())
        for pid, r in labelled.items():
            if r["pair"] in TEST_PAIRS:
                continue
            if r["pair"] == "a_cat__x__a_dog" and 9 <= r["seed"] <= 16:
                continue
            tiles.append({"id": f"{name}-{pid}", "pair": r["pair"], "seed": r["seed"], "method": r["method"],
                          "label": r["label"], "kind": "positive" if r["label"] else r["verdict"],
                          "source": f"{name}-{r['method']}", "check": False,
                          "picture": f"outputs/classifier_guidance_ladder/rung2/{name}/pictures/{pid}.jpg"})
            added[(f"{name}-{r['method']}", r["label"])] += 1
    out = {"rule": base.get("rule", "") + " Plus labelled pool pictures: strict API faithful pair clean; hybrid, single, "
                   "wrong pair reject; damaged left out.",
           "sources": sorted({t["source"] for t in tiles}),
           "held_out_pairs": sorted(set(base["held_out_pairs"]) | set(TEST_PAIRS)), "tiles": tiles}
    args.out.write_text(json.dumps(out, indent=0))
    for (src, lab), n in sorted(added.items()):
        print(f"  {src:22s} {'clean' if lab else 'reject'}: {n}")
    print(f"{len(tiles)} tiles ({len(base['tiles'])} cat x dog, {sum(added.values())} added); "
          f"{len({t['pair'] for t in tiles})} pairs; held out {out['held_out_pairs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
