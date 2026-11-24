#!/usr/bin/env python3
"""Step M5's overfit rung: pick 16 training pictures from one list and write them twice, as a training list and as a check list.

The 16 are 8 clean and 8 rejects, drawn with a fixed seed from tiles whose latent and three augmented copies exist. The trainer reads
the training list's tiles that are not marked `check`, and checks against the other file's tiles that are, so the checker is scored on
exactly the pictures it trained on. The rule is in artifacts/notes/progress/method-search/01b-balance-the-checker-data/design.md, the
ladder's overfit rung.

Usage: POE_REPAIR_MOUNT=<scratch root> python m5_overfit_tiles.py --list <training_set_LIST.json> --out <dir>
"""
import argparse
import json
import random
from pathlib import Path

from poe_repair import paths

RUNG2 = paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2"   # the trainer's own rule
PER_LABEL = 8


def has_latents(t, rung2):
    latent = paths.MOUNT_ROOT / t["latent"] if "/" in t["latent"] else rung2 / "training_set/latents" / f"{t['latent']}.pt"
    return latent.exists() and all((rung2 / "dataset-v4/aug" / f"{t['id']}__a{k}.pt").exists() for k in range(3))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--list", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    data = json.loads(args.list.read_text())
    pool = [t for t in data["tiles"] if not t.get("check") and has_latents(t, RUNG2)]
    rng = random.Random(0)
    picked = []
    for label in (1, 0):
        group = sorted((t for t in pool if t["label"] == label), key=lambda t: t["id"])
        if len(group) < PER_LABEL:
            raise SystemExit(f"only {len(group)} tiles with label {label} have latents")
        picked += rng.sample(group, PER_LABEL)
    args.out.mkdir(parents=True, exist_ok=True)
    train = {k: v for k, v in data.items() if k != "tiles"} | {"tiles": [dict(t, check=False) for t in picked]}
    check = {k: v for k, v in data.items() if k != "tiles"} | {"tiles": [dict(t, check=True) for t in picked]}
    (args.out / "overfit-train.json").write_text(json.dumps(train, indent=1))
    (args.out / "overfit-check.json").write_text(json.dumps(check, indent=1))
    print(json.dumps({"tiles": len(picked), "pairs": len({t["pair"] for t in picked}),
                      "ids": [t["id"] for t in picked]}))


if __name__ == "__main__":
    main()
