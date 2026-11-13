#!/usr/bin/env python
"""Keep only the generated photos that hold what their prompt asked for.

SD 3.5 medium is asked for "a photo of a cat and a dog" and often draws one animal, or one animal
wearing the other's face. Those are the exact failures this project studies, so letting them into
the training pool would teach the cat condition that a cat-and-dog scene is a single hybrid
animal, which is the opposite of what the pilot is testing.

The screen is this repo's existing instrument, unchanged: GroundingDINO counts distinct animal
instances, and the CLIP crop classifier names each one, because the detector can count animals
and cannot tell a cat from a dog (64 of 64 false dog calls on single-cat renders).

  both   kept when exactly two instances are found and one is called cat and the other dog
  cat    kept when exactly one instance is found and it is called cat
  dog    kept when exactly one instance is found and it is called dog

An uncertain call from the classifier is a rejection, not a guess. Writes screened.json beside
the images: one row per image with its counts, calls and verdict, so a keep can be checked.

    ~/envs-sd35/bin/python scripts/census/inclusive_captions_screen_generated.py \
      --kind both --dir /datasets/.../inclusive_captions_generated/both
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

from poe_repair.experiments.compose_scorer_validation.detection_scorer import (  # noqa: E402
    count_instances,
)
from scripts.census.crop_classifier_cat_vs_dog import ZeroShot, crop  # noqa: E402

WANTED = {"both": ("cat", "dog"), "cat": ("cat",), "dog": ("dog",)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=sorted(WANTED), required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    directory = Path(args.dir)
    images = sorted(directory.glob("*.png"))
    print(f"{len(images)} images to screen in {directory}", flush=True)

    device = torch.device(args.device)
    namer = ZeroShot(device)
    wanted = sorted(WANTED[args.kind])

    rows, kept = [], 0
    for i, path in enumerate(images):
        n, boxes = count_instances(path, query="animal", device=device)
        calls = []
        if n == len(wanted):
            with Image.open(path) as im:
                rgb = im.convert("RGB")
                calls = [namer(crop(rgb, b["box"]))["call"] for b in boxes]
        verdict = "keep" if sorted(calls) == wanted else "reject"
        kept += verdict == "keep"
        rows.append({"file": str(path), "n_instances": n, "calls": calls, "verdict": verdict})
        if (i + 1) % 25 == 0:
            print(f"{i + 1} screened, {kept} kept", flush=True)

    (directory / "screened.json").write_text(json.dumps(
        {"kind": args.kind, "wanted": wanted, "n": len(rows), "kept": kept, "rows": rows},
        indent=2))
    print(f"done: {kept} of {len(rows)} kept, written to {directory / 'screened.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
