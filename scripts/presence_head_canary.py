#!/usr/bin/env python
"""Does the classifier-guided composer leave plain product-of-experts exactly where it was?

Every column of this experiment is read against plain product. If the new sampler changes the
baseline even slightly, the comparison measures the plumbing instead of the method, and a
float32-against-float16 difference of the same combine already costs about three grey levels over
50 steps here, so "close enough" is not a pass.

Renders the same cell three ways and compares the bytes:
    plain          the existing poe composer
    weight 0       the new composer with no head at all
    weight 0, head the new composer carrying an untrained head, which is what the sweep passes

Writes, under MOUNT_ROOT/outputs/presence_head/canary/:
    canary.json   the three checksums, the largest absolute pixel difference, and the verdict
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poe_repair import paths  # noqa: E402
from poe_repair.composers import poe as poe_composer  # noqa: E402
from poe_repair.composers import poe_classifier_guided  # noqa: E402
from poe_repair.config import RunConfig  # noqa: E402
from poe_repair.experiments.presence_head.head import PresenceHead  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import PairSeedCell  # noqa: E402

PAIR_SLUG = "a_cat__x__a_dog"
PROMPT_A, PROMPT_B = "a cat", "a dog"
FAMILY = "outputs/presence_head/canary"


def digest(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def largest_pixel_difference(a: Path, b: Path) -> dict:
    with Image.open(a) as ia, Image.open(b) as ib:
        xa = np.asarray(ia.convert("RGB"), dtype=np.int16)
        xb = np.asarray(ib.convert("RGB"), dtype=np.int16)
    d = np.abs(xa - xb)
    return {"max_abs_diff_0_255": int(d.max()),
            "fraction_of_pixels_differing": round(float((d > 0).mean()), 6)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=9)
    args = ap.parse_args()

    out_root = paths.MOUNT_ROOT / FAMILY
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = RunConfig()
    ctx = make_ctx(output_root=out_root)
    print(f"node {socket.gethostname()}  device {ctx.device}  dtype {ctx.dtype}  "
          f"steps {ctx.num_inference_steps}  guidance {ctx.guidance_scale}", flush=True)

    cell = PairSeedCell(
        pair_dir=cfg.paths.pilot_dir / f"seed_{args.seed}" / PAIR_SLUG,
        pair_slug=PAIR_SLUG, prompt_a=PROMPT_A, prompt_b=PROMPT_B,
        seed=args.seed, regime="collision", height=1024, width=1024, grid_assets={},
    )

    print("rendering plain product...", flush=True)
    plain = poe_composer.run(cell, ctx, exp_name="presence_head/canary", overwrite=True)

    print("rendering the new composer at weight 0, no head...", flush=True)
    zero_no_head = poe_classifier_guided.run(
        cell, ctx, head=None, weight=0.0,
        exp_name="presence_head/canary_zero_no_head", overwrite=True)

    print("rendering the new composer at weight 0 carrying an untrained head...", flush=True)
    head = PresenceHead().to(ctx.device).eval()
    zero_with_head = poe_classifier_guided.run(
        cell, ctx, head=head, weight=0.0,
        exp_name="presence_head/canary_zero_with_head", overwrite=True)

    checks = {
        "weight 0, no head": largest_pixel_difference(plain, zero_no_head),
        "weight 0, untrained head": largest_pixel_difference(plain, zero_with_head),
    }
    identical = all(c["max_abs_diff_0_255"] == 0 for c in checks.values())
    result = {
        "what_this_measures": "whether the classifier-guided composer reproduces plain "
                              "product-of-experts exactly when the classifier weight is zero",
        "pair": PAIR_SLUG, "seed": args.seed,
        "prompts": [PROMPT_A, PROMPT_B],
        "guidance_scale": ctx.guidance_scale,
        "num_inference_steps": ctx.num_inference_steps,
        "checksums": {"plain": digest(plain),
                      "weight 0, no head": digest(zero_no_head),
                      "weight 0, untrained head": digest(zero_with_head)},
        "pixel_checks": checks,
        "identical": identical,
        "verdict": ("the composer leaves the baseline byte-identical at weight 0, so every later "
                    "column is read against an unmoved reference"
                    if identical else
                    "the composer changed the baseline at weight 0; the sweep must not run until "
                    "this is fixed, because every column would measure the plumbing"),
        "paths": {"plain": str(plain), "zero_no_head": str(zero_no_head),
                  "zero_with_head": str(zero_with_head)},
        "node": socket.gethostname(),
    }
    (out_root / "canary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
