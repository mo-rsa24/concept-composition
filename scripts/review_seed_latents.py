"""Save the starting latent x_T of seeds 9 to 16 that wave 1 used, as plain tensors, for baselines run outside this package.

Protocol rule 2: every method on SDXL starts a seed from the same x_T. Wave 1's joint, PoE and adapter tiles read it
through poe_repair._sdxl.runtime.load_shared_latents on a_cat__x__a_dog's cache. The cache holds no entry for seeds
9 to 16 (no run trained on them), so that function draws from a CPU generator seeded with the seed number, in fp16.
This script calls the same function with the same pair, so the saved tensor is the one wave 1 started from, and
prints whether it came from the cache or from that draw. ToMe runs in its own environment without this package, so
it reads these files instead.

    python scripts/review_seed_latents.py            # writes <OUT>/seedNN.pt, one [1, 4, 128, 128] fp16 tensor each
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(os.environ.get("POE_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))

from poe_repair._sdxl.runtime import load_shared_latents  # noqa: E402
from poe_repair.experiments.interaction_term.cell import cell_from_slug  # noqa: E402

OUT = Path(os.environ.get("REVIEW_LATENTS_OUT", "/datasets/mmolefe/poe_repair_min/outputs/paper_review/latents"))
NOISE_PAIR = "a_cat__x__a_dog"
SEEDS = range(9, 17)
# Only in_channels is read, and only when the cache has no entry for the seed.
STAND_IN = {"unet": SimpleNamespace(config=SimpleNamespace(in_channels=4))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        cell = cell_from_slug(NOISE_PAIR, seed)
        cached = (cell.pair_dir / "grid_assets.json").exists()
        x, _ = load_shared_latents(pair_dir=cell.pair_dir, seed=seed, device=torch.device("cpu"),
                                   dtype=torch.float16, models=STAND_IN, height=1024, width=1024)
        path = OUT / f"seed{seed:02d}.pt"
        torch.save(x.cpu(), path)
        digest = hashlib.md5(x.cpu().numpy().tobytes()).hexdigest()[:12]
        print(f"seed {seed}: {'cache' if cached else 'seeded CPU draw'} {tuple(x.shape)} {x.dtype} "
              f"std {x.float().std():.4f} md5 {digest} -> {path}", flush=True)


if __name__ == "__main__":
    main()
