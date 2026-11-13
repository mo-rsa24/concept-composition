"""PoE + Langevin on a named list of cells outside protocol rule 6's subset, for the appendix's best- and
worst-cases grids (1-lora-correction/report/paper-review/protocol.md, rule 5).

The same call as scripts/review_langevin_subset.py: c = 3, k = 20 over all 50 levels, from the starting latent
every other method used for that seed (rule 2). Tiles go to <OUT>/langevin_extra/<pair>/seedNN/poe_langevin.png,
apart from the subset, so the subset's rows in R1 and R3 stay exactly what rule 6 says. With
LANGEVIN_TILES=langevin_subset it fills the subset's own missing cells instead, skipping any that exist.

    python 1-lora-correction/scripts/review_langevin_cells.py --cells <file of 'concept a|concept b|seed' lines>
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("POE_REPO", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "showcase"))
OUT = Path(os.environ.get("REVIEW_BASELINE_OUT", "/datasets/mmolefe/poe_repair_min/outputs/paper_review"))
TILES = OUT / os.environ.get("LANGEVIN_TILES", "langevin_extra")   # "langevin_subset" for the subset's own cells
NOISE_PAIR = "a_cat__x__a_dog"
C, K = 3.0, 20


def make_dir(d: Path, tries: int = 10) -> None:
    for i in range(tries):
        try:
            d.mkdir(parents=True, exist_ok=True)
            return
        except (FileExistsError, FileNotFoundError) as e:
            print(f"{d}: {type(e).__name__} on try {i + 1}, waiting 15 s", flush=True)
            time.sleep(15)
            os.listdir(d.parent.parent)
    d.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    args = ap.parse_args()
    import torch
    from poe_repair.composers._helpers import encode_pair, init_latents_for_cell
    from poe_repair.experiments._eval_common import cell_for
    from poe_repair.experiments.interaction_term.cell import cell_from_slug
    from poe_repair.methods._poe_langevin import run_poe_langevin
    from poe_repair.methods._sampling import write_decoded_image
    from poe_repair.run import make_ctx

    mine = [(a, b, int(s)) for a, b, s in (l.strip().split("|") for l in open(args.cells) if l.strip())]
    print(f"{len(mine)} cells", flush=True)
    ctx = make_ctx()
    n = ctx.num_inference_steps
    for a, b, seed in mine:
        cell = cell_for(a, b, seed)
        out = TILES / cell.pair_slug / f"seed{seed:02d}" / "poe_langevin.png"
        if out.exists():
            print(f"{out}: already rendered", flush=True)
            continue
        init, sigma = init_latents_for_cell(cell_from_slug(NOISE_PAIR, seed), ctx)
        emb = encode_pair(cell, ctx)
        t0 = time.time()
        res = run_poe_langevin(seq_a=emb["seq_a"], pool_a=emb["pool_a"], seq_b=emb["seq_b"],
                               pool_b=emb["pool_b"], seq_e=emb["seq_e"], pool_e=emb["pool_e"],
                               k=K, c=C, noise_seed=seed, init_latents=init, models=ctx.models,
                               scheduler=ctx.scheduler, guidance_scale=ctx.guidance_scale,
                               num_inference_steps=n, height=cell.height, width=cell.width,
                               euler_init_noise_sigma=sigma, device=ctx.device, dtype=ctx.dtype)
        make_dir(out.parent)
        write_decoded_image(res.image, out)
        out.with_suffix(".json").write_text(json.dumps({
            "model": "SDXL", "method": "poe_langevin", "adapter": None, "sampler": "DDIM", "steps": n,
            "guidance": ctx.guidance_scale, "resolution": cell.width, "pair": cell.pair_slug,
            "prompt_a": a, "prompt_b": b, "seed": seed, "group": os.environ.get("LANGEVIN_GROUP", "best-worst grids"), "setting": f"c={C:g},k={K}",
            "nfe": None, "unet_calls_per_branch_set": n * (K + 1), "noise_from": f"{NOISE_PAIR} seed {seed} (protocol rule 2)",
            "corrector_noise_seed": seed, "seconds": round(time.time() - t0, 1)}, indent=2))
        del res
        gc.collect()
        torch.cuda.empty_cache()
        print(f"wrote {out}  ({time.time() - t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
