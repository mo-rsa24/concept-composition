"""PoE + Langevin on protocol rule 6's labelled subset, for R1 and R3 (1-lora-correction/report/paper-review/protocol.md).

The subset is the benchmark's 66 animals prompts at seeds 9 to 12 (264 cells, group "animals") and the
held-out hard group's three pairs at seeds 9 to 16 (24 cells, group "hard"): 288 cells. Each renders
PoE + Langevin at c = 3 and k = 20 over all 50 levels, the call 1-lora-correction/scripts/review_r6_renders.py makes for
Figure 5, from the starting latent every other method used for that seed (protocol rule 2).

Cells are split by index into --shards parts and this process renders part --shard, so several cards
can share the subset without two of them ever writing the same cell. A tile that exists is skipped.

    python scripts/review_langevin_subset.py --shard 0 --shards 2

Tiles land as <OUT>/langevin_subset/<pair>/seedNN/poe_langevin.png with a JSON sidecar carrying
protocol rule 9's columns.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("POE_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "showcase"))
HERE = Path(__file__).resolve().parent

OUT = Path(os.environ.get("REVIEW_BASELINE_OUT", "/datasets/mmolefe/poe_repair_min/outputs/paper_review"))
TILES = OUT / "langevin_subset"
NOISE_PAIR = "a_cat__x__a_dog"
C, K = 3.0, 20
HARD = [("a cow", "a horse"), ("a seal", "a walrus"), ("a suitcase", "a desk fan")]


def make_dir(d: Path, tries: int = 10) -> None:
    """mkdir that survives two cards creating one folder at once over NFS.

    A node's cached view can say "File exists" and "No such file" for the same folder for a while;
    waiting and re-reading the parent refreshes it.
    """
    for i in range(tries):
        try:
            d.mkdir(parents=True, exist_ok=True)
            return
        except (FileExistsError, FileNotFoundError) as e:
            print(f"{d}: {type(e).__name__} on try {i + 1}, waiting 15 s", flush=True)
            time.sleep(15)
            os.listdir(d.parent.parent)
    d.mkdir(parents=True, exist_ok=True)


def cells() -> list[tuple[str, str, int, str]]:
    rows = [l.strip().split("|") for l in open(HERE / "review_benchmark_pairs.txt")
            if l.strip() and not l.startswith("#")]
    animals = [(a, b) for a, b, _, group in rows if group == "animals"]
    assert len(animals) == 66, len(animals)
    out = [(a, b, s, "animals") for a, b in animals for s in (9, 10, 11, 12)]
    out += [(a, b, s, "hard") for a, b in HARD for s in range(9, 17)]
    assert len(out) == 288, len(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--shards", type=int, required=True)
    args = ap.parse_args()

    import torch
    from disallowed_subjects import check
    from poe_repair.composers._helpers import encode_pair, init_latents_for_cell
    from poe_repair.experiments._eval_common import cell_for
    from poe_repair.experiments.interaction_term.cell import cell_from_slug
    from poe_repair.methods._poe_langevin import run_poe_langevin
    from poe_repair.methods._sampling import write_decoded_image
    from poe_repair.run import make_ctx

    mine = [c for i, c in enumerate(cells()) if i % args.shards == args.shard]
    for a, b, _, _ in mine:          # frog and mouse run under the pair ledger's 2026-11-10 decision
        check(a, allow_ask_first=True)
        check(b, allow_ask_first=True)
    print(f"shard {args.shard} of {args.shards}: {len(mine)} cells", flush=True)

    ctx = make_ctx()
    n = ctx.num_inference_steps
    for a, b, seed, group in mine:
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
            "prompt_a": a, "prompt_b": b, "seed": seed, "group": group, "setting": f"c={C:g},k={K}",
            "nfe": None, "unet_calls_per_branch_set": n * (K + 1), "noise_from": f"{NOISE_PAIR} seed {seed} (protocol rule 2)",
            "corrector_noise_seed": seed, "seconds": round(time.time() - t0, 1),
        }, indent=2))
        del res
        gc.collect()
        torch.cuda.empty_cache()
        print(f"wrote {out}  ({time.time() - t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
