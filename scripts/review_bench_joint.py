"""The joint prompt on a benchmark's own prompts (slot R14, compute job 13), one tile per prompt and seed.

Each prompt is drawn exactly as the benchmark wrote it, with plain CFG, from the shared starting
latent of its seed (protocol rule 2: the latent `a_cat__x__a_dog` borrows at that seed), 50 DDIM steps,
guidance 7.5, 1024 by 1024, fp16. These are the calls of the joint column in
scripts/shared/concept_pair_sheet.py (`--column mono`), with the prompt text passed whole rather than built
from two concepts, so the counting, spatial and complex prompts, which have no pair, render the same way.

    python scripts/review_bench_joint.py --set geneval_two

Writes <OUT>/bench_joint/<set>/<idx>/seedNN/joint.png and a .json with protocol rule 9's columns, where
<idx> is the prompt's line number in the benchmark file. A tile that exists is skipped.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_bench_sets import SETS, load  # noqa: E402

OUT = Path(os.environ.get("REVIEW_BASELINE_OUT", "/datasets/mmolefe/poe_repair_min/outputs/paper_review"))
NOISE_PAIR = "a_cat__x__a_dog"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=sorted(SETS))
    ap.add_argument("--shard", default="0/1", help="k/n: render every n-th prompt starting at k, to split a set over cards")
    a = ap.parse_args()
    k, n = (int(x) for x in a.shard.split("/"))

    import torch

    from poe_repair.composers._helpers import init_latents_for_cell
    from poe_repair.experiments.interaction_term.cell import cell_from_slug
    from poe_repair.methods._sampling import run_cfg, write_decoded_image
    from poe_repair.run import make_ctx
    from poe_repair.runtime import encode_prompt_sdxl

    kept, dropped = load(a.set)
    seeds = SETS[a.set]["seeds"]
    print(f"{a.set}: {len(kept)} prompts kept, {len(dropped)} dropped by the subject lists, seeds {seeds}, "
          f"shard {k}/{n}", flush=True)
    ctx = make_ctx()
    enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = enc("")
    for seed in seeds:  # the latent this run draws must be the one every other method saw (protocol rule 2)
        init, _ = init_latents_for_cell(cell_from_slug(NOISE_PAIR, seed), ctx)
        saved = OUT / "latents" / f"seed{seed:02d}.pt"
        if saved.exists() and not torch.equal(init.cpu().to(torch.float16), torch.load(saved, map_location="cpu").to(torch.float16).reshape(init.shape)):
            raise SystemExit(f"seed {seed}: starting latent differs from {saved}")
        print(f"seed {seed}: starting latent {'matches ' + str(saved) if saved.exists() else 'has no saved copy'}", flush=True)
    for item in kept[k::n]:
        seq_j = pool_j = None
        for seed in seeds:
            png = OUT / "bench_joint" / a.set / f"{item['idx']:04d}" / f"seed{seed:02d}" / "joint.png"
            if png.exists():
                print(f"{png}: already rendered", flush=True)
                continue
            if seq_j is None:
                seq_j, pool_j = enc(item["prompt"])
            init, sigma = init_latents_for_cell(cell_from_slug(NOISE_PAIR, seed), ctx)
            t0 = time.perf_counter()
            res = run_cfg(init_latents=init, models=ctx.models, scheduler=ctx.scheduler,
                          seq_cond=seq_j, pool_cond=pool_j, seq_e=seq_e, pool_e=pool_e,
                          guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
                          height=1024, width=1024, euler_init_noise_sigma=sigma, device=ctx.device, dtype=ctx.dtype)
            # Folders made on this NFS can stay invisible for seconds and two cards race on them, so the
            # write is retried with the folder re-made and its parents re-listed (as review_baseline_cell.py).
            for attempt in range(10):
                try:
                    png.parent.mkdir(parents=True, exist_ok=True)
                    write_decoded_image(res.image, png)
                    break
                except OSError as e:
                    print(f"{png}: {type(e).__name__} on try {attempt + 1}, waiting 10 s", flush=True)
                    time.sleep(10)
                    for d in (png.parent.parent.parent, png.parent.parent):
                        try:
                            os.listdir(d)
                        except OSError:
                            pass
            else:
                print(f"{png}: NOT WRITTEN after 10 tries; a later pass picks it up", flush=True)
                continue
            secs = time.perf_counter() - t0
            png.with_suffix(".json").write_text(json.dumps({
                "model": "SDXL", "method": "joint", "adapter": "-", "sampler": "DDIM",
                "steps": ctx.num_inference_steps, "guidance": ctx.guidance_scale, "resolution": "1024x1024",
                "pair": "-", "seed": seed, "group": SETS[a.set]["bench"], "setting": a.set,
                "nfe": "", "set": a.set, "idx": item["idx"], "prompt": item["prompt"],
                "meta": item["meta"], "noise_from": NOISE_PAIR, "seconds": round(secs, 1)}, indent=1))
            print(f"{a.set} {item['idx']:04d} seed {seed}: {secs:.1f} s -> {png}", flush=True)
            del res
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
