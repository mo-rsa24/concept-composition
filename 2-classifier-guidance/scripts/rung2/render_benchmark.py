#!/usr/bin/env python
"""The benchmark's 48 training pairs, re-rendered for rung 2's larger discriminator training set.

For every training pair of ``pair_pool_v58.yaml`` (never cat x dog, never seeds 9 to 16) and each
seed given, two renders from the seed's own starting noise:
    joint     the pair's joint prompt ("a bear and a boar"), ordinary guidance
    product   the plain product of its two prompts
Each saves its picture and its final latent, and each pair gets one sheet (seeds as rows) to call.
The starting noise comes from the seed number, as every rung 2 render does; the benchmark's own
cached noise is on /datasets and is not the same, so these are new pictures to be called afresh.

--part K --of N renders every N-th pair starting at K, so two cards can split the list.

Writes, under MOUNT_ROOT/outputs/classifier_guidance_ladder/rung2/benchmark/:
    tiles/<pair>__seedNN__<joint|product>.png, latents/..., sheets/<pair>.png, manifest-partK.json
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import matplotlib
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poe_repair import paths  # noqa: E402
from poe_repair.composers._helpers import init_latents_for_cell  # noqa: E402
from poe_repair.methods._sampling import run_cfg, run_cfg_poe, write_decoded_image  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import encode_prompt_sdxl  # noqa: E402
import classifier_guidance_from_the_paper as single  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[3] / "artifacts/_shared/cross_pair_pool_configs"
OUT = paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2/benchmark"
HELD_OUT_SEEDS = range(9, 17)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 9)))
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    args = ap.parse_args()
    if any(s in HELD_OUT_SEEDS for s in args.seeds):
        raise SystemExit("guard: seeds 9 to 16 are held out and never enter training")
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    pairs = yaml.safe_load(open(CONFIGS / "pair_pool_v58.yaml"))["train"]
    prompts = yaml.safe_load(open(CONFIGS / "pair_prompts_v58.yaml"))
    if "a_cat__x__a_dog" in pairs:
        raise SystemExit("guard: cat x dog is the held-out evaluation pair")
    mine = pairs[args.part::args.of]
    print(f"{socket.gethostname()}: {torch.cuda.get_device_name(0)}, {len(mine)} of {len(pairs)} pairs, "
          f"seeds {args.seeds}", flush=True)
    single.disk_guard(OUT)
    for d in ("tiles", "latents", "sheets"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    ctx = make_ctx(output_root=OUT / "raw")
    enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = enc("")
    common = dict(models=ctx.models, scheduler=ctx.scheduler, seq_e=seq_e, pool_e=pool_e,
                  guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
                  height=1024, width=1024, device=ctx.device, dtype=ctx.dtype)
    manifest = []
    for pair in mine:
        p = prompts[pair]
        seq_j, pool_j = enc(p["joint_prompt"])
        seq_a, pool_a = enc(p["prompt_a"])
        seq_b, pool_b = enc(p["prompt_b"])
        t0 = time.perf_counter()
        for seed in args.seeds:
            stem = f"{pair}__seed{seed:02d}"
            init, sigma = init_latents_for_cell(single.cell_for(seed), ctx)
            o = run_cfg(init_latents=init, euler_init_noise_sigma=sigma, seq_cond=seq_j, pool_cond=pool_j, **common)
            write_decoded_image(o.image, OUT / "tiles" / f"{stem}__joint.png")
            torch.save(o.latents.detach().cpu(), OUT / "latents" / f"{stem}__joint.pt")
            init, sigma = init_latents_for_cell(single.cell_for(seed), ctx)
            o = run_cfg_poe(init_latents=init, euler_init_noise_sigma=sigma, seq_a=seq_a, pool_a=pool_a,
                            seq_b=seq_b, pool_b=pool_b, **common)
            write_decoded_image(o.image, OUT / "tiles" / f"{stem}__product.png")
            torch.save(o.latents.detach().cpu(), OUT / "latents" / f"{stem}__product.pt")
            manifest += [{"pair": pair, "seed": seed, "column": c, "prompt": pr} for c, pr in
                         (("joint", p["joint_prompt"]), ("product", f"{p['prompt_a']} x {p['prompt_b']}"))]
        fig, axes = plt.subplots(2, len(args.seeds), figsize=(2.6 * len(args.seeds), 5.6))
        for c, seed in enumerate(args.seeds):
            for r, col in enumerate(("joint", "product")):
                ax = axes[r][c]
                with Image.open(OUT / "tiles" / f"{pair}__seed{seed:02d}__{col}.png") as im:
                    ax.imshow(im.convert("RGB").resize((360, 360)))
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"seed {seed}", fontsize=10)
                if c == 0:
                    ax.set_ylabel(f"\"{p['joint_prompt']}\"" if col == "joint" else "plain product", fontsize=9)
        fig.suptitle(pair, fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(OUT / "sheets" / f"{pair}.png", dpi=90)
        plt.close(fig)
        (OUT / f"manifest-part{args.part}.json").write_text(json.dumps(manifest, indent=1))
        print(f"{pair}: {2 * len(args.seeds)} renders in {time.perf_counter() - t0:.0f} s", flush=True)
    print("BENCHMARK DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
