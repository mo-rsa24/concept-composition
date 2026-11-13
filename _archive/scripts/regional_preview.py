#!/usr/bin/env python
"""Check the regional sampler against plain product, then render both overlap readings on two seeds.

Two jobs, in this order, because the second is worthless if the first fails.

**The identity check.** With both masks flat at 0.5 everywhere, the regional sampler predicts

    eps_uncond + w * (0.5 * d_A + 0.5 * d_B)  =  eps_uncond + (w/2) * (d_A + d_B)

which is exactly what plain product predicts at half the guidance. So the regional sampler at
guidance 7.5 with flat masks and ``run_cfg_poe`` at guidance 3.75 must render the same picture.
Any real difference is a bug in the new sampler, and the check is run on real renders rather
than on a stub so the scheduler, the precision and the decode are all in it. The bar is in the
source: mean absolute pixel difference on a 0 to 255 scale, under ``MAX_IDENTITY_DIFF``, which is
the fp16 noise floor this repo has already measured for the same combine over 50 steps.

**The two readings.** The oracle layout rendered on two seeds under each overlap reading, with
the joint prompt beside them. This is the figure the overlap choice is made from, and it is made
before the sweep rather than argued about.

Every render here starts from the same cached noise as every other column of its row, so the
only thing differing down a row is the sampler.

    python scripts/regional_preview.py --seeds 9 19
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from poe_repair.composers import poe_regional  # noqa: E402
from poe_repair.composers._helpers import (  # noqa: E402
    encode_pair, get_joint_embeds, init_latents_for_cell,
)
from poe_repair.composers._regional_masks import LATENT, READINGS  # noqa: E402
from poe_repair.experiments._eval_common import cell_for  # noqa: E402
from poe_repair.methods._poe_regional import run_cfg_poe_regional  # noqa: E402
from poe_repair.methods._sampling import run_cfg, run_cfg_poe, write_decoded_image  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402

OUT_DIR = REPO / "artifacts/results/is-the-layout-the-missing-piece"
RENDER_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/is_the_layout_the_missing_piece")
PROMPT_A, PROMPT_B = "a cat", "a dog"

# The bar, in the source so it cannot move after the answer. A float32-against-float16 comparison
# of the same combine over 50 steps gives about three grey levels, so anything under that is the
# precision floor and not a difference in the operator.
MAX_IDENTITY_DIFF = 3.0

MAX_FOREIGN_MIB_DEFAULT = 1024
NEEDED_MIB = 12288        # SDXL at 1024 px with a three-branch batch
MIN_FREE_DISK_FRACTION = 0.10


def guard_disk(out_dir: Path) -> None:
    import shutil
    out_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(out_dir)
    free = usage.free / usage.total
    print(f"disk guard: {out_dir} has {usage.free / 1024 ** 3:.1f} GiB free of "
          f"{usage.total / 1024 ** 3:.1f} GiB ({free:.1%})", flush=True)
    if free < MIN_FREE_DISK_FRACTION:
        raise SystemExit(f"guard: under {MIN_FREE_DISK_FRACTION:.0%} free; refusing to write")


def guard_device(max_foreign_mib: int) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("guard: no CUDA; this would render silently on the CPU")
    free_b, total_b = torch.cuda.mem_get_info(0)
    used_mib, free_mib = (total_b - free_b) / 1024 ** 2, free_b / 1024 ** 2
    if free_mib < NEEDED_MIB:
        raise SystemExit(f"guard: {free_mib:.0f} MiB free, under the {NEEDED_MIB} MiB SDXL needs")
    if used_mib > max_foreign_mib:
        raise SystemExit(f"guard: {used_mib:.0f} MiB held by other processes, over the "
                         f"{max_foreign_mib} MiB limit; pass --max-foreign-mib to share")
    print(f"gpu guard: {used_mib:.0f} MiB foreign, {free_mib:.0f} MiB free", flush=True)


def mean_abs_diff(a: Path, b: Path) -> float:
    """Mean absolute pixel difference on a 0 to 255 scale."""
    with Image.open(a) as ia, Image.open(b) as ib:
        x = np.asarray(ia.convert("RGB"), dtype=np.float64)
        y = np.asarray(ib.convert("RGB"), dtype=np.float64)
    return float(np.abs(x - y).mean())


def identity_check(ctx, seed: int, out_dir: Path) -> dict:
    """Flat 0.5 masks at guidance w must equal plain product at guidance w/2."""
    cell = cell_for(PROMPT_A, PROMPT_B, seed)
    init, sigma = init_latents_for_cell(cell, ctx)
    emb = encode_pair(cell, ctx)
    common = dict(models=ctx.models, scheduler=ctx.scheduler,
                  num_inference_steps=ctx.num_inference_steps,
                  height=cell.height, width=cell.width, euler_init_noise_sigma=sigma,
                  device=ctx.device, dtype=ctx.dtype)
    flat = torch.full((1, 1, LATENT, LATENT), 0.5, device=ctx.device, dtype=ctx.dtype)

    regional_path = out_dir / "identity_regional_flat_masks.png"
    plain_path = out_dir / "identity_plain_poe_half_guidance.png"
    out_dir.mkdir(parents=True, exist_ok=True)

    res = run_cfg_poe_regional(
        init_latents=init, seq_a=emb["seq_a"], pool_a=emb["pool_a"],
        seq_b=emb["seq_b"], pool_b=emb["pool_b"], seq_e=emb["seq_e"], pool_e=emb["pool_e"],
        mask_a=flat, mask_b=flat, guidance_scale=ctx.guidance_scale, **common)
    write_decoded_image(res.image, regional_path)
    del res; gc.collect(); torch.cuda.empty_cache()

    res = run_cfg_poe(
        init_latents=init, seq_a=emb["seq_a"], pool_a=emb["pool_a"],
        seq_b=emb["seq_b"], pool_b=emb["pool_b"], seq_e=emb["seq_e"], pool_e=emb["pool_e"],
        guidance_scale=ctx.guidance_scale / 2.0, **common)
    write_decoded_image(res.image, plain_path)
    del res; gc.collect(); torch.cuda.empty_cache()

    diff = mean_abs_diff(regional_path, plain_path)
    verdict = "held" if diff <= MAX_IDENTITY_DIFF else "did not hold"
    print(f"identity check on seed {seed}: mean absolute pixel difference {diff:.3f} of 255, "
          f"bar {MAX_IDENTITY_DIFF} -> {verdict}", flush=True)
    return {"seed": seed, "mean_abs_pixel_diff": diff, "bar": MAX_IDENTITY_DIFF,
            "verdict": verdict, "regional_flat": str(regional_path),
            "plain_poe_half_guidance": str(plain_path)}


def boxes_for(record: dict) -> tuple[list[int], list[int]]:
    by_call = {b["call"]: b for b in record["boxes"]}
    return by_call["cat"]["box_cells"], by_call["dog"]["box_cells"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[9, 19])
    ap.add_argument("--bootstrap", type=int, nargs="+", default=[0, 15],
                    help="one column per value: how many opening steps each region is denoised "
                         "against a flat colour instead of against the other animal")
    ap.add_argument("--reading", default="even",
                    help="the overlap reading; the two were indistinguishable on the first "
                         "preview, so only one is rendered now")
    ap.add_argument("--boxes", type=Path, default=OUT_DIR / "boxes.json")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--render-root", type=Path, default=RENDER_ROOT)
    ap.add_argument("--max-foreign-mib", type=int, default=MAX_FOREIGN_MIB_DEFAULT)
    ap.add_argument("--skip-identity", action="store_true",
                    help="skip the identity check; only for a re-run that already passed it")
    args = ap.parse_args()

    print(f"node {socket.gethostname()}  CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}", flush=True)
    guard_disk(args.render_root)
    guard_device(args.max_foreign_mib)

    data = json.loads(args.boxes.read_text())
    selected = {r["seed"]: r for r in data["seeds"] if r.get("selected")}
    missing = [s for s in args.seeds if s not in selected]
    if missing:
        raise SystemExit(f"seeds {missing} are not selected in {args.boxes}")

    ctx = make_ctx(output_root=args.render_root)
    print(f"guidance {ctx.guidance_scale}, {ctx.num_inference_steps} steps, dtype {ctx.dtype}",
          flush=True)

    record = {"identity_check": None, "panels": []}
    if not args.skip_identity:
        record["identity_check"] = identity_check(ctx, args.seeds[0], args.out / "identity")
        if record["identity_check"]["verdict"] != "held":
            (args.out / "overlap-readings.json").write_text(json.dumps(record, indent=2))
            raise SystemExit("the identity check did not hold; not rendering anything else")

    cols = 1 + len(args.bootstrap)
    fig, axes = plt.subplots(len(args.seeds), cols, figsize=(cols * 3.6, len(args.seeds) * 3.9),
                             squeeze=False)
    for row, seed in enumerate(args.seeds):
        cell = cell_for(PROMPT_A, PROMPT_B, seed)
        cat_box, dog_box = boxes_for(selected[seed])

        joint_path = args.render_root / "joint" / f"seed{seed:02d}.png"
        if not joint_path.exists():
            init, sigma = init_latents_for_cell(cell, ctx)
            emb = encode_pair(cell, ctx)
            seq_j, pool_j = get_joint_embeds(cell, ctx)
            res = run_cfg(init_latents=init, models=ctx.models, scheduler=ctx.scheduler,
                          seq_cond=seq_j, pool_cond=pool_j,
                          seq_e=emb["seq_e"], pool_e=emb["pool_e"],
                          guidance_scale=ctx.guidance_scale,
                          num_inference_steps=ctx.num_inference_steps,
                          height=cell.height, width=cell.width,
                          euler_init_noise_sigma=sigma, device=ctx.device, dtype=ctx.dtype)
            joint_path.parent.mkdir(parents=True, exist_ok=True)
            write_decoded_image(res.image, joint_path)
            del res; gc.collect(); torch.cuda.empty_cache()
        with Image.open(joint_path) as im:
            axes[row][0].imshow(im.convert("RGB"))
        axes[row][0].set_title(f"seed {seed}: the joint prompt", fontsize=9)

        for col, steps in enumerate(args.bootstrap, start=1):
            path = poe_regional.run(cell, ctx, cat_box=cat_box, dog_box=dog_box,
                                    condition="oracle", reading=args.reading,
                                    bootstrap_steps=steps)
            with Image.open(path) as im:
                axes[row][col].imshow(im.convert("RGB"))
            label = ("no bootstrapping" if steps == 0
                     else f"{steps} bootstrapping steps of {ctx.num_inference_steps}")
            axes[row][col].set_title(f"oracle layout, {label}", fontsize=9)
            record["panels"].append({"seed": seed, "reading": args.reading,
                                     "bootstrap_steps": steps, "path": str(path)})
            print(f"seed {seed} bootstrap {steps}: {path}", flush=True)
            gc.collect(); torch.cuda.empty_cache()
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])

    inches = len(args.seeds) * 3.9
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.85 / inches])
    fig.suptitle("the oracle layout with and without bootstrapping, beside the joint prompt "
                 "that supplied it\nsame starting noise down each row; guidance 7.5, 50 DDIM "
                 "steps; bootstrapping denoises each region against a flat colour for its "
                 "opening steps",
                 fontsize=11, y=1 - 0.15 / inches, va="top")
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "overlap-readings.png", dpi=130)
    plt.close(fig)
    (args.out / "overlap-readings.json").write_text(json.dumps(record, indent=2))
    print(f"wrote {args.out / 'overlap-readings.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
