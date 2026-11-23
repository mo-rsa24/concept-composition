#!/usr/bin/env python3
"""Step M9, rung 1: the oracle-layout sheet on 8 cat and dog seeds.

  refs      the joint prompt and the plain product, in a process that never imports the regional composer
  regional  the composer check, then right boxes, random boxes, swapped prompts and the same prompt in both

The design is artifacts/notes/progress/method-search/04-bind-each-body/design.md.

Usage: python m9_rung1.py refs|regional --boxes <boxes.json> --out <dir> [--seeds 9 12]
"""
import argparse
import json
import shutil
import socket
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from poe_repair.composers._helpers import encode_pair, get_joint_embeds, init_latents_for_cell  # noqa: E402
from poe_repair.experiments._eval_common import cell_for  # noqa: E402
from poe_repair.methods._sampling import run_cfg, run_cfg_poe, write_decoded_image  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402

PROMPT_A, PROMPT_B = "a cat", "a dog"
EXPERIMENT = "rung1"
READING = "even"
CONDITIONS = ("oracle", "random", "swapped", "same")
MAX_PIXEL_DIFF = 3.0   # of 255, the project's reproduction limit
RANDOM_BOX_SEED = 20261123
SOURCE_JOINT = Path("/datasets/mmolefe/poe_repair_min/outputs/is_the_layout_the_missing_piece/joint")


def mean_abs(a, b):
    with Image.open(a) as ia, Image.open(b) as ib:
        x = np.asarray(ia.convert("RGB"), dtype=np.float64)
        y = np.asarray(ib.convert("RGB").resize(ia.size), dtype=np.float64)
    return float(np.abs(x - y).mean())


def read_boxes(path):
    data = json.loads(Path(path).read_text())
    out = {}
    for record in data["seeds"]:
        if record.get("selected"):
            named = {box["call"]: box["box_cells"] for box in record["boxes"]}
            if set(named) != {"cat", "dog"}:
                raise SystemExit(f"seed {record['seed']}: boxes are not one cat and one dog")
            out[record["seed"]] = named
    if sorted(out) != sorted(data["selected_seeds"]) or len(out) != 8:
        raise SystemExit(f"expected 8 selected seeds, found {sorted(out)}")
    return out


def refs(args, boxes, ctx):
    from poe_repair.composers import poe
    if "poe_repair.composers.poe_regional" in sys.modules:
        raise SystemExit("the reference process must not import the regional composer")
    folder = args.out / EXPERIMENT / "joint"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in sorted(boxes):
        cell = cell_for(PROMPT_A, PROMPT_B, seed)
        path = folder / f"seed{seed:02d}.png"
        if not path.exists():
            init, sigma = init_latents_for_cell(cell, ctx)
            emb = encode_pair(cell, ctx)
            seq, pool = get_joint_embeds(cell, ctx)
            result = run_cfg(init_latents=init, models=ctx.models, scheduler=ctx.scheduler, seq_cond=seq, pool_cond=pool,
                             seq_e=emb["seq_e"], pool_e=emb["pool_e"], guidance_scale=ctx.guidance_scale,
                             num_inference_steps=ctx.num_inference_steps, height=cell.height, width=cell.width,
                             euler_init_noise_sigma=sigma, device=ctx.device, dtype=ctx.dtype)
            write_decoded_image(result.image, path)
        poe.run(cell, ctx, exp_name=EXPERIMENT)
        source = SOURCE_JOINT / f"seed{seed:02d}.png"
        diff = mean_abs(path, source)
        rows.append({"seed": seed, "joint_vs_box_source_of_255": round(diff, 3), "passes": diff <= MAX_PIXEL_DIFF})
        print(json.dumps(rows[-1]), flush=True)
    (args.out / "refs_check.json").write_text(json.dumps({"limit_of_255": MAX_PIXEL_DIFF, "seeds": rows}, indent=1))
    if not all(r["passes"] for r in rows):
        raise SystemExit("a joint render does not match the picture its boxes came from")


def composer_check(args, ctx):
    """Masks of 0.5 at guidance g are the plain product at guidance g / 2."""
    from poe_repair.methods._poe_regional import run_cfg_poe_regional
    cell = cell_for(PROMPT_A, PROMPT_B, 9)
    init, sigma = init_latents_for_cell(cell, ctx)
    emb = encode_pair(cell, ctx)
    shared = dict(init_latents=init, models=ctx.models, scheduler=ctx.scheduler, seq_a=emb["seq_a"], pool_a=emb["pool_a"],
                  seq_b=emb["seq_b"], pool_b=emb["pool_b"], seq_e=emb["seq_e"], pool_e=emb["pool_e"],
                  num_inference_steps=ctx.num_inference_steps, height=cell.height, width=cell.width,
                  euler_init_noise_sigma=sigma, device=ctx.device, dtype=ctx.dtype)
    half = torch.full((1, 1, 128, 128), 0.5, device=ctx.device, dtype=ctx.dtype)
    regional = run_cfg_poe_regional(mask_a=half, mask_b=half, guidance_scale=ctx.guidance_scale, **shared)
    plain = run_cfg_poe(guidance_scale=ctx.guidance_scale / 2, **shared)
    folder = args.out / EXPERIMENT / "composer-check"
    folder.mkdir(parents=True, exist_ok=True)
    write_decoded_image(regional.image, folder / "regional-half-masks.png")
    write_decoded_image(plain.image, folder / "plain-half-guidance.png")
    diff = mean_abs(folder / "regional-half-masks.png", folder / "plain-half-guidance.png")
    result = {"seed": 9, "mean_abs_pixel_diff_of_255": round(diff, 4), "limit_of_255": MAX_PIXEL_DIFF,
              "max_abs_latent_diff": float((regional.latents.float() - plain.latents.float()).abs().max()),
              "passes": diff <= MAX_PIXEL_DIFF}
    (args.out / "composer_check.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result), flush=True)
    if not result["passes"]:
        raise SystemExit("the regional composer does not reduce to the plain product")


def regional(args, boxes, ctx):
    from poe_repair.composers import poe_regional
    composer_check(args, ctx)
    for seed in sorted(boxes):
        cell = cell_for(PROMPT_A, PROMPT_B, seed)
        for condition in CONDITIONS:
            path = poe_regional.run(cell, ctx, cat_box=boxes[seed]["cat"], dog_box=boxes[seed]["dog"], condition=condition,
                                    reading=READING, random_seed=RANDOM_BOX_SEED + seed if condition == "random" else None,
                                    exp_name=EXPERIMENT)
            print(f"seed {seed} {condition} {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["refs", "regional"])
    ap.add_argument("--boxes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="*")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("torch sees no CUDA card")
    usage = shutil.disk_usage(args.out.parent)
    if usage.free / usage.total < 0.10:
        raise SystemExit("under 10% free on the output filesystem")
    boxes = read_boxes(args.boxes)
    if args.seeds:
        boxes = {s: boxes[s] for s in args.seeds}
    print(f"{socket.gethostname()} {torch.cuda.get_device_name(0)} stage={args.stage} seeds={sorted(boxes)}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    ctx = make_ctx(output_root=args.out)
    (refs if args.stage == "refs" else regional)(args, boxes, ctx)
    print("finished", args.stage, flush=True)


if __name__ == "__main__":
    main()
