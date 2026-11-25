#!/usr/bin/env python3
"""Loop 12 step Z8, the screening pass: plain PoE and the joint prompt on the tuning pairs, from which the failure cases are picked.

Design: artifacts/notes/progress/leakage/08-first-correction-test/design.md, "The screening pass". Plain PoE is
eps_A + eps_B - eps_empty, each expert at classifier-free guidance 7.5; the joint prompt is plain classifier-free guidance 7.5
on "a {A} and a {B}". 50 DDIM steps, 1024 px, the per-seed starting noise every pair shares (render_joint_sheet.py's).

Before any screening picture, the identity check: the joint prompt drawn as the product with the joint prompt as both experts
at half the guidance scale (u + w/2 (c - u) + w/2 (c - u) = u + w (c - u)), through run_poe_classifier_guided at strength 0,
the sampler Z8's push uses, against run_cfg at guidance w, both at the model's own precision, on the first pair and seed.
It passes when the mean absolute pixel difference is under 1 grey level of 255; the largest difference is reported.

  python z8_screen.py --splits <json> --out <dir> [--seeds 6001 6002 6003 6004] [--identity-only]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "rung2"))
sys.path.insert(0, str(HERE.parents[2] / "scripts" / "shared"))
from poe_repair.composers._helpers import init_latents_for_cell  # noqa: E402
from poe_repair.methods._classifier_guided import run_poe_classifier_guided  # noqa: E402
from poe_repair.methods._sampling import run_cfg, run_cfg_poe  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import encode_prompt_sdxl  # noqa: E402
import classifier_guidance_from_the_paper as single  # noqa: E402
from disallowed_subjects import is_allowed  # noqa: E402
from train_readout_pair import joint_prompt  # noqa: E402

KEPT_OFF = {"a_giraffe__x__a_tortoise"}  # counted elsewhere, never drawn on a sheet or used as a case


def subjects(pair):
    return [p.replace("_", " ") for p in pair.split("__x__")]


def to_img(o):
    return (o.image[0].float().clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()


def log_wandb(name, config, summary):
    try:
        import wandb
        run = wandb.init(project="poe-repair-animals-compose", entity="prime_lab", name=name, group="loop-12-z8",
                         config=config, settings=wandb.Settings(init_timeout=60))
        run.summary.update({k: v for k, v in summary.items() if isinstance(v, (int, float, str, bool))})
        rid = run.id
        run.finish()
        return rid
    except Exception as e:  # the run's files stay on /datasets either way
        print(f"wandb not logged: {e}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[6001, 6002, 6003, 6004])
    ap.add_argument("--identity-only", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    pairs = [p for p in json.loads(args.splits.read_text())["tuning_pairs"] if p not in KEPT_OFF]
    for p in pairs:
        assert is_allowed(joint_prompt(p)), p
    args.out.mkdir(parents=True, exist_ok=True)
    single.disk_guard(args.out)
    ctx = make_ctx(output_root=args.out / "raw")
    enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = enc("")
    w = ctx.guidance_scale
    common = dict(models=ctx.models, scheduler=ctx.scheduler, seq_e=seq_e, pool_e=pool_e, num_inference_steps=ctx.num_inference_steps,
                  height=1024, width=1024, device=ctx.device, dtype=ctx.dtype)

    # identity check
    pair, seed = pairs[0], args.seeds[0]
    seq_j, pool_j = enc(joint_prompt(pair))
    init, sigma = init_latents_for_cell(single.cell_for(seed), ctx)
    ref = to_img(run_cfg(init_latents=init, euler_init_noise_sigma=sigma, seq_cond=seq_j, pool_cond=pool_j, guidance_scale=w, **common))
    init, sigma = init_latents_for_cell(single.cell_for(seed), ctx)
    as_poe = to_img(run_poe_classifier_guided(init_latents=init, euler_init_noise_sigma=sigma, seq_a=seq_j, pool_a=pool_j,
                                              seq_b=seq_j, pool_b=pool_j, guidance_scale=w / 2, strength=0.0, **common))
    diff = np.abs(ref.astype(int) - as_poe.astype(int))
    identity = {"pair": pair, "seed": seed, "guidance": w, "dtype": str(ctx.dtype), "device": torch.cuda.get_device_name(0),
                "mean_abs_pixel_diff": float(diff.mean()), "max_abs_pixel_diff": int(diff.max()),
                "passes": bool(diff.mean() < 1.0)}
    (args.out / "identity.json").write_text(json.dumps(identity, indent=1) + "\n")
    print("IDENTITY " + json.dumps(identity), flush=True)
    if not identity["passes"] or args.identity_only:
        return 0 if identity["passes"] else 7

    t0, times, n = time.perf_counter(), [], 0
    for pair in pairs:
        a, b = subjects(pair)
        seq_a, pool_a = enc(a); seq_b, pool_b = enc(b); seq_j, pool_j = enc(joint_prompt(pair))
        for seed in args.seeds:
            for base in ("poe", "joint"):
                tile = args.out / base / pair / f"seed-{seed}.png"
                if tile.exists():
                    continue
                tile.parent.mkdir(parents=True, exist_ok=True)
                s = time.perf_counter()
                init, sigma = init_latents_for_cell(single.cell_for(seed), ctx)
                if base == "poe":
                    o = run_cfg_poe(init_latents=init, euler_init_noise_sigma=sigma, seq_a=seq_a, pool_a=pool_a, seq_b=seq_b,
                                    pool_b=pool_b, guidance_scale=w, **common)
                else:
                    o = run_cfg(init_latents=init, euler_init_noise_sigma=sigma, seq_cond=seq_j, pool_cond=pool_j, guidance_scale=w, **common)
                Image.fromarray(to_img(o)).save(tile)
                times.append(time.perf_counter() - s); n += 1
                print(f"{n} {base} {pair} seed={seed} {times[-1]:.1f}s", flush=True)
    summary = {"rendered": n, "median_seconds": float(np.median(times)) if times else 0.0, "identity_mean_abs": identity["mean_abs_pixel_diff"],
               "identity_max_abs": identity["max_abs_pixel_diff"], "total_seconds": time.perf_counter() - t0}
    summary["wandb_run"] = log_wandb("loop-12-z8-screen", {"pairs": pairs, "seeds": args.seeds, "guidance": w,
                                                            "steps": ctx.num_inference_steps}, summary)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
