#!/usr/bin/env python3
"""The binding plan's option A: the LoRA's separation plus each animal's own prediction inside its box.

Plan: 1-lora-correction/binding-plan.md, section 2. Rows 1 (the LoRA at weight 1) and 4 (the plain product) are Z10's pictures
on the same pairs, seeds, sampler and card type; this script draws rows 2 and 3:

  row 2  own   the plain product; inside animal A's box only, A's own guided prediction, inside B's box only, B's, at steps 5 to 30
  row 3  both  the LoRA-corrected product (weight 1) everywhere; inside each animal's box only, that animal's own prediction, steps 5 to 30

"Own prediction" is guided_eps(eps_A, eps_uncond, 7.5) from the untouched U-Net, the single-prompt classifier-free guidance of that
animal; the LoRA-corrected product takes "off" from the untouched U-Net and "on" from a copy carrying the LoRA (the off switch leaks
on this checkpoint). Where the two boxes overlap, and outside both, the cell keeps the row's product.

Checks before any scored picture, in one process:
  1. with both boxes empty, row 3's code gives the LoRA-at-weight-1 picture byte for byte (z10_dial.run_two_unet), and row 2's code
     gives the plain product byte for byte;
  2. the LoRA-at-weight-1 picture drawn here equals Z10's saved one for the same pair and seed; if not, rows 1 and 4 are drawn here too.

  python option_a.py --boxes <boxes.json> --pairs <pairs.json> --adapter <lora_step_030000.pt> --z10 <z10 renders dir> --out <dir>
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "2-classifier-guidance" / "scripts"))
sys.path.insert(0, str(REPO / "2-classifier-guidance" / "scripts" / "leakage"))
sys.path.insert(0, str(REPO / "scripts" / "shared"))
from poe_repair.composers._helpers import init_latents_for_cell  # noqa: E402
from poe_repair.experiments import _adapter_shape  # noqa: E402
from poe_repair.methods._sampling import SamplerOutputs, add_time_ids, run_cfg_poe  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import (LatentTrajectoryCollector, ddim_prev_from_x0_eps, decode_latents, encode_prompt_sdxl, guided_eps,  # noqa: E402
                                poe_eps, tweedie_mean)
import classifier_guidance_from_the_paper as single  # noqa: E402
from z10_dial import run_two_unet, slug, to_img  # noqa: E402

SEEDS = (6001, 6002, 6003, 6004)
WINDOW = range(5, 31)  # sampler steps 5 to 30, after the layout forms, as in Z7
GRID = 128


def box_mask(box, device):
    m = torch.zeros(GRID, GRID, device=device, dtype=torch.bool)
    if box:
        x0, y0, x1, y1 = [int(round(v / 8)) for v in box]
        m[max(y0, 0):min(y1, GRID), max(x0, 0):min(x1, GRID)] = True
    return m


@torch.no_grad()
def run_boxed(*, base, adapted, lam, box_a, box_b, init_latents, models, scheduler, seq_a, pool_a, seq_b, pool_b, seq_e, pool_e,
              guidance_scale, num_inference_steps, height, width, euler_init_noise_sigma, device, dtype):
    """z10_dial.run_two_unet, with each animal's own prediction inside its box only at steps 5 to 30. Empty boxes give run_two_unet."""
    scheduler.set_timesteps(num_inference_steps)
    latents = (init_latents / euler_init_noise_sigma).to(device=device, dtype=dtype)
    tracker = LatentTrajectoryCollector(num_inference_steps, 1, latents.shape[1], latents.shape[2], latents.shape[3])
    pe = torch.cat([seq_a, seq_b, seq_e], dim=0)
    pool = torch.cat([pool_a, pool_b, pool_e], dim=0)
    cond = {"text_embeds": pool, "time_ids": add_time_ids(height=height, width=width, batch_size=3, device=device, dtype=dtype)}
    ma, mb = box_mask(box_a, device), box_mask(box_b, device)
    only_a, only_b = (ma & ~mb)[None, None], (mb & ~ma)[None, None]

    def forward(unet, latent_input, timestep):
        noise = unet(latent_input, timestep, encoder_hidden_states=pe, added_cond_kwargs=cond, timestep_cond=None).sample
        a, b, u = noise.chunk(3)
        ga, gb = guided_eps(a, u, guidance_scale), guided_eps(b, u, guidance_scale)
        return poe_eps(ga, gb, u), ga, gb
    for step_index, timestep in enumerate(scheduler.timesteps):
        latent_input = scheduler.scale_model_input(latents.repeat(3, 1, 1, 1), timestep)
        eps_p, own_a, own_b = forward(base, latent_input, timestep)
        if lam != 0.0:
            delta = forward(adapted, latent_input, timestep)[0] - eps_p
            eps_p = eps_p + float(lam) * delta
        if step_index in WINDOW and (only_a.any() or only_b.any()):
            eps_p = torch.where(only_a, own_a, eps_p)
            eps_p = torch.where(only_b, own_b, eps_p)
        alpha_bar_t = scheduler.alphas_cumprod[int(timestep.item())].to(device=device, dtype=dtype)
        x0 = tweedie_mean(latents, alpha_bar_t, eps_p)
        tracker.store_step(step_index, latents, eps_p, float(step_index) / float(num_inference_steps), int(timestep.item()))
        latents = ddim_prev_from_x0_eps(scheduler=scheduler, timestep=timestep, step_index=step_index, x0=x0, eps=eps_p)
    tracker.store_final(latents)
    return SamplerOutputs(latents=latents, image=decode_latents(models, latents).cpu(), tracker=tracker, extras={})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--boxes", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--z10", type=Path, required=True, help="Z10's renders folder: <pair>/seed-<seed>-w1.00.png and -w0.00.png")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    prompts = json.loads(args.pairs.read_text())["pairs"]
    boxes = json.loads(args.boxes.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    single.disk_guard(args.out)
    ctx = make_ctx(output_root=args.out / "raw")
    enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = enc("")
    common = dict(models=ctx.models, scheduler=ctx.scheduler, seq_e=seq_e, pool_e=pool_e, guidance_scale=ctx.guidance_scale,
                  num_inference_steps=ctx.num_inference_steps, height=1024, width=1024, device=ctx.device, dtype=ctx.dtype)

    def experts(prompt):
        a, b = prompt.split(" and ")
        sa, pa = enc(a); sb, pb = enc(b)
        return dict(seq_a=sa, pool_a=pa, seq_b=sb, pool_b=pb)
    base = ctx.models["unet"]
    adapted = copy.deepcopy(base)
    info = _adapter_shape.attach(adapted, args.adapter); adapted.eval()
    print("ADAPTER " + json.dumps({k: info[k] for k in ("target_modules", "rank", "n_matched", "checkpoint_step")}, default=list), flush=True)

    def noise(seed):
        return init_latents_for_cell(single.cell_for(seed), ctx)

    # check 1: empty boxes reproduce the two samplers byte for byte
    e0 = experts(prompts[0]); s0 = SEEDS[0]
    mx = lambda x, y: int(np.abs(x.astype(int) - y.astype(int)).max())
    i, sg = noise(s0); lora1 = to_img(run_two_unet(base=base, adapted=adapted, lam=1.0, init_latents=i, euler_init_noise_sigma=sg, **e0, **common))
    i, sg = noise(s0); both_empty = to_img(run_boxed(base=base, adapted=adapted, lam=1.0, box_a=None, box_b=None, init_latents=i, euler_init_noise_sigma=sg, **e0, **common))
    i, sg = noise(s0); plain = to_img(run_cfg_poe(init_latents=i, euler_init_noise_sigma=sg, **e0, **common))
    i, sg = noise(s0); own_empty = to_img(run_boxed(base=base, adapted=adapted, lam=0.0, box_a=None, box_b=None, init_latents=i, euler_init_noise_sigma=sg, **e0, **common))
    # check 2: Z10's saved picture reproduces on this card
    saved = np.array(Image.open(args.z10 / slug(prompts[0]) / f"seed-{s0}-w1.00.png").convert("RGB"))
    checks = {"empty_boxes_both_vs_lora_max_abs": mx(both_empty, lora1), "empty_boxes_own_vs_plain_max_abs": mx(own_empty, plain),
              "lora_here_vs_z10_saved_max_abs": mx(lora1, saved), "device": torch.cuda.get_device_name(0)}
    checks["check_1_passes"] = checks["empty_boxes_both_vs_lora_max_abs"] == 0 and checks["empty_boxes_own_vs_plain_max_abs"] == 0
    checks["z10_reproduces"] = checks["lora_here_vs_z10_saved_max_abs"] == 0
    print("CHECKS " + json.dumps(checks), flush=True)
    (args.out / "checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    if not checks["check_1_passes"]:
        raise SystemExit("check 1 failed: empty boxes do not reproduce the samplers")
    rows = [("own", 0.0, True), ("both", 1.0, True)]
    if not checks["z10_reproduces"]:
        rows = [("lora", 1.0, False), ("plain", 0.0, False)] + rows
        print("Z10's pictures do not reproduce here; rows 1 and 4 are drawn in this job", flush=True)
    times, n = [], 0
    for prompt in prompts:
        e = experts(prompt); sl = slug(prompt)
        for seed in SEEDS:
            b = boxes[sl][str(seed)]
            for row, lam, use_boxes in rows:
                tile = args.out / row / sl / f"seed-{seed}.png"
                if tile.exists():
                    continue
                tile.parent.mkdir(parents=True, exist_ok=True)
                i, sg = noise(seed); t0 = time.perf_counter()
                o = run_boxed(base=base, adapted=adapted, lam=lam, box_a=b.get("A") if use_boxes else None,
                              box_b=b.get("B") if use_boxes else None, init_latents=i, euler_init_noise_sigma=sg, **e, **common)
                Image.fromarray(to_img(o)).save(tile)
                times.append(time.perf_counter() - t0); n += 1
                print(f"{n} {row} {prompt} seed={seed} {times[-1]:.1f}s", flush=True)
    summary = {**checks, "rows": [r[0] for r in rows], "rendered": n, "median_seconds": float(np.median(times)) if times else 0.0}
    try:
        import wandb
        run = wandb.init(project="poe-repair-animals-compose", entity="prime_lab", name="lora-binding-option-a", group="lora-binding",
                         config={"pairs": prompts, "seeds": SEEDS, "window": [5, 30], "adapter": str(args.adapter)},
                         settings=wandb.Settings(init_timeout=60))
        run.summary.update({k: v for k, v in summary.items() if isinstance(v, (int, float, bool))}); summary["wandb_run"] = run.id; run.finish()
    except Exception as ex:
        print(f"wandb not logged: {ex}", flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
