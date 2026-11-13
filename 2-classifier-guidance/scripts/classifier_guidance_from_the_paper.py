#!/usr/bin/env python
"""Classifier guidance on the plainest case: SDXL's empty prompt pushed toward "dog".

The two ends come first. The empty prompt alone is what guidance starts from; "a dog" under
ordinary guidance is the look a guided picture should reach. Both share each seed's starting noise
so a pushed picture can be read against both.

With ``--push`` the empty prompt is then guided toward "dog" by the strong presence head, in the
form of Dhariwal and Nichol's Algorithm 2 with nothing rescaled. Strength 0 must reproduce the
empty-prompt render byte for byte. The four strengths are chosen from the gradient measured along
the empty-prompt run, so that the median push is PUSH_PERCENTS of the model's own prediction: the
paper's strengths belong to the paper's classifier, whose gradient is far larger than this head's.

Writes, under MOUNT_ROOT/outputs/classifier_guidance_from_the_paper/:
    tiles/seedNN__<column>.png   every full-resolution render
    two-ends.png                 the seeds down, the two ends across
    dog-sheet.png                the seeds down; no prompt, "a dog", and the four strengths across
    push.json                    the strength chosen for each percent, and per render the gradient,
                                 the push and the prediction's size at every step
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from poe_repair import paths  # noqa: E402
from poe_repair.composers._helpers import init_latents_for_cell  # noqa: E402
from poe_repair.config import RunConfig  # noqa: E402
from poe_repair.methods._sampling import run_cfg, write_decoded_image  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import PairSeedCell, encode_prompt_sdxl  # noqa: E402

SEEDS = (9, 10, 11, 12)
PUSH_PERCENTS = (1, 3, 10, 30)
HEAD_CKPT = paths.MOUNT_ROOT / "outputs/strong_presence_head/strong-01/checkpoints/head_step_004000.pt"
OUT = paths.MOUNT_ROOT / "outputs/classifier_guidance_from_the_paper"
DOG_CLASSIFIER_ROOT = paths.MOUNT_ROOT / "outputs/dog_classifier"
MIN_FREE_DISK_PERCENT = 10


def disk_guard(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    u = shutil.disk_usage(root)
    if 100.0 * u.free / u.total < MIN_FREE_DISK_PERCENT:
        raise SystemExit(f"guard: {root} is nearly full")


def cell_for(seed: int) -> PairSeedCell:
    # The cat x dog cell is used only for its per-seed starting noise, which every pair shares.
    return PairSeedCell(pair_dir=RunConfig().paths.pilot_dir / f"seed_{seed}" / "a_cat__x__a_dog",
                        pair_slug="a_cat__x__a_dog", prompt_a="a cat", prompt_b="a dog",
                        seed=seed, regime="collision", height=1024, width=1024, grid_assets={})


def render(ctx, seed, seq_c, pool_c, seq_e, pool_e, path: Path) -> Path:
    init, sigma = init_latents_for_cell(cell_for(seed), ctx)
    out = run_cfg(init_latents=init, models=ctx.models, scheduler=ctx.scheduler,
                  seq_cond=seq_c, pool_cond=pool_c, seq_e=seq_e, pool_e=pool_e,
                  guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
                  height=1024, width=1024, euler_init_noise_sigma=sigma,
                  device=ctx.device, dtype=ctx.dtype)
    write_decoded_image(out.image, path)
    return path


def sheet(columns, path: Path, title: str) -> None:
    fig, axes = plt.subplots(len(SEEDS), len(columns), figsize=(len(columns) * 3.0, len(SEEDS) * 3.1))
    for r, seed in enumerate(SEEDS):
        for c, (label, key) in enumerate(columns):
            ax = axes[r][c]
            with Image.open(OUT / "tiles" / f"seed{seed:02d}__{key}.png") as im:
                ax.imshow(im.convert("RGB").resize((360, 360)))
            if r == 0:
                ax.set_title(label, fontsize=12)
            if c == 0:
                ax.text(-0.06, 0.5, f"seed {seed}", transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0.03, 0, 1, 0.95])
    fig.savefig(path, dpi=100)
    plt.close(fig)


def load_head(ctx, seq_e, pool_e):
    from poe_repair.experiments.presence_head.strong_head import (
        SpeciesHead, StrongPresenceHead, UNetMidFeatures,
    )
    state = torch.load(HEAD_CKPT, map_location="cpu")
    head = SpeciesHead(len(state["species"])).to(ctx.device)
    head.load_state_dict(state["state_dict"])
    return StrongPresenceHead(UNetMidFeatures(ctx.models["unet"], seq_e, pool_e), head,
                              state["species"], target_species=("dog",))


def load_dog_classifier(ctx, seq_e, pool_e, run: str, step: int):
    """Plan 18's classifier: a LoRA'd copy of the UNet trunk plus the attention-pool head."""
    from diffusers import UNet2DConditionModel
    from poe_repair.experiments.presence_head.dog_classifier import (
        AttentionPoolHead, DogClassifier, attach_trunk_adapters,
    )
    from poe_repair.experiments.presence_head.strong_head import UNetMidFeatures
    ckpt = DOG_CLASSIFIER_ROOT / run / "checkpoints" / f"step_{step:06d}.pt"
    state = torch.load(ckpt, map_location="cpu")
    unet = UNet2DConditionModel.from_pretrained(
        RunConfig().model_id, subfolder="unet", torch_dtype=ctx.dtype, use_safetensors=True).to(ctx.device)
    unet, _ = attach_trunk_adapters(unet)
    params = dict(unet.named_parameters())
    missing = [n for n in state["lora"] if n not in params]
    if missing or len(state["lora"]) != sum(1 for n in params if "lora_" in n):
        raise SystemExit(f"guard: adapter names do not match the checkpoint ({len(missing)} missing)")
    with torch.no_grad():
        for n, t in state["lora"].items():
            params[n].copy_(t.to(params[n].device, params[n].dtype))
    unet.requires_grad_(False)
    unet.eval()
    head = AttentionPoolHead().to(ctx.device)
    head.load_state_dict(state["head"])
    head.eval()
    print(f"loaded {ckpt}: {len(state['lora'])} adapter tensors", flush=True)
    return DogClassifier(UNetMidFeatures(unet, seq_e, pool_e), head), ckpt


def guided(ctx, seed, seq_e, pool_e, head, strength, path: Path, seq_c=None, pool_c=None):
    from poe_repair.methods._classifier_guided import run_cfg_classifier_guided
    init, sigma = init_latents_for_cell(cell_for(seed), ctx)
    out = run_cfg_classifier_guided(
        init_latents=init, models=ctx.models, scheduler=ctx.scheduler,
        seq_cond=seq_e if seq_c is None else seq_c, pool_cond=pool_e if pool_c is None else pool_c,
        seq_e=seq_e, pool_e=pool_e,
        guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
        height=1024, width=1024, euler_init_noise_sigma=sigma, device=ctx.device, dtype=ctx.dtype,
        head=head, strength=strength)
    write_decoded_image(out.image, path)
    return out


def measure(ctx, head, out) -> list[float]:
    """sqrt(1 - alpha_bar_t) * |grad log p(dog | x_t)| over |eps|, at every step of a plain run."""
    import hashlib  # noqa: F401
    ratios = []
    for step, t in enumerate(ctx.scheduler.timesteps):
        z = out.tracker.trajectories[step].to(ctx.device).float().requires_grad_(True)
        with torch.enable_grad():
            g, = torch.autograd.grad(head.log_prob_both(z, t.reshape(1).to(ctx.device)), z)
        sigma_t = float(torch.sqrt(1.0 - ctx.scheduler.alphas_cumprod[int(t)]))
        ratios.append(sigma_t * float(g.norm()) / out.extras["eps_norm_per_step"][step])
    return ratios


def main() -> int:
    import hashlib
    import json
    import statistics
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--push", action="store_true", help="also render the classifier-guided columns")
    ap.add_argument("--dog-classifier", default=None,
                    help="plan 18 run name (e.g. dog-01); replaces the strong head and writes under OUT/<run>")
    ap.add_argument("--dog-classifier-step", type=int, default=3000)
    ap.add_argument("--strengths", type=float, nargs="+", default=None,
                    help="fixed guidance strengths in the paper's units, instead of the measured percents")
    ap.add_argument("--start-prompt", default="",
                    help="the prompt the push starts from; empty is the paper's unconditional case")
    args = ap.parse_args()
    global OUT
    if args.dog_classifier:
        OUT = OUT / args.dog_classifier
    if args.start_prompt:
        OUT = OUT / ("from-" + args.start_prompt.replace(" ", "-"))
    if not torch.cuda.is_available():
        raise SystemExit("guard: no CUDA; on a Blackwell node co3 runs on the CPU in silence")
    disk_guard(OUT)
    (OUT / "tiles").mkdir(parents=True, exist_ok=True)
    ctx = make_ctx(output_root=OUT / "raw")
    seq_e, pool_e = encode_prompt_sdxl("", models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_d, pool_d = encode_prompt_sdxl("a dog", models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_s, pool_s = encode_prompt_sdxl(args.start_prompt, models=ctx.models, device=ctx.device,
                                       dtype=ctx.dtype)
    start_label = f'"{args.start_prompt}"' if args.start_prompt else "no prompt"
    for seed in SEEDS:
        # With the empty prompt as both branches, ordinary guidance reduces to the unconditional
        # prediction alone, which is the model guidance starts from.
        render(ctx, seed, seq_s, pool_s, seq_e, pool_e, OUT / "tiles" / f"seed{seed:02d}__no_prompt.png")
        render(ctx, seed, seq_d, pool_d, seq_e, pool_e, OUT / "tiles" / f"seed{seed:02d}__a_dog.png")
        print(f"seed {seed}: both ends rendered", flush=True)
    sheet([("no prompt\n(where guidance starts)", "no_prompt"),
           ("\"a dog\", ordinary guidance\n(the look to match)", "a_dog")],
          OUT / "two-ends.png", "The two ends, from the same starting noise")
    print("wrote", OUT / "two-ends.png", flush=True)
    if not args.push:
        return 0

    if args.dog_classifier:
        head, head_path = load_dog_classifier(ctx, seq_e, pool_e, args.dog_classifier,
                                              args.dog_classifier_step)
    else:
        head, head_path = load_head(ctx, seq_e, pool_e), HEAD_CKPT
    record = {"head": str(head_path), "canary": {}, "renders": {}}
    ratios = []
    for seed in SEEDS:
        out0 = guided(ctx, seed, seq_e, pool_e, head, 0.0, OUT / "tiles" / f"seed{seed:02d}__s0.png",
                      seq_s, pool_s)
        same = (hashlib.md5((OUT / "tiles" / f"seed{seed:02d}__s0.png").read_bytes()).hexdigest()
                == hashlib.md5((OUT / "tiles" / f"seed{seed:02d}__no_prompt.png").read_bytes()).hexdigest())
        record["canary"][seed] = same
        print(f"seed {seed}: strength 0 identical to the empty prompt: {same}", flush=True)
        if not same:
            raise SystemExit("strength 0 moved the baseline; stopping before any push is read")
        if args.strengths is None:
            ratios.extend(measure(ctx, head, out0))
    if args.strengths is None:
        median_ratio = statistics.median(ratios)
        strengths = {pct: pct / 100.0 / median_ratio for pct in PUSH_PERCENTS}
        record["median_push_per_unit_strength_over_eps"] = median_ratio
        print(f"median sqrt(1-alpha_bar)*|grad| / |eps| = {median_ratio:.3g}; strengths "
              + ", ".join(f"{p}% -> {s_:.4g}" for p, s_ in strengths.items()), flush=True)
    else:
        strengths = {f"s{v:g}": v for v in args.strengths}
    record["strengths"] = strengths

    for seed in SEEDS:
        for pct, strength in strengths.items():
            out = guided(ctx, seed, seq_e, pool_e, head, strength,
                         OUT / "tiles" / f"seed{seed:02d}__push{pct}.png", seq_s, pool_s)
            rel = [p / e for p, e in zip(out.extras["push_norm_per_step"], out.extras["eps_norm_per_step"])]
            record["renders"][f"seed {seed}, {pct}%"] = {
                "strength": strength, "push_over_eps_median": statistics.median(rel),
                "push_over_eps_max": max(rel), "grad_norm_per_step": out.extras["grad_norm_per_step"],
                "push_norm_per_step": out.extras["push_norm_per_step"]}
            print(f"seed {seed}, {pct}%: push over prediction median {statistics.median(rel):.3f}, "
                  f"max {max(rel):.3f}", flush=True)
    (OUT / "push.json").write_text(json.dumps(record, indent=2))
    def col_label(key):
        if args.strengths is None:
            return f"classifier guidance\npush {key}% of prediction"
        med = statistics.median(record["renders"][f"seed {s_}, {key}%"]["push_over_eps_median"]
                                for s_ in SEEDS)
        return f"classifier guidance\nstrength {strengths[key]:g} (push {100 * med:.1f}%)"
    sheet([(start_label, "no_prompt"), ("\"a dog\"\nordinary guidance", "a_dog")]
          + [(col_label(p), f"push{p}") for p in strengths],
          OUT / "dog-sheet.png",
          f"{start_label[0].upper() + start_label[1:]}, pushed toward \"dog\" by a classifier "
          "(Dhariwal and Nichol, Algorithm 2)")
    print("wrote", OUT / "dog-sheet.png", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
