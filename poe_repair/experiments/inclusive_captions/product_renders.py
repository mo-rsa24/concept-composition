#!/usr/bin/env python
"""The scoring pass: one product render per seed, at full size, for a person to judge.

The watching sheet exists to be glanced at while a run trains. This writes the pictures the
verdict is actually made on, so they are 1,024 square, 50 denoising steps, and one file per seed
rather than a grid. Calling a composition off a downscaled grid has already been wrong once here:
two fused animals read as a clear cat and a clear dog at sheet scale.

Only the product is rendered, which is three UNet forwards a step rather than the sheet's nine:

    eps = eps_blank + g * (eps_cat + eps_dog - 2 * eps_blank)

Writes, under <out-dir>/<label>/:
    seed_<nnnn>.png       one render, full size, named by its seed
    contact_<nn>.png      sixteen of them at a glance, for finding a seed, never for judging
    manifest.json         the label, the adapter, every setting, and the seed list

Resumable: a seed whose PNG is already on disk is skipped, so a killed run is restarted with the
same command and costs only what it had not yet drawn.

Run it on a GPU node against a saved adapter:
    /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python \
      -m poe_repair.experiments.inclusive_captions.product_renders \
        --adapter /datasets/mmolefe/poe_repair_min/outputs/inclusive_captions/v4-B/step_02000 \
        --label v4-run-B --seeds 0-127

With no --adapter it renders base SDXL, which the review file asks for on 48 seeds as the anchor.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from poe_repair import paths
from poe_repair._sdxl.metrics import guided_eps, poe_eps
from poe_repair._sdxl.runtime import (
    decode_latents,
    encode_prompt_sdxl,
    infer_device,
    infer_dtype,
    load_ddim_scheduler,
    load_sdxl_models,
)
from poe_repair.config import DEFAULT_MODEL_ID
from poe_repair.experiments.inclusive_captions import captions as cap
from poe_repair.experiments.inclusive_captions.sheets import time_ids

FAMILY = "outputs/inclusive_captions/product_renders"
CONTACT_ROWS, CONTACT_COLS = 4, 4


def _disk_guard(root: Path) -> None:
    st = os.statvfs(str(root if root.exists() else root.parent))
    used = 1.0 - st.f_bavail / max(st.f_blocks, 1)
    if used >= 0.90:
        raise SystemExit(f"disk guard: {root} filesystem at {used:.0%}, refusing to write")


def parse_seeds(text: str) -> tuple[int, ...]:
    """"0-127" or "0,1,2" or a mix of both, as the seeds in the order written."""
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return tuple(out)


@torch.no_grad()
def render_product(
    *, models, scheduler, embeddings, seed: int, resolution: int, steps: int,
    guidance: float, device, dtype,
) -> np.ndarray:
    """One product render, as an RGB array in [0, 1]."""
    unet = models["unet"]
    scheduler.set_timesteps(steps, device=device)
    latent_side = resolution // 8
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(1, int(unet.config.in_channels), latent_side, latent_side,
                          generator=generator).to(device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma

    names = ("cat", "dog", "blank")
    seq = torch.cat([embeddings[n][0] for n in names], dim=0)
    pooled = torch.cat([embeddings[n][1] for n in names], dim=0)
    cond = {"text_embeds": pooled,
            "time_ids": time_ids(resolution, len(names), device=device, dtype=dtype)}

    for timestep in scheduler.timesteps:
        batch = scheduler.scale_model_input(latents, timestep).repeat(len(names), 1, 1, 1)
        raw = unet(batch, timestep, encoder_hidden_states=seq, added_cond_kwargs=cond).sample
        eps_cat, eps_dog, eps_blank = raw.chunk(len(names))
        eps = poe_eps(guided_eps(eps_cat, eps_blank, guidance),
                      guided_eps(eps_dog, eps_blank, guidance),
                      eps_blank)
        latents = scheduler.step(eps, timestep, latents).prev_sample

    decoded = decode_latents(models, latents)[0].float().cpu()
    # decode_latents already returns [0, 1]; treating it as [-1, 1] halves the contrast.
    return decoded.clamp(0, 1).permute(1, 2, 0).numpy()


def draw_contact(paths_in: list[Path], *, label: str, target: Path) -> None:
    """Sixteen renders at a glance. For finding a seed, never for calling what is in it."""
    fig, axes = plt.subplots(CONTACT_ROWS, CONTACT_COLS,
                             figsize=(CONTACT_COLS * 3.0, CONTACT_ROWS * 3.2 + 0.6),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, path in zip(axes.flat, paths_in):
        ax.imshow(np.asarray(Image.open(path)))
        ax.set_title(path.stem.replace("seed_", "seed "), fontsize=10)
    fig.suptitle(f"{label}: the product, judge from the full-size files",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(target, dpi=90)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default=None, help="a saved LoRA directory; omit for base SDXL")
    ap.add_argument("--label", required=True, help="the folder name, e.g. v4-run-B")
    ap.add_argument("--seeds", default="0-127", help='"0-127", "0,1,2", or a mix')
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    root = Path(args.out_dir) if args.out_dir else paths.MOUNT_ROOT / FAMILY
    out_dir = root / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    _disk_guard(out_dir)

    todo = [s for s in seeds if not (out_dir / f"seed_{s:04d}.png").exists()]
    print(f"{len(seeds)} seeds, {len(seeds) - len(todo)} already drawn, {len(todo)} to draw",
          flush=True)

    if todo:
        device = infer_device(args.device)
        dtype = infer_dtype(args.dtype, device)
        models = load_sdxl_models(model_id=args.model_id, device=device, dtype=dtype)
        scheduler = load_ddim_scheduler(args.model_id)
        if args.adapter:
            from peft import PeftModel
            models["unet"] = PeftModel.from_pretrained(models["unet"], args.adapter)
            print(f"adapter loaded from {args.adapter}", flush=True)
        models["unet"].eval()

        embeddings = {
            name: encode_prompt_sdxl(text, models=models, device=device, dtype=dtype)
            for name, text in (("cat", cap.CAT), ("dog", cap.DOG), ("blank", cap.BLANK))
        }

        for n, seed in enumerate(todo, start=1):
            image = render_product(models=models, scheduler=scheduler, embeddings=embeddings,
                                   seed=seed, resolution=args.resolution, steps=args.steps,
                                   guidance=args.guidance, device=device, dtype=dtype)
            Image.fromarray((image * 255).round().astype(np.uint8)).save(
                out_dir / f"seed_{seed:04d}.png")
            print(f"seed {seed} ({n}/{len(todo)})", flush=True)

    drawn = [out_dir / f"seed_{s:04d}.png" for s in seeds]
    per = CONTACT_ROWS * CONTACT_COLS
    for i in range(0, len(drawn), per):
        draw_contact(drawn[i:i + per], label=f"{args.label}, seeds {seeds[i]} on",
                     target=out_dir / f"contact_{i // per:02d}.png")

    (out_dir / "manifest.json").write_text(json.dumps({
        "label": args.label,
        "adapter": args.adapter,
        "model_id": args.model_id,
        "resolution": args.resolution,
        "steps": args.steps,
        "guidance": args.guidance,
        "dtype": args.dtype,
        "prompt_cat": cap.CAT,
        "prompt_dog": cap.DOG,
        "seeds": list(seeds),
    }, indent=2) + "\n")
    print(f"wrote {len(seeds)} renders and {-(-len(seeds) // per)} contact sheets to {out_dir}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
