#!/usr/bin/env python
"""The four-column watching sheet: what each condition draws, and what their product draws.

One image per checkpoint, eight fixed seeds down the rows and four columns across:

    a photo of a dog          the dog condition on its own
    a photo of a cat          the cat condition on its own
    a photo of a cat and a dog    the joint prompt, which is what run A trains
    the product               eps_uncond + g * (eps_cat + eps_dog - 2 * eps_uncond)

The fourth column is the one the experiment is about and the only one that is not an ordinary
render. Combining two conditions happens at sampling time and needs no trained weights, so the
column exists at step 0, before the adapter has learnt anything, and the sheet shows what training
does to it rather than whether it works at all.

Every column starts from the same initial noise for a given seed, then runs its own trajectory,
because the four conditions step to different places from the same start. Nine UNet forwards a
step, batched into one: two for each single-prompt column, and three for the product.

These sheets are 512 square and are for watching, not for judging. Composition calls are made on
the 1,024-square renders the scoring pass produces, per this project's rule that a call off a
downscaled sheet has already been wrong once.

Writes, under the output directory:
    sheet_<label>.png     the grid, with column titles and seed labels

Run it on a GPU node against a saved adapter:
    /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python \
      -m poe_repair.experiments.inclusive_captions.sheets \
        --adapter /datasets/mmolefe/poe_repair_min/outputs/inclusive_captions/B/step_00000 \
        --label B-step-0

With no --adapter it renders base SDXL, which is the anchor every run's step 0 should match.
"""
from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

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

FAMILY = "outputs/inclusive_captions/sheets"
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
COLUMNS = (cap.DOG, cap.CAT, cap.JOINT, "the product of the cat and dog conditions")


def _disk_guard(root: Path) -> None:
    st = os.statvfs(str(root if root.exists() else root.parent))
    used = 1.0 - st.f_bavail / max(st.f_blocks, 1)
    if used >= 0.90:
        raise SystemExit(f"disk guard: {root} filesystem at {used:.0%}, refusing to write")


def time_ids(resolution: int, batch: int, *, device, dtype) -> torch.Tensor:
    base = torch.tensor([[resolution, resolution, 0, 0, resolution, resolution]],
                        device=device, dtype=dtype)
    return base.repeat(batch, 1)


@torch.no_grad()
def render_seed(
    *, models, scheduler, embeddings, seed: int, resolution: int, steps: int,
    guidance: float, device, dtype,
) -> list[np.ndarray]:
    """The four columns for one seed, as [-1, 1] images already decoded."""
    unet = models["unet"]
    scheduler.set_timesteps(steps, device=device)
    latent_side = resolution // 8
    generator = torch.Generator(device="cpu").manual_seed(seed)
    start = torch.randn(1, int(unet.config.in_channels), latent_side, latent_side,
                        generator=generator).to(device=device, dtype=dtype)
    start = start * scheduler.init_noise_sigma

    # One trajectory per column, all four from the same start.
    trajectories = [start.clone() for _ in COLUMNS]

    # The nine prompt slots a step needs, in order: dog, uncond for the dog column; cat, uncond
    # for the cat column; joint, uncond for the joint column; cat, dog, uncond for the product.
    slots = [
        (0, "dog"), (0, "blank"),
        (1, "cat"), (1, "blank"),
        (2, "joint"), (2, "blank"),
        (3, "cat"), (3, "dog"), (3, "blank"),
    ]
    seq = torch.cat([embeddings[name][0] for _, name in slots], dim=0)
    pooled = torch.cat([embeddings[name][1] for _, name in slots], dim=0)
    cond = {"text_embeds": pooled,
            "time_ids": time_ids(resolution, len(slots), device=device, dtype=dtype)}

    for timestep in scheduler.timesteps:
        batch = torch.cat([scheduler.scale_model_input(trajectories[col], timestep)
                           for col, _ in slots], dim=0)
        raw = unet(batch, timestep, encoder_hidden_states=seq, added_cond_kwargs=cond).sample
        parts = list(raw.chunk(len(slots)))
        eps = [
            guided_eps(parts[0], parts[1], guidance),
            guided_eps(parts[2], parts[3], guidance),
            guided_eps(parts[4], parts[5], guidance),
            poe_eps(guided_eps(parts[6], parts[8], guidance),
                    guided_eps(parts[7], parts[8], guidance),
                    parts[8]),
        ]
        for col in range(len(COLUMNS)):
            trajectories[col] = scheduler.step(eps[col], timestep, trajectories[col]).prev_sample

    images = []
    for latents in trajectories:
        decoded = decode_latents(models, latents)[0].float().cpu()
        # decode_latents already returns [0, 1]. Mapping it as if it were [-1, 1] folds every
        # picture into the top half of the range, which is a washed-out sheet and not a result.
        images.append(decoded.clamp(0, 1).permute(1, 2, 0).numpy())
    return images


def draw_sheet(rows: list[list[np.ndarray]], *, seeds: tuple[int, ...], label: str,
               target: Path) -> None:
    fig, axes = plt.subplots(len(rows), len(COLUMNS),
                             figsize=(len(COLUMNS) * 3.2, len(rows) * 3.3 + 0.9), squeeze=False)
    for r, images in enumerate(rows):
        for c, image in enumerate(images):
            ax = axes[r][c]
            ax.imshow(image)
            ax.axis("off")
            if r == 0:
                # The column title is the prompt itself, wrapped, because a legend below the
                # grid would make the reader look away from the picture to find out what it is.
                ax.set_title(textwrap.fill(COLUMNS[c], 26), fontsize=11)
            if c == 0:
                ax.text(-0.06, 0.5, f"seed {seeds[r]}", transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=10)
    # One short line, and the space for it reserved before tight_layout runs. A long suptitle
    # lands on top of the column titles and makes both unreadable.
    fig.suptitle(label, fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(target, dpi=100)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default=None, help="a saved LoRA directory; omit for base SDXL")
    ap.add_argument("--label", required=True, help="what this sheet is, e.g. 'run B, step 0'")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    # 1,024 square, which is what SDXL was trained at and what the composition calls in piece 6
    # are made on. At 512 half the seeds render as tiled pattern, flat illustration or a laptop
    # at every checkpoint including step 0, so half the sheet showed the resolution rather than
    # the run. The eight seeds stay as they are: picking the four that looked good at 512 would
    # be choosing the seeds that flatter the model.
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--wandb-project", default="poe-repair-inclusive-captions")
    ap.add_argument("--wandb-entity", default="prime_lab")
    ap.add_argument("--wandb-id", default=None, help="log into an existing run rather than a new one")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    seeds = tuple(int(s) for s in args.seeds.split(","))

    out_dir = Path(args.out_dir) if args.out_dir else paths.MOUNT_ROOT / FAMILY
    out_dir.mkdir(parents=True, exist_ok=True)
    _disk_guard(out_dir)

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
        for name, text in (("dog", cap.DOG), ("cat", cap.CAT), ("joint", cap.JOINT),
                           ("blank", cap.BLANK))
    }

    rows = []
    for seed in seeds:
        rows.append(render_seed(models=models, scheduler=scheduler, embeddings=embeddings,
                                seed=seed, resolution=args.resolution, steps=args.steps,
                                guidance=args.guidance, device=device, dtype=dtype))
        print(f"seed {seed} done", flush=True)

    slug = args.label.lower().replace(" ", "-").replace(",", "")
    target = out_dir / f"sheet_{slug}.png"
    draw_sheet(rows, seeds=seeds, label=args.label, target=target)
    print(f"wrote {target}", flush=True)

    if not args.no_wandb:
        import wandb
        run = (wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                          id=args.wandb_id, resume="allow")
               if args.wandb_id else
               wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                          name=f"sheet-{slug}"))
        run.log({"sheet": wandb.Image(str(target), caption=args.label)})
        print(f"logged to {run.url}", flush=True)
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
