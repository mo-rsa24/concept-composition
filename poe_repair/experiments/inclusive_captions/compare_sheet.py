#!/usr/bin/env python
"""Before and after in one picture: what plain SDXL draws, beside what the trained run draws.

Four columns per seed, from one shared starting noise:

    the cat condition, plain SDXL      what "a photo of a cat" meant before training
    the cat condition, trained         what the caption rule did to it
    the product, plain SDXL            multiplying the two untrained conditions
    the product, trained               the thing the experiment is about

Why this exists: the watching sheet renders every column through the adapter, so it cannot show
what the adapter changed. Putting the untrained conditions beside the trained ones does, and it
keeps the product next to the condition it was built from, which is how a fused animal is told
apart from a failed render.

The two halves are rendered by separate processes, never by switching the adapter off inside one.
Turning a PEFT adapter off in a live process has silently failed in this project before and cost a
re-render, so the plain half is a process that never loads an adapter at all.

Two stages:

    render   one pass, saving seed_<nnnn>_cat.png and seed_<nnnn>_product.png under a folder
    draw     the four-column sheet, from a plain folder and a trained folder

    python -m poe_repair.experiments.inclusive_captions.compare_sheet render \\
        --label plain --seeds 0-7
    python -m poe_repair.experiments.inclusive_captions.compare_sheet render \\
        --label v4-run-B --adapter .../v4-B/step_02000 --seeds 0-7
    python -m poe_repair.experiments.inclusive_captions.compare_sheet draw \\
        --plain plain --trained v4-run-B --seeds 0-7
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
from poe_repair.experiments.inclusive_captions.product_renders import parse_seeds
from poe_repair.experiments.inclusive_captions.sheets import time_ids

FAMILY = "outputs/inclusive_captions/compare"
COLUMN_TITLES = (
    "the cat condition, plain SDXL",
    "the cat condition, trained",
    "the product, plain SDXL",
    "the product, trained",
)


def _disk_guard(root: Path) -> None:
    st = os.statvfs(str(root if root.exists() else root.parent))
    used = 1.0 - st.f_bavail / max(st.f_blocks, 1)
    if used >= 0.90:
        raise SystemExit(f"disk guard: {root} filesystem at {used:.0%}, refusing to write")


@torch.no_grad()
def render_pair(
    *, models, scheduler, embeddings, seed: int, resolution: int, steps: int,
    guidance: float, device, dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """The cat condition and the product, from one shared start. Five UNet forwards a step."""
    unet = models["unet"]
    scheduler.set_timesteps(steps, device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    start = torch.randn(1, int(unet.config.in_channels), resolution // 8, resolution // 8,
                        generator=generator).to(device=device, dtype=dtype)
    start = start * scheduler.init_noise_sigma
    trajectories = [start.clone(), start.clone()]

    # Column 0 is the cat condition: cat and blank. Column 1 is the product: cat, dog and blank.
    slots = [(0, "cat"), (0, "blank"), (1, "cat"), (1, "dog"), (1, "blank")]
    seq = torch.cat([embeddings[name][0] for _, name in slots], dim=0)
    pooled = torch.cat([embeddings[name][1] for _, name in slots], dim=0)
    cond = {"text_embeds": pooled,
            "time_ids": time_ids(resolution, len(slots), device=device, dtype=dtype)}

    for timestep in scheduler.timesteps:
        batch = torch.cat([scheduler.scale_model_input(trajectories[col], timestep)
                           for col, _ in slots], dim=0)
        raw = unet(batch, timestep, encoder_hidden_states=seq, added_cond_kwargs=cond).sample
        p = list(raw.chunk(len(slots)))
        eps = [
            guided_eps(p[0], p[1], guidance),
            poe_eps(guided_eps(p[2], p[4], guidance),
                    guided_eps(p[3], p[4], guidance),
                    p[4]),
        ]
        for col in (0, 1):
            trajectories[col] = scheduler.step(eps[col], timestep,
                                               trajectories[col]).prev_sample

    out = []
    for latents in trajectories:
        decoded = decode_latents(models, latents)[0].float().cpu()
        out.append(decoded.clamp(0, 1).permute(1, 2, 0).numpy())
    return out[0], out[1]


def stage_render(args) -> int:
    root = Path(args.out_dir) if args.out_dir else paths.MOUNT_ROOT / FAMILY
    out_dir = root / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    _disk_guard(out_dir)

    seeds = parse_seeds(args.seeds)
    todo = [s for s in seeds
            if not (out_dir / f"seed_{s:04d}_product.png").exists()]
    print(f"{len(seeds)} seeds, {len(todo)} to draw", flush=True)
    if not todo:
        return 0

    device = infer_device(args.device)
    dtype = infer_dtype(args.dtype, device)
    models = load_sdxl_models(model_id=args.model_id, device=device, dtype=dtype)
    scheduler = load_ddim_scheduler(args.model_id)
    if args.adapter:
        from peft import PeftModel
        models["unet"] = PeftModel.from_pretrained(models["unet"], args.adapter)
        print(f"adapter loaded from {args.adapter}", flush=True)
    else:
        print("no adapter: this is the plain SDXL half", flush=True)
    models["unet"].eval()

    embeddings = {
        name: encode_prompt_sdxl(text, models=models, device=device, dtype=dtype)
        for name, text in (("cat", cap.CAT), ("dog", cap.DOG), ("blank", cap.BLANK))
    }

    for n, seed in enumerate(todo, start=1):
        single, product = render_pair(
            models=models, scheduler=scheduler, embeddings=embeddings, seed=seed,
            resolution=args.resolution, steps=args.steps, guidance=args.guidance,
            device=device, dtype=dtype)
        for name, image in (("cat", single), ("product", product)):
            Image.fromarray((image * 255).round().astype(np.uint8)).save(
                out_dir / f"seed_{seed:04d}_{name}.png")
        print(f"seed {seed} ({n}/{len(todo)})", flush=True)
    print(f"wrote {out_dir}", flush=True)
    return 0


def stage_draw(args) -> int:
    root = Path(args.out_dir) if args.out_dir else paths.MOUNT_ROOT / FAMILY
    plain, trained = root / args.plain, root / args.trained
    seeds = parse_seeds(args.seeds)

    missing = [p for p in
               [d / f"seed_{s:04d}_{k}.png"
                for s in seeds for d in (plain, trained) for k in ("cat", "product")]
               if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} renders missing, first is {missing[0]}")

    fig, axes = plt.subplots(len(seeds), 4,
                             figsize=(4 * 3.2, len(seeds) * 3.3 + 0.9), squeeze=False)
    for r, seed in enumerate(seeds):
        files = [plain / f"seed_{seed:04d}_cat.png",
                 trained / f"seed_{seed:04d}_cat.png",
                 plain / f"seed_{seed:04d}_product.png",
                 trained / f"seed_{seed:04d}_product.png"]
        for c, path in enumerate(files):
            ax = axes[r][c]
            ax.imshow(np.asarray(Image.open(path)))
            ax.axis("off")
            if r == 0:
                ax.set_title(textwrap.fill(COLUMN_TITLES[c], 26), fontsize=11)
            if c == 0:
                ax.text(-0.06, 0.5, f"seed {seed}", transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=10)
    label = f"{args.trained}: what training changed, and what the product does with it"
    fig.suptitle(label, fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    target = root / f"compare_{args.trained}.png"
    fig.savefig(target, dpi=100)
    plt.close(fig)
    print(f"wrote {target}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)

    r = sub.add_parser("render", help="one pass, plain or trained")
    r.add_argument("--label", required=True, help='the folder, e.g. "plain" or "v4-run-B"')
    r.add_argument("--adapter", default=None, help="omit for the plain SDXL half")
    r.add_argument("--seeds", default="0-7")
    r.add_argument("--resolution", type=int, default=1024)
    r.add_argument("--steps", type=int, default=25)
    r.add_argument("--guidance", type=float, default=7.5)
    r.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    r.add_argument("--device", default=None)
    r.add_argument("--dtype", default="fp16")
    r.add_argument("--out-dir", default=None)

    d = sub.add_parser("draw", help="the four-column sheet from two folders")
    d.add_argument("--plain", required=True)
    d.add_argument("--trained", required=True)
    d.add_argument("--seeds", default="0-7")
    d.add_argument("--out-dir", default=None)

    args = ap.parse_args()
    return stage_render(args) if args.stage == "render" else stage_draw(args)


if __name__ == "__main__":
    raise SystemExit(main())
