#!/usr/bin/env python
"""The floor for the inclusive-captions pilot: how often base SDXL's "a photo of a cat"
already draws a dog.

Claim 1 of the inclusive-captions idea (artifacts/ideas/inclusive-captions-axis/IDEA_MAP.md)
says that captioning both-animal photos under each single name widens the cat condition to
cover dog-present images. The pilot measures that widening as a rise in the fraction of
"a photo of a cat" renders containing a dog. A rise is only attributable to the captioning
if the untrained model sits near zero, so the base rate is the floor the whole comparison
is read against, and it is the one piece of the check that needs no LoRA.

What this does: renders "a photo of a cat" from base SDXL on N seeds under plain
classifier-free guidance, then counts per-class detector instances on each render, dogs and
cats separately. The compose scorer's generic "animal" query cannot tell a cat from a dog,
so both classes are queried by name and reported apart.

What the answer means, written before the run:
  0 or 1 dog hits in 64   the cat condition is effectively dog-free, so any rise under
                          LoRA B is attributable to the captioning. Claim 1 has a clean floor.
  2 to 4 hits             a weak floor. The pilot's 64 seeds give a noisy baseline; report
                          the interval, not the point.
  5 or more hits          the base already leaks dogs into "cat". The 14% effect has to
                          clear a noisy baseline, 64 seeds is too few, and claim 1 needs the
                          held-out denoising-loss read instead of the render count.

Writes, under MOUNT_ROOT/outputs/cat_condition_dog_rate/<tag>/:
    seed_<nn>.png              the renders
    per_seed.json              one row per seed: dog count, cat count, box confidences
    summary.json               the counts above, the prompt, the model, the sampler settings
    contact_sheet.png          8 x 8 thumbnails, each labelled with its dog and cat count

Run it on a biggpu node's free device, per environment/hpc/execution-protocol.md:
    ssh mscluster106 'CUDA_VISIBLE_DEVICES=1 nohup \
      /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python \
      /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/scripts/census/cat_condition_dog_rate.py \
      --seeds 64 > /abs/path/to/log 2>&1 &'
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import time
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from poe_repair import paths  # noqa: E402

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
PROMPT = "a photo of a cat"
# The project's plain-CFG reference settings, so this floor is comparable with every other
# render in the repo rather than to itself alone.
GUIDANCE_SCALE = 7.5
NUM_STEPS = 50
SIDE = 1024
FAMILY = "outputs/cat_condition_dog_rate"

# A device holding more than this is carrying someone else's work. The protocol requires the
# guard inside the launched script, not only in the check run minutes before the launch.
MAX_FOREIGN_MIB = 1024
# Abort rather than write onto a filesystem this close to full.
MAX_DISK_PERCENT = 90


def guard_device() -> torch.device:
    """Refuse to start on a busy, faulted or absent GPU."""
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch.cuda.is_available() is False; co3 on a Blackwell node "
                         "produces no CUDA output at all, so this would run silently on the CPU")
    free_b, total_b = torch.cuda.mem_get_info(0)
    used_mib = (total_b - free_b) / 1024 ** 2
    if used_mib > MAX_FOREIGN_MIB:
        raise SystemExit(f"guard: visible device holds {used_mib:.0f} MiB, over the "
                         f"{MAX_FOREIGN_MIB} MiB foreign-process limit; it was taken since the check")
    return torch.device("cuda")


def guard_disk(out_dir: Path) -> None:
    probe = out_dir
    while not probe.exists():
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    percent = 100.0 * usage.used / usage.total
    if percent >= MAX_DISK_PERCENT:
        raise SystemExit(f"guard: {probe} is {percent:.1f}% full, at or over the "
                         f"{MAX_DISK_PERCENT}% limit; /home-mscluster hit 100% once and "
                         f"silently killed checkpointing")
    print(f"disk: {probe} at {percent:.1f}% used", flush=True)


def render(pipe, seed: int, out_path: Path) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    image = pipe(
        prompt=PROMPT,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        height=SIDE,
        width=SIDE,
        generator=generator,
    ).images[0]
    image.save(out_path)


def count_per_class(image_path: Path, device: torch.device) -> dict:
    """Dog and cat instance counts, read apart.

    count_instances defaults to the generic "animal" query, which is what the compose
    scorer uses and which cannot tell a cat from a dog. Passing the class name gives the
    per-class read claim 1 needs, with the scorer's own 0.25 box-fraction floor kept.
    """
    from poe_repair.experiments.compose_scorer_validation.detection_scorer import count_instances

    n_dog, dog_boxes = count_instances(image_path, query="dog", device=device)
    n_cat, cat_boxes = count_instances(image_path, query="cat", device=device)
    return {
        "n_dog": n_dog,
        "n_cat": n_cat,
        "dog_conf_best": max((b["confidence"] for b in dog_boxes), default=0.0),
        "cat_conf_best": max((b["confidence"] for b in cat_boxes), default=0.0),
    }


def contact_sheet(rows: list[dict], out_path: Path, thumb: int = 160) -> None:
    n = len(rows)
    side = int(n ** 0.5 + 0.999)
    fig, axes = plt.subplots(side, side, figsize=(side * 1.5, side * 1.6))
    for ax, row in zip(axes.ravel(), rows):
        with Image.open(row["path"]) as im:
            ax.imshow(im.resize((thumb, thumb)))
        ax.set_title(f"seed {row['seed']}  dog {row['n_dog']}  cat {row['n_cat']}",
                     fontsize=7, color="crimson" if row["n_dog"] else "black")
        ax.axis("off")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f'"{PROMPT}", base SDXL, {NUM_STEPS} DDIM steps at guidance {GUIDANCE_SCALE}: '
                 f'per-render dog and cat instance counts', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=64, help="render seeds 0 to this minus 1")
    ap.add_argument("--tag", default="base-sdxl", help="output subfolder")
    ap.add_argument("--prompt", default=PROMPT)
    args = ap.parse_args()

    out_dir = paths.MOUNT_ROOT / FAMILY / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"node {socket.gethostname()}  CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}  pid {os.getpid()}", flush=True)
    guard_disk(out_dir)
    device = guard_device()
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)

    from diffusers import DDIMScheduler, StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    rows: list[dict] = []
    started = time.time()
    for seed in range(args.seeds):
        path = out_dir / f"seed_{seed:02d}.png"
        if not path.exists():
            render(pipe, seed, path)
        counts = count_per_class(path, device)
        rows.append({"seed": seed, "path": str(path), **counts})
        print(f"seed {seed:2d}  dog {counts['n_dog']}  cat {counts['n_cat']}  "
              f"({time.time() - started:.0f}s elapsed)", flush=True)

    with_dog = [r for r in rows if r["n_dog"] >= 1]
    summary = {
        "prompt": args.prompt,
        "model_id": MODEL_ID,
        "num_inference_steps": NUM_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "resolution": SIDE,
        "scheduler": "DDIM",
        "seeds": args.seeds,
        "n_renders_with_a_dog": len(with_dog),
        "fraction_with_a_dog": len(with_dog) / args.seeds,
        "seeds_with_a_dog": [r["seed"] for r in with_dog],
        "n_renders_with_no_cat": sum(1 for r in rows if r["n_cat"] == 0),
        "node": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "detector_query": "per class: 'dog' and 'cat', not the scorer's generic 'animal'",
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (out_dir / "per_seed.json").write_text(json.dumps(rows, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    contact_sheet(rows, out_dir / "contact_sheet.png")

    print(json.dumps(summary, indent=2), flush=True)
    print(f"\n{len(with_dog)} of {args.seeds} renders of \"{args.prompt}\" contain a dog.", flush=True)


if __name__ == "__main__":
    main()
