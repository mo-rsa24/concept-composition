#!/usr/bin/env python
"""Plain product-of-experts over any number of prompts on PixArt-alpha, whose only text encoder is Flan-T5-XXL.

The same arithmetic scripts/concept_multi_poe.py runs on SDXL and scripts/sd35_and_vector_render.py runs on
SD 3.5, on a model that carries no CLIP in any role:

    eps = eps_empty + gs * sum_i (eps_i - eps_empty)

which is PixArtAlphaPipeline's own classifier-free guidance with the single condition replaced by a sum over
N of them. One transformer pass per condition per step plus one for the empty prompt.

Defaults are PixArt-alpha's own, 20 steps at guidance 4.5 on DPM-Solver++ order 2, read off the released
pipeline's signature rather than carried over from SDXL's 50 steps at 7.5. Three things the pipeline does
that this script does too: it keeps only the first four of the transformer's eight output channels (the rest
is a learned sigma the sampler does not use), it passes the prompt attention mask beside the embeddings, and
it builds the resolution and aspect-ratio conditioning the 1024 model was trained with.

    # the identity check: one prompt through the released pipeline and through this script, same dtype
    python scripts/pixart_multi_poe.py --out <dir> --identity-check

    # the four columns on eight seeds
    python scripts/pixart_multi_poe.py --out <dir> --a "a cat" --b "a dog" --joint "a cat and a dog" \
        --seeds 1 2 3 4 5 6 7 8

Tiles land as <out>/<a>__x__<b>/seedNN/<column>.png with a JSON sidecar; an existing tile is skipped.
Starting noise is the seed's own draw through the pipeline's own helper, so the reference render and this
script's single-condition render start from the same latent.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path

MODEL_ROOT = Path(os.environ.get("PIXART_ROOT", "/datasets/mmolefe/poe_repair_min/models/PixArt-XL-2-1024-MS"))

# column key, the conditions summed into it. One condition is ordinary classifier-free guidance; two are
# the product this project's whole failure is about.
COLUMNS = [
    ("a", ["a"]),
    ("b", ["b"]),
    ("poe", ["a", "b"]),
    ("joint", ["joint"]),
]

MIN_FREE_GB = 2.0
MAX_USED_PCT = 90.0


def slug(word: str) -> str:
    return word.replace(" ", "_")


def disk_guard(path: Path) -> None:
    """Guard the filesystem `path` actually resolves to, never an assumed one (poe-disk-001)."""
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    free_gb = usage.free / 1024 ** 3
    used_pct = 100.0 * usage.used / usage.total
    print(f"disk guard on {path.resolve()}: {free_gb:.1f} GB free, {used_pct:.0f}% used", flush=True)
    if free_gb < MIN_FREE_GB or used_pct >= MAX_USED_PCT:
        sys.exit(f"disk guard refused {path.resolve()}: need {MIN_FREE_GB} GB free and under {MAX_USED_PCT}% used")


def load(device: str, dtype) -> object:
    from diffusers import PixArtAlphaPipeline

    if not (MODEL_ROOT / "model_index.json").exists():
        sys.exit(f"no PixArt-alpha weights at {MODEL_ROOT}; set PIXART_ROOT or fetch them there")
    pipe = PixArtAlphaPipeline.from_pretrained(MODEL_ROOT, torch_dtype=dtype, local_files_only=True).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def encode(pipe, conds: dict[str, str], device: str):
    """Each condition's embeddings and mask, plus the empty prompt's, exactly as the pipeline encodes them."""
    import torch

    out, empty = {}, None
    with torch.no_grad():
        for key, text in conds.items():
            e, m, e0, m0 = pipe.encode_prompt(text, do_classifier_free_guidance=True, negative_prompt="",
                                              device=device, clean_caption=True, max_sequence_length=120)
            out[key] = (e, m)
            if empty is None:
                empty = (e0, m0)
    return out, empty


def render_column(pipe, names, cond, empty, seed, steps, gs, size, device, dtype):
    """One tile: the summed guidance over `names`, sampled by the pipeline's own scheduler."""
    import torch
    from diffusers.utils.torch_utils import randn_tensor

    tr, vae, sched = pipe.transformer, pipe.vae, pipe.scheduler
    n = len(names) + 1  # the conditions, then the empty prompt last
    embs = torch.cat([cond[k][0] for k in names] + [empty[0]], dim=0).to(dtype)
    masks = torch.cat([cond[k][1] for k in names] + [empty[1]], dim=0)

    added = {"resolution": None, "aspect_ratio": None}
    if tr.config.sample_size == 128:
        res = torch.tensor([size, size]).repeat(n, 1).to(device=device, dtype=dtype)
        ar = torch.tensor([1.0]).repeat(n, 1).to(device=device, dtype=dtype)
        added = {"resolution": res, "aspect_ratio": ar}

    gen = torch.Generator(device=device).manual_seed(seed)
    shape = (1, tr.config.in_channels, size // pipe.vae_scale_factor, size // pipe.vae_scale_factor)
    lat = randn_tensor(shape, generator=gen, device=torch.device(device), dtype=dtype) * sched.init_noise_sigma

    sched.set_timesteps(steps, device=device)  # order-2 DPM-Solver++ keeps state, so every column resets it
    with torch.no_grad():
        for t in sched.timesteps:
            inp = sched.scale_model_input(lat.repeat(n, 1, 1, 1), t)
            eps = tr(inp, encoder_hidden_states=embs, encoder_attention_mask=masks,
                     timestep=t.expand(n), added_cond_kwargs=added, return_dict=False)[0]
            e_u = eps[-1:]
            e = e_u + gs * sum(eps[k:k + 1] - e_u for k in range(n - 1))
            if tr.config.out_channels // 2 == tr.config.in_channels:
                e = e.chunk(2, dim=1)[0]  # the second half is a learned sigma the sampler does not use
            lat = sched.step(e, t, lat, return_dict=False)[0]
        image = vae.decode(lat / vae.config.scaling_factor, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def identity_check(pipe, out: Path, prompt: str, seed: int, steps: int, gs: float, size: int, device, dtype) -> None:
    """The released pipeline's render of one prompt, beside this script's single-condition render of it."""
    import numpy as np
    import torch

    out.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator(device=device).manual_seed(seed)
    ref = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=gs, height=size, width=size,
               generator=gen, use_resolution_binning=False, clean_caption=True).images[0]
    ref.save(out / "identity-pipeline.png")

    cond, empty = encode(pipe, {"a": prompt}, device)
    mine = render_column(pipe, ["a"], cond, empty, seed, steps, gs, size, device, dtype)
    mine.save(out / "identity-script.png")

    a, b = np.asarray(ref, dtype=np.int16), np.asarray(mine, dtype=np.int16)
    diff = np.abs(a - b)
    record = {"prompt": prompt, "seed": seed, "steps": steps, "guidance": gs, "size": size,
              "dtype": str(dtype), "max_abs_diff_grey_levels": int(diff.max()),
              "fraction_of_pixels_differing": float((diff > 0).mean()),
              "mean_abs_diff_grey_levels": float(diff.mean())}
    (out / "identity-check.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1), flush=True)
    print(f"wrote {out}/identity-pipeline.png and {out}/identity-script.png", flush=True)


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--a", default="a cat")
    ap.add_argument("--b", default="a dog")
    ap.add_argument("--joint", default="a cat and a dog")
    ap.add_argument("--seeds", nargs="+", type=int, default=[])
    ap.add_argument("--steps", type=int, default=20, help="PixArt-alpha's own default, not SDXL's 50")
    ap.add_argument("--guidance", type=float, default=4.5, help="PixArt-alpha's own default, not SDXL's 7.5")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--identity-check", action="store_true",
                    help="render one prompt both ways and report the largest pixel difference, then stop")
    args = ap.parse_args()

    disk_guard(args.out)
    device, dtype = "cuda", torch.float16
    if not torch.cuda.is_available():
        sys.exit("no CUDA device visible; this would run on the CPU with no error (poe-launch-002)")
    pipe = load(device, dtype)
    print(f"text encoder {type(pipe.text_encoder).__name__}, {sum(p.numel() for p in pipe.text_encoder.parameters()) / 1e9:.1f}B params; "
          f"scheduler {type(pipe.scheduler).__name__}", flush=True)

    if args.identity_check:
        identity_check(pipe, args.out, args.a, args.seeds[0] if args.seeds else 1,
                       args.steps, args.guidance, args.size, device, dtype)
        return

    cond, empty = encode(pipe, {"a": args.a, "b": args.b, "joint": args.joint}, device)
    root = args.out / f"{slug(args.a)}__x__{slug(args.b)}"
    for seed in args.seeds:
        for key, names in COLUMNS:
            path = root / f"seed{seed:02d}" / f"{key}.png"
            if path.exists():
                print(f"{path}: already rendered", flush=True)
                continue
            image = render_column(pipe, names, cond, empty, seed, args.steps, args.guidance, args.size,
                                  device, dtype)
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            path.with_suffix(".json").write_text(json.dumps(
                {"column": key, "conditions": [{"a": args.a, "b": args.b, "joint": args.joint}[k] for k in names],
                 "a": args.a, "b": args.b, "joint": args.joint, "seed": seed, "steps": args.steps,
                 "guidance": args.guidance, "size": args.size, "model": str(MODEL_ROOT),
                 "text_encoder": type(pipe.text_encoder).__name__,
                 "scheduler": type(pipe.scheduler).__name__, "dtype": str(dtype)}, indent=1))
            print(f"wrote {path}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
