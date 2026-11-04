"""Regional PoE composer: each expert governs its own region of the latent grid.

Beside [the plain product composer](poe.py), which it does not touch. Plain product lets both
experts constrain every cell; this one hands each expert a soft mask and lets it constrain only
its own, which turns the product over shared coordinates into a product over disjoint ones.

The masks come from a layout, and a layout has four sources here, which are the four conditions
the experiment compares:

``oracle``
    the two animals' boxes detected in the joint prompt's own successful render, each named by
    the crop classifier. The experiment.
``random``
    two boxes of the same areas as the oracle's, placed at random. Separates "a layout" from
    "the right layout".
``swapped``
    the oracle boxes with the two prompts exchanged. Separates "two regions" from "the right
    region for each animal".
``same``
    the oracle boxes with prompt A in both. Separates "regional masking makes two objects" from
    "it makes the two named objects".

Method name format::

    poe_regional_<condition>_<reading>          # reading: even | smaller-wins
    poe_regional_<condition>_<reading>_b<NN>    # with NN bootstrapping steps
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from poe_repair.composers._helpers import (
    cell_output_dir,
    encode_pair,
    init_latents_for_cell,
)
from poe_repair.composers._regional_masks import LATENT, MaskPair, masks_from_boxes
from poe_repair.methods._poe_regional import run_cfg_poe_regional
from poe_repair.methods._sampling import write_decoded_image
from poe_repair.run import MethodCtx
from poe_repair.runtime import PairSeedCell, encode_prompt_sdxl, write_json

CONDITIONS = ("oracle", "random", "swapped", "same")
BOOTSTRAP_COLOUR = 0.5   # mid grey, the constant background each region forms against


def constant_colour_latent(ctx: MethodCtx, cell: PairSeedCell) -> torch.Tensor:
    """The VAE encoding of a flat mid-grey image, which bootstrapping composites each region over.

    The inverse of ``decode_latents_to_tensor``: that maps a latent to [0, 1] pixels through
    ``latents / scaling_factor + shift_factor``, so the encoder's mean is scaled back the same way.
    """
    vae = ctx.models["vae"]
    pixels = torch.full((1, 3, cell.height, cell.width), BOOTSTRAP_COLOUR * 2.0 - 1.0,
                        device=ctx.device, dtype=vae.dtype)
    with torch.no_grad():
        mean = vae.encode(pixels).latent_dist.mean
    shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
    return ((mean - shift) * vae.config.scaling_factor).to(dtype=ctx.dtype)


def method_name_for(*, condition: str, reading: str, bootstrap_steps: int = 0) -> str:
    name = f"poe_regional_{condition}_{reading}"
    if bootstrap_steps:
        name += f"_b{int(bootstrap_steps):02d}"
    return name


def random_boxes_like(box_a, box_b, rng: np.random.Generator, grid: int = LATENT):
    """Two boxes of the same width and height as the originals, placed at random.

    Shape is held and position is not, so the control differs from the oracle on where the
    regions are and on nothing else. Overlap is left to chance, exactly as the oracle's is.
    """
    out = []
    for box in (box_a, box_b):
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        w, h = x1 - x0, y1 - y0
        nx = int(rng.integers(0, max(1, grid - w + 1)))
        ny = int(rng.integers(0, max(1, grid - h + 1)))
        out.append((nx, ny, nx + w, ny + h))
    return out[0], out[1]


def masks_for(condition: str, cat_box, dog_box, *, reading: str,
              rng: np.random.Generator | None = None) -> tuple[MaskPair, tuple, tuple]:
    """The mask pair for one condition, and the two boxes it was built from.

    ``swapped`` and ``same`` keep the oracle's boxes and change only which prompt each region
    carries, so the masks themselves are the oracle's; the swap happens in the embeddings the
    caller passes.
    """
    if condition == "random":
        if rng is None:
            raise ValueError("the random-box control needs a seeded generator, so it is recorded")
        cat_box, dog_box = random_boxes_like(cat_box, dog_box, rng)
    return masks_from_boxes(cat_box, dog_box, reading=reading), tuple(cat_box), tuple(dog_box)


def _as_tensor(mask: np.ndarray, device, dtype) -> torch.Tensor:
    return torch.from_numpy(mask).to(device=device, dtype=dtype)[None, None]


def run(
    cell: PairSeedCell,
    ctx: MethodCtx,
    *,
    cat_box,
    dog_box,
    condition: str = "oracle",
    reading: str = "even",
    bootstrap_steps: int = 0,
    random_seed: int | None = None,
    init_from: PairSeedCell | None = None,
    exp_name: str = "is_the_layout_the_missing_piece",
    overwrite: bool = False,
) -> Path:
    """Render one cell under one condition; return the image path.

    ``cat_box`` and ``dog_box`` are the oracle boxes in latent cells. ``init_from`` lets the
    starting noise come from another cell, so every column of a row starts identically.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, not {condition!r}")
    method_name = method_name_for(condition=condition, reading=reading,
                                  bootstrap_steps=bootstrap_steps)
    out_dir = cell_output_dir(ctx, exp_name, method_name, cell)
    image_path = out_dir / f"{method_name}.png"
    if image_path.exists() and not overwrite:
        return image_path

    init_latents, euler_sigma = init_latents_for_cell(init_from or cell, ctx)
    emb = encode_pair(cell, ctx)
    seq_cat, pool_cat = emb["seq_a"], emb["pool_a"]
    seq_dog, pool_dog = emb["seq_b"], emb["pool_b"]
    if condition == "swapped":
        seq_cat, pool_cat, seq_dog, pool_dog = seq_dog, pool_dog, seq_cat, pool_cat
    elif condition == "same":
        seq_dog, pool_dog = seq_cat, pool_cat

    rng = np.random.default_rng(random_seed) if random_seed is not None else None
    pair, used_cat_box, used_dog_box = masks_for(condition, cat_box, dog_box,
                                                 reading=reading, rng=rng)

    bootstrap_latent = constant_colour_latent(ctx, cell) if bootstrap_steps else None

    out = run_cfg_poe_regional(
        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
        seq_a=seq_cat, pool_a=pool_cat,
        seq_b=seq_dog, pool_b=pool_dog,
        seq_e=emb["seq_e"], pool_e=emb["pool_e"],
        mask_a=_as_tensor(pair.a, ctx.device, ctx.dtype),
        mask_b=_as_tensor(pair.b, ctx.device, ctx.dtype),
        guidance_scale=ctx.guidance_scale,
        num_inference_steps=ctx.num_inference_steps,
        height=cell.height, width=cell.width,
        euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype,
        bootstrap_steps=bootstrap_steps,
        bootstrap_latent=bootstrap_latent,
    )
    write_decoded_image(out.image, image_path)
    write_json(out_dir / f"summary_{method_name}.json", {
        "method": method_name,
        "condition": condition,
        "overlap_reading": reading,
        "pair_slug": cell.pair_slug,
        "seed": cell.seed,
        "noise_from_seed": (init_from or cell).seed,
        "noise_from_pair": (init_from or cell).pair_slug,
        "prompt_in_cat_region": cell.prompt_b if condition == "swapped" else cell.prompt_a,
        "prompt_in_dog_region": (cell.prompt_a if condition in ("swapped", "same")
                                 else cell.prompt_b),
        "cat_box_cells": list(used_cat_box),
        "dog_box_cells": list(used_dog_box),
        "random_box_seed": random_seed,
        "guidance_scale": ctx.guidance_scale,
        "num_inference_steps": ctx.num_inference_steps,
        "image_path": str(image_path),
        **out.extras,
    })
    return image_path
