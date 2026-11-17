"""Classifier-guided PoE composer.

Wraps ``run_poe_classifier_guided``: plain product-of-experts sampling with the presence head's
push added at every step, in Dhariwal and Nichol's unrescaled form and capped at a fraction of the
product prediction's own norm.

Method name format::

    poe_classifier_st<NNNNN>              # NNNNN = round(strength * 100)
    poe_classifier_st<NNNNN>_s<NNNNNN>    # with the head's training step, when probing mid-run
"""
from __future__ import annotations

from pathlib import Path

from poe_repair.composers._helpers import (
    cell_output_dir,
    encode_pair,
    init_latents_for_cell,
)
from poe_repair.methods._classifier_guided import PUSH_CAP_FRACTION, run_poe_classifier_guided
from poe_repair.methods._sampling import write_decoded_image
from poe_repair.run import MethodCtx
from poe_repair.runtime import PairSeedCell, write_json

METHOD_NAME = "poe_classifier"


def method_name_for(*, strength: float, train_step: int | None = None) -> str:
    name = f"{METHOD_NAME}_st{int(round(float(strength) * 100)):05d}"
    if train_step is not None:
        name += f"_s{int(train_step):06d}"
    return name


def run(
    cell: PairSeedCell,
    ctx: MethodCtx,
    *,
    head=None,
    weight: float = 0.0,
    strength: float = 0.0,
    push_cap: float | None = PUSH_CAP_FRACTION,
    guidance_window: tuple[int, int] | None = None,
    train_step: int | None = None,
    exp_name: str = "presence_head/sweep",
    overwrite: bool = False,
    return_outputs: bool = False,
):
    """Render one cell at one classifier strength; return the image path.

    ``weight`` is accepted only so the sampler can refuse a nonzero value from an old caller.
    """
    method_name = method_name_for(strength=strength, train_step=train_step)
    out_dir = cell_output_dir(ctx, exp_name, method_name, cell)
    image_path = out_dir / f"{method_name}.png"
    if image_path.exists() and not overwrite and not return_outputs:
        return image_path

    init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
    emb = encode_pair(cell, ctx)
    out = run_poe_classifier_guided(
        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
        seq_a=emb["seq_a"], pool_a=emb["pool_a"],
        seq_b=emb["seq_b"], pool_b=emb["pool_b"],
        seq_e=emb["seq_e"], pool_e=emb["pool_e"],
        guidance_scale=ctx.guidance_scale,
        num_inference_steps=ctx.num_inference_steps,
        height=cell.height, width=cell.width,
        euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype,
        head=head, weight=float(weight), strength=float(strength), push_cap=push_cap,
        guidance_window=guidance_window,
    )
    write_decoded_image(out.image, image_path)
    write_json(
        out_dir / f"summary_{method_name}.json",
        {
            "method": method_name,
            "pair_slug": cell.pair_slug,
            "seed": cell.seed,
            "pair": [cell.prompt_a, cell.prompt_b],
            "guidance_scale": ctx.guidance_scale,
            "num_inference_steps": ctx.num_inference_steps,
            "head_train_step": train_step,
            **out.extras,
        },
    )
    if return_outputs:
        return image_path, out
    return image_path
