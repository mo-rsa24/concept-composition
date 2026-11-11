"""Joint-prompt Langevin composer (PRZE-2).

Wraps ``run_joint_langevin``: the joint prompt's CFG sampler with ``k`` Langevin steps per level
on the joint score. ``k = 0`` is the plain joint-prompt sampler, byte for byte.

Method name format::

    joint_langevin_k<NNN>_c<NNN>              # NNN = k, and round(c * 1000)
"""

from __future__ import annotations

from poe_repair.composers._helpers import (
    cell_output_dir,
    encode_pair,
    get_joint_embeds,
    init_latents_for_cell,
)
from poe_repair.methods._corrector_drivers import run_joint_langevin
from poe_repair.methods._sampling import write_decoded_image
from poe_repair.run import MethodCtx
from poe_repair.runtime import PairSeedCell, write_json


def method_name_for(*, k: int, c: float) -> str:
    return f"joint_langevin_k{int(k):03d}_c{int(round(float(c) * 1000)):03d}"


def run(cell: PairSeedCell, ctx: MethodCtx, *, k: int, c: float,
        noise_seed: int | None = None, decode: bool = True,
        exp_name: str = "paper_review/correctors", overwrite: bool = False,
        return_outputs: bool = False):
    """Run the joint-prompt corrector on one cell and return the image path."""
    method_name = method_name_for(k=k, c=c)
    out_dir = cell_output_dir(ctx, exp_name, method_name, cell)
    image_path = out_dir / f"{method_name}.png"
    summary_path = out_dir / f"summary_{method_name}.json"
    if image_path.exists() and not overwrite and not return_outputs:
        return image_path
    init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
    emb = encode_pair(cell, ctx)
    seq_j, pool_j = get_joint_embeds(cell, ctx)
    out = run_joint_langevin(
        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
        seq_j=seq_j, pool_j=pool_j, seq_e=emb["seq_e"], pool_e=emb["pool_e"],
        guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
        height=cell.height, width=cell.width, euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype, k=int(k), c=float(c),
        noise_seed=int(cell.seed if noise_seed is None else noise_seed), decode=decode,
    )

    if decode and out.image is not None:
        write_decoded_image(out.image, image_path)
    write_json(summary_path, {
        "method": method_name, "pair_slug": cell.pair_slug, "seed": cell.seed,
        "image_path": str(image_path) if decode else None,
        "pair": [cell.prompt_a, cell.prompt_b],
        "guidance_scale": ctx.guidance_scale,
        "num_inference_steps": ctx.num_inference_steps,
        **out.extras,
    })
    if return_outputs:
        return image_path, out
    return image_path
