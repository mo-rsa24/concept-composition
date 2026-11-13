"""CLIP-guided PoE composer: the product nudged toward two things in frame.

Wraps ``run_poe_clip_guided``. Method name format::

    poe_clip_w<NNNN>_win<s>-<e>     # NNNN = round(weight * 1000), window [s, e)
"""
from __future__ import annotations

from poe_repair.composers._helpers import cell_output_dir, encode_pair, init_latents_for_cell
from poe_repair.methods._clip_guided import DEFAULT_WINDOW, run_poe_clip_guided
from poe_repair.methods._sampling import write_decoded_image
from poe_repair.run import MethodCtx
from poe_repair.runtime import PairSeedCell, write_json

METHOD_NAME = "poe_clip"


def method_name_for(*, weight: float, window=DEFAULT_WINDOW) -> str:
    return f"{METHOD_NAME}_w{int(round(float(weight) * 1000)):04d}_win{window[0]}-{window[1]}"


def run(cell: PairSeedCell, ctx: MethodCtx, *, score=None, weight: float = 0.0,
        window=DEFAULT_WINDOW, exp_name: str = "clip_guided", overwrite: bool = False,
        return_outputs: bool = False):
    method_name = method_name_for(weight=weight, window=window)
    out_dir = cell_output_dir(ctx, exp_name, method_name, cell)
    image_path = out_dir / f"{method_name}.png"
    if image_path.exists() and not overwrite and not return_outputs:
        return image_path
    init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
    emb = encode_pair(cell, ctx)
    out = run_poe_clip_guided(
        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
        seq_a=emb["seq_a"], pool_a=emb["pool_a"], seq_b=emb["seq_b"], pool_b=emb["pool_b"],
        seq_e=emb["seq_e"], pool_e=emb["pool_e"],
        guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
        height=cell.height, width=cell.width, euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype, score=score, weight=float(weight),
        guidance_window=tuple(window),
    )
    write_decoded_image(out.image, image_path)
    write_json(out_dir / f"summary_{method_name}.json", {
        "method": method_name, "pair_slug": cell.pair_slug, "seed": cell.seed,
        "pair": [cell.prompt_a, cell.prompt_b], "guidance_scale": ctx.guidance_scale,
        "num_inference_steps": ctx.num_inference_steps, **out.extras,
    })
    if return_outputs:
        return image_path, out
    return image_path
