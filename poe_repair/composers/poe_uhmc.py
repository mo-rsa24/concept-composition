"""Unadjusted HMC composer on the product of experts (PRZE-2).

Wraps ``run_poe_uhmc``: product-of-experts sampling with ``k`` unadjusted HMC iterations of ``L``
leapfrog steps per level. ``k = 0`` is plain product of experts, byte for byte.

Method name format::

    poe_uhmc_k<NNN>_c<NNN>_L<N>              # NNN = k, and round(c * 1000)
"""

from __future__ import annotations

from poe_repair.composers._helpers import cell_output_dir, encode_pair, init_latents_for_cell
from poe_repair.methods._corrector_drivers import run_poe_uhmc
from poe_repair.methods._sampling import write_decoded_image
from poe_repair.run import MethodCtx
from poe_repair.runtime import PairSeedCell, write_json


def method_name_for(*, k: int, c: float, n_leapfrog: int) -> str:
    return f"poe_uhmc_k{int(k):03d}_c{int(round(float(c) * 1000)):03d}_L{int(n_leapfrog)}"


def run(cell: PairSeedCell, ctx: MethodCtx, *, k: int, c: float, n_leapfrog: int = 5,
        noise_seed: int | None = None, decode: bool = True,
        exp_name: str = "paper_review/correctors", overwrite: bool = False,
        return_outputs: bool = False):
    """Run the unadjusted HMC corrector on one cell and return the image path."""
    method_name = method_name_for(k=k, c=c, n_leapfrog=n_leapfrog)
    out_dir = cell_output_dir(ctx, exp_name, method_name, cell)
    image_path = out_dir / f"{method_name}.png"
    summary_path = out_dir / f"summary_{method_name}.json"
    if image_path.exists() and not overwrite and not return_outputs:
        return image_path
    init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
    emb = encode_pair(cell, ctx)
    out = run_poe_uhmc(
        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
        seq_a=emb["seq_a"], pool_a=emb["pool_a"], seq_b=emb["seq_b"], pool_b=emb["pool_b"],
        seq_e=emb["seq_e"], pool_e=emb["pool_e"],
        guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
        height=cell.height, width=cell.width, euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype, k=int(k), c=float(c), n_leapfrog=int(n_leapfrog),
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
