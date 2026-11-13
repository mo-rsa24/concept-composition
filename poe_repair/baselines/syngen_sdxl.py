"""SynGen (Rassin et al. 2023) ported to SDXL: the joint-prompt sampler with SynGen's latent update on steps 0 to 24.

Plan: _archive/1-lora-correction/artifacts/drips/answer-iclr-reviews/routes/plan-sd15-baselines-and-syngen-sdxl.md, Plan 2.

`run_cfg_syngen` is the body of `run_cfg` in poe_repair/methods/_sampling.py (plain single-prompt CFG on SDXL, the sampler
behind the joint column of Table 1) with SynGen's `_syngen_step` inserted at the top of each of the first 25 of 50 steps,
as in the released pipeline: one conditional forward pass with gradients, the cross-attention maps of that pass, SynGen's
loss on them, and z <- z - step_size * dL/dz when the loss is not zero. `run_cfg` itself is not edited, and its trajectory
collector is left out because nothing reads it here.

What is SynGen's own code, imported unedited from the clone at /datasets/mmolefe/repos/Linguistic-Binding-in-Diffusion-Models
(commit 50b8eab): the loss (`SynGenDiffusionPipeline._compute_loss`, `_attribution_loss`, `_calculate_losses`, and through
them compute_loss.py's positive and negative losses), `_get_attention_maps_list`, and diffusers' `AttentionStore` and
`AttendExciteAttnProcessor`, which SynGen imports the same way. The parse is SynGen's too, run beforehand in its own
environment (1-lora-correction/scripts/review_syngen_indices.py), because co3 has no spaCy; the released module's `import spacy` is given an
empty stand-in here so the module loads, and no spaCy call is made.

What is adapted for SDXL, and why:
- the attention maps are read at 32 by 32 (SDXL at 1024 has no 16 by 16 layer; 32 by 32 is its coarsest, 60 of its 70
  cross-attention layers), from cross-attention layers only. The store's processors are attached for the gradient pass
  alone and the original processors are put back before the CFG pass, so the CFG pass is exactly `run_cfg`'s.
- the U-Net runs in fp16 as every Table 1 column does (protocol rule 2); the latent the gradient is taken against is an
  fp32 copy, the maps and the loss are computed in fp32, and a non-finite gradient aborts the cell rather than writing it.
- the gradient pass gets the conditional half of SDXL's pooled embedding and size ids through `added_cond_kwargs`.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import torch

from poe_repair.methods._sampling import add_time_ids
from poe_repair.runtime import ddim_prev_from_x0_eps, decode_latents, guided_eps, tweedie_mean

SYNGEN_DIR = Path("/datasets/mmolefe/repos/Linguistic-Binding-in-Diffusion-Models")
SYNGEN_COMMIT = "50b8eab175dc9b31ce9c7c4b2eba28c817fc440e"
ATTN_RES = (32, 32)
NUM_INTERVENTION_STEPS = 25


def import_syngen():
    """The released syngen_diffusion_pipeline module, loaded in an environment without spaCy."""
    if str(SYNGEN_DIR) not in sys.path:
        sys.path.insert(0, str(SYNGEN_DIR))
    try:
        import spacy  # noqa: F401
    except ImportError:
        sys.modules["spacy"] = types.ModuleType("spacy")
    # SynGen imports Attend-and-Excite from diffusers' pre-0.20 path; newer diffusers keeps the same
    # classes in their own package, so the old name points at it.
    old = "diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_attend_and_excite"
    try:
        importlib.import_module(old)
    except ModuleNotFoundError:
        sys.modules[old] = importlib.import_module(
            "diffusers.pipelines.stable_diffusion_attend_and_excite.pipeline_stable_diffusion_attend_and_excite")
    import syngen_diffusion_pipeline as sgp
    return sgp


class SynGenLoss:
    """Carries the two attributes SynGen's loss methods read from the pipeline, with those methods bound unchanged."""

    def __init__(self, sgp, tokenizer) -> None:
        P = sgp.SynGenDiffusionPipeline
        self.tokenizer = tokenizer
        self.subtrees_indices = None
        self.maps_list = sgp._get_attention_maps_list
        self._compute_loss = types.MethodType(P._compute_loss, self)
        self._attribution_loss = types.MethodType(P._attribution_loss, self)
        self._calculate_losses = types.MethodType(P._calculate_losses, self)


class SynGenNonFinite(FloatingPointError):
    pass


def _store_processors(unet, store) -> dict:
    from diffusers.pipelines.stable_diffusion_attend_and_excite.pipeline_stable_diffusion_attend_and_excite import (
        AttendExciteAttnProcessor,
    )
    procs, n = dict(unet.attn_processors), 0
    for name in procs:
        if ".attn2." not in name:
            continue
        place = "mid" if name.startswith("mid_block") else "up" if name.startswith("up_blocks") else "down"
        procs[name] = AttendExciteAttnProcessor(attnstore=store, place_in_unet=place)
        n += 1
    store.num_att_layers = n
    return procs


def _syngen_step(*, unet, latents, timestep, seq_cond, cond1, loss_fn: SynGenLoss, prompt, indices, step_size,
                 originals, dtype):
    from diffusers.pipelines.stable_diffusion_attend_and_excite.pipeline_stable_diffusion_attend_and_excite import (
        AttentionStore,
    )
    store = AttentionStore(ATTN_RES)
    unet.set_attn_processor(_store_processors(unet, store))  # set_attn_processor pops from the dict it is given
    try:
        with torch.enable_grad():
            lat32 = latents.detach().float().requires_grad_(True)
            unet(lat32.to(dtype), timestep, encoder_hidden_states=seq_cond, added_cond_kwargs=cond1,
                 timestep_cond=None)
            if not indices:  # no modifier in the prompt: SynGen's loss is zero and the latent is not moved
                return latents, 0.0
            maps = store.aggregate_attention(from_where=("up", "down", "mid")).float()
            loss_fn.subtrees_indices = indices
            loss = loss_fn._compute_loss(attention_maps=loss_fn.maps_list(maps), prompt=prompt)
            value = float(loss)
            if loss != 0 and step_size != 0:
                grad = torch.autograd.grad(loss.requires_grad_(True), [lat32])[0]
                if not torch.isfinite(grad).all() or not torch.isfinite(torch.as_tensor(value)):
                    raise SynGenNonFinite(f"non-finite gradient or loss ({value}) at timestep {int(timestep)}")
                lat32 = lat32 - step_size * grad
            return lat32.detach().to(dtype), value
    finally:
        unet.set_attn_processor(dict(originals))


@torch.no_grad()
def run_cfg_syngen(
    *,
    init_latents: torch.Tensor,
    models: dict,
    scheduler,
    seq_cond: torch.Tensor, pool_cond: torch.Tensor,
    seq_e: torch.Tensor, pool_e: torch.Tensor,
    guidance_scale: float,
    num_inference_steps: int,
    height: int, width: int,
    euler_init_noise_sigma: float,
    device: torch.device, dtype: torch.dtype,
    prompt: str,
    indices: list,
    loss_fn: SynGenLoss,
    step_size: float = 20.0,
    num_intervention_steps: int = NUM_INTERVENTION_STEPS,
) -> tuple[torch.Tensor, list[float]]:
    """Returns the decoded image in [0, 1] (as run_cfg's) and the SynGen loss seen at each update step."""
    scheduler.set_timesteps(num_inference_steps)
    latents = (init_latents / euler_init_noise_sigma).to(device=device, dtype=dtype)
    pe = torch.cat([seq_cond, seq_e], dim=0)
    pool = torch.cat([pool_cond, pool_e], dim=0)
    cond = {"text_embeds": pool, "time_ids": add_time_ids(height=height, width=width, batch_size=2, device=device,
                                                          dtype=dtype)}
    cond1 = {"text_embeds": pool_cond, "time_ids": add_time_ids(height=height, width=width, batch_size=1,
                                                               device=device, dtype=dtype)}
    unet = models["unet"]
    originals = dict(unet.attn_processors)
    losses: list[float] = []
    for step_index, timestep in enumerate(scheduler.timesteps):
        if step_index < num_intervention_steps:
            latents, value = _syngen_step(unet=unet, latents=latents, timestep=timestep, seq_cond=seq_cond,
                                          cond1=cond1, loss_fn=loss_fn, prompt=prompt, indices=indices,
                                          step_size=step_size, originals=originals, dtype=dtype)
            losses.append(value)
        latent_input = scheduler.scale_model_input(latents.repeat(2, 1, 1, 1), timestep)
        noise = unet(
            latent_input, timestep, encoder_hidden_states=pe,
            added_cond_kwargs=cond, timestep_cond=None,
        ).sample
        eps_cond_raw, eps_uncond = noise.chunk(2)
        eps_t = guided_eps(eps_cond_raw, eps_uncond, guidance_scale)
        alpha_bar_t = scheduler.alphas_cumprod[int(timestep.item())].to(device=device, dtype=dtype)
        x0 = tweedie_mean(latents, alpha_bar_t, eps_t)
        latents = ddim_prev_from_x0_eps(
            scheduler=scheduler, timestep=timestep, step_index=step_index,
            x0=x0, eps=eps_t,
        )
    image = decode_latents(models, latents).cpu()
    return image, losses
