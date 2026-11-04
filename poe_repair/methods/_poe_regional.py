"""The regional sampler: each expert governs its own cells instead of all of them.

Plain product of experts predicts, at every cell,

    eps = eps_uncond + w * (d_A + d_B),      d_X = eps_X - eps_uncond

so both experts constrain every cell. The only image satisfying both is one animal that reads as
either, which is the chimera this project is about. This sampler predicts, at every cell,

    eps = eps_uncond + w * (m_A * d_A + m_B * d_B),      m_A + m_B = 1

with ``m_A`` and ``m_B`` soft masks on the latent grid. Where the masks are hard, the product is
over disjoint coordinates, which is the condition under which score addition is exact
(Projective Composition, arXiv 2502.04549, Definition 5.2).

**This is MultiDiffusion's operator, arrived at from the other side.** MultiDiffusion (arXiv
2302.08113) fuses per-region *latents*, ``x_{t-1} = sum_i m_i * x_{t-1}^i``. Given a shared
``x_t``, the DDIM step is affine in epsilon, so fusing latents with weights that sum to 1 is the
same as fusing the epsilons with those weights. Fusing epsilons costs one batched UNet pass
instead of one pass per region, which is why it is written this way here.

**Bootstrapping, default off.** The one piece of MultiDiffusion the plain fusion leaves out. For
the first ``bootstrap_steps`` steps, each expert sees its own region composited over a constant
colour rather than over the other animal, so an object forms inside a small box instead of the
box filling with background. It costs the batched pass its sharing, since the two conditional
branches then run on different inputs. Off by default, because turning it on is a change to the
operator and the plan records when it was turned on and why.
"""

from __future__ import annotations

import torch

from poe_repair.methods._sampling import SamplerOutputs, add_time_ids
from poe_repair.runtime import (
    LatentTrajectoryCollector,
    ddim_prev_from_x0_eps,
    decode_latents,
    tweedie_mean,
)


@torch.no_grad()
def run_cfg_poe_regional(
    *,
    init_latents: torch.Tensor,
    models: dict,
    scheduler,
    seq_a: torch.Tensor, pool_a: torch.Tensor,
    seq_b: torch.Tensor, pool_b: torch.Tensor,
    seq_e: torch.Tensor, pool_e: torch.Tensor,
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
    guidance_scale: float,
    num_inference_steps: int,
    height: int, width: int,
    euler_init_noise_sigma: float,
    device: torch.device, dtype: torch.dtype,
    bootstrap_steps: int = 0,
    bootstrap_latent: torch.Tensor | None = None,
) -> SamplerOutputs:
    """eps = eps_uncond + w * (m_A * d_A + m_B * d_B), with m_A + m_B = 1 at every cell.

    ``mask_a`` and ``mask_b`` are ``(1, 1, H/8, W/8)`` and broadcast over the four latent
    channels, so an expert's weight is per cell and not per channel.
    """
    if bootstrap_steps and bootstrap_latent is None:
        raise ValueError("bootstrap_steps > 0 needs a bootstrap_latent to composite against")
    scheduler.set_timesteps(num_inference_steps)
    latents = (init_latents / euler_init_noise_sigma).to(device=device, dtype=dtype)
    m_a = mask_a.to(device=device, dtype=dtype)
    m_b = mask_b.to(device=device, dtype=dtype)
    worst = float((m_a + m_b - 1.0).abs().max())
    if worst > 1e-3:
        raise ValueError(f"the two masks must sum to 1 at every cell; worst deviation {worst:.2e}")

    tracker = LatentTrajectoryCollector(
        num_inference_steps, 1, latents.shape[1], latents.shape[2], latents.shape[3]
    )
    pe = torch.cat([seq_a, seq_b, seq_e], dim=0)
    pool = torch.cat([pool_a, pool_b, pool_e], dim=0)
    cond = {
        "text_embeds": pool,
        "time_ids": add_time_ids(
            height=height, width=width, batch_size=3, device=device, dtype=dtype,
        ),
    }
    unet = models["unet"]
    for step_index, timestep in enumerate(scheduler.timesteps):
        if step_index < bootstrap_steps:
            # Each expert sees its own region over a constant colour, noised to this timestep,
            # so a small box forms an object rather than filling with what is around it.
            alpha_bar = scheduler.alphas_cumprod[int(timestep.item())].to(
                device=device, dtype=dtype)
            noise = torch.randn(bootstrap_latent.shape, device=device, dtype=dtype,
                                generator=None)
            bg = alpha_bar.sqrt() * bootstrap_latent + (1 - alpha_bar).sqrt() * noise
            batch = torch.cat([
                m_a * latents + (1 - m_a) * bg,
                m_b * latents + (1 - m_b) * bg,
                latents,
            ], dim=0)
        else:
            batch = latents.repeat(3, 1, 1, 1)
        latent_input = scheduler.scale_model_input(batch, timestep)
        noise_pred = unet(
            latent_input, timestep, encoder_hidden_states=pe,
            added_cond_kwargs=cond, timestep_cond=None,
        ).sample
        eps_a_raw, eps_b_raw, eps_uncond = noise_pred.chunk(3)
        d_a = eps_a_raw - eps_uncond
        d_b = eps_b_raw - eps_uncond
        eps_r = eps_uncond + guidance_scale * (m_a * d_a + m_b * d_b)
        alpha_bar_t = scheduler.alphas_cumprod[int(timestep.item())].to(
            device=device, dtype=dtype)
        x0 = tweedie_mean(latents, alpha_bar_t, eps_r)
        tracker.store_step(
            step_index, latents, eps_r,
            float(step_index) / float(num_inference_steps),
            int(timestep.item()),
        )
        latents = ddim_prev_from_x0_eps(
            scheduler=scheduler, timestep=timestep, step_index=step_index,
            x0=x0, eps=eps_r,
        )
    tracker.store_final(latents)
    image = decode_latents(models, latents).cpu()
    return SamplerOutputs(
        latents=latents, image=image, tracker=tracker,
        extras={"mask_sum_max_deviation": worst, "bootstrap_steps": int(bootstrap_steps),
                "cat_share_of_grid": float(m_a.mean())},
    )
