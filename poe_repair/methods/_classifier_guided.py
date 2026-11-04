"""Plain product-of-experts sampling with a presence classifier's gradient added per step.

The product of two prompts is maximised by one animal that reads as both, because two captions
compete for the same probability mass. A presence classifier does not compete: "a cat instance is
present" and "a dog instance is present" are separate yes-or-no questions with separate
normalisers, so a frame holding two animals can satisfy both. Adding the gradient of the sum of
their log probabilities is therefore not a third expert; it is a reweighting that can put mass
where both experts agree there is none, which is the one thing no amount of re-weighting the two
experts can do.

Per step:
  1. One three-branch frozen UNet forward over (A, B, empty) gives the product prediction, exactly
     as ``run_cfg_poe`` computes it.
  2. The head reads the current latent and its timestep and gives log p(cat) + log p(dog).
  3. That gradient with respect to the latent is rescaled to a fixed fraction of the product
     prediction's own norm and subtracted, which is classifier guidance in noise-prediction form.
  4. The ordinary DDIM step follows.

At ``weight == 0`` the head is never called and nothing touches the prediction, so the rollout is
byte-identical to ``run_cfg_poe``. That canary is the gate: a composer that moves the baseline
makes every column of the comparison meaningless.

The normalisation is per step and the renders are judged by eye. A norm guard on an earlier
corrector in this repo reported healthy numbers while the samples were being destroyed, so no
statistic over the gradient may stand in for looking at the picture.
"""
from __future__ import annotations

import torch

from poe_repair.methods._sampling import SamplerOutputs, add_time_ids
from poe_repair.runtime import (
    LatentTrajectoryCollector,
    decode_latents,
    ddim_prev_from_x0_eps,
    guided_eps,
    poe_eps,
    tweedie_mean,
)

# Below this the gradient is treated as absent rather than rescaled, so a flat head cannot be
# blown up into a full-size step by the normalisation.
MIN_GRAD_NORM = 1e-8


def run_poe_classifier_guided(
    *,
    init_latents: torch.Tensor,
    models: dict,
    scheduler,
    seq_a: torch.Tensor, pool_a: torch.Tensor,
    seq_b: torch.Tensor, pool_b: torch.Tensor,
    seq_e: torch.Tensor, pool_e: torch.Tensor,
    guidance_scale: float,
    num_inference_steps: int,
    height: int, width: int,
    euler_init_noise_sigma: float,
    device: torch.device, dtype: torch.dtype,
    head=None,
    weight: float = 0.0,
    guidance_window: tuple[int, int] | None = None,
) -> SamplerOutputs:
    """``run_cfg_poe`` plus ``weight`` times the head's normalised gradient at each step."""
    scheduler.set_timesteps(num_inference_steps)
    latents = (init_latents / euler_init_noise_sigma).to(device=device, dtype=dtype)
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
    head_was_training = None
    if head is not None:
        head_was_training = head.training
        head.eval()

    grad_norm_per_step: list[float] = []
    applied_per_step: list[float] = []

    for step_index, timestep in enumerate(scheduler.timesteps):
        with torch.no_grad():
            latent_input = scheduler.scale_model_input(latents.repeat(3, 1, 1, 1), timestep)
            noise = unet(
                latent_input, timestep, encoder_hidden_states=pe,
                added_cond_kwargs=cond, timestep_cond=None,
            ).sample
            eps_a_raw, eps_b_raw, eps_uncond = noise.chunk(3)
            eps_a = guided_eps(eps_a_raw, eps_uncond, guidance_scale)
            eps_b = guided_eps(eps_b_raw, eps_uncond, guidance_scale)
            eps_p = poe_eps(eps_a, eps_b, eps_uncond)

        in_window = guidance_window is None or (
            guidance_window[0] <= step_index < guidance_window[1])
        # The canary lives here: at weight 0 nothing below runs and eps_p is untouched.
        if head is not None and weight != 0.0 and in_window:
            with torch.enable_grad():
                z = latents.detach().float().requires_grad_(True)
                t_batch = timestep.reshape(1).to(device=device).expand(z.shape[0])
                log_p = head.log_prob_both(z, t_batch)
                grad = torch.autograd.grad(log_p, z)[0]
            grad_norm = float(grad.norm().item())
            grad_norm_per_step.append(grad_norm)
            if grad_norm > MIN_GRAD_NORM:
                # The weight is a step size in units of the product prediction's own norm, so it
                # means the same thing at every noise level without a schedule.
                step = (grad / grad_norm) * eps_p.float().norm() * float(weight)
                eps_p = (eps_p.float() - step).to(dtype)
                applied_per_step.append(float(step.norm().item()))
            else:
                applied_per_step.append(0.0)
        else:
            grad_norm_per_step.append(0.0)
            applied_per_step.append(0.0)

        with torch.no_grad():
            alpha_bar_t = scheduler.alphas_cumprod[int(timestep.item())].to(
                device=device, dtype=dtype)
            x0 = tweedie_mean(latents, alpha_bar_t, eps_p)
            tracker.store_step(
                step_index, latents, eps_p,
                float(step_index) / float(num_inference_steps),
                int(timestep.item()),
            )
            latents = ddim_prev_from_x0_eps(
                scheduler=scheduler, timestep=timestep, step_index=step_index,
                x0=x0, eps=eps_p,
            )

    tracker.store_final(latents)
    with torch.no_grad():
        image = decode_latents(models, latents).cpu()
    if head is not None and head_was_training:
        head.train()
    return SamplerOutputs(
        latents=latents, image=image, tracker=tracker,
        extras={
            "classifier_weight": float(weight),
            "guidance_window": list(guidance_window) if guidance_window else None,
            "grad_norm_per_step": grad_norm_per_step,
            "applied_norm_per_step": applied_per_step,
        },
    )
