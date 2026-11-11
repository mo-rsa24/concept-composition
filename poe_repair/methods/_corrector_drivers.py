"""Two more correctors for the review's sampler comparison (PRZE-2).

``run_joint_langevin`` is ``run_poe_langevin``'s loop driven by the joint prompt's guided
prediction instead of the product's: at each of the 50 DDIM levels the latent is settled by ``k``
unadjusted Langevin steps on the joint score, then the ordinary DDIM step is taken from the
settled point. ``k = 0`` runs the exact ops of ``run_cfg`` with the joint embedding, so it is
byte-identical to it.

``run_poe_uhmc`` is unadjusted Hamiltonian Monte Carlo on the product's score. At each level it
runs ``k`` iterations, each one a fresh momentum ``p ~ N(0, I)`` and ``L`` leapfrog steps of size
``eps_t = sqrt(2 c beta_t)``, with no accept or reject step (SDXL gives a score, not an energy, so
the Metropolis correction cannot be computed). Then the ordinary DDIM step. ``k = 0`` runs the
exact ops of ``run_cfg_poe``. With ``L = 1`` one iteration is one Langevin step of size
``delta_t = eps_t^2 / 2 = c beta_t``, the same rule ``run_poe_langevin`` uses, so the two samplers
share ``c``. ``k`` counts iterations, so a level costs ``k * L`` UNet calls plus the one DDIM call.

Both keep ``run_poe_langevin``'s precision rules: the chain runs in float32, the UNet sees float16,
and the DDIM step is taken on the float16 latent with the reference sampler's own ops.
"""

from __future__ import annotations

import math

import torch

from poe_repair.methods._sampling import SamplerOutputs, add_time_ids
from poe_repair.methods._poe_langevin import corrector_active
from poe_repair.runtime import (
    LatentTrajectoryCollector,
    decode_latents,
    ddim_prev_from_x0_eps,
    guided_eps,
    poe_eps,
    tweedie_mean,
)


def _norm(x: torch.Tensor) -> float:
    return float(x.detach().float().norm().item())


def _two_branch(unet, scheduler, seq_c, pool_c, seq_e, pool_e, guidance_scale, height, width,
                device, dtype):
    """``run_cfg``'s per-step call, op for op."""
    pe = torch.cat([seq_c, seq_e], dim=0)
    pool = torch.cat([pool_c, pool_e], dim=0)
    cond = {"text_embeds": pool,
            "time_ids": add_time_ids(height=height, width=width, batch_size=2,
                                     device=device, dtype=dtype)}

    def forward(x16, timestep):
        latent_input = scheduler.scale_model_input(x16.repeat(2, 1, 1, 1), timestep)
        noise = unet(latent_input, timestep, encoder_hidden_states=pe,
                     added_cond_kwargs=cond, timestep_cond=None).sample
        eps_cond_raw, eps_uncond = noise.chunk(2)
        return guided_eps(eps_cond_raw, eps_uncond, guidance_scale)
    return forward


def _three_branch(unet, scheduler, seq_a, pool_a, seq_b, pool_b, seq_e, pool_e, guidance_scale,
                  height, width, device, dtype):
    """``run_cfg_poe``'s per-step call, op for op."""
    pe = torch.cat([seq_a, seq_b, seq_e], dim=0)
    pool = torch.cat([pool_a, pool_b, pool_e], dim=0)
    cond = {"text_embeds": pool,
            "time_ids": add_time_ids(height=height, width=width, batch_size=3,
                                     device=device, dtype=dtype)}

    def forward(x16, timestep):
        latent_input = scheduler.scale_model_input(x16.repeat(3, 1, 1, 1), timestep)
        noise = unet(latent_input, timestep, encoder_hidden_states=pe,
                     added_cond_kwargs=cond, timestep_cond=None).sample
        eps_a_raw, eps_b_raw, eps_uncond = noise.chunk(3)
        eps_a = guided_eps(eps_a_raw, eps_uncond, guidance_scale)
        eps_b = guided_eps(eps_b_raw, eps_uncond, guidance_scale)
        return poe_eps(eps_a, eps_b, eps_uncond)
    return forward


def _langevin_level(x32, score_of, delta, n_inner, gen, dtype):
    noise_std = math.sqrt(2.0 * delta)
    for _ in range(n_inner):
        score = score_of(x32.to(dtype))
        z = torch.randn(x32.shape, generator=gen, device=x32.device, dtype=torch.float32)
        x32 = x32 + delta * score + noise_std * z
    return x32


def _uhmc_level(x32, score_of, step, n_inner, n_leapfrog, gen, dtype):
    score = score_of(x32.to(dtype)) if n_inner > 0 else None
    for _ in range(n_inner):
        p = torch.randn(x32.shape, generator=gen, device=x32.device, dtype=torch.float32)
        p = p + 0.5 * step * score
        for leap in range(n_leapfrog):
            x32 = x32 + step * p
            score = score_of(x32.to(dtype))
            p = p + (step if leap < n_leapfrog - 1 else 0.5 * step) * score
    return x32


@torch.no_grad()
def _corrected_sampler(*, forward, kernel, init_latents, models, scheduler, num_inference_steps,
                       euler_init_noise_sigma, device, dtype, k, c, n_leapfrog,
                       corrector_window, noise_seed, decode, rule):
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if n_leapfrog < 1:
        raise ValueError(f"L must be >= 1, got {n_leapfrog}")
    scheduler.set_timesteps(num_inference_steps)
    latents = (init_latents / euler_init_noise_sigma).to(device=device, dtype=dtype)
    tracker = LatentTrajectoryCollector(
        num_inference_steps, 1, latents.shape[1], latents.shape[2], latents.shape[3]
    )
    gen = torch.Generator(device=device).manual_seed(int(noise_seed))
    calls = {"n": 0}

    def counted(x16, timestep):
        calls["n"] += 1
        return forward(x16, timestep)

    per_step: list[dict] = []
    for step_index, timestep in enumerate(scheduler.timesteps):
        t_int = int(timestep.item())
        alpha_bar_t = scheduler.alphas_cumprod[t_int].to(device=device, dtype=dtype)
        beta_t = float(scheduler.betas[t_int].item())
        sigma_t = math.sqrt(max(1.0 - float(alpha_bar_t.item()), 1e-12))
        n_inner = int(k) if corrector_active(step_index, corrector_window) else 0
        delta = float(c) * beta_t
        row = {"step_index": int(step_index), "timestep": t_int, "beta_t": beta_t,
               "k_applied": n_inner, "delta_t": delta if n_inner > 0 else 0.0}

        if n_inner > 0:
            def score_of(x16, _t=timestep, _s=sigma_t):
                return -counted(x16, _t).float() / _s
            x_start32 = latents.float()
            if kernel == "langevin":
                x32 = _langevin_level(x_start32.clone(), score_of, delta, n_inner, gen, dtype)
            else:
                x32 = _uhmc_level(x_start32.clone(), score_of, math.sqrt(2.0 * delta), n_inner,
                                  n_leapfrog, gen, dtype)
            start_norm = _norm(x_start32)
            row["chain_disp_rel"] = _norm(x32 - x_start32) / max(start_norm, 1e-12)
            row["latent_norm_rel"] = _norm(x32) / max(start_norm, 1e-12)
            latents = x32.to(dtype)
        per_step.append(row)

        # The settled point. This call is the reference sampler's own per-step call, and at
        # k = 0 it is the only call made.
        eps = counted(latents, timestep)
        x0 = tweedie_mean(latents, alpha_bar_t, eps)
        tracker.store_step(step_index, latents, eps,
                           float(step_index) / float(num_inference_steps), t_int)
        latents = ddim_prev_from_x0_eps(scheduler=scheduler, timestep=timestep,
                                        step_index=step_index, x0=x0, eps=eps)
    tracker.store_final(latents)
    image = decode_latents(models, latents).cpu() if decode else None
    return SamplerOutputs(
        latents=latents, image=image, tracker=tracker,
        extras={
            "k": int(k), "c": float(c), "L": int(n_leapfrog) if kernel == "uhmc" else None,
            "corrector_window": (None if corrector_window is None
                                 else [int(corrector_window[0]), int(corrector_window[1])]),
            "noise_seed": int(noise_seed), "nfe": int(calls["n"]),
            "step_size_rule": rule, "per_step": per_step,
        },
    )


def run_joint_langevin(*, init_latents, models, scheduler, seq_j, pool_j, seq_e, pool_e,
                       guidance_scale, num_inference_steps, height, width,
                       euler_init_noise_sigma, device, dtype, k, c,
                       corrector_window=None, noise_seed=0, decode=True) -> SamplerOutputs:
    """Joint-prompt CFG with ``k`` Langevin steps per level; ``k = 0`` is ``run_cfg``."""
    forward = _two_branch(models["unet"], scheduler, seq_j, pool_j, seq_e, pool_e,
                          guidance_scale, height, width, device, dtype)
    return _corrected_sampler(
        forward=forward, kernel="langevin", init_latents=init_latents, models=models,
        scheduler=scheduler, num_inference_steps=num_inference_steps,
        euler_init_noise_sigma=euler_init_noise_sigma, device=device, dtype=dtype, k=k, c=c,
        n_leapfrog=1, corrector_window=corrector_window, noise_seed=noise_seed, decode=decode,
        rule="delta_t = c * beta_t; x <- x + delta_t * (-eps_J / sqrt(1 - alpha_bar_t)) + sqrt(2 delta_t) z",
    )


def run_poe_uhmc(*, init_latents, models, scheduler, seq_a, pool_a, seq_b, pool_b, seq_e, pool_e,
                 guidance_scale, num_inference_steps, height, width, euler_init_noise_sigma,
                 device, dtype, k, c, n_leapfrog=5, corrector_window=None, noise_seed=0,
                 decode=True) -> SamplerOutputs:
    """PoE with ``k`` unadjusted HMC iterations of ``L`` leapfrog steps per level; ``k = 0`` is ``run_cfg_poe``."""
    forward = _three_branch(models["unet"], scheduler, seq_a, pool_a, seq_b, pool_b, seq_e, pool_e,
                            guidance_scale, height, width, device, dtype)
    return _corrected_sampler(
        forward=forward, kernel="uhmc", init_latents=init_latents, models=models,
        scheduler=scheduler, num_inference_steps=num_inference_steps,
        euler_init_noise_sigma=euler_init_noise_sigma, device=device, dtype=dtype, k=k, c=c,
        n_leapfrog=n_leapfrog, corrector_window=corrector_window, noise_seed=noise_seed,
        decode=decode,
        rule="eps_t = sqrt(2 c beta_t); per iteration p ~ N(0, I), L leapfrog steps on -eps_PoE / sqrt(1 - alpha_bar_t), no accept step",
    )
