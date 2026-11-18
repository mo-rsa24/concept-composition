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
  3. The push is strength * sqrt(1 - alpha_bar_t) * grad, Dhariwal and Nichol's Algorithm 2 with
     nothing rescaled, scaled down if needed so its norm is at most ``PUSH_CAP_FRACTION`` of the
     product prediction's norm, then subtracted.
  4. The ordinary DDIM step follows.

At ``strength == 0`` the head is never called and nothing touches the prediction, so the rollout
is byte-identical to ``run_cfg_poe``. That canary is the gate: a composer that moves the baseline
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

# The push may be at most this fraction of the product prediction's norm, per step. Dhariwal and
# Nichol (arXiv 2105.05233) cap nothing, so this value is ours: route step 9's criterion sets the
# final one. On plan 17's dog sheet a cap of 0.1 never fires at strengths 10 and 30 and fires on
# some step of every render at 100 and 300.
PUSH_CAP_FRACTION = 0.1


def capped_push(push: torch.Tensor, eps: torch.Tensor, cap: float | None):
    """Scale ``push`` down so its norm is at most ``cap`` times the norm of ``eps``.

    Scaling keeps the direction; only the length changes. Returns the push to apply, the norm
    before the cap, and whether the cap fired. ``cap=None`` switches the cap off.
    """
    raw_norm = push.norm()
    if cap is None:
        return push, float(raw_norm), False
    limit = float(cap) * eps.float().norm()
    if raw_norm > limit:
        return push * (limit / raw_norm), float(raw_norm), True
    return push, float(raw_norm), False


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
    strength: float = 0.0,
    push_cap: float | None = PUSH_CAP_FRACTION,
    random_push_norms: list[float] | None = None,
    random_generator: torch.Generator | None = None,
    adapter_lambda: float | None = None,
    recurrence: int = 1,
    push_cap_late: float | None = None,
    push_cap_late_from: int = 10,
    recurrence_window: tuple[int, int] | None = None,
    recurrence_generator: torch.Generator | None = None,
    backward_steps: int = 0,
    eta: float = 0.0,
    eta_generator: torch.Generator | None = None,
    backward_lr: float = 0.05,
) -> SamplerOutputs:
    """``run_cfg_poe`` plus a classifier push at each step.

    With ``adapter_lambda`` set, the UNet must carry the rank-32 adapter
    (``noise_search.adapter.attach_and_load_lora``) and the base prediction becomes
    eps_off + adapter_lambda * (eps_on - eps_off), the PoE prediction with the adapter switched off
    and on, as ``noise_search.adapter.run_adapter_candidates`` computes it. The adapter is switched
    off again before the push, so a checker that reads the UNet's features reads the frozen SDXL it
    was trained on, and it is left off when the function returns. At ``adapter_lambda`` 0 both
    predictions are still computed and the adapter's is discarded. ``None`` is the old path.

    With ``random_push_norms`` the classifier is not called: at step ``i`` the push is a fresh
    random direction from ``random_generator``, scaled to norm ``random_push_norms[i]``. Fed the
    applied norms a guided run logged, this is the size-matched random control; only the push's
    direction differs from the guided run.

    The push is Dhariwal and Nichol's Algorithm 2 with nothing rescaled:
    eps_PoE - strength * sqrt(1 - alpha_bar_t) * grad log p(y | x_t), shortened to at most
    ``push_cap`` times the norm of eps_PoE. ``weight`` named the older form, which rescaled the
    gradient to the prediction's norm and over-pushed by about four orders of magnitude; it is
    removed, and a nonzero value raises so an old caller cannot run it silently.
    """
    if weight != 0.0:
        raise ValueError(
            "the rescaled 'weight' push was removed because it over-pushed by about 44,000 times; "
            "pass 'strength' (Dhariwal and Nichol's scale) instead")
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

    push_i = 0  # the random push's running index: a repeated step takes a new norm each time
    grad_norm_per_step: list[float] = []
    push_before_cap_per_step: list[float] = []
    applied_per_step: list[float] = []
    capped_per_step: list[bool] = []
    eps_norm_per_step: list[float] = []

    if adapter_lambda is not None:
        from poe_repair.experiments.noise_search.adapter import adapter_disable, adapter_enable

    def product_eps(timestep) -> torch.Tensor:
        latent_input = scheduler.scale_model_input(latents.repeat(3, 1, 1, 1), timestep)
        noise = unet(
            latent_input, timestep, encoder_hidden_states=pe,
            added_cond_kwargs=cond, timestep_cond=None,
        ).sample
        eps_a_raw, eps_b_raw, eps_uncond = noise.chunk(3)
        eps_a = guided_eps(eps_a_raw, eps_uncond, guidance_scale)
        eps_b = guided_eps(eps_b_raw, eps_uncond, guidance_scale)
        return poe_eps(eps_a, eps_b, eps_uncond)

    for step_index, timestep in enumerate(scheduler.timesteps):
        reps = recurrence if (recurrence > 1 and recurrence_window is not None
                              and recurrence_window[0] <= step_index < recurrence_window[1]) else 1
        for rep in range(reps):
            if rep > 0:
                # Universal Guidance's self-recurrence (Bansal et al., Algorithm 1): the guided step
                # from x_t to x_prev is undone by adding back the noise between the two levels,
                # z_t = sqrt(a_t / a_prev) x_prev + sqrt(1 - a_t / a_prev) noise, and the step is taken again
                with torch.no_grad():
                    a_t = scheduler.alphas_cumprod[int(timestep.item())].to(device=device, dtype=torch.float32)
                    nxt = scheduler.timesteps[step_index + 1] if step_index + 1 < len(scheduler.timesteps) else None
                    a_prev = (scheduler.alphas_cumprod[int(nxt.item())] if nxt is not None
                              else torch.as_tensor(scheduler.final_alpha_cumprod)).to(device=device, dtype=torch.float32)
                    ratio = a_t / a_prev
                    n = torch.randn(latents.shape, generator=recurrence_generator).to(device=device, dtype=torch.float32)
                    latents = (ratio.sqrt() * latents.float() + (1 - ratio).sqrt() * n).to(dtype)
            with torch.no_grad():
                if adapter_lambda is None:
                    eps_p = product_eps(timestep)
                else:
                    adapter_disable(unet)
                    eps_off = product_eps(timestep)
                    adapter_enable(unet)
                    eps_on = product_eps(timestep)
                    adapter_disable(unet)
                    eps_p = eps_off + float(adapter_lambda) * (eps_on - eps_off)
            eps_norm_per_step.append(float(eps_p.float().norm()))

            in_window = guidance_window is None or (
                guidance_window[0] <= step_index < guidance_window[1])
            if random_push_norms is not None:
                direction = torch.randn(eps_p.shape, generator=random_generator).to(
                    device=device, dtype=torch.float32)
                push = direction * (float(random_push_norms[push_i]) / direction.norm())
                eps_p = (eps_p.float() - push).to(dtype)
                grad_norm_per_step.append(0.0)
                push_before_cap_per_step.append(float(push.norm()))
                applied_per_step.append(float(push.norm()))
                capped_per_step.append(False)
                push_i += 1
            # The canary lives here: at strength 0 nothing below runs and eps_p is untouched.
            elif head is not None and strength != 0.0 and in_window:
                with torch.enable_grad():
                    z = latents.detach().float().requires_grad_(True)
                    t_batch = timestep.reshape(1).to(device=device).expand(z.shape[0])
                    alpha_bar = scheduler.alphas_cumprod[int(timestep.item())].to(device=device).float()
                    if getattr(head, "reads_estimate", False):
                        # a picture checker: it decodes the estimate of the finished picture from x_t and
                        # this step's product prediction, held fixed (rung2/clip_push.py)
                        log_p = head.log_prob_both(z, t_batch, eps=eps_p.detach().float(), alpha_bar=alpha_bar)
                    else:
                        log_p = head.log_prob_both(z, t_batch)
                    grad = torch.autograd.grad(log_p, z)[0]
                sigma_t = torch.sqrt(1.0 - alpha_bar)
                if backward_steps > 0 and getattr(head, "reads_estimate", False):
                    # Universal Guidance's backward guidance (Bansal et al., eq. 7 to 9): several Adam steps on a change
                    # delta to the estimate of the finished latent, scored as a finished picture, then carried into
                    # the noise prediction: x0 + delta needs eps - sqrt(alpha_bar / (1 - alpha_bar)) * delta
                    with torch.enable_grad():
                        x0_hat = ((latents.float() - sigma_t * eps_p.float()) / alpha_bar.sqrt()).detach()
                        delta = torch.zeros_like(x0_hat, requires_grad=True)
                        bopt = torch.optim.Adam([delta], lr=float(backward_lr))
                        for _ in range(int(backward_steps)):
                            bopt.zero_grad(set_to_none=True)
                            (-head.log_prob_both(x0_hat + delta, t_batch)).backward()
                            bopt.step()
                    raw = (alpha_bar.sqrt() / sigma_t) * delta.detach()
                else:
                    raw = float(strength) * sigma_t * grad
                push, push_before_cap, capped = capped_push(
                    raw, eps_p,
                    push_cap_late if (push_cap_late is not None and step_index >= push_cap_late_from) else push_cap)
                eps_p = (eps_p.float() - push).to(dtype)
                grad_norm_per_step.append(float(grad.norm()))
                push_before_cap_per_step.append(push_before_cap)
                applied_per_step.append(float(push.norm()))
                capped_per_step.append(capped)
            else:
                grad_norm_per_step.append(0.0)
                push_before_cap_per_step.append(0.0)
                applied_per_step.append(0.0)
                capped_per_step.append(False)

            with torch.no_grad():
                alpha_bar_t = scheduler.alphas_cumprod[int(timestep.item())].to(
                    device=device, dtype=dtype)
                x0 = tweedie_mean(latents, alpha_bar_t, eps_p)
                tracker.store_step(
                    step_index, latents, eps_p,
                    float(step_index) / float(num_inference_steps),
                    int(timestep.item()),
                )
                if eta > 0:
                    # a partly random sampler (DDIM, Song et al. eq. 12), its noise fixed per seed by eta_generator,
                    # so each seed is still one trajectory; the MNIST step 7b loop found eta 1.0 lifted the push
                    from poe_repair.experiments.twisted_smc.sampler import ddim_step_eta
                    latents, _ = ddim_step_eta(scheduler=scheduler, timestep=timestep, step_index=step_index,
                                               x_t=latents, x0=x0, eps=eps_p, eta=float(eta), generator=eta_generator)
                else:
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
            "strength": float(strength),
            "push_cap_fraction": None if push_cap is None else float(push_cap),
            "guidance_window": list(guidance_window) if guidance_window else None,
            "grad_norm_per_step": grad_norm_per_step,
            "push_before_cap_norm_per_step": push_before_cap_per_step,
            "applied_norm_per_step": applied_per_step,
            "capped_per_step": capped_per_step,
            "eps_norm_per_step": eps_norm_per_step,
            "random_push": random_push_norms is not None,
            "adapter_lambda": None if adapter_lambda is None else float(adapter_lambda),
            "recurrence": int(recurrence),
            "backward_steps": int(backward_steps),
            "eta": float(eta),
            "backward_lr": float(backward_lr),
            "push_cap_late": None if push_cap_late is None else float(push_cap_late),
            "push_cap_late_from": int(push_cap_late_from),
            "recurrence_window": list(recurrence_window) if recurrence_window else None,
        },
    )


def run_cfg_classifier_guided(
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
    head=None,
    strength: float = 0.0,
) -> SamplerOutputs:
    """Ordinary guidance plus classifier guidance exactly as Dhariwal and Nichol write it.

    Algorithm 2 of arXiv 2105.05233: the noise prediction becomes
    eps - strength * sqrt(1 - alpha_bar_t) * grad log p(y | x_t), on every step, with nothing
    rescaled. The noise factor shrinks the push as the picture clears. With the empty prompt in
    both branches the base prediction is the unconditional one, which is the paper's case.

    At ``strength == 0`` the head is never called, so the rollout is byte-identical to ``run_cfg``.
    """
    scheduler.set_timesteps(num_inference_steps)
    latents = (init_latents / euler_init_noise_sigma).to(device=device, dtype=dtype)
    tracker = LatentTrajectoryCollector(
        num_inference_steps, 1, latents.shape[1], latents.shape[2], latents.shape[3]
    )
    pe = torch.cat([seq_cond, seq_e], dim=0)
    pool = torch.cat([pool_cond, pool_e], dim=0)
    cond = {"text_embeds": pool,
            "time_ids": add_time_ids(height=height, width=width, batch_size=2,
                                     device=device, dtype=dtype)}
    unet = models["unet"]
    if head is not None:
        head.eval()
    grad_norm, push_norm, eps_norm = [], [], []
    for step_index, timestep in enumerate(scheduler.timesteps):
        with torch.no_grad():
            latent_input = scheduler.scale_model_input(latents.repeat(2, 1, 1, 1), timestep)
            noise = unet(latent_input, timestep, encoder_hidden_states=pe,
                         added_cond_kwargs=cond, timestep_cond=None).sample
            eps_cond, eps_uncond = noise.chunk(2)
            eps = guided_eps(eps_cond, eps_uncond, guidance_scale)
            alpha_bar_t = scheduler.alphas_cumprod[int(timestep.item())].to(device=device, dtype=dtype)
        eps_norm.append(float(eps.float().norm()))
        if head is not None and strength != 0.0:
            with torch.enable_grad():
                z = latents.detach().float().requires_grad_(True)
                log_p = head.log_prob_both(z, timestep.reshape(1).to(device))
                grad = torch.autograd.grad(log_p, z)[0]
            sigma_t = torch.sqrt(1.0 - alpha_bar_t.float())
            push = float(strength) * sigma_t * grad
            eps = (eps.float() - push).to(dtype)
            grad_norm.append(float(grad.norm()))
            push_norm.append(float(push.norm()))
        else:
            grad_norm.append(0.0)
            push_norm.append(0.0)
        with torch.no_grad():
            x0 = tweedie_mean(latents, alpha_bar_t, eps)
            tracker.store_step(step_index, latents, eps,
                               float(step_index) / float(num_inference_steps), int(timestep.item()))
            latents = ddim_prev_from_x0_eps(scheduler=scheduler, timestep=timestep,
                                            step_index=step_index, x0=x0, eps=eps)
    tracker.store_final(latents)
    with torch.no_grad():
        image = decode_latents(models, latents).cpu()
    return SamplerOutputs(latents=latents, image=image, tracker=tracker,
                          extras={"strength": float(strength), "grad_norm_per_step": grad_norm,
                                  "push_norm_per_step": push_norm, "eps_norm_per_step": eps_norm})
