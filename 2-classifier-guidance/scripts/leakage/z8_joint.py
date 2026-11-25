"""Loop 12 step Z8: the joint prompt as a base for the checker's push.

``run_cfg`` (poe_repair/methods/_sampling.py), plain classifier-free guidance on one prompt, with the push of
``run_poe_classifier_guided`` (poe_repair/methods/_classifier_guided.py) added after the guided prediction, in the same
order and with the same arithmetic as that function's plain path (no recurrence, adapter, backward steps or eta). At
strength 0 with no random push, nothing below the prediction runs, so the picture must equal ``run_cfg``'s byte for byte;
z8_screen.py checks that before anything is drawn. The model is called on [joint prompt, empty prompt], a batch of two,
as ``run_cfg`` calls it: drawing the joint prompt through the three-entry product sampler instead differs by a mean of
3.2 grey levels over 50 steps at half precision, because the batch of three rounds differently.
"""
from __future__ import annotations

import torch

from poe_repair.methods._classifier_guided import PUSH_CAP_FRACTION, capped_push
from poe_repair.methods._sampling import SamplerOutputs, add_time_ids
from poe_repair.runtime import LatentTrajectoryCollector, ddim_prev_from_x0_eps, decode_latents, guided_eps, tweedie_mean


def run_joint_classifier_guided(*, init_latents, models, scheduler, seq_cond, pool_cond, seq_e, pool_e, guidance_scale,
                                num_inference_steps, height, width, euler_init_noise_sigma, device, dtype, head=None,
                                guidance_window=None, strength=0.0, push_cap=PUSH_CAP_FRACTION, random_push_norms=None,
                                random_generator=None) -> SamplerOutputs:
    scheduler.set_timesteps(num_inference_steps)
    latents = (init_latents / euler_init_noise_sigma).to(device=device, dtype=dtype)
    tracker = LatentTrajectoryCollector(num_inference_steps, 1, latents.shape[1], latents.shape[2], latents.shape[3])
    pe = torch.cat([seq_cond, seq_e], dim=0)
    pool = torch.cat([pool_cond, pool_e], dim=0)
    cond = {"text_embeds": pool, "time_ids": add_time_ids(height=height, width=width, batch_size=2, device=device, dtype=dtype)}
    unet = models["unet"]
    if head is not None:
        was_training = head.training
        head.eval()
    push_i, grad_norms, before_cap, applied, capped_list = 0, [], [], [], []
    for step_index, timestep in enumerate(scheduler.timesteps):
        with torch.no_grad():
            latent_input = scheduler.scale_model_input(latents.repeat(2, 1, 1, 1), timestep)
            noise = unet(latent_input, timestep, encoder_hidden_states=pe, added_cond_kwargs=cond, timestep_cond=None).sample
            eps_cond_raw, eps_uncond = noise.chunk(2)
            eps_t = guided_eps(eps_cond_raw, eps_uncond, guidance_scale)
        in_window = guidance_window is None or guidance_window[0] <= step_index < guidance_window[1]
        if random_push_norms is not None:
            direction = torch.randn(eps_t.shape, generator=random_generator).to(device=device, dtype=torch.float32)
            push = direction * (float(random_push_norms[push_i]) / direction.norm())
            eps_t = (eps_t.float() - push).to(dtype)
            grad_norms.append(0.0); before_cap.append(float(push.norm())); applied.append(float(push.norm())); capped_list.append(False)
            push_i += 1
        elif head is not None and strength != 0.0 and in_window:
            with torch.enable_grad():
                z = latents.detach().float().requires_grad_(True)
                t_batch = timestep.reshape(1).to(device=device).expand(z.shape[0])
                alpha_bar = scheduler.alphas_cumprod[int(timestep.item())].to(device=device).float()
                log_p = head.log_prob_both(z, t_batch)
                grad = torch.autograd.grad(log_p, z)[0]
            sigma_t = torch.sqrt(1.0 - alpha_bar)
            push, push_before_cap, capped = capped_push(float(strength) * sigma_t * grad, eps_t, push_cap)
            eps_t = (eps_t.float() - push).to(dtype)
            grad_norms.append(float(grad.norm())); before_cap.append(push_before_cap); applied.append(float(push.norm()))
            capped_list.append(capped)
        else:
            grad_norms.append(0.0); before_cap.append(0.0); applied.append(0.0); capped_list.append(False)
        with torch.no_grad():
            alpha_bar_t = scheduler.alphas_cumprod[int(timestep.item())].to(device=device, dtype=dtype)
            x0 = tweedie_mean(latents, alpha_bar_t, eps_t)
            tracker.store_step(step_index, latents, eps_t, float(step_index) / float(num_inference_steps), int(timestep.item()))
            latents = ddim_prev_from_x0_eps(scheduler=scheduler, timestep=timestep, step_index=step_index, x0=x0, eps=eps_t)
    tracker.store_final(latents)
    with torch.no_grad():
        image = decode_latents(models, latents).cpu()
    if head is not None and was_training:
        head.train()
    return SamplerOutputs(latents=latents, image=image, tracker=tracker, extras={
        "strength": float(strength), "push_cap_fraction": None if push_cap is None else float(push_cap),
        "guidance_window": list(guidance_window) if guidance_window else None, "grad_norm_per_step": grad_norms,
        "push_before_cap_norm_per_step": before_cap, "applied_norm_per_step": applied, "capped_per_step": capped_list,
        "random_push": random_push_norms is not None})
