"""Plain product-of-experts sampling, nudged toward two things in frame by CLIP.

The product of "a cat" and "a dog" draws one fused animal. CLIP, asked how much more a picture
looks like "a cat and a dog" than like "one animal that is half cat and half dog", scored every
one of 8 two-animal renders above every one of 8 fused renders on this repo's own cat x dog cell,
including three that held two dogs or two cats. So the score separates two things in frame from
one fused thing, which is co-occurrence, and it needs no training.

Per step, inside the guidance window:
  1. The frozen three-branch UNet forward gives the product prediction, as ``run_cfg_poe`` does.
  2. The clean picture this step is heading for is decoded with its graph kept.
  3. CLIP scores it: similarity to the pair caption minus similarity to the fused caption.
  4. The gradient of that score with respect to the clean latent is rescaled to a fixed fraction
     of the product prediction's own norm and subtracted from the prediction. The UNet is not
     differentiated through: the prediction is held fixed, so the gradient reaches the latent
     only through the clean estimate, which is the cheap form of guidance on a decoded image.
  5. The ordinary DDIM step follows.

At ``weight == 0`` nothing below the UNet forward runs, so the rollout is byte-identical to
``run_cfg_poe``. The window defaults to the first 20 steps because that is where this repo has
measured the layout being decided; decoding with a graph at every one of 50 steps would cost
more than the rest of the render and push on steps whose layout is already fixed.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from poe_repair.methods._sampling import SamplerOutputs, add_time_ids
from poe_repair.runtime import (
    LatentTrajectoryCollector,
    decode_latents,
    ddim_prev_from_x0_eps,
    guided_eps,
    poe_eps,
    tweedie_mean,
)

PAIR_CAPTION = "a cat and a dog"
FUSED_CAPTION = "one animal that is half cat and half dog"
TEMPLATES = ("{}", "a photograph showing {}", "an image of {}")
DEFAULT_WINDOW = (0, 20)
MIN_GRAD_NORM = 1e-8


class CoOccurrenceScore:
    """CLIP's pair-minus-fused similarity for a decoded picture, kept differentiable.

    With ``augment`` above zero the score is averaged over that many random crops and flips of
    the picture. On seed 9, the single-framing gradient at weight 1.0 raised the score to +0.053,
    above every real two-animal render, by painting dots over one cartoon dog: a pattern that
    fools CLIP at one exact framing. A trick of that kind stops paying when it has to work
    across framings, which is the standard defence, and it costs only CLIP passes, not decodes.
    """

    def __init__(self, reward, pair_caption: str = PAIR_CAPTION,
                 fused_caption: str = FUSED_CAPTION, augment: int = 0,
                 min_crop: float = 0.7, seed: int = 0) -> None:
        # ``reward`` is a PluralityReward, reused for its frozen CLIP and its differentiable
        # image preparation rather than loading a second copy.
        self.reward = reward
        self.augment = int(augment)
        self.min_crop = float(min_crop)
        self.generator = torch.Generator().manual_seed(seed)
        self.pair_dir = self._direction(pair_caption)
        self.fused_dir = self._direction(fused_caption)
        self.pair_caption, self.fused_caption = pair_caption, fused_caption

    def _direction(self, phrase: str) -> torch.Tensor:
        dirs = torch.cat([self.reward.text_embed(t.format(phrase)) for t in TEMPLATES], dim=0)
        return F.normalize(dirs.mean(dim=0, keepdim=True), dim=-1)

    def _views(self, image: torch.Tensor) -> list[torch.Tensor]:
        if self.augment <= 0:
            return [image]
        _, _, h, w = image.shape
        views = []
        for _ in range(self.augment):
            frac = self.min_crop + (1 - self.min_crop) * float(torch.rand(1, generator=self.generator))
            ch, cw = int(h * frac), int(w * frac)
            top = int(torch.randint(0, h - ch + 1, (1,), generator=self.generator))
            left = int(torch.randint(0, w - cw + 1, (1,), generator=self.generator))
            v = image[:, :, top:top + ch, left:left + cw]
            if float(torch.rand(1, generator=self.generator)) < 0.5:
                v = torch.flip(v, dims=[3])
            views.append(v)
        return views

    def __call__(self, image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        pairs, fuseds = [], []
        for v in self._views(image):
            f = self.reward.clip_image_embed(v)
            pairs.append((f * self.pair_dir).sum())
            fuseds.append((f * self.fused_dir).sum())
        pair = torch.stack(pairs).mean()
        fused = torch.stack(fuseds).mean()
        return pair - fused, {"to_pair": float(pair), "to_fused": float(fused)}


def run_poe_clip_guided(
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
    score=None,
    weight: float = 0.0,
    guidance_window: tuple[int, int] = DEFAULT_WINDOW,
) -> SamplerOutputs:
    """``run_cfg_poe`` plus ``weight`` times CLIP's normalised co-occurrence gradient."""
    from poe_repair._sdxl.sdipc_utils import decode_latents_with_grad

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
    per_step: list[dict] = []

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
            alpha_bar_t = scheduler.alphas_cumprod[int(timestep.item())].to(
                device=device, dtype=dtype)

        row = {"step": step_index, "guided": False}
        in_window = guidance_window[0] <= step_index < guidance_window[1]
        # The canary lives here: at weight 0 nothing below runs and eps_p is untouched.
        if score is not None and weight != 0.0 and in_window:
            with torch.enable_grad():
                x0 = tweedie_mean(latents, alpha_bar_t, eps_p).detach().float().requires_grad_(True)
                image = decode_latents_with_grad(models["vae"], x0.to(dtype))
                value, parts = score(image)
                grad, = torch.autograd.grad(value, x0)
            grad_norm = float(grad.norm().item())
            row.update({"guided": True, "score": float(value), **parts, "grad_norm": grad_norm})
            if grad_norm > MIN_GRAD_NORM:
                # Raising the clean estimate along the gradient means lowering the noise
                # prediction along it, since x0 = (x - sigma * eps) / alpha. The weight is a step
                # size in units of the prediction's own norm.
                step = (grad / grad_norm) * eps_p.float().norm() * float(weight)
                eps_p = (eps_p.float() - step).to(dtype)
            del image, grad
        per_step.append(row)

        with torch.no_grad():
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
    return SamplerOutputs(
        latents=latents, image=image, tracker=tracker,
        extras={"clip_weight": float(weight), "guidance_window": list(guidance_window),
                "per_step": per_step},
    )
