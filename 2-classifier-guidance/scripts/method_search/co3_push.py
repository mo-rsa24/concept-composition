"""Add the checker's push to CO3 without changing a file of the authors' code (step M1, Paper 2).

CO3's main prediction is the only UNet call in a step with a batch of two (unconditional, joint prompt). Its corrector calls
use one entry per concept plus two, so at least three. PushingUNet wraps the UNet and, on that one call, subtracts the push
from both entries:

    guided = (u - p) + g * ((c - p) - (u - p)) = u + g * (c - u) - p

so the push leaves CO3's guided prediction and its unconditional prediction each lower by p, and the CFG++ step moves the
latent the way one prediction filling both roles does in the kept recipe's DDIM step.
"""
import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput


class PushingUNet(torch.nn.Module):
    """The UNet, with an optional push applied to CO3's main two-entry prediction."""

    def __init__(self, unet):
        super().__init__()
        self.inner = unet
        self.pusher = None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("inner"), name)

    def forward(self, sample, timestep, *args, **kwargs):
        out = self.inner(sample, timestep, *args, **kwargs)
        if self.pusher is None or sample.shape[0] != 2:
            return out
        eps = out["sample"]
        push = self.pusher(sample[:1], timestep, eps).to(device=eps.device, dtype=eps.dtype)
        return UNet2DConditionOutput(sample=eps - push)


class Pusher:
    """One render's push. mode: "push" (the checker), "zero" (the checker's path, times zero) or "random" (matched lengths)."""

    def __init__(self, checker, guidance, strength, cap, window, mode="push", norms=None, generator=None):
        self.checker, self.guidance, self.strength, self.cap = checker, float(guidance), float(strength), float(cap)
        self.window, self.mode, self.norms, self.generator = tuple(window), mode, norms, generator
        self.step, self.alpha_bar = None, None
        self.applied, self.capped, self.main_calls = [], [], {}

    def __call__(self, x, timestep, eps):
        self.main_calls[self.step] = self.main_calls.get(self.step, 0) + 1
        if self.main_calls[self.step] > 1:
            raise RuntimeError(f"CO3 made more than one two-entry UNet call on step {self.step}")
        if not self.window[0] <= self.step < self.window[1]:
            self.applied.append(0.0)
            self.capped.append(False)
            return torch.zeros_like(x)
        if self.mode == "random":
            direction = torch.randn(x.shape, generator=self.generator).to(device=x.device, dtype=torch.float32)
            push = direction * (float(self.norms[self.step]) / direction.norm())
            self.applied.append(float(push.norm()))
            self.capped.append(False)
            return push
        with torch.enable_grad():
            z = x.detach().float().requires_grad_(True)
            log_p = self.checker.log_prob_both(z, timestep.reshape(1).to(x.device))
            grad = torch.autograd.grad(log_p, z)[0]
        sigma = (1.0 - self.alpha_bar) ** 0.5
        strength = 0.0 if self.mode == "zero" else self.strength
        raw = strength * sigma * grad
        uncond, cond = eps[:1].float(), eps[1:2].float()
        limit = self.cap * (uncond + self.guidance * (cond - uncond)).norm()
        raw_norm = raw.norm()
        fired = bool(raw_norm > limit)
        push = raw * (limit / raw_norm) if fired else raw
        self.applied.append(float(push.norm()))
        self.capped.append(fired)
        return push


def attach(model):
    """Wrap the UNet once and make each step tell the current pusher where it is. Returns the wrapper."""
    wrapper = PushingUNet(model.unet)
    model.unet = wrapper
    original = model.denoise_step

    def denoise_step(x, t, step):
        if wrapper.pusher is not None:
            wrapper.pusher.step = int(step)
            wrapper.pusher.alpha_bar = float(model.alpha(t))
        return original(x, t, step)

    model.denoise_step = denoise_step
    return wrapper
