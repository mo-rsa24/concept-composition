"""A fine-tuned CLIP checker as a push head: it scores the sampler's decoded estimate of the finished picture.

The readout checkers read SDXL's own features at the noisy latent. This one reads pictures, so at each step it
forms the sampler's estimate of the finished latent, x0 = (x_t - sqrt(1 - alpha_bar_t) eps) / sqrt(alpha_bar_t),
decodes it with the VAE, resizes it to the checker's input, and returns log p(clean cat and dog). The push
function passes the product's noise prediction eps (``reads_estimate``); eps is held fixed, so the gradient
with respect to x_t is the gradient with respect to x0 divided by sqrt(alpha_bar_t) and no UNet backward pass
is needed. The decode runs in a float32 copy of the VAE, because gradients through the half-precision decoder
can overflow. Called without eps (best-of-2's final pick) the latent is read as already finished.

Checkpoints are finetune_checker.py's or finetune_estimates.py's best.pt.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
from finetune_checker import SPECS, Checker  # noqa: E402
from poe_repair._sdxl.sdipc_utils import decode_latents_with_grad  # noqa: E402


class ClipEstimateChecker(nn.Module):
    reads_estimate = True

    def __init__(self, ckpt: Path, vae, device):
        super().__init__()
        state = torch.load(ckpt, map_location="cpu")
        self.net = Checker(state["model"], n_out=state.get("n_out", 1))
        self.net.load_state_dict(state["state"])
        self.net = self.net.float().eval().to(device).requires_grad_(False)
        spec = SPECS[state["model"]]
        self.res = spec["res"]
        self.register_buffer("mean", torch.tensor(spec["mean"], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(spec["std"], device=device).view(1, 3, 1, 1))
        object.__setattr__(self, "vae32", copy.deepcopy(vae).float().eval().requires_grad_(False))
        self.epoch = state.get("epoch")

    def forward(self, x_t, timestep, eps=None, alpha_bar=None):
        x_t = x_t.float()
        x0 = x_t if eps is None else (x_t - (1 - alpha_bar).sqrt() * eps.float()) / alpha_bar.sqrt()
        img = decode_latents_with_grad(self.vae32, x0)
        img = F.interpolate(img, size=(self.res, self.res), mode="bicubic", antialias=True, align_corners=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = self.net((img - self.mean) / self.std)
        return logit.float()

    def log_prob_both(self, x_t, timestep, eps=None, alpha_bar=None):
        # one output: log p(clean pair); two (the species checker): log p(a clear cat) + log p(a clear dog)
        return F.logsigmoid(self.forward(x_t, timestep, eps, alpha_bar)).sum()
