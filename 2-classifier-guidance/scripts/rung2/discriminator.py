"""Rung 2's pair-aware discriminator: "is this noisy latent a cat and a dog apart?"

Input: a noisy SDXL latent (B, 4, 128, 128) and its timestep. SDXL's own UNet, frozen and read
on the empty prompt, runs its down and middle blocks and hands over the middle block's output
(B, 1280, 32, 32). Plan 18's attention-pool head, the only trained part, turns that into one
logit: the log-odds of "a cat and a dog apart" against "the product's render, a lone cat, or a
lone dog".

The UNet is shared with the sampler. ``UNetMidFeatures`` arms its forward hook only while it reads
features, so the sampler's own forward pass is untouched, and nothing in the UNet is trained.

``log_prob_both`` is the name the product sampler calls. Here it returns the summed logit, so the
sampler's push is the log-odds gradient the rung 2 review file defines. Plan 18's classifier
returned log p instead, whose gradient fades to zero as p nears 1.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from poe_repair.experiments.presence_head.dog_classifier import AttentionPoolHead
from poe_repair.experiments.presence_head.strong_head import UNetMidFeatures


class PairDiscriminator(nn.Module):
    def __init__(self, unet, seq_e: torch.Tensor, pool_e: torch.Tensor) -> None:
        super().__init__()
        self.features = UNetMidFeatures(unet, seq_e, pool_e)
        self.head = AttentionPoolHead()

    def head_logit(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(feat)[:, 0]

    def forward(self, x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.head_logit(self.features(x_t, timestep))

    def log_prob_both(self, x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.forward(x_t, timestep).sum()


def head_parameter_count(model: PairDiscriminator) -> int:
    return sum(p.numel() for p in model.head.parameters())
