"""A noise-aware dog classifier in the shape Dhariwal and Nichol use for guidance.

Their classifier is the diffusion model's own downsampling trunk with an attention pool on top,
trained on noisy pictures at the model's noise levels. Here the trunk is SDXL's down and middle
blocks, trained through LoRA adapters on their attention projections, read on the empty prompt.

It runs on its own copy of the UNet. The sampler's copy is never modified, so guidance cannot leak
the adapter into the model it is guiding, which an earlier sampler in this repo did by leaving an
adapter enabled.

``log_prob_both`` keeps the signature the guided samplers call, and here returns log p(dog | x_t).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from poe_repair.experiments.presence_head.strong_head import UNetMidFeatures

LORA_TARGET_REGEX = r".*(down_blocks|mid_block).*\.(to_q|to_k|to_v|to_out\.0)"


def attach_trunk_adapters(unet, rank: int = 16, alpha: int = 16):
    """LoRA on the attention projections of the down and middle blocks only, in fp32."""
    from peft import LoraConfig, inject_adapter_in_model

    unet.requires_grad_(False)
    config = LoraConfig(r=rank, lora_alpha=alpha, target_modules=LORA_TARGET_REGEX,
                        lora_dropout=0.0, bias="none")
    unet = inject_adapter_in_model(config, unet)
    trainable = []
    for name, p in unet.named_parameters():
        if "lora_" in name:
            p.data = p.data.float()
            p.requires_grad_(True)
            trainable.append(p)
    return unet, trainable


class AttentionPoolHead(nn.Module):
    """Middle-block features (B, 1280, 32, 32) to one dog logit, pooled by attention at 8 x 8."""

    def __init__(self, in_channels: int = 1280, width: int = 512, heads: int = 8) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, stride=2, padding=1), nn.GroupNorm(32, width), nn.SiLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.GroupNorm(32, width), nn.SiLU(),
        )
        self.pos = nn.Parameter(torch.randn(1, 64 + 1, width) / math.sqrt(width))
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.out = nn.Linear(width, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        h = self.down(feat.float())                       # (B, W, 8, 8)
        tokens = h.flatten(2).transpose(1, 2)              # (B, 64, W)
        query = tokens.mean(dim=1, keepdim=True)           # the paper's pool queries the mean token
        x = torch.cat([query, tokens], dim=1) + self.pos
        pooled, _ = self.attn(x[:, :1], x, x)
        return self.out(pooled[:, 0])                      # (B, 1)


class DogClassifier(nn.Module):
    def __init__(self, features: UNetMidFeatures, head: AttentionPoolHead) -> None:
        super().__init__()
        self.features, self.head = features, head

    def forward(self, x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x_t, timestep))

    def log_prob_both(self, x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return F.logsigmoid(self.forward(x_t, timestep)[:, 0]).sum()
