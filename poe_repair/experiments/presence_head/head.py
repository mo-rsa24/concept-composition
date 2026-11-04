"""The presence head: two independent yes-or-no answers about a noisy latent.

``forward(x_t, timestep) -> (B, 2)`` logits, for "a cat instance is present" and "a dog instance
is present". Each is its own sigmoid with its own loss, so the two never share a normaliser and a
frame holding two animals can score high on both. That is the whole reason this is not a
two-class head: a two-way softmax has to spend its mass on one species, and measured on 14 fused
cat x dog renders it chose one, confidently, on 13.

The body is ``TwistHead``'s FiLM convolution stack from the twisted-SMC experiment, which already
reads SDXL latents conditioned on a timestep. Two changes: the last layer gives two logits rather
than one, and there is no text conditioning at all, because "is a cat in this picture" does not
depend on what the picture was asked to be.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from poe_repair.experiments.twisted_smc.twist import FiLMBlock, sinusoidal_embedding

LABEL_NAMES = ("cat_present", "dog_present")


class PresenceHead(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 4,
        width: int = 64,
        time_dim: int = 128,
        emb_dim: int = 256,
        dropout: float = 0.1,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.stem = nn.Conv2d(in_channels, width, 3, padding=1)
        chans = [width, 2 * width, 4 * width, 4 * width, 4 * width, 4 * width]
        self.blocks = nn.ModuleList(
            [FiLMBlock(chans[i], chans[i + 1], emb_dim, dropout) for i in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Linear(chans[num_blocks] + emb_dim, emb_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(emb_dim, len(LABEL_NAMES)),
        )

    def forward(self, x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(sinusoidal_embedding(timestep, self.time_dim))
        h = self.stem(x_t.float())
        for block in self.blocks:
            h = block(h, emb)
        pooled = h.mean(dim=(2, 3))
        return self.head(torch.cat([pooled, emb], dim=-1))

    def log_prob_both(self, x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """log p(cat present) + log p(dog present), summed over the batch.

        The sum is what classifier guidance differentiates: two independent questions, both
        wanted true, so their log probabilities add. Nothing here couples the two.
        """
        logits = self.forward(x_t, timestep)
        return nn.functional.logsigmoid(logits).sum()

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
