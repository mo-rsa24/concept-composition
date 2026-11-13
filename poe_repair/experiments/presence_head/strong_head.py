"""The strong presence head: per-species yes-or-no answers read off SDXL's own middle layer.

The weak head learned to see half-noisy pictures from 873 images and stayed near guessing at the
noisy steps that decide the layout (0.68 on cat and 0.56 on dog over sampler steps 0 to 19). SDXL
already sees half-noisy pictures at every noise level, so this head reads the frozen UNet's middle
block, run on the empty prompt so the features describe the picture rather than a request, and
trains only a small convolutional head on top.

The answers are inclusive and independent: one sigmoid per species, so a picture holding a cat and
a dog is yes to both, and a smooshed cat-dog, labelled neither, is no to both. Guidance
differentiates the sum of the log probabilities of the two target species with respect to the
noisy latent, through the head and the UNet's down and middle blocks. The up blocks never run.

``StrongPresenceHead.log_prob_both`` has the same signature as ``PresenceHead.log_prob_both``, so
``run_poe_classifier_guided`` drives either one unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from poe_repair.methods._sampling import add_time_ids


class _StopAfterMid(Exception):
    """Raised from the middle block's hook so the UNet never spends time on its up blocks."""


class UNetMidFeatures(nn.Module):
    """SDXL's frozen middle-block output for a noisy latent at a timestep, on the empty prompt."""

    def __init__(self, unet, seq_e: torch.Tensor, pool_e: torch.Tensor, *,
                 height: int = 1024, width: int = 1024) -> None:
        super().__init__()
        self.unet = unet
        self.register_buffer("seq_e", seq_e, persistent=False)
        self.register_buffer("pool_e", pool_e, persistent=False)
        self.height, self.width = height, width
        self._feat = None
        # The UNet is shared with the sampler, so the hook acts only while this module is reading
        # features. Left armed, it stopped the sampler's own forward pass halfway as well.
        self._active = False
        self._hook = unet.mid_block.register_forward_hook(self._grab)

    def _grab(self, module, inputs, output):
        if not self._active:
            return None
        self._feat = output
        raise _StopAfterMid

    def forward(self, x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        b = x_t.shape[0]
        dtype = self.unet.dtype
        cond = {"text_embeds": self.pool_e.expand(b, -1).to(dtype),
                "time_ids": add_time_ids(height=self.height, width=self.width, batch_size=b,
                                         device=x_t.device, dtype=dtype)}
        self._active = True
        try:
            self.unet(x_t.to(dtype), timestep, encoder_hidden_states=self.seq_e.expand(b, -1, -1).to(dtype),
                      added_cond_kwargs=cond, timestep_cond=None)
        except _StopAfterMid:
            pass
        finally:
            self._active = False
        feat, self._feat = self._feat, None
        return feat


class SpeciesHead(nn.Module):
    """Middle-block features (B, 1280, 32, 32) to one logit per species."""

    def __init__(self, n_species: int, in_channels: int = 1280, width: int = 256,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 1), nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
        )
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, n_species))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        h = self.net(feat.float())
        return self.out(h.mean(dim=(2, 3)))


class StrongPresenceHead(nn.Module):
    def __init__(self, features: UNetMidFeatures, head: SpeciesHead, species: list[str],
                 target_species: tuple[str, str] = ("cat", "dog")) -> None:
        super().__init__()
        self.features, self.head = features, head
        self.species = list(species)
        self.target_idx = [self.species.index(s) for s in target_species]

    def forward(self, x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x_t, timestep))

    def log_prob_both(self, x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """log p(first target present) + log p(second target present), summed over the batch."""
        logits = self.forward(x_t, timestep)
        return F.logsigmoid(logits[:, self.target_idx]).sum()
