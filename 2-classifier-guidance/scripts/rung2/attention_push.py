#!/usr/bin/env python
"""The layout push: move the product's "cat" and "dog" attention apart, so the two animals are drawn in two places.

PoE draws one fused animal because both experts put their animal in the same place: measured on cat x dog
seeds 1 to 8, the top 10% of the "cat" map (cat expert) and of the "dog" map (dog expert) overlap with an
intersection over union of 0.53 to 0.81 through the drawing, against about 0.25 in joint drawings that
composed (attention_overlap.py). A classifier's push (v4c) mostly added cat and dog features to one body.
This push targets the layout instead.

Its score, for a half-drawn latent x_t at timestep t, is minus the correlation between two maps over the
32 x 32 grid:
    cat map  the cat expert ("a cat") read on x_t: cross-attention to the word "cat", averaged over heads
             and over every cross-attention layer working at 32 x 32, renormalised per position over the
             prompt's word tokens (start token dropped), as Attend-and-Excite and attention_overlap.py do
    dog map  the same for the dog expert ("a dog") and the word "dog"
A correlation of 1 means both words light up the same places. The push is the gradient of the score, so it
lowers the correlation. It plugs into the result sheet as a "head" (log_prob_both returns the score), so the
canary, the size-matched random push, the cap and the window all apply unchanged.

Known risk, read on the pictures: the correlation can also fall when one map fades out (one animal gone).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from poe_repair.methods._sampling import add_time_ids


class _KeepCrossAttention:
    """The plain attention computation, keeping the 32 x 32 layers' probabilities with their gradients."""

    def __init__(self, kept: list):
        self.kept = kept

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):
        batch, seq, _ = hidden_states.shape
        query = attn.head_to_batch_dim(attn.to_q(hidden_states))
        key = attn.head_to_batch_dim(attn.to_k(encoder_hidden_states))
        value = attn.head_to_batch_dim(attn.to_v(encoder_hidden_states))
        probs = attn.get_attention_scores(query, key, attention_mask)
        if seq == 32 * 32:
            self.kept.append(probs.view(batch, attn.heads, seq, -1).mean(1).float())
        out = attn.batch_to_head_dim(torch.bmm(probs, value))
        return attn.to_out[1](attn.to_out[0](out))


def token_index(tokenizer, prompt: str, word: str) -> int:
    ids = tokenizer(prompt).input_ids
    target = tokenizer(word, add_special_tokens=False).input_ids
    assert len(target) == 1, (word, target)
    return ids.index(target[0])


class AttentionSeparation(nn.Module):
    def __init__(self, unet, tokenizer, seq_a, pool_a, seq_b, pool_b, word_a="cat", word_b="dog",
                 prompt_a="a cat", prompt_b="a dog"):
        super().__init__()
        self.unet = unet
        self.seq = torch.cat([seq_a, seq_b])
        self.pool = torch.cat([pool_a, pool_b])
        self.i_a, self.i_b = token_index(tokenizer, prompt_a, word_a), token_index(tokenizer, prompt_b, word_b)
        self.n_a, self.n_b = len(tokenizer(prompt_a).input_ids) - 2, len(tokenizer(prompt_b).input_ids) - 2

    def maps(self, x_t, timestep):
        """(cat map, dog map), each (1024,), for one latent."""
        kept, originals = [], dict(self.unet.attn_processors)
        self.unet.set_attn_processor({n: (_KeepCrossAttention(kept) if ".attn2." in n else p) for n, p in originals.items()})
        try:
            x = x_t.to(self.seq.dtype).repeat(2, 1, 1, 1)
            t = timestep.reshape(-1)[:1].repeat(2)
            cond = {"text_embeds": self.pool,
                    "time_ids": add_time_ids(height=1024, width=1024, batch_size=2, device=x.device, dtype=x.dtype)}
            self.unet(x, t, encoder_hidden_states=self.seq, added_cond_kwargs=cond)
        finally:
            self.unet.set_attn_processor(originals)
        m = torch.stack(kept).mean(0)                                   # (2, 1024, 77)
        wa = m[0, :, 1:1 + self.n_a]
        wb = m[1, :, 1:1 + self.n_b]
        a = (wa / wa.sum(-1, keepdim=True))[:, self.i_a - 1]
        b = (wb / wb.sum(-1, keepdim=True))[:, self.i_b - 1]
        return a, b

    def correlation(self, x_t, timestep):
        a, b = self.maps(x_t, timestep)
        a, b = a - a.mean(), b - b.mean()
        return (a * b).sum() / (a.norm() * b.norm() + 1e-8)

    def forward(self, x_t, timestep):
        return (-self.correlation(x_t[:1], timestep)).reshape(1)

    def log_prob_both(self, x_t, timestep):
        return -self.correlation(x_t[:1], timestep)
