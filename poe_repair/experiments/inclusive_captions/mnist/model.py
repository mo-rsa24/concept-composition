"""The MNIST stage's denoiser: one label-conditional U-Net on 1 x 32 x 64 canvases, and the toy's
noise schedule.

Three levels (32 x 64 with c channels, 16 x 32 with 2c, 8 x 16 with 4c), two residual blocks per
level, one self-attention block at the bottom. The label is looked up in a 56-row table (ten digits,
45 joint labels, blank) and added to the noise-level embedding, which enters every residual block.
The network predicts the noise.

The schedule is the toy's (3-understanding-compositionality/.../toy_sampler_hybrids.py): T = 200 steps,
cosine alpha-bar with s = 0.008, clipped to [1e-5, 0.9999]; training draws k uniformly from 1 to 200.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

T = 200


def make_abar(device="cpu"):
    s = 0.008
    tt = torch.linspace(0, 1, T + 1, dtype=torch.float64)
    abar = torch.cos((tt + s) / (1 + s) * math.pi / 2) ** 2 / math.cos(s / (1 + s) * math.pi / 2) ** 2
    return abar.clamp(1e-5, 0.9999).float().to(device)


def step_embedding(k, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=k.device) / half)
    a = k.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], 1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(e)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.n, self.qkv, self.out, self.h = nn.GroupNorm(8, c), nn.Conv2d(c, 3 * c, 1), nn.Conv2d(c, c, 1), heads

    def forward(self, x):
        b, c, hh, ww = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(b, 3, self.h, c // self.h, hh * ww).unbind(1)
        a = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.out(a.transpose(-1, -2).reshape(b, c, hh, ww))


class UNet(nn.Module):
    def __init__(self, c=64, n_labels=56):
        super().__init__()
        self.c = c
        td = 4 * c
        self.temb = nn.Sequential(nn.Linear(c, td), nn.SiLU(), nn.Linear(td, td))
        self.lemb = nn.Embedding(n_labels, td)
        ch = [c, 2 * c, 4 * c]
        self.inp = nn.Conv2d(1, c, 3, padding=1)
        self.down = nn.ModuleList()
        cin = c
        for i, co in enumerate(ch):
            self.down.append(nn.ModuleList([ResBlock(cin, co, td), ResBlock(co, co, td),
                                            nn.Conv2d(co, co, 3, stride=2, padding=1) if i < 2 else nn.Identity()]))
            cin = co
        self.mid1, self.mid_attn, self.mid2 = ResBlock(ch[2], ch[2], td), Attention(ch[2]), ResBlock(ch[2], ch[2], td)
        self.up = nn.ModuleList()
        for i, co in reversed(list(enumerate(ch))):
            self.up.append(nn.ModuleList([ResBlock(cin + co, co, td), ResBlock(co + co, co, td),
                                          nn.Upsample(scale_factor=2, mode="nearest") if i > 0 else nn.Identity()]))
            cin = co
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x, k, label):
        e = self.temb(step_embedding(k, self.c)) + self.lemb(label)
        h = self.inp(x)
        skips = []
        for r1, r2, dn in self.down:
            h = r1(h, e); skips.append(h)
            h = r2(h, e); skips.append(h)
            h = dn(h)
        h = self.mid2(self.mid_attn(self.mid1(h, e)), e)
        for r1, r2, up in self.up:
            h = r1(torch.cat([h, skips.pop()], 1), e)
            h = r2(torch.cat([h, skips.pop()], 1), e)
            h = up(h)
        return self.out(h)
