#!/usr/bin/env python3
"""Paper 1's core idea on a toy that answers in minutes: red and blue squares on 16 by 16 pictures.

The question. Composing separately trained experts by the plain sum D_red + D_blue - D_empty draws a red and a blue object that
overlap and paints the overlap purple. Does a learned interaction term added to that sum (Method A), or a classifier push on top of it
(Methods C and B), make the two come out as a clean touching pair, one in front of the other where they overlap?

The pictures. 16 by 16 RGB on the -1 to 1 scale, background black (-1, -1, -1). A red square is (1, -1, -1), a blue square (-1, -1, 1),
side SIDE (4) pixels, at every one of the 13 by 13 top-left positions. Four sets, enumerated exactly:
  empty   the one black picture
  red     one red square, every position (169)
  blue    one blue square, every position (169)
  joint   one red and one blue square touching: with dr, dc the row and column offsets of the two top-left corners, either the squares
          share an edge with no gap (max(|dr|, |dc|) = SIDE and min(|dr|, |dc|) < SIDE; one picture each), or they overlap by one or
          two pixels (max(|dr|, |dc|) is SIDE - 1 or SIDE - 2, so the overlap's thinner side is 1 or 2 pixels), drawn twice, once with
          red in front (its pixels win the overlap) and once with blue in front.
Because the background is black, a red-set picture differs from the background only in the R channel and a blue-set picture only in
B, so the exact plain sum at an overlap pixel is R = 1, G = -1, B = 1: purple.

Exact denoisers. For a finite set {x0_i} under Gaussian noise of level sigma, the ideal denoiser is
    D(x, sigma) = sum_i softmax_i(-||x - x0_i||^2 / (2 sigma^2)) x0_i,
computed here in float64. These are the exact experts and the exact joint model.

Conventions copied from the CLEVR scripts so a toy result transfers (paper1_sample.py, paper1_interaction.py, paper1_fourway.py):
  sampler   EDM2's deterministic Heun, 32 steps, sigma 80 to 0.002, rho 7, one starting noise per seed shared by every row
  networks  EDM preconditioning with sigma_data 0.5: D = c_skip x + c_out F(c_in x, c_noise), c_noise = log(sigma) / 4 in 32 sine
            and cosine features; loss weight (sigma^2 + 0.25) / (0.5 sigma)^2 with EDM2's learned uncertainty; sigma = exp(-0.8 +
            1.6 n) during training (edm2-img64-xs)
  Method A  D_new = D_red + D_blue - D_empty + delta, delta = c_out(sigma) F(c_in x, D_red, D_blue, D_empty; sigma), F's last
            convolution zero at the start; every tenth joint scene held out; the update with the lowest validation loss is kept
  Method C  a real-versus-drawn head, logit = 4 tanh(raw / 4); drawn = the plain-sum sampler's own states after each Heun step at
            that step's sigma; push D' = D + w sigma^2 grad logit at every evaluation
  Method B  four logits (empty, red only, blue only, both), each 4 tanh(raw / 4), trained on real pictures with sigma log-uniform on
            [0.002, 80]; push D' = D + w sigma^2 grad [l11 - logaddexp(l10, l11) - logaddexp(l01, l11) + logsumexp(l)] at every
            evaluation, which is grad [log q11 - log(q10 + q11) - log(q01 + q11)]
What differs from CLEVR: no count label (one object per colour), the experts and heads are small U-Nets trained here from scratch,
the heads read the noisy picture directly instead of the experts' features, and the red and blue experts read disjoint channels
(R and B), so shared starting noise does not pull their squares together the way it pulls CLEVR objects together.

Metrics, from pixels, on the final pictures clamped to [-1, 1]:
  clean     the nearest joint-set picture lies within CLEAN_L2 (1.0) in L2 over all 768 values. One pixel wrong by a full colour in
            one channel is distance 2, so a clean sample matches a touching pair on every pixel to within a fraction of a colour
  purple    a pixel with R > 0.5 and B > 0.5
  red, blue a red pixel has R > 0.5 and B <= 0.5; a blue pixel B > 0.5 and R <= 0.5
  contact   some blue pixel lies in the 3 by 3 neighbourhood of a red pixel
  one-each  the red pixels form exactly one 4-connected component and the blue pixels exactly one

Subcommands, one per rung, run in order (each writes its section of results.json and its picture grid):
  sets      print each set's size
  rung0     exact joint and exact plain sum
  experts   train the three neural experts and report their error against the exact denoisers per sigma band
  rung1     the neural plain sum
  rung2     Method A at N = 16, 64 and all, plus the zero-delta control
  rung2b    Method A at N = 256, 1024 and 4096 (the data curve), and the plot of clean-pair share against N
  rung3     Method C at w = 1, 4, 16, 64
  rung4     Method B at w = 1, 4, 16, 64

  /home/molef/micromamba/envs/voice-agent/bin/python 2-classifier-guidance/scripts/paper1/toy/squares_toy.py rung0
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "outputs" / "paper1-toy"                       # checkpoints and sample tensors (git-ignored)
RESULTS = REPO / "artifacts" / "results" / "does-the-interaction-term-work-on-a-small-toy"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

H = 16
SIDE = 4
BG = (-1.0, -1.0, -1.0)
RED = (1.0, -1.0, -1.0)
BLUE = (-1.0, -1.0, 1.0)

STEPS, SIGMA_MIN, SIGMA_MAX, RHO = 32, 0.002, 80.0, 7.0      # paper1_sample.py
P_MEAN, P_STD, SIGMA_DATA = -0.8, 1.6, 0.5                   # paper1_interaction.py
LOGIT_BOUND = 4.0                                             # paper1_disc.py
SIGMA_LO, SIGMA_HI = 0.002, 80.0                              # paper1_fourway.py

N_SEEDS = 256
EVAL_SEEDS = list(range(N_SEEDS))
DRAWN_SEEDS = list(range(100000, 101024))                     # Method C's negatives, never an evaluation seed
GRID_SEEDS = 8
CLEAN_L2 = 1.0
PURPLE_T = 0.5
SIGMA_BANDS = ((80.0, 10.0), (10.0, 2.0), (2.0, 0.5), (0.5, 0.1), (0.1, 0.002))
BAND_DRAWS = 2048
VAL_NTH = 10
WEIGHTS = (1, 4, 16, 64)

BARS = ("On 256 shared seeds, a method WORKS on the toy if its clean-pair share is at least 0.8 and at least 0.5 above the neural "
        "plain sum's; it is PARTIAL if it is at least 0.2 above the plain sum but misses WORKS; it FAILS otherwise.")


# -------------------------------------------------------------------------------------------------------------------------------------
# The sets


def _blank(n=1):
    return torch.tensor(BG).view(1, 3, 1, 1).repeat(n, 1, H, H)


def _paint(img, r, c, colour):
    img[:, r:r + SIDE, c:c + SIDE] = torch.tensor(colour).view(3, 1, 1)


def positions():
    return [(r, c) for r in range(H - SIDE + 1) for c in range(H - SIDE + 1)]


def empty_set():
    return _blank(1)


def single_set(colour):
    pos = positions()
    pics = _blank(len(pos))
    for i, (r, c) in enumerate(pos):
        _paint(pics[i], r, c, colour)
    return pics


def joint_set():
    """(pictures, metadata). Metadata rows: red r, red c, blue r, blue c, kind (0 edge, 1 overlap 1, 2 overlap 2), front (0 none,
    1 red in front, 2 blue in front)."""
    pics, meta = [], []
    for rr, rc in positions():
        for br, bc in positions():
            dr, dc = abs(rr - br), abs(rc - bc)
            hi, lo = max(dr, dc), min(dr, dc)
            if hi == SIDE and lo < SIDE:
                img = _blank(1)[0]
                _paint(img, rr, rc, RED)
                _paint(img, br, bc, BLUE)
                pics.append(img)
                meta.append((rr, rc, br, bc, 0, 0))
            elif hi in (SIDE - 1, SIDE - 2):
                for front in (1, 2):
                    img = _blank(1)[0]
                    if front == 1:
                        _paint(img, br, bc, BLUE)
                        _paint(img, rr, rc, RED)
                    else:
                        _paint(img, rr, rc, RED)
                        _paint(img, br, bc, BLUE)
                    pics.append(img)
                    meta.append((rr, rc, br, bc, SIDE - hi, front))
    pics = torch.stack(pics)
    uniq = torch.unique(pics.flatten(1), dim=0)
    assert len(uniq) == len(pics), "two touching configurations drew the same picture"
    return pics, np.array(meta)


def joint_split():
    """(train pool indices, validation indices) into joint_set(): a fixed permutation (seed 0) of the enumeration, every tenth
    scene in that order held out. The first N of the train pool are Method A's N scenes."""
    n = len(joint_set()[0])
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).tolist()
    val = [perm[i] for i in range(0, n, VAL_NTH)]
    train = [perm[i] for i in range(n) if i % VAL_NTH != 0]
    return train, val


# -------------------------------------------------------------------------------------------------------------------------------------
# Exact denoisers


class Exact:
    """D(x, sigma) = sum_i softmax_i(-||x - x0_i||^2 / (2 sigma^2)) x0_i over a finite set, in float64."""

    def __init__(self, pics, device=DEV):
        self.X = pics.flatten(1).double().to(device)
        self.sq = (self.X ** 2).sum(1)

    def __call__(self, x, sigma, chunk=512):
        out = []
        s = torch.as_tensor(sigma, dtype=torch.float64, device=self.X.device).reshape(-1)
        for i in range(0, len(x), chunk):
            xf = x[i:i + chunk].flatten(1).double().to(self.X.device)
            si = s if len(s) == 1 else s[i:i + chunk]
            d2 = ((xf ** 2).sum(1, keepdim=True) - 2 * xf @ self.X.T + self.sq[None]).clamp_min(0)
            w = torch.softmax(-d2 / (2 * si.view(-1, 1) ** 2), 1)
            out.append((w @ self.X).float())
        return torch.cat(out).view(x.shape).to(x.device)


# -------------------------------------------------------------------------------------------------------------------------------------
# Sampler (paper1_sample.sigmas and heun)


def sigmas(device=DEV):
    i = torch.arange(STEPS, dtype=torch.float64)
    t = (SIGMA_MAX ** (1 / RHO) + i / (STEPS - 1) * (SIGMA_MIN ** (1 / RHO) - SIGMA_MAX ** (1 / RHO))) ** RHO
    return torch.cat([t, torch.zeros(1, dtype=torch.float64)]).float().to(device)


def heun(denoise, x, after_step=None):
    t = sigmas(x.device)
    for i in range(STEPS):
        t_cur, t_next = t[i], t[i + 1]
        d_cur = (x - denoise(x, t_cur, i, False)) / t_cur
        x_next = x + (t_next - t_cur) * d_cur
        if i < STEPS - 1:
            d_eval = denoise(x_next, t_next, i, True)
            x_next = x + (t_next - t_cur) * 0.5 * (d_cur + (x_next - d_eval) / t_next)
        x = x_next
        if after_step is not None:
            after_step(i, x, t_next)
    return x


def starting_noise(seeds, device=DEV):
    """x at sigma 80, one generator per seed, so every row starts from the same noise at a seed."""
    eps = torch.stack([torch.randn(3, H, H, generator=torch.Generator().manual_seed(int(s))) for s in seeds])
    return (eps * SIGMA_MAX).to(device)


def plainsum_of(d_r, d_b, d_e):
    """The plain sum's floating-point operations, shared by every row that contains it."""
    return d_r + d_b - d_e


def plainsum_denoiser(red, blue, empty):
    def denoise(x, s, step, second):
        return plainsum_of(red(x, s), blue(x, s), empty(x, s))
    return denoise


def joint_denoiser(exact_joint):
    def denoise(x, s, step, second):
        return exact_joint(x, s)
    return denoise


def sample(denoise, seeds=EVAL_SEEDS, batch=256):
    outs = []
    for i in range(0, len(seeds), batch):
        outs.append(heun(denoise, starting_noise(seeds[i:i + batch])).detach())
    return torch.cat(outs)


# -------------------------------------------------------------------------------------------------------------------------------------
# Metrics


def masks(x):
    x = x.clamp(-1, 1)
    r, b = x[:, 0], x[:, 2]
    purple = (r > PURPLE_T) & (b > PURPLE_T)
    red = (r > PURPLE_T) & (b <= PURPLE_T)
    blue = (b > PURPLE_T) & (r <= PURPLE_T)
    return red, blue, purple


def components(mask):
    from scipy import ndimage
    return np.array([ndimage.label(m)[1] for m in mask.cpu().numpy()])


def nearest(x, pics):
    """For each picture, the L2 distance to the nearest member of pics and that member's index, in float64."""
    X = pics.flatten(1).double().to(DEV)
    sq = (X ** 2).sum(1)
    dists, idx = [], []
    for i in range(0, len(x), 256):
        xf = x[i:i + 256].clamp(-1, 1).flatten(1).double().to(DEV)
        d2 = ((xf ** 2).sum(1, keepdim=True) - 2 * xf @ X.T + sq[None]).clamp_min(0)
        m, j = d2.min(1)
        dists.append(m.sqrt().cpu())
        idx.append(j.cpu())
    return torch.cat(dists), torch.cat(idx)


def metrics(x, jpics, train_idx=None):
    d, idx = nearest(x, jpics)
    red, blue, purple = masks(x)
    near_red = F.max_pool2d(red.float()[:, None], 3, 1, 1)[:, 0] > 0
    contact = (near_red & blue).flatten(1).any(1)
    n_red, n_blue = components(red), components(blue)
    clean = d <= CLEAN_L2
    pc = purple.flatten(1).sum(1).float()
    out = {
        "n": len(x),
        "clean_share": float(clean.float().mean()),
        "distinct_clean_scenes": int(len(set(idx[clean].tolist()))),
        "nearest_l2_quantiles": {q: round(float(d.quantile(float(q))), 3) for q in ("0.1", "0.25", "0.5", "0.75", "0.9")},
        "nearest_l2_bins": {lab: float(((d > lo) & (d <= hi)).float().mean()) for lab, lo, hi in
                            (("<=0.25", -1, 0.25), ("0.25-1", 0.25, 1.0), ("1-2", 1.0, 2.0), ("2-4", 2.0, 4.0),
                             ("4-8", 4.0, 8.0), (">8", 8.0, 1e9))},
        "purple_pixels_mean": float(pc.mean()),
        "share_with_purple": float((pc > 0).float().mean()),
        "contact_share": float(contact.float().mean()),
        "one_each_share": float(((n_red == 1) & (n_blue == 1)).mean()),
    }
    if train_idx is not None and clean.any():
        tr = set(train_idx)
        out["clean_that_are_training_scenes"] = float(np.mean([int(i) in tr for i in idx[clean].tolist()]))
    return out


def verdict(clean, base):
    if clean >= 0.8 and clean - base >= 0.5:
        return "WORKS"
    if clean - base >= 0.2:
        return "PARTIAL"
    return "FAILS"


# -------------------------------------------------------------------------------------------------------------------------------------
# Networks (paper1_interaction.py's block, smaller)

FOURIER = 32
EMB = 128
GROUPS = 8


def c_in(s):
    return 1 / (SIGMA_DATA ** 2 + s ** 2).sqrt()


def c_out(s):
    return s * SIGMA_DATA / (s ** 2 + SIGMA_DATA ** 2).sqrt()


def c_skip(s):
    return SIGMA_DATA ** 2 / (s ** 2 + SIGMA_DATA ** 2)


def loss_weight(s):
    return (s ** 2 + SIGMA_DATA ** 2) / (s * SIGMA_DATA) ** 2


def draw_sigma(n, gen=None, device=DEV):
    return (torch.randn(n, generator=gen, device=device) * P_STD + P_MEAN).exp()


def fourier(sigma):
    c_noise = sigma.float().log()[:, None] / 4
    freqs = math.pi * 2.0 ** torch.linspace(0, 7, FOURIER // 2, device=sigma.device)
    return torch.cat([(c_noise * freqs).sin(), (c_noise * freqs).cos()], 1)


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.norm1, self.film1, self.conv1 = nn.GroupNorm(GROUPS, cin), nn.Linear(EMB, 2 * cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2, self.film2, self.conv2 = nn.GroupNorm(GROUPS, cout), nn.Linear(EMB, 2 * cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, h, e):
        s, b = self.film1(e)[:, :, None, None].chunk(2, 1)
        y = self.conv1(F.silu(self.norm1(h) * (1 + s) + b))
        s, b = self.film2(e)[:, :, None, None].chunk(2, 1)
        y = self.conv2(F.silu(self.norm2(y) * (1 + s) + b))
        return self.skip(h) + y


class UNet(nn.Module):
    """16 -> 8 -> 4 -> 8 -> 16, every block conditioned on sigma; at 4 by 4 a 3 by 3 convolution sees the whole picture."""

    def __init__(self, cin, cout, w=64, zero_out=False):
        super().__init__()
        self.cond = nn.Sequential(nn.Linear(FOURIER, EMB), nn.SiLU(), nn.Linear(EMB, EMB), nn.SiLU())
        self.conv_in = nn.Conv2d(cin, w, 3, padding=1)
        self.e1, self.e2 = Block(w, w), Block(w, 2 * w)
        self.m1, self.m2 = Block(2 * w, 2 * w), Block(2 * w, 2 * w)
        self.d2, self.d1 = Block(4 * w, 2 * w), Block(3 * w, w)
        self.conv_out = nn.Conv2d(w, cout, 3, padding=1)
        if zero_out:
            nn.init.zeros_(self.conv_out.weight)
            nn.init.zeros_(self.conv_out.bias)
        self.logvar_linear = nn.Linear(FOURIER, 1)
        nn.init.zeros_(self.logvar_linear.weight)
        nn.init.zeros_(self.logvar_linear.bias)

    def forward(self, h, sigma):
        e = self.cond(fourier(sigma))
        h1 = self.e1(self.conv_in(h), e)
        h2 = self.e2(F.avg_pool2d(h1, 2), e)
        h3 = self.m2(self.m1(F.avg_pool2d(h2, 2), e), e)
        h = self.d2(torch.cat([F.interpolate(h3, scale_factor=2, mode="nearest"), h2], 1), e)
        h = self.d1(torch.cat([F.interpolate(h, scale_factor=2, mode="nearest"), h1], 1), e)
        return self.conv_out(h)

    def logvar(self, sigma):
        return self.logvar_linear(fourier(sigma)).view(-1, 1, 1, 1)


class Expert(nn.Module):
    """EDM-preconditioned denoiser D = c_skip x + c_out F(c_in x, c_noise)."""

    def __init__(self, w=64):
        super().__init__()
        self.net = UNet(3, 3, w)

    def forward(self, x, sigma):
        sigma = torch.as_tensor(sigma, device=x.device, dtype=torch.float32).reshape(-1).expand(len(x))
        s4 = sigma.view(-1, 1, 1, 1)
        return c_skip(s4) * x + c_out(s4) * self.net(c_in(s4) * x, sigma)


class Delta(nn.Module):
    """Method A's interaction term: c_out(sigma) F(c_in x, D_red, D_blue, D_empty; sigma), F's last convolution zero at the start."""

    def __init__(self, w=64):
        super().__init__()
        self.net = UNet(12, 3, w, zero_out=True)

    def forward(self, x, sigma, d_r, d_b, d_e):
        sigma = torch.as_tensor(sigma, device=x.device, dtype=torch.float32).reshape(-1).expand(len(x))
        s4 = sigma.view(-1, 1, 1, 1)
        return c_out(s4) * self.net(torch.cat([c_in(s4) * x, d_r, d_b, d_e], 1), sigma)


class Head(nn.Module):
    """Noise-conditioned classifier on c_in x: 16 -> 8 -> 4, grid average, linear to nout raw scores, each 4 tanh(raw / 4)."""

    def __init__(self, nout, w=48):
        super().__init__()
        self.cond = nn.Sequential(nn.Linear(FOURIER, EMB), nn.SiLU(), nn.Linear(EMB, EMB), nn.SiLU())
        self.conv_in = nn.Conv2d(3, w, 3, padding=1)
        self.b1, self.b2, self.b3 = Block(w, w), Block(w, 2 * w), Block(2 * w, 2 * w)
        self.lin = nn.Linear(2 * w, nout)

    def forward(self, x, sigma):
        sigma = torch.as_tensor(sigma, device=x.device, dtype=torch.float32).reshape(-1).expand(len(x))
        e = self.cond(fourier(sigma))
        h = self.b1(self.conv_in(c_in(sigma.view(-1, 1, 1, 1)) * x), e)
        h = self.b2(F.avg_pool2d(h, 2), e)
        h = self.b3(F.avg_pool2d(h, 2), e)
        raw = self.lin(h.mean((2, 3)))
        return LOGIT_BOUND * torch.tanh(raw / LOGIT_BOUND)


def fourway_ratio(logits):
    """log q11 - log(q10 + q11) - log(q01 + q11) from the four logits [l00, l10, l01, l11], without forming q."""
    l00, l10, l01, l11 = logits.unbind(1)
    return l11 - torch.logaddexp(l10, l11) - torch.logaddexp(l01, l11) + torch.logsumexp(logits, 1)


# -------------------------------------------------------------------------------------------------------------------------------------
# Composed denoisers with a learned term or a push


def interaction_denoiser(red, blue, empty, delta, zero_delta=False):
    scale = 0.0 if zero_delta else 1.0

    def denoise(x, s, step, second):
        d_r, d_b, d_e = red(x, s), blue(x, s), empty(x, s)
        return plainsum_of(d_r, d_b, d_e) + scale * delta(x, s, d_r, d_b, d_e)
    return denoise


def push_denoiser(base, score, w):
    """D' = D + w sigma^2 grad_x score(x, sigma) at every evaluation; score returns one number per picture."""
    def denoise(x, s, step, second):
        d = base(x, s, step, second)
        with torch.enable_grad():
            xx = x.detach().requires_grad_(True)
            g, = torch.autograd.grad(score(xx, s).sum(), xx)
        return d + w * s ** 2 * g
    return denoise


# -------------------------------------------------------------------------------------------------------------------------------------
# Bookkeeping


def load_results():
    p = RESULTS / "results.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_results(key, value):
    RESULTS.mkdir(parents=True, exist_ok=True)
    res = load_results()
    res[key] = value
    res["bars"] = BARS
    res["device"] = str(DEV) + (f" ({torch.cuda.get_device_name()})" if DEV.type == "cuda" else "")
    (RESULTS / "results.json").write_text(json.dumps(res, indent=1))


def grid(rows, labels, path, scale=8, gap=2):
    """One row per method, GRID_SEEDS seeds as columns, nearest-neighbour upscaled, a label column on the left."""
    from PIL import Image, ImageDraw
    lw = 150
    tile = H * scale
    img = Image.new("RGB", (lw + GRID_SEEDS * (tile + gap), len(rows) * (tile + gap) + 14), "white")
    dr = ImageDraw.Draw(img)
    for c in range(GRID_SEEDS):
        dr.text((lw + c * (tile + gap) + 2, 1), f"seed {EVAL_SEEDS[c]}", fill="black")
    for r, (x, lab) in enumerate(zip(rows, labels)):
        y = 14 + r * (tile + gap)
        dr.text((4, y + tile // 2 - 6), lab, fill="black")
        arr = ((x[:GRID_SEEDS].clamp(-1, 1).cpu().numpy().transpose(0, 2, 3, 1) + 1) * 127.5).round().astype(np.uint8)
        for c in range(GRID_SEEDS):
            img.paste(Image.fromarray(arr[c]).resize((tile, tile), Image.NEAREST), (lw + c * (tile + gap), y))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def table(rows):
    head = "| Row | Clean pair share | Distinct clean scenes | Median nearest L2 | Seeds with purple | Purple pixels (mean) | Contact | One each |"
    lines = [head, "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in rows:
        lines.append(f"| {name} | {m['clean_share']:.3f} | {m['distinct_clean_scenes']} | {m['nearest_l2_quantiles']['0.5']:.2f} | "
                     f"{m['share_with_purple']:.3f} | {m['purple_pixels_mean']:.2f} | {m['contact_share']:.3f} | {m['one_each_share']:.3f} |")
    return "\n".join(lines)


def sets_and_exact():
    e, r, b = empty_set(), single_set(RED), single_set(BLUE)
    j, meta = joint_set()
    return (e, r, b, j, meta), (Exact(e), Exact(r), Exact(b), Exact(j))


def save_samples(name, x):
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(x.cpu(), OUT / f"samples-{name}.pt")


def load_samples(name):
    return torch.load(OUT / f"samples-{name}.pt")


# -------------------------------------------------------------------------------------------------------------------------------------
# Training


def edm_loss(D, x0, sigma, logvar):
    w = loss_weight(sigma.view(-1, 1, 1, 1))
    return (w / logvar.exp() * (D - x0) ** 2 + logvar).mean()


def train_expert(pics, updates, batch=256, lr=5e-4, seed=0, log=None):
    torch.manual_seed(seed)
    model = Expert().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    X = pics.to(DEV)
    curve = []
    for u in range(updates):
        x0 = X[torch.randint(len(X), (batch,), device=DEV)]
        s = draw_sigma(batch)
        D = model(x0 + s.view(-1, 1, 1, 1) * torch.randn_like(x0), s)
        loss = edm_loss(D, x0, s, model.net.logvar(s))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if u % 500 == 0 or u == updates - 1:
            curve.append((u, float(loss.detach())))
            if log:
                print(f"  {log} update {u} loss {float(loss):.4f}", flush=True)
    return model.eval().requires_grad_(False), curve


def band_draws(pics, band, n=BAND_DRAWS, seed=7):
    """n noisy pictures from members of pics, sigma log-uniform inside the band, fixed by seed."""
    g = torch.Generator().manual_seed(seed)
    hi, lo = band
    s = torch.exp(torch.empty(n).uniform_(math.log(lo), math.log(hi), generator=g))
    x0 = pics[torch.randint(len(pics), (n,), generator=g)]
    x = x0 + s.view(-1, 1, 1, 1) * torch.randn(x0.shape, generator=g)
    return x.to(DEV), s.to(DEV)


@torch.no_grad()
def band_error(fn_a, fn_b, pics):
    """Per sigma band: mean squared difference per value between two denoisers, and the mean square of fn_b (its size)."""
    out = {}
    for band in SIGMA_BANDS:
        x, s = band_draws(pics, band)
        a, b = [], []
        for i in range(0, len(x), 256):
            a.append(fn_a(x[i:i + 256], s[i:i + 256]))
            b.append(fn_b(x[i:i + 256], s[i:i + 256]))
        a, b = torch.cat(a), torch.cat(b)
        out[f"{band[0]}-{band[1]}"] = {"mse": float(((a - b) ** 2).mean()), "size_of_reference": float((b ** 2).mean())}
    return out


def load_experts():
    nets = []
    for name in ("red", "blue", "empty"):
        m = Expert().to(DEV)
        m.load_state_dict(torch.load(OUT / f"expert-{name}.pt", map_location=DEV))
        nets.append(m.eval().requires_grad_(False))
    return nets


# -------------------------------------------------------------------------------------------------------------------------------------
# Subcommands


def cmd_sets(a):
    (e, r, b, j, meta), _ = sets_and_exact()
    kinds = {k: int((meta[:, 4] == v).sum()) for k, v in (("edge", 0), ("overlap 1", 1), ("overlap 2", 2))}
    tr, va = joint_split()
    info = {"empty": len(e), "red": len(r), "blue": len(b), "joint": len(j), "joint_by_kind": kinds,
            "joint_train_pool": len(tr), "joint_validation": len(va), "side": SIDE, "size": H}
    print(json.dumps(info, indent=1))
    save_results("sets", info)


def cmd_rung0(a):
    t0 = time.time()
    (e, r, b, j, meta), (Ee, Er, Eb, Ej) = sets_and_exact()
    xj = sample(joint_denoiser(Ej))
    xp = sample(plainsum_denoiser(Er, Eb, Ee))
    save_samples("exact-joint", xj)
    save_samples("exact-plainsum", xp)
    rows = [("exact joint", metrics(xj, j)), ("exact plain sum", metrics(xp, j))]
    print(table(rows))
    grid([xj, xp], ["exact joint", "exact plain sum"], RESULTS / "rung0-exact.png")
    save_results("rung0", {"rows": dict(rows), "seconds": round(time.time() - t0, 1)})


def cmd_experts(a):
    t0 = time.time()
    (e, r, b, j, meta), (Ee, Er, Eb, Ej) = sets_and_exact()
    info = {}
    for name, pics, ex, ups in (("red", r, Er, a.updates), ("blue", b, Eb, a.updates), ("empty", e, Ee, a.updates // 4)):
        t1 = time.time()
        model, curve = train_expert(pics, ups, seed={"red": 1, "blue": 2, "empty": 3}[name], log=name)
        OUT.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), OUT / f"expert-{name}.pt")
        err = band_error(model, ex, pics)
        info[name] = {"updates": ups, "seconds": round(time.time() - t1, 1), "loss_curve": curve, "error_vs_exact_by_band": err}
        print(name, json.dumps(err))
    info["seconds"] = round(time.time() - t0, 1)
    save_results("experts", info)


def cmd_rung1(a):
    t0 = time.time()
    (e, r, b, j, meta), (Ee, Er, Eb, Ej) = sets_and_exact()
    red, blue, empty = load_experts()
    with torch.no_grad():
        xn = sample(plainsum_denoiser(red, blue, empty))
    save_samples("neural-plainsum", xn)
    xj, xp = load_samples("exact-joint"), load_samples("exact-plainsum")
    # the neural plain sum's denoiser against the exact plain sum's, on joint scenes noised (where the sampler must go)
    err = band_error(lambda x, s: plainsum_of(red(x, s), blue(x, s), empty(x, s)),
                     lambda x, s: plainsum_of(Er(x, s), Eb(x, s), Ee(x, s)), j)
    rows = [("exact joint (ceiling)", metrics(xj, j)), ("exact plain sum", metrics(xp, j)), ("neural plain sum", metrics(xn, j))]
    print(table(rows))
    grid([xj, xp, xn], ["exact joint", "exact plain sum", "neural plain sum"], RESULTS / "rung1-neural-plain-sum.png")
    save_results("rung1", {"rows": dict(rows), "plainsum_error_vs_exact_plainsum_by_band": err,
                           "seconds": round(time.time() - t0, 1)})


def train_delta(nets, jpics, train_idx, val_idx, updates, batch=256, lr=5e-4, seed=0, val_every=250):
    red, blue, empty = nets
    torch.manual_seed(seed)
    model = Delta().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    X = jpics[train_idx].to(DEV)
    V = jpics[val_idx]
    g = torch.Generator().manual_seed(12345)
    v0 = V.repeat(2, 1, 1, 1)
    vs = (torch.randn(len(v0), generator=g) * P_STD + P_MEAN).exp()
    vn = torch.randn(v0.shape, generator=g)

    @torch.no_grad()
    def val_loss():
        tot = 0.0
        for i in range(0, len(v0), 512):
            x0, s, n = v0[i:i + 512].to(DEV), vs[i:i + 512].to(DEV), vn[i:i + 512].to(DEV)
            x = x0 + s.view(-1, 1, 1, 1) * n
            d_r, d_b, d_e = red(x, s), blue(x, s), empty(x, s)
            D = plainsum_of(d_r, d_b, d_e) + model(x, s, d_r, d_b, d_e)
            tot += float((loss_weight(s.view(-1, 1, 1, 1)) * (D - x0) ** 2).mean((1, 2, 3)).sum())
        return tot / len(v0)

    best = (val_loss(), 0, {k: v.clone() for k, v in model.state_dict().items()})
    curve = [(0, None, best[0])]
    for u in range(1, updates + 1):
        x0 = X[torch.randint(len(X), (batch,), device=DEV)]
        s = draw_sigma(batch)
        x = x0 + s.view(-1, 1, 1, 1) * torch.randn_like(x0)
        with torch.no_grad():
            d_r, d_b, d_e = red(x, s), blue(x, s), empty(x, s)
        D = plainsum_of(d_r, d_b, d_e) + model(x, s, d_r, d_b, d_e)
        loss = edm_loss(D, x0, s, model.net.logvar(s))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if u % val_every == 0 or u == updates:
            vl = val_loss()
            curve.append((u, float(loss.detach()), vl))
            if u % 1000 == 0:
                print(f"  delta N={len(train_idx)} update {u} loss {float(loss.detach()):.4f} val {vl:.4f}", flush=True)
            if vl < best[0]:
                best = (vl, u, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    return model.eval().requires_grad_(False), {"best_update": best[1], "best_val_loss": best[0], "curve": curve}


def run_method_a(ns, updates, key, picture, zero_control, tag=""):
    """Method A at each N in ns (0 means the whole training pool); results under results.json[key], a grid at picture."""
    t0 = time.time()
    (e, r, b, j, meta), (Ee, Er, Eb, Ej) = sets_and_exact()
    nets = load_experts()
    red, blue, empty = nets
    tr, va = joint_split()
    xn = load_samples("neural-plainsum")
    base = metrics(xn, j)
    rows, xs, labels = [("neural plain sum", base)], [load_samples("exact-joint"), xn], ["exact joint", "neural plain sum"]
    info = {"updates": updates}
    exact_resid = lambda x, s: Ej(x, s) - plainsum_of(red(x, s), blue(x, s), empty(x, s))  # noqa: E731
    for n in ns:
        t1 = time.time()
        idx = tr if n == 0 else tr[:n]
        name = ("all" if n == 0 else str(n)) + tag
        model, tinfo = train_delta(nets, j, idx, va, updates)
        torch.save(model.state_dict(), OUT / f"delta-N{name}.pt")
        with torch.no_grad():
            x = sample(interaction_denoiser(red, blue, empty, model))
        save_samples(f"methodA-N{name}", x)
        m = metrics(x, j, train_idx=idx)
        m["verdict"] = verdict(m["clean_share"], base["clean_share"])
        err = band_error(lambda xx, s: model(xx, s, red(xx, s), blue(xx, s), empty(xx, s)), exact_resid, j[va])
        info[f"N={name}"] = {"n_train_scenes": len(idx), "train": tinfo, "residual_error_vs_exact_by_band": err,
                             "seconds": round(time.time() - t1, 1)}
        rows.append((f"Method A, N = {name} ({len(idx)} scenes)" + (f", {updates} updates" if tag else ""), m))
        xs.append(x)
        labels.append(f"A, N={name}")
        print(rows[-1][0], m["clean_share"], m["verdict"], json.dumps(err), flush=True)
        if zero_control and n == ns[-1]:
            with torch.no_grad():
                xz = sample(interaction_denoiser(red, blue, empty, model, zero_delta=True))
            mz = metrics(xz, j)
            mz["verdict"] = verdict(mz["clean_share"], base["clean_share"])
            mz["byte_identical_to_neural_plain_sum"] = bool(torch.equal(xz.cpu(), xn.cpu()))
            rows.append(("zero-delta control", mz))
            xs.append(xz)
            labels.append("zero delta")
    print(table(rows))
    grid(xs, labels, RESULTS / picture)
    info["rows"] = dict(rows)
    info["seconds"] = round(time.time() - t0, 1)
    save_results(key, info)


def cmd_rung2(a):
    run_method_a(a.ns, a.updates, "rung2", "rung2-method-a.png", zero_control=True)


def data_curve():
    """Clean-pair share against the number of joint training scenes, log x, from results.json's rung2 and rung2b rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = load_results()
    pts = []
    for key in ("rung2", "rung2b"):
        for k, v in res.get(key, {}).items():
            if k.startswith("N="):
                pts.append((v["n_train_scenes"], res[key]["rows"][next(r for r in res[key]["rows"] if f"({v['n_train_scenes']} scenes)" in r)]))
    pts.sort(key=lambda p: p[0])
    base = res["rung1"]["rows"]["neural plain sum"]["clean_share"]
    exact_ps = res["rung0"]["rows"]["exact plain sum"]["clean_share"]
    n = [p[0] for p in pts]
    c = [p[1]["clean_share"] for p in pts]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ax.axhspan(base + 0.2, 1.0, color="#f2e6c9", alpha=0.5, lw=0)
    ax.axhspan(max(0.8, base + 0.5), 1.0, color="#cfe8cf", alpha=0.8, lw=0)
    ax.axhline(base, color="#888", lw=1, ls="--")
    ax.axhline(exact_ps, color="#b06", lw=1, ls=":")
    ax.plot(n, c, "o-", color="#1f4e8c", lw=1.8, ms=6)
    for xi, yi in zip(n, c):
        ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax.text(n[0], 0.81, "WORKS (at least 0.8 and 0.5 above the plain sum)", fontsize=7.5, va="bottom", color="#2c6b2c")
    ax.text(n[0], base + 0.21, "PARTIAL (at least 0.2 above the plain sum)", fontsize=7.5, va="bottom", color="#8a6d1f")
    ax.text(n[-1], base + 0.01, f"neural plain sum {base:.3f}", fontsize=7.5, va="bottom", ha="right", color="#555")
    ax.text(n[-1], exact_ps + 0.01, f"exact plain sum {exact_ps:.3f}", fontsize=7.5, va="bottom", ha="right", color="#b06")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlabel("joint training scenes N (log scale; 11,329 is the whole training pool)")
    ax.set_ylabel("clean-pair share on 256 shared seeds")
    ax.set_title("Method A: clean-pair share against the number of joint scenes", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "rung2-data-curve.png")
    print("data curve", list(zip(n, c)))


def cmd_rung2b(a):
    run_method_a(a.ns, a.updates, "rung2b", "rung2b-method-a.png", zero_control=False)
    data_curve()


def cmd_rung2c(a):
    """Method A trained longer, one change from rung 2b: the number of updates. Files carry a -u<updates> tag, so the 5,000-update
    models, samples, rows and data curve are left as they are."""
    run_method_a(a.ns, a.updates, "rung2c", "rung2c-method-a-longer.png", zero_control=False, tag=f"-u{a.updates}")


def cmd_rung3(a):
    t0 = time.time()
    (e, r, b, j, meta), (Ee, Er, Eb, Ej) = sets_and_exact()
    red, blue, empty = load_experts()
    tr, va = joint_split()
    xn = load_samples("neural-plainsum")
    base = metrics(xn, j)
    t = sigmas()
    # drawn: the neural plain sum's state after every Heun step i = 0..30, at sigma t[i+1]
    states, ssig = [], []

    def keep(i, x, s):
        if i < STEPS - 1:
            states.append(x.detach().clone())
            ssig.append(torch.full((len(x),), float(s), device=DEV))
    with torch.no_grad():
        for i in range(0, len(DRAWN_SEEDS), 256):
            heun(plainsum_denoiser(red, blue, empty), starting_noise(DRAWN_SEEDS[i:i + 256]), after_step=keep)
    S, Ssig = torch.cat(states), torch.cat(ssig)
    hold = torch.zeros(len(S), dtype=torch.bool)
    seed_of = torch.arange(len(S)) % 256 + 256 * (torch.arange(len(S)) // (256 * (STEPS - 1)))
    hold[seed_of % VAL_NTH == 0] = True
    S_tr, Sig_tr, S_va, Sig_va = S[~hold.to(DEV)], Ssig[~hold.to(DEV)], S[hold.to(DEV)], Ssig[hold.to(DEV)]
    R_tr, R_va = j[tr].to(DEV), j[va].to(DEV)
    sig_set = t[1:STEPS]
    torch.manual_seed(0)
    head = Head(1).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, betas=(0.9, 0.99))
    half = a.batch // 2
    t1 = time.time()
    for u in range(a.updates):
        k = torch.randint(len(S_tr), (half,), device=DEV)
        xf, sf = S_tr[k], Sig_tr[k]
        sr = sig_set[torch.randint(len(sig_set), (half,), device=DEV)]
        x0 = R_tr[torch.randint(len(R_tr), (half,), device=DEV)]
        xr = x0 + sr.view(-1, 1, 1, 1) * torch.randn_like(x0)
        logit = head(torch.cat([xr, xf]), torch.cat([sr, sf]))[:, 0]
        y = torch.cat([torch.ones(half, device=DEV), torch.zeros(half, device=DEV)])
        loss = F.binary_cross_entropy_with_logits(logit, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if u % 1000 == 0:
            print(f"  head C update {u} loss {float(loss):.4f}", flush=True)
    train_seconds = time.time() - t1
    head.eval().requires_grad_(False)
    torch.save(head.state_dict(), OUT / "head-C.pt")
    # read-only check: balanced accuracy per sigma band on held-out seeds' states and held-out joint scenes
    acc = {}
    g = torch.Generator(device=DEV).manual_seed(5)
    with torch.no_grad():
        for hi, lo in SIGMA_BANDS:
            sel = (Sig_va <= hi) & (Sig_va > lo)
            if sel.sum() == 0:
                continue
            xf, sf = S_va[sel][:2048], Sig_va[sel][:2048]
            x0 = R_va[torch.randint(len(R_va), (len(xf),), device=DEV, generator=g)]
            xr = x0 + sf.view(-1, 1, 1, 1) * torch.randn(x0.shape, device=DEV, generator=g)
            acc[f"{hi}-{lo}"] = round(0.5 * float((head(xr, sf)[:, 0] > 0).float().mean() + (head(xf, sf)[:, 0] <= 0).float().mean()), 3)
    print("head C held-out balanced accuracy by band", acc)
    xa = load_samples("methodA-Nall")                         # rung 2's Method A on the same training pool, for comparison
    ma = metrics(xa, j, train_idx=tr)
    ma["verdict"] = verdict(ma["clean_share"], base["clean_share"])
    rows = [("neural plain sum", base), ("Method A, N = all (rung 2)", ma)]
    xs, labels = [load_samples("exact-joint"), xn, xa], ["exact joint", "neural plain sum", "A, N=all"]
    for w in WEIGHTS:
        x = sample(push_denoiser(plainsum_denoiser(red, blue, empty), lambda xx, s: head(xx, s)[:, 0], w))
        save_samples(f"methodC-w{w}", x)
        m = metrics(x, j)
        m["verdict"] = verdict(m["clean_share"], base["clean_share"])
        rows.append((f"Method C, w = {w}", m))
        xs.append(x)
        labels.append(f"C, w={w}")
        print(rows[-1][0], m["clean_share"], m["verdict"], flush=True)
    print(table(rows))
    grid(xs, labels, RESULTS / "rung3-method-c.png")
    save_results("rung3", {"rows": dict(rows), "drawn_states": len(S), "head_updates": a.updates,
                           "head_train_seconds": round(train_seconds, 1), "head_heldout_balanced_accuracy_by_band": acc,
                           "seconds": round(time.time() - t0, 1)})


def cmd_rung4(a):
    t0 = time.time()
    (e, r, b, j, meta), (Ee, Er, Eb, Ej) = sets_and_exact()
    red, blue, empty = load_experts()
    tr, va = joint_split()
    xn = load_samples("neural-plainsum")
    base = metrics(xn, j)
    every = lambda n: ([i for i in range(n) if i % VAL_NTH != 0], [i for i in range(n) if i % VAL_NTH == 0])  # noqa: E731
    rtr, rva = every(len(r))
    classes_tr = [e.to(DEV), r[rtr].to(DEV), b[rtr].to(DEV), j[tr].to(DEV)]
    classes_va = [e.to(DEV), r[rva].to(DEV), b[rva].to(DEV), j[va].to(DEV)]
    torch.manual_seed(0)
    head = Head(4).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, betas=(0.9, 0.99))
    q = a.batch // 4

    def draw(classes, q, gen=None):
        xs, ys, ss = [], [], []
        for c, P in enumerate(classes):
            s = torch.exp(torch.empty(q, device=DEV).uniform_(math.log(SIGMA_LO), math.log(SIGMA_HI), generator=gen))
            x0 = P[torch.randint(len(P), (q,), device=DEV, generator=gen)]
            xs.append(x0 + s.view(-1, 1, 1, 1) * torch.randn(x0.shape, device=DEV, generator=gen))
            ys.append(torch.full((q,), c, device=DEV))
            ss.append(s)
        return torch.cat(xs), torch.cat(ss), torch.cat(ys)
    t1 = time.time()
    for u in range(a.updates):
        x, s, y = draw(classes_tr, q)
        loss = F.cross_entropy(head(x, s), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if u % 1000 == 0:
            print(f"  head B update {u} loss {float(loss):.4f}", flush=True)
    train_seconds = time.time() - t1
    head.eval().requires_grad_(False)
    torch.save(head.state_dict(), OUT / "head-B.pt")
    acc = {}
    g = torch.Generator(device=DEV).manual_seed(5)
    with torch.no_grad():
        for hi, lo in SIGMA_BANDS:
            per = []
            for c, P in enumerate(classes_va):
                s = torch.exp(torch.empty(512, device=DEV).uniform_(math.log(lo), math.log(hi), generator=g))
                x0 = P[torch.randint(len(P), (512,), device=DEV, generator=g)]
                x = x0 + s.view(-1, 1, 1, 1) * torch.randn(x0.shape, device=DEV, generator=g)
                per.append(float((head(x, s).argmax(1) == c).float().mean()))
            acc[f"{hi}-{lo}"] = {"per_class (empty, red, blue, both)": [round(p, 3) for p in per], "balanced": round(sum(per) / 4, 3)}
    print("head B held-out accuracy by band", json.dumps(acc))
    xa = load_samples("methodA-Nall")                         # rung 2's Method A on the same training pool, for comparison
    ma = metrics(xa, j, train_idx=tr)
    ma["verdict"] = verdict(ma["clean_share"], base["clean_share"])
    rows = [("neural plain sum", base), ("Method A, N = all (rung 2)", ma)]
    xs, labels = [load_samples("exact-joint"), xn, xa], ["exact joint", "neural plain sum", "A, N=all"]
    for w in WEIGHTS:
        x = sample(push_denoiser(plainsum_denoiser(red, blue, empty), lambda xx, s: fourway_ratio(head(xx, s)), w))
        save_samples(f"methodB-w{w}", x)
        m = metrics(x, j)
        m["verdict"] = verdict(m["clean_share"], base["clean_share"])
        rows.append((f"Method B, w = {w}", m))
        xs.append(x)
        labels.append(f"B, w={w}")
        print(rows[-1][0], m["clean_share"], m["verdict"], flush=True)
    print(table(rows))
    grid(xs, labels, RESULTS / "rung4-method-b.png")
    save_results("rung4", {"rows": dict(rows), "head_updates": a.updates, "head_train_seconds": round(train_seconds, 1),
                           "head_heldout_accuracy_by_band": acc, "seconds": round(time.time() - t0, 1)})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sets")
    sub.add_parser("rung0")
    q = sub.add_parser("experts")
    q.add_argument("--updates", type=int, default=6000)
    sub.add_parser("rung1")
    q = sub.add_parser("rung2")
    q.add_argument("--ns", type=int, nargs="+", default=[16, 64, 0], help="0 means the whole training pool")
    q.add_argument("--updates", type=int, default=5000)
    q = sub.add_parser("rung2c")
    q.add_argument("--ns", type=int, nargs="+", default=[1024, 0])
    q.add_argument("--updates", type=int, default=20000)
    q = sub.add_parser("rung2b")
    q.add_argument("--ns", type=int, nargs="+", default=[256, 1024, 4096])
    q.add_argument("--updates", type=int, default=5000)
    for name in ("rung3", "rung4"):
        q = sub.add_parser(name)
        q.add_argument("--updates", type=int, default=6000)
        q.add_argument("--batch", type=int, default=256)
        q.add_argument("--lr", type=float, default=5e-4)
    a = p.parse_args()
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
