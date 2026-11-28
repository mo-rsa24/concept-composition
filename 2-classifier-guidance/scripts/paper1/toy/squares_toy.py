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
  rung2c    Method A trained 20,000 updates at N = 1,024 and all
  rung3c    Method C at rung 2c's budget (20,000 updates, N = 1,024 and all), drawn states from 1,024 seeds (a) and 8,192 (b)
  rung4b    Method B at rung 2c's budget, the score unchanged
  rung5     Method D: rung 2c's Method A whose delta also reads the experts' inner decoder features
  rung6     every composed sampler from saved models, drawn plain and with a Langevin corrector at 5 and 20 steps per noise level
            (step 0.05 sigma^2, score (D - x) / sigma^2 from the row's own composed denoiser, push included)
  rungG1, rungG2, rungG3   the guidance-head ladder at N = 1,024 (see "The guidance ladder" below)
  cells     guidance cells CELL@N in order (G1, G2, G3, HL1, HL3, HL4), each read plain and with rung 6's corrector
  readG     sampling only: the plain and corrector reads for cells whose key has none (read2-<key>)
  rungG4    G4's single changes on the best deployable guidance cell at N = 1,024
  guidance_plan   readG, then the cells, then rungG4, in one job

  /home/molef/micromamba/envs/voice-agent/bin/python 2-classifier-guidance/scripts/paper1/toy/squares_toy.py rung0
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[4]
# checkpoints and sample tensors (git-ignored), and the results folder; SQUARES_TOY_OUT and SQUARES_TOY_RESULTS move them (on the
# cluster both point at /datasets/mmolefe/poe_repair_min/toys/paper1/)
OUT = Path(os.environ.get("SQUARES_TOY_OUT", REPO / "outputs" / "paper1-toy"))
RESULTS = Path(os.environ.get("SQUARES_TOY_RESULTS", REPO / "artifacts" / "results" / "does-the-interaction-term-work-on-a-small-toy"))
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


def heun(denoise, x, after_step=None, correct=None):
    """correct(i, x, sigma) -> x, if given, runs after every step that lands above sigma 0 (all but the last); rung 6's
    corrector. Without it the operations are the plain sampler's."""
    t = sigmas(x.device)
    for i in range(STEPS):
        t_cur, t_next = t[i], t[i + 1]
        d_cur = (x - denoise(x, t_cur, i, False)) / t_cur
        x_next = x + (t_next - t_cur) * d_cur
        if i < STEPS - 1:
            d_eval = denoise(x_next, t_next, i, True)
            x_next = x + (t_next - t_cur) * 0.5 * (d_cur + (x_next - d_eval) / t_next)
        x = x_next
        if correct is not None and i < STEPS - 1:
            x = correct(i, x, t_next)
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
# Rung 6's corrector: toy_caption_corrector.draw's Langevin step, moved to this toy's EDM denoisers.
# There, after each reverse step lands at noise level j >= 1, CORR_STEPS times: z <- z + eta s + sqrt(2 eta) noise, with s the score
# of the same composed noise estimate at level j and eta = CORR_SIZE times the noise variance at j. Here the level is the sigma a
# Heun step lands on (every step but the last, which lands on 0), the score is (D(x, sigma) - x) / sigma^2 with D the row's own
# composed denoiser (push included), and eta = CORRECTOR_SIZE sigma^2. That script draws its noise from one generator seeded
# training seed + 7; here each evaluation seed gets its own generator, seeded CORRECTOR_SEED_OFFSET + seed, so a seed's pictures do
# not depend on how seeds are batched, and the offset keeps clear of the evaluation seeds (0 to 255) and Method C's drawn seeds
# (100,000 to 108,191), whose generators make starting noise (seed + 7 would reuse seed 7's starting noise as seed 0's
# corrector noise).

CORRECTOR_SIZE = 0.05
CORRECTOR_SEED_OFFSET = 200000


def langevin_step(denoise, x, s, size, noise, step=0):
    """One corrector step at sigma s: x + eta score + sqrt(2 eta) noise, score = (D(x, s) - x) / s^2, eta = size s^2. The
    denoiser is called as a second evaluation, so a push's record (first evaluations only) is left as the plain sampler's."""
    score = (denoise(x, s, step, True) - x) / s ** 2
    eta = size * s ** 2
    return x + eta * score + (2 * eta).sqrt() * noise


def corrector(denoise, seeds, steps, size=CORRECTOR_SIZE):
    """heun's correct hook: after Heun step i lands at sigma, `steps` Langevin steps at that sigma. At each level every seed's own
    generator draws (steps, 3, H, H) noise. With steps 0 nothing is drawn and x is returned untouched."""
    gens = [torch.Generator().manual_seed(CORRECTOR_SEED_OFFSET + int(s)) for s in seeds]

    def correct(i, x, sigma):
        if steps == 0:
            return x
        noise = torch.stack([torch.randn(steps, 3, H, H, generator=g) for g in gens]).to(x.device)
        for k in range(steps):
            x = langevin_step(denoise, x, sigma, size, noise[:, k], step=i + 1)
        return x
    return correct


def sample_corrected(denoise, steps, size=CORRECTOR_SIZE, seeds=EVAL_SEEDS, batch=256, device=DEV):
    """sample() with the corrector at `steps` per noise level; the same starting noise per seed as every other row."""
    outs = []
    for i in range(0, len(seeds), batch):
        chunk = seeds[i:i + batch]
        outs.append(heun(denoise, starting_noise(chunk, device), correct=corrector(denoise, chunk, steps, size)).detach())
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

    def forward(self, h, sigma, feats=False):
        """With feats, also the decoder's inner feature maps [4 by 4 (after the middle blocks), 8 by 8 (after d2), 16 by 16
        (after d1)]; the output is computed by the same operations either way."""
        e = self.cond(fourier(sigma))
        h1 = self.e1(self.conv_in(h), e)
        h2 = self.e2(F.avg_pool2d(h1, 2), e)
        h3 = self.m2(self.m1(F.avg_pool2d(h2, 2), e), e)
        h8 = self.d2(torch.cat([F.interpolate(h3, scale_factor=2, mode="nearest"), h2], 1), e)
        h16 = self.d1(torch.cat([F.interpolate(h8, scale_factor=2, mode="nearest"), h1], 1), e)
        out = self.conv_out(h16)
        return (out, [h3, h8, h16]) if feats else out

    def logvar(self, sigma):
        return self.logvar_linear(fourier(sigma)).view(-1, 1, 1, 1)


class Expert(nn.Module):
    """EDM-preconditioned denoiser D = c_skip x + c_out F(c_in x, c_noise)."""

    def __init__(self, w=64):
        super().__init__()
        self.net = UNet(3, 3, w)

    def forward(self, x, sigma, feats=False):
        sigma = torch.as_tensor(sigma, device=x.device, dtype=torch.float32).reshape(-1).expand(len(x))
        s4 = sigma.view(-1, 1, 1, 1)
        if feats:
            out, fs = self.net(c_in(s4) * x, sigma, feats=True)
            return c_skip(s4) * x + c_out(s4) * out, fs
        return c_skip(s4) * x + c_out(s4) * self.net(c_in(s4) * x, sigma)


class Delta(nn.Module):
    """Method A's interaction term: c_out(sigma) F(c_in x, D_red, D_blue, D_empty; sigma), F's last convolution zero at the start."""
    wants_feats = False

    def __init__(self, w=64):
        super().__init__()
        self.net = UNet(12, 3, w, zero_out=True)

    def forward(self, x, sigma, d_r, d_b, d_e, feats=None):
        sigma = torch.as_tensor(sigma, device=x.device, dtype=torch.float32).reshape(-1).expand(len(x))
        s4 = sigma.view(-1, 1, 1, 1)
        return c_out(s4) * self.net(torch.cat([c_in(s4) * x, d_r, d_b, d_e], 1), sigma)


class FusedUNet(UNet):
    """Method D's network: UNet whose decoder also reads the three frozen experts' inner decoder features. At each resolution
    (4, 8, 16) the red, blue and empty experts' maps are concatenated and a 1 by 1 convolution projects them to `proj` channels,
    which are concatenated into the decoder at that resolution: beside the middle output at 4 by 4, beside the skip at 8 by 8 and
    at 16 by 16. Everything else (encoder, conditioning, zero last convolution, logvar) is UNet's."""

    def __init__(self, cin, cout, w=64, expert_w=64, proj=32, zero_out=True):
        super().__init__(cin, cout, w, zero_out=zero_out)
        self.p4 = nn.Conv2d(3 * 2 * expert_w, proj, 1)
        self.p8 = nn.Conv2d(3 * 2 * expert_w, proj, 1)
        self.p16 = nn.Conv2d(3 * expert_w, proj, 1)
        self.d2 = Block(2 * w + proj + 2 * w + proj, 2 * w)
        self.d1 = Block(2 * w + w + proj, w)

    def forward(self, h, sigma, ef):
        """ef: [[red 4, red 8, red 16], [blue ...], [empty ...]], the experts' feature lists in that order."""
        f4, f8, f16 = (torch.cat([f[k] for f in ef], 1) for k in range(3))
        e = self.cond(fourier(sigma))
        h1 = self.e1(self.conv_in(h), e)
        h2 = self.e2(F.avg_pool2d(h1, 2), e)
        h3 = torch.cat([self.m2(self.m1(F.avg_pool2d(h2, 2), e), e), self.p4(f4)], 1)
        h = self.d2(torch.cat([F.interpolate(h3, scale_factor=2, mode="nearest"), h2, self.p8(f8)], 1), e)
        h = self.d1(torch.cat([F.interpolate(h, scale_factor=2, mode="nearest"), h1, self.p16(f16)], 1), e)
        return self.conv_out(h)


class DeltaD(nn.Module):
    """Method D's interaction term: Method A's delta whose network also reads the experts' inner decoder features (FusedUNet)."""
    wants_feats = True

    def __init__(self, w=64, proj=32, expert_w=64):
        super().__init__()
        self.net = FusedUNet(12, 3, w, expert_w=expert_w, proj=proj)

    def forward(self, x, sigma, d_r, d_b, d_e, feats=None):
        sigma = torch.as_tensor(sigma, device=x.device, dtype=torch.float32).reshape(-1).expand(len(x))
        s4 = sigma.view(-1, 1, 1, 1)
        return c_out(s4) * self.net(torch.cat([c_in(s4) * x, d_r, d_b, d_e], 1), sigma, feats)


def experts_out(nets, x, s, feats=False):
    """(d_red, d_blue, d_empty, features or None). With feats, each expert runs once and returns its inner decoder maps too."""
    if not feats:
        return nets[0](x, s), nets[1](x, s), nets[2](x, s), None
    (d_r, f_r), (d_b, f_b), (d_e, f_e) = (n(x, s, feats=True) for n in nets)
    return d_r, d_b, d_e, [f_r, f_b, f_e]


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
    feats = getattr(delta, "wants_feats", False)

    def denoise(x, s, step, second):
        if not feats:
            d_r, d_b, d_e = red(x, s), blue(x, s), empty(x, s)
            return plainsum_of(d_r, d_b, d_e) + scale * delta(x, s, d_r, d_b, d_e)
        d_r, d_b, d_e, ef = experts_out((red, blue, empty), x, s, feats=True)
        return plainsum_of(d_r, d_b, d_e) + scale * delta(x, s, d_r, d_b, d_e, ef)
    return denoise


def push_denoiser(base, score, w, record=None):
    """D' = D + w sigma^2 grad_x score(x, sigma) at every evaluation; score returns one number per picture. With a record list,
    each first evaluation appends (step, sigma, root mean square of the push per value, mean score); the output is unchanged."""
    def denoise(x, s, step, second):
        d = base(x, s, step, second)
        with torch.enable_grad():
            xx = x.detach().requires_grad_(True)
            sc = score(xx, s)
            g, = torch.autograd.grad(sc.sum(), xx)
        push = w * s ** 2 * g
        if record is not None and not second:
            record.append((int(step), float(s), float(push.pow(2).mean().sqrt()), float(sc.detach().mean())))
        return d + push
    return denoise


# -------------------------------------------------------------------------------------------------------------------------------------
# Bookkeeping


def load_results():
    p = RESULTS / "results.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_results(key, value):
    """Read, add one key, write, under an exclusive lock, so two rungs running at once cannot drop each other's key."""
    import fcntl
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "results.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        res = load_results()
        res[key] = value
        res["bars"] = BARS
        res["device"] = str(DEV) + (f" ({torch.cuda.get_device_name()})" if DEV.type == "cuda" else "")
        tmp = RESULTS / "results.json.tmp"
        tmp.write_text(json.dumps(res, indent=1))
        tmp.replace(RESULTS / "results.json")


def require_new(key, paths):
    """Stop before any compute if the results key or any output file already exists: a new run never overwrites an old one."""
    if key in load_results():
        raise SystemExit(f"results.json already has key {key!r}; pick a new key")
    taken = [str(p) for p in paths if Path(p).exists()]
    if taken:
        raise SystemExit("these files already exist: " + ", ".join(taken))


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


def train_delta(nets, jpics, train_idx, val_idx, updates, batch=256, lr=5e-4, seed=0, val_every=250, model_cls=None):
    """model_cls None is Method A's Delta. A model whose class has wants_feats (DeltaD) also gets the experts' inner features;
    for Delta the calls below run the same operations as before that option existed."""
    red, blue, empty = nets
    torch.manual_seed(seed)
    model = (model_cls or Delta)().to(DEV)
    feats = getattr(model, "wants_feats", False)
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
            d_r, d_b, d_e, ef = experts_out(nets, x, s, feats)
            D = plainsum_of(d_r, d_b, d_e) + model(x, s, d_r, d_b, d_e, ef)
            tot += float((loss_weight(s.view(-1, 1, 1, 1)) * (D - x0) ** 2).mean((1, 2, 3)).sum())
        return tot / len(v0)

    best = (val_loss(), 0, {k: v.clone() for k, v in model.state_dict().items()})
    curve = [(0, None, best[0])]
    for u in range(1, updates + 1):
        x0 = X[torch.randint(len(X), (batch,), device=DEV)]
        s = draw_sigma(batch)
        x = x0 + s.view(-1, 1, 1, 1) * torch.randn_like(x0)
        with torch.no_grad():
            d_r, d_b, d_e, ef = experts_out(nets, x, s, feats)
        D = plainsum_of(d_r, d_b, d_e) + model(x, s, d_r, d_b, d_e, ef)
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


def run_method_a(ns, updates, key, picture, zero_control, tag="", method="A", model_cls=None, compare=None):
    """Method A at each N in ns (0 means the whole training pool); results under results.json[key], a grid at picture. method "D"
    with model_cls DeltaD runs Method D the same way, files named deltaD-... and methodD-...; compare maps a name to an earlier
    samples file whose row and picture are shown beside each N (rung 5 shows rung 2c's Method A)."""
    t0 = time.time()
    prefix = "delta" if method == "A" else f"delta{method}"
    names = [("all" if n == 0 else str(n)) + tag for n in ns]
    require_new(key, [OUT / f"{prefix}-N{nm}.pt" for nm in names] + [OUT / f"samples-method{method}-N{nm}.pt" for nm in names]
                + [RESULTS / picture])
    (e, r, b, j, meta), (Ee, Er, Eb, Ej) = sets_and_exact()
    nets = load_experts()
    red, blue, empty = nets
    tr, va = joint_split()
    xn = load_samples("neural-plainsum")
    base = metrics(xn, j)
    rows, xs, labels = [("neural plain sum", base)], [load_samples("exact-joint"), xn], ["exact joint", "neural plain sum"]
    info = {"updates": updates, "method": method}
    exact_resid = lambda x, s: Ej(x, s) - plainsum_of(red(x, s), blue(x, s), empty(x, s))  # noqa: E731
    for n in ns:
        t1 = time.time()
        idx = tr if n == 0 else tr[:n]
        name = ("all" if n == 0 else str(n)) + tag
        if compare and name in compare:
            xc = load_samples(compare[name][1])
            mc = metrics(xc, j, train_idx=idx)
            mc["verdict"] = verdict(mc["clean_share"], base["clean_share"])
            rows.append((compare[name][0], mc))
            xs.append(xc)
            labels.append(compare[name][2])
            t1 = time.time()
        model, tinfo = train_delta(nets, j, idx, va, updates, model_cls=model_cls)
        train_seconds = round(time.time() - t1, 1)
        torch.save(model.state_dict(), OUT / f"{prefix}-N{name}.pt")
        with torch.no_grad():
            x = sample(interaction_denoiser(red, blue, empty, model))
        save_samples(f"method{method}-N{name}", x)
        m = metrics(x, j, train_idx=idx)
        m["verdict"] = verdict(m["clean_share"], base["clean_share"])
        feats = getattr(model, "wants_feats", False)

        def delta_of(xx, s):
            d_r, d_b, d_e, ef = experts_out(nets, xx, s, feats)
            return model(xx, s, d_r, d_b, d_e, ef)
        err = band_error(delta_of, exact_resid, j[va])
        info[f"N={name}"] = {"n_train_scenes": len(idx), "train": tinfo, "residual_error_vs_exact_by_band": err,
                             "parameters": sum(p.numel() for p in model.parameters()), "train_seconds": train_seconds,
                             "seconds": round(time.time() - t1, 1)}
        rows.append((f"Method {method}, N = {name} ({len(idx)} scenes)" + (f", {updates} updates" if tag else ""), m))
        xs.append(x)
        labels.append(f"{method}, N={name}")
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

# -------------------------------------------------------------------------------------------------------------------------------------
# Matched budgets: Methods C and B trained as long as rung 2c's Method A (20,000 updates), and Method D


DRAWN_SEEDS_MORE = list(range(100000, 108192))                # rung 3c variant b; its first 1,024 are DRAWN_SEEDS
LOGIT_STEPS = (2, 8, 14, 20, 26)                              # states after these Heun steps sit at sigma 46.2, 13.3, 2.90, 0.415, 0.028


def picture_extras(x):
    """Median over seeds of the brightest R and the brightest B, and the share of seeds with no red, blue or purple pixel at all."""
    x = x.clamp(-1, 1)
    red, blue, purple = masks(x)
    empty = ~(red | blue | purple).flatten(1).any(1)
    return {"median_brightest_R": round(float(x[:, 0].flatten(1).max(1).values.median()), 3),
            "median_brightest_B": round(float(x[:, 2].flatten(1).max(1).values.median()), 3),
            "share_empty": float(empty.float().mean())}


def drawn_states(nets, seeds):
    """The neural plain sum's state after every Heun step i = 0..30 (at sigma t[i+1]) from each seed, in rung 3's layout (batches of
    256 seeds, then step, then seed in the batch), so the first 1,024 seeds' block is rung 3's drawn set exactly.
    Returns (states, their sigmas, each state's position in seeds)."""
    per, n = STEPS - 1, len(seeds)
    S = torch.empty(n * per, 3, H, H, device=DEV)
    Ssig = torch.empty(n * per, device=DEV)
    seed_of = torch.empty(n * per, dtype=torch.long)
    with torch.no_grad():
        for b0 in range(0, n, 256):
            chunk = seeds[b0:b0 + 256]
            nb, off = len(chunk), b0 * per

            def keep(i, x, s):
                if i < per:
                    S[off + i * nb: off + (i + 1) * nb] = x
                    Ssig[off + i * nb: off + (i + 1) * nb] = float(s)
                    seed_of[off + i * nb: off + (i + 1) * nb] = torch.arange(b0, b0 + nb)
            heun(plainsum_denoiser(*nets), starting_noise(chunk), after_step=keep)
    return S, Ssig, seed_of


def train_head(nout, draw_batch, val_batches, loss_fn, updates, lr, val_every=250, log=""):
    """Rung 3 and 4's head recipe (seed 0, AdamW, learning rate lr) run for `updates`, with the held-out loss on fixed batches
    measured every val_every updates as rung 2's delta does. Returns (final head, the head at the lowest held-out loss, info)."""
    torch.manual_seed(0)
    head = Head(nout).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, betas=(0.9, 0.99))

    @torch.no_grad()
    def val_loss():
        head.eval()
        tot, cnt = 0.0, 0
        for x, s, y in val_batches:
            tot += float(loss_fn(head(x, s), y, reduction="sum"))
            cnt += len(y)
        head.train()
        return tot / cnt
    best = (val_loss(), 0, {k: v.clone() for k, v in head.state_dict().items()})
    curve = [(0, None, best[0])]
    t1 = time.time()
    for u in range(1, updates + 1):
        x, s, y = draw_batch()
        loss = loss_fn(head(x, s), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if u % val_every == 0 or u == updates:
            vl = val_loss()
            curve.append((u, float(loss.detach()), vl))
            if u % 2000 == 0:
                print(f"  head {log} update {u} loss {float(loss.detach()):.4f} val {vl:.4f}", flush=True)
            if vl < best[0]:
                best = (vl, u, {k: v.clone() for k, v in head.state_dict().items()})
    train_seconds = round(time.time() - t1, 1)
    final = head.eval().requires_grad_(False)
    best_head = Head(nout).to(DEV)
    best_head.load_state_dict(best[2])
    info = {"updates": updates, "train_seconds": train_seconds, "best_update": best[1], "best_val_loss": best[0],
            "final_val_loss": curve[-1][2], "val_every": val_every, "curve": curve}
    return final, best_head.eval().requires_grad_(False), info


def push_rows(nets, score, sample_tag, label, j, base_clean):
    """Sample the plain sum pushed by score at every weight; rows with the toy's metrics plus brightness and empty share, and the
    push's size per step (root mean square per value and mean score over the 256 seeds, first evaluation of each step)."""
    rows, xs, traj = [], [], {}
    for w in WEIGHTS:
        rec = []
        x = sample(push_denoiser(plainsum_denoiser(*nets), score, w, record=rec))
        save_samples(f"{sample_tag}-w{w}", x)
        m = metrics(x, j)
        m["verdict"] = verdict(m["clean_share"], base_clean)
        m.update(picture_extras(x))
        rows.append((f"{label}, w = {w}", m))
        xs.append(x)
        traj[f"w={w}"] = [(st_, round(sg, 4), round(rms, 5), round(sc, 4)) for st_, sg, rms, sc in rec]
        print(rows[-1][0], m["clean_share"], m["verdict"], m["median_brightest_R"], m["median_brightest_B"], m["share_empty"],
              flush=True)
    return rows, xs, traj


def cmd_rung3c(a):
    """Method C at rung 2c's budget: the same head, data recipe and push as rung 3, trained a.updates updates, at N = 1,024 and all
    joint scenes as the real examples. Variant a draws from rung 3's 1,024 seeds; variant b from 8,192 seeds (more drawn data)."""
    t0 = time.time()
    U = a.updates
    names = {n: ("all" if n == 0 else str(n)) for n in a.ns}
    variants = {"a": len(DRAWN_SEEDS), "b": len(DRAWN_SEEDS_MORE)}
    files = [RESULTS / f"rung3c-method-c-matched-N{nm}.png" for nm in names.values()]
    for v in variants:
        for nm in names.values():
            files += [OUT / f"head-C{v}-N{nm}-u{U}.pt", OUT / f"head-C{v}-N{nm}-u{U}-best.pt"]
            files += [OUT / f"samples-methodC{v}-N{nm}-u{U}{b}-w{w}.pt" for w in WEIGHTS for b in ("", "-best")]
    require_new("rung3c", files)
    (e, r, b, j, meta), _ = sets_and_exact()
    nets = load_experts()
    tr, va = joint_split()
    xn = load_samples("neural-plainsum")
    base = metrics(xn, j)
    t = sigmas()
    sig_set = t[1:STEPS]
    t1 = time.time()
    S_all, Sig_all, seed_all = drawn_states(nets, DRAWN_SEEDS_MORE)
    states_seconds = round(time.time() - t1, 1)
    print(f"drawn states {len(S_all)} in {states_seconds} s", flush=True)
    R_va = j[va].to(DEV)
    info = {"updates": U, "batch": a.batch, "lr": a.lr, "drawn_states_seconds_8192_seeds": states_seconds}
    grids = {nm: ([load_samples("exact-joint"), xn], ["exact joint", "neural plain sum"]) for nm in names.values()}
    all_rows = [("neural plain sum", base)]
    for nm in names.values():
        xa = load_samples(f"methodA-N{nm}-u20000")
        ma = metrics(xa, j, train_idx=tr if nm == "all" else tr[:int(nm)])
        ma["verdict"] = verdict(ma["clean_share"], base["clean_share"])
        all_rows.append((f"Method A, N = {nm}, 20000 updates (rung 2c)", ma))
        grids[nm][0].append(xa)
        grids[nm][1].append(f"A, N={nm} (2c)")
    for v, nseeds in variants.items():
        k = len(DRAWN_SEEDS) * (STEPS - 1) if v == "a" else len(S_all)
        S, Ssig, seed_of = S_all[:k], Sig_all[:k], seed_all[:k]
        hold = (seed_of % VAL_NTH == 0).to(DEV)
        S_tr, Sig_tr, S_va, Sig_va = S[~hold], Ssig[~hold], S[hold], Ssig[hold]
        n_tr_seeds, n_va_seeds = int((torch.arange(nseeds) % VAL_NTH != 0).sum()), int((torch.arange(nseeds) % VAL_NTH == 0).sum())
        g = torch.Generator(device=DEV).manual_seed(11)
        kv = torch.randperm(len(S_va), device=DEV, generator=g)[:4096]
        vf, vsf = S_va[kv], Sig_va[kv]
        vsr = sig_set[torch.randint(len(sig_set), (len(kv),), device=DEV, generator=g)]
        vx0 = R_va[torch.randint(len(R_va), (len(kv),), device=DEV, generator=g)]
        vr = vx0 + vsr.view(-1, 1, 1, 1) * torch.randn(vx0.shape, device=DEV, generator=g)
        vx, vs = torch.cat([vr, vf]), torch.cat([vsr, vsf])
        vy = torch.cat([torch.ones(len(kv), device=DEV), torch.zeros(len(kv), device=DEV)])
        val_batches = [(vx[i:i + 1024], vs[i:i + 1024], vy[i:i + 1024]) for i in range(0, len(vx), 1024)]
        for n, nm in names.items():
            t1 = time.time()
            idx = tr if n == 0 else tr[:n]
            R_tr = j[idx].to(DEV)
            half = a.batch // 2

            def draw_batch():
                kk = torch.randint(len(S_tr), (half,), device=DEV)
                xf, sf = S_tr[kk], Sig_tr[kk]
                sr = sig_set[torch.randint(len(sig_set), (half,), device=DEV)]
                x0 = R_tr[torch.randint(len(R_tr), (half,), device=DEV)]
                xr = x0 + sr.view(-1, 1, 1, 1) * torch.randn_like(x0)
                y = torch.cat([torch.ones(half, device=DEV), torch.zeros(half, device=DEV)])
                return torch.cat([xr, xf]), torch.cat([sr, sf]), y

            def bce(logits, y, reduction="mean"):
                return F.binary_cross_entropy_with_logits(logits[:, 0], y, reduction=reduction)
            head, best_head, hinfo = train_head(1, draw_batch, val_batches, bce, U, a.lr, log=f"C{v} N={nm}")
            torch.save(head.state_dict(), OUT / f"head-C{v}-N{nm}-u{U}.pt")
            torch.save(best_head.state_dict(), OUT / f"head-C{v}-N{nm}-u{U}-best.pt")
            entry = {"drawn_seeds": nseeds, "drawn_train_seeds": n_tr_seeds, "drawn_heldout_seeds": n_va_seeds,
                     "drawn_states": len(S), "n_real_scenes": len(idx), "head": hinfo}
            for which, hd in (("final", head), ("best", best_head)):
                if which == "best" and hinfo["best_update"] == U:
                    entry["best"] = "the lowest held-out loss is at the last update, so the best head is the final head"
                    continue
                acc, logit_table = {}, {}
                gg = torch.Generator(device=DEV).manual_seed(5)
                with torch.no_grad():
                    for hi, lo in SIGMA_BANDS:
                        sel = (Sig_va <= hi) & (Sig_va > lo)
                        if sel.sum() == 0:
                            continue
                        xf, sf = S_va[sel][:2048], Sig_va[sel][:2048]
                        x0 = R_va[torch.randint(len(R_va), (len(xf),), device=DEV, generator=gg)]
                        xr = x0 + sf.view(-1, 1, 1, 1) * torch.randn(x0.shape, device=DEV, generator=gg)
                        acc[f"{hi}-{lo}"] = round(0.5 * float((hd(xr, sf)[:, 0] > 0).float().mean()
                                                            + (hd(xf, sf)[:, 0] <= 0).float().mean()), 3)
                    first = (seed_of < 256)
                    for i in LOGIT_STEPS:
                        s_i = float(t[i + 1])
                        at = (Ssig == s_i)
                        tr_sel = (at & first.to(DEV) & ~hold)
                        va_sel = (at & first.to(DEV) & hold)
                        x0 = R_va[:256]
                        xr = x0 + s_i * torch.randn(x0.shape, device=DEV, generator=gg)
                        logit_table[f"{s_i:.3f}"] = {
                            "drawn_training_seeds": round(float(hd(S[tr_sel], Ssig[tr_sel])[:, 0].mean()), 2),
                            "drawn_heldout_seeds": round(float(hd(S[va_sel], Ssig[va_sel])[:, 0].mean()), 2),
                            "real_heldout_scenes_noised": round(float(hd(xr, torch.full((256,), s_i, device=DEV))[:, 0].mean()), 2),
                            "n_training_seeds": int(tr_sel.sum()), "n_heldout_seeds": int(va_sel.sum())}
                tag = "" if which == "final" else "-best"
                lab = f"C({v}) N={nm}" + ("" if which == "final" else f" best@{hinfo['best_update']}")
                rows, xs, traj = push_rows(nets, lambda xx, s, hd=hd: hd(xx, s)[:, 0], f"methodC{v}-N{nm}-u{U}{tag}", lab, j,
                                           base["clean_share"])
                print(f"C{v} N={nm} {which} accuracy {acc}", json.dumps(logit_table), flush=True)
                entry[which] = {"heldout_balanced_accuracy_by_band": acc, "mean_logit_by_sigma": logit_table,
                                "rows": dict(rows), "push_by_step": traj}
                all_rows += rows
                if which == "final":
                    grids[nm][0].extend(xs)
                    grids[nm][1].extend([f"C({v}), w={w}" for w in WEIGHTS])
            entry["seconds"] = round(time.time() - t1, 1)
            info[f"C{v}-N={nm}"] = entry
    print(table(all_rows))
    for nm, (xs, labels) in grids.items():
        grid(xs, labels, RESULTS / f"rung3c-method-c-matched-N{nm}.png")
    info["rows"] = dict(all_rows)
    info["seconds"] = round(time.time() - t0, 1)
    save_results("rung3c", info)


def cmd_rung4b(a):
    """Method B at rung 2c's budget: rung 4's four-way head, classes and score unchanged, trained a.updates updates, with the both
    class drawn from N = 1,024 or all joint training scenes."""
    t0 = time.time()
    U = a.updates
    names = {n: ("all" if n == 0 else str(n)) for n in a.ns}
    files = [RESULTS / "rung4b-method-b-matched.png"]
    for nm in names.values():
        files += [OUT / f"head-B-N{nm}-u{U}.pt", OUT / f"head-B-N{nm}-u{U}-best.pt"]
        files += [OUT / f"samples-methodB-N{nm}-u{U}{b}-w{w}.pt" for w in WEIGHTS for b in ("", "-best")]
    require_new("rung4b", files)
    (e, r, b, j, meta), _ = sets_and_exact()
    nets = load_experts()
    tr, va = joint_split()
    xn = load_samples("neural-plainsum")
    base = metrics(xn, j)
    every = lambda n: ([i for i in range(n) if i % VAL_NTH != 0], [i for i in range(n) if i % VAL_NTH == 0])  # noqa: E731
    rtr, rva = every(len(r))
    classes_va = [e.to(DEV), r[rva].to(DEV), b[rva].to(DEV), j[va].to(DEV)]
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
    vx, vs, vy = draw(classes_va, 1024, torch.Generator(device=DEV).manual_seed(11))
    val_batches = [(vx[i:i + 1024], vs[i:i + 1024], vy[i:i + 1024]) for i in range(0, len(vx), 1024)]

    def ce(logits, y, reduction="mean"):
        return F.cross_entropy(logits, y, reduction=reduction)
    info = {"updates": U, "batch": a.batch, "lr": a.lr}
    xs_grid, labels = [load_samples("exact-joint"), xn], ["exact joint", "neural plain sum"]
    all_rows = [("neural plain sum", base)]
    fixed = {"confident touching pair (both +4, rest -4)": [-4, -4, -4, 4], "four even logits": [0, 0, 0, 0],
             "confident empty (neither +4, rest -4)": [4, -4, -4, -4], "confident red only": [-4, 4, -4, -4]}
    info["score_on_fixed_logits"] = {k: round(float(fourway_ratio(torch.tensor([v], dtype=torch.float32))[0]), 3)
                                     for k, v in fixed.items()}
    for n, nm in names.items():
        t1 = time.time()
        idx = tr if n == 0 else tr[:n]
        xa = load_samples(f"methodA-N{nm}-u20000")
        ma = metrics(xa, j, train_idx=idx)
        ma["verdict"] = verdict(ma["clean_share"], base["clean_share"])
        all_rows.append((f"Method A, N = {nm}, 20000 updates (rung 2c)", ma))
        xs_grid.append(xa)
        labels.append(f"A, N={nm} (2c)")
        classes_tr = [e.to(DEV), r[rtr].to(DEV), b[rtr].to(DEV), j[idx].to(DEV)]
        head, best_head, hinfo = train_head(4, lambda: draw(classes_tr, q), val_batches, ce, U, a.lr, log=f"B N={nm}")
        torch.save(head.state_dict(), OUT / f"head-B-N{nm}-u{U}.pt")
        torch.save(best_head.state_dict(), OUT / f"head-B-N{nm}-u{U}-best.pt")
        entry = {"n_both_scenes": len(idx), "head": hinfo}
        for which, hd in (("final", head), ("best", best_head)):
            if which == "best" and hinfo["best_update"] == U:
                entry["best"] = "the lowest held-out loss is at the last update, so the best head is the final head"
                continue
            acc, score_by_class = {}, {}
            g = torch.Generator(device=DEV).manual_seed(5)
            with torch.no_grad():
                for hi, lo in SIGMA_BANDS:
                    per, sc = [], []
                    for c, P in enumerate(classes_va):
                        s = torch.exp(torch.empty(512, device=DEV).uniform_(math.log(lo), math.log(hi), generator=g))
                        x0 = P[torch.randint(len(P), (512,), device=DEV, generator=g)]
                        x = x0 + s.view(-1, 1, 1, 1) * torch.randn(x0.shape, device=DEV, generator=g)
                        lg = hd(x, s)
                        per.append(float((lg.argmax(1) == c).float().mean()))
                        sc.append(round(float(fourway_ratio(lg).mean()), 3))
                    acc[f"{hi}-{lo}"] = {"per_class (empty, red, blue, both)": [round(p_, 3) for p_ in per],
                                         "balanced": round(sum(per) / 4, 3)}
                    score_by_class[f"{hi}-{lo}"] = dict(zip(("empty", "red only", "blue only", "both"), sc))
            tag = "" if which == "final" else "-best"
            lab = f"B N={nm}" + ("" if which == "final" else f" best@{hinfo['best_update']}")
            rows, xs, traj = push_rows(nets, lambda xx, s, hd=hd: fourway_ratio(hd(xx, s)), f"methodB-N{nm}-u{U}{tag}", lab, j,
                                       base["clean_share"])
            print(f"B N={nm} {which} accuracy", json.dumps(acc), "score by class", json.dumps(score_by_class), flush=True)
            entry[which] = {"heldout_accuracy_by_band": acc, "mean_score_by_class_and_band": score_by_class, "rows": dict(rows),
                            "push_by_step": traj}
            all_rows += rows
            if which == "final":
                xs_grid.extend(xs)
                labels.extend([f"B N={nm}, w={w}" for w in WEIGHTS])
        entry["seconds"] = round(time.time() - t1, 1)
        info[f"B-N={nm}"] = entry
    print(table(all_rows))
    grid(xs_grid, labels, RESULTS / "rung4b-method-b-matched.png")
    info["rows"] = dict(all_rows)
    info["seconds"] = round(time.time() - t0, 1)
    save_results("rung4b", info)


def cmd_rung5(a):
    """Method D: rung 2c's Method A with one change, the delta network also reads the experts' inner decoder features (DeltaD).
    Same loss, data, seed, updates, held-out rule and sampling; files carry the -u<updates> tag."""
    tag = f"-u{a.updates}"
    compare = {}
    for n in a.ns:
        nm = "all" if n == 0 else str(n)
        compare[nm + tag] = (f"Method A, N = {nm}, 20000 updates (rung 2c)", f"methodA-N{nm}-u20000", f"A, N={nm} (2c)")
    run_method_a(a.ns, a.updates, "rung5", "rung5-method-d.png", zero_control=True, tag=tag, method="D", model_cls=DeltaD,
                 compare=compare)


# -------------------------------------------------------------------------------------------------------------------------------------
# Rung 6: every composed sampler again, with a Langevin corrector at each noise level


RUNG6_STEPS = (0, 5, 20)                                     # corrector steps per noise level; 0 is the plain sampler, 20 is judged
RUNG6_JUDGED = 20
RUNG6_N = 1024
RUNG6_WEIGHTS = (16, 64)
RUNG6_WRITTEN_BEFORE = """# Rung 6: the corrector on every composed sampler (written before the run)

Every row is drawn on the 256 shared seeds from the same starting noise three ways: the plain Heun sampler (0 corrector steps),
5 and 20 Langevin corrector steps per noise level, step 0.05 of the noise variance (toy_caption_corrector.py's corrector, score
(D - x) / sigma^2 from each row's own composed denoiser, push included). Rows: exact joint (sanity), exact plain sum, neural plain
sum, Method A (N = 1,024, 20,000 updates, rung 2c's saved delta), Method D (N = 1,024, 20,000 updates, rung 5's saved model),
Method C (rung 3c's C(b) final head at N = 1,024, w = 16 and 64), Method B (rung 4b's final head at N = 1,024, w = 16 and 64).

The bars are the toy's, unchanged: a row WORKS if its clean-pair share is at least 0.8 and at least 0.5 above the neural plain
sum's plain-sampler share (0.000); PARTIAL if at least 0.2 above it but short of WORKS; FAILS otherwise.

The supervisor's prediction, judged at 20 corrector steps (5 steps reported beside it, not judged):
  - neural plain sum with the corrector: clean-pair share below 0.2
  - Method A with the corrector: above its plain-sampler 0.891
  - Method D with the corrector: above its plain-sampler 0.887
  - Method B and Method C with the corrector, at both weights: below 0.2
"""
RUNG6_PREDICTED = {                                           # row prefix -> (comparison, threshold), checked on the judged setting
    "neural plain sum": ("below", 0.2),
    "Method A": ("above", 0.891),
    "Method D": ("above", 0.887),
    "Method C": ("below", 0.2),
    "Method B": ("below", 0.2),
}


def rung6_rows():
    """(row name, files it needs under OUT, builder taking the exact denoisers and returning the row's composed denoiser, the
    plain sampler's saved samples name). Builders load their models only when called."""
    nm = f"N{RUNG6_N}-u20000"
    experts = [OUT / f"expert-{c}.pt" for c in ("red", "blue", "empty")]
    rows = []

    def exact_joint(ex):
        return joint_denoiser(ex[3])

    def exact_plain(ex):
        return plainsum_denoiser(ex[1], ex[2], ex[0])
    rows.append(("exact joint", [], exact_joint, "exact-joint"))
    rows.append(("exact plain sum", [], exact_plain, "exact-plainsum"))

    def neural_plain(ex):
        return plainsum_denoiser(*load_experts())
    rows.append(("neural plain sum", experts, neural_plain, "neural-plainsum"))

    def method(cls, path):
        def build(ex):
            model = cls().to(DEV)
            model.load_state_dict(torch.load(path, map_location=DEV))
            return interaction_denoiser(*load_experts(), model.eval().requires_grad_(False))
        return build
    rows.append((f"Method A, N = {RUNG6_N}, 20000 updates", experts + [OUT / f"delta-{nm}.pt"],
                 method(Delta, OUT / f"delta-{nm}.pt"), f"methodA-{nm}"))
    rows.append((f"Method D, N = {RUNG6_N}, 20000 updates", experts + [OUT / f"deltaD-{nm}.pt"],
                 method(DeltaD, OUT / f"deltaD-{nm}.pt"), f"methodD-{nm}"))

    def pushed(nout, path, score_of, w):
        def build(ex):
            head = Head(nout).to(DEV)
            head.load_state_dict(torch.load(path, map_location=DEV))
            head = head.eval().requires_grad_(False)
            return push_denoiser(plainsum_denoiser(*load_experts()), lambda xx, s: score_of(head(xx, s)), w)
        return build
    for w in RUNG6_WEIGHTS:
        rows.append((f"Method C, C(b) N = {RUNG6_N} final head, w = {w}", experts + [OUT / f"head-Cb-{nm}.pt"],
                     pushed(1, OUT / f"head-Cb-{nm}.pt", lambda lg: lg[:, 0], w), f"methodCb-{nm}-w{w}"))
    for w in RUNG6_WEIGHTS:
        rows.append((f"Method B, N = {RUNG6_N} final head, w = {w}", experts + [OUT / f"head-B-{nm}.pt"],
                     pushed(4, OUT / f"head-B-{nm}.pt", fourway_ratio, w), f"methodB-{nm}-w{w}"))
    return rows


def rung6_slug(name):
    import re
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def rung6_prediction_checks(rows, steps):
    """For each row drawn at `steps` corrector steps whose name starts with a predicted prefix: the prediction, the clean-pair
    share, and whether it held."""
    out = {}
    for name, m in rows.items():
        if not name.endswith(f", {steps} corrector steps"):
            continue
        for prefix, (how, thr) in RUNG6_PREDICTED.items():
            if name.startswith(prefix):
                c = m["clean_share"]
                out[name] = {"predicted": f"{how} {thr}", "clean_share": c, "held": bool(c < thr if how == "below" else c > thr)}
    return out


def cmd_rung6(a):
    """Rung 6: each composed sampler drawn plain and with the corrector at 5 and 20 steps per noise level, from saved models only.
    The predictions and bars are written to RESULTS/rung6-written-before-the-run.md and results.json before any sampling."""
    t0 = time.time()
    steps_list = sorted(set(a.steps))
    rows_spec = rung6_rows()
    missing = {name: [str(p) for p in need if not Path(p).exists()] for name, need, _, _ in rows_spec}
    missing = {k: v for k, v in missing.items() if v}
    todo = [r for r in rows_spec if r[0] not in missing]
    sample_files = [OUT / f"samples-rung6-{rung6_slug(name)}-k{k}.pt" for name, _, _, _ in todo for k in steps_list]
    before = RESULTS / "rung6-written-before-the-run.md"
    require_new("rung6", sample_files + [RESULTS / "rung6-corrector.png", before])
    for name, files in missing.items():
        print(f"SKIP {name}: missing {', '.join(files)}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    before.write_text(RUNG6_WRITTEN_BEFORE)
    save_results("rung6", {"status": "written before the run; no result yet", "written_before_the_run": RUNG6_WRITTEN_BEFORE,
                           "corrector_steps": steps_list, "corrector_size": a.size, "skipped_missing_files": missing})
    print(RUNG6_WRITTEN_BEFORE, flush=True)

    (e, r, b, j, meta), ex = sets_and_exact()
    tr, va = joint_split()
    base_file = OUT / "samples-neural-plainsum.pt"
    base = metrics(load_samples("neural-plainsum"), j)["clean_share"] if base_file.exists() else 0.0
    rows, xs, labels, plain_check, timing = {}, [], [], {}, {}
    for name, _, build, plain_name in todo:
        denoise = build(ex)
        idx = tr[:RUNG6_N] if name.startswith(("Method A", "Method D")) else None
        for k in steps_list:
            t1 = time.time()
            with torch.no_grad():
                x = sample_corrected(denoise, k, a.size)
            row = f"{name}, {k} corrector steps"
            timing[row] = round(time.time() - t1, 1)
            save_samples(f"rung6-{rung6_slug(name)}-k{k}", x)
            m = metrics(x, j, train_idx=idx)
            m["verdict"] = verdict(m["clean_share"], base)
            m.update(picture_extras(x))
            rows[row] = m
            xs.append(x)
            labels.append(f"{name.split(',')[0]}{', w=' + name.split('w = ')[1] if 'w = ' in name else ''}, k={k}")
            if k == 0 and plain_name and (OUT / f"samples-{plain_name}.pt").exists():
                xp = load_samples(plain_name)[:len(x)].to(x.device)
                plain_check[name] = {"saved_samples": f"samples-{plain_name}.pt", "byte_identical": bool(torch.equal(x, xp)),
                                     "max_abs_difference": float((x - xp).abs().max())}
            print(f"{row}: clean {m['clean_share']:.3f} {m['verdict']} purple seeds {m['share_with_purple']:.3f} "
                  f"empty {m['share_empty']:.3f} ({timing[row]} s)", flush=True)
    print(table(list(rows.items())))
    grid(xs, labels, RESULTS / "rung6-corrector.png")
    checks = rung6_prediction_checks(rows, RUNG6_JUDGED)
    reported = rung6_prediction_checks(rows, 5) if 5 in steps_list else {}
    print("predictions at 20 steps:", json.dumps(checks, indent=1), flush=True)
    info = {"written_before_the_run": RUNG6_WRITTEN_BEFORE, "corrector_steps": steps_list, "corrector_size": a.size,
            "corrector_noise": f"one generator per evaluation seed, seeded {CORRECTOR_SEED_OFFSET} + seed",
            "judged_steps": RUNG6_JUDGED, "neural_plain_sum_plain_sampler_clean_share": base,
            "skipped_missing_files": missing, "plain_sampler_against_saved_samples": plain_check,
            "prediction_checks_at_20_steps": checks, "prediction_checks_at_5_steps_reported_not_judged": reported,
            "row_seconds": timing, "rows": rows, "seconds": round(time.time() - t0, 1)}
    save_results("rung6", info)


# -------------------------------------------------------------------------------------------------------------------------------------
# The guidance ladder (G1 to G5): does classifier guidance help composition when it is done properly?
#
# The interaction term is the gradient of a scalar. With r(x, sigma) = log p_joint(x; sigma) - [log p_red + log p_blue - log p_empty]
# (each density the noised one at sigma), Tweedie's formula gives D_joint - D_plain_sum = sigma^2 grad_x r. So a head phi(x, sigma)
# whose push is sigma^2 grad_x phi can represent exactly what Method A's delta learns. Each rung below keeps that head and changes one
# thing about how it is trained or run:
#   G1  the head reads all three frozen experts' inner decoder features (rung 5's FusedUNet path), trained by Method A's EDM2
#       denoising loss on D_plain_sum + push, 1,024 joint scenes, 20,000 updates, Method A's optimiser and held-out rule
#   G2  the same head trained as the CLEVR head was: real joint scenes (noised at the sampler's levels) against the plain-sum
#       sampler's own states (rung 3c's variant b drawn set, 8,192 seeds), bounded logit, plus the alignment loss
#       1 - cos(sigma^2 grad logit, x0 - D_plain_sum(x_t)) on a quarter of each batch's real scenes at weight 1; the push is
#       w sigma^2 grad logit with w chosen on tuning seeds 10,000 to 10,255 (never an evaluation seed)
#   G3  the same head with its push regressed onto the exact interaction D_exact_joint(x_t) - D_plain_sum(x_t): the ceiling
#   G4  on the best of G1 to G3, one change each: EMA of the weights, whole-pixel shift augmentation, 3 times the updates, a small
#       vision transformer in place of the conv network
#   G5  the Langevin corrector (rung 6's, 20 steps per level, step 0.05 sigma^2) on the winner


TUNING_SEEDS = list(range(10000, 10256))                      # G2's weight choice; disjoint from evaluation, drawn and corrector seeds
G_N = 1024
G_UPDATES = 20000
ALIGN_SHARE_TOY = 0.25                                        # paper1_disc.py ALIGN_SHARE
ALIGN_WEIGHT_TOY = 1.0                                        # paper1_disc.py ALIGN_WEIGHT_ON
EMA_DECAY = 0.999
SHIFT_MAX = 4                                                 # paper1_disc.py's whole-pixel shift, up to 4 each way
G_PREDICTIONS = {                                             # artifacts/notes/progress/setup/supervisor.md, entry 2026-11-28 04:20 UTC
    "G1": "works, 0.75 to 0.90, near A's 0.891",
    "G2": "fails, under 0.2",
    "G3": "works, at least 0.9: the ceiling for any guidance head",
    "G4": "each within 0.05 of its parent, except 3 times the updates on G2, which may rise",
}


class ViTNet(nn.Module):
    """A small vision transformer with FusedUNet's inputs and output shape: the 12-channel picture input cut into patch by patch
    tokens, the three experts' inner decoder features brought to the token grid (4 by 4 maps upsampled, 16 by 16 maps averaged),
    projected and added to each token, then `depth` transformer blocks whose norms are scaled and shifted by sigma (adaLN, as in
    DiT), and a linear map from each token back to its patch's pixels, zero at the start."""

    def __init__(self, cin, cout, patch=2, dim=128, depth=4, heads=4, expert_w=64, proj=32):
        super().__init__()
        self.patch, self.cout, self.n, self.heads = patch, cout, H // patch, heads
        self.cond = nn.Sequential(nn.Linear(FOURIER, EMB), nn.SiLU(), nn.Linear(EMB, EMB), nn.SiLU())
        self.embed = nn.Conv2d(cin, dim, patch, stride=patch)
        self.p4 = nn.Conv2d(3 * 2 * expert_w, proj, 1)
        self.p8 = nn.Conv2d(3 * 2 * expert_w, proj, 1)
        self.p16 = nn.Conv2d(3 * expert_w, proj, 1)
        self.feat = nn.Linear(3 * proj, dim)
        self.pos = nn.Parameter(torch.randn(1, self.n * self.n, dim) * 0.02)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            blk = nn.ModuleDict({"n1": nn.LayerNorm(dim, elementwise_affine=False), "qkv": nn.Linear(dim, 3 * dim), "proj": nn.Linear(dim, dim),
                                 "n2": nn.LayerNorm(dim, elementwise_affine=False),
                                 "mlp": nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)),
                                 "ada": nn.Linear(EMB, 6 * dim)})
            nn.init.zeros_(blk["ada"].weight)
            nn.init.zeros_(blk["ada"].bias)
            self.blocks.append(blk)
        self.n_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.ada_out = nn.Linear(EMB, 2 * dim)
        self.out = nn.Linear(dim, cout * patch * patch)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.logvar_linear = nn.Linear(FOURIER, 1)
        nn.init.zeros_(self.logvar_linear.weight)
        nn.init.zeros_(self.logvar_linear.bias)

    def attend(self, qkv):
        """Softmax attention written out, because the fused attention kernels need not support the second derivative that
        training through grad_x phi takes."""
        b, n, _ = qkv.shape
        q, k, v = qkv.view(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), -1)
        return (a @ v).transpose(1, 2).reshape(b, n, -1)

    def forward(self, h, sigma, ef):
        f4, f8, f16 = (torch.cat([f[k] for f in ef], 1) for k in range(3))
        g = self.n
        fs = [F.interpolate(self.p4(f4), size=(g, g), mode="nearest") if f4.shape[-1] < g else F.adaptive_avg_pool2d(self.p4(f4), g),
              F.interpolate(self.p8(f8), size=(g, g), mode="nearest") if f8.shape[-1] < g else F.adaptive_avg_pool2d(self.p8(f8), g),
              F.adaptive_avg_pool2d(self.p16(f16), g)]
        e = self.cond(fourier(sigma))
        t = self.embed(h).flatten(2).transpose(1, 2) + self.feat(torch.cat(fs, 1).flatten(2).transpose(1, 2)) + self.pos
        for blk in self.blocks:
            s1, b1, g1, s2, b2, g2 = blk["ada"](e)[:, None].chunk(6, -1)
            y = blk["n1"](t) * (1 + s1) + b1
            t = t + g1 * blk["proj"](self.attend(blk["qkv"](y)))
            y = blk["n2"](t) * (1 + s2) + b2
            t = t + g2 * blk["mlp"](y)
        s, b = self.ada_out(e)[:, None].chunk(2, -1)
        t = self.out(self.n_out(t) * (1 + s) + b)                                      # (B, g*g, cout*p*p)
        p = self.patch
        return t.view(-1, g, g, self.cout, p, p).permute(0, 3, 1, 4, 2, 5).reshape(-1, self.cout, H, H)

    def logvar(self, sigma):
        return self.logvar_linear(fourier(sigma)).view(-1, 1, 1, 1)


class GuidanceHead(nn.Module):
    """One scalar phi(x, sigma) per picture, reading the noisy picture, the three experts' outputs and their inner decoder features.

    The push is sigma^2 grad_x phi, in the units of a denoiser correction (the same units as Method A's delta), because
    D_joint - D_plain_sum = sigma^2 grad_x r with r the log ratio of the joint density to the product of experts; so c(sigma) = sigma^2
    and phi stands for r. Inside, phi = (sigma_data / sigma) <c_in x, F(c_in x, D_red, D_blue, D_empty, features; sigma)>, summed
    over every value: with that scale the part of the push that does not differentiate F is c_out(sigma) F, Method A's delta form
    exactly, so the head starts with A's inductive bias and the rest of the gradient keeps the push a true gradient. F's last layer
    is zero at the start, so phi and the push start at 0. With bound, the logit is bound tanh(phi / bound) (the CLEVR head's bounded
    logit, used by G2 only)."""
    wants_feats = True

    def __init__(self, arch="conv", bound=None, w=64, proj=32, expert_w=64):
        super().__init__()
        self.arch, self.bound = arch, bound
        self.net = FusedUNet(12, 3, w, expert_w=expert_w, proj=proj) if arch == "conv" else ViTNet(12, 3, expert_w=expert_w, proj=proj)

    def forward(self, x, sigma, d_r, d_b, d_e, ef):
        sigma = torch.as_tensor(sigma, device=x.device, dtype=torch.float32).reshape(-1).expand(len(x))
        s4 = sigma.view(-1, 1, 1, 1)
        xin = c_in(s4) * x
        Fx = self.net(torch.cat([xin, d_r, d_b, d_e], 1), sigma, ef)
        phi = SIGMA_DATA / sigma * (xin * Fx).flatten(1).sum(1)
        return self.bound * torch.tanh(phi / self.bound) if self.bound else phi


def guided_parts(nets, head, x, s, create_graph=False):
    """(D_plain_sum detached, push = sigma^2 grad_x phi, phi). The experts run with x requiring grad, so the gradient flows through
    their outputs and features as well as the head; the experts' own weights are frozen. create_graph keeps the push differentiable
    in the head's weights (training through grad_x phi)."""
    with torch.enable_grad():
        xx = x.detach().requires_grad_(True)
        d_r, d_b, d_e, ef = experts_out(nets, xx, s, feats=True)
        phi = head(xx, s, d_r, d_b, d_e, ef)
        g, = torch.autograd.grad(phi.sum(), xx, create_graph=create_graph)
    sig = torch.as_tensor(s, device=x.device, dtype=torch.float32).reshape(-1).expand(len(x)).view(-1, 1, 1, 1)
    d_ps = plainsum_of(d_r, d_b, d_e).detach()
    return d_ps, sig ** 2 * g, phi


def guidance_denoiser(nets, head, w=1.0, record=None):
    """D_plain_sum + w sigma^2 grad_x phi at every evaluation. With a record list, each first evaluation appends (step, sigma, root
    mean square of the push per value, mean phi); the output is unchanged."""
    def denoise(x, s, step, second):
        d_ps, push, phi = guided_parts(nets, head, x, s)
        push = w * push.detach()
        if record is not None and not second:
            record.append((int(step), float(s), float(push.pow(2).mean().sqrt()), float(phi.detach().mean())))
        return d_ps + push
    return denoise


def scene_shift(x0, max_shift, gen=None):
    """Each scene moved by its own whole-pixel shift, up to max_shift each way, drawn uniformly among the shifts that keep every
    coloured pixel inside the picture, so a shifted touching pair is again a touching pair (a member of the joint set). The
    background is one colour, so rolling the picture is the move."""
    if max_shift <= 0:
        return x0
    n = len(x0)
    fg = (x0 > -0.5).any(1)                                                             # (n, H, H) coloured pixels
    rows, cols = fg.any(2), fg.any(1)
    ar = torch.arange(H, device=x0.device)
    top = torch.where(rows, ar, H).min(1).values
    bot = torch.where(rows, ar, -1).max(1).values
    lef = torch.where(cols, ar, H).min(1).values
    rig = torch.where(cols, ar, -1).max(1).values
    lo_r, hi_r = (-top).clamp(min=-max_shift), (H - 1 - bot).clamp(max=max_shift)
    lo_c, hi_c = (-lef).clamp(min=-max_shift), (H - 1 - rig).clamp(max=max_shift)
    u = torch.rand(n, 2, device=x0.device, generator=gen)
    dr = lo_r + (u[:, 0] * (hi_r - lo_r + 1).float()).floor().long().clamp(max=(hi_r - lo_r))
    dc = lo_c + (u[:, 1] * (hi_c - lo_c + 1).float()).floor().long().clamp(max=(hi_c - lo_c))
    ri = (ar[None, :] - dr[:, None]) % H                                                # source row for each target row
    ci = (ar[None, :] - dc[:, None]) % H
    out = x0.gather(2, ri[:, None, :, None].expand(n, 3, H, H))
    return out.gather(3, ci[:, None, None, :].expand(n, 3, H, H))


STATE_SEEDS = list(range(300000, 300256))                     # the guided sampler's own states for L3 and L4; no other seed set uses these
L1_BAND_GAIN = 4.0                                            # L1: the contact band and occlusion region weigh 1 + 4 = 5, every other pixel 1
L3_REWARD = {"weight": 0.1, "tau": 0.1, "batch": 128, "refresh": 1000}
L3_UPDATES, L3_LR = 5000, 1e-4
L4_ADV = {"weight": 0.1, "ramp": 2000, "r1": 1.0, "batch": 64, "d_lr": 2e-4, "sigma_lo": 0.05, "sigma_hi": 0.5, "refresh": 1000}


def contact_band_weights(jpics, meta, gain=L1_BAND_GAIN):
    """L1's per-pixel loss weights, one map per joint scene: 1 + gain on the contact band (pixels within one pixel of both the
    visible red and the visible blue square, which is the seam where they touch) and on the occlusion region (where the two
    squares' full extents overlap, so one hides the other), 1 elsewhere. Built from the scene's own pixels and its metadata."""
    red, blue, _ = masks(jpics)
    near = lambda m: F.max_pool2d(m.float()[:, None], 3, 1, 1)[:, 0] > 0  # noqa: E731
    band = near(red) & near(blue)
    full = torch.zeros(len(jpics), 2, H, H, dtype=torch.bool)
    for i, (rr, rc, br, bc, _, _) in enumerate(meta.tolist()):
        full[i, 0, rr:rr + SIDE, rc:rc + SIDE] = True
        full[i, 1, br:br + SIDE, bc:bc + SIDE] = True
    band = band | (full[:, 0] & full[:, 1])
    return (1 + gain * band.float())[:, None]


def soft_reward(x, tau=0.1):
    """L3's measured reward on a decoded picture, differentiable: a soft contact score minus a soft purple penalty. Soft red, blue
    and purple masks use sigmoid((value - 0.5) / tau) in place of the reader's thresholds at 0.5; contact = sum over pixels of the
    3 by 3 max of soft red times soft blue, scored 1 - exp(-contact) (0 with no contact, near 1 with a full seam); the penalty is the
    soft purple pixel count divided by SIDE (one purple square side costs 1)."""
    pr, pb = torch.sigmoid((x[:, 0] - PURPLE_T) / tau), torch.sigmoid((x[:, 2] - PURPLE_T) / tau)
    red, blue, purple = pr * (1 - pb), pb * (1 - pr), pr * pb
    contact = (F.max_pool2d(red[:, None], 3, 1, 1)[:, 0] * blue).flatten(1).sum(1)
    return (1 - torch.exp(-contact)) - purple.flatten(1).sum(1) / SIDE


def guided_states(nets, head, seeds, w=1.0):
    """The guided sampler's state after every Heun step but the last (at sigma t[i+1]), for each seed: (states, sigmas)."""
    S, Sg = [], []

    def keep_(i, x, s):
        if i < STEPS - 1:
            S.append(x.detach().clone())
            Sg.append(torch.full((len(x),), float(s), device=x.device))
    with torch.no_grad():
        for b0 in range(0, len(seeds), 256):
            heun(guidance_denoiser(nets, head, w), starting_noise(seeds[b0:b0 + 256]), after_step=keep_)
    return torch.cat(S), torch.cat(Sg)


class FeatureDisc(nn.Module):
    """L4's discriminator on the three frozen experts' inner decoder features of a lightly noised picture (sigma conditioned): the
    16 by 16 maps projected and run through a block, pooled to 8 by 8 beside the projected 8 by 8 maps, then to 4 by 4 beside the
    projected 4 by 4 maps, averaged, one raw score (hinge loss, no bound)."""

    def __init__(self, expert_w=64, w=64):
        super().__init__()
        self.cond = nn.Sequential(nn.Linear(FOURIER, EMB), nn.SiLU(), nn.Linear(EMB, EMB), nn.SiLU())
        self.p16, self.p8, self.p4 = nn.Conv2d(3 * expert_w, w, 1), nn.Conv2d(6 * expert_w, w, 1), nn.Conv2d(6 * expert_w, 2 * w, 1)
        self.b16, self.b8, self.b4 = Block(w, w), Block(2 * w, 2 * w), Block(4 * w, 2 * w)
        self.lin = nn.Linear(2 * w, 1)

    def forward(self, ef, sigma):
        f4, f8, f16 = (torch.cat([f[k] for f in ef], 1) for k in range(3))
        e = self.cond(fourier(sigma))
        h = self.b16(self.p16(f16), e)
        h = self.b8(torch.cat([F.avg_pool2d(h, 2), self.p8(f8)], 1), e)
        h = self.b4(torch.cat([F.avg_pool2d(h, 2), self.p4(f4)], 1), e)
        return self.lin(h.mean((2, 3)))[:, 0]


def train_guidance(nets, jpics, exact_joint, train_idx, val_idx, signal, bound=None, arch="conv", ema=None, shift=0,
                   updates=G_UPDATES, batch=256, lr=5e-4, seed=0, val_every=250, drawn=None, log="", pixel_weights=None,
                   init_state=None, reward=None, adv=None, keep="best"):
    """Train a GuidanceHead. signal "denoise" (G1): Method A's EDM2 loss on D_plain_sum + push against x0. "exact" (G3): the same
    loss against D_exact_joint(x_t), so the push is regressed onto the exact interaction. "classifier" (G2): real (label 1) against
    drawn (label 0) by binary cross-entropy on the bounded logit, plus the alignment loss; drawn = (states, sigmas, seed index).
    Every signal uses Method A's optimiser (Adam, lr 5e-4, betas 0.9 and 0.99, seed 0, batch 256) and keeps the weights at the
    lowest held-out loss of the signal's own kind, measured every val_every updates. With ema, the held-out loss and the kept
    weights are the exponential moving average's (decay ema per update).

    The head x L cells change only what trains the head (signal "denoise"):
      pixel_weights (L1)  one weight map per joint scene, multiplying the denoising loss per pixel
      init_state, reward (L3)  start from a trained head (G1's) and add -reward["weight"] times soft_reward of the decoded one-step
                          estimate D_plain_sum + push at the guided sampler's own states (STATE_SEEDS, refreshed every
                          reward["refresh"] updates with the current head)
      adv (L4)            a FeatureDisc on the frozen experts' inner features learns real joint scenes against the decoded one-step
                          estimates at the guided sampler's own states (hinge loss, R1 penalty on the real scenes), and the head's
                          push is trained to raise the discriminator's score beside the denoising loss, ramped in over adv["ramp"]
    keep "final" keeps the last weights (the reward and adversarial cells, whose held-out denoising loss is not their objective)."""
    import copy
    red, blue, empty = nets
    torch.manual_seed(seed)
    head = GuidanceHead(arch, bound).to(DEV)
    if init_state is not None:
        head.load_state_dict(init_state)
    avg = copy.deepcopy(head).requires_grad_(False) if ema else None
    opt = torch.optim.Adam(head.parameters(), lr=lr, betas=(0.9, 0.99))
    X = jpics[train_idx].to(DEV)
    PW = pixel_weights[train_idx].to(DEV) if pixel_weights is not None else None
    V = jpics[val_idx]
    g = torch.Generator().manual_seed(12345)
    v0 = V.repeat(2, 1, 1, 1)
    vs = (torch.randn(len(v0), generator=g) * P_STD + P_MEAN).exp()
    vn = torch.randn(v0.shape, generator=g)
    sig_set = sigmas()[1:STEPS]
    logvar = head.net.logvar

    if signal == "classifier":
        S, Ssig, seed_of = drawn
        hold = (seed_of % VAL_NTH == 0).to(DEV)
        S_tr, Sig_tr, S_va, Sig_va = S[~hold], Ssig[~hold], S[hold], Ssig[hold]
        gv = torch.Generator(device=DEV).manual_seed(11)
        kv = torch.randperm(len(S_va), device=DEV, generator=gv)[:2048]
        cvf, cvsf = S_va[kv], Sig_va[kv]
        cvsr = sig_set[torch.randint(len(sig_set), (len(kv),), device=DEV, generator=gv)]
        cvx0 = V.to(DEV)[torch.randint(len(V), (len(kv),), device=DEV, generator=gv)]
        cvr = cvx0 + cvsr.view(-1, 1, 1, 1) * torch.randn(cvx0.shape, device=DEV, generator=gv)
        half = batch // 2
        na = max(1, int(half * ALIGN_SHARE_TOY))

    def denoise_loss_val(model, exact_target=False):
        tot = 0.0
        for i in range(0, len(v0), 512):
            x0, s, n = v0[i:i + 512].to(DEV), vs[i:i + 512].to(DEV), vn[i:i + 512].to(DEV)
            x = x0 + s.view(-1, 1, 1, 1) * n
            d_ps, push, _ = guided_parts(nets, model, x, s)
            tgt = exact_joint(x, s) if exact_target else x0
            tot += float((loss_weight(s.view(-1, 1, 1, 1)) * (d_ps + push - tgt) ** 2).mean((1, 2, 3)).sum())
        return tot / len(v0)

    def align_loss(model, x0, s, gen=None, create_graph=False):
        xt = x0 + s.view(-1, 1, 1, 1) * torch.randn(x0.shape, device=DEV, generator=gen)
        d_ps, push, _ = guided_parts(nets, model, xt, s, create_graph=create_graph)
        return (1 - F.cosine_similarity(push.flatten(1), (x0 - d_ps).flatten(1), dim=1))

    def classifier_val(model):
        with torch.no_grad():
            bce, acc = 0.0, 0.0
            for xs_, ss_, y in ((cvr, cvsr, 1.0), (cvf, cvsf, 0.0)):
                lg = []
                for i in range(0, len(xs_), 512):
                    d_r, d_b, d_e, ef = experts_out(nets, xs_[i:i + 512], ss_[i:i + 512], feats=True)
                    lg.append(model(xs_[i:i + 512], ss_[i:i + 512], d_r, d_b, d_e, ef))
                lg = torch.cat(lg)
                bce += float(F.binary_cross_entropy_with_logits(lg, torch.full_like(lg, y), reduction="sum"))
                acc += float(((lg > 0) == bool(y)).float().mean()) / 2
        al = float(sum(align_loss(model, cvx0[i:i + 512], cvsr[i:i + 512], torch.Generator(device=DEV).manual_seed(13 + i)).sum()
                       for i in range(0, len(cvx0), 512)))
        return bce / (2 * len(kv)) + ALIGN_WEIGHT_TOY * al / len(kv), acc, al / len(kv)

    def val(model):
        model.eval()
        out = {"denoise_val": denoise_loss_val(model)}
        if signal == "exact":
            out["exact_val"] = denoise_loss_val(model, exact_target=True)
            out["select"] = out["exact_val"]
        elif signal == "classifier":
            out["select"], out["balanced_accuracy"], out["align_val"] = classifier_val(model)
        else:
            out["select"] = out["denoise_val"]
        model.train()
        return out

    states = None
    if adv:
        disc = FeatureDisc().to(DEV)
        d_opt = torch.optim.Adam(disc.parameters(), lr=adv["d_lr"], betas=(0.0, 0.99))

        def disc_score(img, sd):
            xn = img + sd.view(-1, 1, 1, 1) * torch.randn_like(img)
            _, _, _, ef = experts_out(nets, xn, sd, feats=True)
            return disc(ef, sd)

        def disc_sigma(k):
            return torch.exp(torch.empty(k, device=DEV).uniform_(math.log(adv["sigma_lo"]), math.log(adv["sigma_hi"])))
    refresh = (reward or adv or {}).get("refresh")
    side = {}

    watched = avg if ema else head
    v = val(watched)
    best = (v["select"], 0, {k: t.clone() for k, t in watched.state_dict().items()})
    curve = [(0, None, v)]
    t1 = time.time()
    for u in range(1, updates + 1):
        if refresh and (u - 1) % refresh == 0:
            states = guided_states(nets, head, STATE_SEEDS)
        if signal in ("denoise", "exact"):
            ii = torch.randint(len(X), (batch,), device=DEV)
            x0 = scene_shift(X[ii], shift)
            s = draw_sigma(batch)
            x = x0 + s.view(-1, 1, 1, 1) * torch.randn_like(x0)
            d_ps, push, _ = guided_parts(nets, head, x, s, create_graph=True)
            tgt = x0
            if signal == "exact":
                with torch.no_grad():
                    tgt = exact_joint(x, s)
            if PW is None:
                loss = edm_loss(d_ps + push, tgt, s, logvar(s))
            else:
                lv = logvar(s)
                loss = (loss_weight(s.view(-1, 1, 1, 1)) / lv.exp() * PW[ii] * (d_ps + push - tgt) ** 2 + lv).mean()
            if reward:
                kk = torch.randint(len(states[0]), (reward["batch"],), device=DEV)
                rd, rp, _ = guided_parts(nets, head, states[0][kk], states[1][kk], create_graph=True)
                rr = soft_reward(rd + rp, reward["tau"])
                loss = loss - reward["weight"] * rr.mean()
                side["reward"] = float(rr.mean().detach())
            if adv:
                kk = torch.randint(len(states[0]), (adv["batch"],), device=DEV)
                fd, fp, _ = guided_parts(nets, head, states[0][kk], states[1][kk], create_graph=True)
                fake = fd + fp
                ramp = min(1.0, u / adv["ramp"])
                loss = loss - adv["weight"] * ramp * disc_score(fake, disc_sigma(len(fake))).mean()
        else:
            kk = torch.randint(len(S_tr), (half,), device=DEV)
            xf, sf = S_tr[kk], Sig_tr[kk]
            ip = torch.randint(len(X), (half,), device=DEV)
            sr = sig_set[torch.randint(len(sig_set), (half,), device=DEV)]
            x0 = scene_shift(X[ip], shift)
            xr = x0 + sr.view(-1, 1, 1, 1) * torch.randn_like(x0)
            xb, sb = torch.cat([xr, xf]), torch.cat([sr, sf])
            with torch.no_grad():
                d_r, d_b, d_e, ef = experts_out(nets, xb, sb, feats=True)
            lg = head(xb, sb, d_r, d_b, d_e, ef)
            y = torch.cat([torch.ones(half, device=DEV), torch.zeros(half, device=DEV)])
            loss = F.binary_cross_entropy_with_logits(lg, y)
            ia = ip[:na]                                                                # unshifted, as paper1_disc.py aligns
            loss = loss + ALIGN_WEIGHT_TOY * align_loss(head, X[ia], sr[:na], create_graph=True).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if adv:
            real = X[torch.randint(len(X), (adv["batch"],), device=DEV)].clone().requires_grad_(True)
            dr = disc_score(real, disc_sigma(len(real)))
            df = disc_score(fake.detach(), disc_sigma(len(fake)))
            gr, = torch.autograd.grad(dr.sum(), real, create_graph=True)
            d_loss = F.relu(1 - dr).mean() + F.relu(1 + df).mean() + adv["r1"] / 2 * gr.pow(2).flatten(1).sum(1).mean()
            d_opt.zero_grad(set_to_none=True)
            d_loss.backward()
            d_opt.step()
            side.update({"d_real": float(dr.mean().detach()), "d_fake": float(df.mean().detach()), "d_loss": float(d_loss.detach())})
        if ema:
            with torch.no_grad():
                for pa, pm in zip(avg.parameters(), head.parameters()):
                    pa.lerp_(pm, 1 - ema)
                for ba, bm in zip(avg.buffers(), head.buffers()):
                    ba.copy_(bm)
        if u % val_every == 0 or u == updates:
            v = val(watched)
            v.update(side)
            curve.append((u, float(loss.detach()), v))
            if u % 1000 == 0:
                print(f"  head {log} update {u} loss {float(loss.detach()):.4f} val {json.dumps({k: round(x, 4) for k, x in v.items()})} "
                      f"({time.time() - t1:.0f} s)", flush=True)
            if v["select"] < best[0]:
                best = (v["select"], u, {k: t.clone() for k, t in watched.state_dict().items()})
    train_seconds = round(time.time() - t1, 1)
    final_state = {k: t.clone() for k, t in watched.state_dict().items()}
    if keep == "best":
        watched.load_state_dict(best[2])
    info = {"keep": keep, "denoise_val_at_start": curve[0][2]["denoise_val"], "denoise_val_at_end": curve[-1][2]["denoise_val"],
            "denoise_val_of_kept_weights": val(watched)["denoise_val"], "signal": signal, "bound": bound, "arch": arch, "ema": ema, "shift": shift, "updates": updates, "batch": batch, "lr": lr,
            "optimiser": "Adam, betas (0.9, 0.99), Method A's", "best_update": best[1], "best_val_select": best[0],
            "train_seconds": train_seconds, "curve": curve, "parameters": sum(p.numel() for p in head.parameters())}
    return watched.eval().requires_grad_(False), final_state, info


def g_files(name):
    return OUT / f"head-{name}.pt", OUT / f"samples-{name}.pt"


def g_base_rows(j, tr, n=G_N):
    """Rows and pictures every guidance cell shows beside its own: exact joint, neural plain sum, Method A at the same N (rung 2c's
    20,000-update run when it exists, else rung 2b's 5,000-update run), rung 5's Method D at N = 1,024."""
    xn = load_samples("neural-plainsum")
    base = metrics(xn, j)
    rows, xs, labels = [("neural plain sum", base)], [load_samples("exact-joint"), xn], ["exact joint", "neural plain sum"]
    cands = [(f"methodA-N{n}-u20000", f"Method A, N = {n}, 20000 updates (rung 2c)", "A (2c)"),
             (f"methodA-N{n}", f"Method A, N = {n}, 5000 updates (rung 2b)", "A (2b)"),
             (f"methodD-N{n}-u20000", f"Method D, N = {n}, 20000 updates (rung 5)", "D (rung 5)")]
    have_a = False
    for nm, lab, short in cands:
        if nm.startswith("methodA") and have_a:
            continue
        if (OUT / f"samples-{nm}.pt").exists():
            x = load_samples(nm)
            m = metrics(x, j, train_idx=tr[:n])
            m["verdict"] = verdict(m["clean_share"], base["clean_share"])
            rows.append((lab, m))
            xs.append(x)
            labels.append(short)
            have_a = have_a or nm.startswith("methodA")
    return base, rows, xs, labels


def choose_weight(nets, head, j, base_clean):
    """G2's weight: the weight of rung 3's grid (1, 4, 16, 64) whose pictures on the tuning seeds have the highest clean-pair share,
    ties to the smaller weight. Only tuning seeds choose; the evaluation seeds are read after the choice."""
    tuning = {}
    for w in WEIGHTS:
        with torch.no_grad():
            x = sample(guidance_denoiser(nets, head, w), seeds=TUNING_SEEDS)
        tuning[w] = metrics(x, j)
        print(f"  tuning seeds, w = {w}: clean {tuning[w]['clean_share']:.3f} purple seeds {tuning[w]['share_with_purple']:.3f}", flush=True)
    chosen = max(WEIGHTS, key=lambda w: (tuning[w]["clean_share"], -w))
    return chosen, {f"w={w}": {"clean_share": m["clean_share"], "share_with_purple": m["share_with_purple"],
                               "contact_share": m["contact_share"]} for w, m in tuning.items()}


def g_prediction_check(rung, clean, parent_clean=None, parent=None, variant=None):
    if rung == "G1":
        held = 0.75 <= clean <= 0.90
    elif rung == "G2":
        held = clean < 0.2
    elif rung == "G3":
        held = clean >= 0.9
    elif rung == "G4":
        held = (clean >= parent_clean - 0.05) if (variant == "u60000" and parent == "G2") else abs(clean - parent_clean) <= 0.05
    else:
        return {"predicted": "no prediction recorded for this cell on the supervisor page", "clean_share": clean}
    out = {"predicted": G_PREDICTIONS[rung], "clean_share": clean, "held": bool(held)}
    if parent_clean is not None:
        out.update({"parent": parent, "parent_clean_share": parent_clean, "difference": round(clean - parent_clean, 4)})
    return out


READ_STEPS = (0, RUNG6_JUDGED)                                # every cell is read plain (0) and with rung 6's corrector (20 per level)


def two_reads(nets, head, ws, j, idx, base_clean, name):
    """Each weight in ws read on the 256 shared seeds with the plain Heun sampler and with rung 6's Langevin corrector (20 steps per
    level, step 0.05 sigma^2, corrector noise seeded 200,000 + seed). Verdicts for both reads against the same bar (the neural
    plain sum's plain-sampler share). Samples saved as samples-<name>-w<w>-k<k>.pt. Returns (reads, {(w, k): pictures})."""
    reads, pics = {}, {}
    for w in ws:
        for k in READ_STEPS:
            f = OUT / f"samples-{name}-w{w:g}-k{k}.pt"
            t1 = time.time()
            with torch.no_grad():
                x = sample(guidance_denoiser(nets, head, w)) if k == 0 else sample_corrected(guidance_denoiser(nets, head, w), k)
            if not f.exists():
                torch.save(x.cpu(), f)
            m = metrics(x, j, train_idx=idx)
            m["verdict"] = verdict(m["clean_share"], base_clean)
            m.update(picture_extras(x))
            m["seconds"] = round(time.time() - t1, 1)
            reads.setdefault(f"w={w:g}", {})["plain" if k == 0 else f"corrector {k} steps"] = m
            pics[(w, k)] = x
            print(f"  read w = {w:g}, {'plain' if k == 0 else f'corrector {k}'}: clean {m['clean_share']:.3f} {m['verdict']} "
                  f"empty {m['share_empty']:.3f} purple seeds {m['share_with_purple']:.3f} ({m['seconds']} s)", flush=True)
    return reads, pics


def read_summary(reads):
    keep = ("clean_share", "verdict", "share_empty", "share_with_purple", "contact_share")
    return {w: {r: {k: m[k] for k in keep} for r, m in by.items()} for w, by in reads.items()}


def build_train_kw(cfg, jpics, meta):
    """The train_guidance keywords for a cell's JSON config (train_kw)."""
    kw = dict(cfg)
    out = {k: kw[k] for k in ("arch", "ema", "shift", "updates", "lr", "reward", "adv", "keep") if k in kw and kw[k] is not None}
    if kw.get("pixel_band_gain"):
        out["pixel_weights"] = contact_band_weights(jpics, meta, kw["pixel_band_gain"])
    if kw.get("init_from"):
        out["init_state"] = torch.load(kw["init_from"], map_location=DEV)
    return out


def run_cell(key, cell, name, picture, signal, bound, n=G_N, train_kw=None, parent=None, parent_clean=None, variant=None,
             change=None, smoke=False):
    """Train one GuidanceHead cell on N joint scenes, read it twice (plain and with the corrector; G2-style cells at the weight
    chosen on tuning seeds first, then every grid weight), write results.json[key] with both reads, the prediction check on the
    plain read at the chosen weight, and the picture."""
    t0 = time.time()
    train_kw = dict(train_kw or {})
    ckpt, _ = g_files(name)
    require_new(key, [ckpt, RESULTS / picture])
    (e, r, b, j, meta), (Ee, Er, Eb, Ej) = sets_and_exact()
    nets = load_experts()
    tr, va = joint_split()
    idx = tr[:n]
    base, rows, xs, labels = g_base_rows(j, tr, n)
    pcell = cell if n == G_N else f"{cell} at N = {n}"           # the recorded predictions are for N = 1,024
    info = {"cell": cell, "n_train_scenes": n, "the_one_change": change,
            "prediction_written_before_the_run": G_PREDICTIONS.get(pcell, "none recorded on the supervisor page")}
    drawn = None
    if signal == "classifier":
        t1 = time.time()
        drawn = drawn_states(nets, DRAWN_SEEDS_MORE if not smoke else DRAWN_SEEDS[:8])
        info["drawn_states"] = {"states": len(drawn[0]), "seconds": round(time.time() - t1, 1)}
    kw = build_train_kw(train_kw, j, meta)
    if smoke:
        kw["updates"] = min(kw.get("updates", G_UPDATES), 4)
        for d in ("reward", "adv"):
            if d in kw:
                kw[d] = dict(kw[d], refresh=2, ramp=2) if d == "adv" else dict(kw[d], refresh=2)
    head, final_state, tinfo = train_guidance(nets, j, Ej, idx, va, signal, bound, drawn=drawn, log=name,
                                              val_every=250 if not smoke else 2, **kw)
    torch.save(head.state_dict(), ckpt)
    torch.save(final_state, OUT / f"head-{name}-final.pt")
    info["train"] = tinfo
    w, ws = 1.0, [1.0]
    if signal == "classifier":
        w, info["tuning"] = choose_weight(nets, head, j, base["clean_share"])
        info["weight_rule"] = ("the weight of (1, 4, 16, 64) with the highest clean-pair share on tuning seeds 10,000 to 10,255, ties "
                               "to the smaller weight")
        ws = [w] + [x for x in WEIGHTS if x != w]
    info["weight"] = w
    reads, pics = two_reads(nets, head, ws, j, idx, base["clean_share"], name)
    info["reads"] = reads
    info["reads_summary"] = read_summary(reads)
    plain = reads[f"w={w:g}"]["plain"]
    for k in READ_STEPS:
        lab = f"{cell} N={n}" + (f" {variant}" if variant else "") + (f", w = {w:g}" if signal == "classifier" else "")
        rows.append((lab + (", plain" if k == 0 else f", corrector {k}"), reads[f"w={w:g}"]["plain" if k == 0 else f"corrector {k} steps"]))
        xs.append(pics[(w, k)])
        labels.append(f"{cell}{' ' + variant if variant else ''}" + (f" w={w:g}" if signal == "classifier" else "") + f", k={k}")

    def push_of(xx, s):
        return w * guided_parts(nets, head, xx, s)[1]
    exact_resid = lambda xx, s: Ej(xx, s) - plainsum_of(nets[0](xx, s), nets[1](xx, s), nets[2](xx, s))  # noqa: E731
    info["push_error_vs_exact_interaction_by_band"] = band_error(push_of, exact_resid, j[va])
    info["prediction_check"] = g_prediction_check(pcell, plain["clean_share"], parent_clean, parent, variant)
    info["checkpoint"] = str(ckpt)
    info["samples"] = str(OUT / f"samples-{name}-w{w:g}-k0.pt")
    info["picture"] = str(RESULTS / picture)
    info["config"] = {"name": name, "cell": cell, "signal": signal, "bound": bound, "n": n, "weight": w, "parent": parent,
                      "variant": variant, "train_kw": train_kw, "arch": train_kw.get("arch", "conv")}
    print(table(rows), flush=True)
    print(f"{key}: {json.dumps(info['reads_summary'][f'w={w:g}'])}; prediction {json.dumps(info['prediction_check'])}", flush=True)
    grid(xs, labels, RESULTS / picture)
    info["rows"] = dict(rows)
    info["seconds"] = round(time.time() - t0, 1)
    save_results(key, info)
    return plain["clean_share"]


G_RUNGS = {   # cell -> (signal, bound, train_kw, the one change)
    "G1": ("denoise", None, {}, "a scalar guidance head on the experts' features, push sigma^2 grad phi, trained by Method A's loss"),
    "G2": ("classifier", LOGIT_BOUND, {}, "G1's head trained as the CLEVR head was: real against drawn, bounded logit, alignment loss"),
    "G3": ("exact", None, {}, "G1's head with its push regressed onto the exact interaction (the ceiling)"),
    "HL1": ("denoise", None, {"pixel_band_gain": L1_BAND_GAIN},
            f"G1 with per-pixel denoising-loss weights 1 + {L1_BAND_GAIN:g} on the contact band and the occlusion region"),
    "HL3": ("denoise", None, {"init_from": "G1", "reward": L3_REWARD, "updates": L3_UPDATES, "lr": L3_LR, "keep": "final"},
            f"G1's head fine-tuned {L3_UPDATES} updates (lr {L3_LR:g}) with -{L3_REWARD['weight']:g} times the soft contact-minus-purple "
            "reward of the decoded one-step estimate at the guided sampler's own states, the denoising loss kept as the anchor"),
    "HL4": ("denoise", None, {"adv": L4_ADV, "ema": EMA_DECAY, "keep": "final"},
            f"G1 trained beside a discriminator on the experts' inner features (hinge, R1 {L4_ADV['r1']:g}, head EMA {EMA_DECAY}): "
            f"real joint scenes against the decoded one-step estimates at the guided sampler's own states, the push trained to fool "
            f"it at weight {L4_ADV['weight']:g} (ramped over {L4_ADV['ramp']} updates) beside the denoising loss"),
}
G4_VARIANTS = {   # variant -> (change to the parent's train_kw, the one change)
    "ema": (lambda kw: dict(kw, ema=EMA_DECAY), f"EMA of the weights, decay {EMA_DECAY} per update, held-out choice on the EMA"),
    "shift": (lambda kw: dict(kw, shift=SHIFT_MAX), f"whole-pixel shift augmentation of the joint scenes, up to {SHIFT_MAX} each way"),
    "u60000": (lambda kw: dict(kw, updates=3 * kw.get("updates", G_UPDATES)), "3 times the updates"),
    "vit": (lambda kw: dict(kw, arch="vit"), "a small vision transformer (patch 2, width 128, 4 blocks, 4 heads) in place of the conv network"),
}


def cell_key(cell, n, smoke=False):
    """G1 to G3 at N = 1,024 keep their first keys (rungG1 ...); every other cell carries its N."""
    k = f"rung{cell}" if (cell in ("G1", "G2", "G3") and n == G_N) else f"rung{cell}-N{n}"
    return k + ("-smoke" if smoke else "")


def cell_name(cell, n, train_kw, smoke=False):
    return f"{cell}-N{n}-u{train_kw.get('updates', G_UPDATES)}" + ("-smoke" if smoke else "")


def run_named_cell(cell, n, smoke=False):
    signal, bound, train_kw, change = G_RUNGS[cell]
    key = cell_key(cell, n, smoke)
    if key in load_results():
        print(f"skip {key}: already in results.json", flush=True)
        return
    train_kw = dict(train_kw)
    if train_kw.get("init_from") == "G1":
        src = load_results().get(cell_key("G1", n, smoke))
        if not src:
            print(f"skip {key}: G1 at N = {n} has no result yet to start from", flush=True)
            return
        train_kw["init_from"] = src["checkpoint"]
    name = cell_name(cell, n, train_kw, smoke)
    run_cell(key, cell, name, f"{key}-guidance{'' if not smoke else ''}.png", signal, bound, n, train_kw, change=change, smoke=smoke)


def cmd_cells(a):
    """Guidance cells in order, each CELL@N (CELL one of G1, G2, G3, HL1, HL3, HL4); a cell whose key exists is skipped, so a
    resubmitted job carries on where the last stopped."""
    for tok in a.cells:
        cell, n = tok.split("@")
        run_named_cell(cell, int(n), a.smoke)


def cmd_rungG(a):
    """G1, G2 or G3 at N = 1,024 (a.cmd names it)."""
    run_named_cell(a.cmd[4:], G_N, a.smoke)


def guidance_cells(res, smoke=False):
    """Every trained guidance cell in results.json (keys rungG..., rungHL...), smoke keys only with smoke."""
    return {k: v for k, v in res.items() if k.startswith(("rungG1", "rungG2", "rungG3", "rungG4", "rungHL")) and isinstance(v, dict)
            and "config" in v and k.endswith("-smoke") == smoke}


def cell_train_kw(entry):
    """A cell's train_kw, also for the first G1 to G3 keys, whose config predates train_kw."""
    cfg = entry["config"]
    if "train_kw" in cfg:
        return dict(cfg["train_kw"])
    return {k: cfg[k] for k in ("arch", "ema", "shift", "updates") if cfg.get(k) not in (None, 0, "conv") or k == "updates"}


def cmd_rungG4(a):
    """G4: on the best guidance cell at N = 1,024, one change at a time, each its own key. auto picks the highest plain-read
    clean-pair share among the cells trained without the exact joint denoiser (G3 is the ceiling, not a recipe that exists off the
    toy); ties go to the cell listed first."""
    res = load_results()
    cells = guidance_cells(res, a.smoke)
    if a.parent == "auto":
        pool = {k: v for k, v in cells.items() if v["config"].get("signal") != "exact" and v["config"].get("n", G_N) == G_N
                and not k.startswith("rungG4")}
        if not pool:
            raise SystemExit("no guidance cell to build on yet")
        pk = max(pool, key=lambda k: pool[k]["prediction_check"]["clean_share"])
    else:
        pk = a.parent
    parent = res[pk]
    pcfg = parent["config"]
    pcell = pcfg.get("cell") or pk[4:].split("-")[0]
    parent_clean = parent["prediction_check"]["clean_share"]
    print(f"G4 parent {pk} (plain clean {parent_clean:.3f}); pool {json.dumps({k: v['prediction_check']['clean_share'] for k, v in cells.items()})}", flush=True)
    for variant in a.variants:
        fn, change = G4_VARIANTS[variant]
        kw = fn(cell_train_kw(parent))
        if variant == "vit" and kw.get("init_from"):
            print(f"skip G4 {variant} on {pk}: the parent starts from a conv checkpoint, so a transformer has nothing to start from", flush=True)
            continue
        key = f"rungG4-{pcell}-N{pcfg.get('n', G_N)}-{variant}" + ("-smoke" if a.smoke else "")
        if key in res:
            print(f"skip {key}: already in results.json", flush=True)
            continue
        name = f"G4-{pcell}-N{pcfg.get('n', G_N)}-{variant}-u{kw.get('updates', G_UPDATES)}" + ("-smoke" if a.smoke else "")
        run_cell(key, "G4", name, f"{key}.png", pcfg["signal"], pcfg["bound"], pcfg.get("n", G_N), kw, parent=pcell,
                 parent_clean=parent_clean, variant=variant, change=change, smoke=a.smoke)


def cmd_readG(a):
    """Sampling only, no training: every guidance cell whose own key has no corrector read (the first G1 to G3 keys) read twice,
    plain and with rung 6's corrector, into a new key read2-<cell key>; G2-style cells at every grid weight."""
    res = load_results()
    (e, r, b, j, meta), _ = sets_and_exact()
    nets = load_experts()
    tr, va = joint_split()
    base = metrics(load_samples("neural-plainsum"), j)["clean_share"]
    for k, v in guidance_cells(res, a.smoke).items():
        if "reads" in v or f"read2-{k}" in res:
            continue
        t0 = time.time()
        cfg = v["config"]
        n = cfg.get("n", G_N)
        head = GuidanceHead(cfg.get("arch", "conv"), cfg["bound"]).to(DEV)
        head.load_state_dict(torch.load(v["checkpoint"], map_location=DEV))
        head = head.eval().requires_grad_(False)
        w = cfg["weight"]
        ws = [w] + ([x for x in WEIGHTS if x != w] if cfg["signal"] == "classifier" else [])
        print(f"read2 {k} at weights {ws}", flush=True)
        reads, pics = two_reads(nets, head, ws, j, tr[:n], base, cfg["name"])
        xp = torch.load(v["samples"]).to(pics[(w, 0)].device)
        info = {"cell_key": k, "weights": ws, "reads": reads, "reads_summary": read_summary(reads),
                "plain_read_against_the_cell_samples": {"byte_identical": bool(torch.equal(pics[(w, 0)], xp)),
                                                        "max_abs_difference": float((pics[(w, 0)] - xp).abs().max())},
                "picture": str(RESULTS / f"read2-{k}.png"), "seconds": round(time.time() - t0, 1)}
        grid([load_samples("exact-joint"), load_samples("neural-plainsum")] + [pics[(x, kk)] for x in ws for kk in READ_STEPS],
             ["exact joint", "neural plain sum"] + [f"{k[4:]} w={x:g}, k={kk}" for x in ws for kk in READ_STEPS], RESULTS / f"read2-{k}.png")
        print(f"read2-{k}: {json.dumps(info['reads_summary'])}", flush=True)
        save_results(f"read2-{k}", info)


def cmd_guidance_plan(a):
    """One job for the whole guidance column, in priority order, each step skipped when its key exists: the second read of every
    trained cell (sampling only), the cells in a.cells, a second read again (for cells trained in this job, a no-op), then G4's
    single changes on the best deployable cell at N = 1,024."""
    from argparse import Namespace
    cmd_readG(Namespace(smoke=a.smoke))
    cmd_cells(Namespace(cells=a.cells, smoke=a.smoke))
    if not a.no_g4:
        cmd_rungG4(Namespace(parent="auto", variants=list(G4_VARIANTS), smoke=a.smoke))


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
    for name in ("rung3c", "rung4b"):
        q = sub.add_parser(name)
        q.add_argument("--ns", type=int, nargs="+", default=[1024, 0], help="0 means the whole training pool")
        q.add_argument("--updates", type=int, default=20000)
        q.add_argument("--batch", type=int, default=256)
        q.add_argument("--lr", type=float, default=5e-4)
    q = sub.add_parser("rung5")
    q.add_argument("--ns", type=int, nargs="+", default=[1024, 0])
    q.add_argument("--updates", type=int, default=20000)
    q = sub.add_parser("rung6")
    q.add_argument("--steps", type=int, nargs="+", default=list(RUNG6_STEPS), help="corrector steps per noise level; 0 is plain")
    q.add_argument("--size", type=float, default=CORRECTOR_SIZE, help="corrector step as a fraction of sigma squared")
    for name in ("rungG1", "rungG2", "rungG3"):
        q = sub.add_parser(name)
        q.add_argument("--smoke", action="store_true", help="a quick check: keys and files end in -smoke")
    q = sub.add_parser("cells")
    q.add_argument("--cells", nargs="+", required=True, help="CELL@N, CELL one of G1 G2 G3 HL1 HL3 HL4, e.g. HL4@1024")
    q.add_argument("--smoke", action="store_true")
    q = sub.add_parser("rungG4")
    q.add_argument("--parent", default="auto", help="a results.json cell key, or auto (the best deployable cell at N = 1,024)")
    q.add_argument("--variants", nargs="+", default=list(G4_VARIANTS), choices=list(G4_VARIANTS))
    q.add_argument("--smoke", action="store_true")
    q = sub.add_parser("guidance_plan")
    q.add_argument("--cells", nargs="+", default=["G1@1024", "G2@1024", "G3@1024", "HL4@1024", "HL1@1024", "HL3@1024", "G1@256",
                                                  "HL4@256", "HL1@256", "HL3@256"])
    q.add_argument("--no-g4", action="store_true")
    q.add_argument("--smoke", action="store_true")
    q = sub.add_parser("readG")
    q.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    fn = "cmd_rungG" if a.cmd in ("rungG1", "rungG2", "rungG3") else "cmd_" + a.cmd
    globals()[fn](a)


if __name__ == "__main__":
    main()
