#!/usr/bin/env python3
"""Paper 1's Method A: a learned interaction term, trained directly by denoising on real joint scenes.

The combined denoiser, in EDM2's denoised-picture form (D is the network's guess of the clean picture, on the -1 to 1 scale):

    D_new(x, sigma) = D_red(x, sigma) + D_blue(x, sigma) - D_empty(x, sigma) + delta(x, sigma, count, D_red, D_blue, D_empty)

The three experts are frozen and called exactly as paper1_sample.plainsum_denoiser calls them: red and blue each told the count k as
EDM2's one-hot class vector, the empty-scene model told nothing. delta is a small U-Net, the only thing trained. Its output is its
raw 3-channel map F multiplied by c_out(sigma) = sigma * 0.5 / sqrt(sigma^2 + 0.25), as EDM2 scales its own network's output
(networks_edm2.py Precond.forward, lines 302 to 310), and F's last convolution starts at zero, so an untrained delta is exactly 0 and
D_new starts as the plain sum exactly.

delta's inputs. Twelve picture channels at 64 by 64: the noisy picture scaled as EDM2's preconditioning scales its input,
c_in(sigma) x = x / sqrt(sigma^2 + 0.25), then D_red, D_blue and D_empty. A conditioning vector from c_noise = log(sigma) / 4 (EDM2's
own noise label) in 32 fixed sine and cosine features, joined with the count as a one-hot of 3, through a two-layer network to 256
values; every residual block takes a scale and shift from it.

delta's shape (plans/paper-1-clevr-touching/plans/05-a-learned-interaction-term.md, "Route C: a correction network"), at base width
w (WIDTH, 96): a 3 by 3 convolution to w channels; two residual blocks at 64 by 64 (w); a 2 by 2 average pool; two residual blocks at
32 by 32 (2w); a nearest upsample to 64 by 64 joined with the earlier w-channel map; two residual blocks at w; a 3 by 3 convolution to
3 channels whose weights and bias start at zero. Each residual block is [GroupNorm with 8 groups, scale and shift, SiLU, 3 by 3
convolution] twice, with a 1 by 1 convolution on the skip when the channel count changes. No attention and no pooling layer over the
whole picture: through its convolutions each output pixel sees a window of 38 by 38 input pixels (measured by the gradient of the
centre pixel with the norms switched off: rows and columns 13 to 50); GroupNorm's per-picture mean and variance add a weak link to
every pixel. Weights (weight_count): 2,968,739 at w = 96; 1,551,555 at the plan's w = 64; 4,838,531 at w = 128. EDM2's learned loss
uncertainty (one value per sigma, the logvar of training_loop.py line 40) adds a linear layer on the same 32 sine and cosine
features: 33 weights, trained with delta, never used when sampling.

Training (the train subcommand). The loss settings are the experts' own, read from the EDM2 code they were trained with:
  sigma = exp(P_mean + P_std * n), n standard normal, fresh per example, P_mean = -0.8 and P_std = 1.6: the edm2-img64-xs preset
          (train_edm2.py line 33), passed to the loss at train_edm2.py line 76; m7_train.sbatch trains every expert with
          --preset=edm2-img64-xs and no --P_mean or --P_std. (EDM2's published -0.4 and 1.0 are its 512-pixel presets.)
  weight lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2 with sigma_data = 0.5 (EDM2Loss, training_loop.py lines
          29 and 37; Precond's default, networks_edm2.py line 281), and the loss lambda / exp(u(sigma)) * (D_new - x0)^2 + u(sigma),
          u the learned uncertainty (training_loop.py line 40). Since lambda = 1 / c_out^2, lambda * (D_new - x0)^2 is
          (F - (x0 - D_sum) / c_out)^2: the plan's denoising target.
  The EDM2 checkout these line numbers are from: git 4bf8162 (the laptop's copy at
  /home/molef/PhD/replications/projective-composition/reference/edm2; the cluster's at paper1_sample.EDM2_DIR).
Ten percent of the positives are held out (every tenth scene in render-index order, identical pictures kept on one side); every
VAL_EVERY updates, and at update 0 (where D_new is the plain sum) and the last update, the validation loss is the weighted squared
error lambda * (D_new - x0)^2, averaged over pixels, scenes and VAL_DRAWS fixed noise draws per scene (no uncertainty term, so the
numbers compare across updates). The checkpoint with the lowest validation loss is kept as interaction.pt; update 0 can win, in
which case the kept delta is zero and Method A is the plain sum.

Sampling (the sample subcommand). paper1_sample.heun with D_new at all 32 steps (63 evaluations) from paper1_sample.starting_noise at
each seed, the plain sum's own starting noise, writing <run>/count-<k>/<seed:06d>.png and timing.jsonl, the layout
paper1_overlap_stats.py and paper1_sample.py contact read. --zero-delta multiplies delta by 0 inside the same denoiser: the plain sum
through this code (the control); with an untrained delta it must write byte-identical pictures to paper1_sample.py plainsum.

The Langevin corrector (--corrector-steps K, --corrector-size 0.05). Copied from the squares toy's rung 6
(toy/squares_toy.py: langevin_step, corrector), itself the inclusive-captions project's toy_caption_corrector: after every Heun step
that lands above sigma 0 (heun's after_step hook, so 31 of the 32 steps), K steps at that sigma of
    x <- x + eta * score + sqrt(2 eta) * z,   score = (D(x, sigma) - x) / sigma^2,   eta = size * sigma^2,
with D the same composed denoiser the row samples with (the plain sum plus delta, or the plain sum under --zero-delta). Each seed's
z comes from its own CPU generator seeded CORRECTOR_SEED_OFFSET + seed (200,000 + seed), drawing (K, 3, 64, 64) per noise level, so a
seed's pictures do not depend on how seeds are batched. K = 0 (the default) passes no hook to heun: the sampler's operations, and
its pictures, are the uncorrected ones byte for byte. Each corrector step is one more denoiser call, so K steps cost 63 + 31 K calls
per picture against the plain sampler's 63.

Adversarial training (train --adv-weight lam; 0, the default, is Method A exactly: no discriminator is built, no extra random
number is drawn, and the training loop runs the same operations in the same order). Ported from the squares toy's L4
(toy/loss_ladder.py: FeatDisc, train_delta_x with adv_weight, run_L4, the L4_* constants), one change from Method A:
  The discriminator (FeatDisc below) reads a picture y re-noised at sigma_d, log-uniform on ADV_SIGMA_D, and the three frozen
  experts' inner decoder features at (y, sigma_d): the 16 by 16 and 8 by 8 decoder blocks paper1_disc.py's head reads (the block
  choice in paper1-first-disc-k64/blocks.json, ADV_BLOCK16 and ADV_BLOCK8), from red, blue and the empty-scene model, each projected
  by a 1 by 1 convolution and joined into a small encoder over c_in(sigma_d) y conditioned on sigma_d and the count. One raw score
  per picture, no bound (as the toy's L4).
  Each update, after Method A's denoising loss is formed on the batch: real joint scenes x0 and the delta's own current one-step
  estimates D_new (the same noise draw n_f re-noises every estimate) go through the discriminator; it takes one Adam step (lr
  ADV_D_LR, betas ADV_D_BETAS) on the hinge loss relu(1 - real) + relu(1 + fake), plus R1 = ADV_R1_EVERY * 0.5 * ADV_R1 *
  |grad_y real score|^2 on real pictures every ADV_R1_EVERY updates. Then delta is trained on the denoising loss plus
  lam * (-score of D_new + sigma_d n_f), the gradient flowing through the frozen experts' features into D_new.
  The expert features of D_new + sigma_d n_f are computed once: the discriminator's step reads them detached, the delta's term reads
  them live. The experts are frozen, so this is the same arithmetic as the toy's two separate passes. To fit a 24 GB card the experts
  run ADV_CHUNK pictures at a time, the live features through gradient checkpointing, and R1 is taken chunk by chunk with the
  gradients added; values and gradients are those of the whole batch at once.
  The discriminator's sigma_d and n_f come from their own CPU generator (seed + ADV_SEED_OFFSET), so the positives, sigmas and noise
  of the denoising loss are the same draws as Method A's at the same seed.
  delta's weights are averaged with EMA ADV_EMA after every update. The validation loss is read on the averaged weights and the
  lowest one is kept as interaction.pt (the kept row); the averaged weights at the last update are interaction-final.pt (the judged
  row, since with an adversarial loss the held-out denoising loss is no longer what is trained), the raw last weights
  interaction-final-raw.pt. Every 250 updates train.json records (update, discriminator loss, mean real score, mean fake score,
  delta's adversarial term).
  Two cheaper options (the toy's L4c, toy/loss_ladder.py run_L4c): --adv-r1-head-only takes R1 with the expert features
  computed without gradients and detached, so R1's gradient reaches y only through the discriminator's pixel path; --adv-experts 2
  has the discriminator read red's and blue's features only. Left out, both are L4 as above.
  The weight is chosen between ADV_WEIGHTS on tuning seeds with the final averaged delta (paper1_tuning_clean.py), as on the toy.

Usage (cluster, through paper1_gpu.sbatch; folders absolute or relative to /datasets/mmolefe/poe_repair_min/paper1/):
  python paper1_interaction.py train --run paper1-methodA-k384 --positives paper1-diag-pack-k384/k384-s1
  python paper1_interaction.py sample --run paper1-methodA-k384-rows --checkpoint paper1-methodA-k384/interaction.pt \
      --split paper1-first-lone-red/split.json --set evaluation --side touching --counts 1,2,3
  python paper1_interaction.py sample --run paper1-methodA-zero-rows --checkpoint paper1-methodA-k384/interaction.pt --zero-delta \
      --split paper1-first-lone-red/split.json --set evaluation --side touching --counts 1,2,3
  python paper1_interaction.py sample --run paper1-methodA-k384-k20-rows --checkpoint paper1-methodA-k384/interaction.pt \
      --corrector-steps 20 --split paper1-first-lone-red/split.json --set evaluation --side touching --counts 1,2,3
"""
import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper1_sample as ps  # noqa: E402
import paper1_disc as pdisc  # noqa: E402

DEV = ps.DEV

# ---------------------------------------------------------------------------------------------------------------------------------------
# The experts' loss settings (see the docstring for where each is read)

P_MEAN, P_STD = -0.8, 1.6     # edm2-img64-xs preset, train_edm2.py line 33
SIGMA_DATA = 0.5              # EDM2Loss and Precond default
LOSS_SOURCE = ("P_mean -0.8 and P_std 1.6 from the edm2-img64-xs preset (EDM2 train_edm2.py line 33, given to EDM2Loss at line 76; "
               "m7_train.sbatch trains the experts with --preset=edm2-img64-xs and no override); sigma_data 0.5 from EDM2Loss "
               "(training/training_loop.py line 29) and Precond (training/networks_edm2.py line 281); weight (sigma^2 + 0.25) / "
               "(0.5 sigma)^2 and the learned uncertainty from training_loop.py lines 37 and 40; EDM2 git 4bf8162")

# Training defaults
UPDATES = 4000
BATCH = 64
LR = 2e-4                     # the plan's route C rate; Adam with EDM2's betas (0.9, 0.99)
BETAS = (0.9, 0.99)
WIDTH = 96
GROUPS = 8
EMB = 256
FOURIER = 32
VAL_NTH = 10                  # every tenth scene (render-index order) is held out: 10 percent
VAL_EVERY = 250
VAL_DRAWS = 4                 # fixed noise draws per held-out scene
VAL_SEED = 12345
LOSS_CURVE_EVERY = 50         # train.json's loss curve: the mean training loss over each block of this many updates
WANDB_ENTITY, WANDB_PROJECT = pdisc.WANDB_ENTITY, pdisc.WANDB_PROJECT

# Adversarial training (--adv-weight; the toy's L4 constants, toy/loss_ladder.py line 95)
ADV_WEIGHTS = (0.003, 0.03)   # the weights tried; the choice is made on tuning seeds with the final averaged delta
ADV_R1, ADV_R1_EVERY = 1.0, 4
ADV_EMA = 0.999
ADV_SIGMA_D = (0.05, 1.0)     # the discriminator's re-noising sigma, log-uniform on this range
ADV_D_LR, ADV_D_BETAS = 2e-4, (0.0, 0.99)
ADV_D_WIDTH, ADV_D_PROJ = 48, 32   # the toy FeatDisc's widths
ADV_BLOCK8, ADV_BLOCK16 = "8x8_block1", "16x16_block3"   # paper1-first-disc-k64/blocks.json's chosen pair
ADV_SEED_OFFSET = 1000        # the discriminator's own generator is seeded seed + this
ADV_CURVE_EVERY = 250
ADV_CHUNK = 8                 # pictures per pass through the experts' features, so the graphs fit a 24 GB card (same arithmetic)


def c_in(sigma):
    return 1 / (SIGMA_DATA ** 2 + sigma ** 2).sqrt()


def c_out(sigma):
    return sigma * SIGMA_DATA / (sigma ** 2 + SIGMA_DATA ** 2).sqrt()


def loss_weight(sigma):
    """EDM2's lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma sigma_data)^2, which is 1 / c_out(sigma)^2."""
    return (sigma ** 2 + SIGMA_DATA ** 2) / (sigma * SIGMA_DATA) ** 2


def draw_sigma(n, gen):
    """sigma = exp(P_mean + P_std * n), n standard normal: EDM2Loss's noise levels for edm2-img64-xs."""
    return (torch.randn(n, generator=gen) * P_STD + P_MEAN).exp()


# ---------------------------------------------------------------------------------------------------------------------------------------
# delta


def fourier(sigma):
    """32 fixed sine and cosine features of c_noise = log(sigma) / 4, frequencies 2^0 to 2^7 times pi spread over 16 octave steps."""
    c_noise = sigma.float().log()[:, None] / 4
    freqs = math.pi * 2.0 ** torch.linspace(0, 7, FOURIER // 2, device=sigma.device)
    return torch.cat([(c_noise * freqs).sin(), (c_noise * freqs).cos()], 1)


class Block(nn.Module):
    """[GroupNorm, scale and shift from the conditioning, SiLU, 3 by 3 convolution] twice, plus the skip."""

    def __init__(self, cin, cout, emb=EMB, groups=GROUPS):
        super().__init__()
        self.norm1, self.film1, self.conv1 = nn.GroupNorm(groups, cin), nn.Linear(emb, 2 * cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2, self.film2, self.conv2 = nn.GroupNorm(groups, cout), nn.Linear(emb, 2 * cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, h, e):
        s, b = self.film1(e)[:, :, None, None].chunk(2, 1)
        y = self.conv1(F.silu(self.norm1(h) * (1 + s) + b))
        s, b = self.film2(e)[:, :, None, None].chunk(2, 1)
        y = self.conv2(F.silu(self.norm2(y) * (1 + s) + b))
        return self.skip(h) + y


class InteractionNet(nn.Module):
    """delta(x, sigma, count, D_red, D_blue, D_empty) = c_out(sigma) * F(c_in(sigma) x, D_red, D_blue, D_empty; sigma, count)."""

    def __init__(self, width=WIDTH, groups=GROUPS):
        super().__init__()
        self.width, w = width, width
        self.cond = nn.Sequential(nn.Linear(FOURIER + 3, EMB), nn.SiLU(), nn.Linear(EMB, EMB), nn.SiLU())
        self.conv_in = nn.Conv2d(12, w, 3, padding=1)
        self.enc = nn.ModuleList([Block(w, w, groups=groups), Block(w, w, groups=groups)])
        self.mid = nn.ModuleList([Block(w, 2 * w, groups=groups), Block(2 * w, 2 * w, groups=groups)])
        self.dec = nn.ModuleList([Block(3 * w, w, groups=groups), Block(w, w, groups=groups)])
        self.conv_out = nn.Conv2d(w, 3, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
        self.logvar_linear = nn.Linear(FOURIER, 1)            # EDM2's learned loss uncertainty u(sigma); loss only
        nn.init.zeros_(self.logvar_linear.weight)
        nn.init.zeros_(self.logvar_linear.bias)

    def raw(self, x, sigma, count, d_red, d_blue, d_empty):
        """F, the 3-channel map before c_out. sigma (n,), count (n,) in 1 to 3, pictures (n, 3, 64, 64)."""
        s4 = sigma.float().view(-1, 1, 1, 1)
        e = self.cond(torch.cat([fourier(sigma.float()), F.one_hot(count.long() - 1, 3).float()], 1))
        h = self.conv_in(torch.cat([c_in(s4) * x.float(), d_red.float(), d_blue.float(), d_empty.float()], 1))
        for blk in self.enc:
            h = blk(h, e)
        skip = h
        h = F.avg_pool2d(h, 2)
        for blk in self.mid:
            h = blk(h, e)
        h = torch.cat([F.interpolate(h, scale_factor=2, mode="nearest"), skip], 1)
        for blk in self.dec:
            h = blk(h, e)
        return self.conv_out(h)

    def forward(self, x, sigma, count, d_red, d_blue, d_empty):
        return c_out(sigma.float().view(-1, 1, 1, 1)) * self.raw(x, sigma, count, d_red, d_blue, d_empty)

    def logvar(self, sigma):
        return self.logvar_linear(fourier(sigma.float())).view(-1, 1, 1, 1)


def weight_count(model, with_logvar=False):
    """delta's weights; EDM2's uncertainty layer (33 weights) only when asked."""
    return sum(p.numel() for name, p in model.named_parameters() if with_logvar or not name.startswith("logvar_linear"))


# ---------------------------------------------------------------------------------------------------------------------------------------
# The combined denoiser


def expert_guesses(nets, x, sigma, labels):
    """D_red, D_blue, D_empty at x, called exactly as plainsum_denoiser calls them (red, then blue, then the empty-scene model)."""
    red, blue, empty = nets
    return red(x, sigma, labels), blue(x, sigma, labels), empty(x, sigma, None)


def combined(d_r, d_b, d_e, delta=None, scale=1.0):
    """D_new = (D_red + D_blue - D_empty) + scale * delta, the plain sum's floating-point operations first."""
    plain = (d_r + d_b - d_e).float()
    return plain if delta is None else plain + scale * delta


def interaction_denoiser(nets, model, k, n, zero_delta=False):
    """D_new for paper1_sample.heun at count k for n pictures; --zero-delta multiplies delta by 0 in the same code path."""
    labels = ps.one_hot(k, n)
    count = torch.full((n,), k, device=DEV)
    scale = 0.0 if zero_delta else 1.0

    def denoise(x, s, step, second=False):
        sv = s.expand(len(x))
        d_r, d_b, d_e = expert_guesses(nets, x, sv, labels)
        return combined(d_r, d_b, d_e, model(x, sv, count, d_r, d_b, d_e), scale)
    return denoise


# ---------------------------------------------------------------------------------------------------------------------------------------
# The Langevin corrector (toy/squares_toy.py rung 6; see the docstring)

CORRECTOR_SIZE = 0.05
CORRECTOR_SEED_OFFSET = 200000


def langevin_step(denoise, x, s, size, noise, step=0):
    """One corrector step at sigma s: x + eta score + sqrt(2 eta) noise, score = (D(x, s) - x) / s^2, eta = size s^2."""
    score = (denoise(x, s, step, True) - x) / s ** 2
    eta = size * s ** 2
    return x + eta * score + (2 * eta).sqrt() * noise


def corrector(denoise, seeds, steps, size=CORRECTOR_SIZE):
    """paper1_sample.heun's after_step hook: after Heun step i lands at sigma, `steps` Langevin steps at that sigma. At each level
    every seed's own generator draws (steps, 3, 64, 64) noise. With steps 0 nothing is drawn and x is returned untouched."""
    gens = [torch.Generator().manual_seed(CORRECTOR_SEED_OFFSET + int(s)) for s in seeds]
    level = [0]                                              # heun calls the hook once per step, in order, from step 0

    def correct(x, sigma):
        i = level[0]
        level[0] += 1
        if steps == 0:
            return x
        noise = torch.stack([torch.randn(steps, 3, 64, 64, generator=g) for g in gens]).to(x.device)
        for k in range(steps):
            x = langevin_step(denoise, x, sigma, size, noise[:, k], step=i + 1)
        return x
    return correct


# ---------------------------------------------------------------------------------------------------------------------------------------
# Data and the held-out split


def split_scenes(pics, names, nth=VAL_NTH):
    """(train indices, validation indices). Pictures with identical pixels form one group, so a scene rendered twice never sits on
    both sides; groups in order of their smallest render index, every nth group (ranks 0, nth, 2 nth, ...) held out."""
    groups = {}
    for i in range(len(names)):
        key = hashlib.sha256(pics[i].numpy().tobytes()).hexdigest()
        groups.setdefault(key, []).append(i)
    ordered = sorted(groups.values(), key=lambda g: min(pdisc.file_number(names[i]) for i in g))
    val = sorted(i for rank, g in enumerate(ordered) if rank % nth == 0 for i in g)
    train = sorted(set(range(len(names))) - set(val))
    return train, val


def validation_set(pics, counts, gen_seed=VAL_SEED, draws=VAL_DRAWS):
    """Each held-out picture repeated draws times with a fixed sigma and noise, so every validation reading uses the same draws."""
    gen = torch.Generator().manual_seed(gen_seed)
    x0 = pics.repeat(draws, 1, 1, 1)
    sigma = draw_sigma(len(x0), gen)
    noise = torch.randn(x0.shape, generator=gen)
    return x0, counts.repeat(draws), sigma, noise


@torch.no_grad()
def validation_loss(nets, model, val, batch=128):
    """Mean of lambda(sigma) (D_new - x0)^2 over pixels, channels, pictures and draws."""
    x0, counts, sigma, noise = val
    total, n = 0.0, 0
    for i in range(0, len(x0), batch):
        xb, cb, sb = x0[i:i + batch].to(DEV), counts[i:i + batch].to(DEV), sigma[i:i + batch].to(DEV)
        xt = xb + sb.view(-1, 1, 1, 1) * noise[i:i + batch].to(DEV)
        d_r, d_b, d_e = expert_guesses(nets, xt, sb, F.one_hot(cb.long() - 1, 3).float())
        d_new = combined(d_r, d_b, d_e, model(xt, sb, cb, d_r, d_b, d_e))
        total += float((loss_weight(sb.view(-1, 1, 1, 1)) * (d_new - xb) ** 2).sum())
        n += xb.numel()
    return total / max(1, n)


def training_loss(nets, model, x0, count, sigma, noise, return_d=False):
    """EDM2Loss on D_new: lambda / exp(u) * (D_new - x0)^2 + u, averaged over every element; the experts run without gradients.
    With return_d, also D_new itself (with its graph), for the adversarial term."""
    s4 = sigma.view(-1, 1, 1, 1)
    xt = x0 + s4 * noise
    with torch.no_grad():
        d_r, d_b, d_e = expert_guesses(nets, xt, sigma, F.one_hot(count.long() - 1, 3).float())
    d_new = combined(d_r, d_b, d_e, model(xt, sigma, count, d_r, d_b, d_e))
    logvar = model.logvar(sigma)
    loss = ((loss_weight(s4) / logvar.exp()) * (d_new - x0) ** 2 + logvar).mean()
    return (loss, d_new) if return_d else loss


# ---------------------------------------------------------------------------------------------------------------------------------------
# The discriminator for adversarial training (the toy's FeatDisc on CLEVR; see the docstring)


class FeatDisc(nn.Module):
    """Scores a picture y at re-noising level sigma_d and count k. The three frozen experts (red and blue told k, the empty-scene
    model told nothing) run on (y, sigma_d) up to their 16 by 16 decoder block; their ADV_BLOCK16 outputs, joined, are projected by
    a 1 by 1 convolution and joined to the pixel encoder at 16 by 16, and their ADV_BLOCK8 outputs likewise at 8 by 8. The pixel
    encoder: c_in(sigma_d) y, a 3 by 3 convolution to w channels at 64 by 64, a residual block, then average pools and residual
    blocks at 32 (w), 16 (2w, with the 16 by 16 features) and 8 (2w, with the 8 by 8 features), the mean over the grid and one
    linear layer. Gradients flow through the frozen experts to y."""

    def __init__(self, nets, w=ADV_D_WIDTH, proj=ADV_D_PROJ, experts=3):
        super().__init__()
        self._feats = [pdisc.ExpertFeatures(n, (ADV_BLOCK8, ADV_BLOCK16)) for n in list(nets)[:experts]]   # 2: red and blue only
        with torch.no_grad():                                   # the feature widths, read from one picture
            f = self.features(torch.zeros(1, 3, 64, 64, device=DEV), torch.ones(1, device=DEV), torch.ones(1, device=DEV))
        c16, c8 = f[0].shape[1], f[1].shape[1]
        self.cond = nn.Sequential(nn.Linear(FOURIER + 3, EMB), nn.SiLU(), nn.Linear(EMB, EMB), nn.SiLU())
        self.p16, self.p8 = nn.Conv2d(c16, proj, 1), nn.Conv2d(c8, proj, 1)
        self.conv_in = nn.Conv2d(3, w, 3, padding=1)
        self.b64, self.b32 = Block(w, w), Block(w, w)
        self.b16, self.b8 = Block(w + proj, 2 * w), Block(2 * w + proj, 2 * w)
        self.lin = nn.Linear(2 * w, 1)
        self.feature_channels = (c16, c8)

    def features(self, y, sigma, count, checkpointed=False):
        """(the joined 16 by 16 maps, the joined 8 by 8 maps) of red, blue and empty at (y, sigma), ADV_CHUNK pictures at a time.
        checkpointed: keep only each chunk's inputs and run the expert again during the backward pass (gradient checkpointing),
        so the graph through the experts is held for one expert and one chunk at a time; the values and gradients are the same."""
        labels = F.one_hot(count.long() - 1, 3).float()
        use_ckpt = checkpointed and torch.is_grad_enabled() and y.requires_grad
        out16, out8 = [], []
        for i in range(0, len(y), ADV_CHUNK):
            per = []
            for fx, lab in zip(self._feats, (labels, labels, None)):
                fn = lambda yy, ss, ll, fx=fx: tuple(fx(yy, ss, ll)[b] for b in (ADV_BLOCK16, ADV_BLOCK8))
                a = (y[i:i + ADV_CHUNK], sigma[i:i + ADV_CHUNK], None if lab is None else lab[i:i + ADV_CHUNK])
                per.append(torch.utils.checkpoint.checkpoint(fn, *a, use_reentrant=False) if use_ckpt else fn(*a))
            out16.append(torch.cat([f[0] for f in per], 1))
            out8.append(torch.cat([f[1] for f in per], 1))
        return torch.cat(out16), torch.cat(out8)

    def head(self, y, feats, sigma, count):
        e = self.cond(torch.cat([fourier(sigma.float()), F.one_hot(count.long() - 1, 3).float()], 1))
        h = self.b64(self.conv_in(c_in(sigma.float().view(-1, 1, 1, 1)) * y.float()), e)
        h = self.b32(F.avg_pool2d(h, 2), e)
        h = self.b16(torch.cat([F.avg_pool2d(h, 2), self.p16(feats[0])], 1), e)
        h = self.b8(torch.cat([F.avg_pool2d(h, 2), self.p8(feats[1])], 1), e)
        return self.lin(h.mean((2, 3)))[:, 0]

    def forward(self, y, sigma, count):
        return self.head(y, self.features(y, sigma, count), sigma, count)


class Adversary:
    """The discriminator, its optimiser and its own generator; step() runs one update's discriminator step and returns delta's
    adversarial term (with its graph to D_new) and the numbers recorded."""

    def __init__(self, nets, seed, r1_head_only=False, experts=3):
        self.disc = FeatDisc(nets, experts=experts).to(DEV)
        self.r1_head_only = r1_head_only
        self.opt = torch.optim.Adam(self.disc.parameters(), lr=ADV_D_LR, betas=ADV_D_BETAS)
        self.gen = torch.Generator().manual_seed(seed + ADV_SEED_OFFSET)
        self.lo, self.hi = math.log(ADV_SIGMA_D[0]), math.log(ADV_SIGMA_D[1])

    def step(self, update, x0, count, d_new):
        n = len(x0)
        sd = torch.empty(n).uniform_(self.lo, self.hi, generator=self.gen).exp().to(DEV)
        n_r = torch.randn(x0.shape, generator=self.gen).to(DEV)
        n_f = torch.randn(x0.shape, generator=self.gen).to(DEV)
        s4 = sd.view(-1, 1, 1, 1)
        y_r = x0 + s4 * n_r
        y_f = d_new + s4 * n_f                                   # with its graph to delta
        r1 = update % ADV_R1_EVERY == 0
        f_f = self.disc.features(y_f, sd, count, checkpointed=True)
        with torch.no_grad():
            f_r = self.disc.features(y_r, sd, count)
        self.disc.requires_grad_(True)
        l_r = self.disc.head(y_r, f_r, sd, count)
        l_f = self.disc.head(y_f.detach(), tuple(f.detach() for f in f_f), sd, count)
        loss_d = F.relu(1 - l_r).mean() + F.relu(1 + l_f).mean()
        self.opt.zero_grad(set_to_none=True)
        loss_d.backward()
        loss_d = float(loss_d.detach())
        if r1 and self.r1_head_only:  # R1 through the head only: the real features above are detached, so y reaches it by pixels
            yg = y_r.detach().requires_grad_(True)
            g, = torch.autograd.grad(self.disc.head(yg, f_r, sd, count).sum(), yg, create_graph=True)
            term = ADV_R1_EVERY * 0.5 * ADV_R1 * g.pow(2).flatten(1).sum(1).mean()
            term.backward()
            loss_d += float(term.detach())
        elif r1:                     # R1 on real pictures, chunk by chunk; each chunk's share of the batch mean, gradients added
            for i in range(0, n, ADV_CHUNK):
                yc = y_r[i:i + ADV_CHUNK].detach().requires_grad_(True)
                lc = self.disc(yc, sd[i:i + ADV_CHUNK], count[i:i + ADV_CHUNK])
                g, = torch.autograd.grad(lc.sum(), yc, create_graph=True)
                term = ADV_R1_EVERY * 0.5 * ADV_R1 * g.pow(2).flatten(1).sum(1).sum() / n
                term.backward()
                loss_d += float(term.detach())
        self.opt.step()
        self.disc.requires_grad_(False)
        g_adv = -self.disc.head(y_f, f_f, sd, count).mean()
        return g_adv, (loss_d, float(l_r.detach().mean()), float(l_f.detach().mean()), float(g_adv.detach()))


# ---------------------------------------------------------------------------------------------------------------------------------------
# Checkpoint


def save_checkpoint(path, model, settings, update, val):
    torch.save({"delta": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "width": model.width,
                "groups": GROUPS, "update": update, "validation_loss": val, "settings": settings}, path)


def load_interaction(path):
    """interaction.pt to an InteractionNet on DEV in eval mode with no gradients, and the checkpoint's record."""
    ckpt = torch.load(path, map_location="cpu")
    model = InteractionNet(ckpt["width"], ckpt.get("groups", GROUPS))
    model.load_state_dict(ckpt["delta"])
    return model.to(DEV).eval().requires_grad_(False), {k: v for k, v in ckpt.items() if k != "delta"}


# ---------------------------------------------------------------------------------------------------------------------------------------
# train


def cmd_train(args):
    folder = ps.run_dir(args)
    pos_dir = pdisc.resolve(args.root, args.positives)
    pics, counts, names = pdisc.load_folder(pos_dir)
    train_idx, val_idx = split_scenes(pics, names)
    shared = {names[i] for i in train_idx} & {names[i] for i in val_idx}
    if shared:
        raise SystemExit(f"{len(shared)} scenes on both sides of the split")
    tr_x, tr_c = pics[train_idx], counts[train_idx]
    val = validation_set(pics[val_idx], counts[val_idx])
    nets, _ = ps.load_three(args)

    torch.manual_seed(args.seed)
    model = InteractionNet(args.width).to(DEV).train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=BETAS)
    settings = {"positives": args.positives, "positives_count": len(names), "train_scenes": len(train_idx),
                "validation_scenes": len(val_idx), "validation_names": [names[i] for i in val_idx],
                "split_rule": f"identical pictures grouped; groups in render-index order; every {VAL_NTH}th group (ranks 0, {VAL_NTH}, ...) held out",
                "updates": args.updates, "batch": args.batch, "lr": args.lr, "betas": list(BETAS), "seed": args.seed, "width": args.width,
                "groups": GROUPS, "P_mean": P_MEAN, "P_std": P_STD, "sigma_data": SIGMA_DATA, "loss_source": LOSS_SOURCE,
                "val_every": args.val_every, "val_draws": VAL_DRAWS, "val_seed": VAL_SEED,
                "delta_weights": weight_count(model), "uncertainty_weights": weight_count(model, True) - weight_count(model),
                "red_net": args.red_net, "blue_net": args.blue_net, "empty_net": args.empty_net}
    adv = args.adv_weight > 0
    if args.adv_weight < 0:
        raise SystemExit("--adv-weight must be 0 or more")
    ema, adversary, judged = None, None, model
    if adv:                                                  # adversarial training (see the docstring); nothing here at 0
        ema = copy.deepcopy(model).requires_grad_(False)
        judged = ema
        adversary = Adversary(nets, args.seed, r1_head_only=args.adv_r1_head_only, experts=args.adv_experts)
        settings.update({"adv_weight": args.adv_weight, "adv_r1": ADV_R1, "adv_r1_every": ADV_R1_EVERY, "adv_ema": ADV_EMA,
                         "adv_sigma_d": list(ADV_SIGMA_D), "adv_d_lr": ADV_D_LR, "adv_d_betas": list(ADV_D_BETAS),
                         "adv_d_width": ADV_D_WIDTH, "adv_d_proj": ADV_D_PROJ, "adv_blocks": [ADV_BLOCK16, ADV_BLOCK8],
                         "adv_feature_channels": list(adversary.disc.feature_channels),
                         "adv_d_weights": sum(p.numel() for p in adversary.disc.parameters()),
                         "adv_seed": args.seed + ADV_SEED_OFFSET, "adv_loss": "hinge",
                         "adv_r1_head_only": bool(args.adv_r1_head_only), "adv_experts": args.adv_experts,
                         "judged": "interaction-final.pt, the EMA weights at the last update; interaction.pt keeps the EMA "
                                   "weights with the lowest validation loss"})
    print(json.dumps({k: v for k, v in settings.items() if k != "validation_names"}), flush=True)
    wb = None
    if args.wandb:
        try:
            import wandb
            wb = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=args.run, job_type="paper1-methodA",
                            config={k: v for k, v in settings.items() if k != "validation_names"})
        except Exception as error:   # the curves still go to train.json
            print(json.dumps({"wandb": f"not logging: {error}"}), flush=True)

    val_curve, best = [], {"update": None, "validation_loss": math.inf}

    def validate(update):
        judged.eval()
        v = validation_loss(nets, judged, val)
        judged.train()
        val_curve.append({"update": update, "validation_loss": round(v, 6)})
        if v < best["validation_loss"]:
            best.update(update=update, validation_loss=v)
            save_checkpoint(folder / "interaction.pt", judged, settings, update, v)
        print(json.dumps({"update": update, "validation_loss": round(v, 6), "best_update": best["update"]}), flush=True)
        if wb is not None:
            wb.log({"validation_loss": v}, step=update)

    validate(0)                                              # delta is 0 here: the plain sum's own validation loss
    gen = torch.Generator().manual_seed(args.seed)
    losses, begin, rate, adv_curve = [], time.time(), None, []
    for update in range(1, args.updates + 1):
        idx = torch.randint(len(tr_x), (args.batch,), generator=gen)                 # positives with replacement
        sigma = draw_sigma(args.batch, gen)
        noise = torch.randn((args.batch, 3, 64, 64), generator=gen)
        if not adv:
            loss = training_loss(nets, model, tr_x[idx].to(DEV), tr_c[idx].to(DEV), sigma.to(DEV), noise.to(DEV))
        else:
            x0b, cb = tr_x[idx].to(DEV), tr_c[idx].to(DEV)
            loss, d_new = training_loss(nets, model, x0b, cb, sigma.to(DEV), noise.to(DEV), return_d=True)
            g_adv, numbers = adversary.step(update, x0b, cb, d_new)
            denoising = float(loss.detach())
            loss = loss + args.adv_weight * g_adv
            if update % ADV_CURVE_EVERY == 0:
                adv_curve.append([update, round(denoising, 6), *(round(v, 6) for v in numbers)])
                print(json.dumps({"update": update, "denoising_loss": round(denoising, 6), "disc_loss": round(numbers[0], 4),
                                  "disc_real": round(numbers[1], 4), "disc_fake": round(numbers[2], 4),
                                  "adv_term": round(numbers[3], 4)}), flush=True)
                if wb is not None:
                    wb.log({"denoising_loss": denoising, "disc_loss": numbers[0], "disc_real": numbers[1], "disc_fake": numbers[2],
                            "adv_term": numbers[3]}, step=update)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if adv:
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - ADV_EMA)
        losses.append(float(loss.detach()))
        if wb is not None and update % 10 == 0:
            wb.log({"loss": float(np.mean(losses[-10:]))}, step=update)
        if update == pdisc.TIMING_UPDATES:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            rate = (time.time() - begin) / update
            pdisc.stop_if_over(folder, {"stage": "methodA-train", "seconds_per_update": round(rate, 5),
                                        "hours_per_1000_updates": round(rate * 1000 / 3600, 4), "updates": args.updates,
                                        "already_gpu_hours": args.already_gpu_hours, "other_gpu_hours": args.other_gpu_hours,
                                        "projected_gpu_hours": pdisc.projection(rate, args.updates, args.already_gpu_hours,
                                                                                args.other_gpu_hours)},
                               stop_override=args.gpu_hours_stop)
        if update % args.val_every == 0 or update == args.updates:
            validate(update)
    seconds = time.time() - begin
    result = {"updates": args.updates, "seconds": round(seconds, 1), "seconds_per_update_first_200": rate,
              "loss_curve_every": LOSS_CURVE_EVERY,
              "loss_curve": [round(float(np.mean(losses[i:i + LOSS_CURVE_EVERY])), 6) for i in range(0, len(losses), LOSS_CURVE_EVERY)],
              "validation_curve": val_curve, "validation_loss_plain_sum": val_curve[0]["validation_loss"],
              "best_update": best["update"], "best_validation_loss": round(best["validation_loss"], 6),
              "checkpoint": str(folder / "interaction.pt"), "settings": settings, "device": ps.device_name(),
              "wandb_run": wb.url if wb is not None else None}
    if adv:
        save_checkpoint(folder / "interaction-final.pt", ema, settings, args.updates, val_curve[-1]["validation_loss"])
        save_checkpoint(folder / "interaction-final-raw.pt", model, settings, args.updates, None)
        result.update({"final_checkpoint": str(folder / "interaction-final.pt"),
                       "final_raw_checkpoint": str(folder / "interaction-final-raw.pt"),
                       "final_validation_loss": val_curve[-1]["validation_loss"],
                       "adversarial_curve_columns": ["update", "denoising loss", "disc loss", "disc mean real score",
                                                     "disc mean fake score", "delta's adversarial term"],
                       "adversarial_curve": adv_curve})
    (folder / "train.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("updates", "seconds", "best_update", "best_validation_loss", "validation_loss_plain_sum")}),
          flush=True)
    if wb is not None:
        wb.finish()


# ---------------------------------------------------------------------------------------------------------------------------------------
# sample


@torch.no_grad()
def cmd_sample(args):
    if not args.checkpoint and not args.zero_delta:
        raise SystemExit("give --checkpoint interaction.pt (or --zero-delta for the plain sum through this code)")
    if args.corrector_steps < 0 or args.corrector_size <= 0:
        raise SystemExit("--corrector-steps must be 0 or more and --corrector-size above 0")
    folder = ps.run_dir(args)
    nets, encoder = ps.load_three(args)
    if args.checkpoint:
        model, record = load_interaction(pdisc.resolve(args.root, args.checkpoint))
    else:                                                    # zero-delta with no checkpoint: an untrained delta, multiplied by 0
        model, record = InteractionNet().to(DEV).eval().requires_grad_(False), {"update": None, "width": WIDTH}
    t = ps.sigmas()
    counts = ps.parse_counts(args.counts)
    pictures, begin = 0, time.time()
    for k in counts:
        for batch in ps.batches(ps.seeds_for(args, k), args.batch):
            x, _ = ps.starting_noise(batch)
            denoise = interaction_denoiser(nets, model, k, len(batch), zero_delta=args.zero_delta)
            hook = corrector(denoise, batch, args.corrector_steps, args.corrector_size) if args.corrector_steps > 0 else None
            x = ps.heun(denoise, x, t, after_step=hook)
            ps.save(ps.to_uint8(encoder, x), folder / f"count-{k}", batch)
            pictures += len(batch)
    seconds = time.time() - begin
    ps.log_json(folder, "timing", {"stage": "methodA-sample", "zero_delta": bool(args.zero_delta), "checkpoint": args.checkpoint,
                                   "checkpoint_update": record.get("update"), "width": record.get("width"), "counts": counts,
                                   "corrector_steps": args.corrector_steps,
                                   "corrector_size": args.corrector_size if args.corrector_steps > 0 else None,
                                   "corrector_noise": (f"one CPU generator per seed, seeded {CORRECTOR_SEED_OFFSET} + seed"
                                                       if args.corrector_steps > 0 else None),
                                   "denoiser_calls_per_picture": 2 * ps.STEPS - 1 + (ps.STEPS - 1) * args.corrector_steps,
                                   "seed_choice": {"seeds": args.seeds, "split": args.split, "set": args.set, "side": args.side,
                                                   "limit": args.limit},
                                   "pictures": pictures, "seconds": round(seconds, 1),
                                   "seconds_per_picture": round(seconds / max(1, pictures), 4), "device": ps.device_name()})


# ---------------------------------------------------------------------------------------------------------------------------------------


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    def common(p, batch):
        p.add_argument("--run", required=True, help="the run name; outputs go to <root>/<run>/")
        p.add_argument("--root", default=str(ps.PAPER1_ROOT), help="where runs write (default %(default)s)")
        p.add_argument("--red-net", default=str(ps.RED_NET))
        p.add_argument("--blue-net", default=str(ps.BLUE_NET))
        p.add_argument("--empty-net", default=str(ps.EMPTY_NET))
        p.add_argument("--edm2", default=str(ps.EDM2_DIR), help="EDM2 checkout, for its pickles and seeding")
        p.add_argument("--batch", type=int, default=batch)
        return p

    p = common(sub.add_parser("train"), BATCH)
    p.add_argument("--positives", required=True, help="a pack folder of real joint scenes, holding count-<k>/<index:06d>.png")
    p.add_argument("--updates", type=int, default=UPDATES)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--seed", type=int, default=0, help="the random start and the order of examples")
    p.add_argument("--width", type=int, default=WIDTH, help="delta's base width (default %(default)s: 2,968,739 weights)")
    p.add_argument("--val-every", type=int, default=VAL_EVERY, help="updates between validation readings (default %(default)s)")
    p.add_argument("--already-gpu-hours", type=float, default=0.0, help="GPU-hours already spent, counted toward the stop")
    p.add_argument("--other-gpu-hours", type=float, default=0.0, help="GPU-hours still to come after training, counted toward the stop")
    p.add_argument("--gpu-hours-stop", type=float, default=None,
                   help="replace the first result's 4 GPU-hour cost stop for this run (a spend decision, recorded in projection)")
    p.add_argument("--no-wandb", dest="wandb", action="store_false", help="do not log to W&B")
    p.add_argument("--adv-weight", type=float, default=0.0,
                   help="the adversarial term's weight (default 0: Method A exactly, no discriminator built); see the docstring")
    p.add_argument("--adv-r1-head-only", action="store_true",
                   help="R1 with the expert features detached, through the discriminator's pixel path only (default: through the experts)")
    p.add_argument("--adv-experts", type=int, default=3, choices=(2, 3),
                   help="experts the discriminator reads: 3 red, blue and empty (default), 2 red and blue")

    p = common(sub.add_parser("sample"), ps.BATCH)
    p.add_argument("--checkpoint", help="interaction.pt from train")
    p.add_argument("--zero-delta", action="store_true", help="multiply delta by 0: the plain sum through this code (the control)")
    p.add_argument("--counts", default="1,2,3")
    p.add_argument("--seeds", help="seed ranges such as 20000-20255; or choose them from a split file below")
    p.add_argument("--split", help="split.json from paper1_sample.py split")
    p.add_argument("--set", choices=ps.SEED_SETS)
    p.add_argument("--side", choices=["touching", "apart"], default="touching")
    p.add_argument("--limit", type=int, help="the first N seeds in seed order, per count")
    p.add_argument("--corrector-steps", type=int, default=0,
                   help="Langevin corrector steps after each Heun step above sigma 0 (default 0: no corrector)")
    p.add_argument("--corrector-size", type=float, default=CORRECTOR_SIZE,
                   help="corrector step as a fraction of sigma squared (default %(default)s)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    {"train": cmd_train, "sample": cmd_sample}[args.stage](args)


if __name__ == "__main__":
    main()
