#!/usr/bin/env python3
"""Paper 1's head: a small network on the frozen red and blue experts' decoder features that tells real joint scenes from the region
sampler's own pictures, its training loop, the choice of which decoder blocks feed it, and its check.

Settings: plans/paper-1-clevr-touching/plans/03-full-plan.md, "The models" ("The main discriminator" and the settings table's training
rows: noise, positives, negatives, loss and optimiser). The table's layer and size rows describe the from-scratch ablation, not this
head. First-result sizes: plans/paper-1-clevr-touching/plans/02-first-result.md, tasks 9 and 10. Guardrails adopted before the first
result: artifacts/notes/guarding-the-first-results/paper-1.md, "The guardrails, step by step" (1a, 8) and enhancements 1, 3, 4 and 5.

One rule over every decision here: held-out real scenes may inform, never choose. The block choice reads only the 64 training scenes
and the region sampler's pictures at the tuning seeds; every held-out number is printed for reading only.

The head. Both experts run frozen on the noisy picture x at its noise level sigma, each told its own count (k red, k blue). From
each, one decoder block's output at 8 by 8 (512 channels in edm2-img64-xs) is upsampled to 16 by 16 by nearest neighbour and joined
along the channels with one decoder block's output at 16 by 16 (384 channels): red 16, red 8, blue 16, blue 8, 1,792 channels. Two
blocks of [3 by 3 convolution to 256 channels, GroupNorm with 8 groups, a scale and shift from log sigma and the count, SiLU] follow;
then the average over the 16 by 16 grid and one linear layer give a raw score, and the logit is LOGIT_BOUND * tanh(raw / LOGIT_BOUND),
so it cannot grow without limit and neither can the push taken through it. Only the head is trained.

Training additions (guardrails): every example is shifted by a random whole number of pixels, up to SHIFT_MAX each way with the edges
repeated, real and drawn alike, never mirrored (a mirror would put the shadows on the wrong side); negatives are drawn with weights
that give them the positives' touching share in each count; every GAUGE_EVERY updates the overfitting gauge is written (read only);
an optional alignment loss (off by default) pulls sigma^2 * grad logit toward the denoising target x0 - D_region(x_t).

Input layout. Every picture folder holds count-<k>/ sub-folders of 64 by 64 PNGs (k = each colour's count, 1 to 3), the layout of
paper1_sample.py's runs. Joint scenes from paper1_pack.py are named by render index (<index:06d>.png), carry touching.json beside
them, and their per-colour masks under masks/count-<k>/<index:06d>-red.png and -blue.png. Negatives are named by seed, and their
touching or apart status comes from the lone samples' split.json, seed set "negatives".

Subcommands:
  choose  for each candidate pair of decoder blocks, 4-fold cross-validation inside the 64 training scenes against the region
          sampler's pictures at the tuning seeds; the highest mean balanced accuracy wins (CHOOSE_RULE). Beside it, per sigma band, a
          logistic probe on raw noisy pixels and on each candidate's features, scored by the same folds. Writes blocks.json
  train   train the head on the chosen blocks; logs the loss and the gauge to W&B; times the first 200 updates and stops (exit 3) if
          the first result's projection passes FIRST_GPU_HOURS_STOP; writes head.pt, train.json and overfit.json
  check   balanced accuracy in the four sigma bands on all held-out positives against held-out negatives, beside training accuracy,
          and on touching-only and apart-only pairs (guardrail 7), read only (accuracy.json, accuracy.png); the depth-order
          preference (guardrail 2b, read only, depth_preference.json); then the direction check on the training scenes, with the
          held-out scenes beside it (direction.json, direction.png), which exits 4 when it fails, so the job stops before tuning

Usage (cluster, through paper1_train.sbatch run as bash on the held node; folders relative to /datasets/mmolefe/poe_repair_min/paper1/):
  python paper1_disc.py choose --run paper1-first-disc-k64 --positives paper1-first-pack/k64-s1 --negatives paper1-first-disc-k64/tuning-region
  python paper1_disc.py train  --run paper1-first-disc-k64 --positives paper1-first-pack/k64-s1 --negatives paper1-first-negatives/train \
      --negatives-split paper1-first-negatives/split.json --gauge-negatives paper1-first-negatives/heldout \
      --blocks-file paper1-first-disc-k64/blocks.json --seed 0
  python paper1_disc.py check  --run paper1-first-disc-check --head paper1-first-disc-k64/head.pt --positives paper1-first-pack/k64-s1 \
      --negatives paper1-first-negatives/train --heldout-positives paper1-first-pack/heldout --heldout-negatives paper1-first-negatives/heldout
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper1_sample as ps  # noqa: E402

# ---------------------------------------------------------------------------------------------------------------------------------------
# Settings from the full plan's table (training rows), each also a default argument below

SIGMA_LO, SIGMA_HI = 0.28, 10.5       # training noise: sigma log-uniform in [0.28, 10.5], fresh per example; covers both push windows
BATCH_POS, BATCH_NEG = 64, 64         # batch 128: 64 positives and 64 negatives, so the two classes weigh equally
LR, WEIGHT_DECAY = 2e-4, 0.01         # AdamW
UPDATES = 4000
WIDTH = 256                           # the head's channels, fixed across every k
GROUPS = 8
COND_HIDDEN = 128                     # the small network that turns (log sigma, count) into each block's scale and shift
WANDB_ENTITY, WANDB_PROJECT = "prime_lab", "poe-repair-animals-compose"

# Guardrails (artifacts/notes/guarding-the-first-results/paper-1.md)
SHIFT_MAX = 4                         # enhancement 3: whole-pixel shifts up to 4 each way, edges repeated, both classes; no mirror
LOGIT_BOUND = 4.0                     # enhancement 4: logit = 4 tanh(raw / 4); 0 turns the bound off
REWEIGHT_TOUCHING = True              # enhancement 5: negatives weighted to the positives' touching share per count
ALIGN_WEIGHT = 0.0                    # enhancement 1, off by default: weight of 1 - cos(sigma^2 grad logit, x0 - D_region(x_t))
ALIGN_WEIGHT_ON = 1.0                 # the weight the aligned rerun uses: the full plan's aligned variant, "weight 1 fixed before any
                                      # cosine is seen"; the first result trains with it when the direction check fails
ALIGN_SHARE = 0.25                    # the alignment loss reads a quarter of each batch's positives
GAUGE_EVERY = 250                     # guardrail 1a: the overfitting gauge every 250 updates
GAUGE_NEGATIVES = 256                 # held-out negatives the gauge reads, for reading only

# The four sigma bands of the check (03-full-plan.md, step 25), highest noise first
BANDS = ((10.5, 3.8), (3.8, 1.6), (1.6, 0.6), (0.6, 0.28))

# The block choice, fixed before it runs. Candidates: the middle and the last decoder block at each of the two levels.
CANDIDATES_8 = ("8x8_block1", "8x8_block3")
CANDIDATES_16 = ("16x16_block1", "16x16_block3")
CHOOSE_FOLDS = 4                      # 4-fold cross-validation inside the 64 training scenes: train on 48, read the other 16
CHOOSE_UPDATES = 250                  # each fold of each candidate trains this many updates; 4 candidates x 4 folds = one full training
CHOOSE_SEED = 0
CHOOSE_RULE = ("4-fold cross-validation inside the k training scenes (folds by render index: scene of rank r goes to fold r mod 4, so "
               "sizes stay balanced) against the region sampler's pictures at the tuning seeds, split into 4 folds by seed the same way; "
               "each fold trains a head for CHOOSE_UPDATES on the other three and reads balanced accuracy on its own, in each of the "
               "four sigma bands; the candidate with the highest mean over folds and bands wins; on an exact tie the earlier candidate "
               "in the order (8x8_block1, 16x16_block1), (8x8_block1, 16x16_block3), (8x8_block3, 16x16_block1), (8x8_block3, 16x16_block3). "
               "No held-out scene is read.")
PROBE_DRAWS = 4                       # guardrail 8: each picture is noised this many times per band for the linear probes
PROBE_STEPS = 300                     # full-batch steps of the probe's logistic regression
CHECK_REPEATS = 2                     # each picture is noised this many times per band in the check
DIRECTION_DRAWS = 8                   # the direction check: 8 noise draws per scene per band
LOW_BANDS = (2, 3)                    # the two lowest sigma bands (1.6 to 0.6 and 0.6 to 0.28), where the direction check is judged
DEPTH_SEEDS_PER_COUNT = 32            # the depth-order preference reads the first 32 touching tuning seeds per count (96)
DEPTH_DRAWS = 4                       # noise draws per pair per band for the depth-order preference
TIMING_UPDATES = 200
# The first result's cost table without step 9, at the upper end of each range: steps 2 (0.2), 3 (0.1), 8 (0.65), 10 (0.2), 11 (0.5),
# 13 (0.4), plus the guardrails' about 20 GPU-minutes (0.33): 2.38
OTHER_GPU_HOURS = 2.38

DEV = ps.DEV


# ---------------------------------------------------------------------------------------------------------------------------------------
# Frozen expert features


class _Stop(Exception):
    pass


class ExpertFeatures:
    """Decoder block outputs of one frozen EDM2 network, captured by forward hooks.

    The hooks act only inside __call__, so the same network still denoises normally in the sampler. The forward pass stops after the
    last block asked for, so the 32 by 32 and 64 by 64 decoder levels never run. Gradients flow back to x for the push.
    """

    def __init__(self, net, names):
        self.net, self.names, self.active, self.out = net, list(names), False, {}
        order = list(net.unet.dec.keys())
        missing = [n for n in self.names if n not in order]
        if missing:
            raise SystemExit(f"no decoder block {missing}; this network has {order}")
        self.last = max(self.names, key=order.index)
        for n in self.names:
            net.unet.dec[n].register_forward_hook(self._hook(n))

    def _hook(self, name):
        def hook(module, inputs, output):
            if self.active:
                self.out[name] = output
                if name == self.last:
                    raise _Stop
        return hook

    def __call__(self, x, sigma, labels):
        self.out, self.active = {}, True
        try:
            self.net(x, sigma, labels)
        except _Stop:
            pass
        finally:
            self.active = False
        return {n: self.out[n].float() for n in self.names}


def joined(red_feats, blue_feats, block8, block16):
    """red 16, red 8 upsampled, blue 16, blue 8 upsampled, along the channels: (n, 2 * (384 + 512), 16, 16) for edm2-img64-xs."""
    up = lambda f: F.interpolate(f, size=f.shape[-1] * 2, mode="nearest")
    return torch.cat([red_feats[block16], up(red_feats[block8]), blue_feats[block16], up(blue_feats[block8])], 1)


# ---------------------------------------------------------------------------------------------------------------------------------------
# The head


class Head(nn.Module):
    def __init__(self, in_channels, width=WIDTH, groups=GROUPS, cond_hidden=COND_HIDDEN, bound=LOGIT_BOUND):
        super().__init__()
        self.bound = bound
        self.cond = nn.Sequential(nn.Linear(4, cond_hidden), nn.SiLU(), nn.Linear(cond_hidden, 4 * width))
        self.conv1, self.norm1 = nn.Conv2d(in_channels, width, 3, padding=1), nn.GroupNorm(groups, width)
        self.conv2, self.norm2 = nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(groups, width)
        self.out = nn.Linear(width, 1)

    def forward(self, feats, sigma, count):
        """feats (n, C, 16, 16); sigma (n,); count (n,) each colour's count, 1 to 3. Returns the bounded logit (n,)."""
        c = torch.cat([sigma.float().log()[:, None], F.one_hot(count.long() - 1, 3).float()], 1)
        s1, b1, s2, b2 = (v[:, :, None, None] for v in self.cond(c).chunk(4, 1))
        h = F.silu(self.norm1(self.conv1(feats)) * (1 + s1) + b1)
        h = F.silu(self.norm2(self.conv2(h)) * (1 + s2) + b2)
        raw = self.out(h.mean((2, 3)))[:, 0]
        return self.bound * torch.tanh(raw / self.bound) if self.bound > 0 else raw


class Discriminator:
    """The frozen experts plus the head: logit(x, sigma, count), differentiable in x."""

    def __init__(self, red_net, blue_net, block8, block16, head=None):
        self.block8, self.block16 = block8, block16
        self.red = ExpertFeatures(red_net, (block8, block16))
        self.blue = ExpertFeatures(blue_net, (block8, block16))
        self.head = head

    def features(self, x, sigma, count):
        labels = F.one_hot(count.long() - 1, 3).float()
        s = sigma.expand(len(x)) if sigma.dim() == 0 else sigma
        return joined(self.red(x, s, labels), self.blue(x, s, labels), self.block8, self.block16)

    def logit(self, x, sigma, count):
        s = sigma.expand(len(x)) if sigma.dim() == 0 else sigma
        return self.head(self.features(x, s, count), s, count)


def weight_count(module):
    return sum(p.numel() for p in module.parameters())


def load_discriminator(path, red_net, blue_net):
    """A trained head from head.pt, wired to the two frozen experts; eval mode, no gradients on its weights."""
    ckpt = torch.load(path, map_location=DEV)
    disc = Discriminator(red_net, blue_net, ckpt["block8"], ckpt["block16"])
    disc.head = Head(ckpt["in_channels"], ckpt["width"], bound=ckpt.get("logit_bound", LOGIT_BOUND)).to(DEV)
    disc.head.load_state_dict(ckpt["head"])
    disc.head.eval().requires_grad_(False)
    return disc


# ---------------------------------------------------------------------------------------------------------------------------------------
# Data


def load_folder(folder):
    """Every PNG under folder/count-<k>/ as (pictures on -1 to 1, (n, 3, 64, 64)), counts (n,), names ("count-<k>/<file>")."""
    pics, counts, names = [], [], []
    for k in ps.COUNTS:
        sub = Path(folder) / f"count-{k}"
        if not sub.is_dir():
            continue
        for p in sorted(sub.glob("*.png")):
            pics.append(np.asarray(Image.open(p).convert("RGB")))
            counts.append(k)
            names.append(f"count-{k}/{p.name}")
    if not pics:
        raise SystemExit(f"no pictures under {folder}/count-<k>/")
    x = torch.from_numpy(np.stack(pics)).permute(0, 3, 1, 2).float() / 127.5 - 1
    return x, torch.tensor(counts), names


def file_number(name):
    """count-<k>/<number:06d>.png to the number: a render index for joint scenes, a seed for sampled pictures."""
    return int(Path(name).stem)


def load_masks(folder, names):
    """The renderer's per-colour masks of joint scenes, dilated by 1 pixel like the region sampler's: two (n, 1, 64, 64) tensors."""
    out = []
    for colour in ("red", "blue"):
        paths = [Path(folder) / "masks" / f"{n[:-4]}-{colour}.png" for n in names]
        missing = [str(q) for q in paths if not q.is_file()]
        if missing:
            raise SystemExit(f"{len(missing)} {colour} mask files missing under {folder}/masks/, first {missing[0]}; paper1_pack.py writes them")
        m = np.stack([np.asarray(Image.open(q).convert("L")) > 127 for q in paths])
        m = torch.from_numpy(m).float()[:, None]
        out.append((F.max_pool2d(m, 3, 1, 1) > 0).float())
    return out


def resolve(root, folder):
    """A folder given relative to the paper1 root, or absolute."""
    p = Path(folder)
    return p if p.is_absolute() else Path(root) / p


def noised(x, sigma, gen):
    """x = clean + sigma * noise, noise from gen."""
    return x + sigma.view(-1, 1, 1, 1) * torch.randn(x.shape, generator=gen).to(x.device)


def log_uniform(n, lo, hi, gen):
    return torch.exp(torch.empty(n).uniform_(math.log(lo), math.log(hi), generator=gen))


def shifted(x, max_shift, gen):
    """Each picture moved by its own random whole-pixel shift in [-max_shift, max_shift] each way, edges repeated (enhancement 3)."""
    if max_shift <= 0:
        return x
    n, _, h, w = x.shape
    padded = F.pad(x, (max_shift,) * 4, mode="replicate")
    dy = torch.randint(0, 2 * max_shift + 1, (n,), generator=gen).tolist()
    dx = torch.randint(0, 2 * max_shift + 1, (n,), generator=gen).tolist()
    return torch.stack([padded[i, :, dy[i]:dy[i] + h, dx[i]:dx[i] + w] for i in range(n)])


def balanced_accuracy(logits, labels):
    called = logits > 0
    pos, neg = labels > 0.5, labels < 0.5
    return 0.5 * (float((called & pos).sum()) / max(1, int(pos.sum())) + float((~called & neg).sum()) / max(1, int(neg.sum())))


def touching_weights(pos_folder, neg_names, neg_counts, split_path, set_name="negatives"):
    """Sampling weights that give the negatives the positives' touching share in each count (enhancement 5).

    Positives' shares come from <positives>/touching.json (paper1_pack.py); the negatives' touching or apart status from split.json.
    Within a count, a touching negative weighs p / q and an apart one (1 - p) / (1 - q), p and q the positives' and negatives'
    touching shares there; the counts keep their own totals. Returns (weights, report).
    """
    pos_share = {k: v["touching_share"] for k, v in json.loads((Path(pos_folder) / "touching.json").read_text())["per_count"].items()}
    split = json.loads(Path(split_path).read_text())[set_name]
    weights, report = torch.ones(len(neg_names)), {}
    for k in ps.COUNTS:
        key = f"count-{k}"
        idx = [i for i, c in enumerate(neg_counts.tolist()) if c == k]
        if not idx or key not in pos_share:
            continue
        touching = set(split[key]["touching"])
        is_t = [file_number(neg_names[i]) in touching for i in idx]
        p, q = pos_share[key], sum(is_t) / len(is_t)
        if 0 < q < 1:
            wt, wa = p / q, (1 - p) / (1 - q)
        else:
            wt = wa = 1.0
        for i, t in zip(idx, is_t):
            weights[i] = wt if t else wa
        report[key] = {"positives_touching_share": round(p, 4), "negatives_touching_share": round(q, 4),
                       "weight_touching": round(wt, 4), "weight_apart": round(wa, 4), "negatives": len(idx),
                       "note": "" if 0 < q < 1 else "negatives all touching or all apart in this count: left unweighted"}
    return weights, report


# ---------------------------------------------------------------------------------------------------------------------------------------
# The denoising target and the alignment loss (enhancement 1, off by default)


def region_target(nets3, x0, xt, sigma, count, m_red, m_blue):
    """x0 - D_region(x_t), with D_region the region sampler's combination built from the renderer's own masks."""
    labels = F.one_hot(count.long() - 1, 3).float()
    with torch.no_grad():
        d = ps.region_denoiser(nets3, labels, labels, m_red, m_blue)(xt, sigma, 0)
    return x0 - d


def alignment_loss(disc, nets3, x0, sigma, count, mask_red, mask_blue, gen):
    """Enhancement 1: mean over pictures of 1 - cos(sigma^2 grad_x logit, x0 - D_region(x_t)) at x_t = x0 + sigma * noise.

    The gradient is taken with create_graph, so the loss trains the head through it (Z11's penalty, MMDisCo's denoising loss);
    D_region is the region sampler's combination from the three frozen models with the renderer's masks, dilated by 1 pixel."""
    xt = noised(x0, sigma, gen)
    target = region_target(nets3, x0, xt, sigma, count, mask_red, mask_blue)
    push = push_direction(disc, xt, sigma, count, create_graph=True)
    return (1 - F.cosine_similarity(push.flatten(1), target.flatten(1), dim=1)).mean()


def push_direction(disc, xt, sigma, count, create_graph=False):
    """sigma^2 * grad_x logit at xt."""
    with torch.enable_grad():
        xx = xt.detach().requires_grad_(True)
        g, = torch.autograd.grad(disc.logit(xx, sigma, count).sum(), xx, create_graph=create_graph)
    return sigma.view(-1, 1, 1, 1) ** 2 * g


# ---------------------------------------------------------------------------------------------------------------------------------------
# Training


def train_head(disc, pos, neg, args, updates, seed, neg_weights=None, log=None, on_timing=None, gauge=None, align=None):
    """Train disc.head from a fresh start. pos and neg are (pictures, counts). Returns the training record.

    neg_weights: sampling weights for the negatives (None: uniform). gauge(update): called every GAUGE_EVERY updates.
    align: (nets3, masks_red, masks_blue) when the alignment loss is on; the masks index like pos.
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    in_channels = disc.features(pos[0][:1].to(DEV), torch.tensor([1.0], device=DEV), pos[1][:1].to(DEV)).shape[1]
    disc.head = Head(in_channels, args.width, bound=args.logit_bound).to(DEV).train()
    opt = torch.optim.AdamW(disc.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    begin, losses, align_losses, rate = time.time(), [], [], None
    for update in range(1, updates + 1):
        ip = torch.randint(len(pos[0]), (args.batch_pos,), generator=gen)                       # positives with replacement
        if neg_weights is None:
            ineg = torch.randint(len(neg[0]), (args.batch_neg,), generator=gen)                 # negatives with replacement
        else:
            ineg = torch.multinomial(neg_weights, args.batch_neg, replacement=True, generator=gen)
        x0 = shifted(torch.cat([pos[0][ip], neg[0][ineg]]), args.shift, gen).to(DEV)
        count = torch.cat([pos[1][ip], neg[1][ineg]]).to(DEV)
        label = torch.cat([torch.ones(args.batch_pos), torch.zeros(args.batch_neg)]).to(DEV)
        sigma = log_uniform(len(x0), args.sigma_lo, args.sigma_hi, gen).to(DEV)
        x = noised(x0, sigma, gen)
        with torch.no_grad():
            feats = disc.features(x, sigma, count)
        loss = F.binary_cross_entropy_with_logits(disc.head(feats, sigma, count), label)
        if align is not None and args.align_weight > 0:
            nets3, mr, mb = align
            na = max(1, int(args.batch_pos * ALIGN_SHARE))
            ia = ip[:na]
            xa0 = pos[0][ia].to(DEV)                               # unshifted, so the renderer's masks line up
            sa = log_uniform(na, args.sigma_lo, args.sigma_hi, gen).to(DEV)
            a_loss = alignment_loss(disc, nets3, xa0, sa, pos[1][ia].to(DEV), mr[ia].to(DEV), mb[ia].to(DEV), gen)
            loss = loss + args.align_weight * a_loss
            align_losses.append(float(a_loss.detach()))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if log is not None and update % 10 == 0:
            rec = {"loss": float(np.mean(losses[-10:])), "update": update}
            if align_losses:
                rec["align_loss"] = float(np.mean(align_losses[-10:]))
            log(rec)
        if gauge is not None and update % GAUGE_EVERY == 0:
            disc.head.eval()
            gauge(update)
            disc.head.train()
        if update == TIMING_UPDATES:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            rate = (time.time() - begin) / TIMING_UPDATES
            if on_timing is not None:
                on_timing(rate)
    return {"updates": updates, "seed": seed, "in_channels": in_channels, "head_weights": weight_count(disc.head),
            "seconds": round(time.time() - begin, 1), "seconds_per_update_first_200": rate,
            "loss_first_100": float(np.mean(losses[:100])), "loss_last_100": float(np.mean(losses[-100:])),
            "align_loss_mean": float(np.mean(align_losses)) if align_losses else None,
            "loss_mean_by_500": [round(float(np.mean(losses[i:i + 500])), 4) for i in range(0, len(losses), 500)]}


@torch.no_grad()
def band_logits(disc, pics, counts, band, repeats, gen, batch=256):
    hi, lo = band
    out = []
    for _ in range(repeats):
        for i in range(0, len(pics), batch):
            x0, c = pics[i:i + batch].to(DEV), counts[i:i + batch].to(DEV)
            sigma = log_uniform(len(x0), lo, hi, gen).to(DEV)
            out.append(disc.logit(noised(x0, sigma, gen), sigma, c).cpu())
    return torch.cat(out)


def band_accuracy(disc, pos, neg, seed, repeats=CHECK_REPEATS):
    """Balanced accuracy in each sigma band: every picture noised `repeats` times at sigma log-uniform within the band."""
    gen = torch.Generator().manual_seed(seed)
    out = []
    for band in BANDS:
        lp = band_logits(disc, pos[0], pos[1], band, repeats, gen)
        ln = band_logits(disc, neg[0], neg[1], band, repeats, gen)
        acc = balanced_accuracy(torch.cat([lp, ln]), torch.cat([torch.ones(len(lp)), torch.zeros(len(ln))]))
        out.append({"band": list(band), "balanced_accuracy": round(acc, 4), "positives": len(pos[0]) * repeats,
                    "negatives": len(neg[0]) * repeats})
    return out


def projection(seconds_per_update, updates_left, already_hours, other):
    return other + already_hours + seconds_per_update * updates_left / 3600


def stop_if_over(folder, record, aligned=False):
    from paper1_first_bars import FIRST_GPU_HOURS_STOP, FIRST_GPU_HOURS_STOP_ALIGNED
    stop = FIRST_GPU_HOURS_STOP_ALIGNED if aligned else FIRST_GPU_HOURS_STOP
    record["first_gpu_hours_stop"] = stop
    record["aligned"] = aligned
    record["over"] = record["projected_gpu_hours"] > stop
    ps.log_json(folder, "projection", record)
    if record["over"]:
        (folder / "stop.json").write_text(json.dumps(record, indent=1) + "\n")
        print(f"STOP: the first result projects {record['projected_gpu_hours']:.2f} GPU-hours, past {stop}; "
              "the owner decides", flush=True)
        raise SystemExit(3)


def load_pair(args):
    """The two frozen experts."""
    ps.use_edm2(args.edm2)
    return ps.load_net(args.red_net)[0], ps.load_net(args.blue_net)[0]


def load_nets_for(args):
    """The models a training run needs: the two experts, and the empty-scene model only when the alignment loss is on."""
    return load_three_nets(args) if args.align_weight > 0 else list(load_pair(args))


def load_three_nets(args):
    """The two experts and the empty-scene model, for the region target."""
    red, blue = load_pair(args)
    if not Path(args.empty_net).is_file():
        raise SystemExit(f"no empty-scene model at {args.empty_net}")
    return [red, blue, ps.load_net(args.empty_net)[0]]


# ---------------------------------------------------------------------------------------------------------------------------------------
# The block choice


def folds_of(names, n_folds):
    """Fold of each picture: rank by file number (render index or seed), rank mod n_folds."""
    order = sorted(range(len(names)), key=lambda i: file_number(names[i]))
    fold = [0] * len(names)
    for rank, i in enumerate(order):
        fold[i] = rank % n_folds
    return torch.tensor(fold)


def fit_probe(train_x, train_y, steps=PROBE_STEPS):
    """A logistic regression (one linear layer) on fixed vectors, classes weighted equally, AdamW with a little weight decay."""
    mean, std = train_x.mean(0, keepdim=True), train_x.std(0, keepdim=True).clamp_min(1e-6)
    lin = nn.Linear(train_x.shape[1], 1).to(DEV)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-2, weight_decay=1e-2)
    pos_w = (train_y < 0.5).sum() / (train_y > 0.5).sum().clamp_min(1)
    xs = (train_x - mean) / std
    for _ in range(steps):
        loss = F.binary_cross_entropy_with_logits(lin(xs)[:, 0], train_y, pos_weight=pos_w)
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda x: lin((x - mean) / std)[:, 0].detach()


@torch.no_grad()
def probe_vectors(disc, pics, counts, band, gen, raw, batch=256):
    """Per band, PROBE_DRAWS noisy copies of each picture: flattened raw pixels, or the joined features averaged over the grid."""
    hi, lo = band
    out = []
    for _ in range(PROBE_DRAWS):
        for i in range(0, len(pics), batch):
            x0, c = pics[i:i + batch].to(DEV), counts[i:i + batch].to(DEV)
            sigma = log_uniform(len(x0), lo, hi, gen).to(DEV)
            x = noised(x0, sigma, gen)
            out.append(x.flatten(1) if raw else disc.features(x, sigma, c).mean((2, 3)))
    return torch.cat(out)


def probes(disc_by_candidate, pos, neg, pos_fold, neg_fold):
    """Guardrail 8: per sigma band, a logistic probe on raw noisy pixels and on each candidate's features, by the choice's folds."""
    result = {}
    any_disc = next(iter(disc_by_candidate.values()))
    for band in BANDS:
        row = {}
        for name, disc, raw in [("raw pixels", any_disc, True)] + [(n, d, False) for n, d in disc_by_candidate.items()]:
            gen = torch.Generator().manual_seed(77)
            vp = probe_vectors(disc, pos[0], pos[1], band, gen, raw)
            vn = probe_vectors(disc, neg[0], neg[1], band, gen, raw)
            fp, fn = pos_fold.repeat(PROBE_DRAWS), neg_fold.repeat(PROBE_DRAWS)
            accs = []
            for f in range(CHOOSE_FOLDS):
                tx = torch.cat([vp[fp != f], vn[fn != f]]).float()
                ty = torch.cat([torch.ones(int((fp != f).sum())), torch.zeros(int((fn != f).sum()))]).to(DEV)
                with torch.enable_grad():
                    probe = fit_probe(tx, ty)
                rx = torch.cat([vp[fp == f], vn[fn == f]]).float()
                ry = torch.cat([torch.ones(int((fp == f).sum())), torch.zeros(int((fn == f).sum()))])
                accs.append(balanced_accuracy(probe(rx).cpu(), ry))
            row[name] = round(float(np.mean(accs)), 4)
        result[f"{band[0]}-{band[1]}"] = row
        print(json.dumps({"probe_band": f"{band[0]}-{band[1]}", **row}), flush=True)
    return result


def cmd_choose(args):
    folder = ps.run_dir(args)
    red, blue = load_pair(args)
    pos_data, neg_data = load_folder(resolve(args.root, args.positives)), load_folder(resolve(args.root, args.negatives))
    pos, neg = pos_data[:2], neg_data[:2]
    pos_fold, neg_fold = folds_of(pos_data[2], CHOOSE_FOLDS), folds_of(neg_data[2], CHOOSE_FOLDS)
    candidates = [(b8, b16) for b8 in CANDIDATES_8 for b16 in CANDIDATES_16]
    results, discs, begin = [], {}, time.time()

    def timing(rate):
        stop_if_over(folder, {"stage": "choose", "seconds_per_update": rate,
                              "projected_gpu_hours": projection(rate, len(candidates) * CHOOSE_FOLDS * args.choose_updates + args.updates,
                                                                0.0, args.other_gpu_hours)})
    first = True
    for i, (b8, b16) in enumerate(candidates):
        disc = Discriminator(red, blue, b8, b16)
        per_fold = []
        for f in range(CHOOSE_FOLDS):
            tp, tn = (pos[0][pos_fold != f], pos[1][pos_fold != f]), (neg[0][neg_fold != f], neg[1][neg_fold != f])
            rp, rn = (pos[0][pos_fold == f], pos[1][pos_fold == f]), (neg[0][neg_fold == f], neg[1][neg_fold == f])
            train_head(disc, tp, tn, args, args.choose_updates, CHOOSE_SEED, on_timing=timing if first else None)
            first = False
            disc.head.eval()
            per_fold.append(band_accuracy(disc, rp, rn, seed=1000 + 10 * i + f))
        bands = [{"band": list(b), "balanced_accuracy": round(float(np.mean([pf[j]["balanced_accuracy"] for pf in per_fold])), 4)}
                 for j, b in enumerate(BANDS)]
        mean = round(float(np.mean([b["balanced_accuracy"] for b in bands])), 4)
        results.append({"block8": b8, "block16": b16, "mean_balanced_accuracy": mean, "bands": bands, "folds": per_fold})
        discs[f"{b8}+{b16}"] = disc
        print(json.dumps({"candidate": [b8, b16], "mean_balanced_accuracy": mean}), flush=True)
    best = max(range(len(results)), key=lambda j: (results[j]["mean_balanced_accuracy"], -j))
    probe_result = probes(discs, pos, neg, pos_fold, neg_fold)
    out = {"rule": CHOOSE_RULE, "folds": CHOOSE_FOLDS, "choose_updates": args.choose_updates, "positives": args.positives,
           "positives_count": len(pos[0]), "negatives": args.negatives, "negatives_count": len(neg[0]), "heldout_read": 0,
           "candidates": results, "chosen": {"block8": results[best]["block8"], "block16": results[best]["block16"]},
           "linear_probes_by_band": probe_result, "seconds": round(time.time() - begin, 1), "device": ps.device_name()}
    (folder / "blocks.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"chosen": out["chosen"], "seconds": out["seconds"]}), flush=True)


# ---------------------------------------------------------------------------------------------------------------------------------------
# Training run


def cmd_train(args):
    folder = ps.run_dir(args)
    if args.blocks_file:
        blocks = json.loads(resolve(args.root, args.blocks_file).read_text())
        chose_seconds = blocks.get("seconds", 0.0)
        block8, block16 = blocks["chosen"]["block8"], blocks["chosen"]["block16"]
    else:
        block8, block16, chose_seconds = args.block8, args.block16, 0.0
    aligning = args.align_weight > 0
    nets = load_nets_for(args)
    pos_dir = resolve(args.root, args.positives)
    pos_data, neg_data = load_folder(pos_dir), load_folder(resolve(args.root, args.negatives))
    pos, neg = pos_data[:2], neg_data[:2]
    disc = Discriminator(nets[0], nets[1], block8, block16)

    neg_weights, reweight_report = None, {"reweighted": False}
    if args.reweight_touching:
        if not args.negatives_split:
            raise SystemExit("--negatives-split is required while negatives are reweighted to the positives' touching share")
        neg_weights, report = touching_weights(pos_dir, neg_data[2], neg_data[1], resolve(args.root, args.negatives_split))
        reweight_report = {"reweighted": True, "per_count": report}
        print(json.dumps({"touching_reweight": report}), flush=True)

    gauge_neg = None
    if args.gauge_negatives:
        g = load_folder(resolve(args.root, args.gauge_negatives))
        order = sorted(range(len(g[2])), key=lambda i: (g[1][i].item(), file_number(g[2][i])))
        pick = torch.tensor(order[::max(1, len(order) // GAUGE_NEGATIVES)][:GAUGE_NEGATIVES])
        gauge_neg = (g[0][pick], g[1][pick])

    settings = {k: getattr(args, k) for k in ("sigma_lo", "sigma_hi", "batch_pos", "batch_neg", "lr", "weight_decay", "updates", "width",
                                              "shift", "logit_bound", "reweight_touching", "align_weight")}
    settings.update({"mirror": False, "block8": block8, "block16": block16, "seed": args.seed, "positives": args.positives,
                     "negatives": args.negatives, "positives_count": len(pos[0]), "negatives_count": len(neg[0]),
                     "touching_reweight": reweight_report})
    print(json.dumps({"settings": settings}), flush=True)
    wb = None
    try:
        import wandb
        wb = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=args.run, config=settings, job_type="paper1-head")
    except Exception as error:   # the loss still goes to train.jsonl
        print(json.dumps({"wandb": f"not logging: {error}"}), flush=True)

    def log(rec):
        with open(folder / "train.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        if wb is not None:
            wb.log(rec, step=rec["update"])

    gauge_records, gauge_gen = [], torch.Generator().manual_seed(99)

    def gauge(update):
        """Guardrail 1a, read only: ADA's r_t = mean(sign(logit)) on the positives, mean logits, accuracy against held-out negatives."""
        sig = lambda n: log_uniform(n, args.sigma_lo, args.sigma_hi, gauge_gen).to(DEV)
        with torch.no_grad():
            xp, cp = pos[0].to(DEV), pos[1].to(DEV)
            sp = sig(len(xp))
            lp = disc.logit(noised(xp, sp, gauge_gen), sp, cp).cpu()
            xn, cn = neg[0][:len(xp)].to(DEV), neg[1][:len(xp)].to(DEV)
            sn = sig(len(xn))
            ln = disc.logit(noised(xn, sn, gauge_gen), sn, cn).cpu()
            rec = {"update": update, "r_t": round(float(lp.sign().mean()), 4), "mean_logit_positives": round(float(lp.mean()), 4),
                   "mean_logit_negatives": round(float(ln.mean()), 4)}
            if gauge_neg is not None:
                xh, ch = gauge_neg[0].to(DEV), gauge_neg[1].to(DEV)
                sh = sig(len(xh))
                lh = disc.logit(noised(xh, sh, gauge_gen), sh, ch).cpu()
                rec["balanced_accuracy_positives_vs_heldout_negatives"] = round(
                    balanced_accuracy(torch.cat([lp, lh]), torch.cat([torch.ones(len(lp)), torch.zeros(len(lh))])), 4)
        gauge_records.append(rec)
        (folder / "overfit.json").write_text(json.dumps({"read_only": True, "every": GAUGE_EVERY, "records": gauge_records}, indent=1) + "\n")
        print(json.dumps({"gauge": rec}), flush=True)
        if wb is not None:
            wb.log({f"gauge/{k}": v for k, v in rec.items() if k != "update"}, step=update)

    def timing(rate):
        already = chose_seconds / 3600 + args.already_gpu_hours
        stop_if_over(folder, {"stage": "train", "seconds_per_update": rate, "chose_hours": chose_seconds / 3600,
                              "already_gpu_hours": args.already_gpu_hours,
                              "projected_gpu_hours": projection(rate, args.updates, already, args.other_gpu_hours)}, aligned=aligning)
    align = None
    if aligning:
        mr, mb = load_masks(pos_dir, pos_data[2])
        align = (nets, mr, mb)
    record = train_head(disc, pos, neg, args, args.updates, args.seed, neg_weights=neg_weights, log=log, on_timing=timing,
                        gauge=gauge, align=align)
    print(json.dumps({"head_weights": record["head_weights"]}), flush=True)
    torch.save({"head": disc.head.state_dict(), "block8": block8, "block16": block16, "in_channels": record["in_channels"],
                "width": args.width, "logit_bound": args.logit_bound, "settings": settings}, folder / "head.pt")
    result = {"settings": settings, "training": record, "loss_falls_over_first_1000":
              record["loss_mean_by_500"][1] < record["loss_mean_by_500"][0] if len(record["loss_mean_by_500"]) > 1 else None,
              "wandb_run": wb.url if wb is not None else None, "device": ps.device_name()}
    (folder / "train.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("loss_falls_over_first_1000", "wandb_run")}), flush=True)
    if wb is not None:
        wb.finish()


# ---------------------------------------------------------------------------------------------------------------------------------------
# The check (read only)


def by_side(disc, args):
    """Guardrail 7, read only: held-out balanced accuracy per band on touching-only and on apart-only pairs.

    Positives are touching or apart by the renderer's masks (paper1_pack.py's touching.json, the 1-pixel rule); negatives by the lone
    samples' split.json, seed set "heldout-negatives". Each side pairs as many positives and negatives as both have.
    """
    pdir = resolve(args.root, args.heldout_positives)
    pics, counts, names = load_folder(pdir)
    flags = json.loads((pdir / "touching.json").read_text())["files"]
    npics, ncounts, nnames = load_folder(resolve(args.root, args.heldout_negatives))
    split = json.loads(resolve(args.root, args.heldout_negatives_split).read_text())["heldout-negatives"]
    touching_seeds = {k: set(v["touching"]) for k, v in split.items()}
    pos_t = torch.tensor([bool(flags[nm]) for nm in names])
    neg_t = torch.tensor([file_number(nm) in touching_seeds[nm.split("/")[0]] for nm in nnames])
    out = {}
    for side, pm, nm_ in (("touching", pos_t, neg_t), ("apart", ~pos_t, ~neg_t)):
        m = min(int(pm.sum()), int(nm_.sum()))
        if m == 0:
            out[side] = {"pairs": 0, "bands": [], "note": "no pictures on this side"}
            continue
        pi, ni = pm.nonzero()[:m, 0], nm_.nonzero()[:m, 0]
        out[side] = {"pairs": m, "bands": band_accuracy(disc, (pics[pi], counts[pi]), (npics[ni], ncounts[ni]), seed=21)}
    return out


def direction_cosines(disc, nets3, data, folder, seed, draws=DIRECTION_DRAWS, batch=64):
    """Per sigma band, the cosine between the push sigma^2 grad_x logit and the denoising target x0 - D_region(x_t), with D_region
    from the renderer's masks; beside it the shuffled control, each push against the next scene's target (Z11's control).
    Returns {band: {"real": [cosines], "shuffled": [cosines]}}."""
    pics, counts, names = data
    m_red, m_blue = load_masks(folder, names)
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for hi, lo in BANDS:
        real, shuffled = [], []
        for _ in range(draws):
            pushes, targets = [], []
            for i in range(0, len(pics), batch):
                x0, c = pics[i:i + batch].to(DEV), counts[i:i + batch].to(DEV)
                sigma = log_uniform(len(x0), lo, hi, gen).to(DEV)
                xt = noised(x0, sigma, gen)
                targets.append(region_target(nets3, x0, xt, sigma, c, m_red[i:i + batch].to(DEV), m_blue[i:i + batch].to(DEV)).flatten(1).cpu())
                pushes.append(push_direction(disc, xt, sigma, c).flatten(1).detach().cpu())
            p_all, t_all = torch.cat(pushes), torch.cat(targets)
            real += F.cosine_similarity(p_all, t_all, dim=1).tolist()
            shuffled += F.cosine_similarity(p_all, t_all.roll(1, 0), dim=1).tolist()
        out[f"{hi}-{lo}"] = {"real": real, "shuffled": shuffled}
    return out


def direction_check(disc, nets3, args, folder):
    """The direction check (02-first-result.md task 10): judged on the k training scenes only; held-out printed for reading.
    Fails when the training scenes' median real cosine is under FIRST_DIRECTION_COS_MIN in both of the two lowest bands."""
    from paper1_first_bars import FIRST_DIRECTION_COS_MIN
    med = lambda v: round(float(np.median(v)), 4)
    train_dir = resolve(args.root, args.positives)
    train = direction_cosines(disc, nets3, load_folder(train_dir), train_dir, seed=31)
    held_dir = resolve(args.root, args.heldout_positives)
    held = direction_cosines(disc, nets3, load_folder(held_dir), held_dir, seed=32)
    bands = [f"{hi}-{lo}" for hi, lo in BANDS]
    summary = {b: {"training_real": med(train[b]["real"]), "training_shuffled": med(train[b]["shuffled"]),
                   "heldout_real": med(held[b]["real"]), "heldout_shuffled": med(held[b]["shuffled"])} for b in bands}
    low = [bands[i] for i in LOW_BANDS]
    under = {b: summary[b]["training_real"] < FIRST_DIRECTION_COS_MIN for b in low}
    passes = not all(under.values())
    out = {"rule": f"fails when the median cosine on the training scenes is under {FIRST_DIRECTION_COS_MIN} in both of the bands {low}; "
                   "held-out cosines are for reading only", "draws_per_scene_per_band": DIRECTION_DRAWS,
           "head": args.head, "by_band": summary, "low_bands_under": under, "passes": passes,
           "next_if_fail": f"train again with --align-weight {ALIGN_WEIGHT_ON} into a new run folder and run this check again; "
                           "a second fail stops for the owner"}
    (folder / "direction.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"direction": {b: summary[b]["training_real"] for b in bands}, "passes": passes}), flush=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        for key, style in (("training_real", "o-"), ("training_shuffled", "o--"), ("heldout_real", "s-"), ("heldout_shuffled", "s--")):
            ax.plot(bands, [summary[b][key] for b in bands], style, label=key.replace("_", " "))
        ax.axhline(FIRST_DIRECTION_COS_MIN, color="grey", lw=0.8)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("sigma band (noise level, highest first)")
        ax.set_ylabel("median cosine, push against denoising target")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(folder / "direction.png", dpi=150)
    except Exception as error:
        print(json.dumps({"plot": f"not drawn: {error}"}), flush=True)
    return passes


@torch.no_grad()
def depth_preference(disc, args, folder):
    """Guardrail 2b, read only: on the first 32 touching tuning seeds per count, the head's logit on the lower-on-top paste against
    the upper-on-top paste of the same lone samples, both noised the same way, averaged over DEPTH_DRAWS draws; per band, the share
    of seeds where the lower-on-top paste scores higher (0.5 is no preference). Seeds whose two pastes are identical are left out."""
    empty = ps.load_mean_empty(resolve(args.root, args.mean_empty))
    split = json.loads(resolve(args.root, args.tuning_split).read_text())["tuning"]
    lower, upper, counts = [], [], []
    for k in ps.COUNTS:
        seeds = sorted(split[f"count-{k}"]["touching"])[:DEPTH_SEEDS_PER_COUNT]
        red = ps.load_pictures(resolve(args.root, args.red_lone) / f"count-{k}", seeds)
        blue = ps.load_pictures(resolve(args.root, args.blue_lone) / f"count-{k}", seeds)
        lo_paste, _ = ps.paste_lower_on_top(red, blue, empty, lower=True)
        up_paste, _ = ps.paste_lower_on_top(red, blue, empty, lower=False)
        lower.append(lo_paste); upper.append(up_paste); counts += [k] * len(seeds)
    to_x = lambda t: (torch.cat(t).permute(0, 3, 1, 2) * 2 - 1)
    xl, xu, cnt = to_x(lower), to_x(upper), torch.tensor(counts)
    differ = (xl - xu).flatten(1).abs().amax(1) > 0
    xl, xu, cnt = xl[differ], xu[differ], cnt[differ]
    gen = torch.Generator().manual_seed(41)
    out = {}
    for hi, lo in BANDS:
        diff = torch.zeros(len(xl))
        for _ in range(DEPTH_DRAWS):
            sigma = log_uniform(len(xl), lo, hi, gen)
            noise = torch.randn(xl.shape, generator=gen)
            for i in range(0, len(xl), 128):
                sl = slice(i, i + 128)
                s, c = sigma[sl].to(DEV), cnt[sl].to(DEV)
                a = disc.logit((xl[sl] + s.view(-1, 1, 1, 1).cpu() * noise[sl]).to(DEV), s, c).cpu()
                b = disc.logit((xu[sl] + s.view(-1, 1, 1, 1).cpu() * noise[sl]).to(DEV), s, c).cpu()
                diff[sl] += a - b
        out[f"{hi}-{lo}"] = round(float((diff > 0).float().mean()), 4) if len(diff) else None
    result = {"rule": "share of touching tuning seeds where the head scores the lower-on-top paste above the upper-on-top paste, "
                      "same noise for both, logits summed over draws; 0.5 is no preference", "seeds": int(differ.sum()),
              "seeds_with_identical_pastes_left_out": int((~differ).sum()), "draws": DEPTH_DRAWS, "share_by_band": out,
              "gates": "nothing"}
    (folder / "depth_preference.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"depth_preference": out}), flush=True)


def cmd_check(args):
    folder = ps.run_dir(args)
    nets3 = load_three_nets(args)
    red, blue = nets3[0], nets3[1]
    disc = load_discriminator(resolve(args.root, args.head), red, blue)
    sets = {name: load_folder(resolve(args.root, getattr(args, name)))[:2]
            for name in ("positives", "negatives", "heldout_positives", "heldout_negatives")}
    n = min(len(sets["heldout_positives"][0]), len(sets["heldout_negatives"][0]))
    gen = torch.Generator().manual_seed(7)
    take = lambda s, m: tuple(v[torch.randperm(len(s[0]), generator=gen)[:m]] for v in s)
    held_pos, held_neg = take(sets["heldout_positives"], n), take(sets["heldout_negatives"], n)
    train_neg = take(sets["negatives"], len(sets["positives"][0]))
    held = band_accuracy(disc, held_pos, held_neg, seed=11)
    trained = band_accuracy(disc, sets["positives"], train_neg, seed=12)
    out = {"bands": [list(b) for b in BANDS], "heldout": held, "training": trained, "heldout_positives": n, "heldout_negatives": n,
           "training_positives": len(sets["positives"][0]), "repeats": CHECK_REPEATS, "head": args.head, "gates": "nothing"}
    if args.heldout_negatives_split:
        out["heldout_by_side"] = by_side(disc, args)        # guardrail 7, read only
        print(json.dumps({"heldout_by_side": {side: [b["balanced_accuracy"] for b in v["bands"]]
                                              for side, v in out["heldout_by_side"].items()}}), flush=True)
    (folder / "accuracy.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"heldout": [b["balanced_accuracy"] for b in held], "training": [b["balanced_accuracy"] for b in trained]}), flush=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = [f"{hi}-{lo}" for hi, lo in BANDS]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(labels, [b["balanced_accuracy"] for b in held], marker="o", label=f"held-out ({n} real, {n} drawn)")
        ax.plot(labels, [b["balanced_accuracy"] for b in trained], marker="s", label=f"training ({out['training_positives']} real)")
        ax.set_ylim(0.5, 1.0)
        ax.set_xlabel("sigma band (noise level, highest first)")
        ax.set_ylabel("balanced accuracy")
        ax.legend()
        fig.tight_layout()
        fig.savefig(folder / "accuracy.png", dpi=150)
    except Exception as error:
        print(json.dumps({"plot": f"not drawn: {error}"}), flush=True)
    if args.tuning_split:
        depth_preference(disc, args, folder)
    if not direction_check(disc, nets3, args, folder):
        print("DIRECTION CHECK FAILED: the push does not point toward the change that would make the picture right; "
              f"train again with --align-weight {ALIGN_WEIGHT_ON}", flush=True)
        raise SystemExit(4)


# ---------------------------------------------------------------------------------------------------------------------------------------


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    def common(p):
        p.add_argument("--run", required=True)
        p.add_argument("--root", default=str(ps.PAPER1_ROOT))
        p.add_argument("--red-net", default=str(ps.RED_NET))
        p.add_argument("--blue-net", default=str(ps.BLUE_NET))
        p.add_argument("--empty-net", default=str(ps.EMPTY_NET))
        p.add_argument("--edm2", default=str(ps.EDM2_DIR))
        return p

    def training(p):
        p.add_argument("--positives", required=True, help="folder of count-<k>/ joint scenes at 64 pixels (the k-scene subset)")
        p.add_argument("--negatives", required=True, help="folder of count-<k>/ region-sampler pictures")
        p.add_argument("--sigma-lo", type=float, default=SIGMA_LO)
        p.add_argument("--sigma-hi", type=float, default=SIGMA_HI)
        p.add_argument("--batch-pos", type=int, default=BATCH_POS)
        p.add_argument("--batch-neg", type=int, default=BATCH_NEG)
        p.add_argument("--lr", type=float, default=LR)
        p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
        p.add_argument("--updates", type=int, default=UPDATES)
        p.add_argument("--width", type=int, default=WIDTH)
        p.add_argument("--shift", type=int, default=SHIFT_MAX, help="largest whole-pixel shift each way, both classes; 0 turns it off")
        p.add_argument("--logit-bound", type=float, default=LOGIT_BOUND, help="logit = c tanh(raw / c); 0 turns the bound off")
        p.add_argument("--align-weight", type=float, default=ALIGN_WEIGHT, help="weight of the alignment loss; 0 (default) is off")
        p.add_argument("--other-gpu-hours", type=float, default=OTHER_GPU_HOURS,
                       help="the first result's other GPU steps, for the projection against FIRST_GPU_HOURS_STOP")

    p = common(sub.add_parser("choose"))
    training(p)
    p.add_argument("--choose-updates", type=int, default=CHOOSE_UPDATES)
    p.set_defaults(reweight_touching=False)

    p = common(sub.add_parser("train"))
    training(p)
    p.add_argument("--blocks-file", help="blocks.json from choose")
    p.add_argument("--block8", default=CANDIDATES_8[-1])
    p.add_argument("--block16", default=CANDIDATES_16[-1])
    p.add_argument("--seed", type=int, default=0, help="the random start")
    p.add_argument("--negatives-split", help="the lone samples' split.json holding the seed set 'negatives'")
    p.add_argument("--no-reweight-touching", dest="reweight_touching", action="store_false",
                   help="draw negatives uniformly instead of at the positives' touching share")
    p.set_defaults(reweight_touching=REWEIGHT_TOUCHING)
    p.add_argument("--gauge-negatives", help="held-out negatives for the overfitting gauge (read only)")
    p.add_argument("--already-gpu-hours", type=float, default=0.0,
                   help="GPU-hours the first result has already spent on heads (the aligned rerun passes the first head's)")

    p = common(sub.add_parser("check"))
    p.add_argument("--head", required=True)
    p.add_argument("--positives", required=True, help="the head's own training scenes")
    p.add_argument("--negatives", required=True, help="the training negatives")
    p.add_argument("--heldout-positives", required=True)
    p.add_argument("--heldout-negatives", required=True)
    p.add_argument("--heldout-negatives-split", help="split.json holding the seed set 'heldout-negatives', for the touching and apart split")
    p.add_argument("--tuning-split", help="split.json holding the seed set 'tuning', for the depth-order preference (guardrail 2b)")
    p.add_argument("--red-lone", help="the red lone run's folder, for the depth-order preference")
    p.add_argument("--blue-lone", help="the blue lone run's folder")
    p.add_argument("--mean-empty", help="mean-empty.pt, for the depth-order preference's pastes")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    {"choose": cmd_choose, "train": cmd_train, "check": cmd_check}[args.stage](args)


if __name__ == "__main__":
    main()
