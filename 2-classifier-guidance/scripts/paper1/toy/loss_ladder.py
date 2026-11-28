#!/usr/bin/env python3
"""The loss ladder (L1 to L5): does a better training objective, or a discriminator, put more of the learned signal onto the
interaction between the two composed experts than Method A's plain denoising loss does?

Method A (squares_toy.py) is the parent: D_new = D_red + D_blue - D_empty + delta, delta trained by the EDM2 denoising loss on N joint
scenes, the update with the lowest held-out denoising loss kept. Every run below differs from that parent by one change, at the
same N, the same update budget and the same seed, and is judged on the toy's 256 shared evaluation seeds under the toy's bars:

  parent  Method A's recipe rerun at this N and update budget (the reading every L row is compared with)
  L1      per-pixel loss weights: 1 + L1_KAPPA on the contact band (pixels within one pixel of both the full red square and the full
          blue square, which covers the seam between touching squares and the whole occluded overlap), renormalised to mean 1 per
          scene, so the band's pixels count 1 + L1_KAPPA times the others
  L2      training noise levels shifted toward those where placement is decided: log sigma ~ N(L2_P_MEAN, 1.6) instead of
          N(-0.8, 1.6), so the median training sigma moves from 0.45 to 2.0
  L3      the parent's kept delta fine-tuned for L3_UPDATES updates with a measured reward on its one-step estimate D_new(x_t):
          a soft contact score (1 - exp(-sum of blue-near-red pixels)) minus L3_PURPLE times the soft purple pixel count, the
          denoising loss kept as an anchor; the reward weight is chosen on tuning seeds from L3_WEIGHTS. A control row fine-tunes
          the same way with the reward weight 0, so the reward is the only difference from it. The held-out denoising loss is
          reported before and after (a rise is the sign of reward gaming)
  L4      adversarial training: Method A's training with a discriminator beside it. The discriminator reads the frozen experts'
          inner decoder features (projected-GAN style) of a picture re-noised at a small sigma_d, learns real joint scenes against
          the delta's own current one-step estimates with the hinge loss and an R1 penalty on real pictures, and the delta is
          trained to fool it beside the denoising loss; the delta's weights are averaged (EMA) and the averaged weights are judged.
          The adversarial weight is chosen on tuning seeds from L4_WEIGHTS
  L5      discriminator guidance (Kim et al., arXiv 2211.17091) at sampling time: D' = D_base + w sigma^2 grad_x logit(x, sigma),
          logit a discriminator between real joint scenes and a fake set, both noised to the same sampler level, BCE, bounded
          logit (Kim et al.'s clip at 1e-5 from 0 and 1), guidance off below sigma L5_T_MID. w chosen on tuning seeds

The fake set for L5 is a switch (the caveats table in
2-classifier-guidance/artifacts/drips/read-refining-generative-process-discriminator/READ_MAP.md, line 116 on):
  product-exact    drawn exactly from the product of the noised experts at each level. The red and blue experts read disjoint
                   channels on a black background, so p_red^s p_blue^s / p_empty^s is the equal-weight mixture of N(r + b - e,
                   s^2 I) over every red picture r and blue picture b: the noised plain-sum picture set. The best discriminator's
                   s^2 grad logit is then D_joint - D_plain_sum exactly, the interaction term with no leftover (caveat C1). Its
                   guidance is added to the neural plain sum. This is the main L5 row
  product-mcmc     the same product reached by MCMC: the neural plain sum's sampler with L5_MCMC_STEPS Langevin steps (step 0.05
                   sigma^2) at every level, its state at each level kept (caveat C2's route on SDXL)
  finished-noised  the paper's fake set: the neural plain sum sampler's finished pictures noised forward (leaves a leftover, C1)
  rounds on A      on top of Method A: round r's fake set is the current sampler's (A, then A guided by the earlier rounds) states
                   after MCMC at each level from fresh seeds; a new discriminator each round; its guidance added on top

Caveats and the control or knob each gets here (C3, C4 and C8 cannot arise on the toy, which says so in results):
  C1  fake-set kind is the switch above; the correction error against the exact interaction per sigma band is reported per row
  C2  L5_MCMC_STEPS (MCMC steps per level) for the product-mcmc and on-A fake sets
  C3  real and fake pictures are drawn by the same renderer with the same colours, so no style gap exists to learn
  C4  the real set is the enumerated joint set, which contains no failures
  C5  the training sigma is drawn uniformly over the sampler's 32 levels and shared by the real and fake halves, so sigma carries
      no label; the correction error is reported per band, high noise first
  C6  every row is judged by pixel metrics the discriminator never sees; the discriminator's own mean logit on the final samples is
      reported beside them
  C7  w swept over L5_WEIGHTS on the tuning seeds, every w's tuning share reported
  C9  correction error per band (mean squared error, size of the true correction, mean cosine) beside the training loss
  C10 guidance window: on from sigma 80 down to L5_T_MID

Bookkeeping. Nothing existing is overwritten: each run stops before any compute if its results key or any of its files exists.
Results go to <root>/results-loss-ladder.json (toy) or <root>/results.json (harder toy), checkpoints, samples and pictures to
<root>/loss-ladder/. Each run writes its prediction check against the ranges recorded in artifacts/notes/progress/setup/supervisor.md,
entry 2026-11-28 04:27 UTC, and the corrector row (rung 6's Langevin corrector, 20 steps per level, step 0.05 sigma^2).

  python loss_ladder.py ladder --world toy --runs parent,L1,L2,L3,L4,L5
  CUDA_VISIBLE_DEVICES= python loss_ladder.py ladder --world toy --smoke     # a minutes-long CPU check into the scratch folder
"""
import argparse
import copy
import fcntl
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("SQUARES_TOY_SRC", str(HERE)))
import squares_toy as st  # noqa: E402

DEV = st.DEV
TOY_ROOT = Path(os.environ.get("LADDER_TOY_ROOT", "/datasets/mmolefe/poe_repair_min/toys/paper1"))

LADDER_N = 256
LADDER_UPDATES = 5000                # Method A's recipe at N = 256 (rung 2b): its kept update was 1,500 of 5,000
TUNING_SEEDS = list(range(10000, 10256))   # weight choices; disjoint from evaluation (0 to 255), drawn and corrector seeds
FAKE_SEED_BASE = 300000              # L5 fake pools; each pool its own block of 4,096
CORR_STEPS = 20                      # rung 6's judged corrector setting

L1_KAPPA = 4.0
L2_P_MEAN = 0.7
L3_UPDATES, L3_LR, L3_WEIGHTS, L3_PURPLE, L3_TAU = 2000, 1e-4, (0.01, 0.1), 0.25, 0.1
L4_WEIGHTS, L4_R1, L4_R1_EVERY, L4_EMA, L4_SIGMA_D, L4_D_LR = (0.003, 0.03), 1.0, 4, 0.999, (0.05, 1.0), 2e-4
L5_UPDATES, L5_LR, L5_BATCH, L5_WEIGHTS = 5000, 2e-4, 256, (0.5, 1.0, 2.0)
L5_BOUND, L5_T_MID, L5_MCMC_STEPS, L5_ROUNDS, L5_POOL_SEEDS = math.log((1 - 1e-5) / 1e-5), 0.01, 20, 2, 1024
L5_KINDS = ("product-exact", "product-mcmc", "finished-noised", "rounds-on-A")

# artifacts/notes/progress/setup/supervisor.md, 2026-11-28 04:27 UTC, at 256 scenes with A at 0.609: (kind, low, high).
# "share" checks the run's clean touching share; "gain" checks its gain over the parent's share.
PREDICTIONS = {
    "L1": ("share", 0.65, 0.80, "rises, 0.65 to 0.80"),
    "L2": ("share", 0.62, 0.75, "rises a little, 0.62 to 0.75"),
    "L3": ("share", 0.65, 0.85, "0.65 to 0.85, with its held-out denoising loss rising"),
    "L4": ("share", 0.60, 0.80, "0.60 to 0.80"),
    "L5": ("gain", 0.0, 0.10, "adds 0 to 0.10 on top of A"),
}
PARENT_AT_PREDICTION = 0.609

SMOKE = False
R1_CHUNK = None      # --r1-chunk: L4's R1 term on sub-batches of this many real pictures (same gradient, less memory); None whole batch


# -------------------------------------------------------------------------------------------------------------------------------------
# The world a ladder runs in: the existing toy, or the harder toy (harder_toy.make_world)


def square_masks(meta, h, side):
    """Full red and full blue square masks (occluded pixels included) for each joint scene, from its metadata row."""
    n = len(meta)
    red = torch.zeros(n, h, h, dtype=torch.bool)
    blue = torch.zeros(n, h, h, dtype=torch.bool)
    for i, (rr, rc, br, bc) in enumerate(meta[:, :4].tolist()):
        red[i, rr:rr + side, rc:rc + side] = True
        blue[i, br:br + side, bc:bc + side] = True
    return red, blue


def contact_band(meta, h, side):
    red, blue = square_masks(meta, h, side)
    near = lambda m: F.max_pool2d(m.float()[:, None], 3, 1, 1)[:, 0] > 0  # noqa: E731
    return near(red) & near(blue)


def toy_world():
    root = TOY_ROOT
    st.OUT = root
    st.RESULTS = root
    (e, r, b, j, meta), ex = st.sets_and_exact()
    tr, va = st.joint_split()
    nets = st.load_experts()
    w = SimpleNamespace(name="toy", h=st.H, side=st.SIDE, e=e, r=r, b=b, j=j, meta=meta, ex=ex, tr=tr, va=va, nets=nets,
                        root=root, results=root / "results-loss-ladder.json", ckpt=root / "loss-ladder",
                        bars=st.BARS, verdict=st.verdict, grid_scale=8, eval_seeds=st.EVAL_SEEDS, tuning_seeds=TUNING_SEEDS,
                        base_x=st.load_samples("neural-plainsum"))
    return w


def finish_world(w):
    """Smoke settings, the neural plain sum's reading on the evaluation seeds, and the exact joint row for pictures."""
    if SMOKE:
        scratch = Path(os.environ.get("LADDER_SMOKE_DIR", "/tmp/loss-ladder-smoke"))
        w.results, w.ckpt = scratch / "results.json", scratch / "loss-ladder"
        w.eval_seeds, w.tuning_seeds = w.eval_seeds[:8], w.tuning_seeds[:8]
    w.base_x = w.base_x[:len(w.eval_seeds)]
    w.base = st.metrics(w.base_x, w.j)["clean_share"]
    w.ckpt.mkdir(parents=True, exist_ok=True)
    return w


def make_world(name):
    if name == "toy":
        return finish_world(toy_world())
    import harder_toy
    return finish_world(harder_toy.make_world())


# -------------------------------------------------------------------------------------------------------------------------------------
# Bookkeeping


def load_results(w):
    return json.loads(w.results.read_text()) if w.results.exists() else {}


def save_result(w, key, value):
    """Read, add one key, write, under an exclusive lock (squares_toy.save_results' pattern), into this ladder's own file."""
    w.results.parent.mkdir(parents=True, exist_ok=True)
    with open(w.results.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        res = load_results(w)
        res[key] = value
        res["bars"] = w.bars
        res["device"] = str(DEV) + (f" ({torch.cuda.get_device_name()})" if DEV.type == "cuda" else "")
        tmp = w.results.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(res, indent=1))
        tmp.replace(w.results)


def require_new(w, key, paths):
    if key in load_results(w):
        raise SystemExit(f"{w.results} already has key {key!r}; pick a new key")
    taken = [str(p) for p in paths if Path(p).exists()]
    if taken:
        raise SystemExit("these files already exist: " + ", ".join(taken))


def files_of(w, key, names):
    return {n: w.ckpt / f"{key}-{n}" for n in names}


# -------------------------------------------------------------------------------------------------------------------------------------
# Training: Method A's train_delta with each rung's one change as an option. With no option set the operations are train_delta's.


def make_val_loss(w):
    """Method A's held-out denoising loss: every validation scene twice, sigma and noise fixed by generator 12345."""
    g = torch.Generator().manual_seed(12345)
    V = w.j[w.va]
    v0 = V.repeat(2, 1, 1, 1)
    vs = (torch.randn(len(v0), generator=g) * st.P_STD + st.P_MEAN).exp()
    vn = torch.randn(v0.shape, generator=g)

    @torch.no_grad()
    def val_loss(model, by_band=False):
        tot, bands = 0.0, {}
        for i in range(0, len(v0), 512):
            x0, s, n = v0[i:i + 512].to(DEV), vs[i:i + 512].to(DEV), vn[i:i + 512].to(DEV)
            x = x0 + s.view(-1, 1, 1, 1) * n
            d_r, d_b, d_e, _ = st.experts_out(w.nets, x, s)
            D = st.plainsum_of(d_r, d_b, d_e) + model(x, s, d_r, d_b, d_e)
            per = (st.loss_weight(s.view(-1, 1, 1, 1)) * (D - x0) ** 2).mean((1, 2, 3))
            tot += float(per.sum())
            if by_band:
                for hi, lo in st.SIGMA_BANDS:
                    sel = (s <= hi) & (s > lo)
                    a = bands.setdefault(f"{hi}-{lo}", [0.0, 0])
                    a[0] += float(per[sel].sum())
                    a[1] += int(sel.sum())
        if by_band:
            return tot / len(v0), {k: (v[0] / v[1] if v[1] else None) for k, v in bands.items()}
        return tot / len(v0)
    return val_loss


def soft_reward(D):
    """(soft contact score, soft purple pixel count) per picture, on the toy's pixel rules with a temperature L3_TAU."""
    R, B = D[:, 0], D[:, 2]
    hiR, hiB = torch.sigmoid((R - st.PURPLE_T) / L3_TAU), torch.sigmoid((B - st.PURPLE_T) / L3_TAU)
    red, blue, purple = hiR * (1 - hiB), hiB * (1 - hiR), hiR * hiB
    near_red = F.max_pool2d(red[:, None], 3, 1, 1)[:, 0]
    contact = 1 - torch.exp(-(near_red * blue).flatten(1).sum(1))
    return contact, purple.flatten(1).sum(1)


class FeatDisc(nn.Module):
    """A discriminator that reads c_in(s) y and the three frozen experts' inner decoder features at (y, s), projected-GAN style:
    at full, half and quarter resolution the red, blue and empty experts' maps are concatenated, projected by a 1 by 1 convolution,
    and concatenated into a small encoder conditioned on s. Output one raw score per picture, bounded by bound tanh(raw / bound)
    when bound is given. The experts are frozen; gradients flow through them to the picture."""

    def __init__(self, nets, w=48, proj=32, expert_w=64, bound=None):
        super().__init__()
        self._nets = list(nets)
        self.bound = bound
        self.cond = nn.Sequential(nn.Linear(st.FOURIER, st.EMB), nn.SiLU(), nn.Linear(st.EMB, st.EMB), nn.SiLU())
        self.p_hi = nn.Conv2d(3 * expert_w, proj, 1)
        self.p_mid = nn.Conv2d(3 * 2 * expert_w, proj, 1)
        self.p_lo = nn.Conv2d(3 * 2 * expert_w, proj, 1)
        self.conv_in = nn.Conv2d(3, w, 3, padding=1)
        self.b1, self.b2, self.b3 = st.Block(w + proj, w), st.Block(w + proj, 2 * w), st.Block(2 * w + proj, 2 * w)
        self.lin = nn.Linear(2 * w, 1)

    def forward(self, y, s):
        s = torch.as_tensor(s, device=y.device, dtype=torch.float32).reshape(-1).expand(len(y))
        fs = [n(y, s, feats=True)[1] for n in self._nets]
        f_lo, f_mid, f_hi = (torch.cat([f[k] for f in fs], 1) for k in range(3))
        e = self.cond(st.fourier(s))
        h = self.b1(torch.cat([self.conv_in(st.c_in(s.view(-1, 1, 1, 1)) * y), self.p_hi(f_hi)], 1), e)
        h = self.b2(torch.cat([F.avg_pool2d(h, 2), self.p_mid(f_mid)], 1), e)
        h = self.b3(torch.cat([F.avg_pool2d(h, 2), self.p_lo(f_lo)], 1), e)
        raw = self.lin(h.mean((2, 3)))[:, 0]
        return raw if self.bound is None else self.bound * torch.tanh(raw / self.bound)


def train_delta_x(w, idx, updates, seed=0, lr=5e-4, batch=256, val_every=250, pixel_w=None, p_mean=None, init=None,
                  reward_weight=None, adv_weight=None, log="", r1_chunk=None):
    """Method A's train_delta (same seed, optimiser, batch, held-out rule) with one option set:
      pixel_w        (len(j), 1, H, H) per-pixel weights on the squared error (L1)
      p_mean         the training log-sigma mean (L2)
      init           a state dict to start from (L3)
      reward_weight  the soft reward's weight, the denoising loss kept (L3)
      adv_weight     the adversarial term's weight beside the denoising loss, with a FeatDisc trained alongside and EMA weights (L4)
    Returns (states {"best", "final"} (EMA weights when adv_weight is set, plus "raw_final"), info)."""
    red, blue, empty = w.nets
    torch.manual_seed(seed)
    model = st.Delta().to(DEV)
    if init is not None:
        model.load_state_dict(init)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    X = w.j[idx].to(DEV)
    PW = pixel_w[idx].to(DEV) if pixel_w is not None else None
    val_loss = make_val_loss(w)
    adv = adv_weight is not None
    ema = copy.deepcopy(model).requires_grad_(False) if adv else None
    judged = ema if adv else model
    if adv:
        disc = FeatDisc(w.nets).to(DEV)
        opt_d = torch.optim.Adam(disc.parameters(), lr=L4_D_LR, betas=(0.0, 0.99))
        lo, hi = math.log(L4_SIGMA_D[0]), math.log(L4_SIGMA_D[1])
        adv_curve = []
    best = (val_loss(judged), 0, {k: v.clone() for k, v in judged.state_dict().items()})
    curve = [(0, None, best[0])]
    extra = []
    for u in range(1, updates + 1):
        k = torch.randint(len(X), (batch,), device=DEV)
        x0 = X[k]
        s = (torch.randn(batch, device=DEV) * st.P_STD + (st.P_MEAN if p_mean is None else p_mean)).exp()
        x = x0 + s.view(-1, 1, 1, 1) * torch.randn_like(x0)
        with torch.no_grad():
            d_r, d_b, d_e = red(x, s), blue(x, s), empty(x, s)
        D = st.plainsum_of(d_r, d_b, d_e) + model(x, s, d_r, d_b, d_e)
        logvar = model.net.logvar(s)
        if PW is None:
            loss = st.edm_loss(D, x0, s, logvar)
        else:
            lw = st.loss_weight(s.view(-1, 1, 1, 1))
            loss = (lw / logvar.exp() * PW[k] * (D - x0) ** 2 + logvar).mean()
        if reward_weight is not None:
            contact, purple = soft_reward(D)
            rew = contact - L3_PURPLE * purple
            loss = loss - reward_weight * rew.mean()
            if u % 250 == 0:
                extra.append((u, float(contact.mean()), float(purple.mean())))
        if adv:
            sd = torch.exp(torch.empty(batch, device=DEV).uniform_(lo, hi))
            n_r, n_f = torch.randn_like(x0), torch.randn_like(x0)
            y_r = x0 + sd.view(-1, 1, 1, 1) * n_r
            y_f = D.detach() + sd.view(-1, 1, 1, 1) * n_f
            disc.requires_grad_(True)
            r1 = u % L4_R1_EVERY == 0
            if r1 and r1_chunk:
                # The same R1 term computed on sub-batches of r1_chunk real pictures, each backward accumulated into the same
                # gradient, so the result equals the whole-batch term while the double backward through the experts holds only
                # one sub-batch in memory (the harder toy's 32 by 32 batch does not fit otherwise).
                l_r, l_f = disc(y_r, sd), disc(y_f, sd)
                loss_d = F.relu(1 - l_r).mean() + F.relu(1 + l_f).mean()
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                pen_total = 0.0
                for c0 in range(0, batch, r1_chunk):
                    yc = y_r[c0:c0 + r1_chunk].detach().requires_grad_(True)
                    lc = disc(yc, sd[c0:c0 + r1_chunk])
                    g, = torch.autograd.grad(lc.sum(), yc, create_graph=True)
                    pen = L4_R1_EVERY * 0.5 * L4_R1 * g.pow(2).flatten(1).sum(1).sum() / batch
                    pen.backward()
                    pen_total += float(pen.detach())
                loss_d = loss_d.detach() + pen_total
                opt_d.step()
            else:
                if r1:
                    y_r.requires_grad_(True)
                l_r, l_f = disc(y_r, sd), disc(y_f, sd)
                loss_d = F.relu(1 - l_r).mean() + F.relu(1 + l_f).mean()
                if r1:
                    g, = torch.autograd.grad(l_r.sum(), y_r, create_graph=True)
                    loss_d = loss_d + L4_R1_EVERY * 0.5 * L4_R1 * g.pow(2).flatten(1).sum(1).mean()
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()
            disc.requires_grad_(False)
            g_adv = -disc(D + sd.view(-1, 1, 1, 1) * n_f, sd).mean()
            loss = loss + adv_weight * g_adv
            if u % 250 == 0:
                adv_curve.append((u, float(loss_d.detach()), float(l_r.detach().mean()), float(l_f.detach().mean()),
                                  float(g_adv.detach())))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if adv:
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - L4_EMA)
        if u % val_every == 0 or u == updates:
            vl = val_loss(judged)
            curve.append((u, float(loss.detach()), vl))
            if u % 1000 == 0:
                print(f"  {log} update {u} loss {float(loss.detach()):.4f} val {vl:.4f}", flush=True)
            if vl < best[0]:
                best = (vl, u, {kk: v.clone() for kk, v in judged.state_dict().items()})
    states = {"best": best[2], "final": {kk: v.clone() for kk, v in judged.state_dict().items()}}
    if adv:
        states["raw_final"] = {kk: v.clone() for kk, v in model.state_dict().items()}
    info = {"best_update": best[1], "best_val_loss": best[0], "final_val_loss": curve[-1][2], "curve": curve}
    if extra:
        info["reward_curve (update, soft contact, soft purple pixels)"] = extra
    if adv:
        info["adversarial_curve (update, disc loss, disc real, disc fake, generator term)"] = adv_curve
    return states, info


def delta_of(state):
    m = st.Delta().to(DEV)
    m.load_state_dict(state)
    return m.eval().requires_grad_(False)


# -------------------------------------------------------------------------------------------------------------------------------------
# Evaluation


def a_denoiser(w, delta):
    return st.interaction_denoiser(*w.nets, delta)


def draw(denoise, seeds):
    with torch.no_grad():
        return st.sample(denoise, seeds=seeds)


def evaluate(w, denoise, idx, corrector=True):
    """(row, samples, corrected samples or None): the toy's metrics on the evaluation seeds plus picture extras and the verdict,
    and the same with rung 6's corrector at CORR_STEPS steps per level."""
    t0 = time.time()
    x = draw(denoise, w.eval_seeds)
    m = st.metrics(x, w.j, train_idx=idx)
    m.update(st.picture_extras(x))
    m["verdict"] = w.verdict(m["clean_share"], w.base)
    m["sampling_seconds"] = round(time.time() - t0, 1)
    xc = None
    if corrector:
        t0 = time.time()
        with torch.no_grad():
            xc = st.sample_corrected(denoise, 2 if SMOKE else CORR_STEPS, seeds=w.eval_seeds)
        mc = st.metrics(xc, w.j, train_idx=idx)
        mc.update(st.picture_extras(xc))
        mc["verdict"] = w.verdict(mc["clean_share"], w.base)
        mc["sampling_seconds"] = round(time.time() - t0, 1)
        mc["corrector"] = f"{CORR_STEPS} Langevin steps per level, step {st.CORRECTOR_SIZE} sigma^2, same seeds"
        m["with_corrector_20_steps"] = mc
    return m, x, xc


def tuning_share(w, denoise):
    return st.metrics(draw(denoise, w.tuning_seeds), w.j)["clean_share"]


def correction_error(w, cfn):
    """Per sigma band, on noised held-out joint scenes: the learned correction cfn(x, s) against the exact interaction
    D_joint - D_plain_sum (exact denoisers): mean squared error, mean square of the exact interaction, mean cosine per picture."""
    Ee, Er, Eb, Ej = w.ex
    out = {}
    for band in st.SIGMA_BANDS:
        x, s = st.band_draws(w.j[w.va], band, n=256 if SMOKE else st.BAND_DRAWS)
        a, b = [], []
        for i in range(0, len(x), 256):
            xx, ss = x[i:i + 256], s[i:i + 256]
            a.append(cfn(xx, ss).detach())
            with torch.no_grad():
                b.append(Ej(xx, ss) - st.plainsum_of(Er(xx, ss), Eb(xx, ss), Ee(xx, ss)))
        a, b = torch.cat(a), torch.cat(b)
        cos = F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1)
        out[f"{band[0]}-{band[1]}"] = {"mse": float(((a - b) ** 2).mean()), "size_of_exact": float((b ** 2).mean()),
                                       "mean_cosine": float(cos.mean())}
    return out


def delta_correction(w, delta):
    def cfn(x, s):
        with torch.no_grad():
            d_r, d_b, d_e, _ = st.experts_out(w.nets, x, s)
            return delta(x, s, d_r, d_b, d_e)
    return cfn


def prediction_check(rung, clean, parent_clean):
    kind, lo, hi, text = PREDICTIONS[rung]
    val = clean if kind == "share" else clean - parent_clean
    return {"predicted": text, "checked_on": "clean touching share" if kind == "share" else "gain over the parent",
            "value": round(val, 4), "range": [lo, hi], "held": bool(lo <= val <= hi),
            "note": f"the range was written with the parent at {PARENT_AT_PREDICTION}; this run's parent reads {parent_clean:.3f}"}


def picture(w, rows, labels, path):
    xs = [st.load_samples("exact-joint")[:len(w.eval_seeds)] if w.name == "toy" else w.exact_joint_x, w.base_x] + rows
    st.grid(xs, ["exact joint", "neural plain sum"] + labels, path, scale=w.grid_scale)


def parent_row(w, n, updates):
    res = load_results(w)
    key = f"{w.name}-N{n}-u{updates}-parent-A"
    if key not in res:
        raise SystemExit(f"run the parent first: {key} is not in {w.results}")
    return key, res[key], torch.load(w.ckpt / f"{key}-best.pt", map_location=DEV)


# -------------------------------------------------------------------------------------------------------------------------------------
# The runs


def run_trained(w, n, updates, rung, label, **opts):
    """parent, L1, L2: one training at Method A's recipe with the rung's option; the kept (lowest held-out loss) delta is judged,
    the final update's delta reported beside it."""
    t0 = time.time()
    key = f"{w.name}-N{n}-u{updates}-{rung}"
    fs = files_of(w, key, ["best.pt", "final.pt", "samples.pt", "samples-corrected.pt", "final-samples.pt"])
    pic = w.ckpt / f"{key}.png"
    require_new(w, key, list(fs.values()) + [pic])
    idx = w.tr[:n]
    states, info = train_delta_x(w, idx, updates, log=f"{rung} N={n}", **opts)
    train_s = time.time() - t0
    torch.save(states["best"], fs["best.pt"])
    torch.save(states["final"], fs["final.pt"])
    delta = delta_of(states["best"])
    row, x, xc = evaluate(w, a_denoiser(w, delta), idx)
    torch.save(x.cpu(), fs["samples.pt"])
    torch.save(xc.cpu(), fs["samples-corrected.pt"])
    row_final, xf, _ = evaluate(w, a_denoiser(w, delta_of(states["final"])), idx, corrector=True)
    torch.save(xf.cpu(), fs["final-samples.pt"])
    out = {"rung": rung, "change_from_method_a": label, "n_train_scenes": len(idx), "updates": updates, "train": info,
           "judged_row (kept delta, lowest held-out loss)": row, "final_update_row (reported, not judged)": row_final,
           "correction_error_vs_exact_by_band": correction_error(w, delta_correction(w, delta)),
           "files": {k: str(v) for k, v in fs.items()}, "picture": str(pic), "train_seconds": round(train_s, 1)}
    if rung != "parent-A":
        pkey, prow, _ = parent_row(w, n, updates)
        pc = prow["judged_row (kept delta, lowest held-out loss)"]["clean_share"]
        out["parent"] = {"key": pkey, "clean_share": pc}
        out["prediction_check"] = prediction_check(rung.split("-")[0], row["clean_share"], pc)
    elif w.name == "toy":
        out["earlier_reading_rung2b_N256_u5000"] = 0.609
    picture(w, [x, xc, xf], [f"{rung} kept", f"{rung} kept +corr", f"{rung} final"], pic)
    out["gpu_seconds"] = round(time.time() - t0, 1)
    save_result(w, key, out)
    print(f"{key}: clean {row['clean_share']:.3f} {row['verdict']} purple seeds {row['share_with_purple']:.3f} "
          f"empty {row['share_empty']:.3f}; with corrector {row['with_corrector_20_steps']['clean_share']:.3f}", flush=True)
    return out


def pixel_weights(w):
    band = contact_band(w.meta, w.h, w.side).float()
    pw = 1 + L1_KAPPA * band
    return (pw / pw.mean((1, 2), keepdim=True))[:, None]


def run_L3(w, n, updates):
    """Fine-tune the parent's kept delta with the reward at each weight in L3_WEIGHTS and with weight 0 (the control); the weight
    is chosen on tuning seeds; held-out denoising loss before and after."""
    t0 = time.time()
    key = f"{w.name}-N{n}-u{updates}-L3"
    lams = (0.0,) + L3_WEIGHTS
    fs = files_of(w, key, [f"lam{l}.pt" for l in lams] + [f"lam{l}-samples.pt" for l in lams] + ["samples-corrected.pt"])
    pic = w.ckpt / f"{key}.png"
    require_new(w, key, list(fs.values()) + [pic])
    pkey, prow, pstate = parent_row(w, n, updates)
    pc = prow["judged_row (kept delta, lowest held-out loss)"]["clean_share"]
    idx = w.tr[:n]
    val_loss = make_val_loss(w)
    before = val_loss(delta_of(pstate), by_band=True)
    runs, xs, labels, xcs = {}, [], [], {}
    upd = 20 if SMOKE else L3_UPDATES
    for lam in lams:
        states, info = train_delta_x(w, idx, upd, lr=L3_LR, init=pstate, reward_weight=lam, log=f"L3 lam={lam}")
        torch.save(states["final"], fs[f"lam{lam}.pt"])
        delta = delta_of(states["final"])
        den = a_denoiser(w, delta)
        row, x, xcl = evaluate(w, den, idx, corrector=True)
        torch.save(x.cpu(), fs[f"lam{lam}-samples.pt"])
        xcs[lam] = xcl
        after = val_loss(delta, by_band=True)
        runs[lam] = {"train": info, "row": row, "tuning_clean_share": tuning_share(w, den),
                     "heldout_denoising_loss_after": after[0], "heldout_by_band_after": after[1]}
        xs.append(x)
        labels.append(f"L3 lam={lam}")
        print(f"L3 lam={lam}: tuning {runs[lam]['tuning_clean_share']:.3f} eval {row['clean_share']:.3f} "
              f"held-out loss {before[0]:.4f} -> {after[0]:.4f}", flush=True)
    chosen = max(L3_WEIGHTS, key=lambda l: (runs[l]["tuning_clean_share"], -l))
    delta = delta_of(torch.load(fs[f"lam{chosen}.pt"], map_location=DEV))
    xc = xcs[chosen]
    torch.save(xc.cpu(), fs["samples-corrected.pt"])
    judged = runs[chosen]["row"]
    mc = judged["with_corrector_20_steps"]
    out = {"rung": "L3", "change_from_method_a": f"the parent's kept delta fine-tuned {upd} updates at learning rate {L3_LR} with "
           f"reward (soft contact - {L3_PURPLE} soft purple pixels, temperature {L3_TAU}) at weight lam, denoising loss kept; "
           "lam = 0 is the control (fine-tuned the same way with no reward)",
           "parent": {"key": pkey, "clean_share": pc}, "lam_chosen_on_tuning_seeds": chosen,
           "judged_row": judged, "control_row_lam0": runs[0.0]["row"],
           "heldout_denoising_loss_before": before[0], "heldout_by_band_before": before[1],
           "runs": {str(k): v for k, v in runs.items()},
           "correction_error_vs_exact_by_band": correction_error(w, delta_correction(w, delta)),
           "prediction_check": prediction_check("L3", judged["clean_share"], pc),
           "reward_gaming_check": {"heldout_loss_before": before[0], "heldout_loss_after": runs[chosen]["heldout_denoising_loss_after"],
                                   "rose": bool(runs[chosen]["heldout_denoising_loss_after"] > before[0]),
                                   "control_after": runs[0.0]["heldout_denoising_loss_after"]},
           "files": {k: str(v) for k, v in fs.items()}, "picture": str(pic)}
    picture(w, xs + [xc], labels + [f"L3 lam={chosen} +corr"], pic)
    out["gpu_seconds"] = round(time.time() - t0, 1)
    save_result(w, key, out)
    print(f"{key}: chosen lam {chosen}, clean {judged['clean_share']:.3f} {judged['verdict']}, with corrector "
          f"{mc['clean_share']:.3f}", flush=True)
    return out


def run_L4(w, n, updates):
    """Method A's training with the adversarial term at each weight in L4_WEIGHTS; EMA weights judged (the kept and the final);
    the weight chosen on tuning seeds with the final EMA delta."""
    t0 = time.time()
    key = f"{w.name}-N{n}-u{updates}-L4"
    fs = files_of(w, key, [f"lam{l}-{k}.pt" for l in L4_WEIGHTS for k in ("best", "final", "raw_final")]
                  + [f"lam{l}-samples.pt" for l in L4_WEIGHTS] + ["samples-corrected.pt"])
    pic = w.ckpt / f"{key}.png"
    require_new(w, key, list(fs.values()) + [pic])
    pkey, prow, _ = parent_row(w, n, updates)
    pc = prow["judged_row (kept delta, lowest held-out loss)"]["clean_share"]
    idx = w.tr[:n]
    runs, xs, labels, xcs = {}, [], [], {}
    for lam in L4_WEIGHTS:
        t1 = time.time()
        states, info = train_delta_x(w, idx, updates, adv_weight=lam, log=f"L4 lam={lam}", r1_chunk=R1_CHUNK)
        for k in ("best", "final", "raw_final"):
            torch.save(states[k], fs[f"lam{lam}-{k}.pt"])
        den_f = a_denoiser(w, delta_of(states["final"]))
        row_f, x, xcl = evaluate(w, den_f, idx, corrector=True)
        torch.save(x.cpu(), fs[f"lam{lam}-samples.pt"])
        xcs[lam] = xcl
        row_b, _, _ = evaluate(w, a_denoiser(w, delta_of(states["best"])), idx, corrector=True)
        runs[lam] = {"train": info, "row_final_ema": row_f, "row_kept_ema (lowest held-out loss)": row_b,
                     "tuning_clean_share_final_ema": tuning_share(w, den_f), "seconds": round(time.time() - t1, 1)}
        xs.append(x)
        labels.append(f"L4 lam={lam}")
        print(f"L4 lam={lam}: tuning {runs[lam]['tuning_clean_share_final_ema']:.3f} eval final EMA {row_f['clean_share']:.3f} "
              f"kept EMA {row_b['clean_share']:.3f}", flush=True)
    chosen = max(L4_WEIGHTS, key=lambda l: (runs[l]["tuning_clean_share_final_ema"], -l))
    delta = delta_of(torch.load(fs[f"lam{chosen}-final.pt"], map_location=DEV))
    xc = xcs[chosen]
    torch.save(xc.cpu(), fs["samples-corrected.pt"])
    judged = runs[chosen]["row_final_ema"]
    mc = judged["with_corrector_20_steps"]
    out = {"rung": "L4", "change_from_method_a": "Method A's training plus a hinge-loss adversarial term at weight lam: a "
           "discriminator on the frozen experts' inner decoder features of pictures re-noised at sigma_d log-uniform on "
           f"{L4_SIGMA_D}, real joint scenes against the delta's current one-step estimates, R1 {L4_R1} on real pictures every "
           f"{L4_R1_EVERY} updates, discriminator Adam {L4_D_LR} betas (0, 0.99); the delta's weights averaged with EMA {L4_EMA}, "
           "the final averaged delta judged (with an adversarial loss the held-out denoising loss is no longer what is trained)",
           "parent": {"key": pkey, "clean_share": pc}, "lam_chosen_on_tuning_seeds": chosen, "judged_row": judged,
           "runs": {str(k): v for k, v in runs.items()},
           "correction_error_vs_exact_by_band": correction_error(w, delta_correction(w, delta)),
           "prediction_check": prediction_check("L4", judged["clean_share"], pc),
           "files": {k: str(v) for k, v in fs.items()}, "picture": str(pic)}
    picture(w, xs + [xc], labels + [f"L4 lam={chosen} +corr"], pic)
    out["gpu_seconds"] = round(time.time() - t0, 1)
    save_result(w, key, out)
    print(f"{key}: chosen lam {chosen}, clean {judged['clean_share']:.3f} {judged['verdict']}, with corrector "
          f"{mc['clean_share']:.3f}", flush=True)
    return out


# --- L5 ------------------------------------------------------------------------------------------------------------------------------


def sampler_levels():
    return st.sigmas()[:st.STEPS]


def product_pictures(w):
    """Every red picture plus every blue picture minus the empty one: the plain-sum picture set, whose noised mixture is the
    product of the noised exact experts (see the module docstring)."""
    return w.r, w.b, w.e[0]


def mcmc_pool(denoise, seeds, steps):
    """(states (levels, seeds, 3, H, H), sigmas): the sampler's starting noise at sigma 80, then its state after every Heun step
    that lands above 0, each followed by `steps` Langevin steps at that level (rung 6's corrector)."""
    levels = sampler_levels()
    out = []
    for b0 in range(0, len(seeds), 256):
        chunk = seeds[b0:b0 + 256]
        x0 = st.starting_noise(chunk)
        keep = [x0.clone()]

        def after(i, x, s):
            if i < st.STEPS - 1:
                keep.append(x.detach().clone())
        with torch.no_grad():
            st.heun(denoise, x0, after_step=after, correct=st.corrector(denoise, chunk, steps))
        out.append(torch.stack(keep))
    return torch.cat(out, 1), levels


def fake_drawer(w, kind, base, seed_block):
    """A function n -> (fake pictures at sampler levels, their sigmas) for the fake-set kind, and a description."""
    levels = sampler_levels()
    seeds = list(range(FAKE_SEED_BASE + 4096 * seed_block, FAKE_SEED_BASE + 4096 * seed_block + (32 if SMOKE else L5_POOL_SEEDS)))
    if kind == "product-exact":
        R, B, E = (t.to(DEV) for t in product_pictures(w))

        def drawer(n):
            s = levels[torch.randint(len(levels), (n,), device=DEV)]
            x0 = R[torch.randint(len(R), (n,), device=DEV)] + B[torch.randint(len(B), (n,), device=DEV)] - E
            return x0 + s.view(-1, 1, 1, 1) * torch.randn_like(x0), s
        return drawer, "exact draws from the product of the noised exact experts at each level", None
    if kind == "finished-noised":
        Fp = draw(base, seeds).to(DEV)

        def drawer(n):
            s = levels[torch.randint(len(levels), (n,), device=DEV)]
            x0 = Fp[torch.randint(len(Fp), (n,), device=DEV)]
            return x0 + s.view(-1, 1, 1, 1) * torch.randn_like(x0), s
        return drawer, f"the base sampler's finished pictures from seeds {seeds[0]} to {seeds[-1]}, noised forward", Fp
    pool, _ = mcmc_pool(base, seeds, 2 if SMOKE else L5_MCMC_STEPS)
    pool = pool.to(DEV)

    def drawer(n):
        li = torch.randint(len(levels), (n,), device=DEV)
        si = torch.randint(pool.shape[1], (n,), device=DEV)
        return pool[li, si], levels[li]
    return drawer, (f"the base sampler's state at each level after {L5_MCMC_STEPS} Langevin steps (step {st.CORRECTOR_SIZE} "
                    f"sigma^2), seeds {seeds[0]} to {seeds[-1]}"), pool[-1]


def train_disc(w, idx, drawer, updates, seed=0):
    """BCE between real joint scenes (the N training scenes) and the fake set, each half of a batch at the same sampler levels."""
    torch.manual_seed(seed)
    disc = FeatDisc(w.nets, bound=L5_BOUND).to(DEV)
    opt = torch.optim.AdamW(disc.parameters(), lr=L5_LR, betas=(0.9, 0.99), weight_decay=0.01)
    Rj = w.j[idx].to(DEV)
    half = L5_BATCH // 2
    curve = []
    for u in range(1, updates + 1):
        xf, s = drawer(half)
        x0 = Rj[torch.randint(len(Rj), (half,), device=DEV)]
        xr = x0 + s.view(-1, 1, 1, 1) * torch.randn_like(x0)
        logit = disc(torch.cat([xr, xf]), torch.cat([s, s]))
        y = torch.cat([torch.ones(half, device=DEV), torch.zeros(half, device=DEV)])
        loss = F.binary_cross_entropy_with_logits(logit, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if u % 250 == 0 or u == updates:
            curve.append((u, float(loss.detach())))
            if u % 1000 == 0:
                print(f"  disc update {u} loss {float(loss.detach()):.4f}", flush=True)
    return disc.eval().requires_grad_(False), curve


def disc_accuracy(w, disc, drawer, idx):
    """Balanced accuracy per sigma band on fresh draws: held-out real joint scenes against the fake set."""
    out = {}
    levels = sampler_levels()
    with torch.no_grad():
        xf, s = drawer(4096 if not SMOKE else 256)
        Rv = w.j[w.va].to(DEV)
        x0 = Rv[torch.randint(len(Rv), (len(xf),), device=DEV)]
        xr = x0 + s.view(-1, 1, 1, 1) * torch.randn_like(x0)
        lr_, lf_ = [], []
        for i in range(0, len(xf), 512):
            lr_.append(disc(xr[i:i + 512], s[i:i + 512]))
            lf_.append(disc(xf[i:i + 512], s[i:i + 512]))
        lr_, lf_ = torch.cat(lr_), torch.cat(lf_)
    for hi, lo in st.SIGMA_BANDS:
        sel = (s <= hi) & (s > lo)
        if sel.any():
            out[f"{hi}-{lo}"] = round(0.5 * float((lr_[sel] > 0).float().mean() + (lf_[sel] <= 0).float().mean()), 3)
    del levels
    return out


def guided(base, disc, wgt, t_mid=L5_T_MID):
    """D' = D_base + w sigma^2 grad_x logit at every evaluation with sigma >= t_mid."""
    def denoise(x, s, step, second):
        d = base(x, s, step, second)
        if float(s) < t_mid:
            return d
        with torch.enable_grad():
            xx = x.detach().requires_grad_(True)
            g, = torch.autograd.grad(disc(xx, s).sum(), xx)
        return d + wgt * s ** 2 * g
    return denoise


def disc_correction(disc):
    def cfn(x, s):
        with torch.enable_grad():
            xx = x.detach().requires_grad_(True)
            g, = torch.autograd.grad(disc(xx, s).sum(), xx)
        return s.view(-1, 1, 1, 1) ** 2 * g
    return cfn


def mean_logit(disc, x):
    with torch.no_grad():
        return float(torch.cat([disc(x[i:i + 256].to(DEV), st.SIGMA_MIN) for i in range(0, len(x), 256)]).mean())


def l5_one(w, idx, kind, base, block, tag):
    """Train one discriminator on the fake set of `kind` drawn from `base`, choose w on tuning seeds, evaluate."""
    t0 = time.time()
    drawer, desc, _ = fake_drawer(w, kind, base, block)
    disc, curve = train_disc(w, idx, drawer, 20 if SMOKE else L5_UPDATES)
    tuning = {}
    for wg in L5_WEIGHTS:
        tuning[str(wg)] = tuning_share(w, guided(base, disc, wg))
    chosen = max(L5_WEIGHTS, key=lambda v: (tuning[str(v)], -v))
    den = guided(base, disc, chosen)
    row, x, xc = evaluate(w, den, idx)
    info = {"fake_set": kind, "fake_set_drawn_as": desc, "disc_updates": len(curve) and curve[-1][0], "disc_curve": curve,
            "disc_heldout_balanced_accuracy_by_band": disc_accuracy(w, disc, drawer, idx),
            "tuning_clean_share_by_w": tuning, "w_chosen_on_tuning_seeds": chosen, "row": row,
            "correction_error_vs_exact_by_band": correction_error(w, disc_correction(disc)),
            "disc_mean_logit_on_final_samples_at_sigma_min": mean_logit(disc, x),
            "seconds": round(time.time() - t0, 1)}
    print(f"L5 {tag}: w {chosen} tuning {tuning} eval {row['clean_share']:.3f} {row['verdict']} with corrector "
          f"{row['with_corrector_20_steps']['clean_share']:.3f}", flush=True)
    return disc, chosen, den, info, x, xc


def run_L5(w, n, updates, kinds):
    t0 = time.time()
    key = f"{w.name}-N{n}-u{updates}-L5"
    names = [k for k in kinds if k != "rounds-on-A"] + ([f"round{r}-on-A" for r in range(1, L5_ROUNDS + 1)]
                                                       if "rounds-on-A" in kinds else [])
    fs = files_of(w, key, [f"{nm}-disc.pt" for nm in names] + [f"{nm}-samples.pt" for nm in names]
                  + [f"{nm}-samples-corrected.pt" for nm in names])
    pic = w.ckpt / f"{key}.png"
    require_new(w, key, list(fs.values()) + [pic])
    pkey, prow, pstate = parent_row(w, n, updates)
    pc = prow["judged_row (kept delta, lowest held-out loss)"]["clean_share"]
    idx = w.tr[:n]
    plain = st.plainsum_denoiser(*w.nets)
    rows, xs, labels = {}, [], []
    for block, kind in enumerate(k for k in kinds if k != "rounds-on-A"):
        disc, wg, _, info, x, xc = l5_one(w, idx, kind, plain, block, kind)
        info["base"] = "neural plain sum"
        info["prediction_check (gain over the parent A)"] = prediction_check("L5", info["row"]["clean_share"], pc)
        torch.save(disc.state_dict(), fs[f"{kind}-disc.pt"])
        torch.save(x.cpu(), fs[f"{kind}-samples.pt"])
        torch.save(xc.cpu(), fs[f"{kind}-samples-corrected.pt"])
        rows[kind] = info
        xs += [x, xc]
        labels += [f"L5 {kind[:13]}", f"L5 {kind[:13]} +corr"]
        save_result(w, key + "-partial", {"status": "running", "rows_so_far": rows})
    if "rounds-on-A" in kinds:
        base = a_denoiser(w, delta_of(pstate))
        for r in range(1, L5_ROUNDS + 1):
            nm = f"round{r}-on-A"
            disc, wg, base, info, x, xc = l5_one(w, idx, "product-mcmc", base, 10 + r, nm)
            info["base"] = "Method A (the parent)" + "".join(f" + round {q} guidance" for q in range(1, r))
            info["fake_set"] = "the current sampler's states after MCMC at each level (fresh seeds this round)"
            info["prediction_check (gain over the parent A)"] = prediction_check("L5", info["row"]["clean_share"], pc)
            torch.save(disc.state_dict(), fs[f"{nm}-disc.pt"])
            torch.save(x.cpu(), fs[f"{nm}-samples.pt"])
            torch.save(xc.cpu(), fs[f"{nm}-samples-corrected.pt"])
            rows[nm] = info
            xs += [x, xc]
            labels += [f"L5 {nm}", f"L5 {nm} +corr"]
            save_result(w, key + "-partial", {"status": "running", "rows_so_far": rows})
    main = "product-exact" if "product-exact" in rows else next(iter(rows))
    out = {"rung": "L5", "change_from_method_a": "no delta: discriminator guidance on the neural plain sum (fake-set rows), or on "
           "top of Method A (round rows)", "main_row": main, "judged_row": rows[main]["row"],
           "prediction_check": rows[main]["prediction_check (gain over the parent A)"], "parent": {"key": pkey, "clean_share": pc},
           "rows": rows, "files": {k: str(v) for k, v in fs.items()}, "picture": str(pic),
           "caveats_without_a_toy_knob": {"C3": "real and fake are drawn by the same renderer and colours: no style gap exists",
                                          "C4": "the real set is the enumerated joint set: it holds no failures",
                                          "C8": "no attention maps on the toy"}}
    picture(w, xs, labels, pic)
    out["gpu_seconds"] = round(time.time() - t0, 1)
    save_result(w, key, out)
    return out


# -------------------------------------------------------------------------------------------------------------------------------------


WRITTEN_BEFORE = """# The loss ladder on the {world} at N = {n} joint scenes, {updates} updates (written before the run)

Every row is drawn on the {seeds} shared evaluation seeds. The bars: {bars}
Parent: Method A's recipe rerun at N = {n}, {updates} updates (rung 2b read 0.609 at N = 256, 5,000 updates on the toy).
Predictions (artifacts/notes/progress/setup/supervisor.md, 2026-11-28 04:27 UTC; for the harder toy each range is read as the same
gain over its parent, the range minus 0.609):
  L1  per-pixel weights on the contact band and occlusion edge: rises, 0.65 to 0.80
  L2  training noise shifted toward sigma above 2: rises a little, 0.62 to 0.75
  L3  reward fine-tune with the denoising anchor: 0.65 to 0.85, its held-out denoising loss rising
  L4  adversarial training on the experts' features: 0.60 to 0.80
  L5  discriminator guidance: adds 0 to 0.10 on top of A
Main L5 row: the fake set drawn exactly from the product of the noised experts at each level. The Langevin corrector (20 steps per
level, step 0.05 sigma^2) is reported on every judged row.
"""


def cmd_ladder(a):
    global SMOKE
    SMOKE = a.smoke
    w = make_world(a.world)
    n, updates = a.n, (20 if SMOKE else a.updates)
    before = w.ckpt / f"{w.name}-N{n}-u{updates}-written-before-the-run.md"
    text = WRITTEN_BEFORE.format(world=w.name, n=n, updates=updates, seeds=len(w.eval_seeds), bars=w.bars)
    bkey = f"{w.name}-N{n}-u{updates}-written-before-the-run"
    if bkey not in load_results(w):
        if before.exists() and before.read_text() != text:
            raise SystemExit(f"{before} exists with different text")
        before.write_text(text)
        save_result(w, bkey, text)
    print(text, flush=True)
    t0 = time.time()
    for run in a.runs.split(","):
        t1 = time.time()
        if run == "parent":
            run_trained(w, n, updates, "parent-A", "none: Method A's recipe")
        elif run == "L1":
            run_trained(w, n, updates, "L1", f"per-pixel loss weights 1 + {L1_KAPPA} on the contact band, mean 1 per scene",
                        pixel_w=pixel_weights(w))
        elif run == "L2":
            run_trained(w, n, updates, "L2", f"training log-sigma mean {L2_P_MEAN} instead of {st.P_MEAN}", p_mean=L2_P_MEAN)
        elif run == "L3":
            run_L3(w, n, updates)
        elif run == "L4":
            run_L4(w, n, updates)
        elif run == "L5":
            run_L5(w, n, updates, a.l5_kinds.split(","))
        else:
            raise SystemExit(f"unknown run {run}")
        print(f"[{run} took {time.time() - t1:.0f} s]", flush=True)
    print(f"ladder done in {time.time() - t0:.0f} s", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("ladder")
    q.add_argument("--world", default="toy", choices=("toy", "harder"))
    q.add_argument("--n", type=int, default=LADDER_N)
    q.add_argument("--updates", type=int, default=LADDER_UPDATES)
    q.add_argument("--runs", default="parent,L1,L2,L3,L4,L5")
    q.add_argument("--l5-kinds", default=",".join(L5_KINDS))
    q.add_argument("--smoke", action="store_true", help="tiny sizes into LADDER_SMOKE_DIR, for a CPU check")
    q.add_argument("--r1-chunk", type=int, default=0, help="L4's R1 on sub-batches of this size (0: whole batch)")
    a = p.parse_args()
    global R1_CHUNK
    R1_CHUNK = a.r1_chunk or None
    {"ladder": cmd_ladder}[a.cmd](a)


if __name__ == "__main__":
    main()
