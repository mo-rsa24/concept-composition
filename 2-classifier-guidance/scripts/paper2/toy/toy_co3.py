#!/usr/bin/env python3
"""A toy CO3 base for Paper 2's miniature: a product of per-object experts, sampled with Heun plus a Langevin corrector.

Why this exists. The miniature's dials were built on one conditional base model that reads the whole condition at once. Paper 2's
real system composes with CO3 on SDXL: one network evaluated once per concept, the evaluations summed as a product of experts, and
a corrector run between sampler steps. This file builds that sampler at toy scale so the dial stages in dials_toy.py can run on it
unchanged, and the only difference between the two rows of every stage is the base and its sampler.

The experts. One network, the same CondUNet as the conditional base (width 128, self-attention at 8 by 8 and 4 by 4, so a dial wraps
exactly the same layers), trained from scratch on the same 200,000 training canvases with a partial condition of 17 numbers:
  numbers 0 to 13   one combination k present on the canvas with its multiplicity m on the canvas: count m at k, at k's shape (8 or
                    9) and at k's colour (10 to 13), every other combination, shape and colour zero
  numbers 14 to 16  the canvas's total object count n (1, 2 or 3), one-hot
With probability P_EMPTY the 14 combination numbers are all zero, which trains the empty expert p(x | n objects). So expert (k, m, n)
models a canvas of n objects that holds m copies of combination k, whatever the other objects are, and the empty expert models a
canvas of n objects.

Why one network with partial conditions rather than reusing the conditional base with one-object conditions. The conditional base
reads "RS" as "a canvas holding exactly one red square and nothing else", so a sum of two such experts asks for two incompatible
canvases, and the base was never trained on an all-zero condition, so it has no empty expert to subtract. Training one network on
partial conditions gives experts whose product is the composition asked: p(x | n) times the product over k of p(k appears m times |
x, n), which for n objects and these combinations is the canvas holding exactly the asked objects. The total count n is shared by
every expert because a product of "contains" experts cannot say how many objects there are (two copies of one combination are one
expert with m = 2, never the same expert used twice); the joint condition itself is never shown to an expert.

The composition. For a condition with E distinct combinations (the stages pass the same 14-number full condition they pass the
conditional base), D = sum over the E experts of D_k - (E - 1) D_empty, the empty expert at the same n. With E = 1 this is the one
expert and the empty expert is not evaluated.

The sampler. EDM2's deterministic Heun (32 steps, sigma 80 to 0.002, the same arithmetic as toy_model.heun_range), then after every
step that lands above sigma 0, CORR_STEPS Langevin steps at that sigma: x <- x + eta score + sqrt(2 eta) z, score = (D(x, sigma) - x)
/ sigma^2 from the composed denoiser, eta = CORR_SIZE sigma^2 (Paper 1's squares toy, rung 6). CO3's resampling is not included: this
toy has the product and the corrector only. In sample(), each seed's corrector noise comes from its own generator seeded
CORR_SEED_OFFSET + seed, so a seed's picture does not depend on batching and every row of a stage (frozen, dial, random dial, stack)
shares the corrector noise as it shares the starting noise. Where no seeds are known (a dial's gradient training rollouts and the
correction's state bank), the corrector noise comes from torch's global generator.

install(tm) swaps toy_model's load_base, sample and heun_range for these, which is the whole interface the stages use; the composed
model's forward takes (x, sigma, full condition) like the conditional base's, so the stages, dials, closeness term and correction
see the same calls.

  /home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python 2-classifier-guidance/scripts/paper2/toy/toy_co3.py train --updates 40000
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toy_canvas as tc  # noqa: E402
import toy_model as tm  # noqa: E402

st = tm.st
DEV = tm.DEV

PART_DIM = 17                      # 14 partial-condition numbers + the total count one-hot (n = 1, 2, 3)
MAX_N = 3
P_EMPTY = 0.2                      # share of training examples given the empty condition
CORR_STEPS = 5                     # Langevin steps after every Heun step above sigma 0 (rung 6 read 5 and 20)
CORR_SIZE = 0.05                   # eta = CORR_SIZE sigma^2, rung 6's step
CORR_SEED_OFFSET = 50_000_000      # clear of every seed range in toy_canvas.SEEDS and the stage-1 reads

# ---- stage 1 on the composed sampler: the bars are dials_toy's, unchanged; the seeds are new and fixed here before the first read
CO3_TEST_SEED0 = 23_072            # test seeds 23,072 to 24,095
CO3_TUNE_SEED0 = 11_536            # tuning seeds 11,536 to 12,047


# -------------------------------------------------------------------------------------------------------------------------------------
# The expert network and the composed denoiser


class CO3Denoiser(tm.Denoiser):
    """The expert network (CondUNet on 17 partial-condition numbers) whose forward composes the experts a full condition asks for."""

    def __init__(self, w=128, attn=True):
        nn.Module.__init__(self)
        self.net = tm.CondUNet(w, cond_dim=PART_DIM, attn=attn)

    def expert(self, x, sigma, part):
        """One expert evaluation per row: part is (N, 17)."""
        return tm.Denoiser.forward(self, x, sigma, part)

    def forward(self, x, sigma, cond):
        """cond: (N, 14) full conditions, as the stages pass the conditional base. Returns sum_k D_k - (E - 1) D_empty per row."""
        sigma = torch.as_tensor(sigma, device=x.device, dtype=torch.float32).reshape(-1).expand(len(x))
        bi, part, ei, empty, E = expert_conditions(cond)
        Dall = self.expert(torch.cat([x[bi], x[ei]]), torch.cat([sigma[bi], sigma[ei]]), torch.cat([part, empty]))
        De, D0 = Dall[:len(bi)], Dall[len(bi):]
        out = torch.zeros_like(x).index_add(0, bi, De)
        if len(ei):
            out = out.index_add(0, ei, -(E[ei] - 1).to(x.dtype).view(-1, 1, 1, 1) * D0)
        return out


def expert_conditions(cond):
    """(row of each expert, its 17-number condition, rows needing the empty expert, their empty conditions, experts per row)."""
    counts = cond[:, :8].round().long()
    n = counts.sum(1)
    if (n < 1).any() or (n > MAX_N).any():
        raise ValueError(f"the composed base takes 1 to {MAX_N} objects; asked {sorted(set(n.tolist()))}")
    present = counts > 0
    E = present.sum(1)
    bi, ki = present.nonzero(as_tuple=True)
    m = counts[bi, ki].to(cond.dtype)
    r = torch.arange(len(bi), device=cond.device)
    part = torch.zeros(len(bi), PART_DIM, device=cond.device, dtype=cond.dtype)
    part[r, ki] = m
    part[r, 8 + ki // 4] = m
    part[r, 10 + ki % 4] = m
    part[r, 14 + n[bi] - 1] = 1
    ei = (E > 1).nonzero()[:, 0]
    empty = torch.zeros(len(ei), PART_DIM, device=cond.device, dtype=cond.dtype)
    empty[torch.arange(len(ei), device=cond.device), 14 + n[ei] - 1] = 1
    return bi, part, ei, empty, E


def training_parts(C, gen=None):
    """Partial conditions for training canvases C (B, 14 counts): one present combination with its multiplicity, or the empty
    condition with probability P_EMPTY, always with the canvas's total count."""
    counts = C[:, :8]
    n = counts.sum(1).long()
    B = len(C)
    score = torch.rand(B, 8, device=C.device, generator=gen) + (counts > 0).float()     # present combinations score above 1
    k = score.argmax(1)
    m = counts[torch.arange(B, device=C.device), k]
    keep = (torch.rand(B, device=C.device, generator=gen) >= P_EMPTY).float()
    part = torch.zeros(B, PART_DIM, device=C.device)
    r = torch.arange(B, device=C.device)
    part[r, k] = m * keep
    part[r, 8 + k // 4] = m * keep
    part[r, 10 + k % 4] = m * keep
    part[r, 14 + n - 1] = 1
    return part


def load_base(path, dials=(), dial_states=None):
    """toy_model.load_base for the composed base: a fresh frozen expert network, wrapped, with only the named dials loaded."""
    cfg = json.loads(Path(path).with_suffix(".json").read_text())
    if cfg.get("kind") != "co3":
        raise ValueError(f"{path} is not a composed (co3) base; its sidecar says {cfg}")
    m = CO3Denoiser(cfg["width"], cfg["attn"]).to(DEV)
    m.load_state_dict(torch.load(path, map_location=DEV))
    tm.freeze_base(m)
    m.eval()
    tm.wrap(m)
    for d in dials:
        tm.load_dial(m, d, dial_states[d])
        for p in tm.dial_params(m, d):
            p.requires_grad_(False)
    return m


# -------------------------------------------------------------------------------------------------------------------------------------
# Heun plus the Langevin corrector


def heun_range(denoise, x, start=0, stop=st.STEPS, on_step=None, gens=None):
    """toy_model.heun_range with CORR_STEPS Langevin steps after every Heun step that lands above sigma 0. gens: one torch.Generator
    per row for the corrector noise, or None for torch's global generator."""
    t = st.sigmas(x.device)
    for i in range(start, stop):
        t_cur, t_next = t[i], t[i + 1]
        d_cur = (x - denoise(x, t_cur, i, False)) / t_cur
        x_next = x + (t_next - t_cur) * d_cur
        if i < st.STEPS - 1:
            d_eval = denoise(x_next, t_next, i, True)
            x_next = x + (t_next - t_cur) * 0.5 * (d_cur + (x_next - d_eval) / t_next)
        x = x_next
        if i < st.STEPS - 1 and CORR_STEPS:
            if gens is None:
                noise = torch.randn((CORR_STEPS,) + tuple(x.shape), device=x.device).transpose(0, 1)
            else:
                noise = torch.stack([torch.randn(CORR_STEPS, *x.shape[1:], generator=g) for g in gens]).to(x.device)
            eta = CORR_SIZE * t_next ** 2
            for k in range(CORR_STEPS):
                score = (denoise(x, t_next, i + 1, True) - x) / t_next ** 2
                x = x + eta * score + (2 * eta).sqrt() * noise[:, k]
        if on_step is not None:
            on_step(i, x, t_next)
    return x


@torch.no_grad()
def sample(model, conds, seeds, batch=1024):
    """toy_model.sample on the composed sampler: the same starting noise per seed, and per-seed corrector noise."""
    out = []
    cv = tc.cond_vector(conds).to(DEV)
    for i in range(0, len(seeds), batch):
        chunk = seeds[i:i + batch]
        gens = [torch.Generator().manual_seed(CORR_SEED_OFFSET + int(s)) for s in chunk]
        x = st.starting_noise(chunk)
        out.append(heun_range(tm.denoiser_for(model, cv[i:i + batch]), x, gens=gens).cpu())
    return torch.cat(out)


def install(tm_module):
    """Point toy_model's base loader, sampler and step loop at the composed base, for dials_toy.py's stages."""
    tm_module.load_base = load_base
    tm_module.sample = sample
    tm_module.heun_range = heun_range


# -------------------------------------------------------------------------------------------------------------------------------------
# Training the expert network


def save_results(results_dir, key, value):
    p = Path(results_dir) / "results.json"
    res = json.loads(p.read_text()) if p.exists() else {}
    if key in res:
        sys.exit(f"stopped: results key {key} exists; never overwritten")
    res[key] = value
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))


def cmd_train(a):
    out = Path(a.out)
    path = out / "co3-base.pt"
    if path.exists():
        sys.exit(f"stopped: {path} exists; move it to its own name first")
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(Path(a.canvases))
    X = torch.from_numpy(d["idx"]).to(DEV)
    C = torch.from_numpy(d["cond"]).float().to(DEV)
    pal = tc.PALETTE.to(DEV)
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    torch.manual_seed(0)
    model = CO3Denoiser(a.width, a.attn).to(DEV)
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.99))
    curve = []
    for u in range(a.updates):
        lr = a.lr * min(1.0, (a.updates - u) / (0.2 * a.updates))          # last fifth decays linearly to zero, as the base's
        for g in opt.param_groups:
            g["lr"] = lr
        k = torch.randint(len(X), (a.batch,), device=DEV)
        x0 = pal[X[k].long()].permute(0, 3, 1, 2)
        s = st.draw_sigma(a.batch)
        part = training_parts(C[k])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
            D = model.expert(x0 + s.view(-1, 1, 1, 1) * torch.randn_like(x0), s, part)
        loss = st.edm_loss(D.float(), x0, s, model.net.logvar(s).float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - a.ema)
        if u % 500 == 0 or u == a.updates - 1:
            curve.append((u, round(float(loss.detach()), 4)))
            print(f"  co3 expert update {u} loss {float(loss.detach()):.4f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save(ema.state_dict(), path)
    path.with_suffix(".json").write_text(json.dumps({"kind": "co3", "width": a.width, "attn": a.attn, "p_empty": P_EMPTY,
                                                     "corr_steps": CORR_STEPS, "corr_size": CORR_SIZE}) + "\n")
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    info = {"updates": a.updates, "width": a.width, "attn": a.attn, "batch": a.batch, "lr": a.lr, "ema": a.ema, "p_empty": P_EMPTY,
            "parameters": sum(p.numel() for p in model.parameters()), "loss_curve": curve, "seconds": round(time.time() - t0, 1)}
    if a.results:
        save_results(a.results, a.key, info)
    print(json.dumps({k: v for k, v in info.items() if k != "loss_curve"}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("train")
    q.add_argument("--updates", type=int, default=40_000)
    q.add_argument("--width", type=int, default=128)
    q.add_argument("--no-attn", dest="attn", action="store_false")
    q.add_argument("--batch", type=int, default=256)
    q.add_argument("--lr", type=float, default=1e-3)
    q.add_argument("--ema", type=float, default=0.999)
    q.add_argument("--out", default=os.environ.get("PAPER2_TOY_WORK", "outputs/paper2-toy/co3"))
    q.add_argument("--canvases", default=str(Path(os.environ.get("PAPER2_TOY_OUT", "outputs/paper2-toy")) / "train_canvases.npz"))
    q.add_argument("--results", default=os.environ.get("PAPER2_TOY_RESULTS", ""))
    q.add_argument("--key", default="co3_base_training")
    a = p.parse_args()
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
