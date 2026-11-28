#!/usr/bin/env python3
"""Times the parts of one adversarial-training update (paper1_interaction.py train --adv-weight), to choose a cheaper variant.

Runs --updates updates of the same arithmetic as paper1_interaction.Adversary.step plus Method A's denoising loss, on the real
positives at batch 64, with torch.cuda.synchronize() around each part, and writes <run>/profile.json with the mean seconds per
update of each part:
  denoising        Method A's training loss forward (experts without gradients, delta with its graph)
  fake_features    the three experts' features of D_new + sigma_d n_f, checkpointed (forward only; the recompute lands in fooling)
  disc_step        real features without gradients, the head on real and fake, the hinge loss backward
  r1               the R1 penalty through the experts, chunk by chunk (timed on R1 updates only, then spread over ADV_R1_EVERY)
  fooling          delta's fooling term, the backward of the whole loss (the checkpoint recompute through the experts included),
                   both optimiser steps and the EMA
Then, for the candidates, on --variant-updates more batches each (no training step needed for these timings):
  r1_head_only     R1 with the expert features computed without gradients and detached, so the gradient reaches y only
                   through the head's pixel path (per R1 update)
  features_3 / features_2   the feature pass with the three experts, and with red and blue only (no gradients)
  fake_features_2_fwd_bwd / fake_features_3_fwd_bwd   forward and backward of the checkpointed feature pass with 2 and 3 experts

  python paper1_adv_profile.py --run paper1-adv-profile --positives paper1-pack-joint2-touching/s1 --updates 200
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper1_sample as ps  # noqa: E402
import paper1_disc as pdisc  # noqa: E402
import paper1_interaction as pi  # noqa: E402

DEV = pi.DEV


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    def __init__(self):
        self.total, self.count = {}, {}

    def __call__(self, name, t0):
        sync()
        dt = time.time() - t0
        self.total[name] = self.total.get(name, 0.0) + dt
        self.count[name] = self.count.get(name, 0) + 1
        return time.time()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", default=str(ps.PAPER1_ROOT))
    ap.add_argument("--positives", required=True)
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--variant-updates", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5, help="updates run before timing starts")
    ap.add_argument("--batch", type=int, default=pi.BATCH)
    ap.add_argument("--adv-weight", type=float, default=0.03)
    ap.add_argument("--red-net", default=str(ps.RED_NET))
    ap.add_argument("--blue-net", default=str(ps.BLUE_NET))
    ap.add_argument("--empty-net", default=str(ps.EMPTY_NET))
    ap.add_argument("--edm2", default=str(ps.EDM2_DIR))
    args = ap.parse_args(argv)
    folder = ps.run_dir(args)
    pics, counts, _ = pdisc.load_folder(pdisc.resolve(args.root, args.positives))
    nets, _ = ps.load_three(args)
    torch.manual_seed(0)
    model = pi.InteractionNet().to(DEV).train()
    opt = torch.optim.Adam(model.parameters(), lr=pi.LR, betas=pi.BETAS)
    ema = __import__("copy").deepcopy(model).requires_grad_(False)
    adv = pi.Adversary(nets, 0)
    disc = adv.disc
    gen = torch.Generator().manual_seed(0)
    T, n = Timer(), args.batch

    def batch():
        idx = torch.randint(len(pics), (n,), generator=gen)
        return (pics[idx].to(DEV), counts[idx].to(DEV), pi.draw_sigma(n, gen).to(DEV),
                torch.randn((n, 3, 64, 64), generator=gen).to(DEV))

    def noised(x0, d_new):
        sd = torch.empty(n).uniform_(adv.lo, adv.hi, generator=adv.gen).exp().to(DEV)
        s4 = sd.view(-1, 1, 1, 1)
        return sd, x0 + s4 * torch.randn_like(x0), d_new + s4 * torch.randn_like(x0)

    for u in range(1, args.warmup + args.updates + 1):
        timing = u > args.warmup
        tm = T if timing else (lambda name, t0: time.time())
        sync()
        t = time.time()
        x0, cb, sigma, noise = batch()
        loss, d_new = pi.training_loss(nets, model, x0, cb, sigma, noise, return_d=True)
        t = tm("denoising", t)
        sd, y_r, y_f = noised(x0, d_new)
        f_f = disc.features(y_f, sd, cb, checkpointed=True)
        t = tm("fake_features", t)
        with torch.no_grad():
            f_r = disc.features(y_r, sd, cb)
        disc.requires_grad_(True)
        l_r = disc.head(y_r, f_r, sd, cb)
        l_f = disc.head(y_f.detach(), tuple(f.detach() for f in f_f), sd, cb)
        loss_d = F.relu(1 - l_r).mean() + F.relu(1 + l_f).mean()
        adv.opt.zero_grad(set_to_none=True)
        loss_d.backward()
        t = tm("disc_step", t)
        if u % pi.ADV_R1_EVERY == 0:
            for i in range(0, n, pi.ADV_CHUNK):
                yc = y_r[i:i + pi.ADV_CHUNK].detach().requires_grad_(True)
                lc = disc(yc, sd[i:i + pi.ADV_CHUNK], cb[i:i + pi.ADV_CHUNK])
                g, = torch.autograd.grad(lc.sum(), yc, create_graph=True)
                (pi.ADV_R1_EVERY * 0.5 * pi.ADV_R1 * g.pow(2).flatten(1).sum(1).sum() / n).backward()
            t = tm("r1_per_r1_update", t)
        adv.opt.step()
        disc.requires_grad_(False)
        g_adv = -disc.head(y_f, f_f, sd, cb).mean()
        total = loss + args.adv_weight * g_adv
        opt.zero_grad()
        total.backward()
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - pi.ADV_EMA)
        t = tm("fooling", t)
        if u % 50 == 0:
            print(json.dumps({"update": u, **{k: round(T.total[k] / T.count[k], 4) for k in T.total}}), flush=True)

    # Candidate timings
    feats = disc._feats
    for _ in range(args.variant_updates):
        x0, cb, sigma, noise = batch()
        with torch.no_grad():
            loss, d_new = pi.training_loss(nets, model, x0, cb, sigma, noise, return_d=True)
        sd, y_r, _ = noised(x0, d_new)
        labels = F.one_hot(cb.long() - 1, 3).float()
        disc.requires_grad_(True)
        sync()
        t = time.time()
        with torch.no_grad():
            f_r = disc.features(y_r, sd, cb)
        yc = y_r.detach().requires_grad_(True)
        lc = disc.head(yc, f_r, sd, cb)
        g, = torch.autograd.grad(lc.sum(), yc, create_graph=True)
        (pi.ADV_R1_EVERY * 0.5 * pi.ADV_R1 * g.pow(2).flatten(1).sum(1).mean()).backward()
        t = T("r1_head_only_per_r1_update", t)
        adv.opt.zero_grad(set_to_none=True)
        disc.requires_grad_(False)
        for k, use in ((3, feats), (2, feats[:2])):
            with torch.no_grad():
                for i in range(0, n, pi.ADV_CHUNK):
                    for fx, lab in zip(use, (labels, labels, None)):
                        fx(y_r[i:i + pi.ADV_CHUNK], sd[i:i + pi.ADV_CHUNK], None if lab is None else lab[i:i + pi.ADV_CHUNK])
            t = T(f"features_{k}_no_grad", t)
            yg = y_r.detach().requires_grad_(True)
            outs = []
            for i in range(0, n, pi.ADV_CHUNK):
                for fx, lab in zip(use, (labels, labels, None)):
                    fn = lambda yy, ss, ll, fx=fx: tuple(fx(yy, ss, ll)[b] for b in (pi.ADV_BLOCK16, pi.ADV_BLOCK8))
                    a = (yg[i:i + pi.ADV_CHUNK], sd[i:i + pi.ADV_CHUNK], None if lab is None else labels[i:i + pi.ADV_CHUNK])
                    outs += list(torch.utils.checkpoint.checkpoint(fn, *a, use_reentrant=False))
            sum(o.float().mean() for o in outs).backward()
            t = T(f"fake_features_{k}_fwd_bwd", t)

    mean = {k: round(T.total[k] / T.count[k], 4) for k in T.total}
    per_update = {k: mean[k] for k in ("denoising", "fake_features", "disc_step", "fooling")}
    per_update["r1_spread"] = round(mean["r1_per_r1_update"] / pi.ADV_R1_EVERY, 4)
    out = {"seconds_mean": mean, "counts": T.count, "seconds_per_update_by_part": per_update,
           "seconds_per_update_total": round(sum(per_update.values()), 4), "batch": n, "adv_chunk": pi.ADV_CHUNK,
           "r1_every": pi.ADV_R1_EVERY, "device": ps.device_name(), "updates_timed": args.updates,
           "variant_updates": args.variant_updates}
    (folder / "profile.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
