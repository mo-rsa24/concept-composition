#!/usr/bin/env python3
"""Loop 12 step Z9: arm 2 (the checker push) and arm 3 (the interaction push), with arm 3's controls.

Design: artifacts/notes/progress/leakage/09-dcfm-colored-mnist/design.md, "The three arms" and "The interaction push, in full".
Every push is -strength * sqrt(1 - abar_t) * grad_x(score), capped per picture at 0.1 of the prediction's length (strength 10,
Paper 1's kept settings), added to arm 1's composed prediction (z9_sample.Experts.eps).

  checker-data    arm 1 on the 50 seen combinations, seeds 1,000 to 1,039, called by the judge (the checker's drawn pictures)
  checker-train   the pair-aware checker on both frozen experts' decoder features, seen combinations only
  checker-arm     control 1 (strength 0 gives arm 1's pictures byte for byte), then arm 2 on the novel combinations
  ratio-train     the interaction classifier and its shuffled-label twin, on real training digits of the training pairs
  ratio-held      the held-back seen pairs: interaction push against the random push, head accuracy per step
  ratio-novel     the novel combinations: interaction push, controls a (heads only), b (random), c (shuffled twin);
                  the seen combinations for control e
  pooled-arm      control d: the pooled model's own plain product on the novel combinations
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "mnist"))
from canvas_unet import ABAR, T  # noqa: E402
from z9_data import colourise, load_mnist, reference  # noqa: E402
from z9_sample import (SEEDS, Experts, draw_set, fid, fid_stats, inception, judge_right, load_judge, log_wandb, sample,  # noqa: E402
                       to_uint8)
from z9_train import NONE, ColourUNet  # noqa: E402

STRENGTH, CAP = 10.0, 0.1
LOGIT_CLIP = float(np.log((1 - 1e-5) / 1e-5))


def time_features(t):
    a = ABAR.to(t.device)[t]
    lsnr = torch.log(a / (1 - a))[:, None]
    f = torch.exp(torch.linspace(0, 3, 8, device=t.device))[None] * lsnr / 10
    return torch.cat([f.sin(), f.cos()], 1)


def sigma(t):
    return (1 - ABAR.to(t.device)[t]).sqrt()[:, None, None, None]


# ---------------------------------------------------------------- arm 2: the checker

class CheckerHead(nn.Module):
    """Paper 1's pair-aware head (mnist/checker.py, Head with expert features), with a digit and a colour embedding."""

    def __init__(self):
        super().__init__()
        c = 256  # both experts' 128-channel decoder features, side by side
        self.down = nn.Conv2d(c, c, 3, stride=2, padding=1)
        self.body = nn.Sequential(nn.Conv2d(2 * c, 128, 3, padding=1), nn.SiLU(), nn.Conv2d(128, 128, 3, padding=1), nn.SiLU())
        self.demb, self.cemb = nn.Embedding(10, 64), nn.Embedding(10, 64)
        self.mlp = nn.Sequential(nn.Linear(256 + 16 + 64, 256), nn.SiLU(), nn.Linear(256, 1))

    def forward(self, feats, t, d, c):
        h = self.body(torch.cat([feats[0], self.down(feats[1])], 1))
        z = torch.cat([h.mean((2, 3)), h.amax((2, 3)), time_features(t), self.demb(d) + self.cemb(c)], 1)
        return self.mlp(z)[:, 0]


def expert_features(experts, x, t):
    """Both experts' decoder features at 8 x 8 and 16 x 16, asked with both labels none."""
    none = torch.full((x.shape[0],), NONE, device=x.device)
    f = [m(x, t, none, none, features=True)[1] for m in experts.models]
    return [torch.cat([f[0][k], f[1][k]], 1) for k in (0, 1)]


def checker_push(experts, head, d, c, s=STRENGTH):
    def push(x, t, eps):
        if s == 0:
            return torch.zeros_like(eps)
        with torch.enable_grad():
            xx = x.detach().requires_grad_(True)
            dd = torch.full((x.shape[0],), d, device=x.device); cc = torch.full((x.shape[0],), c, device=x.device)
            logit = head(expert_features(experts, xx, t), t, dd, cc)
            g, = torch.autograd.grad(logit.sum(), xx)
        inside = (logit.detach().abs() < LOGIT_CLIP).float()[:, None, None, None]
        return -s * sigma(t) * g * inside
    return push


def checker_data(args, split, dev):
    experts = Experts(args.experts, split, dev); judge = load_judge(args.judge, dev)
    imgs, dd, cc = [], [], []
    for d, c in [tuple(k) for k in split["seen"]]:
        x, _ = sample(lambda xx, tt: experts.eps(xx, tt, d, c), list(range(1000, 1040)), dev)
        imgs.append(to_uint8(x)); dd += [d] * 40; cc += [c] * 40
    imgs, dd, cc = np.concatenate(imgs), np.array(dd), np.array(cc)
    right = judge_right(judge, imgs, dd, cc, dev)
    out = args.out / "checker"; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "drawn.npz", images=imgs, digits=dd, colours=cc, right=right)
    print(json.dumps({"drawn": int(len(imgs)), "judged_right": float(right.mean())}), flush=True)


def checker_train(args, split, dev):
    experts = Experts(args.experts, split, dev)
    images, labels = load_mnist(args.mnist, "train")
    drawn = np.load(args.out / "checker" / "drawn.npz")
    seen = np.array(split["seen"])
    seen_set = {tuple(k) for k in split["seen"]}
    by_digit = {d: np.flatnonzero(labels == d) for d in range(10)}
    rng = np.random.default_rng(3); torch.manual_seed(3)
    head = CheckerHead().to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    dpos, dneg = np.flatnonzero(drawn["right"]), np.flatnonzero(~drawn["right"])

    def real(n, kind):
        k = seen[rng.integers(0, len(seen), n)]
        d_ask, c_ask = k[:, 0].copy(), k[:, 1].copy()
        d_pic, c_pic = d_ask.copy(), c_ask.copy()
        if kind == "wrong_colour":
            c_pic = (c_ask + rng.integers(1, 10, n)) % 10
        elif kind == "wrong_digit":
            d_pic = (d_ask + rng.integers(1, 10, n)) % 10
        x = colourise(images[[rng.choice(by_digit[int(v)]) for v in d_pic]], c_pic)
        return x, d_ask, c_ask

    for step in range(1, args.steps + 1):
        n = 64
        xp, dp, cp = real(n, "right")
        xw1, dw1, cw1 = real(n // 2, "wrong_colour")
        xw2, dw2, cw2 = real(n // 2, "wrong_digit")
        ip = rng.choice(dpos, n) if len(dpos) else np.array([], int)
        ineg = rng.choice(dneg, n) if len(dneg) else np.array([], int)
        to = lambda u8: drawn["images"][u8].astype(np.float32) / 127.5 - 1
        x = np.concatenate([xp, to(ip), xw1, xw2, to(ineg)])
        d = np.concatenate([dp, drawn["digits"][ip], dw1, dw2, drawn["digits"][ineg]])
        c = np.concatenate([cp, drawn["colours"][ip], cw1, cw2, drawn["colours"][ineg]])
        y = np.concatenate([np.ones(n + len(ip)), np.zeros(n + len(ineg))]).astype(np.float32)
        x = torch.from_numpy(x).to(dev)
        t = torch.randint(0, T, (len(x),), device=dev)
        a = ABAR.to(dev)[t][:, None, None, None]
        xt = a.sqrt() * x + (1 - a).sqrt() * torch.randn_like(x)
        with torch.no_grad():
            feats = expert_features(experts, xt, t)
        logit = head(feats, t, torch.from_numpy(d.astype(np.int64)).to(dev), torch.from_numpy(c.astype(np.int64)).to(dev))
        loss = F.binary_cross_entropy_with_logits(logit, torch.from_numpy(y).to(dev))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            print(json.dumps({"step": step, "loss": round(loss.item(), 4)}), flush=True)
    # held-back check: real test digits of seen combinations, right against wrong colour, at three noise levels
    timg, tlab = load_mnist(args.mnist, "test")
    seen_l = [tuple(k) for k in split["seen"]]
    xs, ds, cs = reference(timg, tlab, seen_l, 20, seed=11)
    xw, _, _ = reference(timg, tlab, [(d, (c + 3) % 10) for d, c in seen_l], 20, seed=11)  # the asked digit in a wrong colour
    acc = {}
    head.eval()
    for tval in (100, 400, 700):
        with torch.no_grad():
            right_ok, wrong_ok = [], []
            for arr, want in ((xs, 1), (xw, 0)):
                x = torch.from_numpy(arr).to(dev); t = torch.full((len(x),), tval, device=dev)
                a = ABAR.to(dev)[t][:, None, None, None]
                xt = a.sqrt() * x + (1 - a).sqrt() * torch.randn_like(x)
                lg = head(expert_features(experts, xt, t), t, torch.from_numpy(ds.astype(np.int64)).to(dev),
                          torch.from_numpy(cs.astype(np.int64)).to(dev))
                (right_ok if want else wrong_ok).append(((lg > 0).float() == want).float().mean().item())
            acc[f"t{tval}"] = {"right_called_right": right_ok[0], "wrong_colour_called_wrong": wrong_ok[0]}
    torch.save({"head": head.state_dict(), "steps": args.steps}, args.out / "checker" / "head.pt")
    (args.out / "checker" / "accuracy.json").write_text(json.dumps(acc, indent=1) + "\n")
    print(json.dumps({"held_back_seen_accuracy": acc}), flush=True)


def load_checker(args, dev):
    head = CheckerHead().to(dev).eval()
    head.load_state_dict(torch.load(args.out / "checker" / "head.pt", map_location=dev, weights_only=False)["head"])
    head.requires_grad_(False)
    return head


# ---------------------------------------------------------------- arm 3: the interaction classifier

class Interaction(nn.Module):
    """Digit head, colour head, and a rank-r interaction head u_d^T W(x, t) v_c, on noisy pictures."""

    def __init__(self, r=4):
        super().__init__()
        self.r = r
        self.body = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.SiLU(), nn.Conv2d(64, 64, 3, padding=1), nn.SiLU(), nn.AvgPool2d(2),
                                  nn.Conv2d(64, 128, 3, padding=1), nn.SiLU(), nn.Conv2d(128, 128, 3, padding=1), nn.SiLU(), nn.AvgPool2d(2),
                                  nn.Conv2d(128, 128, 3, padding=1), nn.SiLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.mix = nn.Sequential(nn.Linear(128 + 16, 256), nn.SiLU())
        self.digit, self.colour = nn.Linear(256, 10), nn.Linear(256, 10)
        self.w = nn.Linear(256, r * r)
        self.u, self.v = nn.Embedding(10, r), nn.Embedding(10, r)

    def forward(self, x, t):
        h = self.mix(torch.cat([self.body(x), time_features(t)], 1))
        return self.digit(h), self.colour(h), self.w(h).view(-1, self.r, self.r)

    def pair_score(self, W, d, c):
        """u_d^T W v_c for given d, c (tensors of the batch's length)."""
        return torch.einsum("br,brs,bs->b", self.u(d), W, self.v(c))


def ratio_train(args, split, dev, shuffled):
    images, labels = load_mnist(args.mnist, "train")
    held = {tuple(k) for k in split["ratio_held_back"]}
    train_pairs = np.array([k for k in split["seen"] if tuple(k) not in held])
    by_digit = {d: np.flatnonzero(labels == d) for d in range(10)}
    rng = np.random.default_rng(5 if not shuffled else 6); torch.manual_seed(5 if not shuffled else 6)
    net = Interaction().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    pd, pc = torch.from_numpy(train_pairs[:, 0]).to(dev), torch.from_numpy(train_pairs[:, 1]).to(dev)
    for step in range(1, args.steps + 1):
        k = rng.integers(0, len(train_pairs), 256)
        d, c = train_pairs[k, 0], train_pairs[k, 1]
        x = torch.from_numpy(colourise(images[[rng.choice(by_digit[int(v)]) for v in d]], c)).to(dev)
        if shuffled:  # labels shuffled across the batch, so no label matches its picture beyond chance
            k = k[rng.permutation(len(k))]; d, c = train_pairs[k, 0], train_pairs[k, 1]
        t = torch.randint(0, T, (len(x),), device=dev)
        a = ABAR.to(dev)[t][:, None, None, None]
        ld, lc, W = net(a.sqrt() * x + (1 - a).sqrt() * torch.randn_like(x), t)
        # interaction logits over the training pairs only
        lp = torch.einsum("pr,brs,ps->bp", net.u(pd), W, net.v(pc))
        target = torch.from_numpy(k.astype(np.int64)).to(dev)
        loss = F.cross_entropy(ld, torch.from_numpy(d.astype(np.int64)).to(dev)) + \
            F.cross_entropy(lc, torch.from_numpy(c.astype(np.int64)).to(dev)) + F.cross_entropy(lp, target)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            print(json.dumps({"shuffled": shuffled, "step": step, "loss": round(loss.item(), 4)}), flush=True)
    out = args.out / "ratio"; out.mkdir(parents=True, exist_ok=True)
    torch.save({"net": net.state_dict(), "train_pairs": train_pairs.tolist(), "shuffled": shuffled},
               out / ("classifier-shuffled.pt" if shuffled else "classifier.pt"))


def load_ratio(args, dev, shuffled=False):
    net = Interaction().to(dev).eval()
    net.load_state_dict(torch.load(args.out / "ratio" / ("classifier-shuffled.pt" if shuffled else "classifier.pt"),
                                   map_location=dev, weights_only=False)["net"])
    net.requires_grad_(False)
    return net


def interaction_push(net, d, c, s=STRENGTH):
    def push(x, t, eps):
        with torch.enable_grad():
            xx = x.detach().requires_grad_(True)
            _, _, W = net(xx, t)
            dd = torch.full((x.shape[0],), d, device=x.device); cc = torch.full((x.shape[0],), c, device=x.device)
            g, = torch.autograd.grad(net.pair_score(W, dd, cc).sum(), xx)
        return -s * sigma(t) * g
    return push


def matched_push(direction_fn, lengths_by_combo, d, c):
    """A push whose length at each step and picture is the interaction push's (after its cap); direction from direction_fn."""
    state = {"i": 0}
    L = lengths_by_combo[(d, c)]  # (pictures, steps)

    def push(x, t, eps):
        g = direction_fn(x, t)
        n = g.flatten(1).norm(dim=1).clamp_min(1e-12)
        want = torch.from_numpy(L[:, state["i"]]).to(x.device).float()
        state["i"] += 1
        return g * (want / n)[:, None, None, None]
    return push


def heads_direction(net, d, c):
    def f(x, t):
        with torch.enable_grad():
            xx = x.detach().requires_grad_(True)
            ld, lc, _ = net(xx, t)
            s = F.log_softmax(ld, 1)[:, d] + F.log_softmax(lc, 1)[:, c]
            g, = torch.autograd.grad(s.sum(), xx)
        return -g  # the same sign convention as a push: added to eps, so minus the ascent direction
    return f


def random_direction(seed_base):
    gen = torch.Generator().manual_seed(seed_base)

    def f(x, t):
        return torch.randn(x.shape, generator=gen).to(x.device)
    return f


def mcnemar_one_sided(a_right, b_right):
    """Exact one-sided McNemar: is a right more often than b on matched pictures? Returns (a_only, b_only, p)."""
    from scipy.stats import binom
    a_only = int((a_right & ~b_right).sum()); b_only = int((b_right & ~a_right).sum())
    n = a_only + b_only
    p = float(binom.sf(a_only - 1, n, 0.5)) if n else 1.0
    return a_only, b_only, p


def head_accuracy_record(net, d, c, log):
    def rec(i, x, t):
        with torch.no_grad():
            ld, lc, W = net(x, t)
        log.setdefault(i, []).append({"digit": float((ld.argmax(1) == d).float().mean()), "colour": float((lc.argmax(1) == c).float().mean())})
    return rec


def run_arm(experts, combos, dev, push_for, record_for=None, per=100):
    imgs, dd, cc, lens = [], [], [], {}
    for d, c in combos:
        pf = push_for(d, c)
        rec = record_for(d, c) if record_for else None
        x, applied = sample(lambda xx, tt: experts.eps(xx, tt, d, c), SEEDS[:per], dev, push_fn=pf, cap=CAP, record=rec)
        imgs.append(to_uint8(x)); dd += [d] * per; cc += [c] * per
        if applied is not None:
            lens[(d, c)] = applied.numpy()
    return np.concatenate(imgs), np.array(dd), np.array(cc), lens


def score_arm(name, imgs, d, c, judge, dev, base_right, out, net_fid=None, s_real=None):
    right = judge_right(judge, imgs, d, c, dev)
    a_only, b_only, p = mcnemar_one_sided(right, base_right)
    res = {"accuracy": float(right.mean()), "lead_points": 100 * float(right.mean() - base_right.mean()),
           "won": a_only, "lost": b_only, "p_one_sided": p}
    if net_fid is not None:
        res["fid"] = fid(fid_stats(net_fid, imgs, dev), s_real)
    np.savez_compressed(out / f"{name}.npz", images=imgs, digits=d, colours=c, right=right)
    print(json.dumps({name: res}), flush=True)
    return res, right


def real_stats(args, split, name, net, dev):
    timg, tlab = load_mnist(args.mnist, "test")
    real, _, _ = reference(timg, tlab, [tuple(k) for k in split[name]], 200, seed=7)
    blocks = to_uint8(torch.from_numpy(real)).reshape(len(split[name]), 200, 3, 32, 32)
    return fid_stats(net, blocks[:, :100].reshape(-1, 3, 32, 32), dev)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["checker-data", "checker-train", "checker-arm", "ratio-train", "ratio-held", "ratio-novel", "pooled-arm"])
    ap.add_argument("--mnist", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--experts", type=Path, required=True)
    ap.add_argument("--judge", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=3000)
    args = ap.parse_args()
    split = json.loads(args.split.read_text())
    dev = "cuda"
    t0 = time.time()
    if args.stage == "checker-data":
        return checker_data(args, split, dev)
    if args.stage == "checker-train":
        return checker_train(args, split, dev)
    if args.stage == "ratio-train":
        ratio_train(args, split, dev, False); ratio_train(args, split, dev, True); return
    experts = Experts(args.experts, split, dev); judge = load_judge(args.judge, dev)
    plain_novel = np.load(args.out / "plain" / "novel-right.npy")
    novel = [tuple(k) for k in split["novel"]]
    summary = {}
    if args.stage == "checker-arm":
        head = load_checker(args, dev)
        out = args.out / "checker"
        # control 1 on a 5-combination subset, one seed each
        ok = True
        for d, c in novel[:5]:
            a, _ = sample(lambda xx, tt: experts.eps(xx, tt, d, c), [0], dev)
            b, _ = sample(lambda xx, tt: experts.eps(xx, tt, d, c), [0], dev, push_fn=checker_push(experts, head, d, c, s=0.0), cap=CAP)
            ok &= bool((to_uint8(a) == to_uint8(b)).all())
        summary["control_1_zero_push_identical"] = ok
        print(json.dumps({"control_1": ok}), flush=True)
        if not ok:
            raise SystemExit("control 1 failed: a push of zero changed the pictures")
        net = inception(dev)
        imgs, d, c, lens = run_arm(experts, novel, dev, lambda d, c: checker_push(experts, head, d, c))
        res, _ = score_arm("arm-2-novel", imgs, d, c, judge, dev, plain_novel, out, net, real_stats(args, split, "novel", net, dev))
        summary["arm_2"] = res
    elif args.stage == "ratio-held":
        net = load_ratio(args, dev); out = args.out / "ratio"
        held = [tuple(k) for k in split["ratio_held_back"]]
        log = {}
        imgs, d, c, lens = run_arm(experts, held, dev, lambda d, c: interaction_push(net, d, c),
                                   record_for=lambda d, c: head_accuracy_record(net, d, c, log))
        right_push = judge_right(judge, imgs, d, c, dev)
        imgs_r, _, _, _ = run_arm(experts, held, dev, lambda dd_, cc_: matched_push(random_direction(7_000 + 100 * dd_ + cc_), lens, dd_, cc_))
        right_rand = judge_right(judge, imgs_r, d, c, dev)
        imgs_p, _, _, _ = run_arm(experts, held, dev, lambda dd_, cc_: None)
        right_plain = judge_right(judge, imgs_p, d, c, dev)
        a_only, b_only, p = mcnemar_one_sided(right_push, right_rand)
        per_step = {int(i): {k: float(np.mean([e[k] for e in v])) for k in ("digit", "colour")} for i, v in sorted(log.items())}
        summary["held_back"] = {"push": float(right_push.mean()), "random": float(right_rand.mean()), "plain": float(right_plain.mean()),
                                "lead_points": 100 * float(right_push.mean() - right_rand.mean()), "won": a_only, "lost": b_only,
                                "p_one_sided": p, "passes": bool(100 * (right_push.mean() - right_rand.mean()) >= 5 and p < 0.05),
                                "head_accuracy_per_step": per_step}
        np.savez_compressed(out / "held-back.npz", push=imgs, random=imgs_r, plain=imgs_p, digits=d, colours=c)
        np.save(out / "held-back-lengths.npy", np.stack([lens[k] for k in held]))
        print(json.dumps({k: v for k, v in summary["held_back"].items() if k != "head_accuracy_per_step"}), flush=True)
    elif args.stage == "ratio-novel":
        net = load_ratio(args, dev); shuf = load_ratio(args, dev, shuffled=True); out = args.out / "ratio"
        fnet = inception(dev); s_real = real_stats(args, split, "novel", fnet, dev)
        imgs, d, c, lens = run_arm(experts, novel, dev, lambda dd_, cc_: interaction_push(net, dd_, cc_))
        summary["arm_3"], right3 = score_arm("arm-3-novel", imgs, d, c, judge, dev, plain_novel, out, fnet, s_real)
        imgs_a, _, _, _ = run_arm(experts, novel, dev, lambda dd_, cc_: matched_push(heads_direction(net, dd_, cc_), lens, dd_, cc_))
        summary["control_a_heads_only"], right_a = score_arm("control-a-novel", imgs_a, d, c, judge, dev, plain_novel, out, fnet, s_real)
        a_only, b_only, p = mcnemar_one_sided(right3, right_a)
        summary["arm_3_against_control_a"] = {"lead_points": 100 * float(right3.mean() - right_a.mean()), "won": a_only, "lost": b_only, "p_one_sided": p}
        imgs_b, _, _, _ = run_arm(experts, novel, dev, lambda dd_, cc_: matched_push(random_direction(8_000 + 100 * dd_ + cc_), lens, dd_, cc_))
        summary["control_b_random"], _ = score_arm("control-b-novel", imgs_b, d, c, judge, dev, plain_novel, out, fnet, s_real)
        imgs_c, _, _, _ = run_arm(experts, novel, dev, lambda dd_, cc_: interaction_push(shuf, dd_, cc_))
        summary["control_c_shuffled"], _ = score_arm("control-c-novel", imgs_c, d, c, judge, dev, plain_novel, out, fnet, s_real)
        # control e on the seen combinations
        seen = [tuple(k) for k in split["seen"]]
        plain_seen = np.load(args.out / "plain" / "seen-right.npy")
        imgs_s, ds, cs, lens_s = run_arm(experts, seen, dev, lambda dd_, cc_: interaction_push(net, dd_, cc_))
        right_s = judge_right(judge, imgs_s, ds, cs, dev)
        imgs_sr, _, _, _ = run_arm(experts, seen, dev, lambda dd_, cc_: matched_push(random_direction(9_000 + 100 * dd_ + cc_), lens_s, dd_, cc_))
        right_sr = judge_right(judge, imgs_sr, ds, cs, dev)
        summary["control_e_already_right_broken"] = {"interaction": int((plain_seen & ~right_s).sum()), "random": int((plain_seen & ~right_sr).sum()),
                                                     "plain_right": int(plain_seen.sum())}
        print(json.dumps({"control_e": summary["control_e_already_right_broken"]}), flush=True)
    elif args.stage == "pooled-arm":
        ck = torch.load(args.experts / "holder-pooled" / "unet.pt", map_location=dev, weights_only=False)
        m = ColourUNet(ck["ch"]).to(dev).eval(); m.load_state_dict(ck["ema"]); m.requires_grad_(False)

        def eps_pooled(x, t, d, c):
            full = lambda v: torch.full((x.shape[0],), v, device=dev, dtype=torch.long)
            e0 = m(x, t, full(NONE), full(NONE))
            return e0 + (m(x, t, full(d), full(NONE)) - e0) + (m(x, t, full(NONE), full(c)) - e0)
        imgs, dd, cc = [], [], []
        for d, c in novel:
            x, _ = sample(lambda xx, tt: eps_pooled(xx, tt, d, c), SEEDS, dev)
            imgs.append(to_uint8(x)); dd += [d] * 100; cc += [c] * 100
        fnet = inception(dev)
        summary["control_d_pooled"], _ = score_arm("control-d-novel", np.concatenate(imgs), np.array(dd), np.array(cc), judge, dev,
                                                   plain_novel, args.out / "ratio", fnet, real_stats(args, split, "novel", fnet, dev))
    summary["seconds"] = round(time.time() - t0, 1)
    (args.out / f"{args.stage}.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    flat = {f"{k}/{kk}": vv for k, v in summary.items() if isinstance(v, dict) for kk, vv in v.items() if isinstance(vv, (int, float, bool))}
    print(json.dumps({"wandb_run": log_wandb(f"loop-12-z9-{args.stage}", {"stage": args.stage}, flat)}), flush=True)


if __name__ == "__main__":
    main()
