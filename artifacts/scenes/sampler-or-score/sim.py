"""Two-error toy for the reading 6 page.

Experts follow Catastrophic Compositional Generation p9: P_a1 = N(0, diag(10, 1)),
P_a2 = N(0, diag(1, 10)). The base P_0 = N(0, c I) sets the composed target
P_a1 P_a2 / P_0. c = 1 is the paper's factorized out-of-distribution setting.
c = 1.25 is this page's own illustrative non-factorized setting.

Eight cells: sampler error (c = 1 or 1.25) x score error (analytic or trained
scores) x corrector (none, or Feynman-Kac weights with K = 16 particles).
"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
A, B = 10.0, 1.0
EXPERT_VARS = {1: (A, B), 2: (B, A)}
SIGMA_MAX, SIGMA_MIN, N_STEPS = 50.0, 0.02, 400
N_OUT, K_FKC, SEEDS = 2000, 16, (0, 1, 2, 3, 4)
N_TRAIN, TRAIN_STEPS, BATCH, WIDTH = 5000, 14000, 1024, 128


def variances(label, c):
    return (c, c) if label == 0 else EXPERT_VARS[label]


def target_var(c, u=0.0):
    return 1.0 / (1 / (A + u) + 1 / (B + u) - 1 / (c + u))


def silu(h):
    return h / (1 + np.exp(-h))


def dsilu(h):
    e = np.exp(-h)
    return (1 + e + h * e) / (1 + e) ** 2


class Net:
    """Conditional noise predictor: inputs are scaled x, log sigma and a one-hot label."""

    def __init__(self, rng):
        sizes = [6, WIDTH, WIDTH, WIDTH, 2]
        self.W = [rng.normal(0, 1 / np.sqrt(a), (a, b)).astype(np.float32) for a, b in zip(sizes, sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.W[-1] *= 0.1
        params = self.W + self.b
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    @staticmethod
    def features(x, sigma, label):
        n = len(x)
        scale = np.sqrt(sigma**2 + 5.0)
        f = np.empty((n, 6), np.float32)
        f[:, :2] = x / scale[:, None]
        f[:, 2] = np.log(sigma) / 4
        f[:, 3:] = 0
        f[np.arange(n), 3 + label] = 1
        return f

    def forward(self, f):
        h = f
        for W, b in zip(self.W[:-1], self.b[:-1]):
            h = silu(h @ W + b)
        return h @ self.W[-1] + self.b[-1]

    def step(self, f, target, lr):
        acts, pre = [f], []
        h = f
        for W, b in zip(self.W[:-1], self.b[:-1]):
            z = h @ W + b
            pre.append(z)
            h = silu(z)
            acts.append(h)
        out = h @ self.W[-1] + self.b[-1]
        err = out - target
        loss = float((err**2).sum(1).mean())
        n = len(f)
        grad_h = 2 * err / n
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        gW[-1] = acts[-1].T @ grad_h
        gb[-1] = grad_h.sum(0)
        back = grad_h @ self.W[-1].T
        for i in range(len(self.W) - 2, -1, -1):
            back = back * dsilu(pre[i])
            gW[i] = acts[i].T @ back
            gb[i] = back.sum(0)
            back = back @ self.W[i].T
        self.t += 1
        for j, (p, g) in enumerate(zip(self.W + self.b, gW + gb)):
            self.m[j] = 0.9 * self.m[j] + 0.1 * g
            self.v[j] = 0.999 * self.v[j] + 0.001 * g * g
            mh = self.m[j] / (1 - 0.9**self.t)
            vh = self.v[j] / (1 - 0.999**self.t)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)
        return loss

    def score(self, x, sigma, label):
        s = np.full(len(x), sigma, np.float32)
        lab = np.full(len(x), label)
        return -self.forward(self.features(x.astype(np.float32), s, lab)) / sigma


def train(c, seed=0):
    rng = np.random.default_rng(1000 + seed)
    data = [rng.normal(size=(N_TRAIN, 2)) * np.sqrt(variances(l, c)) for l in range(3)]
    net = Net(rng)
    t0 = time.time()
    tail = []
    for it in range(TRAIN_STEPS):
        lab = rng.integers(0, 3, BATCH)
        idx = rng.integers(0, N_TRAIN, BATCH)
        x0 = np.empty((BATCH, 2))
        for l in range(3):
            m = lab == l
            x0[m] = data[l][idx[m]]
        sigma = np.exp(rng.uniform(np.log(SIGMA_MIN), np.log(SIGMA_MAX * 1.2), BATCH))
        z = rng.normal(size=(BATCH, 2))
        xt = x0 + sigma[:, None] * z
        lr = 1e-3 * (0.05 ** (it / TRAIN_STEPS))
        loss = net.step(net.features(xt, sigma, lab), z.astype(np.float32), lr)
        tail.append(loss)
        if (it + 1) % 3500 == 0:
            print(f"train c={c} step {it+1} loss {np.mean(tail[-500:]):.4f} t={time.time()-t0:.0f}s", flush=True)
    return net, data


def scores(x, sigma, c, net):
    if net is None:
        u = sigma**2
        return [-x / (np.array(variances(l, c)) + u) for l in range(3)]
    return [net.score(x, sigma, l).astype(np.float64) for l in range(3)]


def resample(G, K, rng):
    """Systematic resampling within each group of K particles."""
    G = G.reshape(-1, K)
    w = np.exp(G - G.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    cum = np.cumsum(w, 1)
    cum[:, -1] = 1.0
    pos = (rng.uniform(size=(len(G), 1)) + np.arange(K)) / K
    idx = (pos[:, :, None] > cum[:, None, :]).sum(2)
    return (idx + K * np.arange(len(G))[:, None]).ravel()


def sample(c, net, K, seed, snaps=None):
    rng = np.random.default_rng(seed)
    sig = np.geomspace(SIGMA_MAX, SIGMA_MIN, N_STEPS + 1)
    x = rng.normal(size=(N_OUT * K, 2)) * np.sqrt(target_var(c, SIGMA_MAX**2))
    saved = {}
    next_snaps = sorted(snaps or [], reverse=True)
    for i in range(N_STEPS):
        while next_snaps and sig[i] <= next_snaps[0]:
            saved[next_snaps.pop(0)] = (float(sig[i]), x[: 350 * K : K].copy(), float(x.var()))
        s0, s1, s2 = scores(x, float(sig[i]), c, net)
        s = s1 + s2 - s0
        du = sig[i] ** 2 - sig[i + 1] ** 2
        if K > 1:
            G = 0.5 * du * ((s**2).sum(1) - (s1**2).sum(1) - (s2**2).sum(1) + (s0**2).sum(1))
            G = np.nan_to_num(G, nan=-1e6, posinf=1e6, neginf=-1e6)
            idx = resample(G, K, rng)
            x, s = x[idx], s[idx]
        x = x + du * s + np.sqrt(du) * rng.normal(size=x.shape)
    if snaps:
        saved[min(snaps)] = (float(sig[-1]), x[: 350 * K : K].copy(), float(x.var()))
    return x[::K], saved


def sw2(a, b, rng, n_proj=256):
    th = rng.uniform(0, np.pi, n_proj)
    d = np.stack([np.cos(th), np.sin(th)], 1)
    pa, pb = np.sort(a @ d.T, 0), np.sort(b @ d.T, 0)
    return float(np.sqrt(((pa - pb) ** 2).mean()))


def main():
    t0 = time.time()
    nets, train_data = {}, {}
    for c in (1.0, 1.25):
        nets[c], train_data[c] = train(c)
    cells, points = [], {}
    snapshots = None
    for c, sampler_on in ((1.0, False), (1.25, True)):
        for learned in (False, True):
            for corrected in (False, True):
                K = K_FKC if corrected else 1
                dist, varx, vary = [], [], []
                for seed in SEEDS:
                    want = [8, 4, 2, 1, 0.5, 0.25, 0.1, 0.02] if (sampler_on and not learned and not corrected and seed == 0) else None
                    x, snap = sample(c, nets[c] if learned else None, K, seed, want)
                    rng = np.random.default_rng(5000 + seed)
                    ref = rng.normal(size=(N_OUT, 2)) * np.sqrt(target_var(c))
                    dist.append(sw2(x, ref, rng))
                    varx.append(float(x[:, 0].var()))
                    vary.append(float(x[:, 1].var()))
                    if seed == 0:
                        points[f"{int(sampler_on)}{int(learned)}{int(corrected)}"] = np.round(x[:450], 2).tolist()
                    if want:
                        snapshots = [
                            {"sigma": v[0], "target_var": target_var(c, v[0] ** 2), "sampled_var": v[2], "pts": np.round(v[1], 2).tolist()}
                            for k, v in sorted(snap.items(), key=lambda kv: -kv[0])
                        ]
                cells.append({
                    "key": f"{int(sampler_on)}{int(learned)}{int(corrected)}",
                    "base_var": c, "target_var": target_var(c), "sampler_error": sampler_on,
                    "score_error": learned, "corrector": corrected, "K": K,
                    "sw2_mean": float(np.mean(dist)), "sw2_sd": float(np.std(dist, ddof=1)),
                    "sw2_seeds": dist, "var_x": float(np.mean(varx)), "var_y": float(np.mean(vary)),
                })
                print(cells[-1]["key"], f"SW2 {np.mean(dist):.3f} +- {np.std(dist, ddof=1):.3f} var {np.mean(varx):.2f},{np.mean(vary):.2f} t={time.time()-t0:.0f}s", flush=True)

    floor = []
    for seed in SEEDS:
        rng = np.random.default_rng(7000 + seed)
        a = rng.normal(size=(N_OUT, 2)) * np.sqrt(10.0)
        b = rng.normal(size=(N_OUT, 2)) * np.sqrt(10.0)
        floor.append(sw2(a, b, rng))

    # Score error field for the factorized setting, composed score, at several noise levels.
    grid = np.linspace(-9, 9, 37)
    gx, gy = np.meshgrid(grid, grid)
    g = np.stack([gx.ravel(), gy.ravel()], 1)
    fields = {}
    rng = np.random.default_rng(11)
    tgt = rng.normal(size=(20000, 2)) * np.sqrt(10.0)
    def covered(p):
        m = [(p**2 / np.array(variances(l, 1.0))).sum(1) <= 9.21 for l in range(3)]
        return m[0] | m[1] | m[2]
    cov_t = covered(tgt)
    for sigma in (0.1, 0.5, 1.0, 2.0):
        def err(p):
            s = scores(p, sigma, 1.0, nets[1.0])
            t = scores(p, sigma, 1.0, None)
            return np.linalg.norm((s[1] + s[2] - s[0]) - (t[1] + t[2] - t[0]), axis=1)
        e_t = err(tgt)
        fields[str(sigma)] = {
            "err": np.round(err(g).reshape(37, 37), 3).tolist(),
            "mean_err_covered": float(e_t[cov_t].mean()),
            "mean_err_uncovered": float(e_t[~cov_t].mean()),
        }
        print(f"sigma {sigma}: composed score error, covered {e_t[cov_t].mean():.3f}, uncovered {e_t[~cov_t].mean():.3f}", flush=True)

    # Theorem 3 check: the log-weight drift is zero in the factorized setting with exact scores.
    drift = []
    for sigma in np.geomspace(SIGMA_MAX, SIGMA_MIN, 50):
        p = np.random.default_rng(3).normal(size=(2000, 2)) * np.sqrt(target_var(1.0, sigma**2))
        s0, s1, s2 = scores(p, float(sigma), 1.0, None)
        s = s1 + s2 - s0
        drift.append(float(np.abs((s**2).sum(1) - (s1**2).sum(1) - (s2**2).sum(1) + (s0**2).sum(1)).max()))

    out = {
        "cells": cells, "points": points, "snapshots": snapshots, "fields": fields, "grid": grid.tolist(),
        "train": {str(c): [np.round(d[:260], 2).tolist() for d in train_data[c]] for c in train_data},
        "target_samples": {"1.0": np.round(np.random.default_rng(21).normal(size=(450, 2)) * np.sqrt(target_var(1.0)), 2).tolist(),
                           "1.25": np.round(np.random.default_rng(21).normal(size=(450, 2)) * np.sqrt(target_var(1.25)), 2).tolist()},
        "share_target_uncovered": float((~cov_t).mean()),
        "sw2_floor_mean": float(np.mean(floor)), "sw2_floor_sd": float(np.std(floor, ddof=1)),
        "theorem3_max_abs_drift": float(max(drift)),
        "config": {"sigma_max": SIGMA_MAX, "sigma_min": SIGMA_MIN, "steps": N_STEPS, "n_out": N_OUT, "K": K_FKC,
                   "seeds": len(SEEDS), "n_train_per_condition": N_TRAIN, "train_steps": TRAIN_STEPS, "width": WIDTH},
    }
    with open(os.path.join(HERE, "toy_results.json"), "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"floor {np.mean(floor):.3f} uncovered {(~cov_t).mean():.3f} thm3 drift {max(drift):.2e} total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
