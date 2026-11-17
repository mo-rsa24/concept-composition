"""Can a classifier's push fix the product? guided-diffusion's push, ported to the exact toy.

The toy is condition D of toy_sampler_hybrids.py at pair weight 0.01: the exact product draws 300 of
300 crisp pairs and reverse diffusion on the summed noised scores draws 0. The push enters the
sampler through toy_sampler_hybrids.guided_eps, the condition_score form of guided-diffusion
(gaussian_diffusion.py:371): eps <- eps - scale * sqrt(1 - abar_t) * grad log c(x_t).

Pieces, one subcommand each:
  port   scale 0 against unguided draw by draw, and the true correction as the push

    /home/molef/micromamba/envs/sdxl-local/bin/python -I toy_classifier_push.py port
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toy_sampler_hybrids as t           # the experts, the schedule, the sampler, the labelling library
import toy_noised_product as tn           # the true noised product, closed form

PAIR_WEIGHT = 0.01
SEED = 7                                  # every count in this walk draws its starting noise from this seed
OUT = t.OUT.parents[1] / "results" / "the-exact-toy" / "can-a-classifier-push-fix-the-product"

_, A, B = t.hand_weighted(PAIR_WEIGHT)
TRUE = tn.TrueProduct(A, B)


def summed(x, k):
    """The product's score as practised: the two experts' noised scores added."""
    return A.score(x, k) + B.score(x, k)


def true_correction(x, k):
    """The score of the true noised product minus the summed scores: what the sampler drops."""
    return TRUE.score(x, k) - summed(x, k)


def counts(x):
    labs = t.label(x)
    return {c: labs.count(c) for c in t.CATS}


def sample(push=None, scale=0.0, score=summed):
    return t.reverse(score, push=push, scale=scale, rng=np.random.default_rng(SEED))


def port():
    OUT.mkdir(parents=True, exist_ok=True)
    x_plain = sample()

    # Bit-for-bit, in the same loop: at scale 0 the guided step must hand back the product's own noise
    # unchanged at every step. Two separate runs are compared too, but numpy's summation order can
    # move the last bit between processes, so that comparison is reported, not judged bitwise.
    same = {"steps": 0, "bitwise": 0}
    typed = t.guided_eps

    def checked(eps, grad, a_t, scale):
        out = typed(eps, grad, a_t, scale)
        same["steps"] += 1
        same["bitwise"] += int(np.array_equal(out, eps))
        return out

    t.guided_eps = checked
    try:
        x_s0 = sample(true_correction, 0.0)
    finally:
        t.guided_eps = typed
    x_s1 = sample(true_correction, 1.0)
    x_true = sample(score=TRUE.score)
    x_half = sample(true_correction, 0.5)

    differ = int((np.abs(x_s0 - x_plain).max(1) > 0).sum())
    rows = {
        "unguided": counts(x_plain),
        "true correction, scale 0": counts(x_s0),
        "true correction, scale 0.5": counts(x_half),
        "true correction, scale 1": counts(x_s1),
        "true noised-product score, no push": counts(x_true),
    }
    checks = {
        f"scale 0, same loop: steps where the guided noise equals the product's bit for bit, of {same['steps']}":
            same["bitwise"],
        "scale 0 against a separate unguided run: draws that differ, of 300": differ,
        "scale 0 against unguided: largest pixel difference": float(np.abs(x_s0 - x_plain).max()),
        "scale 1 against the true-score sampler: largest pixel difference": float(np.abs(x_s1 - x_true).max()),
    }
    for k, v in checks.items():
        print(f"{k}: {v}")
    for name, c in rows.items():
        print(f"{name:38s} crisp {c['two objects, crisp']:3d}  overlaid {c['hybrid: overlaid']:3d}  "
              f"faint {c['two faint objects']:3d}  single {c['one circle'] + c['one square']:3d}")
    (OUT / "port.json").write_text(json.dumps(
        {"what": "the push hook checked: scale 0 against unguided draw by draw, and the true correction "
                 "(true noised-product score minus summed scores) entered as the push",
         "pair_weight": PAIR_WEIGHT, "seed": SEED, "draws": t.N, "steps": t.T,
         "checks": checks, "counts": rows}, indent=1))
    np.savez_compressed(OUT / "port-samples.npz", unguided=x_plain, scale0=x_s0, half=x_half,
                        scale1=x_s1, true_score=x_true)

    cols = [("unguided", x_plain), ("true correction, scale 0", x_s0), ("true correction, scale 1", x_s1)]
    fig, axes = plt.subplots(1, 3, figsize=(9, 1.9))
    for ax, (name, x) in zip(axes, cols):
        ax.imshow(np.clip(x[0], 0, 1).reshape(t.H, t.W, 3), interpolation="nearest")
        ax.set_title(f"{name}\n{t.label(x[:1])[0]}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"draw 0 of seed {SEED}, pair weight {PAIR_WEIGHT}", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "draw-0-unguided-and-pushed.png", dpi=150)
    print("wrote", OUT)


# ---------------------------------------------------------------------------------------------------
# Piece 2: the separate-label classifier, "circle present" and "square present"
# ---------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ABAR = torch.tensor(t.abar, dtype=torch.float32, device=DEV)
SCALES = [1.0, 2.0, 5.0, 10.0]
CHECK_STEPS = [10, 50, 100, 150, 190, 200]


def picture_bank():
    """The seven pictures, their weights in the data and their two labels.

    The data is half circle-expert draws and half square-expert draws, so each single shape carries
    (1 - w) / 6 and the pair carries w. Labels: circle present, square present.
    """
    mu = np.stack(list(t.CIRCLES.values()) + list(t.SQUARES.values()) + [t.PAIR])
    w = PAIR_WEIGHT
    pi = np.array([(1 - w) / 6] * 6 + [w])
    has = np.array([[1, 0]] * 3 + [[0, 1]] * 3 + [[1, 1]], dtype=np.float32)
    return mu, pi, has


MU, PI, HAS = picture_bank()
MU_T = torch.tensor(MU, dtype=torch.float32, device=DEV)
LOGPI_T = torch.log(torch.tensor(PI, dtype=torch.float32, device=DEV))
HAS_T = torch.tensor(HAS, device=DEV)


def noised_batch(n, gen, steps=None):
    """n clean draws from the bank, each noised forward to a step drawn uniformly from 1 to T."""
    idx = torch.multinomial(torch.tensor(PI, device=DEV, dtype=torch.float32), n, replacement=True, generator=gen)
    x0 = MU_T[idx] + t.SIGMA0 * torch.randn(n, t.D, device=DEV, generator=gen)
    k = steps if steps is not None else torch.randint(1, t.T + 1, (n,), device=DEV, generator=gen)
    a = ABAR[k][:, None]
    return a.sqrt() * x0 + (1 - a).sqrt() * torch.randn(n, t.D, device=DEV, generator=gen), k, HAS_T[idx]


def time_features(k):
    """Sixteen sine and cosine features of the step's log signal-to-noise ratio."""
    a = ABAR[k]
    lsnr = torch.log(a / (1 - a))[:, None]
    f = torch.exp(torch.linspace(0, 3, 8, device=DEV))[None] * lsnr / 10
    return torch.cat([f.sin(), f.cos()], 1)


class Net(nn.Module):
    """An MLP from scratch on the noised picture and its step, the paper's own LSGM and 2D-toy route."""

    def __init__(self, out):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(t.D + 16, 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(),
                                  nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, out))

    def forward(self, x, k):
        return self.body(torch.cat([x, time_features(k)], 1))


def best_log_probs(x, k):
    """The closed-form best classifier: log p(circle present | x_t) and log p(square present | x_t)."""
    a = ABAR[k][:, None]
    var = a * t.SIGMA0 ** 2 + 1 - a
    l = LOGPI_T[None] - 0.5 * ((x[:, None, :] - a.sqrt()[:, :, None] * MU_T[None]) ** 2).sum(-1) / var
    total = torch.logsumexp(l, 1)
    neg = torch.tensor(-1e30, device=DEV)
    return torch.stack([torch.logsumexp(torch.where(HAS_T[:, j][None] > 0, l, neg), 1) - total
                        for j in range(2)], 1)


def separate_train():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    gen = torch.Generator(device=DEV).manual_seed(0)
    net = Net(2).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for it in range(4000):
        x, k, y = noised_batch(1024, gen)
        loss = F.binary_cross_entropy_with_logits(net(x, k), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 1000 == 0 or it == 3999:
            print(f"iteration {it:4d}  loss {loss.item():.4f}", flush=True)
    torch.save(net.state_dict(), OUT / "separate-label-classifier.pt")

    table = {}
    with torch.no_grad():
        for s in CHECK_STEPS:
            x, k, y = noised_batch(4000, gen, torch.full((4000,), s, device=DEV))
            pt = torch.sigmoid(net(x, k)) > 0.5
            pb = best_log_probs(x, k).exp() > 0.5
            table[s] = {"trained": [float((pt[:, j] == (y[:, j] > 0)).float().mean()) for j in range(2)],
                        "best possible": [float((pb[:, j] == (y[:, j] > 0)).float().mean()) for j in range(2)]}
    print("accuracy on 4,000 fresh draws per step (circle present, square present)")
    print("step   trained          best possible")
    for s, r in table.items():
        print(f"{s:4d}   {r['trained'][0]:.3f}  {r['trained'][1]:.3f}     {r['best possible'][0]:.3f}  {r['best possible'][1]:.3f}")
    (OUT / "separate-label-train.json").write_text(json.dumps(
        {"what": "a two-output MLP trained with binary cross-entropy on noised draws from the seven pictures "
                 "at the experts' weights (each single shape (1 - w)/6, the pair w), steps uniform 1 to 200, "
                 "4,000 iterations of 1,024; accuracy per step on 4,000 fresh draws, beside the closed-form best",
         "pair_weight": PAIR_WEIGHT, "accuracy_circle_square": table}, indent=1))


def separate_push(which):
    """The push: the gradient of log p(circle present | x_t) + log p(square present | x_t)."""
    if which == "trained":
        net = Net(2).to(DEV)
        net.load_state_dict(torch.load(OUT / "separate-label-classifier.pt", map_location=DEV, weights_only=True))
        net.eval()
        logp = lambda x, k: F.logsigmoid(net(x, k))
    else:
        logp = best_log_probs

    def push(x, k):
        xt = torch.tensor(x, dtype=torch.float32, device=DEV, requires_grad=True)
        kk = torch.full((len(x),), k, device=DEV, dtype=torch.long)
        g, = torch.autograd.grad(logp(xt, kk).sum(), xt)
        return g.double().cpu().numpy()
    return push


def separate_guide():
    rows = {"unguided": counts(sample())}
    for which in ["trained", "best possible"]:
        push = separate_push(which)
        for s in SCALES:
            x = sample(push, s)
            rows[f"{which}, scale {s:g}"] = counts(x)
            if which == "trained" and s == 10.0:
                np.savez_compressed(OUT / "separate-label-scale10-samples.npz", x=x)
    for name, c in rows.items():
        print(f"{name:28s} crisp {c['two objects, crisp']:3d}  overlaid {c['hybrid: overlaid']:3d}  "
              f"faint {c['two faint objects']:3d}  single {c['one circle'] + c['one square']:3d}", flush=True)
    worst = max(c["two objects, crisp"] for n, c in rows.items() if n.startswith("trained"))
    print(f"bar, separate-label at most 15 of 300 at every scale: highest {worst} -> "
          f"{'pass' if worst <= 15 else 'miss'}")
    (OUT / "separate-label-guide.json").write_text(json.dumps(
        {"what": "reverse diffusion on the summed scores with the separate-label push at each scale, the "
                 "trained classifier and the closed-form best, seed 7, 300 draws, uncapped",
         "pair_weight": PAIR_WEIGHT, "seed": SEED, "counts": rows,
         "bar": {"rule": "trained separate-label push at most 15 of 300 crisp pairs at every scale in 1, 2, 5, 10",
                 "highest": worst, "verdict": "pass" if worst <= 15 else "miss"}}, indent=1))


if __name__ == "__main__":
    {"port": port, "separate-train": separate_train, "separate-guide": separate_guide}[sys.argv[1]]()
