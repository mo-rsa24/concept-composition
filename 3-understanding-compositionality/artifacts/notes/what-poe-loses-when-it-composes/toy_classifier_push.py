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


# ---------------------------------------------------------------------------------------------------
# Piece 3: the pair-aware discriminator, Kim et al. 2023's recipe
# Real: exact product draws (side by side). Fake: the summed-score sampler's finished draws (stacked).
# Both noised forward to a step drawn uniformly from 1 to T, the loss weighted per step by
# lambda = beta_k / (1 - abar_k), Kim's g^2 / sigma^2, normalised to mean 1 over the 200 steps.
# ---------------------------------------------------------------------------------------------------
N_TRAIN, N_HELD = 3000, 500
_k = np.arange(1, t.T + 1)
_lam = (1 - t.abar[_k] / t.abar[_k - 1]) / (1 - t.abar[_k])
LAMBDA = torch.tensor(np.concatenate([[0.0], _lam / _lam.mean()]), dtype=torch.float32, device=DEV)


def exact_draws(n, seed):
    """n draws from the exact product at step 0: a mixture over every pairing, variance sigma0^2 / 2."""
    rng = np.random.default_rng(seed)
    w = np.exp(TRUE.logw - TRUE.logw.max()); w /= w.sum()
    idx = rng.choice(len(TRUE.mu), size=n, p=w)
    return TRUE.mu[idx] + np.sqrt(TRUE.v0) * rng.standard_normal((n, t.D))


def disc_data():
    """3,500 exact draws and 3,500 sampler draws, 3,000 of each to train on and 500 held out.

    Seeds 101 and 201 to 205, never seed 7, which every count in this walk is judged on.
    """
    f = OUT / "discriminator-data.npz"
    if not f.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        real = exact_draws(N_TRAIN + N_HELD, 101)
        fake = np.concatenate([t.reverse(summed, rng=np.random.default_rng(201 + i), n=700) for i in range(5)])
        np.savez_compressed(f, real=real, fake=fake,
                            real_counts=json.dumps(counts(real)), fake_counts=json.dumps(counts(fake)))
    d = np.load(f)
    return d["real"], d["fake"]


def noise_to(x0, k, gen):
    a = ABAR[k][:, None]
    return a.sqrt() * x0 + (1 - a).sqrt() * torch.randn(x0.shape, device=DEV, generator=gen)


def disc_loss(logit_real, logit_fake, weight_real, weight_fake):
    """Binary cross-entropy with real labelled 1 and fake labelled 0, each term weighted by lambda at its step.

    logit_real, logit_fake    the network's output, log(d / (1 - d)), on a batch of each, shape (B,)
    weight_real, weight_fake  lambda at each example's step, shape (B,)
    returns one number: the mean over both halves of the batch
    """
    raise NotImplementedError("type the discriminator's loss here")


def log_odds_grad(net, x, k):
    """The push: the gradient of log(d / (1 - d)) with respect to x_t, where d = sigmoid(net(x, k)).

    x  torch tensor (B, D) of noised pictures; k  torch tensor (B,) of steps
    returns a torch tensor (B, D)
    """
    raise NotImplementedError("type the log-odds gradient here")


def disc_train():
    real, fake = disc_data()
    R = torch.tensor(real[:N_TRAIN], dtype=torch.float32, device=DEV)
    Fk = torch.tensor(fake[:N_TRAIN], dtype=torch.float32, device=DEV)
    torch.manual_seed(1)
    gen = torch.Generator(device=DEV).manual_seed(1)
    net = Net(1).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    B = 512
    for it in range(4000):
        ir = torch.randint(N_TRAIN, (B,), device=DEV, generator=gen)
        jf = torch.randint(N_TRAIN, (B,), device=DEV, generator=gen)
        kr = torch.randint(1, t.T + 1, (B,), device=DEV, generator=gen)
        kf = torch.randint(1, t.T + 1, (B,), device=DEV, generator=gen)
        loss = disc_loss(net(noise_to(R[ir], kr, gen), kr)[:, 0], net(noise_to(Fk[jf], kf, gen), kf)[:, 0],
                         LAMBDA[kr], LAMBDA[kf])
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 1000 == 0 or it == 3999:
            print(f"iteration {it:4d}  loss {loss.item():.4f}", flush=True)
    torch.save(net.state_dict(), OUT / "pair-aware-discriminator.pt")

    Rh = torch.tensor(real[N_TRAIN:], dtype=torch.float32, device=DEV)
    Fh = torch.tensor(fake[N_TRAIN:], dtype=torch.float32, device=DEV)
    table = {}
    with torch.no_grad():
        for s in CHECK_STEPS:
            k = torch.full((N_HELD,), s, device=DEV)
            lr_, lf = net(noise_to(Rh, k, gen), k)[:, 0], net(noise_to(Fh, k, gen), k)[:, 0]
            table[s] = {"real called real": float((lr_ > 0).float().mean()),
                        "fake called fake": float((lf < 0).float().mean())}
    # the typed log-odds gradient against a central finite difference, on 4 held-out pairs at step 100
    k = torch.full((4,), 100, device=DEV)
    x = noise_to(Rh[:4], k, gen)
    v = torch.randn(x.shape, device=DEV, generator=gen); v /= v.norm(dim=1, keepdim=True)
    g = log_odds_grad(net, x, k)
    h = 1e-2
    with torch.no_grad():
        fd = (net(x + h * v, k)[:, 0] - net(x - h * v, k)[:, 0]) / (2 * h)
    rel = float(((g * v).sum(1) - fd).abs().max() / fd.abs().max())
    print(f"log-odds gradient against a finite difference, 4 pairs at step 100: largest relative error {rel:.1e}")
    table["gradient_check_relative_error"] = rel

    print("held-out accuracy, 500 exact pairs and 500 sampler draws never trained on")
    print("step   pairs called real   sampler draws called fake")
    for s in CHECK_STEPS:
        r = table[s]
        print(f"{s:4d}   {r['real called real']:.3f}              {r['fake called fake']:.3f}")
    (OUT / "pair-aware-train.json").write_text(json.dumps(
        {"what": "a one-output MLP trained with lambda-weighted binary cross-entropy: 3,000 exact product draws "
                 "(real) against 3,000 finished draws of the summed-score sampler (fake), each noised to a step "
                 "uniform 1 to 200, 4,000 iterations of 512 + 512; accuracy on 500 + 500 held out per step",
         "pair_weight": PAIR_WEIGHT, "accuracy": table}, indent=1))


if __name__ == "__main__":
    {"port": port, "separate-train": separate_train, "separate-guide": separate_guide,
     "disc-train": disc_train}[sys.argv[1]]()
