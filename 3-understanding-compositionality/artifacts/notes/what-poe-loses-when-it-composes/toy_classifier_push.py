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
    return 0.5 * ((weight_real * F.softplus(-logit_real)).mean() + (weight_fake * F.softplus(logit_fake)).mean())


def log_odds_grad(net, x, k):
    """The push: the gradient of log(d / (1 - d)) with respect to x_t, where d = sigmoid(net(x, k)).

    x  torch tensor (B, D) of noised pictures; k  torch tensor (B,) of steps
    returns a torch tensor (B, D)
    """
    x = x.detach().requires_grad_(True)
    g, = torch.autograd.grad(net(x, k)[:, 0].sum(), x)
    return g


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


# ---------------------------------------------------------------------------------------------------
# Piece 4: guide the sampler with the pair-aware push, and the bar
# ---------------------------------------------------------------------------------------------------
WEIGHTS = [0.5, 1.0, 2.0]          # the bar is read at w = 1, fixed before the run
LOGIT_CLIP = float(np.log((1 - 1e-5) / 1e-5))   # Kim et al. clip d to [1e-5, 1 - 1e-5]; past it the push is 0


def pair_push():
    net = Net(1).to(DEV)
    net.load_state_dict(torch.load(OUT / "pair-aware-discriminator.pt", map_location=DEV, weights_only=True))
    net.eval()

    def push(x, k):
        xt = torch.tensor(x, dtype=torch.float32, device=DEV)
        kk = torch.full((len(x),), k, device=DEV, dtype=torch.long)
        g = log_odds_grad(net, xt, kk)
        with torch.no_grad():
            inside = (net(xt, kk)[:, 0].abs() < LOGIT_CLIP).float()[:, None]
        return (g * inside).double().cpu().numpy()
    return push


def pair_guide():
    plain = sample()
    rows = {"unguided": counts(plain)}
    push = pair_push()
    keep = {"unguided": plain}
    for w in WEIGHTS:
        x = sample(push, w)
        rows[f"pair-aware, w {w:g}"] = counts(x)
        keep[f"w{w:g}"] = x
    for name, c in rows.items():
        print(f"{name:20s} crisp {c['two objects, crisp']:3d}  overlaid {c['hybrid: overlaid']:3d}  "
              f"faint {c['two faint objects']:3d}  single {c['one circle'] + c['one square']:3d}", flush=True)
    c1 = rows["pair-aware, w 1"]
    ok = c1["two objects, crisp"] >= 150 and c1["hybrid: overlaid"] <= rows["unguided"]["hybrid: overlaid"]
    print(f"bar, pair-aware at w = 1: crisp {c1['two objects, crisp']} (at least 150), overlaid "
          f"{c1['hybrid: overlaid']} (no more than unguided's {rows['unguided']['hybrid: overlaid']}) -> "
          f"{'pass' if ok else 'miss'}")
    np.savez_compressed(OUT / "pair-aware-guide-samples.npz", **keep)
    (OUT / "pair-aware-guide.json").write_text(json.dumps(
        {"what": "reverse diffusion on the summed scores with the pair-aware discriminator's log-odds gradient "
                 "as the push, at weights 0.5, 1 and 2, every step, uncapped, the logit clipped at +-11.5 as Kim "
                 "et al.; seed 7, 300 draws",
         "pair_weight": PAIR_WEIGHT, "seed": SEED, "counts": rows,
         "bar": {"rule": "at w = 1, at least 150 of 300 crisp pairs and overlaid no more than unguided",
                 "crisp": c1["two objects, crisp"], "overlaid": c1["hybrid: overlaid"],
                 "verdict": "pass" if ok else "miss"}}, indent=1))


# ---------------------------------------------------------------------------------------------------
# Piece 5: the diagnosis
#   1. along the guided run at w = 1, the push's cosine with two targets, every 10 steps:
#      the true correction (true noised-product score minus summed scores), and
#      the sampler's own ratio (true noised-product score minus the score of the sampler's finished
#      draws noised forward, an exact Gaussian mixture over the 3,500 draws in discriminator-data.npz)
#   2. the minimal-pair gap: the log-odds on a side-by-side picture minus on a stacked one at the same
#      positions' average, same noise, every 10 steps, against a discriminator trained on shuffled labels
#   3. the cap sweep: the push capped at kappa times the size of the product's own step
# ---------------------------------------------------------------------------------------------------
DIAG_STEPS = list(range(200, 0, -10))
KAPPAS = [0.1, 0.3, 1.0]


def sampler_ratio_score(x, k, fake_t):
    """Score of the sampler's finished draws noised to step k: a Gaussian mixture, one component per draw."""
    a = t.abar[k]
    X = torch.tensor(x, dtype=torch.float64, device=DEV)
    M = np.sqrt(a) * fake_t                                        # the components' means
    d2 = (X ** 2).sum(1, keepdim=True) - 2 * X @ M.T + (M ** 2).sum(1)[None]
    r = torch.softmax(-d2 / (2 * (1 - a)), 1)
    return (-(X - r @ M) / (1 - a)).cpu().numpy()


def cosine(u, v):
    return (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-300)


def train_shuffled():
    """The same recipe with real and fake labels shuffled: the floor a gap has to clear."""
    real, fake = disc_data()
    both = np.concatenate([real[:N_TRAIN], fake[:N_TRAIN]])
    perm = np.random.default_rng(3).permutation(len(both))
    R = torch.tensor(both[perm[:N_TRAIN]], dtype=torch.float32, device=DEV)
    Fk = torch.tensor(both[perm[N_TRAIN:]], dtype=torch.float32, device=DEV)
    torch.manual_seed(1)
    gen = torch.Generator(device=DEV).manual_seed(1)
    net = Net(1).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(4000):
        ir = torch.randint(N_TRAIN, (512,), device=DEV, generator=gen)
        jf = torch.randint(N_TRAIN, (512,), device=DEV, generator=gen)
        kr = torch.randint(1, t.T + 1, (512,), device=DEV, generator=gen)
        kf = torch.randint(1, t.T + 1, (512,), device=DEV, generator=gen)
        loss = disc_loss(net(noise_to(R[ir], kr, gen), kr)[:, 0], net(noise_to(Fk[jf], kf, gen), kf)[:, 0],
                         LAMBDA[kr], LAMBDA[kf])
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


def capped(push, kappa):
    """Cap the push at kappa times the size of the summed score, draw by draw (w = 1)."""
    def p(x, k):
        g = push(x, k)
        lim = kappa * np.linalg.norm(summed(x, k), axis=1)
        f = np.minimum(1.0, lim / np.maximum(np.linalg.norm(g, axis=1), 1e-300))
        return g * f[:, None]
    return p


def diagnose():
    real, fake = disc_data()
    fake_t = torch.tensor(fake, dtype=torch.float64, device=DEV)
    push = pair_push()

    # 1. cosines along the guided run at w = 1, recorded in a hook that does not change the run
    rec = {}

    def watched(x, k):
        g = push(x, k)
        if k in DIAG_STEPS:
            tc = true_correction(x, k)
            sr = TRUE.score(x, k) - sampler_ratio_score(x, k, fake_t)
            rec[k] = {"true correction": cosine(g, tc), "sampler's own ratio": cosine(g, sr),
                      "push size / summed score size": np.linalg.norm(g, axis=1) / np.linalg.norm(summed(x, k), axis=1),
                      "true correction size / summed score size": np.linalg.norm(tc, axis=1) / np.linalg.norm(summed(x, k), axis=1)}
        return g
    x_w1 = sample(watched, 1.0)
    cos_rows = {k: {n: [float(np.percentile(v, q)) for q in (25, 50, 75)] for n, v in rec[k].items()} for k in DIAG_STEPS}
    print("median over 300 draws, along the guided run at w = 1")
    print("step   cos with true correction   cos with sampler's own ratio   push / summed   true corr / summed")
    for k in DIAG_STEPS[::2]:
        r = cos_rows[k]
        tc, sr = r["true correction"][1], r["sampler's own ratio"][1]
        ps, ts = r["push size / summed score size"][1], r["true correction size / summed score size"][1]
        print(f"{k:4d}   {tc:6.3f}                     {sr:6.3f}                         {ps:6.3f}          {ts:6.3f}")

    # 2. the minimal-pair gap, with the shuffled-label floor and the separate-label classifier beside it
    shuffled = train_shuffled()
    sep = Net(2).to(DEV)
    sep.load_state_dict(torch.load(OUT / "separate-label-classifier.pt", map_location=DEV, weights_only=True)); sep.eval()
    pair_net = Net(1).to(DEV)
    pair_net.load_state_dict(torch.load(OUT / "pair-aware-discriminator.pt", map_location=DEV, weights_only=True)); pair_net.eval()
    side = torch.tensor(t.PAIR, dtype=torch.float32, device=DEV)
    stacked = torch.tensor(np.mean([(t.CIRCLES[p] + t.SQUARES[p]) / 2 for p in t.XPOS], 0), dtype=torch.float32, device=DEV)
    stacked_one = torch.tensor((t.CIRCLES["centre"] + t.SQUARES["centre"]) / 2, dtype=torch.float32, device=DEV)
    gen = torch.Generator(device=DEV).manual_seed(5)
    gap = {}
    with torch.no_grad():
        for k in DIAG_STEPS:
            kk = torch.full((256,), k, device=DEV)
            e = torch.randn(256, t.D, device=DEV, generator=gen)
            a = ABAR[k]
            xs, xo = a.sqrt() * side + (1 - a).sqrt() * e, a.sqrt() * stacked_one + (1 - a).sqrt() * e
            gap[k] = {"pair-aware discriminator": float((pair_net(xs, kk) - pair_net(xo, kk)).mean()),
                      "shuffled labels": float((shuffled(xs, kk) - shuffled(xo, kk)).mean()),
                      "separate-label classifier": float((F.logsigmoid(sep(xs, kk)).sum(1) - F.logsigmoid(sep(xo, kk)).sum(1)).mean())}
    print("minimal-pair gap: log-odds on the side-by-side pair minus on the circle stacked on the square (centre), "
          "mean over 256 shared noise draws")
    print("step   pair-aware   shuffled labels   separate-label")
    for k in DIAG_STEPS[::2]:
        g = gap[k]
        print(f"{k:4d}   {g['pair-aware discriminator']:9.2f}   {g['shuffled labels']:14.2f}   {g['separate-label classifier']:13.2f}")

    # 3. the cap sweep, the pair-aware push and the true correction
    caps = {}
    for kappa in KAPPAS:
        caps[f"pair-aware push, w 1, cap {kappa:g}"] = counts(sample(capped(push, kappa), 1.0))
        caps[f"true correction, cap {kappa:g}"] = counts(sample(capped(true_correction, kappa), 1.0))
    print("cap sweep, crisp pairs of 300 (overlaid)")
    for n, c in caps.items():
        print(f"  {n:34s} {c['two objects, crisp']:3d}  ({c['hybrid: overlaid']})")

    # 4. the clip: share of draws whose logit is past +-11.5 (push set to 0), split by how the draw ends
    logits = {}
    def logged(x, k):
        with torch.no_grad():
            logits[k] = pair_net(torch.tensor(x, dtype=torch.float32, device=DEV),
                                 torch.full((len(x),), k, device=DEV))[:, 0].cpu().numpy()
        return push(x, k)
    ends_crisp = np.array(t.label(sample(logged, 1.0))) == "two objects, crisp"
    clip = {k: {"ends crisp: clipped high": float(np.mean(logits[k][ends_crisp] >= LOGIT_CLIP)),
                "ends not crisp: clipped low": float(np.mean(logits[k][~ends_crisp] <= -LOGIT_CLIP))}
            for k in DIAG_STEPS}
    print("the clip, share of draws whose push is set to 0")
    for k in DIAG_STEPS[::2]:
        print(f"{k:4d}   ends crisp {clip[k]['ends crisp: clipped high']:.2f}   ends not crisp {clip[k]['ends not crisp: clipped low']:.2f}")

    json.dump({"what": "piece 5 diagnosis on the exact toy at pair weight 0.01, seed 7, 300 draws",
               "clip_share_by_final_label": clip,
               "cosines_along_guided_run_w1_percentiles_25_50_75": cos_rows,
               "guided_run_w1_counts": counts(x_w1),
               "minimal_pair_gap": gap, "cap_sweep": caps,
               "notes": {"sampler's own ratio": "true noised-product score minus the score of the 3,500 finished sampler "
                                               "draws noised forward, an exact mixture over those draws; at small steps "
                                               "the mixture is one spike per draw, so read the late steps loosely",
                         "stacked picture": "the circle and the square averaged pixel by pixel at the centre position"}},
              open(OUT / "diagnosis.json", "w"), indent=1)
    figures_for_diagnosis(cos_rows, gap)
    print("wrote", OUT / "diagnosis.json")


def figures_for_diagnosis(cos_rows, gap):
    ks = DIAG_STEPS
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for name, col in [("sampler's own ratio", "#c86a0a"), ("true correction", "#1f5fd0")]:
        q = np.array([cos_rows[k][name] for k in ks])
        ax.fill_between(ks, q[:, 0], q[:, 2], color=col, alpha=0.18, lw=0)
        ax.plot(ks, q[:, 1], color=col, lw=2)
        ax.text(ks[2], q[2, 1] + 0.04, f"against the {name}", color=col, fontsize=9)
    ax.set_xlim(200, 0); ax.set_ylim(-0.1, 1.0)
    ax.set_xlabel("reverse step (200 is pure noise, 0 the finished picture)")
    ax.set_ylabel("cosine of the push with the target")
    ax.set_title("The discriminator's push along the guided run at w = 1: median of 300 draws, band 25th to 75th", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "push-cosine-per-step.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for name, col in [("pair-aware discriminator", "#c86a0a"), ("separate-label classifier", "#7a4fb0"),
                      ("shuffled labels", "#888888")]:
        v = [gap[k][name] for k in ks]
        ax.plot(ks, v, color=col, lw=2)
        ax.text(ks[3], v[3] + (-1.6 if name == "shuffled labels" else 0.6), name, color=col, fontsize=9)
    ax.axhline(0, color="#bbbbbb", lw=1)
    ax.set_xlim(200, 0)
    ax.set_xlabel("noise step (200 is pure noise)")
    ax.set_ylabel("log-odds: side by side minus stacked")
    ax.set_title("Minimal-pair gap: how much more each network prefers the pair to the stacked picture, same noise", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "minimal-pair-gap-per-step.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------------------------------
# Beside the bar, not deciding it
#   inclusive: the inclusive-captions route on the toy. A picture holding both shapes is also captioned
#              with each single name, which on this toy is the pair's weight inside each expert. The
#              same sampler on the same seed-7 noises, no push, at rising pair weights.
#   no-clip:   the pair-aware push at w = 1 and 2 with Kim et al.'s clip removed
# ---------------------------------------------------------------------------------------------------
INCLUSIVE_WEIGHTS = [0.01, 0.03, 0.1, 0.2, 0.3, 0.5]


def inclusive():
    rows = {}
    for pw in INCLUSIVE_WEIGHTS:
        _, A2, B2 = t.hand_weighted(pw)
        x = t.reverse(lambda x, k: A2.score(x, k) + B2.score(x, k), rng=np.random.default_rng(SEED))
        rows[f"{pw:g}"] = counts(x)
        c = rows[f"{pw:g}"]
        print(f"pair weight {pw:<5g} no push   crisp {c['two objects, crisp']:3d}  overlaid {c['hybrid: overlaid']:3d}", flush=True)
    (OUT / "inclusive-captions-comparison.json").write_text(json.dumps(
        {"what": "the summed-score sampler with no push at rising pair weights (the share of each expert's mass "
                 "on the side-by-side picture, the toy's version of captioning a both-shapes picture with each "
                 "single name), seed 7, 300 draws; compare with the pair-aware push at pair weight 0.01 in "
                 "pair-aware-guide.json",
         "seed": SEED, "counts_by_pair_weight": rows}, indent=1))


def no_clip():
    net = Net(1).to(DEV)
    net.load_state_dict(torch.load(OUT / "pair-aware-discriminator.pt", map_location=DEV, weights_only=True))
    net.eval()

    def push(x, k):
        xt = torch.tensor(x, dtype=torch.float32, device=DEV)
        return log_odds_grad(net, xt, torch.full((len(x),), k, device=DEV, dtype=torch.long)).double().cpu().numpy()
    rows = {}
    for w in [1.0, 2.0]:
        rows[f"w {w:g}, no clip"] = counts(sample(push, w))
        c = rows[f"w {w:g}, no clip"]
        print(f"pair-aware, w {w:g}, no clip   crisp {c['two objects, crisp']:3d}  overlaid {c['hybrid: overlaid']:3d}  "
              f"faint {c['two faint objects']:3d}  single {c['one circle'] + c['one square']:3d}", flush=True)
    (OUT / "pair-aware-no-clip.json").write_text(json.dumps(
        {"what": "the pair-aware push with the logit clip removed, seed 7, 300 draws, pair weight 0.01; beside the "
                 "bar, which is read with the clip in pair-aware-guide.json", "seed": SEED, "counts": rows}, indent=1))


if __name__ == "__main__":
    {"port": port, "separate-train": separate_train, "separate-guide": separate_guide,
     "disc-train": disc_train, "pair-guide": pair_guide, "diagnose": diagnose,
     "inclusive": inclusive, "no-clip": no_clip}[sys.argv[1]]()
