"""Du et al.'s Figure 2, redone on red circles and blue squares: does the target or the sampler make hybrids?

Each condition is an ideal expert: an equal-weight mixture of Gaussians, spread SIGMA0, centred on its
training images (16 x 32 RGB, values 0 to 1). Under the variance-preserving forward process a mixture of
Gaussians stays one, so every diffused density and score is exact.

  A  product, singles only          "a red circle" x "a blue square", each trained on its shape alone
  B  product, "contains" captions   as A, plus the side-by-side pair image inside both conditions
  C  mixture (OR)                    Du's bottom row: circle or square
  D  product, pair at weight w        as B, but the pair image's weight inside both conditions is --pair-weight,
                                      the rest split equally over that condition's three single shapes
  E  product, competing captions       three captions score every image, softmax over captions gives p(caption|image),
                                      Bayes against a uniform image prior gives each caption's condition, and the two
                                      single-caption conditions are multiplied. --caption-form picks how a caption
                                      scores an image, --caption-tau its softmax temperature

For each, three ways to sample:
  exact            draw from the composed density at t = 0 directly (closed form)
  reverse diff.    DDPM ancestral steps on the summed diffused scores (PoE as practised; exact for C)
  annealed MALA    the same reverse steps, each followed by Metropolis-adjusted Langevin steps on the
                   product (or mixture) of the two diffused densities at that noise level

Each final sample is labelled by its nearest picture in a library of every single shape, the pair, and
every pixel-average of a circle and a square, so a blend is named as one.

    /home/molef/miniconda3/bin/python3 -I toy_sampler_hybrids.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

H, W = 16, 32
SIGMA0 = 0.1          # each expert's spread around its training images
T = 200               # reverse steps
N = 300               # samples per method
MALA_K = 5            # Metropolis-adjusted Langevin steps per reverse step
MALA_STEP = 0.05      # step size as a fraction of the diffused variance at that step
RNG = np.random.default_rng(20261022)
OUT = Path(__file__).resolve().parent
RED, BLUE = np.array([0.85, 0.18, 0.15]), np.array([0.16, 0.35, 0.80])
XPOS = {"left": 6.0, "centre": 16.0, "right": 26.0}


def draw(objs) -> np.ndarray:
    y, x = np.mgrid[0:H, 0:W] + 0.5
    img = np.ones((H, W, 3))
    for kind, cx in objs:
        if kind == "circle":
            img[(x - cx) ** 2 + (y - 8) ** 2 <= 3.6 ** 2] = RED
        else:
            img[(abs(x - cx) <= 3.2) & (abs(y - 8) <= 3.2)] = BLUE
    return img.ravel()


CIRCLES = {p: draw([("circle", c)]) for p, c in XPOS.items()}
SQUARES = {p: draw([("square", c)]) for p, c in XPOS.items()}
PAIR = draw([("circle", 10.0), ("square", 22.0)])          # "a red circle and a blue square"
D = PAIR.size

# cosine schedule, alpha-bar from about 1 to about 0
s = 0.008
tt = np.linspace(0, 1, T + 1)
abar = np.cos((tt + s) / (1 + s) * np.pi / 2) ** 2 / np.cos(s / (1 + s) * np.pi / 2) ** 2
abar = np.clip(abar, 1e-5, 0.9999)


class Mixture:
    """Equal-weight (or given-weight) Gaussian mixture, diffused exactly to any step."""

    def __init__(self, means, weights=None):
        self.mu = np.stack(means)
        self.logw = np.log(np.asarray(weights if weights is not None else np.ones(len(means)) / len(means)))

    def _parts(self, x, k):
        a = abar[k]
        var = a * SIGMA0 ** 2 + (1 - a)
        diff = x[:, None, :] - np.sqrt(a) * self.mu[None]                      # N x M x D
        logp = self.logw[None] - 0.5 * (diff ** 2).sum(-1) / var - 0.5 * D * np.log(2 * np.pi * var)
        return logp, diff, var

    def logp(self, x, k):
        lp, _, _ = self._parts(x, k)
        m = lp.max(1, keepdims=True)
        return (m + np.log(np.exp(lp - m).sum(1, keepdims=True)))[:, 0]

    def score(self, x, k):
        lp, diff, var = self._parts(x, k)
        r = np.exp(lp - lp.max(1, keepdims=True)); r /= r.sum(1, keepdims=True)
        return -(r[:, :, None] * diff).sum(1) / var


def composed(kind, A, B):
    if kind == "product":
        return (lambda x, k: A.logp(x, k) + B.logp(x, k)), (lambda x, k: A.score(x, k) + B.score(x, k))
    def lp(x, k):
        return np.logaddexp(A.logp(x, k), B.logp(x, k)) - np.log(2)
    def sc(x, k):
        la, lb = A.logp(x, k), B.logp(x, k)
        wa = 1 / (1 + np.exp(np.clip(lb - la, -700, 700)))
        return wa[:, None] * A.score(x, k) + (1 - wa)[:, None] * B.score(x, k)
    return lp, sc


def exact_samples(kind, A, B):
    """Exact draws at t = 0. Product of two mixtures: a mixture over pairs of components."""
    if kind == "product":
        mi, mj = A.mu[:, None], B.mu[None]
        logw = A.logw[:, None] + B.logw[None] - ((mi - mj) ** 2).sum(-1) / (4 * SIGMA0 ** 2)
        w = np.exp(logw - logw.max()); w /= w.sum()
        means = ((mi + mj) / 2).reshape(-1, D); var = SIGMA0 ** 2 / 2
        idx = RNG.choice(len(means), size=N, p=w.ravel())
    else:
        means = np.concatenate([A.mu, B.mu]); var = SIGMA0 ** 2
        idx = RNG.integers(len(means), size=N)
    return means[idx] + np.sqrt(var) * RNG.standard_normal((N, D))


def reverse(score, logp=None, mala=False):
    x = RNG.standard_normal((N, D))
    for k in range(T, 0, -1):
        a_t, a_prev = abar[k], abar[k - 1]
        alpha = a_t / a_prev; beta = 1 - alpha
        eps = -np.sqrt(1 - a_t) * score(x, k)
        mean = (x - beta / np.sqrt(1 - a_t) * eps) / np.sqrt(alpha)
        sig = np.sqrt(beta * (1 - a_prev) / (1 - a_t)) if k > 1 else 0.0
        x = mean + sig * RNG.standard_normal(x.shape)
        if mala:
            j = k - 1
            step = MALA_STEP * (abar[j] * SIGMA0 ** 2 + 1 - abar[j])
            for _ in range(MALA_K):
                g = score(x, j)
                prop = x + step * g + np.sqrt(2 * step) * RNG.standard_normal(x.shape)
                gp = score(prop, j)
                fwd = -((prop - x - step * g) ** 2).sum(1) / (4 * step)
                bwd = -((x - prop - step * gp) ** 2).sum(1) / (4 * step)
                acc = np.log(RNG.random(N)) < logp(prop, j) - logp(x, j) + bwd - fwd
                x[acc] = prop[acc]
                ACCEPT.append(acc.mean())
    return x


ACCEPT: list = []
KEEP: dict = {}


def exact_one(M):
    """Six exact draws from one condition alone, for the first two columns of the Du-style figure."""
    idx = RNG.choice(len(M.mu), size=6, p=np.exp(M.logw) / np.exp(M.logw).sum())
    return np.clip(M.mu[idx] + SIGMA0 * RNG.standard_normal((6, D)), 0, 1)


def mosaic(xs):
    """Six 16 x 32 samples stacked in a column of three by two, with white gutters."""
    tiles = [x.reshape(H, W, 3) for x in xs]
    g = np.ones((3 * H + 4, 2 * W + 2, 3))
    for i, t in enumerate(tiles):
        r, c = divmod(i, 2)
        g[r * (H + 2): r * (H + 2) + H, c * (W + 2): c * (W + 2) + W] = t
    return g


def du_figure(tag=""):
    """Du et al.'s Figure 2 layout: p1, operator, p2, =, exact, reverse diffusion, annealed MALA."""
    names = list(KEEP)
    cols = ["p1", "op1", "p2", "eq", "exact", "reverse diffusion", "annealed MALA"]
    heads = ['"a red circle"', "", '"a blue square"', "", "exact result", "reverse diffusion", "annealed MALA"]
    fig, axes = plt.subplots(len(names), 7, figsize=(15, 2.5 * len(names) + 0.6),
                             gridspec_kw={"width_ratios": [3, 0.6, 3, 0.6, 3, 3, 3]})
    axes = np.atleast_2d(axes)
    for r, name in enumerate(names):
        k = KEEP[name]
        for c, col in enumerate(cols):
            ax = axes[r, c]; ax.set_xticks([]); ax.set_yticks([])
            if col in ("op1", "eq"):
                ax.axis("off")
                ax.text(0.5, 0.5, ("×" if k["op"] == "x" else "+") if col == "op1" else "=",
                        fontsize=28, ha="center", va="center", color="#5d6470", transform=ax.transAxes)
                continue
            ax.imshow(mosaic(k[col]), interpolation="nearest")
            for sp in ax.spines.values():
                sp.set_color("#1f2328")
            if r == 0:
                ax.set_title(heads[c], fontsize=11)
        axes[r, 0].set_ylabel(name.split(":")[0] + ":\n" + name.split(": ", 1)[1].replace(", ", ",\n"), fontsize=9,
                              rotation=0, ha="right", va="center", labelpad=8)
    fig.suptitle("Du et al.'s product and sum rows, redone on red circles and blue squares with exact experts",
                 fontsize=12)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.88, bottom=0.02, wspace=0.08, hspace=0.12)
    fig.savefig(OUT / "figures" / f"fig-toy-du-layout{tag}.png", dpi=120)


# library for labelling: every single shape, the pair, and every circle-square pixel average
LIB, LAB = [], []
for p, im in CIRCLES.items():
    LIB.append(im); LAB.append("one circle")
for p, im in SQUARES.items():
    LIB.append(im); LAB.append("one square")
LIB.append(PAIR); LAB.append("two objects, crisp")
for pc, ci in CIRCLES.items():
    for ps, sq in SQUARES.items():
        LIB.append((ci + sq) / 2)
        LAB.append("hybrid: overlaid" if pc == ps else "two faint objects")
LIB = np.stack(LIB)
CATS = ["one circle", "one square", "two objects, crisp", "two faint objects", "hybrid: overlaid"]


def label(x):
    d = ((x[:, None, :] - LIB[None]) ** 2).sum(-1)
    return [LAB[i] for i in d.argmin(1)]


SETUPS = {
    "A: product, singles only": ("product",
                                 Mixture(list(CIRCLES.values())), Mixture(list(SQUARES.values()))),
    "B: product, pair also captioned with each name": ("product",
                                 Mixture(list(CIRCLES.values()) + [PAIR], [0.3, 0.3, 0.3, 0.1]),
                                 Mixture(list(SQUARES.values()) + [PAIR], [0.3, 0.3, 0.3, 0.1])),
    "C: mixture (OR), Du's bottom row": ("mixture",
                                 Mixture(list(CIRCLES.values())), Mixture(list(SQUARES.values()))),
}


def hand_weighted(w):
    """Condition D: the pair image inside both conditions at weight w, 1 - w split over the three singles.

    At w = 0 the pair component is dropped rather than given log(0), which makes D exactly condition A.
    """
    if w <= 0:
        return "product", Mixture(list(CIRCLES.values())), Mixture(list(SQUARES.values()))
    ws = [(1 - w) / 3] * 3 + [w]
    return ("product",
            Mixture(list(CIRCLES.values()) + [PAIR], ws),
            Mixture(list(SQUARES.values()) + [PAIR], ws))


CAPTIONS = ["a circle", "a square", "a circle and a square"]
CAPTION_SHAPES = {"a circle": {"circle"},
                  "a square": {"square"},
                  "a circle and a square": {"circle", "square"}}
CAPTION_IMAGES = {"a circle": list(CIRCLES.values()),
                  "a square": list(SQUARES.values()),
                  "a circle and a square": [PAIR]}
IMAGE_BANK = list(CIRCLES.values()) + list(SQUARES.values()) + [PAIR]
IMAGE_SHAPES = [{"circle"}] * 3 + [{"square"}] * 3 + [{"circle", "square"}]


def _lse(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def caption_match(form, tau):
    """match(caption, image) for all three captions over the seven-image bank.

    Form 1 scores an image by how close it is to the images the caption describes, so an extra
    object costs its pixels. Form 2 counts named shapes present, named shapes missing and present
    shapes the caption does not name, so an extra object costs one unit.
    """
    m = np.zeros((len(CAPTIONS), len(IMAGE_BANK)))
    for i, c in enumerate(CAPTIONS):
        for j, x in enumerate(IMAGE_BANK):
            if form == 1:
                d = np.array([((x - mu) ** 2).sum() for mu in CAPTION_IMAGES[c]])
                m[i, j] = _lse(-d / (2 * tau ** 2))
            else:
                named, present = CAPTION_SHAPES[c], IMAGE_SHAPES[j]
                m[i, j] = (len(named & present) - len(named - present) - len(present - named)) / tau
    return m


def caption_experts(form=1, tau=None):
    """Condition E: the product of the two single-caption conditions, with the third caption competing."""
    tau = (5.0 if form == 1 else 1.0) if tau is None else tau
    m = caption_match(form, tau)
    w = np.exp(m - _lse(m, axis=0)[None])          # p(caption | image), softmax over the three captions
    w = w / w.sum(axis=1, keepdims=True)           # Bayes with a uniform image prior: p(image | caption)
    setup = ("product", Mixture(IMAGE_BANK, list(w[0])), Mixture(IMAGE_BANK, list(w[1])))
    report = {"form": form, "tau": tau,
              "implied_pair_weight_in_circle": float(w[0, -1]),
              "implied_pair_weight_in_square": float(w[1, -1]),
              "p_caption_given_pair_image": {c: float(v) for c, v in
                                             zip(CAPTIONS, np.exp(m[:, -1] - _lse(m[:, -1])))}}
    return setup, report


def main(only=None, tag="", pair_weight=0.1, caption_form=1, caption_tau=None):
    results, shown, acc_rate = {}, {}, {}
    setups = {k: v for k, v in SETUPS.items() if only is None or k[0] in only}
    if only is not None and "D" in only:
        setups[f"D: product, pair at weight {pair_weight:g} in both"] = hand_weighted(pair_weight)
    if only is not None and "E" in only:
        setup, rep = caption_experts(caption_form, caption_tau)
        setups[f"E: product, competing captions, form {rep['form']}, tau {rep['tau']:g}"] = setup
        print("condition E:", json.dumps(rep, indent=1))
    for name, (kind, A, B) in setups.items():
        lp, sc = composed(kind, A, B)
        runs = {"exact": exact_samples(kind, A, B),
                "reverse diffusion": reverse(sc),
                "annealed MALA": reverse(sc, lp, mala=True)}
        acc_rate[name] = float(np.mean(ACCEPT)); ACCEPT.clear()
        KEEP[name] = {"p1": exact_one(A), "p2": exact_one(B), "op": "x" if kind == "product" else "+",
                      **{m: np.clip(x[:6], 0, 1) for m, x in runs.items()}}
        results[name] = {}
        for m, x in runs.items():
            labs = label(x)
            results[name][m] = {c: labs.count(c) for c in CATS}
            shown[(name, m)] = np.clip(x[:8], 0, 1)
            print(f"{name:48s} {m:18s} " + "  ".join(f"{c}: {results[name][m][c]}" for c in CATS if results[name][m][c]))
    (OUT / f"toy-sampler-hybrids{tag}.json").write_text(json.dumps(
        {"sigma0": SIGMA0, "steps": T, "samples": N, "mala_steps": MALA_K, "mala_step_fraction": MALA_STEP,
         "mala_acceptance": acc_rate, "image": [H, W, 3], "counts": results}, indent=1))
    print("MALA acceptance:", acc_rate)
    du_figure(tag)

    methods = ["exact", "reverse diffusion", "annealed MALA"]
    fig, axes = plt.subplots(len(setups) * len(methods), 9, figsize=(17, 1.1 * len(setups) * len(methods) + 1.0),
                             gridspec_kw={"width_ratios": [3.2] + [1] * 8})
    r = 0
    for name in setups:
        for m in methods:
            ax = axes[r, 0]
            c = results[name][m]
            top = "\n".join(f"{k}: {v}" for k, v in c.items() if v)
            ax.text(0, 0.5, f"{name}\n{m}\n{top}", fontsize=8, va="center", transform=ax.transAxes)
            ax.axis("off")
            for j in range(8):
                axes[r, j + 1].imshow(shown[(name, m)][j].reshape(H, W, 3), interpolation="nearest")
                axes[r, j + 1].axis("off")
            r += 1
    fig.suptitle(f"Red circles and blue squares, exact experts: 8 samples per row, counts out of {N}", fontsize=12)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.08, hspace=0.25)
    fig.savefig(OUT / "figures" / f"fig-toy-sampler-hybrids{tag}.png", dpi=110)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="setup letters to run, e.g. B")
    ap.add_argument("--mala-k", type=int, default=MALA_K)
    ap.add_argument("--mala-step", type=float, default=MALA_STEP)
    ap.add_argument("--tag", default="")
    ap.add_argument("--pair-weight", type=float, default=0.1,
                    help="condition D: the pair image's weight inside each condition")
    ap.add_argument("--caption-form", type=int, default=1, choices=(1, 2),
                    help="condition E: 1 scores an image by distance, 2 by which shapes it got right")
    ap.add_argument("--caption-tau", type=float, default=None,
                    help="condition E: the softmax temperature; 5 for form 1, 1 for form 2")
    a = ap.parse_args()
    MALA_K, MALA_STEP = a.mala_k, a.mala_step
    main(a.only, a.tag, a.pair_weight, a.caption_form, a.caption_tau)
