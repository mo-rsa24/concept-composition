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
    x_s0 = sample(true_correction, 0.0)
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
        "scale 0 against unguided: draws that differ, of 300": differ,
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


if __name__ == "__main__":
    {"port": port}[sys.argv[1]]()
