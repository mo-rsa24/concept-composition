"""When does adding scores draw two things? An exact toy with no trained model.

Usage: check_score_addition.py <out.json> <out.png>
A picture is two patches, and each patch holds nothing, a cat vector or a dog vector. Each "expert" is a finite set of clean
pictures, so its noise prediction at every step has a closed form. Only the meaning of the single-word experts changes between
the settings. The numbers illustrate the mathematics and measure nothing about SDXL.
"""
import json
import pathlib
import sys

import numpy as np

OUT, PNG = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
E, C, D = np.array([0.0, 0.0]), np.array([1.0, 0.6]), np.array([0.6, 1.0])  # the cat and dog vectors resemble each other
H = (C + D) / 2
pic = lambda a, b: np.concatenate([a, b])
CE, EC, DE, ED, CD, DC = pic(C, E), pic(E, C), pic(D, E), pic(E, D), pic(C, D), pic(D, C)
SETTINGS = {
    "alone": {"label": "\"a cat\" means a cat alone", "null": [CE, EC, DE, ED, CD, DC], "a": [CE, EC], "b": [DE, ED], "joint": [CD, DC]},
    "contains": {"label": "\"a cat\" means a picture containing a cat", "null": [CE, EC, DE, ED, CD, DC],
                 "a": [CE, EC, CD, DC], "b": [DE, ED, CD, DC], "joint": [CD, DC]},
    "regions": {"label": "each word owns its own patch", "null": [pic(p, q) for p in (E, C) for q in (E, D)],
                "a": [CE, CD], "b": [ED, CD], "joint": [CD]},
}
JOINT = [CD, DC]

betas = np.linspace(1e-4, 0.02, 1000)
abar = np.cumprod(1 - betas)
steps = np.round(np.linspace(999, 0, 50)).astype(int)


def eps(x, atoms, t):
    """The exact noise prediction of a uniform mixture of clean points, for a batch x of shape (n, 4)."""
    atoms = np.asarray(atoms)
    a, s2 = abar[t] ** 0.5, 1 - abar[t]
    log_w = -((x[:, None] - a * atoms[None]) ** 2).sum(-1) / (2 * s2)
    log_w -= log_w.max(1, keepdims=True)
    w = np.exp(log_w)
    w /= w.sum(1, keepdims=True)
    return (x - a * (w[..., None] * atoms[None]).sum(1)) / s2 ** 0.5


def sample(fn, n=4000, seed=0):
    x = np.random.default_rng(seed).standard_normal((n, 4))
    for i, t in enumerate(steps):
        e = fn(x, t)
        a, ap = abar[t], abar[steps[i + 1]] if i + 1 < len(steps) else 1.0
        x = ap ** 0.5 * (x - (1 - a) ** 0.5 * e) / a ** 0.5 + (1 - ap) ** 0.5 * e
    return x


NAMES = np.array(["empty", "cat", "dog", "hybrid"])
PATCHES = np.stack([E, C, D, H])


def call(x):
    left = ((x[:, None, :2] - PATCHES[None]) ** 2).sum(-1).argmin(1)
    right = ((x[:, None, 2:] - PATCHES[None]) ** 2).sum(-1).argmin(1)
    two = ((left == 1) & (right == 2)) | ((left == 2) & (right == 1))
    hybrid = (left == 3) | (right == 3)
    one = ((left == 0) & np.isin(right, [1, 2])) | ((right == 0) & np.isin(left, [1, 2]))
    pairs, counts = np.unique(np.stack([left, right], 1), axis=0, return_counts=True)
    top = pairs[counts.argmax()]
    return {"cat_and_dog": float(two.mean()), "any_hybrid_patch": float(hybrid.mean()), "one_subject": float(one.mean()),
            "most_common": f"{NAMES[top[0]]} | {NAMES[top[1]]}"}


results = {}
for key, cfg in SETTINGS.items():
    fn = lambda x, t: eps(x, cfg["a"], t) + eps(x, cfg["b"], t) - eps(x, cfg["null"], t)
    r = call(sample(fn))
    probe = np.random.default_rng(1)
    curve = []
    for t in steps:
        x0 = np.asarray(cfg["joint"])[probe.integers(0, len(cfg["joint"]), 2000)]
        xt = abar[t] ** 0.5 * x0 + (1 - abar[t]) ** 0.5 * probe.standard_normal((2000, 4))
        added, target, null = fn(xt, t), eps(xt, cfg["joint"], t), eps(xt, cfg["null"], t)
        curve.append(float(np.linalg.norm(target - added, axis=1).mean() / np.linalg.norm(target - null, axis=1).mean()))
    results[key] = {"label": cfg["label"], **r, "gap_by_step": curve, "mean_gap_steps_0_to_15": float(np.mean(curve[:16]))}
results["joint"] = {"label": "the joint prompt, for reference", **call(sample(lambda x, t: eps(x, JOINT, t)))}
results["steps"] = steps.tolist()
OUT.write_text(json.dumps(results, indent=1))
for key in ("alone", "contains", "regions", "joint"):
    r = results[key]
    print(f"{key:9s} cat+dog {r['cat_and_dog']:.3f} hybrid {r['any_hybrid_patch']:.3f} one {r['one_subject']:.3f} "
          f"most common [{r['most_common']}] gap0-15 {r.get('mean_gap_steps_0_to_15', float('nan')):.3f}")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

COL = {"empty": "#eeeeea", "cat": "#c86a0a", "dog": "#1f5fd0", "hybrid": "#7a4fa3"}
fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.4, 4.3), dpi=170, gridspec_kw={"width_ratios": [1.25, 1]}, facecolor="#ffffff")
rows = ["alone", "contains", "regions", "joint"]
for j, key in enumerate(rows):
    y = len(rows) - 1 - j
    r = results[key]
    left, right = r["most_common"].split(" | ")
    ax.add_patch(Rectangle((-0.34, y - 0.27), 0.14, 0.54, color=COL[left]))
    ax.add_patch(Rectangle((-0.195, y - 0.27), 0.14, 0.54, color=COL[right]))
    ax.barh(y, r["cat_and_dog"], height=0.5, color="#1d7a3e")
    ax.text(r["cat_and_dog"] + 0.015, y, f"{100 * r['cat_and_dog']:.0f}%", va="center", fontsize=10)
    ax.text(-0.36, y, r["label"], ha="right", va="center", fontsize=9.5)
ax.set_xlim(-1.32, 1.13), ax.set_ylim(-0.6, 3.6), ax.set_yticks([]), ax.set_xticks([0, 0.5, 1], ["0", "50%", "100%"])
ax.spines[["top", "right", "left"]].set_visible(False)
ax.set_xlabel("Samples with one cat patch and one dog patch", fontsize=9.5, loc="right")
ax.set_title("The same addition, three meanings of a word", fontsize=12, fontweight="bold", loc="left")
for key, colour, name, y in (("alone", "#c0262d", "cat alone", 2.13), ("contains", "#c86a0a", "containing a cat", 0.63),
                             ("regions", "#1d7a3e", "own patch: exactly 0", 0.13)):
    bx.plot(range(36), results[key]["gap_by_step"][:36], color=colour, lw=2.4)
    bx.text(18, y, name, fontsize=9.5, ha="center")
bx.set_ylim(-0.08, 2.5)
bx.set_xlabel("Sampling step (0 is pure noise)", fontsize=9.5)
bx.set_ylabel("$\\|r_t\\|$ / $\\|\\epsilon_{joint} - \\epsilon_\\varnothing\\|$", fontsize=10)
bx.set_title("How far addition is from the joint prompt", fontsize=12, fontweight="bold", loc="left")
bx.spines[["top", "right"]].set_visible(False), bx.grid(axis="y", color="#e8e7e3", lw=0.8)
fig.tight_layout()
fig.savefig(PNG)
