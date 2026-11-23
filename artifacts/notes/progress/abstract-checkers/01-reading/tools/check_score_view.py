"""Compute and check every number in 02-score-view.md on small discrete examples.

Usage: check_score_view.py <out.json> <out.png>
CPU only, no model, no data. The clean "images" are a few points and the forward process is the Gaussian one, so every
posterior, label probability and gradient below has a closed form. Each closed-form gradient is compared with a central
finite difference, and the script stops if any differs by more than 1e-6.
"""
import json
import pathlib
import sys

import numpy as np

OUT, PNG = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
A, S = 0.6, 0.8  # sqrt(alpha_bar_t) and sigma_t; 0.6^2 + 0.8^2 = 1


def posterior(x, atoms, a=A, s=S):
    """p(x_0 = atom | x_t = x) under a uniform prior. x has shape (d,) and atoms (n, d)."""
    log_w = -((x - a * atoms) ** 2).sum(-1) / (2 * s ** 2)
    log_w -= log_w.max()
    w = np.exp(log_w)
    return w / w.sum()


def log_p(x, atoms, mask, a=A, s=S):
    return np.log(posterior(x, atoms, a, s)[mask].sum())


def grad_log_p(x, atoms, mask, a=A, s=S):
    """(a / s^2) (E[x_0 | y, x_t] - E[x_0 | x_t])."""
    w = posterior(x, atoms, a, s)
    mean_all = (w[:, None] * atoms).sum(0)
    mean_y = (w[mask, None] * atoms[mask]).sum(0) / w[mask].sum()
    return a / s ** 2 * (mean_y - mean_all)


def grad_log_odds(x, atoms, mask, a=A, s=S):
    """(a / s^2) (E[x_0 | y = 1, x_t] - E[x_0 | y = 0, x_t])."""
    w = posterior(x, atoms, a, s)
    mean_1 = (w[mask, None] * atoms[mask]).sum(0) / w[mask].sum()
    mean_0 = (w[~mask, None] * atoms[~mask]).sum(0) / w[~mask].sum()
    return a / s ** 2 * (mean_1 - mean_0)


def finite(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def check(closed, numeric, name):
    gap = float(np.abs(closed - numeric).max())
    assert gap < 1e-6, (name, gap)
    return gap


def case(name, atoms, m1, m2, x):
    p1, p2 = np.exp(log_p(x, atoms, m1)), np.exp(log_p(x, atoms, m2))
    both = m1 & m2
    joint = np.exp(log_p(x, atoms, both))
    g1, g2, gj = grad_log_p(x, atoms, m1), grad_log_p(x, atoms, m2), grad_log_p(x, atoms, both)
    for mask, g, n in ((m1, g1, "1"), (m2, g2, "2"), (both, gj, "joint")):
        check(g, finite(lambda z: log_p(z, atoms, mask), x), f"{name} grad {n}")
    return {"p1": p1, "p2": p2, "joint": joint, "product": p1 * p2, "covariance": joint - p1 * p2,
            "log_gap": np.log(joint) - np.log(p1) - np.log(p2),
            "grad_1": g1.tolist(), "grad_2": g2.tolist(), "grad_joint": gj.tolist(), "grad_sum": (g1 + g2).tolist(),
            "grad_error": float(np.linalg.norm(g1 + g2 - gj))}


results = {"sqrt_alpha_bar": A, "sigma": S}

# Example images: 0 fused, 1 one subject, 2 two subjects in the wrong order, 3 two subjects in the right order.
line = np.array([[-3.0], [-1.0], [1.0], [3.0]])
x = np.array([0.5])
two = np.array([False, False, True, True])
right_order = np.array([False, False, False, True])
weights = posterior(x, line)
results["worked_posterior"] = weights.tolist()
results["nested"] = case("nested", line, two, right_order, x)

# Chain rule: log p(two) + log p(right order | two) equals the log of the joint.
p_order_given_two = weights[3] / weights[two].sum()
results["chain_rule"] = {"p_order_given_two": float(p_order_given_two),
                         "log_two_plus_log_conditional": float(np.log(weights[two].sum()) + np.log(p_order_given_two)),
                         "log_joint": float(np.log(weights[3]))}
assert abs(results["chain_rule"]["log_two_plus_log_conditional"] - results["chain_rule"]["log_joint"]) < 1e-12
grad_two, grad_right = grad_log_p(x, line, two), grad_log_p(x, line, right_order)
grad_conditional = grad_right - grad_two
check(grad_conditional, finite(lambda z: log_p(z, line, right_order) - log_p(z, line, two), x), "conditional")
results["chain_rule"].update({"grad_two": float(grad_two[0]), "grad_right_order": float(grad_right[0]),
                              "grad_right_order_given_two": float(grad_conditional[0])})

# Two labels independent in the clean data that read the same coordinate.
positive = line[:, 0] > 0
large = np.abs(line[:, 0]) > 2
assert abs((positive & large).mean() - positive.mean() * large.mean()) < 1e-12
results["same_coordinate"] = case("same coordinate", line, positive, large, x)

# Two labels on separate coordinates with a factorised prior: the sum of logs is exact.
grid = np.array([[u, v] for u in (-1.0, 1.0) for v in (-1.0, 1.0)])
results["separate_coordinates"] = case("separate", grid, grid[:, 0] > 0, grid[:, 1] > 0, np.array([0.3, -0.2]))
assert abs(results["separate_coordinates"]["log_gap"]) < 1e-12 and results["separate_coordinates"]["grad_error"] < 1e-12

# Log-probability against log-odds for the "two subjects" label.
g_p, g_odds = grad_log_p(x, line, two), grad_log_odds(x, line, two)
check(g_odds, finite(lambda z: log_p(z, line, two) - np.log(1 - np.exp(log_p(z, line, two))), x), "log odds")
p_two = float(weights[two].sum())
assert abs(g_p[0] - (1 - p_two) * g_odds[0]) < 1e-12
results["log_odds"] = {"p": p_two, "grad_log_p": float(g_p[0]), "grad_log_odds": float(g_odds[0]), "one_minus_p": 1 - p_two}

# The shared cap: five orthogonal pushes of norm 3 under a limit of 0.4 * 10 = 4.
limit, size = 4.0, 3.0
share = {}
for k in (5, 4):
    raw = size * k ** 0.5
    share[k] = size * min(1.0, limit / raw)
results["cap"] = {"limit": limit, "each_before": size, "applied_each_five": share[5], "applied_each_four": share[4],
                  "rise_percent": 100 * (share[4] / share[5] - 1), "own_limit_each": limit / 5,
                  "total_if_orthogonal": limit / 5 * 5 ** 0.5, "total_if_aligned": limit}

# How large the gap is across noise levels, on noised versions of the clean images that hold both labels, which is
# where guidance is trying to go. Averaged over the forward process with a fixed seed.
rng = np.random.default_rng(0)
sigmas = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.995, 0.9999])
curves = {"nested": [], "same_coordinate": [], "separate_coordinates": []}
for s in sigmas:
    a = (1 - s ** 2) ** 0.5
    for name, atoms, m1, m2 in (("nested", line, two, right_order), ("same_coordinate", line, positive, large),
                                ("separate_coordinates", grid, grid[:, 0] > 0, grid[:, 1] > 0)):
        idx = rng.choice(np.flatnonzero(m1 & m2), 4000)
        xs = a * atoms[idx] + s * rng.standard_normal((4000, atoms.shape[1]))
        gaps = []
        for z in xs:
            w = posterior(z, atoms, a, s)
            p1, p2, pj = w[m1].sum(), w[m2].sum(), w[m1 & m2].sum()
            gaps.append(abs(np.log(pj) - np.log(p1) - np.log(p2)))
        curves[name].append(float(np.mean(gaps)))
results["gap_by_noise"] = {"sigma": sigmas.tolist(), **curves}
results["steps_above_800"] = int(sum(981 - 20 * i > 800 for i in range(16)))

# SDXL's scaled-linear schedule at the pushed steps, with 50 evenly spaced steps starting at t = 981 (inferred).
betas = np.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000) ** 2
alpha_bar = np.cumprod(1 - betas)
results["schedule"] = {str(i): {"t": 981 - 20 * i, "sqrt_alpha_bar": float(alpha_bar[981 - 20 * i] ** 0.5),
                                "sigma": float((1 - alpha_bar[981 - 20 * i]) ** 0.5),
                                "signal_to_noise": float((alpha_bar[981 - 20 * i] / (1 - alpha_bar[981 - 20 * i])) ** 0.5)}
                       for i in (0, 5, 9, 10, 15)}

OUT.write_text(json.dumps(results, indent=1))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7.6, 4.3), dpi=170, facecolor="#ffffff")
styles = {"nested": ("#d1495b", "one label implies the other"),
          "same_coordinate": ("#1f77b4", "independent labels, same pixels"),
          "separate_coordinates": ("#6b6a65", "independent labels, separate pixels")}
LABEL_AT = {"nested": (9, (-8, 4), "right"), "same_coordinate": (8, (0, -27), "center"),
            "separate_coordinates": (5, (0, 7), "center")}
for name, (colour, label) in styles.items():
    ax.plot(sigmas, curves[name], color=colour, lw=2, marker="o", ms=4)
    at, off, ha = LABEL_AT[name]
    ax.annotate(label, (sigmas[at], curves[name][at]), xytext=off, textcoords="offset points", ha=ha,
                fontsize=9, color="#111111")
ax.set_xlabel("Noise level $\\sigma_t$ (right is earlier in sampling)", fontsize=10)
ax.set_ylabel("Error of the summed logs (nats)", fontsize=10)
ax.set_xlim(0.48, 1.01)
ax.axvspan(0.9515, 0.9971, color="#f1efe8", zorder=0)
ax.text(0.945, 0.665, "SDXL steps 0 to 15", ha="right", fontsize=9, color="#111111")
ax.set_title("Adding logs errs when two labels read the same pixels", fontsize=12, fontweight="bold", loc="left")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e8e7e3", lw=0.8)
fig.tight_layout()
fig.savefig(PNG)

print(json.dumps({k: v for k, v in results.items() if k != "gap_by_noise"}, indent=1))
print("gap by noise", json.dumps(results["gap_by_noise"]))
