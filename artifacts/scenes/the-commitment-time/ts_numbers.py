"""Every number on the commitment-time page, computed on the 2D two-Gaussian toy.

Toy: classes at +m and -m, isotropic variance sigma^2 = 1, |m|^2 = 19, so the data covariance is
I + m m^T and its largest eigenvalue is 20. Forward: x_t = e^{-t} x_0 + sqrt(1 - e^{-2t}) z.
Backward (Biroli et al. Eq. 3): y(t - dt) = y + (y + 2 S(y, t)) dt + sqrt(2 dt) z, with the exact
mixture score of their Eq. 19. Cloning (their Eq. 6 and Section B.3): at time t one backward
trajectory is copied into two, which continue with independent noise to t = 0.
"""
import json
import math
import os
import time

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
SIGMA2 = 1.0
M2 = 19.0
LAM = SIGMA2 + M2
ANGLE = math.radians(30)  # the direction of m is a drawing choice; no number depends on it
M = math.sqrt(M2) * np.array([math.cos(ANGLE), math.sin(ANGLE)])
U = M / np.linalg.norm(M)
TF = 4.0
DT = 0.002
N_PAIRS = 40000
GRID = np.round(np.arange(0.1, 3.0001, 0.1), 1)


def gamma(t):
    return SIGMA2 * np.exp(-2 * t) + 1 - np.exp(-2 * t)


def score(x, t):
    g = gamma(t)
    e = np.exp(-t)
    return -x / g + M[None, :] * (e / g) * np.tanh((x @ M) * e / g)[:, None]


def step(x, t, rng):
    return x + (x + 2 * score(x, t)) * DT + math.sqrt(2 * DT) * rng.standard_normal(x.shape)


def sample_marginal(t, n, rng):
    s = rng.choice([-1.0, 1.0], size=n)[:, None]
    return s * M[None, :] * math.exp(-t) + math.sqrt(gamma(t)) * rng.standard_normal((n, 2))


def phi_exact(t):
    """Biroli Eq. 6, by trapezoid on a fine grid along the main direction."""
    y = np.linspace(-20, 20, 40001)
    mt, g = math.sqrt(M2) * np.exp(-t), gamma(t)
    gp = np.exp(-(y - mt) ** 2 / (2 * g)) / np.sqrt(2 * np.pi * g)
    gm = np.exp(-(y + mt) ** 2 / (2 * g)) / np.sqrt(2 * np.pi * g)
    return float(0.5 * (np.trapezoid if hasattr(np, "trapezoid") else np.trapz)((gp ** 2 + gm ** 2) / (gp + gm), y))


def run_clone(y0, t0, rng):
    x = y0.copy()
    n = int(round(t0 / DT))
    for i in range(n):
        x = step(x, t0 - i * DT, rng)
    return x


def bisect(f, a, b):
    fa = f(a)
    for _ in range(80):
        c = 0.5 * (a + b)
        fc = f(c)
        if (fa > 0) == (fc > 0):
            a, fa = c, fc
        else:
            b = c
    return 0.5 * (a + b)


def main():
    t0 = time.time()
    r = {}
    C = SIGMA2 * np.eye(2) + np.outer(M, M)
    ev = np.linalg.eigvalsh(C)
    r["eig_small"], r["eig_large"] = float(ev[0]), float(ev[1])
    rng = np.random.default_rng(20261126)
    big = sample_marginal(0.0, 2_000_000, rng)
    r["sample_lambda_2M"] = float(np.linalg.eigvalsh(np.cov(big.T))[-1])
    Ts = 0.5 * math.log(LAM)
    r["T_s"] = Ts
    r["ln_lambda"] = math.log(LAM)
    r["e_minus_2Ts"] = math.exp(-2 * Ts)
    r["signal_at_Ts"] = LAM * math.exp(-2 * Ts)
    r["noise_var_at_Ts"] = 1 - math.exp(-2 * Ts)
    r["t_signal_equals_true_noise"] = 0.5 * math.log(LAM + 1)
    r["t_curvature_logp_zero"] = 0.5 * math.log(M2 + 1 - SIGMA2)
    r["t_paper_eq23_2dmu2"] = 0.5 * math.log(2 * M2)
    r["doubling_adds"] = 0.5 * math.log(2)
    r["ten_times_adds"] = 0.5 * math.log(10)
    r["Ts_table"] = {str(l): 0.5 * math.log(l) for l in [2, 5, 10, 20, 40, 80, 100, 200, 1000, 10000]}
    r["shrink_table"] = {str(t): {"e-2t": math.exp(-2 * t), "signal": LAM * math.exp(-2 * t)}
                         for t in [0, 0.5, 1, 1.5, 2, 3]}
    r["phi_at_Ts_exact"] = phi_exact(Ts)
    r["t_phi_0775"] = bisect(lambda t: phi_exact(t) - 0.775, 0.2, 3.0)
    r["phi_at_m_et_1"] = phi_exact(0.5 * math.log(M2))
    r["t_where_phi_0.95"] = bisect(lambda t: phi_exact(t) - 0.95, 0.05, 3.0)
    r["t_where_phi_0.55"] = bisect(lambda t: phi_exact(t) - 0.55, 0.2, 4.0)
    r["paper_leading"] = {str(N): {"half": 0.5 * math.log(N), "quarter": 0.25 * math.log(N)}
                          for N in [100, 1024, 16384, 1000000]}

    # Population cloning curve: one parent ensemble run backward from TF, cloned at each grid time.
    rng = np.random.default_rng(20261127)
    x = sample_marginal(TF, N_PAIRS, rng)
    saved = {}
    n_steps = int(round(TF / DT))
    grid_idx = {int(round((TF - g) / DT)): float(g) for g in GRID}
    for i in range(n_steps + 1):
        if i in grid_idx:
            saved[grid_idx[i]] = x.copy()
        if i < n_steps:
            x = step(x, TF - i * DT, rng)
    r["parent_end_frac_plus"] = float(np.mean(x @ M > 0))
    curve = []
    for g in GRID:
        g = float(g)
        y = saved[g]
        a = run_clone(y, g, rng)
        b = run_clone(y, g, rng)
        same = (a @ M > 0) == (b @ M > 0)
        p = float(np.mean(same))
        se = float(np.sqrt(p * (1 - p) / N_PAIRS))
        curve.append({"t": g, "sim": round(p, 4), "se": round(se, 4), "exact": round(phi_exact(g), 4)})
    r["curve"] = curve
    r["curve_max_abs_gap"] = max(abs(c["sim"] - c["exact"]) for c in curve)
    r["curve_max_gap_in_se"] = max(abs(c["sim"] - c["exact"]) / max(c["se"], 1e-4) for c in curve)
    r["fine_exact"] = [{"t": round(float(t), 3), "phi": round(phi_exact(float(t)), 4)}
                       for t in np.arange(0.0, 3.001, 0.025)]

    # One drawn parent with 16 clones per split time, for the fan picture. Seed is the first tried.
    rng = np.random.default_rng(7)
    p = sample_marginal(TF, 1, rng)
    every = int(round(0.05 / DT))
    parent = {}
    for i in range(n_steps + 1):
        tnow = TF - i * DT
        if i % every == 0 and tnow <= 3.0001:
            parent[round(tnow, 2)] = float((p @ U)[0])
        if i in grid_idx:
            saved[("p", grid_idx[i])] = p.copy()
        if i < n_steps:
            p = step(p, tnow, rng)
    fans = {}
    for g in GRID:
        g = float(g)
        c = np.repeat(saved[("p", g)], 16, axis=0)
        n = int(round(g / DT))
        tr = [np.round(c @ U, 2).tolist()]
        for i in range(n):
            c = step(c, g - i * DT, rng)
            if (i + 1) % every == 0:
                tr.append(np.round(c @ U, 2).tolist())
        ends = c @ M > 0
        fans[f"{g:.1f}"] = {"paths": np.array(tr).T.tolist(), "plus": int(ends.sum())}
    r["fan_plus_counts"] = {k: v["plus"] for k, v in fans.items()}

    # Cloud points for the drawing: fixed draws so the picture moves smoothly with t.
    rng = np.random.default_rng(11)
    n = 360
    s = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)[:, None]
    e0 = rng.standard_normal((n, 2))
    z = rng.standard_normal((n, 2))
    x0 = s * M[None, :] + e0
    # Same-mean, different-variance drawing for the last frame (Achilli Fig. 2 uses delta = 0.3).
    rng = np.random.default_rng(13)
    a = rng.standard_normal((150, 2)) * math.sqrt(0.7)
    b = rng.standard_normal((150, 2)) * math.sqrt(1.3)

    page = {
        "m": np.round(M, 4).tolist(), "u": np.round(U, 4).tolist(),
        "x0": np.round(x0, 2).tolist(), "z": np.round(z, 2).tolist(), "cls": s[:, 0].astype(int).tolist(),
        "curve": curve, "exact": r["fine_exact"], "parent": {f"{k:.2f}": round(v, 2) for k, v in parent.items()},
        "fans": fans, "sameMeanA": np.round(a, 2).tolist(), "sameMeanB": np.round(b, 2).tolist(),
    }
    with open(os.path.join(OUT, "ts_page_data.json"), "w") as f:
        json.dump(page, f, separators=(",", ":"))
    r["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "ts_numbers.json"), "w") as f:
        json.dump(r, f, indent=1)
    show = {k: v for k, v in r.items() if k not in ("curve", "fine_exact")}
    print(json.dumps(show, indent=1))
    print("t   sim    se     exact")
    for c in curve:
        print(f'{c["t"]:.1f} {c["sim"]:.4f} {c["se"]:.4f} {c["exact"]:.4f}')
    print("page data KB", os.path.getsize(os.path.join(OUT, "ts_page_data.json")) / 1024)


if __name__ == "__main__":
    main()
