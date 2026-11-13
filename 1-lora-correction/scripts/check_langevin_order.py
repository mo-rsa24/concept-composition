"""Measure what the order of corrector and step changes, on the two-Gaussian toy.

The same experts as ``check_langevin_two_gaussians.py``, N(-1, 0.5) and N(1, 0.8), are diffused
along SDXL's own schedule (scaled-linear betas 0.00085 to 0.012 over 1000 timesteps, 50 DDIM levels
at 981, 961, ..., 1). At each level the product-of-experts score is the sum of the two diffused
experts' exact scores, which is what ``run_poe_langevin`` gets from the UNet.

Two orders of the same pieces, each with k Langevin steps of size c * beta_t per level:

    corrector first (ours, ``_poe_langevin.py``): correct at level t, then the DDIM step to t-1
    step first (the paper's):                      the DDIM step to t-1, then correct at t-1

and plain product-of-experts (k = 0) as the reference. Each run's final samples are scored against
the product of the clean experts, N(-0.2308, 0.3077), by their mean, their variance and the KL
divergence from the target.

    python3 1-lora-correction/scripts/check_langevin_order.py
"""

from __future__ import annotations

import math

import numpy as np

MU = (-1.0, 1.0)
VAR = (0.5, 0.8)
N_CHAINS = 200_000
SEED = 0
SETTINGS = ((3.0, 20), (1.0, 20), (10.0, 20))   # (c, k); c = 3 with k = 20 is the working point


def sdxl_schedule(n_levels=50):
    betas = np.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000) ** 2
    abar = np.cumprod(1.0 - betas)
    step = 1000 // n_levels
    timesteps = np.arange(0, n_levels)[::-1] * step + 1          # leading spacing, steps_offset 1
    return betas, abar, timesteps


def poe_score(x, a):
    """Exact product-of-experts score at cumulative alpha a."""
    s = np.zeros_like(x)
    for m, v in zip(MU, VAR):
        s -= (x - math.sqrt(a) * m) / (a * v + 1.0 - a)
    return s


def ddim_step(x, a, a_prev):
    eps = -math.sqrt(1.0 - a) * poe_score(x, a)
    x0 = (x - math.sqrt(1.0 - a) * eps) / math.sqrt(a)
    return math.sqrt(a_prev) * x0 + math.sqrt(1.0 - a_prev) * eps


def correct(x, a, beta, c, k, rng):
    delta = c * beta
    for _ in range(k):
        x = x + delta * poe_score(x, a) + math.sqrt(2.0 * delta) * rng.standard_normal(x.shape)
    return x


def sample(order, c, k, seed):
    rng = np.random.default_rng(seed)
    betas, abar, ts = sdxl_schedule()
    x = rng.standard_normal(N_CHAINS)
    for i, t in enumerate(ts):
        a = abar[t]
        a_prev = abar[ts[i + 1]] if i + 1 < len(ts) else abar[0]   # set_alpha_to_one = False
        if order == "corrector first":
            x = correct(x, a, betas[t], c, k, rng)
            x = ddim_step(x, a, a_prev)
        else:
            x = ddim_step(x, a, a_prev)
            t_prev = ts[i + 1] if i + 1 < len(ts) else 0
            x = correct(x, a_prev, betas[t_prev], c, k, rng)
    return x


def kl_to(m, v, mu, var):
    return 0.5 * (v / var + (m - mu) ** 2 / var - 1.0 + math.log(var / v))


def main() -> None:
    prec = sum(1.0 / v for v in VAR)
    var = 1.0 / prec
    mu = var * sum(m / v for m, v in zip(MU, VAR))
    print(f"target: product of the clean experts, N({mu:.4f}, {var:.4f}); "
          f"{N_CHAINS} chains, SDXL schedule, 50 DDIM levels\n")
    header = f"{'run':<18} {'c':>4} {'k':>3} {'mean':>8} {'var':>8} {'KL to target':>13}"
    print(header)
    print("-" * len(header))
    rows = [("plain PoE", 0.0, 0)] + [(o, c, k) for c, k in SETTINGS
                                       for o in ("corrector first", "step first")]
    for name, c, k in rows:
        x = sample("corrector first", c, k, SEED) if k == 0 else sample(name, c, k, SEED)
        m, v = float(x.mean()), float(x.var(ddof=1))
        print(f"{name:<18} {c:>4g} {k:>3d} {m:>8.4f} {v:>8.4f} {kl_to(m, v, mu, var):>13.5f}")


if __name__ == "__main__":
    main()
