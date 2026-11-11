"""Check the Langevin corrector's update on a toy whose answer is known exactly.

Two Gaussian experts, N(-1, 0.5) and N(1, 0.8) (variance as the second number), multiply into
N(-0.2308, 0.3077). The corrector in ``poe_repair/methods/_poe_langevin.py`` takes

    x <- x + delta * score(x) + sqrt(2 delta) * z,      z ~ N(0, 1)

with score the sum of the two experts' scores. On a Gaussian of variance V this chain does not
settle at V: its own stationary variance is V / (1 - delta / (2 V)), and its stationary mean is the
product's mean exactly. The check runs many independent chains at four step sizes and passes a step
size when the chains' mean and variance each sit within 3 Monte Carlo standard errors of those
exact values. The last column is the gap between the chain's exact variance and 0.3077, the bias the
step size alone causes.

    python3 scripts/check_langevin_two_gaussians.py
"""

from __future__ import annotations

import math
import sys

import numpy as np

MU_A, VAR_A = -1.0, 0.5
MU_B, VAR_B = 1.0, 0.8
STEP_SIZES = (0.005, 0.02, 0.05, 0.1)
N_CHAINS = 200_000
TOLERANCE_SE = 3.0          # the pass bar, in Monte Carlo standard errors
SEED = 0


def product_of_gaussians(mu_a, var_a, mu_b, var_b):
    precision = 1.0 / var_a + 1.0 / var_b
    var = 1.0 / precision
    return var * (mu_a / var_a + mu_b / var_b), var


def score(x):
    return -(x - MU_A) / VAR_A - (x - MU_B) / VAR_B


def run_chains(delta, n_chains, n_steps, rng):
    x = rng.standard_normal(n_chains)
    noise_std = math.sqrt(2.0 * delta)
    for _ in range(n_steps):
        x = x + delta * score(x) + noise_std * rng.standard_normal(n_chains)
    return x


def main() -> int:
    mu, var = product_of_gaussians(MU_A, VAR_A, MU_B, VAR_B)
    rng = np.random.default_rng(SEED)
    print(f"product of N({MU_A}, {VAR_A}) and N({MU_B}, {VAR_B}) = N({mu:.4f}, {var:.4f})")
    print(f"{N_CHAINS} independent chains per step size; pass = within {TOLERANCE_SE:g} standard errors\n")
    header = (f"{'delta':>6} {'steps':>6} {'mean':>8} {'exact':>8} {'var':>8} {'exact':>8} "
              f"{'mean SE':>8} {'var SE':>8} {'bias vs 0.3077':>15}  verdict")
    print(header)
    print("-" * len(header))
    all_pass = True
    for delta in STEP_SIZES:
        n_steps = math.ceil(20 * var / delta)        # 20 relaxation times, each about V / delta steps
        x = run_chains(delta, N_CHAINS, n_steps, rng)
        var_exact = var / (1.0 - delta / (2.0 * var))
        m, v = float(x.mean()), float(x.var(ddof=1))
        se_mean = math.sqrt(var_exact / N_CHAINS)
        se_var = var_exact * math.sqrt(2.0 / (N_CHAINS - 1))
        ok = abs(m - mu) <= TOLERANCE_SE * se_mean and abs(v - var_exact) <= TOLERANCE_SE * se_var
        all_pass &= ok
        print(f"{delta:>6g} {n_steps:>6d} {m:>8.4f} {mu:>8.4f} {v:>8.4f} {var_exact:>8.4f} "
              f"{se_mean:>8.5f} {se_var:>8.5f} {var_exact - var:>+15.4f}  {'PASS' if ok else 'MISS'}")
    print()
    print("PASS: all four step sizes" if all_pass else "MISS: at least one step size")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
