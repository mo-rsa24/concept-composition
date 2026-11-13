"""Annealed Hamiltonian Monte Carlo as a third corrector, beside the Langevin one already here.

At every reverse step the chain takes HMC_K Hamiltonian moves on the diffused log-density at that
noise level: a momentum draw, HMC_L leapfrog steps at a size scaled to that level's spread, and a
Metropolis accept on the total energy. The gradient-evaluation budget is HMC_K x HMC_L, set to match
the Langevin corrector's HMC_K x 1 so the two are compared at the same cost.
"""
import numpy as np
import toy_sampler_hybrids as t

ACCEPT_HMC = []


def reverse_hmc(score, logp, hmc_k=1, hmc_l=5, eps_frac=0.15, n=None):
    n = t.N if n is None else n
    x = t.RNG.standard_normal((n, t.D))
    for k in range(t.T, 0, -1):
        a_t, a_prev = t.abar[k], t.abar[k - 1]
        alpha = a_t / a_prev
        beta = 1 - alpha
        eps_pred = -np.sqrt(1 - a_t) * score(x, k)
        mean = (x - beta / np.sqrt(1 - a_t) * eps_pred) / np.sqrt(alpha)
        sig = np.sqrt(beta * (1 - a_prev) / (1 - a_t)) if k > 1 else 0.0
        x = mean + sig * t.RNG.standard_normal(x.shape)

        j = k - 1
        step = eps_frac * np.sqrt(t.abar[j] * t.SIGMA0 ** 2 + 1 - t.abar[j])
        for _ in range(hmc_k):
            p = t.RNG.standard_normal(x.shape)
            u0 = -logp(x, j)
            h0 = u0 + 0.5 * (p ** 2).sum(1)
            xp, pp = x.copy(), p.copy()
            pp = pp + 0.5 * step * score(xp, j)          # -grad U = score
            for s in range(hmc_l):
                xp = xp + step * pp
                pp = pp + (step if s < hmc_l - 1 else 0.5 * step) * score(xp, j)
            h1 = -logp(xp, j) + 0.5 * (pp ** 2).sum(1)
            acc = t.RNG.random(len(x)) < np.exp(np.clip(h0 - h1, -700, 0))
            ACCEPT_HMC.append(acc.mean())
            x = np.where(acc[:, None], xp, x)
    return x
