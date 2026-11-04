"""Is the samplers' failure the sum of noised scores, or reverse diffusion itself?

Both experts are Gaussian mixtures, so their product at step 0 is a Gaussian mixture over every
pairing (component i of the circle expert with component j of the square expert): mean (mi + mj) / 2,
variance SIGMA0^2 / 2, weight wA_i * wB_j * exp(-|mi - mj|^2 / 4 SIGMA0^2). Noising that mixture
forward gives the TRUE score of the noised product at every step, in closed form.

The samplers in toy_sampler_hybrids.py instead add the two experts' own noised scores. That sum is
the score of (noised A) x (noised B), which is also a mixture over pairings, but its pairing weights
at step k use the noised distance: exp(-abar_k |mi - mj|^2 / 4 (abar_k SIGMA0^2 + 1 - abar_k)).

Two reads, both on hand-weighted condition D at the given pair weights:
  1. the share of the pair image's self-pairing in each mixture's weights, step by step
  2. 300 reverse-diffusion draws with each score, the same sampler, labelled by the template library

    python3 -I toy_noised_product.py
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import toy_sampler_hybrids as t

OUT = t.OUT.parents[1] / "results" / "how-much-co-occurrence-the-product-needs"
WEIGHTS = [0.001, 0.01, 0.1, 0.3]


def pairing(A, B, k, true):
    """Pairing weights (normalised) and the index of the pair image's self-pairing."""
    a = 1.0 if true else t.abar[k]
    var = t.SIGMA0 ** 2 if true else a * t.SIGMA0 ** 2 + 1 - a
    d2 = ((A.mu[:, None] - B.mu[None]) ** 2).sum(-1)
    lw = A.logw[:, None] + B.logw[None] - a * d2 / (4 * var)
    w = np.exp(lw - lw.max()); w /= w.sum()
    return w


class TrueProduct:
    """The step-0 product mixture, noised exactly to step k."""
    def __init__(self, A, B):
        self.mu = ((A.mu[:, None] + B.mu[None]) / 2).reshape(-1, t.D)
        self.logw = np.log(np.maximum(pairing(A, B, 0, True).ravel(), 1e-300))
        self.v0 = t.SIGMA0 ** 2 / 2

    def score(self, x, k):
        a = t.abar[k]; var = a * self.v0 + 1 - a
        diff = x[:, None, :] - np.sqrt(a) * self.mu[None]
        lp = self.logw[None] - 0.5 * (diff ** 2).sum(-1) / var
        r = np.exp(lp - lp.max(1, keepdims=True)); r /= r.sum(1, keepdims=True)
        return -(r[:, :, None] * diff).sum(1) / var


def main():
    out = {}
    for w in WEIGHTS:
        _, A, B = t.hand_weighted(w)
        pi = len(A.mu) - 1                                    # the pair image is the last component
        share_true = float(pairing(A, B, 0, True)[pi, pi])
        share_sum = [float(pairing(A, B, k, False)[pi, pi]) for k in range(t.T + 1)]
        _, sc_sum = t.composed("product", A, B)
        x_sum = t.reverse(sc_sum)
        x_true = t.reverse(TrueProduct(A, B).score)
        l_sum, l_true = t.label(x_sum), t.label(x_true)
        out[f"{w:g}"] = {
            "pair self-pairing share, true product (every step)": share_true,
            "pair self-pairing share, summed noised scores, by step": share_sum,
            "reverse diffusion, summed noised scores": {c: l_sum.count(c) for c in t.CATS},
            "reverse diffusion, true noised-product score": {c: l_true.count(c) for c in t.CATS}}
        t.KEEP[f"{w:g}"] = (x_sum[:8], x_true[:8])
        print(f"w={w:g}: pair share true {share_true:.3f}, summed at step 0 {share_sum[0]:.3f} / "
              f"step 100 {share_sum[100]:.4f} / step 200 {share_sum[200]:.4f} | crisp of 300: summed "
              f"{l_sum.count('two objects, crisp')}, true {l_true.count('two objects, crisp')}", flush=True)
    np.savez_compressed(OUT / "toy-noised-product-samples.npz",
                        **{f"{w}_{i}": s for w, ss in t.KEEP.items() for i, s in enumerate(ss)})
    json.dump({"what": "reverse diffusion on condition D with the summed noised expert scores against "
                       "the true score of the noised product, same sampler, 300 draws each",
               "steps": t.T, "results": out}, open(OUT / "toy-noised-product.json", "w"), indent=1)
    print("wrote", OUT / "toy-noised-product.json")


if __name__ == "__main__":
    main()
