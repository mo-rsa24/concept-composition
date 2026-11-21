"""The miniature of the product of experts and the interaction term it drops: an exactly solvable toy.

A canvas is two cells, left (L) and right (R). Each cell holds two numbers: how much "4" is in it and how much "7" is in it.
x = (4 in L, 7 in L, 4 in R, 7 in R), so four numbers in all. A digit drawn is 1 in its channel, 0 elsewhere.
Every density is a mixture of narrow Gaussians (width 0.1) around the canvases it can draw, so every score at every noise
level is exact (no network, no training).

  expert A  (label 4): a 4 in the left cell, or a 4 in the right cell, one half each
  expert B  (label 7): a 7 in the left cell, or a 7 in the right cell, one half each
  base      (no label): any of those four single-digit canvases, one quarter each
  joint     (labels 4 and 7 together): a 4 in one cell and a 7 in the other, one half each
Neither expert ever draws two digits, so the product of the two experts has no mass on the joint's canvases.
"""
import numpy as np

SIGMA0 = 0.1
A_MEANS = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], float)          # 4 left, 4 right
B_MEANS = np.array([[0, 1, 0, 0], [0, 0, 0, 1]], float)          # 7 left, 7 right
BASE_MEANS = np.vstack([A_MEANS, B_MEANS])
JOINT_MEANS = np.array([[1, 0, 0, 1], [0, 1, 1, 0]], float)       # 4 left + 7 right, 7 left + 4 right
K = 50                                                             # sampler steps, step 0 is the noisiest
U = 0.995 * (1 - np.arange(K) / (K - 1))
ABAR = np.cos(np.pi / 2 * U) ** 2                                   # cosine schedule, ABAR[0] about 6e-5, ABAR[K-1] = 1
LEFT, RIGHT = [0, 1], [2, 3]


def score(x, ab, means):
    """Exact score of the noised mixture at signal level ab: sum_j posterior_j * (-(x - sqrt(ab) mu_j) / v)."""
    a = np.sqrt(ab); v = ab * SIGMA0 ** 2 + (1 - ab)
    d = x[:, None, :] - a * means[None]
    logit = -(d ** 2).sum(-1) / (2 * v)
    logit -= logit.max(1, keepdims=True)
    r = np.exp(logit); r /= r.sum(1, keepdims=True)
    return -(r[..., None] * d).sum(1) / v, r


def eps_of(s, ab):
    return -np.sqrt(1 - ab) * s


def poe_score(x, ab):
    return score(x, ab, A_MEANS)[0] + score(x, ab, B_MEANS)[0] - score(x, ab, BASE_MEANS)[0]


def joint_score(x, ab):
    return score(x, ab, JOINT_MEANS)[0]


def sample(eps_fn, noise, init):
    """The project's sampler (DDIM with eta 1) on the toy; noise is (K, N, 4), init is (N, 4)."""
    x = init.copy()
    for k in range(K):
        ab = ABAR[k]; abp = ABAR[k + 1] if k + 1 < K else 1.0
        e = eps_fn(x, k)
        x0 = (x - np.sqrt(1 - ab) * e) / np.sqrt(ab)
        if k + 1 < K:
            sig = np.sqrt((1 - abp) / (1 - ab)) * np.sqrt(1 - ab / abp)
            x = np.sqrt(abp) * x0 + np.sqrt(max(1 - abp - sig ** 2, 0)) * e + sig * noise[k]
        else:
            x = np.sqrt(abp) * x0 + np.sqrt(1 - abp) * e
    return x


def classify(x):
    on = x > 0.5
    aL, bL, aR, bR = on.T
    two_apart = (aL & bR & ~bL & ~aR) | (aR & bL & ~aL & ~bR)
    same_cell = ((aL & bL) & ~(aR | bR)) | ((aR & bR) & ~(aL | bL))
    one = on.sum(1) == 1
    out = np.full(len(x), "other", dtype=object)
    out[one] = "one digit"; out[same_cell] = "both in one cell"; out[two_apart] = "two apart"
    return out


def hide_eps(layout_mode, hide, blank_seed=777, lock_k=10):
    """Region composition. layout_mode: 'coin' (a coin per canvas, held), 'vote' (E0e: summed score-gap votes for the first
    lock_k steps, then held), 'vote-each-step' (re-chosen at every step). hide: replace the other cell of each expert's input
    by the empty canvas at the current noise level."""
    state = {}

    def f(x, k):
        n = len(x); ab = ABAR[k]; a = np.sqrt(ab); sg = np.sqrt(1 - ab)
        if "coin" not in state:
            state["coin"] = np.random.default_rng(5).integers(2, size=n).astype(bool)
            state["cum"] = np.zeros(n)
        sA, sB = score(x, ab, A_MEANS)[0], score(x, ab, B_MEANS)[0]
        s0 = score(x, ab, BASE_MEANS)[0]
        gA, gB = np.abs(sA - s0), np.abs(sB - s0)                                  # the score gaps (label minus no label)
        vote = (gA[:, LEFT].sum(1) + gB[:, RIGHT].sum(1)) - (gA[:, RIGHT].sum(1) + gB[:, LEFT].sum(1))
        if layout_mode == "coin":
            lay = state["coin"]
        elif layout_mode == "vote":
            if k < lock_k:
                state["cum"] = state["cum"] + vote
            lay = state["cum"] >= 0
        else:
            lay = vote >= 0
        if hide:
            z = np.random.default_rng(blank_seed + k).standard_normal(x.shape)
            blank = sg * z                                                              # empty canvas (all zeros) at this noise level
            xa, xb = x.copy(), x.copy()
            xa[:, RIGHT] = np.where(lay[:, None], blank[:, RIGHT], xa[:, RIGHT]); xa[:, LEFT] = np.where(lay[:, None], xa[:, LEFT], blank[:, LEFT])
            xb[:, LEFT] = np.where(lay[:, None], blank[:, LEFT], xb[:, LEFT]); xb[:, RIGHT] = np.where(lay[:, None], xb[:, RIGHT], blank[:, RIGHT])
            sA, sB = score(xa, ab, A_MEANS)[0], score(xb, ab, B_MEANS)[0]
        s = np.empty_like(x)
        s[:, LEFT] = np.where(lay[:, None], sA[:, LEFT], sB[:, LEFT])
        s[:, RIGHT] = np.where(lay[:, None], sB[:, RIGHT], sA[:, RIGHT])
        return eps_of(s, ab)
    return f
