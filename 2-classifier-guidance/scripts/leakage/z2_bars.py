"""Bars for Loop 12 step Z2, fixed before any predictor is computed.

Design: artifacts/notes/progress/leakage/02-what-predicts-borrowing/design.md.
"""
import numpy as np


def _ranks(a):
    """Ranks from 1, ties given their average rank."""
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float)
    r[order] = np.arange(1, len(a) + 1)
    for v in np.unique(a):
        m = a == v
        r[m] = r[m].mean()
    return r


def spearman(x, y):
    rx, ry = _ranks(np.asarray(x, float)), _ranks(np.asarray(y, float))
    return float(np.corrcoef(rx, ry)[0, 1])

N_TESTS = 7                               # text once; attention overlap and score-map similarity in each of 3 windows
ALPHA_ONE_SIDED = 0.05 / N_TESTS          # 0.0071 per test; rho_threshold() gives 0.706 for these counts
N_PERMUTATIONS = 100_000
N_BOOTSTRAP = 10_000
PICTURE_MATCH_MAX_MEAN_ABS = 3.0          # of 255, against M1's CO3 picture, the limit M9 used
DAAM_CUT = 0.4                            # of each map's own maximum (DAAM p. 7)
CORRECTED_STEPS = tuple(range(6))         # CO3's num_ts_to_correct = 6
READ_STEPS = CORRECTED_STEPS + (10, 15, 20, 25, 30, 35, 40, 45)
WINDOWS = {"early": CORRECTED_STEPS, "middle": (10, 15, 20), "late": (25, 30, 35, 40, 45)}  # owner's choice, 2026-11-24

# W2d's final calls, borrowed of 8, per tuning pair (part-binding-calls.csv)
BORROWED = {
    "a_cheetah__x__a_cougar": 7, "a_duck__x__a_goose": 7, "a_puma__x__a_condor": 7,
    "a_boar__x__a_pheasant": 4, "a_bobcat__x__a_quail": 4, "a_seal__x__a_gull": 4,
    "a_crow__x__a_raven": 0, "an_eagle__x__a_fox": 0, "a_lighthouse__x__a_sailboat": 0,
    "a_zebra__x__a_flamingo": 0, "a_giraffe__x__a_tortoise": 0, "a_hippopotamus__x__a_swallow": 0,
}


def verdict(predictor: dict, seed: int = 0):
    """Spearman rho over pairs, one-sided permutation p in both directions, bootstrap 95% interval, and the call."""
    pairs = sorted(BORROWED)
    x = np.array([predictor[p] for p in pairs], float)
    y = np.array([BORROWED[p] for p in pairs], float)
    rho = spearman(x, y)
    rng = np.random.default_rng(seed)
    null = np.array([spearman(x, rng.permutation(y)) for _ in range(N_PERMUTATIONS)])
    p_up, p_down = (null >= rho - 1e-12).mean(), (null <= rho + 1e-12).mean()
    boot = []
    for _ in range(N_BOOTSTRAP):
        i = rng.integers(0, len(pairs), len(pairs))
        if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0:
            boot.append(spearman(x[i], y[i]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    call = "support" if rho > 0 and p_up < ALPHA_ONE_SIDED else "wrong way" if rho < 0 and p_down < ALPHA_ONE_SIDED else "null"
    return dict(rho=float(rho), p_up=float(p_up), p_down=float(p_down), ci=(float(lo), float(hi)), call=call)


def rho_threshold(seed: int = 0):
    """The smallest rho that reaches the bar with these tied counts, from the permutation null of a predictor with no ties."""
    y = np.array(list(BORROWED.values()), float)
    rng = np.random.default_rng(seed)
    x = np.arange(len(y), dtype=float)
    null = np.sort([spearman(x, rng.permutation(y)) for _ in range(N_PERMUTATIONS)])
    return float(null[int(np.ceil((1 - ALPHA_ONE_SIDED) * N_PERMUTATIONS))])


if __name__ == "__main__":
    print(f"rho needed for support: {rho_threshold():.3f}")
