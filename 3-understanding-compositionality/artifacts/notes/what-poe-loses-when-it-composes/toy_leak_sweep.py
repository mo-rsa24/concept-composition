"""The second dial: how much weight each expert puts on the other's shapes before composition dies.

The first dial asked how much the two experts must agree. This one holds that agreement fixed and
asks how much they may disagree: the circle expert is given weight `leak` on each single square and
the square expert the same on each single circle, with the pair image held at --pair-weight and the
rest split over that expert's own shapes. Condition E showed a caption model produces a leak of this
kind without anyone choosing it; this measures what any given amount of it costs.
"""
import json, sys, time
import numpy as np
sys.path.insert(0, "/home/molef/PhD/poe_repair_min-dog-and-rain/artifacts/notes/what-poe-loses-when-it-composes")
import toy_sampler_hybrids as t

OUT = "/home/molef/PhD/poe_repair_min-dog-and-rain/artifacts/results/how-much-co-occurrence-the-product-needs"
SEED, W = 20261022, 0.1
SAMPLED_AT = [0.0, 0.003, 0.01, 0.1]


def leaky(leak, w=W):
    """Each expert: its own three shapes, the other's three at `leak` each, the pair at `w`."""
    own = (1 - w - 3 * leak) / 3
    circ, sq = list(t.CIRCLES.values()), list(t.SQUARES.values())
    wa = [own] * 3 + [leak] * 3 + [w]
    wb = [leak] * 3 + [own] * 3 + [w]
    return ("product", t.Mixture(circ + sq + [t.PAIR], wa), t.Mixture(circ + sq + [t.PAIR], wb))


def closed_form(leak, w=W):
    kind, A, B = leaky(leak, w)
    mi, mj = A.mu[:, None], B.mu[None]
    lw = A.logw[:, None] + B.logw[None] - ((mi - mj) ** 2).sum(-1) / (4 * t.SIGMA0 ** 2)
    p = np.exp(lw - lw.max()); p /= p.sum()
    means = ((mi + mj) / 2).reshape(-1, t.D)
    out = {c: 0.0 for c in t.CATS}
    for lab, q in zip(t.label(means), p.ravel()):
        out[lab] += float(q)
    return out


grid = sorted(set(list(np.geomspace(1e-5, 0.3, 240)) + SAMPLED_AT[1:]))
cf = {f"{g:.6g}": closed_form(g) for g in grid}
dies = next((g for g in grid if cf[f"{g:.6g}"]["two objects, crisp"] < 0.5), None)
print("closed-form: the crisp share drops below half at leak =", dies, flush=True)

counts = {}
for leak in SAMPLED_AT:
    t0 = time.time(); t.RNG = np.random.default_rng(SEED)
    kind, A, B = leaky(leak)
    lp, sc = t.composed(kind, A, B)
    runs = {"exact": t.exact_samples(kind, A, B), "reverse diffusion": t.reverse(sc)}
    counts[f"{leak:g}"] = {m: {c: t.label(x).count(c) for c in t.CATS} for m, x in runs.items()}
    print(f"leak {leak:g} ({time.time()-t0:.0f}s)", json.dumps(counts[f"{leak:g}"]), flush=True)

json.dump({"what": "each expert's weight on each of the other's three single shapes, with the pair image held at 0.1",
           "pair_weight_held_at": W, "samples_per_cell": t.N, "rng_seed_per_cell": SEED,
           "closed_form_leak_where_crisp_share_drops_below_half": dies,
           "sampled_counts": counts,
           "closed_form_curve": cf},
          open(f"{OUT}/toy-leak-sweep.json", "w"), indent=1)
print("wrote", f"{OUT}/toy-leak-sweep.json")
