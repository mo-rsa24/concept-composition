"""Condition D over eight pair weights, three samplers, plus the closed-form crisp share."""
import json, sys, time
import numpy as np
sys.path.insert(0, "/home/molef/PhD/poe_repair_min-dog-and-rain/artifacts/notes/what-poe-loses-when-it-composes")
import toy_sampler_hybrids as t

WEIGHTS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5]
SEED = 20261022

def closed_form(w):
    """Noiseless share of the exact product's mass whose pairing mean labels as a crisp pair."""
    kind, A, B = t.hand_weighted(w)
    mi, mj = A.mu[:, None], B.mu[None]
    logw = A.logw[:, None] + B.logw[None] - ((mi - mj) ** 2).sum(-1) / (4 * t.SIGMA0 ** 2)
    p = np.exp(logw - logw.max()); p /= p.sum()
    means = ((mi + mj) / 2).reshape(-1, t.D)
    labs = t.label(means)
    out = {c: 0.0 for c in t.CATS}
    for lab, q in zip(labs, p.ravel()):
        out[lab] += float(q)
    return out

counts, acc, timing = {}, {}, {}
for w in WEIGHTS:
    t0 = time.time()
    t.RNG = np.random.default_rng(SEED)
    kind, A, B = t.hand_weighted(w)
    lp, sc = t.composed(kind, A, B)
    runs = {"exact": t.exact_samples(kind, A, B),
            "reverse diffusion": t.reverse(sc),
            "annealed MALA": t.reverse(sc, lp, mala=True)}
    acc[f"{w:g}"] = float(np.mean(t.ACCEPT)); t.ACCEPT.clear()
    counts[f"{w:g}"] = {m: {c: t.label(x).count(c) for c in t.CATS} for m, x in runs.items()}
    timing[f"{w:g}"] = round(time.time() - t0, 1)
    print(f"w={w:g} ({timing[f'{w:g}']}s)", json.dumps(counts[f"{w:g}"]), flush=True)

grid = sorted(set(list(np.geomspace(1e-5, 0.5, 400)) + WEIGHTS[1:]))
cf = {f"{g:.6g}": closed_form(g) for g in grid}
crossing = None
for g in grid:
    if cf[f"{g:.6g}"]["two objects, crisp"] > 0.5:
        crossing = g
        break
grid_threshold = next((f"{w:g}" for w in WEIGHTS
                       if counts[f"{w:g}"]["exact"]["two objects, crisp"] / t.N > 0.5), None)

out = {"what": "condition D: the pair image inside both experts at weight w, 1-w split over that expert's three singles",
       "samples_per_cell": t.N, "sigma0": t.SIGMA0, "steps": t.T,
       "mala_steps": t.MALA_K, "mala_step_fraction": t.MALA_STEP,
       "rng_seed_per_weight": SEED,
       "note_on_seeding": "the stream is reset to the same seed at every weight, so sampler noise is shared across the curve and w=0 reproduces condition A exactly",
       "weights": [f"{w:g}" for w in WEIGHTS], "seconds_per_weight": timing,
       "counts": counts, "mala_acceptance": acc,
       "condition_b_pair_weight": 0.1,
       "grid_threshold_exact_crisp_over_half": grid_threshold,
       "closed_form": {"what": "share of the exact product's mixture mass whose pairing mean falls in each category, noise ignored",
                       "crossing_crisp_over_half": crossing,
                       "curve": cf}}
P = "/home/molef/PhD/poe_repair_min-dog-and-rain/artifacts/notes/what-poe-loses-when-it-composes/toy-sampler-cooccurrence-sweep.json"
open(P, "w").write(json.dumps(out, indent=1))
print("grid threshold:", grid_threshold, "closed-form crossing:", crossing)
print("wrote", P)
