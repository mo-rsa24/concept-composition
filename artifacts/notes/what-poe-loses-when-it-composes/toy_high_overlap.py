"""Where does a sampler finally draw the pair? The first dial, carried past 0.5 to 0.99.

As the pair image's weight approaches 1 each expert becomes that picture and little else, so the
two stop being a circle expert and a square expert. The run therefore records, beside each count,
what fraction of each expert's own mass is still on its single shapes.
"""
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import toy_sampler_hybrids as t
OUT = HERE.parents[1] / "results" / "how-much-co-occurrence-the-product-needs"
SEED = 20261022
WEIGHTS = [0.5, 0.65, 0.8, 0.9, 0.95, 0.99]

counts, acc = {}, {}
for w in WEIGHTS:
    t0 = time.time(); t.RNG = np.random.default_rng(SEED)
    kind, A, B = t.hand_weighted(w)
    lp, sc = t.composed(kind, A, B)
    runs = {"exact": t.exact_samples(kind, A, B), "reverse diffusion": t.reverse(sc)}
    t.ACCEPT.clear()
    runs["annealed MALA"] = t.reverse(sc, lp, mala=True)
    acc[f"{w:g}"] = round(float(np.mean(t.ACCEPT)), 4); t.ACCEPT.clear()
    counts[f"{w:g}"] = {m: {c: t.label(x).count(c) for c in t.CATS} for m, x in runs.items()}
    counts[f"{w:g}"]["mass still on this expert's own single shapes"] = round(1 - w, 4)
    print(f"w={w:g} ({time.time()-t0:.0f}s) "
          + " | ".join(f"{m} {counts[f'{w:g}'][m]['two objects, crisp']}" for m in runs), flush=True)

json.dump({"what": "the pair image's weight inside both experts, carried from 0.5 to 0.99",
           "samples_per_cell": t.N, "rng_seed_per_weight": SEED,
           "note": "at weight w each expert keeps only 1 - w of its mass on its own three single shapes, "
                   "so near 1 neither is a single-shape expert any more",
           "counts": counts, "mala_acceptance": acc},
          open(OUT / "toy-high-overlap-sweep.json", "w"), indent=1)
print("wrote", OUT / "toy-high-overlap-sweep.json")
