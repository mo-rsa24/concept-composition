"""Samples for the model 1 picture grids: the formula experts at several weights on the
both-shapes picture, and at several leak amounts, drawn exactly and by reverse diffusion with
summed scores. Same random stream as the sweeps (seed 20261022 reset per setting). 300 samples per
cell; the count is the template library's "two objects, crisp" (exact pictures, no VAE)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import toy_sampler_hybrids as t
import toy_leak_sweep as lk

OUT = Path(__file__).resolve().parents[2] / "results" / "how-much-co-occurrence-the-product-needs"
res, keep = {"weight": {}, "leak": {}}, {}
for group, values, build in (("weight", [0.0, 0.001, 0.01, 0.1, 0.3, 0.9], t.hand_weighted),
                             ("leak", [0.0, 0.003, 0.01, 0.1], lk.leaky)):
    for v in values:
        t.RNG = np.random.default_rng(20261022)
        kind, A, B = build(v)
        lp, sc = t.composed(kind, A, B)
        for name, x in (("exact", t.exact_samples(kind, A, B)), ("sampler", t.reverse(sc))):
            labs = t.label(x)
            res[group][f"{v:g}_{name}"] = {c: labs.count(c) for c in t.CATS}
            keep[f"{group}_{v:g}_{name}"] = np.clip(x[:8], 0, 1)
            print(group, v, name, res[group][f"{v:g}_{name}"], flush=True)
np.savez_compressed(OUT / "toy-formula-samples.npz", **keep)
json.dump({"what": "formula experts: exact draws and reverse-diffusion draws with summed scores, 300 each",
           "results": res}, open(OUT / "toy-formula-samples.json", "w"), indent=1)
print("wrote", OUT / "toy-formula-samples.json")
