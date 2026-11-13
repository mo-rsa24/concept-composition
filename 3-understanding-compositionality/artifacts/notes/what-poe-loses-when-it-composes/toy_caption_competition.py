"""Condition E in both forms, and condition D re-run at each form's implied pair weight."""
import json, os, sys, time
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/molef/PhD/poe_repair_min-dog-and-rain/artifacts/notes/what-poe-loses-when-it-composes")
import toy_sampler_hybrids as t
SEED = 20261022
OUT = "/home/molef/PhD/poe_repair_min-dog-and-rain/artifacts/results/how-much-co-occurrence-the-product-needs"
os.makedirs(OUT, exist_ok=True)

reports, conds = {}, []
for form in (1, 2):
    setup, rep = t.caption_experts(form)
    reports[f"form {form}"] = rep
    conds.append((f"E: competing captions, form {form}, tau {rep['tau']:g}", setup))
for form in (1, 2):
    w = reports[f"form {form}"]["implied_pair_weight_in_circle"]
    conds.append((f"D: hand-weighted at form {form}'s implied w = {w:.4f}", t.hand_weighted(w)))

counts, acc, rows = {}, {}, []
for name, (kind, A, B) in conds:
    t0 = time.time(); t.RNG = np.random.default_rng(SEED)
    lp, sc = t.composed(kind, A, B)
    runs = {"exact": t.exact_samples(kind, A, B), "reverse diffusion": t.reverse(sc),
            "annealed MALA": t.reverse(sc, lp, mala=True)}
    acc[name] = float(np.mean(t.ACCEPT)); t.ACCEPT.clear()
    counts[name] = {}
    for m, x in runs.items():
        labs = t.label(x)
        counts[name][m] = {c: labs.count(c) for c in t.CATS}
        rows.append((name, m, labs.count("two objects, crisp"), np.clip(x[:8], 0, 1)))
    print(f"{name} ({time.time()-t0:.0f}s)", json.dumps(counts[name]), flush=True)

taus = {}
for form in (1, 2):
    grid = np.geomspace(0.25, 20, 25) if form == 1 else np.geomspace(0.1, 10, 25)
    taus[f"form {form}"] = {f"{x:.4g}": t.caption_experts(form, float(x))[1]["implied_pair_weight_in_circle"]
                            for x in grid}

json.dump({"what": "condition E: three captions compete; each expert is read off p(caption|image) by Bayes with a uniform image prior",
           "samples_per_cell": t.N, "rng_seed_per_condition": SEED,
           "forms": {"1": "a caption scores an image by distance to the images it describes",
                     "2": "a caption scores an image by named shapes present, named shapes missing and unnamed shapes present"},
           "reports": reports, "counts": counts, "mala_acceptance": acc,
           "implied_pair_weight_vs_tau": taus},
          open(f"{OUT}/toy-caption-competition.json", "w"), indent=1)

fig, axes = plt.subplots(len(rows), 9, figsize=(16, 1.05 * len(rows) + 0.8),
                         gridspec_kw={"width_ratios": [4.4] + [1] * 8})
for r, (name, m, crisp, imgs) in enumerate(rows):
    axes[r, 0].text(0, 0.5, f"{name}\n{m}\ncrisp pairs: {crisp} of 300", fontsize=8,
                    va="center", transform=axes[r, 0].transAxes)
    axes[r, 0].axis("off")
    for j in range(8):
        axes[r, j + 1].imshow(imgs[j].reshape(t.H, t.W, 3), interpolation="nearest")
        axes[r, j + 1].axis("off")
fig.suptitle("Competing captions against the same weight set by hand", fontsize=12)
fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.08, hspace=0.3)
fig.savefig(f"{OUT}/competing-captions.png", dpi=110)
print("wrote", f"{OUT}/competing-captions.png")
