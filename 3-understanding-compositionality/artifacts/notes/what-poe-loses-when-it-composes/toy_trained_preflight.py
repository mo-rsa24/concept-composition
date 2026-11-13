"""Before any product is read: can one expert reproduce the pictures it was trained on?

Trains a circle-only expert at four step budgets and shows what it draws. An expert that cannot
return its own training set cannot be used to test composition, so this is the check that should
have run before toy_trained_experts.py.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import toy_sampler_hybrids as t
import toy_trained_experts as tr
OUT = HERE.parents[1] / "results" / "how-much-co-occurrence-the-product-needs"
BUDGETS = [6000, 20000, 60000]

rows = []
for steps in BUDGETS:
    tr.STEPS = steps
    rng = np.random.default_rng(tr.SEED)
    t0 = time.time()
    net, loss = tr.train(tr.build_set(tr.CIRCLES, 0.0, rng), tr.SEED)
    torch.manual_seed(tr.SEED)
    x = tr.sample(lambda z, k: net(z, k))
    c = tr.counts(x)
    rows.append((f"circle expert, {steps} steps", c, np.clip(x[:8], 0, 1), round(loss, 5),
                 round(time.time() - t0, 1)))
    print(f"{steps} steps ({rows[-1][4]}s) loss {loss:.5f}: {json.dumps(c)}", flush=True)

rows.append(("the pictures it was trained on", {"one circle": 300}, np.clip(
    tr.CIRCLES[np.random.default_rng(0).integers(3, size=8)]
    + t.SIGMA0 * np.random.default_rng(1).standard_normal((8, t.D)), 0, 1), None, None))

fig, axes = plt.subplots(len(rows), 9, figsize=(16, 1.25 * len(rows) + 1.0),
                         gridspec_kw={"width_ratios": [4.2] + [1] * 8})
for r, (name, c, imgs, loss, secs) in enumerate(rows):
    top = ",  ".join(f"{k} {v}" for k, v in c.items() if v)
    extra = f"\nfinal loss {loss}, {secs}s" if loss is not None else ""
    axes[r, 0].text(0, 0.5, f"{name}\n{top} of 300{extra}", fontsize=9, va="center",
                    transform=axes[r, 0].transAxes)
    axes[r, 0].axis("off")
    for j in range(8):
        axes[r, j + 1].imshow(imgs[j].reshape(t.H, t.W, 3), interpolation="nearest")
        axes[r, j + 1].axis("off")
fig.suptitle("Can one trained expert return the pictures it was trained on? Circles only, no squares anywhere",
             fontsize=13)
fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02, wspace=0.08, hspace=0.35)
fig.savefig(OUT / "can-one-expert-learn-its-own-pictures.png", dpi=110, facecolor="white")
json.dump({"what": "a circle-only expert at three step budgets, 300 draws each",
           "budgets": BUDGETS,
           "counts": {str(s): rows[i][1] for i, s in enumerate(BUDGETS)},
           "final_loss": {str(s): rows[i][3] for i, s in enumerate(BUDGETS)}},
          open(OUT / "toy-trained-preflight.json", "w"), indent=1)
print("wrote", OUT / "can-one-expert-learn-its-own-pictures.png")
