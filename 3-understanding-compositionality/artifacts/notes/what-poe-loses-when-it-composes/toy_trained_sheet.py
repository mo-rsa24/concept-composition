"""What two separately trained networks draw, alone and multiplied, at three shared fractions."""
import sys
from pathlib import Path
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import toy_sampler_hybrids as t, toy_trained_experts as tr
OUT = HERE.parents[1] / "results" / "how-much-co-occurrence-the-product-needs"

rows = []
for frac in (0.0, 0.1, 0.5):
    rng = np.random.default_rng(tr.SEED)
    A, _ = tr.train(tr.build_set(tr.CIRCLES, frac, rng), tr.SEED)
    B, _ = tr.train(tr.build_set(tr.SQUARES, frac, rng), tr.SEED + 1)
    ea, eb = tr.eps_of(A), tr.eps_of(B)
    for name, fn in (("the circle network alone", ea), ("the square network alone", eb),
                     ("the two multiplied", lambda x, k: ea(x, k) + eb(x, k))):
        torch.manual_seed(tr.SEED)
        x = tr.sample(fn)
        labs = t.label(x)
        top = max(t.CATS, key=labs.count)
        rows.append((f"shared pictures: {frac:g}\n{name}", f"{top}, {labs.count(top)} of 300",
                     np.clip(x[:8], 0, 1)))
        print(rows[-1][0].replace("\n", " | "), "->", rows[-1][1], flush=True)

fig, axes = plt.subplots(len(rows), 9, figsize=(16, 1.15 * len(rows) + 1.0),
                         gridspec_kw={"width_ratios": [3.6] + [1] * 8})
for r, (name, top, imgs) in enumerate(rows):
    axes[r, 0].text(0, 0.5, f"{name}\n{top}", fontsize=9, va="center", transform=axes[r, 0].transAxes)
    axes[r, 0].axis("off")
    for j in range(8):
        axes[r, j + 1].imshow(imgs[j].reshape(t.H, t.W, 3), interpolation="nearest")
        axes[r, j + 1].axis("off")
fig.suptitle("Two networks trained separately, each alone and then multiplied", fontsize=13)
fig.subplots_adjust(left=0.01, right=0.99, top=0.945, bottom=0.015, wspace=0.08, hspace=0.3)
fig.savefig(OUT / "what-two-trained-networks-draw.png", dpi=110, facecolor="white")
print("wrote", OUT / "what-two-trained-networks-draw.png")
