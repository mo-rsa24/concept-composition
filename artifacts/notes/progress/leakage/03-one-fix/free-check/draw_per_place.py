"""Free check for Z3: does CO3's own per-place signal find the borrowing body?

For each cell of the 128 x 128 latent, at the last corrector pass of steps 1, 3, 4 and 5, the distance between each animal's
noise prediction and the joint prompt's (length over the 4 channels), and their difference: blue where the first animal's
prediction is closer to the joint one, red where the second's is. Records: the CO3 capture run (Slurm job 94450, code snapshot
44b51af4), passes.npz and co3.png for each pair at seed 6001, in the folder given.
Usage: draw_per_place.py <co3-inside folder>
"""
import json
import sys

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = sys.argv[1]
PAIRS = [("a_cheetah__x__a_cougar", "cheetah", "cougar", "borrowed: cougar has cheetah spots"),
         ("an_eagle__x__a_fox", "eagle", "fox", "own parts")]
STEPS = [1, 3, 4, 5]
fig, ax = plt.subplots(len(PAIRS), 1 + len(STEPS), figsize=(4.2 * (1 + len(STEPS)), 4.6 * len(PAIRS)))
numbers = {}
for r, (pair, a, b, w2d) in enumerate(PAIRS):
    d = np.load(f"{R}/{pair}/seed-6001/passes.npz")
    steps, eps = d["step"], d["eps"].astype(np.float32)
    ax[r, 0].imshow(Image.open(f"{R}/{pair}/seed-6001/co3.png"))
    ax[r, 0].set_title(f"CO3's finished picture, seed 6001\n{a} x {b}; W2d: {w2d}", fontsize=10)
    numbers[pair] = {}
    for c, s in enumerate(STEPS, start=1):
        i = int(np.where(steps == s)[0][-1])
        _, joint, ea, eb = eps[i]
        diff = np.linalg.norm(ea - joint, axis=0) - np.linalg.norm(eb - joint, axis=0)
        v = float(np.percentile(np.abs(diff), 98))
        ax[r, c].imshow(diff, cmap="coolwarm", vmin=-v, vmax=v)
        share = float((diff < 0).mean())
        numbers[pair][s] = {"share_first_closer": round(share, 3), "p98_abs": round(v, 4)}
        ax[r, c].set_title(f"step {s}: blue = {a} closer to the joint,\nred = {b} closer; blue on {share:.0%} of cells", fontsize=9)
for x in ax.ravel():
    x.set_xticks([]); x.set_yticks([])
fig.suptitle("Free check: per cell, which animal's noise prediction is closer to the joint prompt's, at CO3's corrected steps "
             "(colour scale: the 98th percentile of each map's difference)", fontsize=11)
plt.tight_layout()
plt.savefig("per-place-distance-maps.png", dpi=80)
json.dump(numbers, open("per-place-distance-maps.json", "w"), indent=1)
print(json.dumps(numbers))
