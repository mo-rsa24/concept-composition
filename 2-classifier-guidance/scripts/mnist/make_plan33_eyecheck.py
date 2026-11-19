"""Plan 33, the final read counted by eye: every tile the judge did not accept on the weakest pairs, enlarged, labelled pair:index.
Writes eyecheck-N.png and eyecheck-index.json (sheet -> labels). Run with the sdxl-local python from 2-classifier-guidance/."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "scripts/mnist")
from judge import load as load_judge
import two_experts as te
F = Path("data/mnist-canvas/33-mnist-experts-at-inference/final")
net = load_judge(Path("data/mnist-canvas/32-mnist-two-experts/judge-v3/judge.pt"), "cpu")
cv = np.load(F / "final-crop-both-lang5c3/canvases.npz")
cnt = json.load(open(F / "final-crop-both-lang5c3/counts.json"))["per_pair"]
weak = sorted((v["two clear, right, apart"], k) for k, v in cnt.items() if v["two clear, right, apart"] < 58)
tiles = []
for _, pair in weak:
    a, b = (int(v) for v in pair.split("-"))
    for i in range(64):
        if te.call_two(te.reads(net, cv[pair][i], "cpu"), (a, b)) != te.CALLS[0]:
            tiles.append((pair, i, cv[pair][i]))
print(len(tiles), "tiles to call")
per, sheets, index = 30, [], {}
for s in range(0, len(tiles), per):
    chunk = tiles[s:s + per]
    fig, axs = plt.subplots(5, 6, figsize=(14, 7.6))
    for ax in axs.flat:
        ax.axis("off")
    for ax, (pair, i, c) in zip(axs.flat, chunk):
        ax.imshow(c, cmap="gray", vmin=0, vmax=255, interpolation="nearest"); ax.set_title(f"{pair} #{i}", fontsize=13, color="#c8680a")
    fig.tight_layout(); n = s // per + 1
    fig.savefig(F / f"eyecheck-{n}.png", dpi=80); plt.close(fig)
    index[n] = [f"{p}:{i}" for p, i, _ in chunk]
json.dump(index, open(F / "eyecheck-index.json", "w"), indent=1)
print("sheets", len(index))
