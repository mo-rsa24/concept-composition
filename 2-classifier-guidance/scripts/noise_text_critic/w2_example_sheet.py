"""Six finished cat-and-dog pictures with NoisyCLIP's cosine to each counting sentence at step 49, from W2's saved scores.

The first picture of each kind in SHA-256 order of its id is shown, three that NoisyCLIP calls right and the first miss of each
two-animal and one-animal kind. Usage: python w2_example_sheet.py <folder holding scores.npz and the six jpgs> <out dir>
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
src, out = Path(sys.argv[1]), Path(sys.argv[2])
S = np.load(src / "scores.npz")
keys, sents, steps = list(S["keys"]), list(S["sentences"]), list(S["steps"])
res = json.loads((REPO / "artifacts/notes/progress/noise-text-critic/02-can-it-read/result.json").read_text())
k49 = steps.index(49)
thr = res["counting"]["noisyclip"]["thresholds"][k49]
right = "a photo of a cat and a dog, two separate animals"
wrong = ["a photo of one animal", "one animal that is part cat and part dog"]
pics = [("more_catdog", 239, "clean pair", 1), ("more_catdog", 230, "hybrid", 0), ("more_catdog2", 669, "single animal", 0),
        ("more_catdog2", 696, "clean pair", 1), ("more_catdog2", 637, "two cats", 1), ("more_catdog2", 644, "hybrid (cat with a dog's paw)", 0)]
ids = {"more_catdog": "more_catdog", "more_catdog2": "more2"}
fig, axes = plt.subplots(2, 6, figsize=(21, 7.6), gridspec_kw={"height_ratios": [1.2, 1]})
rows = []
for c, (folder, seed, kind, y) in enumerate(pics):
    pid = f"{ids[folder]}-a_cat__x__a_dog__seed{seed}__joint"
    cos = S["noisyclip"][k49, :, keys.index(pid), :].mean(0)
    vals = [float(cos[sents.index(right)])] + [float(cos[sents.index(w)]) for w in wrong]
    score = vals[0] - max(vals[1:])
    call = int(score > thr)
    ok = call == y
    rows.append({"picture": pid, "label": kind, "cosines": [round(v, 4) for v in vals], "score": round(score, 4), "right": bool(ok)})
    ax = axes[0, c]
    ax.imshow(Image.open(src / f"{folder}-a_cat__x__a_dog__seed{seed}__joint.jpg"))
    ax.set_axis_off()
    ax.set_title(f"seed {seed}: {kind}", fontsize=10.5)
    ax.text(0.5, -0.03, ("called two animals" if call else "called one animal") + (", right" if ok else ", wrong"),
            transform=ax.transAxes, ha="center", va="top", fontsize=11, fontweight="bold", color="#1d7a3e" if ok else "#c0262d")
    b = axes[1, c]
    b.barh(range(3), vals, color=["#1f5fd0", "#9aa5b1", "#9aa5b1"])
    b.set_yticks(range(3), ["two separate animals\n(the right sentence)", "one animal", "part cat, part dog"] if c == 0 else [""] * 3, fontsize=9)
    b.invert_yaxis()
    b.set_xlim(0.10, 0.27)
    for k, v in enumerate(vals):
        b.text(v + 0.002, k, f"{v:.3f}", va="center", fontsize=8.5)
    b.set_xlabel("cosine to the sentence", fontsize=9)
    b.set_title(f"right minus best wrong = {score:+.3f}", fontsize=9.5)
    for s in ("top", "right"):
        b.spines[s].set_visible(False)
fig.text(0.5, 0.985, f"NoisyCLIP's counting score at step 49 of 50. It calls 'two animals' when the score is above {thr:+.4f}, the threshold "
         "W2 set on the other half of the seeds. Over all 220 pictures it is right on 107 of 109 clean pairs, 23 of 46 two-of-a-kind, "
         "18 of 29 hybrids and 1 of 1 single.", ha="center", va="top", fontsize=11, wrap=True)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "noisyclip-counting-on-six-finished-pictures.png", dpi=100)
(out / "noisyclip-counting-on-six-finished-pictures.json").write_text(json.dumps(
    {"step": 49, "threshold": thr, "rows": rows, "source": "loop-9/w2/read/scores.npz, W&B 70g3qw2f"}, indent=1))
print(json.dumps(rows))
