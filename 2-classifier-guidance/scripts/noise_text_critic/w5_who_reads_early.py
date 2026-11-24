"""W5: readability against sampler step on the sampler's real states, from W2c's saved result (group 3)."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
O = "/home/molef/PhD/concept-composition/artifacts/notes/progress/noise-text-critic"
r = json.load(open(f"{O}/02c-fair-test/result.json"))["group_b"]
steps, res = r["steps"], r["clean_pair"]
rows = [("readout", "SDXL's own features (a small trained reader)", "#1d7a3e", 2.6),
        ("latent_clip", "Latent CLIP, sentence score", "#7a4fc9", 1.6),
        ("noisyclip", "NoisyCLIP, sentence score", "#1f5fd0", 1.6),
        ("frozen_clip", "frozen CLIP, sentence score", "#8a939c", 1.6)]
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.axvspan(-0.3, 15.3, color="#fff1e0", zorder=0)
ax.text(7.5, 0.965, "steps 0 to 15: where a push would act", ha="center", va="top", fontsize=10, color="#c86a0a")
ax.axhline(0.75, color="#c0262d", lw=1, ls="--"); ax.text(29.8, 0.755, "bar 0.75", color="#c0262d", fontsize=9, va="bottom", ha="right")
ax.axhline(0.5, color="0.6", lw=1, ls=":"); ax.text(29.8, 0.495, "chance 0.5", color="0.45", fontsize=9, va="top", ha="right")
ends = {}
for key, name, colour, lw in rows:
    ys = res[key]["balanced_accuracy_reading_half"]
    ax.plot(steps, ys, "o-", ms=3.2, lw=lw, color=colour)
    ends[key] = (ys[-1], name, colour)
place = {"readout": 0.87, "noisyclip": 0.605, "latent_clip": 0.545, "frozen_clip": 0.455}
for key, (y, name, colour) in ends.items():
    ax.text(20.5, place[key], name, color=colour, fontsize=9.5, va="center")
ax.set_xlim(-0.3, 30)
ax.set_xticks(range(0, 21, 5))
ax.set_ylim(0.35, 1.0)
ax.set_xlabel("sampler step (0 = pure noise; SDXL draws 50 steps)")
ax.set_ylabel("balanced accuracy: clean cat and dog\nagainst a failed picture")
ax.set_title("On the sampler's real states, only SDXL's own features read the picture early", fontsize=11.5, loc="left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0.01, 0.01, "195 held-out cat and dog runs (105 clean), threshold set on half the seeds and read on the other half. W2c, W&B etbecql0.",
         fontsize=8, color="0.4")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{O}/who-reads-early.png", dpi=130)
