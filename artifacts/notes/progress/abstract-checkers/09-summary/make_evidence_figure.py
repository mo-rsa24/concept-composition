import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
base = "06-does-it-compose"
plain, push = mpimg.imread(f"{base}/unseen-plain.png"), mpimg.imread(f"{base}/unseen-push.png")
h, w = plain.shape[:2]
crop = lambda im: im[int(0.045*h):int(0.045*h + 0.385*h), int(0.03*w):, :3]
ink, mute, A = "#1f1f1f", "#6b6b6b", "#2a78d6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
f = plt.figure(figsize=(14, 7.2), dpi=160); f.patch.set_facecolor("#fcfcfb")
g = f.add_gridspec(2, 2, width_ratios=[1.55, 1], hspace=0.22, wspace=0.18)
for r, (im, t) in enumerate([(plain, "Plain product: one fused glyph per canvas"), (push, "Same seeds with the count checker's push: two digits")]):
    ax = f.add_subplot(g[r, 0]); ax.imshow(crop(im)); ax.axis("off"); ax.set_title(t, loc="left", color=ink, fontsize=13)
ax = f.add_subplot(g[:, 1]); ax.set_facecolor("#fcfcfb")
rows = ["Plain\nproduct", "Random\npush", "Best of\n4 seeds", "Count\nchecker\npush"]
vals = [0, 1, 0, 43]
ax.bar(range(4), vals, color=[mute, mute, mute, A], width=0.62)
ax.errorbar([3], [43], yerr=1.96 * 7.58, color=ink, capsize=6, lw=1.5)
for i, v in enumerate(vals):
    ax.text(i, v + (16.5 if i == 3 else 1.5), str(v), ha="center", color=ink, fontsize=14, fontweight="bold" if i == 3 else None)
ax.set_xticks(range(4)); ax.set_xticklabels(rows, fontsize=11); ax.set_ylim(0, 66)
ax.set_ylabel("clean pairs of 320 (digits 5 to 9, never seen)", color=ink, labelpad=8)
ax.spines[["top", "right"]].set_visible(False); ax.spines[["left", "bottom"]].set_color("#bdbdbd"); ax.tick_params(colors=mute)
ax.set_title("Only the checker's push composes", loc="left", color=ink, fontsize=13)
ax.text(3.33, 30, "±1.96 standard\nerrors of the lead", color=mute, fontsize=9.5, va="center")
f.text(0.01, 0.01, "Sheets: pairs 5 and 6, 5 and 7, 5 and 8, 5 and 9 (first 8 of 32 seeds each); a green frame means the judge calls it two clear, right digits apart. "
       "Source: V3 run, W&B qhusod91. A rerun on another node gave 41.", color=mute, fontsize=9.5)
f.savefig("09-summary/evidence-at-a-glance.png", facecolor=f.get_facecolor(), bbox_inches="tight")
