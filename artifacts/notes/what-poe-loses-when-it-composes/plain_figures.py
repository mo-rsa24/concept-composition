"""Simple figures for the supervisor write-up: few words, one job per figure.

Reads the saved samples and counts in artifacts/results/how-much-co-occurrence-the-product-needs/
and writes PNGs to its plain-figures/ folder. Numbers for the exact-formula sweeps are the tables in
that folder's README.md (copied below with the row they come from)."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import toy_sampler_hybrids as t

REPO = HERE.parents[2]
R = HERE.parents[1] / "results" / "how-much-co-occurrence-the-product-needs"
OUT = R / "plain-figures"
PILOT = Path(sys.argv[1]) if len(sys.argv) > 1 else None
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False})
INK, MUTE, GOOD, BAD = "#222222", "#777777", "#1a7f37", "#b42318"
CIRC, SQ, JOINT = '"a red circle"', '"a blue square"', '"a red circle and a blue square"'


def tile(x):
    return np.clip(np.asarray(x).reshape(t.H, t.W, 3), 0, 1)


def strip(ax, imgs, gap=6, slots=None):
    """Lay 16x32 pictures side by side in one axis, each with a thin grey frame, white gaps between."""
    h, w = t.H, t.W
    n = slots or len(imgs)
    canvas = np.ones((h + 2, n * (w + 2 + gap) - gap, 3))
    for i, im in enumerate(imgs):
        x0 = i * (w + 2 + gap)
        canvas[:, x0:x0 + w + 2] = 0.72
        canvas[1:h + 1, x0 + 1:x0 + w + 1] = tile(im)
    ax.imshow(canvas, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("wrote", OUT / name)


def fig_problem():
    seeds = [2, 3, 6, 7]
    joint = [Image.open(REPO / f"artifacts/results/does-the-hybrid-fit-both-captions/joint-tiles/seed{s:02d}.png") for s in seeds]
    poe = [Image.open(REPO / f"artifacts/results/does-poe-of-cats-and-dogs-match-the-joint/tiles/cat__x__dog/seed{s:02d}/poe.png") for s in seeds]
    fig, axs = plt.subplots(2, 4, figsize=(11, 6))
    for j in range(4):
        for i, im in enumerate((joint[j], poe[j])):
            axs[i, j].imshow(im); axs[i, j].axis("off")
        axs[0, j].set_title(f"seed {seeds[j]}", color=MUTE, fontsize=10)
    fig.text(0.01, 0.73, 'One prompt\n"a cat and a dog"', fontsize=13, va="center")
    fig.text(0.01, 0.30, 'PoE\n"a cat" × "a dog"', fontsize=13, va="center")
    fig.subplots_adjust(left=0.17, wspace=0.04, hspace=0.06)
    fig.suptitle("SDXL: one prompt draws a cat and a dog, PoE draws one fused animal", x=0.55, fontsize=14)
    save(fig, "1-the-problem-on-sdxl.png")


def fig_training():
    c, s = list(t.CIRCLES.values()), list(t.SQUARES.values())
    import toy_caption_dial_ldm as m
    p = list(m.PAIRS)
    fig, axs = plt.subplots(3, 2, figsize=(11, 4.6))
    rows = ((CIRC, c, c[:2] + [p[0]]), (SQ, s, s[:2] + [p[1]]), (JOINT, p, p))
    for i, (cap, left, right) in enumerate(rows):
        strip(axs[i, 0], left[:3], slots=3)
        strip(axs[i, 1], right[:3], slots=3)
        axs[i, 0].set_ylabel(cap, rotation=0, ha="right", va="center", fontsize=11)
        axs[i, 0].yaxis.set_label_coords(-0.04, 0.5)
    for ax in (axs[0, 1], axs[1, 1]):   # outline the both-shapes picture now under a single caption
        ax.add_patch(Rectangle((2 * (t.W + 8) - 1.5, -1.5), t.W + 4, t.H + 4.5, fill=False, ec=GOOD, lw=3, clip_on=False))
    axs[0, 0].set_title("Both-shapes pictures carry only the joint caption\n(2,000 pictures under each caption)", fontsize=11)
    axs[0, 1].set_title("Some both-shapes pictures also carry each single caption\n(10% share: about 200 more under each single caption)", fontsize=11)
    fig.suptitle("What the model is trained on: pictures and their captions", fontsize=14, y=1.04)
    fig.subplots_adjust(wspace=0.08, hspace=0.25)
    save(fig, "2-the-training-pictures.png")


def fig_main():
    d = json.load(open(R / "toy-caption-dial-ldm.json"))["results"]
    S = np.load(R / "toy-caption-dial-ldm-samples.npz")
    shares = ["0", "0.1", "0.3", "0.6", "1"]
    cols = [(CIRC, 0, "a red circle alone", "red only", "red only"),
            (SQ, 1, "a blue square alone", "blue only", "blue only"),
            ('PoE: "a red circle" × "a blue square"', 2, "the two single captions multiplied", "red and blue apart", "apart"),
            (JOINT, 3, "the joint caption", "red and blue apart", "apart")]
    fig, axs = plt.subplots(len(shares), 4, figsize=(15, 7.2))
    for i, sh in enumerate(shares):
        for j, (title, idx, key, cat, word) in enumerate(cols):
            ax = axs[i, j]; strip(ax, S[f"{sh}_{idx}"][:4])
            n = d[sh][key]["blob scorer"][cat]
            ax.set_xlabel(f"{n} of 300 {word}", fontsize=11, color=INK, fontweight="bold" if j == 2 else "normal")
            if i == 0: ax.set_title(title, fontsize=11.5)
        axs[i, 0].set_ylabel(f"{float(sh):.0%}", rotation=0, ha="right", va="center", fontsize=13)
        axs[i, 0].yaxis.set_label_coords(-0.04, 0.5)
    fig.text(0.035, 0.5, "share of both-shapes pictures\nalso given each single caption", rotation=90, va="center", ha="center", fontsize=11, color=MUTE)
    fig.suptitle("One caption-conditioned latent diffusion model per row: PoE draws both shapes only once some\n"
                 "both-shapes training pictures carry the single captions", fontsize=14, y=1.0)
    fig.subplots_adjust(left=0.1, wspace=0.08, hspace=0.55)
    save(fig, "3-the-main-result.png")


def fig_bars():
    d = json.load(open(R / "toy-caption-dial-ldm.json"))["results"]
    shares = ["0", "0.1", "0.3", "0.6", "1"]
    apart = [d[s]["the two single captions multiplied"]["blob scorer"]["red and blue apart"] for s in shares]
    joint = [d[s]["the joint caption"]["blob scorer"]["red and blue apart"] for s in shares]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(shares))
    ax.bar(x - 0.2, joint, 0.38, color="#c9c9c9", label="one prompt: " + JOINT)
    ax.bar(x + 0.2, apart, 0.38, color="#3b6fb6", label='PoE: "a red circle" × "a blue square"')
    for xi, v in zip(x, apart): ax.text(xi + 0.2, v + 5, str(v), ha="center", fontsize=10)
    ax.set_xticks(x, [f"{float(s):.0%}" for s in shares])
    ax.set_xlabel("share of both-shapes pictures also given each single caption")
    ax.set_ylabel("samples with a separate red circle\nand blue square (of 300)")
    ax.set_ylim(0, 330); ax.legend(frameon=False, loc="upper left", fontsize=9.5, bbox_to_anchor=(0, 1.12))
    save(fig, "4-the-count-by-share.png")


def fig_exact():
    # README.md, "The sweep, weight by weight" and "How far the dial has to go" (reverse diffusion column)
    w = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 0.65, 0.8, 0.9]
    exact = [300] * len(w)
    rd = [0, 0, 0, 1, 10, 122, 238, 281, 298, 300]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(w, exact, "-o", color=GOOD, label="exact answer, computed from the formula")
    ax.plot(w, rd, "-o", color="#3b6fb6", label="what the step-by-step sampler draws")
    ax.set_xscale("log"); ax.set_ylim(-10, 330)
    ax.set_xlabel("probability each expert gives the both-shapes picture (log scale)")
    ax.set_ylabel("samples that are the both-shapes\npicture (of 300)")
    ax.legend(frameon=False, loc="center right", fontsize=9.5)
    ax.set_title("Experts written as formulas: PoE wants the pair at 0.1%,\nthe sampler only finds it near 30%", fontsize=12)
    save(fig, "5-formula-experts-answer-against-sampler.png")


def fig_leak():
    # README.md, "The second dial" table, exact column; pair probability held at 0.1
    leak = [0.0001, 0.003, 0.01, 0.1]; crisp = [300, 201, 113, 20]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(leak, crisp, "-o", color=BAD)
    ax.set_xscale("log"); ax.set_ylim(-10, 330)
    ax.set_xticks(leak, ["0", "0.3%", "1%", "10%"])
    ax.set_xlabel("probability the circle expert gives each single square\n(and the square expert each single circle)")
    ax.set_ylabel("exact answer: samples that are\nthe both-shapes picture (of 300)")
    ax.set_title("A little probability on the other shape turns PoE's answer into one shape", fontsize=12)
    save(fig, "6-experts-that-also-score-the-other-shape.png")


def fig_sampler():
    S = np.load(R / "toy-noised-product-samples.npz")
    j = json.load(open(R / "toy-noised-product.json"))["results"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 1.9))
    for ax, i, name, key in ((axs[0], 0, "adding the two experts' noise estimates", "reverse diffusion, summed noised scores"),
                             (axs[1], 1, "the correct noise estimate for the product", "reverse diffusion, true noised-product score")):
        strip(ax, S[f"0.1_{i}"][:6])
        n = j["0.1"][key]["two objects, crisp"]
        ax.set_title(name, fontsize=11.5); ax.set_xlabel(f"{n} of 300 are the both-shapes picture", fontsize=11,
                                                       color=GOOD if n >= 150 else BAD, fontweight="bold")
    fig.suptitle("Same sampler, same experts (both give the pair 10%): only the noise estimate differs", fontsize=13, y=1.02)
    save(fig, "7-where-the-sampler-loses-the-pair.png")


def fig_factorized():
    S = np.load(R / "toy-factorized-samples.npz")
    f = json.load(open(R / "toy-factorized.json"))["results"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 3.6))
    for c, bg in enumerate(("grey", "white")):
        for r, (idx, name, key) in enumerate(((1, "exact answer", "exact composition, by the pairing drawn"),
                                               (2, "step-by-step sampler", "reverse diffusion, summed noised scores"))):
            ax = axs[r, c]; strip(ax, S[f"{bg}_{idx}"][:5])
            n = f[bg][key]["unseen layouts"]
            ax.set_xlabel(f"{n} of 300 in a layout no training picture holds", fontsize=10.5,
                          color=GOOD if n >= 100 else BAD, fontweight="bold")
            if c == 0: ax.set_ylabel(name, rotation=0, ha="right", va="center"); ax.yaxis.set_label_coords(-0.03, 0.5)
        axs[0, c].set_title(f"{bg} background", fontsize=12)
    fig.suptitle("Circle expert × square expert ÷ empty-background expert, no both-shapes picture in training",
                 fontsize=13, y=1.04)
    fig.subplots_adjust(hspace=0.45, wspace=0.08)
    save(fig, "8-composing-with-no-both-shapes-pictures.png")


def fig_pilot():
    if PILOT is None: return
    seeds = [0, 1, 3, 5, 6, 7]
    rows = (("plain SDXL", "plain"), ("trained, both-animal photos\ncaptioned only jointly", "v4-run-A"),
            ("trained, both-animal photos\nalso captioned \"cat\" or \"dog\"", "v4-run-B"))
    fig, axs = plt.subplots(3, len(seeds), figsize=(13, 6.8))
    for i, (lab, d) in enumerate(rows):
        for j, s in enumerate(seeds):
            axs[i, j].imshow(Image.open(PILOT / d / f"seed_{s:04d}_product.png")); axs[i, j].axis("off")
            if i == 0: axs[i, j].set_title(f"seed {s}", color=MUTE, fontsize=10)
        fig.text(0.005, 0.78 - i * 0.29, lab, fontsize=11.5, va="center")
    fig.subplots_adjust(left=0.2, wspace=0.03, hspace=0.05)
    fig.suptitle('SDXL, PoE "a cat" × "a dog", same seeds: one animal in every picture, before and after training',
                 x=0.58, fontsize=13)
    save(fig, "10-sdxl-inclusive-captions-pilot.png")


def fig_threshold():
    p = R / "toy-caption-threshold.json"
    if not p.exists(): return
    d = _threshold_rows()
    ks = [int(k) for k in d]
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(ks))
    for xi, k in zip(xs, ks):
        v = [r["apart"] for r in d[str(k)]]
        ax.scatter([xi] * len(v), v, color="#3b6fb6", s=45, zorder=3)
        ax.plot([xi - 0.2, xi + 0.2], [np.median(v)] * 2, color=INK, lw=2)
    ax.set_xticks(xs, [str(k) for k in ks]); ax.set_ylim(-10, 330)
    ax.set_xlabel("number of both-shapes pictures also given each single caption (of 2,000)")
    ax.set_ylabel("PoE samples with a separate red circle\nand blue square (of 300)")
    ax.set_title("Each dot is one training, the line their median", fontsize=11.5)
    save(fig, "9-how-many-pictures-it-takes.png")


def _threshold_rows():
    rows = {}
    for f in ("toy-caption-threshold.json", "toy-caption-threshold-600-1200-2000.json"):
        if (R / f).exists():
            rows.update(json.load(open(R / f))["results"])
    return dict(sorted(rows.items(), key=lambda kv: int(kv[0])))


def fig_what_poe_draws():
    rows = _threshold_rows()
    if not rows: return
    cats = [("red and blue apart", "separate red circle and blue square", "#2f6fbf"),
            ("purple or overlaid", "purple hybrid", "#8e4fa8"),
            ("single", "one shape only", "#bdbdbd"),
            ("nothing readable", "faint smear", "#ece6f0")]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    x, ticks, labels = 0, [], []
    for k, trainings in rows.items():
        xs = []
        for r in trainings:
            c = dict(r["counts"]["product"]); c["single"] = c["red only"] + c["blue only"]
            bottom = 0
            for key, name, col in cats:
                ax.bar(x, c[key], 0.8, bottom=bottom, color=col, edgecolor="white", lw=0.5,
                       label=name if x == 0 else None)
                bottom += c[key]
            xs.append(x); x += 1
        ticks.append(np.mean(xs)); labels.append(k); x += 1.2
    ax.set_xticks(ticks, labels); ax.set_ylim(0, 300); ax.set_yticks([0, 100, 200, 300])
    ax.set_xlabel("number of the 2,000 both-shapes training pictures also captioned \"a red circle\" and \"a blue square\"\n(three trainings at each number, one bar each)")
    ax.set_ylabel("what PoE draws\n(300 samples per bar)")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=10)
    ax.set_title("Without the single captions PoE draws a purple hybrid; with enough of them it starts drawing both shapes",
                 fontsize=12.5, pad=34)
    save(fig, "0-what-poe-draws-as-the-captions-change.png")


def fig_samples_side_by_side():
    npz = {}
    for f in ("toy-caption-threshold-samples.npz", "toy-caption-threshold-600-1200-2000-samples.npz"):
        if (R / f).exists():
            npz.update(dict(np.load(R / f)))
    rows = _threshold_rows()
    picks = [("0", 20261023), ("200", 20261023), ("2000", None)]
    if "2000" not in rows: return
    best = max(rows["2000"], key=lambda r: r["apart"])["seed"]
    picks[2] = ("2000", best)
    fig, axs = plt.subplots(1, 3, figsize=(15, 1.9))
    for ax, (k, sd) in zip(axs, picks):
        strip(ax, npz[f"{k}_{sd}_product"][:4])
        r = next(r for r in rows[k] if r["seed"] == sd)
        best = "" if k == "0" else ", the best of its three trainings"
        ax.set_title(f"{int(k):,} pictures{best}", fontsize=11)
        ax.set_xlabel(f"{r['apart']} of 300 separate, {r['counts']['product']['purple or overlaid']} of 300 purple",
                      fontsize=10.5)
    fig.suptitle('PoE "a red circle" × "a blue square": first four samples, by how many both-shapes pictures\nalso carry the single captions', fontsize=12.5, y=1.1)
    save(fig, "0b-poe-samples-side-by-side.png")


def fig_corrector():
    files = [("toy-caption-corrector.json", "toy-caption-corrector-samples.npz"),
             ("toy-caption-corrector-no-single-captions.json", "toy-caption-corrector-no-single-captions-samples.npz")]
    rows, npz = [], {}
    for fj, fn in files:
        if (R / fj).exists():
            rows += json.load(open(R / fj))["results"]; npz.update(dict(np.load(R / fn)))
    if not rows: return
    rows.sort(key=lambda r: (r["pictures"], r["seed"]))
    cols = [("plain", "plain sampling"), ("main", "5 corrector steps per level"), ("stronger", "20 corrector steps per level")]
    fig, axs = plt.subplots(len(rows), 3, figsize=(14, 1.25 * len(rows) + 0.8))
    for i, r in enumerate(rows):
        for j, (key, title) in enumerate(cols):
            ax = axs[i, j]; strip(ax, npz[f"{r['pictures']}_{r['seed']}_{key}"][:4])
            n = r[key]["product"]["red and blue apart"]
            ax.set_xlabel(f"{n} of 300 separate", fontsize=10.5, fontweight="bold")
            if i == 0: ax.set_title(title, fontsize=11.5)
        axs[i, 0].set_ylabel(f"{r['pictures']:,} pictures,\nmodel {SEEDS_SHORT[r['seed']]}", rotation=0,
                             ha="right", va="center", fontsize=10.5)
        axs[i, 0].yaxis.set_label_coords(-0.04, 0.5)
    fig.suptitle('PoE "a red circle" × "a blue square" in the caption model, with and without a Langevin corrector\n'
                 "(rows: how many both-shapes pictures also carry the single captions)", fontsize=12.5, y=1.0)
    fig.subplots_adjust(hspace=0.75, wspace=0.08, left=0.13)
    save(fig, "11-the-corrector.png")


SEEDS_SHORT = {20261022: "A", 20261023: "B", 20261024: "C"}


if __name__ == "__main__":
    for f in (fig_problem, fig_training, fig_main, fig_exact, fig_leak, fig_sampler,
              fig_factorized, fig_pilot, fig_what_poe_draws, fig_samples_side_by_side, fig_corrector):
        f()
