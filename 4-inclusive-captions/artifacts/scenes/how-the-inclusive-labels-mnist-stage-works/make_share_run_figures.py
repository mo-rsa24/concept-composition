"""The MNIST share run (plan 03 of scope 11) at guidance 3, in the layout of r6.

s1  one row per share: what training saw, what "7" alone draws, what the 7 x 8 product draws, and the
    product's count of two separate right digits
s2  the product against what "7" alone draws, every share, three trainings as dots and a line through
    their medians, with the trained share x 256 as a dashed line

Layout and helpers (Fig with its crossing check, the palette) are imported from make_mnist_stage_figures.py,
never edited. Every count comes from counts.json in
4-inclusive-captions/artifacts/results/does-the-share-of-two-digit-pictures-help-the-product-compose-on-mnist/;
every tile mark from scores.csv beside it; every tile from data/mnist/share-samples (share 0: plan 02's
data/mnist/samples/dial-0-seed-<t>.npz). The training table is computed from ShareCanvasMaker's rule
(canvases.py) and MNIST's training labels, as expected draws, not counted draws.

    /home/molef/micromamba/envs/sdxl-local/bin/python -I make_share_run_figures.py [s1 s2]
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_mnist_stage_figures as base  # noqa: E402
from make_mnist_stage_figures import BLUE, GREEN, GREY, INK, ORANGE, RED, Fig  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402  (Agg already selected by the base script)

REPO = HERE.parents[3]
RES = REPO / "4-inclusive-captions/artifacts/results/does-the-share-of-two-digit-pictures-help-the-product-compose-on-mnist"
COUNTS = json.load(open(RES / "counts.json"))
C, V = COUNTS["counts"], COUNTS["verdict"]
sys.path.insert(0, str(REPO / "poe_repair/experiments/inclusive_captions/mnist"))
import share_bar as bar  # noqa: E402

SHARES = ["0", "0.1", "0.2", "0.4", "0.8"]
TRAIN = 1
TWO, WITH2, ALONE = "two separate right digits", "right digit with a second", "right digit alone"

# what training saw: ShareCanvasMaker's rule (canvases.py), expected draws per model
STEPS, BATCH = 6000, 128                       # 4-inclusive-captions/scripts/mnist_share.sbatch: --steps 6000; train.py BATCH
CANVASES = STEPS * BATCH
BLANK_RATE, TWO_SHARE, SEVEN_EIGHT = 0.10, 0.10, 0.5   # canvases.py
_raw = Path(os.environ.get("MNIST_RAW", "/home/molef/PhD/replications/original-superdiff/data/MNIST/raw"))
_y = np.frombuffer(open(_raw / "train-labels-idx1-ubyte", "rb").read()[8:], dtype=np.uint8)
FREQ = np.bincount(_y, minlength=10) / len(_y)
SINGLE = CANVASES * (1 - BLANK_RATE - TWO_SHARE * (1 - BLANK_RATE))


def trained(s):
    s = float(s)
    seven, eight = SINGLE * FREQ[7], SINGLE * FREQ[8]
    return {'"7"': round(seven), "   of which show an 8": round(seven * s), '"8"': round(eight),
            '"7 and 8"': round(CANVASES * TWO_SHARE * (1 - BLANK_RATE) * SEVEN_EIGHT), "blank": round(CANVASES * BLANK_RATE)}


def n(s, t, row, call):
    return C[f"{s}|{t}|{row}"].get(call, 0)


def three(s, row, call):
    return [n(s, t, row, call) for t in (1, 2, 3)]


def tiles(s, t):
    p = (REPO / f"data/mnist/samples/dial-0-seed-{t}.npz") if s == "0" else (REPO / f"data/mnist/share-samples/share-{s}-seed-{t}.npz")
    return np.load(p)


CALLS = {}
for r in csv.DictReader(open(RES / "scores.csv")):
    CALLS[(f"{float(r['share']):g}", int(r["training_seed"]), r["row"], int(r["seed"]))] = r["call"]


def pct(s):
    return f"{100 * float(s):.0f}%"


SHORT2 = {"damaged": "damaged", "one digit": "one", "fused": "fused", "two digits with a wrong identity": "wrong", TWO: "✓"}
SHORT1 = {"damaged": "damaged", "wrong digit": "wrong", ALONE: "alone", WITH2: "+ second"}


# ------------------------------------------------------------------ s1: one row per share
def s1():
    W, H = 2330, 1185
    f = Fig(W, H, "What training saw, what \"7\" alone draws, and what the 7 × 8 product draws, at each share",
            f"One model per row (training {TRAIN}), guidance 3, the same 256 starting noises in every row. Each model trained on "
            f"{CANVASES:,} canvases built fresh from MNIST's 60,000 training digits; the table gives the expected draws per model, "
            "computed from canvases.py's rule, not counted.")
    ys = [960, 775, 590, 405, 220]
    hy = 1090
    f.text(120, hy, "the share", id="h1", color=GREY, size=11.5)
    f.text(400, hy, "what training saw (draws per model)", id="h2", color=GREY, size=11.5)
    f.text(856, hy, "what \"7\" alone draws: seeds 0 to 3", id="h3", color=GREY, size=11.5)
    f.text(1190, hy, "with a second digit", id="h3b", color=GREY, size=11.5)
    f.text(1619, hy, "what the 7 × 8 product draws: seeds 0 to 5, the same seeds in every row", id="h4", color=GREY, size=11.5)
    f.text(2170, hy, "two separate right digits", id="h5", color=INK, size=11.5, weight="bold")
    f.text(2170, hy - 22, "plain product", id="h5b", color=GREY, size=10)
    for r, s in enumerate(SHARES):
        y = ys[r]
        sv = V["shares"][s]
        npz = tiles(s, TRAIN)
        # 1. the share
        f.text(120, y + 80, pct(s), id=f"s{r}", color=INK, size=26, weight="bold")
        f.text(120, y + 42, "of \"7\" training pictures\nalso show an 8", id=f"sl{r}", color=GREY, size=10)
        if s == "0":
            note, col = "plan 02's dial-0 models", GREY
        elif sv["counts"]:
            note, col = "counts toward the verdict", GREEN
        else:
            note, col = f"does not count: \"7\" alone\n{min(sv['seven_alone'])} under its line of {sv['purity_floor']}", RED
        f.text(120, y + 2, note, id=f"sn{r}", color=col, size=10.5)
        # 2. the table
        f.box(f"tb{r}", 230, y - 5, 340, 125, "", fc="white", ec="#d5dbe1")
        for i, (nm, v) in enumerate(trained(s).items()):
            yy = y + 98 - i * 23
            hot = i == 1
            f.text(248, yy, nm, id=f"tn{r}{i}", color=INK if hot else GREY, size=10.5, ha="left", weight="bold" if hot else "normal")
            f.text(555, yy, f"{v:,}", id=f"tv{r}{i}", color=INK if hot else GREY, size=10.5, ha="right",
                   weight="bold" if hot else "normal")
        f.arrow([(570, y + 57), (620, y + 57)], f"tb{r}", f"a{r}0", color="#c3cbd3")
        # 3. "7" alone, seeds 0 to 3
        for q in range(4):
            x = 625 + q * 118
            call = CALLS[(s, TRAIN, "7 alone", q)]
            hot = call == WITH2
            f.img(f"a{r}{q}", npz["7 alone"][q], x, y + 30, 108, 54, ec=ORANGE if hot else "#c9d1d9")
            f.text(x + 54, y + 12, SHORT1[call], id=f"am{r}{q}", color=ORANGE if hot else GREY, size=9.5,
                   weight="bold" if hot else "normal")
        w2 = n(s, TRAIN, "7 alone", WITH2)
        f.text(1190, y + 72, f"{w2} of 256", id=f"w{r}", color=ORANGE, size=16)
        f.text(1190, y + 46, f"trained: {pct(s)}, {256 * float(s):.0f} of 256", id=f"wt{r}", color=GREY, size=9.5)
        f.text(1190, y + 24, "three trainings: " + ", ".join(map(str, three(s, "7 alone", WITH2))), id=f"w3{r}",
               color=GREY, size=9)
        # 4. the 7 x 8 product, seeds 0 to 5
        for q in range(6):
            x = 1295 + q * 118
            call = CALLS[(s, TRAIN, "7 x 8 plain", q)]
            ok = call == TWO
            f.img(f"i{r}{q}", npz["7 x 8 plain"][q], x, y + 30, 108, 54, ec=GREEN if ok else "#c9d1d9")
            f.text(x + 54, y + 12, SHORT2[call], id=f"m{r}{q}", color=GREEN if ok else GREY,
                   size=13 if ok else 9.5, weight="bold" if ok else "normal")
        f.arrow([(1993, y + 57), (2045, y + 57)], f"i{r}5", f"res{r}", color="#c3cbd3")
        # 5. the result
        plain = n(s, TRAIN, "7 x 8 plain", TWO)
        corr = n(s, TRAIN, "7 x 8 corrected", TWO)
        f.text(2170, y + 80, f"{plain} of 256", id=f"res{r}", color=INK, size=24, weight="bold")
        f.text(2170, y + 46, f"corrected: {corr}", id=f"rc{r}", color=GREY, size=11)
        f.text(2170, y + 22, "three trainings: " + ", ".join(map(str, sv["plain_7x8_two"])) + "; corrected "
               + ", ".join(map(str, sv["corrected_7x8_two"])), id=f"r3{r}", color=GREY, size=8.5)
    counting = [s for s in SHARES if s != "0" and V["shares"][s]["counts"]]
    means = {s: np.mean(V["shares"][s]["corrected_7x8_two"]) for s in counting}
    f.text(W / 2, 135, f"Verdict {V['verdict']}: a share counts only if \"7\" alone and \"8\" alone keep their lone digit on at least "
                       f"95% of 256 × (1 − share) seeds in all three trainings, and null means no counting share's corrected 7 × 8 "
                       f"averages {bar.NULL_MAX_MEAN} of 256. Only " + ", ".join(pct(s) for s in counting) + " counts: mean "
                       + ", ".join(f"{m:.0f}" for m in means.values()) + ".", id="foot1", color=INK, size=12)
    p20, w20 = n("0.2", TRAIN, "7 x 8 plain", TWO), n("0.2", TRAIN, "7 alone", WITH2)
    p40, w40 = n("0.4", TRAIN, "7 x 8 plain", TWO), n("0.4", TRAIN, "7 alone", WITH2)
    f.text(W / 2, 103, f"The product's pairs track the pairs each single label already draws: plain 7 × 8 {p20} against \"7\" alone's "
                       f"{w20} at 20%, {p40} against {w40} at 40% (training {TRAIN}); \"7\" alone draws a second digit far more often "
                       "than it was trained to.", id="foot2", color=INK, size=12)
    f.text(W / 2, 68, "Marks are the pinned judge's calls. Under \"7\" alone: alone, + second (the 7 with a second digit beside it), wrong "
                      "(wrong digit), damaged. Under the product: ✓ two separate right digits; one: one digit; wrong: two digits, wrong "
                      "identity.", id="foot3", color=GREY, size=10.5)
    f.check_and_save("s1-one-row-per-share")


# ------------------------------------------------------------------ s2: the product against what each label draws alone
def s2():
    fig = plt.figure(figsize=(1500 / 100, 860 / 100), dpi=100)
    ax = fig.add_axes([0.07, 0.11, 0.70, 0.72])
    x = np.arange(len(SHARES))
    series = [("7 alone", WITH2, BLUE, '"7" alone: the 7 with\na second digit beside it'),
              ("7 x 8 plain", TWO, RED, "7 × 8 product, plain:\ntwo separate right digits"),
              ("7 x 8 corrected", TWO, ORANGE, "7 × 8 product, corrected:\ntwo separate right digits")]
    ends = {}
    for row, call, col, lab in series:
        vals = np.array([three(s, row, call) for s in SHARES])
        med = np.median(vals, axis=1)
        ax.plot(x, med, color=col, lw=2.2, zorder=3)
        for k, off in enumerate((-0.07, 0, 0.07)):
            ax.scatter(x + off, vals[:, k], s=34, color=col, edgecolor="white", lw=0.8, zorder=4)
        ends[lab] = (med[-1], col)
    tr = np.array([256 * float(s) for s in SHARES])
    ax.plot(x, tr, color=GREY, lw=1.1, ls="--", zorder=2)
    ends["trained share × 256\n(\"7\" pictures showing an 8)"] = (tr[-1], GREY)
    # labels to the right of the last point, nudged apart so none sits on another
    items = sorted(ends.items(), key=lambda kv: kv[1][0])
    placed = []
    for lab, (yv, col) in items:
        yy = yv if not placed else max(yv, placed[-1] + 30)
        placed.append(yy)
        ax.annotate(lab, xy=(x[-1] + 0.12, yv), xytext=(x[-1] + 0.3, yy), color=col, size=11.5, va="center",
                    annotation_clip=False, arrowprops=dict(arrowstyle="-", color=col, lw=0.8) if abs(yy - yv) > 4 else None)
    ax.set_xticks(x, [pct(s) for s in SHARES], size=12)
    ax.set_xlabel("share of \"7\" training pictures that also show an 8", size=12, color=INK)
    ax.set_ylabel("count of 256 seeds (pinned judge)", size=12, color=INK)
    ax.set_ylim(-5, 262)
    ax.set_xlim(-0.3, len(SHARES) - 0.7)
    ax.set_yticks([0, 64, 128, 192, 256])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#9aa5b1")
    ax.tick_params(colors=INK)
    ax.grid(axis="y", color="#eceff2", lw=0.8, zorder=0)
    fig.text(0.02, 0.95, "The product draws pairs about as often as \"7\" alone already does", size=20, weight="bold", color=INK)
    fig.text(0.02, 0.905, "Guidance 3, MNIST share run. Each dot is one training (three per share), the line runs through their median; "
                          "share 0 is plan 02's dial-0 models.", size=12, color=GREY)
    fig.text(0.02, 0.025, "Counts from counts.json (scores.csv). \"7\" alone counts the judge's \"right digit with a second\"; the judge "
                          "does not check that the second digit is an 8.", size=10.5, color=GREY)
    base.OUT.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(base.OUT / f"s2-the-product-against-what-each-label-draws-alone.{ext}", dpi=100 if ext == "png" else None)
    plt.close(fig)
    print("wrote s2-the-product-against-what-each-label-draws-alone")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["s1", "s2"]:
        globals()[name]()
