"""Figures for the supervisor write-up, all in one layout.

Every figure has: a title that states the finding, one line saying what the rows and columns are,
framed samples with one count under each cell, and a "Look at" line naming the outlined cells.
Reads saved samples and counts from artifacts/results/how-much-co-occurrence-the-product-needs/ and
writes PNGs to its plain-figures/ folder. Pass the folder of SDXL pilot renders as the first
argument to draw the pilot figure."""
import json, sys, textwrap
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
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
INK, MUTE, LOOK = "#222222", "#666666", "#e08a00"
RUN = {20261022: "run 1", 20261023: "run 2", 20261024: "run 3"}


def tile(x):
    return np.clip(np.asarray(x).reshape(t.H, t.W, 3), 0, 1)


def strip(ax, imgs, gap=6, slots=None):
    h, w = t.H, t.W
    n = slots or len(imgs)
    canvas = np.ones((h + 2, n * (w + 2 + gap) - gap, 3))
    for i, im in enumerate(imgs):
        x0 = i * (w + 2 + gap)
        canvas[:, x0:x0 + w + 2] = 0.72
        canvas[1:h + 1, x0 + 1:x0 + w + 1] = tile(im)
    ax.imshow(canvas, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)


def outline(fig, ax):
    """Amber box around one cell: the picture and the count under it, not the row label."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [ax.get_window_extent(r)]
    if ax.xaxis.label.get_text(): boxes.append(ax.xaxis.label.get_window_extent(r))
    x0 = min(b.x0 for b in boxes); y0 = min(b.y0 for b in boxes)
    x1 = max(b.x1 for b in boxes); y1 = max(b.y1 for b in boxes)
    inv = fig.transFigure.inverted()
    (fx0, fy0), (fx1, fy1) = inv.transform((x0, y0)), inv.transform((x1, y1))
    fig.patches.append(Rectangle((fx0 - 0.004, fy0 - 0.006), fx1 - fx0 + 0.008, fy1 - fy0 + 0.012,
                                 transform=fig.transFigure, fill=False, ec=LOOK, lw=3))


def frame(fig, title, how, look, height, extra_top=0.45, bottom_in=0.55):
    width = int(fig.get_size_inches()[0] * 10.5)
    """Title, wrapped how-to-read line, and the Look-at line; returns the top fraction for the axes."""
    how = "\n".join(textwrap.wrap(how, width))
    look = "\n".join(textwrap.wrap("Look at: " + look, int(width * 0.92)))
    n_title, n_how = title.count("\n") + 1, how.count("\n") + 1
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1 - 0.12 / height, va="top")
    y_how = 1 - (0.12 + 0.3 * n_title + 0.08) / height
    fig.text(0.5, y_how, how, ha="center", va="top", fontsize=10.5, color=MUTE, linespacing=1.3)
    fig.text(0.5, 0.15 / height, look, ha="center", va="bottom", fontsize=11.5, color=LOOK, fontweight="bold")
    top_in = 0.12 + 0.3 * n_title + 0.08 + 0.21 * n_how + extra_top
    return 1 - top_in / height, bottom_in / height, top_in


def grid(name, title, how, look, rows, cols, cell, marks=(), w=14.0, rowh=1.15, left=0.16, n=4):
    """rows, cols: labels. cell(i, j) -> (pictures, count text). marks: cells to outline."""
    n_title = title.count("\n") + 1; n_how = len(textwrap.wrap(how, int(w * 10.5)))
    height = rowh * len(rows) + 0.12 + 0.3 * n_title + 0.08 + 0.21 * n_how + 0.45 + 0.75
    fig, axs = plt.subplots(len(rows), len(cols), figsize=(w, height), squeeze=False)
    for i, rl in enumerate(rows):
        for j, cl in enumerate(cols):
            got = cell(i, j)
            imgs, txt = got[0], got[1]
            strip(axs[i, j], imgs[:n], slots=n)
            if len(got) > 2:                     # outline individual pictures inside the cell
                for q, on in enumerate(got[2][:n]):
                    if on:
                        axs[i, j].add_patch(Rectangle((q * (t.W + 8) - 1.5, -1.5), t.W + 4, t.H + 4,
                                                      fill=False, ec=LOOK, lw=2.5, clip_on=False))
            axs[i, j].set_xlabel(txt, fontsize=10.5, color=INK)
            if i == 0: axs[i, j].set_title(cl, fontsize=11.5, pad=14)
        axs[i, 0].set_ylabel(rl, rotation=0, ha="right", va="center", fontsize=10.5)
        axs[i, 0].yaxis.set_label_coords(-0.03, 0.5)
    top, bottom, _ = frame(fig, title, how, look, height, bottom_in=0.75)
    fig.subplots_adjust(left=left, right=0.99, top=top, bottom=bottom, hspace=0.8, wspace=0.07)
    for i, j in marks: outline(fig, axs[i, j])
    fig.savefig(OUT / name, dpi=200, facecolor="white"); plt.close(fig)
    print("wrote", OUT / name)


def photos(name, title, how, look, rows, files, coltitles, marks=(), left=0.2):
    n_title = title.count("\n") + 1; n_how = len(textwrap.wrap(how, 136))
    height = 2.3 * len(rows) + 0.12 + 0.3 * n_title + 0.08 + 0.21 * n_how + 0.45 + 0.55
    fig, axs = plt.subplots(len(rows), len(files[0]), figsize=(13, height), squeeze=False)
    for i, row in enumerate(files):
        for j, f in enumerate(row):
            axs[i, j].imshow(Image.open(f)); axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
            for sp in axs[i, j].spines.values(): sp.set_visible(False)
            if i == 0: axs[i, j].set_title(coltitles[j], fontsize=10, color=MUTE)
        axs[i, 0].set_ylabel(rows[i], rotation=0, ha="right", va="center", fontsize=11)
    top, bottom, _ = frame(fig, title, how, look, height)
    fig.subplots_adjust(left=left, right=0.99, top=top, bottom=bottom, wspace=0.03, hspace=0.06)
    for i, j in marks: outline(fig, axs[i, j])
    fig.savefig(OUT / name, dpi=170, facecolor="white"); plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- the problem
def fig_problem():
    seeds = [2, 3, 6, 7]
    joint = [REPO / f"artifacts/results/does-the-hybrid-fit-both-captions/joint-tiles/seed{s:02d}.png" for s in seeds]
    poe = [REPO / f"artifacts/results/does-poe-of-cats-and-dogs-match-the-joint/tiles/cat__x__dog/seed{s:02d}/poe.png" for s in seeds]
    photos("01-the-problem-on-sdxl.png",
           "SDXL: one prompt draws a cat and a dog. PoE draws one fused animal.",
           "Each column starts from the same random noise (seed). Top: one prompt. Bottom: PoE of two prompts.",
           "the bottom row: every picture is a single animal mixing cat and dog parts",
           ['One prompt:\n"a cat and a dog"', 'PoE:\n"a cat" × "a dog"'], [joint, poe],
           [f"seed {s}" for s in seeds], marks=[(1, j) for j in range(4)])


# ---------------------------------------------------------------- model 1
def fig_formula():
    S = np.load(R / "toy-formula-samples.npz"); J = json.load(open(R / "toy-formula-samples.json"))["results"]
    ws = ["0", "0.001", "0.01", "0.1", "0.3", "0.9"]
    lab = {"0": "0%", "0.001": "0.1%", "0.01": "1%", "0.1": "10%", "0.3": "30%", "0.9": "90%"}
    def cell(i, j):
        k = f"{ws[i]}_{['exact', 'sampler'][j]}"
        return S[f"weight_{k}"], f"{J['weight'][k]['two objects, crisp']} of 300 are the both-shapes picture"
    grid("02-formula-experts-what-poe-wants.png",
         "Model 1, experts written as formulas: PoE wants the both-shapes picture once each expert\n"
         "gives it 0.1%, but the step-by-step sampler only finds it near 90%",
         "Rows: the share of each expert's probability on the both-shapes picture. Left: the exact PoE answer, "
         "computed from the formula. Right: a diffusion sampler aiming at the same PoE.",
         "the 0.1%, 1% and 10% rows: the exact answer is the pair every time, the sampler almost never",
         [f"{lab[w]} on the\nboth-shapes picture" for w in ws],
         ["exact PoE answer (formula)", "step-by-step diffusion sampler"], cell,
         marks=[(r, c) for r in (1, 2, 3) for c in (0, 1)], left=0.17)


def fig_leak():
    S = np.load(R / "toy-formula-samples.npz"); J = json.load(open(R / "toy-formula-samples.json"))["results"]
    ls = ["0", "0.003", "0.01", "0.1"]
    lab = {"0": "0%", "0.003": "0.3%", "0.01": "1%", "0.1": "10%"}
    def cell(i, j):
        k = f"{ls[i]}_exact"; c = J["leak"][k]
        txt = (f"{c['two objects, crisp']} of 300 the pair, "
               f"{c['one circle'] + c['one square']} of 300 one shape")
        return S[f"leak_{k}"], txt
    grid("03-formula-experts-that-like-the-other-shape.png",
         "Model 1: when each expert also likes the other shape a little,\nPoE's exact answer becomes one shape",
         "Each expert keeps 10% on the both-shapes picture. Rows: how much it also gives to each of the other "
         "expert's single-shape pictures. Exact PoE answer only.",
         "the pair count falling from 300 to 20 as the other-shape share rises from 0 to 10%",
         [f"{lab[v]} on each of the\nother expert's shapes" for v in ls],
         ["exact PoE answer (formula)"], cell, marks=[(3, 0)], w=10, left=0.27)


def fig_sampler():
    S = np.load(R / "toy-noised-product-samples.npz"); J = json.load(open(R / "toy-noised-product.json"))["results"]
    keys = ["reverse diffusion, summed noised scores", "reverse diffusion, true noised-product score"]
    ws = ["0.01", "0.1", "0.3"]
    def cell(i, j):
        return S[f"{ws[i]}_{j}"], f"{J[ws[i]][keys[j]]['two objects, crisp']} of 300 are the pair"
    grid("04-where-the-sampler-loses-the-pair.png",
         "Model 1: the sampler loses the pair where it adds the two experts' noise estimates",
         "Same experts, same sampler, same starting noise. Left: the usual sum of the two noise estimates. "
         "Right: the correct noise estimate for the product, computed from the formula.",
         "the right column: with the correct estimate the sampler finds the pair 300 of 300 times",
         [f"{float(w):.0%} on the\nboth-shapes picture" for w in ws],
         ["adding the two experts' estimates (usual PoE)", "the correct estimate for the product"], cell,
         marks=[(r, 1) for r in range(3)], left=0.15)


# ---------------------------------------------------------------- model 3
def fig_training():
    import toy_caption_dial_ldm as m
    c, s, p = list(t.CIRCLES.values()), list(t.SQUARES.values()), list(m.PAIRS)
    fig, axs = plt.subplots(3, 2, figsize=(13, 4.8))
    rows = (('"a red circle"', c, c[:2] + [p[0]]), ('"a blue square"', s, s[:2] + [p[1]]),
            ('"a red circle and a blue square"', p, p))
    for i, (cap, left, right) in enumerate(rows):
        strip(axs[i, 0], left[:3], slots=3); strip(axs[i, 1], right[:3], slots=3)
        axs[i, 0].set_ylabel(cap, rotation=0, ha="right", va="center", fontsize=11)
        axs[i, 0].yaxis.set_label_coords(-0.03, 0.5)
    axs[0, 0].set_title("0 relabelled: both-shapes pictures\ncarry only the joint caption", fontsize=11)
    axs[0, 1].set_title("some relabelled: a both-shapes picture also\ncarries each single caption", fontsize=11)
    frame(fig, "Model 3's training pictures: 2,000 per caption, and the one thing we change",
          "Each row is a caption and three of the pictures trained under it. 6,000 pictures in all: 2,000 circles, "
          "2,000 squares, 2,000 both-shapes.",
          "the outlined pictures: a both-shapes picture also filed under each single caption", 4.8)
    fig.subplots_adjust(left=0.26, right=0.99, top=0.72, bottom=0.12, wspace=0.08, hspace=0.35)
    for ax in (axs[0, 1], axs[1, 1]):
        ax.add_patch(Rectangle((2 * (t.W + 8) - 1.5, -1.5), t.W + 4, t.H + 4.5, fill=False, ec=LOOK, lw=3, clip_on=False))
    fig.savefig(OUT / "05-the-training-pictures.png", dpi=200, facecolor="white"); plt.close(fig)
    print("wrote", OUT / "05-the-training-pictures.png")


def _threshold():
    rows, npz = {}, {}
    for f in ("toy-caption-threshold", "toy-caption-threshold-600-1200-2000"):
        rows.update(json.load(open(R / f"{f}.json"))["results"]); npz.update(dict(np.load(R / f"{f}-samples.npz")))
    return dict(sorted(rows.items(), key=lambda kv: int(kv[0]))), npz


def fig_what_poe_draws():
    rows, npz = _threshold()
    ks = list(rows)
    seeds = [20261022, 20261023, 20261024]
    def cell(i, j):
        r = next(r for r in rows[ks[i]] if r["seed"] == seeds[j])
        return npz[f"{ks[i]}_{seeds[j]}_product"], f"{r['apart']} of 300 separate"
    marks = [(i, j) for i, k in enumerate(ks) for j, sd in enumerate(seeds)
             if next(r for r in rows[k] if r["seed"] == sd)["apart"] >= 30]
    grid("06-what-poe-draws-in-the-caption-model.png",
         'Model 3: PoE of "a red circle" × "a blue square" draws a purple blob until enough\n'
         "both-shapes training pictures also carry the single captions",
         "Each row: how many of the 2,000 both-shapes training pictures were also captioned \"a red circle\" and "
         "\"a blue square\". Each column: one training run (same data, different random seed). "
         "Count: of 300 PoE samples, how many show a separate red circle and blue square.",
         "the outlined cells (30 or more of 300 separate): none until 200 relabelled pictures",
         [f"{int(k):,} relabelled" for k in ks], ["training run 1", "training run 2", "training run 3"],
         cell, marks=marks, rowh=0.95, left=0.12)


def fig_all_captions():
    d = json.load(open(R / "toy-caption-dial-ldm.json"))["results"]
    S = np.load(R / "toy-caption-dial-ldm-samples.npz")
    shares = ["0", "0.1", "0.3", "0.6", "1"]
    pics = {"0": "0", "0.1": "about 200", "0.3": "about 600", "0.6": "about 1,200", "1": "2,000"}
    cols = [('"a red circle"', 0, "a red circle alone", "red only", "only red"),
            ('"a blue square"', 1, "a blue square alone", "blue only", "only blue"),
            ('PoE: "a red circle" × "a blue square"', 2, "the two single captions multiplied", "red and blue apart", "separate"),
            ('one prompt: "a red circle and a blue square"', 3, "the joint caption", "red and blue apart", "separate")]
    def cell(i, j):
        _, idx, key, cat, word = cols[j]
        return S[f"{shares[i]}_{idx}"], f"{d[shares[i]][key]['blob scorer'][cat]} of 300 {word}"
    grid("07-every-caption-side-by-side.png",
         "Model 3, one training run per row: the model draws each caption correctly,\nand only PoE depends on the relabelling",
         "Rows: how many of the 2,000 both-shapes pictures were also given the single captions. "
         "Columns: four ways of asking the same trained model.",
         "the PoE column: 0 of 300 separate with no relabelled pictures, while the one-prompt column is near 300 in every row",
         [f"{pics[s]} relabelled" for s in shares], [c[0] for c in cols], cell,
         marks=[(i, 2) for i in range(5)], w=16.5, left=0.125)


def fig_corrector():
    rows, npz = [], {}
    for f in ("toy-caption-corrector", "toy-caption-corrector-no-single-captions"):
        rows += json.load(open(R / f"{f}.json"))["results"]; npz.update(dict(np.load(R / f"{f}-samples.npz")))
    rows.sort(key=lambda r: (r["pictures"], r["seed"]))
    keys = ["plain", "main", "stronger"]
    def cell(i, j):
        r = rows[i]
        return npz[f"{r['pictures']}_{r['seed']}_{keys[j]}"], f"{r[keys[j]]['product']['red and blue apart']} of 300 separate"
    grid("08-the-corrector.png",
         "Model 3 with a Langevin corrector: the pair reaches 300 of 300, but only in models\n"
         "trained with relabelled pictures",
         "Each row: one trained model (how many both-shapes pictures were relabelled, which training run). "
         "Columns: the same model sampled three ways, from the same starting noise.",
         "the outlined cells: 300 of 300 with 20 corrector steps where pictures were relabelled, 0 where they were not",
         [f"{r['pictures']:,} relabelled,\n{RUN[r['seed']]}" for r in rows],
         ["plain sampling", "5 corrector steps per noise level", "20 corrector steps per noise level"], cell,
         marks=[(i, 2) for i in range(len(rows))], rowh=1.0, left=0.13)


# ---------------------------------------------------------------- model 4
def fig_factorized():
    S = np.load(R / "toy-factorized-samples.npz"); f = json.load(open(R / "toy-factorized.json"))["results"]
    bgs = ["grey", "white"]
    spec = [(1, "exact composition, by the pairing drawn"), (2, "reverse diffusion, summed noised scores")]
    def cell(i, j):
        idx, key = spec[j]
        return S[f"{bgs[i]}_{idx}"], f"{f[bgs[i]][key]['unseen layouts']} of 300 in a new layout"
    grid("09-composing-with-no-both-shapes-pictures.png",
         "Model 4, circle × square ÷ background, no both-shapes picture anywhere:\n"
         "it draws new layouts on grey, and one dark blob on white",
         "Rows: background colour. Columns: the exact answer from the formula, and the diffusion sampler. "
         "Count: samples showing a circle and a square in a layout no training picture holds.",
         "the grey row: about 200 of 300 are new layouts; the white row: almost none",
         ["grey background", "white background"], ["exact answer (formula)", "step-by-step diffusion sampler"],
         cell, marks=[(0, 0), (0, 1)], n=5, left=0.12)


# ---------------------------------------------------------------- back to SDXL
def fig_pilot():
    if PILOT is None: return
    seeds = [0, 1, 3, 5, 6, 7]
    dirs = ["plain", "v4-run-A", "v4-run-B"]
    photos("10-sdxl-inclusive-captions-pilot.png",
           'SDXL, PoE "a cat" × "a dog": one animal in every picture, with or without the single captions',
           "Each column starts from the same random noise (seed). Rows: plain SDXL, and SDXL after two LoRA "
           "fine-tunes on 550 photos that differ only in how the 150 cat-and-dog photos are captioned.",
           "the bottom row: clean single cats or dogs, where the middle row has fused cat-dogs",
           ["plain SDXL", 'fine-tuned:\nboth-animal photos only\n"a cat and a dog"',
            'fine-tuned:\nboth-animal photos also\n"cat" or "dog"'],
           [[PILOT / d / f"seed_{s:04d}_product.png" for s in seeds] for d in dirs],
           [f"seed {s}" for s in seeds], marks=[(2, j) for j in range(6)], left=0.2)


def fig_dataset():
    """Real training pictures drawn at random from each caption's pool, at 0, 200 and 2,000
    relabelled, with each pool's true counts. Both-shapes pictures outlined."""
    import toy_caption_dial_ldm as m
    import toy_caption_threshold as th
    ks = [0, 200, 2000]
    caps = [('"a red circle"', m.CIRCLE), ('"a blue square"', m.SQUARE), ('"a red circle and\na blue square"', m.BOTH)]
    per = m.IMAGES // 3
    data = {}
    for k in ks:
        x, c = th.corpus_k(k, np.random.default_rng(20261022))
        x = x.permute(0, 2, 3, 1).reshape(len(x), -1).cpu().numpy()
        both = np.zeros(len(x), bool); both[2 * per:] = True        # pairs block and every relabelled copy
        data[k] = (x, c.cpu().numpy(), both)
    pick = np.random.default_rng(7)
    cells = {}
    for i, (_, cid) in enumerate(caps):
        for j, k in enumerate(ks):
            x, c, both = data[k]
            pool = np.flatnonzero(c == cid)
            nb = int(both[pool].sum())
            n_both = int(round(8 * nb / len(pool)))          # drawn in proportion to the pool
            pb, ps = pool[both[pool]], pool[~both[pool]]
            idx = np.concatenate([pick.choice(pb, n_both, replace=False) if n_both else np.array([], int),
                                  pick.choice(ps, 8 - n_both, replace=False) if 8 - n_both else np.array([], int)])
            idx = pick.permutation(idx)
            hl = both[idx] if cid != m.BOTH else np.zeros(8, bool)
            cells[i, j] = (x[idx], hl, f"{len(pool):,} pictures, {nb:,} with both shapes ({nb / len(pool):.0%})")
    def cell(i, j):
        return cells[i, j][0], cells[i, j][2], cells[i, j][1]
    fig_name = "05b-the-training-set-with-and-without-relabelling.png"
    grid(fig_name,
         "Model 3's training set: relabelling puts both-shapes pictures under the single captions",
         "Rows: the caption a picture is trained under. Columns: how many of the 2,000 both-shapes pictures were "
         "relabelled. Each cell: 8 real training pictures from that caption's pool, drawn so the share holding both "
         "shapes (outlined) matches the pool, and the pool's true counts.",
         'the "a red circle" and "a blue square" rows: 0% both-shapes without relabelling, 9% at 200, 50% at 2,000',
         [cp[0] for cp in caps], [f"{k:,} relabelled" for k in ks], cell, n=8, w=18, rowh=1.1, left=0.1)
    print("tiles with both shapes per cell:", {f"{caps[i][0][:14]}|{ks[j]}": int(cells[i, j][1].sum()) for (i, j) in cells})


if __name__ == "__main__":
    for f in (fig_problem, fig_formula, fig_leak, fig_sampler, fig_training, fig_dataset, fig_what_poe_draws,
              fig_all_captions, fig_corrector, fig_factorized, fig_pilot):
        f()
