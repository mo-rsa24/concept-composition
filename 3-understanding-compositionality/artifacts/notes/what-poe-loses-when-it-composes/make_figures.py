"""Draw the figures for 3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/what-poe-loses-when-it-composes.md.

Two kinds of figure, kept apart:

  from renders   fig-causal-renders.png crops real SDXL tiles and prints the reader's calls
  computed       every other figure is exact arithmetic on small Gaussians, or counts read
                 from the Re-LAION-2B-en search; where a number is a chosen illustration
                 rather than a measurement, the figure says so in its own title

    /home/molef/miniconda3/bin/python3 -I 3-understanding-compositionality/artifacts/notes/what-poe-loses-when-it-composes/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent / "figures"
CAMEL = REPO / "3-understanding-compositionality/artifacts/results/does-a-camel-in-the-rain-come-out-wet"
DOG_SHEET = REPO / "3-understanding-compositionality/artifacts/results/does-a-dog-in-the-rain-come-out-wet/a-dog-and-rain.png"
# tile boxes found by scanning the dog sheet for non-white runs: columns a, b, joint, PoE, SuperDiff
DOG_COLS = [(250, 550), (558, 858), (890, 1190), (1198, 1498), (1506, 1806)]
DOG_ROWS = {1: (132, 432), 2: (440, 740), 4: (748, 1048), 9: (1056, 1356), 10: (1364, 1664)}

WET, DRY, INK, MUTED = "#2a72b8", "#b8862b", "#1f2328", "#5d6470"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False})


def calls() -> dict:
    out = {}
    for f in ("asked-directly-calls.json", "a-camel-and-rain-calls.json"):
        for c in json.loads((CAMEL / f).read_text())["calls"]:
            out[(c["seed"], c["column"])] = c["call"]
    return out


def dog_tile(seed: int, col: int) -> Image.Image:
    (x0, x1), (y0, y1) = DOG_COLS[col], DOG_ROWS[seed]
    return Image.open(DOG_SHEET).convert("RGB").crop((x0, y0, x1, y1))


def fig_causal_renders(seeds=(1, 4)) -> None:
    """Rows: camel and dog at each seed. Columns: the six prompts the causal test reads."""
    C = calls()
    heads = ['"a ___" alone', '"rain" alone', '"a ___ in the rain"\nasked directly',
             '"a ___ and rain"\njoint prompt', 'PoE of "a ___"\nand "rain"', '"a wet ___"\nasked directly']
    fig, axes = plt.subplots(2 * len(seeds), 6, figsize=(13.2, 4.6 * len(seeds)))
    for r, (seed, animal) in enumerate([(s, a) for s in seeds for a in ("camel", "dog")]):
        t = CAMEL / "tiles"
        if animal == "camel":
            tiles = [t / f"single/a_camel/seed{seed:02d}.png", t / f"single/rain/seed{seed:02d}.png",
                     t / f"single/a_camel_in_the_rain/seed{seed:02d}.png", t / f"a_camel__x__rain/seed{seed:02d}/mono.png",
                     t / f"a_camel__x__rain/seed{seed:02d}/poe.png", t / f"single/a_wet_camel/seed{seed:02d}.png"]
            imgs = [Image.open(p).convert("RGB") for p in tiles]
            lab = [None, None, C.get((seed, "a camel in the rain")), C.get((seed, "joint")),
                   C.get((seed, "poe")), C.get((seed, "a wet camel"))]
        else:
            imgs = [dog_tile(seed, 0), dog_tile(seed, 1),
                    Image.open(t / f"single/a_dog_in_the_rain/seed{seed:02d}.png").convert("RGB"),
                    dog_tile(seed, 2), dog_tile(seed, 3),
                    Image.open(t / f"single/a_wet_dog/seed{seed:02d}.png").convert("RGB")]
            lab = [None, None, C.get((seed, "a dog in the rain")), "no per-seed call yet",
                   "no per-seed call yet", C.get((seed, "a wet dog"))]
        for c, (im, l) in enumerate(zip(imgs, lab)):
            ax = axes[r, c]
            ax.imshow(im.resize((300, 300)))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(heads[c], fontsize=10.5)
            if c == 0:
                ax.set_ylabel(f"{animal}, seed {seed}", fontsize=11)
            if l:
                col = WET if l == "wet" else (DRY if l == "dry" else MUTED)
                ax.set_xlabel(f"reader's call: {l}", color=col, fontsize=10)
    fig.suptitle("The same rain, two animals: SDXL, 50 DDIM steps, guidance 7.5, shared starting noise per seed",
                 fontsize=12, y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "fig-causal-renders.png", dpi=110)
    plt.close(fig)


def shapes(kind: str, n: int = 16) -> np.ndarray:
    """A 16 x 16 RGB image on white: a red circle, a blue square, or both side by side."""
    y, x = np.mgrid[0:n, 0:n] + 0.5
    img = np.ones((n, n, 3))
    def circle(cx, cy, r=3.6):
        img[(x - cx) ** 2 + (y - cy) ** 2 <= r * r] = (0.85, 0.18, 0.15)
    def square(cx, cy, h=3.2):
        img[(abs(x - cx) <= h) & (abs(y - cy) <= h)] = (0.16, 0.35, 0.80)
    if kind == "circle":
        circle(8, 8)
    elif kind == "square":
        square(8, 8)
    elif kind == "pair":
        circle(4.2, 8); square(11.8, 8)
    return img


def fig_exact_product_is_a_blend() -> None:
    """Each condition a Gaussian of spread sigma around its one training picture.
    The product of two such Gaussians is a Gaussian at their average, which is the overlay."""
    c, s, pair = shapes("circle"), shapes("square"), shapes("pair")
    product_mean = (c + s) / 2                                    # exact: N(c,s2) N(s,s2) ~ N((c+s)/2, s2/2)
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.4))
    for ax, im, t in zip(axes, [c, s, product_mean, pair],
                         ['"a red circle"\nits one training picture', '"a blue square"\nits one training picture',
                          'exact product of the two\n(mean of the product Gaussian)', '"a circle and a square"\nwhat the joint caption saw']):
        ax.imshow(im, interpolation="nearest"); ax.set_title(t, fontsize=10.5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("#d9d5cb")
    axes[2].set_xlabel("every pixel is the average of the two", fontsize=10, color=MUTED)
    fig.suptitle("The exact product of two centred single-object conditions is an overlay, not two objects",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    fig.savefig(OUT / "fig-exact-product-is-a-blend.png", dpi=120)
    plt.close(fig)


def gauss2(X, Y, mu, sd):
    return np.exp(-0.5 * (((X - mu[0]) / sd[0]) ** 2 + ((Y - mu[1]) / sd[1]) ** 2))


def product_mean(mu_a, sd_a, mu_b, sd_b):
    """Precision-weighted mean, one coordinate at a time (diagonal Gaussians)."""
    ta, tb = 1 / np.square(sd_a), 1 / np.square(sd_b)
    return (ta * np.asarray(mu_a) + tb * np.asarray(mu_b)) / (ta + tb)


def fig_precision_picture() -> None:
    """Two coordinates stand for two parts of the image. Illustrative values, chosen by hand."""
    X, Y = np.mgrid[-2.2:2.2:400j, -2.2:2.2:400j]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))
    cases = [("cat and dog: both claim the subject pixels", [(-1, 0), (0.22, 1.6), "a cat", "#9a5bd6"],
              [(1, 0), (0.22, 1.6), "a dog", "#c46b1a"]),
             ("dog and rain: each claims its own pixels", [(1, 0), (0.22, 1.6), "a dog", "#c46b1a"],
              [(0, 1), (1.6, 0.22), "rain", WET])]
    for ax, (title, A, B) in zip(axes, cases):
        for mu, sd, name, col in (A, B):
            ax.contour(X, Y, gauss2(X, Y, mu, sd), levels=[0.135, 0.6], colors=[col], linewidths=[1.4, 2.2])
            tx, ty = (mu[0] + 0.12, mu[1] + 1.25) if sd[0] < 1 else (-2.05, mu[1] + 0.3)
            ax.text(tx, ty, name, color=col, fontsize=12)
        m = product_mean(A[0], A[1], B[0], B[1])
        P = gauss2(X, Y, A[0], A[1]) * gauss2(X, Y, B[0], B[1])
        ax.contourf(X, Y, P / P.max(), levels=[0.6, 1.01], colors=[INK], alpha=0.25)
        ax.plot(*m, "o", color=INK, ms=8)
        ax.annotate(f"product\n({m[0]:+.2f}, {m[1]:+.2f})", m, xytext=(m[0] - 1.6, m[1] - 1.5),
                    arrowprops=dict(arrowstyle="->", color=INK), fontsize=11)
        ax.set_xlabel("subject pixels: cat  ←  →  dog")
        ax.set_ylabel("background pixels: dry street  ←  →  rain")
        ax.set_title(title, fontsize=11.5)
        ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2); ax.set_aspect("equal")
    fig.suptitle("Where each condition is sure (narrow) and where it does not care (wide). Illustrative spreads.",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig-why-rain-and-dog-do-not-fuse.png", dpi=120)
    plt.close(fig)


def fig_mixture_reweighting() -> None:
    """A subject's pictures are a mixture of dry and wet states; the effect condition reweights them."""
    pi = np.logspace(-5, 0, 400)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    for rho, ls in ((1, ":"), (10, "--"), (100, "-"), (1000, "-.")):
        ax.plot(pi, pi * rho / (pi * rho + 1 - pi), ls, color=WET, lw=2, label=f"ρ = {rho}")
    ax.plot(pi, pi, color=MUTED, lw=0.8)
    ax.set_xscale("log"); ax.set_ylim(0, 1.02)
    ax.set_xlabel("π, share of the subject's own pictures that are wet (log scale)")
    ax.set_ylabel("share of the product that is wet")
    ax.set_title("The product amplifies wet pictures that exist.\nIt cannot make them from π = 0.", fontsize=11.5)
    ax.legend(title="how much more the effect\ncondition likes wet than dry", fontsize=9, title_fontsize=9)
    ax = axes[1]
    labels = ["dog\nπ = 0.05", "camel\nπ = 0.0005", "camel\nπ = 0"]
    pis = [0.05, 0.0005, 0.0]
    rho = 100
    after = [p * rho / (p * rho + 1 - p) for p in pis]
    x = np.arange(3)
    ax.bar(x - 0.18, pis, 0.36, color=MUTED, label="before: the subject alone")
    ax.bar(x + 0.18, after, 0.36, color=WET, label=f"after: times the effect (ρ = {rho})")
    for i, (b, a) in enumerate(zip(pis, after)):
        ax.text(i + 0.18, a + 0.02, f"{a:.2f}", ha="center", fontsize=10)
    ax.set_xticks(x, labels); ax.set_ylim(0, 1.05); ax.set_ylabel("wet share")
    ax.set_title("Same effect, three subjects. π values are illustrative,\nnot measured: the census measures them.", fontsize=11.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig-mixture-reweighting.png", dpi=120)
    plt.close(fig)


def fig_caption_dial() -> None:
    """Each condition is a mixture over its training pictures (Gaussian, spread sigma, in pixel space).
    A share q of each condition's pictures are the pair image, captioned with that single name.
    The product of two mixtures is a mixture over pairs of components; we report its mass on the
    pair-with-pair component, i.e. two separate objects. Exact, no sampling."""
    c, s, pair = shapes("circle").ravel(), shapes("square").ravel(), shapes("pair").ravel()
    # singles jitter one pixel left or right so the overlay is not the only option
    def shifted(img, dx):
        return np.roll(img.reshape(16, 16, 3), dx, axis=1).ravel()
    A_single = [shifted(c.copy(), d) for d in (-1, 0, 1)]
    B_single = [shifted(s.copy(), d) for d in (-1, 0, 1)]
    qs = np.linspace(0, 0.5, 51)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for sigma, ls in ((0.6, "-"), (1.2, "--"), (2.4, ":")):
        mass = []
        for q in qs:
            A = A_single + [pair]; wA = np.r_[np.full(3, (1 - q) / 3), q]
            B = B_single + [pair]; wB = np.r_[np.full(3, (1 - q) / 3), q]
            logw = np.array([[np.log(wa + 1e-300) + np.log(wb + 1e-300) - np.sum((a - b) ** 2) / (4 * sigma ** 2)
                              for b, wb in zip(B, wB)] for a, wa in zip(A, wA)])
            w = np.exp(logw - logw.max()); w /= w.sum()
            mass.append(w[-1, -1])
        ax.plot(qs * 100, mass, ls, color=INK, lw=2, label=f"σ = {sigma}")
    ax.set_xlabel("q: % of each condition's pictures that are the pair, captioned with that one name")
    ax.set_ylabel("exact product mass on two separate objects")
    ax.set_title("The caption dial, computed exactly on 16 x 16 images", fontsize=11.5)
    ax.legend(title="spread of each condition", fontsize=9, title_fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig-caption-dial.png", dpi=120)
    plt.close(fig)


def fig_caption_evidence() -> None:
    """Counts measured on Re-LAION-2B-en research-safe, 2026-10-21 (caption-counts.md and the pilot)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1, 1.25, 1.35]})
    ax = axes[0]
    vals = [2_097_693_557, 14_396_194, 784_797]
    ax.bar(["all\ncaptions", "dog, dogs,\npuppy", "camel, camels,\ndromedary"], vals, color=[MUTED, "#c46b1a", DRY])
    ax.set_yscale("log"); ax.set_ylabel("captions (log scale)")
    for i, v in enumerate(vals):
        ax.text(i, v * 1.3, f"{v:,}", ha="center", fontsize=9.5)
    ax.set_title("Whole-word caption hits, all 128 files", fontsize=11)
    ax = axes[1]
    cats = ["wet dog food", '"raining cats\nand dogs"', "rain gear for dogs", "other mentions", "a drawing", "an actual\nwet dog"]
    n = [16, 3, 2, 2, 1, 1]
    ax.barh(cats[::-1], n[::-1], color=[WET] + [MUTED] * 5)
    for i, v in enumerate(n[::-1]):
        ax.text(v + 0.2, i, str(v), va="center")
    ax.set_xlabel("captions, of a random 25")
    ax.set_title("A wet word plus camel or dog: 2,574 hits in 4 files.\nWhat a random 25 of them are", fontsize=11)
    ax = axes[2]
    steps = ["sampled", "image still loads", "CLIP: the animal\nis there", "CLIP: wet", "of those, wet\nby eye"]
    camel = [100, 63, 43, 4, 0]; dog = [100, 59, 47, 3, 0]
    y = np.arange(len(steps))[::-1]
    ax.barh(y + 0.2, camel, 0.4, color=DRY, label="camel captions")
    ax.barh(y - 0.2, dog, 0.4, color="#c46b1a", label="dog captions")
    for yy, a, b in zip(y, camel, dog):
        ax.text(a + 1, yy + 0.2, str(a), va="center", fontsize=9); ax.text(b + 1, yy - 0.2, str(b), va="center", fontsize=9)
    ax.set_yticks(y, steps); ax.set_xlim(0, 112)
    ax.set_title("The pilot: 100 random captions per animal", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    fig.suptitle("What the public stand-in for SDXL's captions shows (Re-LAION-2B-en research-safe, searched 2026-10-21)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig-caption-evidence.png", dpi=110)
    plt.close(fig)


# Every image the write-up and the two routes show, copied as a thumbnail beside them, so the VS Code
# preview loads it even when the workspace root is elsewhere. Each <img> still links to its original.
THUMBS = REPO / "3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/figures"
SHOWN = [OUT / n for n in ("du-2023-product-sampling.png", "bradley-2025-projective-composition.png",
                           "fig-causal-renders.png", "fig-why-rain-and-dog-do-not-fuse.png",
                           "fig-exact-product-is-a-blend.png", "fig-mixture-reweighting.png",
                           "fig-caption-evidence.png", "fig-caption-dial.png",
                           "fig-toy-du-layout.png", "fig-toy-sampler-hybrids.png", "fig-one-pixel-product.png")] + [
    REPO / "context/images/world/01-poe-chimera-example.png",
    REPO / "context/images/world/02-mono-ceiling-example.png",
    CAMEL / "pilot-ten-walk-pictures.jpg", CAMEL / "pilot-clip-called-wet.jpg",
    DOG_SHEET]


def thumbnails(width: int = 1100) -> None:
    THUMBS.mkdir(exist_ok=True)
    for src in SHOWN:
        if not src.exists():
            print("missing, skipped:", src); continue
        im = Image.open(src).convert("RGB")
        if im.width > width:
            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        dst = THUMBS / (src.stem + ".jpg")
        im.save(dst, quality=86)
        print(dst.relative_to(REPO), f"{dst.stat().st_size // 1024} KB")


if __name__ == "__main__":
    import sys
    if "--thumbnails" in sys.argv:
        thumbnails(); raise SystemExit
    OUT.mkdir(exist_ok=True)
    fig_causal_renders()
    fig_exact_product_is_a_blend()
    fig_precision_picture()
    fig_mixture_reweighting()
    fig_caption_dial()
    fig_caption_evidence()
    print("\n".join(sorted(p.name for p in OUT.glob("fig-*.png"))))
    thumbnails()
