"""A scorer that survives a VAE: find the red pixels and the blue pixels, then say where they sit.

The labelling library in toy_sampler_hybrids.py compares a picture to pixel-exact templates, which
works on the exact toy and fails on decoded latents, where edges soften and shapes shift by a pixel.
This scorer reads colour instead. Each pixel goes to the nearest of four colours (white background,
the circle's red, the square's blue, and the purple half-way between them, which is what an overlaid
hybrid looks like). Then:

  red and blue apart   at least MIN_PX red pixels and MIN_PX blue pixels, centroids at least
                       MIN_GAP pixels apart along x, and fewer than MIN_PX purple pixels
  red only             red present, blue and purple absent
  blue only            blue present, red and purple absent
  purple or overlaid   MIN_PX purple pixels, or red and blue present with centroids closer than MIN_GAP
  nothing readable     none of the above

The library's own categories are not changed; this is a second reader used where the first fails.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import toy_sampler_hybrids as t

WHITE = np.ones(3)
PURPLE = (t.RED + t.BLUE) / 2
PALETTE = np.stack([WHITE, t.RED, t.BLUE, PURPLE])
MIN_PX, MIN_GAP, MAX_DIST = 12, 6.0, 0.35
CATS = ["red and blue apart", "red only", "blue only", "purple or overlaid", "nothing readable"]


def _box3(img):
    pad = np.pad(img, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
    return sum(pad[:, i:i + t.H, j:j + t.W] for i in range(3) for j in range(3)) / 9


def read(x, bg=None, smooth=False, purple=True):
    """x is (n, H*W*3) in roughly 0 to 1. Returns one category per picture and the per-picture details.
    bg is the background colour, white unless given. smooth averages each pixel with its 3 x 3
    neighbours first. purple=False drops the purple reference, which a grey background needs: there
    purple sits 0.24 from the background, so noise and soft red edges both get called purple."""
    img = np.asarray(x).reshape(-1, t.H, t.W, 3)
    if smooth:
        img = _box3(img)
    pal = PALETTE if bg is None else np.stack([np.asarray(bg, float), t.RED, t.BLUE, PURPLE])
    if not purple:
        pal = pal[:3]
    d = ((img[..., None, :] - pal) ** 2).sum(-1) ** 0.5               # n, H, W, 4
    near = d.argmin(-1)
    near[d.min(-1) > MAX_DIST] = -1
    xs = np.arange(t.W)[None, None, :] + 0.5
    out, info = [], []
    for k in range(len(img)):
        red, blue, pur = (near[k] == 1), (near[k] == 2), (near[k] == 3)
        nr, nb, npu = int(red.sum()), int(blue.sum()), int(pur.sum())
        cr = float((xs[0] * red).sum() / nr) if nr else None
        cb = float((xs[0] * blue).sum() / nb) if nb else None
        if nr >= MIN_PX and nb >= MIN_PX:
            cat = ("red and blue apart" if abs(cr - cb) >= MIN_GAP and npu < MIN_PX
                   else "purple or overlaid")
        elif npu >= MIN_PX:
            cat = "purple or overlaid"
        elif nr >= MIN_PX:
            cat = "red only"
        elif nb >= MIN_PX:
            cat = "blue only"
        else:
            cat = "nothing readable"
        out.append(cat)
        info.append({"red px": nr, "blue px": nb, "purple px": npu, "red x": cr, "blue x": cb})
    return out, info


def counts(x):
    labs, _ = read(x)
    return {c: labs.count(c) for c in CATS}


KNOWN = {  # every training picture with the category this scorer must give it
    **{f"circle {p}": (v, "red only") for p, v in t.CIRCLES.items()},
    **{f"square {p}": (v, "blue only") for p, v in t.SQUARES.items()},
    "circle left, square right": (t.draw([("circle", t.XPOS["left"]), ("square", t.XPOS["right"])]),
                                  "red and blue apart"),
    "square left, circle right": (t.draw([("square", t.XPOS["left"]), ("circle", t.XPOS["right"])]),
                                  "red and blue apart"),
    **{f"overlaid at {p}": ((t.CIRCLES[p] + t.SQUARES[p]) / 2, "purple or overlaid") for p in t.XPOS},
}


def self_check(noise=0.1, n=200, seed=0):
    """Score n noisy copies of each known picture; return the fraction scored right per picture."""
    rng = np.random.default_rng(seed)
    return {name: sum(c == want for c in read(img + noise * rng.standard_normal((n, img.size)))[0]) / n
            for name, (img, want) in KNOWN.items()}


if __name__ == "__main__":
    for name, frac in self_check().items():
        print(f"{name:28s} {frac:.3f}")
