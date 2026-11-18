"""Step 7b's judge: calls a finished canvas on the ladder, reading each blob as a single digit.

Route step 7b, plan 24 task 1.3. The judge never trains on canvases and never pushes. It learns on
28 x 28 crops only: MNIST's 60,000 training digits, plus an eleventh class, "two digits on one
spot", made by overlaying two different training digits at a random shift of up to 6 pixels. Every
crop, in training and on a canvas, goes through the same preparation (`judge_sheet.crop28`): scaled
so its longer side is 20 pixels and centred by centre of mass, as MNIST digits are.

The fused test, beyond confidence: a blob is fused when the network reads it as the eleventh class,
or reads no digit at confidence 0.9 or more. A blob much wider than one digit (two digits touching)
is first split at the column with the least ink in its middle third.

The ladder: one digit < fused < two digits, wrong identity < two clear, right digits.
- two blobs, each a confident single digit, one of each asked digit: two clear, right digits
- two blobs otherwise, neither fused: two digits, wrong identity
- any fused blob: fused
- one blob, a confident single digit: one digit
- no blob, or more than two: damaged

Usage:
    python judge.py train --mnist <raw dir> --weights <pt>
    python judge.py sheet --mnist <raw dir> --weights <pt> --out <dir>   # the 64-tile gate sheet
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canvas_sheet import H, W, D, Canvases, load_mnist  # noqa: E402
from judge_sheet import blobs, crop28  # noqa: E402

CONFIDENT = 0.9
TWO_ON_ONE_SPOT = 10
SPLIT_WIDTH = 30          # a blob at least this wide is tried as two touching digits
LADDER = ["damaged", "one digit", "fused", "two digits, wrong identity", "two clear, right digits"]


def judge_net():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 7 * 7, 256), nn.ReLU(), nn.Linear(256, 11))


def prepared(img28):
    """A 28 x 28 uint8 picture through the same blob preparation a canvas blob gets."""
    ys, xs = np.nonzero(img28 > 0)
    return crop28(img28, ys, xs)[0]


def overlay(a, b, rng):
    """Two digits on one spot, the second shifted by up to 6 pixels each way."""
    pad = np.zeros((40, 40), np.uint8)
    pad[6:34, 6:34] = a
    dy, dx = rng.integers(-6, 7, 2)
    pad[6 + dy:34 + dy, 6 + dx:34 + dx] = np.maximum(pad[6 + dy:34 + dy, 6 + dx:34 + dx], b)
    ys, xs = np.nonzero(pad > 0)
    return crop28(pad, ys, xs)[0]


def crop_set(images, labels, n_overlay, rng):
    xs = [prepared(im) for im in images]
    ys = list(labels.astype(int))
    by = {d: np.flatnonzero(labels == d) for d in range(10)}
    for _ in range(n_overlay):
        d1, d2 = rng.choice(10, 2, replace=False)
        xs.append(overlay(images[rng.choice(by[d1])], images[rng.choice(by[d2])], rng))
        ys.append(TWO_ON_ONE_SPOT)
    return torch.stack(xs), torch.tensor(ys)


def train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = load_mnist(args.mnist)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    x, y = crop_set(images, labels, 30000, rng)
    net = judge_net().to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for epoch in range(8 if args.augment == "strong" else 6 if args.augment else 4):
        net.train()
        for idx in torch.randperm(len(x)).split(256):
            xb = x[idx].to(dev)
            if args.augment == "strong":
                xb = thicken(shape(xb), strong=True)
            elif args.augment:
                xb = thicken(xb)
            loss = nn.functional.cross_entropy(net(xb), y[idx].to(dev))
            opt.zero_grad(), loss.backward(), opt.step()
    net.eval()
    # held out: MNIST's test digits, and overlays built only from test digits
    timg = np.fromfile(args.mnist / "t10k-images-idx3-ubyte", np.uint8, offset=16).reshape(-1, D, D)
    tlbl = np.fromfile(args.mnist / "t10k-labels-idx1-ubyte", np.uint8, offset=8)
    tx, ty = crop_set(timg, tlbl, 5000, np.random.default_rng(1))
    with torch.no_grad():
        pred = torch.cat([net(c.to(dev)).argmax(1).cpu() for c in tx.split(1000)])
    single = float((pred[:10000] == ty[:10000]).float().mean())
    two = float((pred[10000:] == TWO_ON_ONE_SPOT).float().mean())
    false_two = float((pred[:10000] == TWO_ON_ONE_SPOT).float().mean())
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), args.weights)
    res = {"test_single_digit_accuracy": round(single, 4), "test_two_on_one_spot_caught": round(two, 4),
           "test_single_digits_called_two_on_one_spot": round(false_two, 4)}
    (args.weights.parent / "judge_test.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


def thicken(x, strong=False):
    """Stroke thickening (once or twice) or blur, at random, on single digits and overlays alike, so
    ink thickness is no cue for "two digits on one spot"; the canvas model draws thick strokes.
    strong: also three times thicker, and thinner."""
    r = torch.rand(x.shape[0], device=x.device)[:, None, None, None]
    once = nn.functional.max_pool2d(x, 3, 1, 1)
    twice = nn.functional.max_pool2d(once, 3, 1, 1)
    blur = nn.functional.avg_pool2d(once, 3, 1, 1)
    if strong:
        thrice = nn.functional.max_pool2d(twice, 3, 1, 1)
        thin = -nn.functional.max_pool2d(-x, 3, 1, 1)
        return torch.where(r < 0.25, once, torch.where(r < 0.45, twice, torch.where(r < 0.55, thrice,
               torch.where(r < 0.65, blur, torch.where(r < 0.75, thin, x)))))
    return torch.where(r < 0.3, once, torch.where(r < 0.45, twice, torch.where(r < 0.6, blur, x)))


def shape(x):
    """Random rotation up to 15 degrees, shear up to 0.3 and scale 0.85 to 1.15, per crop."""
    b = x.shape[0]
    a = (torch.rand(b, device=x.device) - 0.5) * (2 * 15 * 3.14159 / 180)
    sh = (torch.rand(b, device=x.device) - 0.5) * 0.6
    sc = 0.85 + torch.rand(b, device=x.device) * 0.3
    th = torch.zeros(b, 2, 3, device=x.device)
    th[:, 0, 0] = torch.cos(a) / sc
    th[:, 0, 1] = (-torch.sin(a) + sh) / sc
    th[:, 1, 0] = torch.sin(a) / sc
    th[:, 1, 1] = torch.cos(a) / sc
    g = nn.functional.affine_grid(th, x.shape, align_corners=False)
    return nn.functional.grid_sample(x, g, align_corners=False)


def load(weights, dev):
    net = judge_net().to(dev)
    net.load_state_dict(torch.load(weights, map_location=dev))
    return net.eval()


def split_wide(canvas, ys, xs):
    """Split a blob at the column with the least ink in its middle third, if it is wide."""
    x0, x1 = xs.min(), xs.max()
    if x1 - x0 + 1 < SPLIT_WIDTH:
        return [(ys, xs)]
    lo, hi = x0 + (x1 - x0) // 3, x0 + 2 * (x1 - x0) // 3
    cols = np.arange(lo, hi + 1)
    ink = np.array([canvas[ys[xs == c], c].astype(float).sum() for c in cols])
    cut = cols[int(ink.argmin())]
    left, right = xs < cut, xs > cut
    halves = [(ys[left], xs[left]), (ys[right], xs[right])]
    return [h for h in halves if len(h[0]) >= 25] or [(ys, xs)]


@torch.no_grad()
def judge(net, canvas, asked, dev):
    """The ladder call for one uint8 canvas, and per blob (ys, xs, digit or 10, confidence)."""
    parts = []
    for ys, xs in blobs(canvas):
        parts += split_wide(canvas, ys, xs)
    reads = []
    for ys, xs in parts:
        p = torch.softmax(net(crop28(canvas, ys, xs).to(dev)), 1)[0].cpu().numpy()
        reads.append((ys, xs, int(p.argmax()), float(p.max())))
    fused = [r[2] == TWO_ON_ONE_SPOT or r[3] < CONFIDENT for r in reads]
    if len(reads) == 0 or len(reads) > 2:
        call = "damaged"
    elif any(fused):
        call = "fused"
    elif len(reads) == 1:
        call = "one digit"
    else:
        ok = sorted(r[2] for r in reads) == sorted(asked)
        call = "two clear, right digits" if ok else "two digits, wrong identity"
    return call, reads


def gate_tiles(cv, rng):
    """64 pasted canvases from MNIST's test digits, asked for 7 and 8, in six kinds."""
    tiles = []
    for _ in range(16):
        tiles.append(("7 beside 8", cv.two(7, 8)))
    for _ in range(16):
        c = np.zeros((H, W), np.uint8)
        x = int(rng.integers(8, W - D - 8))
        y = int(rng.integers(0, H - D + 1))
        dx, dy = int(rng.integers(-5, 6)), int(rng.integers(-2, 3))
        c[y:y + D, x:x + D] = cv.digit(7)
        y2, x2 = min(max(y + dy, 0), H - D), x + dx
        c[y2:y2 + D, x2:x2 + D] = np.maximum(c[y2:y2 + D, x2:x2 + D], cv.digit(8))
        tiles.append(("7 and 8 on one spot", c))
    for _ in range(8):
        tiles.append(("7 and 8 touching", touching(cv, rng)))
    for d in (7, 8) * 6:
        tiles.append((f"a {d} alone", cv.one(d)))
    for a, b in [(7, 1)] * 4 + [(7, 7)] * 4 + [(8, 3)] * 4:
        tiles.append((f"{a} beside {b}", cv.two(a, b)))
    assert len(tiles) == 64, len(tiles)
    order = rng.permutation(len(tiles))
    return [tiles[i] for i in order]


def touching(cv, rng):
    c = np.zeros((H, W), np.uint8)
    y = int(rng.integers(0, H - D + 1))
    c[y:y + D, 4:4 + D] = cv.digit(7)
    gap = int(rng.integers(13, 18))
    c[y:y + D, 4 + gap:4 + gap + D] = np.maximum(c[y:y + D, 4 + gap:4 + gap + D], cv.digit(8))
    return c


def sheet(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = load(args.weights, dev)
    timg = np.fromfile(args.mnist / "t10k-images-idx3-ubyte", np.uint8, offset=16).reshape(-1, D, D)
    tlbl = np.fromfile(args.mnist / "t10k-labels-idx1-ubyte", np.uint8, offset=8)
    rng = np.random.default_rng(64)
    tiles = gate_tiles(Canvases(timg, tlbl, rng), rng)
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "gate_tiles.npy", np.stack([t for _, t in tiles]))

    # blind sheet: numbers only, for the eye calls
    fig, axs = plt.subplots(16, 4, figsize=(16, 32))
    for i, (ax, (_, img)) in enumerate(zip(axs.flat, tiles)):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"tile {i + 1}", fontsize=13)
    fig.suptitle("The judge's gate: 64 pasted canvases from MNIST's test digits, asked for 7 and 8.\n"
                 "Call each tile: one digit, fused, two digits wrong identity, two clear right digits.",
                 fontsize=15, weight="bold")
    fig.savefig(args.out / "gate-tiles-blind.png", dpi=80, bbox_inches="tight")
    plt.close(fig)

    with open(args.out / "judge_calls.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["tile", "built_as", "judge_call", "blob_reads"])
        for i, (built, img) in enumerate(tiles):
            call, reads = judge(net, img, (7, 8), dev)
            wr.writerow([i + 1, built, call, "; ".join(
                f"{'two-on-one-spot' if r[2] == TWO_ON_ONE_SPOT else r[2]} {r[3]:.2f}" for r in reads)])
    print(f"wrote {args.out}/gate-tiles-blind.png, gate_tiles.npy and judge_calls.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "sheet"])
    ap.add_argument("--mnist", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--augment", nargs="?", const="basic", default=None,
                    help="thicken or blur crops at random while training; 'strong' adds shape changes")
    args = ap.parse_args()
    train(args) if args.cmd == "train" else sheet(args)


if __name__ == "__main__":
    main()
