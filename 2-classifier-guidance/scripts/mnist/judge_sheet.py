"""Prototype of step 7b's judge, and the sheet that shows how it calls a canvas.

The judge never sees a canvas while it learns. It is a small CNN trained on single 28 x 28 MNIST
training digits. On a canvas it finds each separate patch of ink (a blob, by flood fill), crops
the blob into a 28 x 28 square, and asks the CNN which digit it is. The call on the ladder:

    one digit < fused < two digits, wrong identity < two clear, right digits

Rules (prototype, to be checked against eye calls before any count is used):
- two blobs, each read confidently (top probability >= CONFIDENT) as the two asked digits,
  one each: two clear, right digits
- two blobs otherwise: two digits, wrong identity
- one blob read confidently as one digit: one digit
- one blob not read confidently: fused
- no blob, or more than two: damaged

The ladder pictures on the sheet are built by pasting real digits. They are illustrations of
each call, not model outputs: no diffusion model exists yet.

Usage:
    python scripts/mnist/judge_sheet.py --mnist <raw dir> --out <png> --weights <pt>
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from canvas_sheet import Canvases, load_mnist  # noqa: E402

CONFIDENT = 0.9
INK = 40          # a pixel above this (0 to 255) counts as ink
MIN_BLOB = 25     # blobs smaller than this many pixels are specks, ignored


def judge_net():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10))


def train_judge(images, labels, weights: Path, device):
    net = judge_net().to(device)
    if weights.exists():
        net.load_state_dict(torch.load(weights, map_location=device))
        return net.eval()
    x = torch.tensor(images, dtype=torch.float32).unsqueeze(1) / 255.0
    y = torch.tensor(labels, dtype=torch.long)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    g = torch.Generator().manual_seed(0)
    for epoch in range(2):
        for idx in torch.randperm(len(x), generator=g).split(256):
            loss = nn.functional.cross_entropy(net(x[idx].to(device)), y[idx].to(device))
            opt.zero_grad(), loss.backward(), opt.step()
    weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), weights)
    return net.eval()


def blobs(canvas):
    """Connected patches of ink, 8-neighbour flood fill. Returns a list of (ys, xs) arrays."""
    ink = canvas > INK
    seen = np.zeros_like(ink)
    found = []
    for y0, x0 in zip(*np.nonzero(ink)):
        if seen[y0, x0]:
            continue
        stack, ys, xs = [(y0, x0)], [], []
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            ys.append(y), xs.append(x)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    v, u = y + dy, x + dx
                    if 0 <= v < ink.shape[0] and 0 <= u < ink.shape[1] and ink[v, u] and not seen[v, u]:
                        seen[v, u] = True
                        stack.append((v, u))
        if len(ys) >= MIN_BLOB:
            found.append((np.array(ys), np.array(xs)))
    return found


def crop28(canvas, ys, xs):
    """The blob alone, prepared the way MNIST digits are: scaled so its longer side is 20 pixels,
    then placed in a 28 x 28 square with its centre of mass at the centre."""
    mask = np.zeros_like(canvas)
    mask[ys, xs] = canvas[ys, xs]
    patch = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    h, w = patch.shape
    scale = 20.0 / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    t = torch.tensor(patch)[None, None]
    small = nn.functional.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False)[0, 0]
    out = torch.zeros(28, 28)
    total = small.sum().clamp(min=1e-6)
    cy = float((small.sum(1) * torch.arange(nh)).sum() / total)
    cx = float((small.sum(0) * torch.arange(nw)).sum() / total)
    oy = int(round(14 - cy)), int(round(14 - cx))
    oy = (min(max(oy[0], 0), 28 - nh), min(max(oy[1], 0), 28 - nw))
    out[oy[0]:oy[0] + nh, oy[1]:oy[1] + nw] = small
    return (out / 255.0)[None, None]


@torch.no_grad()
def judge(net, canvas, asked, device):
    found = blobs(canvas)
    reads = []
    for ys, xs in found:
        p = torch.softmax(net(crop28(canvas, ys, xs).to(device)), 1)[0].cpu().numpy()
        reads.append((ys, xs, int(p.argmax()), float(p.max())))
    if len(reads) == 0 or len(reads) > 2:
        call = "damaged"
    elif len(reads) == 1:
        call = "one digit" if reads[0][3] >= CONFIDENT else "fused"
    else:
        digits = sorted(r[2] for r in reads)
        ok = all(r[3] >= CONFIDENT for r in reads) and digits == sorted(asked)
        call = "two clear, right digits" if ok else "two digits, wrong identity"
    return call, reads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnist", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    images, labels = load_mnist(args.mnist)
    net = train_judge(images, labels, args.weights, device)
    test_imgs = np.fromfile(args.mnist / "t10k-images-idx3-ubyte", np.uint8, offset=16).reshape(-1, 28, 28)
    test_lbls = np.fromfile(args.mnist / "t10k-labels-idx1-ubyte", np.uint8, offset=8)
    with torch.no_grad():
        pred = net(torch.tensor(test_imgs[:, None], dtype=torch.float32, device=device) / 255.0).argmax(1)
    test_acc = float((pred.cpu().numpy() == test_lbls).mean())

    cv = Canvases(images, labels, np.random.default_rng(1))
    asked = (7, 8)
    ladder = [
        ("one digit", "a 7 alone", cv.one(7)),
        ("fused", "7 and 8 pasted on one spot", cv.overlaid(7, 8)),
        ("two digits, wrong identity", "7 beside 1", cv.two(7, 1)),
        ("two digits, wrong identity", "7 beside 7", cv.two(7, 7)),
        ("two clear, right digits", "7 beside 8", cv.two(7, 8)),
    ]
    extra = [("7 beside 8", "two clear, right digits", cv.two(7, 8)) for _ in range(4)] + \
            [("7 and 8 on one spot", "fused", cv.overlaid(7, 8)) for _ in range(4)] + \
            [("an 8 alone", "one digit", cv.one(8)) for _ in range(2)]

    colours = {"one digit": "#7f7f7f", "fused": "#d62728", "two digits, wrong identity": "#ff7f0e",
               "two clear, right digits": "#2ca02c", "damaged": "#000000"}
    fig = plt.figure(figsize=(15, 8.5))
    sf_a, sf_b = fig.subfigures(2, 1, height_ratios=[1.0, 1.7], hspace=0.02)

    sf_a.suptitle("1. The four calls on the ladder, asked for 7 and 8 (canvases built by pasting real "
                  "digits; illustrations of each call, not model outputs)", fontsize=12.5, weight="bold")
    axs = sf_a.subplots(1, 5, gridspec_kw={'wspace': 0.08})
    for ax, (meant, how, img) in zip(axs, ladder):
        call, reads = judge(net, img, asked, device)
        draw(ax, img, reads, f"built as: {how}\nmeant to be: {meant}\njudge says: {call}",
             colours[call])

    sf_b.suptitle(f"2. The prototype judge on more canvases asked for 7 and 8. It is a small CNN "
                  f"trained only on single digits\n(2 passes over MNIST's 60,000 training digits; "
                  f"{test_acc:.1%} right on the 10,000 test digits). Each box is one blob, with the "
                  f"digit it read and how sure it was.", fontsize=12, weight="bold")
    axs = sf_b.subplots(2, 5, gridspec_kw={'wspace': 0.08, 'hspace': 0.25})
    tally = {}
    agree = 0
    agree = 0
    for ax, (how, meant, img) in zip(axs.flat, extra):
        call, reads = judge(net, img, asked, device)
        tally[call] = tally.get(call, 0) + 1
        agree += call == meant
        draw(ax, img, reads, f"built as: {how}\njudge says: {call}" + ("" if call == meant else "  ✗"),
             colours[call])
    sf_b.text(0.5, -0.02, f"The judge agrees with how the canvas was built on {agree} of 10. Step 7b's "
              f"gate needs at least 58 of 64 against your eye calls before any count is used.",
              ha="center", fontsize=11.5)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}; judge test accuracy {test_acc:.4f}; panel 2 calls {tally}; agree {agree} of 10")


def draw(ax, img, reads, title, colour):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(title, fontsize=8.5, color=colour)
    for s in ax.spines.values():
        s.set_edgecolor(colour), s.set_linewidth(2.5)
    for ys, xs, d, conf in reads:
        ax.add_patch(patches.Rectangle((xs.min() - 0.5, ys.min() - 0.5), xs.max() - xs.min() + 1,
                                       ys.max() - ys.min() + 1, fill=False, ec="#1f77b4", lw=1.2))
        ax.text(xs.min(), ys.min() - 1, f"{d} ({conf:.2f})", color="#1f77b4", fontsize=8,
                va="bottom", weight="bold")


if __name__ == "__main__":
    main()
