"""Zoom ladder level 0.5: the three networks of rung 2 on SDXL's cat x dog product, one row each.

The SDXL twin of make_three_networks_in_step_7b.py (the route walk's figures folder), drawn in the same
register and with the same colour key: row 1 is the generator being fixed (one sampler step of the
product of experts), row 2 the pair-aware checker that pushes it, row 3 the judge that only scores
(the Claude API, then the learner's own calls). Each row reads left to right: what goes in, the network
as one box, what comes out, what it is used for.

What is real and what is not:
  - every picture is a real file copied into inputs/ by the session that drew this figure: SDXL joint
    drawings and their API labels from api-labels/more_catdog2.jsonl (picks.json), a plan 23 product
    drawing, a COCO photo and a single-cat photo from the inclusive-captions pool, and two tiles of
    result-sheet-catdog (held-out seed 9 plain; seed 10 pushed at strength 10 by iteration 0's checker);
  - the noisy picture in rows 1 and 2 is illustrative: noise is added to the plain tile in pixels at
    alpha_bar = 0.5 so the animal still shows, while the sampler's x_t is a 4 x 128 x 128 latent;
  - sizes are read from the code: 50 DDIM steps and guidance 7.5 (poe_repair/config.py), PoE as
    eps_a + eps_b - eps_uncond (poe_repair/_sdxl/metrics.py), the push cap 0.1
    (poe_repair/methods/_classifier_guided.py), the tapped blocks and the head
    (scripts/rung2/train_count_head.py Readout, train_readout_pair.py PairHead; 780,417 weights summed
    from its layer shapes);
  - numbers: held-out balanced accuracy 0.737 (readout-catdog-more2, 220 pictures of
    eval_set_catdog_v2.json), training list size from training_set_catdog_more2.json, the API's
    agreement with hand calls 0.789 (api-labels/validate.jsonl).

The figure is refused (nothing written) by the base module's check if any line crosses a label, an
image or a node it is not joined to, or if two labels overlap.

Usage (torch is needed only because the base module imports it):
    CUDA_VISIBLE_DEVICES= uv run --no-project --with torch --with matplotlib --with numpy --with pillow \
        python make_three_networks_in_rung_2.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mp  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
REF = HERE.parents[1] / "route-classifier-guided-poe-from-toy-to-animals" / "figures"
sys.path.insert(0, str(REF))
import make_judge_network_view as base  # noqa: E402

INPUT, HIDDEN, OUTPUT, OFF = base.INPUT, base.HIDDEN, base.OUTPUT, base.OFF
TEXT, MUTED, EDGE = base.TEXT, base.MUTED, base.EDGE
PUSH = "#c86a0a"
W, H = 2400, 1560
base.W, base.H = W, H
tint = base._tint
IN = HERE / "inputs"
ALPHA_BAR = 0.5


def load(name, side):
    im = Image.open(IN / name).convert("RGB").resize((side, side), Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


def noisy(img, seed=0):
    x0 = img * 2 - 1
    eps = np.random.default_rng(seed).standard_normal(img.shape).astype(np.float32)
    return np.clip((ALPHA_BAR ** 0.5 * x0 + (1 - ALPHA_BAR) ** 0.5 * eps + 1) / 2, 0, 1)


def picture(F, name, img, x, y, side, edge):
    F.ax.imshow(img, extent=(x, x + side, y + side, y), interpolation="bilinear", zorder=4)
    F.ax.add_patch(mp.Rectangle((x, y), side, side, fill=False, ec=edge, lw=1.6, zorder=5))
    F.rect(name, x, y, x + side, y + side)
    return x + side, y + side


def chip(F, name, x0, y0, x1, y1, s, color=INPUT, size=13):
    F.ax.add_patch(mp.FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=8",
                                     fc=tint(color, 0.9), ec=color, lw=1.2, zorder=3))
    F.rect("box-" + name, x0, y0, x1, y1)
    F.text((x0 + x1) / 2, (y0 + y1) / 2, s, size, color, bold=True, inside=True)


def panel(F, name, x0, y0, x1, y1, color, dashed=False, glow=False, fill="white"):
    if glow:
        for k, a in ((10, 0.05), (6, 0.08), (3, 0.12)):
            F.ax.add_patch(mp.Rectangle((x0 - k, y0 - k), x1 - x0 + 2 * k, y1 - y0 + 2 * k, fc=color,
                                        alpha=a, lw=0, zorder=1))
    F.ax.add_patch(mp.FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=10",
                                     fc=fill, ec=color, lw=1.4, ls=(0, (6, 4)) if dashed else "-", zorder=2))
    F.rect("box-" + name, x0, y0, x1, y1)


def arrow(F, x0, x1, y, color=OFF, lw=2.6, allow=()):
    F.line([(x0, y), (x1, y)], color, lw, allow=set(allow), arrow=True)


def network_box(F, x0, y0, x1, y1):
    F.ax.add_patch(mp.FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=12",
                                     fill=False, ec=tint(TEXT, 0.5), lw=1.0, zorder=1))


def unet_bars(F, tag, x, yc, scale=1.0):
    """The frozen SDXL U-Net as five grey bars (down, bottom, up), faint skip links. Returns the right edge."""
    hs = [h * scale for h in (110, 76, 46, 76, 110)]
    step, bw = 32 * scale, 17 * scale
    for i, h in enumerate(hs):
        F.ax.add_patch(mp.Rectangle((x + i * step, yc - h / 2), bw, h, fc=tint(OFF, 0.8), ec=OFF, lw=0.9,
                                    zorder=3))
    for i in range(2):
        a, b = x + bw / 2 + i * step, x + bw / 2 + (4 - i) * step
        top = yc - hs[i] / 2 - 8 * scale - i * 6 * scale
        F.ax.plot([a, a, b, b], [yc - hs[i] / 2, top, top, yc - hs[4 - i] / 2], color=tint(OFF, 0.4),
                  lw=0.8, zorder=2)
    right = x + 4 * step + bw
    F.rect(f"{tag}-unet", x, yc - hs[0] / 2 - 22 * scale, right, yc + hs[0] / 2)
    return right


def row_title(F, y, num, title, note):
    F.ax.plot([40, W - 40], [y - 28, y - 28], color=tint(OFF, 0.6), lw=1.0, zorder=1)
    t = F.text(40, y, f"{num}   {title}", 21, ha="left", bold=True)
    F.fig.canvas.draw()
    bb = t.get_window_extent(F.fig.canvas.get_renderer())
    x_end = F.ax.transData.inverted().transform((bb.x1, bb.y0))[0]
    F.text(x_end + 24, y + 1, note, 14, MUTED, ha="left")


def column_heads(F, y):
    for x, s in ((60, "what goes in"), (650, "the network, as one box"), (1270, "what comes out"),
                 (1730, "what it is used for")):
        F.text(x, y, s.upper(), 12, MUTED, ha="left", bold=True)


def draw(N):
    F = base.Fig()
    ax = F.ax
    F.text(40, 42, "Rung 2 has three networks with three different jobs", 30, ha="left", bold=True)
    F.text(40, 86, "Row 1 is the drawing we are fixing; row 2 pushes it while it draws; row 3 only scores "
           "finished pictures, to label the data and count the results.", 16, MUTED, ha="left")
    column_heads(F, 130)

    # ================= Row 1: the generator being fixed =================
    row_title(F, 190, "1", "The generator: one step of the cat × dog product",
              "the same frozen SDXL run three times, its predictions added; this is what draws fused cat-dogs")
    yc = 345
    rx, ry = picture(F, "r1-in", N["noisy"], 60, yc - 95, 190, INPUT)
    F.text(60, ry + 20, "the noisy picture x_t", 13, INPUT, ha="left", bold=True)
    F.text(60, ry + 42, "illustrative: noise added in pixels", 11, MUTED, ha="left")
    F.text(60, ry + 60, "(the sampler's x_t is a 4 × 128 × 128 latent)", 11, MUTED, ha="left")
    chip(F, "r1-step", 400, yc - 120, 585, yc - 84, "step t of 50", size=12)
    chip(F, "r1-cat", 400, yc - 72, 585, yc - 36, "\"a cat\"", size=12)
    chip(F, "r1-dog", 400, yc - 24, 585, yc + 12, "\"a dog\"", size=12)
    chip(F, "r1-null", 400, yc + 24, 585, yc + 60, "empty prompt", size=12)
    arrow(F, 593, 643, yc - 30)
    network_box(F, 650, yc - 135, 1170, yc + 138)
    ys = [yc - 85, yc - 15, yc + 55]
    for k, (y, lab) in enumerate(zip(ys, ("ε_cat", "ε_dog", "ε_null"))):
        r = unet_bars(F, f"r1-u{k}", 680, y, scale=0.48)
        F.circle(f"r1-e{k}", 900, y, 12, HIDDEN)
        F.line([(r + 8, y), (888, y)], OFF, 1.4, allow={f"r1-u{k}", f"r1-e{k}"}, arrow=True)
        F.text(925, y, lab, 13, HIDDEN, ha="left", bold=True)
    F.circle("r1-sum", 1060, yc - 15, 20, HIDDEN, glow=True)
    F.text(1060, yc - 15, "+", 18, HIDDEN, bold=True, inside=True)
    for k, y in enumerate(ys):
        a, b = np.array([985.0, y]), np.array([1060.0, yc - 15])
        u = (b - a) / np.linalg.norm(b - a)
        F.line([tuple(a), tuple(b - u * 20)], EDGE, 1.0, allow={"r1-sum"})
    F.text(700, yc + 102, "SDXL U-Net, frozen, run three times", 12, OFF, ha="left", bold=True)
    F.text(700, yc + 120, "each prediction with classifier-free guidance 7.5", 11, MUTED, ha="left")
    arrow(F, 1083, 1254, yc - 15, OFF, allow={"r1-sum", "box-r1-out"})
    panel(F, "r1-out", 1262, yc - 60, 1458, yc + 30, OUTPUT)
    F.text(1360, yc - 32, "ONE noise guess", 13, OUTPUT, bold=True, inside=True)
    F.text(1360, yc - 6, "4 × 128 × 128", 12, MUTED, inside=True)
    F.text(1270, yc + 56, r"$\epsilon = \epsilon_{\rm cat} + \epsilon_{\rm dog} - \epsilon_{\rm null}$", 16,
           OUTPUT, ha="left")
    F.text(1270, yc + 86, "both experts pull toward one body", 12, TEXT, ha="left")
    arrow(F, 1466, 1722, yc - 15, OFF, allow={"box-r1-out", "box-r1-use"})
    F.text(1590, yc - 40, "a DDIM step", 12, OFF, bold=True)
    panel(F, "r1-use", 1730, yc - 120, 2360, yc + 120, OFF)
    F.text(1755, yc - 92, "50 such steps, then the VAE: the picture", 14, TEXT, ha="left", bold=True, inside=True)
    picture(F, "r1-plain", N["plain"], 1755, yc - 70, 160, OUTPUT)
    for i, s in enumerate(("Plain product, held-out seed 9:", "one fused cat-dog face.",
                           "Plain product gives one animal", "on 8 of 8 held-out seeds.")):
        F.text(1940, yc - 50 + i * 26, s, 13, TEXT if i < 2 else MUTED, ha="left", inside=True)

    # ================= Row 2: the checker =================
    row_title(F, 560, "2", "The checker: the claim (iteration 2)",
              "it asks \"is this becoming a clean cat and dog?\" and its gradient is the push")
    yc = 715
    rx, ry = picture(F, "r2-in", N["noisy"], 60, yc - 95, 190, INPUT)
    F.text(60, ry + 20, "the same noisy picture x_t", 13, INPUT, ha="left", bold=True)
    F.text(60, ry + 42, "read at any of the 50 steps", 11, MUTED, ha="left")
    chip(F, "r2-step", 400, yc - 60, 585, yc - 24, "step t", size=12)
    chip(F, "r2-pair", 400, yc - 6, 585, yc + 30, "\"a cat and a dog\"", size=12)
    arrow(F, 593, 643, yc - 15)
    network_box(F, 650, yc - 135, 1170, yc + 125)
    r = unet_bars(F, "r2-unet", 675, yc - 20, scale=1.0)
    taps = [(675 + 2 * 32 + 8, "1280 × 32²"), (675 + 3 * 32 + 8, "1280 × 64²"), (675 + 4 * 32 + 8, "640 × 128²")]
    for k, (tx, lab) in enumerate(taps):
        F.circle(f"r2-t{k}", tx, yc + 62, 6, HIDDEN, glow=True)
        F.line([(tx, yc + 36 if k else yc + 5), (tx, yc + 56)], HIDDEN, 1.0, allow={f"r2-t{k}", "r2-unet-unet"})
    F.text(680, yc + 86, "SDXL U-Net, frozen", 12, OFF, ha="left", bold=True)
    F.text(680, yc + 106, "taps: middle, up 0, up 1", 11, MUTED, ha="left")
    hx, ox = 950, 1080
    hy = [yc - 65, yc - 35, yc - 5, yc + 25]
    F.line([(r + 10, yc - 20), (hx - 30, yc - 20)], OFF, 2.0, allow={"r2-unet-unet"}, arrow=True)
    for i, y in enumerate(hy):
        F.circle(f"r2-h{i}", hx, y, 10, HIDDEN)
    F.circle("r2-o", ox, yc - 20, 14, OUTPUT, glow=True)
    for i, y in enumerate(hy):
        a, b = np.array([hx, y]), np.array([ox, yc - 20])
        u = (b - a) / np.linalg.norm(b - a)
        F.line([tuple(a + u * 10), tuple(b - u * 14)], EDGE, 0.7, allow={f"r2-h{i}", "r2-o"})
    F.text(1010, yc + 62, "small head, learnt", 12, HIDDEN, bold=True)
    F.text(1010, yc + 82, f"{N['head_weights']:,} weights", 11, MUTED)
    F.text(1010, yc + 102, "the only part trained", 11, MUTED)
    arrow(F, 1096, 1262, yc - 20, OFF, allow={"r2-o", "box-r2-out"})
    panel(F, "r2-out", 1270, yc - 60, 1450, yc + 20, OUTPUT)
    F.text(1360, yc - 32, "ONE number", 14, OUTPUT, bold=True, inside=True)
    F.text(1360, yc - 6, "log-odds", 12, MUTED, inside=True)
    F.text(1270, yc + 48, "high: becoming a clean cat and dog", 12, TEXT, ha="left")
    F.text(1270, yc + 68, f"held-out accuracy {N['acc']:.3f} on {N['n_eval']} pictures", 12, TEXT, ha="left")
    arrow(F, 1458, 1722, yc - 20, PUSH, allow={"box-r2-out", "box-r2-use"})
    F.text(1590, yc - 44, "its gradient ∇", 12, PUSH, bold=True)
    panel(F, "r2-use", 1730, yc - 130, 2360, yc + 130, PUSH, glow=True)
    F.text(1755, yc - 104, "the push, subtracted at every sampler step", 14, TEXT, ha="left", bold=True,
           inside=True)
    F.text(2045, yc - 66, r"$\epsilon_{\rm cat} + \epsilon_{\rm dog} - \epsilon_{\rm null}"
           r" - s\,\sqrt{1-\bar\alpha_t}\,\nabla\,{\rm logit}$", 21, PUSH, inside=True)
    F.text(2045, yc - 30, "capped at 0.1 of |ε| per step; s sets its strength", 11, MUTED, inside=True)
    picture(F, "r2-pushed", N["pushed"], 1755, yc - 8, 125, PUSH)
    for i, s in enumerate(("Held-out seed 10, pushed at s = 10:", "two bodies where the plain",
                           "product drew one fused face", "(iteration 0's checker).")):
        F.text(1905, yc + 8 + i * 25, s, 12, TEXT, ha="left", inside=True)

    # trained on
    ty = 880
    F.text(60, ty, "Trained on", 14, TEXT, ha="left", bold=True)
    F.text(200, ty, f"({N['n_train']:,} cat × dog pictures, {N['n_clean']} clean; every one labelled by row 3, "
           "noised to a random step, and scored)", 11, MUTED, ha="left")
    F.text(60, ty + 32, "label 1: one normal cat and one normal dog, apart", 12, HIDDEN, ha="left", bold=True)
    F.text(640, ty + 32, "label 0: everything else, including the product's own draws", 12, OUTPUT,
           ha="left", bold=True)
    y0, side = ty + 52, 120
    tiles = [("clean_sdxl", 60, "SDXL joint drawing", HIDDEN), ("clean_photo", 250, "real photo (COCO)", HIDDEN),
             ("poe_hybrid", 640, "product's draw, fused", OUTPUT), ("two_cats", 850, "two cats", OUTPUT),
             ("one_animal", 1060, "one animal", OUTPUT), ("drawing", 1270, "a drawing", OUTPUT)]
    for key, x, cap, col in tiles:
        picture(F, f"ex-{key}", N["ex"][key], x, y0, side, col)
        F.text(x, y0 + side + 18, cap, 11, col, ha="left")

    # ================= Row 3: the judge =================
    row_title(F, 1130, "3", "The judge: scoring only",
              "the Claude API labels every picture; your calls from full-size tiles have the last word")
    yc = 1280
    rx, ry = picture(F, "r3-in", N["judge_img"], 60, yc - 95, 190, INPUT)
    F.text(60, ry + 20, "a FINISHED picture, never a noisy one", 13, INPUT, ha="left", bold=True)
    F.text(60, ry + 42, "an SDXL joint drawing, seed 586", 11, MUTED, ha="left")
    chip(F, "r3-pair", 400, yc - 20, 585, yc + 16, "\"a cat\" and \"a dog\"", size=12)
    arrow(F, 593, 643, yc)
    network_box(F, 650, yc - 125, 1170, yc + 125)
    panel(F, "r3-q", 665, yc - 110, 1155, yc + 85, OFF, dashed=True, fill="none")
    F.text(680, yc - 90, "CLAUDE API: NOT TRAINED BY US; FIXED QUESTIONS", 10, OFF, ha="left",
           bold=True, inside=True)
    for i, s in enumerate(("How many separate bodies?", "Is one clearly and only a cat?",
                           "Is one clearly and only a dog?", "Does any body mix both, in parts or in build?",
                           "Is it an intact photo-like picture?")):
        F.text(690, yc - 60 + i * 28, f"·  {s}", 12, TEXT, ha="left", inside=True)
    F.text(910, yc + 106, "strict rule; real-only for photos", 11, MUTED)
    arrow(F, 1180, 1262, yc, OFF)
    j = N["judge"]
    F.text(1270, yc - 100, "its answers for this picture", 13, OUTPUT, ha="left", bold=True)
    for i, s in enumerate((f"bodies: {j['bodies']}", f"a cat: {'yes' if j['a_present'] else 'no'};  "
                           f"a dog: {'yes' if j['b_present'] else 'no'}", f"mixes both: {'yes' if j['hybrid'] else 'no'}")):
        F.text(1270, yc - 72 + i * 24, s, 13, OUTPUT, ha="left")
    panel(F, "r3-call", 1270, yc + 6, 1650, yc + 120, OFF, dashed=True, fill="none")
    F.text(1285, yc + 26, "ONE call", 10, OFF, ha="left", bold=True, inside=True)
    F.text(1285, yc + 52, f"{j['verdict']}", 15, OUTPUT, ha="left", bold=True, inside=True)
    words, lines, cur = j["note"].split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 50:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    for i, s in enumerate(lines[:2]):
        F.text(1285, yc + 80 + i * 18, s, 10, MUTED, ha="left", inside=True)
    arrow(F, 1658, 1722, yc + 60, OFF, allow={"box-r3-call", "box-r3-use"})
    panel(F, "r3-use", 1730, yc - 95, 2360, yc + 120, OFF)
    F.text(1755, yc - 68, "labels and counts only", 14, TEXT, ha="left", bold=True, inside=True)
    for i, s in enumerate(("It labels every training and evaluation picture.",
                           "It never pushes the sampler, so the push",
                           "cannot learn to fool the labeller.",
                           f"Agreement with your hand calls: {N['api_agree']:.3f} (balanced).",
                           "Push sheets are called by you, from full-size tiles.")):
        F.text(1755, yc - 32 + i * 28, s, 13, TEXT if i < 3 else MUTED, ha="left", inside=True)

    # ================= Footer =================
    F.text(40, 1470, "Every picture is a real file (inputs/): SDXL joint drawings with their Claude API labels "
           "(api-labels/more_catdog2.jsonl), a plan 23 product drawing, COCO photos, and two tiles of "
           "result-sheet-catdog.", 12, MUTED, ha="left")
    F.text(40, 1496, "Illustrative: the pixel noise on rows 1 and 2's input. Sizes from poe_repair/config.py, "
           "_sdxl/metrics.py, methods/_classifier_guided.py and scripts/rung2; accuracy from readout-catdog-more2 "
           "on eval_set_catdog_v2.json.", 12, MUTED, ha="left")
    F.text(40, 1522, "Drawn by make_three_networks_in_rung_2.py, in the register of "
           "make_three_networks_in_step_7b.py.", 12, MUTED, ha="left")
    return F


def numbers():
    picks = json.loads((IN / "picks.json").read_text())
    rung2 = HERE.parents[3] / "scripts" / "rung2"
    train = json.loads((rung2 / "training_set_catdog_more2.json").read_text())["tiles"]
    ev = json.loads((rung2 / "eval_set_catdog_v2.json").read_text())["tiles"]
    plain = load("plain_seed09.png", 256)
    return dict(noisy=noisy(plain), plain=plain, pushed=load("pushed_seed10_s10.png", 256),
                ex={k: load(f"{k}.jpg", 192) for k in ("clean_sdxl", "two_cats", "drawing")}
                | {"clean_photo": load("clean_photo.jpg", 192), "poe_hybrid": load("poe_hybrid.jpg", 192),
                   "one_animal": load("one_animal.jpg", 192)},
                judge_img=load("judge_hybrid.jpg", 256), judge=picks["judge_hybrid"],
                head_weights=780417, acc=0.737, n_eval=len(ev), n_train=len(train),
                n_clean=sum(t["label"] for t in train), api_agree=0.789)


def main():
    N = numbers()
    F = draw(N)
    base.check(F)
    F.fig.savefig(HERE / "three-networks-in-rung-2.png", dpi=100)
    F.fig.savefig(HERE / "three-networks-in-rung-2.svg")
    print("wrote three-networks-in-rung-2.png and .svg;", N["n_train"], "training pictures,", N["n_eval"], "evaluation")


if __name__ == "__main__":
    main()
