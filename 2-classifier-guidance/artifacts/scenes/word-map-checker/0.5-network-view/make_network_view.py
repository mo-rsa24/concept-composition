#!/usr/bin/env python
"""The word-map checker's network view: wm0 to wm6, each written as SVG and PNG.

Every size is read from ../numbers/numbers.json. Level 0.5 shows structure and sizes; worked values
belong to the miniature. Heat maps here are schematic and each figure that holds one says so.

Run:
  PYTHONDONTWRITEBYTECODE=1 $HOME/.cache/word-map-ladder-venv/bin/python make_network_view.py [wm3 ...]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from figure_kit import (Fig, FADED_FILL, FADED_TEXT, GREY, HIDDEN, HIDDEN_FILL, INK, INPUT, INPUT_FILL, ORANGE, ORANGE_FILL,  # noqa: E402
                        OUTPUT, OUTPUT_FILL, WHITE, tex_number)

N = json.loads((ROOT / "numbers" / "numbers.json").read_text())
SDXL = N["word_maps"]["sdxl"]["value"]
HEAD_SD = N["head"]["sd2_config_on_sdxl_1300_maps_proposal"]
HEAD_FLUX = N["head"]["flux_dev_config_on_sdxl_1300_maps_proposal"]
STYLES = {"input": (INPUT_FILL, INPUT), "hidden": (HIDDEN_FILL, HIDDEN), "output": (OUTPUT_FILL, OUTPUT),
          "push": (ORANGE_FILL, ORANGE), "off": (FADED_FILL, GREY), "plain": (WHITE, GREY)}
GAP = 7


class B:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.l, self.r = (x, y + h / 2), (x + w, y + h / 2)
        self.t, self.b = (x + w / 2, y), (x + w / 2, y + h)


def box(f, x, y, w, h, lines, kind="plain", size=20, glow=False, dash=None):
    fill, stroke = STYLES[kind]
    if glow:
        f.halo(x, y, w, h, stroke)
    f.rect(x, y, w, h, fill, stroke, 1.6, 8, dash=dash)
    step = size * (1.75 if any(l.startswith("$") for l in lines) else 1.34)
    top = y + h / 2 - step * (len(lines) - 1) / 2 + size * 0.34
    for i, line in enumerate(lines):
        yy = top + i * step
        if line.startswith("$"):
            f.maths(line.strip("$"), x + w / 2, yy - size * 0.34, size + 2, stroke if kind != "plain" else INK)
        else:
            colour = INK if kind != "off" else "#566472"
            f.words(x + w / 2, yy, line, size, colour, 600 if i == 0 and len(lines) > 1 else 400, "middle")
    return B(x, y, w, h)


def go(f, p, q, color=GREY, width=2.2, head=10):
    dx, dy = q[0] - p[0], q[1] - p[1]
    d = math.hypot(dx, dy)
    ux, uy = dx / d, dy / d
    f.arrow(p[0] + GAP * ux, p[1] + GAP * uy, q[0] - GAP * ux, q[1] - GAP * uy, color, width, head)


def path(f, pts, color=GREY, width=2.2, head=10):
    a, b = pts[0], pts[1]
    d = math.hypot(b[0] - a[0], b[1] - a[1])
    first = (a[0] + GAP * (b[0] - a[0]) / d, a[1] + GAP * (b[1] - a[1]) / d)
    y, z = pts[-2], pts[-1]
    d = math.hypot(z[0] - y[0], z[1] - y[1])
    last = (z[0] - GAP * (z[0] - y[0]) / d, z[1] - GAP * (z[1] - y[1]) / d)
    f.route([first, *pts[1:-1], last], color, width, head)


def title(f, text, sub=None):
    f.words(40, 48, text, 31, INK, 700)
    if sub:
        f.words(40, 84, sub, 20, "#566472")


def tag(f, x, y, text, color=ORANGE):
    f.words(x, y, text, 18, color, 600)


def heat(f, x, y, size, values, color):
    """A small grid of cells on a plain background; darker means more of the map's mass."""
    n = len(values)
    c = size / n
    top = max(max(row) for row in values) or 1
    cells = []
    for j, row in enumerate(values):
        for i, v in enumerate(row):
            cells.append(f'<rect x="{x + i * c:.2f}" y="{y + j * c:.2f}" width="{c + .2:.2f}" height="{c + .2:.2f}" '
                         f'fill="{color}" fill-opacity="{0.06 + 0.9 * v / top:.3f}"/>')
    f.rect(x, y, size, size, WHITE, GREY, 1.0)
    f.raw("".join(cells))
    f.raw(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" fill="none" stroke="{GREY}" stroke-width="1"/>')
    return B(x, y, size, size)


def blob(n, cx, cy, s):
    return [[math.exp(-(((i + .5) / n - cx) ** 2 + ((j + .5) / n - cy) ** 2) / (2 * s * s)) for i in range(n)] for j in range(n)]


def wm0():
    f = Fig("wm0-where-the-checker-sits", 1560, 640)
    title(f, "Where the checker would sit in one step of our sampler",
          "SDXL's product of experts with a Learn-to-Steer head added. This wiring is our proposal and has not been run.")
    x = box(f, 40, 185, 165, 120, ["noisy latent", "$x_t$"], "input")
    u = box(f, 275, 150, 300, 190, ["frozen SDXL U-Net", "one pass per prompt:", "\"a cat\", \"a dog\", \"\""], "off")
    e = box(f, 650, 185, 250, 120, ["product of experts", r"$\varepsilon_{\text{PoE}}$"], "hidden")
    f.circle(985, 245, 28, ORANGE_FILL, ORANGE, 2)
    f.maths(r"-", 985, 245, 30, ORANGE)
    d = box(f, 1075, 185, 170, 120, ["DDIM", "step"], "plain")
    n = box(f, 1320, 185, 190, 120, ["next latent", r"$x_{t-1}$"], "input")
    go(f, x.r, u.l), go(f, u.r, e.l), go(f, e.r, (957, 245)), go(f, (1013, 245), d.l), go(f, d.r, n.l)
    m = box(f, 275, 440, 300, 130, ["word maps", "\"cat\" from its prompt,", "\"dog\" from its prompt"], "hidden")
    c = box(f, 650, 440, 250, 130, ["the checker", "reads only", "the two words' maps"], "hidden", glow=True)
    p = box(f, 960, 440, 290, 130, ["capped push", r"$\lambda\,\sigma_t\,\nabla_{x_t}\log\hat p[r]$"], "push")
    go(f, u.b, m.t, HIDDEN), go(f, m.r, c.l, HIDDEN), go(f, c.r, p.l, ORANGE)
    go(f, (985, 440), (985, 273), ORANGE)
    f.words(1006, 372, "subtracted from the prediction", 19, ORANGE, 600)
    f.words(440, 398, "read while the U-Net runs", 18, HIDDEN, 600)
    f.write(HERE, "One sampler step: the latent through the frozen U-Net to the product's noise prediction, minus a capped push computed by a checker that reads only the word maps, then a DDIM step.")


def wm1():
    f = Fig("wm1-three-inputs-and-where-the-maps-come-from", 1640, 900)
    title(f, "Three inputs go in, and each word's attention maps are read on the way",
          "SDXL's sizes, read from its config. The U-Net stays closed here: only its cross-attention layers are marked.")
    c = box(f, 40, 150, 200, 110, ["caption", "\"a cat\""], "input")
    te = box(f, 310, 135, 280, 140, ["two CLIP text encoders", "widths 768 and 1280", "stacked per token"], "off")
    wv = box(f, 660, 130, 270, 76, ["word vectors", "77 tokens × 2048"], "hidden", size=18)
    sv = box(f, 660, 222, 270, 66, ["summary vector, 1280"], "hidden", size=18)
    go(f, c.r, te.l, INPUT), go(f, (590, 170), wv.l, HIDDEN), go(f, (590, 245), sv.l, HIDDEN)
    t = box(f, 40, 365, 200, 100, ["timestep", "$t$"], "input")
    te2 = box(f, 310, 365, 280, 100, ["waves, then a small", "network"], "off")
    tv = box(f, 660, 365, 270, 100, ["time vector", "1280"], "hidden")
    go(f, t.r, te2.l, INPUT), go(f, te2.r, tv.l, HIDDEN)
    f.circle(1010, 327, 20, HIDDEN_FILL, HIDDEN, 1.8)
    f.maths("+", 1010, 327, 22, HIDDEN)
    path(f, [sv.r, (1010, 255), (1010, 307)], HIDDEN)
    path(f, [tv.r, (1010, 415), (1010, 347)], HIDDEN)
    x = box(f, 40, 590, 200, 120, ["noisy latent", "$x_t$", "4 × 128 × 128"], "input", size=19)
    u = box(f, 1090, 150, 245, 590, ["frozen U-Net", "(not opened here)"], "off")
    path(f, [x.r, (640, 650), (1050, 650), (1090, 650)], INPUT)
    path(f, [wv.r, (1060, 168), (1060, 220), (1090, 220)], HIDDEN)
    f.words(960, 124, "enter by cross-attention", 18, HIDDEN, 600, "middle")
    path(f, [(1030, 327), (1090, 327)], HIDDEN)
    f.words(800, 515, "summary + time joins every ResBlock", 18, HIDDEN, 600, "middle")
    rows = SDXL["per_grid"]
    for i, r in enumerate(rows):
        yy = 250 + i * 260
        f.circle(1335, yy + 40, 11, HIDDEN, HIDDEN, 1.5)
        b = box(f, 1395, yy, 215, 130, [f"{r['grid']} × {r['grid']} grid", f"{r['layers']} layers", f"× {r['heads']} heads"], "hidden", size=19)
        f.line(1353, yy + 40, 1388, yy + 40, HIDDEN, 2.2)
        f.words(1502, yy + 158, f"= {r['maps_per_word']:,} maps", 21, HIDDEN, 700, "middle")
    f.words(40, 800, f"Each word gets {SDXL['maps_per_word']:,} maps from {SDXL['layers']} cross-attention layers.", 25, INK, 700)
    f.words(40, 836, "Counted from SDXL's config.", 19, "#566472")
    f.write(HERE, "A caption, a timestep and the noisy latent enter a closed U-Net; its 64 by 64 and 32 by 32 cross-attention layers each give every word a stack of maps, 1300 in all.")


def wm2():
    f = Fig("wm2-from-one-layer-to-a-words-stack", 1600, 720)
    title(f, "One head of one layer gives each word one map, and all of them are kept",
          "The heat maps are schematic. The counts at the bottom are real.")
    f.words(40, 145, "1. each patch shares", 22, INK, 700)
    f.words(40, 175, "1 among the tokens", 22, INK, 700)
    tokens = ["start", "a", "cat", "end"]
    weights = [.10, .08, .74, .08]
    bx, by, bw = 60, 290, 64
    for i, (tok, w) in enumerate(zip(tokens, weights)):
        h = 105 * w / .74
        col, fill = (OUTPUT, OUTPUT_FILL) if tok == "cat" else (GREY, FADED_FILL)
        f.rect(bx + i * (bw + 16), by + 40 - h, bw, h, fill, col, 1.4)
        f.words(bx + i * (bw + 16) + bw / 2, by + 70, tok, 18, col, 600 if tok == "cat" else 400, "middle")
    f.words(60, 400, "one patch's softmax row", 18, "#566472")
    f.words(60, 428, "(schematic weights)", 18, "#566472")
    f.words(450, 145, "2. the \"cat\" column", 22, INK, 700)
    f.words(450, 175, "is the word's map", 22, INK, 700)
    a = heat(f, 470, 205, 180, blob(8, .36, .46, .2), OUTPUT)
    f.words(560, 420, "one map", 19, OUTPUT, 600, "middle")
    f.words(560, 446, "at the layer's grid", 18, "#566472", 400, "middle")
    go(f, (394, 290), a.l, OUTPUT)
    f.words(760, 145, "3. resize to 16 × 16,", 22, INK, 700)
    f.words(760, 175, "divide by its sum", 22, INK, 700)
    b = heat(f, 790, 205, 180, blob(16, .36, .46, .2), OUTPUT)
    go(f, a.r, b.l, OUTPUT)
    f.maths(r"\hat A^{(\ell)}=\frac{A^{(\ell)}}{\sum_{p}A^{(\ell)}[p]}", 880, 455, 25, OUTPUT)
    f.words(1090, 145, "4. keep every one", 22, INK, 700)
    for i in range(6):
        off = 11 * (5 - i)
        heat(f, 1110 + off, 205 + off, 150, blob(8, .36 + .025 * i, .46 - .015 * i, .2), HIDDEN)
    f.words(1215, 440, "L × H maps", 20, HIDDEN, 700, "middle")
    f.words(1215, 468, "for this word", 18, "#566472", 400, "middle")
    go(f, b.r, (1110, 300), HIDDEN)
    f.card(40, 500, 1520, 180, "#fbfbfc", GREY, 1.2, 8)
    f.words(70, 540, "Maps each word hands the head", 22, INK, 700)
    w = N["word_maps"]
    cols = [("SD 1.4", w["sd_1_4"]["value"]["maps_per_word"], "counted"), ("SD 2.1", w["sd_2_1"]["value"]["maps_per_word"], "inferred"),
            ("FLUX", w["flux"]["value"]["maps_per_word"], "from its code"), ("SDXL", SDXL["maps_per_word"], "our proposal")]
    for i, (name, value, how) in enumerate(cols):
        xx = 110 + i * 370
        col = ORANGE if name == "SDXL" else INK
        f.words(xx, 590, name, 21, col, 700)
        f.words(xx, 628, f"{value:,}", 29, col, 700)
        f.words(xx + 120, 628, how, 18, "#566472")
    f.write(HERE, "Each patch shares one unit of attention across tokens; the cat column is a map; it is resized to 16 by 16 and renormalised, and every layer and head's map is kept: 128 on SD 1.4, 195 on SD 2.1 (inferred), 1368 on FLUX, 1300 on SDXL.")


def wm3():
    f = Fig("wm3-the-head-reads-the-two-stacks", 1720, 980)
    a, c = HEAD_SD["aggregator"], HEAD_SD["classifier"]
    title(f, "The head: pool each word's stack into one map, then read the pair",
          "Sizes from the authors' SD 2.1 config, with SDXL's 1,300 maps per word as our proposal. The aggregator's weights are shared by both words.")
    for row, (word, yy, tint) in enumerate((("subject \"cat\"", 150, OUTPUT), ("object \"dog\"", 400, INPUT))):
        n = 5
        for i in range(n):
            off = 7 * (n - 1 - i)
            heat(f, 40 + off, yy + 15 + off, 120, blob(8, .36 if row == 0 else .62, .46, .19), tint)
        f.words(40, yy + 200, word, 19, tint, 700)
        f.words(40, yy + 226, "1,300 × 16 × 16", 18, "#566472")
        p = box(f, 240, yy + 20, 210, 145, ["each map", "is one token:", "flatten, project", "256 → 256"], "hidden", size=17)
        tr = box(f, 520, yy + 20, 245, 145, ["transformer", f"{a['num_layers']} layers, {a['nhead']} heads", "step vector added", "at every layer"], "hidden", size=17)
        q = box(f, 835, yy + 20, 225, 145, ["learned query", f"one attention layer,", f"{a['layer_attn_nheads']} heads", "1,300 → 1 token"], "hidden", size=17)
        o = box(f, 1130, yy + 20, 170, 145, ["linear", "256 → 256", "reshape"], "hidden", size=17)
        m = heat(f, 1375, yy + 37, 112, blob(16, .36 if row == 0 else .62, .46, .19), tint)
        f.words(1431, yy + 178, "one 16 × 16 map", 17, tint, 600, "middle")
        go(f, (190, yy + 92), p.l, HIDDEN), go(f, p.r, tr.l, HIDDEN), go(f, tr.r, q.l, HIDDEN), go(f, q.r, o.l, HIDDEN), go(f, o.r, m.l, HIDDEN)
    yy = 680
    s = box(f, 40, yy, 200, 150, ["stack the pair", "2 × 16 × 16", "flatten to 512"], "hidden", size=18)
    pr = box(f, 310, yy, 170, 150, ["project", "512 → 256"], "hidden", size=18)
    tr = box(f, 550, yy, 300, 150, ["transformer", f"{c['num_layers']} layers, {c['nhead']} heads", "over 1 token: the whole", "pair is a single vector"], "hidden", size=17)
    tv = box(f, 920, yy, 200, 150, ["join step vector", "256 + 16"], "hidden", size=18)
    mlp = box(f, 1190, yy, 200, 150, ["MLP", "256 → 128 → 5"], "hidden", size=18)
    o = box(f, 1460, yy, 220, 150, ["5 logits", "above, below, left,", "right, neutral"], "output", size=17)
    go(f, s.r, pr.l, HIDDEN), go(f, pr.r, tr.l, HIDDEN), go(f, tr.r, tv.l, HIDDEN), go(f, tv.r, mlp.l, HIDDEN), go(f, mlp.r, o.l, OUTPUT)
    f.line(1493, 243, 1570, 243, HIDDEN, 2.2)
    f.line(1493, 493, 1570, 493, HIDDEN, 2.2)
    f.line(1570, 243, 1570, 652, HIDDEN, 2.2)
    f.line(1570, 652, 140, 652, HIDDEN, 2.2)
    f.arrow(140, 652, 140, 673, HIDDEN, 2.2, 10)
    f.words(40, 890, f"Trained weights: {HEAD_SD['trained_weights']['total']:,} on the SD config, {HEAD_FLUX['trained_weights']['total']:,} on the FLUX.1-dev config.", 23, INK, 700)
    f.words(40, 926, "The time input is the step number 0 to 49, snapped up to one of 5, 10, 15, 20, 25 or 49, looked up in a 50-row table.", 19, "#566472")
    f.write(HERE, "Each word's 1300 maps become tokens, pass a two-layer transformer and a learned-query attention layer, and come out as one 16 by 16 map. The pair is stacked, flattened to 512, projected to 256, passed through a one-token transformer, joined with a step vector and read by an MLP into five logits.")


def wm4():
    f = Fig("wm4-training-on-the-right-and-wrong-word", 1640, 780)
    title(f, "Why each training picture is inverted twice",
          "If every caption states the truth, the head can read the relation word's trace in the maps and ignore the layout.")
    img = box(f, 40, 300, 225, 170, ["training picture", "dog left of a cat"], "input")
    rows = ((150, "right caption", "\"a dog to the left of a cat\"", HIDDEN), (470, "wrong caption", "\"a dog above a cat\"", OUTPUT))
    for yy, name, cap, col in rows:
        c = box(f, 345, yy, 355, 130, [name, cap], "plain", size=19)
        path(f, [img.t if yy < 300 else img.b, (152, yy + 65), c.l], GREY)
        i = box(f, 775, yy, 200, 130, ["invert, read", "the word maps"], "off", size=19)
        go(f, c.r, i.l)
        l = box(f, 1050, yy, 245, 130, ["label", "\"left of\""], "output", size=19)
        go(f, i.r, l.l, OUTPUT)
    h = box(f, 1370, 300, 230, 170, ["the head", "cross-entropy", "against the picture"], "hidden", size=18, glow=True)
    for yy in (215, 535):
        path(f, [(1295, yy), (1332, yy), (1332, 385), h.l], OUTPUT)
    v = N["paper"]["leakage_fix"]["value"]
    f.words(40, 690, f"With the wrong caption added, GenEval position rose from {v['off']} to {v['on']} on FLUX.1-schnell validation (p. 8, Table 4).", 22, INK, 700)
    f.words(40, 728, "GenEval position is the fraction of images with the right relation, 0 to 1. The paper reports no direct measure of leakage after the fix.", 19, "#566472")
    f.write(HERE, "A training picture of a dog left of a cat is inverted under the right caption and a wrong one. Both sets of word maps carry the label left of, so the head cannot use the relation word.")


def wm5():
    f = Fig("wm5-two-ways-to-apply-the-push", 1640, 880)
    p = N["paper"]
    title(f, "The same gradient can enter the sampler in two ways",
          "Top: what Learn-to-Steer does on its models. Bottom: what our sampler does with a head today. SDXL with this head is our proposal.")
    f.words(40, 150, "Learn-to-Steer: move the latent itself, many times", 24, INPUT, 700)
    x = box(f, 40, 180, 150, 110, ["latent", "$x_t$"], "input")
    u = box(f, 255, 180, 215, 110, ["frozen model", "prompt branch only"], "off", size=18)
    h = box(f, 535, 180, 170, 110, ["maps,", "then head"], "hidden", size=19)
    l = box(f, 770, 180, 225, 110, ["loss", r"$-\log\hat p[r]$"], "output")
    go(f, x.r, u.l), go(f, u.r, h.l), go(f, h.r, l.l, OUTPUT)
    path(f, [l.b, (882, 350), (115, 350), x.b], ORANGE)
    f.words(420, 382, "gradient back to the latent", 19, ORANGE, 600, "middle")
    b = box(f, 1060, 165, 540, 215, ["update the latent directly", r"$x_t\leftarrow x_t-\eta\,\nabla_{x_t}\mathcal L_t$", f"step size {p['step_size']['value']['SD 2.1 and Flux']} (SD 2.1, Flux), no cap stated",
                                     "up to 16 updates per step in the code (15 printed)", "on the first 25 of 50 steps"], "push", size=19)
    f.line(40, 450, 1600, 450, "#d4d9df", 1.2)
    f.words(40, 500, "Our sampler: change the noise prediction, once, with a cap", 24, OUTPUT, 700)
    x = box(f, 40, 530, 150, 110, ["latent", "$x_t$"], "input")
    u = box(f, 255, 530, 215, 110, ["frozen U-Net", "3 prompts"], "off", size=18)
    e = box(f, 535, 530, 170, 110, ["product", r"$\varepsilon_{\text{PoE}}$"], "hidden", size=19)
    d = box(f, 770, 530, 225, 110, ["push, then", "one DDIM step"], "plain", size=19)
    go(f, x.r, u.l), go(f, u.r, e.l), go(f, e.r, d.l, HIDDEN)
    path(f, [(362, 640), (362, 690), (882, 690), (882, 640)], ORANGE)
    f.words(395, 718, "the checker's gradient, shortened", 19, ORANGE, 600)
    f.words(395, 746, "to at most c of the prediction's norm", 19, ORANGE, 600)
    b = box(f, 1060, 515, 540, 215, ["adjust the prediction", r"$\varepsilon\leftarrow\varepsilon_{\text{PoE}}-\lambda\,\sigma_t\,\nabla_{x_t}\log\hat p[r]$", "shortened to at most $c$ of its norm".replace("$c$", "c"),
                                     "one push per sampler step", "the code's default c is 0.1; proposed: 0.4"], "push", size=19)
    a = N["ours"]["attention_separation_push"]["value"]
    f.words(40, 812, f"Our own hand-written attention-separation push drew two animals on {a['two_animals']} of {a['seeds']} SDXL seeds.", 23, OUTPUT, 700)
    f.words(40, 846, "Seeds 9 to 16, steps 0 to 19, draft calls, no verdict recorded. It used a correlation loss, so it does not test a trained head.", 19, "#566472")
    f.write(HERE, "Two lanes. Learn-to-Steer updates the latent itself up to 16 times per step at step size 5. Our sampler subtracts a capped push from the product's noise prediction once per step. Our hand-written attention-separation push drew two animals on 0 of 8 SDXL seeds.")


def wm6():
    f = Fig("wm6-one-joint-prompt-against-two-prompts", 1640, 830)
    o = N["ours"]["overlap_of_top_10_percent"]["value"]
    title(f, "Learn-to-Steer's maps come from one prompt. Ours would come from two.",
          "This is the gap a head trained on joint prompts would have to cross. The heat maps are schematic.")
    f.card(40, 120, 760, 500, "#fbfbfc", GREY, 1.2, 8)
    f.card(840, 120, 760, 500, "#fbfbfc", GREY, 1.2, 8)
    f.words(70, 165, "One joint prompt", 25, INPUT, 700)
    f.words(70, 197, "\"a cat to the left of a dog\"", 20, INK)
    f.words(870, 165, "Our product: two prompts", 25, OUTPUT, 700)
    f.words(870, 197, "\"a cat\" and \"a dog\", one U-Net pass each", 20, INK)
    a = heat(f, 110, 240, 190, blob(16, .28, .5, .16), OUTPUT)
    b = heat(f, 500, 240, 190, blob(16, .72, .5, .16), INPUT)
    f.words(205, 462, "\"cat\"", 21, OUTPUT, 700, "middle")
    f.words(595, 462, "\"dog\"", 21, INPUT, 700, "middle")
    f.words(70, 515, "Both words share one softmax at each patch,", 20, INK)
    f.words(70, 545, "so where one wins the other loses.", 20, INK)
    c = heat(f, 910, 240, 190, blob(16, .47, .5, .18), OUTPUT)
    d = heat(f, 1300, 240, 190, blob(16, .53, .5, .18), INPUT)
    f.words(1005, 462, "\"cat\"", 21, OUTPUT, 700, "middle")
    f.words(1395, 462, "\"dog\"", 21, INPUT, 700, "middle")
    f.words(870, 515, "Each word has a softmax to itself, so nothing", 20, INK)
    f.words(870, 545, "stops both from filling the same body.", 20, INK)
    f.words(40, 690, "Overlap of the top 10% of the cat and dog maps, 1 means the same places (median over 8 seeds, steps 2 to 40):", 22, INK, 700)
    f.words(40, 730, f"our product {o['product_median_range'][0]} to {o['product_median_range'][1]}", 26, OUTPUT, 700)
    f.words(560, 730, f"joint prompt {o['joint_median_range'][0]} to {o['joint_median_range'][1]}", 26, INPUT, 700)
    f.words(40, 776, "Measured on SDXL here, with \"a cat and a dog\" as the joint prompt. Learn-to-Steer never ran on SDXL or on a product of experts.", 19, "#566472")
    f.write(HERE, "Left: under one joint prompt the cat and dog maps separate because they share a softmax. Right: under two prompts each map has its own softmax and both fill one body. Measured overlap of the top 10 percent is 0.506 to 0.814 for the product and 0.225 to 0.394 for the joint prompt.")


FIGURES = dict(wm0=wm0, wm1=wm1, wm2=wm2, wm3=wm3, wm4=wm4, wm5=wm5, wm6=wm6)

if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGURES)
    failed = []
    for name in names:
        try:
            FIGURES[name]()
        except RuntimeError as err:
            failed.append(name)
            lines = str(err).splitlines()
            print(lines[0])
            print("\n".join(lines[1:14]))
    if failed:
        sys.exit(f"failed: {failed}")
