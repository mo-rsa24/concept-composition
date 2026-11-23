#!/usr/bin/env python
"""The miniature's figures, mm0 to mm5. Every number is read from miniature_answer_key.json.

Run:
  PYTHONDONTWRITEBYTECODE=1 $HOME/.cache/word-map-ladder-venv/bin/python make_miniature_figures.py [mm2 ...]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from figure_kit import (Fig, FADED_FILL, GREY, HIDDEN, HIDDEN_FILL, INK, INPUT, INPUT_FILL, ORANGE, ORANGE_FILL,  # noqa: E402
                        OUTPUT, OUTPUT_FILL, WHITE)

K = json.loads((HERE / "miniature_answer_key.json").read_text())
STYLES = {"input": (INPUT_FILL, INPUT), "hidden": (HIDDEN_FILL, HIDDEN), "output": (OUTPUT_FILL, OUTPUT),
          "push": (ORANGE_FILL, ORANGE), "off": (FADED_FILL, GREY), "plain": (WHITE, GREY)}
MUTED = "#566472"
GAP = 7


class B:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.l, self.r = (x, y + h / 2), (x + w, y + h / 2)
        self.t, self.b = (x + w / 2, y), (x + w / 2, y + h)


def box(f, x, y, w, h, lines, kind="plain", size=20):
    fill, stroke = STYLES[kind]
    f.rect(x, y, w, h, fill, stroke, 1.6, 8)
    step = size * (1.75 if any(l.startswith("$") for l in lines) else 1.34)
    top = y + h / 2 - step * (len(lines) - 1) / 2 + size * 0.34
    for i, line in enumerate(lines):
        yy = top + i * step
        if line.startswith("$"):
            f.maths(line.strip("$"), x + w / 2, yy - size * 0.34, size + 2, stroke if kind != "plain" else INK)
        else:
            f.words(x + w / 2, yy, line, size, INK, 600 if i == 0 and len(lines) > 1 else 400, "middle")
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
    f.words(40, 48, text, 30, INK, 700)
    if sub:
        f.words(40, 84, sub, 20, MUTED)


def grid(f, x, y, cell, values, color, digits=3, lo=None, hi=None, size=None):
    """A 2 x 2 grid with each patch's value printed in it; darker means larger."""
    lo = min(values) if lo is None else lo
    hi = max(values) if hi is None else hi
    span = (hi - lo) or 1
    cells = []
    for n, v in enumerate(values):
        cx, cy = x + (n % 2) * cell, y + (n // 2) * cell
        cells.append(f'<rect x="{cx}" y="{cy}" width="{cell}" height="{cell}" fill="{color}" fill-opacity="{0.05 + 0.5 * (v - lo) / span:.3f}" '
                     f'stroke="{GREY}" stroke-width="1"/>')
    f.rect(x, y, 2 * cell, 2 * cell, WHITE, GREY, 1.2)
    f.raw("".join(cells))
    size = size or int(cell * 0.27)
    for n, v in enumerate(values):
        cx, cy = x + (n % 2) * cell + cell / 2, y + (n // 2) * cell + cell / 2 + size * 0.36
        f.words(cx, cy, f"{v:.{digits}f}", size, INK, 500, "middle")
    return B(x, y, 2 * cell, 2 * cell)


def mm0():
    f = Fig("mm0-the-whole-miniature", 1720, 730)
    fw, g = K["forward_fused"], K["gradient_fused"]
    title(f, "The whole method on a 2 × 2 latent", "Forward along the top, the gradient back along the bottom. Weights are set by hand, so this shows the parts and predicts nothing about SDXL.")
    x = K["latent_start"]
    f.words(40, 145, "noisy latent", 21, INPUT, 700)
    f.words(40, 175, "cat channel", 18, INPUT, 600)
    a = grid(f, 40, 185, 62, x["cat"], INPUT, 0, -1, 2)
    f.words(40, 340, "dog channel", 18, INPUT, 600)
    b = grid(f, 40, 350, 62, x["dog"], INPUT, 0, -1, 2)
    f.words(245, 145, "pooled word maps", 21, HIDDEN, 700)
    f.words(245, 175, "cat", 18, OUTPUT, 600)
    c = grid(f, 245, 185, 74, fw["pooled_cat"], OUTPUT, 3, 0, .5)
    f.words(245, 340 + 22, "dog", 18, INPUT, 600)
    d = grid(f, 245, 372, 74, fw["pooled_dog"], INPUT, 3, 0, .5)
    go(f, (164, 247), (245, 247), HIDDEN), go(f, (164, 412), (245, 434), HIDDEN)
    h = box(f, 480, 245, 225, 190, ["two hidden units", "centre-of-mass gaps", r"$\Delta_u=0$", r"$\Delta_v=0$"], "hidden", 19)
    go(f, (393, 259), (480, 310), HIDDEN), go(f, (393, 446), (480, 380), HIDDEN)
    lg = box(f, 780, 245, 245, 190, ["5 logits", "scale κ = 4", "all 0 here"], "hidden", 19)
    go(f, h.r, lg.l, HIDDEN)
    p = box(f, 1100, 245, 250, 190, ["probabilities", "0.2 on all 5 classes", "no class means \"fused\""], "output", 19)
    go(f, lg.r, p.l, OUTPUT)
    l = box(f, 1425, 245, 255, 190, ["loss, target \"left of\"", r"$-\log\hat p[\text{left}]$", f"= {fw['loss']:.4f}"], "output", 19)
    go(f, p.r, l.l, OUTPUT)
    path(f, [l.b, (1552, 560), (102, 560), (102, 474)], ORANGE)
    f.words(330, 545, "gradient of the loss, back through head and maps into the latent", 21, ORANGE, 700)
    f.words(40, 615, "gradient on the cat channel", 19, ORANGE, 600)
    f.words(620, 615, "gradient on the dog channel", 19, ORANGE, 600)
    txt = lambda v: "  ".join(f"{u:+.4f}" for u in v)
    f.words(40, 650, txt(g["cat"]), 22, INK, 500)
    f.words(620, 650, txt(g["dog"]), 22, INK, 500)
    f.words(1170, 650, "patch order: top left, top right,", 18, MUTED)
    f.words(1170, 676, "bottom left, bottom right", 18, MUTED)
    f.write(HERE, "A fused body in both channels of a 2 by 2 latent gives identical pooled maps, zero centre-of-mass gaps, 0.2 on all five classes and a loss of 1.6094; the gradient runs back to the latent.")


def mm1():
    f = Fig("mm1-from-latent-to-pooled-map", 1680, 800)
    m, fw = K["maps"], K["forward_fused"]
    title(f, "How the cat's latent becomes its pooled map", "Each patch splits one unit between \"cat\" and the background token, whose key is 0.")
    a = grid(f, 40, 170, 80, K["latent_start"]["cat"], INPUT, 0, -1, 2)
    f.words(40, 150, "cat channel x", 20, INPUT, 700)
    f.maths(r"A^{(\ell)}[p]=\frac{e^{k_\ell x_p}}{1+e^{k_\ell x_p}}", 460, 130, 25, HIDDEN)
    b = grid(f, 370, 170, 80, m["cat_layer1"], HIDDEN, 3, 0, 1)
    f.words(370, 355, "layer 1, key 2", 19, HIDDEN, 600)
    f.words(370, 381, f"sums to {m['layer1_sum']:.4f}", 18, MUTED)
    c = grid(f, 370, 410, 80, m["cat_layer2"], HIDDEN, 3, 0, 1)
    f.words(370, 595, "layer 2, key 1", 19, HIDDEN, 600)
    f.words(370, 621, f"sums to {m['layer2_sum']:.4f}", 18, MUTED)
    path(f, [(200, 250), (370, 250)], HIDDEN)
    f.route([(285, 250), (285, 490), (363, 490)], HIDDEN, 2.2, 10)
    f.maths(r"\hat A=\frac{A}{\sum_p A[p]}", 860, 130, 25, HIDDEN)
    d = grid(f, 770, 170, 80, m["cat_layer1_renormalised"], HIDDEN, 3, 0, .5)
    e = grid(f, 770, 410, 80, m["cat_layer2_renormalised"], HIDDEN, 3, 0, .5)
    go(f, b.r, d.l, HIDDEN), go(f, c.r, e.l, HIDDEN)
    f.words(770, 355, "adds to 1 over the grid", 19, HIDDEN, 600)
    f.words(770, 595, "adds to 1 over the grid", 19, HIDDEN, 600)
    f.maths(r"\bar A=\tfrac34\hat A^{(1)}+\tfrac14\hat A^{(2)}", 1270, 130, 25, OUTPUT)
    g = grid(f, 1170, 280, 90, fw["pooled_cat"], OUTPUT, 3, 0, .5)
    go(f, (930, 250), (1170, 340), OUTPUT), go(f, (930, 490), (1170, 400), OUTPUT)
    f.words(1170, 495, "pooled cat map", 20, OUTPUT, 700)
    f.words(1170, 523, "stands for the learned query", 18, MUTED)
    f.card(40, 660, 1600, 110, "#fbfbfc", GREY, 1.2, 8)
    f.words(70, 702, "Why two prompts matter:", 21, INK, 700)
    f.words(400, 702, f"on the fused top-left patch, \"cat\" takes {m['cat_layer1'][0]:.4f} of layer 1 in its own prompt,", 21, INK)
    f.words(400, 738, f"and {m['joint_prompt_cat_layer1'][0]:.4f} when \"dog\" competes for the same patch in one joint prompt.", 21, INK)
    f.write(HERE, "The cat channel through two layers' softmaxes, each map divided by its sum, then mixed three quarters and one quarter into one pooled map. On the fused patch cat takes 0.982 in its own prompt and 0.4977 in a joint prompt.")


def mm2():
    f = Fig("mm2-the-head-reads-a-relation", 1680, 830)
    fw = K["forward_fused"]
    lp = K["lts_first_update"]
    title(f, "The head turns two maps into a relation", "Two hidden units measure how far apart the maps' centres of mass sit. Their weights are the patch positions.")
    f.words(40, 140, "fused body", 22, OUTPUT, 700)
    a = grid(f, 40, 160, 72, fw["pooled_cat"], OUTPUT, 3, 0, .5)
    b = grid(f, 40, 340, 72, fw["pooled_dog"], INPUT, 3, 0, .5)
    f.words(40, 520, "top grid: cat's map", 18, MUTED)
    f.words(40, 546, "bottom grid: dog's map", 18, MUTED)
    u = box(f, 300, 180, 300, 130, [r"$\Delta_u=\sum_p u_p(\bar A_s[p]-\bar A_o[p])$", f"= {fw['hx']:.4f}"], "hidden", 18)
    v = box(f, 300, 360, 300, 130, [r"$\Delta_v=\sum_p v_p(\bar A_s[p]-\bar A_o[p])$", f"= {fw['hy']:.4f}"], "hidden", 18)
    go(f, (184, 232), u.l, HIDDEN), go(f, (184, 412), v.l, HIDDEN)
    rel = K["relations"]
    forms = [r"\kappa\Delta_v", r"-\kappa\Delta_v", r"-\kappa\Delta_u", r"\kappa\Delta_u", "0"]
    for i, (name, form) in enumerate(zip(rel, forms)):
        yy = 140 + i * 82
        col = OUTPUT if name == "left of" else GREY
        f.rect(720, yy, 245, 66, WHITE, col, 1.6 if name == "left of" else 1.2, 8)
        f.words(735, yy + 41, name, 19, col, 700)
        f.maths(form, 890, yy + 32, 21, col)
        f.words(1000, yy + 41, f"logit {fw['logits'][i]:.1f}", 19, INK)
        f.words(1150, yy + 41, f"p = {fw['p'][i]:.1f}", 19, OUTPUT if name == "left of" else INK, 600)
    go(f, u.r, (720, 201), HIDDEN), go(f, v.r, (720, 365), HIDDEN)
    l = box(f, 1320, 230, 320, 180, ["loss for \"left of\"", r"$-\log 0.2$", f"= {fw['loss']:.4f}"], "output", 20)
    go(f, (1262, 320), l.l, OUTPUT)
    f.card(40, 590, 1600, 200, "#fbfbfc", GREY, 1.2, 8)
    f.words(70, 632, "After one push of step size 5", 22, ORANGE, 700)
    c = grid(f, 70, 648, 62, lp["pooled_cat"], OUTPUT, 3, 0, .5, 15)
    d = grid(f, 240, 648, 62, lp["pooled_dog"], INPUT, 3, 0, .5, 15)
    f.words(440, 685, f"The cat's mass has moved left and the dog's right, so Δu = {lp['hx']:.4f}.", 22, INK)
    f.words(440, 725, f"The chance of \"left of\" rises from {fw['p'][2]:.1f} to {lp['p_left']:.4f}, and the loss falls to {lp['loss']:.4f}.", 22, INK)
    f.write(HERE, "Pooled cat and dog maps feed two hidden units, the horizontal and vertical centre-of-mass gaps; these give five logits and probabilities. Identical maps give 0.2 on every class. One step of size 5 gives left of a probability of 0.7961.")


def mm3():
    f = Fig("mm3-the-gradient-back-to-the-latent", 1680, 700)
    g = K["gradient_fused"]
    title(f, "The gradient lands hardest on the patches the map has not decided",
          "The chain rule, from the loss back to each number of the latent. A finite difference agrees to better than 0.0000001.")
    steps = [("loss to hidden units", [r"\partial\mathcal L/\partial\Delta_u=" + f"{g['d_loss_d_hx_hy'][0]:.0f}", r"\partial\mathcal L/\partial\Delta_v=" + f"{g['d_loss_d_hx_hy'][1]:.0f}"]),
             ("hidden units to pooled map", [r"\partial\Delta_u/\partial\bar A_s[p]=u_p", r"\partial\Delta_u/\partial\bar A_o[p]=-u_p"]),
             ("pooled map to latent", [r"\partial A[p]/\partial x_p=k\,A[p]\,(1-A[p])", r"\text{largest where }A\approx 0.5"])]
    for i, (name, forms) in enumerate(steps):
        x = 40 + i * 545
        f.rect(x, 130, 505, 190, ORANGE_FILL, ORANGE, 1.6, 8)
        f.words(x + 252, 170, name, 21, INK, 700, "middle")
        f.maths(forms[0], x + 252, 215, 23, ORANGE)
        f.maths(forms[1], x + 252, 270, 23, ORANGE)
    for i in range(2):
        go(f, (545 + i * 545, 225), (585 + i * 545, 225), ORANGE)
    f.words(40, 380, "slope of layer 1's map", 21, HIDDEN, 700)
    f.words(40, 408, "how much the map can still move", 18, MUTED)
    a = grid(f, 40, 425, 110, g["slope_of_map_layer1"], HIDDEN, 4, 0, .25, 22)
    f.words(420, 380, "gradient on the cat channel", 21, ORANGE, 700)
    f.words(420, 408, "negative: this number will go up", 18, MUTED)
    b = grid(f, 420, 425, 110, g["cat"], ORANGE, 4, -.4, .4, 22)
    f.words(800, 380, "gradient on the dog channel", 21, ORANGE, 700)
    f.words(800, 408, "the mirror image of the cat's", 18, MUTED)
    c = grid(f, 800, 425, 110, g["dog"], ORANGE, 4, -.4, .4, 22)
    f.words(1190, 455, "The hot top row is saturated,", 22, INK, 600)
    f.words(1190, 490, f"so its slope is {g['slope_of_map_layer1'][0]:.4f}.", 22, INK)
    f.words(1190, 545, "The cool bottom row is undecided,", 22, INK, 600)
    f.words(1190, 580, f"so its slope is {g['slope_of_map_layer1'][2]:.4f}.", 22, INK)
    f.words(1190, 635, f"Gradient norm: {g['norm']:.4f}.", 22, ORANGE, 700)
    f.write(HERE, "Three chain-rule steps from the loss to the latent, then the map's slope at each patch beside the gradient on the cat and dog channels. The bottom row, where the map is undecided, takes the largest gradient.")


def mm4():
    f = Fig("mm4-sixteen-updates-against-one-capped-push", 1700, 880)
    L, P, D = K["lts_loop"], K["product_step"], K["dose"]
    title(f, "The same gradient, two doses", "The miniature's numbers. Top: Learn-to-Steer's loop on the latent. Bottom: one capped push on the product's prediction.")
    f.words(40, 145, "Chance of \"left of\" going into each of Learn-to-Steer's 16 updates, step size 5", 22, INPUT, 700)
    rows = L["rows"]
    x0, y0, w, hmax = 90, 370, 46, 180
    f.line(x0 - 10, y0 + 5, x0 + 16 * (w + 12) + 10, y0 + 5, GREY, 1.2)
    for i, q in enumerate(rows):
        h = hmax * q["p_left"]
        f.rect(x0 + i * (w + 12), y0 - h, w, h, OUTPUT_FILL, OUTPUT, 1.2)
        f.words(x0 + i * (w + 12) + w / 2, y0 + 26, str(i), 16, MUTED, 400, "middle")
        if i in (0, 1, 15):
            f.words(x0 + i * (w + 12) + w / 2, y0 - h - 8, f"{q['p_left']:.2f}", 17, OUTPUT, 700, "middle")
    f.words(x0, y0 + 54, "update number; the final chance, after update 15, is on the right", 17, MUTED)
    f.words(1090, 200, "Latent after the loop", 21, ORANGE, 700)
    grid(f, 1090, 215, 66, L["final_latent"]["cat"], INPUT, 2, -3.1, 2.6, 17)
    grid(f, 1270, 215, 66, L["final_latent"]["dog"], INPUT, 2, -3.1, 2.6, 17)
    f.words(1090, 372, "cat channel", 18, MUTED)
    f.words(1270, 372, "dog channel", 18, MUTED)
    f.words(1460, 262, "final chance", 18, MUTED)
    f.words(1460, 300, f"{L['final_p'][2]:.4f}", 30, OUTPUT, 700)
    f.line(40, 455, 1660, 455, "#d4d9df", 1.2)
    f.words(40, 500, "One push on the product's prediction, then one DDIM step", 22, OUTPUT, 700)
    a = box(f, 40, 525, 330, 150, ["raw push", r"$\lambda\,\sigma_t\,\nabla\log\hat p[r]$", f"norm {P['raw_norm']:.4f}"], "push", 19)
    b = box(f, 450, 525, 330, 150, ["cap", r"$c\,\lVert\varepsilon_{\text{PoE}}\rVert$", f"{P['cap']} × {P['eps_norm']:.4f} = {P['limit']:.4f}"], "plain", 19)
    c = box(f, 860, 525, 330, 150, ["applied", f"scaled by {P['scale_applied']:.4f}", f"norm {P['push_norm']:.4f}"], "push", 19)
    d = box(f, 1270, 525, 390, 150, ["after the DDIM step", f"chance of \"left of\": {P['pushed_after']['p_left']:.4f}", f"without a push: {P['plain_after']['p_left']:.4f}"], "output", 19)
    go(f, a.r, b.l, ORANGE), go(f, b.r, c.l, ORANGE), go(f, c.r, d.l, ORANGE)
    f.words(40, 750, f"One sampler step, change in the latent due to steering: {D['one_capped_push_then_ddim']['latent_change_norm']:.4f} with the capped push, "
                     f"{D['lts_16_passes']['latent_change_norm']:.4f} with the 16 updates.", 22, INK, 700)
    f.words(40, 790, "The two are not a fair race: the weights are hand-set and the step sizes are the methods' own. The gap shows why a dose must be tested before a head is judged.", 19, MUTED)
    f.write(HERE, "Top: the chance of left of across sixteen updates of step size 5, reaching 0.9936. Bottom: one push capped at 0.4 of the prediction's norm, then a DDIM step, reaching 0.2367.")


def mm5():
    f = Fig("mm5-three-ways-it-breaks", 1720, 760)
    L, C = K["broken_leakage"], K["broken_correlation"]
    title(f, "Three ways the checker misleads", "All three are the miniature's numbers, computed in the key.")
    w = 545
    for i, (head, col) in enumerate((("1. It reads the word", OUTPUT), ("2. A correlation loss has no pull", ORANGE), ("3. No class for fused", GREY))):
        x = 40 + i * (w + 20)
        f.card(x, 115, w, 600, "#fbfbfc", GREY, 1.2, 8)
        f.words(x + 24, 158, head, 24, col, 700)
    x = 64
    f.words(x, 200, "Two small heads learn from h (the map)", 19, INK)
    f.words(x, 226, "and τ (the trace of the prompt's word).", 19, INK)
    c, b = L["captions_only_weights"], L["both_prompts_weights"]
    t = L["test_truly_left_prompt_says_right"]
    box(f, x, 250, 240, 150, ["true captions", f"weight on h: {c['a_on_h']:.2f}", f"weight on τ: {c['b_on_tau']:.2f}"], "output", 18)
    box(f, x + 262, 250, 235, 150, ["both captions", f"weight on h: {b['a_on_h']:.2f}", f"weight on τ: {b['b_on_tau']:.2f}"], "hidden", 18)
    f.words(x, 450, "Test: truly left, prompt says \"right of\".", 20, INK, 700)
    f.words(x, 492, "chance of \"right of\":", 19, MUTED)
    f.words(x, 540, f"{t['captions_only_p_right']:.4f}", 31, OUTPUT, 700)
    f.words(x, 570, "true captions only", 18, MUTED)
    f.words(x + 262, 540, f"{t['both_prompts_p_right']:.4f}", 31, HIDDEN, 700)
    f.words(x + 262, 570, "both captions", 18, MUTED)
    f.words(x, 640, "The first head obeys the word.", 20, OUTPUT, 600)
    f.words(x, 670, "The second reads the picture.", 20, HIDDEN, 600)
    x = 40 + w + 20 + 24
    f.words(x, 200, "Identical maps have correlation 1, its", 19, INK)
    f.words(x, 226, "maximum, so the gradient is exactly 0.", 19, INK)
    box(f, x, 250, 240, 150, ["correlation", "on the fused body", f"grad norm {C['gradient_norm_fused']:.4f}"], "push", 18)
    box(f, x + 262, 250, 235, 150, ["the head", "on the fused body", f"grad norm {C['head_gradient_norm_fused']:.4f}"], "hidden", 18)
    n = C["after_nudge_of_0.1_on_cat_top_left"]
    f.words(x, 450, "Nudge the cat's top-left number by 0.1:", 20, INK, 700)
    f.words(x, 492, "correlation's gradient norm is only", 19, MUTED)
    f.words(x, 540, f"{n['gradient_norm']:.6f}", 31, ORANGE, 700)
    f.words(x, 620, "On SDXL the maps are not identical", 19, INK)
    f.words(x, 646, "(overlap 0.506 to 0.814), so this is the", 19, INK)
    f.words(x, 672, "limit case, not the cause of the 0 of 8.", 19, INK)
    x = 40 + 2 * (w + 20) + 24
    f.words(x, 200, "Identical maps give every class the", 19, INK)
    f.words(x, 226, "same probability.", 19, INK)
    for i, name in enumerate(K["relations"]):
        h = 120
        f.rect(x + 10 + i * 96, 400 - h, 70, h, FADED_FILL, GREY, 1.2)
        f.words(x + 45 + i * 96, 268, f"{K['broken_no_fused_class']['p_on_identical_maps'][i]:.1f}", 20, INK, 600, "middle")
        f.words(x + 45 + i * 96, 428, name.split()[0], 16, MUTED, 400, "middle")
    f.words(x, 490, "The head can say \"I don't know which side\".", 19, INK)
    f.words(x, 520, "It has no way to say \"these are one body\".", 19, INK)
    f.words(x, 590, "Learn-to-Steer has no fused class.", 20, OUTPUT, 700)
    f.words(x, 620, "Fusion is the failure our project", 19, INK)
    f.words(x, 646, "targets, so a head for it needs one", 19, INK)
    f.words(x, 672, "(inferred).", 19, INK)
    f.write(HERE, "Three panels: a head trained only on true captions follows the prompt's word; a correlation loss has zero gradient on identical maps while the head's is 0.7664; identical maps give 0.2 on all five classes and no class means fused.")


FIGURES = dict(mm0=mm0, mm1=mm1, mm2=mm2, mm3=mm3, mm4=mm4, mm5=mm5)

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
            print("\n".join(l[:165] for l in lines[1:12]))
    if failed:
        sys.exit(f"failed: {failed}")
