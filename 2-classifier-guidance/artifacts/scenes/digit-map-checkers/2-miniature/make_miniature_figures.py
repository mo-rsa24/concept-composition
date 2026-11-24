"""Figures mm1 to mm3 for the miniature, drawn from miniature_answer_key.json only.

Run after miniature.py: ~/.claude/venvs/figures/bin/python -I make_miniature_figures.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figure_kit import (COUNT, COUNT_FILL, FADED_FILL, GREY, HIDDEN, HIDDEN_FILL, IDENT, IDENT_FILL, INK, INPUT,  # noqa: E402
                        INPUT_FILL, ORANGE, ORANGE_FILL, OUTPUT, OUTPUT_FILL, WHITE, Fig)

K = json.loads((HERE / "miniature_answer_key.json").read_text())
CW, CH = 64, 34


def cells(f, x, y, vals, color, fill, oid, size=13, fmt="{:+.2f}"):
    """A row of number cells; returns the ids an arrow touching the row may cross."""
    ids = []
    for i, v in enumerate(vals):
        cid = f.card(x + i * CW, y, CW, CH, fill, color, 1.0, 3, f"{oid}{i}")
        ids += [cid, f.text(x + i * CW + CW / 2, y + CH / 2, fmt.format(v), size, color, ha="center", va="center", oid=f"{oid}{i}t")]
    return ids


def head_label(f, x, y, s, color, oid, sub=None):
    f.text(x, y, s, 14, color, "bold", oid=oid)
    if sub:
        f.text(x, y + 18, sub, 11, GREY, oid=oid + "s")


def mm1():
    f = Fig("mm1-the-forward-pass-with-every-number", 1720, 700)
    f.text(30, 42, "The miniature's forward pass: two canvases, every number", 24, INK, "bold", oid="ti")
    f.text(30, 72, "A 4-pixel canvas (left half | right half), asked 1 and 3, at the real t = 480. Weights are set by hand; only the noise level and time features are real.", 13, GREY, oid="sub")
    for r, name in enumerate(("apart", "wrong")):
        c = K["canvases"][name]
        y = 150 + r * 260
        shows = "1 | 3" if name == "apart" else "3 | 2"
        f.text(30, y + 20, name, 18, INPUT, "bold", oid=f"n{r}")
        f.text(30, y + 42, f"shows {shows}", 12, GREY, oid=f"n{r}s")
        head_label(f, 150, y - 14, "$x_0$", INPUT, f"h0{r}")
        a = cells(f, 150, y, c["x0"], INPUT, INPUT_FILL, f"x0{r}-", fmt="{:+.0f}")
        head_label(f, 150, y + 66, "$x_t$", INPUT, f"h1{r}")
        b = cells(f, 150, y + 80, c["xt"], INPUT, INPUT_FILL, f"xt{r}-")
        f.arrow(278, y + CH, 278, y + 80, GREY, allowed=a + b, name=f"noise{r}")
        f.text(290, y + 62, "$0.308\\,x_0 + 0.951\\,\\varepsilon$", 12, GREY, oid=f"nz{r}")
        head_label(f, 520, y - 14, "map $m_a$ (digit 1)", HIDDEN, f"h2{r}")
        ma = cells(f, 520, y, c["m_a"], HIDDEN, HIDDEN_FILL, f"ma{r}-")
        head_label(f, 520, y + 66, "map $m_b$ (digit 3)", HIDDEN, f"h3{r}")
        mb = cells(f, 520, y + 80, c["m_b"], HIDDEN, HIDDEN_FILL, f"mb{r}-")
        f.route([(406, y + 97), (460, y + 97), (460, y + 17), (520, y + 17)], GREY, 1.2, b + ma, f"x-ma{r}")
        f.arrow(406, y + 97, 520, y + 97, GREY, allowed=b + mb, name=f"x-mb{r}")
        f.text(520, y + 140, "each row adds to 4: asking for a digit changes", 11, GREY, oid=f"sum{r}")
        f.text(520, y + 156, "the guess most where it is bright and that digit has ink", 11, GREY, oid=f"sum{r}b")
        head_label(f, 890, y - 14, "hidden values h", HIDDEN, f"h4{r}", None)
        f.text(890, y + 2 + 6, "conv, SiLU, scale, shift: ch 1 | ch 2", 11, GREY, oid=f"chn{r}")
        hl = cells(f, 890, y + 18, c["h"][0], HIDDEN, WHITE, f"hl{r}-")
        hr = cells(f, 890, y + 60, c["h"][1], HIDDEN, WHITE, f"hr{r}-")
        f.text(1024, y + 40, "left half", 11, GREY, oid=f"lh{r}")
        f.text(1024, y + 82, "right half", 11, GREY, oid=f"rh{r}")
        f.route([(776, y + 17), (840, y + 17), (840, y + 35), (890, y + 35)], HIDDEN, 1.2, ma + hl, f"m-h{r}")
        f.route([(776, y + 97), (840, y + 97), (840, y + 77), (890, y + 77)], HIDDEN, 1.2, mb + hr, f"m-h2{r}")
        head_label(f, 1120, y - 14, "mean | mean | max | max", HIDDEN, f"h5{r}")
        pl = cells(f, 1120, y + 38, c["mean"] + c["max"], HIDDEN, HIDDEN_FILL, f"pl{r}-")
        f.arrow(1090, y + 55, 1120, y + 55, HIDDEN, allowed=pl, name=f"h-p{r}")
        lg = f.card(1430, y + 30, 120, 50, OUTPUT_FILL, OUTPUT, 1.2, 8, f"lg{r}")
        lgt = f.text(1490, y + 55, f"ℓ = {c['logit']:+.3f}", 14, OUTPUT, "bold", "center", "center", oid=f"lg{r}t")
        f.arrow(1376, y + 55, 1430, y + 55, HIDDEN, allowed=pl + [lg, lgt], name=f"p-l{r}")
        f.text(1490, y + 104, f"p = σ(ℓ) = {c['p']:.3f}", 13, OUTPUT, ha="center", oid=f"pp{r}")
    f.text(30, 670, "The two canvases reach logits 0.36 apart only because their noisy pixels differ. The head has not been trained; mm2 shows one training step on each branch.", 13, INK, oid="foot")
    f.write(HERE)


def mm2():
    f = Fig("mm2-one-training-step-per-branch", 1800, 760)
    f.text(30, 42, "One training step on each branch, same two canvases, same starting weights", 24, INK, "bold", oid="ti")
    f.text(30, 72, "Each dot is a canvas's logit before (hollow) and after (filled) one plain gradient step of size 0.05 on the two-canvas batch.", 13, GREY, oid="sub")
    x0, x1, lo, hi = 260, 1400, -0.25, 1.75

    def px(v):
        return x0 + (v - lo) / (hi - lo) * (x1 - x0)
    for r, (branch, col, fill) in enumerate((("count", COUNT, COUNT_FILL), ("identity", IDENT, IDENT_FILL))):
        B = K["branches"][branch]
        y = 170 + r * 270
        f.text(30, y + 10, branch.upper(), 18, col, "bold", oid=f"b{r}")
        f.text(30, y + 32, "labels: apart " + str(B["rows"]["apart"]["y"]) + ", wrong " + str(B["rows"]["wrong"]["y"]), 12, GREY, oid=f"b{r}l")
        f.text(30, y + 52, f"batch loss {B['batch_loss_before']:.3f} → {B['batch_loss_after']:.3f}", 12, INK, oid=f"b{r}m")
        axis_y = y + 110
        f.line(x0, axis_y, x1, axis_y, GREY, 1.0, name=f"ax{r}")
        for v in (0.0, 0.5, 1.0, 1.5):
            f.line(px(v), axis_y, px(v), axis_y + 6, GREY, 1.0, name=f"tk{r}{v}")
            f.text(px(v), axis_y + 24, f"{v:.1f}", 11, GREY, ha="center", oid=f"tl{r}{v}")
        f.text(x1, axis_y + 44, "logit ℓ", 12, GREY, ha="right", oid=f"axl{r}")
        for j, name in enumerate(("apart", "wrong")):
            row = B["rows"][name]
            yy = y + 20 + j * 44
            b, a = px(row["logit"]), px(row["logit_after_one_step"])
            c0 = f.circle(b, yy, 7, WHITE, col, 1.6, oid=f"c{r}{j}b")
            c1 = f.circle(a, yy, 7, col, col, 1.6, oid=f"c{r}{j}a")
            f.arrow(b + (9 if a > b else -9), yy, a + (-9 if a > b else 9), yy, col, 1.6, allowed=[c0, c1], name=f"mv{r}{j}")
            lab_x = max(a, b) + 16
            f.text(lab_x, yy + 5, f"{name}: {row['logit']:+.3f} → {row['logit_after_one_step']:+.3f}   (∂L/∂ℓ = {row['dloss_dlogit']:+.3f})",
                   12, INK, oid=f"lab{r}{j}")
        gap0 = B["rows"]["apart"]["logit"] - B["rows"]["wrong"]["logit"]
        gap1 = B["rows"]["apart"]["logit_after_one_step"] - B["rows"]["wrong"]["logit_after_one_step"]
        f.text(30, y + 76, f"apart minus wrong: {gap0:.4f} → {gap1:.4f}", 12, col, "bold", oid=f"gap{r}")
    f.box(30, 650, 1540, 84, WHITE, INK, 1.0, 8, "note", "card")
    f.text(50, 680, "Count asks for both logits up, so both move right. Identity asks for apart up and wrong down; the wrong canvas's larger error (0.587 against −0.316)",
           13, INK, oid="n1")
    f.text(50, 704, "wins the shared weights, so both logits move left and the gap between them barely changes. The head has to find a difference between the two canvases' maps to separate them.",
           13, INK, oid="n2")
    f.write(HERE)


def mm3():
    f = Fig("mm3-the-push-on-the-four-pixels", 1660, 640)
    f.text(30, 42, "The push on the four pixels", 24, INK, "bold", oid="ti")
    f.text(30, 72, "Gradient of the logit with respect to $x_t$, through the head, the maps and the frozen expert; then scaled to 0.4 of the product's length. Starting weights.", 13, GREY, oid="sub")
    for r, name in enumerate(("apart", "wrong")):
        p, c = K["push"][name], K["canvases"][name]
        y = 140 + r * 220
        f.text(30, y + 22, name, 18, INPUT, "bold", oid=f"n{r}")
        cols = [("$x_t$", c["xt"], INPUT, INPUT_FILL), ("$g = \\partial\\ell/\\partial x_t$", p["g"], ORANGE, ORANGE_FILL),
                ("$\\varepsilon_{\\rm product}$", c["eps_product"], OUTPUT, WHITE), ("push", p["push"], ORANGE, WHITE),
                ("$\\hat\\varepsilon$ = product − push", p["eps_hat"], OUTPUT, OUTPUT_FILL)]
        prev = None
        for j, (lab, vals, col, fill) in enumerate(cols):
            x = 150 + j * 300
            f.text(x, y - 8, lab, 14, col, "bold", oid=f"l{r}{j}")
            ids = cells(f, x, y + 6, vals, col, fill, f"c{r}{j}-")
            if prev and j != 2:          # the product is not computed from g
                f.arrow(prev + 4, y + 23, x - 4, y + 23, GREY, allowed=ids, name=f"a{r}{j}")
            prev = x + 4 * CW
        f.text(150, y + 70, f"‖g‖ = {p['norm_g']:.3f},  ‖ε_product‖ = {p['norm_eps_product']:.3f},  ‖push‖ / ‖ε_product‖ = {p['push_over_product']:.2f}",
               12, GREY, oid=f"s{r}")
        f.text(150, y + 92, f"A step of 0.01 along g moves ℓ from {p['logit_at_xt']:+.4f} to {p['logit_at_xt_plus_small_step_along_g']:+.4f}.",
               12, ORANGE, oid=f"s{r}b")
        f.text(150, y + 114, "The clean-canvas guess x̂₀ = (x_t − 0.951 ε̂)/0.308 moves by +3.09 × push: "
               + ", ".join(f"{v:+.2f}" for v in p["x0_hat_shift"]), 12, GREY, oid=f"s{r}c")
    f.text(30, 600, "The push uses only g's direction. On the apart canvas it brightens pixel 2 and pixel 3 and darkens pixel 1; nothing here says this draws better digits, since the head is untrained.",
           13, INK, oid="foot")
    f.write(HERE)


if __name__ == "__main__":
    for fn in (mm1, mm2, mm3):
        fn()
