"""Level 0: Paper 1's method as one annotated overview, one real digit canvas followed from input to output.

The canvas is the 'method with the checker push' tile for digits 4 and 7, seed 0, in step M1's sheet
(artifacts/notes/progress/method-search/01-beat-the-baseline/paper-1-four-rows-first-two-seeds.png, third row, fifth tile),
from the run W&B vekm3sfh. The noise on the input copy is drawn for show. Each expert's strip of samples is cut from the
same sheet's paste row, which holds each expert's own samples drawn alone. Sizes and counts come from data/paper1_numbers.json.

Run: ~/.claude/venvs/figures/bin/python scripts/make_overview.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

import figure_kit as K  # noqa: E402
from figure_kit import S, ae  # noqa: E402

N = K.NUMBERS
SZ, RS = N["sizes"], N["results"]["digits_halves_vs_paste"]
OUT = K.SCENE / "figures" / "level0-paper-1-overview"
GREY_FILL = "#e8ebee"


def text(ax, x, y, t, size=9.0, color=None, **kw):
    kw.setdefault("ha", "center"); kw.setdefault("va", "center")
    ax.text(x, y, t, fontsize=size, color=color or S["ink"], family=S["font"], zorder=6, **kw)


def canvas(ax, img, x, y, w, edge, lw=1.8):
    ax.imshow(img, extent=(x, x + w, y, y + w / 2), cmap="Greys_r", vmin=0, vmax=1, zorder=3, interpolation="nearest")
    ax.add_patch(Rectangle((x, y), w, w / 2, fill=False, ec=edge, lw=lw, zorder=4))


def sample_strip(ax, halves, x, y, w):
    """Five 32 x 32 halves side by side: one expert's own samples."""
    cell = w / len(halves)
    for i, h in enumerate(halves):
        ax.imshow(h, extent=(x + i * cell + 0.02, x + (i + 1) * cell - 0.02, y, y + cell - 0.04), cmap="Greys_r", vmin=0, vmax=1,
                  zorder=5, interpolation="nearest")


def main():
    K.reset()
    fig, ax = plt.subplots(figsize=(16.6, 11.0))
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.955)
    fig.patch.set_facecolor(S["bg"]); ax.set_facecolor(S["bg"])
    ax.set_xlim(-0.05, 16.6); ax.set_ylim(0.05, 10.5); ax.axis("off")

    for x0, x1, name in ((0.2, 2.9, "What goes in"), (3.2, 6.0, "Experts trained apart"), (6.3, 8.95, "Merge by region"),
                         (9.25, 11.85, "Checker reads $x_t$"), (12.1, 16.2, "One sampler step")):
        ae.bracket(ax, x0, x1, 9.95, name, S)

    # ---- what goes in: the real canvas with noise drawn for show
    out_img = K.sheet_tile(2, 4)                                     # method with the checker push, digits 4 and 7, seed 0
    canvas(ax, K.noisy(out_img, 0.55), 0.2, 5.3, 2.6, S["cin"])
    text(ax, 1.5, 7.05, "half-drawn canvas $x_t$", 10.5, weight="bold")
    text(ax, 1.5, 6.72, f"{SZ['digit_canvas']['value'][1]} x {SZ['digit_canvas']['value'][2]} pixels, grey", 8.8)
    text(ax, 1.3, 4.9, "the run's own finished 4 and 7,\nwith noise added for show", 8.4, style="italic")
    win = (1.98, 5.42, 0.62, 1.06)                                   # around the 7
    ax.add_patch(Rectangle(win[:2], win[2], win[3], fill=False, ec=S["cout"], lw=1.6, zorder=6))

    # ---- two experts that never met, each with the holder of its data
    paste_row = [K.sheet_tile(0, c) for c in (0, 2, 4, 6, 8)]       # tiles '1 9', '3 6', '4 7', '2 8', '0 5': A's digit left
    halves_a = [t[:, :32] for t in paste_row]
    halves_b = [t[:, 32:] for t in paste_row]
    ae.card(ax, 3.3, 8.75, 2.6, 0.72, "Holder A\nMNIST digits 0 to 4 only", S, kind="in", size=9.4)
    ae.card(ax, 3.3, 2.75, 2.6, 0.72, "Holder B\nMNIST digits 5 to 9 only", S, kind="in", size=9.4)
    for (top, bottom), label_y in (((8.73, 8.33), 8.53), ((3.49, 4.03), 3.76)):
        K.line(ax, [(3.75, top), (3.75, bottom)], S["fixed"], lw=1.3, ls=(0, (4, 3)), name="trains expert")
        text(ax, 4.75, label_y, "trains once, then frozen", 8.2, color=S["fixed"], style="italic")
    for y0, name, who, halves in ((6.95, "Expert A", "a canvas U-Net, 0 to 4", halves_a), (4.35, "Expert B", "a canvas U-Net, 5 to 9", halves_b)):
        ax.add_patch(FancyBboxPatch((3.3, y0), 2.6, 1.35, boxstyle="round,pad=0.02,rounding_size=0.09", lw=1.6,
                                    edgecolor=S["chid"], facecolor=S["fhid"], zorder=3))
        text(ax, 4.6, y0 + 1.12, name, 10.5, weight="bold")
        text(ax, 4.6, y0 + 0.84, who, 8.6)
        sample_strip(ax, halves, 3.45, y0 + 0.1, 2.3)
    text(ax, 4.6, 6.62, "each box shows its own samples", 7.8, style="italic")
    text(ax, 4.6, 6.33, "no picture holds digits from both", 8.4, weight="bold", color=S["cout"])
    K.line(ax, [(2.84, 5.95), (3.08, 5.95), (3.08, 7.62), (3.28, 7.62)], S["cin"], lw=1.5, name="x_t to expert A")
    K.line(ax, [(2.84, 5.95), (3.08, 5.95), (3.08, 5.02), (3.28, 5.02)], S["cin"], lw=1.5, name="x_t to expert B")

    # ---- merge by region
    fx, fy, fw, fh = 6.5, 5.6, 2.3, 1.15
    ax.add_patch(Rectangle((fx, fy), fw / 2, fh, facecolor="#cfe8d6", edgecolor=S["chid"], lw=1.4, zorder=3))
    ax.add_patch(Rectangle((fx + fw / 2, fy), fw / 2, fh, facecolor="#e6f3e9", edgecolor=S["chid"], lw=1.4, zorder=3))
    text(ax, fx + fw / 4, fy + 0.72, r"$\epsilon^A_t$", 13)
    text(ax, fx + fw / 4, fy + 0.3, "kept on $R_A$", 8.6)
    text(ax, fx + 3 * fw / 4, fy + 0.72, r"$\epsilon^B_t$", 13)
    text(ax, fx + 3 * fw / 4, fy + 0.3, "kept on $R_B$", 8.6)
    K.line(ax, [(5.92, 7.62), (7.07, 7.62), (7.07, fy + fh + 0.02)], S["chid"], lw=1.6, name="expert A's prediction to the merge")
    K.line(ax, [(5.92, 5.02), (8.22, 5.02), (8.22, fy - 0.02)], S["chid"], lw=1.6, name="expert B's prediction to the merge")
    text(ax, 6.4, 7.86, "its noise prediction", 8.2, ha="left")
    text(ax, 6.4, 4.78, "its noise prediction", 8.2, ha="left")
    text(ax, 7.65, 8.55, "each expert also sees the\nother half blanked", 8.0, style="italic")
    lock = int(SZ["layout_lock_t"]["value"])
    text(ax, 7.65, 4.05, f"which half each expert takes is\nvoted early and held from t = {lock}", 8.2)

    # ---- the checker
    ae.trapezoid(ax, 9.4, 11.6, 6.1, 2.3, 1.25, S["chid"], "#bfe5ca",
                 [("Checker $C_t$", "bold"), ("a small CNN, told the step", "normal"), ("trained on pastes of the", "normal"),
                  ("experts' own samples", "normal")], S)
    K.line(ax, [(2.84, 5.95), (3.08, 5.95), (3.08, 2.35), (9.15, 2.35), (9.15, 5.55), (9.38, 5.55)], S["cin"], lw=1.5, name="x_t to the checker")
    text(ax, 5.3, 2.12, "the same $x_t$ goes to the checker", 8.4, color=S["cin"])

    # the push lane, orange
    ae.card(ax, 9.3, 3.15, 2.2, 1.05, "push\n" + r"$-\lambda_t\,\sigma_t\,\nabla_{x_t}\log C_t$", S, size=10.5, ec=S["accent"], fc="#fdeedd")
    K.line(ax, [(10.4, 5.42), (10.4, 4.24)], S["accent"], lw=2.2, name="checker gradient to the push")
    text(ax, 10.52, 4.83, "gradient taken\nat $x_t$", 8.2, color=S["accent"], ha="left", weight="bold")

    # merge result and push into the composed prediction
    K.line(ax, [(fx + fw + 0.02, 6.45), (9.0, 6.45), (9.0, 9.0), (12.08, 9.0)], S["chid"], lw=1.6, name="merge to the composed prediction")
    K.line(ax, [(11.52, 3.67), (11.95, 3.67), (11.95, 8.62), (12.08, 8.62)], S["accent"], lw=2.4, ms=14, name="push to the composed prediction")
    ae.card(ax, 12.1, 8.35, 4.0, 0.9, r"composed prediction $\hat\epsilon_t$" + "\nthe merge, plus the push", S, size=10.0, ec=S["ink"], fc="white")

    # the sampler step box
    ax.add_patch(FancyBboxPatch((12.1, 4.3), 4.0, 3.45, boxstyle="round,pad=0.02,rounding_size=0.1", fc="white", ec=S["ink"], lw=1.0, zorder=2))
    K.line(ax, [(14.1, 8.32), (14.1, 7.8)], S["ink"], lw=1.2, name="prediction into the step")
    text(ax, 14.1, 7.42, "One sampler step", 11.5, weight="bold")
    text(ax, 14.1, 6.78, r"$\hat\epsilon_t = \mathrm{merge}(\epsilon^A_t|_{R_A},\ \epsilon^B_t|_{R_B})$", 13.5)
    text(ax, 14.1, 6.18, r"$-\ \lambda_t\,\sigma_t\,\nabla_{x_t}\log C_t(\mathrm{both}\mid x_t)$", 13.5, color=S["accent"])
    text(ax, 14.1, 5.55, r"$x_{t-1}$ = DDIM step from $x_t$ with $\hat\epsilon_t$", 10.5)
    st, cap = SZ["push_strength"]["value"], SZ["push_cap"]["value"]
    text(ax, 14.1, 4.82, f"in the run: {int(SZ['sampler_steps']['value'])} steps, " + r"$\lambda$" + f" = {st:g},\n"
         f"each push cut to at most {cap:g} of " + r"$\|\hat\epsilon_t\|$", 9.0)

    # the output, real
    K.line(ax, [(14.1, 4.28), (14.1, 3.62)], S["ink"], lw=1.2, name="step to the output")
    canvas(ax, out_img, 12.8, 2.25, 2.6, S["cout"], lw=2.0)
    ax.add_patch(Rectangle((12.8 + (win[0] - 0.2), 2.25 + (win[1] - 5.3)), win[2], win[3], fill=False, ec=S["cout"], lw=1.3, ls=":", zorder=6))
    text(ax, 14.1, 1.92, f"$x_0$ after {int(SZ['sampler_steps']['value'])} steps: a 4 and a 7 apart", 9.4, weight="bold")
    text(ax, 14.1, 1.6, "real output, step M1, W&B vekm3sfh", 8.2, style="italic")
    ae.card(ax, 12.1, 0.2, 4.0, 1.12,
            f"on 5 tuning pairs x 64 seeds, clean pairs of 320:\npaste {RS['paste']}, this method {RS['method_checker_push']}, "
            f"no push {RS['method_no_push']}\nlead {RS['lead_push_over_paste']} under its bar {RS['bar']}: a null", S, size=9.0, ec=S["accent"], fc="#fff6ec")

    # ---- inset: inside the checker
    ix, iy, iw, ih = 3.3, 0.25, 5.65, 1.6
    ax.add_patch(FancyBboxPatch((ix, iy), iw, ih, boxstyle="round,pad=0.02,rounding_size=0.1", lw=1.5, edgecolor=S["chid"],
                                facecolor="#f4faf5", zorder=1, gid="frame"))
    K.line(ax, [(9.4, 4.95), (ix + iw, iy + ih)], S["chid"], lw=1.0, ls=":", arrow=False, name="inset connector")
    text(ax, ix + 0.12, iy + ih - 0.17, "Inside the pair checker", 9.3, ha="left", weight="bold", color=S["chid"])
    gx, gy, gw = ix + 0.2, iy + 0.28, 1.5
    ax.imshow(K.noisy(out_img, 0.55), extent=(gx, gx + gw, gy, gy + gw / 2), cmap="Greys_r", vmin=0, vmax=1, zorder=3, interpolation="nearest")
    for k in range(1, 16):
        ax.plot([gx + k * gw / 16] * 2, [gy, gy + gw / 2], color="#b9d8c2", lw=0.4, zorder=4)
    for k in range(1, 8):
        ax.plot([gx, gx + gw], [gy + k * gw / 16] * 2, color="#b9d8c2", lw=0.4, zorder=4)
    ax.add_patch(Rectangle((gx + 12 * gw / 16, gy + 3 * gw / 16), gw / 16, gw / 16, facecolor=S["cout"], zorder=5))
    text(ax, gx + gw / 2, gy - 0.12, "32 x 64, then 8 x 16", 7.0)
    K.line(ax, [(gx + gw + 0.06, gy + 0.38), (gx + gw + 0.34, gy + 0.38)], S["chid"], ms=8, name="inset 1")
    for k in range(5):
        ax.add_patch(Rectangle((5.35 + k * 0.17, gy + 0.03), 0.12, 0.72, facecolor=S["chid"], alpha=0.35 + k * 0.12, lw=0, zorder=3))
    text(ax, 5.73, gy + 0.93, "5 convolutions", 7.0)
    text(ax, 5.73, gy - 0.12, "64, 64, 128, 128, 128", 7.0)
    K.line(ax, [(6.24, gy + 0.38), (6.5, gy + 0.38)], S["chid"], ms=8, name="inset 2")
    text(ax, 7.08, gy + 0.55, "step scales and\nshifts the maps", 7.0)
    text(ax, 7.08, gy + 0.1, "mean, max: 256", 7.0)
    K.line(ax, [(7.66, gy + 0.38), (7.92, gy + 0.38)], S["chid"], ms=8, name="inset 3")
    ax.add_patch(FancyBboxPatch((8.08, gy + 0.24), 0.28, 0.28, boxstyle="circle,pad=0.0", fc="#fbe3e3", ec=S["cout"], lw=1.3, zorder=5))
    text(ax, 8.55, gy + 0.85, "1 answer:\nboth, apart", 7.0)
    text(ax, ix + iw / 2 + 0.6, iy + ih - 0.17, f"{SZ['pair_checker_weights']['value']:,} trained weights", 7.6, color=S["chid"])
    K.line(ax, [(win[0] + 0.57, win[1] - 0.01), (win[0] + 0.57, 1.05), (gx - 0.01, 1.05)], S["cout"], lw=1.0, arrow=False, name="trace of the red window")

    fig.text(0.5, 0.985, "Paper 1: two experts that never met, merged by region and pushed by a checker trained on their own samples",
             ha="center", va="top", fontsize=14.5, weight="bold", color=S["ink"], family=S["font"])
    K.save(fig, str(OUT))


if __name__ == "__main__":
    main()
