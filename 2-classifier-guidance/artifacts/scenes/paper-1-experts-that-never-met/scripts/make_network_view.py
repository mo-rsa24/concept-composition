"""Level 0.5: three network-view figures for what Paper 1 adds on top of the PoE ladder's network view.

  a  the region merge: each expert sees the canvas with the other half blanked, and its prediction is kept on its own half
  b  the checker reading the noisy canvas, its gradient as the push in its own orange lane, joined to the merged prediction
  c  the two settings side by side: regions where each concept owns a place, the plain product where it cannot

Clean cheat-sheet register: thin outlines, the part in question glows, the rest fades. No numbers on edges (level 2 holds
those). Real pictures: the digit canvas and expert halves from step M1's sheet, and the two-colour CLEVR renders from step
M10's smoke sheet. Noise maps and blanked halves are drawn for show and say so.

Run: ~/.claude/venvs/figures/bin/python scripts/make_network_view.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

import figure_kit as K  # noqa: E402
from figure_kit import KEY, S, ae  # noqa: E402

N = K.NUMBERS
SZ = N["sizes"]
FIG = K.SCENE / "figures"
FADE_FILL, FADE_EDGE = "#f1f2f4", "#b8c0c8"


def new(w, h):
    K.reset()
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.92)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, w); ax.set_ylim(0, h * 0.92 / 1.0); ax.axis("off")
    return fig, ax


def text(ax, x, y, t, size=9.5, color=None, **kw):
    kw.setdefault("ha", "center"); kw.setdefault("va", "center")
    ax.text(x, y, t, fontsize=size, color=color or S["ink"], family=S["font"], zorder=7, **kw)


def box(ax, x, y, w, h, ec, fc, lw=1.3, glow=False):
    if glow:
        for k, a in ((0.14, 0.10), (0.08, 0.16)):
            ax.add_patch(FancyBboxPatch((x - k, y - k), w + 2 * k, h + 2 * k, boxstyle="round,pad=0.02,rounding_size=0.12",
                                        lw=0, facecolor=ec, alpha=a, zorder=1, gid="frame"))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", lw=lw, edgecolor=ec, facecolor=fc, zorder=3))


def img(ax, a, x, y, w, edge, cmap="Greys_r", lw=1.4, h=None):
    h = h or w * a.shape[0] / a.shape[1]
    ax.imshow(a, extent=(x, x + w, y, y + h), cmap=cmap, vmin=0, vmax=1, zorder=3, interpolation="nearest")
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=edge, lw=lw, zorder=4))
    return h


def title(fig, t):
    fig.text(0.5, 0.975, t, ha="center", va="top", fontsize=13.5, weight="bold", color=S["ink"], family=S["font"])


def formula(ax, x, y, w, h, lines):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec=S["ink"], lw=0.9, zorder=4))
    step = h / (len(lines) + 1)
    for i, (t, size, color) in enumerate(lines):
        text(ax, x + w / 2, y + h - step * (i + 1), t, size, color=color)


# ---------------------------------------------------------------------------------------------------------------- a
def fig_a():
    fig, ax = new(15.0, 7.2)
    title(fig, "The region merge: each expert sees only its own half, and its prediction is kept only there")
    x0 = K.sheet_tile(2, 4)
    xt = K.noisy(x0, 0.55)
    rng = np.random.default_rng(3)
    blank = K.noisy(np.zeros_like(x0), 0.55, seed=3)                          # an empty canvas at this noise level, for show
    xa, xb = xt.copy(), xt.copy()
    xa[:, 32:] = blank[:, 32:]; xb[:, :32] = blank[:, :32]
    noise_map = np.clip(0.5 + 1.2 * (xt - 0.6 * x0 - 0.2), 0, 1)              # the noise each would remove, for show

    img(ax, xt, 0.3, 2.65, 2.4, S["cin"])
    text(ax, 1.5, 4.1, "$x_t$, the noisy canvas", 10, weight="bold")
    text(ax, 1.5, 2.35, "1 x 32 x 64", 8.6)

    # the two blanked copies
    for y, a, lab, who in ((4.15, xa, r"$\tilde x^A_t$: right half blanked", "A"), (0.95, xb, r"$\tilde x^B_t$: left half blanked", "B")):
        img(ax, a, 3.6, y, 2.2, S["cin"])
        side = 4.7 + (0.55 if who == "A" else -0.55)
        text(ax, 4.7, y + 1.32, lab, 9.2)
    K.line(ax, [(2.74, 3.25), (3.2, 3.25), (3.2, 4.7), (3.56, 4.7)], S["cin"], name="x_t to A's copy")
    K.line(ax, [(2.74, 3.25), (3.2, 3.25), (3.2, 1.5), (3.56, 1.5)], S["cin"], name="x_t to B's copy")
    text(ax, 4.7, 3.25, "the blanked half is an empty\ncanvas noised to the same level", 8.2, style="italic", color=S["fixed"])

    # the experts
    for y, name in ((4.2, "Expert A"), (1.0, "Expert B")):
        box(ax, 6.4, y, 1.7, 1.0, S["chid"], S["fhid"])
        text(ax, 7.25, y + 0.62, name, 10.5, weight="bold")
        text(ax, 7.25, y + 0.3, "finished U-Net", 8.6)
        K.line(ax, [(5.84, y + 0.5), (6.36, y + 0.5)], S["edge"], name=f"copy into {name}")
        K.line(ax, [(8.14, y + 0.5), (8.66, y + 0.5)], S["chid"], name=f"{name} out")

    # each prediction, the kept half glowing and the dropped half faded
    for y, keep_left, lab in ((4.15, True, r"$\epsilon^A_t(\tilde x^A_t)$"), (0.95, False, r"$\epsilon^B_t(\tilde x^B_t)$")):
        img(ax, noise_map, 8.7, y, 2.2, S["chid"], cmap="Greens")
        ax.add_patch(Rectangle((8.7 + (1.1 if keep_left else 0), y), 1.1, 1.1, facecolor="white", alpha=0.78, lw=0, zorder=5))
        text(ax, 9.8, y + 1.32, lab + (", keep the left half $R_A$" if keep_left else ", keep the right half $R_B$"), 9.2)
    text(ax, 9.8, 3.25, "the faded half is thrown away", 8.2, style="italic", color=S["fixed"])

    # joined
    left = np.zeros_like(noise_map); left[:, :32] = noise_map[:, :32]; left[:, 32:] = noise_map[:, 32:]
    img(ax, left, 12.0, 2.65, 2.4, S["chid"], cmap="Greens")
    ax.plot([13.2, 13.2], [2.65, 3.85], color=S["chid"], lw=1.2, ls=(0, (3, 2)), zorder=5)
    text(ax, 12.6, 2.45, "$R_A$", 9.5); text(ax, 13.8, 2.45, "$R_B$", 9.5)
    text(ax, 13.2, 4.1, r"merged $\hat\epsilon_t$", 10, weight="bold")
    K.line(ax, [(10.94, 4.7), (11.45, 4.7), (11.45, 3.5), (11.96, 3.5)], S["chid"], name="A's half to the merge")
    K.line(ax, [(10.94, 1.5), (11.45, 1.5), (11.45, 3.0), (11.96, 3.0)], S["chid"], name="B's half to the merge")

    formula(ax, 6.2, 5.75, 8.6, 0.62, [(r"$\hat\epsilon_t = m\odot\epsilon^A_t(\tilde x^A_t) + (1-m)\odot\epsilon^B_t(\tilde x^B_t)$,"
                                              r"   $\tilde x^A_t = m\odot x_t + (1-m)\odot \mathrm{blank}_t$", 11.5, S["ink"])])
    text(ax, 3.3, 6.06, "$m$ is 1 on the left half, 0 on the right", 9.0)
    lock = int(SZ["layout_lock_t"]["value"])
    text(ax, 7.5, 0.35, f"which expert takes the left half is voted from the experts' own predictions early, then held from t = {lock};"
                        " the noise maps are drawn for show", 8.4, color=S["fixed"])
    K.save(fig, str(FIG / "level05-the-region-merge"))


# ---------------------------------------------------------------------------------------------------------------- b
def fig_b():
    fig, ax = new(15.0, 7.0)
    title(fig, "The checker reads the noisy canvas, and its gradient comes back as the push in its own lane")
    x0 = K.sheet_tile(2, 4); xt = K.noisy(x0, 0.55)
    # forward lane, top
    img(ax, xt, 0.3, 4.1, 2.4, S["cin"])
    text(ax, 1.5, 5.55, "$x_t$", 11, weight="bold")
    # the merge, faded: it is figure a
    box(ax, 3.4, 5.25, 2.6, 0.9, FADE_EDGE, FADE_FILL)
    text(ax, 4.7, 5.7, "the region merge (figure a)\ngives the merged $\\hat\\epsilon_t$", 9, color=S["fixed"])
    K.line(ax, [(2.74, 4.95), (3.05, 4.95), (3.05, 5.7), (3.36, 5.7)], FADE_EDGE, name="x_t to the merge")
    # the two checkers, glowing
    box(ax, 3.4, 3.35, 2.6, 1.2, S["chid"], S["fhid"], glow=True)
    text(ax, 4.7, 4.2, "pair checker", 10.5, weight="bold")
    text(ax, 4.7, 3.82, "told the step and the asked pair;\none logit: both there, apart", 8.4)
    box(ax, 3.4, 1.85, 2.6, 1.2, S["chid"], S["fhid"], glow=True)
    text(ax, 4.7, 2.7, "separate-digit checker", 10.5, weight="bold")
    text(ax, 4.7, 2.32, "told the step; ten logits,\none per digit: is it there?", 8.4)
    K.line(ax, [(2.74, 4.95), (3.05, 4.95), (3.05, 3.95), (3.36, 3.95)], S["cin"], name="x_t to the pair checker")
    K.line(ax, [(2.74, 4.95), (3.05, 4.95), (3.05, 2.45), (3.36, 2.45)], S["cin"], name="x_t to the separate checker")
    # what each answer becomes
    box(ax, 6.7, 3.45, 2.1, 1.0, S["cout"], "#fbe3e3")
    text(ax, 7.75, 3.95, r"$z_{\mathrm{pair}}$", 12)
    box(ax, 6.7, 1.95, 2.1, 1.0, S["cout"], "#fbe3e3")
    text(ax, 7.75, 2.45, r"$\log\sigma(z_a) + \log\sigma(z_b)$", 10.5)
    K.line(ax, [(6.04, 3.95), (6.66, 3.95)], S["chid"], name="pair logit")
    K.line(ax, [(6.04, 2.45), (6.66, 2.45)], S["chid"], name="separate logits")
    box(ax, 9.4, 2.7, 1.9, 1.0, S["cout"], "#fbe3e3")
    text(ax, 10.35, 3.2, "sum: how sure\nboth are there", 9)
    K.line(ax, [(8.84, 3.95), (9.1, 3.95), (9.1, 3.35), (9.36, 3.35)], S["cout"], name="pair into the sum")
    K.line(ax, [(8.84, 2.45), (9.1, 2.45), (9.1, 3.05), (9.36, 3.05)], S["cout"], name="separate into the sum")

    # the push lane, orange, below everything with clear space
    yl = 1.0
    K.line(ax, [(10.35, 2.66), (10.35, yl), (1.5, yl), (1.5, 4.06)], S["accent"], lw=2.2, ms=14, name="gradient back to x_t")
    text(ax, 6.0, yl + 0.27, r"backpropagate through both checkers to the pixels of $x_t$: $\nabla_{x_t}$ (sum)", 9.4, color=S["accent"], weight="bold")

    # scaling and joining, right
    box(ax, 12.0, 2.55, 2.75, 1.3, S["accent"], "#fdeedd", glow=True)
    text(ax, 13.37, 3.42, r"push $= -\lambda\,\sigma_t\,\nabla_{x_t}(\cdot)$", 11)
    st, cap = SZ["push_strength"]["value"], SZ["push_cap"]["value"]
    text(ax, 13.37, 2.95, f"cut to at most {cap:g} of " + r"$\|\hat\epsilon_t\|$", 9)
    K.line(ax, [(0.75, 4.06), (0.75, 0.4), (13.37, 0.4), (13.37, 2.5)], S["accent"], lw=2.2, ms=14, name="gradient to the scaling")
    text(ax, 6.0, 0.64, "the gradient has the canvas's shape, one number per pixel, and goes to the push", 9.0, color=S["accent"])
    box(ax, 12.0, 4.85, 2.75, 1.0, S["ink"], "white")
    text(ax, 13.37, 5.35, r"$\hat\epsilon_t$ + push", 12, weight="bold")
    K.line(ax, [(6.04, 5.7), (11.96, 5.7)], FADE_EDGE, name="merge to the sum")
    K.line(ax, [(13.37, 3.9), (13.37, 4.8)], S["accent"], lw=2.2, name="push into the prediction")
    text(ax, 13.37, 6.35, f"{int(SZ['sampler_steps']['value'])} steps, " + r"$\lambda$" + f" = {st:g} in step M1", 9)
    text(ax, 7.6, 6.2, "the forward path is the top lane; the push is computed in the bottom lane and only joins at the end",
         8.6, color=S["fixed"], style="italic")
    K.save(fig, str(FIG / "level05-the-checker-push-lane"))


# ---------------------------------------------------------------------------------------------------------------- c
def fig_c():
    fig, ax = new(16.0, 7.6)
    title(fig, "Where regions apply and where they cannot: the push lands on the merge, or on the plain product")
    R = N["results"]
    # left side
    box(ax, 0.25, 0.35, 7.55, 6.1, S["chid"], "#f7fbf8")
    text(ax, 4.02, 6.05, "Each concept owns a place: merge by region", 12, weight="bold", color=S["chid"])
    x0 = K.sheet_tile(2, 4)
    img(ax, x0, 0.6, 3.6, 2.8, S["cout"])
    ax.add_patch(Rectangle((0.6, 3.6), 1.4, 1.4, facecolor=KEY["experts"], alpha=0.16, lw=0, zorder=5))
    text(ax, 1.3, 3.35, "$R_A$", 10); text(ax, 2.7, 3.35, "$R_B$", 10)
    text(ax, 2.0, 5.35, "digits on halves", 10.5, weight="bold")
    d = R["digits_halves_vs_paste"]
    text(ax, 5.55, 4.55, f"clean pairs of 320:\npaste {d['paste']}, method {d['method_checker_push']}\n"
                         f"lead {d['lead_push_over_paste']}, bar {d['bar']}: null", 9.2, ha="center")
    text(ax, 2.0, 2.65, "CLEVR, one object per fixed place", 10.5, weight="bold")
    c = R["clevr_fixed_places_paste"]
    text(ax, 2.0, 2.05, f"paste alone puts every object in place:\n{c['two_experts'][0]:,} of {c['two_experts'][1]:,} with two experts", 9.2)
    formula(ax, 4.1, 1.0, 3.45, 1.35, [(r"$\hat\epsilon = \mathrm{merge}(\epsilon^A|_{R_A},\epsilon^B|_{R_B})$", 10.5, S["ink"]),
                                        (r"$-\ \lambda\,\sigma_t\,\nabla\log C_t$", 10.5, S["accent"])])
    text(ax, 4.02, 0.6, "with the push off, fixed regions amount to two samples pasted side by side", 8.6, style="italic", color=S["fixed"])

    # right side
    box(ax, 8.2, 0.35, 7.55, 6.1, S["accent"], "#fffaf4")
    text(ax, 11.97, 6.05, "No place to own: the plain product, plus the push", 12, weight="bold", color=S["accent"])
    w = 1.35
    for i, (row, col, lab) in enumerate(((1, 0, "red expert's data"), (2, 0, "blue expert's data"), (3, 1, "asked: 2 red, 1 blue"))):
        h = img(ax, K.clevr_tile(row, col), 8.5 + i * (w + 0.22), 3.85, w, KEY["inputs"] if i < 2 else KEY["outputs"], cmap=None)
        text(ax, 8.5 + i * (w + 0.22) + w / 2, 3.6, lab, 8.2)
    text(ax, 10.45, 5.35, "two-colour CLEVR, told what and not where", 10.5, weight="bold")
    text(ax, 10.45, 3.12, "real renders; the right one has both colours and\nonly the scorer sees such scenes. No expert has trained yet.", 8.4, style="italic")
    # Colored MNIST: drawn for show
    dig = K.sheet_tile(0, 4)[:, :32]
    rgb = np.stack([dig * 0.95, dig * 0.35, dig * 0.15], -1)
    img(ax, rgb, 13.7, 3.85, 1.3, KEY["inputs"], cmap=None)
    text(ax, 14.35, 5.35, "Colored MNIST", 10.5, weight="bold")
    text(ax, 14.35, 3.32, "digit and colour on one\nobject: drawn for show,\nnot built here", 8.4, style="italic")
    formula(ax, 8.55, 1.0, 4.3, 1.35, [(r"$\hat\epsilon = \epsilon_\varnothing + \sum_i(\epsilon_{y_i} - \epsilon_\varnothing)$", 10.5, S["ink"]),
                                        (r"$-\ \lambda\,\sigma_t\,\nabla\log C_t$", 10.5, S["accent"])])
    text(ax, 14.3, 1.68, "the same push,\non the plain\nproduct", 9.2, color=S["accent"])
    text(ax, 11.97, 0.6, "Colored MNIST is DCFM's setting; on two-colour CLEVR a mask paste is still the control to beat", 8.6, style="italic", color=S["fixed"])
    K.save(fig, str(FIG / "level05-regions-or-plain-product"))


if __name__ == "__main__":
    fig_a(); fig_b(); fig_c()
