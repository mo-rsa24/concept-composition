"""The moving pictures for toys C to J of triad_toys.py, one small gif each, on the toy's own numbers.

C the cap   D the random step   E a skip   F scale and shift   G mean and max   H the chain through a frozen
layer   I the size of a gradient   J a running average
Colours as the rest of the explainer: blue input, green hidden, red output, orange push, grey frozen or off.
Usage: python triad_pictures.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = Path(__file__).resolve().parent
BLUE, GREEN, RED, ORANGE, GREY, INK = "#1f5fd0", "#1d7a3e", "#c0262d", "#c86a0a", "#6b7a88", "#1f2a33"


def save(fig, upd, frames, name, fps=12):
    ani = FuncAnimation(fig, upd, frames=frames, interval=1000 / fps)
    ani.save(HERE / name, writer=PillowWriter(fps=fps), dpi=70)
    plt.close(fig)


def hold(n, k=8):
    """Frame indices that rest on the last frame for k extra frames."""
    return list(range(n)) + [n - 1] * k


def pic_c():
    fig, ax = plt.subplots(figsize=(6, 3.4)); ax.set_xlim(-0.1, 1.15); ax.set_ylim(-0.5, 0.5); ax.axis("off")
    ax.annotate("", xy=(1, 0.2), xytext=(0, 0.2), arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=3))
    ax.text(0.5, 0.28, "the product's size 1.0", color=GREEN, ha="center")
    ax.axvline(0.1, ymin=0.05, ymax=0.4, color=INK, ls=":"); ax.text(0.1, -0.42, "lid: 0.1", color=INK, ha="center")
    arr = ax.annotate("", xy=(0, -0.1), xytext=(0, -0.1), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=3))
    txt = ax.text(0.5, -0.3, "", ha="center", color=ORANGE, size=11)
    sizes = np.r_[np.linspace(0.02, 0.6, 40)]

    def upd(i):
        s = sizes[min(i, len(sizes) - 1)]; c = min(s, 0.1 * 1.0)
        arr.xy = (c, -0.1); arr.set_position((0, -0.1))
        txt.set_text(f"push before {s:.2f}  after the lid {c:.2f}")
    save(fig, upd, hold(len(sizes)), "toy-c-the-cap.gif")


def pic_d():
    T = 1000; abar = np.cumprod(1 - np.linspace(1e-4, 0.02, T)); steps = np.round(np.linspace(T - 1, 0, 50)).astype(int)
    rng = np.random.default_rng(3)
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    paths = {}
    for eta, ax in zip((0, 1), axs):
        ps = []
        for s in range(6):
            x = np.random.default_rng(s).standard_normal(); xs = [x]
            for i in range(49):
                a, ap = abar[steps[i]], abar[steps[i + 1]]
                eps = (x - np.sqrt(a) * 0.8) / np.sqrt(1 - a)            # data is the single value 0.8
                x0 = (x - np.sqrt(1 - a) * eps) / np.sqrt(a)
                sg = eta * np.sqrt((1 - ap) / (1 - a)) * np.sqrt(1 - a / ap)
                x = np.sqrt(ap) * x0 + np.sqrt(max(1 - ap - sg ** 2, 0)) * eps + sg * rng.standard_normal(); xs.append(x)
            ps.append(xs)
        paths[eta] = np.array(ps)
        ax.set_title(f"eta {eta}: " + ("same start, same path" if eta == 0 else "fresh noise each step"), size=10)
        ax.set_xlim(0, 49); ax.set_xlabel("sampler step")
    lines = {eta: [axs[k].plot([], [], color=ORANGE if eta else BLUE, lw=1.5)[0] for _ in range(6)] for k, eta in enumerate((0, 1))}
    axs[0].set_ylim(-3, 3)

    def upd(i):
        for eta in (0, 1):
            for j, ln in enumerate(lines[eta]):
                ln.set_data(range(i + 1), paths[eta][j, :i + 1])
    save(fig, upd, hold(50, 10), "toy-d-the-random-step.gif", 14)


def pic_e():
    x = np.array([0, 4, 4, 0.0]); down = x.reshape(2, 2).mean(1); up = np.repeat(down, 2)
    fig, axs = plt.subplots(1, 4, figsize=(9, 2.8))
    stages = [("the strip", x, BLUE), ("squeezed: pairs averaged", down, GREEN), ("spread back: edges lost", up, GREEN),
              ("with the copy joined: exact", x, GREEN)]
    for k, (ax, (t, v, c)) in enumerate(zip(axs, stages)):
        ax.set_title(t, size=9); ax.set_ylim(0, 4.8); ax.set_xticks([]); ax.set_yticks([])

    def upd(i):
        for k, (ax, (t, v, c)) in enumerate(zip(axs, stages)):
            ax.cla(); ax.set_title(t, size=9); ax.set_ylim(0, 4.8); ax.set_xticks([]); ax.set_yticks([])
            if i >= 3 * k:
                ax.bar(range(len(v)), v, color=c, width=0.8 if len(v) == 4 else 1.6)
                ax.text(0.5, 0.92, ("error 8" if k == 2 else "error 0" if k == 3 else ""), transform=ax.transAxes,
                        ha="center", color=RED if k == 2 else GREEN, size=11)
    save(fig, upd, hold(14, 6), "toy-e-a-skip.gif", 4)


def pic_f():
    h = np.array([2.0, -1.0]); fig, ax = plt.subplots(figsize=(5.5, 3.2)); ax.set_ylim(-3, 6)
    gs = np.r_[np.linspace(0, 0.5, 15), np.linspace(0.5, -1, 25)]; bs = np.r_[np.full(15, 0) + np.linspace(0, 1, 15), np.linspace(1, 0, 25)]

    def upd(i):
        ax.cla(); ax.set_ylim(-3, 6); ax.axhline(0, color=GREY, lw=0.6)
        g, b = gs[i], bs[i]; out = h * (1 + g) + b
        ax.bar([0, 1], h, 0.35, color=GREY, label="h"); ax.bar([0.4, 1.4], out, 0.35, color=GREEN, label="h x (1 + g) + b")
        ax.set_xticks([0.2, 1.2]); ax.set_xticklabels(["channel 1", "channel 2"]); ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax.set_title(f"the pair sets g = {g:+.2f}, b = {b:+.2f}", size=10)
    save(fig, upd, hold(len(gs), 6), "toy-f-scale-and-shift.gif")


def pic_g():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7, 3), gridspec_kw={"width_ratios": [1.2, 1]})
    vals = np.r_[np.linspace(0, 6, 30)]

    def upd(i):
        v = vals[min(i, 29)]; m = np.zeros((2, 3)); m[0, 2] = v
        a1.cla(); a1.imshow(m, cmap="Greens", vmin=0, vmax=6); a1.set_xticks([]); a1.set_yticks([]); a1.set_title("one bright cell", size=10)
        a1.text(2, 0, f"{v:.1f}", ha="center", va="center", color="white" if v > 2 else INK)
        a2.cla(); a2.bar([0, 1], [m.mean(), m.max()], color=[GREY, ORANGE]); a2.set_ylim(0, 6.5)
        a2.set_xticks([0, 1]); a2.set_xticklabels(["mean", "max"]); a2.set_title(f"mean {m.mean():.2f}   max {m.max():.2f}", size=10)
    save(fig, upd, hold(30, 8), "toy-g-mean-and-max.gif")


def pic_h():
    fig, ax = plt.subplots(figsize=(6.2, 3.6)); xs = np.linspace(0, 2, 100)
    pos = np.r_[np.linspace(0.2, 1.8, 40)]

    def upd(i):
        x = pos[min(i, 39)]; ax.cla(); ax.plot(xs, 3 * 2 * xs + 1, color=INK, lw=2, label="answer = 3 f(x) + 1,  f(x) = 2x (frozen)")
        ax.plot([x], [6 * x + 1], "o", color=ORANGE, ms=9)
        ax.plot(xs, 6 * x + 1 + 6 * (xs - x), color=ORANGE, lw=1.5, ls="--")
        ax.plot(xs, np.full_like(xs, 6 * x + 1), color=GREY, lw=1.2, ls=":")
        ax.text(0.05, 11, f"slope through the frozen layer: 3 x 2 = 6", color=ORANGE, size=10)
        ax.text(0.05, 9.3, f"slope if it were cut off (dotted): 0", color=GREY, size=10)
        ax.set_ylim(0, 14); ax.set_xlabel("x, the canvas"); ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, upd, hold(40, 8), "toy-h-through-a-frozen-layer.gif")


def pic_i():
    fig, ax = plt.subplots(figsize=(4.6, 3.6)); ax.set_xlim(-0.3, 4.6); ax.set_ylim(-0.3, 4.6); ax.set_aspect("equal")

    def upd(i):
        t = min(i, 24) / 24; ax.cla(); ax.set_xlim(-0.3, 4.6); ax.set_ylim(-0.3, 4.6); ax.set_aspect("equal")
        ax.annotate("", xy=(3 * t, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.5))
        ax.annotate("", xy=(3 * t, 4 * t), xytext=(3 * t, 0), arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.5))
        ax.annotate("", xy=(3 * t, 4 * t), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.5))
        ax.text(1.5 * t, -0.25, "pixel 1: 3", color=BLUE, ha="center", size=9); ax.text(3 * t + 0.1, 2 * t, "pixel 2: 4", color=GREEN, size=9)
        ax.set_title(f"size = sqrt(3^2 + 4^2) = 5   now {5 * t:.1f}", size=10, color=ORANGE)
        ax.axis("off")
    save(fig, upd, hold(25, 8), "toy-i-the-size-of-a-gradient.gif")


def pic_j():
    fig, ax = plt.subplots(figsize=(6, 3.4)); n = 60
    a9 = [1 - 0.9 ** k for k in range(n)]; a999 = [1 - 0.999 ** k for k in range(n)]

    def upd(i):
        i = min(i, n - 1); ax.cla(); ax.set_ylim(0, 1.05); ax.set_xlim(0, n)
        ax.plot(range(i + 1), a9[:i + 1], color=ORANGE, lw=2, label="decay 0.9: after 3 steps 0.271")
        ax.plot(range(i + 1), a999[:i + 1], color=BLUE, lw=2, label="decay 0.999: after 3 steps 0.003")
        ax.axhline(1, color=GREY, lw=0.8, ls=":"); ax.legend(frameon=False, fontsize=8, loc="center right")
        ax.set_xlabel("training steps"); ax.set_title("the averaged weight chasing a target of 1", size=10)
    save(fig, upd, hold(n, 8), "toy-j-a-running-average.gif")


if __name__ == "__main__":
    for f in (pic_c, pic_d, pic_e, pic_f, pic_g, pic_h, pic_i, pic_j):
        f(); print("drew", f.__name__)
