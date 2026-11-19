"""The toys behind the 🧸 and 🔨 lines of the five explainer pages: each hand answer is computed and asserted.

Every toy is ordinary (soup, light bulbs, a four-pixel strip), small enough to do on paper, and its numbers are
the ones the page quotes. The build is this file: it must reproduce each hand answer, and it writes
toy_numbers.json and two moving pictures.

Toys: A sigmoid and softmax   B why the log-odds keeps pushing   C the cap   D the random sampler step
      E a skip   F scale-and-shift   G mean and max   H a frozen layer the gradient still passes   I the size of a
      gradient   J a running average
Usage: python triad_toys.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = Path(__file__).resolve().parent
sig = lambda z: 1 / (1 + np.exp(-np.asarray(z, float)))
out = {}
close = lambda a, b, tol=1e-3: abs(a - b) < tol

# A. Three tasters each give the soup a raw saltiness score z (any real number). A sigmoid turns one score into a chance.
out["A"] = {"z": [-2, 0, 2, 3], "sigmoid": [round(float(sig(z)), 4) for z in (-2, 0, 2, 3)]}
assert close(sig(0), 0.5) and close(sig(2), 0.8808) and close(sig(-2), 0.1192) and close(sig(3), 0.9526)
# the same two scores through a softmax share one total of 1
two = np.array([3.0, 3.0])
sm = np.exp(two) / np.exp(two).sum()
out["A"] |= {"two_scores": [3, 3], "softmax": sm.round(4).tolist(), "sigmoid_pair": sig(two).round(4).tolist(),
             "sigmoid_pair_sum": round(float(sig(two).sum()), 4)}
assert close(sm[0], 0.5) and close(sig(two).sum(), 1.9052)

# B. A bulb is lit with chance d = sigma(l). Push l up by a little: how fast does log d grow, and how fast does log(d/(1-d))?
rows = []
for d in (0.5, 0.9, 0.99):
    l = np.log(d / (1 - d))
    slope_logd = 1 - d                         # d/dl log sigma(l) = 1 - sigma(l)
    h = 1e-6
    num = (np.log(sig(l + h)) - np.log(sig(l))) / h
    assert close(num, slope_logd, 1e-4)
    rows.append({"d": d, "logit": round(float(l), 3), "slope_of_log_d": round(slope_logd, 4), "slope_of_log_odds": 1.0})
out["B"] = rows

# C. The cap: a push is scaled so its size is at most 0.1 of the product's size.
cap = lambda push, prod, c=0.1: push * min(1.0, c * np.linalg.norm(prod) / max(np.linalg.norm(push), 1e-12))
prod = np.array([0.6, 0.8])                                   # size 1.0
big, small = np.array([0.24, 0.32]), np.array([0.03, 0.04])   # sizes 0.4 and 0.05
out["C"] = {"product_size": 1.0, "push_size_before": [0.4, 0.05],
            "push_size_after": [round(float(np.linalg.norm(cap(big, prod))), 3), round(float(np.linalg.norm(cap(small, prod))), 3)]}
assert close(np.linalg.norm(cap(big, prod)), 0.1) and close(np.linalg.norm(cap(small, prod)), 0.05)

# D. The random sampler step: the real 50-step schedule, step 25 to 26 (t = 489 to 464).
T = 1000
abar = np.cumprod(1 - np.linspace(1e-4, 0.02, T))
steps = np.round(np.linspace(T - 1, 0, 50)).astype(int)
a, ap = abar[steps[25]], abar[steps[26]]
sigma = lambda eta: eta * np.sqrt((1 - ap) / (1 - a)) * np.sqrt(1 - a / ap)
out["D"] = {"t": int(steps[25]), "t_next": int(steps[26]), "abar": round(float(a), 4), "abar_next": round(float(ap), 4),
            "fresh_noise_size_eta0": round(float(sigma(0)), 4), "fresh_noise_size_eta1": round(float(sigma(1)), 4),
            "fresh_noise_size_eta05": round(float(sigma(0.5)), 4)}
assert sigma(0) == 0 and 0 < sigma(1) ** 2 <= 1 - ap          # fresh noise must fit inside the step's noise budget

# E. A skip: a four-pixel strip [0,4,4,0]; the encoder averages pairs (size 4 > 2), the decoder repeats them back.
x = np.array([0, 4, 4, 0.0])
down = x.reshape(2, 2).mean(1)
up = np.repeat(down, 2)
out["E"] = {"strip": x.tolist(), "after_down": down.tolist(), "after_up": up.tolist(),
            "error_without_skip": float(np.abs(x - up).sum()), "error_with_skip": 0.0}
assert out["E"]["error_without_skip"] == 8.0

# F. Scale and shift: h * (1 + g) + b, with g and b made from the asked pair.
h = np.array([2.0, -1.0])
film = lambda g, b: h * (1 + g) + b
out["F"] = {"h": h.tolist(), "pair_A_g0.5_b1": film(0.5, 1).tolist(), "pair_B_g-1_b0": film(-1, 0).tolist(), "no_pair": h.tolist()}
assert film(0.5, 1).tolist() == [4.0, -0.5] and film(-1, 0).tolist() == [0.0, -0.0]

# G. Mean and max over a 2 x 3 map with one bright cell.
m = np.array([[0, 0, 6], [0, 0, 0.0]])
out["G"] = {"map": m.tolist(), "mean": float(m.mean()), "max": float(m.max())}
assert m.mean() == 1.0 and m.max() == 6.0

# H. A frozen layer f(x) = 2x and a head 3 f + 1: the answer's slope with respect to x is 3 * 2 = 6.
f = lambda x: 2 * x
ans = lambda x: 3 * f(x) + 1
fd = (ans(1.001) - ans(1.0)) / 0.001
out["H"] = {"x": 1.0, "answer": ans(1.0), "slope_by_hand": 6.0, "slope_by_finite_difference": round(float(fd), 3),
            "slope_if_the_frozen_layer_is_cut_off": 0.0}
assert close(fd, 6.0, 1e-2) and ans(1.0) == 7.0

# I. The size of a gradient: [3, 4] has size 5.
g = np.array([3.0, 4.0])
out["I"] = {"gradient": g.tolist(), "size": float(np.linalg.norm(g))}
assert np.linalg.norm(g) == 5.0

# J. A running average with decay 0.9, target 1, start 0: after 3 steps 1 - 0.9**3; with 0.999 it barely moves.
w = 0.0
for _ in range(3):
    w = 0.9 * w + 0.1 * 1
w2 = 0.0
for _ in range(3):
    w2 = 0.999 * w2 + 0.001 * 1
out["J"] = {"decay_0.9_after_3": round(w, 4), "decay_0.999_after_3": round(w2, 4)}
assert close(w, 1 - 0.9 ** 3, 1e-9)

(HERE / "toy_numbers.json").write_text(json.dumps(out, indent=1))
print("all hand answers reproduced:", ", ".join(out))

# ---------------------------------------------------------------- moving pictures
BLUE, ORANGE, GREY, INK, GREEN = "#1f5fd0", "#c86a0a", "#6b7a88", "#1f2a33", "#1d7a3e"


def gif_sigmoid():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9, 3.6))
    zz = np.linspace(-6, 6, 200)
    ax.plot(zz, sig(zz), color=INK, lw=2); ax.set_xlabel("raw score z (the logit)"); ax.set_ylabel("chance = sigma(z)")
    dot, = ax.plot([], [], "o", color=ORANGE, ms=10); txt = ax.text(-5.8, 0.85, "", color=ORANGE, size=11)
    bx.set_xlim(-0.6, 1.6); bx.set_ylim(0, 1.05); bx.set_xticks([0, 1]); bx.set_xticklabels(["digit 4", "digit 7"])
    bx.set_title("two scores, both at z", size=10)
    bars_s = bx.bar([-0.2, 0.8], [0, 0], 0.35, color=ORANGE, label="sigmoid each: can both be high")
    bars_m = bx.bar([0.2, 1.2], [0, 0], 0.35, color=GREY, label="softmax: must share 1")
    bx.legend(frameon=False, fontsize=8, loc="upper left")
    zs = np.r_[np.linspace(-4, 4, 40), np.linspace(4, -4, 40)]

    def upd(i):
        z = zs[i]; dot.set_data([z], [sig(z)])
        txt.set_text(f"z = {z:+.1f}   sigma = {float(sig(z)):.3f}")
        for b_, v in zip(bars_s, (sig(z), sig(z))): b_.set_height(v)
        for b_ in bars_m: b_.set_height(0.5)
        return []
    ani = FuncAnimation(fig, upd, frames=len(zs), interval=70)
    ani.save(HERE / "toy-a-sigmoid-and-softmax.gif", writer=PillowWriter(fps=14), dpi=70); plt.close(fig)


def gif_slope():
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ll = np.linspace(-5, 8, 300)
    ax.plot(ll, np.log(sig(ll)), color=ORANGE, lw=2.2, label="log d")
    ax.plot(ll, ll, color=BLUE, lw=2.2, ls="--", label="log d/(1-d) = the logit")
    ax.set_ylim(-6, 6); ax.axhline(0, color=GREY, lw=0.6); ax.set_xlabel("the logit l"); ax.legend(frameon=False, loc="upper left")
    pt_o, = ax.plot([], [], "o", color=ORANGE, ms=9); pt_b, = ax.plot([], [], "o", color=BLUE, ms=9)
    txt = ax.text(0.2, -5.2, "", size=10, color=INK)
    ls = np.r_[np.linspace(-3, 7, 50), np.linspace(7, -3, 50)]

    def upd(i):
        l = ls[i]; pt_o.set_data([l], [np.log(sig(l))]); pt_b.set_data([l], [l])
        txt.set_text(f"d = {float(sig(l)):.3f}:  slope of log d = {1 - float(sig(l)):.3f},  slope of the log-odds = 1")
        return []
    ani = FuncAnimation(fig, upd, frames=len(ls), interval=70)
    ani.save(HERE / "toy-b-why-the-log-odds-keeps-pushing.gif", writer=PillowWriter(fps=14), dpi=70); plt.close(fig)


gif_sigmoid(); gif_slope()
print("wrote the two moving pictures")
