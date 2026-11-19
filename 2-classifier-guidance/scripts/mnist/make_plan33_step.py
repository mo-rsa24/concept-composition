"""Plan 33: what one sampler step does, drawn from a real run. Asked 4 and 7, seed 0, plan 32's two either-side experts, the half-canvas
crop composition with both pushes (the no-cross referees pair-nc2 and sep-nc), default strength 10, cap 0.1, eta 1.0.
The seed is the first seed of the five-pair cell that the judge calls two clear, right digits (tile 0 of the 4 and 7 sheet).
Run with the sdxl-local python from 2-classifier-guidance/:  python scripts/mnist/make_plan33_step.py
"""
import sys
from pathlib import Path
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, "scripts/mnist")
from canvas_unet import ABAR, NULL, ddim_timesteps, per_sample_cap, start_noise
from checker import load_unet
import pixel_checker as pc
import two_experts as te

M = Path("data/mnist-canvas/33-mnist-experts-at-inference/models")
A, B = load_unet(M / "e04-either/unet.pt", "cpu"), load_unet(M / "e59-either/unet.pt", "cpu")
heads = {"pair": pc.load_head(M / "pair-nc2/head.pt", "cpu"), "sep": pc.load_head(M / "sep-nc/head.pt", "cpu")}
a, b, seed, SHOW = 4, 7, 0, 25
side = int(np.random.default_rng(1000 + seed).integers(2))      # 0: the 0-4 expert takes the left half
ts = ddim_timesteps(50)
x = start_noise([seed], "cpu")
gen = torch.Generator().manual_seed(1_000_003 + seed)
push_pair, push_sep = pc.pair_push(heads["pair"], a, b, 10.0), pc.separate_push(heads["sep"], a, b, 10.0)
ya, yb = torch.tensor([a]), torch.tensor([b])
strip, shown = [], {}
to8 = lambda v: ((v.clamp(-1, 1) + 1) * 127.5).numpy()[0, 0]
for i, t in enumerate(ts):
    tt = t.repeat(1)
    a_t = ABAR[t]; a_prev = ABAR[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
    with torch.no_grad():
        xl, xr = x[..., :32], x[..., 32:]
        eA = (A(xl, tt, ya), A(xr, tt, ya)); eB = (B(xl, tt, yb), B(xr, tt, yb))
        eps = torch.cat([eA[0], eB[1]], -1) if side == 0 else torch.cat([eB[0], eA[1]], -1)
        whole_A, whole_B = A(x, tt, ya), B(x, tt, yb)
    pp, ps = push_pair(x, tt, eps), push_sep(x, tt, eps)
    push, fired = per_sample_cap(pp + ps, eps, 0.1)
    if i == SHOW:
        guess = lambda e: (x - (1 - a_t).sqrt() * e) / a_t.sqrt()
        shown = dict(i=i, t=int(t), x=to8(x), eps_crop=eps, eps_whole_A=whole_A, eps_whole_B=whole_B, guess_crop=to8(guess(eps)),
                     guess_wholeA=to8(guess(whole_A)), guess_wholeB=to8(guess(whole_B)), pp=pp[0, 0].numpy(), ps=ps[0, 0].numpy(),
                     capped=push[0, 0].numpy(), eps_size=float(eps.norm()), push_size=float(push.norm()), raw_size=float((pp + ps).norm()))
    if i in (0, 10, 20, 30, 40):
        strip.append((i, int(t), to8(x)))
    eps = eps + push
    x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
    if i + 1 < len(ts):
        sig = 1.0 * ((1 - a_prev) / (1 - a_t)).sqrt() * (1 - a_t / a_prev).sqrt()
        z = torch.randn(x.shape[1:], generator=gen)[None]
        x = a_prev.sqrt() * x0 + (1 - a_prev - sig ** 2).clamp_min(0).sqrt() * eps + sig * z
    else:
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
strip.append((49, 0, to8(x)))
print("side", side, "step", SHOW, "t", shown["t"], "eps size", round(shown["eps_size"], 2), "raw push", round(shown["raw_size"], 2),
      "capped push", round(shown["push_size"], 2))

# ---------------------------------------------------------------- the figure
INK, GREY, ORANGE, GREEN, BLUE = "#1f2a33", "#6b7a88", "#c8680a", "#1d6b3a", "#1f5fd0"
fig = plt.figure(figsize=(16, 15))
gs = fig.add_gridspec(4, 4, height_ratios=[1.0, 1.15, 1.15, 0.85], hspace=0.55, wspace=0.12, left=0.04, right=0.985, top=0.93, bottom=0.06)
fig.suptitle(f"One sampler step, from a real run (asked {a} and {b}, seed {seed}, step {SHOW + 1} of 50): what each expert sees, and what the pushes push on",
             fontsize=14, weight="bold", color=INK)


def show(ax, img, title=None, cmap="gray", vmin=0, vmax=255, col=GREY):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(col); sp.set_linewidth(1.5)
    if title:
        ax.set_title(title, fontsize=11, color=INK)


# row 1: the noisy canvas and the two crops
ax = fig.add_subplot(gs[0, 0:2]); show(ax, shown["x"], "1. the noisy canvas at this step (what the sampler holds)")
ax.axvline(31.5, color=ORANGE, ls="--", lw=2)
ax.text(16, 4, "left half", ha="center", color=ORANGE, fontsize=11, weight="bold", bbox=dict(fc="white", ec="none", pad=1)); ax.text(48, 4, "right half", ha="center", color=ORANGE, fontsize=11, weight="bold", bbox=dict(fc="white", ec="none", pad=1))
nameL, nameR = ("0-4 expert", "5-9 expert") if side == 0 else ("5-9 expert", "0-4 expert")
ax = fig.add_subplot(gs[0, 2]); show(ax, shown["x"][:, :32], f"2a. the {nameL} is shown ONLY this", col=BLUE)
ax = fig.add_subplot(gs[0, 3]); show(ax, shown["x"][:, 32:], f"2b. the {nameR} is shown ONLY this", col=BLUE)

# row 2: what the experts predict: whole canvas versus crop
guessL, guessR = shown["guess_crop"][:, :32], shown["guess_crop"][:, 32:]
ax = fig.add_subplot(gs[1, 0]); show(ax, shown["guess_wholeA"], "WITHOUT the crop: the 0-4 expert\nlooking at the whole canvas, its clean guess")
ax = fig.add_subplot(gs[1, 1]); show(ax, shown["guess_wholeB"], "WITHOUT the crop: the 5-9 expert\nlooking at the whole canvas, its clean guess")
ax = fig.add_subplot(gs[1, 2:4]); show(ax, shown["guess_crop"], "WITH the crop: each expert's clean guess for its own half, stitched", col=GREEN)
ax.axvline(31.5, color=ORANGE, ls="--", lw=1.5)

# row 3: the pushes
lim = lambda g: float(np.abs(g).max()) or 1.0
ax = fig.add_subplot(gs[2, 0:2]); g = shown["pp"]; show(ax, g, "3a. pair-aware referee's push: which pixels to make brighter (orange)\nor darker (blue) so the canvas looks more like a clear 4 and 7", "PuOr_r", -lim(g), lim(g), ORANGE)
ax = fig.add_subplot(gs[2, 2:4]); g = shown["ps"]; show(ax, g, "3b. separate-label referee's push: the same, asked separately\n'is there a 4?' and 'is there a 7?'", "PuOr_r", -lim(g), lim(g), ORANGE)

# row 4: the whole run
fig.text(0.04, 0.205, "4. the same seed, noise to finished canvas (steps 1, 11, 21, 31, 41, 50):", fontsize=12, weight="bold", color=INK)
inner = gs[3, :].subgridspec(1, 6, wspace=0.08)
for k, (i, t, img) in enumerate(strip):
    ax = fig.add_subplot(inner[0, k]); show(ax, img, f"step {i + 1}", col=GREEN if k == 5 else GREY)
fig.text(0.04, 0.008, f"Real numbers at this step: the experts' prediction has size {shown['eps_size']:.1f}; the two referees' pushes together would have size "
         f"{shown['raw_size']:.1f}, cut to {shown['push_size']:.1f} by the cap (10% of the prediction).\nThe pushes are the two orange-and-blue panels above, added together, scaled and capped, then added to the experts' prediction.",
         fontsize=10, color=GREY)
fig.savefig("data/mnist-canvas/33-mnist-experts-at-inference/one-sampler-step.png", dpi=75)
print("ok")
