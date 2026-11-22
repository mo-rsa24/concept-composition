"""Plan 33: the four steps of E0e drawn from a real run (asked 4 and 7, plan 32's either-side experts, both no-cross pushes).
Two runs of 8 canvases each, written to data/mnist-canvas/33-mnist-experts-at-inference/e0e/:
  locked   E0e: layout summed over the noisy steps (t >= 800) and held       (the setting that read 310 of 320)
  perstep  E0d-b: layout re-chosen at every step                            (the setting that read 56 of 320)
Figures: e0e-1-where-each-expert-wants-its-digit.png, e0e-2-the-layout-flips-or-holds.png, e0e-3-hide-the-other-half-and-join.png
Run with the sdxl-local python from 2-classifier-guidance/:  python scripts/mnist/make_plan33_e0e_steps.py
"""
import sys
from pathlib import Path
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "scripts/mnist")
from canvas_unet import ABAR, ddim_timesteps, per_sample_cap, start_noise
from checker import load_unet
import pixel_checker as pc

M = Path("data/mnist-canvas/33-mnist-experts-at-inference/models")
OUT = Path("data/mnist-canvas/33-mnist-experts-at-inference/e0e"); OUT.mkdir(parents=True, exist_ok=True)
A, B = load_unet(M / "e04-either/unet.pt", "cpu"), load_unet(M / "e59-either/unet.pt", "cpu")
a, b, SEEDS, LOCK_T = 4, 7, list(range(8)), 800
push_pair = pc.pair_push(pc.load_head(M / "pair-nc2/head.pt", "cpu"), a, b, 10.0)
push_sep = pc.separate_push(pc.load_head(M / "sep-nc/head.pt", "cpu"), a, b, 10.0)
ts = ddim_timesteps(50)
to8 = lambda v: ((v.clamp(-1, 1) + 1) * 127.5).numpy()


def run(lock):
    n = len(SEEDS)
    x = start_noise(SEEDS, "cpu"); gen = torch.Generator().manual_seed(4242)
    ya, yb, yn = torch.full((n,), a), torch.full((n,), b), torch.full((n,), 10)
    cum = torch.zeros(n); rec = dict(left=[], raw=[], parts=[], t=[], cum=[], snap={})
    for i, t in enumerate(ts):
        tt = t.repeat(n); a_t = ABAR[t]; a_prev = ABAR[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
        with torch.no_grad():
            eA, eB, nA, nB = A(x, tt, ya), B(x, tt, yb), A(x, tt, yn), B(x, tt, yn)
            dA, dB = (eA - nA).abs(), (eB - nB).abs()
            AL, AR, BL, BR = (dA[..., :32].mean((1, 2, 3)), dA[..., 32:].mean((1, 2, 3)), dB[..., :32].mean((1, 2, 3)), dB[..., 32:].mean((1, 2, 3)))
            sa, sb = AL + BR, AR + BL                                   # A left + B right, against A right + B left
            if lock:
                if int(t) >= LOCK_T:
                    cum = cum + (sa - sb)
                left = cum >= 0
            else:
                left = sa >= sb
            blank = -a_t.sqrt() + (1 - a_t).sqrt() * torch.randn(x.shape, generator=torch.Generator().manual_seed(12345 + int(t)))
            mL = torch.zeros_like(x); mL[..., :32] = 1
            lt = left[:, None, None, None]
            ownA = torch.where(lt, mL, 1 - mL); ownB = 1 - ownA
            xA, xB = ownA * x + (1 - ownA) * blank, ownB * x + (1 - ownB) * blank
            oA, oB = A(xA, tt, ya), B(xB, tt, yb)
            eps = ownA * oA + ownB * oB
        pp, ps = push_pair(x, tt, eps), push_sep(x, tt, eps)
        push, _ = per_sample_cap(pp + ps, eps, 0.1)
        rec["left"].append(left.numpy().copy()); rec["raw"].append((sa >= sb).numpy().copy()); rec["t"].append(int(t)); rec["cum"].append(cum.numpy().copy())
        rec["parts"].append(torch.stack([AL, AR, BL, BR], 1).numpy().copy())
        if lock and i in (3, 25):
            guess = lambda e, xx: (xx - (1 - a_t).sqrt() * e) / a_t.sqrt()
            rec["snap"][i] = dict(t=int(t), x=to8(x[0, 0]), dA=dA[0, 0].numpy(), dB=dB[0, 0].numpy(), xA=to8(xA[0, 0]), xB=to8(xB[0, 0]),
                                  gA=to8(guess(oA, xA)[0, 0]), gB=to8(guess(oB, xB)[0, 0]), gwholeA=to8(guess(eA, x)[0, 0]), gwholeB=to8(guess(eB, x)[0, 0]),
                                  stitched=to8(guess(eps + push, x)[0, 0]), left=bool(left[0]), ownA=ownA[0, 0].numpy())
        eps = eps + push
        x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        if i + 1 < len(ts):
            sig = ((1 - a_prev) / (1 - a_t)).sqrt() * (1 - a_t / a_prev).sqrt()
            x = a_prev.sqrt() * x0 + (1 - a_prev - sig ** 2).clamp_min(0).sqrt() * eps + sig * torch.randn(x.shape, generator=gen)
        else:
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
    rec["final"] = to8(x[:, 0])
    return rec


locked, perstep = run(True), run(False)
INK, GREY, BLUE, ORANGE, GREEN = "#1f2a33", "#6b7a88", "#1f5fd0", "#c8680a", "#1d6b3a"


def show(ax, img, title=None, cmap="gray", vmin=0, vmax=255, col=GREY):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(col); sp.set_linewidth(1.5)
    if title:
        ax.set_title(title, fontsize=10.5, color=INK)


# ---------------------------------------------------------------- figure 1: where each expert wants its digit (step 4 of 50, seed 0)
s = locked["snap"][3]; parts = locked["parts"][3][0] * 1000      # shown in thousandths
fig = plt.figure(figsize=(16, 9.5)); gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.5, wspace=0.35, left=0.04, right=0.985, top=0.9, bottom=0.07)
fig.suptitle(f"E0e step 1 of 4: where does each expert want its digit? (real run, asked {a} and {b}, sampler step 4 of 50, noise level t = {s['t']} of 1000)", fontsize=14, weight="bold", color=INK)
ax = fig.add_subplot(gs[0, 0:2]); show(ax, s["x"], "the noisy canvas both experts are looking at"); ax.axvline(31.5, color=ORANGE, ls="--", lw=2)
for k, tx in enumerate(("left half", "right half")):
    ax.text(16 + 32 * k, 3, tx, ha="center", color=ORANGE, weight="bold", fontsize=11, bbox=dict(fc="white", ec="none", pad=1))
for k, (nm, d, c) in enumerate((("0-4 expert", s["dA"], BLUE), ("5-9 expert", s["dB"], GREEN))):
    ax = fig.add_subplot(gs[0, 2 + k]); show(ax, d, f"{nm}: change in its prediction\nwhen told 'draw a {a if k == 0 else b}' (not 'nothing')", "magma", 0, float(max(s['dA'].max(), s['dB'].max())), c)
    ax.axvline(31.5, color="white", ls="--", lw=1.2)
    ax.text(16, 28, f"{parts[2 * k]:.2f}", ha="center", color="white", weight="bold", fontsize=11); ax.text(48, 28, f"{parts[2 * k + 1]:.2f}", ha="center", color="white", weight="bold", fontsize=11)
ax = fig.add_subplot(gs[1, 0:2]); ax.axis("off")
ax.text(0, 0.98, "The four numbers (the average of each bright map, per half, in thousandths):", fontsize=11.5, weight="bold", color=INK, va="top")
ax.text(0.0, 0.80, f"0-4 expert:  left {parts[0]:.2f}   right {parts[1]:.2f}\n5-9 expert:  left {parts[2]:.2f}   right {parts[3]:.2f}", fontsize=12, color=INK, va="top", family="monospace")
la, lb = parts[0] + parts[3], parts[1] + parts[2]
ax.text(0.0, 0.50, f"layout 1  (0-4 left, 5-9 right):  {parts[0]:.2f} + {parts[3]:.2f} = {la:.2f}\nlayout 2  (0-4 right, 5-9 left):  {parts[1]:.2f} + {parts[2]:.2f} = {lb:.2f}", fontsize=11.5, color=INK, va="top", family="monospace")
ax.text(0.0, 0.18, "The two experts may not share a half, so only these two\nlayouts exist. The bigger total is this step's vote.", fontsize=11, color=GREY, va="top")
ax = fig.add_subplot(gs[1, 2:4]); ax.bar([0, 1], [la, lb], color=[BLUE if la >= lb else GREY, BLUE if lb > la else GREY], width=0.5)
ax.set_xticks([0, 1]); ax.set_xticklabels(["0-4 left,\n5-9 right", "0-4 right,\n5-9 left"], fontsize=11); ax.set_ylabel("sum of the two averages (thousandths)", fontsize=10)
for k, v in enumerate((la, lb)):
    ax.text(k, v, f"{v:.2f}", ha="center", va="bottom", fontsize=11)
ax.set_title("this step's vote", fontsize=11, color=INK); ax.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT / "e0e-1-where-each-expert-wants-its-digit.png", dpi=75); plt.close(fig)

# ---------------------------------------------------------------- figure 2: the layout flips every step, or is held
L, P = np.array(locked["left"]).T, np.array(perstep["left"]).T          # seeds x steps, True: 0-4 expert on the left
fig = plt.figure(figsize=(16, 11)); gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.9], hspace=0.55, wspace=0.12, left=0.06, right=0.985, top=0.9, bottom=0.05)
fig.suptitle("E0e step 2 of 4: the layout vote is summed over the noisy steps, then held (8 canvases, asked 4 and 7, both pushes on)", fontsize=14, weight="bold", color=INK)
cm = matplotlib.colors.ListedColormap([ORANGE, BLUE])
lock_step = int(np.argmax(np.array(locked["t"]) < LOCK_T))
for r, (arr, ttl) in enumerate(((P, "re-chosen at every step (E0d-b): reads 56 of 320 clean pairs on the five tuning pairs"), (L, "summed over steps 1 to %d, then held (E0e): reads 310 of 320" % lock_step))):
    ax = fig.add_subplot(gs[r, :]); ax.imshow(arr.astype(int), cmap=cm, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    ax.set_yticks(range(8)); ax.set_yticklabels([f"canvas {k}" for k in range(8)], fontsize=9); ax.set_xlabel("sampler step (1 = most noise, 50 = clean)", fontsize=10)
    ax.set_title(ttl, fontsize=11.5, color=INK, loc="left")
    ax.axvline(lock_step - 0.5, color="white", lw=2)
ax = fig.add_subplot(gs[2, 0]); ax.axis("off")
ax.text(0, 0.9, "blue cell: the 0-4 expert is given the LEFT half at that step.\norange cell: it is given the RIGHT half.\nA row that keeps changing colour is a canvas whose two experts swap halves while it is drawn.\nThe white line is where the held version stops adding evidence (noise level 800 of 1000).", fontsize=11, color=INK, va="top")
swaps = (np.diff(P.astype(int), axis=1) != 0).sum(1)
ax.text(0, 0.35, f"layout swaps per canvas while it is drawn: re-chosen every step: {swaps.mean():.1f} on average (min {swaps.min()}, max {swaps.max()}); held: {(np.diff(L[:, lock_step:].astype(int), axis=1) != 0).sum()} after the white line, {(np.diff(L[:, :lock_step].astype(int), axis=1) != 0).sum()} across all 8 canvases before it.", fontsize=11, color=INK, va="top")
for k, (rec, nm) in enumerate(((perstep, "re-chosen every step"), (locked, "held"))):
    ax = fig.add_subplot(gs[2, 1]) if k == 0 else ax
    inner = ax.get_subplotspec().subgridspec(1, 8, wspace=0.05) if k == 0 else inner
    if k == 0:
        ax.axis("off")
        for c in range(8):
            a2 = fig.add_subplot(inner[0, c]); show(a2, perstep["final"][c], f"{c}", col=GREY)
        fig.text(0.53, 0.205, "finished canvases, re-chosen every step", fontsize=10.5, color=INK)
fig.savefig(OUT / "e0e-2-the-layout-flips-or-holds.png", dpi=75); plt.close(fig)

# ---------------------------------------------------------------- figure 3: hide the other half, join (step 26 of 50, seed 0)
s = locked["snap"][25]; left = s["left"]
nameL, nameR = ("0-4 expert", "5-9 expert") if left else ("5-9 expert", "0-4 expert")
fig = plt.figure(figsize=(16, 12)); gs = fig.add_gridspec(3, 4, hspace=0.6, wspace=0.14, left=0.04, right=0.985, top=0.9, bottom=0.04)
fig.suptitle(f"E0e steps 3 and 4 of 4: hide the other half, keep each expert's own half, join (real run, sampler step 26 of 50, t = {s['t']})", fontsize=14, weight="bold", color=INK)
ax = fig.add_subplot(gs[0, 0:2]); show(ax, s["x"], f"the noisy canvas; the held layout gives the {nameL} the left half"); ax.axvline(31.5, color=ORANGE, ls="--", lw=2)
ax = fig.add_subplot(gs[0, 2]); show(ax, s["xA"], "what the 0-4 expert is shown:\nits half, and the other half blank + noise", col=BLUE)
ax = fig.add_subplot(gs[0, 3]); show(ax, s["xB"], "what the 5-9 expert is shown:\nits half, and the other half blank + noise", col=GREEN)
ax = fig.add_subplot(gs[1, 0]); show(ax, s["gwholeA"], "WITHOUT hiding: the 0-4 expert's clean\nguess from the whole canvas")
ax = fig.add_subplot(gs[1, 1]); show(ax, s["gwholeB"], "WITHOUT hiding: the 5-9 expert's clean\nguess from the whole canvas")
ax = fig.add_subplot(gs[1, 2]); show(ax, s["gA"], "WITH hiding: the 0-4 expert's\nclean guess", col=BLUE)
ax = fig.add_subplot(gs[1, 3]); show(ax, s["gB"], "WITH hiding: the 5-9 expert's\nclean guess", col=GREEN)
ax = fig.add_subplot(gs[2, 0:2]); show(ax, s["stitched"], "each guess kept on its own half only, joined, with the pushes added", col=GREEN); ax.axvline(31.5, color=ORANGE, ls="--", lw=1.5)
inner = gs[2, 2:4].subgridspec(1, 1); ax = fig.add_subplot(inner[0, 0]); ax.axis("off")
ax.text(0, 0.9, "Why the top-left pair is worse: the expert is shown the other\nexpert's digit, which it never saw in training, and tries to\nerase it or turn it into one of its own digits.\n\nOn the bottom row the experts only ever see one digit's\nworth of ink and an empty half, which is what they trained on.", fontsize=11.5, color=INK, va="top")
fig.savefig(OUT / "e0e-3-hide-the-other-half-and-join.png", dpi=75); plt.close(fig)
print("layout swaps per canvas, re-chosen every step:", swaps.tolist(), "| held:", (np.diff(L.astype(int), axis=1) != 0).sum(1).tolist(), "| lock step", lock_step)
print("ok")
