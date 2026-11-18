"""Plain product on 7 & 8 and the five held-out pairs, and the support check (plan 24 task 1.4).

For one trained canvas U-Net: the product of two digit experts with no push, DDIM 50 steps, seeds 0
to 63 per pair, each canvas called by the judge. The bar is read at g = 1, the Bayes composition
p(x | a) p(x | b) / p(x); g = 3 is drawn beside it as a diagnostic and never read against the bar.
The support check (`--support`): expert 7 alone and expert 4 alone, 1,000 seeds each (seeds 2,000
up, never the evaluation seeds), counting canvases the judge reads as the expert's digit beside one
of its held-out partners (4 for 7; 3 and 7 for 4).

Writes, under <out>/share-<share>/: canvases.npz, calls.csv, counts.json, sheet.png.

Usage:
    python plain_product.py --unet <models>/share-0.01/unet.pt --judge <judge.pt> --out <dir> [--support]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canvas_unet import CanvasUNet, HELD_OUT, sample, to_uint8  # noqa: E402
from judge import LADDER, judge, load as load_judge  # noqa: E402

PAIRS = [(7, 8)] + HELD_OUT
SEEDS = list(range(64))
GS = [1.0, 3.0]
COLOUR = {"damaged": "#000000", "one digit": "#7f7f7f", "fused": "#d62728",
          "two digits, wrong identity": "#ff7f0e", "two clear, right digits": "#2ca02c"}
SHORT = {"damaged": "damaged", "one digit": "one", "fused": "fused",
         "two digits, wrong identity": "wrong pair", "two clear, right digits": "two right"}


def load_unet(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = CanvasUNet(ck["args"]["ch"]).to(dev).eval()
    net.load_state_dict(ck["ema"])
    return net, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", type=Path, required=True)
    ap.add_argument("--judge", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--support", action="store_true")
    args = ap.parse_args()
    dev = "cuda"
    net, ck = load_unet(args.unet, dev)
    jn = load_judge(args.judge, dev)
    share = ck["args"]["share"]
    out = args.out / f"share-{share}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"share {share}, unet step {ck['step']}", flush=True)

    # single experts, a sanity look at what each label draws
    singles = {d: to_uint8(sample(net, range(100, 104), [d], device=dev)[0]) for d in range(10)}

    canv, rows, counts = {}, [], {}
    for g in GS:
        for a, b in PAIRS:
            x = to_uint8(sample(net, SEEDS, [a, b], g=g, device=dev)[0])
            canv[f"g{g}_{a}{b}"] = x
            tally = {k: 0 for k in LADDER}
            for s, c in zip(SEEDS, x):
                call, reads = judge(jn, c, (a, b), dev)
                tally[call] += 1
                rows.append([g, f"{a}&{b}", s, call, "; ".join(f"{r[2]} {r[3]:.2f}" for r in reads)])
            counts[f"g={g} {a}&{b}"] = tally
            print(f"g={g} {a}&{b}: {tally}", flush=True)

    support = {}
    if args.support:
        partners = {7: [4], 4: [3, 7]}
        for d, ps in partners.items():
            found = {p: 0 for p in ps}
            other_pairs = 0
            for start in range(2000, 3000, 250):
                x = to_uint8(sample(net, range(start, start + 250), [d], device=dev)[0])
                for c in x:
                    call, reads = judge(jn, c, (d, -1), dev)
                    digits = [r[2] for r in reads if r[2] != 10 and r[3] >= 0.9]
                    if len(reads) == 2 and len(digits) == 2 and d in digits:
                        other = digits[0] if digits[1] == d else digits[1]
                        if other in found:
                            found[other] += 1
                        else:
                            other_pairs += 1
            support[f"expert {d} alone, 1000 seeds"] = {"held-out partner beside it": found,
                                                        "any other digit beside it": other_pairs}
            print(f"support, expert {d}: {support[f'expert {d} alone, 1000 seeds']}", flush=True)

    np.savez_compressed(out / "canvases.npz", **canv, **{f"single_{d}": v for d, v in singles.items()})
    with open(out / "calls.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["g", "pair", "seed", "judge_call", "blob_reads"])
        w.writerows(rows)
    res = {"share": share, "unet_step": ck["step"], "seeds": "0-63", "counts": counts, "support": support}
    (out / "counts.json").write_text(json.dumps(res, indent=1))
    draw(out, share, ck["step"], singles, canv, rows, counts)


def draw(out, share, step, singles, canv, rows, counts):
    fig = plt.figure(figsize=(22, 27))
    top, mid, bot = fig.subfigures(3, 1, height_ratios=[0.55, 2.2, 1.1], hspace=0.02)
    top.suptitle(f"Share {share}, step {step}. What each digit expert draws alone (seeds 100 to 103)",
                 fontsize=15, weight="bold")
    ax = top.subplots(4, 10, gridspec_kw={"wspace": 0.05, "hspace": 0.05})
    for d in range(10):
        for k in range(4):
            ax[k, d].imshow(singles[d][k], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax[k, d].set_xticks([]), ax[k, d].set_yticks([])
        ax[0, d].set_title(f"expert {d}", fontsize=12)
    calls = {(r[0], r[1], r[2]): r[3] for r in rows}
    c78 = counts["g=1.0 7&8"]
    mid.suptitle(f"Plain product of the 7 and 8 experts, g = 1, seeds 0 to 63. Frame colour is the judge's call: "
                 f"two clear, right digits on {c78['two clear, right digits']} of 64, fused on {c78['fused']}",
                 fontsize=15, weight="bold")
    ax = mid.subplots(8, 8, gridspec_kw={"wspace": 0.06, "hspace": 0.3})
    for s, a in zip(range(64), ax.flat):
        call = calls[(1.0, "7&8", s)]
        a.imshow(canv["g1.0_78"][s], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        a.set_xticks([]), a.set_yticks([])
        a.set_title(f"seed {s}: {SHORT[call]}", fontsize=10, color=COLOUR[call])
        for sp in a.spines.values():
            sp.set_edgecolor(COLOUR[call]), sp.set_linewidth(3)
    bot.suptitle("Every pair, 64 seeds each: the judge's count per call (bar read at g = 1; g = 3 is a diagnostic)",
                 fontsize=15, weight="bold")
    ax = bot.subplots(1, 2, gridspec_kw={"wspace": 0.3})
    for a, g in zip(ax, (1.0, 3.0)):
        names = [k.split(" ")[1] for k in counts if k.startswith(f"g={g}")]
        left = np.zeros(len(names))
        for call in ["two clear, right digits", "two digits, wrong identity", "fused", "one digit", "damaged"]:
            v = np.array([counts[f"g={g} {n}"][call] for n in names])
            a.barh(names, v, left=left, color=COLOUR[call], label=SHORT[call])
            for i, (l, w) in enumerate(zip(left, v)):
                if w >= 4:
                    a.text(l + w / 2, i, str(w), ha="center", va="center", color="white", fontsize=11)
            left += v
        a.set_xlim(0, 64), a.set_xlabel("canvases out of 64 seeds"), a.set_title(f"g = {g}", fontsize=13)
        a.invert_yaxis()
    ax[1].legend(loc="lower right", fontsize=11)
    fig.savefig(out / "sheet.png", dpi=60, bbox_inches="tight")
    print(f"wrote {out}/sheet.png", flush=True)


if __name__ == "__main__":
    main()
