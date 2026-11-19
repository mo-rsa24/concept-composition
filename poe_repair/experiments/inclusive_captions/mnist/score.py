"""Call every drawn tile with this step's judge, check the counts against the bar, draw the figure.

Reads data/mnist/samples/dial-<d>-seed-<s>.npz (8 rows x 256 seeds each). Writes, into
4-inclusive-captions/artifacts/results/do-inclusive-labels-help-the-product-compose-on-mnist/:

  scores.csv        one row per model, row and seed: dial, training seed, row, labels, seed, call
  counts.json       the count of each call per model and row, the bar's checks and the verdict
  the-leak-read.png two panels against the dial: the product's "two separate right digits" (top),
                    "7", "8" and "4" alone drawing their digit alone beside each label's training line
                    (bottom); one dot per training, a line through their mean; the toy's counts in grey

Prints SUPPORTED, NULL or INCONCLUSIVE once all 15 models are scored, and the partial counts before that.
The bar is bar.py, written before any model drew. Runs on the CPU.

Usage (from the repo root):
    python -I poe_repair/experiments/inclusive_captions/mnist/score.py
"""

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
import bar  # noqa: E402
from judge_checks import pair_call, pinned_judge, single_call  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
SAMPLES = REPO / "data/mnist/samples"
OUT = REPO / "4-inclusive-captions/artifacts/results/do-inclusive-labels-help-the-product-compose-on-mnist"
SEEDS_T = (1, 2, 3)
ROWS = {  # row: (kind, asked digits)
    "7 alone": ("single", 7), "8 alone": ("single", 8), "4 alone": ("single", 4),
    "7 and 8": ("pair", (7, 8)), "7 x 8 plain": ("pair", (7, 8)), "7 x 8 corrected": ("pair", (7, 8)),
    "4 x 7 plain": ("pair", (4, 7)), "4 x 7 corrected": ("pair", (4, 7)),
}
TWO = "two separate right digits"
ALONE = "right digit alone"
# Each label's training line: the share of its training draws holding one digit (route step 4).
TRAINING_LINE = {"7": [100, 94.9, 89.7, 85.5, 82.4], "8": [100, 94.7, 88.4, 84.2, 81.4],
                 "4": [100, 99.1, 98.1, 97.2, 96.7]}
TOY_PLAIN, TOY_CORR = [0, 110, 103, 51, 205], [0, 300, 300, 300, 300]   # of 300, the toy's prediction


def dial_of(path):
    return float(path.stem.split("-")[1])


def score_all():
    j, net, sha = pinned_judge()
    rows, counts = [], {}
    for path in sorted(SAMPLES.glob("dial-*-seed-*.npz")):
        dial, ts = dial_of(path), int(path.stem.split("-")[-1])
        d = np.load(path)
        for row, (kind, asked) in ROWS.items():
            calls = [pair_call(j, net, t, asked) if kind == "pair" else single_call(j, net, t, asked)
                     for t in d[row]]
            counts[(dial, ts, row)] = {c: calls.count(c) for c in sorted(set(calls))}
            rows += [{"dial": dial, "training_seed": ts, "row": row, "labels": str(asked), "seed": i, "call": c}
                     for i, c in enumerate(calls)]
    return rows, counts, sha


def n(counts, dial, ts, row, call):
    return counts.get((dial, ts, row), {}).get(call, 0)


def verdict(counts):
    have = {(d, s) for d, s, _ in counts}
    complete = all((d, s) in have for d in bar.DIALS for s in SEEDS_T)
    out = {"models_scored": len(have), "complete": complete, "settings": {}}
    for d in bar.DIALS:
        ss = [s for s in SEEDS_T if (d, s) in have]
        if not ss:
            continue
        pure = all(n(counts, d, s, "7 alone", ALONE) >= bar.PURITY_FLOOR["7"][d] and
                   n(counts, d, s, "8 alone", ALONE) >= bar.PURITY_FLOOR["8"][d] for s in ss)
        out["settings"][str(d)] = {
            "trainings": ss, "counts": pure and len(ss) == 3,
            "seven_alone": [n(counts, d, s, "7 alone", ALONE) for s in ss],
            "eight_alone": [n(counts, d, s, "8 alone", ALONE) for s in ss],
            "corrected_7x8_two": [n(counts, d, s, "7 x 8 corrected", TWO) for s in ss],
            "plain_7x8_two": [n(counts, d, s, "7 x 8 plain", TWO) for s in ss],
            "corrected_4x7_two": [n(counts, d, s, "4 x 7 corrected", TWO) for s in ss],
        }
    st = out["settings"]
    d0 = st.get("0.0", {})
    out["stop_plain_dial0"] = any(c > bar.STOP_PLAIN_DIAL0 for c in d0.get("plain_7x8_two", []))
    if not complete:
        out["verdict"] = "PENDING"
        return out
    dial0_low = all(c <= bar.DIAL0_MAX for c in d0["corrected_7x8_two"])
    supported = [d for d, v in st.items() if d != "0.0" and v["counts"] and dial0_low
                 and all(c >= bar.SUPPORT_MIN for c in v["corrected_7x8_two"])]
    null = all(np.mean(v["corrected_7x8_two"]) < bar.NULL_MAX_MEAN for v in st.values() if v["counts"])
    out["verdict"] = "SUPPORTED" if supported else ("NULL" if null else "INCONCLUSIVE")
    out["supported_settings"] = supported
    if supported:
        d = supported[0]
        out["corrector_not_needed"] = all(c >= bar.CORRECTOR_NOT_NEEDED_MIN for c in st[d]["plain_7x8_two"])
        out["reaches_unseen_pair"] = all(
            c >= bar.UNSEEN_MIN and c >= bar.UNSEEN_FACTOR * c0
            for c, c0 in zip(st[d]["corrected_4x7_two"], d0["corrected_4x7_two"]))
    return out


def figure(counts, path):
    dials = list(bar.DIALS)
    fig, (top, bot) = plt.subplots(2, 1, figsize=(10, 9.5), sharex=True, gridspec_kw=dict(hspace=0.18))
    for ax in (top, bot):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xticks(dials), ax.set_xticklabels([f"{x:g}" for x in dials])
    top.plot(dials, [100 * c / 300 for c in TOY_CORR], color="#6b7a88", ls=":", marker="D", mfc="white")
    top.plot(dials, [100 * c / 300 for c in TOY_PLAIN], color="#6b7a88", ls=":", marker="o", mfc="white")
    lines = [("7 x 8 corrected", "#1f5fd0", "-"), ("7 x 8 plain", "#1f5fd0", "--"),
             ("4 x 7 corrected", "#e09f00", "-"), ("4 x 7 plain", "#e09f00", "--")]
    for row, col, ls in lines:
        xs, means = [], []
        for d in dials:
            v = [n(counts, d, s, row, TWO) or 0 for s in SEEDS_T if (d, s, row) in counts]
            if v:
                top.scatter([d] * len(v), [100 * x / 256 for x in v], color=col, s=18, zorder=3)
                xs.append(d), means.append(100 * np.mean(v) / 256)
        if xs:
            top.plot(xs, means, color=col, ls=ls, lw=2, label=row)
    top.axhline(100 * bar.SUPPORT_MIN / 256, color="#1d7a3e", lw=1, ls=(0, (4, 3)))
    top.text(dials[-1], 100 * bar.SUPPORT_MIN / 256 + 2, "support line, 128 of 256", ha="right", fontsize=9,
             color="#1d7a3e")
    top.set_ylim(-3, 105), top.set_ylabel("two separate right digits (% of 256)")
    top.legend(fontsize=9, loc="upper left", frameon=False)
    top.set_title("Does the product compose? Grey dotted: the toy's counts", loc="left", fontsize=11.5, weight="bold")
    for lab, col in (("7", "#1f5fd0"), ("8", "#c0262d"), ("4", "#e09f00")):
        bot.plot(dials, TRAINING_LINE[lab], color=col, ls="--", lw=1.5)
        for d in dials:
            v = [n(counts, d, s, f"{lab} alone", ALONE) for s in SEEDS_T if (d, s, f"{lab} alone") in counts]
            if v:
                bot.scatter([d] * len(v), [100 * x / 256 for x in v], color=col, s=18, zorder=3)
        bot.text(dials[-1] + 0.02, TRAINING_LINE[lab][-1], f'"{lab}"', color=col, fontsize=10, va="center")
    bot.set_ylim(70, 102), bot.set_ylabel("drew its digit alone (%)\ndashed: its training line")
    bot.set_xlabel("dial: how often a two-digit canvas is also trained under each single label")
    bot.set_title("Do the single labels still draw their own digit alone?", loc="left", fontsize=11.5, weight="bold")
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    torch.set_num_threads(4)
    rows, counts, sha = score_all()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(), w.writerows(rows)
    v = verdict(counts)
    v["judge"] = {"shared_commit": "5bff268f", "weights_sha256": sha,
                  "change": "the 0.90 confidence line applies to one-blob tiles only"}
    json.dump({"verdict": v, "counts": {f"{d:g}|{s}|{r}": c for (d, s, r), c in counts.items()}},
              open(OUT / "counts.json", "w"), indent=1)
    figure(counts, OUT / "the-leak-read.png")
    for d, st in v["settings"].items():
        print(f"dial {float(d):g} · trainings {st['trainings']} · \"7\" alone {st['seven_alone']} · \"8\" alone "
              f"{st['eight_alone']} · 7x8 plain two {st['plain_7x8_two']} · 7x8 corrected two "
              f"{st['corrected_7x8_two']} · 4x7 corrected two {st['corrected_4x7_two']} (of 256)")
    if v["stop_plain_dial0"]:
        print(f"STOP: the plain 7 x 8 product at dial 0 draws more than {bar.STOP_PLAIN_DIAL0} of 256")
    print(f"{v['verdict']} · {v['models_scored']} of 15 models scored · {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
