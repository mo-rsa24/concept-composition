"""The four panels of Figure 10 (slot R9), one PDF and PNG each, for subfigures with their own captions.

  a  failure types from the fixed-prompt judge (scripts/review_judge_failures.py): how many of each
     setting's 424 images got each label (style loss is kept in judge_shares.csv, not plotted)
  b  BLIP-VQA per held-out group, 95% intervals from resampling pairs then seeds
  c  share of the animal-pair images where the detector finds two or more animals
  d  ImageReward over all 424 images per setting

The four settings are the adapter off (plain PoE), on for steps 0 to 10, 0 to 25, and all 50 of 50,
each on the same 424 images (53 held-out pairs, seeds 9 to 16, one starting latent per seed).

Usage:
  python3 scripts/review_window_panels.py --scores <scores dir> --judge <judge dir> --out <results dir>
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from review_window_figure import COLOURS, GROUPS, SETTINGS, cell, load  # noqa: E402

JUDGE_SETTING = {"off": "off", "steps 0-10": "10", "steps 0-25": "25", "all 50": "50"}
TYPES = [("hybrid", "hybrid"), ("one", "one\nconcept"), ("intrusion", "intrusion"),
         ("count error", "count\nerror")]
GROUP_TICKS = ["hard\n(3 pairs)", "easy\n(29 pairs)", "not competing\n(21 pairs)"]


def legend_strip(out):
    """One legend for the whole grid, saved on its own so the four panels keep the same size."""
    fig = plt.figure(figsize=(6.9, 0.25))
    fig.legend(handles=[Patch(color=COLOURS[s], label=lab) for s, lab in SETTINGS], frameon=False,
               ncol=4, fontsize=7, loc="center", title=None)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"panel_legend.{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def bars(ax, groups, value):
    width = 0.2
    for gi, g in enumerate(groups):
        for si, (s, _) in enumerate(SETTINGS):
            v = value(g, s)
            if v is None:
                continue
            x = gi + (si - 1.5) * width
            ax.bar(x, v["mean"], width, color=COLOURS[s])
            if "lo" in v:
                ax.errorbar(x, v["mean"], yerr=[[v["mean"] - v["lo"]], [v["hi"] - v["mean"]]],
                            color="k", lw=0.6, capsize=1.5)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, out, name):
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=10_000)
    a = ap.parse_args()
    plt.rcParams.update({"font.size": 7, "font.family": "serif"})
    rng = random.Random(0)

    # a: judge labels
    key = {r["id"]: r for r in csv.DictReader(open(os.path.join(a.judge, "key.csv")))}
    labels = list(csv.DictReader(open(os.path.join(a.judge, "judge_labels.csv"))))
    per = {s: Counter() for s in JUDGE_SETTING.values()}
    n = Counter()
    for r in labels:
        if not r["label"]:
            continue
        s = JUDGE_SETTING[key[r["id"]]["setting"]]
        n[s] += 1
        per[s][r["label"]] += 1
        per[s]["style"] += r["style_loss"] == "yes"
    shares = {(t, s): {"mean": per[s][t]} for t, _ in TYPES for s in per if n[s]}
    with open(os.path.join(a.out, "judge_shares.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "images", *[t for t, _ in TYPES], "two", "broken", "style_loss"])
        for s in per:
            w.writerow([s, n[s], *[per[s][t] for t, _ in TYPES], per[s]["two"], per[s]["broken"], per[s]["style"]])
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    bars(ax, [t for t, _ in TYPES], lambda t, s: shares.get((t, s)))
    ax.set_xticks(range(len(TYPES)), [lab for _, lab in TYPES])
    ax.set_ylabel("images (of 424)")
    save(fig, a.out, "panel_a_failure_types")

    legend_strip(a.out)
    rows = load(a.scores)
    # b: BLIP-VQA per held-out group
    get = {}
    for s, _ in SETTINGS:
        for g, _ in GROUPS:
            get[(g, s)] = cell([r for r in rows if r["win"] == s and r["group"] == g], "blip", rng, a.draws)
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    bars(ax, [g for g, _ in GROUPS], lambda g, s: get.get((g, s)))
    ax.set_xticks(range(len(GROUPS)), GROUP_TICKS)
    ax.set_ylabel("BLIP-VQA (0 to 1)")
    ax.set_ylim(0, 1)
    save(fig, a.out, "panel_b_blip")

    # c and d: one bar per setting
    for name, k, ylab in (("panel_c_two_animals", "comp", "images with two animals (of 160)"),
                          ("panel_d_imagereward", "ir", "ImageReward")):
        fig, ax = plt.subplots(figsize=(3.3, 1.9))
        for si, (s, lab) in enumerate(SETTINGS):
            v = cell([r for r in rows if r["win"] == s], k, rng, a.draws)
            if k == "comp":  # a count of the animal-pair images, every setting has the same 160
                v = {**v, **{q: v[q] * v["images"] for q in ("mean", "lo", "hi")}}
            ax.bar(si, v["mean"], 0.7, color=COLOURS[s])
            ax.errorbar(si, v["mean"], yerr=[[v["mean"] - v["lo"]], [v["hi"] - v["mean"]]], color="k", lw=0.6, capsize=1.5)
        ax.set_xticks(range(len(SETTINGS)), [lab.replace(" for ", "\n") for _, lab in SETTINGS], fontsize=6)
        ax.set_ylabel(ylab)
        if k == "comp":
            ax.set_ylim(0, 160)
        ax.axhline(0, color="k", lw=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        save(fig, a.out, name)
    print("labelled per setting:", dict(n))


if __name__ == "__main__":
    main()
