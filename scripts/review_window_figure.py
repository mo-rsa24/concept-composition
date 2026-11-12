"""Figure 10 (slot R9): established scores as the adapter acts on fewer steps.

Four settings on the same 424 images each (53 held-out pairs, seeds 9 to 16, one starting latent per
seed): adapter off (plain PoE, the census PoE rows of Table 3), on steps 0 to 10, on steps 0 to 25,
and on all 50 (Table 3's ours rows). Off and all 50 come from scores_long.csv; the two windows from
the *_windows.csv score files the same scorers wrote. Four panels:

  a  BLIP-VQA (T2I-CompBench's disentangled question per noun phrase of "a X and a Y"), mean per
     held-out group with a 95% interval from resampling pairs, then seeds within each pair
  b  Comp.: share of animal-pair images where GroundingDINO finds two or more animals
  c  CLIP score (ViT-L/14), all 424 images per setting, same interval
  d  ImageReward v1.0, all 424 images per setting, same interval

Writes window_scores.csv (one row per setting, group and measure, with the axis columns) and
window_scores.pdf and window_scores.png. Usage:
  python3 scripts/review_window_figure.py --scores <dir holding scores_long.csv and *_windows.csv> --out <results dir>
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from review_build_tables import two_level  # noqa: E402

SETTINGS = [("off", "off\n(PoE)"), ("10", "steps\n0–10"), ("25", "steps\n0–25"), ("50", "all 50")]
GROUPS = [("hard", "hard (3)"), ("easy but unseen", "easy (29)"), ("not competing", "not competing (21)")]
COLOURS = {"off": "#9aa0a6", "10": "#9ecae1", "25": "#4292c6", "50": "#08519c"}


def load(scores: str) -> list[dict]:
    long = list(csv.DictReader(open(os.path.join(scores, "scores_long.csv"))))
    out = []
    for r in long:
        if r["table"] != "R3":
            continue
        if r["method"] == "poe":
            out.append({**r, "win": "off"})
        elif r["method"] == "ours" and r["run"] == "r3_ours":
            out.append({**r, "win": "50"})
    rows = {r["image"]: r for r in csv.DictReader(open(os.path.join(scores, "rows_windows.csv")))}
    blip = {r["image"]: r for r in csv.DictReader(open(os.path.join(scores, "blip_windows.csv")))}
    clip = {r["image"]: r for r in csv.DictReader(open(os.path.join(scores, "clip_ir_windows.csv")))}
    comp = {r["image"]: r for r in csv.DictReader(open(os.path.join(scores, "comp_windows.csv")))}
    for img, r in rows.items():
        out.append({**r, "win": r["setting"].split("=")[1],
                    "blip": blip.get(img, {}).get("blip_vqa", ""),
                    "clip": clip.get(img, {}).get("clip_score", ""),
                    "ir": clip.get(img, {}).get("image_reward", ""),
                    "comp": comp.get(img, {}).get("comp", "")})
    return out


def cell(rows, key, rng, draws):
    by_pair = defaultdict(list)
    for r in rows:
        if r.get(key, "") != "":
            by_pair[r["pair"]].append(float(r[key]))
    if not by_pair:
        return None
    vals = [v for vs in by_pair.values() for v in vs]
    lo, hi = two_level(by_pair, rng, draws)
    return {"mean": mean(vals), "lo": lo, "hi": hi, "pairs": len(by_pair), "images": len(vals)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=10_000)
    a = ap.parse_args()
    rows = load(a.scores)
    rng = random.Random(0)
    res = []
    for w, _ in SETTINGS:
        sub = [r for r in rows if r["win"] == w]
        for g, _ in GROUPS:
            c = cell([r for r in sub if r["group"] == g], "blip", rng, a.draws)
            if c:
                res.append({"setting": w, "group": g, "measure": "blip", **c})
        for k in ("comp", "clip", "ir"):
            c = cell(sub, k, rng, a.draws)
            if c:
                res.append({"setting": w, "group": "all", "measure": k, **c})
    os.makedirs(a.out, exist_ok=True)
    axes_cols = {"model": "SDXL", "method": "ours", "adapter": "v58-03-self_s030000", "sampler": "DDIM",
                 "steps": "50", "guidance": "7.5", "resolution": "1024x1024", "pair": "all", "seed": "9-16"}
    with open(os.path.join(a.out, "window_scores.csv"), "w", newline="") as f:
        cols = list(axes_cols) + ["group", "setting", "nfe", "measure", "mean", "lo", "hi", "pairs", "images"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in res:
            w.writerow({**axes_cols, "group": r["group"],
                        "setting": "adapter off" if r["setting"] == "off" else f"adapter_steps={r['setting']}",
                        "nfe": "", "measure": r["measure"], "mean": f"{r['mean']:.4f}", "lo": f"{r['lo']:.4f}",
                        "hi": f"{r['hi']:.4f}", "pairs": r["pairs"], "images": r["images"]})

    plt.rcParams.update({"font.size": 7, "font.family": "serif"})
    fig, axs = plt.subplots(1, 4, figsize=(6.9, 2.0), gridspec_kw={"width_ratios": [2.2, 1, 1, 1]})
    get = {(r["setting"], r["group"], r["measure"]): r for r in res}
    width = 0.2
    for gi, (g, glabel) in enumerate(GROUPS):
        for si, (s, _) in enumerate(SETTINGS):
            r = get.get((s, g, "blip"))
            if not r:
                continue
            x = gi + (si - 1.5) * width
            axs[0].bar(x, r["mean"], width, color=COLOURS[s], label=SETTINGS[si][1].replace("\n", " ") if gi == 0 else None)
            axs[0].errorbar(x, r["mean"], yerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]], color="k", lw=0.6, capsize=1.5)
    axs[0].set_xticks(range(len(GROUPS)), [g[1] for g in GROUPS])
    axs[0].set_ylabel("BLIP-VQA (0 to 1)")
    axs[0].set_ylim(0, 1)
    axs[0].set_title("a  composition, per held-out group", loc="left")
    axs[0].legend(frameon=False, fontsize=6, ncol=2, loc="upper left")
    for ax, (m, title, ylab) in zip(axs[1:], [("comp", "b  two animals found", "share of images"),
                                              ("clip", "c  CLIP score", "CLIP score"),
                                              ("ir", "d  ImageReward", "ImageReward")]):
        for si, (s, slabel) in enumerate(SETTINGS):
            r = get.get((s, "all", m))
            if not r:
                continue
            ax.bar(si, r["mean"], 0.7, color=COLOURS[s])
            ax.errorbar(si, r["mean"], yerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]], color="k", lw=0.6, capsize=1.5)
        ax.set_xticks(range(len(SETTINGS)), [s[1] for s in SETTINGS], fontsize=5.5)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylab)
        if m == "comp":
            ax.set_ylim(0, 1)
        ax.axhline(0, color="k", lw=0.4)
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, f"window_scores.{ext}"), dpi=300)
    for r in res:
        print(r["setting"], r["group"], r["measure"], {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in ("mean", "lo", "hi", "pairs", "images")})


if __name__ == "__main__":
    main()
