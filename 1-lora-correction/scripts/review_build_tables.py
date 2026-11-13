"""Tables 1 and 2 (slots R1 and R3) from the paper-review score files.

Joins every rows_*.csv (image, prompt, rule 9 axis columns) with the score files keyed by image
(clip_ir_*.csv: clip_score, image_reward; blip_*.csv: blip_vqa; comp.csv: comp) into one long
file, one line per image. Then:

  R1 (Table 1): the benchmark runs (r1_*, baseline r1 folders), mean CLIP, BLIP-VQA and ImageReward
      per method and set, and Comp. for the animals set; one CSV row per method and set.
  R3 (Table 2): the held-out runs (census joint and PoE, r3_ours, baseline r3 folders), BLIP-VQA per
      method and group, 95% interval by resampling pairs, then seeds within each pair.
  The bar (protocol, "The bar for R1"): ours minus PoE in BLIP-VQA per prompt and seed, averaged
      per prompt, then per set; 95% interval by resampling prompts, then seeds. Written to the
      sidecar with its verdict, never into a table.

A cell is marked partial in the sidecar when fewer images were scored than the design has
(R1: 4 seeds per prompt; R3: 8 seeds per pair). Nothing is imputed.

Usage:
  python3 1-lora-correction/scripts/review_build_tables.py --scores <scores dir> --out <results dir>
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
from collections import defaultdict
from statistics import mean

AXES = ["model", "method", "adapter", "sampler", "steps", "guidance", "resolution",
        "pair", "seed", "group", "setting", "nfe"]
NAME = {"joint": "Joint prompt", "poe": "PoE", "co3": "CO3", "tome": "ToMe", "langevin": "PoE + Langevin", "ours": "Ours"}
ORDER = ["joint", "poe", "co3", "tome", "langevin", "ours"]
SETS = {"animals": 66, "animals_objects": 144, "objects": 66}
GROUPS = {"hard": ("hard", 3), "easy but unseen": ("easy", 29), "not competing": ("not_competing", 21)}
DRAWS = 10_000
# The bar for R1, as committed in 1-lora-correction/report/paper-review/protocol.md before any score: the interval on
# ours minus PoE in BLIP-VQA, per set. Constants here so a change shows in a diff.
BAR_LEVEL = 0.95
BAR_ZERO = 0.0
MAX_MISSING = 0.10


def read(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        out += list(csv.DictReader(open(f)))
    return out


def two_level(groups: dict, rng: random.Random, draws: int) -> tuple[float, float]:
    """groups: {outer key: [values]}; resample outer keys, then values within each; mean of all."""
    keys = list(groups)
    stats = []
    for _ in range(draws):
        vals = []
        for k in (rng.choice(keys) for _ in keys):
            v = groups[k]
            vals += [rng.choice(v) for _ in v]
        stats.append(mean(vals))
    stats.sort()
    lo = stats[int((1 - BAR_LEVEL) / 2 * draws)]
    hi = stats[int((1 + BAR_LEVEL) / 2 * draws) - 1]
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=DRAWS)
    a = ap.parse_args()
    S = a.scores
    rng = random.Random(0)

    rows = {r["image"]: r for r in read(os.path.join(S, "rows_batch*.csv"))}
    clip = {r["image"]: r for r in read(os.path.join(S, "clip_ir_batch*.csv"))}
    blip = {r["image"]: r for r in read(os.path.join(S, "blip_batch*.csv"))}
    comp = {r["image"]: r for r in read(os.path.join(S, "comp*.csv"))}

    long = []
    for img, r in rows.items():
        if r["run"] == "early_steps":
            continue
        rec = dict(r)
        rec["clip"] = clip.get(img, {}).get("clip_score", "")
        rec["ir"] = clip.get(img, {}).get("image_reward", "")
        rec["blip"] = blip.get(img, {}).get("blip_vqa", "")
        rec["comp"] = comp.get(img, {}).get("comp", "")
        rec["table"] = "R1" if r["run"].startswith("r1") or r["run"].endswith("/r1") else "R3"
        long.append(rec)
    os.makedirs(a.out, exist_ok=True)
    cols = ["image", "prompt", "run", "table"] + AXES + ["clip", "blip", "ir", "comp"]
    with open(os.path.join(S, "scores_long.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(long)

    side = {"r1": {}, "r3": {}, "bar": {}}

    # Table 1
    t1 = []
    for m in [m for m in ORDER if any(x["method"] == m and x["table"] == "R1" for x in long)]:
        for s, n_prompts in SETS.items():
            cell = [x for x in long if x["table"] == "R1" and x["method"] == m and x["group"] == s]
            out = {k: "" for k in AXES}
            out.update({"model": "SDXL", "method": NAME[m], "sampler": "DDIM", "steps": "50", "guidance": "7.5",
                        "resolution": "1024x1024", "pair": "all", "seed": "9-12", "group": s, "setting": "-",
                        "adapter": cell[0]["adapter"] if cell else "-"})
            counts = {}
            for k in ("clip", "blip", "ir") + (("comp",) if s == "animals" else ()):
                v = [float(x[k]) for x in cell if x[k] != ""]
                out[k] = f"{mean(v):.4f}" if v else ""
                counts[k] = len(v)
            t1.append(out)
            side["r1"][f"{NAME[m]} / {s}"] = {"expected": n_prompts * 4, "scored": counts,
                                              "partial": any(c < n_prompts * 4 for c in counts.values())}
    with open(os.path.join(a.out, "main_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AXES + ["clip", "blip", "ir", "comp"])
        w.writeheader(); w.writerows(t1)

    # Table 2
    t2 = []
    for m in [m for m in ORDER if any(x["method"] == m and x["table"] == "R3" for x in long)]:
        for g, (key, n_pairs) in GROUPS.items():
            by_pair = defaultdict(list)
            for x in long:
                if x["table"] == "R3" and x["method"] == m and x["group"] == g and x["blip"] != "":
                    by_pair[x["pair"]].append(float(x["blip"]))
            out = {k: "" for k in AXES}
            out.update({"model": "SDXL", "method": NAME[m], "sampler": "DDIM", "steps": "50", "guidance": "7.5",
                        "resolution": "1024x1024", "pair": "all", "seed": "9-16", "group": key, "setting": "-",
                        "adapter": "v58-03-self_s030000" if m == "ours" else "-"})
            if by_pair:
                allv = [v for vs in by_pair.values() for v in vs]
                lo, hi = two_level(by_pair, rng, a.draws)
                out["blip"] = f"{mean(allv):.4f}"
                out["blip_ci"] = f"{(hi - lo) / 2:.4f}"
            t2.append(out)
            n = sum(len(v) for v in by_pair.values())
            side["r3"][f"{NAME[m]} / {g}"] = {"expected": n_pairs * 8, "scored": n, "pairs": len(by_pair),
                                              "partial": n < n_pairs * 8}
    with open(os.path.join(a.out, "held_out_ablation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AXES + ["blip", "blip_ci"])
        w.writeheader(); w.writerows(t2)

    # The bar: ours minus PoE, paired by prompt and seed
    idx = {(x["method"], x["pair"], x["seed"]): x for x in long if x["table"] == "R1" and x["blip"] != ""}
    verdicts = {}
    for s, n_prompts in SETS.items():
        by_prompt = defaultdict(list)
        for (m, pair, seed), x in idx.items():
            if m == "ours" and x["group"] == s and ("poe", pair, seed) in idx:
                by_prompt[pair].append(float(x["blip"]) - float(idx[("poe", pair, seed)]["blip"]))
        n = sum(len(v) for v in by_prompt.values())
        if not by_prompt:
            side["bar"][s] = {"pairs_scored": 0}
            verdicts[s] = "no data"
            continue
        lo, hi = two_level(by_prompt, rng, a.draws)
        d = mean([v for vs in by_prompt.values() for v in vs])
        missing = 1 - n / (n_prompts * 4)
        verdicts[s] = "above" if lo > BAR_ZERO else ("below" if hi < BAR_ZERO else "covers")
        side["bar"][s] = {"mean_diff": round(d, 4), "ci": [round(lo, 4), round(hi, 4)],
                          "paired_images": n, "expected": n_prompts * 4, "missing_share": round(missing, 4)}
    miss = any(side["bar"].get(s, {}).get("missing_share", 1) > MAX_MISSING for s in SETS)
    if miss:
        verdict = "INCONCLUSIVE (more than 10% of cells missing)"
    elif all(v == "above" for v in verdicts.values()):
        verdict = "SUPPORTS"
    elif all(v == "covers" for v in verdicts.values()) or any(v == "below" for v in verdicts.values()):
        verdict = "NULL"
    else:
        verdict = "INCONCLUSIVE"
    side["bar"]["verdict"] = verdict
    side["draws"] = a.draws
    with open(os.path.join(a.out, "tables_sidecar.json"), "w") as f:
        json.dump(side, f, indent=1)
    print(f"{len(long)} images in scores_long.csv; Table 1 {len(t1)} rows; Table 2 {len(t2)} rows; bar: {verdict}")


if __name__ == "__main__":
    main()
