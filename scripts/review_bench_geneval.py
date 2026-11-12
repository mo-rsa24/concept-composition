"""GenEval's own evaluator on one Table 2 set (slot R14), all four methods.

Sets: geneval_two (two_object), geneval_colorattr (color_attr), geneval_position (position), as
scripts/review_bench_sets.py defines them. For each method this lays the renders out the way GenEval's
evaluation/evaluate_images.py reads its own generations (<dir>/<prompt index>/metadata.jsonl holding
the benchmark's line for that prompt unchanged, <dir>/<prompt index>/samples/<seed>.png; the
evaluator skips any file not named digits.png), runs that script unchanged at its defaults
(Mask2Former Swin-S on COCO, confidence 0.3; CLIP ViT-L-14 for colours; position threshold 0.1), and
writes one row per image. GenEval's score for a task is the fraction of its images judged correct
(evaluation/summary_scores.py).

Renders, matched to their benchmark line by the concepts in their sidecar (PoE, ours, CO3) or by the
index (joint). A 0-byte file is a relay placeholder and is skipped and counted:
    joint  <OUT>/bench_joint/<set>/<idx>/seedNN/joint.png
    poe    <OUT>/bench_poe/<set>/<pair>/seedNN/poe.png
    ours   <OUT>/bench_ours/<set>/<pair>/seedNN/ours_<tag>_w50.png
    co3    <OUT>/co3_default/<set>/<pair>/seedNN/co3.png

Usage (geneval env, one free card):
  python scripts/review_bench_geneval.py --set geneval_two --work <fresh dir> --out geneval_two_images.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from review_bench_sets import SETS, load, split  # noqa: E402

OUT = os.environ.get("REVIEW_BASELINE_OUT", "/datasets/mmolefe/poe_repair_min/outputs/paper_review")
GENEVAL = os.environ.get("GENEVAL", "/datasets/mmolefe/repos/geneval")
MMDET = os.environ.get("MMDET", "/datasets/mmolefe/repos/mmdetection")
MODELS = os.environ.get("GENEVAL_MODELS", "/datasets/mmolefe/repos/geneval_models")
CONFIG = os.path.join(MMDET, "configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py")
GENEVAL_SETS = [k for k, v in SETS.items() if v["bench"] == "geneval" and k != "geneval_count"]
METHODS = ["joint", "poe", "ours", "co3"]
PATTERN = {
    "joint": "bench_joint/{set}/*/seed*/joint.png",
    "poe": "bench_poe/{set}/*/seed*/poe.png",
    "ours": "bench_ours/{set}/*/seed*/ours_*_w50.png",
    "co3": "co3_default/{set}/*/seed*/co3.png",
}


def renders(name: str, method: str, kept: list[dict]) -> tuple[list[tuple[dict, int, str]], int]:
    """([(benchmark item, seed, png)], count skipped as 0-byte placeholders or with no sidecar yet)."""
    by_idx = {x["idx"]: x for x in kept}
    by_pair = {split(x)[:2]: x for x in kept}
    seeds = set(SETS[name]["seeds"])
    found, empty, seen = [], 0, set()
    for png in sorted(glob.glob(os.path.join(OUT, PATTERN[method].format(set=name)))):
        if os.path.getsize(png) == 0 or not os.path.exists(png[:-4] + ".json"):
            empty += 1
            continue
        side = json.load(open(png[:-4] + ".json"))
        seed = int(side["seed"])
        if method == "joint":
            item = by_idx.get(int(side["idx"]))
            if item is None or item["prompt"] != side["prompt"]:
                sys.exit(f"{png}: idx {side['idx']} prompt {side['prompt']!r} is not a kept {name} prompt")
        else:
            item = by_pair.get((side["prompt_a"], side["prompt_b"]))
            if item is None:
                sys.exit(f"{png}: concepts {side['prompt_a']!r} | {side['prompt_b']!r} are not a kept {name} pair")
        if seed not in seeds:
            continue
        key = (item["idx"], seed)
        if key in seen:
            sys.exit(f"{png}: a second {method} render of prompt {item['idx']} seed {seed}")
        seen.add(key)
        found.append((item, seed, png))
    return found, empty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=GENEVAL_SETS)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    a = ap.parse_args()
    work = os.path.abspath(a.work)
    if os.path.exists(work):
        sys.exit(f"{work} exists; give a fresh --work folder")
    kept, _ = load(a.set)
    lines = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                              "data/prompt-sets/geneval/evaluation_metadata.jsonl")).read().splitlines()
    expected = len(kept) * len(SETS[a.set]["seeds"])

    rows, timing = [], {}
    for method in a.methods:
        found, empty = renders(a.set, method, kept)
        for item, seed, png in found:
            d = os.path.join(work, method, f"{item['idx']:05d}")
            if not os.path.isdir(d):
                os.makedirs(os.path.join(d, "samples"))
                open(os.path.join(d, "metadata.jsonl"), "w").write(lines[item["idx"]].strip())
            link = os.path.join(d, "samples", f"{seed:04d}.png")
            os.symlink(png, link)
            rows.append({"method": method, "set": a.set, "tag": item["meta"]["tag"], "idx": item["idx"],
                         "prompt": item["prompt"], "seed": seed, "image": png, "link": link})
        print(f"{method}: {len(found)} of {expected} images ({empty} skipped: empty or no sidecar)", flush=True)
        if found:
            t0 = time.time()
            subprocess.run([sys.executable, os.path.join(GENEVAL, "evaluation", "evaluate_images.py"),
                            os.path.join(work, method), "--outfile", os.path.join(work, f"{method}_results.jsonl"),
                            "--model-config", CONFIG, "--model-path", MODELS], check=True)
            timing[method] = (time.time() - t0, len(found))

    result = {}
    for method in a.methods:
        path = os.path.join(work, f"{method}_results.jsonl")
        if os.path.exists(path):
            for line in open(path):
                x = json.loads(line)
                result[x["filename"]] = (int(bool(x["correct"])), x.get("reason", ""))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "set", "tag", "idx", "prompt", "seed", "image", "correct", "reason"])
        w.writeheader()
        for r in rows:
            c, why = result.get(r.pop("link"), ("", "not evaluated"))
            w.writerow({**r, "correct": c, "reason": why})
    got = list(csv.DictReader(open(a.out)))
    for method in a.methods:
        vals = [int(r["correct"]) for r in got if r["method"] == method and r["correct"] != ""]
        unscored = sum(1 for r in got if r["method"] == method and r["correct"] == "")
        if vals:
            sec, n = timing[method]
            print(f"{a.set} {method:6s} {sum(vals) / len(vals):.4f}  ({sum(vals)} / {len(vals)} images, "
                  f"{unscored} unscored; {sec:.0f} s incl. model load, {sec / n:.2f} s/image)")


if __name__ == "__main__":
    main()
