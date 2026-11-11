"""GenEval's own evaluator on the slot R14 renders (compute job 13): two-object and counting, per method.

For each method this lays the renders out the way GenEval's evaluation/evaluate_images.py reads its own
generations (<dir>/<prompt index>/metadata.jsonl holding the benchmark's line for that prompt unchanged,
<dir>/<prompt index>/samples/<seed>.png), runs that script unchanged at its defaults (Mask2Former Swin-S
on COCO; confidence 0.3, 0.9 for counting), and writes one row per image. GenEval's score for a task is
the fraction of its images judged correct (evaluation/summary_scores.py).

Renders are found through their sidecars, and each is matched to its benchmark line by prompt text:
    joint  <OUT>/bench_joint/geneval_{two,count}/<idx>/seedNN/joint.png        ("prompt")
    poe    <OUT>/bench_poe/geneval_two/<pair>/seedNN/poe.png                   ("prompt_a" and "prompt_b")
    ours   <OUT>/bench_ours/geneval_two/<pair>/seedNN/ours_<tag>_w50.png
    co3    <OUT>/co3_default/geneval_two/<pair>/seedNN/co3.png

Usage (geneval env, an RTX 3090, as protocol.md's scorer table says):
  python scripts/review_bench_geneval.py --work <fresh dir> --out geneval_images.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys

OUT = os.environ.get("REVIEW_BASELINE_OUT", "/datasets/mmolefe/poe_repair_min/outputs/paper_review")
GENEVAL = os.environ.get("GENEVAL", "/datasets/mmolefe/repos/geneval")
MMDET = os.environ.get("MMDET", "/datasets/mmolefe/repos/mmdetection")
MODELS = os.environ.get("GENEVAL_MODELS", "/datasets/mmolefe/repos/geneval_models")
CONFIG = os.path.join(MMDET, "configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py")
META = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data/prompt-sets/geneval/evaluation_metadata.jsonl")
METHODS = {
    "joint": ["bench_joint/geneval_two/*/seed*/joint.json", "bench_joint/geneval_count/*/seed*/joint.json"],
    "poe": ["bench_poe/geneval_two/*/seed*/poe.json"],
    "ours": ["bench_ours/geneval_two/*/seed*/ours_*_w50.json"],
    "co3": ["co3_default/geneval_two/*/seed*/co3.json"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    a = ap.parse_args()
    work = os.path.abspath(a.work)
    if os.path.exists(work):
        sys.exit(f"{work} exists; give a fresh --work folder")

    by_prompt = {}
    for idx, line in enumerate(open(META)):
        if line.strip():
            m = json.loads(line)
            if m["tag"] in ("two_object", "counting"):
                by_prompt[m["prompt"]] = (idx, line.strip(), m["tag"])

    rows = []
    for method in a.methods:
        found = 0
        for pattern in METHODS[method]:
            for js in sorted(glob.glob(os.path.join(OUT, pattern))):
                png = js[:-5] + ".png"
                if not os.path.exists(png):
                    continue
                side = json.load(open(js))
                prompt = side.get("prompt") if method == "joint" else f"{side['prompt_a']} and {side['prompt_b']}"
                if prompt not in by_prompt:
                    sys.exit(f"{js}: prompt {prompt!r} is not a GenEval two_object or counting prompt")
                idx, line, tag = by_prompt[prompt]
                d = os.path.join(work, method, f"{idx:05d}")
                if not os.path.isdir(d):
                    os.makedirs(os.path.join(d, "samples"))
                    open(os.path.join(d, "metadata.jsonl"), "w").write(line)
                img = os.path.join(d, "samples", f"seed{int(side['seed']):02d}.png")
                os.symlink(png, img)
                rows.append({"method": method, "tag": tag, "idx": idx, "prompt": prompt, "seed": side["seed"],
                             "image": png, "link": img})
                found += 1
        print(f"{method}: {found} images", flush=True)
        if found:
            subprocess.run([sys.executable, os.path.join(GENEVAL, "evaluation", "evaluate_images.py"),
                            os.path.join(work, method), "--outfile", os.path.join(work, f"{method}_results.jsonl"),
                            "--model-config", CONFIG, "--model-path", MODELS], check=True)

    result = {}
    for method in a.methods:
        path = os.path.join(work, f"{method}_results.jsonl")
        if os.path.exists(path):
            for line in open(path):
                x = json.loads(line)
                result[x["filename"]] = (int(bool(x["correct"])), x.get("reason", ""))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "tag", "idx", "prompt", "seed", "image", "correct", "reason"])
        w.writeheader()
        for r in rows:
            c, why = result.get(r.pop("link"), ("", "not evaluated"))
            w.writerow({**r, "correct": c, "reason": why})
    for method in a.methods:
        for tag in ("two_object", "counting"):
            vals = [int(r["correct"]) for r in csv.DictReader(open(a.out))
                    if r["method"] == method and r["tag"] == tag and r["correct"] != ""]
            if vals:
                print(f"{method:6s} {tag:11s} {sum(vals) / len(vals):.4f}  ({sum(vals)} / {len(vals)} images)")


if __name__ == "__main__":
    main()
