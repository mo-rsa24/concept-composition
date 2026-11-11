"""GenEval's two-object check for the paper-review renders, through GenEval's own evaluator.

GenEval (djghosh13/geneval, evaluation/evaluate_images.py) runs Mask2Former Swin-S on COCO,
keeps detections above 0.3 confidence, and passes a two_object image when each named COCO
class is found at least once. This driver only lays images out the way that script reads
them (<n>/metadata.jsonl, <n>/samples/0000.png) with tag two_object, runs it unchanged in
the geneval env at its default options, and maps each result back to its input row.

Only prompts whose two concepts name COCO classes can be judged; the class for each concept
comes from the rows file (class_a, class_b), and an empty class means the row is skipped and
written with geneval_two_object empty and the reason "not a COCO class".

Usage (geneval env, non-Blackwell GPU):
  python scripts/review_score_geneval.py --in rows.csv --out scores.csv --work <dir>
rows.csv has columns image,prompt,class_a,class_b.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys

GENEVAL = os.environ.get("GENEVAL", "/datasets/mmolefe/repos/geneval")
MMDET = os.environ.get("MMDET", "/datasets/mmolefe/repos/mmdetection")
MODELS = os.environ.get("GENEVAL_MODELS", "/datasets/mmolefe/repos/geneval_models")
CONFIG = os.path.join(MMDET, "configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.inp)))
    work = os.path.abspath(a.work)
    if os.path.exists(work):
        sys.exit(f"{work} exists; give a fresh --work folder")
    judged = {}
    for n, r in enumerate(rows):
        if not (r["class_a"] and r["class_b"]):
            continue
        d = os.path.join(work, f"{n:05d}")
        os.makedirs(os.path.join(d, "samples"))
        meta = {"tag": "two_object", "prompt": r["prompt"],
                "include": [{"class": r["class_a"], "count": 1}, {"class": r["class_b"], "count": 1}]}
        json.dump(meta, open(os.path.join(d, "metadata.jsonl"), "w"))
        img = os.path.join(d, "samples", "0000.png")
        os.symlink(os.path.abspath(r["image"]), img)
        judged[img] = n

    res_path = os.path.join(work, "results.jsonl")
    subprocess.run([sys.executable, os.path.join(GENEVAL, "evaluation", "evaluate_images.py"), work,
                    "--outfile", res_path, "--model-config", CONFIG, "--model-path", MODELS], check=True)

    out = {}
    for line in open(res_path):
        x = json.loads(line)
        out[judged[x["filename"]]] = (int(bool(x["correct"])), x.get("reason", ""), x.get("details", ""))
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "prompt", "class_a", "class_b", "geneval_two_object", "geneval_reason", "geneval_detected"])
        for n, r in enumerate(rows):
            if n in out:
                c, why, det = out[n]
                w.writerow([r["image"], r["prompt"], r["class_a"], r["class_b"], c, why, det])
            else:
                w.writerow([r["image"], r["prompt"], r["class_a"], r["class_b"], "", "not a COCO class", ""])
    print(open(a.out).read())


if __name__ == "__main__":
    main()
