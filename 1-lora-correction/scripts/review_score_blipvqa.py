"""BLIP-VQA for the paper-review renders, through T2I-CompBench's own code, unchanged.

T2I-CompBench (Karine-Huang/T2I-CompBench, BLIPvqa_eval/BLIP_vqa.py) splits the prompt
into spaCy noun phrases, asks BLIP base VQA (model_base_vqa_capfilt_large, 480 px)
"<phrase>?" for each, and multiplies the yes-probabilities. This driver only stages
images under the file names that script expects (<prompt>_<n>.png), runs it in the
t2icomp env, and maps the per-image answers back to the input rows.

Passes beyond a prompt's noun-phrase count ask an empty question that the script scores as 1, so
--np-num 3 equals the default 8 on the paper-review prompts (323 have 2 noun phrases, one has 1, one 3).

Usage (t2icomp env, non-Blackwell GPU):
  python 1-lora-correction/scripts/review_score_blipvqa.py --in rows.csv --out scores.csv --work <dir>
rows.csv has columns image,prompt.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

T2I = os.environ.get("T2I_COMPBENCH", "/datasets/mmolefe/repos/T2I-CompBench")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--np-num", type=int, default=8,
                    help="question passes; any value at least the most noun phrases in a prompt gives the same scores")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.inp)))
    work = os.path.abspath(a.work)
    if os.path.exists(os.path.join(work, "samples")):
        sys.exit(f"{work}/samples exists; give a fresh --work folder")
    os.makedirs(os.path.join(work, "samples"))
    for n, r in enumerate(rows):
        assert "_" not in r["prompt"], r["prompt"]  # the script splits the file name on "_"
        os.symlink(os.path.abspath(r["image"]), os.path.join(work, "samples", f"{r['prompt']}_{n}.png"))

    subprocess.run([sys.executable, "BLIP_vqa.py", "--out_dir", work, "--np_num", str(a.np_num)],
                   cwd=os.path.join(T2I, "BLIPvqa_eval"), check=True)

    # The script sorts samples by the trailing index, so question_id n is row n.
    res = json.load(open(os.path.join(work, "annotation_blip", "vqa_result.json")))
    by_id = {int(x["question_id"]): float(x["answer"]) for x in res}
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "prompt", "blip_vqa"])
        for n, r in enumerate(rows):
            w.writerow([r["image"], r["prompt"], f"{by_id[n]:.4f}"])
    print(open(a.out).read())


if __name__ == "__main__":
    main()
