"""T2I-CompBench's own BLIP-VQA attribute-binding evaluator on one Table 2 set (slot R14), all four methods.

Sets: t2i_color, t2i_shape, t2i_texture (100 two-concept prompts each by the fixed draw in
scripts/review_bench_sets.py, seeds 9 and 10). BLIPvqa_eval/BLIP_vqa.py runs unchanged at its
defaults: it splits each prompt into noun phrases with spaCy, asks BLIP-VQA "<noun phrase>?" for each,
and scores an image by the product of the yes-probabilities. The benchmark's score for the category is
the mean over images.

BLIP_vqa.py reads <work>/samples/<prompt>_<n>.png, takes the prompt from the text before the first
underscore, and numbers its answers by the files' rank when sorted by n. So each method gets its own
work folder and n runs 0, 1, 2, ... over that method's images, which makes the answer's question_id
equal to n.

Renders, matched to their benchmark line by the concepts in their sidecar (PoE, ours, CO3) or by the
index (joint). A 0-byte file is a relay placeholder and is skipped and counted:
    joint  <OUT>/bench_joint/<set>/<idx>/seedNN/joint.png
    poe    <OUT>/bench_poe/<set>/<pair>/seedNN/poe.png
    ours   <OUT>/bench_ours/<set>/<pair>/seedNN/ours_<tag>_w50.png
    co3    <OUT>/co3_default/<set>/<pair>/seedNN/co3.png

Usage (t2icomp env, one free card):
  python scripts/review_bench_t2i.py --set t2i_color --work <fresh dir> --out t2i_color_images.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from review_bench_geneval import METHODS, renders  # noqa: E402
from review_bench_sets import SETS, load  # noqa: E402

T2I = os.environ.get("T2I_COMPBENCH", "/datasets/mmolefe/repos/T2I-CompBench")
BLIP_SETS = ["t2i_color", "t2i_shape", "t2i_texture"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=BLIP_SETS)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    a = ap.parse_args()
    work = os.path.abspath(a.work)
    if os.path.exists(work):
        sys.exit(f"{work} exists; give a fresh --work folder")
    kept, _ = load(a.set)
    expected = len(kept) * len(SETS[a.set]["seeds"])

    rows = []
    for method in a.methods:
        found, empty = renders(a.set, method, kept)
        print(f"{method}: {len(found)} of {expected} images ({empty} skipped: empty or no sidecar)", flush=True)
        if not found:
            continue
        mdir = os.path.join(work, method)
        os.makedirs(os.path.join(mdir, "samples"))
        found.sort(key=lambda t: (t[0]["idx"], t[1]))
        for n, (item, seed, png) in enumerate(found):
            assert "_" not in item["prompt"] and "/" not in item["prompt"], item["prompt"]
            os.symlink(png, os.path.join(mdir, "samples", f"{item['prompt']}_{n:06d}.png"))
        t0 = time.time()
        print("+ BLIP_vqa.py --out_dir", mdir, flush=True)
        subprocess.run([sys.executable, "BLIP_vqa.py", "--out_dir", mdir], cwd=os.path.join(T2I, "BLIPvqa_eval"), check=True)
        sec = time.time() - t0
        res = json.load(open(os.path.join(mdir, "annotation_blip", "vqa_result.json")))
        by_id = {int(x["question_id"]): float(x["answer"]) for x in res}
        assert sorted(by_id) == list(range(len(found))), (len(by_id), len(found))
        for n, (item, seed, png) in enumerate(found):
            rows.append({"method": method, "set": a.set, "idx": item["idx"], "prompt": item["prompt"],
                         "seed": seed, "image": png, "score": f"{by_id[n]:.4f}"})
        v = [by_id[n] for n in range(len(found))]
        print(f"{a.set} {method:6s} {sum(v) / len(v):.4f}  ({len(v)} images; {sec:.0f} s incl. model load, "
              f"{sec / len(v):.2f} s/image)", flush=True)

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "set", "idx", "prompt", "seed", "image", "score"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
