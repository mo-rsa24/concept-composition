"""Slot R14's result file from the per-image benchmark scores (compute job 13).

One row per method and benchmark, with protocol rule 9's twelve axis columns, in the layout
scripts/refresh_paper_slots.py's SCHEMA["R14"] pivots: group `geneval` carries geneval_two and
geneval_count, group `t2icompbench` carries t2i_spatial and t2i_complex. Each value is the
benchmark's own score over the images rendered: GenEval's fraction of images judged correct per task;
T2I-CompBench's mean per-image score from its own evaluator. A cell left empty is either not applicable
(PoE, ours and CO3 run the two-object prompts only, the user's decision of 2026-11-11) or not scored yet,
and the slot prints it as a dash.

    python3 scripts/review_bench_table.py --geneval geneval_images.csv [--t2i t2i_spatial_images.csv t2i_complex_images.csv] --out standard_benchmarks.csv

--t2i takes rows method,set,image,score with set t2i_spatial or t2i_complex.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

NAMES = {"joint": "Joint prompt", "poe": "PoE", "co3": "CO3", "ours": "Ours"}
ADAPTER = {"ours": "v58-03-self_s030000"}
SEEDS = {"geneval": "9-12", "t2icompbench": "9-16,59-60"}
AXES = ["model", "method", "adapter", "sampler", "steps", "guidance", "resolution",
        "pair", "seed", "group", "setting", "nfe"]
METRICS = ["geneval_two", "geneval_count", "t2i_spatial", "t2i_complex"]
GROUP_OF = {"geneval_two": "geneval", "geneval_count": "geneval", "t2i_spatial": "t2icompbench",
            "t2i_complex": "t2icompbench"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geneval", required=True)
    ap.add_argument("--t2i", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    vals = defaultdict(list)  # (method, metric) -> per-image scores
    for r in csv.DictReader(open(a.geneval)):
        if r["correct"] != "":
            vals[(r["method"], {"two_object": "geneval_two", "counting": "geneval_count"}[r["tag"]])].append(float(r["correct"]))
    for path in a.t2i:
        for r in csv.DictReader(open(path)):
            if r["score"] != "":
                vals[(r["method"], r["set"])].append(float(r["score"]))

    rows = []
    for method in NAMES:
        for group in ("geneval", "t2icompbench"):
            row = {"model": "SDXL", "method": NAMES[method], "adapter": ADAPTER.get(method, "-"),
                   "sampler": "DDIM", "steps": 50, "guidance": 7.5, "resolution": "1024x1024", "pair": "all",
                   "seed": SEEDS[group], "group": group, "setting": "-", "nfe": ""}
            for m in METRICS:
                v = vals.get((method, m)) if GROUP_OF[m] == group else None
                row[m] = f"{sum(v) / len(v):.4f}" if v else ""
                if GROUP_OF[m] == group:
                    row[f"{m}_n"] = len(v) if v else 0
            rows.append(row)
    fields = AXES + METRICS + [f"{m}_n" for m in METRICS]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    print(open(a.out).read())


if __name__ == "__main__":
    main()
