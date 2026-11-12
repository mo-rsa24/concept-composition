"""Slot R14's result file (Table 2) from the per-image benchmark scores.

One row per method and benchmark, with protocol rule 9's twelve axis columns, in the layout
scripts/refresh_paper_slots.py's SCHEMA["R14"] pivots: group `geneval` carries geneval_two,
geneval_colorattr and geneval_position; group `t2icompbench` carries t2i_color, t2i_shape and
t2i_texture. Each value is the benchmark's own score over the images scored: GenEval's fraction of
images judged correct per task, T2I-CompBench's mean BLIP-VQA score per category. A set not scored
yet is left empty and the slot prints it as a dash; `<set>_n` beside it counts the images it rests on.

    python3 scripts/review_bench_table.py --images <dir>/*_images.csv --out standard_benchmarks.csv

Each per-image CSV has columns method, set and either correct (GenEval) or score (T2I-CompBench).
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

NAMES = {"joint": "Joint prompt", "co3": "CO3", "poe": "PoE", "ours": "Ours"}
ADAPTER = {"ours": "v58-03-self_s030000"}
SEEDS = {"geneval": "9-12", "t2icompbench": "9-10"}
AXES = ["model", "method", "adapter", "sampler", "steps", "guidance", "resolution",
        "pair", "seed", "group", "setting", "nfe"]
METRICS = ["geneval_two", "geneval_colorattr", "geneval_position", "t2i_color", "t2i_shape", "t2i_texture"]
GROUP_OF = {m: ("geneval" if m.startswith("geneval") else "t2icompbench") for m in METRICS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    vals = defaultdict(list)  # (method, set) -> per-image scores
    for path in a.images:
        for r in csv.DictReader(open(path)):
            v = r.get("correct", r.get("score", ""))
            if v != "":
                vals[(r["method"], r["set"])].append(float(v))

    rows = []
    for method in NAMES:
        for group in ("geneval", "t2icompbench"):
            row = {"model": "SDXL", "method": NAMES[method], "adapter": ADAPTER.get(method, "-"),
                   "sampler": "DDIM", "steps": 50, "guidance": 7.5, "resolution": "1024x1024", "pair": "all",
                   "seed": SEEDS[group], "group": group, "setting": "-", "nfe": ""}
            for m in METRICS:
                if GROUP_OF[m] == group:
                    v = vals.get((method, m))
                    row[m] = f"{sum(v) / len(v):.4f}" if v else ""
                    row[f"{m}_n"] = len(v) if v else 0
            if method == "co3":  # as its sidecars record: CO3's own corrector sampler under CFG++ at 0.8
                row.update(sampler="DDIM with CO3 corrector, CFG++", guidance=0.8)
            rows.append(row)
    fields = AXES + METRICS + [f"{m}_n" for m in METRICS]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    print(open(a.out).read())


if __name__ == "__main__":
    main()
