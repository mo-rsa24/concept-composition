"""Rows file for the early-steps window renders, so the paper-review scorers can read them.

Each window render (adapter on steps 0 to 10 or 0 to 25 of 50) has a sidecar JSON beside it naming
its pair, seed and adapter_steps. The prompt scored and the held-out group come from the r3_ours rows
already scored for Table 3, so the window images are scored against the same joint-prompt text as
the all-50-steps images. Writes rows_windows.csv in the scores folder, in the columns every
scorer reads.

Usage (on mscluster, co3 python):
  python 1-lora-correction/scripts/review_window_rows.py
"""
from __future__ import annotations

import csv
import glob
import json
import os

OUT = "/datasets/mmolefe/poe_repair_min/outputs/paper_review"
SCORES = f"{OUT}/scores"
COLS = ["image", "prompt", "run", "model", "method", "adapter", "sampler", "steps", "guidance",
        "resolution", "pair", "seed", "group", "setting", "nfe"]


def main() -> None:
    ref = {}
    for f in sorted(glob.glob(f"{SCORES}/rows_*.csv")):
        for r in csv.DictReader(open(f)):
            if r.get("run") == "r3_ours":
                ref[r["pair"]] = (r["prompt"], r["group"])
    rows = []
    for side in sorted(glob.glob(f"{OUT}/early_steps/*/seed*/ours_*_w*.json")):
        d = json.load(open(side))
        png = side[:-5] + ".png"
        if not os.path.isfile(png) or d["pair"] not in ref:
            continue
        prompt, group = ref[d["pair"]]
        rows.append({"image": png, "prompt": prompt, "run": f"r3_window{d['adapter_steps']}",
                     "model": "SDXL", "method": "ours", "adapter": d["tag"], "sampler": "DDIM",
                     "steps": str(d["steps"]), "guidance": str(d["guidance"]), "resolution": "1024x1024",
                     "pair": d["pair"], "seed": str(d["seed"]), "group": group,
                     "setting": f"adapter_steps={d['adapter_steps']}", "nfe": ""})
    with open(f"{SCORES}/rows_windows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    by = {}
    for r in rows:
        by[r["setting"]] = by.get(r["setting"], 0) + 1
    print(f"rows_windows.csv: {len(rows)} rows {by}; pairs {len({r['pair'] for r in rows})}; reference pairs {len(ref)}")


if __name__ == "__main__":
    main()
