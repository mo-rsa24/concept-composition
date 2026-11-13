"""One row per rendered image of the paper-review runs, with protocol rule 9's axis columns.

Walks the run folders under the paper-review output root, reads each tile's JSON sidecar, and
writes rows.csv for the scorers: image, the full prompt the image is scored against (the
benchmark prompt as written, "{a} and {b}" or "{a} with {b}"; "{a} and {b}" for held-out
pairs), then model, method, adapter, sampler, steps, guidance, resolution, pair, seed, group,
setting, nfe. nfe is left empty for the wave 1 sidecars, which carry no count; the baseline
sidecars written by 1-lora-correction/scripts/review_baseline_cell.py carry their method, sampler, steps, guidance,
setting and counted nfe, and those values are kept. Rows whose image is already in --skip (a scores
file) are left out, so a later call picks up only new tiles.

Usage (any python3 on the cluster):
  python3 1-lora-correction/scripts/review_score_rows.py --root <paper_review dir> --census <review_step4/census>
      --repo <repo> --folders r1_joint r1_poe early_steps --out rows.csv [--skip scores_long.csv]
  Baseline folders name their set as a subfolder: --folders co3_default/r1 co3_default/r3 superdiff_default/r1 ...
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re

AXES = ["model", "method", "adapter", "sampler", "steps", "guidance", "resolution",
        "pair", "seed", "group", "setting", "nfe"]


def slug(c: str) -> str:
    return c.strip().replace(" ", "_")


def benchmark(repo):
    out = {}
    for line in open(os.path.join(repo, "1-lora-correction/scripts/review_benchmark_pairs.txt")):
        if line.startswith("#") or not line.strip():
            continue
        a, b, conn, group = line.strip().split("|")
        out[f"{slug(a)}__x__{slug(b)}"] = (f"{a} {conn} {b}", group)
    return out


def held_out(repo):
    """Concepts with their articles from scripts/review_held_out_pairs.txt; group from held-out-set.md."""
    groups = {}
    text = open(os.path.join(repo, "1-lora-correction/report/paper-review/held-out-set.md")).read()
    for g, pair in re.findall(r"^\| (hard|easy but unseen|not competing) \| ([^|]+?) \|", text, re.M):
        pair = re.sub(r"\s*\(.*\)$", "", pair)
        groups[tuple(p.strip() for p in pair.split("×"))] = g
    bare = lambda c: re.sub(r"^(a|an) ", "", c.strip())
    out = {}
    for line in open(os.path.join(repo, "scripts/review_held_out_pairs.txt")):
        if line.startswith("#") or not line.strip():
            continue
        a, b = [c.strip() for c in line.strip().split("|")]
        out[f"{slug(a)}__x__{slug(b)}"] = (f"{a} and {b}", groups[(bare(a), bare(b))])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--census", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--folders", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip")
    a = ap.parse_args()

    bench, held = benchmark(a.repo), held_out(a.repo)
    done = set()
    if a.skip and os.path.exists(a.skip):
        done = {r["image"] for r in csv.DictReader(open(a.skip))}

    rows, unknown = [], set()
    for folder in a.folders:
        base = a.census if folder == "census" else os.path.join(a.root, folder)
        for js in sorted(glob.glob(os.path.join(base, "*", "seed*", "*.json"))):
            img = js[:-5] + ".png"
            if not os.path.exists(img) or img in done:
                continue
            m = json.load(open(js))
            pair = m["pair"]
            if folder.startswith("r1_") or folder.endswith("/r1"):
                if pair not in bench:
                    unknown.add(pair); continue
                prompt, group = bench[pair]
            else:
                if pair not in held:
                    unknown.add(pair); continue
                prompt, group = held[pair]
            col = m.get("column")
            if col is None:  # a baseline sidecar from 1-lora-correction/scripts/review_baseline_cell.py
                rows.append({"image": img, "prompt": prompt, "run": folder, "model": m["model"],
                             "method": m["method"], "adapter": m["adapter"], "sampler": m["sampler"],
                             "steps": m["steps"], "guidance": m["guidance"],
                             "resolution": f"{m['resolution']}x{m['resolution']}", "pair": pair,
                             "seed": m["seed"], "group": group, "setting": m["setting"], "nfe": m["nfe"]})
                continue
            method = {"mono": "joint", "poe": "poe", "ours": "ours"}[col]
            adapter = m.get("tag", "-") if col == "ours" else "-"
            setting = f"adapter_steps={m['adapter_steps']}" if col == "ours" and "adapter_steps" in m else "-"
            size = m.get("size", [1024, 1024])
            rows.append({"image": img, "prompt": prompt, "run": folder, "model": "SDXL",
                         "method": method, "adapter": adapter, "sampler": "DDIM",
                         "steps": m.get("steps", 50), "guidance": m.get("guidance", 7.5),
                         "resolution": f"{size[0]}x{size[1]}", "pair": pair,
                         "seed": m["seed"], "group": group, "setting": setting, "nfe": ""})

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "prompt", "run"] + AXES)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {a.out}; {len(done)} skipped as scored; "
          f"{len(unknown)} pairs in no list: {sorted(unknown)[:8]}")
    for folder in a.folders:
        print(folder, sum(r["run"] == folder for r in rows))


if __name__ == "__main__":
    main()
