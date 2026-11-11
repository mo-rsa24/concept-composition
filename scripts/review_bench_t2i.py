"""T2I-CompBench's own evaluators on the slot R14 joint renders (compute job 13): spatial and complex.

spatial  UniDet_eval/2D_spatial_eval.py, unchanged: UniDet (RS200) finds the two objects and scores
         their left/right/top/bottom/near relation; one score per image.
complex  3_in_1_eval/3_in_1.py, unchanged, over BLIPvqa_eval/BLIP_vqa.py (attributes),
         UniDet_eval/2D_spatial_eval.py --complex (spatial) and CLIPScore_eval/CLIP_similarity.py
         --complex (actions): per prompt, the mean of the scores its kind of relation calls for.

Every scorer reads <work>/samples/<prompt>_<n>.png and orders its results by n, and 3-in-1 takes
images 10*i to 10*i+9 as prompt i of complex_val.txt. So the images are linked in as n = 10 * (the
prompt's position among the kept prompts) + (the seed's position), and 3-in-1 reads a copy of the
prompt lists holding only the kept prompts, in the benchmark's order.

Usage (t2icomp env, an RTX 3090 or RTX 8000):
  python scripts/review_bench_t2i.py --set t2i_spatial --work <fresh dir> --out t2i_spatial_images.csv
  python scripts/review_bench_t2i.py --set t2i_complex --work <fresh dir> --out t2i_complex_images.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from review_bench_sets import SETS, T2I as PROMPTS, load  # noqa: E402

T2I = os.environ.get("T2I_COMPBENCH", "/datasets/mmolefe/repos/T2I-CompBench")
OUT = os.environ.get("REVIEW_BASELINE_OUT", "/datasets/mmolefe/poe_repair_min/outputs/paper_review")


def run(args, cwd):
    print("+", " ".join(args), f"(in {cwd})", flush=True)
    subprocess.run([sys.executable] + args, cwd=cwd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["t2i_spatial", "t2i_complex"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    work = os.path.abspath(a.work)
    if os.path.exists(work):
        sys.exit(f"{work} exists; give a fresh --work folder")
    samples = os.path.join(work, "samples")
    os.makedirs(samples)

    kept, _ = load(a.set)
    seeds = sorted(SETS[a.set]["seeds"])
    images, missing = {}, []
    for pos, item in enumerate(kept):
        assert "_" not in item["prompt"] and "/" not in item["prompt"], item["prompt"]
        for k, seed in enumerate(seeds):
            png = os.path.join(OUT, "bench_joint", a.set, f"{item['idx']:04d}", f"seed{seed:02d}", "joint.png")
            if not os.path.exists(png):
                missing.append(png)
                continue
            n = 10 * pos + k
            os.symlink(png, os.path.join(samples, f"{item['prompt']}_{n:06d}.png"))
            images[n] = (png, item["prompt"], seed)
    if missing:
        sys.exit(f"{len(missing)} renders missing, e.g. {missing[0]}; every prompt needs all {len(seeds)} images")

    unidet = os.path.join(T2I, "UniDet_eval")
    if a.set == "t2i_spatial":
        run(["2D_spatial_eval.py", "--outpath", work], unidet)
        res = os.path.join(work, "labels", "annotation_obj_detection_2d", "vqa_result.json")
    else:
        data = os.path.join(work, "dataset")
        os.makedirs(data)
        keep = {x["prompt"] for x in kept}
        for name in ("complex_val.txt", "complex_val_spatial.txt", "complex_val_action.txt"):
            src = os.path.join(T2I, "examples", "dataset", name)
            lines = [ln for ln in open(src).read().splitlines() if ln.strip()]
            if name == "complex_val.txt":
                lines = [ln for ln in lines if ln.strip() in keep]
                assert len(lines) == len(kept), (len(lines), len(kept))
            open(os.path.join(data, name), "w").write("\n".join(lines) + "\n")
        run(["BLIP_vqa.py", "--out_dir", work], os.path.join(T2I, "BLIPvqa_eval"))
        run(["2D_spatial_eval.py", "--outpath", work, "--complex", "True"], unidet)
        run(["CLIPScore_eval/CLIP_similarity.py", "--outpath", work, "--complex", "True"], T2I)
        run(["3_in_1.py", "--outpath", work, "--data_path", data], os.path.join(T2I, "3_in_1_eval"))
        res = os.path.join(work, "annotation_3_in_1", "vqa_result.json")

    by_id = {int(x["question_id"]): float(x["answer"]) for x in json.load(open(res))}
    assert len(by_id) == len(images), (len(by_id), len(images))
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "set", "idx", "prompt", "seed", "image", "score"])
        for n in sorted(images):
            png, prompt, seed = images[n]
            w.writerow(["joint", a.set, png.split("/")[-3], prompt, seed, png, f"{by_id[n]:.4f}"])
    vals = list(by_id.values())
    print(f"{a.set}: mean {sum(vals) / len(vals):.4f} over {len(vals)} images -> {a.out}")


if __name__ == "__main__":
    main()
