#!/usr/bin/env python3
"""Step M5: can the kept checker tell CO3's two-clear pictures from its borrowed-part pictures, step by step?

The design is artifacts/notes/progress/method-search/01b-balance-the-checker-data/design.md, section "The check that the checker can
see CO3's failure". CO3 reruns at its released settings on M1's 12 tuning pairs at seeds 6001 to 6008. At steps 0, 5, 10, 15, 20, 25 and 30
the kept checker reads the step's latent under the joint prompt, and its outputs are recorded. Nothing is pushed. Each final picture must
match M1's within 3 of 255, and the labels are M1's blind calls on CO3 alone.

  read   the 96 runs, writing reads.jsonl and the pictures
  score  balanced accuracy, area under the curve and a per-pair table per step, against the bar

Usage: python m5_read_co3.py read --head <pt> --splits <json> --m1-co3 <dir> --out <dir> [--redo-from <reads.jsonl>]
       python m5_read_co3.py score --out <dir> [--also <dir> ...] --calls <csv> --key <json>

M1 drew half its pairs on a Quadro RTX 8000 and half on an RTX 3090, and a CO3 run reproduces only on the card type that drew it, so the
rerun of the non-reproduced trajectories runs on an RTX 8000 into its own folder and the score merges the two.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
READ_STEPS = (0, 5, 10, 15, 20, 25, 30)
MAX_PIXEL_DIFF = 3.0      # of 255, the project's reproduction limit
BAR, FLOOR = 0.75, 0.65   # written before the read
MIN_REPRODUCED = 80


def read(args):
    import torch
    from PIL import Image
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parent / "rung2"))
    from co3_common import load, load_parser, set_prompt, start_noise
    from composers.utils_custom import seed_everything
    from disallowed_subjects import is_allowed
    from paper2_rows import build_checker, point_checker, tuning
    from train_readout_pair import joint_prompt
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput

    pairs, seeds = tuning(args.splits)
    prompts = {pair: joint_prompt(pair) for pair in pairs}
    if any(not is_allowed(p) for p in prompts.values()):
        raise SystemExit("the subject guard refuses a prompt")
    parser_nlp = load_parser()
    model, config = load(prompts[pairs[0]], parser_nlp, args.out / "_config")
    real_unet = model.unet
    real_unet.requires_grad_(False)
    checker, head_step = build_checker(model, real_unet, args.head, prompts[pairs[0]])
    state = {"step": None, "reads": {}}

    class ReadingUNet(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.inner = unet

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(super().__getattr__("inner"), name)

        def forward(self, sample, timestep, *a, **k):
            out = self.inner(sample, timestep, *a, **k)
            if sample.shape[0] == 2 and state["step"] in READ_STEPS:
                with torch.no_grad():
                    o = checker.head(checker.reader(sample[:1].to(checker.reader.seq_e.dtype), timestep.reshape(1)))
                state["reads"][state["step"]] = [float(v) for v in o.reshape(-1).float().cpu()]
            return out if isinstance(out, UNet2DConditionOutput) else out

    model.unet = ReadingUNet(real_unet)
    original = model.denoise_step

    def denoise_step(x, t, step):
        state["step"] = int(step)
        return original(x, t, step)

    model.denoise_step = denoise_step
    args.out.mkdir(parents=True, exist_ok=True)
    redo = None
    if args.redo_from:   # rerun only what an earlier read could not reproduce, into this new folder
        redo = {r["key"] for r in map(json.loads, args.redo_from.read_text().splitlines()) if not r["reproduced"]}
        print(f"rerunning {len(redo)} trajectories not reproduced in {args.redo_from}", flush=True)
    done = {json.loads(l)["key"] for l in (args.out / "reads.jsonl").read_text().splitlines()} if (args.out / "reads.jsonl").exists() else set()
    with open(args.out / "reads.jsonl", "a") as log:
        for pair in pairs:
            set_prompt(model, config, prompts[pair], parser_nlp)
            point_checker(model, checker, prompts[pair])
            for seed in seeds:
                key = f"{pair}|{seed}"
                if key in done or (redo is not None and key not in redo):
                    continue
                if args.limit is not None and len(done) >= args.limit:
                    return
                state["reads"] = {}
                begin = time.time()
                seed_everything(seed)
                image = model.sample_loop(start_noise(model, seed))[0]
                folder = args.out / "renders" / pair
                folder.mkdir(parents=True, exist_ok=True)
                image.save(folder / f"seed-{seed}.png")
                with Image.open(args.m1_co3 / pair / f"seed-{seed}.png") as old:
                    diff = float(np.abs(np.asarray(image.convert("RGB"), dtype=np.float64)
                                        - np.asarray(old.convert("RGB"), dtype=np.float64)).mean())
                row = {"key": key, "pair": pair, "seed": seed, "head_step": head_step, "pixel_diff_to_m1_of_255": round(diff, 3),
                       "reproduced": diff <= MAX_PIXEL_DIFF, "reads": {str(s): v for s, v in state["reads"].items()},
                       "seconds": round(time.time() - begin, 1)}
                log.write(json.dumps(row) + "\n")
                log.flush()
                done.add(key)
                print(f"{key} diff {diff:.2f} steps read {sorted(state['reads'])} {row['seconds']} s", flush=True)


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def score(args):
    key = json.loads(args.key.read_text())
    clear = {}
    for row in csv.DictReader(open(args.calls, newline="")):
        k = key[row["id"]]
        if k["row"] == "co3":
            clear[f"{k['pair']}|{k['seed']}"] = row["rung"] == "two_clear"
    best = {}
    for folder in [args.out] + list(args.also or []):
        for r in map(json.loads, (folder / "reads.jsonl").read_text().splitlines()):
            if r["key"] not in best or (r["reproduced"] and not best[r["key"]]["reproduced"]):
                best[r["key"]] = r
    rows = list(best.values())
    kept = [r for r in rows if r["reproduced"] and r["key"] in clear]
    per_step = {}
    for step in READ_STEPS:
        use = [r for r in kept if str(step) in r["reads"]]
        s = [r["reads"][str(step)][0] for r in use]
        lab = [clear[r["key"]] for r in use]
        pos = [x > 0 for x, l in zip(s, lab) if l]
        neg = [x <= 0 for x, l in zip(s, lab) if not l]
        pairs = {}
        for r, x, l in zip(use, s, lab):
            pairs.setdefault(r["pair"], {"clear_mean": [], "other_mean": []})["clear_mean" if l else "other_mean"].append(x)
        per_step[step] = {"pictures": len(use), "balanced_accuracy": round((np.mean(pos) + np.mean(neg)) / 2, 4),
                          "area_under_curve": round(auc(s, lab), 4),
                          "per_pair_mean_score": {p: {k: (round(float(np.mean(v)), 3) if v else None) for k, v in d.items()}
                                                  for p, d in sorted(pairs.items())}}
    best = max(per_step, key=lambda s: per_step[s]["balanced_accuracy"])
    top = per_step[best]["balanced_accuracy"]
    if len(kept) < MIN_REPRODUCED:
        verdict = "inconclusive: too few trajectories reproduce"
    elif top >= BAR:
        verdict = "the checker sees the failure"
    elif top < FLOOR:
        verdict = "it does not"
    else:
        verdict = "inconclusive"
    result = {"runs": len(rows), "reproduced_and_labelled": len(kept), "clear": sum(clear[r["key"]] for r in kept),
              "bar": BAR, "floor": FLOOR, "best_step": best, "best_balanced_accuracy": top, "verdict": verdict, "per_step": per_step}
    (args.out / ("read-score-merged.json" if args.also else "read-score.json")).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("runs", "reproduced_and_labelled", "clear", "best_step", "best_balanced_accuracy", "verdict")}))
    print({s: (v["balanced_accuracy"], v["area_under_curve"]) for s, v in per_step.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["read", "score"])
    ap.add_argument("--head", type=Path)
    ap.add_argument("--splits", type=Path)
    ap.add_argument("--m1-co3", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--calls", type=Path)
    ap.add_argument("--key", type=Path)
    ap.add_argument("--redo-from", type=Path, help="an earlier reads.jsonl; rerun only its trajectories that did not reproduce")
    ap.add_argument("--also", type=Path, nargs="*", help="score: further read folders, merged, a reproduced row winning")
    ap.add_argument("--limit", type=int, help="stop after this many runs, for the smoke rung")
    args = ap.parse_args()
    {"read": read, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
