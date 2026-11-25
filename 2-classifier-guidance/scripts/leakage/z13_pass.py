#!/usr/bin/env python3
"""Loop 12 step Z13: Z7's clean-up pass with each animal's own prediction in its automatic box, against the same pass over the whole
picture, against CO3 alone, at seeds 6025 to 6032.

Design: artifacts/notes/progress/leakage/13-clean-up-pass-in-boxes/design.md. The pass is z3_regions.Corrector unchanged (row "own" with
boxes, row "whole" without), at steps 5 to 30; this script only loops over the 12 tuning pairs and the fresh seeds, which z3_regions'
own loop refuses.

  render   --rows co3|whole|own [--boxes <json>] --out <dir> [--shard i --num-shards n]
  control  --boxes <json> --out <dir>      row own with empty boxes against row whole, byte for byte; the latent entering step 5
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "rung2"))
import z3_regions  # noqa: E402

z3_regions.STEPS = tuple(range(5, 31))  # Z7's window
SEEDS = tuple(range(6025, 6033))
ROW_TO_CORRECTOR = {"co3": None, "whole": "whole", "own": "own"}


def setup(args, first_prompt):
    from co3_common import load, load_parser
    parser_nlp = load_parser()
    model, config = load(first_prompt, parser_nlp, args.out / "_config")
    model.unet.requires_grad_(False)
    corr = z3_regions.attach(model)
    return model, config, parser_nlp, corr


def prompts_for(pairs):
    from disallowed_subjects import is_allowed
    from train_readout_pair import joint_prompt
    prompts = {p: joint_prompt(p) for p in pairs}
    refused = [p for p in prompts.values() if not is_allowed(p)]
    if refused:
        raise SystemExit(f"the subject guard refuses: {refused}")
    return prompts


def make_dir(folder):
    for _ in range(5):
        try:
            folder.mkdir(parents=True, exist_ok=True)
            return
        except FileExistsError:
            time.sleep(2)
    if not folder.is_dir():
        raise SystemExit(f"cannot create {folder}")


def boxes_for(boxes, pair, seed):
    b = boxes.get(pair, {}).get(str(seed), {}) if boxes else {}
    return b.get("A"), b.get("B")


def render_stage(args):
    from co3_common import set_prompt
    pairs = json.loads(args.splits.read_text())["tuning_pairs"][args.shard::args.num_shards]
    prompts = prompts_for(pairs)
    boxes = json.loads(args.boxes.read_text()) if args.rows == "own" else {}
    model, config, parser_nlp, corr = setup(args, prompts[pairs[0]])
    row = ROW_TO_CORRECTOR[args.rows]
    done, secs = 0, []
    for pair in pairs:
        assert set_prompt(model, config, prompts[pair], parser_nlp) == 2, pair
        for seed in SEEDS:
            folder = args.out / args.rows / pair
            path, meta = folder / f"seed-{seed}.png", folder / f"seed-{seed}.json"
            if path.exists() and meta.exists():
                continue
            b = boxes_for(boxes, pair, seed) if row == "own" else (None, None)
            img, sec = z3_regions.render(model, corr, seed, row, b)
            make_dir(folder)
            img.save(path)
            meta.write_text(json.dumps({"row": args.rows, "pair": pair, "seed": seed, "seconds": round(sec, 2), "boxes": list(b),
                                        "steps": list(z3_regions.STEPS), "beta": z3_regions.BETA, "passes": corr.log}) + "\n")
            done += 1; secs.append(sec)
            print(f"{done} {args.rows} {pair} seed={seed} {sec:.1f}s", flush=True)
    summary = {"row": args.rows, "shard": args.shard, "rendered": done, "median_seconds": float(np.median(secs)) if secs else None}
    summary["wandb_run"] = z3_regions.log_wandb(f"loop-12-z13-{args.rows}-{args.shard}",
                                                {"pairs": pairs, "seeds": SEEDS, "row": args.rows, "steps": list(z3_regions.STEPS)}, summary)
    (args.out / f"summary-{args.rows}-{args.shard}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary), flush=True)


def control_stage(args):
    from co3_common import set_prompt
    pairs = json.loads(args.splits.read_text())["tuning_pairs"]
    prompts = prompts_for(pairs)
    boxes = json.loads(args.boxes.read_text())
    model, config, parser_nlp, corr = setup(args, prompts[pairs[0]])
    pair, seed = args.control_pair, SEEDS[0]
    assert set_prompt(model, config, prompts[pair], parser_nlp) == 2
    first = z3_regions.STEPS[0]
    z3_regions.render(model, corr, seed, None, (None, None)); x_co3 = corr.entry[first]
    img_w, _ = z3_regions.render(model, corr, seed, "whole", (None, None)); x_whole = corr.entry[first]
    img_e, _ = z3_regions.render(model, corr, seed, "own", (None, None))
    b = boxes_for(boxes, pair, seed)
    img_o, _ = z3_regions.render(model, corr, seed, "own", b); x_own = corr.entry[first]
    diff = lambda u, v: int(np.abs(np.asarray(u, np.int16) - np.asarray(v, np.int16)).max())
    res = {"pair": pair, "seed": seed, "device": torch.cuda.get_device_name(0), "first_step": first,
           "empty_boxes_own_vs_whole_max_pixel": diff(img_w, img_e),
           "entering_first_step_whole_vs_co3_max_abs": float((x_whole - x_co3).abs().max()),
           "entering_first_step_own_vs_co3_max_abs": float((x_own - x_co3).abs().max()),
           "own_vs_whole_max_pixel": diff(img_w, img_o), "boxes": list(b)}
    res["passes"] = (res["empty_boxes_own_vs_whole_max_pixel"] == 0 and res["entering_first_step_whole_vs_co3_max_abs"] == 0.0
                     and res["entering_first_step_own_vs_co3_max_abs"] == 0.0)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "control.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res), flush=True)
    raise SystemExit(0 if res["passes"] else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["render", "control"])
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--boxes", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", default="co3", choices=list(ROW_TO_CORRECTOR))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--control-pair", default="a_duck__x__a_goose")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("torch sees no CUDA card")
    {"render": render_stage, "control": control_stage}[args.stage](args)


if __name__ == "__main__":
    main()
