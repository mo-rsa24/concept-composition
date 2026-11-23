#!/usr/bin/env python3
"""Step M9, rung 2: CO3 with its own regions, swapped regions and random regions from step 10.

The design is artifacts/notes/progress/method-search/04-bind-each-body/design.md.

  check   the zero-binding check, in one process on one card
  read    the step-10 boxes on 12 pairs at 2 seeds, with no picture rendered, against the bar on the masks
  render  the three rows for each pair and seed, skipping finished pictures

Usage: python m9_rung2.py check --splits <json> --m1-co3 <dir> --out <dir>
       python m9_rung2.py read --splits <json> --out <dir>
       python m9_rung2.py render --splits <json> --out <dir> [--shard 0 --num-shards 2] [--pairs-limit 1 --seeds-limit 1]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "rung2"))
from co3_common import load, load_parser, set_prompt, start_noise  # noqa: E402
from composers.utils_custom import seed_everything  # noqa: E402
from disallowed_subjects import is_allowed  # noqa: E402
from co3_bind import START_STEP, Binder, MasksRead, attach  # noqa: E402
from train_readout_pair import joint_prompt  # noqa: E402

ZERO_BINDING_MAX_ABS = 1e-5
ROWS = ("own", "swapped", "random")
READ_SEEDS = 2
MAX_BOX_SHARE = 0.8         # a box over this share of the grid gives its subject no region of its own
MAX_READ_IOU = 0.8          # two boxes overlapping this much are one region
MAX_DEGENERATE_SHARE = 0.25 # the most reads that may fail either limit before rung 2 stops at its masks


def tuning(splits, pairs_limit=None, seeds_limit=None):
    data = json.loads(Path(splits).read_text())
    pairs, seeds = data["tuning_pairs"], data["tuning_seeds"]
    return pairs[:pairs_limit] if pairs_limit else pairs, seeds[:seeds_limit] if seeds_limit else seeds


def subject_words(pair):
    """'a_puma__x__a_condor' -> ('puma', 'condor')."""
    a, b = pair.split("__x__")
    return a.split("_")[-1], b.split("_")[-1]


def binder_for(model, pair, prompt, mode, box_seed=None):
    word_a, word_b = subject_words(pair)
    return Binder(model, prompt, word_a, word_b, mode, box_seed)


def final_latent(model, start):
    x = start.clone()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for index, timestep in enumerate(model.scheduler.timesteps):
            x = model.denoise_step(x, timestep, index)
    return x


def check(args):
    pairs, seeds = tuning(args.splits)
    parser_nlp = load_parser()
    prompts = {pair: joint_prompt(pair) for pair in pairs}
    refused = [p for p in prompts.values() if not is_allowed(p)]
    if refused:
        raise SystemExit(f"the subject guard refuses: {refused}")
    pair, seed = pairs[0], seeds[0]
    model, config = load(prompts[pair], parser_nlp, args.out / "_config")
    model.unet.requires_grad_(False)
    wrapper = attach(model)
    concepts = {p: set_prompt(model, config, text, parser_nlp) for p, text in prompts.items()}
    order = {p: list(model.prompts_single[:2]) for p in [pairs[-1]]}
    for p, text in prompts.items():
        set_prompt(model, config, text, parser_nlp)
        binder_for(model, p, text, "zero")
    set_prompt(model, config, prompts[pair], parser_nlp)
    start = start_noise(model, seed)

    def run(binder):
        wrapper.binder = binder
        seed_everything(seed)
        latent = final_latent(model, start)
        wrapper.binder = None
        return latent

    plain = run(None)
    repeat = run(None)
    zero = binder_for(model, pair, prompts[pair], "zero")
    zero_latent = run(zero)
    own = binder_for(model, pair, prompts[pair], "own")
    own_latent = run(own)
    seed_everything(seed)
    args.out.mkdir(parents=True, exist_ok=True)
    picture = args.out / "plain-co3-this-process.png"
    model.sample_loop(start_noise(model, seed))[0].save(picture)
    earlier = args.m1_co3 / pair / f"seed-{seed}.png"
    with Image.open(picture) as new, Image.open(earlier) as old:
        pixel_diff = float(np.abs(np.asarray(new.convert("RGB"), dtype=np.float64) - np.asarray(old.convert("RGB"), dtype=np.float64)).mean())
    result = {"pair": pair, "seed": seed, "device": torch.cuda.get_device_name(0), "dtype": str(wrapper.inner.dtype),
              "start_step": START_STEP, "concepts": concepts, "pairs_without_two_concepts": [p for p, n in concepts.items() if n != 2],
              "concept_order_example": order, "plain_repeat_max_abs": float((repeat - plain).abs().max()),
              "zero_binding_max_abs": float((zero_latent - plain).abs().max()), "zero_bar": ZERO_BINDING_MAX_ABS,
              "own_regions_max_abs": float((own_latent - plain).abs().max()), "own_regions_record": own.record,
              "two_entry_calls_per_step": sorted(set(own.main_calls.values())),
              "mean_abs_pixel_diff_to_m1_co3_of_255": round(pixel_diff, 4)}
    result["zero_passes"] = result["zero_binding_max_abs"] < ZERO_BINDING_MAX_ABS and not result["pairs_without_two_concepts"]
    (args.out / "check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if result["zero_passes"] else 1)


def is_degenerate(record):
    return (record["box_a_share_of_grid"] >= MAX_BOX_SHARE or record["box_b_share_of_grid"] >= MAX_BOX_SHARE
            or record["read_boxes_iou"] >= MAX_READ_IOU)


def read(args):
    """Read the step-10 boxes on 12 pairs at 2 seeds, with no picture rendered, and apply the bar on the masks."""
    pairs, seeds = tuning(args.splits, seeds_limit=READ_SEEDS)
    parser_nlp = load_parser()
    prompts = {pair: joint_prompt(pair) for pair in pairs}
    refused = [p for p in prompts.values() if not is_allowed(p)]
    if refused:
        raise SystemExit(f"the subject guard refuses: {refused}")
    model, config = load(prompts[pairs[0]], parser_nlp, args.out / "_config")
    model.unet.requires_grad_(False)
    wrapper = attach(model)
    args.out.mkdir(parents=True, exist_ok=True)
    rows, maps = [], {}
    for pair in pairs:
        set_prompt(model, config, prompts[pair], parser_nlp)
        word_a, word_b = subject_words(pair)
        for seed in seeds:
            binder = Binder(model, prompts[pair], word_a, word_b, "zero", stop_after_read=True)
            wrapper.binder = binder
            seed_everything(seed)
            try:
                final_latent(model, start_noise(model, seed))
                raise SystemExit("the render finished without reading the masks")
            except MasksRead:
                pass
            finally:
                wrapper.binder = None
            rows.append({"pair": pair, "seed": seed, **binder.record, "degenerate": is_degenerate(binder.record)})
            maps[f"{pair}|{seed}|a"], maps[f"{pair}|{seed}|b"] = binder.maps
            print(json.dumps(rows[-1]), flush=True)
    degenerate = sum(r["degenerate"] for r in rows)
    result = {"reads": len(rows), "degenerate": degenerate, "max_degenerate_share": MAX_DEGENERATE_SHARE,
              "max_box_share": MAX_BOX_SHARE, "max_read_iou": MAX_READ_IOU,
              "passes": degenerate / len(rows) <= MAX_DEGENERATE_SHARE, "rows": rows}
    (args.out / "reads.json").write_text(json.dumps(result, indent=1) + "\n")
    np.savez_compressed(args.out / "maps.npz", **maps)
    print(json.dumps({k: result[k] for k in ("reads", "degenerate", "passes")}), flush=True)
    raise SystemExit(0 if result["passes"] else 1)


def render(args):
    pairs, seeds = tuning(args.splits, args.pairs_limit, args.seeds_limit)
    order = json.loads(args.splits.read_text())["tuning_pairs"]
    pairs = pairs[args.shard::args.num_shards]
    parser_nlp = load_parser()
    prompts = {pair: joint_prompt(pair) for pair in pairs}
    refused = [p for p in prompts.values() if not is_allowed(p)]
    if refused:
        raise SystemExit(f"the subject guard refuses: {refused}")
    model, config = load(prompts[pairs[0]], parser_nlp, args.out / "_config")
    model.unet.requires_grad_(False)
    wrapper = attach(model)
    done = 0
    for pair in pairs:
        concepts = set_prompt(model, config, prompts[pair], parser_nlp)
        if concepts != 2:
            raise SystemExit(f"{pair}: CO3 found {concepts} concepts")
        for seed in seeds:
            for row in ROWS:
                folder = args.out / "renders" / row / pair
                image_path, meta_path = folder / f"seed-{seed}.png", folder / f"seed-{seed}.json"
                if image_path.exists() and meta_path.exists():
                    continue
                box_seed = 91_000 + 1_000 * order.index(pair) + seed
                binder = binder_for(model, pair, prompts[pair], row, box_seed if row == "random" else None)
                begin = time.time()
                seed_everything(seed)
                wrapper.binder = binder
                image = model.sample_loop(start_noise(model, seed))[0]
                wrapper.binder = None
                seconds = time.time() - begin
                folder.mkdir(parents=True, exist_ok=True)
                image.save(image_path)
                meta = {"row": row, "pair": pair, "prompt": prompts[pair], "seed": seed, "seconds": round(seconds, 2),
                        "start_step": START_STEP, "box_seed": box_seed if row == "random" else None, **binder.record}
                meta_path.write_text(json.dumps(meta) + "\n")
                done += 1
                print(f"{done} {row} {pair} seed={seed} {seconds:.1f}s iou={binder.record['read_boxes_iou']}", flush=True)
    print(f"finished shard={args.shard} rendered={done}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["check", "read", "render"])
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--m1-co3", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--pairs-limit", type=int)
    ap.add_argument("--seeds-limit", type=int)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("torch sees no CUDA card")
    {"check": check, "read": read, "render": render}[args.stage](args)


if __name__ == "__main__":
    main()
