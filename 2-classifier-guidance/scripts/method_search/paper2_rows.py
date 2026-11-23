#!/usr/bin/env python3
"""Step M1, Paper 2: CO3 alone, CO3 with the kept push, and CO3 with a size-matched random push.

The design is artifacts/notes/progress/method-search/01-beat-the-baseline/design.md.

  check   the concept parse on all tuning prompts and the zero-push check, in one process on one card
  render  the three rows for each pair and seed, skipping finished pictures

Usage: python paper2_rows.py check --head <pt> --splits <json> --out <dir>
       python paper2_rows.py render --head <pt> --splits <json> --out <dir> [--shard 0 --num-shards 2] [--pairs-limit 1 --seeds-limit 1]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "rung2"))
from co3_common import load, load_parser, set_prompt, start_noise  # noqa: E402
from composers.utils_custom import seed_everything  # noqa: E402
from disallowed_subjects import is_allowed  # noqa: E402
from co3_push import Pusher, attach  # noqa: E402
from poe_repair.runtime import encode_prompt_sdxl  # noqa: E402
from train_readout_pair import PairHead, ReadoutPairClassifier, joint_prompt  # noqa: E402

STRENGTH = 30.0
CAP = 0.4
WINDOW = (0, 16)
ZERO_PUSH_MAX_ABS = 1e-5
ROWS = ("co3", "push", "random")


def tuning(splits, pairs_limit=None, seeds_limit=None):
    data = json.loads(Path(splits).read_text())
    pairs, seeds = data["tuning_pairs"], data["tuning_seeds"]
    return pairs[:pairs_limit] if pairs_limit else pairs, seeds[:seeds_limit] if seeds_limit else seeds


def build_checker(model, real_unet, head_path, first_prompt):
    state = torch.load(head_path, map_location="cpu", weights_only=False)
    if not state.get("pair_prompt") or state.get("trunk_lora"):
        raise SystemExit("this script expects the kept checker: joint prompt, no trunk adapters")
    head = PairHead(n_out=state.get("heads", 1))
    head.load_state_dict(state["head"])
    seq, pool = encode(model, first_prompt)
    checker = ReadoutPairClassifier(real_unet, seq, pool, head.float().eval()).to(model.unet.device)
    for parameter in checker.parameters():
        parameter.requires_grad_(False)
    return checker, state["step"]


def encode(model, prompt):
    models = {"tokenizer": model.tokenizer, "tokenizer_2": model.tokenizer_2, "text_encoder": model.text_encoder,
              "text_encoder_2": model.text_encoder_2}
    with torch.no_grad():
        return encode_prompt_sdxl(prompt, models=models, device=model.unet.device, dtype=torch.float16)


def point_checker(model, checker, prompt):
    checker.reader.seq_e, checker.reader.pool_e = encode(model, prompt)


def final_latent(model, start):
    x = start.clone()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for index, timestep in enumerate(model.scheduler.timesteps):
            x = model.denoise_step(x, timestep, index)
    return x


def make_pusher(config, checker, mode, norms=None, generator=None):
    return Pusher(checker, config.guidance_scale, STRENGTH, CAP, WINDOW, mode, norms, generator)


def check(args):
    pairs, seeds = tuning(args.splits)
    parser_nlp = load_parser()
    prompts = {pair: joint_prompt(pair) for pair in pairs}
    refused = [p for p in prompts.values() if not is_allowed(p)]
    if refused:
        raise SystemExit(f"the subject guard refuses: {refused}")
    model, config = load(prompts[pairs[0]], parser_nlp, args.out / "_config")
    real_unet = model.unet
    real_unet.requires_grad_(False)
    wrapper = attach(model)
    checker, head_step = build_checker(model, real_unet, args.head, prompts[pairs[0]])
    concepts = {pair: set_prompt(model, config, prompt, parser_nlp) for pair, prompt in prompts.items()}

    pair, seed = pairs[0], seeds[0]
    set_prompt(model, config, prompts[pair], parser_nlp)
    point_checker(model, checker, prompts[pair])
    start = start_noise(model, seed)

    def run(pusher):
        wrapper.pusher = pusher
        seed_everything(seed)
        latent = final_latent(model, start)
        wrapper.pusher = None
        return latent

    plain = run(None)
    repeat = run(None)
    zero_pusher = make_pusher(config, checker, "zero")
    zero = run(zero_pusher)
    real_pusher = make_pusher(config, checker, "push")
    pushed = run(real_pusher)
    result = {"pair": pair, "seed": seed, "device": torch.cuda.get_device_name(0), "dtype": str(real_unet.dtype),
              "head_step": head_step, "concepts": concepts, "pairs_without_two_concepts": [p for p, n in concepts.items() if n != 2],
              "plain_repeat_max_abs": float((repeat - plain).abs().max()),
              "zero_push_max_abs": float((zero - plain).abs().max()), "zero_bar": ZERO_PUSH_MAX_ABS,
              "real_push_max_abs": float((pushed - plain).abs().max()),
              "pushed_steps": sum(1 for n in real_pusher.applied if n > 0), "cap_fired_steps": sum(real_pusher.capped),
              "two_entry_calls_per_step": sorted(set(real_pusher.main_calls.values()))}
    result["zero_passes"] = result["zero_push_max_abs"] < ZERO_PUSH_MAX_ABS
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if result["zero_passes"] else 1)


def render(args):
    pairs, seeds = tuning(args.splits, args.pairs_limit, args.seeds_limit)
    pairs = pairs[args.shard::args.num_shards]
    parser_nlp = load_parser()
    prompts = {pair: joint_prompt(pair) for pair in pairs}
    refused = [p for p in prompts.values() if not is_allowed(p)]
    if refused:
        raise SystemExit(f"the subject guard refuses: {refused}")
    model, config = load(prompts[pairs[0]], parser_nlp, args.out / "_config")
    real_unet = model.unet
    real_unet.requires_grad_(False)
    wrapper = attach(model)
    checker, _ = build_checker(model, real_unet, args.head, prompts[pairs[0]])
    done = 0
    for pair_index, pair in enumerate(pairs):
        concepts = set_prompt(model, config, prompts[pair], parser_nlp)
        if concepts != 2:
            print(f"skipping {pair}: {concepts} concepts", flush=True)
            continue
        point_checker(model, checker, prompts[pair])
        for seed in seeds:
            norms = None
            for row in ROWS:
                folder = args.out / "renders" / row / pair
                image_path, meta_path = folder / f"seed-{seed}.png", folder / f"seed-{seed}.json"
                if image_path.exists() and meta_path.exists():
                    if row == "push":
                        norms = json.loads(meta_path.read_text())["applied_norms"]
                    continue
                if row == "co3":
                    pusher = None
                elif row == "push":
                    pusher = make_pusher(config, checker, "push")
                else:
                    if norms is None:
                        raise SystemExit(f"no push lengths for {pair} seed {seed}")
                    generator = torch.Generator().manual_seed(90_000 + 1_000 * args.splits_index(pair) + seed)
                    pusher = make_pusher(config, checker, "random", norms, generator)
                begin = time.time()
                seed_everything(seed)
                wrapper.pusher = pusher
                image = model.sample_loop(start_noise(model, seed))[0]
                wrapper.pusher = None
                seconds = time.time() - begin
                folder.mkdir(parents=True, exist_ok=True)
                image.save(image_path)
                applied = pusher.applied if pusher else []
                meta = {"row": row, "pair": pair, "prompt": prompts[pair], "seed": seed, "seconds": round(seconds, 2),
                        "concepts": concepts, "applied_norms": applied, "cap_fired": pusher.capped if pusher else [],
                        "strength": STRENGTH if row == "push" else 0.0, "cap": CAP, "window": list(WINDOW)}
                meta_path.write_text(json.dumps(meta) + "\n")
                if row == "push":
                    norms = applied
                done += 1
                print(f"{done} {row} {pair} seed={seed} {seconds:.1f}s", flush=True)
    print(f"finished shard={args.shard} rendered={done}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["check", "render"])
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--pairs-limit", type=int)
    ap.add_argument("--seeds-limit", type=int)
    args = ap.parse_args()
    order = json.loads(args.splits.read_text())["tuning_pairs"]
    args.splits_index = order.index
    if not torch.cuda.is_available():
        raise SystemExit("torch sees no CUDA card")
    (check if args.stage == "check" else render)(args)


if __name__ == "__main__":
    main()
