#!/usr/bin/env python3
"""Loop 12 step Z8 proper: each case redrawn on its own base without a push (once), with the checker's push (four draws) and
with a random push of the same size (four draws).

Design: artifacts/notes/progress/leakage/08-first-correction-test/design.md. The push is the kept recipe: the multi-pair
readout checker reading SDXL with the pair's joint prompt, strength 30, capped at 0.4 of the prediction's length, sampler
steps 0 to 15. Draw k of the random push takes the lengths draw k of the push applied, step by step, and a direction from its
own generator.

  sdxl  plain PoE (run_poe_classifier_guided) and the joint prompt (z8_joint.run_joint_classifier_guided), 50 DDIM steps,
        guidance 7.5, the per-seed starting noise every pair shares
  co3   CO3 at its released settings with Loop 6's pusher (paper2_rows.py, co3_push.py), unchanged

  python z8_cases.py sdxl --head <pt> --out <dir>
  python z8_cases.py co3  --head <pt> --out <dir>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "rung2"))
sys.path.insert(0, str(HERE.parent / "method_search"))
sys.path.insert(0, str(HERE.parents[2] / "scripts" / "shared"))
from disallowed_subjects import is_allowed  # noqa: E402
from train_readout_pair import joint_prompt  # noqa: E402

STRENGTH, CAP, WINDOW, DRAWS = 30.0, 0.4, (0, 16), 4
# Fixed in the design before any Z8 picture: plain PoE and the joint prompt from the screening pass by the pick rule,
# CO3 from W2d's labels (both labellers agreeing)
CASES = {
    "poe": [("a_puma__x__a_condor", 6001), ("a_zebra__x__a_flamingo", 6001), ("a_seal__x__a_gull", 6001)],
    "joint": [("a_puma__x__a_condor", 6001), ("a_zebra__x__a_flamingo", 6001), ("a_seal__x__a_gull", 6001)],
    "co3": [("a_cheetah__x__a_cougar", 6001), ("a_puma__x__a_condor", 6001), ("a_duck__x__a_goose", 6002),
            ("a_zebra__x__a_flamingo", 6001), ("an_eagle__x__a_fox", 6001)],
}
BASE_INDEX = {"poe": 1, "joint": 2, "co3": 3}


def rgen(base, case_i, seed, draw):
    return torch.Generator(device="cpu").manual_seed(80_000 + 10_000 * BASE_INDEX[base] + 1_000 * case_i + 10 * (seed - 6000) + draw)


def save(out, base, arm, pair, seed, draw, img, meta):
    folder = out / base / pair / f"seed-{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    name = arm if draw is None else f"{arm}-{draw}"
    img.save(folder / f"{name}.png")
    (folder / f"{name}.json").write_text(json.dumps(meta) + "\n")


def done(out, base, pair, seed, name):
    f = out / base / pair / f"seed-{seed}"
    return (f / f"{name}.png").exists() and (f / f"{name}.json").exists()


def sdxl(args):
    from poe_repair.composers._helpers import init_latents_for_cell
    from poe_repair.methods._classifier_guided import run_poe_classifier_guided
    from poe_repair.run import make_ctx
    from poe_repair.runtime import encode_prompt_sdxl
    from train_readout_pair import PairHead, ReadoutPairClassifier
    import classifier_guidance_from_the_paper as single
    from z8_joint import run_joint_classifier_guided

    single.disk_guard(args.out)
    ctx = make_ctx(output_root=args.out / "raw")
    enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = enc("")
    state = torch.load(args.head, map_location="cpu", weights_only=False)
    if not state.get("pair_prompt") or state.get("trunk_lora"):
        raise SystemExit("expects the kept checker: joint prompt, no trunk adapters")
    head = PairHead(n_out=state.get("heads", 1)); head.load_state_dict(state["head"])
    common = dict(models=ctx.models, scheduler=ctx.scheduler, seq_e=seq_e, pool_e=pool_e, guidance_scale=ctx.guidance_scale,
                  num_inference_steps=ctx.num_inference_steps, height=1024, width=1024, device=ctx.device, dtype=ctx.dtype)
    to_img = lambda o: Image.fromarray((o.image[0].float().clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy())
    n = 0
    for base in ("poe", "joint"):
        for case_i, (pair, seed) in enumerate(CASES[base]):
            jp = joint_prompt(pair); assert is_allowed(jp), pair
            seq_j, pool_j = enc(jp)
            a, b = [p.replace("_", " ") for p in pair.split("__x__")]
            if base == "poe":
                seq_a, pool_a = enc(a); seq_b, pool_b = enc(b)
                run = lambda **kw: run_poe_classifier_guided(seq_a=seq_a, pool_a=pool_a, seq_b=seq_b, pool_b=pool_b, **kw, **common)
            else:
                run = lambda **kw: run_joint_classifier_guided(seq_cond=seq_j, pool_cond=pool_j, **kw, **common)
            disc = ReadoutPairClassifier(ctx.models["unet"], seq_j, pool_j, head.float().eval()).to(ctx.device)
            arms = [("without", None)] + [(arm, k) for k in range(DRAWS) for arm in ("push", "random")]
            norms = {}
            for arm, k in arms:
                name = arm if k is None else f"{arm}-{k}"
                if done(args.out, base, pair, seed, name):
                    if arm == "push":
                        norms[k] = json.loads((args.out / base / pair / f"seed-{seed}" / f"{name}.json").read_text())["applied_norms"]
                    continue
                init, sigma = init_latents_for_cell(single.cell_for(seed), ctx)
                s = time.time()
                if arm == "without":
                    o = run(init_latents=init, euler_init_noise_sigma=sigma, strength=0.0)
                elif arm == "push":
                    o = run(init_latents=init, euler_init_noise_sigma=sigma, head=disc, strength=STRENGTH, push_cap=CAP,
                            guidance_window=WINDOW)
                    norms[k] = o.extras["applied_norm_per_step"]
                else:
                    o = run(init_latents=init, euler_init_noise_sigma=sigma, random_push_norms=norms[k],
                            random_generator=rgen(base, case_i, seed, k))
                sec = time.time() - s
                meta = {"base": base, "arm": arm, "draw": k, "pair": pair, "prompt": jp, "seed": seed, "seconds": round(sec, 2),
                        "applied_norms": o.extras.get("applied_norm_per_step", []), "cap_fired": o.extras.get("capped_per_step", []),
                        "strength": STRENGTH if arm == "push" else 0.0, "cap": CAP, "window": list(WINDOW)}
                save(args.out, base, arm, pair, seed, k, to_img(o), meta)
                n += 1
                print(f"{n} {base} {name} {pair} seed={seed} {sec:.1f}s", flush=True)
    return n


def co3(args):
    from co3_common import load, load_parser, set_prompt, start_noise
    from composers.utils_custom import seed_everything
    from co3_push import attach
    from paper2_rows import build_checker, make_pusher, point_checker

    parser_nlp = load_parser()
    cases = CASES["co3"]
    prompts = {p: joint_prompt(p) for p, _ in cases}
    assert all(is_allowed(v) for v in prompts.values())
    model, config = load(prompts[cases[0][0]], parser_nlp, args.out / "_config")
    real_unet = model.unet; real_unet.requires_grad_(False)
    wrapper = attach(model)
    checker, _ = build_checker(model, real_unet, args.head, prompts[cases[0][0]])
    n = 0
    for case_i, (pair, seed) in enumerate(cases):
        concepts = set_prompt(model, config, prompts[pair], parser_nlp)
        assert concepts == 2, (pair, concepts)
        point_checker(model, checker, prompts[pair])
        norms = {}
        for arm, k in [("without", None)] + [(arm, k) for k in range(DRAWS) for arm in ("push", "random")]:
            name = arm if k is None else f"{arm}-{k}"
            if done(args.out, "co3", pair, seed, name):
                if arm == "push":
                    norms[k] = json.loads((args.out / "co3" / pair / f"seed-{seed}" / f"{name}.json").read_text())["applied_norms"]
                continue
            pusher = None if arm == "without" else (make_pusher(config, checker, "push") if arm == "push"
                                                     else make_pusher(config, checker, "random", norms[k], rgen("co3", case_i, seed, k)))
            s = time.time()
            seed_everything(seed)
            wrapper.pusher = pusher
            img = model.sample_loop(start_noise(model, seed))[0]
            wrapper.pusher = None
            sec = time.time() - s
            applied = pusher.applied if pusher else []
            if arm == "push":
                norms[k] = applied
            meta = {"base": "co3", "arm": arm, "draw": k, "pair": pair, "prompt": prompts[pair], "seed": seed, "seconds": round(sec, 2),
                    "concepts": concepts, "applied_norms": applied, "cap_fired": pusher.capped if pusher else [],
                    "strength": STRENGTH if arm == "push" else 0.0, "cap": CAP, "window": list(WINDOW)}
            save(args.out, "co3", arm, pair, seed, k, img, meta)
            n += 1
            print(f"{n} co3 {name} {pair} seed={seed} {sec:.1f}s", flush=True)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("part", choices=["sdxl", "co3"])
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = sdxl(args) if args.part == "sdxl" else co3(args)
    print(json.dumps({"part": args.part, "rendered": n, "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
