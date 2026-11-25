#!/usr/bin/env python3
"""Loop 12 step Z10: the adapter's correction at six weights on eight animal pairs it never trained on.

Design: artifacts/notes/progress/leakage/10-co-occurrence-dial/design.md. The sampled prediction is
eps_off + lambda * (eps_on - eps_off), the product of experts with the adapter switched off and on
(run_poe_classifier_guided's adapter_lambda path, no push), classifier-free guidance 7.5 per expert, 50 DDIM steps.

  smoke   the first pair and seed at all six weights, controls 1 and 2, seconds per picture
  full    8 pairs x seeds 6001 to 6004 x 6 weights, plus control 3 (badger x magpie, a training pair, at weights 0 and 1)

  python z10_dial.py <smoke|full> --pairs <pairs.json> --adapter <lora_step_030000.pt> --out <dir>
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
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "scripts" / "shared"))
from poe_repair.composers._helpers import init_latents_for_cell  # noqa: E402
from poe_repair.experiments import _adapter_shape  # noqa: E402
from poe_repair.methods._classifier_guided import run_poe_classifier_guided  # noqa: E402
from poe_repair.methods._sampling import run_cfg_poe  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import encode_prompt_sdxl  # noqa: E402
import classifier_guidance_from_the_paper as single  # noqa: E402
from disallowed_subjects import is_allowed  # noqa: E402

WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
SEEDS = (6001, 6002, 6003, 6004)
TRAINING_PAIR = "a badger and a magpie"  # the first pair under train: in pair_pool_v58.yaml


def to_img(o):
    return (o.image[0].float().clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()


def slug(prompt):
    return prompt.replace(" and ", "__x__").replace(" ", "_")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["diagnose", "smoke", "full"])
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    prompts = json.loads(args.pairs.read_text())["pairs"]
    assert len(prompts) == 8 and all(is_allowed(p) for p in prompts + [TRAINING_PAIR])
    out = args.out / args.stage; out.mkdir(parents=True, exist_ok=True)
    single.disk_guard(out)
    ctx = make_ctx(output_root=out / "raw")
    enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = enc("")
    common = dict(models=ctx.models, scheduler=ctx.scheduler, seq_e=seq_e, pool_e=pool_e, guidance_scale=ctx.guidance_scale,
                  num_inference_steps=ctx.num_inference_steps, height=1024, width=1024, device=ctx.device, dtype=ctx.dtype)

    def experts(prompt):
        a, b = prompt.split(" and ")
        sa, pa = enc(a); sb, pb = enc(b)
        return dict(seq_a=sa, pool_a=pa, seq_b=sb, pool_b=pb)

    if args.stage == "diagnose":  # which of sampler code and adapter switch breaks control 1
        e0 = experts(prompts[0])
        def go(fn, **kw):
            init, sigma = init_latents_for_cell(single.cell_for(SEEDS[0]), ctx)
            return to_img(fn(init_latents=init, euler_init_noise_sigma=sigma, **e0, **common, **kw))
        a = go(run_cfg_poe)
        b = go(run_poe_classifier_guided, strength=0.0)
        _adapter_shape.attach(ctx.models["unet"], args.adapter, disable=True)
        c = go(run_poe_classifier_guided, strength=0.0, adapter_lambda=0.0)
        d = go(run_cfg_poe)
        ctx.models["unet"].enable_adapters()
        e = go(run_cfg_poe)
        ctx.models["unet"].disable_adapters()
        f = go(run_cfg_poe)
        cmp = lambda x, y: {"max": int(np.abs(x.astype(int) - y.astype(int)).max()), "mean": float(np.abs(x.astype(int) - y.astype(int)).mean())}
        res = {"a_plain_vs_b_guided_no_adapter": cmp(a, b), "b_vs_c_guided_weight_0_attached": cmp(b, c),
               "a_vs_d_plain_attached_disabled": cmp(a, d), "a_vs_e_plain_adapter_enabled": cmp(a, e),
               "a_vs_f_plain_disabled_again": cmp(a, f), "c_vs_e": cmp(c, e)}
        out.joinpath("diagnose.json").write_text(json.dumps(res, indent=1) + "\n")
        for k, im in zip("abcdef", (a, b, c, d, e, f)):
            Image.fromarray(im).save(out / f"diagnose-{k}.png")
        print("DIAGNOSE " + json.dumps(res), flush=True)
        return

    # control 1, part 1: the plain product before any adapter is attached
    e0 = experts(prompts[0])
    init, sigma = init_latents_for_cell(single.cell_for(SEEDS[0]), ctx)
    plain = to_img(run_cfg_poe(init_latents=init, euler_init_noise_sigma=sigma, **e0, **common))
    # control 2: the whole adapter, in the shape the checkpoint holds
    info = _adapter_shape.attach(ctx.models["unet"], args.adapter, disable=True)
    print("ADAPTER " + json.dumps({k: v for k, v in info.items() if k != "rank_pattern"}), flush=True)
    init, sigma = init_latents_for_cell(single.cell_for(SEEDS[0]), ctx)
    at0 = to_img(run_poe_classifier_guided(init_latents=init, euler_init_noise_sigma=sigma, adapter_lambda=0.0, strength=0.0, **e0, **common))
    diff = np.abs(plain.astype(int) - at0.astype(int))
    controls = {"control_1_weight_0_is_plain_max_abs": int(diff.max()), "control_1_passes": bool(diff.max() == 0),
                "control_2_adapter": {k: info[k] for k in ("target_modules", "rank", "n_matched", "n_loaded", "checkpoint_step")}}
    print("CONTROLS " + json.dumps(controls, default=list), flush=True)
    if not controls["control_1_passes"]:
        (out / "controls.json").write_text(json.dumps(controls, indent=1, default=list) + "\n")
        raise SystemExit("control 1 failed: weight 0 is not the plain product")

    jobs = [(p, s, w) for p in prompts[:1] for s in SEEDS[:1] for w in WEIGHTS] if args.stage == "smoke" else \
        [(p, s, w) for p in prompts for s in SEEDS for w in WEIGHTS] + [(TRAINING_PAIR, s, w) for s in SEEDS for w in (0.0, 1.0)]
    times, n = [], 0
    cache = {}
    for prompt, seed, w in jobs:
        tile = out / "renders" / slug(prompt) / f"seed-{seed}-w{w:.2f}.png"
        if tile.exists():
            continue
        tile.parent.mkdir(parents=True, exist_ok=True)
        if prompt not in cache:
            cache = {prompt: experts(prompt)}
        init, sigma = init_latents_for_cell(single.cell_for(seed), ctx)
        t0 = time.perf_counter()
        o = run_poe_classifier_guided(init_latents=init, euler_init_noise_sigma=sigma, adapter_lambda=float(w), strength=0.0,
                                      **cache[prompt], **common)
        Image.fromarray(to_img(o)).save(tile)
        times.append(time.perf_counter() - t0); n += 1
        print(f"{n} {prompt} seed={seed} w={w:.2f} {times[-1]:.1f}s", flush=True)
    summary = {**controls, "rendered": n, "median_seconds": float(np.median(times)) if times else 0.0,
               "projected_gpu_hours_full": round(float(np.median(times)) * 200 / 3600, 2) if times else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=list) + "\n")
    try:
        import wandb
        run = wandb.init(project="poe-repair-animals-compose", entity="prime_lab", name=f"loop-12-z10-{args.stage}", group="loop-12-z10",
                         config={"weights": WEIGHTS, "seeds": SEEDS, "pairs": prompts, "adapter": str(args.adapter)},
                         settings=wandb.Settings(init_timeout=60))
        run.summary.update({k: v for k, v in summary.items() if isinstance(v, (int, float, bool))}); summary["wandb_run"] = run.id; run.finish()
    except Exception as e:
        print(f"wandb not logged: {e}", flush=True)
    print(json.dumps(summary, default=list), flush=True)


if __name__ == "__main__":
    main()
