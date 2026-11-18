#!/usr/bin/env python
"""Rung 2's result sheet: plain product, best-of-N, the guided push and a size-matched random push.

Every column is SDXL's product of "a cat" and "a dog", 50 steps, on one card, and only how the
discriminator is used changes between columns:

    plain       the product from the seed's own starting noise
    best_of_2   two plain products, the seed's own noise and one fresh draw (generator
                20,000 + seed); the discriminator's logit on each final latent at the last sampler
                timestep picks one. N = 2 from plan 21's timing.
    s10, s30, s100      the product pushed by the discriminator's log-odds gradient at that strength,
                        capped at 0.1 of the product prediction's norm
    r10, r30, r100      the same seed and strength, the push replaced by a fresh random direction
                        scaled to the norm the guided push had at that step

Before any push is read, strength 0 with the discriminator attached must match the plain product
byte for byte on the first seed, or the run stops.

Writes, under MOUNT_ROOT/outputs/classifier_guidance_ladder/rung2/results/:
    tiles/seedNN__<column>.png, tiles/seedNN__draw2.png, result-sheet.png, result.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import time
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from poe_repair import paths  # noqa: E402
from poe_repair.composers._helpers import init_latents_for_cell  # noqa: E402
from poe_repair.methods._classifier_guided import PUSH_CAP_FRACTION, run_poe_classifier_guided  # noqa: E402
from poe_repair.methods._sampling import run_cfg_poe, write_decoded_image  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import encode_prompt_sdxl  # noqa: E402
import classifier_guidance_from_the_paper as single  # noqa: E402
from discriminator import PairDiscriminator  # noqa: E402

RUNG2 = paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2"
OUT = RUNG2 / "results"
HEAD = paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2/discriminator/final-all-seeds/head_step_001500.pt"


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 9, 10, 11, 12])
    ap.add_argument("--strengths", type=float, nargs="+", default=[10, 30, 100])
    ap.add_argument("--head", type=Path, default=HEAD)
    ap.add_argument("--readout", action="store_true",
                    help="--head is a train_readout_pair.py checkpoint; SDXL is read with the joint prompt when it was trained so")
    ap.add_argument("--push-cap", type=float, default=PUSH_CAP_FRACTION)
    ap.add_argument("--window", type=int, nargs=2, default=None, metavar=("FIRST", "STOP"),
                    help="push only on sampler steps FIRST <= step < STOP; the random push follows, since it copies "
                         "the guided push's size step by step")
    ap.add_argument("--out-name", default="results", help="folder under rung2/ for this sheet")
    ap.add_argument("--reuse-baselines", default="",
                    help="folder under rung2/ whose plain and best-of-2 tiles this sheet reuses (same head only)")
    args = ap.parse_args()
    global OUT
    OUT = RUNG2 / args.out_name
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card; co3 on a Blackwell runs on the CPU in silence")
    print(f"{socket.gethostname()}: {torch.cuda.get_device_name(0)}, seeds {args.seeds}", flush=True)
    single.disk_guard(OUT)
    (OUT / "tiles").mkdir(parents=True, exist_ok=True)
    ctx = make_ctx(output_root=OUT / "raw")
    enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = enc("")
    seq_a, pool_a = enc("a cat")
    seq_b, pool_b = enc("a dog")
    common = dict(models=ctx.models, scheduler=ctx.scheduler, seq_a=seq_a, pool_a=pool_a,
                  seq_b=seq_b, pool_b=pool_b, seq_e=seq_e, pool_e=pool_e,
                  guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
                  height=1024, width=1024, device=ctx.device, dtype=ctx.dtype)

    state = torch.load(args.head, map_location="cpu")
    if args.readout:
        from train_readout_pair import PairHead, ReadoutPairClassifier
        seq_r, pool_r = enc("a cat and a dog") if state.get("pair_prompt") else (seq_e, pool_e)
        head = PairHead()
        head.load_state_dict(state["head"])
        disc = ReadoutPairClassifier(ctx.models["unet"], seq_r, pool_r, head.float().eval()).to(ctx.device)
        print(f"loaded readout checker {args.head} (step {state['step']}, joint prompt {bool(state.get('pair_prompt'))})", flush=True)
    else:
        disc = PairDiscriminator(ctx.models["unet"], seq_e, pool_e).to(ctx.device)
        disc.head.load_state_dict(state["head"])
        disc.head.float().eval()
        print(f"loaded {args.head} (trained on seeds {state['split']['train_seeds']})", flush=True)
    ctx.scheduler.set_timesteps(ctx.num_inference_steps)
    t_last = ctx.scheduler.timesteps[-1].reshape(1).to(ctx.device)

    @torch.no_grad()
    def score(latents):
        return float(disc(latents.float().to(ctx.device), t_last)[0])

    def own_noise(seed):
        return init_latents_for_cell(single.cell_for(seed), ctx)

    def timed(fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        o = fn()
        torch.cuda.synchronize()
        return o, time.perf_counter() - t0

    tile = lambda seed, col: OUT / "tiles" / f"seed{seed:02d}__{col}.png"
    base = RUNG2 / args.reuse_baselines if args.reuse_baselines else None
    base_record = json.loads((base / "result.json").read_text()) if base else None
    if base_record and base_record["head"] != str(args.head):
        raise SystemExit("guard: reused baselines were picked by a different discriminator")
    record = {"push_cap": args.push_cap, "window": args.window, "reused_baselines": str(base) if base else None,
              "host": socket.gethostname(), "card": torch.cuda.get_device_name(0), "head": str(args.head),
              "seeds": args.seeds, "strengths": args.strengths, "best_of_n": 2, "canary": {}, "seeds_detail": {}}

    first = args.seeds[0]
    init, sigma = own_noise(first)
    o = run_cfg_poe(init_latents=init, euler_init_noise_sigma=sigma, **common)
    write_decoded_image(o.image, tile(first, "plain"))
    init, sigma = own_noise(first)
    run_s0 = run_poe_classifier_guided(init_latents=init, euler_init_noise_sigma=sigma, head=disc,
                                       strength=0.0, **common)
    write_decoded_image(run_s0.image, tile(first, "s0"))
    same = md5(tile(first, "plain")) == md5(tile(first, "s0"))
    record["canary"] = {"seed": first, "strength_0_identical_to_plain": same}
    print(f"canary seed {first}: strength 0 identical to plain product: {same}", flush=True)
    if not same:
        raise SystemExit("strength 0 moved the product; stopping before any push is read")

    for seed in args.seeds:
        d = {"seconds": {}}
        if base:
            for col in ("plain", "draw2", "best_of_2"):
                Image.open(base / "tiles" / f"seed{seed:02d}__{col}.png").save(tile(seed, col))
            old_d = base_record["seeds_detail"][str(seed)]
            d["best_of_2"] = old_d["best_of_2"]
            d["seconds"].update({k: old_d["seconds"][k] for k in ("plain", "draw2")})
            pick, scores = d["best_of_2"]["picked"], d["best_of_2"]["logits"]
        else:
            init, sigma = own_noise(seed)
            plain, d["seconds"]["plain"] = timed(lambda: run_cfg_poe(init_latents=init, euler_init_noise_sigma=sigma, **common))
            write_decoded_image(plain.image, tile(seed, "plain"))
            g = torch.Generator(device="cpu").manual_seed(20_000 + seed)
            noise2 = torch.randn(1, 4, 128, 128, generator=g).to(device=ctx.device, dtype=ctx.dtype)
            draw2, d["seconds"]["draw2"] = timed(lambda: run_cfg_poe(init_latents=noise2, euler_init_noise_sigma=1.0, **common))
            write_decoded_image(draw2.image, tile(seed, "draw2"))
            scores = {"own_noise": score(plain.latents), "draw2": score(draw2.latents)}
            pick = max(scores, key=scores.get)
            Image.open(tile(seed, "plain" if pick == "own_noise" else "draw2")).save(tile(seed, "best_of_2"))
            d["best_of_2"] = {"logits": scores, "picked": pick}
        for s_ in args.strengths:
            init, sigma = own_noise(seed)
            guided, d["seconds"][f"s{s_:g}"] = timed(lambda: run_poe_classifier_guided(
                init_latents=init, euler_init_noise_sigma=sigma, head=disc, strength=s_,
                push_cap=args.push_cap, guidance_window=tuple(args.window) if args.window else None, **common))
            write_decoded_image(guided.image, tile(seed, f"s{s_:g}"))
            norms = guided.extras["applied_norm_per_step"]
            init, sigma = own_noise(seed)
            rg = torch.Generator(device="cpu").manual_seed(30_000 + 100 * seed + int(s_))
            rand, d["seconds"][f"r{s_:g}"] = timed(lambda: run_poe_classifier_guided(
                init_latents=init, euler_init_noise_sigma=sigma, random_push_norms=norms,
                random_generator=rg, **common))
            write_decoded_image(rand.image, tile(seed, f"r{s_:g}"))
            ratio = [r / n if n > 0 else 1.0 for r, n in zip(rand.extras["applied_norm_per_step"], norms)]
            capped = guided.extras["capped_per_step"]
            d[f"s{s_:g}"] = {"applied_norm_per_step": norms, "capped_per_step": capped,
                             "cap_fired_fraction": sum(capped) / len(capped),
                             "push_over_eps_per_step": [a / e for a, e in zip(norms, guided.extras["eps_norm_per_step"])],
                             "final_logit": score(guided.latents)}
            d[f"r{s_:g}"] = {"random_over_guided_norm_min": min(ratio), "random_over_guided_norm_max": max(ratio),
                             "final_logit": score(rand.latents)}
            print(f"seed {seed}, s {s_:g}: cap fired on {100 * sum(capped) / len(capped):.0f}% of steps; "
                  f"random/guided norm {min(ratio):.4f} to {max(ratio):.4f}; logit guided "
                  f"{d[f's{s_:g}']['final_logit']:.2f}, random {d[f'r{s_:g}']['final_logit']:.2f}", flush=True)
        record["seeds_detail"][seed] = d
        print(f"seed {seed}: best of 2 picked {pick} (logits {scores['own_noise']:.2f}, {scores['draw2']:.2f})",
              flush=True)
        (OUT / "result.json").write_text(json.dumps(record, indent=2))

    secs = [r["seconds"] for r in record["seeds_detail"].values()]
    record["realised_gpu_seconds"] = {
        "best_of_2_per_seed_median": sorted(x["plain"] + x["draw2"] for x in secs)[len(secs) // 2],
        "guided_per_seed_median": {f"s{s_:g}": sorted(x[f"s{s_:g}"] for x in secs)[len(secs) // 2]
                                   for s_ in args.strengths}}
    (OUT / "result.json").write_text(json.dumps(record, indent=2))

    cols = [("plain product", "plain"), ("best of 2,\npicked by the\ndiscriminator", "best_of_2")] + \
           [(f"guided\ns = {s_:g}", f"s{s_:g}") for s_ in args.strengths] + \
           [(f"random push,\nsame size, s = {s_:g}", f"r{s_:g}") for s_ in args.strengths]
    fig, axes = plt.subplots(len(args.seeds), len(cols), figsize=(len(cols) * 2.6, len(args.seeds) * 2.7))
    for r, seed in enumerate(args.seeds):
        for c, (label, key) in enumerate(cols):
            ax = axes[r][c]
            with Image.open(tile(seed, key)) as im:
                ax.imshow(im.convert("RGB").resize((320, 320)))
            if r == 0:
                ax.set_title(label, fontsize=10)
            if c == 0:
                part = "training seed" if seed <= 8 else "held-out seed"
                ax.text(-0.06, 0.5, f"seed {seed}\n{part}", transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Rung 2 ({args.out_name}, cap {args.push_cap:g}"
                 f"{f', push on steps {args.window[0]} to {args.window[1] - 1}' if args.window else ''}): the cat x dog product, seeds "
                 f"{', '.join(str(x) for x in args.seeds)} (9 to 16 held out)", fontsize=12)
    fig.tight_layout(rect=[0.02, 0, 1, 0.96])
    fig.savefig(OUT / "result-sheet.png", dpi=100)
    plt.close(fig)
    print("RESULTS DONE", OUT / "result-sheet.png", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
