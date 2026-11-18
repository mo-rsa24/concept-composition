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

With --adapter the sheet is the rank-32 adapter instead, one row per condition and one column per
seed. The base prediction is eps_off + lambda * (eps_on - eps_off), PoE with the adapter off and on:

    a<L>            the adapter alone at lambda L, for each --adapter-alone-lambdas
    a<L>s10, ...    the adapter at --adapter-lambda plus the checker's push at that strength
    a<L>r10, ...    the adapter at --adapter-lambda plus the size-matched random push

The checker reads SDXL with the adapter switched off, the features it was trained on. Two more
checks run on the first seed before any row: adapter lambda 0 with push 0 must match the plain
product byte for byte, and push 0 at each lambda must match ``noise_search.adapter``'s
adapter-alone sampler byte for byte. --identity-only stops after them. --adapter takes a
checkpoint path, or ``untrained`` (attached, adds nothing) or ``random`` (fixed random output
matrices, a stand-in that moves the picture so the second check is not trivially true).

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
import numpy as np
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
from poe_repair.runtime import decode_latents, encode_prompt_sdxl  # noqa: E402
import classifier_guidance_from_the_paper as single  # noqa: E402
from discriminator import PairDiscriminator  # noqa: E402

RUNG2 = paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2"
OUT = RUNG2 / "results"
HEAD = paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2/discriminator/final-all-seeds/head_step_001500.pt"


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def max_pixel_diff(p: Path, q: Path) -> int:
    """Largest difference on the 0-255 scale over every pixel and channel of two tiles."""
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(q).convert("RGB"), dtype=np.int16)
    return int(np.abs(a - b).max())


def adapter_rows(args, ctx, common, disc, score, own_noise, timed, tile, record) -> int:
    """The --adapter sheet: identity checks on the first seed, then one row per condition."""
    from poe_repair.experiments.noise_search import adapter as ad

    unet = ctx.models["unet"]
    ckpt = None if args.adapter in ("untrained", "random") else Path(args.adapter)
    info = ad.attach_and_load_lora(unet, ckpt)
    if args.adapter == "random":
        g = torch.Generator().manual_seed(0)
        for name, p in unet.named_parameters():
            if "lora_B" in name:
                p.data.copy_((torch.randn(p.shape, generator=g) * 0.01).to(p.dtype))
    ad.adapter_disable(unet)
    record.update({"adapter": args.adapter, "adapter_loaded_tensors": info["n_loaded"],
                   "adapter_checkpoint_step": info["checkpoint_step"], "adapter_lambda": args.adapter_lambda,
                   "adapter_alone_lambdas": args.adapter_alone_lambdas, "adapter_identity": {}})
    print(f"adapter {args.adapter}: {info['n_loaded']} tensors loaded, step {info['checkpoint_step']}", flush=True)

    def adapter_run(seed, lam, **kw):
        init, sigma = own_noise(seed)
        return run_poe_classifier_guided(init_latents=init, euler_init_noise_sigma=sigma, head=disc,
                                         adapter_lambda=lam, push_cap=args.push_cap,
                                         guidance_window=tuple(args.window) if args.window else None, **kw, **common)

    # Identity 1: the adapter at lambda 0 with no push is the plain product.
    first = args.seeds[0]
    write_decoded_image(adapter_run(first, 0.0, strength=0.0).image, tile(first, "a0s0"))
    ident = record["adapter_identity"]
    ident["lambda0_push0_vs_plain"] = {"identical": md5(tile(first, "a0s0")) == md5(tile(first, "plain")),
                                       "max_pixel_diff": max_pixel_diff(tile(first, "a0s0"), tile(first, "plain"))}
    print(f"identity seed {first}: adapter lambda 0, push 0 vs plain product: {ident['lambda0_push0_vs_plain']}", flush=True)

    # Identity 2: push 0 at each lambda is noise_search.adapter's adapter-alone sampler.
    s = {k: common[k] for k in ("models", "scheduler", "seq_a", "pool_a", "seq_b", "pool_b", "seq_e", "pool_e",
                                "guidance_scale", "num_inference_steps", "height", "width", "device", "dtype")}
    for lam in sorted(set(args.adapter_alone_lambdas) | {args.adapter_lambda}):
        init, sigma = own_noise(first)
        with torch.no_grad():
            ref = ad.run_adapter_candidates(init_latents=init / sigma, lambda_value=lam, unet_chunk=1, **s)
            write_decoded_image(decode_latents(ctx.models, ref).cpu(), tile(first, f"a{lam:g}_ref"))
        ad.adapter_disable(unet)
        write_decoded_image(adapter_run(first, lam, strength=0.0).image, tile(first, f"a{lam:g}s0"))
        new, old = tile(first, f"a{lam:g}s0"), tile(first, f"a{lam:g}_ref")
        ident[f"lambda{lam:g}_push0_vs_adapter_alone"] = {
            "identical": md5(new) == md5(old), "max_pixel_diff": max_pixel_diff(new, old),
            "max_pixel_diff_from_plain": max_pixel_diff(new, tile(first, "plain"))}
        print(f"identity seed {first}: adapter lambda {lam:g}, push 0 vs adapter-alone sampler: "
              f"{ident[f'lambda{lam:g}_push0_vs_adapter_alone']}", flush=True)
    (OUT / "result.json").write_text(json.dumps(record, indent=2))
    if not all(v["identical"] for v in ident.values()):
        raise SystemExit("an identity check failed; stopping before any row is drawn")
    if args.identity_only:
        print("IDENTITY DONE", OUT / "result.json", flush=True)
        return 0

    L = args.adapter_lambda
    rows = [(f"adapter alone\nλ = {lam:g}", f"a{lam:g}") for lam in args.adapter_alone_lambdas] + \
           [(f"adapter λ = {L:g}\n+ checker push\ns = {s_:g}", f"a{L:g}s{s_:g}") for s_ in args.strengths] + \
           [(f"adapter λ = {L:g}\n+ random push,\nsame size, s = {s_:g}", f"a{L:g}r{s_:g}") for s_ in args.strengths]
    for seed in args.seeds:
        d = {"seconds": {}}
        for lam in args.adapter_alone_lambdas:
            out, d["seconds"][f"a{lam:g}"] = timed(lambda: adapter_run(seed, lam, strength=0.0))
            write_decoded_image(out.image, tile(seed, f"a{lam:g}"))
            d[f"a{lam:g}"] = {"final_logit": score(out.latents)}
        for s_ in args.strengths:
            key = f"a{L:g}s{s_:g}"
            guided, d["seconds"][key] = timed(lambda: adapter_run(seed, L, strength=s_))
            write_decoded_image(guided.image, tile(seed, key))
            norms, capped = guided.extras["applied_norm_per_step"], guided.extras["capped_per_step"]
            rg = torch.Generator(device="cpu").manual_seed(30_000 + 100 * seed + int(s_))
            rkey = f"a{L:g}r{s_:g}"
            rand, d["seconds"][rkey] = timed(lambda: adapter_run(seed, L, random_push_norms=norms, random_generator=rg))
            write_decoded_image(rand.image, tile(seed, rkey))
            ratio = [r / n if n > 0 else 1.0 for r, n in zip(rand.extras["applied_norm_per_step"], norms)]
            d[key] = {"applied_norm_per_step": norms, "capped_per_step": capped,
                      "cap_fired_fraction": sum(capped) / len(capped),
                      "push_over_eps_per_step": [a / e for a, e in zip(norms, guided.extras["eps_norm_per_step"])],
                      "final_logit": score(guided.latents)}
            d[rkey] = {"random_over_guided_norm_min": min(ratio), "random_over_guided_norm_max": max(ratio),
                       "final_logit": score(rand.latents)}
            print(f"seed {seed}, adapter {L:g}, s {s_:g}: cap fired on {100 * sum(capped) / len(capped):.0f}% of steps; "
                  f"random/guided norm {min(ratio):.4f} to {max(ratio):.4f}; logit guided "
                  f"{d[key]['final_logit']:.2f}, random {d[rkey]['final_logit']:.2f}", flush=True)
        record["seeds_detail"][seed] = d
        (OUT / "result.json").write_text(json.dumps(record, indent=2))

    fig, axes = plt.subplots(len(rows), len(args.seeds), figsize=(len(args.seeds) * 2.6, len(rows) * 2.7))
    for r, (label, key) in enumerate(rows):
        for c, seed in enumerate(args.seeds):
            ax = axes[r][c]
            with Image.open(tile(seed, key)) as im:
                ax.imshow(im.convert("RGB").resize((320, 320)))
            if r == 0:
                ax.set_title(f"seed {seed}", fontsize=10)
            if c == 0:
                ax.text(-0.06, 0.5, label, transform=ax.transAxes, va="center", ha="right", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Rung 2 ({args.out_name}): the cat x dog product with the rank-32 adapter ({args.adapter}), "
                 f"push cap {args.push_cap:g}, seeds {args.seeds[0]} to {args.seeds[-1]}", fontsize=12)
    fig.tight_layout(rect=[0.08, 0, 1, 0.96])
    fig.savefig(OUT / "result-sheet.png", dpi=100)
    plt.close(fig)
    print("RESULTS DONE", OUT / "result-sheet.png", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 9, 10, 11, 12])
    ap.add_argument("--strengths", type=float, nargs="+", default=[10, 30, 100])
    ap.add_argument("--head", type=Path, default=HEAD)
    ap.add_argument("--attention", action="store_true",
                    help="the layout push of attention_push.py instead of a trained head; --head is ignored")
    ap.add_argument("--readout", action="store_true",
                    help="--head is a train_readout_pair.py checkpoint; SDXL is read with the joint prompt when it was trained so")
    ap.add_argument("--push-cap", type=float, default=PUSH_CAP_FRACTION)
    ap.add_argument("--window", type=int, nargs=2, default=None, metavar=("FIRST", "STOP"),
                    help="push only on sampler steps FIRST <= step < STOP; the random push follows, since it copies "
                         "the guided push's size step by step")
    ap.add_argument("--out-name", default="results", help="folder under rung2/ for this sheet")
    ap.add_argument("--reuse-baselines", default="",
                    help="folder under rung2/ whose plain and best-of-2 tiles this sheet reuses (same head only)")
    ap.add_argument("--adapter", default=None,
                    help="draw on top of the rank-32 adapter: a checkpoint path, 'untrained' or 'random'")
    ap.add_argument("--adapter-lambda", type=float, default=1.0, help="adapter strength under the push rows")
    ap.add_argument("--adapter-alone-lambdas", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--identity-only", action="store_true", help="with --adapter: stop after the identity checks")
    ap.add_argument("--clip", type=Path, default=None,
                    help="push with a fine-tuned CLIP checker (finetune_checker.py or finetune_estimates.py best.pt) "
                         "reading the decoded estimate of the finished picture; --head is ignored")
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

    state = {} if (args.attention or args.clip) else torch.load(args.head, map_location="cpu")
    if args.clip:
        from clip_push import ClipEstimateChecker
        disc = ClipEstimateChecker(args.clip, ctx.models["vae"], ctx.device)
        args.head = args.clip
        print(f"CLIP push: {args.clip} (epoch {disc.epoch}) reads the decoded estimate of the finished picture", flush=True)
    elif args.attention:
        from attention_push import AttentionSeparation
        disc = AttentionSeparation(ctx.models["unet"], ctx.models["tokenizer"], seq_a, pool_a, seq_b, pool_b)
        args.head = Path("attention-separation")
        print("layout push: minus the correlation of the cat expert's 'cat' map and the dog expert's 'dog' map", flush=True)
    elif args.readout:
        from train_readout_pair import PairHead, ReadoutPairClassifier
        seq_r, pool_r = enc("a cat and a dog") if state.get("pair_prompt") else (seq_e, pool_e)
        head = PairHead(n_out=state.get("heads", 1))
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
    if args.adapter:
        return adapter_rows(args, ctx, common, disc, score, own_noise, timed, tile, record)

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
