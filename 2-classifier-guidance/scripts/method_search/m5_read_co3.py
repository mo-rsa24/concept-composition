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
       python m5_read_co3.py read ... --shard <k> --num-shards 2 --save-latents      keep the states, on the card type that drew shard k
       python m5_read_co3.py heads --heads <name>=<pt> ... --latents-from <dir> ... --out <dir> --calls <csv> --key <json>

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
    if args.seeds:                                    # fresh seeds, such as 6009-6016, that no loop has read
        lo, hi = (int(v) for v in args.seeds.split("-"))
        seeds = list(range(lo, hi + 1))
        if set(seeds) & set(range(7001, 7009)):
            raise SystemExit("seeds 7001 to 7008 are held out")
    pairs = pairs[args.shard::args.num_shards]       # M1's split: shard 0 drew on a Quadro RTX 8000, shard 1 on an RTX 3090
    prompts = {pair: joint_prompt(pair) for pair in pairs}
    if any(not is_allowed(p) for p in prompts.values()):
        raise SystemExit("the subject guard refuses a prompt")
    parser_nlp = load_parser()
    model, config = load(prompts[pairs[0]], parser_nlp, args.out / "_config")
    real_unet = model.unet
    real_unet.requires_grad_(False)
    checker, head_step = build_checker(model, real_unet, args.head, prompts[pairs[0]])
    state = {"step": None, "reads": {}, "states": {}}

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
                if args.save_latents:
                    state["states"][state["step"]] = {"x": sample[:1].detach().to(torch.float16).cpu(), "t": int(timestep.reshape(-1)[0])}
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
                state["reads"], state["states"] = {}, {}
                begin = time.time()
                seed_everything(seed)
                image = model.sample_loop(start_noise(model, seed))[0]
                folder = args.out / "renders" / pair
                folder.mkdir(parents=True, exist_ok=True)
                image.save(folder / f"seed-{seed}.png")
                if args.save_latents:
                    (args.out / "latents" / pair).mkdir(parents=True, exist_ok=True)
                    torch.save(state["states"], args.out / "latents" / pair / f"seed-{seed}.pt")
                if args.m1_co3:
                    with Image.open(args.m1_co3 / pair / f"seed-{seed}.png") as old:
                        diff = float(np.abs(np.asarray(image.convert("RGB"), dtype=np.float64)
                                            - np.asarray(old.convert("RGB"), dtype=np.float64)).mean())
                else:                                   # fresh seeds: no earlier picture to reproduce
                    diff = 0.0
                row = {"key": key, "pair": pair, "seed": seed, "head_step": head_step, "pixel_diff_to_m1_of_255": round(diff, 3),
                       "reproduced": diff <= MAX_PIXEL_DIFF, "fresh": not args.m1_co3, "reads": {str(s): v for s, v in state["reads"].items()},
                       "seconds": round(time.time() - begin, 1)}
                log.write(json.dumps(row) + "\n")
                log.flush()
                done.add(key)
                print(f"{key} diff {diff:.2f} steps read {sorted(state['reads'])} {row['seconds']} s", flush=True)


def heads(args):
    """Score further checkers on the saved states of earlier reads, with no CO3 run. The kept checker is read again as a control:
    its scores must match the recorded ones, so the saved states stand in for the trajectories."""
    import torch
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parent / "rung2"))
    from co3_common import load, load_parser
    from paper2_rows import build_checker, point_checker
    from train_readout_pair import joint_prompt

    source = {}
    for folder in args.latents_from:
        for r in map(json.loads, (folder / "reads.jsonl").read_text().splitlines()):
            path = folder / "latents" / r["pair"] / f"seed-{r['seed']}.pt"
            if path.exists() and (r["key"] not in source or (r["reproduced"] and not source[r["key"]][0]["reproduced"])):
                source[r["key"]] = (r, path)
    print(f"{len(source)} saved trajectories, {sum(r['reproduced'] for r, _ in source.values())} reproduced", flush=True)
    first = next(iter(source.values()))[0]["pair"]
    model, config = load(joint_prompt(first), load_parser(), args.out / "_config")
    real_unet = model.unet
    real_unet.requires_grad_(False)
    for spec in args.heads:
        name, path = spec.split("=", 1)
        checker, head_step = build_checker(model, real_unet, Path(path), joint_prompt(first))
        out = args.out / name
        out.mkdir(parents=True, exist_ok=True)
        largest = 0.0
        with open(out / "reads.jsonl", "w") as log:
            for key, (r, latent_path) in sorted(source.items()):
                point_checker(model, checker, joint_prompt(r["pair"]))
                saved = torch.load(latent_path, weights_only=False)
                reads = {}
                for step, st in sorted(saved.items()):
                    with torch.no_grad():
                        x = st["x"].to(real_unet.device)
                        t = torch.tensor([st["t"]], device=real_unet.device)
                        o = checker.head(checker.reader(x.to(checker.reader.seq_e.dtype), t))
                    reads[str(step)] = [float(v) for v in o.reshape(-1).float().cpu()]
                    if str(step) in r["reads"]:
                        largest = max(largest, abs(reads[str(step)][0] - r["reads"][str(step)][0]))
                log.write(json.dumps({**{k: r[k] for k in ("key", "pair", "seed", "pixel_diff_to_m1_of_255", "reproduced")},
                                      "head": path, "head_step": head_step, "reads": reads}) + "\n")
        (out / "control.json").write_text(json.dumps({"head": path, "largest_difference_to_recorded_kept_scores": largest}) + "\n")
        print(f"{name}: {len(source)} read; largest difference to the kept checker's recorded scores {largest:.4f}", flush=True)
        score(argparse.Namespace(out=out, also=None, calls=args.calls, key=args.key))


PERPLACE_STEPS = (5, 10, 15, 20, 25, 30)
PERPLACE_CALL = 0.30          # "two clear" when the less-represented animal's share of cells is at least this (reviewer, 2026-11-25)
CFGPP_SCALE = 0.8             # CO3's released guidance, used for the one-step clean estimate the judge reads


def perplace(args):
    """Step M12: the per-place prediction difference on CO3's saved states, with no training. At each saved step, SDXL predicts the
    noise under the empty, joint, first-animal and second-animal prompts. Per latent cell, the first animal's distance to the joint
    prediction minus the second's, each over the 4 channels; the score is the share of cells closer to the less-represented animal.
    Beside it, the one-step clean estimate is decoded for the judge (a diagnostic only)."""
    import torch
    from PIL import Image
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parent / "rung2"))
    sys.path.insert(0, str(HERE.parents[2]))
    from co3_common import load, load_parser
    from paper2_rows import encode
    from poe_repair.methods._sampling import add_time_ids
    from train_readout_pair import joint_prompt
    if args.count:
        from poe_repair.experiments.compose_scorer_validation.detection_scorer import count_instances

    source = {}
    for folder in args.latents_from:
        for r in map(json.loads, (folder / "reads.jsonl").read_text().splitlines()):
            path = folder / "latents" / r["pair"] / f"seed-{r['seed']}.pt"
            if path.exists() and r["reproduced"]:
                source[r["key"]] = (r, path)
    print(f"{len(source)} reproduced trajectories with saved states", flush=True)
    first = next(iter(source.values()))[0]["pair"]
    model, _ = load(joint_prompt(first), load_parser(), args.out / "_config")
    unet = model.unet
    vae = model.vae.to(torch.float32)      # SDXL's VAE overflows in fp16, so it decodes in fp32 (environment/overview.md)
    vae_dtype = torch.float32
    abar = model.scheduler.alphas_cumprod.to(unet.device)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "clean-estimates").mkdir(exist_ok=True)
    done = set()
    if (args.out / "reads.jsonl").exists():
        done = {json.loads(l)["key"] for l in (args.out / "reads.jsonl").read_text().splitlines()}
    with open(args.out / "reads.jsonl", "a") as log:
        for key, (r, latent_path) in sorted(source.items()):
            if key in done:
                continue
            a, b = (w.replace("_", " ") for w in r["pair"].split("__x__"))
            prompts = ["", joint_prompt(r["pair"]), a, b]
            seqs, pools = zip(*(encode(model, q) for q in prompts))
            seq, pool = torch.cat(seqs), torch.cat(pools)
            saved = torch.load(latent_path, weights_only=False)
            reads, shares, counts = {}, {}, {}
            for step in PERPLACE_STEPS:
                st = saved[step]
                x = st["x"].to(unet.device, torch.float16).repeat(4, 1, 1, 1)
                t = torch.full((4,), st["t"], device=unet.device)
                cond = {"text_embeds": pool, "time_ids": add_time_ids(height=1024, width=1024, batch_size=4, device=unet.device,
                                                                         dtype=torch.float16)}
                with torch.no_grad():
                    eps = unet(x, t, encoder_hidden_states=seq, added_cond_kwargs=cond).sample.float()
                empty, joint, ea, eb = eps
                diff = (ea - joint).norm(dim=0) - (eb - joint).norm(dim=0)
                first_closer = float((diff < 0).float().mean())
                shares[str(step)] = round(first_closer, 4)
                reads[str(step)] = [min(first_closer, 1 - first_closer)]
                with torch.no_grad():
                    a_t = abar[st["t"]]
                    guided = empty + CFGPP_SCALE * (joint - empty)
                    x0 = (x[0].float() - (1 - a_t).sqrt() * guided) / a_t.sqrt()
                    img = vae.decode((x0[None] / vae.config.scaling_factor).to(vae_dtype)).sample[0].float()
                img = ((img.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
                folder = args.out / "clean-estimates" / f"step-{step:02d}" / r["pair"]
                folder.mkdir(parents=True, exist_ok=True)
                Image.fromarray(img).save(folder / f"seed-{r['seed']}.png")
                if args.count and not r["pair"].startswith("a_lighthouse"):
                    n, _ = count_instances(folder / f"seed-{r['seed']}.png", device=torch.device(unet.device))
                    counts[str(step)] = int(n)
            log.write(json.dumps({**{k: r[k] for k in ("key", "pair", "seed", "pixel_diff_to_m1_of_255", "reproduced")},
                                  "reads": reads, "first_animal_closer_share": shares, "animal_count_on_clean_estimate": counts}) + "\n")
            log.flush()
            print(f"{key} less-represented share by step {[round(v[0], 3) for v in reads.values()]}", flush=True)
    if args.calls:
        score(argparse.Namespace(out=args.out, also=None, calls=args.calls, key=args.key, threshold=PERPLACE_CALL, steps=PERPLACE_STEPS))


EXPECTED_SEEDS = (8001, 8002, 8003, 8004)
EXPECTED_STEPS, EXPECTED_GUIDANCE, EXPECTED_MIN_CONF = 30, 5.0, 0.30


def expected(args):
    """Step M12's second reader: how much of the picture each subject fills when plain SDXL draws it alone. For each subject word,
    "a <subject>" at seeds 8001 to 8004, 30 steps, guidance 5.0, 1024 by 1024, from the SDXL weights CO3 loads; the detector, asked for the
    subject's own word, gives the highest-confidence box at 0.30 or more, and its area fraction. Writes expected.json with each pair's
    expected shares."""
    import torch
    from diffusers import StableDiffusionXLPipeline
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parent / "rung2"))
    sys.path.insert(0, str(HERE.parents[2]))
    from co3_common import load, load_parser
    from paper2_rows import tuning
    from train_readout_pair import joint_prompt
    from poe_repair.experiments.residual_between_mono_and_poe.metrics import detect_boxes

    pairs, _ = tuning(args.splits)
    model, _ = load(joint_prompt(pairs[0]), load_parser(), args.out / "_config")
    import copy
    vae32 = copy.deepcopy(model.vae).to(torch.float32)   # SDXL's VAE overflows in fp16, so the pipeline stops at latents and this decodes
    pipe = StableDiffusionXLPipeline(vae=model.vae, text_encoder=model.text_encoder,
                                     text_encoder_2=model.text_encoder_2, tokenizer=model.tokenizer, tokenizer_2=model.tokenizer_2,
                                     unet=model.unet, scheduler=model.scheduler)
    pipe.set_progress_bar_config(disable=True)
    args.out.mkdir(parents=True, exist_ok=True)
    subjects = sorted({w for p in pairs for w in p.split("__x__")})
    areas = {}
    for subject in subjects:
        word = subject.split("_", 1)[1].replace("_", " ")
        per_seed = {}
        for seed in EXPECTED_SEEDS:
            path = args.out / "alone" / subject / f"seed-{seed}.png"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                g = torch.Generator(device=model.unet.device).manual_seed(seed)
                with torch.no_grad():
                    latents = pipe(subject.replace("_", " "), num_inference_steps=EXPECTED_STEPS, guidance_scale=EXPECTED_GUIDANCE,
                                   height=1024, width=1024, generator=g, output_type="latent").images
                    pixels = vae32.decode(latents.to(torch.float32) / vae32.config.scaling_factor).sample[0]
                pixels = ((pixels.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
                from PIL import Image
                Image.fromarray(pixels).save(path)
            boxes = [d for d in detect_boxes(path, [word], device=torch.device(model.unet.device)) if d["confidence"] >= EXPECTED_MIN_CONF]
            if boxes:
                x0, y0, x1, y1 = max(boxes, key=lambda d: d["confidence"])["box"]
                per_seed[seed] = round((x1 - x0) * (y1 - y0) / (1024 * 1024), 4)
        areas[subject] = {"per_seed": per_seed, "area": round(float(np.mean(list(per_seed.values()))), 4) if per_seed else None}
        print(subject, areas[subject], flush=True)
    result = {"seeds": EXPECTED_SEEDS, "steps": EXPECTED_STEPS, "guidance": EXPECTED_GUIDANCE, "min_confidence": EXPECTED_MIN_CONF,
              "areas": areas, "pairs": {}}
    for pair in pairs:
        a, b = pair.split("__x__")
        fa, fb = areas[a]["area"], areas[b]["area"]
        share = 0.5 if fa is None or fb is None else fa / (fa + fb)
        result["pairs"][pair] = {"first_expected_share": round(share, 4), "fallback": fa is None or fb is None}
    (args.out / "expected.json").write_text(json.dumps(result, indent=1) + "\n")


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
    threshold = getattr(args, "threshold", 0.0)
    for step in getattr(args, "steps", None) or READ_STEPS:
        use = [r for r in kept if str(step) in r["reads"]]
        s = [r["reads"][str(step)][0] for r in use]
        lab = [clear[r["key"]] for r in use]
        pos = [x >= threshold if threshold else x > 0 for x, l in zip(s, lab) if l]
        neg = [not (x >= threshold if threshold else x > 0) for x, l in zip(s, lab) if not l]
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
    # Board question 96 asks whether the gate should be the area under the curve; until the owner answers, the gate above stands, and a
    # checker that reaches the bar only on ranking is reported as that, beside the verdict, deciding nothing.
    auc_best = max(per_step, key=lambda s: per_step[s]["area_under_curve"])
    auc_top = per_step[auc_best]["area_under_curve"]
    ranking_only = top < BAR and auc_top >= BAR
    result = {"runs": len(rows), "reproduced_and_labelled": len(kept), "clear": sum(clear[r["key"]] for r in kept),
              "bar": BAR, "floor": FLOOR, "best_step": best, "best_balanced_accuracy": top, "verdict": verdict,
              "best_auc_step": auc_best, "best_area_under_curve": auc_top,
              "outcome_on_ranking": "reaches the bar on ranking only (question 96)" if ranking_only else None, "per_step": per_step}
    (args.out / ("read-score-merged.json" if args.also else "read-score.json")).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("runs", "reproduced_and_labelled", "clear", "best_step", "best_balanced_accuracy", "verdict",
                                           "best_auc_step", "best_area_under_curve", "outcome_on_ranking")}))
    print({s: (v["balanced_accuracy"], v["area_under_curve"]) for s, v in per_step.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["read", "score", "heads", "perplace", "expected"])
    ap.add_argument("--head", type=Path)
    ap.add_argument("--splits", type=Path)
    ap.add_argument("--m1-co3", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--calls", type=Path)
    ap.add_argument("--key", type=Path)
    ap.add_argument("--redo-from", type=Path, help="an earlier reads.jsonl; rerun only its trajectories that did not reproduce")
    ap.add_argument("--also", type=Path, nargs="*", help="score: further read folders, merged, a reproduced row winning")
    ap.add_argument("--seeds", help="read: a range such as 6009-6016 in place of the tuning seeds, with no earlier picture to match")
    ap.add_argument("--count", action="store_true", help="perplace: count animals on each clean estimate with the validated detector")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--save-latents", action="store_true", help="read: keep each read step's state, so later checkers need no CO3 run")
    ap.add_argument("--heads", nargs="*", help="heads: name=path for each checker to score on the saved states")
    ap.add_argument("--latents-from", type=Path, nargs="*", help="heads: read folders holding reads.jsonl and latents/")
    ap.add_argument("--limit", type=int, help="stop after this many runs, for the smoke rung")
    args = ap.parse_args()
    {"read": read, "score": score, "heads": heads, "perplace": perplace, "expected": expected}[args.stage](args)


if __name__ == "__main__":
    main()
