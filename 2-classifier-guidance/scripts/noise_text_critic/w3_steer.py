#!/usr/bin/env python
"""Step W3: push CO3 with the gradient of one right-minus-wrong sentence score.

The design is noise-text-critic/03-does-it-steer/design.md.

  choose  pick the model, property and steps from W2's result.json, by the design's rule
  clamp   on W2's unflipped pictures, compare the hard clamp with the soft clamp and record the choice
  check   the concept parse on all tuning prompts, the zero push, and the push length
  render  CO3 alone, CO3 with the push, and a random push of the same length

Usage: python w3_steer.py choose --w2 <result.json> --out <dir>
       python w3_steer.py clamp --w2-prep <dir> --out <dir>
       python w3_steer.py check --splits <json> --out <dir>
       python w3_steer.py render --splits <json> --out <dir> [--pairs-limit 1 --seeds-limit 1]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "method_search"))
from common import (check as guard, image_features, latent_clip_image, latent_clip_text, latent_to_rgb, load_clip,  # noqa: E402
                    load_latent_clip, rgb_to_clip, text_features)

# Fixed in 03-does-it-steer/design.md before any W2 score.
PUSH_LENGTH = 0.4
READ_GATE = 0.6
TIE = 0.005
SATURATION_BAR = 0.05
SOFT_COSINE_BAR = 0.99
SOFT_BETA = 0.05
ZERO_PUSH_MAX_ABS = 1e-5
ROWS = ("co3", "push", "random")
CANDIDATE_MODELS = ("noisyclip", "latent_clip")
CANDIDATE_PROPERTIES = ("counting", "prompt_match")
OBJECTS = {"lighthouse", "sailboat"}
PLURAL = {"puma": "pumas", "condor": "condors", "zebra": "zebras", "flamingo": "flamingos", "seal": "seals", "gull": "gulls",
          "hippopotamus": "hippopotamuses", "swallow": "swallows", "giraffe": "giraffes", "tortoise": "tortoises",
          "bobcat": "bobcats", "quail": "quails", "duck": "ducks", "goose": "geese", "crow": "crows", "raven": "ravens",
          "eagle": "eagles", "fox": "foxes", "lighthouse": "lighthouses", "sailboat": "sailboats", "boar": "boars",
          "pheasant": "pheasants", "cheetah": "cheetahs", "cougar": "cougars"}


def pair_words(pair: str):
    (art_a, a), (art_b, b) = (s.split("_", 1) for s in pair.split("__x__"))
    return art_a, a.replace("_", " "), art_b, b.replace("_", " ")


def joint_prompt(pair: str) -> str:
    art_a, a, art_b, b = pair_words(pair)
    return f"{art_a} {a} and {art_b} {b}"


def sentences(pair: str, prop: str) -> tuple[str, tuple[str, ...]]:
    art_a, a, art_b, b = pair_words(pair)
    phrase = f"{art_a} {a} and {art_b} {b}"
    noun = "object" if a in OBJECTS and b in OBJECTS else "animal"
    if prop == "counting":
        right, wrong = f"a photo of {phrase}, two separate {noun}s", (f"a photo of one {noun}", f"one {noun} that is part {a} and part {b}")
    elif prop == "prompt_match":
        right, wrong = f"a photo of {phrase}", (f"a photo of two {PLURAL[a]}", f"a photo of two {PLURAL[b]}", "a photo of a horse and a sheep")
    else:
        raise ValueError(prop)
    for s in (right, *wrong):
        guard(s)
    return right, wrong


def choose(result: dict) -> dict:
    """The design's rule. It reads only the threshold half."""
    steps = result["steps"]
    main = [i for i, s in enumerate(steps) if s <= 15]
    rows = []
    for model in CANDIDATE_MODELS:
        for prop in CANDIDATE_PROPERTIES:
            ba = result[prop].get(model, {}).get("balanced_accuracy_threshold_half")
            if ba is None:
                continue
            values = [ba[i] for i in main]
            rows.append({"model": model, "property": prop, "mean_threshold_half": sum(values) / len(values),
                         "push_steps": [steps[i] for i in main if ba[i] >= READ_GATE]})
    usable = [r for r in rows if r["push_steps"]]
    if not usable:
        return {"chosen": None, "candidates": rows, "why": "no candidate reads at 0.6 on any step from 0 to 15"}
    best = max(r["mean_threshold_half"] for r in usable)
    close = [r for r in usable if best - r["mean_threshold_half"] <= TIE]
    noisy = [r for r in close if r["model"] == "noisyclip"]
    chosen = max(noisy or close, key=lambda r: r["mean_threshold_half"])
    return {"chosen": chosen, "candidates": rows}


def soft_clamp(v: torch.Tensor, beta: float = SOFT_BETA) -> torch.Tensor:
    """Close to v inside 0 to 255, close to the limit outside, with a gradient everywhere."""
    return v - F.softplus(beta * (v - 255.0)) / beta + F.softplus(-beta * v) / beta


class SentenceScore:
    """The checker score D(x): the right sentence's cosine minus the highest wrong sentence's."""

    def __init__(self, which: str, device, clamp: str = "hard"):
        self.which, self.device, self.clamp = which, device, clamp
        if which == "latent_clip":
            self.model, self.tokenizer = load_latent_clip(device)
        else:
            self.model, self.tokenizer = load_clip("noisyclip" if which == "noisyclip" else "frozen", device)
        self.right, self.wrong = None, None

    def embed(self, texts):
        if self.which == "latent_clip":
            return latent_clip_text(self.model, self.tokenizer, texts, self.device)
        return text_features(self.model, self.tokenizer, texts, self.device)

    def set_sentences(self, right: str, wrong) -> None:
        self.right, self.wrong = self.embed([right]), self.embed(list(wrong))

    def image(self, x: torch.Tensor) -> torch.Tensor:
        if self.which == "latent_clip":
            return latent_clip_image(self.model, x)
        rgb = latent_to_rgb(x)
        rgb = soft_clamp(rgb) if self.clamp == "soft" else rgb.clamp(0, 255)
        return image_features(self.model, rgb_to_clip(rgb))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        f = self.image(x)
        return (f @ self.right.T)[:, 0] - (f @ self.wrong.T).max(1).values


class SentencePusher:
    """One render's push, at a fixed length. mode: "push", "zero" (the same path times zero) or "random"."""

    def __init__(self, score, guidance, steps, mode="push", norms=None, generator=None, length=PUSH_LENGTH):
        self.score, self.guidance, self.steps, self.mode = score, float(guidance), set(steps), mode
        self.norms, self.generator, self.length = norms, generator, float(length)
        self.step, self.alpha_bar = None, None
        self.applied, self.capped, self.main_calls, self.dead = [], [], {}, 0

    def __call__(self, x, timestep, eps):
        self.main_calls[self.step] = self.main_calls.get(self.step, 0) + 1
        if self.main_calls[self.step] > 1:
            raise RuntimeError(f"CO3 made more than one two-entry UNet call on step {self.step}")
        self.capped.append(False)
        if self.step not in self.steps:
            self.applied.append(0.0)
            return torch.zeros_like(x)
        if self.mode == "random":
            direction = torch.randn(x.shape, generator=self.generator).to(device=x.device, dtype=torch.float32)
            push = direction * (float(self.norms[self.step]) / direction.norm())
            self.applied.append(float(push.norm()))
            return push
        with torch.enable_grad(), torch.autocast(device_type=x.device.type, enabled=False):
            z = x.detach().float().requires_grad_(True)
            grad = torch.autograd.grad(self.score(z).sum(), z)[0]
        norm = grad.norm()
        if not torch.isfinite(norm) or norm == 0:
            self.dead += 1
            self.applied.append(0.0)
            return torch.zeros_like(x)
        uncond, cond = eps[:1].float(), eps[1:2].float()
        size = self.length * (uncond + self.guidance * (cond - uncond)).norm()
        push = grad / norm * size * (0.0 if self.mode == "zero" else 1.0)
        self.applied.append(float(push.norm()))
        return push


def stage_choose(args) -> int:
    result = choose(json.loads(args.w2.read_text()))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "choice.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1), flush=True)
    return 0


@torch.no_grad()
def stage_clamp(args) -> int:
    from common import LOOP
    from poe_repair._sdxl.runtime import load_ddim_scheduler

    choice = json.loads((args.out / "choice.json").read_text())["chosen"]
    if choice is None:
        raise SystemExit("no score was chosen, so W3 renders nothing")
    if choice["model"] == "latent_clip":
        out = {"clamp": "none", "why": "Latent CLIP reads the latent directly"}
    else:
        device = torch.device("cuda")
        derived = torch.load((args.w2_prep or LOOP / "w2/prep") / "derived.pt", map_location="cpu", weights_only=True)
        keep = [i for i, k in enumerate(derived["ids"]) if k.endswith("::unflipped")]
        z = derived["latents"][keep].float().to(device)
        scheduler = load_ddim_scheduler("stabilityai/stable-diffusion-xl-base-1.0")
        scheduler.set_timesteps(50)
        hard, soft = SentenceScore("noisyclip", device, "hard"), None
        saturated, cosines = [], []
        for step in range(16):
            gen = torch.Generator().manual_seed(7000 + step)
            x = scheduler.add_noise(z, torch.randn(z.shape, generator=gen).to(device),
                                    scheduler.timesteps[step].repeat(len(z)).to(device))
            rgb = latent_to_rgb(x)
            saturated.append(float(((rgb < 0) | (rgb > 255)).float().mean()))
            a = image_features(hard.model, rgb_to_clip(rgb.clamp(0, 255)))
            b = image_features(hard.model, rgb_to_clip(soft_clamp(rgb)))
            cosines.append(float((a * b).sum(-1).mean()))
        share, cosine = sum(saturated) / 16, sum(cosines) / 16
        use_soft = share > SATURATION_BAR and cosine >= SOFT_COSINE_BAR
        out = {"clamp": "soft" if use_soft else "hard", "saturated_share": share, "soft_hard_cosine": cosine,
               "saturated_by_step": saturated, "cosine_by_step": cosines, "bars": {"saturation": SATURATION_BAR, "cosine": SOFT_COSINE_BAR}}
    (args.out / "clamp.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, list)}), flush=True)
    return 0


def tuning(splits: Path, pairs_limit=None, seeds_limit=None):
    data = json.loads(splits.read_text())
    pairs, seeds = data["tuning_pairs"], data["tuning_seeds"]
    return pairs[:pairs_limit] if pairs_limit else pairs, seeds[:seeds_limit] if seeds_limit else seeds


def setup(args, first_prompt):
    from co3_common import load, load_parser
    from co3_push import attach

    parser = load_parser()
    model, config = load(first_prompt, parser, args.out / "_config")
    model.unet.requires_grad_(False)
    wrapper = attach(model)
    choice = json.loads((args.out / "choice.json").read_text())["chosen"]
    if choice is None:
        raise SystemExit("no score was chosen, so W3 renders nothing")
    clamp = json.loads((args.out / "clamp.json").read_text())["clamp"]
    score = SentenceScore(choice["model"], model.unet.device, "soft" if clamp == "soft" else "hard")
    return parser, model, config, wrapper, choice, score


def final_latent(model, start, denoise_seed):
    from composers.utils_custom import seed_everything

    seed_everything(denoise_seed)
    x = start.clone()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for index, timestep in enumerate(model.scheduler.timesteps):
            x = model.denoise_step(x, timestep, index)
    return x


def stage_check(args) -> int:
    from co3_common import set_prompt, start_noise

    pairs, seeds = tuning(args.splits)
    prompts = {p: joint_prompt(p) for p in pairs}
    parser, model, config, wrapper, choice, score = setup(args, prompts[pairs[0]])
    for p in pairs:
        sentences(p, choice["property"])
    concepts = {p: set_prompt(model, config, q, parser) for p, q in prompts.items()}
    pair, seed = pairs[0], seeds[0]
    set_prompt(model, config, prompts[pair], parser)
    score.set_sentences(*sentences(pair, choice["property"]))
    start = start_noise(model, seed)

    def run(pusher):
        wrapper.pusher = pusher
        latent = final_latent(model, start, seed)
        wrapper.pusher = None
        return latent

    plain, repeat = run(None), run(None)
    zero = run(SentencePusher(score, config.guidance_scale, choice["push_steps"], "zero"))
    real = SentencePusher(score, config.guidance_scale, choice["push_steps"], "push")
    pushed = run(real)
    result = {"pair": pair, "seed": seed, "device": torch.cuda.get_device_name(0), "concepts": concepts,
              "pairs_without_two_concepts": [p for p, n in concepts.items() if n != 2],
              "plain_repeat_max_abs": float((repeat - plain).abs().max()), "zero_push_max_abs": float((zero - plain).abs().max()),
              "zero_bar": ZERO_PUSH_MAX_ABS, "real_push_max_abs": float((pushed - plain).abs().max()),
              "pushed_steps": sum(1 for n in real.applied if n > 0), "steps_asked": len(choice["push_steps"]), "dead_gradient_steps": real.dead}
    result["zero_passes"] = result["zero_push_max_abs"] < ZERO_PUSH_MAX_ABS
    (args.out / "check.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["zero_passes"] else 1


def stage_render(args) -> int:
    from co3_common import set_prompt, start_noise
    from composers.utils_custom import seed_everything

    pairs, seeds = tuning(args.splits, args.pairs_limit, args.seeds_limit)
    order = json.loads(args.splits.read_text())["tuning_pairs"]
    parser, model, config, wrapper, choice, score = setup(args, joint_prompt(pairs[0]))
    done = 0
    for pair in pairs:
        prompt = joint_prompt(pair)
        guard(prompt)
        concepts = set_prompt(model, config, prompt, parser)
        if concepts != 2:
            print(f"skipping {pair}: {concepts} concepts", flush=True)
            continue
        score.set_sentences(*sentences(pair, choice["property"]))
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
                    pusher = SentencePusher(score, config.guidance_scale, choice["push_steps"], "push")
                else:
                    if norms is None:
                        raise SystemExit(f"no push lengths for {pair} seed {seed}")
                    generator = torch.Generator().manual_seed(190_000 + 1_000 * order.index(pair) + seed)
                    pusher = SentencePusher(score, config.guidance_scale, choice["push_steps"], "random", norms, generator)
                begin = time.time()
                seed_everything(seed)
                wrapper.pusher = pusher
                image = model.sample_loop(start_noise(model, seed))[0]
                wrapper.pusher = None
                folder.mkdir(parents=True, exist_ok=True)
                image.save(image_path)
                applied = pusher.applied if pusher else []
                meta_path.write_text(json.dumps({"row": row, "pair": pair, "prompt": prompt, "seed": seed, "seconds": round(time.time() - begin, 2),
                                                 "concepts": concepts, "applied_norms": applied, "model": choice["model"],
                                                 "property": choice["property"], "push_steps": choice["push_steps"], "length": PUSH_LENGTH,
                                                 "dead_gradient_steps": pusher.dead if pusher else 0}) + "\n")
                if row == "push":
                    norms = applied
                done += 1
                print(f"{done} {row} {pair} seed={seed} {time.time() - begin:.1f}s", flush=True)
    print(f"finished rendered={done}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["choose", "clamp", "check", "render"])
    ap.add_argument("--w2", type=Path)
    ap.add_argument("--w2-prep", type=Path)
    ap.add_argument("--splits", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs-limit", type=int)
    ap.add_argument("--seeds-limit", type=int)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage != "choose" and not torch.cuda.is_available():
        raise SystemExit("torch sees no CUDA card")
    return {"choose": stage_choose, "clamp": stage_clamp, "check": stage_check, "render": stage_render}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
