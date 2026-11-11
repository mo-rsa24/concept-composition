"""Render ToMe (Hu et al., NeurIPS 2024) on SDXL for the review response, from its official code, unedited.

Runs in the `tome` environment against the official repository cloned at
/datasets/mmolefe/poe_repair_min/third_party/ToMe (https://github.com/hutaihang/ToMe, commit f256a49a). The
pipeline is called exactly as its run_demo.py calls it, with two differences at the call site only: `latents=`
carries the seed's cached x_T (scripts/review_seed_latents.py, protocol rule 2), which the pipeline scales by its
scheduler's init_noise_sigma itself; and the token groups are built per pair here, because RunConfig1 sets them by
hand for its one demo prompt. A group is the last token of each concept phrase as the noun and the words before it,
less the article, as its attributes ("a snowy mountain" binds "snowy" to "mountain"; "a lion" binds nothing).

The pipeline keeps SDXL's own scheduler, so the sampler differs from rule 2's DDIM and the column is marked mixed.

    python scripts/review_tome_cell.py --setting default --pairs "a lion|a meerkat" --seeds 9

Writes <OUT>/tome_<setting>/<pair slug>/seedNN/tome.png and .json with protocol rule 9's columns.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import time
from pathlib import Path

import torch

TOME_DIR = Path("/datasets/mmolefe/poe_repair_min/third_party/ToMe")
OUT = Path("/datasets/mmolefe/poe_repair_min/outputs/paper_review")
LATENTS = OUT / "latents"
ARTICLES = {"a", "an", "the"}
NEGATIVE = "low res, ugly, blurry, artifact, unreal"  # run_demo.py's own negative prompt


def slug(a: str, b: str) -> str:
    s = lambda t: re.sub(r"[^a-z0-9]+", "_", t.strip().lower()).strip("_")
    return f"{s(a)}__x__{s(b)}"


def token_groups(tokenizer, a: str, b: str, joiner: str) -> tuple[str, list, int]:
    """The joint prompt, ToMe's [[noun], [attributes]] groups over its token positions, and its token count."""
    prompt = f"{a} {joiner} {b}"
    ids = tokenizer(prompt)["input_ids"][1:-1]
    groups, pos = [], 1  # position 0 is the start token
    for i, phrase in enumerate([a, joiner, b]):
        n = len(tokenizer(phrase)["input_ids"]) - 2
        if i != 1:
            words = phrase.split()
            head = words[-1]
            n_head = len(tokenizer(head)["input_ids"]) - 2
            noun = [pos + n - n_head]
            attrs, p = [], pos
            for w in words[:-1]:
                k = len(tokenizer(w)["input_ids"]) - 2
                if w.lower() not in ARTICLES:
                    attrs.extend(range(p, p + k))
                p += k
            groups.append([noun, attrs])
        pos += n
    assert pos - 1 == len(ids), f"token count mismatch for {prompt!r}: {pos - 1} != {len(ids)}"
    return prompt, groups, len(ids)


class UNetCounter:
    def __init__(self, unet) -> None:
        self.calls = self.rows = 0
        inner = unet.forward

        def counted(sample, *args, **kwargs):
            self.calls += 1
            self.rows += int(sample.shape[0])
            return inner(sample, *args, **kwargs)

        unet.forward = counted


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--setting", required=True)
    p.add_argument("--config", default="RunConfig1", choices=["RunConfig1", "RunConfig2"])
    p.add_argument("--knobs", default="{}", help="JSON overrides of the RunConfig fields")
    p.add_argument("--pairs", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--joiner", default="and")
    p.add_argument("--group", default="tuning")
    args = p.parse_args()

    sys.path.insert(0, str(TOME_DIR))
    from configs import demo_config
    from pipe_tome import tomePipeline
    from utils import ptp_utils
    from utils.ptp_utils import AttentionStore

    cfg = dataclasses.replace(getattr(demo_config, args.config)(), **json.loads(args.knobs))
    pipe = tomePipeline.from_pretrained(cfg.model_path, torch_dtype=torch.float16, variant="fp16",
                                        safety_checker=None).to("cuda")
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    counter = UNetCounter(pipe.unet)
    sampler = type(pipe.scheduler).__name__
    knobs = {k: (str(v) if isinstance(v, Path) else v) for k, v in dataclasses.asdict(cfg).items()
             if k not in {"prompt", "token_indices", "prompt_anchor", "prompt_merged", "prompt_length", "seeds",
                          "output_path"}}

    for pair in args.pairs:
        a, b = (x.strip() for x in pair.split("|"))
        prompt, groups, length = token_groups(pipe.tokenizer, a, b, args.joiner)
        for seed in args.seeds:
            png = OUT / f"tome_{args.setting}" / slug(a, b) / f"seed{seed:02d}" / "tome.png"
            if png.exists():
                print(f"{slug(a, b)} seed {seed}: already rendered", flush=True)
                continue
            png.parent.mkdir(parents=True, exist_ok=True)
            x_t = torch.load(LATENTS / f"seed{seed:02d}.pt").to("cuda", torch.float16)
            controller = AttentionStore()
            ptp_utils.register_attention_control(pipe, controller)
            counter.calls = counter.rows = 0
            t0 = time.perf_counter()
            image = pipe(prompt=prompt, guidance_scale=cfg.guidance_scale, latents=x_t,
                         num_inference_steps=cfg.n_inference_steps, attention_store=controller,
                         indices_to_alter=groups, prompt_anchor=[a, b], attention_res=cfg.attention_res,
                         run_standard_sd=cfg.run_standard_sd, thresholds=cfg.thresholds,
                         scale_factor=cfg.scale_factor, scale_range=cfg.scale_range, prompt3=prompt,
                         prompt_length=length, token_refinement_steps=cfg.token_refinement_steps,
                         attention_refinement_steps=cfg.attention_refinement_steps,
                         tome_control_steps=cfg.tome_control_steps, eot_replace_step=cfg.eot_replace_step,
                         use_pose_loss=cfg.use_pose_loss, negative_prompt=NEGATIVE).images[0]
            secs = time.perf_counter() - t0
            image.save(png)
            rec = {"model": "SDXL", "method": "tome", "adapter": "none", "sampler": sampler,
                   "steps": cfg.n_inference_steps, "guidance": cfg.guidance_scale, "resolution": 1024,
                   "pair": slug(a, b), "seed": seed, "group": args.group, "setting": args.setting,
                   "nfe": counter.rows, "unet_calls": counter.calls, "prompt": prompt, "token_groups": groups,
                   "prompt_length": length, "config": args.config, "knobs": knobs,
                   "noise": f"cached x_T {LATENTS / f'seed{seed:02d}.pt'}", "tome_commit": "f256a49a",
                   "seconds": round(secs, 1)}
            png.with_suffix(".json").write_text(json.dumps(rec, indent=1, default=str))
            print(f"{slug(a, b)} seed {seed}: {secs:.1f} s, nfe {counter.rows} ({counter.calls} calls), "
                  f"groups {groups} -> {png}", flush=True)


if __name__ == "__main__":
    main()
