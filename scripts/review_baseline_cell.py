"""Render CO3 or SuperDiff on SDXL for the review response, one tile per pair and seed, with protocol rule 9's columns.

CO3 runs from the vendored code under composition/debottam_co3, unedited. Its own entry point draws its starting
noise inside Co3.run_sampling, so this script calls the same steps sample_co3.py calls and then hands
Co3.sample_loop the cached x_T of the seed (scripts/review_seed_latents.py), which keeps CO3's column on the same
noise as the joint prompt, PoE and the adapter (protocol rule 2). Its later resampling noise still comes from the
seed number through seed_everything, as in its own code.

SuperDiff runs from poe_repair/composers/superdiff.py, its own noise from the seed number and its own stochastic
sampler, so its column is mixed under protocol rule 7 whatever x_T it starts from.

Every U-Net forward call is counted (protocol rule 6): `unet_calls` is the number of calls and `nfe` the number of
latent rows evaluated, which is the calls times the batch.

    python scripts/review_baseline_cell.py --method co3 --setting default --pairs "a lion|a meerkat" --seeds 9
    python scripts/review_baseline_cell.py --method superdiff --setting default --pairs "a lion|a meerkat" --seeds 9

Writes <OUT>/<method>_<setting>/<pair slug>/seedNN/<method>.png and .json. A tile that exists is skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("POE_REPO", Path(__file__).resolve().parents[1]))
CO3_DIR = REPO / "composition" / "debottam_co3"
OUT = Path(os.environ.get("REVIEW_BASELINE_OUT", "/datasets/mmolefe/poe_repair_min/outputs/paper_review"))
LATENTS = OUT / "latents"

# CO3's published default: the settings of its own run_sample_co3.sh, which its README says to run.
CO3_DEFAULT = dict(guidance_scale=0.8, num_ts_to_correct=6, num_latent_corrector_steps=5, num_resampling_steps=3,
                   corrector_algo="co3-hybrid", modulate_comp_weights=True, beta=0.9, lmda=0.8)
# SuperDiff's published default: the official SDXL port's 200 steps and no clamp on kappa (the clamp is this
# repo's addition for 50-step runs); guidance 7.5 as everywhere in rule 2.
SUPERDIFF_DEFAULT = dict(steps=200, guidance_scale=7.5, kappa_clamp=False)


def slug(a: str, b: str) -> str:
    s = lambda t: re.sub(r"[^a-z0-9]+", "_", t.strip().lower()).strip("_")
    return f"{s(a)}__x__{s(b)}"


class UNetCounter:
    """Counts forward calls on one U-Net instance without editing the class it belongs to."""

    def __init__(self, unet) -> None:
        self.calls = 0
        self.rows = 0
        inner = unet.forward

        def counted(sample, *args, **kwargs):
            self.calls += 1
            self.rows += int(sample.shape[0])
            return inner(sample, *args, **kwargs)

        unet.forward = counted

    def reset(self) -> None:
        self.calls = self.rows = 0


def row(method: str, sampler: str, steps: int, guidance: float, a: str, b: str, seed: int, setting: str,
        counter: UNetCounter, noise: str, knobs: dict, seconds: float, group: str) -> dict:
    return {"model": "SDXL", "method": method, "adapter": "none", "sampler": sampler, "steps": steps,
            "guidance": guidance, "resolution": 1024, "pair": slug(a, b), "seed": seed, "group": group,
            "setting": setting, "nfe": counter.rows, "unet_calls": counter.calls, "prompt_a": a, "prompt_b": b,
            "noise": noise, "knobs": knobs, "seconds": round(seconds, 1)}


def run_co3(pairs, seeds, setting, knobs, group):
    import torch

    sys.path.insert(0, str(CO3_DIR))
    from composers.Co3 import Co3
    from composers.config import Co3Config
    from composers.utils_custom import seed_everything

    out_root = OUT / f"co3_{setting}"
    a0, b0 = pairs[0]
    cfg = Co3Config(prompt=f"{a0}+{b0}+{a0} and {b0}", prompt_orig=f"{a0} and {b0}", n_timesteps=50,
                    sd_version="xl", resolution_h=1024, resolution_w=1024, negative_prompt="",
                    output_path=str(out_root / "_co3_scratch"), output_path_all=str(out_root / "_co3_scratch"),
                    **knobs)
    co3 = Co3(cfg)
    counter = UNetCounter(co3.unet)
    for a, b in pairs:
        for seed in seeds:
            png = out_root / slug(a, b) / f"seed{seed:02d}" / "co3.png"
            if png.exists():
                print(f"{slug(a, b)} seed {seed}: already rendered", flush=True)
                continue
            png.parent.mkdir(parents=True, exist_ok=True)
            cfg.prompt, cfg.prompt_orig, cfg.seed = f"{a}+{b}+{a} and {b}", f"{a} and {b}", seed
            seed_everything(seed)
            co3.config = cfg
            co3.prepare_prompts(cfg)
            co3.prepare_embeds()
            co3.init_sampling()
            x_t = torch.load(LATENTS / f"seed{seed:02d}.pt").to(co3.unet.device)
            counter.reset()
            t0 = time.perf_counter()
            img = co3.sample_loop(x_t * co3.scheduler.init_noise_sigma)
            secs = time.perf_counter() - t0
            img[0].save(png)
            rec = row("co3", "DDIM with CO3 corrector, CFG++", 50, knobs["guidance_scale"], a, b, seed, setting,
                      counter, f"cached x_T {LATENTS / f'seed{seed:02d}.pt'}", knobs, secs, group)
            png.with_suffix(".json").write_text(json.dumps(rec, indent=1))
            print(f"{slug(a, b)} seed {seed}: {secs:.1f} s, nfe {counter.rows} ({counter.calls} calls) -> {png}",
                  flush=True)


def run_superdiff(pairs, seeds, setting, knobs, group):
    from poe_repair.composers import superdiff
    from poe_repair.config import RunConfig
    from poe_repair.experiments._eval_common import cell_for
    from poe_repair.run import MethodCtx
    from poe_repair.runtime import infer_device, infer_dtype

    sys.path.insert(0, str(REPO))
    cfg = RunConfig()
    device = infer_device(None)
    out_root = OUT / f"superdiff_{setting}"
    ctx = MethodCtx(models={}, scheduler=None, output_root=out_root / "_superdiff_raw", device=device,
                    dtype=infer_dtype(cfg.dtype, device), guidance_scale=knobs["guidance_scale"],
                    num_inference_steps=knobs["steps"], joint_template=cfg.joint_template)
    models = superdiff._load_superdiff_models(ctx.device, ctx.dtype)
    counter = UNetCounter(models["unet"])
    for a, b in pairs:
        for seed in seeds:
            png = out_root / slug(a, b) / f"seed{seed:02d}" / "superdiff.png"
            if png.exists():
                print(f"{slug(a, b)} seed {seed}: already rendered", flush=True)
                continue
            png.parent.mkdir(parents=True, exist_ok=True)
            counter.reset()
            t0 = time.perf_counter()
            raw = superdiff.run(cell_for(a, b, seed), ctx, exp_name="review", overwrite=True,
                                 kappa_clamp=knobs["kappa_clamp"])
            secs = time.perf_counter() - t0
            png.write_bytes(Path(raw).read_bytes())
            rec = row("superdiff", "SuperDiff AND, Euler-Maruyama (stochastic)", knobs["steps"],
                      knobs["guidance_scale"], a, b, seed, setting, counter, "own, torch.cuda.manual_seed(seed)",
                      knobs, secs, group)
            png.with_suffix(".json").write_text(json.dumps(rec, indent=1))
            print(f"{slug(a, b)} seed {seed}: {secs:.1f} s, nfe {counter.rows} ({counter.calls} calls) -> {png}",
                  flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["co3", "superdiff"])
    p.add_argument("--setting", required=True, help="'default' or a grid setting name; names the output folder")
    p.add_argument("--knobs", default="{}", help="JSON overrides of the method's default knobs")
    p.add_argument("--pairs", nargs="+", required=True, help='"a lion|a meerkat" ...')
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--group", default="tuning", help="tuning, R1 or R3")
    args = p.parse_args()
    pairs = [tuple(x.strip() for x in s.split("|")) for s in args.pairs]
    sys.path.insert(0, str(REPO))
    if args.method == "co3":
        run_co3(pairs, args.seeds, args.setting, {**CO3_DEFAULT, **json.loads(args.knobs)}, args.group)
    else:
        run_superdiff(pairs, args.seeds, args.setting, {**SUPERDIFF_DEFAULT, **json.loads(args.knobs)}, args.group)


if __name__ == "__main__":
    main()
