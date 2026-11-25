#!/usr/bin/env python3
"""Step M13's plain-PoE half: draw plain PoE exactly as Loop 12's Z8 screening pass does, and keep the state at steps 0, 5, 10 and 15.

Design: artifacts/notes/progress/method-search/09-gradient-direction-check/design.md. Plain PoE is run_cfg_poe, eps_A + eps_B - eps_empty
with each expert at classifier-free guidance 7.5, 50 steps, 1024 px, from the per-seed starting noise every pair shares, as in
2-classifier-guidance/scripts/leakage/z8_screen.py. The sampler is not changed: a thin wrapper around the UNet records the latent it is
given the first time it sees each timestep, which is the state at that step. The output has the layout m5_read_co3.py gradcos reads:
reads.jsonl and latents/<pair>/seed-<seed>.pt holding {step: {"x", "t"}}.

  python m13_poe_states.py --splits <json> --out <dir> [--seeds 6001 6002 6003 6004] [--limit N]
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
sys.path.insert(0, str(HERE.parent / "rung2"))
sys.path.insert(0, str(HERE.parents[2] / "scripts" / "shared"))
from poe_repair.composers._helpers import init_latents_for_cell  # noqa: E402
from poe_repair.methods._sampling import run_cfg_poe  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import encode_prompt_sdxl  # noqa: E402
import classifier_guidance_from_the_paper as single  # noqa: E402
from disallowed_subjects import is_allowed  # noqa: E402
from train_readout_pair import joint_prompt  # noqa: E402

SAVE_STEPS = (0, 5, 10, 15)


class StateRecorder(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.inner, self.seen, self.states = unet, [], {}

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("inner"), name)

    def forward(self, sample, timestep, *a, **k):
        t = int(timestep.reshape(-1)[0]) if torch.is_tensor(timestep) else int(timestep)
        if t not in self.seen:
            self.seen.append(t)
            step = len(self.seen) - 1
            if step in SAVE_STEPS:
                self.states[step] = {"x": sample[:1].detach().to(torch.float16).cpu(), "t": t}
        return self.inner(sample, timestep, *a, **k)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[6001, 6002, 6003, 6004])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    pairs = json.loads(args.splits.read_text())["tuning_pairs"]
    for p in pairs:
        assert is_allowed(joint_prompt(p)), p
    args.out.mkdir(parents=True, exist_ok=True)
    single.disk_guard(args.out)
    ctx = make_ctx(output_root=args.out / "raw")
    enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = enc("")
    recorder = StateRecorder(ctx.models["unet"])
    models = dict(ctx.models, unet=recorder)
    common = dict(models=models, scheduler=ctx.scheduler, seq_e=seq_e, pool_e=pool_e, num_inference_steps=ctx.num_inference_steps,
                  height=1024, width=1024, device=ctx.device, dtype=ctx.dtype)
    done = set()
    if (args.out / "reads.jsonl").exists():
        done = {json.loads(l)["key"] for l in (args.out / "reads.jsonl").read_text().splitlines()}
    with open(args.out / "reads.jsonl", "a") as log:
        for pair in pairs:
            a, b = (w.replace("_", " ") for w in pair.split("__x__"))
            seq_a, pool_a = enc(a)
            seq_b, pool_b = enc(b)
            for seed in args.seeds:
                key = f"{pair}|{seed}"
                if key in done:
                    continue
                if args.limit is not None and len(done) >= args.limit:
                    return 0
                recorder.seen, recorder.states = [], {}
                begin = time.time()
                init, sigma = init_latents_for_cell(single.cell_for(seed), ctx)
                o = run_cfg_poe(init_latents=init, euler_init_noise_sigma=sigma, seq_a=seq_a, pool_a=pool_a, seq_b=seq_b, pool_b=pool_b,
                                guidance_scale=ctx.guidance_scale, **common)
                image = (o.image[0].float().clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
                (args.out / "renders" / pair).mkdir(parents=True, exist_ok=True)
                Image.fromarray(image).save(args.out / "renders" / pair / f"seed-{seed}.png")
                (args.out / "latents" / pair).mkdir(parents=True, exist_ok=True)
                torch.save(recorder.states, args.out / "latents" / pair / f"seed-{seed}.pt")
                row = {"key": key, "pair": pair, "seed": seed, "reproduced": True, "sampler": "plain PoE, run_cfg_poe",
                       "guidance": ctx.guidance_scale, "steps": ctx.num_inference_steps, "saved_steps": sorted(recorder.states),
                       "seconds": round(time.time() - begin, 1)}
                log.write(json.dumps(row) + "\n")
                log.flush()
                done.add(key)
                print(f"{key} saved {sorted(recorder.states)} {row['seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
