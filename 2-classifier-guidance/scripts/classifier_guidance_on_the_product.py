#!/usr/bin/env python
"""The cat x dog product pushed by a classifier, in the paper's form, against the joint prompt.

Columns: the joint prompt "a cat and a dog" under ordinary guidance (what co-occurrence looks like
when the model is told), the plain product of "a cat" and "a dog", then the product pushed at each
strength. The push is Dhariwal and Nichol's Algorithm 2 with nothing rescaled. Strength 0 must
reproduce the plain product byte for byte, or the run stops before any push is read.

The target is the sum of log p(present) over the classifiers given: the dog classifier alone asks
"is there a dog", and with a cat classifier beside it the sum asks "is there a cat and a dog".

Writes, under MOUNT_ROOT/outputs/classifier_guidance_on_the_product/<name>/:
    tiles/seedNN__<column>.png, sheet.png, push.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from poe_repair import paths  # noqa: E402
from poe_repair.composers._helpers import init_latents_for_cell  # noqa: E402
from poe_repair.methods._sampling import run_cfg_poe, write_decoded_image  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import encode_prompt_sdxl  # noqa: E402
import classifier_guidance_from_the_paper as single  # noqa: E402

SEEDS = single.SEEDS
OUT_ROOT = paths.MOUNT_ROOT / "outputs/classifier_guidance_on_the_product"
CLASSIFIER_ROOTS = {"dog": paths.MOUNT_ROOT / "outputs/dog_classifier",
                    "cat": paths.MOUNT_ROOT / "outputs/cat_classifier"}


class SumOfClassifiers(torch.nn.Module):
    def __init__(self, members):
        super().__init__()
        self.members = torch.nn.ModuleList(members)

    def log_prob_both(self, x_t, timestep):
        return sum(m.log_prob_both(x_t, timestep[:1]) for m in self.members)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="output folder name")
    ap.add_argument("--classifier", action="append", required=True,
                    help="species:run:step, e.g. dog:dog-01:3000; repeat to sum several")
    ap.add_argument("--strengths", type=float, nargs="+", default=[10, 30, 100, 300])
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("guard: no CUDA; on a Blackwell node co3 runs on the CPU in silence")
    out = OUT_ROOT / args.name
    single.disk_guard(out)
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    ctx = make_ctx(output_root=out / "raw")
    enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = enc("")
    seq_a, pool_a = enc("a cat")
    seq_b, pool_b = enc("a dog")
    seq_j, pool_j = enc("a cat and a dog")

    def plain_product(seed, path):
        init, sigma = init_latents_for_cell(single.cell_for(seed), ctx)
        o = run_cfg_poe(init_latents=init, models=ctx.models, scheduler=ctx.scheduler,
                        seq_a=seq_a, pool_a=pool_a, seq_b=seq_b, pool_b=pool_b,
                        seq_e=seq_e, pool_e=pool_e, guidance_scale=ctx.guidance_scale,
                        num_inference_steps=ctx.num_inference_steps, height=1024, width=1024,
                        euler_init_noise_sigma=sigma, device=ctx.device, dtype=ctx.dtype)
        write_decoded_image(o.image, path)

    def pushed(seed, head, strength, path):
        from poe_repair.methods._classifier_guided import run_poe_classifier_guided
        init, sigma = init_latents_for_cell(single.cell_for(seed), ctx)
        o = run_poe_classifier_guided(
            init_latents=init, models=ctx.models, scheduler=ctx.scheduler,
            seq_a=seq_a, pool_a=pool_a, seq_b=seq_b, pool_b=pool_b, seq_e=seq_e, pool_e=pool_e,
            guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
            height=1024, width=1024, euler_init_noise_sigma=sigma, device=ctx.device,
            dtype=ctx.dtype, head=head, strength=strength)
        write_decoded_image(o.image, path)
        return o

    for seed in SEEDS:
        single.render(ctx, seed, seq_j, pool_j, seq_e, pool_e, out / "tiles" / f"seed{seed:02d}__joint.png")
        plain_product(seed, out / "tiles" / f"seed{seed:02d}__product.png")
        print(f"seed {seed}: joint prompt and plain product rendered", flush=True)

    members = []
    for spec in args.classifier:
        species, run, step = spec.split(":")
        single.DOG_CLASSIFIER_ROOT = CLASSIFIER_ROOTS[species]
        m, _ = single.load_dog_classifier(ctx, seq_e, pool_e, run, int(step))
        members.append(m)
    head = SumOfClassifiers(members)

    record = {"classifiers": args.classifier, "canary": {}, "renders": {}}
    for seed in SEEDS:
        pushed(seed, head, 0.0, out / "tiles" / f"seed{seed:02d}__s0.png")
        same = (hashlib.md5((out / "tiles" / f"seed{seed:02d}__s0.png").read_bytes()).hexdigest()
                == hashlib.md5((out / "tiles" / f"seed{seed:02d}__product.png").read_bytes()).hexdigest())
        record["canary"][seed] = same
        print(f"seed {seed}: strength 0 identical to the plain product: {same}", flush=True)
        if not same:
            raise SystemExit("strength 0 moved the product; stopping before any push is read")
    for seed in SEEDS:
        for s_ in args.strengths:
            o = pushed(seed, head, s_, out / "tiles" / f"seed{seed:02d}__s{s_:g}.png")
            rel = [p / e for p, e in zip(o.extras["applied_norm_per_step"], o.extras["eps_norm_per_step"])]
            record["renders"][f"seed {seed}, strength {s_:g}"] = {
                "push_over_eps_median": statistics.median(rel), "push_over_eps_max": max(rel),
                "grad_norm_per_step": o.extras["grad_norm_per_step"]}
            print(f"seed {seed}, strength {s_:g}: push over prediction median "
                  f"{statistics.median(rel):.3f}, max {max(rel):.3f}", flush=True)
    (out / "push.json").write_text(json.dumps(record, indent=2))

    def label(s_):
        med = statistics.median(record["renders"][f"seed {x}, strength {s_:g}"]["push_over_eps_median"]
                                for x in SEEDS)
        return f"product pushed\nstrength {s_:g} (push {100 * med:.1f}%)"
    single.OUT = out
    target = " + ".join(c.split(":")[0] for c in args.classifier)
    single.sheet([("\"a cat and a dog\"\nordinary guidance", "joint"),
                  ("plain product\n\"a cat\" x \"a dog\"", "product")]
                 + [(label(s_), f"s{s_:g}") for s_ in args.strengths],
                 out / "sheet.png", f"The cat x dog product pushed toward {target} present")
    print("wrote", out / "sheet.png", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
