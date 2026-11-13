#!/usr/bin/env python
"""The ceiling: is the third expert's target reachable by ANY embedding, or by none?

Every x so far was constructed. This one is searched for. At each sampling step, with the sampler
standing on the latent it has actually reached, optimise an embedding x directly so that the UNet's
prediction under x matches the one term that would make the three-expert product equal the guided
joint:

    target(x_t, t) = eps_J + 2 eps_empty - eps_A - eps_B

Nothing in the text encoders trains; they are never called in the loop. The free parameters are the
77 x 2048 sequence and the 1280 pooled vector themselves.

Why this is the decisive run. If a searched-for x still cannot hit the target, no formula and no
learned predictor will either, and text-space x is finished. If it can, the target is reachable and
the remaining job is predicting it, which is a different and much easier question.

Two modes, and the pair of them is the measurement:

    --mode per-step     a fresh x at every step. The ceiling: the most any embedding can do.
    --mode constant     one x shared across all 50 steps, optimised against every step's target
                        at once. The ceiling for a FIXED x, which is what every built variant is.

The gap between the two is how much of the failure is state dependence rather than expressiveness.

    python scripts/and_vector_per_step_x.py --mode per-step --seeds 2 6 --iters 50

Writes <out>/per-step-x-<mode>-seedNN.json (the relative error at every step, before and after
optimisation, beside the same number for the best existing third expert) and the rendered tile.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("POE_REPO", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from concept_pair_sheet import NOISE_PAIR, concept_slug  # noqa: E402

# The verdict bars, written here before the first run so they cannot move after the answer is seen.
# Read on the median over steps of ||eps(x*) - target|| / ||target||.
REACHABLE_BELOW = 0.5      # the conditioning channel can express the target
NOT_EXPRESSIBLE_ABOVE = 0.9  # it cannot, and a fixed or learned x is finished


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="dog")
    ap.add_argument("--b", default="cat")
    ap.add_argument("--joint", default="dog and cat")
    ap.add_argument("--seeds", nargs="+", type=int, default=[2])
    ap.add_argument("--mode", choices=("per-step", "constant"), default="per-step")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=REPO / "artifacts/results/the-positional-and-vector")
    args = ap.parse_args()

    import torch

    from poe_repair.composers._helpers import init_latents_for_cell
    from poe_repair.experiments.interaction_term.cell import cell_from_slug
    from poe_repair.methods._sampling import add_time_ids, write_decoded_image
    from poe_repair.run import make_ctx
    from poe_repair.runtime import (ddim_prev_from_x0_eps, decode_latents, encode_prompt_sdxl,
                                    tweedie_mean)

    ctx = make_ctx()
    assert torch.cuda.is_available(), "no CUDA device; this run needs a GPU"
    unet, sched, w = ctx.models["unet"], ctx.scheduler, ctx.guidance_scale
    unet.requires_grad_(False)
    enc = lambda p: tuple(t.float() for t in encode_prompt_sdxl(p, models=ctx.models,
                                                                device=ctx.device, dtype=ctx.dtype))
    e = {k: enc(p) for k, p in (("a", args.a), ("b", args.b), ("joint", args.joint), ("empty", ""))}
    slug = f"{concept_slug(args.a)}__x__{concept_slug(args.b)}"
    args.out.mkdir(parents=True, exist_ok=True)

    def predict(lat, t, seq, pool, cell):
        cond = {"text_embeds": pool.to(ctx.dtype),
                "time_ids": add_time_ids(height=cell.height, width=cell.width, batch_size=seq.shape[0],
                                         device=ctx.device, dtype=ctx.dtype)}
        return unet(sched.scale_model_input(lat.repeat(seq.shape[0], 1, 1, 1), t), t,
                    encoder_hidden_states=seq.to(ctx.dtype), added_cond_kwargs=cond,
                    timestep_cond=None).sample

    for seed in args.seeds:
        cell = cell_from_slug(NOISE_PAIR, seed)
        init, sigma = init_latents_for_cell(cell, ctx)
        sched.set_timesteps(ctx.num_inference_steps)
        lat = (init / sigma).to(device=ctx.device, dtype=ctx.dtype)

        # The shared starting point and, in constant mode, the one x being optimised: the joint
        # prompt's own embedding, which is the best third expert anyone has rendered.
        x_seq = e["joint"][0].clone().requires_grad_(True)
        x_pool = e["joint"][1].clone().requires_grad_(True)
        shared_opt = torch.optim.Adam([x_seq, x_pool], lr=args.lr) if args.mode == "constant" else None

        steps = []
        for i, t in enumerate(sched.timesteps):
            with torch.no_grad():
                four = torch.cat([e[k][0] for k in ("a", "b", "joint", "empty")], dim=0)
                fourp = torch.cat([e[k][1] for k in ("a", "b", "joint", "empty")], dim=0)
                p = predict(lat, t, four, fourp, cell).float()
                eps_a, eps_b, eps_j, eps_0 = p[0:1], p[1:2], p[2:3], p[3:4]
                target = eps_j + 2 * eps_0 - eps_a - eps_b
                tnorm = target.norm()

            if args.mode == "per-step":
                x_seq = e["joint"][0].clone().requires_grad_(True)
                x_pool = e["joint"][1].clone().requires_grad_(True)
                opt = torch.optim.Adam([x_seq, x_pool], lr=args.lr)
            else:
                opt = shared_opt

            with torch.no_grad():
                start_err = float((predict(lat, t, e["joint"][0], e["joint"][1], cell).float()
                                   - target).norm() / tnorm)

            for _ in range(args.iters):
                opt.zero_grad(set_to_none=True)
                err = (predict(lat, t, x_seq, x_pool, cell).float() - target).norm() / tnorm
                err.backward()
                opt.step()

            with torch.no_grad():
                eps_x = predict(lat, t, x_seq.detach(), x_pool.detach(), cell).float()
                end_err = float((eps_x - target).norm() / tnorm)
                eg = (eps_0 + w * ((eps_a - eps_0) + (eps_b - eps_0) + (eps_x - eps_0))).to(ctx.dtype)
                abar = sched.alphas_cumprod[int(t.item())].to(device=ctx.device, dtype=ctx.dtype)
                lat = ddim_prev_from_x0_eps(scheduler=sched, timestep=t, step_index=i,
                                            x0=tweedie_mean(lat, abar, eg), eps=eg)

            steps.append({"step": i, "timestep": int(t.item()),
                          "relative_error_joint_prompt_as_x": start_err,
                          "relative_error_optimised_x": end_err})
            print(f"seed {seed} step {i:>2}: joint-as-x {start_err:.3f} -> optimised {end_err:.3f}",
                  flush=True)

        with torch.no_grad():
            img = decode_latents(ctx.models, lat).cpu()
        png = args.out / f"per-step-x-{args.mode}-seed{seed:02d}.png"
        write_decoded_image(img, png)

        import statistics
        med = statistics.median(s["relative_error_optimised_x"] for s in steps)
        base = statistics.median(s["relative_error_joint_prompt_as_x"] for s in steps)
        verdict = ("reachable" if med < REACHABLE_BELOW else
                   "not expressible" if med > NOT_EXPRESSIBLE_ABOVE else "inconclusive")
        rec = {"a": args.a, "b": args.b, "joint": args.joint, "seed": seed, "mode": args.mode,
               "iters": args.iters, "lr": args.lr, "guidance": w,
               "steps": ctx.num_inference_steps, "noise_from": NOISE_PAIR, "slug": slug,
               "bars": {"reachable_below": REACHABLE_BELOW,
                        "not_expressible_above": NOT_EXPRESSIBLE_ABOVE},
               "median_relative_error_optimised": med,
               "median_relative_error_joint_prompt_as_x": base,
               "verdict": verdict, "per_step": steps}
        (args.out / f"per-step-x-{args.mode}-seed{seed:02d}.json").write_text(json.dumps(rec, indent=1))
        print(f"\nseed {seed} [{args.mode}]: median relative error {med:.3f} "
              f"(joint prompt as x: {base:.3f}) -> {verdict}\nwrote {png}\n", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
