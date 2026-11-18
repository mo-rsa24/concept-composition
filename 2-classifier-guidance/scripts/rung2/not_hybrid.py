#!/usr/bin/env python
"""Plan 27: cat AND dog AND NOT hybrid, with SDXL giving the direction and the checker giving the weight.

Logical guidance (Alesiani et al., arXiv 2602.05549, Table 1) writes the score of NOT psi as
-(pi / (1 - pi)) s(psi), where s(psi) is the conditional-minus-unconditional score of psi and pi the posterior
probability of psi at the current step. Its gradient-free guidance takes the direction from the diffusion
model's own prompts and only the weight from a classifier, so no classifier gradient is used.

Each sampler step here: SDXL's noise predictions for "a cat", "a dog", the empty prompt and HYBRID_PROMPT;
the product as in plain PoE (each prompt with classifier-free guidance 7.5, eps_cat + eps_dog - eps_null); the
readout checker of iteration 4 (readout-catdog-more3-long, one output, 0.755) gives pi = 1 - sigmoid(clean-pair
logit), the probability the state is not heading for a clean cat and dog, read with no gradient; and

    eps = eps_PoE - lam * k * 7.5 * (eps_hybrid - eps_null),   k = pi / (1 - pi), pi clamped at 0.95.

The term is limited to the norm of eps_PoE (a guard against a runaway step; how often it fires is recorded).
Columns per held-out seed (9 to 16), one trajectory each from the seed's own noise:
    plain        plain PoE (run_cfg_poe)
    lam0         this loop at lam 0: must match plain to within fp16 noise, the check that the loop is PoE
    neg-k1       an ordinary negative prompt: k = 1, lam 1.0, no checker
    neg-k19      the hybrid prompt at the checker run's size (k fixed at 19, lam 0.03), no checker
    w0.01, w0.03, w0.1   the checker-weighted term at lam 0.01, 0.03 and 0.1

Writes, under MOUNT_ROOT/outputs/classifier_guidance_ladder/rung2/not-hybrid/:
    tiles/seedNN__<column>.png, record.json, sheet.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from poe_repair import paths  # noqa: E402
from poe_repair.composers._helpers import init_latents_for_cell  # noqa: E402
from poe_repair.methods._sampling import add_time_ids, run_cfg_poe, write_decoded_image  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import (ddim_prev_from_x0_eps, decode_latents, encode_prompt_sdxl, guided_eps,  # noqa: E402
                                poe_eps, tweedie_mean)
import classifier_guidance_from_the_paper as single  # noqa: E402
from train_readout_pair import PairHead, ReadoutPairClassifier  # noqa: E402

RUNG2 = paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2"
OUT = RUNG2 / "not-hybrid"
HEAD = RUNG2 / "readout_pair/readout-catdog-more3-long/head_step_002000.pt"
HYBRID_PROMPT = "a hybrid animal, half cat and half dog"
SEEDS = list(range(9, 17))
# (column, lam, weight): "checker" uses k = pi / (1 - pi); a number fixes k. On PoE's own trajectory the checker reads
# "not heading for a clean pair" with pi at the 0.95 clamp on most steps, so k sits near 19: "neg-k19" is the same
# size with no checker, isolating what the per-step weight adds; "neg-k1" is an ordinary negative prompt.
COLUMNS = [("lam0", 0.0, "checker"), ("neg-k1", 1.0, 1.0), ("neg-k19", 0.03, 19.0),
           ("w0.01", 0.01, "checker"), ("w0.03", 0.03, "checker"), ("w0.1", 0.1, "checker")]
PI_MAX, GUARD = 0.95, 1.0


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    (OUT / "tiles").mkdir(parents=True, exist_ok=True)
    ctx = make_ctx(output_root=OUT / "raw")
    enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    (sa, pa), (sb, pb), (se, pe), (sh, ph) = enc("a cat"), enc("a dog"), enc(""), enc(HYBRID_PROMPT)
    unet, g = ctx.models["unet"], ctx.guidance_scale
    # PoE's three prompts as one batch of 3, exactly as run_cfg_poe and the push loop call them, so lam 0 reproduces
    # plain PoE; the hybrid prompt as its own call (a batch of 4 changes the fp16 arithmetic and the picture)
    seq, pool = torch.cat([sa, sb, se]), torch.cat([pa, pb, pe])
    cond = {"text_embeds": pool, "time_ids": add_time_ids(height=1024, width=1024, batch_size=3,
                                                          device=ctx.device, dtype=ctx.dtype)}
    cond_h = {"text_embeds": ph, "time_ids": add_time_ids(height=1024, width=1024, batch_size=1,
                                                          device=ctx.device, dtype=ctx.dtype)}
    state = torch.load(HEAD, map_location="cpu")
    seq_r, pool_r = enc("a cat and a dog") if state.get("pair_prompt") else (se, pe)
    head = PairHead(n_out=state.get("heads", 1))
    head.load_state_dict(state["head"])
    checker = ReadoutPairClassifier(unet, seq_r, pool_r, head.float().eval()).to(ctx.device)
    common = dict(models=ctx.models, scheduler=ctx.scheduler, seq_a=sa, pool_a=pa, seq_b=sb, pool_b=pb, seq_e=se,
                  pool_e=pe, guidance_scale=g, num_inference_steps=ctx.num_inference_steps, height=1024, width=1024,
                  device=ctx.device, dtype=ctx.dtype)

    @torch.no_grad()
    def sample(init, sigma, lam, weight):
        sched = ctx.scheduler
        sched.set_timesteps(ctx.num_inference_steps)
        latents = (init / sigma).to(device=ctx.device, dtype=ctx.dtype)
        ks, guard_fired = [], 0
        for i, t in enumerate(sched.timesteps):
            x_in = sched.scale_model_input(latents.repeat(3, 1, 1, 1), t)
            e = unet(x_in, t, encoder_hidden_states=seq, added_cond_kwargs=cond, timestep_cond=None).sample
            ea, eb, eu = e.chunk(3)
            eps = poe_eps(guided_eps(ea, eu, g), guided_eps(eb, eu, g), eu)
            if lam > 0:
                eh = unet(sched.scale_model_input(latents, t), t, encoder_hidden_states=sh,
                          added_cond_kwargs=cond_h, timestep_cond=None).sample
                if weight == "checker":
                    logit = float(checker(latents.float(), t.reshape(1).to(ctx.device)).reshape(-1)[0])
                    pi = min(1.0 - 1.0 / (1.0 + np.exp(-logit)), PI_MAX)
                    k = pi / (1.0 - pi)
                else:
                    k = float(weight)
                term = lam * k * g * (eh.float() - eu.float())
                limit = GUARD * float(eps.float().norm())
                if float(term.norm()) > limit:
                    term, guard_fired = term * (limit / float(term.norm())), guard_fired + 1
                eps = (eps.float() - term).to(ctx.dtype)
                ks.append(k)
            ab = sched.alphas_cumprod[int(t.item())].to(device=ctx.device, dtype=ctx.dtype)
            x0 = tweedie_mean(latents, ab, eps)
            latents = ddim_prev_from_x0_eps(scheduler=sched, timestep=t, step_index=i, x0=x0, eps=eps)
        return decode_latents(ctx.models, latents).cpu(), {"k_per_step": ks, "guard_fired_steps": guard_fired}

    record = {"head": str(HEAD), "hybrid_prompt": HYBRID_PROMPT, "pi_max": PI_MAX, "guard": GUARD, "seeds": {}}
    t0 = time.time()
    for seed in SEEDS:
        rec = {}
        init, sigma = init_latents_for_cell(single.cell_for(seed), ctx)
        o = run_cfg_poe(init_latents=init, euler_init_noise_sigma=sigma, **common)
        write_decoded_image(o.image, OUT / "tiles" / f"seed{seed:02d}__plain.png")
        for name, lam, weight in COLUMNS:
            init, sigma = init_latents_for_cell(single.cell_for(seed), ctx)
            img, info = sample(init, sigma, lam, weight)
            write_decoded_image(img, OUT / "tiles" / f"seed{seed:02d}__{name}.png")
            rec[name] = {"lam": lam, "weight": weight, "guard_fired_steps": info["guard_fired_steps"],
                         "k_steps_0_to_9": info["k_per_step"][:10], "k_mean": float(np.mean(info["k_per_step"])) if info["k_per_step"] else 0.0}
        a = np.asarray(Image.open(OUT / "tiles" / f"seed{seed:02d}__plain.png").convert("RGB"), dtype=np.int16)
        b = np.asarray(Image.open(OUT / "tiles" / f"seed{seed:02d}__lam0.png").convert("RGB"), dtype=np.int16)
        rec["lam0_vs_plain_max_pixel_diff"] = int(np.abs(a - b).max())
        rec["lam0_vs_plain_mean_pixel_diff"] = float(np.abs(a - b).mean())
        record["seeds"][seed] = rec
        (OUT / "record.json").write_text(json.dumps(record, indent=1))
        print(f"seed {seed}: lam 0 vs plain mean pixel difference {rec['lam0_vs_plain_mean_pixel_diff']:.2f} "
              f"(max {rec['lam0_vs_plain_max_pixel_diff']}); checker k mean {rec['w0.03']['k_mean']:.2f}, steps 0 to 9 "
              f"{[round(k, 1) for k in rec['w0.03']['k_steps_0_to_9']]}; guard fired on {rec['w0.1']['guard_fired_steps']} "
              f"steps at lam 0.1  ({time.time() - t0:.0f} s)", flush=True)

    T, font = 256, ImageFont.load_default(size=20)
    cols = ["plain"] + [c[0] for c in COLUMNS]
    sheet = Image.new("RGB", (len(cols) * T, 30 + len(SEEDS) * T), "white")
    d = ImageDraw.Draw(sheet)
    for i, c in enumerate(cols):
        d.text((i * T + 4, 6), c, fill="black", font=font)
    for r, seed in enumerate(SEEDS):
        for i, c in enumerate(cols):
            sheet.paste(Image.open(OUT / "tiles" / f"seed{seed:02d}__{c}.png").convert("RGB").resize((T, T)), (i * T, 30 + r * T))
    sheet.save(OUT / "sheet.png")
    print("NOT HYBRID DONE", OUT / "sheet.png", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
