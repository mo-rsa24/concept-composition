"""Three concepts (slot R11): the joint prompt, the plain product and the adapter on four kinds of triple.

Columns are the kind of triple (animal-animal-animal, animal-object-animal, object-object-object,
animal-animal-object), one triple each; rows are the joint prompt, the plain three-concept product
and the product through the adapter of record. Every tile starts from the shared seed-10 latent
(``a_cat__x__a_dog``'s cached latent, protocol rule 2) with 50 DDIM steps at guidance 7.5, and the
adapter is on for all 50 steps (protocol rule 1).

The three-concept product guides each concept against the one null prediction and subtracts the
null twice, the two-concept rule of Appendix B extended to three:
    eps = u + w * (c1 + c2 + c3 - 3u)

The joint and product tiles are rendered before the adapter is attached, because an attached
adapter stays active and would leak into them.

    python 1-lora-correction/scripts/review_three_concepts.py --render
    python 1-lora-correction/scripts/review_three_concepts.py --sheet

Tiles: <OUT>/three_concepts/<kind>/<column>.png with a JSON sidecar; a tile that exists is skipped.
Sheet: <OUT>/three_concepts/three_concepts.png with three_concepts.json beside it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("POE_REPO", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "showcase"))

OUT = Path(os.environ.get("REVIEW_BASELINE_OUT", "/datasets/mmolefe/poe_repair_min/outputs/paper_review"))
R11 = OUT / "three_concepts"
NOISE_PAIR = "a_cat__x__a_dog"
SEED = 10
CHECKPOINT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/cohort3/v58-03-self/checkpoints/lora_step_030000.pt")
TRIPLES = [  # (kind slug, column label, three concept prompts); no concept is in a training pair (pair_pool_v58.yaml)
    ("animal-animal-animal", "animal, animal, animal", ("a cat", "a dog", "a penguin")),
    ("animal-object-animal", "animal, object, animal", ("a dog", "a chair", "a cat")),
    ("object-object-object", "object, object, object", ("a teapot", "a lamp", "a book")),
    ("animal-animal-object", "animal, animal, object", ("a cat", "a dog", "an umbrella")),
]
ROWS = [("joint", "joint prompt"), ("product", "plain product"), ("ours", "ours")]


def joint_text(prompts: tuple[str, str, str]) -> str:
    return f"{prompts[0]}, {prompts[1]} and {prompts[2]}"


def render() -> None:
    import torch

    from poe_repair.composers._helpers import init_latents_for_cell
    from poe_repair.experiments.interaction_term.cell import cell_from_slug
    from poe_repair.methods._sampling import add_time_ids, run_cfg, write_decoded_image
    from poe_repair.run import make_ctx
    from poe_repair.runtime import (ddim_prev_from_x0_eps, decode_latents, encode_prompt_sdxl,
                                    guided_eps, tweedie_mean)
    import lambda_boundary_probe as lbp

    ctx = make_ctx()
    n, w = ctx.num_inference_steps, ctx.guidance_scale
    init, sigma = init_latents_for_cell(cell_from_slug(NOISE_PAIR, SEED), ctx)
    H = W = 1024
    enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = enc("")
    unet = ctx.models["unet"]

    @torch.no_grad()
    def product(prompts) -> torch.Tensor:
        embs = [enc(p) for p in prompts] + [(seq_e, pool_e)]
        sched = ctx.scheduler
        sched.set_timesteps(n)
        latents = (init / sigma).to(device=ctx.device, dtype=ctx.dtype)
        for i, t in enumerate(sched.timesteps):
            outs = []
            for chunk in (embs[:2], embs[2:]):   # two forward passes of two, to stay inside 24 GB
                pe = torch.cat([s for s, _ in chunk]); pool = torch.cat([p for _, p in chunk])
                cond = {"text_embeds": pool, "time_ids": add_time_ids(
                    height=H, width=W, batch_size=len(chunk), device=ctx.device, dtype=ctx.dtype)}
                x = sched.scale_model_input(latents.repeat(len(chunk), 1, 1, 1), t)
                outs.append(unet(x, t, encoder_hidden_states=pe, added_cond_kwargs=cond,
                                 timestep_cond=None).sample)
            c1, c2, c3, u = torch.cat(outs).chunk(4)
            eps = guided_eps(c1, u, w) + guided_eps(c2, u, w) + guided_eps(c3, u, w) - 2 * u
            ab = sched.alphas_cumprod[int(t.item())].to(device=ctx.device, dtype=ctx.dtype)
            x0 = tweedie_mean(latents, ab, eps)
            latents = ddim_prev_from_x0_eps(scheduler=sched, timestep=t, step_index=i, x0=x0, eps=eps)
        return decode_latents(ctx.models, latents).cpu()

    def save(img, kind, col, prompts, extra) -> None:
        out = R11 / kind / f"{col}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        write_decoded_image(img, out)
        out.with_suffix(".json").write_text(json.dumps({
            "kind": kind, "column": col, "prompts": list(prompts), "joint_prompt": joint_text(prompts),
            "seed": SEED, "noise_from": NOISE_PAIR, "steps": n, "guidance": w, **extra}, indent=1))
        print(f"{kind} {col}: {out}", flush=True)

    for kind, _, prompts in TRIPLES:          # adapter not attached yet
        if not (R11 / kind / "joint.png").exists():
            seq_j, pool_j = enc(joint_text(prompts))
            res = run_cfg(init_latents=init, models=ctx.models, scheduler=ctx.scheduler,
                          seq_cond=seq_j, pool_cond=pool_j, seq_e=seq_e, pool_e=pool_e,
                          guidance_scale=w, num_inference_steps=n, height=H, width=W,
                          euler_init_noise_sigma=sigma, device=ctx.device, dtype=ctx.dtype)
            save(res.image, kind, "joint", prompts, {"checkpoint": None})
        if not (R11 / kind / "product.png").exists():
            save(product(prompts), kind, "product", prompts, {"checkpoint": None})

    lbp.LORA_RANK = lbp.LORA_ALPHA = 32
    info = lbp._attach_and_load_lora(unet, CHECKPOINT)
    if int(info["n_matched"]) != 420 or int(info["n_loaded"]) == 0:
        raise SystemExit(f"adapter did not load as trained: matched {info['n_matched']}, loaded {info['n_loaded']}")
    print(f"attached {CHECKPOINT} (step {info['checkpoint_step']}, {info['n_matched']} projections)", flush=True)
    for kind, _, prompts in TRIPLES:
        if not (R11 / kind / "ours.png").exists():
            save(product(prompts), kind, "ours", prompts,
                 {"checkpoint": str(CHECKPOINT), "adapter_steps": n})


def sheet() -> None:
    from PIL import Image, ImageDraw, ImageFont

    tile, pad, left, top = 256, 6, 150, 60
    W = left + len(TRIPLES) * (tile + pad)
    H = top + len(ROWS) * (tile + pad)
    im = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except OSError:
        f = ImageFont.load_default()
    for c, (kind, label, prompts) in enumerate(TRIPLES):
        x = left + c * (tile + pad)
        dr.text((x, 8), label, fill="black", font=f)
        dr.text((x, 30), ", ".join(p.split(" ", 1)[1] for p in prompts), fill="gray", font=f)
        for r, (col, rlabel) in enumerate(ROWS):
            y = top + r * (tile + pad)
            if c == 0:
                dr.text((8, y + tile // 2 - 8), rlabel, fill="black", font=f)
            p = R11 / kind / f"{col}.png"
            if p.exists():
                im.paste(Image.open(p).convert("RGB").resize((tile, tile)), (x, y))
            else:
                dr.rectangle([x, y, x + tile, y + tile], outline="red")
                dr.text((x + 10, y + 10), "missing", fill="red", font=f)
    out = R11 / "three_concepts.png"
    im.save(out)
    out.with_suffix(".json").write_text(json.dumps({
        "seed": SEED, "noise_from": NOISE_PAIR, "checkpoint": str(CHECKPOINT),
        "columns": [{"kind": k, "prompts": list(p)} for k, _, p in TRIPLES],
        "rows": [c for c, _ in ROWS]}, indent=1))
    print(f"sheet: {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--sheet", action="store_true")
    a = ap.parse_args()
    if a.render:
        render()
    if a.sheet:
        sheet()
