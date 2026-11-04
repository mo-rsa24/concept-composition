#!/usr/bin/env python
"""Does a text-space "and" vector, added as a third expert, make PoE draw both animals? One sheet.

    x_raw = E(joint) - E(a) - E(b)            the whiteboard's arithmetic
    x_fix = E(joint) - E(a) - E(b) + E("")    the same with the start token cancelled
    x_aff = E(joint) - E(a) - E(b) + 2 E("")  coefficients summing to 1: if the UNet were affine in its
                                              condition, its prediction under x_aff would be exactly
                                              eps_J - eps_A - eps_B + 2 eps_empty, the term PoE lacks

Every column is the repo's plain guided sum over N conditions (scripts/concept_multi_poe.py,
eps = eps_empty + w * sum_i (eps_i - eps_empty)); one condition is ordinary classifier-free guidance.
So a, b, the joint prompt, PoE, x alone and the three-expert products all come from one loop, one
sampler and, per seed, the same starting noise (init_latents_for_cell on `a_cat__x__a_dog`). The
arithmetic runs in fp32 and is cast to the model dtype after.

    python scripts/and_vector_sheet.py --render --a dog --b cat --joint "dog and cat" --seeds 1 2 3 4 5 6 7 8
    python scripts/and_vector_sheet.py --sheet <tiles_dir> <out.png> --a dog --b cat --joint "dog and cat" --seeds 1 2 3 4 5 6 7 8

Tiles land as <OUT>/and_vector/<a>__x__<b>/seedNN/<column>.png with a JSON sidecar; an existing tile
is skipped.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("POE_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from concept_pair_sheet import NOISE_PAIR, OUT, concept_slug, write_sidecar  # noqa: E402

# Column key, header lines, and the conditions it sums: "a", "b", "joint" name prompts, "x_raw",
# "x_fix" and "x_aff" name the arithmetic vectors, and any other name is encoded as a prompt itself.
# "name*0.5" scales that condition's guidance term by 0.5.
COLUMNS = [
    ("a", ["{a}", "alone"], ["a"]),
    ("b", ["{b}", "alone"], ["b"]),
    ("joint", ["Joint prompt", "{joint}"], ["joint"]),
    ("poe", ["PoE", "{a}, {b}"], ["a", "b"]),
    ("x_raw", ["x_raw alone", "E(m) - E(a) - E(b)"], ["x_raw"]),
    ("x_fix", ["x_fix alone", 'x_raw + E("")'], ["x_fix"]),
    ("poe_x_raw", ["PoE", "{a}, {b}, x_raw"], ["a", "b", "x_raw"]),
    ("poe_x_fix", ["PoE", "{a}, {b}, x_fix"], ["a", "b", "x_fix"]),
    ("x_aff", ["x_aff alone", 'x_raw + 2E("")'], ["x_aff"]),
    ("poe_x_aff", ["PoE", "{a}, {b}, x_aff"], ["a", "b", "x_aff"]),
    ("poe_single", ["PoE", '{a}, {b}, "single animal"'], ["a", "b", "single animal"]),
    ("poe_two", ["PoE", '{a}, {b}, "two animals"'], ["a", "b", "two animals"]),
]

# The phrase search: real phrases as the third expert, and x_fix turned down, beside the references.
PHRASES = COLUMNS[:4] + [COLUMNS[11]] + [
    ("poe_pair", ["PoE", '{a}, {b}, "a pair of animals"'], ["a", "b", "a pair of animals"]),
    ("poe_side", ["PoE", '{a}, {b}, "side by side"'], ["a", "b", "side by side"]),
    ("poe_together", ["PoE", '{a}, {b}, "together"'], ["a", "b", "together"]),
    ("poe_next_to", ["PoE", '{a}, {b}, "a dog next to a cat"'], ["a", "b", "a dog next to a cat"]),
    ("poe_different", ["PoE", '{a}, {b}, "two different animals"'], ["a", "b", "two different animals"]),
    ("poe_joint", ["PoE", '{a}, {b}, {joint}'], ["a", "b", "joint"]),
    ("poe_x_fix_half", ["PoE", "{a}, {b}, 0.5 x_fix"], ["a", "b", "x_fix*0.5"]),
    ("poe_x_fix_quarter", ["PoE", "{a}, {b}, 0.25 x_fix"], ["a", "b", "x_fix*0.25"]),
]
# One vector at rising weight: PoE of a, b and lam * x_fix, lam from 0.1 to 1 (1 is COLUMNS' poe_x_fix).
SWEEP = COLUMNS[:4] + [
    ("poe_x_fix_tenth", ["lam = 0.1", "PoE {a}, {b}, lam x_fix"], ["a", "b", "x_fix*0.1"]),
    ("poe_x_fix_quarter", ["lam = 0.25", "PoE {a}, {b}, lam x_fix"], ["a", "b", "x_fix*0.25"]),
    ("poe_x_fix_half", ["lam = 0.5", "PoE {a}, {b}, lam x_fix"], ["a", "b", "x_fix*0.5"]),
    ("poe_x_fix_three_quarters", ["lam = 0.75", "PoE {a}, {b}, lam x_fix"], ["a", "b", "x_fix*0.75"]),
    ("poe_x_fix", ["lam = 1", "PoE {a}, {b}, lam x_fix"], ["a", "b", "x_fix"]),
]
SETS = {"vectors": (COLUMNS, 'An "and" vector as a third expert'),
        "phrases": (PHRASES, "A phrase as the third expert"),
        "sweep": (SWEEP, "x_fix as the third expert, at rising weight"),
        "curate": ([COLUMNS[2]], "The joint prompt, one tile per seed, to pass by eye")}


def split(cond: str) -> tuple[str, float]:
    name, _, scale = cond.partition("*")
    return name, float(scale or 1)


def tile(root: Path, slug: str, seed: int, column: str) -> Path:
    return root / "and_vector" / slug / f"seed{seed:02d}" / f"{column}.png"


def render(columns: list, a: str, b: str, joint: str, seeds: list[int]) -> None:
    import torch

    from poe_repair.composers._helpers import init_latents_for_cell
    from poe_repair.experiments.interaction_term.cell import cell_from_slug
    from poe_repair.methods._sampling import add_time_ids, write_decoded_image
    from poe_repair.run import make_ctx
    from poe_repair.runtime import (ddim_prev_from_x0_eps, decode_latents, encode_prompt_sdxl,
                                    tweedie_mean)

    ctx = make_ctx()
    slug = f"{concept_slug(a)}__x__{concept_slug(b)}"
    enc = lambda p: tuple(t.float() for t in encode_prompt_sdxl(p, models=ctx.models, device=ctx.device,
                                                                 dtype=ctx.dtype))
    e = {"a": enc(a), "b": enc(b), "joint": enc(joint)}
    e0 = enc("")
    e["x_raw"] = tuple(m - p - q for m, p, q in zip(e["joint"], e["a"], e["b"]))
    e["x_fix"] = tuple(x + z for x, z in zip(e["x_raw"], e0))
    e["x_aff"] = tuple(x + 2 * z for x, z in zip(e["x_raw"], e0))
    for _, _, conds in columns:
        for c, _ in map(split, conds):
            if c not in e:
                e[c] = enc(c)
    unet, sched, w = ctx.models["unet"], ctx.scheduler, ctx.guidance_scale

    for seed in seeds:
        cell = cell_from_slug(NOISE_PAIR, seed)
        init, sigma = init_latents_for_cell(cell, ctx)
        for key, _, conds in columns:
            out = tile(OUT, slug, seed, key)
            if out.exists():
                print(f"{out}: already rendered", flush=True)
                continue
            names, scales = zip(*map(split, conds))
            embs = [e[c] for c in names] + [e0]
            seq = torch.cat([s for s, _ in embs], dim=0).to(ctx.dtype)
            pool = torch.cat([p for _, p in embs], dim=0).to(ctx.dtype)
            n = len(embs)
            cond = {"text_embeds": pool, "time_ids": add_time_ids(height=cell.height, width=cell.width,
                                                                  batch_size=n, device=ctx.device,
                                                                  dtype=ctx.dtype)}
            sched.set_timesteps(ctx.num_inference_steps)
            lat = (init / sigma).to(device=ctx.device, dtype=ctx.dtype)
            with torch.no_grad():
                for i, t in enumerate(sched.timesteps):
                    eps = unet(sched.scale_model_input(lat.repeat(n, 1, 1, 1), t), t,
                               encoder_hidden_states=seq, added_cond_kwargs=cond, timestep_cond=None).sample
                    e_u = eps[-1:]
                    eg = e_u + w * sum(scales[k] * (eps[k:k + 1] - e_u) for k in range(n - 1))
                    abar = sched.alphas_cumprod[int(t.item())].to(device=ctx.device, dtype=ctx.dtype)
                    x0 = tweedie_mean(lat, abar, eg)
                    lat = ddim_prev_from_x0_eps(scheduler=sched, timestep=t, step_index=i, x0=x0, eps=eg)
                image = decode_latents(ctx.models, lat).cpu()
            out.parent.mkdir(parents=True, exist_ok=True)
            write_decoded_image(image, out)
            write_sidecar(out, {"column": key, "conditions": conds, "a": a, "b": b, "joint": joint,
                                "seed": seed, "noise_from": NOISE_PAIR, "steps": ctx.num_inference_steps,
                                "guidance": w, "size": [cell.width, cell.height]})
            gc.collect()
            torch.cuda.empty_cache()
            print(f"wrote {out}", flush=True)


def sheet(columns: list, title: str, tiles: Path, out: Path, a: str, b: str, joint: str, seeds: list[int]) -> None:
    from main_study_grids import grid

    slug = f"{concept_slug(a)}__x__{concept_slug(b)}"
    fill = lambda s: s.format(a=f'"{a}"', b=f'"{b}"', joint=f'"{joint}"')
    columns = [(key, [fill(h) for h in head]) for key, head, _ in columns]
    rows = [(f"seed {s}", lambda key, s=s: tile(tiles, slug, s, key)) for s in seeds]
    out.parent.mkdir(parents=True, exist_ok=True)
    record = grid(rows, columns, out, title, "", rule_before=4)  # the rule splits what SDXL already does from what x adds
    out.with_suffix(".json").write_text(json.dumps(record, indent=1, default=str))
    print(f"wrote {out}")


def contact(tiles: Path, out: Path, a: str, b: str, seeds: list[int], per_row: int = 8) -> None:
    """The joint prompt's tiles, per_row across, each captioned with its seed, for passing by eye."""
    from PIL import Image, ImageDraw

    slug = f"{concept_slug(a)}__x__{concept_slug(b)}"
    size, pad = 320, 28
    rows = -(-len(seeds) // per_row)
    sheet_img = Image.new("RGB", (per_row * size, rows * (size + pad)), "white")
    draw = ImageDraw.Draw(sheet_img)
    for i, s in enumerate(seeds):
        x, y = (i % per_row) * size, (i // per_row) * (size + pad)
        sheet_img.paste(Image.open(tile(tiles, slug, s, "joint")).convert("RGB").resize((size, size)), (x, y + pad))
        draw.text((x + 6, y + 6), f"seed {s}", fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet_img.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--sheet", nargs=2, type=Path, metavar=("TILES", "OUT_PNG"))
    ap.add_argument("--contact", nargs=2, type=Path, metavar=("TILES", "OUT_PNG"))
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--joint", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--set", choices=sorted(SETS), default="vectors")
    args = ap.parse_args()
    if args.render:
        render(SETS[args.set][0], args.a, args.b, args.joint, args.seeds)
    if args.sheet:
        sheet(*SETS[args.set], *args.sheet, args.a, args.b, args.joint, args.seeds)
    if args.contact:
        contact(*args.contact, args.a, args.b, args.seeds)
