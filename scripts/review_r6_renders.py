"""The Langevin corrector on the four Figure 5 pairs at seed 10: tiles for Figure 5 (slot R2) and a contact sheet.

Four columns are rendered here from one starting latent per row, the one protocol rule 2 names
(``load_shared_latents`` for ``a_cat__x__a_dog`` at the seed): the joint prompt, joint + Langevin,
PoE, and PoE + Langevin, the Langevin columns at protocol rule 3's setting, c = 3 and k = 20 steps per
level over all 50 levels. The fifth column, CO3, is rendered by ``scripts/review_baseline_cell.py
--method co3 --group r6``, which hands CO3 the same starting latent. ``--sheet`` then assembles the
five columns into one image and writes the sidecar listing seeds, noise and settings.

    python scripts/review_r6_renders.py --render
    python scripts/review_r6_renders.py --sheet

Tiles land as <OUT>/r6_correctors/<pair>/seed10/<column>.png with a JSON sidecar; a tile that exists
is skipped. The sheet is <OUT>/r6_correctors/five_columns.png with five_columns.json beside it.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("POE_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))

OUT = Path(os.environ.get("REVIEW_BASELINE_OUT", "/datasets/mmolefe/poe_repair_min/outputs/paper_review"))
R6 = OUT / "r6_correctors"
CO3_TILES = OUT / "co3_default" / "r6"
NOISE_PAIR = "a_cat__x__a_dog"
SEED = 10
C, K = 3.0, 20                      # protocol rule 3, no tuning grid
PAIRS = [("a cat", "a dog"), ("an elephant", "a penguin"),
         ("a tent", "a snowy mountain"), ("a lighthouse", "a stormy sea")]
COLUMNS = [("mono", "joint prompt"), ("mono_langevin", "joint + Langevin"), ("poe", "PoE"),
           ("poe_langevin", "PoE + Langevin"), ("co3", "CO3")]


def render() -> None:
    import torch

    from poe_repair.composers._helpers import encode_pair, get_joint_embeds, init_latents_for_cell
    from poe_repair.experiments._eval_common import cell_for
    from poe_repair.experiments.interaction_term.cell import cell_from_slug
    from poe_repair.methods._corrector_drivers import run_joint_langevin
    from poe_repair.methods._poe_langevin import run_poe_langevin
    from poe_repair.methods._sampling import run_cfg, run_cfg_poe, write_decoded_image
    from poe_repair.run import make_ctx

    ctx = make_ctx()
    n = ctx.num_inference_steps
    for a, b in PAIRS:
        cell = cell_for(a, b, SEED)
        init, sigma = init_latents_for_cell(cell_from_slug(NOISE_PAIR, SEED), ctx)
        latent_sha = hashlib.sha256(init.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]
        saved = torch.load(OUT / "latents" / f"seed{SEED:02d}.pt", map_location="cpu")
        saved = saved if torch.is_tensor(saved) else next(v for v in saved.values() if torch.is_tensor(v))
        same = torch.equal(saved.to(init.dtype).reshape(init.shape), init.detach().cpu())
        print(f"{cell.pair_slug}: starting latent {latent_sha}, same as latents/seed{SEED:02d}.pt "
              f"(CO3's): {same}", flush=True)
        emb = encode_pair(cell, ctx)
        seq_j, pool_j = get_joint_embeds(cell, ctx)
        base = dict(init_latents=init, models=ctx.models, scheduler=ctx.scheduler,
                    guidance_scale=ctx.guidance_scale, num_inference_steps=n, height=cell.height,
                    width=cell.width, euler_init_noise_sigma=sigma, device=ctx.device, dtype=ctx.dtype)
        poe = dict(seq_a=emb["seq_a"], pool_a=emb["pool_a"], seq_b=emb["seq_b"], pool_b=emb["pool_b"],
                   seq_e=emb["seq_e"], pool_e=emb["pool_e"])
        jobs = {
            "mono": (lambda: run_cfg(seq_cond=seq_j, pool_cond=pool_j, seq_e=emb["seq_e"],
                                     pool_e=emb["pool_e"], **base), n, 0),
            "mono_langevin": (lambda: run_joint_langevin(seq_j=seq_j, pool_j=pool_j, seq_e=emb["seq_e"],
                                                         pool_e=emb["pool_e"], k=K, c=C,
                                                         noise_seed=SEED, **base), n * (K + 1), K),
            "poe": (lambda: run_cfg_poe(**poe, **base), n, 0),
            "poe_langevin": (lambda: run_poe_langevin(**poe, k=K, c=C, noise_seed=SEED, **base),
                             n * (K + 1), K),
        }
        for column, (fn, unet_calls, k) in jobs.items():
            out = R6 / cell.pair_slug / f"seed{SEED:02d}" / f"{column}.png"
            if out.exists():
                print(f"{out}: already rendered", flush=True)
                continue
            t0 = time.time()
            res = fn()
            out.parent.mkdir(parents=True, exist_ok=True)
            write_decoded_image(res.image, out)
            out.with_suffix(".json").write_text(json.dumps({
                "model": "SDXL", "method": column, "pair": cell.pair_slug, "prompt_a": a, "prompt_b": b,
                "joint_prompt": ctx.joint_template.format(a=a, b=b), "seed": SEED, "group": "r6",
                "sampler": "DDIM", "steps": n, "guidance": ctx.guidance_scale, "resolution": cell.width,
                "noise_from": f"{NOISE_PAIR} seed {SEED} (protocol rule 2)", "latent_sha256_16": latent_sha, "same_as_co3_latent_file": bool(same),
                "setting": f"c={C:g},k={k}" if k else "default", "corrector_noise_seed": SEED if k else None,
                "unet_calls_per_branch_set": unet_calls, "seconds": round(time.time() - t0, 1),
            }, indent=2))
            del res
            gc.collect()
            torch.cuda.empty_cache()
            print(f"wrote {out}  ({time.time() - t0:.0f} s)", flush=True)


def sheet() -> None:
    from PIL import Image, ImageDraw, ImageFont

    from poe_repair.experiments._eval_common import cell_for

    tile, label_w, head_h = 384, 260, 70
    W, H = label_w + tile * len(COLUMNS), head_h + tile * len(PAIRS)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
    record = {"feeds_slot": "R2", "seed": SEED, "setting": {"c": C, "k": K, "levels": "all 50"},
              "noise": f"{NOISE_PAIR} seed {SEED}, protocol rule 2; CO3 starts from the same latent, "
                       "its later resampling noise from the seed number", "rows": [], "columns": [c[1] for c in COLUMNS]}
    missing = []
    for j, (_, name) in enumerate(COLUMNS):
        d.text((label_w + j * tile + tile // 2, head_h // 2), name, fill="black", font=font, anchor="mm")
    for i, (a, b) in enumerate(PAIRS):
        slug = cell_for(a, b, SEED).pair_slug
        d.text((label_w - 12, head_h + i * tile + tile // 2), f"{a.split(' ', 1)[1]} ×\n{b.split(' ', 1)[1]}",
               fill="black", font=font, anchor="rm", align="right")
        row = {"pair": slug, "tiles": {}}
        for j, (col, _) in enumerate(COLUMNS):
            p = (CO3_TILES / slug / f"seed{SEED:02d}" / "co3.png") if col == "co3" \
                else (R6 / slug / f"seed{SEED:02d}" / f"{col}.png")
            row["tiles"][col] = str(p)
            if p.exists():
                img.paste(Image.open(p).convert("RGB").resize((tile, tile)), (label_w + j * tile, head_h + i * tile))
            else:
                missing.append(str(p))
                d.rectangle([label_w + j * tile + 4, head_h + i * tile + 4, label_w + (j + 1) * tile - 4,
                             head_h + (i + 1) * tile - 4], fill="#d9d9d9")
        record["rows"].append(row)
    record["missing"] = missing
    out = R6 / "five_columns.png"
    img.save(out)
    out.with_suffix(".json").write_text(json.dumps(record, indent=2))
    print(f"wrote {out}; {len(missing)} tile(s) missing", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--sheet", action="store_true")
    args = ap.parse_args()
    if args.render:
        render()
    if args.sheet:
        sheet()
    return 0


if __name__ == "__main__":
    sys.exit(main())
