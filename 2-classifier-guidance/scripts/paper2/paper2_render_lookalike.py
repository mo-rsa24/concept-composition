"""Render the eight new look-alike pairs, for when look-alike pictures run short (species judge steps 0.9 and 0.10).

Two rows, each run in its own process, on paper2_bars.NEW_LOOKALIKE_PAIRS at NEW_LOOKALIKE_SEEDS (8001 to 8008):

  plain   step 0.9, `paper2-render-plain-lookalike`: the plain product of experts on an untouched U-Net, drawn by the
          sampler behind leakage/z10.sbatch (z10_dial.run_two_unet at weight 0, where the adapted copy is never called;
          no adapter is attached here at all), CFG 7.5 per expert, 50 DDIM steps, starting from the per-seed noise its
          cat-and-dog cell gives (drawn from the seed where no cached latent exists). The plain sum is where hybrids and
          two of one kind are common.
  co3     step 0.10, `paper2-render-co3-lookalike`: CO3 alone at its released settings, drawn the way
          method_search/paper2_rows.py draws its co3 row (set_prompt, seed_everything, model.sample_loop on CO3's own
          start_noise); a pair whose prompt CO3's parser does not split into 2 concepts is skipped and named.

Each writes, under --out: renders/<pair slug>/seed-<seed>.png (a picture already there is skipped, so a run resumes),
record.json (every picture with its pair, seed, seconds and the card), and contact-sheet.png (the eight pairs as rows,
the eight seeds as columns). paper2_collect.py --sources new-plain new-co3 turns them into candidates. Every pair passes
scripts/shared/disallowed_subjects.py before anything is drawn. CO3 is built through paper2_co3_patch.py, so it runs on
PyTorch's attention where xformers is missing.

    bash paper2_launch.sh paper2-render-plain-lookalike co3 16000 paper2_render_lookalike.py plain
    bash paper2_launch.sh paper2-render-co3-lookalike co3 16000 paper2_render_lookalike.py co3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from paper2_bars import NEW_LOOKALIKE_PAIRS, NEW_LOOKALIKE_SEEDS  # noqa: E402
from paper2_common import PAPER2_DATASETS, with_article  # noqa: E402

RUN_NAMES = {"plain": "paper2-render-plain-lookalike", "co3": "paper2-render-co3-lookalike"}
TILE = 160


def joint_prompt(a: str, b: str) -> str:
    return f"{with_article(a)} and {with_article(b)}"


def slug(a: str, b: str) -> str:
    return joint_prompt(a, b).replace(" and ", "__x__").replace(" ", "_")


def jobs(pairs=NEW_LOOKALIKE_PAIRS, seeds=NEW_LOOKALIKE_SEEDS) -> list[tuple[str, str, int]]:
    return [(a, b, s) for a, b in pairs for s in seeds]


def tile_path(out: Path, a: str, b: str, seed: int) -> Path:
    return out / "renders" / slug(a, b) / f"seed-{seed}.png"


def guard_subjects(pairs) -> None:
    sys.path.insert(0, str(REPO / "scripts/shared"))
    from disallowed_subjects import is_allowed
    refused = [joint_prompt(a, b) for a, b in pairs if not is_allowed(joint_prompt(a, b))]
    if refused:
        raise SystemExit(f"the subject guard refuses: {refused}")


def contact_sheet(out: Path, pairs, seeds) -> Path:
    page = Image.new("RGB", (TILE * len(seeds) + 170, TILE * len(pairs)), "white")
    d = ImageDraw.Draw(page)
    for r, (a, b) in enumerate(pairs):
        d.text((6, r * TILE + TILE // 2 - 6), f"{a} / {b}", fill="black")
        for c, s in enumerate(seeds):
            p = tile_path(out, a, b, s)
            if p.exists():
                with Image.open(p) as im:
                    page.paste(im.convert("RGB").resize((TILE, TILE)), (170 + c * TILE, r * TILE))
            else:
                d.rectangle((170 + c * TILE, r * TILE, 170 + (c + 1) * TILE - 1, (r + 1) * TILE - 1), outline="red")
    path = out / "contact-sheet.png"
    page.save(path)
    return path


def write_record(out: Path, row: str, done: list[dict], skipped: list[str], extra: dict) -> None:
    rec = out / "record.json"
    old = json.loads(rec.read_text())["pictures"] if rec.exists() else []
    have = {(p["pair"], p["seed"]) for p in done}
    pictures = [p for p in old if (p["pair"], p["seed"]) not in have] + done
    rec.write_text(json.dumps({"row": row, "pairs": [list(p) for p in NEW_LOOKALIKE_PAIRS],
                               "seeds": list(NEW_LOOKALIKE_SEEDS), "skipped_pairs": skipped, **extra,
                               "pictures": sorted(pictures, key=lambda p: (p["pair"], p["seed"]))}, indent=1))


def card() -> str:
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def render_plain(out: Path, todo) -> tuple[list[dict], list[str], dict]:
    import torch
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(HERE.parent))
    sys.path.insert(0, str(HERE.parent / "leakage"))
    import classifier_guidance_from_the_paper as single
    from poe_repair.composers._helpers import init_latents_for_cell
    from poe_repair.run import make_ctx
    from poe_repair.runtime import encode_prompt_sdxl
    from z10_dial import run_two_unet, to_img
    single.disk_guard(out)
    ctx = make_ctx(output_root=out / "raw")
    enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = enc("")
    common = dict(models=ctx.models, scheduler=ctx.scheduler, seq_e=seq_e, pool_e=pool_e,
                  guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps, height=1024,
                  width=1024, device=ctx.device, dtype=ctx.dtype)
    done, cache = [], {}
    for a, b, seed in todo:
        if (a, b) not in cache:
            sa, pa = enc(with_article(a))
            sb, pb = enc(with_article(b))
            cache = {(a, b): dict(seq_a=sa, pool_a=pa, seq_b=sb, pool_b=pb)}
        init, sigma = init_latents_for_cell(single.cell_for(seed), ctx)
        t0 = time.perf_counter()
        o = run_two_unet(base=ctx.models["unet"], adapted=None, lam=0.0, init_latents=init,
                         euler_init_noise_sigma=sigma, **cache[(a, b)], **common)
        path = tile_path(out, a, b, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(to_img(o)).save(path)
        done.append({"pair": slug(a, b), "seed": seed, "picture": str(path), "seconds": round(time.perf_counter() - t0, 1)})
        print(f"{len(done)}/{len(todo)} plain {slug(a, b)} seed {seed} {done[-1]['seconds']}s", flush=True)
    return done, [], {"guidance_scale": ctx.guidance_scale, "steps": ctx.num_inference_steps, "card": card(),
                      "sampler": "leakage/z10_dial.run_two_unet at weight 0, no adapter attached"}


def render_co3(out: Path, todo) -> tuple[list[dict], list[str], dict]:
    for p in (HERE.parent / "method_search", HERE.parent / "rung2", REPO):
        sys.path.insert(0, str(p))
    from paper2_co3_patch import use_pytorch_attention_if_no_xformers
    attention = use_pytorch_attention_if_no_xformers()
    import paper2_rows as rows
    from composers.utils_custom import seed_everything
    parser = rows.load_parser()
    a0, b0, _ = todo[0]
    model, config = rows.load(joint_prompt(a0, b0), parser, out / "_config")
    done, skipped, current = [], [], None
    for a, b, seed in todo:
        if (a, b) != current:
            current = (a, b)
            concepts = rows.set_prompt(model, config, joint_prompt(a, b), parser)
            if concepts != 2:
                skipped.append(f"{slug(a, b)}: CO3's parser found {concepts} concepts")
                print(f"skipping {slug(a, b)}: {concepts} concepts", flush=True)
        if any(s.startswith(slug(a, b) + ":") for s in skipped):
            continue
        t0 = time.time()
        seed_everything(seed)
        image = model.sample_loop(rows.start_noise(model, seed))[0]
        path = tile_path(out, a, b, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        done.append({"pair": slug(a, b), "seed": seed, "picture": str(path), "seconds": round(time.time() - t0, 1)})
        print(f"{len(done)}/{len(todo)} co3 {slug(a, b)} seed {seed} {done[-1]['seconds']}s", flush=True)
    return done, skipped, {"attention": attention, "card": card(),
                           "sampler": "CO3 alone at its released settings, as paper2_rows.py draws its co3 row"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("row", choices=sorted(RUN_NAMES))
    ap.add_argument("--out", type=Path, default=None, help="default /datasets/mmolefe/poe_repair_min/paper2/<run name>")
    ap.add_argument("--pairs-limit", type=int, default=None, help="a smoke: the first N pairs")
    ap.add_argument("--seeds-limit", type=int, default=None, help="a smoke: the first N seeds")
    ap.add_argument("--sheet-only", action="store_true", help="redraw contact-sheet.png from the pictures on disk")
    a = ap.parse_args(argv)
    out = a.out or PAPER2_DATASETS / RUN_NAMES[a.row]
    pairs = NEW_LOOKALIKE_PAIRS[: a.pairs_limit]
    seeds = list(NEW_LOOKALIKE_SEEDS)[: a.seeds_limit]
    guard_subjects(pairs)
    out.mkdir(parents=True, exist_ok=True)
    if not a.sheet_only:
        todo = [(x, y, s) for x, y, s in jobs(pairs, seeds) if not tile_path(out, x, y, s).exists()]
        print(f"{a.row}: {len(jobs(pairs, seeds))} pictures asked, {len(todo)} to draw, the rest already on disk", flush=True)
        if todo:
            done, skipped, extra = (render_plain if a.row == "plain" else render_co3)(out, todo)
            write_record(out, a.row, done, skipped, extra)
    sheet = contact_sheet(out, pairs, seeds)
    have = sum(tile_path(out, x, y, s).exists() for x, y, s in jobs(pairs, seeds))
    print(f"{have} of {len(jobs(pairs, seeds))} on disk; contact sheet {sheet}", flush=True)
    return 0 if have == len(jobs(pairs, seeds)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
