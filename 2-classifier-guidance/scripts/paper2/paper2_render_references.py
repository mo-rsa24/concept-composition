"""Render SDXL's own single-species references for the DreamSim judge: "a photo of a <species>", plain SDXL.

The follow-on to step 0 (no judge met the bars): the DreamSim judge (paper2_judge_dreamsim.py) compares each body crop
with these renders. They are drawn the way the cat census drew its 64 cats, which the crop check reads
(3-understanding-compositionality/scripts/census/cat_condition_dog_rate.py: stabilityai/stable-diffusion-xl-base-1.0
in fp16, DDIM, 50 steps, guidance 7.5, 1024 px, a CPU generator seeded per picture), for every species in
paper2_bars.asked_species() (56) at paper2_bars.REFERENCE_SEEDS (9001 to 9008), so 448 pictures. Those seeds sit apart
from the 64 census cats (seeds 0 to 63), so no picture the crop check reads is its own reference. Every prompt passes
scripts/shared/disallowed_subjects.py before anything is drawn.

Writes under --out (default /datasets/mmolefe/poe_repair_min/paper2/paper2-references-sdxl): <species>/seed-<seed>.png
(a picture already there is skipped, so a run resumes) and record.json (each picture's species, seed, seconds and the
card). Exits 1 until every asked picture is on disk.

    bash paper2_launch.sh paper2-references-sdxl co3 16000 paper2_render_references.py
    bash paper2_launch.sh paper2-references-sdxl co3 16000 paper2_render_references.py --species cat dog --seeds-limit 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from paper2_bars import REFERENCE_SEEDS, asked_species  # noqa: E402
from paper2_common import PAPER2_DATASETS, with_article  # noqa: E402

RUN_NAME = "paper2-references-sdxl"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
NUM_STEPS = 50
GUIDANCE_SCALE = 7.5
SIDE = 1024


def prompt(species: str) -> str:
    return f"a photo of {with_article(species)}"


def ref_path(out: Path, species: str, seed: int) -> Path:
    return out / species.replace(" ", "_") / f"seed-{seed}.png"


def references(out: Path, species: str, n: int | None = None) -> list[Path]:
    """The first n references of a species on disk, in seed order (what the DreamSim judge reads)."""
    paths = [ref_path(out, species, s) for s in REFERENCE_SEEDS][:n]
    return [p for p in paths if p.exists()]


def guard(species: list[str]) -> None:
    sys.path.insert(0, str(REPO / "scripts/shared"))
    from disallowed_subjects import is_allowed
    refused = [prompt(s) for s in species if not is_allowed(prompt(s))]
    if refused:
        raise SystemExit(f"the subject guard refuses: {refused}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--species", nargs="+", default=None, help="default: paper2_bars.asked_species() (56)")
    ap.add_argument("--seeds-limit", type=int, default=None, help="a smoke: the first N seeds")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--list-only", action="store_true", help="print what would be drawn and stop; no torch")
    a = ap.parse_args(argv)
    species = a.species or asked_species()
    unknown = sorted(set(species) - set(asked_species()))
    if unknown:
        raise SystemExit(f"not an asked species: {unknown}")
    seeds = list(REFERENCE_SEEDS)[: a.seeds_limit]
    out = a.out or PAPER2_DATASETS / RUN_NAME
    guard(species)
    jobs = [(sp, s) for sp in species for s in seeds]
    todo = [(sp, s) for sp, s in jobs if not ref_path(out, sp, s).exists()]
    print(f"{len(jobs)} references asked ({len(species)} species, seeds {seeds[0]} to {seeds[-1]}), {len(todo)} to draw",
          flush=True)
    if a.list_only or not todo:
        return 0 if not todo else 1
    import torch
    from diffusers import DDIMScheduler, StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, use_safetensors=True,
                                                     variant="fp16").to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    card = torch.cuda.get_device_name(0)
    rec = out / "record.json"
    done = json.loads(rec.read_text())["pictures"] if rec.exists() else []
    for k, (sp, s) in enumerate(todo):
        t0 = time.time()
        image = pipe(prompt=prompt(sp), num_inference_steps=NUM_STEPS, guidance_scale=GUIDANCE_SCALE, height=SIDE,
                     width=SIDE, generator=torch.Generator(device="cpu").manual_seed(s)).images[0]
        path = ref_path(out, sp, s)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        done.append({"species": sp, "seed": s, "prompt": prompt(sp), "picture": str(path),
                     "seconds": round(time.time() - t0, 1)})
        print(f"{k + 1}/{len(todo)} {prompt(sp)} seed {s} {done[-1]['seconds']}s", flush=True)
        rec.write_text(json.dumps({"model": MODEL_ID, "steps": NUM_STEPS, "guidance_scale": GUIDANCE_SCALE,
                                   "scheduler": "DDIM", "side": SIDE, "card": card, "pictures": done}, indent=1))
    have = sum(ref_path(out, sp, s).exists() for sp, s in jobs)
    print(f"{have} of {len(jobs)} on disk under {out}", flush=True)
    return 0 if have == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
