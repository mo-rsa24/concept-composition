#!/usr/bin/env python
"""Does a single prompt's embedding already say "one animal"? The instrument, and its own check.

Builds a one-vs-many direction from matched prompts, "a cat" against "two cats", over the 81 nouns
the curated pool names, then checks that direction on held-out nouns against a random-direction
floor. Nothing downstream means anything until the floor is cleared, so the check is the point of
this step rather than a formality.

    d = mean over build nouns of ( E("two cats") - E("a cat") )

The read on a held-out noun is that noun's own singular-to-plural displacement along unit(d). The
floor is the same read under 200 random directions of the same shape: a small sample of
high-dimensional vectors lines up with any one direction by chance, so the raw displacement says
nothing on its own.

Four views of the prompt, as scripts/language_probes.py uses them. The pooled view is the primary
read, because adding two pooled vectors is a genuine vector-space addition, while position k does
not hold the same word across two prompts of different length.

Text encoders only, no UNet, fp32 on whatever device is free.

Usage:
    python scripts/one_animal_axis.py
    python scripts/one_animal_axis.py --pool artifacts/_shared/cross_pair_pool_configs/cells_v58.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch

REPO = Path(os.environ.get("POE_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from and_vector_norms import load_text_encoders  # noqa: E402
from poe_repair._sdxl.runtime import encode_prompt_sdxl  # noqa: E402
from poe_repair.config import DEFAULT_MODEL_ID  # noqa: E402

# Where the CLIP-L half of the 2048-wide sequence embedding ends, fixed by the concatenation in
# poe_repair/_sdxl/runtime.py.
CLIP_L_WIDTH = 768

# The axis is usable only this many times above the random-direction floor. In the source so it
# cannot be relaxed after seeing the ratio.
MIN_AXIS_OVER_FLOOR = 3.0

# How many random directions the floor averages over.
N_RANDOM = 200

# Plurals English does not form by adding s. Everything else takes the regular rules below, and the
# generated list is printed so a wrong one is caught by eye rather than by trust.
IRREGULAR = {
    "sheep": "sheep",
    "deer": "deer",
    "moose": "moose",
    "bison": "bison",
    "wolf": "wolves",
    "buffalo": "buffalo",
}


def pluralise(noun: str) -> str:
    """The plural of a noun phrase, formed on its last word."""
    head, _, last = noun.rpartition(" ")
    if last in IRREGULAR:
        plural = IRREGULAR[last]
    elif last.endswith(("s", "x", "z", "ch", "sh")):
        plural = last + "es"
    elif last.endswith("f"):
        plural = last[:-1] + "ves"
    elif last.endswith("y") and last[-2:-1] not in "aeiou":
        plural = last[:-1] + "ies"
    else:
        plural = last + "s"
    return f"{head} {plural}".strip()


def nouns_from_pool(pool_path: Path) -> list[str]:
    """The noun phrases the pool's pair keys name, without their articles."""
    pool = json.loads(pool_path.read_text())
    found = set()
    for key in pool:
        for half in key.split("__x__"):
            found.add(re.sub(r"^an?_", "", half).replace("_", " "))
    return sorted(found)


def views_of(seq: torch.Tensor, pooled: torch.Tensor, n_tok: int) -> dict[str, torch.Tensor]:
    """The four views of one prompt's embedding, each flattened to a vector."""
    s = seq.float()[0]
    return {
        "pooled_bigG": pooled.float().flatten(),
        "seq_clipL": s[:n_tok, :CLIP_L_WIDTH].flatten(),
        "seq_bigG": s[:n_tok, CLIP_L_WIDTH:].flatten(),
        "seq_both": s[:n_tok, :].flatten(),
    }


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm().clamp_min(1e-12)


def axis_and_floor(
    displacements: list[torch.Tensor], build: list[int], held: list[int], generator: torch.Generator
) -> dict:
    """The axis built on one half of the nouns, read on the other, against the random floor."""
    axis = unit(torch.stack([displacements[i] for i in build]).mean(dim=0))
    held_stack = torch.stack([displacements[i] for i in held])

    # The mean held-out singular-to-plural displacement along the axis. This is the unit every later
    # projection is reported in: one of these is one step from "a cat" to "two cats".
    along = held_stack @ axis
    separation = float(along.mean())

    floors = []
    for _ in range(N_RANDOM):
        r = unit(torch.randn(axis.shape, generator=generator, dtype=axis.dtype))
        floors.append(abs(float((held_stack @ r).mean())))
    floor = sum(floors) / len(floors)

    ratio = separation / floor if floor > 0 else float("inf")
    return {
        "separation_along_axis": separation,
        "random_direction_floor": floor,
        "ratio_over_floor": ratio,
        "clears_floor": bool(ratio >= MIN_AXIS_OVER_FLOOR),
        "per_heldout_noun_displacement": [float(x) for x in along],
        "axis_dim": int(axis.numel()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path,
                    default=REPO / "artifacts/_shared/cross_pair_pool_configs/cells_v58.json")
    ap.add_argument("--out", type=Path,
                    default=REPO / "artifacts/results/does-one-prompt-already-say-one-animal")
    ap.add_argument("--seed", type=int, default=0, help="the build/held-out split and the random floor")
    args = ap.parse_args()

    nouns = nouns_from_pool(args.pool)
    prompts = {n: (f"a {n}" if not n[0] in "aeiou" else f"an {n}", f"two {pluralise(n)}") for n in nouns}

    print(f"{len(nouns)} nouns from {args.pool.name}")
    print("the generated prompt pairs, to read by eye:")
    for n in nouns:
        s, p = prompts[n]
        print(f"  {s:28s} | {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_text_encoders(DEFAULT_MODEL_ID)
    for k in ("text_encoder", "text_encoder_2"):
        models[k].to(device)
    tokenizer = models["tokenizer"]

    # One token window for every prompt, so the sequence views share a dimension across nouns.
    n_tok = max(len(tokenizer(t).input_ids) for pair in prompts.values() for t in pair)
    print(f"\ncontent window: {n_tok} positions (longest prompt's end-of-text token)")

    def encode(text: str) -> dict[str, torch.Tensor]:
        seq, pooled = encode_prompt_sdxl(text, models=models, device=device, dtype=torch.float32)
        return views_of(seq.cpu(), pooled.cpu(), n_tok)

    displacement = {view: [] for view in ("pooled_bigG", "seq_clipL", "seq_bigG", "seq_both")}
    for n in nouns:
        s, p = prompts[n]
        vs, vp = encode(s), encode(p)
        for view in displacement:
            displacement[view].append(vp[view] - vs[view])

    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(nouns), generator=generator).tolist()
    build, held = sorted(order[: len(nouns) // 2]), sorted(order[len(nouns) // 2 :])

    per_view = {}
    for view, disps in displacement.items():
        per_view[view] = axis_and_floor(disps, build, held, torch.Generator().manual_seed(args.seed))

    primary = per_view["pooled_bigG"]
    print(f"\nheld-out check, bar is {MIN_AXIS_OVER_FLOOR}x the random floor")
    for view, r in per_view.items():
        mark = "clears" if r["clears_floor"] else "MISSES"
        print(f"  {view:14s} separation {r['separation_along_axis']:10.4f}  "
              f"floor {r['random_direction_floor']:8.4f}  ratio {r['ratio_over_floor']:7.2f}  {mark}")
    print(f"\nprimary read (pooled): ratio {primary['ratio_over_floor']:.2f}, "
          f"{'usable' if primary['clears_floor'] else 'NOT usable, claim one is inconclusive'}")

    args.out.mkdir(parents=True, exist_ok=True)
    record = {
        "what_this_is": "a one-vs-many direction built from matched singular and plural prompts, "
                        "and its held-out check against a random-direction floor",
        "model_id": DEFAULT_MODEL_ID,
        "layer": "hidden_states[-2]",
        "dtype": "float32",
        "bar": {"name": "MIN_AXIS_OVER_FLOOR", "value": MIN_AXIS_OVER_FLOOR,
                "meaning": "held-out singular-to-plural displacement along the axis, divided by the "
                           "mean absolute displacement along a random direction of the same shape"},
        "n_random_directions": N_RANDOM,
        "seed": args.seed,
        "content_window_tokens": n_tok,
        "prompt_pairs": {n: {"singular": prompts[n][0], "plural": prompts[n][1]} for n in nouns},
        "irregular_plurals_used": IRREGULAR,
        "build_nouns": [nouns[i] for i in build],
        "heldout_nouns": [nouns[i] for i in held],
        "per_view": per_view,
        "primary_view": "pooled_bigG",
    }
    path = args.out / "axis-heldout-check.json"
    path.write_text(json.dumps(record, indent=1))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
