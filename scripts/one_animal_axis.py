#!/usr/bin/env python
"""Does a single prompt's embedding already say "one animal"? The instrument, and the read off it.

Builds a one-vs-many direction from matched prompts, "a cat" against "two cats", over the nouns the
curated pool names, then checks that direction on held-out nouns against a random-direction floor.
Nothing downstream means anything until the floor is cleared, so the check is the point of the first
step rather than a formality.

    d = mean over build nouns of ( E("two cats") - E("a cat") )

The read on a held-out noun is that noun's own singular-to-plural displacement along unit(d). The
floor is the same read under 200 random directions of the same shape: a small sample of
high-dimensional vectors lines up with any one direction by chance, so the raw displacement says
nothing on its own.

With --read-pairs it then reads four prompts per pool pair off that axis: the two solo prompts, the
implied sum E(a) + E(b) - E(""), which is what a product of two experts assumes the joint means, and
the joint prompt itself. Positions are in held-out margin units, where one unit is the mean held-out
step from "a cat" to "two cats" and the origin is the midpoint between the held-out singular and
plural clouds. So a held-out singular prompt sits near -0.5, its plural near +0.5, and a number near
0 means the prompt is between the two.

**What the implied sum's position can and cannot tell you.** Projection is linear, so the sum's
position is its parts' positions added with the empty prompt's taken off. Summing two vectors of
similar size therefore moves the projection a long way by arithmetic alone, and a large number there
is not by itself evidence about plurality. The number worth reading is its sign against the solos'.

Four views of the prompt, as scripts/language_probes.py uses them. The pooled view is the primary
read, because adding two pooled vectors is a genuine vector-space addition, while position k does not
hold the same word across two prompts of different length.

Text encoders only, no UNet, fp32 on whatever device is free.

Usage:
    python scripts/one_animal_axis.py
    python scripts/one_animal_axis.py --read-pairs
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(os.environ.get("POE_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

# torch and the repo's own modules load inside the functions that need them, so --figure-only
# redraws the picture from the saved JSON on a machine with neither.

# Where the CLIP-L half of the 2048-wide sequence embedding ends, fixed by the concatenation in
# poe_repair/_sdxl/runtime.py.
CLIP_L_WIDTH = 768

VIEWS = ("pooled_bigG", "seq_clipL", "seq_bigG", "seq_both")
PRIMARY_VIEW = "pooled_bigG"

# The axis is usable only this many times above the random-direction floor. In the source so it
# cannot be relaxed after seeing the ratio.
MIN_AXIS_OVER_FLOOR = 3.0

# How many random directions the floor averages over.
N_RANDOM = 200

# The pair every number is quoted back against. Held out of every training pool on purpose.
TOY_PAIR = "a_cat__x__a_dog"

# Nouns that open with a vowel letter but a consonant sound, so they take "a" and not "an". The
# article is chosen by sound, and spelling is only a proxy for it, so the exceptions are listed
# rather than derived. "ukulele" opens on a "y" sound.
CONSONANT_SOUND = {"ukulele", "unicorn", "ewe", "one", "uniform", "union"}

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


def article(noun: str) -> str:
    """"a" or "an" for a noun phrase, chosen on the sound its first word opens with."""
    first = noun.split(" ")[0]
    if first in CONSONANT_SOUND:
        return "a"
    return "an" if first[0] in "aeiou" else "a"


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


def noun_of(half: str) -> str:
    """The noun phrase a pool key's half names, without its article."""
    return re.sub(r"^an?_", "", half).replace("_", " ")


def pool_pairs(pool_path: Path) -> list[str]:
    return sorted(json.loads(pool_path.read_text()))


def nouns_from_pool(pool_path: Path) -> list[str]:
    found = set()
    for key in pool_pairs(pool_path):
        for half in key.split("__x__"):
            found.add(noun_of(half))
    return sorted(found)


def views_of(seq, pooled, n_tok: int) -> dict:
    """The four views of one prompt's embedding, each flattened to a vector."""
    s = seq.float()[0]
    return {
        "pooled_bigG": pooled.float().flatten(),
        "seq_clipL": s[:n_tok, :CLIP_L_WIDTH].flatten(),
        "seq_bigG": s[:n_tok, CLIP_L_WIDTH:].flatten(),
        "seq_both": s[:n_tok, :].flatten(),
    }


def unit(v):
    return v / v.norm().clamp_min(1e-12)


def axis_and_floor(singular, plural, build, held, generator) -> dict:
    """The axis built on one half of the nouns, read on the other, against the random floor.

    Returns the stats, plus the axis and the two numbers every later position is expressed in: the
    origin (the midpoint between the held-out singular and plural clouds) and the margin unit (the
    mean held-out step from singular to plural).
    """
    disp = [p - s for s, p in zip(singular, plural)]
    axis = unit(torch.stack([disp[i] for i in build]).mean(dim=0))

    held_disp = torch.stack([disp[i] for i in held])
    along = held_disp @ axis
    separation = float(along.mean())

    floors = []
    for _ in range(N_RANDOM):
        r = unit(torch.randn(axis.shape, generator=generator, dtype=axis.dtype))
        floors.append(abs(float((held_disp @ r).mean())))
    floor = sum(floors) / len(floors)
    ratio = separation / floor if floor > 0 else float("inf")

    held_sing = torch.stack([singular[i] for i in held]) @ axis
    held_plur = torch.stack([plural[i] for i in held]) @ axis
    origin = float(((held_sing + held_plur) / 2).mean())

    return {
        "separation_along_axis": separation,
        "random_direction_floor": floor,
        "ratio_over_floor": ratio,
        "clears_floor": bool(ratio >= MIN_AXIS_OVER_FLOOR),
        "origin_raw_projection": origin,
        "margin_unit": separation,
        "heldout_singular_position": float(((held_sing - origin) / separation).mean()),
        "heldout_plural_position": float(((held_plur - origin) / separation).mean()),
        "heldout_singular_spread": float(((held_sing - origin) / separation).std()),
        "heldout_plural_spread": float(((held_plur - origin) / separation).std()),
        "per_heldout_noun_displacement": [float(x) for x in along],
        "axis_dim": int(axis.numel()),
        "_axis": axis,
    }


def strip_tensors(per_view: dict) -> dict:
    return {v: {k: val for k, val in r.items() if not k.startswith("_")} for v, r in per_view.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path,
                    default=REPO / "artifacts/_shared/cross_pair_pool_configs/cells_v58.json")
    ap.add_argument("--out", type=Path,
                    default=REPO / "artifacts/results/does-one-prompt-already-say-one-animal")
    ap.add_argument("--seed", type=int, default=0, help="the build/held-out split and the random floor")
    ap.add_argument("--read-pairs", action="store_true",
                    help="also read the four prompts of every pool pair off the checked axis")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--figure-only", action="store_true",
                    help="redraw the figure from the saved JSON, no encoders and no GPU")
    args = ap.parse_args()

    global torch, load_text_encoders, encode_prompt_sdxl, DEFAULT_MODEL_ID, joint_prompt

    if args.figure_only:
        land = json.loads((args.out / "where-the-prompts-land.json").read_text())
        check = json.loads((args.out / "axis-heldout-check.json").read_text())
        pv = land["primary_view"]
        _figure(land["rows"], land["heldout_singular_position"], land["heldout_plural_position"],
                check["per_view"][pv], pv, args.out)
        return 0

    import torch  # noqa: PLC0415
    from and_vector_norms import load_text_encoders  # noqa: PLC0415
    from poe_repair._sdxl.runtime import encode_prompt_sdxl  # noqa: PLC0415
    from poe_repair.config import DEFAULT_MODEL_ID, joint_prompt  # noqa: PLC0415

    nouns = nouns_from_pool(args.pool)
    prompts = {n: (f"{article(n)} {n}", f"two {pluralise(n)}") for n in nouns}

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

    pairs = pool_pairs(args.pool)
    pair_prompts = {}
    for key in sorted(set(pairs) | {TOY_PAIR}):
        halves = key.split("__x__")
        a, b = (f"{article(noun_of(h))} {noun_of(h)}" for h in halves)
        pair_prompts[key] = (a, b, joint_prompt(a, b))

    # One token window for every prompt in play, so the sequence views share a dimension throughout.
    texts = [t for pair in prompts.values() for t in pair]
    texts += [t for triple in pair_prompts.values() for t in triple] + [""]
    n_tok = max(len(tokenizer(t).input_ids) for t in texts)
    print(f"\ncontent window: {n_tok} positions (longest prompt's end-of-text token)")

    cache: dict[str, dict[str, torch.Tensor]] = {}

    def encode(text: str) -> dict[str, torch.Tensor]:
        if text not in cache:
            seq, pooled = encode_prompt_sdxl(text, models=models, device=device, dtype=torch.float32)
            cache[text] = views_of(seq.cpu(), pooled.cpu(), n_tok)
        return cache[text]

    singular = {v: [] for v in VIEWS}
    plural = {v: [] for v in VIEWS}
    for n in nouns:
        s, p = prompts[n]
        vs, vp = encode(s), encode(p)
        for v in VIEWS:
            singular[v].append(vs[v])
            plural[v].append(vp[v])

    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(nouns), generator=generator).tolist()
    build, held = sorted(order[: len(nouns) // 2]), sorted(order[len(nouns) // 2 :])

    per_view = {}
    for v in VIEWS:
        per_view[v] = axis_and_floor(singular[v], plural[v], build, held,
                                     torch.Generator().manual_seed(args.seed))

    primary = per_view[PRIMARY_VIEW]
    print(f"\nheld-out check, bar is {MIN_AXIS_OVER_FLOOR}x the random floor")
    for v, r in per_view.items():
        mark = "clears" if r["clears_floor"] else "MISSES"
        print(f"  {v:14s} separation {r['separation_along_axis']:10.4f}  "
              f"floor {r['random_direction_floor']:8.4f}  ratio {r['ratio_over_floor']:7.2f}  {mark}")
    print(f"\nprimary read ({PRIMARY_VIEW}): ratio {primary['ratio_over_floor']:.2f}, "
          f"{'usable' if primary['clears_floor'] else 'NOT usable, claim one is inconclusive'}")
    print(f"  held-out singular sits at {primary['heldout_singular_position']:+.3f} "
          f"(spread {primary['heldout_singular_spread']:.3f}) margin units")
    print(f"  held-out plural   sits at {primary['heldout_plural_position']:+.3f} "
          f"(spread {primary['heldout_plural_spread']:.3f}) margin units")

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
        "position_units": "held-out margin units: 1.0 is the mean held-out step from singular to "
                          "plural, and 0.0 is the midpoint between the two held-out clouds",
        "n_random_directions": N_RANDOM,
        "seed": args.seed,
        "content_window_tokens": n_tok,
        "prompt_pairs": {n: {"singular": prompts[n][0], "plural": prompts[n][1]} for n in nouns},
        "irregular_plurals_used": IRREGULAR,
        "consonant_sound_exceptions": sorted(CONSONANT_SOUND),
        "build_nouns": [nouns[i] for i in build],
        "heldout_nouns": [nouns[i] for i in held],
        "per_view": strip_tensors(per_view),
        "primary_view": PRIMARY_VIEW,
    }
    (args.out / "axis-heldout-check.json").write_text(json.dumps(record, indent=1))
    print(f"\nwrote {args.out / 'axis-heldout-check.json'}")

    if not args.read_pairs:
        return 0

    # ---- The read: four prompts per pair, positioned on the checked axis. ----
    empty = encode("")
    rows = []
    for key, (a, b, joint) in sorted(pair_prompts.items()):
        va, vb, vj = encode(a), encode(b), encode(joint)
        row = {"pair": key, "prompt_a": a, "prompt_b": b, "prompt_joint": joint,
               "in_training_pool": key in pairs, "views": {}}
        for v in VIEWS:
            axis, o, u = per_view[v]["_axis"], per_view[v]["origin_raw_projection"], per_view[v]["margin_unit"]
            pos = lambda vec: float(((vec @ axis) - o) / u)
            raw = lambda vec: float(vec @ axis)
            implied_raw = raw(va[v]) + raw(vb[v]) - raw(empty[v])
            row["views"][v] = {
                "a": pos(va[v]),
                "b": pos(vb[v]),
                "joint": pos(vj[v]),
                "implied_sum": float((implied_raw - o) / u),
                "empty": pos(empty[v]),
            }
        rows.append(row)

    pv = PRIMARY_VIEW
    sing_at = primary["heldout_singular_position"]
    plur_at = primary["heldout_plural_position"]
    print(f"\nwhere the prompts sit, {pv}, in margin units "
          f"(held-out singular {sing_at:+.2f}, plural {plur_at:+.2f})")
    toy = next(r for r in rows if r["pair"] == TOY_PAIR)
    tv = toy["views"][pv]
    print(f"  the toy, {TOY_PAIR}:")
    for k in ("a", "b", "joint", "implied_sum", "empty"):
        label = {"a": toy["prompt_a"], "b": toy["prompt_b"], "joint": toy["prompt_joint"],
                 "implied_sum": 'E(a)+E(b)-E("")', "empty": 'the empty prompt ""'}[k]
        print(f"    {label:28s} {tv[k]:+8.3f}")

    import statistics as st
    for k in ("a", "b", "joint", "implied_sum"):
        vals = [r["views"][pv][k] for r in rows]
        print(f"  over {len(vals)} pairs, {k:12s} median {st.median(vals):+8.3f}  "
              f"min {min(vals):+8.3f}  max {max(vals):+8.3f}")

    solo_med = st.median([r["views"][pv]["a"] for r in rows] + [r["views"][pv]["b"] for r in rows])
    joint_med = st.median([r["views"][pv]["joint"] for r in rows])
    print(f"\n  the joint prompt sits {joint_med - solo_med:+.3f} margin units from the solo median")
    print(f"  one margin unit is one held-out step from singular to plural, so a move of less than "
          f"{abs(plur_at - sing_at):.2f} is less than that step")

    out = {
        "what_this_is": "where the two solo prompts, their implied sum and the joint prompt sit on "
                        "the checked one-vs-many axis, per pool pair",
        "position_units": record["position_units"],
        "heldout_singular_position": sing_at,
        "heldout_plural_position": plur_at,
        "caveat_on_the_implied_sum": "projection is linear, so the sum's position is its parts' "
                                     "positions added with the empty prompt's removed. Summing two "
                                     "vectors of similar size moves the projection by arithmetic "
                                     "alone; the reading is its sign against the solos', not its size",
        "primary_view": pv,
        "toy_pair": TOY_PAIR,
        "rows": rows,
    }
    (args.out / "where-the-prompts-land.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out / 'where-the-prompts-land.json'}")

    if not args.no_figure:
        _figure(rows, sing_at, plur_at, primary, pv, args.out)
    return 0


def _figure(rows, sing_at, plur_at, primary, pv, out_dir) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = [("a", 'E(a), "a cat"'), ("b", 'E(b), "a dog"'),
             ("implied_sum", 'E(a)+E(b)-E(""), what the product assumes'),
             ("joint", 'E(joint), "a cat and a dog"')]
    fig, ax = plt.subplots(figsize=(10, 4.2))

    ss, ps = primary["heldout_singular_spread"], primary["heldout_plural_spread"]
    ax.axvspan(sing_at - ss, sing_at + ss, color="#4C72B0", alpha=0.15)
    ax.axvspan(plur_at - ps, plur_at + ps, color="#C44E52", alpha=0.15)
    ax.axvline(sing_at, color="#4C72B0", lw=1)
    ax.axvline(plur_at, color="#C44E52", lw=1)
    ax.text(sing_at, len(kinds) - 0.35, ' held-out singular,\n "a zebra"',
            color="#4C72B0", fontsize=8, va="top", ha="center")
    ax.text(plur_at, len(kinds) - 0.35, ' held-out plural,\n "two zebras"',
            color="#C44E52", fontsize=8, va="top", ha="center")

    # Fix the span now so every row label starts at the same place.
    everything = [r["views"][pv][k] for r in rows for k, _ in kinds]
    lo, hi = min(everything), max(plur_at + ps, max(everything))
    pad = (hi - lo) * 0.06
    ax.set_xlim(lo - pad, hi + pad)

    import random
    gen = random.Random(0)
    toy = next(r for r in rows if r["pair"] == TOY_PAIR)
    for i, (key, label) in enumerate(kinds):
        xs = [r["views"][pv][key] for r in rows]
        ys = [i + gen.random() * 0.3 - 0.15 for _ in xs]
        ax.scatter(xs, ys, s=14, color="#555555", alpha=0.55, linewidths=0)
        tx = toy["views"][pv][key]
        ax.scatter([tx], [i], s=70, color="#DD8452", zorder=5, edgecolor="black", linewidths=0.6)
        ax.annotate(f"cat x dog  {tx:+.2f}", (tx, i), textcoords="offset points",
                    xytext=(0, -16), fontsize=8, color="#8c4b1f", ha="center")
        ax.text(lo - pad * 0.5, i + 0.44, label, fontsize=9, va="bottom", color="#222222")

    ax.set_yticks([])
    ax.set_ylim(-0.6, len(kinds) + 0.1)
    ax.set_xlabel("position on the one-vs-many axis, in held-out margin units\n"
                  "(1.0 is one held-out step from singular to plural; 0.0 is the midpoint)")
    ax.set_title(f"Where each prompt sits on SDXL's one-vs-many direction, {pv}\n"
                 f"{len(rows)} pool pairs, axis checked at "
                 f"{primary['ratio_over_floor']:.0f}x the random floor", fontsize=10)
    fig.tight_layout()
    path = out_dir / "where-the-prompts-land.png"
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
