#!/usr/bin/env python3
"""Paper 2's first result, step 2 (`paper2-first-checks`): the sets, the training window, the trained judge's calls, CO3's
redraw count, the rollout check and checks.txt.

plans/paper-2-dials/plans/03-identity-dial-first-result.md, task 2 and "The pairs and seeds". The judge every read here
uses is the trained species judge (paper2_trained_judge.TrainedJudgeReward on the head from the wider pool), which the
owner chose on 2026-11-27 as the dial's steering signal and the kept judge; the verdict is read from blind calls only.

Subcommands, in the order step 2 runs them:

  sets      no card, no torch. Reads step 0's split.json (its test and choosing pairs) and the 182-cell pool
            (artifacts/_shared/cross_pair_pool_configs/cells_v58.json, 48 pairs) and fixes, in the plan's order:
              tuning   4 look-alike pairs drawn by seed 0 from the ladder's eleven look-alike training-pool pairs
                       (ELEVEN, 04-full-ladder.md "The data"), none on step 0's test side
              test     step 0's look-alike test pairs, topped up to 16 by new look-alike pairs (NEW_CANDIDATES, in
                       order, preferring species no training pair holds), never in training or tuning
              training the 48 pool pairs and the eleven, minus the tuning and test pairs, plus new look-alike pairs
                       (NEW_CANDIDATES after the test top-ups, none holding a species of any test pair) until at least
                       9 are new and at least 16 look-alike pairs train; look-alike pairs weighted so they are drawn at
                       least half the time
              ceiling  16 look-alike training pairs drawn by seed 0
              smoke    4 training pairs drawn by seed 0: 2 look-alike, 2 unlike (two animals of different orders)
            Every pair passes scripts/shared/disallowed_subjects.py; every pair a sheet shows (tuning, test, ceiling)
            also passes its display hold. Writes sets.json, overlaps.txt, species-overlap.csv and the pairs files
            (train, smoke, tuning, test, ceiling and choosing). Exits 1 if any of the plan's pass lines fails.
  window    the card. The plan's step 0.24 read with the trained head as the judge: CO3 drawn on step 0's choosing
            pairs at WINDOW_SEEDS; at every step k the estimate (x_k minus its predicted noise, decoded, as the trainer
            reads it) is called by the head and compared with the head's own call on the finished picture. Writes
            window-read.json (agreement at every step, at the plan's steps 15, 25, 35, 40 and 45, and whether any reaches
            the ladder's 0.75) and window.txt: the contiguous WINDOW_BLOCK-step block, after CO3's corrected steps, with
            the highest mean agreement (ties to the later block), "LO HI" with HI excluded.
  judge     the card. The trained head on every picture of a list (picture, a, b, and any other columns): its kind,
            two-clear probability and six probabilities, and the body counter's boxes. Writes trained-calls.json and
            bodies.json (the boxes in the layout paper2_common.load_bodies reads, so the runner-up judge reads the same
            bodies).
  redraw    no card. Two trained-calls.json files of CO3 drawn twice in two processes, matched by pair and seed: how many
            change kind, against FIRST_REDRAW_KIND_CHANGES_MAX. Writes redraw.json.
  rollout   the card, one process. One tuning pair at one seed: CO3's row as paper2_rows.py draws it (seed_everything,
            sample_loop), twice, then the trainer's own rollout with a dial attached at strength 0, once. The mean
            grey-level difference of each from the first CO3 picture; passes when the trainer's is no larger than the
            second CO3 picture's, or at most 1 grey level of 255. Writes rollout.json and the 3 pictures; exits 3 on a
            failure, which stops the plan.
  checks    no card. checks.txt from the parts: CO3's draws, peak memory (from a one-step training run's summary.json),
            the trainer's first step, the rollout check, the training window, the redraw count and the sets. Exits 1 if
            a part that stops the plan failed or is missing.

    python3 paper2_first_checks.py sets --split <paper2-label-set>/split.json --out <paper2-first-checks>
    bash paper2_launch.sh paper2-first-checks co3 30000 paper2_first_checks.py window \\
        --choosing <paper2-first-checks>/choosing-pairs.txt --judge-head <head.pt> --out <paper2-first-checks>/window
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from paper2_bars import ORDER  # noqa: E402
from paper2_common import REPO, species_from_slug, with_article  # noqa: E402
import paper2_first_result_bars as fb  # noqa: E402

sys.path.insert(0, str(REPO / "scripts" / "shared"))
from disallowed_subjects import is_allowed, is_showable  # noqa: E402

POOL_FILE = REPO / "artifacts/_shared/cross_pair_pool_configs/cells_v58.json"
DRAW_SEED = 0
TUNING_N, TEST_N, CEILING_N = 4, 16, 16
NEW_TRAINING_MIN, LOOKALIKE_TRAINING_MIN = 9, 16
SMOKE_LOOKALIKE, SMOKE_UNLIKE = 2, 2
TRAIN_SEEDS = range(11001, 11065)
TUNING_SEEDS = range(12001, 12009)
TEST_SEEDS = range(13001, 13004)
WINDOW_SEEDS = range(14001, 14005)     # the window read's CO3 draws on the choosing pairs; no other set uses them
WINDOW_BLOCK = 10                      # the owner's rule of 2026-11-27: the contiguous 10-step block of highest agreement
HALFDRAWN_BAR = 0.75                   # the ladder's R3_JUDGE_HALFDRAWN_MIN, recorded beside the block, not applied
PLAN_STEPS = (15, 25, 35, 40, 45)      # the steps species judge step 0.24 names

# The ladder's eleven look-alike training-pool pairs, in the order 04-full-ladder.md "The data" lists them: the rank-32
# pooled adapter's training pairs (its training cache, read 2026-09-08).
ELEVEN = [("wolf", "husky"), ("turtle", "tortoise"), ("rabbit", "hare"), ("lion", "tiger"), ("horse", "zebra"),
          ("gorilla", "chimpanzee"), ("donkey", "pony"), ("dolphin", "porpoise"), ("crow", "raven"),
          ("crocodile", "alligator"), ("cheetah", "cougar")]

# New look-alike pairs, in the order they are taken: first for the test top-ups, then for training. Each is two clearly
# different species of one taxonomic order, in no earlier pair table (paper2_bars.PAIRS, NEW_LOOKALIKE_PAIRS,
# BORROWED_TOPUP_PAIRS, ELEVEN, the 182-cell pool), no species in two of them, every one passing the subject guard and
# its display hold. Mammals and birds first; fish last. A word with a second common meaning (cardinal, kite) or a
# broader class (antelope, vulture) is not used.
NEW_CANDIDATES = [
    ("lynx", "ocelot"), ("orangutan", "baboon"), ("hawk", "osprey"), ("cockatoo", "macaw"), ("okapi", "alpaca"),
    ("mongoose", "ferret"), ("lemur", "gibbon"), ("turkey", "grouse"), ("tapir", "rhinoceros"), ("panda", "skunk"),
    ("squirrel", "capybara"), ("puffin", "sandpiper"), ("elk", "warthog"), ("wombat", "wallaby"), ("ibis", "egret"),
    ("sparrow", "bluebird"), ("mole", "shrew"), ("woodpecker", "toucan"), ("orca", "narwhal"), ("coyote", "dingo"),
    ("seal", "sea lion"), ("gerbil", "marmot"), ("albatross", "petrel"), ("reindeer", "ibex"), ("sloth", "anteater"),
    ("salmon", "trout"),
    ("tuna", "mackerel"), ("goldfish", "carp"),
]

# Orders for every species the pool, the eleven and the candidates name that paper2_bars.ORDER does not.
EXTRA_ORDER = {
    "hippo": "Artiodactyla", "husky": "Carnivora", "turtle": "Testudines", "gorilla": "Primates", "chimpanzee": "Primates",
    "langur": "Primates", "pony": "Perissodactyla", "dolphin": "Cetacea", "porpoise": "Cetacea",
    "crocodile": "Crocodilia", "alligator": "Crocodilia", "bear": "Carnivora", "hyena": "Carnivora",
    "jackal": "Carnivora", "meerkat": "Carnivora", "bison": "Artiodactyla", "buffalo": "Artiodactyla",
    "bull": "Artiodactyla", "deer": "Artiodactyla", "impala": "Artiodactyla", "oryx": "Artiodactyla",
    "beaver": "Rodentia", "chipmunk": "Rodentia", "heron": "Pelecaniformes", "woodpecker": "Piciformes",
    "rooster": "Galliformes", "chicken": "Galliformes", "robin": "Passeriformes", "emu": "Casuariiformes",
    "lynx": "Carnivora", "ocelot": "Carnivora", "orangutan": "Primates", "baboon": "Primates",
    "hawk": "Accipitriformes", "osprey": "Accipitriformes", "cockatoo": "Psittaciformes", "macaw": "Psittaciformes",
    "okapi": "Artiodactyla", "alpaca": "Artiodactyla", "mongoose": "Carnivora", "ferret": "Carnivora",
    "lemur": "Primates", "gibbon": "Primates", "turkey": "Galliformes", "grouse": "Galliformes", "tapir": "Perissodactyla",
    "squirrel": "Rodentia", "capybara": "Rodentia", "puffin": "Charadriiformes", "sandpiper": "Charadriiformes",
    "elk": "Artiodactyla", "warthog": "Artiodactyla", "wombat": "Diprotodontia", "wallaby": "Diprotodontia",
    "ibis": "Pelecaniformes", "egret": "Pelecaniformes", "sparrow": "Passeriformes", "bluebird": "Passeriformes",
    "mole": "Eulipotyphla", "shrew": "Eulipotyphla", "orca": "Cetacea", "narwhal": "Cetacea",
    "salmon": "Salmoniformes", "trout": "Salmoniformes", "tuna": "Scombriformes", "mackerel": "Scombriformes",
    "carp": "Cypriniformes", "dingo": "Carnivora", "sea lion": "Carnivora", "gerbil": "Rodentia", "marmot": "Rodentia",
    "albatross": "Procellariiformes", "petrel": "Procellariiformes", "reindeer": "Artiodactyla", "ibex": "Artiodactyla",
    "anteater": "Pilosa",
}
TAXON = {**ORDER, **EXTRA_ORDER}

# The pool's things that are not animals: a pair holding one is neither look-alike nor unlike.
NOT_ANIMALS = {
    "bicycle", "birdhouse", "birdcage", "hay bale", "cactus", "cooler box", "coat rack", "potted plant", "guitar",
    "traffic cone", "carrot", "ukulele", "lighthouse", "sailboat", "mailbox", "pumpkin", "microscope", "test tube",
    "red barn", "rocking chair", "soccer ball", "park bench", "suitcase", "globe", "teddy bear", "birthday cake",
    "telescope", "beach ball", "typewriter", "windmill", "tractor", "accordion", "scarecrow", "lantern",
}


# ---------------------------------------------------------------------------------------------------------------- pairs

def slug(a: str, b: str) -> str:
    return f"{with_article(a).replace(' ', '_')}__x__{with_article(b).replace(' ', '_')}"


def key(a: str, b: str) -> frozenset:
    return frozenset((a, b))


def pair_type(a: str, b: str) -> str:
    """look-alike (two animals of one order), unlike (two animals of two orders) or object (a thing in the pair)."""
    if a in NOT_ANIMALS or b in NOT_ANIMALS:
        return "object"
    for s in (a, b):
        if s not in TAXON:
            raise SystemExit(f"no taxonomic order for {s!r}: add it to EXTRA_ORDER or NOT_ANIMALS before fixing the sets")
    return "look-alike" if TAXON[a] == TAXON[b] else "unlike"


def guard(a: str, b: str, shown: bool) -> str | None:
    """None when the pair may be used; else why not."""
    prompt = f"{with_article(a)} and {with_article(b)}"
    if not is_allowed(prompt):
        return "fails the subject guard"
    if shown and not is_showable(prompt):
        return "held from display"
    return None


def write_pairs(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(f"{r['slug']} {r.get('weight', 1):g}   # {r['type']}, {r['source']}\n" for r in rows))


def fix_sets(split: dict, pool_slugs: list[str]) -> dict:
    """Every set, in the plan's order, with what was left out and why, and the plan's pass lines."""
    excluded = []

    def entry(a, b, source, s=None):
        return {"slug": s or slug(a, b), "a": a, "b": b, "type": pair_type(a, b), "source": source}

    step0_test = []
    for s in split["test_pairs"]:
        a, b = species_from_slug(s)
        if pair_type(a, b) != "look-alike":
            continue
        why = guard(a, b, shown=True)
        if why:
            excluded.append({"pair": s, "from": "step 0's test side", "why": why})
            continue
        step0_test.append(entry(a, b, "step 0's look-alike test pairs", s))
    if len(step0_test) > TEST_N:
        step0_test = random.Random(DRAW_SEED).sample(step0_test, TEST_N)
    step0_test_keys = {key(*species_from_slug(s)) for s in split["test_pairs"]}

    eleven = []
    for a, b in ELEVEN:
        why = guard(a, b, shown=True)
        if why:
            excluded.append({"pair": slug(a, b), "from": "the eleven", "why": why})
        elif key(a, b) in step0_test_keys:
            excluded.append({"pair": slug(a, b), "from": "the eleven, as a tuning candidate", "why": "on step 0's test side"})
        else:
            eleven.append((a, b))
    if len(eleven) < TUNING_N:
        raise SystemExit(f"only {len(eleven)} of the eleven can be tuning pairs")
    tuning = [entry(a, b, "the eleven, drawn by seed 0") for a, b in random.Random(DRAW_SEED).sample(eleven, TUNING_N)]
    tuning_keys = {key(r["a"], r["b"]) for r in tuning}
    test_keys = {key(r["a"], r["b"]) for r in step0_test}

    base = []
    for s in pool_slugs:
        a, b = species_from_slug(s)
        if guard(a, b, shown=False):
            excluded.append({"pair": s, "from": "the 182-cell pool", "why": guard(a, b, shown=False)})
            continue
        base.append(entry(a, b, "the 182-cell pool", s))
    for a, b in ELEVEN:
        if key(a, b) in {key(r["a"], r["b"]) for r in base}:
            continue
        why = guard(a, b, shown=False)
        if why:
            continue   # already listed as excluded above
        base.append(entry(a, b, "the eleven"))
    base = [r for r in base if key(r["a"], r["b"]) not in tuning_keys | test_keys]
    base_species = {s for r in base for s in (r["a"], r["b"])}

    used = {key(a, b) for a, b in ELEVEN} | {key(*species_from_slug(s)) for s in pool_slugs} | test_keys | tuning_keys
    candidates = []
    for i, (a, b) in enumerate(NEW_CANDIDATES):
        if pair_type(a, b) != "look-alike":
            raise SystemExit(f"candidate {a} and {b} is not look-alike")
        why = guard(a, b, shown=True)
        if why or key(a, b) in used:
            excluded.append({"pair": slug(a, b), "from": "the new candidates", "why": why or "already in a set or pool"})
            continue
        candidates.append((i, a, b))
    need = TEST_N - len(step0_test)
    ranked = sorted(candidates, key=lambda c: (len({c[1], c[2]} & base_species), c[0]))
    top_ups = [entry(a, b, "new look-alike pair, test top-up") for _, a, b in ranked[:need]]
    if len(top_ups) < need:
        raise SystemExit(f"the candidates fill {len(top_ups)} of {need} test top-ups")
    test = step0_test + top_ups
    test_keys = {key(r["a"], r["b"]) for r in test}
    test_species = {s for r in test for s in (r["a"], r["b"])}

    training = [r for r in base if key(r["a"], r["b"]) not in test_keys]
    new_training = []
    for _, a, b in candidates:
        looks = [r for r in training + new_training if r["type"] == "look-alike"]
        n_show = sum(guard(r["a"], r["b"], shown=True) is None for r in looks)
        if len(new_training) >= NEW_TRAINING_MIN and len(looks) >= LOOKALIKE_TRAINING_MIN and n_show >= CEILING_N:
            break
        if key(a, b) in test_keys or {a, b} & test_species:
            continue
        new_training.append(entry(a, b, "new look-alike pair, training"))
    training += new_training
    look = [r for r in training if r["type"] == "look-alike"]
    other = [r for r in training if r["type"] != "look-alike"]
    # unlike and object pairs weigh 1; look-alike pairs weigh their ratio, rounded up to 3 decimals, so they are drawn at
    # least half the time
    w_look = max(1.0, math.ceil(1000 * len(other) / len(look)) / 1000) if look else 1.0
    for r in training:
        r["weight"] = w_look if r["type"] == "look-alike" else 1.0
    look_share = sum(r["weight"] for r in look) / sum(r["weight"] for r in training)

    rng = random.Random(DRAW_SEED)
    showable = sorted([r for r in look if guard(r["a"], r["b"], shown=True) is None], key=lambda r: r["slug"])
    ceiling = [dict(r, source=r["source"] + "; ceiling, drawn by seed 0")
               for r in rng.sample(showable, min(CEILING_N, len(showable)))]
    rng = random.Random(DRAW_SEED)
    unlike = sorted([r for r in training if r["type"] == "unlike"], key=lambda r: r["slug"])
    smoke = rng.sample(sorted(look, key=lambda r: r["slug"]), SMOKE_LOOKALIKE) + rng.sample(unlike, SMOKE_UNLIKE)
    smoke = [dict(r, weight=1.0) for r in smoke]
    choosing = []
    for s in split["choosing_pairs"]:
        a, b = species_from_slug(s)
        choosing.append(entry(a, b, "step 0's choosing pairs", s))

    keys = {name: {key(r["a"], r["b"]) for r in rows} for name, rows in
            (("tuning", tuning), ("test", test), ("training", training), ("ceiling", ceiling))}
    overlaps = {f"{x} and {y}": sorted(" and ".join(sorted(k)) for k in keys[x] & keys[y])
                for i, x in enumerate(keys) for y in list(keys)[i + 1:]}
    species_in_training = {s for r in training for s in (r["a"], r["b"])}
    new_species = {s for r in new_training for s in (r["a"], r["b"])}
    checks = {
        "no set overlaps another (the ceiling is drawn from training)": all(
            not v for k, v in overlaps.items() if k != "training and ceiling"),
        "the ceiling pairs are training pairs": keys["ceiling"] <= keys["training"],
        "4 tuning pairs, all look-alike": len(tuning) == TUNING_N and all(r["type"] == "look-alike" for r in tuning),
        "16 test pairs, all look-alike": len(test) == TEST_N and all(r["type"] == "look-alike" for r in test),
        "16 ceiling pairs, all look-alike": len(ceiling) == CEILING_N and all(r["type"] == "look-alike" for r in ceiling),
        "no species of a test pair is in a new training pair": not (test_species & new_species),
        f"at least {NEW_TRAINING_MIN} new look-alike training pairs": len(new_training) >= NEW_TRAINING_MIN,
        f"at least {LOOKALIKE_TRAINING_MIN} look-alike pairs train": len(look) >= LOOKALIKE_TRAINING_MIN,
        "look-alike pairs drawn at least half the time": look_share >= 0.5,
    }
    species_overlap = [{"pair": r["slug"], "a": r["a"], "b": r["b"], "source": r["source"],
                        "a_in_training": r["a"] in species_in_training, "b_in_training": r["b"] in species_in_training,
                        "species_in_training": (r["a"] in species_in_training) + (r["b"] in species_in_training),
                        "species_novel": not ({r["a"], r["b"]} & species_in_training)} for r in test]
    return {"draw_seed": DRAW_SEED, "seeds": {"training": [TRAIN_SEEDS[0], TRAIN_SEEDS[-1]],
                                              "tuning": [TUNING_SEEDS[0], TUNING_SEEDS[-1]],
                                              "test": [TEST_SEEDS[0], TEST_SEEDS[-1]],
                                              "window": [WINDOW_SEEDS[0], WINDOW_SEEDS[-1]]},
            "tuning": tuning, "test": test, "training": training, "new_training": new_training, "ceiling": ceiling,
            "smoke": smoke, "choosing": choosing, "excluded": excluded, "overlaps": overlaps,
            "species_overlap": species_overlap, "lookalike_weight": w_look, "lookalike_draw_share": round(look_share, 4),
            "counts": {"tuning": len(tuning), "test": len(test), "test_from_step0": len(step0_test),
                       "test_new": len(top_ups), "training": len(training), "training_lookalike": len(look),
                       "training_unlike": sum(r["type"] == "unlike" for r in training),
                       "training_object": sum(r["type"] == "object" for r in training),
                       "new_training": len(new_training), "ceiling": len(ceiling),
                       "species_novel_test": sum(r["species_novel"] for r in species_overlap)},
            "checks": checks, "passed": all(checks.values())}


def cmd_sets(a) -> int:
    split = json.loads(Path(a.split).read_text())
    pool = sorted(json.loads(Path(a.pool).read_text()))
    sets = fix_sets(split, pool)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sets["split"], sets["pool"] = str(a.split), str(a.pool)
    (out / "sets.json").write_text(json.dumps(sets, indent=1))
    for name, rows in (("train", sets["training"]), ("smoke", sets["smoke"]), ("tuning", sets["tuning"]),
                       ("test", sets["test"]), ("ceiling", sets["ceiling"]), ("choosing", sets["choosing"])):
        write_pairs(out / f"{name}-pairs.txt", rows)
    lines = [f"{k}: {len(v)} shared{': ' + ', '.join(v) if v else ''}" for k, v in sets["overlaps"].items()]
    lines.append("species of test pairs in new training pairs: " + (", ".join(sorted(
        {s for r in sets["test"] for s in (r["a"], r["b"])} & {s for r in sets["new_training"] for s in (r["a"], r["b"])}))
        or "none"))
    lines += ["", "left out:"] + [f"  {e['pair']} ({e['from']}): {e['why']}" for e in sets["excluded"]]
    lines += ["", "checks:"] + [f"  {'PASS' if v else 'FAIL'}  {k}" for k, v in sets["checks"].items()]
    (out / "overlaps.txt").write_text("\n".join(lines) + "\n")
    with open(out / "species-overlap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sets["species_overlap"][0]))
        w.writeheader()
        w.writerows(sets["species_overlap"])
    c = sets["counts"]
    print(f"tuning {c['tuning']}; test {c['test']} ({c['test_from_step0']} from step 0, {c['test_new']} new; "
          f"{c['species_novel_test']} species-novel); training {c['training']} ({c['training_lookalike']} look-alike, "
          f"{c['new_training']} of them new; {c['training_unlike']} unlike; {c['training_object']} with an object), "
          f"look-alike weight {sets['lookalike_weight']} so drawn {sets['lookalike_draw_share']:.3f} of the time; "
          f"ceiling {c['ceiling']}")
    print("\n".join(lines))
    print("sets PASSED" if sets["passed"] else "sets FAILED")
    return 0 if sets["passed"] else 1


# ---------------------------------------------------------------------------------------------------------------- the judge

def read_list(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def halves(image):
    """Dry runs only: two boxes, the left and right halves of the picture."""
    _, H, W = image.shape
    return [[0, 0, W // 2, H], [W // 2, 0, W, H]]


def make_judge(a):
    import torch  # noqa: F401
    from paper2_trained_judge import TrainedJudgeReward
    if not a.judge_head or not Path(a.judge_head).is_file():
        raise SystemExit(f"--judge-head {a.judge_head} is not a file")
    return TrainedJudgeReward(Path(a.judge_head), a.device, boxes_fn=halves if a.dry_run else None)


def call_picture(judge, image, a_name: str, b_name: str) -> dict:
    import torch
    from paper2_kinds import KINDS
    with torch.no_grad():
        boxes = judge.boxes(image)
        p = judge.probs(image, a_name, b_name, boxes)
    return {"kind": KINDS[int(p.argmax())], "two_clear_score": round(float(p[KINDS.index("two_clear")]), 6),
            "probs": {k: round(float(v), 6) for k, v in zip(KINDS, p)}, "n_bodies": len(boxes), "boxes": boxes}


def sha256(path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cmd_judge(a) -> int:
    from paper2_trained_judge import load_picture
    rows = read_list(a.pictures)
    judge = make_judge(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pictures, bodies, t0 = [], {}, time.time()
    for i, r in enumerate(rows):
        call = call_picture(judge, load_picture(r["picture"], a.device), r["a"], r["b"])
        pictures.append({**r, **call})
        bodies[r["picture"]] = {"n": call["n_bodies"], "boxes": call["boxes"]}
        if (i + 1) % 50 == 0:
            print(f"  {i + 1} of {len(rows)}", flush=True)
    secs = time.time() - t0
    (out / "trained-calls.json").write_text(json.dumps({
        "judge": "trained", "head": str(a.judge_head), "head_sha256": sha256(a.judge_head), "list": str(a.pictures),
        "seconds_per_picture": round(secs / max(len(rows), 1), 3), "pictures": pictures}, indent=1))
    (out / "bodies.json").write_text(json.dumps({"pictures": bodies, "counter": "the trained judge's own boxes "
                                                 "(BodyBoxes, count_instances queried 'animal')"}, indent=1))
    kinds = Counter(p["kind"] for p in pictures)
    print(f"wrote {out / 'trained-calls.json'} and bodies.json: {len(pictures)} pictures, {dict(kinds)}, "
          f"{secs / max(len(rows), 1):.2f} s a picture")
    return 0


# ---------------------------------------------------------------------------------------------------------------- the window

def choose_block(agreement: list[float], first: int, block: int = WINDOW_BLOCK) -> tuple[int, int, float]:
    """The contiguous `block` steps from `first` on with the highest mean agreement; ties go to the later block."""
    best = None
    for lo in range(first, len(agreement) - block + 1):
        m = statistics.fmean(agreement[lo:lo + block])
        if best is None or m >= best[2] - 1e-12:
            best = (lo, lo + block, m)
    if best is None:
        raise SystemExit(f"no {block}-step block fits between step {first} and step {len(agreement)}")
    return best


def cmd_window(a) -> int:
    import torch
    import paper2_identity_dial as pid
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.dry_run:
        sampler = pid.stand_in_sampler(0)
    else:
        sampler = pid.co3_sampler(a.device, out)
    judge = make_judge(a)
    pairs = pid.read_pairs(a.choosing)
    seeds = pid.parse_seeds(a.seeds) if a.seeds else list(WINDOW_SEEDS)
    n = sampler.n_steps
    records, t0 = [], time.time()
    for pa, pb, _ in pairs:
        sampler.set_pair(pa, pb)
        for s in seeds:
            x = sampler.start(s)
            steps = []
            for k in range(n):
                at = sampler.alpha(k)
                with torch.no_grad():
                    eps = sampler.guided_eps(x, k)
                    est = sampler.decode((x.float() - (1 - at).sqrt() * eps) / at.sqrt()).float()
                c = call_picture(judge, est, pa, pb)
                steps.append({"kind": c["kind"], "two_clear_score": c["two_clear_score"], "n_bodies": c["n_bodies"]})
                x = sampler.rollout(x, k, k + 1, None, n, 0.0)
            finished = sampler.finish(x)
            path = out / "finished" / pid.slug_of(pa, pb) / f"seed-{s}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            pid.to_pil(finished).save(path)
            final = call_picture(judge, finished, pa, pb)
            records.append({"pair": f"{pa}|{pb}", "seed": s, "picture": str(path), "finished_kind": final["kind"],
                            "finished_two_clear_score": final["two_clear_score"], "steps": steps})
            agree = sum(st["kind"] == final["kind"] for st in steps)
            print(f"{pa} and {pb} seed {s}: finished {final['kind']}; the estimate agrees at {agree} of {n} steps "
                  f"({time.time() - t0:.0f} s so far)", flush=True)
    agreement = [statistics.fmean(r["steps"][k]["kind"] == r["finished_kind"] for r in records) for k in range(n)]
    lo, hi, mean = choose_block(agreement, sampler.corrected_steps, a.block)
    result = {
        "judge": "trained", "head": str(a.judge_head), "head_sha256": sha256(a.judge_head),
        "block": a.block,
        "rule": f"the contiguous {a.block}-step block from CO3's first uncorrected step on with the highest mean "
                "agreement between the head's call on the decoded estimate and its own call on the finished picture "
                "(ties to the later block); the owner's rule of 2026-11-27, standing in for the plan's 0.75 bar",
        "pairs": [f"{p}|{q}" for p, q, _ in pairs], "seeds": seeds, "n_pictures": len(records), "n_steps": n,
        "corrected_steps": sampler.corrected_steps,
        "agreement_by_step": {str(k): round(v, 4) for k, v in enumerate(agreement)},
        "two_clear_by_step": {str(k): round(statistics.fmean(r["steps"][k]["two_clear_score"] for r in records), 4)
                              for k in range(n)},
        "plan_steps": {str(k): round(agreement[k], 4) for k in PLAN_STEPS if k < n},
        "plan_bar": HALFDRAWN_BAR,
        "plan_steps_reaching_bar": [k for k in PLAN_STEPS if k < n and agreement[k] >= HALFDRAWN_BAR],
        "steps_reaching_bar": [k for k in range(n) if agreement[k] >= HALFDRAWN_BAR],
        "any_step_reaches_bar": any(v >= HALFDRAWN_BAR for v in agreement),
        "what_the_plan_rule_would_say": ("train in the steps that reach 0.75" if any(
            agreement[k] >= HALFDRAWN_BAR for k in PLAN_STEPS if k < n) else
            "no step of 15, 25, 35, 40 and 45 reaches 0.75: the plan's rule would send the dial to kept draws"),
        "finished_kinds": dict(Counter(r["finished_kind"] for r in records)),
        "window": [lo, hi], "window_mean_agreement": round(mean, 4),
        "seconds": round(time.time() - t0, 1), "pictures": records}
    (out / "window-read.json").write_text(json.dumps(result, indent=1))
    (out / "window.txt").write_text(f"{lo} {hi}\n")
    print("agreement by step: " + " ".join(f"{k}:{v:.2f}" for k, v in enumerate(agreement)))
    print(f"plan steps {result['plan_steps']}; any step at 0.75 or more: {result['any_step_reaches_bar']}")
    print(f"window {lo} to {hi} (mean agreement {mean:.4f}); wrote {out / 'window-read.json'} and window.txt")
    return 0


# ---------------------------------------------------------------------------------------------------------------- the redraw

def cmd_redraw(a) -> int:
    first = {(p["pair"], str(p["seed"])): p for p in json.loads(Path(a.first).read_text())["pictures"]}
    second = {(p["pair"], str(p["seed"])): p for p in json.loads(Path(a.second).read_text())["pictures"]}
    if set(first) != set(second):
        raise SystemExit(f"the two draws hold different pairs and seeds ({len(first)} and {len(second)})")
    rows = [{"pair": k[0], "seed": int(k[1]), "first": first[k]["kind"], "second": second[k]["kind"],
             "changed": first[k]["kind"] != second[k]["kind"]} for k in sorted(first)]
    changed = sum(r["changed"] for r in rows)
    band = fb.redraw_band_needed(changed, len(rows))
    out = {"judge": "the trained head (the kept judge)", "of": len(rows), "changed": changed,
           "bar": fb.FIRST_REDRAW_KIND_CHANGES_MAX, "within_bar": band is None,
           "random_dial_band_needed": band, "pictures": rows}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"CO3 drawn twice: {changed} of {len(rows)} tuning pictures change kind by the trained head "
          f"(bar {fb.FIRST_REDRAW_KIND_CHANGES_MAX}); " + (
              "within the bar" if band is None else
              f"OVER the bar: instruction 1.1 sets FIRST_RANDOM_DIAL_CLOSE to at least {band:.4f}"))
    return 0


# ---------------------------------------------------------------------------------------------------------------- the rollout check

def grey_difference(x, y) -> float:
    import numpy as np
    return float(np.abs(np.asarray(x, dtype="float64") - np.asarray(y, dtype="float64")).mean())


def cmd_rollout(a) -> int:
    """CO3's row twice, then the trainer's strength-0 rollout, with what separates them if they differ.

    Beside the pass rule (unchanged): CO3's own steps run by hand under its sample_loop's autocast, giving its finished
    latent, against the trainer's finished latent (the largest absolute difference, and whether they are equal); that
    hand-run latent decoded by finish_by_model against CO3's first picture (0 when finish is CO3's own decode); and the
    trainer's latent through the reward's fp32 graph decode against CO3's first picture (the decode the finished pictures
    used to take)."""
    import contextlib
    import torch
    import paper2_identity_dial as pid
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pa, pb, _ = pid.read_pairs(a.pairs_file)[0]
    pictures = {}
    if a.dry_run:
        row = pid.stand_in_sampler(0)
        row.set_pair(pa, pb)
        row_model, row_start = row.model, row.start
        seed_all = lambda s: None                                      # noqa: E731
        autocast = contextlib.nullcontext
        path = "stand-in sampler, both rows"
    else:
        for p in (HERE.parent / "method_search", HERE.parent / "rung2", REPO):
            sys.path.insert(0, str(p))
        from paper2_co3_patch import use_pytorch_attention_if_no_xformers
        use_pytorch_attention_if_no_xformers()
        import paper2_rows as rows
        from composers.utils_custom import seed_everything as seed_all
        parser = rows.load_parser()
        prompt = f"{with_article(pa)} and {with_article(pb)}"
        row_model, config = rows.load(prompt, parser, out / "_config")
        rows.set_prompt(row_model, config, prompt, parser)
        row_start = lambda s: rows.start_noise(row_model, s)           # noqa: E731
        autocast = lambda: torch.autocast(device_type="cuda", dtype=torch.float16)   # noqa: E731
        path = "CO3's row: paper2_rows (seed_everything, sample_loop); the trainer: co3_sampler's rollout and finish"
    for name in ("co3-1", "co3-2"):
        seed_all(a.seed)
        pictures[name] = row_model.sample_loop(row_start(a.seed))[0]
    seed_all(a.seed)
    x = row_start(a.seed)
    with torch.no_grad(), autocast():
        for i, t in enumerate(row_model.scheduler.timesteps):
            x = row_model.denoise_step(x, t, i)
    row_latent = x.detach().clone()
    pictures["co3-steps-then-finish"] = pid.to_pil(pid.finish_by_model(row_model, x))
    if a.dry_run:
        sampler = pid.stand_in_sampler(0)
    else:
        row_latent = row_latent.cpu()
        del row_model
        torch.cuda.empty_cache()
        sampler = pid.co3_sampler(a.device, out)
    sampler.set_pair(pa, pb)
    dial = pid.attach_dial(sampler.unet, pid.RANK)
    x = sampler.rollout(sampler.start(a.seed), 0, sampler.n_steps, dial, sampler.n_steps, 0.0)
    pictures["trainer-strength-0"] = pid.to_pil(sampler.finish(x))
    with torch.no_grad():
        graph_decode = pid.to_pil(sampler.decode(x.float()))
    for name, im in pictures.items():
        im.save(out / f"{name}.png")
    trainer_latent = x.detach().to(row_latent.device)
    redraw = grey_difference(pictures["co3-2"], pictures["co3-1"])
    trainer = grey_difference(pictures["trainer-strength-0"], pictures["co3-1"])
    passed = trainer <= redraw or trainer <= 1.0
    result = {"pair": f"{pa}|{pb}", "seed": a.seed, "paths": path,
              "co3_second_minus_first_grey": round(redraw, 4), "trainer_minus_co3_first_grey": round(trainer, 4),
              "rule": "the trainer's strength-0 picture differs from the first CO3 picture by no more than the second "
                      "CO3 picture does, or by at most 1 grey level of 255",
              "passed": passed,
              "latents_equal": bool(row_latent.dtype == trainer_latent.dtype and torch.equal(row_latent, trainer_latent)),
              "latent_dtypes": [str(row_latent.dtype), str(trainer_latent.dtype)],
              "latent_largest_difference": float((row_latent.float() - trainer_latent.float()).abs().max()),
              "co3_steps_then_finish_minus_co3_first_grey": round(
                  grey_difference(pictures["co3-steps-then-finish"], pictures["co3-1"]), 4),
              "trainer_graph_decode_minus_co3_first_grey": round(grey_difference(graph_decode, pictures["co3-1"]), 4)}
    (out / "rollout.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result))
    if not passed:
        print("STOP: the trainer is not rolling out CO3's row (task 2); point the rollout at CO3's row before training")
        return 3
    return 0


# ---------------------------------------------------------------------------------------------------------------- checks.txt

def cmd_checks(a) -> int:
    d = Path(a.dir)
    lines, stops = [], []

    def load(p):
        p = d / p
        return json.loads(p.read_text()) if p.exists() else None
    sets = load("sets.json")
    if sets is None:
        stops.append("sets.json is missing")
    else:
        c = sets["counts"]
        lines.append(f"sets: {'pass' if sets['passed'] else 'FAIL'}; tuning {c['tuning']}, test {c['test']} "
                     f"({c['test_new']} new, {c['species_novel_test']} species-novel), training {c['training']} "
                     f"({c['training_lookalike']} look-alike, {c['new_training']} new), ceiling {c['ceiling']}")
        if not sets["passed"]:
            stops.append("the sets fail a pass line (overlaps.txt)")
    for i in (1, 2):
        dj = load(f"co3-draw-{i}/draw.json")
        n = len(dj["pictures"]) if dj else 0
        lines.append(f"CO3 draw {i}: {n} tuning pictures (no adapter attached, its own process)")
        if n != len(TUNING_SEEDS) * TUNING_N:
            stops.append(f"CO3 draw {i} holds {n} pictures, not {len(TUNING_SEEDS) * TUNING_N}")
    summ = load("memory/summary.json")
    if summ is None:
        stops.append("the one-step memory run's summary.json is missing")
    else:
        m = summ.get("memory", {})
        lines.append(f"memory, one training step on the card: peak {m.get('peak_mib')} MiB of {m.get('card_mib')}; the "
                     f"judge {m.get('judge_mib')} MiB; the task step {m.get('task_step_mib')} MiB above its start; the "
                     f"closeness batch {m.get('closeness_batch_mib')} MiB above its start; SDXL with the dial "
                     f"{m.get('after_model')} MiB")
        if m.get("peak_mib") and m.get("card_mib") and m["peak_mib"] > m["card_mib"]:
            stops.append("one training step does not fit the card")
        log = d / "memory" / "train-log.jsonl"
        if log.exists():
            r = json.loads(log.read_text().splitlines()[0])
            lines.append(f"the trainer's first step: reward {r['reward']:.6f}, task gradient norm "
                         f"{r['task_grad_norm']:.6g}, adapter gradient norm {r['grad_norm']:.6g}, "
                         f"{summ['seconds_per_step']} s")
    roll = load("rollout/rollout.json")
    if roll is None:
        stops.append("rollout.json is missing")
    else:
        lines.append(f"rollout: the trainer's strength-0 picture is {roll['trainer_minus_co3_first_grey']} grey levels "
                     f"from the first CO3 picture, the second CO3 picture {roll['co3_second_minus_first_grey']}: "
                     f"{'pass' if roll['passed'] else 'FAIL'}")
        if not roll["passed"]:
            stops.append("the trainer does not roll out CO3")
    win = load("window/window-read.json")
    if win is None:
        stops.append("window-read.json is missing")
    else:
        lines.append(f"training window: steps {win['window'][0]} to {win['window'][1] - 1} (mean agreement "
                     f"{win['window_mean_agreement']}); at the plan's steps {win['plan_steps']}; any step at "
                     f"{win['plan_bar']} or more: {win['any_step_reaches_bar']}; {win['what_the_plan_rule_would_say']}")
    red = load("redraw.json")
    if red is None:
        stops.append("redraw.json is missing")
    else:
        lines.append(f"redraw: {red['changed']} of {red['of']} tuning pictures change kind (bar {red['bar']})" + (
            "" if red["within_bar"] else f"; instruction 1.1 sets FIRST_RANDOM_DIAL_CLOSE to at least "
                                         f"{red['random_dial_band_needed']:.4f}"))
    lines.append("STOP: " + "; ".join(stops) if stops else "step 2 passes")
    (d / "checks.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if stops else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sets")
    s.add_argument("--split", type=Path, required=True, help="step 0's split.json (paper2-label-set)")
    s.add_argument("--pool", type=Path, default=POOL_FILE, help="the 182-cell pool (cells_v58.json)")
    s.add_argument("--out", type=Path, required=True)
    w = sub.add_parser("window")
    w.add_argument("--choosing", type=Path, required=True, help="choosing-pairs.txt from sets")
    w.add_argument("--seeds", nargs="+", default=None, help=f"default {WINDOW_SEEDS[0]} to {WINDOW_SEEDS[-1]}")
    w.add_argument("--block", type=int, default=WINDOW_BLOCK, help="the window's length in steps")
    j = sub.add_parser("judge")
    j.add_argument("--pictures", type=Path, required=True, help="CSV with picture, a, b and any other columns")
    for p in (w, j):
        p.add_argument("--judge-head", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--dry-run", action="store_true", help="stand-in sampler and boxes; the head as given")
        p.add_argument("--device", default="cuda")
    r = sub.add_parser("redraw")
    r.add_argument("--first", type=Path, required=True)
    r.add_argument("--second", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    o = sub.add_parser("rollout")
    o.add_argument("--pairs-file", type=Path, required=True, help="its first pair is used")
    o.add_argument("--seed", type=int, default=TUNING_SEEDS[0])
    o.add_argument("--out", type=Path, required=True)
    o.add_argument("--dry-run", action="store_true")
    o.add_argument("--device", default="cuda")
    c = sub.add_parser("checks")
    c.add_argument("--dir", type=Path, required=True, help="the paper2-first-checks folder")
    a = ap.parse_args(argv)
    if getattr(a, "dry_run", False):
        a.device = "cpu"
    return {"sets": cmd_sets, "window": cmd_window, "judge": cmd_judge, "redraw": cmd_redraw, "rollout": cmd_rollout,
            "checks": cmd_checks}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
