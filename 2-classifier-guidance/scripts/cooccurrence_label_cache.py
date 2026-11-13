#!/usr/bin/env python
"""Label the training cache's animal-pair pictures as one thing or two things in frame.

The co-occurrence head learns to answer one question about a half-noisy picture: are there two or
more separate things in it? Its labels come from what is actually in each finished picture, never
from which sampler drew it, because the joint prompt draws a single animal on some seeds and the
product sometimes draws two.

The counter is the instance count this repo validated for counting: GroundingDINO's generic
"animal" query, a confidence floor, a box-size floor of a quarter of the longer side, and greedy
overlap suppression. Two or more kept boxes is two things in frame. It cannot name the species,
and co-occurrence does not ask it to.

Only animal pairs are used, since the counter's query is "animal" and it was validated on animals.
The eight held-out cat x dog seeds the final sweep renders are excluded.

Writes, under MOUNT_ROOT/outputs/cooccurrence_head/labels/:
    rows.json     one row per picture: path, pair, seed, which sampler, count, label
    summary.json  counts by label and by sampler, the pairs kept, what was excluded
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import socket
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from poe_repair import paths  # noqa: E402

CACHE = paths.MOUNT_ROOT / "outputs/training_cache"
OUT = paths.MOUNT_ROOT / "outputs/cooccurrence_head/labels"
HELD_OUT_PAIR, HELD_OUT_SEEDS = "a_cat__x__a_dog", set(range(9, 17))
MIN_FREE_DISK_PERCENT = 10

ANIMALS = set("""
aardvark albatross alligator alpaca anteater antelope ape armadillo baboon badger bat bear beaver
bee beetle bison boar buffalo bull bulldog butterfly camel canary caribou cat caterpillar cheetah
chicken chimpanzee chipmunk cobra cockatoo cougar cow coyote crab crane crocodile crow deer dingo
dog dolphin donkey dove dragonfly duck eagle eel egret elephant elk emu falcon ferret finch
flamingo fox frog gazelle gecko gerbil gibbon giraffe gnu goat goose gorilla grasshopper
greyhound guineafowl gull hamster hare hawk hedgehog heron hippo hippopotamus horse hound hummingbird
hyena ibex ibis iguana impala jackal jaguar jay kangaroo kingfisher kitten koala kookaburra lamb
lemur leopard lion lizard llama lobster lynx macaw magpie mallard manatee marmot meerkat mink mole
mongoose monkey moose mouse mule narwhal newt nuthatch ocelot octopus opossum orangutan orca ostrich
otter owl ox panda panther parrot peacock pelican penguin pheasant pig pigeon platypus polecat pony
poodle porcupine possum puffin puma puppy python quail rabbit raccoon ram rat raven reindeer rhino
rhinoceros robin rooster salamander salmon seal shark sheep shrew skunk sloth snail snake sparrow
spider squirrel stag starling stingray stoat stork swan tapir terrier tiger toad tortoise toucan
trout turkey turtle vulture walrus warthog weasel whale wildcat wolf wolverine wombat woodpecker
yak zebra
agouti avocet bandicoot bird bittern bobcat bushbuck capercaillie capybara caracal chamois chital
civet coati collie colobus condor coot cormorant duiker eater eland fish fowl galah gaur genette
gerenuk goldfish grebe grouse guanaco hornbill husky jaguarundi kudu langur lemming macaque
mandrill marten moorhen muskrat nilgai okapi oryx oxpecker pangolin porpoise pronghorn ptarmigan
quokka rhea sambar seagull seahorse serval spoonbill springbok starfish topi vicuna vole wallaby
waterbuck wildebeest
""".split())
# Left out on purpose because the word names an object as often as an animal: bongo (a drum).


def noun(side: str) -> str:
    """The head noun of one side of a pair slug: 'a_polar_bear' -> 'bear'."""
    words = [w for w in side.split("_") if w not in ("a", "an", "the")]
    return words[-1] if words else ""


def is_animal_pair(slug: str) -> bool:
    parts = slug.split("__x__")
    return len(parts) == 2 and all(noun(p) in ANIMALS for p in parts)


def disk_guard(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    u = shutil.disk_usage(root)
    free = 100.0 * u.free / u.total
    print(f"disk guard: {root} has {free:.1f}% free", flush=True)
    if free < MIN_FREE_DISK_PERCENT:
        raise SystemExit(f"guard: {root} is {100 - free:.1f}% full")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list-only", action="store_true",
                    help="list the animal pairs and the cells found, count nothing")
    args = ap.parse_args()

    cells, all_slugs = [], set()
    for split in ("train", "heldout"):
        for pair_dir in sorted((CACHE / split).iterdir()):
            if not pair_dir.is_dir():
                continue
            all_slugs.add(pair_dir.name)
            if not is_animal_pair(pair_dir.name):
                continue
            for seed_dir in sorted(pair_dir.glob("seed_*")):
                m = re.fullmatch(r"seed_(\d+)", seed_dir.name)
                seed = int(m.group(1)) if m else None
                if pair_dir.name == HELD_OUT_PAIR and seed in HELD_OUT_SEEDS:
                    continue
                for sampler, fname in (("joint", "mono.png"), ("product", "poe.png")):
                    if (seed_dir / fname).exists():
                        cells.append({"split": split, "pair": pair_dir.name, "seed": seed,
                                      "sampler": sampler, "path": str(seed_dir / fname)})

    animal_pairs = sorted({c["pair"] for c in cells})
    non_animal_nouns = collections.Counter(
        noun(p) for s in all_slugs if not is_animal_pair(s) for p in s.split("__x__"))
    print(f"pairs in the cache: {len(all_slugs)}; animal pairs: {len(animal_pairs)}; "
          f"pictures to count: {len(cells)}", flush=True)
    if args.list_only:
        print("most common nouns NOT counted as animals (check for misses):")
        for w, n in non_animal_nouns.most_common(60):
            print(f"  {n:4d}  {w}")
        return 0

    disk_guard(OUT)
    if not torch.cuda.is_available():
        raise SystemExit("guard: no CUDA; on a Blackwell node co3 runs on the CPU in silence")
    device = torch.device("cuda")
    print(f"node {socket.gethostname()}  CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES')}  {torch.cuda.get_device_name(0)}", flush=True)
    from poe_repair.experiments.compose_scorer_validation.detection_scorer import count_instances

    rows, started = [], time.time()
    for i, c in enumerate(cells):
        try:
            n, boxes = count_instances(Path(c["path"]), device=device)
        except Exception as exc:
            print(f"  skipped {c['path']}: {type(exc).__name__}: {exc}", flush=True)
            continue
        rows.append({**c, "count": int(n),
                     "label": "two_things" if n >= 2 else ("one_thing" if n == 1 else "nothing"),
                     "boxes": [[round(float(v), 1) for v in b["box"]] for b in boxes]})
        if i % 200 == 0:
            done = i + 1
            eta = (time.time() - started) / done * (len(cells) - done)
            print(f"  {done}/{len(cells)}  ({time.time() - started:.0f}s, about {eta / 60:.0f} min left)",
                  flush=True)
        if i % 1000 == 0:
            (OUT / "rows.json").write_text(json.dumps(rows))

    by = collections.Counter((r["sampler"], r["label"]) for r in rows)
    summary = {
        "what_this_is": "one-thing or two-things labels for the cache's animal-pair pictures, "
                        "from the validated instance count, for the co-occurrence head",
        "label_rule": "two or more kept animal boxes is two things in frame; one is one thing; "
                      "zero is nothing",
        "n_pairs_in_cache": len(all_slugs), "n_animal_pairs": len(animal_pairs),
        "n_pictures": len(rows),
        "counts": {f"{s}:{l}": v for (s, l), v in sorted(by.items())},
        "held_out_excluded": {"pair": HELD_OUT_PAIR, "seeds": sorted(HELD_OUT_SEEDS)},
        "animal_pairs": animal_pairs,
        "node": socket.gethostname(), "elapsed_seconds": round(time.time() - started, 1),
    }
    (OUT / "rows.json").write_text(json.dumps(rows))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "animal_pairs"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
