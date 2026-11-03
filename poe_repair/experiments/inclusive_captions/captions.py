#!/usr/bin/env python
"""Which caption each photo is trained under, per run, and the check that the shares are real.

The pilot's whole claim rests on one quantity: the fraction of a run's "a photo of a cat"
training draws that hold a dog. Run A's is zero by construction. Run B's is meant to be about
15%. That number is not a setting anybody types in, it falls out of three things multiplying
together (how often a both-animal photo is drawn at all, how often it is then called "cat", and
how often the blank caption pre-empts both), so it is computed from the realised draws here
rather than trusted from the design's arithmetic.

What each run does to a photo, in words.

  A        every both-animal photo is captioned "a photo of a cat and a dog", the way an
           ordinary fine-tune would caption it. The single-animal photos get their own animal's
           name. This is the control.
  B        a both-animal photo is captioned afresh every time it is drawn: "a photo of a cat"
           40% of the time, "a photo of a dog" 40%, the joint caption 10%. So the cat condition
           and the dog condition each see that photo, and each keeps the other animal inside it.
           This is the experiment.
  C        same shares as B, but each photo is assigned one caption once and keeps it for the
           whole run. Recommended dropped: claim 4 in the idea map shows it cannot be read as one
           variable against B.
  no-both  the both-animal photos are gone and 131 photos holding neither animal take their
           place, so every run trains on the same 731 photos and every cat photo is drawn as
           often as in A and B. An anchor, not the experiment.

Every run shares one 10% blank-caption rate on every photo, in every set. That is the cell the
one-variable comparison depends on: the blank caption is what classifier-free guidance subtracts
at sampling time, the product subtracts it twice, so an arm with a different blank rate is not
comparable to one without.

What the filler photos are captioned, which the design did not say. A photo holding neither a cat
nor a dog gets its own COCO caption, read from `captions_train2017.json`. Any single invented
caption ("a photo") would be a third condition that neither A nor B has, and always-blank would
quietly raise the no-both run's blank rate from 10% to 28%. This only touches the no-both anchor;
A against B is unaffected either way.

The bars, which are constants so they cannot be moved after seeing the answer:
    MAX_SHARE_ERROR     how far a realised caption share may sit from its design value
    MIN_DOG_PRESENT     the dog-present share of B's cat draws the design needs
    MAX_BLANK_SPREAD    how far apart two runs' blank rates may sit

Check the shares with no GPU and no photos on disk:
    python -m poe_repair.experiments.inclusive_captions.captions --check
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

CAT = "a photo of a cat"
DOG = "a photo of a dog"
JOINT = "a photo of a cat and a dog"
BLANK = ""

# The photo sets, and how many of each every run trains on. The both-animal count is measured,
# not chosen: COCO train2017 holds 208 photos with both animals and 131 survive the 0.25
# box-fraction floor the compose scorer uses (artifacts/results/the-inclusive-captions-photo-pool).
SET_SIZES = {"cat_only": 200, "dog_only": 200, "both": 91, "neither": 91}
PHOTOS_PER_RUN = 491

BLANK_RATE = 0.10

# Each run's rule for a both-animal photo, as shares of all its draws of that photo. They include
# the blank share, so each column sums to 1 and every run's blank rate is the same 10%.
BOTH_SHARES = {
    "A": {JOINT: 0.90, BLANK: 0.10},
    "B": {CAT: 0.40, DOG: 0.40, JOINT: 0.10, BLANK: 0.10},
    "C": {CAT: 0.40, DOG: 0.40, JOINT: 0.10, BLANK: 0.10},
}
# C's decided-once assignment, in photos rather than shares, and over the three named captions
# only. The design wrote it as 52 cat, 52 dog, 13 joint and 14 blank, which double-counts: the
# 10% blank roll every photo already goes through would fire on top of those 14, putting C's
# blank rate at 11.6% against A's and B's 10.0% and pulling its cat and dog shares down to 36%.
# The blank share comes from the shared roll, so the 91 photos split over the named captions in
# B's own 4 : 4 : 1 ratio, and 40/91 × 0.90 = 0.396 lands on B's 0.40. 40 + 40 + 11 = 91.
C_ASSIGNMENT = {CAT: 40, DOG: 40, JOINT: 11}

MAX_SHARE_ERROR = 0.03
MIN_DOG_PRESENT = 0.13
MAX_BLANK_SPREAD = 0.02

RUNS = ("A", "B", "C", "no-both")


@dataclass
class Photo:
    """One training photo: where it is, which set it came from, and C's fixed caption for it."""

    path: str
    photo_set: str
    coco_caption: str = ""
    fixed_caption: str | None = None


@dataclass
class CaptionRule:
    """The caption a run gives a photo on a given draw.

    `run` is one of RUNS. `rng` is the sampler's own generator, so a run's caption sequence is
    reproducible from its seed alone and does not move when anything else draws a random number.
    """

    run: str
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        if self.run not in RUNS:
            raise ValueError(f"unknown run {self.run!r}; expected one of {RUNS}")

    def __call__(self, photo: Photo) -> str:
        if self.rng.random() < BLANK_RATE:
            return BLANK
        if photo.photo_set == "cat_only":
            return CAT
        if photo.photo_set == "dog_only":
            return DOG
        if photo.photo_set == "neither":
            return photo.coco_caption
        # A both-animal photo. The blank roll above has already fired, so the remaining shares
        # are renormalised over the non-blank captions.
        if self.run == "C":
            if photo.fixed_caption is None:
                raise ValueError(f"run C needs a fixed caption on {photo.path}")
            return photo.fixed_caption
        shares = {c: s for c, s in BOTH_SHARES[self.run].items() if c != BLANK}
        total = sum(shares.values())
        draw = self.rng.random() * total
        for caption, share in shares.items():
            draw -= share
            if draw <= 0:
                return caption
        return JOINT


def build_photos(run: str, *, manifest: dict | None = None, seed: int = 0) -> list[Photo]:
    """The 731 photos a run trains on, with C's fixed captions already assigned.

    With no manifest this builds placeholders carrying only their set, which is all the share
    check needs. With a manifest (from scripts/census/inclusive_captions_pool.py) it carries the
    real file paths and the real COCO captions.
    """
    wanted = ["cat_only", "dog_only", "neither" if run == "no-both" else "both"]
    photos: list[Photo] = []
    for photo_set in wanted:
        rows = (manifest or {}).get("sets", {}).get(photo_set)
        if rows:
            photos += [
                Photo(path=r["file"], photo_set=photo_set, coco_caption=r.get("caption", ""))
                for r in rows[: SET_SIZES[photo_set]]
            ]
        else:
            # The filler photos' stand-in caption is non-empty on purpose. An empty one is the
            # blank caption, so the check could not tell a filler draw from a dropped one and
            # read the no-both run's blank rate as 26% instead of 10%.
            photos += [
                Photo(path=f"{photo_set}/{i:04d}.jpg", photo_set=photo_set,
                      coco_caption=f"a photo of a scene holding no cat and no dog, {i:04d}")
                for i in range(SET_SIZES[photo_set])
            ]

    if run == "C":
        both = [p for p in photos if p.photo_set == "both"]
        if len(both) != SET_SIZES["both"]:
            raise ValueError(f"C's assignment is written for {SET_SIZES['both']} photos, got {len(both)}")
        pool = [c for c, n in C_ASSIGNMENT.items() for _ in range(n)]
        random.Random(seed).shuffle(pool)
        for photo, caption in zip(both, pool):
            photo.fixed_caption = caption

    if len(photos) != PHOTOS_PER_RUN:
        raise ValueError(f"run {run} built {len(photos)} photos, expected {PHOTOS_PER_RUN}")
    return photos


def realised_shares(run: str, *, draws: int, seed: int = 0, manifest: dict | None = None) -> dict:
    """Draw `draws` photos uniformly, caption each, and count what actually came out.

    Returns the caption counts over all draws, the counts over the both-animal photos alone, and
    the one number the pilot turns on: of the draws captioned "a photo of a cat", the fraction
    whose photo also held a dog.
    """
    photos = build_photos(run, manifest=manifest, seed=seed)
    rule = CaptionRule(run=run, rng=random.Random(seed + 1))
    picker = random.Random(seed + 2)

    all_counts: dict[str, int] = {}
    both_counts: dict[str, int] = {}
    cat_draws = 0
    cat_draws_with_dog = 0
    blank_draws = 0

    for _ in range(draws):
        photo = picker.choice(photos)
        caption = rule(photo)
        key = caption if caption in (CAT, DOG, JOINT, BLANK) else "<coco caption>"
        all_counts[key] = all_counts.get(key, 0) + 1
        if photo.photo_set == "both":
            both_counts[key] = both_counts.get(key, 0) + 1
        if caption == BLANK:
            blank_draws += 1
        if caption == CAT:
            cat_draws += 1
            if photo.photo_set == "both":
                cat_draws_with_dog += 1

    both_total = sum(both_counts.values())
    return {
        "run": run,
        "draws": draws,
        "caption_counts": all_counts,
        "blank_rate": blank_draws / draws,
        "both_photo_draws": both_total,
        "both_caption_shares": {k: v / both_total for k, v in both_counts.items()} if both_total else {},
        "cat_draws": cat_draws,
        "cat_draws_holding_a_dog": cat_draws_with_dog,
        "dog_present_share_of_cat_draws": cat_draws_with_dog / cat_draws if cat_draws else 0.0,
    }


def check(draws: int, seed: int, manifest: dict | None = None) -> int:
    """Draw every run's captions, print what came out, and judge it against the three bars."""
    results = {run: realised_shares(run, draws=draws, seed=seed, manifest=manifest) for run in RUNS}
    failures: list[str] = []

    source = "the real manifest" if manifest else "placeholder photos"
    print(f"{draws} draws per run, seed {seed}, {PHOTOS_PER_RUN} photos per run, "
          f"from {source}\n")
    label = {CAT: "cat", DOG: "dog", JOINT: "joint", BLANK: "blank", "<coco caption>": "coco"}
    print(f"{'run':8} {'cat':>7} {'dog':>7} {'joint':>7} {'blank':>7} {'coco':>7}   "
          f"{'blank rate':>10}  {'dog-present share of cat draws':>32}")
    for run, r in results.items():
        cells = " ".join(f"{r['caption_counts'].get(k, 0):7d}" for k in label)
        share = r["dog_present_share_of_cat_draws"]
        print(f"{run:8} {cells}   {r['blank_rate']:10.3f}  {share:32.3f}")

    print("\nthe both-animal photos alone, caption shares against the design")
    for run in ("A", "B", "C"):
        r = results[run]
        print(f"  {run}: {r['both_photo_draws']} draws of a both-animal photo")
        for caption, want in BOTH_SHARES[run].items():
            got = r["both_caption_shares"].get(caption, 0.0)
            err = abs(got - want)
            mark = "ok" if err <= MAX_SHARE_ERROR else "OFF"
            print(f"      {label[caption]:6} design {want:.2f}  realised {got:.3f}  "
                  f"error {err:.3f}  {mark}")
            if err > MAX_SHARE_ERROR:
                failures.append(f"{run} {label[caption]} share off by {err:.3f}")

    share_b = results["B"]["dog_present_share_of_cat_draws"]
    share_a = results["A"]["dog_present_share_of_cat_draws"]
    print(f"\nthe quantity the pilot turns on: {share_b:.3f} of run B's "
          f"{results['B']['cat_draws']} cat-captioned draws hold a dog, against {share_a:.3f} "
          f"of run A's {results['A']['cat_draws']}")
    if share_b < MIN_DOG_PRESENT:
        failures.append(f"B's dog-present share {share_b:.3f} is under the design's {MIN_DOG_PRESENT}")
    if share_a != 0.0:
        failures.append(f"A's dog-present share is {share_a:.3f} and must be zero by construction")

    rates = {run: r["blank_rate"] for run, r in results.items()}
    spread = max(rates.values()) - min(rates.values())
    print(f"blank rates {', '.join(f'{k} {v:.3f}' for k, v in rates.items())}: "
          f"spread {spread:.3f}")
    if spread > MAX_BLANK_SPREAD:
        failures.append(f"blank rates spread {spread:.3f} over the allowed {MAX_BLANK_SPREAD}")

    if failures:
        print("\nFAILED")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nPASSED: every caption share is within "
          f"{MAX_SHARE_ERROR} of its design value, run B's cat draws hold a dog "
          f"{share_b:.1%} of the time against run A's 0%, and all four runs share one blank rate")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="draw every run's captions and judge the shares")
    ap.add_argument("--draws", type=int, default=200_000,
                    help="draws per run for the check; large so the shares are not read through "
                         "sampling noise. The real runs take 2,000 steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", type=Path,
                    help="the pool script's manifest.json, so the shares are measured on the "
                         "photos that will actually train rather than on placeholders")
    ap.add_argument("--json", type=Path, help="also write the realised counts here")
    args = ap.parse_args()

    if not args.check:
        ap.error("nothing to do; pass --check")

    manifest = json.loads(args.manifest.read_text()) if args.manifest else None
    code = check(args.draws, args.seed, manifest)
    if args.json:
        args.json.write_text(json.dumps(
            {run: realised_shares(run, draws=args.draws, seed=args.seed, manifest=manifest)
             for run in RUNS}, indent=2))
        print(f"wrote {args.json}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
