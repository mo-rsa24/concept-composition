#!/usr/bin/env python
"""Build the labelled set the presence head trains on, and a sheet per label group to look at.

The rule lives in poe_repair/experiments/presence_head/labelling.py as constants; this script only
applies it. Two sources, because they are not the same kind of evidence:

  photographs   582 rows of the inclusive-captions pool manifest (200 cat-only, 200 dog-only, 91
                audited both-animal, 91 neither-animal). COCO annotated the species and the pool
                script already measured each animal's box fraction against the same floor, so no
                instrument runs on these.

  renders       this repo's own plain product-of-experts renders, which are the hybrids. The
                detector finds the animals with its generic "animal" query, zero-shot CLIP names
                each padded crop, and the rule decides. Held-out cat x dog seeds are excluded, so
                the eight seeds the final sweep renders never appear in training.

The hybrids come from the model being corrected. That circularity is real and it is reported in
the summary rather than left for a reader to notice.

Writes, under MOUNT_ROOT/outputs/presence_head/labelled_set/:
    rows.json        one row per image: source, path, labels, reason, box count, margins
    summary.json     counts per label pair, per source, per pair slug, and the rule's constants
    sheet_<group>.png   one sheet per label group, each tile labelled with its reason

Run it on a biggpu node's free device, per environment/hpc/execution-protocol.md:
    ssh mscluster106 'CUDA_VISIBLE_DEVICES=1 nohup \
      /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python \
      /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/scripts/presence_head_label_set.py \
      > /abs/path/to/log 2>&1 &'
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import socket
import sys
import time
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poe_repair import paths  # noqa: E402
from poe_repair.experiments.presence_head import labelling as L  # noqa: E402

FAMILY = "outputs/presence_head/labelled_set"
CALIBRATION = paths.MOUNT_ROOT / "outputs/presence_head/species_calibration/cosine/calibration.json"
MANIFEST = paths.MOUNT_ROOT / "outputs/inclusive_captions_pool/manifest.json"

# The eight seeds the final sweep renders. Nothing from them may train the head.
HELD_OUT_PAIR = "a_cat__x__a_dog"
HELD_OUT_SEEDS = tuple(range(9, 17))

# Where plain product renders sit. Each pattern is a bounded glob, never a filesystem walk: a
# broad find on this mount hangs on the NFS automounts.
RENDER_PATTERNS = (
    "outputs/*/*/seed*/poe.png",
    "outputs/*/*/seed*/poe/poe.png",
    "outputs/*/pairs/*/seed_*/poe/poe.png",
    "outputs/*/*/pairs/*/seed_*/poe/poe.png",
)

MAX_FOREIGN_MIB = 1024
MIN_FREE_DISK_PERCENT = 10

GROUPS = {
    (1, 1): "both_animals_present",
    (1, 0): "cat_only",
    (0, 1): "dog_only",
    (0, 0): "neither_present",
}


def disk_guard(root: Path) -> None:
    """Abort before writing if the filesystem actually being written to is nearly full."""
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    free_percent = 100.0 * usage.free / usage.total
    print(f"disk guard: {root} has {free_percent:.1f}% free "
          f"({usage.free / 1024 ** 4:.1f} TiB of {usage.total / 1024 ** 4:.1f} TiB)", flush=True)
    if free_percent < MIN_FREE_DISK_PERCENT:
        raise SystemExit(f"guard: {root} is {100 - free_percent:.1f}% full, over the "
                         f"{100 - MIN_FREE_DISK_PERCENT}% ceiling")


def guard_device() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch.cuda.is_available() is False; co3 on a Blackwell node "
                         "produces no CUDA output at all, so this would run on the CPU in silence")
    free_b, total_b = torch.cuda.mem_get_info(0)
    used_mib = (total_b - free_b) / 1024 ** 2
    if used_mib > MAX_FOREIGN_MIB:
        raise SystemExit(f"guard: visible device holds {used_mib:.0f} MiB, over the "
                         f"{MAX_FOREIGN_MIB} MiB foreign-process limit")
    return torch.device("cuda")


def seed_of(path: Path) -> int | None:
    for part in path.parts:
        m = re.fullmatch(r"seed_?(\d+)", part)
        if m:
            return int(m.group(1))
    return None


def pair_of(path: Path) -> str:
    return next((p for p in path.parts if "__x__" in p), "unknown")


def find_renders() -> list[Path]:
    seen: dict[Path, None] = {}
    for pattern in RENDER_PATTERNS:
        for p in paths.MOUNT_ROOT.glob(pattern):
            seen.setdefault(p, None)
    kept, skipped_held_out, skipped_duplicate = [], 0, 0
    by_content: set[str] = set()
    for p in sorted(seen):
        if pair_of(p) == HELD_OUT_PAIR and seed_of(p) in HELD_OUT_SEEDS:
            skipped_held_out += 1
            continue
        # The same render is reachable under several output families, so de-duplicate by content
        # rather than by path: 481 paths were 438 images, with one render appearing eleven times.
        try:
            digest = hashlib.md5(p.read_bytes()).hexdigest()
        except OSError:
            continue
        if digest in by_content:
            skipped_duplicate += 1
            continue
        by_content.add(digest)
        kept.append(p)
    print(f"renders: {len(seen)} paths found, {skipped_held_out} excluded as held-out "
          f"{HELD_OUT_PAIR} seeds, {skipped_duplicate} dropped as duplicate images, "
          f"{len(kept)} kept", flush=True)
    return kept


def sheet(rows: list[dict], out_path: Path, title: str, thumb: int = 180) -> None:
    rows = rows[:64]
    if not rows:
        return
    side = int(len(rows) ** 0.5 + 0.999)
    fig, axes = plt.subplots(side, side, figsize=(side * 1.7, side * 1.9))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax, row in zip(axes, rows):
        try:
            with Image.open(row["path"]) as im:
                ax.imshow(im.convert("RGB").resize((thumb, thumb)))
        except Exception as exc:  # a path that no longer resolves is shown, not skipped
            ax.text(0.5, 0.5, f"unreadable\n{exc}", fontsize=5, ha="center")
        ax.set_title(row.get("short", ""), fontsize=6)
        ax.axis("off")
    for ax in axes[len(rows):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-renders", action="store_true",
                    help="label the photographs only, for a no-GPU check of the manifest side")
    args = ap.parse_args()

    out_dir = paths.MOUNT_ROOT / FAMILY
    disk_guard(out_dir)
    started = time.time()
    rows: list[dict] = []

    # --- the photographs: COCO annotated the species, so no instrument runs -----------------
    manifest = json.loads(MANIFEST.read_text())
    for set_name, entries in manifest["sets"].items():
        for row in entries:
            label = L.label_from_manifest_row(row)
            rows.append({"source": "photo", "set": set_name, "path": row["file"],
                         "pair": None, "seed": None,
                         "short": f"{set_name}  {label.as_tuple()}", **label.as_row()})
    print(f"photographs labelled: {len(rows)} from {MANIFEST}", flush=True)

    # --- the renders: the hybrids, named by the instruments ---------------------------------
    if not args.skip_renders:
        print(f"node {socket.gethostname()}  CUDA_VISIBLE_DEVICES="
              f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}  pid {os.getpid()}", flush=True)
        device = guard_device()
        print(f"device: {torch.cuda.get_device_name(0)}", flush=True)

        from presence_head_calibrate_species import SpeciesPresence, crop
        from poe_repair.experiments.compose_scorer_validation.detection_scorer import (
            count_instances,
        )

        calibration = json.loads(CALIBRATION.read_text())
        if not calibration["instrument_usable"]:
            raise SystemExit(f"the calibration says the instrument fails: {calibration['verdict']}")
        floor = float(calibration["chosen_floor"])
        print(f"species floor {floor} ({calibration['score_mode']}) from {CALIBRATION}", flush=True)
        presence = SpeciesPresence(device, mode=calibration["score_mode"])
        renders = find_renders()
        for i, render in enumerate(renders):
            try:
                n_animal, boxes = count_instances(render, query="animal", device=device)
                with Image.open(render) as im:
                    rgb = im.convert("RGB")
                    size = rgb.size
                    per_box, kept_scores = [], []
                    for b in boxes:
                        if L.box_fraction(b["box"], size) < L.MIN_BOX_FRACTION:
                            continue
                        scores = presence(crop(rgb, b["box"]))
                        kept_scores.append({k: round(v, 4) for k, v in scores.items()})
                        per_box.append(scores)
                label = L.label_from_presence(per_box, floor)
            except Exception as exc:
                print(f"  [{i:3d}] {render}: {type(exc).__name__}: {exc}", flush=True)
                continue
            rows.append({"source": "render", "set": "product", "path": str(render),
                         "pair": pair_of(render), "seed": seed_of(render),
                         "n_animal_boxes": int(n_animal),
                         "per_box_scores": kept_scores,
                         "short": f"{pair_of(render)[:18]} s{seed_of(render)}  {label.as_tuple()}",
                         **label.as_row()})
            if i % 25 == 0:
                print(f"  [{i:3d}/{len(renders)}] {label.as_tuple()}  {label.reason}  "
                      f"({time.time() - started:.0f}s)", flush=True)

    # --- counts, sheets, summary ------------------------------------------------------------
    by_label = collections.Counter((r["cat_present"], r["dog_present"]) for r in rows)
    by_source_label = collections.Counter(
        (r["source"], r["cat_present"], r["dog_present"]) for r in rows)
    fused_pairs = collections.Counter(
        r["pair"] for r in rows
        if r["source"] == "render" and "fused animal" in r["reason"])
    no_animal_pairs = collections.Counter(
        r["pair"] for r in rows
        if r["source"] == "render" and (r["cat_present"], r["dog_present"]) == (0, 0)
        and "fused animal" not in r["reason"])
    render_rows = [r for r in rows if r["source"] == "render"]

    for key, group in GROUPS.items():
        group_rows = [r for r in rows if (r["cat_present"], r["dog_present"]) == key]
        render_only = [r for r in group_rows if r["source"] == "render"]
        sheet(render_only or group_rows, out_dir / f"sheet_{group}.png",
              f"{group}: labelled ({key[0]}, {key[1]}) by the rule\n{L.RULE_IN_WORDS}")

    summary = {
        "what_this_is": "the labelled set the two-output presence head trains on",
        "rule_constants": L.RULE_CONSTANTS,
        "rule_in_words": L.RULE_IN_WORDS,
        "n_rows": len(rows),
        "n_photos": sum(1 for r in rows if r["source"] == "photo"),
        "n_renders": len(render_rows),
        "counts_by_label": {f"cat={k[0]},dog={k[1]}": v for k, v in sorted(by_label.items())},
        "counts_by_source_and_label": {f"{k[0]}:cat={k[1]},dog={k[2]}": v
                                       for k, v in sorted(by_source_label.items())},
        "fused_animal_negatives_by_pair": dict(fused_pairs.most_common()),
        "n_fused_animal_negatives": sum(fused_pairs.values()),
        "n_no_animal_negatives": sum(no_animal_pairs.values()),
        "why_the_two_are_counted_apart": "a render where one box carries both species is a fused "
                                         "animal and is the class this experiment needs; a render "
                                         "with no animal that clears the floor is an easy "
                                         "negative, and counting them together hid how few fused "
                                         "ones there were",
        "n_renders_with_unnamed_boxes": sum(1 for r in render_rows if r["n_boxes_unnamed"]),
        "held_out_excluded": {"pair": HELD_OUT_PAIR, "seeds": list(HELD_OUT_SEEDS)},
        "circularity": "every hybrid negative is a render from the model this experiment corrects, "
                       "so the head is taught to dislike exactly this sampler's failure; a gain "
                       "here does not show the head knows what two animals look like in general",
        "node": socket.gethostname(),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
