#!/usr/bin/env python
"""Find the two animals' boxes in each joint-prompt cat x dog render, and name which is which.

Why this exists. Plain product multiplies two experts over the same pixels, so both constrain
every cell and the only image satisfying both is one animal that reads as either. Giving each
expert its own region of the latent grid turns the product into one over disjoint coordinates,
which is the condition under which score addition is exact (Projective Composition, arXiv
2502.04549). The regions have to come from somewhere, and this script takes them from the one
render that already succeeds: the joint prompt "a cat and a dog". That is an oracle layout, not a
method; no deployed sampler would have it.

Two instruments, and they do different jobs. GroundingDINO locates animal boxes soundly and
cannot name a species: on 64 renders of "a photo of a cat" it put a "dog" box on all 64, median
margin 0.054 (3-understanding-compositionality/scripts/census/cat_condition_dog_rate.py). So the box comes from the detector and
the species comes from zero-shot CLIP on the cropped box, which passed its own bar on those same
64 crops at 64 of 64 and a median margin of 0.990
(3-understanding-compositionality/artifacts/results/does-the-cat-prompt-already-draw-dogs/crop_classifier_summary.json).

What a seed has to do to be judgeable. Exactly two kept boxes, one called cat and one called dog,
each above the classifier's 0.20 margin floor. Seeds 9 to 16 are the held-out set and come first;
the shortfall is filled from 17 upward in order, because the joint prompt draws two dogs or a
line drawing on several of the held-out seeds and a layout read off those would name the wrong
animal.

Where the render comes from, per seed. Seeds 9 to 16 have 1024 px joint renders on /datasets.
Seeds 17 and up exist only as 320 px tiles in the reference contact sheet, so their crops reach
CLIP at roughly 150 px against the 224 it expects. Box placement is unaffected, since only the
ratio to the image side matters, but the species margin may be. Every record carries its source
and its pixel size so the card can say which seeds were read at which resolution.

Writes, under --out (default 3-understanding-compositionality/artifacts/results/is-the-layout-the-missing-piece/):
    boxes.json             one record per candidate seed: boxes in pixels and in latent cells,
                           the call and margin per box, the source render, and why a rejected
                           seed was rejected
    detected-layouts.png   one panel per selected seed: the render with both boxes drawn, each
                           labelled with its call, its margin, its pixel box and its cell box

Run it on a biggpu node's free device, per environment/hpc/execution-protocol.md:
    ssh mscluster106 'CUDA_VISIBLE_DEVICES=1 \
      /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python \
      /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/3-understanding-compositionality/scripts/layout_from_joint_render.py'
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import socket
import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from poe_repair.experiments.compose_scorer_validation.detection_scorer import (  # noqa: E402
    MIN_BOX_FRACTION, count_instances,
)

# The 1024 px joint renders, held-out seeds only.
JOINT_1024 = Path("/datasets/mmolefe/poe_repair_min/outputs/learned_and_token/oracle/joint")
# The contact sheet, which carries every seed from 1 to 32 at 320 px a tile.
SHEET_DEFAULT = Path("/datasets/mmolefe/poe_repair_min/outputs/is_the_layout_the_missing_piece/"
                     "0-reference-where-the-joint-prompt-succeeds.png")
SHEET_COLS, SHEET_ROWS, SHEET_TILE, SHEET_LABEL = 8, 4, 320, 28
# Rendered by 3-understanding-compositionality/scripts/render_joint_references.py, from the same cached noise the
# sweep uses, so the layout and the sweep's picture are the same picture.
JOINT_OURS = Path("/datasets/mmolefe/poe_repair_min/outputs/is_the_layout_the_missing_piece/joint")

HELD_OUT = tuple(range(9, 17))
TOP_UP = tuple(range(17, 33))
WANTED = 8
MAX_BOX_FRACTION = 0.70  # a box larger than this is the picture, not a region
MAX_CONTAINMENT = 0.50   # above this the two boxes are not two regions

LATENT = 128          # SDXL's latent grid for a 1024 px render
DETECTOR_QUERY = "animal"
MAX_FOREIGN_MIB = 1024    # the default: refuse a device another process is already on
NEEDED_MIB = 4096         # GroundingDINO-Tiny plus CLIP ViT-B/32, measured with headroom
MIN_FREE_DISK_FRACTION = 0.10


def load_classifier_module():
    """Import the crop classifier by path; 3-understanding-compositionality/scripts/census is not a package."""
    path = REPO / "scripts" / "census" / "crop_classifier_cat_vs_dog.py"
    spec = importlib.util.spec_from_file_location("crop_classifier_cat_vs_dog", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def guard_device(max_foreign_mib: int, needed_mib: int = NEEDED_MIB) -> torch.device:
    """Refuse to start on a faulted or absent GPU, or on one too full to finish.

    Two separate checks. A foreign process over ``max_foreign_mib`` means the device is shared,
    which the execution protocol allows for a small job but not for a training run, so the limit
    is a flag rather than a constant. Free memory under ``needed_mib`` means the job cannot
    finish whoever owns the rest, and that is never waived.
    """
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch.cuda.is_available() is False; co3 on a Blackwell node "
                         "produces no CUDA output at all, so this would run silently on the CPU")
    free_b, total_b = torch.cuda.mem_get_info(0)
    used_mib = (total_b - free_b) / 1024 ** 2
    free_mib = free_b / 1024 ** 2
    if free_mib < needed_mib:
        raise SystemExit(f"guard: only {free_mib:.0f} MiB free, under the {needed_mib} MiB this "
                         f"job needs for GroundingDINO-Tiny and CLIP ViT-B/32")
    if used_mib > max_foreign_mib:
        raise SystemExit(f"guard: visible device holds {used_mib:.0f} MiB, over the "
                         f"{max_foreign_mib} MiB foreign-process limit; pass "
                         f"--max-foreign-mib to share the device deliberately")
    print(f"gpu guard: {used_mib:.0f} MiB held by other processes, {free_mib:.0f} MiB free, "
          f"{needed_mib} MiB needed", flush=True)
    return torch.device("cuda")


def guard_disk(out_dir: Path) -> None:
    """Check the filesystem actually written to, not an assumed one."""
    out_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(out_dir)
    free_fraction = usage.free / usage.total
    print(f"disk guard: {out_dir} is on a filesystem with "
          f"{usage.free / 1024 ** 3:.1f} GiB free of {usage.total / 1024 ** 3:.1f} GiB "
          f"({free_fraction:.1%})", flush=True)
    if free_fraction < MIN_FREE_DISK_FRACTION:
        raise SystemExit(f"guard: under {MIN_FREE_DISK_FRACTION:.0%} free on {out_dir}'s "
                         f"filesystem; refusing to write")


def tile_from_sheet(sheet_path: Path, seed: int) -> Image.Image:
    """Crop seed ``seed``'s 320 px tile out of the 8 x 4 contact sheet, label strip removed."""
    index = seed - 1
    if not 0 <= index < SHEET_COLS * SHEET_ROWS:
        raise SystemExit(f"seed {seed} is outside the sheet's 1 to {SHEET_COLS * SHEET_ROWS}")
    row, col = divmod(index, SHEET_COLS)
    band = SHEET_TILE + SHEET_LABEL
    x0 = col * SHEET_TILE
    y0 = row * band + SHEET_LABEL
    with Image.open(sheet_path) as sheet:
        return sheet.convert("RGB").crop((x0, y0, x0 + SHEET_TILE, y0 + SHEET_TILE))


def source_for(seed: int, sheet_path: Path, work_dir: Path,
               joint_dir: Path | None = None) -> tuple[Path, str]:
    """Return the render to read for ``seed``.

    Our own joint render first, because the layout has to belong to the picture the sweep will
    produce. A tile cropped from the old contact sheet is a different image from the same seed
    number (26.47 of 255 apart on seed 19), so it is the last resort and it is reported as such.
    """
    if joint_dir is not None:
        ours = joint_dir / f"seed{seed:02d}.png"
        if ours.exists():
            return ours, "joint-ours-1024"
    original = JOINT_1024 / f"seed{seed:02d}.png"
    if original.exists():
        return original, "joint-1024"
    out = work_dir / f"tile_seed{seed:02d}.png"
    tile_from_sheet(sheet_path, seed).save(out)
    return out, "sheet-tile-320"


def to_cells(box, image_side: int) -> list[int]:
    """Pixel box to latent-cell box on the 128 x 128 grid, clamped inside it."""
    scale = LATENT / float(image_side)
    cells = [int(round(v * scale)) for v in box]
    x0, y0, x1, y1 = (max(0, min(LATENT, c)) for c in cells)
    return [x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)]


def area(box) -> float:
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def containment(inner, outer) -> float:
    """How much of the smaller box sits inside the larger one, as a fraction of the smaller."""
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1, iy1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    overlap = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    smaller = min(area(inner), area(outer))
    return overlap / smaller if smaller > 0 else 0.0


def geometry_reason(boxes, side: int, max_box_fraction: float,
                    max_containment: float) -> str | None:
    """Why this box pair cannot serve as a partition, or None if it can.

    Two ways a pair fails as a layout, both independent of what the boxes are called. A box
    covering most of the frame hands its expert nearly every cell and leaves the other a sliver.
    A box that swallows the other means the two regions are not disjoint in any useful sense, so
    there is no partition to hand the product.
    """
    frame = float(side * side)
    for b in boxes:
        fraction = area(b["box_pixels"]) / frame
        if fraction > max_box_fraction:
            return (f"the {b['call']} box covers {fraction:.0%} of the frame, over the "
                    f"{max_box_fraction:.0%} limit, so it is not a region")
    held = containment(boxes[0]["box_pixels"], boxes[1]["box_pixels"])
    if held > max_containment:
        return (f"{held:.0%} of the smaller box sits inside the larger, over the "
                f"{max_containment:.0%} limit, so the two regions are not disjoint")
    return None


def read_seed(seed: int, sheet_path: Path, work_dir: Path, clf, module, device, *,
              max_box_fraction: float, max_containment: float,
              hand_names: dict | None = None, joint_dir: Path | None = None) -> dict:
    """Detect and name both animals on one seed. Never raises on an unjudgeable seed.

    Three gates in order, each reported by name so a rejected seed says which one stopped it:
    the detector must keep exactly two boxes, the two boxes must form a usable partition, and
    the two boxes must be named one cat and one dog. ``hand_names`` overrides the classifier's
    species for a seed, keeping the classifier's own call beside it so the two can be compared.
    """
    render, source = source_for(seed, sheet_path, work_dir, joint_dir)
    n_boxes, kept = count_instances(render, query=DETECTOR_QUERY, device=device)
    with Image.open(render) as im:
        image = im.convert("RGB")
        side = max(image.size)
        frame = float(side * side)
        boxes = []
        for i, det in enumerate(kept):
            call = clf(module.crop(image, det["box"]))
            record = {
                "box_index": i,
                "box_pixels": [float(v) for v in det["box"]],
                "box_cells": to_cells(det["box"], side),
                "detector_confidence": float(det["confidence"]),
                "frame_fraction": area(det["box"]) / frame,
                "clip_call": call["call"],
                "clip_margin": call["margin"],
                **call,
            }
            by_hand = (hand_names or {}).get(str(seed), {}).get(str(i))
            if by_hand is not None:
                record["call"] = by_hand
                record["named_by"] = "hand"
            else:
                record["named_by"] = "clip"
            boxes.append(record)

    if n_boxes != 2:
        reason = f"the detector kept {n_boxes} animal box(es), not 2"
    else:
        reason = geometry_reason(boxes, side, max_box_fraction, max_containment)
        if reason is None:
            calls = sorted(b["call"] for b in boxes)
            if calls != ["cat", "dog"]:
                reason = (f"the two boxes were named {calls[0]} and {calls[1]}, not one cat and "
                          f"one dog")
    return {
        "seed": seed,
        "render_path": str(render),
        "source": source,
        "image_side_px": side,
        "n_animal_boxes": n_boxes,
        "containment": containment(boxes[0]["box_pixels"], boxes[1]["box_pixels"])
                       if len(boxes) == 2 else None,
        "boxes": boxes,
        "judgeable": reason is None,
        "rejected_because": reason,
    }


def draw_crops(records: list[dict], module, out_path: Path) -> None:
    """Every candidate box's crop with the name it was given: what the classifier was handed.

    The crop classifier scored 64 of 64 on single-cat renders and then named both boxes the same
    species on two-animal renders that plainly hold one of each. This sheet is what separates a
    bad box from a bad call, because the crop is the only thing CLIP ever sees.
    """
    crops = [(r, b) for r in records for b in r["boxes"]]
    cols = 8
    rows = (len(crops) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(1.7 * cols, 2.0 * rows), squeeze=False)
    for ax, (rec, b) in zip(axes.ravel(), crops):
        with Image.open(rec["render_path"]) as im:
            ax.imshow(module.crop(im.convert("RGB"), b["box_pixels"]).resize((160, 160)))
        colour = {"cat": "#1f77b4", "dog": "#d62728"}.get(b["clip_call"], "darkorange")
        ax.set_title(f"seed {rec['seed']}  box {b['box_index']}\n{b['clip_call']} "
                     f"{b['clip_margin']:+.2f}  ({b['frame_fraction']:.0%} of frame)",
                     fontsize=7, color=colour)
        ax.axis("off")
    for ax in axes.ravel()[len(crops):]:
        ax.axis("off")
    fig.suptitle("every candidate box, cropped exactly as the classifier received it, with the "
                 "name CLIP gave it\n"
                 "blue cat, red dog, orange uncertain; the margin is p_cat - p_dog and the "
                 "percentage is the box's share of the frame", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def draw_panels(records: list[dict], out_path: Path) -> None:
    chosen = [r for r in records if r.get("selected")]
    cols = 4
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.9 * rows), squeeze=False)
    for ax, rec in zip(axes.ravel(), chosen):
        with Image.open(rec["render_path"]) as im:
            ax.imshow(im.convert("RGB"))
        lines = []
        for b in rec["boxes"]:
            x0, y0, x1, y1 = b["box_pixels"]
            colour = "#1f77b4" if b["call"] == "cat" else "#d62728"
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor=colour, linewidth=2.6))
            ax.text(x0 + 5, y0 + 24, f"{b['call']} {b['margin']:+.2f}", fontsize=9, color="white",
                    bbox=dict(facecolor=colour, edgecolor="none", alpha=0.9, pad=1.6))
            px = ", ".join(f"{v:.0f}" for v in b["box_pixels"])
            cl = ", ".join(str(v) for v in b["box_cells"])
            lines.append(f"{b['call']}: [{px}] px  ->  [{cl}] cells")
        ax.set_title(f"seed {rec['seed']}  ({rec['source']}, {rec['image_side_px']} px)\n"
                     + "\n".join(lines), fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[len(chosen):]:
        ax.axis("off")
    fig.suptitle("the oracle layout: both animals' boxes on the joint prompt's own render, "
                 "located by GroundingDINO and named by CLIP on the crop\n"
                 "blue is the box called cat, red the box called dog; the margin is p_cat - p_dog; "
                 "cells are on the 128 x 128 latent grid", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", type=Path, default=SHEET_DEFAULT,
                    help="the 8 x 4 reference contact sheet, for seeds with no 1024 px render")
    ap.add_argument("--out", type=Path,
                    default=REPO / "3-understanding-compositionality/artifacts/results/is-the-layout-the-missing-piece",
                    help="where boxes.json and detected-layouts.png are written")
    ap.add_argument("--wanted", type=int, default=WANTED, help="how many judgeable seeds to select")
    ap.add_argument("--joint-dir", type=Path, default=JOINT_OURS,
                    help="joint renders made by this repo from the cached noise; the layout "
                         "is detected on these so it belongs to the picture the sweep draws")
    ap.add_argument("--max-box-fraction", type=float, default=MAX_BOX_FRACTION,
                    help="reject a seed where either box covers more than this share of the "
                         "frame; such a box is not a region")
    ap.add_argument("--max-containment", type=float, default=MAX_CONTAINMENT,
                    help="reject a seed where this much of the smaller box sits inside the "
                         "larger; the two regions are then not disjoint")
    ap.add_argument("--names", type=Path, default=None,
                    help="JSON of hand-given species, {\"<seed>\": {\"<box index>\": "
                         "\"cat\"|\"dog\"}}, overriding the classifier. The classifier's own "
                         "call is kept beside it in boxes.json either way.")
    ap.add_argument("--max-foreign-mib", type=int, default=MAX_FOREIGN_MIB,
                    help="how much GPU memory another process may already hold. The default "
                         "refuses a shared device; raise it to share one deliberately, which the "
                         "execution protocol allows for a small job. The separate free-memory "
                         "check cannot be waived.")
    args = ap.parse_args()

    print(f"node {socket.gethostname()}  CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}  pid {os.getpid()}", flush=True)
    guard_disk(args.out)
    device = guard_device(args.max_foreign_mib)
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)

    work_dir = args.out / "source-tiles"
    work_dir.mkdir(parents=True, exist_ok=True)
    module = load_classifier_module()
    clf = module.ZeroShot(device)
    print(f"detector query {DETECTOR_QUERY!r}, kept boxes need confidence >= 0.30 and a longer "
          f"side >= {MIN_BOX_FRACTION:g} of the image; species margin floor "
          f"{module.MIN_MARGIN}", flush=True)

    hand_names = json.loads(args.names.read_text()) if args.names else None
    if hand_names:
        print(f"hand names given for seeds {sorted(hand_names)}; the classifier's call is kept "
              f"beside each one", flush=True)

    # Every candidate is read, never stopping at the first eight, so the crop sheet covers the
    # rejected seeds too. Rejection reasons are only readable against the seeds that passed.
    records = []
    for seed in HELD_OUT + TOP_UP:
        if seed in TOP_UP and not args.sheet.exists():
            print(f"seed {seed:2d}  no 1024 px render and no sheet at {args.sheet}; stopping the "
                  f"top-up", flush=True)
            break
        rec = read_seed(seed, args.sheet, work_dir, clf, module, device,
                        max_box_fraction=args.max_box_fraction,
                        max_containment=args.max_containment,
                        hand_names=hand_names, joint_dir=args.joint_dir)
        records.append(rec)
        shown = "  ".join(f"{b['call']}({b['named_by'][0]}) {b['clip_margin']:+.2f}"
                          for b in rec["boxes"]) or "no box"
        held = "" if rec["containment"] is None else f"held {rec['containment']:.0%}"
        mark = "judgeable" if rec["judgeable"] else rec["rejected_because"]
        print(f"seed {seed:2d}  {rec['source']:15s}  {rec['n_animal_boxes']} box  "
              f"{shown:34s}  {held:9s}  {mark}", flush=True)

    selected = 0
    for rec in records:
        rec["selected"] = rec["judgeable"] and selected < args.wanted
        selected += int(rec["selected"])

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "boxes.json").write_text(json.dumps({
        "what_this_is": "the oracle layout per seed: the two animals' boxes in the joint prompt's "
                        "own render, located by GroundingDINO and named by CLIP on the crop",
        "detector": "IDEA-Research/grounding-dino-tiny, query 'animal', the compose scorer's "
                    "own filters",
        "classifier": f"{module.CLIP_ID}, cat against dog over {len(module.TEMPLATES)} templates, "
                      f"box pad {module.BOX_PAD}, margin floor {module.MIN_MARGIN}",
        "latent_grid": LATENT,
        "geometry_gate": {"max_box_fraction": args.max_box_fraction,
                          "max_containment": args.max_containment},
        "hand_named_seeds": sorted(hand_names) if hand_names else [],
        "held_out_seeds": list(HELD_OUT),
        "wanted": args.wanted,
        "n_selected": selected,
        "selected_seeds": [r["seed"] for r in records if r["selected"]],
        "node": socket.gethostname(),
        "seeds": records,
    }, indent=2))

    draw_crops(records, module, args.out / "candidate-crops.png")
    if selected:
        draw_panels(records, args.out / "detected-layouts.png")
    print(f"\n{selected} of {args.wanted} judgeable seeds selected: "
          f"{[r['seed'] for r in records if r['selected']]}", flush=True)
    print(f"wrote {args.out / 'boxes.json'} and {args.out / 'candidate-crops.png'}"
          + (f" and {args.out / 'detected-layouts.png'}" if selected else ""), flush=True)
    return 0 if selected == args.wanted else 1


if __name__ == "__main__":
    raise SystemExit(main())
