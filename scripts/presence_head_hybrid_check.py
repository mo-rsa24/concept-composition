#!/usr/bin/env python
"""Does the calibrated per-species test see a chimera, where the two-way softmax could not?

Scores this repo's own fused cat x dog product renders with the two independent presence tests
and the floor the calibration chose. A chimera should pass both tests on one box, which is what
the labelling rule needs in order to call it neither animal present.

De-duplicates by file content first: the same render is reachable under several output families,
and counting rows rather than images inflated the earlier pool (33 rows, 14 images).

Writes, under MOUNT_ROOT/outputs/presence_head/hybrid_check/:
    per_render.json   one row per distinct image: p_cat, p_dog, which tests pass, the old margin
    summary.json      the counts, and what fraction of known fused renders pass both
    sheet.png         every render beside its two independent probabilities and its new call
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
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
from presence_head_calibrate_species import (  # noqa: E402
    SpeciesPresence, crop, guard_device, disk_guard,
)

SET_DIR = paths.MOUNT_ROOT / "outputs/presence_head/labelled_set"
CAL_PATH = paths.MOUNT_ROOT / "outputs/presence_head/species_calibration/cosine/calibration.json"
FAMILY = "outputs/presence_head/hybrid_check"


def content_hash(path: str) -> str | None:
    try:
        return hashlib.md5(Path(path).read_bytes()).hexdigest()
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", default="a_cat__x__a_dog")
    args = ap.parse_args()

    out_dir = paths.MOUNT_ROOT / FAMILY
    disk_guard(out_dir)
    calibration = json.loads(CAL_PATH.read_text())
    if not calibration["instrument_usable"]:
        raise SystemExit(f"the calibration says the instrument fails: {calibration['verdict']}")
    floor = float(calibration["chosen_floor"])
    print(f"floor {floor} from {CAL_PATH}", flush=True)

    device = guard_device()
    from poe_repair.experiments.compose_scorer_validation.detection_scorer import count_instances

    presence = SpeciesPresence(device, mode=calibration["score_mode"])
    rows = json.loads((SET_DIR / "rows.json").read_text())
    picked, seen = [], set()
    for r in rows:
        if r["source"] != "render" or r["pair"] != args.pair:
            continue
        h = content_hash(r["path"])
        if h is None or h in seen:
            continue
        seen.add(h)
        picked.append(r)
    print(f"{args.pair}: {len(picked)} distinct images after de-duplicating by content", flush=True)

    out_rows, started = [], time.time()
    for r in picked:
        _, boxes = count_instances(Path(r["path"]), query="animal", device=device)
        with Image.open(r["path"]) as im:
            rgb = im.convert("RGB")
            per_box = [presence(crop(rgb, b["box"])) for b in boxes] or [presence(rgb)]
        p_cat = max(p["cat"] for p in per_box)
        p_dog = max(p["dog"] for p in per_box)
        cat_pass, dog_pass = p_cat >= floor, p_dog >= floor
        call = ("both species on one box: a hybrid" if (cat_pass and dog_pass and len(boxes) < 2)
                else "two boxes, one of each" if (cat_pass and dog_pass)
                else "cat only" if cat_pass else "dog only" if dog_pass else "neither")
        out_rows.append({"path": r["path"], "seed": r["seed"], "n_boxes": len(boxes),
                         "p_cat": round(p_cat, 4), "p_dog": round(p_dog, 4),
                         "cat_pass": cat_pass, "dog_pass": dog_pass, "call": call,
                         "old_softmax_margin": (r.get("margins") or [None])[0],
                         "old_label": [r["cat_present"], r["dog_present"]]})
        print(f"  seed {r['seed']:>3}  boxes {len(boxes)}  p_cat {p_cat:.3f}  p_dog {p_dog:.3f}  "
              f"{call}  ({time.time() - started:.0f}s)", flush=True)

    calls = collections.Counter(x["call"] for x in out_rows)
    both = sum(1 for x in out_rows if x["cat_pass"] and x["dog_pass"])
    summary = {
        "what_this_measures": "whether two independent per-species tests both fire on a fused "
                              "animal, where a two-way softmax named one species on 32 of 33",
        "pair": args.pair, "floor": floor,
        "n_distinct_images": len(out_rows),
        "n_passing_both_tests": both,
        "fraction_passing_both": round(both / max(1, len(out_rows)), 4),
        "calls": dict(calls),
        "n_passing_both_under_old_rule": 0,
        "note": "every one of these renders is a fused animal by eye and the product composes on "
                "0 of 8 held-out seeds, so a test that sees both species here is doing what the "
                "labelling rule needs",
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (out_dir / "per_render.json").write_text(json.dumps(out_rows, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    cols = 5
    n_rows = (len(out_rows) + cols - 1) // cols
    fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 2.7, n_rows * 3.1))
    axes = axes.ravel()
    for ax, r in zip(axes, out_rows):
        with Image.open(r["path"]) as im:
            ax.imshow(im.convert("RGB").resize((320, 320)))
        hybrid = r["cat_pass"] and r["dog_pass"]
        ax.set_title(f"seed {r['seed']}   p(cat) {r['p_cat']:.2f}   p(dog) {r['p_dog']:.2f}\n"
                     f"{'BOTH fire: a hybrid' if hybrid else r['call']}\n"
                     f"old softmax said: {r['old_softmax_margin']:+.2f}"
                     if r["old_softmax_margin"] is not None else "",
                     fontsize=8, color="darkgreen" if hybrid else "crimson")
        ax.axis("off")
    for ax in axes[len(out_rows):]:
        ax.axis("off")
    fig.suptitle(
        f"{args.pair}: {len(out_rows)} distinct fused product renders, each scored by two "
        f"independent presence tests at a floor of {floor}\n"
        f"green means both tests fire, so the rule can call it a hybrid; "
        f"the old two-way softmax margin is printed underneath for comparison",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "sheet.png", dpi=100)
    plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
