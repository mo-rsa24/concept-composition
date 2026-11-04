#!/usr/bin/env python
"""Calibrate a per-species presence test that two animals can both pass at once.

Why this replaces the margin test. The crop classifier is a two-way softmax between cat and dog,
so it divides each score by the sum of exactly those two and has to spend its mass on one side.
On 33 cat x dog product renders, every one a fused animal, it named a confident single species on
32, at margins out to 0.993. A rule built on that margin cannot see a chimera, which is the class
the presence head has to learn.

What replaces it. Two independent questions, each with its own denominator and its own broad
negative set: "is a cat present" is cat against dog and eight unrelated subjects, and "is a dog
present" is dog against cat and the same eight. The other species sits in the negatives as one
option among ten rather than as the only alternative, so a half-cat half-dog scores moderately on
both instead of being forced onto one. Each question is then a floor, not a comparison, and both
can pass.

The bar, written before the run. Calibrate on photographs whose species COCO annotated: 200
cat-only and 200 dog-only, each read through the same detector crop the renders are read through.
The floor is swept and the chosen one is the one that maximises (worst true pass rate) minus
(worst false pass rate). The instrument is usable only if at that floor every true pass rate is at
least MIN_TRUE_PASS and every false pass rate is at most MAX_FALSE_PASS. If no floor clears both,
this reports that the instrument fails and nothing downstream runs on it.

Writes, under MOUNT_ROOT/outputs/presence_head/species_calibration/:
    sweep.json        one row per candidate floor: true and false pass rates per species
    calibration.json  the chosen floor, the rates at it, and the verdict
    sweep.png         pass rate on y against the floor on x, four curves
"""
from __future__ import annotations

import argparse
import json
import os
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "census"))

from poe_repair import paths  # noqa: E402

CLIP_ID = "openai/clip-vit-base-patch32"
FAMILY = "outputs/presence_head/species_calibration"
MANIFEST = paths.MOUNT_ROOT / "outputs/inclusive_captions_pool/manifest.json"

TEMPLATES = ("a photo of a {}", "a {}", "a close-up photo of a {}", "a photo of one {}")

# Each species is asked against the other species and eight unrelated subjects. The other species
# is one negative among many, which is the whole difference from the two-way softmax: the question
# is "is this a cat" rather than "is this a cat or a dog".
DISTRACTORS = ("bird", "horse", "car", "tree", "person", "chair", "landscape", "flower")

# The bar, in source so it cannot move after the answer is seen.
MIN_TRUE_PASS = 0.90   # a cat-only photo must pass the cat test this often
MAX_FALSE_PASS = 0.10  # a dog-only photo must pass the cat test no more often than this
FLOOR_GRID = tuple(round(0.05 * i, 2) for i in range(1, 20))  # 0.05 to 0.95

BOX_PAD = 0.08
N_PER_SPECIES = 200
MAX_FOREIGN_MIB = 1024
MIN_FREE_DISK_PERCENT = 10


def disk_guard(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    free_percent = 100.0 * usage.free / usage.total
    print(f"disk guard: {root} has {free_percent:.1f}% free", flush=True)
    if free_percent < MIN_FREE_DISK_PERCENT:
        raise SystemExit(f"guard: {root} is {100 - free_percent:.1f}% full")


def guard_device() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch.cuda.is_available() is False; this would run on the CPU "
                         "in silence, which is what a Blackwell node does under co3")
    free_b, total_b = torch.cuda.mem_get_info(0)
    used_mib = (total_b - free_b) / 1024 ** 2
    if used_mib > MAX_FOREIGN_MIB:
        raise SystemExit(f"guard: visible device holds {used_mib:.0f} MiB")
    return torch.device("cuda")


class SpeciesPresence:
    """One independent presence probability per species, each with its own broad negative set."""

    def __init__(self, device: torch.device, species=("cat", "dog")) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.species = tuple(species)
        self.model = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(CLIP_ID)
        self.directions = {}
        for s in self.species:
            others = [o for o in self.species if o != s] + list(DISTRACTORS)
            subjects = [s] + others
            prompts = [t.format(subj) for subj in subjects for t in TEMPLATES]
            with torch.no_grad():
                inputs = self.processor(text=prompts, return_tensors="pt", padding=True)
                feats = self.model.get_text_features(
                    input_ids=inputs["input_ids"].to(device),
                    attention_mask=inputs["attention_mask"].to(device),
                )
            feats = feats / feats.norm(dim=-1, keepdim=True)
            per_subject = feats.reshape(len(subjects), len(TEMPLATES), -1).mean(dim=1)
            self.directions[s] = (per_subject / per_subject.norm(dim=-1, keepdim=True))

    def __call__(self, image: Image.Image) -> dict:
        with torch.no_grad():
            inputs = self.processor(images=[image], return_tensors="pt")
            feat = self.model.get_image_features(
                pixel_values=inputs["pixel_values"].to(self.device))
        feat = feat / feat.norm(dim=-1, keepdim=True)
        out = {}
        for s, text in self.directions.items():
            logits = self.model.logit_scale.exp() * feat @ text.T
            # Index 0 is the species itself; the rest are its negatives.
            out[s] = float(logits.softmax(dim=-1)[0, 0].item())
        return out


def crop(image: Image.Image, box) -> Image.Image:
    x0, y0, x1, y1 = box
    pad = BOX_PAD * max(x1 - x0, y1 - y0)
    w, h = image.size
    return image.crop((max(0.0, x0 - pad), max(0.0, y0 - pad),
                       min(float(w), x1 + pad), min(float(h), y1 + pad)))


def best_box_scores(path, presence, count_instances, device) -> dict | None:
    """The highest per-species probability over the animal boxes in one image."""
    try:
        _, boxes = count_instances(Path(path), query="animal", device=device)
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            if not boxes:
                return presence(rgb)
            per_box = [presence(crop(rgb, b["box"])) for b in boxes]
        return {s: max(p[s] for p in per_box) for s in per_box[0]}
    except Exception as exc:
        print(f"  skipped {path}: {type(exc).__name__}: {exc}", flush=True)
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=N_PER_SPECIES)
    args = ap.parse_args()

    out_dir = paths.MOUNT_ROOT / FAMILY
    disk_guard(out_dir)
    print(f"node {socket.gethostname()}  CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}", flush=True)
    device = guard_device()
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)

    from poe_repair.experiments.compose_scorer_validation.detection_scorer import count_instances

    presence = SpeciesPresence(device)
    manifest = json.loads(MANIFEST.read_text())
    started = time.time()

    scores = {"cat": [], "dog": []}
    for truth, set_name in (("cat", "cat_only"), ("dog", "dog_only")):
        entries = manifest["sets"][set_name][:args.n]
        for i, row in enumerate(entries):
            s = best_box_scores(row["file"], presence, count_instances, device)
            if s is not None:
                scores[truth].append(s)
            if i % 50 == 0:
                print(f"  {set_name} {i}/{len(entries)}  ({time.time() - started:.0f}s)", flush=True)
    print(f"scored {len(scores['cat'])} cat-only and {len(scores['dog'])} dog-only photographs",
          flush=True)

    sweep = []
    for floor in FLOOR_GRID:
        row = {"floor": floor}
        for truth in ("cat", "dog"):
            for asked in ("cat", "dog"):
                rate = sum(1 for s in scores[truth] if s[asked] >= floor) / max(1, len(scores[truth]))
                row[f"{truth}_photo_passes_{asked}_test"] = round(rate, 4)
        row["worst_true_pass"] = min(row["cat_photo_passes_cat_test"],
                                     row["dog_photo_passes_dog_test"])
        row["worst_false_pass"] = max(row["cat_photo_passes_dog_test"],
                                      row["dog_photo_passes_cat_test"])
        row["separation"] = round(row["worst_true_pass"] - row["worst_false_pass"], 4)
        sweep.append(row)

    best = max(sweep, key=lambda r: r["separation"])
    passes = (best["worst_true_pass"] >= MIN_TRUE_PASS
              and best["worst_false_pass"] <= MAX_FALSE_PASS)
    verdict = (
        f"usable: at a floor of {best['floor']} every true pass rate is "
        f"{best['worst_true_pass']:.3f} or better and every false pass rate is "
        f"{best['worst_false_pass']:.3f} or lower"
        if passes else
        f"fails: the best floor {best['floor']} gives worst true pass "
        f"{best['worst_true_pass']:.3f} against a bar of {MIN_TRUE_PASS} and worst false pass "
        f"{best['worst_false_pass']:.3f} against a bar of {MAX_FALSE_PASS}; nothing downstream "
        f"may use this instrument"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [r["floor"] for r in sweep]
    for key, style, label in (
        ("cat_photo_passes_cat_test", "-", "cat-only photo passes the cat test (want high)"),
        ("dog_photo_passes_dog_test", "-", "dog-only photo passes the dog test (want high)"),
        ("cat_photo_passes_dog_test", "--", "cat-only photo passes the dog test (want low)"),
        ("dog_photo_passes_cat_test", "--", "dog-only photo passes the cat test (want low)"),
    ):
        ax.plot(xs, [r[key] for r in sweep], style, label=label)
    ax.axvline(best["floor"], color="k", lw=1, label=f"chosen floor {best['floor']}")
    ax.axhline(MIN_TRUE_PASS, color="green", lw=0.8, ls=":", label=f"true-pass bar {MIN_TRUE_PASS}")
    ax.axhline(MAX_FALSE_PASS, color="red", lw=0.8, ls=":", label=f"false-pass bar {MAX_FALSE_PASS}")
    ax.set_xlabel("the floor each independent species probability must clear")
    ax.set_ylabel(f"fraction of photographs passing (n={len(scores['cat'])} per species)")
    ax.set_title("calibrating the per-species presence test on photographs COCO annotated")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "sweep.png", dpi=110)
    plt.close(fig)

    calibration = {
        "what_this_measures": "whether two independent per-species presence tests separate cat "
                              "from dog on photographs whose species COCO annotated",
        "why_not_the_margin": "a two-way softmax must spend its mass on one species, so it named "
                              "a confident species on 32 of 33 fused cat x dog renders",
        "clip_id": CLIP_ID, "templates": list(TEMPLATES), "distractors": list(DISTRACTORS),
        "bars": {"MIN_TRUE_PASS": MIN_TRUE_PASS, "MAX_FALSE_PASS": MAX_FALSE_PASS},
        "n_cat_only": len(scores["cat"]), "n_dog_only": len(scores["dog"]),
        "chosen_floor": best["floor"], "rates_at_chosen_floor": best,
        "instrument_usable": passes, "verdict": verdict,
        "node": socket.gethostname(), "elapsed_seconds": round(time.time() - started, 1),
    }
    (out_dir / "sweep.json").write_text(json.dumps(sweep, indent=2))
    (out_dir / "calibration.json").write_text(json.dumps(calibration, indent=2))
    print(json.dumps(calibration, indent=2), flush=True)


if __name__ == "__main__":
    main()
