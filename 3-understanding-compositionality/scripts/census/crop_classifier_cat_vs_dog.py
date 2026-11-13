#!/usr/bin/env python
"""Can a classifier on the cropped animal box tell a cat from a dog, where the detector cannot?

Why this exists. The base-rate run (3-understanding-compositionality/scripts/census/cat_condition_dog_rate.py, 2026-11-03) put a
"dog" box on all 64 renders of "a photo of a cat", every one of which is a single cat by eye.
GroundingDINO's median confidence was 0.889 on the query "dog" against 0.943 on "cat", the median
margin 0.054, and the sign flipped on 3 renders, so no threshold or margin rule over detector
confidences separates the two classes. Open-vocabulary grounding puts an animal noun on any
animal-shaped region. The instance count is still sound for HOW MANY animals a render holds and is
unusable for WHICH animal each one is.

Every class-level read the inclusive-captions pilot needs depends on fixing that
(4-inclusive-captions/artifacts/ideas/inclusive-captions-axis/IDEA_MAP.md, the "Checks outstanding" entry called
"The instrument"): claim 1 counts dog-present renders of "a photo of a cat", and claim 2 sorts
renders into pair, chimera and single animal, which needs the two animals named apart.

What this does. For each render, the detector locates the animals with its generic "animal" query,
exactly as the compose scorer does, so the box set is the one the pilot would actually score. Each
box is cropped with a small pad and handed to CLIP ViT-B/32 as a zero-shot two-way choice between
cat and dog, averaged over four prompt templates so the call does not hang on one phrasing. The
whole image is classified the same way beside it, which says whether cropping is doing any work.

What the answer means, written before the run. These 64 renders are all cats by eye, so the truth
is known and this measures the instrument rather than the model.
  calls cat on at least 62 of 64 crops and the median margin is at least 0.50
      the instrument works. Claim 1's render-and-count check comes back to life and claim 2's
      three categories become scoreable.
  calls cat on at least 62 but the median margin is under 0.50
      it works but sits close to the line. Report the margin distribution, carry a confidence
      floor, and re-check on real both-animal renders before trusting it on pairs.
  calls dog on 3 or more crops
      zero-shot CLIP fails the same way the detector did. The pilot owes a trained classifier
      (an ImageNet head over the crop), and claim 6's cost rises by that work.
A margin under MIN_MARGIN is reported as uncertain rather than forced into a class, because a
pilot that silently calls coin flips would repeat the mistake this run exists to catch.

Writes, under MOUNT_ROOT/outputs/cat_condition_dog_rate/<tag>/:
    crop_classifier_per_crop.json   one row per box: seed, box, p_cat, p_dog, margin, call,
                                    and the same three numbers for the uncropped image
    crop_classifier_summary.json    the counts above, the margin quantiles, the verdict
    crop_classifier_sheet.png       8 x 8 crops, each labelled with its call and margin

Run it on a biggpu node's free device, per environment/hpc/execution-protocol.md:
    ssh mscluster106 'CUDA_VISIBLE_DEVICES=1 nohup \
      /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python \
      /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/scripts/census/crop_classifier_cat_vs_dog.py \
      > /abs/path/to/log 2>&1 &'
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
import time
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from poe_repair import paths  # noqa: E402

CLIP_ID = "openai/clip-vit-base-patch32"
FAMILY = "outputs/cat_condition_dog_rate"
CLASSES = ("cat", "dog")
# Four phrasings, averaged, so one unlucky template cannot decide a call.
TEMPLATES = ("a photo of a {}", "a {}", "a close-up photo of a {}", "a photo of one {}")
# Pad each detector box by this fraction of its longer side before cropping, so a tight box does
# not cut off the ears or the muzzle that carry the class.
BOX_PAD = 0.08

# The bars, in the source so they cannot be moved after seeing the answer. The truth on these 64
# renders is "cat" on every one, read by eye off the contact sheet.
MIN_CORRECT = 62          # of 64 crops called cat
MIN_MEDIAN_MARGIN = 0.50  # median of p_cat - p_dog
MIN_MARGIN = 0.20         # below this the crop is reported uncertain, not assigned a class

MAX_FOREIGN_MIB = 1024


def guard_device() -> torch.device:
    """Refuse to start on a busy, faulted or absent GPU."""
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch.cuda.is_available() is False; co3 on a Blackwell node "
                         "produces no CUDA output at all, so this would run silently on the CPU")
    free_b, total_b = torch.cuda.mem_get_info(0)
    used_mib = (total_b - free_b) / 1024 ** 2
    if used_mib > MAX_FOREIGN_MIB:
        raise SystemExit(f"guard: visible device holds {used_mib:.0f} MiB, over the "
                         f"{MAX_FOREIGN_MIB} MiB foreign-process limit; it was taken since the check")
    return torch.device("cuda")


class ZeroShot:
    """CLIP as a two-way cat-or-dog choice, averaged over the prompt templates."""

    def __init__(self, device: torch.device) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(CLIP_ID)
        prompts = [t.format(c) for c in CLASSES for t in TEMPLATES]
        with torch.no_grad():
            inputs = self.processor(text=prompts, return_tensors="pt", padding=True)
            feats = self.model.get_text_features(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
            )
        feats = feats / feats.norm(dim=-1, keepdim=True)
        # One averaged, renormalised direction per class.
        per_class = feats.reshape(len(CLASSES), len(TEMPLATES), -1).mean(dim=1)
        self.text = per_class / per_class.norm(dim=-1, keepdim=True)

    def __call__(self, image: Image.Image) -> dict:
        with torch.no_grad():
            inputs = self.processor(images=[image], return_tensors="pt")
            feat = self.model.get_image_features(
                pixel_values=inputs["pixel_values"].to(self.device))
        feat = feat / feat.norm(dim=-1, keepdim=True)
        logits = self.model.logit_scale.exp() * feat @ self.text.T
        p = logits.softmax(dim=-1)[0].tolist()
        p_cat, p_dog = p[0], p[1]
        margin = p_cat - p_dog
        if abs(margin) < MIN_MARGIN:
            call = "uncertain"
        else:
            call = "cat" if margin > 0 else "dog"
        return {"p_cat": p_cat, "p_dog": p_dog, "margin": margin, "call": call}


def crop(image: Image.Image, box) -> Image.Image:
    x0, y0, x1, y1 = box
    pad = BOX_PAD * max(x1 - x0, y1 - y0)
    w, h = image.size
    return image.crop((max(0.0, x0 - pad), max(0.0, y0 - pad),
                       min(float(w), x1 + pad), min(float(h), y1 + pad)))


def sheet(rows: list[dict], out_path: Path, thumb: int = 160) -> None:
    n = len(rows)
    side = int(n ** 0.5 + 0.999)
    fig, axes = plt.subplots(side, side, figsize=(side * 1.5, side * 1.6))
    for ax, row in zip(axes.ravel(), rows):
        with Image.open(row["render_path"]) as im:
            ax.imshow(crop(im.convert("RGB"), row["box"]).resize((thumb, thumb)))
        colour = {"cat": "black", "dog": "crimson", "uncertain": "darkorange"}[row["call"]]
        ax.set_title(f"seed {row['seed']}  {row['call']}  {row['margin']:+.2f}",
                     fontsize=7, color=colour)
        ax.axis("off")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("cropped animal boxes from base SDXL's \"a photo of a cat\", classified cat "
                 "against dog by zero-shot CLIP ViT-B/32\n"
                 f"(label: the call, and p_cat - p_dog; every crop is a cat by eye; "
                 f"margins under {MIN_MARGIN} are reported uncertain)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="base-sdxl", help="subfolder of the base-rate run's output")
    ap.add_argument("--seeds", type=int, default=64)
    args = ap.parse_args()

    out_dir = paths.MOUNT_ROOT / FAMILY / args.tag
    if not out_dir.is_dir():
        raise SystemExit(f"no renders at {out_dir}; run 3-understanding-compositionality/scripts/census/cat_condition_dog_rate.py first")

    print(f"node {socket.gethostname()}  CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}  pid {os.getpid()}", flush=True)
    device = guard_device()
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)

    from poe_repair.experiments.compose_scorer_validation.detection_scorer import count_instances

    clf = ZeroShot(device)
    rows: list[dict] = []
    started = time.time()
    for seed in range(args.seeds):
        render = out_dir / f"seed_{seed:02d}.png"
        if not render.exists():
            print(f"seed {seed:2d}  missing, skipped", flush=True)
            continue
        n_animal, boxes = count_instances(render, query="animal", device=device)
        with Image.open(render) as im:
            whole = clf(im.convert("RGB"))
            for i, b in enumerate(boxes):
                r = clf(crop(im.convert("RGB"), b["box"]))
                rows.append({"seed": seed, "render_path": str(render), "box_index": i,
                             "box": [float(v) for v in b["box"]],
                             "detector_confidence": float(b["confidence"]),
                             "n_animal_boxes": n_animal, **r,
                             "whole_image_p_cat": whole["p_cat"],
                             "whole_image_margin": whole["margin"],
                             "whole_image_call": whole["call"]})
        last = rows[-1] if rows else {"call": "no box", "margin": 0.0}
        print(f"seed {seed:2d}  boxes {n_animal}  crop call {last['call']} "
              f"({last['margin']:+.3f})  whole call {whole['call']} ({whole['margin']:+.3f})  "
              f"({time.time() - started:.0f}s)", flush=True)

    margins = [r["margin"] for r in rows]
    calls = [r["call"] for r in rows]
    n_cat, n_dog = calls.count("cat"), calls.count("dog")
    median_margin = statistics.median(margins) if margins else 0.0
    if n_dog >= 3:
        verdict = ("zero-shot CLIP fails like the detector: the pilot owes a trained classifier "
                   "over the crop, and claim 6's cost rises by that work")
    elif n_cat >= MIN_CORRECT and median_margin >= MIN_MEDIAN_MARGIN:
        verdict = ("the instrument works: claim 1's render-and-count check is usable again and "
                   "claim 2's three categories are scoreable")
    elif n_cat >= MIN_CORRECT:
        verdict = (f"calls are right but the median margin {median_margin:.3f} is under "
                   f"{MIN_MEDIAN_MARGIN}: carry a confidence floor and re-check on real "
                   f"both-animal renders before scoring pairs")
    else:
        verdict = (f"only {n_cat} of {len(rows)} crops called cat, under the bar of {MIN_CORRECT}: "
                   f"not a usable instrument as configured")

    summary = {
        "what_this_measures": "whether a classifier on the cropped animal box separates cat from "
                              "dog on renders the detector called dog on all 64 times",
        "clip_id": CLIP_ID, "templates": list(TEMPLATES), "box_pad_fraction": BOX_PAD,
        "bars": {"min_crops_called_cat": MIN_CORRECT,
                 "min_median_margin": MIN_MEDIAN_MARGIN,
                 "uncertain_below_margin": MIN_MARGIN},
        "ground_truth": "every render is a single cat, read by eye off the base-rate contact sheet",
        "n_crops": len(rows),
        "n_called_cat": n_cat, "n_called_dog": n_dog, "n_uncertain": calls.count("uncertain"),
        "margin_min": min(margins, default=0.0),
        "margin_median": median_margin,
        "margin_max": max(margins, default=0.0),
        "n_crops_margin_below_floor": sum(1 for m in margins if abs(m) < MIN_MARGIN),
        "whole_image_n_called_cat": sum(1 for r in rows if r["whole_image_call"] == "cat"),
        "whole_image_margin_median": statistics.median(
            [r["whole_image_margin"] for r in rows]) if rows else 0.0,
        "verdict": verdict,
        "node": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (out_dir / "crop_classifier_per_crop.json").write_text(json.dumps(rows, indent=2))
    (out_dir / "crop_classifier_summary.json").write_text(json.dumps(summary, indent=2))
    sheet(rows, out_dir / "crop_classifier_sheet.png")

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
