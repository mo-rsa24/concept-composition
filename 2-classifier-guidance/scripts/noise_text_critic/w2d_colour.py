#!/usr/bin/env python
"""Step W2d, colour binding: a red cat and a blue dog against the swap, with whole-animal regions.

  prepare  decode W2's relation pictures, find the two animals with GroundingDINO, cut each out with SAM, recolour
           each picture both ways, encode the copies, and draw the eye-check sheets
  read     score both copies under W2's protocol (only after the eye check passes)

The design is noise-text-critic/02d-binding/design.md, section "Colour binding". Usage: python w2d_colour.py prepare|read [--smoke]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import w2_read as w2  # noqa: E402
from common import (LOOP, REPO, SENTENCES, eval_tiles, image_features, latent_clip_image, latent_to_clip,  # noqa: E402
                    load_clip, load_latent_clip, load_tile_latent, sha)
from w2_prepare import BLUE_HUE, RED_HUE, colour_tile, encode, recolour, write_sheets  # noqa: E402
from w2c_fair import TEMPLATES, phrase  # noqa: E402

# Fixed in 02d-binding/design.md before any recoloured picture.
DETECTOR = "IDEA-Research/grounding-dino-tiny"
SEGMENTER = "facebook/sam-vit-base"
QUERY = "an animal."
MIN_SCORE = 0.30
MIN_BOX_FRACTION = 0.25
NMS_IOU = 0.5
EDGE_PIXELS = 2.0
EYE_BAR = 0.90
READ_BAR = 0.75
OUT = LOOP / "w2d/colour"
EYE_LABELS = REPO / "artifacts/notes/progress/noise-text-critic/02-can-it-read/label-check/eye-labels.csv"
MODELS = ("noisyclip", "frozen_clip", "latent_clip")
STEPS = list(range(16)) + list(range(20, 30)) + [30, 40, 49]
CLEAN = 50   # the clean latent, scored as if it were one more step


def iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


@torch.no_grad()
def two_animals(det, proc, pil, device):
    inputs = proc(images=pil, text=QUERY, return_tensors="pt").to(device)
    out = det(**inputs)
    res = proc.post_process_grounded_object_detection(out, inputs.input_ids, box_threshold=MIN_SCORE, text_threshold=MIN_SCORE,
                                                      target_sizes=[pil.size[::-1]])[0]
    side = MIN_BOX_FRACTION * max(pil.size)
    cands = sorted(((float(s), [float(v) for v in b]) for s, b in zip(res["scores"], res["boxes"])
                    if max(b[2] - b[0], b[3] - b[1]) >= side), key=lambda c: -c[0])
    keep = []
    for s, b in cands:
        if all(iou(b, k[1]) < NMS_IOU for k in keep):
            keep.append((s, b))
    return keep


@torch.no_grad()
def masks_for(sam, sproc, pil, boxes, device):
    inputs = sproc(pil, input_boxes=[boxes], return_tensors="pt").to(device)
    out = sam(**inputs, multimask_output=False)
    m = sproc.image_processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                                                 inputs["reshaped_input_sizes"].cpu())[0]
    return m[:, 0].numpy().astype(bool)                              # (2, H, W)


def soft(cat: np.ndarray, dog: np.ndarray):
    both = cat & dog
    mc = gaussian_filter((cat & ~both).astype(np.float32), EDGE_PIXELS)
    md = gaussian_filter((dog & ~both).astype(np.float32), EDGE_PIXELS)
    total = mc + md
    over = total > 1
    mc[over] /= total[over]
    md[over] /= total[over]
    return mc, md


def prepare(args) -> int:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor
    from poe_repair._sdxl.runtime import decode_latents_to_tensor
    from poe_repair.run import make_ctx

    device = torch.device("cuda")
    records = json.loads((LOOP / "w2/prep/labels.json").read_text())["records"]
    eye = {int(r["index"]): r["relation_right"] == "1" for r in csv.DictReader(open(EYE_LABELS))}
    records = [r for r in records if eye[r["index"]]]
    if args.smoke:
        records = records[:4]
    tiles = {t["id"]: t for t in eval_tiles()}
    out = OUT.with_name("colour-smoke") if args.smoke else OUT
    out.mkdir(parents=True, exist_ok=True)
    ctx = make_ctx(output_root=out / "raw")
    vae = ctx.models["vae"].requires_grad_(False).eval()
    proc, det = AutoProcessor.from_pretrained(DETECTOR), AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR).to(device).eval()
    sproc, sam = SamProcessor.from_pretrained(SEGMENTER), SamModel.from_pretrained(SEGMENTER).to(device).eval()
    kept, left_out, ids, latents, sheet = [], [], [], [], []
    t0 = time.time()
    for r in sorted(records, key=lambda r: sha(r["id"])):
        z0 = load_tile_latent(tiles[r["id"]])
        with torch.no_grad():
            img = decode_latents_to_tensor(vae, z0.to(device))[0].permute(1, 2, 0).float().cpu().numpy()
        pil = Image.fromarray((img * 255).round().astype(np.uint8))
        boxes = two_animals(det, proc, pil, device)
        if len(boxes) != 2:
            left_out.append({"id": r["id"], "boxes": len(boxes)})
            continue
        left, right = sorted((b for _, b in boxes), key=lambda b: (b[0] + b[2]) / 2)
        cat_box, dog_box = (left, right) if r["cat_left"] else (right, left)
        cat_m, dog_m = masks_for(sam, sproc, pil, [cat_box, dog_box], device)
        mc, md = soft(cat_m, dog_m)
        red_blue, blue_red = recolour(img, mc, md, RED_HUE, BLUE_HUE), recolour(img, mc, md, BLUE_HUE, RED_HUE)
        for name, arr in (("clean", img), ("red_cat_blue_dog", red_blue), ("blue_cat_red_dog", blue_red)):
            ids.append(f"{r['id']}::{name}")
            latents.append(encode(vae, arr, device))
        n = len(kept)
        kept.append({"id": r["id"], "seed": r["seed"], "cat_box": cat_box, "dog_box": dog_box,
                     "cat_mask_share": float(cat_m.mean()), "dog_mask_share": float(dog_m.mean()),
                     "overlap_share": float((cat_m & dog_m).mean())})
        if n < 16:
            sheet += [colour_tile(red_blue, n, "red", "blue"), colour_tile(blue_red, n, "blue", "red")]
        print(f"[{time.time() - t0:5.0f}s] {r['id']} cat {cat_m.mean():.3f} dog {dog_m.mean():.3f} overlap {(cat_m & dog_m).mean():.4f}", flush=True)
    torch.save({"ids": ids, "latents": torch.cat(latents)}, out / "copies.pt")
    (out / "regions.json").write_text(json.dumps({"kept": kept, "left_out": left_out, "rule": {
        "detector": DETECTOR, "segmenter": SEGMENTER, "query": QUERY, "min_score": MIN_SCORE,
        "min_box_fraction": MIN_BOX_FRACTION, "nms_iou": NMS_IOU, "edge_pixels": EDGE_PIXELS}}, indent=1))
    pages = write_sheets(sheet, out / "eye-check", "colours")
    print(f"{len(kept)} pictures recoloured, {len(left_out)} left out (not exactly two animals); {pages} eye-check sheets", flush=True)
    print("COLOUR PREPARE DONE", flush=True)
    return 0


def read(args) -> int:
    from poe_repair._sdxl.runtime import load_ddim_scheduler

    device = torch.device("cuda")
    src = OUT.with_name("colour-smoke") if args.smoke else OUT
    data = torch.load(src / "copies.pt", map_location="cpu", weights_only=True)
    ids, Z = data["ids"], data["latents"].float()
    regions = {k["id"]: k for k in json.loads((src / "regions.json").read_text())["kept"]}
    task = [(i, int(k.endswith("::red_cat_blue_dog")), regions[k.split("::")[0]]["seed"])
            for i, k in enumerate(ids) if not k.endswith("::clean")]
    right, wrong = SENTENCES["binding"]
    sentences = [right, *wrong]
    loaded = {m: load_clip("noisyclip" if m == "noisyclip" else "frozen", device) for m in MODELS[:2]}
    loaded["latent_clip"] = load_latent_clip(device)
    from w2c_fair import text_bank
    bank = text_bank(loaded, sentences, device)
    rows = tuple(bank)
    scheduler = load_ddim_scheduler("stabilityai/stable-diffusion-xl-base-1.0")
    scheduler.set_timesteps(50)
    timesteps = scheduler.timesteps.clone().cpu()
    steps = ([0, 15] if args.smoke else STEPS) + [CLEAN]
    draws = 1 if args.smoke else w2.DRAWS

    @torch.no_grad()
    def checker(x):
        pixels = latent_to_clip(x)
        f = {m: image_features(loaded[m][0], pixels).float() for m in MODELS[:2]}
        f["latent_clip"] = latent_clip_image(loaded["latent_clip"][0], x).float()
        return {r: ((f[r.replace("_ens", "")] @ bank[r].T)[:, 0] - (f[r.replace("_ens", "")] @ bank[r].T)[:, 1:].max(-1).values).cpu().numpy()
                for r in rows}

    raw = {r: [] for r in rows}
    t0 = time.time()
    for step in steps:
        per = {r: [] for r in rows}
        for dr in range(1 if step == CLEAN else draws):
            noise = torch.randn(Z.shape, generator=torch.Generator().manual_seed(1000 * step + dr))
            got = {r: [] for r in rows}
            for i in range(0, len(ids), 32):
                z0 = Z[i:i + 32].to(device)
                x = z0 if step == CLEAN else scheduler.add_noise(z0, noise[i:i + 32].to(device), timesteps[step].repeat(z0.shape[0]).to(device))
                for r, v in checker(x).items():
                    got[r].append(v)
            for r in rows:
                per[r].append(np.concatenate(got[r]))
        for r in rows:
            raw[r].append(np.mean(per[r], axis=0))
        print(f"[{time.time() - t0:5.0f}s] step {step} done", flush=True)
    raw = {r: np.stack(v) for r, v in raw.items()}
    w2.MODELS, w2.COMPARED = rows, tuple(r for r in rows if r != "frozen_clip")
    res = w2.analyse(raw, steps, task, args.smoke)
    pos = {k: i for i, k in enumerate(ids)}
    pairs = [(pos[f"{p}::red_cat_blue_dog"], pos[f"{p}::blue_cat_red_dog"]) for p in regions]
    for r in rows:
        if "balanced_accuracy_reading_half" in res.get(r, {}):
            res[r]["clean_reading_half"] = res[r]["balanced_accuracy_reading_half"][-1]
        res[r]["paired_right_above_swap"] = [float(np.mean([raw[r][k, a] > raw[r][k, b] for a, b in pairs])) for k in range(len(steps))]
    out = {"steps": steps, "pictures": len(regions), "sentences": sentences, "templates": TEMPLATES, "bar": READ_BAR,
           "binding": res,
           "rows_that_read_steps_0_to_15": [r for r in rows if res[r].get("mean_steps_0_to_15", 0) >= READ_BAR],
           "rows_that_pass_clean": [r for r in rows if res[r].get("clean_reading_half", 0) >= READ_BAR]}
    (src / "result.json").write_text(json.dumps(out, indent=1))
    for r in rows:
        if "mean_steps_0_to_15" in res[r]:
            p = res[r]["paired_right_above_swap"]
            print(f"{r:16s} mean 0-15 {res[r]['mean_steps_0_to_15']:.3f}  clean {res[r]['clean_reading_half']:.3f}  "
                  f"paired: step 0 {p[0]:.2f}, clean {p[-1]:.2f}", flush=True)
    print("COLOUR READ DONE", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["prepare", "read"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    return {"prepare": prepare, "read": read}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
