#!/usr/bin/env python
"""The per-animal checker, on finished pictures: does looking at each animal separately beat a whole-picture read?

Whole-picture checkers seem to top out near 0.77 (CLIP features plus a linear layer, probe/catdog.json): a
clean pair differs from a reject in local details (which animal each body is, a borrowed ear or build, two
cats instead of a cat and a dog) that one summary of the picture blurs. This reads each animal on its own:
    1. GroundingDINO boxes every animal (count_instances, the project's generic animal query, its own
       confidence floor and overlap suppression);
    2. CLIP ViT-L/14 names each box's crop among "a photo of a cat", "a photo of a dog" and "a photo of an
       animal that is half cat and half dog";
    3. three readings on the same split as the checkers:
       (a) a fixed rule: clean when there are exactly two boxes, one called a cat and the other a dog, each
           with probability at least tau; tau is chosen on the training pictures;
       (b) a logistic regression on per-animal facts: the box count, each of the two largest boxes' three
           probabilities, their sizes as a share of the picture, and their overlap;
       (c) the whole-picture reference: CLIP features of the full picture plus a linear layer.
Fitted on training_set_catdog_more23.json's pictures, scored on eval_set_catdog_v2.json's 220 held-out
pictures (109 clean, 111 rejects), balanced accuracy, clean pair against reject.

Writes, under MOUNT_ROOT/outputs/classifier_guidance_ladder/rung2/probe/: per_animal.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
import augment_encode  # noqa: E402
import probe_frozen_features as pff  # noqa: E402
from poe_repair.experiments.compose_scorer_validation.detection_scorer import count_instances  # noqa: E402

TEXTS = ["a photo of a cat", "a photo of a dog", "a photo of an animal that is half cat and half dog"]


def balanced(pred, y):
    pred, y = np.asarray(pred, bool), np.asarray(y, bool)
    return float((pred[y].mean() + (~pred[~y]).mean()) / 2)


@torch.no_grad()
def read(tiles, device, clip, proc):
    text = clip.get_text_features(**proc(text=TEXTS, return_tensors="pt", padding=True).to(device))
    text = torch.nn.functional.normalize(text, dim=-1)
    out = []
    for i, t in enumerate(tiles):
        path = augment_encode.picture_path(t)
        n, boxes = count_instances(path, device=device)
        im = Image.open(path).convert("RGB")
        w, h = im.size
        boxes = sorted(boxes, key=lambda d: -(d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))[:3]
        reads = []
        for d in boxes:
            x0, y0, x1, y1 = [float(v) for v in d["box"]]
            crop = im.crop((max(0, x0), max(0, y0), min(w, x1), min(h, y1)))
            f = clip.get_image_features(**proc(images=[crop], return_tensors="pt").to(device))
            p = (100 * torch.nn.functional.normalize(f, dim=-1) @ text.T).softmax(-1)[0].tolist()
            reads.append({"p": p, "area": (x1 - x0) * (y1 - y0) / (w * h), "box": [x0 / w, y0 / h, x1 / w, y1 / h]})
        out.append({"n": n, "boxes": reads})
        if i % 200 == 0:
            print(f"  {i} of {len(tiles)}", flush=True)
    return out


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def rule(r, tau):
    if r["n"] != 2 or len(r["boxes"]) < 2:
        return False
    (c0, d0, _), (c1, d1, _) = r["boxes"][0]["p"], r["boxes"][1]["p"]
    return (c0 >= tau and d1 >= tau) or (d0 >= tau and c1 >= tau)


def features(r):
    f = [float(r["n"] == k) for k in (0, 1, 2)] + [float(r["n"] >= 3)]
    for k in range(2):
        b = r["boxes"][k] if k < len(r["boxes"]) else {"p": [0.0, 0.0, 0.0], "area": 0.0}
        f += list(b["p"]) + [b["area"]]
    two = len(r["boxes"]) >= 2
    f += [iou(r["boxes"][0]["box"], r["boxes"][1]["box"]) if two else 0.0]
    if two:  # one cat and one dog, in either order
        (c0, d0, _), (c1, d1, _) = r["boxes"][0]["p"], r["boxes"][1]["p"]
        f += [max(min(c0, d1), min(d0, c1))]
    else:
        f += [0.0]
    return f


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pff.picture_path = augment_encode.picture_path
    train = [t for t in json.loads((HERE / "training_set_catdog_more23.json").read_text())["tiles"]]
    evals = json.loads((HERE / "eval_set_catdog_v2.json").read_text())["tiles"]
    y_tr = np.array([t["label"] for t in train])
    y_te = np.array([t["label"] for t in evals])
    from transformers import CLIPModel, CLIPProcessor
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    print(f"reading {len(train)} training and {len(evals)} held-out pictures", flush=True)
    r_tr, r_te = read(train, device, clip, proc), read(evals, device, clip, proc)

    taus = [round(x, 2) for x in np.arange(0.3, 0.96, 0.05)]
    fit = {tau: balanced([rule(r, tau) for r in r_tr], y_tr) for tau in taus}
    tau = max(fit, key=fit.get)
    rule_te = balanced([rule(r, tau) for r in r_te], y_te)
    x_tr = torch.tensor([features(r) for r in r_tr], dtype=torch.float32)
    x_te = torch.tensor([features(r) for r in r_te], dtype=torch.float32)
    lr_te, lr_tpr, lr_tnr, _ = pff.fit_and_score(x_tr, torch.tensor(y_tr, dtype=torch.float32), x_te,
                                                 torch.tensor(y_te, dtype=torch.float32))
    g_tr, g_te = pff.embed(train, device)["clip"], pff.embed(evals, device)["clip"]
    whole_te, _, _, _ = pff.fit_and_score(g_tr, torch.tensor(y_tr, dtype=torch.float32), g_te,
                                          torch.tensor(y_te, dtype=torch.float32))
    counts = {k: {"clean": int(sum(1 for r, y in zip(r_te, y_te) if y == 1 and r["n"] == k)),
                  "reject": int(sum(1 for r, y in zip(r_te, y_te) if y == 0 and r["n"] == k))} for k in range(5)}
    report = {"train": len(train), "held_out": len(evals), "held_out_clean": int(y_te.sum()),
              "a_rule": {"tau": tau, "training": fit[tau], "held_out": rule_te},
              "b_per_animal_logistic": {"held_out": lr_te, "clean_called_clean": lr_tpr, "rejects_called_reject": lr_tnr},
              "c_whole_picture_clip": {"held_out": whole_te},
              "held_out_box_counts": counts}
    out = pff.OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_animal.json").write_text(json.dumps(report, indent=2))
    print(f"(a) rule, tau {tau}: training {fit[tau]:.3f}, held out {rule_te:.3f}", flush=True)
    print(f"(b) per-animal logistic: held out {lr_te:.3f} (clean {lr_tpr:.2f}, reject {lr_tnr:.2f})", flush=True)
    print(f"(c) whole-picture CLIP: held out {whole_te:.3f}", flush=True)
    print("held-out box counts:", counts, flush=True)
    print("PER-ANIMAL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
