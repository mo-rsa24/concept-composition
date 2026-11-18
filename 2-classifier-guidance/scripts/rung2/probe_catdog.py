#!/usr/bin/env python
"""Is "clean cat and dog against reject" learnable from our cat x dog data at all? The finished-picture upper bound.

Our checkers read noisy SDXL latents through a small head and land near chance on held-out pictures. This
takes noise and the small head out: every finished picture of training_set_catdog.json and
eval_set_catdog.json is embedded by frozen CLIP ViT-L/14 and frozen DINOv2 ViT-B/14 (both trained on
hundreds of millions of photos) and a class-balanced logistic regression is fitted on the training pictures
and scored on the held-out ones, with the model loading and fitting of probe_frozen_features.py.

    high (about 0.9 or more)  the data and labels carry the answer; reading noisy latents is the problem
    low (about 0.7)           the data or labels are the problem; no architecture fixes that

It is also fitted on 25% and 50% of the training pictures (3 draws each, kept in proportion by source and
label): a curve still rising at 100% says more data helps.

Writes, under MOUNT_ROOT/outputs/classifier_guidance_ladder/rung2/probe/: catdog.json
"""
from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import augment_encode  # noqa: E402
import probe_frozen_features as pff  # noqa: E402

HERE = Path(__file__).resolve().parent


def subsample(tiles, frac, seed):
    groups = collections.defaultdict(list)
    for i, t in enumerate(tiles):
        groups[(t["source"], t["label"])].append(i)
    rng, keep = random.Random(seed), []
    for idx in groups.values():
        keep += rng.sample(idx, max(1, round(frac * len(idx))))
    return sorted(keep)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pff.picture_path = augment_encode.picture_path
    train = json.loads((HERE / "training_set_catdog.json").read_text())["tiles"]
    evals = json.loads((HERE / "eval_set_catdog.json").read_text())["tiles"]
    f_tr, f_te = pff.embed(train, device), pff.embed(evals, device)
    y_tr = torch.tensor([float(t["label"]) for t in train])
    y_te = torch.tensor([float(t["label"]) for t in evals])
    report = {"n_train": len(train), "n_eval": len(evals), "eval_clean": int(y_te.sum()), "features": {}}
    for name in f_tr:
        own = pff.fit_and_score(f_tr[name], y_tr, f_tr[name], y_tr)[0]
        bal, tpr, tnr, pred = pff.fit_and_score(f_tr[name], y_tr, f_te[name], y_te)
        by_source = collections.defaultdict(lambda: [0, 0])
        for t, p, y in zip(evals, pred.tolist(), y_te.tolist()):
            by_source[t["source"]][0] += int(p == y)
            by_source[t["source"]][1] += 1
        curve = {}
        for frac in (0.25, 0.5):
            runs = []
            for seed in range(3):
                k = subsample(train, frac, seed)
                runs.append(pff.fit_and_score(f_tr[name][k], y_tr[k], f_te[name], y_te)[0])
            curve[str(frac)] = float(np.mean(runs))
        curve["1.0"] = bal
        report["features"][name] = {"training_pictures_balanced_accuracy": own, "held_out_balanced_accuracy": bal,
                                    "held_out_clean_called_clean": tpr, "held_out_rejects_called_reject": tnr,
                                    "held_out_right_by_source": {k: f"{r} of {n}" for k, (r, n) in by_source.items()},
                                    "learning_curve": curve}
        print(f"{name:26s} training {own:.3f}  held-out {bal:.3f} (clean {tpr:.2f}, reject {tnr:.2f})  "
              f"curve 25% {curve['0.25']:.3f} 50% {curve['0.5']:.3f} 100% {bal:.3f}  "
              f"by source {dict((k, f'{r}/{n}') for k, (r, n) in by_source.items())}", flush=True)
    out = pff.OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "catdog.json").write_text(json.dumps(report, indent=2))
    print("PROBE DONE", out / "catdog.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
