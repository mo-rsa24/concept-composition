#!/usr/bin/env python3
"""Step M7: the scorer that says, for each of the three positions, whether an object is there.

The design is artifacts/notes/progress/method-search/01c-clevr-paste-first/design.md. It trains on real renders of every subset
of the three positions, 250 per subset. The first 200 of each subset train it, and the last 50 are held out, which gives 1,600 and 400.

  overfit  fits 16 real pictures, 2 per subset, and must get at least 15 of 16 right at every position
  train    trains on the 1,600, and must reach 99% per position on the 400 held out

Usage: python clevr_scorer.py overfit|train --renders <dir> --out <dir>
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clevr_data import to_64  # noqa: E402

SUBSETS = 8
PER_SUBSET = 250
TRAIN_PER_SUBSET = 200
MIN_HELD_OUT_ACCURACY = 0.99
MIN_OVERFIT_RIGHT = 15
THRESHOLD = 0.5
EPOCHS = 40
OVERFIT_STEPS = 300
BATCH = 64
LEARNING_RATE = 1e-3
SEED = 20261124


class Scorer(nn.Module):
    """Four convolution blocks over a 64 by 64 picture, then one logit per position."""

    def __init__(self):
        super().__init__()
        layers, channels = [], 3
        for width in (32, 64, 128, 128):
            layers += [nn.Conv2d(channels, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2)]
            channels = width
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, 3))

    def forward(self, x):
        return self.head(self.features(x))


def load(renders, indices):
    """Pictures and position labels for the given per-subset indices, with each label checked against the scene file."""
    pictures, labels = [], []
    for subset in range(SUBSETS):
        name = f"s{subset}"
        bits = [(subset >> position) & 1 for position in range(3)]
        for index in indices:
            stem = f"{name}_{index:06d}"
            scene = json.loads((renders / name / "scenes" / f"{stem}.json").read_text())
            if len(scene["objects"]) != sum(bits):
                raise SystemExit(f"{stem}: {len(scene['objects'])} objects in the scene file, {sum(bits)} in its subset")
            pictures.append(to_64(renders / name / "images" / f"{stem}.png"))
            labels.append(bits)
    x = torch.from_numpy(np.stack(pictures)).permute(0, 3, 1, 2) * 2 - 1
    return x, torch.tensor(labels, dtype=torch.float32)


@torch.no_grad()
def accuracy(model, x, y):
    model.eval()
    logits = torch.cat([model(x[start:start + 256]) for start in range(0, len(x), 256)])
    return ((torch.sigmoid(logits) > THRESHOLD).float() == y).float().mean(0).tolist()


def overfit(args):
    x, y = load(args.renders, [0, 1])
    model, loss_fn = Scorer(), nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    first = None
    model.train()
    for _ in range(OVERFIT_STEPS):
        loss = loss_fn(model(x), y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        first = float(loss) if first is None else first
    per_position = accuracy(model, x, y)
    right = [round(a * len(x)) for a in per_position]
    result = {"pictures": len(x), "first_loss": round(first, 4), "last_loss": round(float(loss), 6), "right_per_position": right,
              "min_right": MIN_OVERFIT_RIGHT, "passes": min(right) >= MIN_OVERFIT_RIGHT}
    (args.out / "overfit.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["passes"] else 1)


def train(args):
    begin = time.time()
    x_train, y_train = load(args.renders, range(TRAIN_PER_SUBSET))
    x_held, y_held = load(args.renders, range(TRAIN_PER_SUBSET, PER_SUBSET))
    run = None
    try:
        import wandb
        run = wandb.init(entity="prime_lab", project="poe-repair-animals-compose", group="loop-6-m7", name="m7-clevr-scorer",
                         config={"epochs": EPOCHS, "batch": BATCH, "lr": LEARNING_RATE, "train": len(x_train), "held_out": len(x_held)},
                         settings=wandb.Settings(x_service_wait=300))
    except Exception as error:   # the scorer's result is written to disk either way
        print(f"W&B unavailable: {error}", flush=True)
    model, loss_fn = Scorer(), nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(SEED)
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(order), BATCH):
            batch = order[start:start + BATCH]
            loss = loss_fn(model(x_train[batch]), y_train[batch])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss) * len(batch)
        held = accuracy(model, x_held, y_held)
        if run:
            run.log({"train/loss": total / len(order), **{f"held_out/accuracy_p{i + 1}": a for i, a in enumerate(held)}}, step=epoch)
        print(f"epoch {epoch} loss {total / len(order):.4f} held-out {[round(a, 4) for a in held]}", flush=True)
    held = accuracy(model, x_held, y_held)
    torch.save({"state": model.state_dict(), "threshold": THRESHOLD, "normalise": "2x-1 on 0-to-1 RGB"}, args.out / "scorer.pt")
    result = {"train": len(x_train), "held_out": len(x_held), "held_out_accuracy_per_position": [round(a, 4) for a in held],
              "min_accuracy": MIN_HELD_OUT_ACCURACY, "passes": min(held) >= MIN_HELD_OUT_ACCURACY,
              "seconds": round(time.time() - begin, 1), "wandb_run": run.id if run else None}
    (args.out / "scorer.json").write_text(json.dumps(result, indent=1) + "\n")
    if run:
        run.summary.update({"verdict": "passes" if result["passes"] else "misses", "bar": MIN_HELD_OUT_ACCURACY})
        run.finish()
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["passes"] else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["overfit", "train"])
    ap.add_argument("--renders", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count())))
    {"overfit": overfit, "train": train}[args.stage](args)


if __name__ == "__main__":
    main()
