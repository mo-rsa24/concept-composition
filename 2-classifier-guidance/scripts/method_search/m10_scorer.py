#!/usr/bin/env python3
"""Step M10: the scorer that counts red and blue objects, each 0 to 3, in a 64 by 64 picture.

The design is artifacts/notes/progress/method-search/05-two-experts-by-colour/design.md. It trains on real renders of every combination of
0 to 3 red with 0 to 3 blue, c00 to c33, 250 per combination: the first 200 of each train it and the last 50 are held out, 3,200 and 800.
The network is M7's scorer (clevr_scorer.py) with its last layer giving four logits for the red count and four for the blue count.

  overfit  fits 16 real pictures, 1 per combination, and must count both colours right on at least 15 of 16
  train    trains on the 3,200, and must count both colours exactly on at least 95% of the 800 held out
  experts  each expert alone, 256 samples per count: at least 90% hold exactly k of its colour and none of the other
  score    counts every picture in a folder of samples, writing counts.jsonl

Usage: python m10_scorer.py overfit|train --renders <m7/s2/renders> --out <dir>
       python m10_scorer.py experts --samples <dir holding r-count-1 ... b-count-3> --out <scorer dir>
       python m10_scorer.py score --samples <dir of png> --out <scorer dir>
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

COMBINATIONS = [(r, b) for r in range(4) for b in range(4)]
PER_SET = 250
TRAIN_PER_SET = 200
MIN_HELD_OUT_EXACT = 0.95
MIN_OVERFIT_RIGHT = 15
EPOCHS = 60
OVERFIT_STEPS = 300
BATCH = 64
LEARNING_RATE = 1e-3
SEED = 20261125
EXPERT_SAMPLES = 256
MIN_EXPERT_RATE = 0.90


class Scorer(nn.Module):
    """Four convolution blocks over a 64 by 64 picture, then four logits for the red count and four for the blue count."""

    def __init__(self):
        super().__init__()
        layers, channels = [], 3
        for width in (32, 64, 128, 128):
            layers += [nn.Conv2d(channels, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2)]
            channels = width
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, 8))

    def forward(self, x):
        out = self.head(self.features(x))
        return out[:, :4], out[:, 4:]


def load(renders, indices):
    """Pictures and (red, blue) counts for the given per-set indices, each count checked against the scene file's colours."""
    pictures, labels = [], []
    for r, b in COMBINATIONS:
        name = f"c{r}{b}"
        for index in indices:
            stem = f"{name}_{index:06d}"
            scene = json.loads((renders / name / "scenes" / f"{stem}.json").read_text())
            colours = [o["color"] for o in scene["objects"]]
            if (colours.count("red"), colours.count("blue"), len(colours)) != (r, b, r + b):
                raise SystemExit(f"{stem}: colours {colours} in the scene file, {r} red and {b} blue in its set")
            pictures.append(to_64(renders / name / "images" / f"{stem}.png"))
            labels.append((r, b))
    x = torch.from_numpy(np.stack(pictures)).permute(0, 3, 1, 2) * 2 - 1
    return x, torch.tensor(labels, dtype=torch.long)


def loss_of(model, x, y):
    red, blue = model(x)
    return nn.functional.cross_entropy(red, y[:, 0]) + nn.functional.cross_entropy(blue, y[:, 1])


@torch.no_grad()
def counts(model, x):
    model.eval()
    out = [model(x[start:start + 256]) for start in range(0, len(x), 256)]
    return torch.stack([torch.cat([o[0] for o in out]).argmax(1), torch.cat([o[1] for o in out]).argmax(1)], 1)


def overfit(args):
    x, y = load(args.renders, [0])
    model = Scorer()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    first = None
    model.train()
    for _ in range(OVERFIT_STEPS):
        loss = loss_of(model, x, y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        first = float(loss) if first is None else first
    right = int((counts(model, x) == y).all(1).sum())
    result = {"pictures": len(x), "first_loss": round(first, 4), "last_loss": round(float(loss), 6), "both_right": right,
              "min_right": MIN_OVERFIT_RIGHT, "passes": right >= MIN_OVERFIT_RIGHT}
    (args.out / "overfit.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["passes"] else 1)


def train(args):
    begin = time.time()
    x_train, y_train = load(args.renders, range(TRAIN_PER_SET))
    x_held, y_held = load(args.renders, range(TRAIN_PER_SET, PER_SET))
    run = None
    try:
        import wandb
        run = wandb.init(entity="prime_lab", project="poe-repair-animals-compose", group="loop-6-m10", name="m10-count-scorer-cosine",
                         config={"epochs": EPOCHS, "batch": BATCH, "lr": LEARNING_RATE, "train": len(x_train), "held_out": len(x_held)},
                         settings=wandb.Settings(x_service_wait=300))
    except Exception as error:   # the scorer's result is written to disk either way
        print(f"W&B unavailable: {error}", flush=True)
    model = Scorer()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)   # a fixed rate swung held-out exact from 0.95 to 0.52
    generator = torch.Generator().manual_seed(SEED)
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(order), BATCH):
            batch = order[start:start + BATCH]
            loss = loss_of(model, x_train[batch], y_train[batch])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss) * len(batch)
        schedule.step()
        exact = float((counts(model, x_held) == y_held).all(1).float().mean())
        if run:
            run.log({"train/loss": total / len(order), "held_out/exact": exact}, step=epoch)
        print(f"epoch {epoch} loss {total / len(order):.4f} held-out exact {exact:.4f}", flush=True)
    called = counts(model, x_held)
    exact = float((called == y_held).all(1).float().mean())
    per_set = {f"c{r}{b}": round(float((called[(y_held[:, 0] == r) & (y_held[:, 1] == b)] == torch.tensor([r, b])).all(1).float().mean()), 3)
               for r, b in COMBINATIONS}
    torch.save({"state": model.state_dict(), "normalise": "2x-1 on 0-to-1 RGB"}, args.out / "scorer.pt")
    result = {"train": len(x_train), "held_out": len(x_held), "held_out_exact": round(exact, 4), "per_set": per_set,
              "red_right": round(float((called[:, 0] == y_held[:, 0]).float().mean()), 4),
              "blue_right": round(float((called[:, 1] == y_held[:, 1]).float().mean()), 4),
              "min_exact": MIN_HELD_OUT_EXACT, "passes": exact >= MIN_HELD_OUT_EXACT,
              "seconds": round(time.time() - begin, 1), "wandb_run": run.id if run else None}
    (args.out / "scorer.json").write_text(json.dumps(result, indent=1) + "\n")
    if run:
        run.summary.update({"verdict": "passes" if result["passes"] else "misses", "bar": MIN_HELD_OUT_EXACT})
        run.finish()
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["passes"] else 1)


def load_scorer(folder):
    state = torch.load(folder / "scorer.pt", map_location="cpu", weights_only=False)
    model = Scorer()
    model.load_state_dict(state["state"])
    return model.eval()


def call_folder(model, folder, count=None):
    paths = sorted(folder.glob("*.png"))
    if count is not None:
        if len(paths) < count:
            raise SystemExit(f"{folder}: {len(paths)} samples, {count} needed")
        paths = paths[:count]
    x = torch.from_numpy(np.stack([to_64(p) for p in paths])).permute(0, 3, 1, 2) * 2 - 1
    return counts(model, x), [p.name for p in paths]


def experts(args):
    """Each expert alone, 256 samples per count: the share holding exactly k of its colour and none of the other, against 90%."""
    model = load_scorer(args.out)
    rows = {}
    for colour, column in (("r", 0), ("b", 1)):
        for k in (1, 2, 3):
            called, _ = call_folder(model, args.samples / f"{colour}-count-{k}", EXPERT_SAMPLES)
            target = torch.tensor([k, 0] if colour == "r" else [0, k])
            share = float((called == target).all(1).float().mean())
            rows[f"{colour}-count-{k}"] = {"samples": EXPERT_SAMPLES, "exactly_k_and_none_of_the_other": round(share, 4),
                                           "own_count_right": round(float((called[:, column] == k).float().mean()), 4),
                                           "any_of_the_other": int((called[:, 1 - column] > 0).sum()), "passes": share >= MIN_EXPERT_RATE}
    result = {"min_rate": MIN_EXPERT_RATE, "experts": rows, "passes": all(r["passes"] for r in rows.values())}
    (args.samples / "experts.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["passes"] else 1)


def score(args):
    model = load_scorer(args.out)
    called, names = call_folder(model, args.samples)
    with open(args.samples / "counts.jsonl", "w") as f:
        for name, (r, b) in zip(names, called.tolist()):
            f.write(json.dumps({"file": name, "red": r, "blue": b}) + "\n")
    print(json.dumps({"pictures": len(names), "folder": str(args.samples)}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["overfit", "train", "experts", "score"])
    ap.add_argument("--renders", type=Path)
    ap.add_argument("--samples", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count())))
    {"overfit": overfit, "train": train, "experts": experts, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
