#!/usr/bin/env python
"""Step W2b: does a linear probe on a frozen model's image vectors read counting and prompt match on unseen pairs?

  extract  read every training and held-out picture at steps 0 to 15, one noise draw shared by all models, and save
           each model's normalised image vectors
  fit      fit one logistic regression per model, property and step on the training split, then read the seen test
           and the 7 unseen pairs once, beside a probe on fixed random numbers

The design is noise-text-critic/02b-linear-probe/design.md.

Usage: python w2b_probe.py extract --out <dir> [--smoke]
       python w2b_probe.py fit --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import REPO, image_features, latent_clip_image, latent_to_clip, load_clip, load_latent_clip, sha  # noqa: E402

sys.path.insert(0, str(REPO / "scripts/shared"))
from disallowed_subjects import is_allowed  # noqa: E402

# The bars, fixed in 02b-linear-probe/design.md before any run.
MIN_BA = 0.75
MAX_GAP = 0.05
FLOOR_BA = 0.55
MIN_PER_CLASS = 30
RANDOM_CEILING = 0.6
L2_GRID = (0.01, 0.1, 1.0, 10.0)
FOLDS = 5
SEEN_SHARE = 0.10

RUNG2 = REPO / "2-classifier-guidance/scripts/rung2"
STEPS = list(range(16))
MODELS = ("noisyclip", "frozen_clip", "latent_clip")
LABELS = {"counting": ({"positive", "wrong pair"}, {"hybrid", "single"}), "prompt_match": ({"positive"}, {"wrong pair"})}


def base_kind(kind: str) -> str:
    return kind.split(" (")[0]


def pictures():
    """Training tiles (split into fit and seen test by hash) and the 7 unseen pairs, after the subject guard."""
    train = json.loads((RUNG2 / "training_set_v4.json").read_text())
    held = json.loads((RUNG2 / "eval_set_v4.json").read_text())
    unseen_pairs = set(held["held_out_pairs"])
    rows = []
    for t in train["tiles"]:
        if is_allowed(t["pair"].replace("__x__", " ")) and t["pair"] not in unseen_pairs:
            split = "seen_test" if int(sha(t["id"])[:8], 16) / 0xFFFFFFFF < SEEN_SHARE else "fit"
            rows.append({**t, "kind": base_kind(t["kind"]), "split": split})
    for t in held["tiles"]:
        if t["pair"] in unseen_pairs and is_allowed(t["pair"].replace("__x__", " ")):
            rows.append({**t, "kind": base_kind(t["kind"]), "split": "unseen"})
    used = set().union(*LABELS["counting"])
    return [r for r in rows if r["kind"] in used]


@torch.no_grad()
def stage_extract(args) -> int:
    from common import find
    from poe_repair._sdxl.runtime import load_ddim_scheduler

    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    device = torch.device("cuda")
    rows, latents, missing = [], [], 0
    for r in pictures():
        try:
            latents.append(torch.load(find(r["latent"]), map_location="cpu", weights_only=True).half())
            rows.append(r)
        except FileNotFoundError:
            missing += 1
    if args.smoke:
        rows, latents = rows[:32], latents[:32]
    Z = torch.cat(latents)
    steps = [0, 15] if args.smoke else STEPS
    scheduler = load_ddim_scheduler("stabilityai/stable-diffusion-xl-base-1.0")
    scheduler.set_timesteps(50)
    timesteps = scheduler.timesteps.clone().cpu()
    models = {m: load_clip("noisyclip" if m == "noisyclip" else "frozen", device) for m in MODELS[:2]}
    models["latent_clip"] = load_latent_clip(device)
    vectors = {m: [] for m in MODELS}
    for step in steps:
        gen = torch.Generator().manual_seed(700_000 + step)
        noise = torch.randn(Z.shape, generator=gen, dtype=torch.float32).half()
        per = {m: [] for m in MODELS}
        for i in range(0, len(Z), 64):
            z0, n = Z[i:i + 64].to(device).float(), noise[i:i + 64].to(device).float()
            x = scheduler.add_noise(z0, n, timesteps[step].repeat(len(z0)).to(device))
            with torch.autocast("cuda", dtype=torch.float16):
                pixels = latent_to_clip(x)
                for m in MODELS[:2]:
                    per[m].append(image_features(models[m][0], pixels).float().cpu())
                per["latent_clip"].append(latent_clip_image(models["latent_clip"][0], x).float().cpu())
        for m in MODELS:
            vectors[m].append(torch.cat(per[m]).half().numpy())
        print(f"step {step} done", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    meta = [{k: r[k] for k in ("id", "pair", "seed", "kind", "split")} for r in rows]
    (args.out / "pictures.json").write_text(json.dumps({"steps": steps, "missing_latents": missing, "pictures": meta}))
    for m in MODELS:
        np.save(args.out / f"vectors-{m}.npy", np.stack(vectors[m]))
    print(f"{len(rows)} pictures, {missing} latents missing", flush=True)
    print("EXTRACT DONE", flush=True)
    return 0


def balanced(pred: np.ndarray, y: np.ndarray) -> float:
    return 0.5 * (float(pred[y == 1].mean()) + float((~pred[y == 0]).mean()))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float) -> tuple[np.ndarray, float]:
    """Class-weighted logistic regression, fitted by L-BFGS on the full batch."""
    Xt, yt = torch.from_numpy(X).double(), torch.from_numpy(y).double()
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    weight = torch.where(yt == 1, 0.5 / yt.mean(), 0.5 / (1 - yt.mean()))
    opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = Xt @ w + b
        loss = (weight * torch.nn.functional.binary_cross_entropy_with_logits(z, yt, reduction="none")).mean() + l2 * (w @ w) / len(yt)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy(), float(b.detach())


def choose_l2(X, y, groups) -> float:
    folds = np.array([int(sha(str(g))[:8], 16) % FOLDS for g in groups])
    best, best_ba = L2_GRID[0], -1.0
    for l2 in L2_GRID:
        scores = []
        for f in range(FOLDS):
            tr, va = folds != f, folds == f
            if len(set(y[tr])) < 2 or len(set(y[va])) < 2:
                continue
            w, b = fit_logistic(X[tr], y[tr], l2)
            scores.append(balanced(X[va] @ w + b > 0, y[va]))
        if scores and np.mean(scores) > best_ba:
            best, best_ba = l2, float(np.mean(scores))
    return best


def probe(vectors: np.ndarray, rows, prop: str) -> dict:
    pos, neg = LABELS[prop]
    keep = np.array([r["kind"] in pos | neg for r in rows])
    y_all = np.array([int(r["kind"] in pos) for r in rows])
    split = np.array([r["split"] for r in rows])
    seeds = np.array([f"{r['pair']}:{r['seed']}" for r in rows])
    counts = {s: {"label_1": int(((split == s) & keep & (y_all == 1)).sum()), "label_0": int(((split == s) & keep & (y_all == 0)).sum())}
              for s in ("fit", "seen_test", "unseen")}
    seen, unseen, l2s = [], [], []
    for k in range(vectors.shape[0]):
        X = vectors[k].astype(np.float64)
        tr = keep & (split == "fit")
        l2 = choose_l2(X[tr], y_all[tr], seeds[tr])
        w, b = fit_logistic(X[tr], y_all[tr], l2)
        for name, out in (("seen_test", seen), ("unseen", unseen)):
            m = keep & (split == name)
            out.append(balanced(X[m] @ w + b > 0, y_all[m]) if len(set(y_all[m])) == 2 else float("nan"))
        l2s.append(l2)
    result = {"counts": counts, "l2": l2s, "seen_per_step": seen, "unseen_per_step": unseen,
              "seen_mean": float(np.nanmean(seen)), "unseen_mean": float(np.nanmean(unseen))}
    enough = all(v >= MIN_PER_CLASS for v in counts["unseen"].values())
    if not enough or max(seen) < FLOOR_BA:
        result["verdict"] = "inconclusive"
    else:
        result["verdict"] = "support" if result["unseen_mean"] >= MIN_BA and abs(result["seen_mean"] - result["unseen_mean"]) <= MAX_GAP else "null"
    return result


def stage_fit(args) -> int:
    meta = json.loads((args.out / "pictures.json").read_text())
    rows = meta["pictures"]
    result = {"steps": meta["steps"], "missing_latents": meta["missing_latents"],
              "bars": {"read": MIN_BA, "gap": MAX_GAP, "floor": FLOOR_BA, "min_per_class": MIN_PER_CLASS, "random_ceiling": RANDOM_CEILING}}
    for m in MODELS:
        v = np.load(args.out / f"vectors-{m}.npy")
        result[m] = {p: probe(v, rows, p) for p in LABELS}
    rng = np.random.default_rng(0)
    fake = rng.standard_normal((len(meta["steps"]), len(rows), 768)).astype(np.float32)
    fake /= np.linalg.norm(fake, axis=-1, keepdims=True)
    result["random_numbers"] = {p: probe(fake, rows, p) for p in LABELS}
    result["random_control_passes"] = all(result["random_numbers"][p]["unseen_mean"] <= RANDOM_CEILING for p in LABELS)
    (args.out / "result.json").write_text(json.dumps(result, indent=1))
    for m in (*MODELS, "random_numbers"):
        for p in LABELS:
            r = result[m][p]
            print(f"{m:13s} {p:13s} seen {r['seen_mean']:.3f} unseen {r['unseen_mean']:.3f} {r['verdict']}", flush=True)
    print("FIT DONE", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["extract", "fit"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    return {"extract": stage_extract, "fit": stage_fit}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
