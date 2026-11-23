#!/usr/bin/env python
"""Step W4: train a small noise-aware image encoder against CLIP's frozen text encoder, and read unseen digits.

Stages, in order:
    smoke    20 updates on the 1% data
    overfit  memorise 16 fixed noisy canvases
    small    1,000 updates on the 1% data, scored on the 1% seen split
    final    8,000 updates on the full data, then read unseen_test once

The training captions name digits 0 to 4 only. The design is noise-text-critic/04-train-our-own/design.md.

Usage:
    python w4_train.py <stage> --data <folder with dataset.npz> --out <folder> [--wandb-name <name>]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mnist"))
sys.path.insert(0, str(HERE.parent / "abstract_checkers"))
from canvas_unet import ABAR, H, W, ddim_timesteps  # noqa: E402

# The bars, fixed in 04-train-our-own/design.md before any run.
MIN_BA = 0.75
MAX_GAP = 0.05
MIN_APART_WRONG = 0.75
OVERFIT_RIGHT = 15
FLOOR_BA = 0.55

DATA_HASHES = {"f70b0fb362e13e255ad332ba14ca93dfb5cf051a5dfd3c86eb30482931d5a5f3",
               "9b70760ad5bed2f8d1b9a0dc6714bbf37f256a9d3486f498b4e4b601278b1b0b"}
ARRAYS = ("x", "asked", "holds", "corner", "src", "kind", "target", "n_digits", "apart")
KINDS = ("apart", "overlaid", "same_half", "single", "wrong")
WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
SEEN = range(5)
STEPS16 = ddim_timesteps(50)[:16]
TAU = 0.07
UPDATES = 8000
BATCH = 256
LR = 2e-4
DRAWS = 4
EVAL_SEED = 990_000
TEXT_MODEL = "openai/clip-vit-large-patch14"


def apart(a: int, b: int) -> str:
    a, b = sorted((a, b))
    return f"the digit {WORDS[a]} and the digit {WORDS[b]}, apart"


def on_top(a: int, b: int) -> str:
    a, b = sorted((a, b))
    return f"the digit {WORDS[a]} on top of the digit {WORDS[b]}"


def same_side(a: int, b: int) -> str:
    a, b = sorted((a, b))
    return f"the digit {WORDS[a]} and the digit {WORDS[b]} on the same side"


def single(a: int) -> str:
    return f"one digit {WORDS[a]}"


def all_captions(digits) -> list[str]:
    out = [single(d) for d in digits]
    for a, b in itertools.combinations(digits, 2):
        out += [apart(a, b), on_top(a, b), same_side(a, b)]
    return out


def true_caption(kind: int, holds) -> str:
    name, (a, b) = KINDS[int(kind)], (int(holds[0]), int(holds[1]))
    if name in ("apart", "wrong"):
        return apart(a, b)
    if name == "overlaid":
        return on_top(a, b)
    if name == "same_half":
        return same_side(a, b)
    return single(a)


def pair_sentences(a: int, b: int) -> tuple[str, list[str]]:
    wrong = [single(a), single(b), on_top(a, b), same_side(a, b)]
    for c in range(10):
        if c not in (a, b):
            wrong += [apart(a, c), apart(b, c)]
    return apart(a, b), wrong


class FilmBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, t_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm = nn.GroupNorm(8, c_out)
        self.film = nn.Linear(t_dim, 2 * c_out)

    def forward(self, x, emb):
        h = self.norm(self.conv(x))
        gamma, beta = self.film(emb).chunk(2, dim=1)
        h = h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return F.avg_pool2d(F.silu(h), 2)


class NoisyDigitEncoder(nn.Module):
    """Noisy canvas and step index to one 768-number vector. The flatten keeps left and right distinct."""

    def __init__(self, out_dim: int = 768, t_dim: int = 128):
        super().__init__()
        self.step = nn.Embedding(len(STEPS16), t_dim)
        widths = (64, 128, 256, 256)
        self.blocks = nn.ModuleList(FilmBlock(c_in, c_out, t_dim) for c_in, c_out in zip((1, *widths[:-1]), widths))
        self.out = nn.Linear(256 * (H // 16) * (W // 16), out_dim)

    def forward(self, x, step_index):
        emb = self.step(step_index)
        for block in self.blocks:
            x = block(x, emb)
        return F.normalize(self.out(x.flatten(1)), dim=-1)


def load_data(folder: Path):
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["data_hash"] not in DATA_HASHES:
        raise SystemExit(f"unexpected data hash {manifest['data_hash']}")
    with np.load(folder / "dataset.npz") as z:
        return {s: {n: z[f"{s}__{n}"] for n in ARRAYS} for s in ("train", "seen_test", "unseen_test")}, manifest["data_hash"]


@torch.no_grad()
def text_bank(sentences: list[str], device) -> torch.Tensor:
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(TEXT_MODEL)
    model = CLIPTextModelWithProjection.from_pretrained(TEXT_MODEL, attn_implementation="eager").to(device).eval()
    out = []
    for i in range(0, len(sentences), 64):
        tok = tokenizer(sentences[i:i + 64], padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)
        out.append(F.normalize(model(**tok).text_embeds, dim=-1))
    return torch.cat(out)


def to_x(a: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(a).float().to(device).div(127.5).sub(1).unsqueeze(1)


def noised(x0, step_index, noise):
    bar = ABAR.to(x0.device)[STEPS16.to(x0.device)[step_index]].view(-1, 1, 1, 1)
    return bar.sqrt() * x0 + (1 - bar).sqrt() * noise


def targets(split, bank_index) -> np.ndarray:
    return np.array([bank_index[true_caption(k, h)] for k, h in zip(split["kind"], split["holds"])], np.int64)


def train(model, split, bank, bank_index, device, updates, seed=0, log_every=500):
    y = torch.from_numpy(targets(split, bank_index)).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    model.train()
    start, recent = time.time(), []
    for update in range(1, updates + 1):
        pick = rng.integers(0, len(split["x"]), BATCH)
        x0 = to_x(split["x"][pick], device)
        step = torch.randint(0, len(STEPS16), (len(pick),), generator=gen, device=device)
        xt = noised(x0, step, torch.randn(x0.shape, generator=gen, device=device))
        loss = F.cross_entropy(model(xt, step) @ bank.T / TAU, y[torch.from_numpy(pick).to(device)])
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        recent.append(loss.item())
        if update % log_every == 0 or update == updates:
            print(f"update {update} loss {np.mean(recent):.4f} ({time.time() - start:.0f}s)", flush=True)
            recent = []
    return time.time() - start


@torch.no_grad()
def checker_scores(model, split, full_bank, full_index, device, draws=DRAWS, batch=512) -> np.ndarray:
    """(canvases, steps): right sentence's cosine minus the highest wrong one's, averaged over noise draws."""
    model.eval()
    n = len(split["x"])
    right = np.zeros(n, np.int64)
    wrong = np.zeros((n, len(full_index)), bool)
    for pair in {(int(a), int(b)) for a, b in split["asked"]}:
        r, w = pair_sentences(*pair)
        mask = (split["asked"][:, 0] == pair[0]) & (split["asked"][:, 1] == pair[1])
        right[mask] = full_index[r]
        wrong[np.ix_(mask, [full_index[s] for s in w])] = True
    right_t, wrong_t = torch.from_numpy(right).to(device), torch.from_numpy(wrong).to(device)
    out = np.zeros((n, len(STEPS16)), np.float32)
    for s in range(len(STEPS16)):
        total = torch.zeros(n, device=device)
        for d in range(draws):
            gen = torch.Generator().manual_seed(EVAL_SEED + 100 * s + d)
            for i in range(0, n, batch):
                x0 = to_x(split["x"][i:i + batch], device)
                step = torch.full((len(x0),), s, device=device, dtype=torch.long)
                xt = noised(x0, step, torch.randn(x0.shape, generator=gen).to(device))
                cos = model(xt, step) @ full_bank.T
                hit = cos.gather(1, right_t[i:i + batch, None])[:, 0]
                miss = cos.masked_fill(~wrong_t[i:i + batch], -2.0).max(1).values
                total[i:i + batch] += hit - miss
        out[:, s] = (total / draws).cpu().numpy()
    return out


def balanced(pred: np.ndarray, y: np.ndarray) -> float:
    return 0.5 * (float(pred[y == 1].mean()) + float((~pred[y == 0]).mean()))


def best_threshold(d: np.ndarray, y: np.ndarray) -> float:
    v = np.unique(d)
    cuts = np.concatenate([[v[0] - 1e-6], (v[:-1] + v[1:]) / 2, [v[-1] + 1e-6]])
    scores = np.array([balanced(d > c, y) for c in cuts])
    best = np.flatnonzero(scores >= scores.max() - 1e-12)
    return float(cuts[best[np.argmin(np.abs(cuts[best]))]])


def seen_halves(split) -> np.ndarray:
    """0 for the even-indexed canvases of each asked pair, 1 for the odd-indexed."""
    half = np.zeros(len(split["x"]), np.int64)
    for a, b in {(int(p), int(q)) for p, q in split["asked"]}:
        idx = np.flatnonzero((split["asked"][:, 0] == a) & (split["asked"][:, 1] == b))
        half[idx[1::2]] = 1
    return half


def summarise(score, split, thresholds, keep=None) -> dict:
    keep = np.ones(len(score), bool) if keep is None else keep
    y, kind = split["target"][keep], split["kind"][keep]
    d = score[keep]
    versus = np.isin(kind, [KINDS.index("apart"), KINDS.index("wrong")])
    per = [balanced(d[:, s] > thresholds[s], y) for s in range(d.shape[1])]
    per_v = [balanced(d[versus, s] > thresholds[s], y[versus]) for s in range(d.shape[1])]
    return {"canvases": int(keep.sum()), "balanced_accuracy": float(np.mean(per)), "apart_vs_wrong": float(np.mean(per_v)),
            "per_step": per, "apart_vs_wrong_per_step": per_v}


def read_seen(model, data, full_bank, full_index, device):
    split = data["seen_test"]
    score = checker_scores(model, split, full_bank, full_index, device)
    half = seen_halves(split)
    thresholds = [best_threshold(score[half == 0, s], split["target"][half == 0]) for s in range(score.shape[1])]
    return summarise(score, split, thresholds, half == 1), thresholds


def save(path: Path, result: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, (list, dict))}), flush=True)


def overfit(data, bank, bank_index, device, out: Path) -> dict:
    split = data["train"]
    rng = np.random.default_rng(0)
    pick = rng.choice(len(split["x"]), 16, replace=False)
    x0 = to_x(split["x"][pick], device)
    y = torch.from_numpy(targets({k: v[pick] for k, v in split.items()}, bank_index)).to(device)
    step = torch.arange(16, device=device)
    noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(7)).to(device)
    xt = noised(x0, step, noise)
    torch.manual_seed(0)
    model = NoisyDigitEncoder().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(800):
        loss = F.cross_entropy(model(xt, step) @ bank.T / TAU, y)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    model.eval()
    with torch.no_grad():
        right = int(((model(xt, step) @ bank.T).argmax(1) == y).sum())
    result = {"right_of_16": right, "needed": OVERFIT_RIGHT, "passes": right >= OVERFIT_RIGHT, "final_loss": loss.item()}
    save(out / "overfit.json", result)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["smoke", "overfit", "small", "final"])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ladder", type=Path, help="the folder holding overfit.json, for the final stage")
    ap.add_argument("--wandb-name", default=None)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    device = "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    data, data_hash = load_data(args.data)
    train_sentences, full_sentences = all_captions(SEEN), all_captions(range(10))
    if any(w in s for s in train_sentences for w in WORDS[5:]):
        raise SystemExit("a training caption names an unseen digit")
    full_bank = text_bank(full_sentences, device)
    full_index = {s: i for i, s in enumerate(full_sentences)}
    bank = full_bank[[full_index[s] for s in train_sentences]]
    bank_index = {s: i for i, s in enumerate(train_sentences)}
    print(json.dumps({"stage": args.stage, "data_hash": data_hash, "train": len(data["train"]["x"]),
                      "train_captions": len(train_sentences), "all_captions": len(full_sentences)}), flush=True)

    if args.stage == "overfit":
        return 0 if overfit(data, bank, bank_index, device, args.out)["passes"] else 1

    torch.manual_seed(0)
    model = NoisyDigitEncoder().to(device)
    if args.stage == "smoke":
        seconds = train(model, data["train"], bank, bank_index, device, 20, log_every=10)
        save(args.out / "smoke.json", {"updates": 20, "seconds": seconds, "seconds_per_update": seconds / 20})
        return 0
    if args.stage == "small":
        seconds = train(model, data["train"], bank, bank_index, device, 1000)
        seen, _ = read_seen(model, data, full_bank, full_index, device)
        save(args.out / "small.json", {"updates": 1000, "seconds": seconds, "seen": seen})
        return 0

    if args.ladder is None or not json.loads((args.ladder / "overfit.json").read_text()).get("passes"):
        result = {"verdict": "inconclusive", "why": "the overfit test did not pass"}
        save(args.out / "final.json", result)
        return 1
    seconds = train(model, data["train"], bank, bank_index, device, UPDATES)
    torch.save({"model": model.state_dict(), "updates": UPDATES, "data_hash": data_hash}, args.out / "encoder.pt")
    seen, thresholds = read_seen(model, data, full_bank, full_index, device)
    if max(seen["per_step"]) < FLOOR_BA:
        result = {"verdict": "inconclusive", "why": "below 0.55 on the seen half at every step", "seen": seen, "train_seconds": seconds}
        save(args.out / "final.json", result)
        return 0
    unseen_score = checker_scores(model, data["unseen_test"], full_bank, full_index, device)
    unseen = summarise(unseen_score, data["unseen_test"], thresholds)
    gap = abs(seen["balanced_accuracy"] - unseen["balanced_accuracy"])
    checks = {"unseen_ba_at_least_0.75": unseen["balanced_accuracy"] >= MIN_BA, "gap_at_most_0.05": gap <= MAX_GAP,
              "apart_vs_wrong_at_least_0.75": unseen["apart_vs_wrong"] >= MIN_APART_WRONG}
    result = {"verdict": "support" if all(checks.values()) else "null", "checks": checks, "gap": gap, "seen": seen,
              "unseen": unseen, "thresholds": thresholds, "train_seconds": seconds, "data_hash": data_hash,
              "timesteps": STEPS16.tolist()}
    if args.wandb_name:
        try:
            import wandb
            run = wandb.init(project="poe-repair-animals-compose", entity="prime_lab", name=args.wandb_name, group="loop9-w4",
                             config={"step": "W4", "updates": UPDATES, "batch": BATCH, "lr": LR, "tau": TAU},
                             settings=wandb.Settings(init_timeout=60))
            run.summary.update({"verdict": result["verdict"], "seen_ba": seen["balanced_accuracy"],
                                "unseen_ba": unseen["balanced_accuracy"], "unseen_apart_vs_wrong": unseen["apart_vs_wrong"]})
            result["wandb_run"] = run.id
            run.finish()
        except Exception as error:  # the numbers are on disk either way
            print(f"W&B logging failed: {error}", flush=True)
    save(args.out / "final.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
