#!/usr/bin/env python
"""Step W2: can NoisyCLIP or frozen CLIP tell a right sentence from a wrong one at steps 0 to 15?

Each picture is noised to each step with four draws, shared by both models, and scored against every sentence.
A property's checker score is the right sentence's cosine minus the highest wrong sentence's. Its threshold is
chosen at each step on one half of the seeds and read on the other half.

Writes under loop-9/w2/read[-smoke]/: scores.npz, result.json, balanced-accuracy-by-step.png.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (AND_SENTENCES, FROZEN, LOOP, NOISYCLIP_DIR, REPO, SENTENCES, all_sentences, eval_tiles,  # noqa: E402
                    image_features, latent_to_clip, latent_to_rgb, load_clip, load_tile_latent, sha, text_features)

sys.path.insert(0, str(REPO))
from poe_repair._sdxl.runtime import load_ddim_scheduler  # noqa: E402

# The bars, fixed in 02-can-it-read/design.md before any score.
READ_BAR = 0.75
SE_MULTIPLE = 1.96
PUSH_GATE = 0.60
SANITY_BAR = 0.60
INPUT_COSINE_BAR = 0.99
MIN_PER_LABEL = 8
N_BOOTSTRAP = 10_000

STEPS = list(range(16))
SANITY_STEPS = [20, 30, 40, 49]
DRAWS = 4
BATCH = 64
MODELS = ("noisyclip", "frozen_clip")
PROPERTIES = ("counting", "prompt_match", "spatial", "position_size", "binding")
TITLES = {"counting": "Counting", "prompt_match": "Prompt match", "spatial": "Spatial relation",
          "position_size": "Position and size", "binding": "Attribute binding"}


def build_items(prep: Path, smoke: bool):
    """Every latent to score, and each property's (item index, label, seed) list."""
    tiles = [t for t in eval_tiles() if t["kind"] in ("positive", "wrong pair", "hybrid", "single")]
    if smoke:
        tiles = sorted(tiles, key=lambda t: sha(t["id"]))[:24]
    keys, latents, tasks = [], [], {p: [] for p in PROPERTIES}
    for t in tiles:
        i = len(keys)
        keys.append(t["id"])
        latents.append(load_tile_latent(t))
        tasks["counting"].append((i, int(t["kind"] in ("positive", "wrong pair")), t["seed"]))
        if t["kind"] in ("positive", "wrong pair"):
            tasks["prompt_match"].append((i, int(t["kind"] == "positive"), t["seed"]))
    derived = torch.load(prep / "derived.pt", map_location="cpu", weights_only=True)
    index = {}
    for k, z in zip(derived["ids"], derived["latents"]):
        index[k] = len(keys)
        keys.append(k)
        latents.append(z[None].float())
    records = json.loads((prep / "labels.json").read_text())["records"]
    left_out = 0
    for r in records:
        u, m = index[f"{r['id']}::unflipped"], index[f"{r['id']}::mirror"]
        tasks["spatial"] += [(u, int(r["cat_left"]), r["seed"]), (m, int(not r["cat_left"]), r["seed"])]
        if r["larger"] is None:
            left_out += 1
        else:
            cat_large = r["larger"] == "cat"
            tasks["position_size"] += [(u, int(cat_large and r["cat_left"]), r["seed"]),
                                       (m, int(cat_large and not r["cat_left"]), r["seed"])]
        tasks["binding"] += [(index[f"{r['id']}::red_cat_blue_dog"], 1, r["seed"]),
                             (index[f"{r['id']}::blue_cat_red_dog"], 0, r["seed"])]
    return keys, torch.cat(latents), tasks, left_out


def split_seeds(task) -> dict:
    """Within each seed's set of labels, seeds sorted by SHA-256 alternate between half 0 and half 1."""
    labels = {}
    for _, y, s in task:
        labels.setdefault(s, set()).add(y)
    groups = {}
    for s, ys in labels.items():
        groups.setdefault(tuple(sorted(ys)), []).append(s)
    half = {}
    for g in groups.values():
        for k, s in enumerate(sorted(g, key=lambda v: sha(str(v)))):
            half[s] = k % 2
    return half


def balanced(pred: np.ndarray, y: np.ndarray) -> float:
    return float(((pred[y == 1]).mean() + (~pred[y == 0]).mean()) / 2)


def best_threshold(d: np.ndarray, y: np.ndarray) -> float:
    v = np.unique(d)
    cuts = np.concatenate([[v[0] - 1e-6], (v[:-1] + v[1:]) / 2, [v[-1] + 1e-6]])
    scores = np.array([balanced(d > c, y) for c in cuts])
    best = np.flatnonzero(scores >= scores.max() - 1e-12)
    return float(cuts[best[np.argmin(np.abs(cuts[best]))]])


def per_seed_counts(d, y, seeds, thresholds, order):
    """(seeds, steps, 4): right label-1, all label-1, right label-0, all label-0."""
    out = np.zeros((len(order), d.shape[0], 4))
    pos = {s: k for k, s in enumerate(order)}
    for st in range(d.shape[0]):
        pred = d[st] > thresholds[st]
        for p, label, s in zip(pred, y, seeds):
            k = pos[s]
            if label == 1:
                out[k, st, 0] += bool(p)
                out[k, st, 1] += 1
            else:
                out[k, st, 2] += not bool(p)
                out[k, st, 3] += 1
    return out


def bootstrap_means(counts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Mean balanced accuracy over steps for each resample of the seeds."""
    c = np.einsum("rs,sti->rti", weights, counts)
    with np.errstate(invalid="ignore", divide="ignore"):
        ba = (c[..., 0] / c[..., 1] + c[..., 2] / c[..., 3]) / 2
    return np.nanmean(ba, axis=1)


def analyse(scores_by_model, steps, task, smoke):
    """scores_by_model[m]: (steps, items) checker score averaged over draws."""
    idx = np.array([i for i, _, _ in task])
    y = np.array([l for _, l, _ in task])
    seeds = np.array([s for _, _, s in task])
    half_of = split_seeds(task)
    half = np.array([half_of[s] for s in seeds])
    counts = {h: {"label_1": int(((half == h) & (y == 1)).sum()), "label_0": int(((half == h) & (y == 0)).sum())} for h in (0, 1)}
    enough = all(v >= MIN_PER_LABEL for h in counts.values() for v in h.values())
    n_main = sum(1 for s in steps if s <= 15)
    result, per_seed = {"counts_by_half": counts, "enough_pictures": enough}, {}
    read_order = sorted({int(s) for s in seeds[half == 1]})
    for m, score in scores_by_model.items():
        d = score[:, idx]
        tr, te = half == 0, half == 1
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            result[m] = {"note": "a half lacks one label"}
            continue
        th = [best_threshold(d[k, tr], y[tr]) for k in range(len(steps))]
        ba_tr = [balanced(d[k, tr] > th[k], y[tr]) for k in range(len(steps))]
        ba_te = [balanced(d[k, te] > th[k], y[te]) for k in range(len(steps))]
        result[m] = {"thresholds": th, "balanced_accuracy_threshold_half": ba_tr, "balanced_accuracy_reading_half": ba_te,
                     "mean_steps_0_to_15": float(np.mean(ba_te[:n_main])),
                     "push_steps": [s for s, b in zip(steps[:n_main], ba_tr[:n_main]) if b >= PUSH_GATE]}
        per_seed[m] = per_seed_counts(d[:n_main][:, te], y[te], seeds[te], th[:n_main], read_order)
    if len(per_seed) == 2:
        rng = np.random.default_rng(0)
        n = len(read_order)
        weights = rng.multinomial(n, np.full(n, 1 / n), size=N_BOOTSTRAP)
        a, b = bootstrap_means(per_seed[MODELS[0]], weights), bootstrap_means(per_seed[MODELS[1]], weights)
        se = float(np.nanstd(a - b, ddof=1))
        gap = result[MODELS[0]]["mean_steps_0_to_15"] - result[MODELS[1]]["mean_steps_0_to_15"]
        result["noisyclip_minus_frozen"] = {"gap": gap, "standard_error": se, "beats_frozen": bool(gap > SE_MULTIPLE * se)}
        for m, boot in zip(MODELS, (a, b)):
            result[m]["standard_error"] = float(np.nanstd(boot, ddof=1))
            result[m]["reads"] = bool(result[m]["mean_steps_0_to_15"] >= READ_BAR)
    return result


@torch.no_grad()
def input_path_check(model, z, scheduler, timesteps, device) -> dict:
    from transformers import CLIPImageProcessor

    proc = CLIPImageProcessor.from_pretrained(FROZEN)
    out = {}
    for step in (0, 15):
        x0 = z[:16].to(device)
        gen = torch.Generator().manual_seed(step + 5000)
        noise = torch.randn(x0.shape, generator=gen).to(device)
        t = timesteps[step].repeat(x0.shape[0]).to(device)
        x = scheduler.add_noise(x0, noise, t)
        ours = image_features(model, latent_to_clip(x))
        rgb = latent_to_rgb(x).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        px = proc(images=list(rgb), return_tensors="pt").pixel_values.to(device)
        theirs = image_features(model, px)
        out[f"step_{step}"] = float((ours * theirs).sum(-1).mean())
    out["passes"] = all(v >= INPUT_COSINE_BAR for k, v in out.items() if k.startswith("step"))
    return out


def compare_checkpoints(a, b) -> dict:
    sa, sb = a.state_dict(), b.state_dict()
    out = {}
    for name, prefix in (("text_model", "text_model."), ("text_projection", "text_projection."),
                         ("visual_projection", "visual_projection."), ("vision_model", "vision_model.")):
        keys = [k for k in sa if k.startswith(prefix) and k in sb and "position_ids" not in k]
        out[name] = {"tensors": len(keys), "differ": sum(not torch.equal(sa[k], sb[k]) for k in keys)}
    out["logit_scale"] = [float(sa["logit_scale"]), float(sb["logit_scale"])]
    return out


def draw(result, steps, path):
    n_main = sum(1 for s in steps if s <= 15)
    fig, axes = plt.subplots(1, 5, figsize=(19, 3.9), sharey=True)
    colour = {"noisyclip": "#1d7a3e", "frozen_clip": "#6b7a88"}
    label = {"noisyclip": "NoisyCLIP", "frozen_clip": "frozen CLIP"}
    for ax, p in zip(axes, PROPERTIES):
        ax.axhline(READ_BAR, color="#c0262d", lw=1, ls="--")
        ax.axhline(0.5, color="0.6", lw=1, ls=":")
        for m in MODELS:
            r = result[p].get(m, {})
            if "balanced_accuracy_reading_half" not in r:
                continue
            ys = r["balanced_accuracy_reading_half"][:n_main]
            ax.plot(steps[:n_main], ys, "o-", ms=3, color=colour[m])
            ax.annotate(f"{label[m]} {r['mean_steps_0_to_15']:.2f}", (steps[n_main - 1], ys[-1]), xytext=(-2, 7 if m == "noisyclip" else -13),
                        textcoords="offset points", ha="right", fontsize=8.5, color=colour[m])
        ax.set_title(TITLES[p], fontsize=11)
        ax.set_xlabel("sampler step (0 = noise)")
        ax.set_ylim(0.2, 1.0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("balanced accuracy, reading half")
    axes[0].text(0, READ_BAR + 0.012, "bar 0.75", color="#c0262d", fontsize=8)
    axes[0].text(0, 0.51, "chance 0.5", color="0.45", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    device = torch.device("cuda")
    out = LOOP / ("w2/read-smoke" if args.smoke else "w2/read")
    out.mkdir(parents=True, exist_ok=True)
    prep = LOOP / ("w2/prep-smoke" if args.smoke else "w2/prep")
    if not (NOISYCLIP_DIR / "model.safetensors").exists():
        from huggingface_hub import snapshot_download
        snapshot_download("asiimo/noisyclip", local_dir=str(NOISYCLIP_DIR), allow_patterns=["config.json", "model.safetensors"])

    sentences = all_sentences()
    pos = {s: i for i, s in enumerate(sentences)}
    keys, Z, tasks, left_out = build_items(prep, args.smoke)
    steps = [0, 15] if args.smoke else STEPS + SANITY_STEPS
    draws = 1 if args.smoke else DRAWS
    scheduler = load_ddim_scheduler("stabilityai/stable-diffusion-xl-base-1.0")
    scheduler.set_timesteps(50)
    timesteps = scheduler.timesteps.clone().cpu()
    print(f"{len(keys)} latents, {len(sentences)} sentences, steps {steps}, {draws} draws; "
          + ", ".join(f"{p} {len(t)}" for p, t in tasks.items()) + f"; no clear size: {left_out}", flush=True)

    loaded = {m: load_clip("noisyclip" if m == "noisyclip" else "frozen", device) for m in MODELS}
    checkpoint = compare_checkpoints(loaded["noisyclip"][0], loaded["frozen_clip"][0])
    print("checkpoint comparison:", json.dumps(checkpoint), flush=True)
    inputs = {m: input_path_check(loaded[m][0], Z, scheduler, timesteps, device) for m in MODELS}
    print("input path:", json.dumps(inputs), flush=True)

    cos = {m: np.zeros((len(steps), draws, len(keys), len(sentences)), np.float32) for m in MODELS}
    t0 = time.time()
    with torch.no_grad():
        text = {m: text_features(loaded[m][0], loaded[m][1], sentences, device) for m in MODELS}
        for a, step in enumerate(steps):
            for dr in range(draws):
                gen = torch.Generator().manual_seed(1000 * step + dr)
                noise = torch.randn(Z.shape, generator=gen)
                for i in range(0, len(keys), BATCH):
                    z0, n = Z[i:i + BATCH].to(device), noise[i:i + BATCH].to(device)
                    x = scheduler.add_noise(z0, n, timesteps[step].repeat(z0.shape[0]).to(device))
                    pixels = latent_to_clip(x)
                    for m in MODELS:
                        cos[m][a, dr, i:i + z0.shape[0]] = (image_features(loaded[m][0], pixels) @ text[m].T).float().cpu().numpy()
            print(f"[{time.time() - t0:6.0f}s] step {step} done", flush=True)
    np.savez_compressed(out / "scores.npz", keys=np.array(keys), sentences=np.array(sentences), steps=np.array(steps), **cos)

    def checker(m, right, wrong):
        c = cos[m]
        return (c[..., pos[right]] - c[..., [pos[w] for w in wrong]].max(-1)).mean(1)

    def and_score(m):
        c = cos[m]
        i = [pos[s] for s in AND_SENTENCES]
        return (c[..., i[0]] - c[..., i[1]] - c[..., i[2]] + c[..., i[3]]).mean(1)

    result = {"steps": steps, "draws": draws, "no_clear_size_left_out": left_out, "checkpoint": checkpoint, "input_path": inputs,
              "bars": {"read": READ_BAR, "se_multiple": SE_MULTIPLE, "push_gate": PUSH_GATE, "sanity": SANITY_BAR,
                       "input_cosine": INPUT_COSINE_BAR, "min_per_label": MIN_PER_LABEL, "bootstrap": N_BOOTSTRAP}}
    for p in PROPERTIES:
        right, wrong = SENTENCES[p]
        result[p] = analyse({m: checker(m, right, wrong) for m in MODELS}, steps, tasks[p], args.smoke)
    result["and_score"] = {p: analyse({m: and_score(m) for m in MODELS}, steps, tasks[p], args.smoke) for p in ("counting", "prompt_match")}
    if 49 in steps:
        k = steps.index(49)
        sane = result["prompt_match"]["noisyclip"]["balanced_accuracy_threshold_half"][k]
        result["published_behaviour"] = {"prompt_match_step_49_threshold_half": sane, "passes": bool(sane > SANITY_BAR)}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    draw(result, steps, out / "balanced-accuracy-by-step.png")

    for p in PROPERTIES:
        r = result[p]
        line = [f"{m}: {r[m]['mean_steps_0_to_15']:.3f}" for m in MODELS if "mean_steps_0_to_15" in r.get(m, {})]
        print(p, r["counts_by_half"], *line, r.get("noisyclip_minus_frozen", ""), flush=True)

    if not args.smoke:
        import wandb
        run = wandb.init(entity="prime_lab", project="poe-repair-animals-compose", name="loop9-w2-read", group="loop9-w2",
                         config={"step": "W2", "models": MODELS, "steps": steps, "draws": draws, "bars": result["bars"]})
        for p in PROPERTIES:
            for m in MODELS:
                if "mean_steps_0_to_15" in result[p].get(m, {}):
                    run.summary[f"{p}/{m}/mean_ba_steps_0_15"] = result[p][m]["mean_steps_0_to_15"]
            if "noisyclip_minus_frozen" in result[p]:
                run.summary[f"{p}/noisyclip_minus_frozen"] = result[p]["noisyclip_minus_frozen"]["gap"]
        run.log({"balanced_accuracy_by_step": wandb.Image(str(out / "balanced-accuracy-by-step.png"))})
        print("W&B run", run.id, flush=True)
        result["wandb_run"] = run.id
        (out / "result.json").write_text(json.dumps(result, indent=1))
        run.finish()
    print("READ DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
