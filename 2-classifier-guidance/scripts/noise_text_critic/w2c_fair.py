#!/usr/bin/env python
"""Step W2c: a positive control and a fair test for W2's null.

Group 1: the kept readout checker (iteration 4) under W2's exact protocol.
Group 2: sentence scores with prompt ensembling, read at steps 0 to 15, 20 to 29, 30, 40 and 49, on W2's pictures and draws.
Group 3: the same readers on the sampler's real states of 195 held-out cat and dog runs, steps 0 to 20.

The design and its bars are noise-text-critic/02c-fair-test/design.md. Pictures, draws, halves, thresholds, balanced
accuracy and the bootstrap come from w2_read.py unchanged.

Usage: python w2c_fair.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import w2_read as w2  # noqa: E402
from common import (LOOP, REPO, SENTENCES, eval_tiles, image_features, latent_clip_image, latent_clip_text,  # noqa: E402
                    latent_to_clip, load_clip, load_latent_clip, text_features)
from disallowed_subjects import check  # noqa: E402

sys.path.insert(0, str(REPO / "2-classifier-guidance/scripts/rung2"))

# Fixed in 02c-fair-test/design.md before any W2c score.
CONTROL_FLOOR = 0.65
READ_BAR = 0.75
TEMPLATES = ("a photo of {}.", "a picture of {}.", "an image of {}.", "a photograph showing {}.", "{}.")
SCRATCH = Path("/home-mscluster/mmolefe/poe_repair_min_scratch/outputs/classifier_guidance_ladder/rung2")
HEAD = SCRATCH / "readout_pair/readout-catdog-more3-long/head_step_002000.pt"
TRAJ = SCRATCH / "trajectories"

CLEAN = ("a photo of a cat and a dog",
         ("a photo of one animal", "one animal that is part cat and part dog", "a photo of two cats",
          "a photo of two dogs", "a photo of a horse and a sheep"))
TASK_SENTENCES = {"counting": SENTENCES["counting"], "prompt_match": SENTENCES["prompt_match"], "clean_pair": CLEAN}
SENTENCE_MODELS = ("noisyclip", "frozen_clip", "latent_clip")
ROWS = (*SENTENCE_MODELS, *(f"{m}_ens" for m in SENTENCE_MODELS), "readout")
STEPS_A = list(range(16)) + list(range(20, 30)) + [30, 40, 49]
STEPS_B = list(range(21))


def phrase(s: str) -> str:
    return s[len("a photo of "):] if s.startswith("a photo of ") else s


def all_task_sentences():
    out = []
    for right, wrong in TASK_SENTENCES.values():
        out += [right, *wrong]
    out = list(dict.fromkeys(out))
    for s in out:
        check(s)
        for tpl in TEMPLATES:
            check(tpl.format(phrase(s)))
    return out


@torch.no_grad()
def text_bank(loaded, sentences, device):
    """Per model, the plain text vectors and the ensembled ones, both normalised, in sentence order."""
    def encode(m, texts):
        if m == "latent_clip":
            return latent_clip_text(*loaded[m], texts, device).float()
        return text_features(loaded[m][0], loaded[m][1], texts, device).float()

    bank = {}
    for m in SENTENCE_MODELS:
        bank[m] = encode(m, sentences)
        ens = []
        for s in sentences:
            v = encode(m, [tpl.format(phrase(s)) for tpl in TEMPLATES])
            ens.append(F.normalize(v.mean(0), dim=-1))
        bank[f"{m}_ens"] = torch.stack(ens)
    return bank


def load_readout(device):
    from poe_repair.run import make_ctx
    from poe_repair.runtime import encode_prompt_sdxl
    from train_count_head import Readout
    from train_readout_pair import PairHead

    ctx = make_ctx(output_root=LOOP / "w2c/raw")
    for n in ("vae", "unet", "text_encoder", "text_encoder_2"):
        ctx.models[n].requires_grad_(False)
        ctx.models[n].eval()
    seq_e, pool_e = encode_prompt_sdxl("", models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    reader = Readout(ctx.models["unet"], seq_e, pool_e).to(ctx.device)
    head = PairHead().to(ctx.device)
    head.load_state_dict(torch.load(HEAD, map_location="cpu", weights_only=True)["head"])
    head.eval()
    return reader, head, ctx.dtype


@torch.no_grad()
def score_batch(x, t, loaded, bank, readout, device):
    """Each row's cosine to every sentence (sentence rows) or logit (readout) for one batch of latents x at timestep t."""
    pixels = latent_to_clip(x)
    feats = {m: image_features(loaded[m][0], pixels).float() for m in ("noisyclip", "frozen_clip")}
    feats["latent_clip"] = latent_clip_image(loaded["latent_clip"][0], x).float()
    out = {}
    for m in SENTENCE_MODELS:
        out[m] = (feats[m] @ bank[m].T).cpu().numpy()
        out[f"{m}_ens"] = (feats[m] @ bank[f"{m}_ens"].T).cpu().numpy()
    reader, head, dtype = readout
    out["readout"] = head(reader(x.to(dtype), t)).float().cpu().numpy()
    return out


def checker_scores(raw, pos, task):
    """raw[row]: (steps, items, sentences) or (steps, items); returns (steps, items) per row."""
    right, wrong = TASK_SENTENCES[task]
    out = {}
    for r, v in raw.items():
        out[r] = v if r == "readout" else v[..., pos[right]] - v[..., [pos[w] for w in wrong]].max(-1)
    return out


def summarise(res, steps):
    for r in ROWS:
        if "balanced_accuracy_reading_half" in res.get(r, {}):
            ba = res[r]["balanced_accuracy_reading_half"]
            win = [b for s, b in zip(steps, ba) if 20 <= s <= 29]
            if win:
                res[r]["mean_steps_20_to_29"] = float(np.mean(win))
    return res


def group_a(args, loaded, bank, pos, readout, device):
    w2.MODELS, w2.COMPARED = ROWS, tuple(r for r in ROWS if r != "frozen_clip")
    prep = LOOP / "w2/prep"
    keys, Z, tasks, _ = w2.build_items(prep, smoke=False)
    tiles = [t for t in eval_tiles() if t["kind"] in ("positive", "wrong pair", "hybrid", "single")]
    tasks["clean_pair"] = [(i, int(t["kind"] == "positive"), t["seed"]) for i, t in enumerate(tiles)]
    n_eval = len(tiles)
    items = 24 if args.smoke else n_eval
    steps = [0, 15] if args.smoke else STEPS_A
    draws = 1 if args.smoke else w2.DRAWS
    from poe_repair._sdxl.runtime import load_ddim_scheduler
    scheduler = load_ddim_scheduler("stabilityai/stable-diffusion-xl-base-1.0")
    scheduler.set_timesteps(50)
    timesteps = scheduler.timesteps.clone().cpu()
    raw = {r: [] for r in ROWS}
    t0 = time.time()
    for step in steps:
        per = {r: [] for r in ROWS}
        for dr in range(draws):
            gen = torch.Generator().manual_seed(1000 * step + dr)   # W2's draw: drawn over the full stack, then cut
            noise = torch.randn(Z.shape, generator=gen)[:items]
            got = {r: [] for r in ROWS}
            for i in range(0, items, 8):
                z0, n = Z[i:min(i + 8, items)].to(device), noise[i:min(i + 8, items)].to(device)
                t = timesteps[step].repeat(z0.shape[0]).to(device)
                x = scheduler.add_noise(z0, n, t)
                for r, v in score_batch(x, t, loaded, bank, readout, device).items():
                    got[r].append(v)
            for r in ROWS:
                per[r].append(np.concatenate(got[r]))
        for r in ROWS:
            raw[r].append(np.mean(per[r], axis=0))
        print(f"[{time.time() - t0:6.0f}s] group A step {step} done", flush=True)
    raw = {r: np.stack(v) for r, v in raw.items()}
    out = {"steps": steps, "draws": draws, "pictures": items}
    for task in TASK_SENTENCES:
        tk = [(i, y, s) for i, y, s in tasks[task] if i < items]
        out[task] = summarise(w2.analyse(checker_scores(raw, pos, task), steps, tk, args.smoke), steps)
    return out, raw


def group_b(args, loaded, bank, pos, readout, device):
    labels = json.loads((TRAJ / "labels.json").read_text())
    kept = json.loads((TRAJ / "timesteps.json").read_text())
    runs = sorted((s, r) for s, r in labels.items() if r["split"] == "held_out" and r["mode"] == "joint" and r["label"] is not None)
    if args.smoke:
        runs = runs[:4] + runs[-4:]
    steps = [0, 15] if args.smoke else STEPS_B
    idx = [kept["kept_steps"].index(s) for s in steps]
    tts = [kept["timesteps"][k] for k in idx]
    states = torch.stack([torch.load(TRAJ / "states" / f"{s}.pt", map_location="cpu", weights_only=True)[idx].half()
                          for s, _ in runs])   # (runs, steps, 4, 128, 128), read once
    raw = {r: [] for r in ROWS}
    for a, (k, tt) in enumerate(zip(idx, tts)):
        got = {r: [] for r in ROWS}
        for i in range(0, len(runs), 8):
            x = states[i:i + 8, a].float().to(device)
            t = torch.full((x.shape[0],), tt, device=device)
            for r, v in score_batch(x, t, loaded, bank, readout, device).items():
                got[r].append(v)
        for r in ROWS:
            raw[r].append(np.concatenate(got[r]))
        print(f"group B step {steps[a]} done", flush=True)
    raw = {r: np.stack(v) for r, v in raw.items()}
    task = [(i, int(r["label"]), int(r["seed"])) for i, (_, r) in enumerate(runs)]
    res = w2.analyse(checker_scores(raw, pos, "clean_pair"), steps, task, args.smoke)
    return {"steps": steps, "runs": len(runs), "clean": sum(y for _, y, _ in task), "clean_pair": res}


def verdict(a, b):
    ctl = a["clean_pair"].get("readout", {})
    gap = a["clean_pair"].get("readout_minus_frozen", {})
    holds = bool(ctl.get("mean_steps_0_to_15", 0) >= CONTROL_FLOOR and gap.get("beats_frozen", False))
    reads = []
    for g, tasks in (("A", a), ("B", b)):
        for task, res in tasks.items():
            if not isinstance(res, dict) or task in ("steps",):
                continue
            for r in ROWS[:-1]:
                v = res.get(r, {}).get("mean_steps_0_to_15")
                if v is not None and v >= READ_BAR:
                    reads.append(f"{g}:{task}:{r}:{v:.3f}")
    state = "a sentence score reads" if reads else ("W2 stands" if holds else "W2 is inconclusive")
    return {"control_holds": holds, "sentence_rows_that_read": reads, "state": state}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    device = torch.device("cuda")
    out = LOOP / ("w2c/smoke" if args.smoke else "w2c/full")
    out.mkdir(parents=True, exist_ok=True)
    sentences = all_task_sentences()
    pos = {s: i for i, s in enumerate(sentences)}
    loaded = {m: load_clip("noisyclip" if m == "noisyclip" else "frozen", device) for m in ("noisyclip", "frozen_clip")}
    loaded["latent_clip"] = load_latent_clip(device)
    bank = text_bank(loaded, sentences, device)
    readout = load_readout(device)
    print(f"{len(sentences)} sentences, {len(TEMPLATES)} wordings each; readout head {HEAD.name}", flush=True)
    a, raw_a = group_a(args, loaded, bank, pos, readout, device)
    b = group_b(args, loaded, bank, pos, readout, device)
    result = {"bars": {"control_floor": CONTROL_FLOOR, "read": READ_BAR}, "templates": TEMPLATES, "head": str(HEAD),
              "group_a": a, "group_b": b}
    result["verdict"] = verdict({k: v for k, v in a.items() if k in TASK_SENTENCES}, {"clean_pair": b["clean_pair"]})
    np.savez_compressed(out / "scores-group-a.npz", **raw_a)
    (out / "result.json").write_text(json.dumps(result, indent=1))
    for task in TASK_SENTENCES:
        print(task, " ".join(f"{r}={a[task][r]['mean_steps_0_to_15']:.3f}" for r in ROWS if "mean_steps_0_to_15" in a[task].get(r, {})), flush=True)
    print("trajectories clean_pair", " ".join(f"{r}={b['clean_pair'][r]['mean_steps_0_to_15']:.3f}" for r in ROWS
                                              if "mean_steps_0_to_15" in b["clean_pair"].get(r, {})), flush=True)
    print("verdict", json.dumps(result["verdict"]), flush=True)
    if not args.smoke:
        import wandb
        run = wandb.init(entity="prime_lab", project="poe-repair-animals-compose", name="loop9-w2c-fair", group="loop9-w2c",
                         config={"step": "W2c", "rows": ROWS, "templates": TEMPLATES, "bars": result["bars"]})
        for task in TASK_SENTENCES:
            for r in ROWS:
                if "mean_steps_0_to_15" in a[task].get(r, {}):
                    run.summary[f"A/{task}/{r}"] = a[task][r]["mean_steps_0_to_15"]
        for r in ROWS:
            if "mean_steps_0_to_15" in b["clean_pair"].get(r, {}):
                run.summary[f"B/clean_pair/{r}"] = b["clean_pair"][r]["mean_steps_0_to_15"]
        run.summary["state"] = result["verdict"]["state"]
        result["wandb_run"] = run.id
        (out / "result.json").write_text(json.dumps(result, indent=1))
        run.finish()
    print("W2C DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
