#!/usr/bin/env python
"""Why does the readout checker stall near 0.6? Accuracy at every noise level, on training and held-out pictures.

Loads one saved head and measures balanced accuracy (clean pair against every reject) at sampler steps
from 0 (pure noise) to 49 (almost finished), on two sets: a fixed random sample of its own training
tiles (64 clean pairs, 64 rejects) and the held-out check tiles. Each tile is noised to the step 4
times and its logits averaged, the same way the check does it.

How to read the two curves:
    training low everywhere            -> the head cannot learn the task from these features
    training high, held-out low        -> it learns the training pairs but does not carry to new pairs
    both high late, both near 0.5 early -> the early picture does not yet say whether it will be a pair

Writes <run dir>/diagnose.json and diagnose.png.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import encode_prompt_sdxl  # noqa: E402
from train_count_head import Readout  # noqa: E402
from train_readout_pair import RUNG2, PairHead, load_latent  # noqa: E402

HERE = Path(__file__).resolve().parent
STEPS = [0, 2, 5, 8, 10, 15, 20, 25, 30, 35, 40, 49]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--head", required=True, help="checkpoint file name inside the run folder")
    ap.add_argument("--train-manifest", type=Path, required=True, help="the tile list this head trained on")
    args = ap.parse_args()
    run = RUNG2 / "readout_pair" / args.run
    ctx = make_ctx(output_root=run / "raw")
    for n in ("vae", "unet", "text_encoder", "text_encoder_2"):
        ctx.models[n].requires_grad_(False)
        ctx.models[n].eval()
    ctx.scheduler.set_timesteps(ctx.num_inference_steps)
    sampler_t = ctx.scheduler.timesteps.clone().cpu()
    seq_e, pool_e = encode_prompt_sdxl("", models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    reader = Readout(ctx.models["unet"], seq_e, pool_e).to(ctx.device)
    head = PairHead().to(ctx.device)
    head.load_state_dict(torch.load(run / args.head)["head"])
    head.eval()

    rng = random.Random(0)
    train = [t for t in json.loads(args.train_manifest.read_text())["tiles"] if not t.get("check")]
    sample = rng.sample([t for t in train if t["label"] == 1], 64) + rng.sample([t for t in train if t["label"] == 0], 64)
    held = [t for t in json.loads((HERE / "training_set_v3.json").read_text())["tiles"] if t["check"]]
    sets = {"training tiles (64 clean pairs, 64 rejects)": sample, "held-out check tiles (20 clean pairs, 23 rejects)": held}

    @torch.no_grad()
    def curve(tiles):
        z = torch.cat([load_latent(t).float() for t in tiles])
        lab = [t["label"] for t in tiles]
        gen, out = torch.Generator().manual_seed(7), []
        for s in STEPS:
            logit = torch.zeros(len(tiles))
            for _ in range(4):
                for i in range(0, len(tiles), 8):
                    z0 = z[i:i + 8].to(ctx.device)
                    t = sampler_t[s].repeat(z0.shape[0]).to(ctx.device)
                    x = ctx.scheduler.add_noise(z0, torch.randn(z0.shape, generator=gen).to(ctx.device), t)
                    logit[i:i + 8] += head(reader(x.to(ctx.dtype), t)).float().cpu()
            yes = (logit > 0).tolist()
            pos = [y for y, l in zip(yes, lab) if l == 1]
            neg = [not y for y, l in zip(yes, lab) if l == 0]
            out.append((sum(pos) / len(pos) + sum(neg) / len(neg)) / 2)
            print(f"step {s:2d}  {out[-1]:.3f}", flush=True)
        return out

    res = {}
    for name, tiles in sets.items():
        print(name, flush=True)
        res[name] = curve(tiles)
    (run / "diagnose.json").write_text(json.dumps({"head": args.head, "sampler_steps": STEPS, "balanced_accuracy": res}, indent=1))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axvspan(0, 19, color="0.92", zorder=0)
    ax.text(9.5, 0.97, "steps the bar is judged on", ha="center", va="top", fontsize=9, color="0.4")
    ax.axhline(0.75, color="tab:red", lw=1, ls="--")
    ax.text(49, 0.76, "bar 0.75", ha="right", va="bottom", fontsize=9, color="tab:red")
    ax.axhline(0.5, color="0.5", lw=1, ls=":")
    ax.text(49, 0.49, "chance", ha="right", va="top", fontsize=9, color="0.5")
    for (name, ys), c in zip(res.items(), ("tab:blue", "tab:orange")):
        ax.plot(STEPS, ys, "o-", color=c)
        ax.text(STEPS[-1] + 0.8, ys[-1], name.split(" (")[0], color=c, fontsize=9, va="center")
    ax.set_xlim(0, 60)
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("sampler step (0 = pure noise, 49 = almost finished)")
    ax.set_ylabel("balanced accuracy, clean pair vs reject")
    ax.set_title(f"{args.run}, {args.head}: is the early picture readable at all?", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(run / "diagnose.png", dpi=150)
    print("DIAGNOSE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
