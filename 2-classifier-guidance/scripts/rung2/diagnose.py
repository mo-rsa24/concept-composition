#!/usr/bin/env python
"""Rung 2 diagnostics: what the pair-aware discriminator reads, and what its push does, step by step.

Part A, through time. For seeds 1 and 3 (training) and 9 and 10 (held out), the product is run
plain, guided at s = 10 and 30, and with the size-matched random push at s = 10. From each run's
own step record (the noisy latent and the prediction the sampler used at every step):
    - the discriminator's logit at every one of the 50 steps;
    - at 8 shown steps, the predicted finished picture (Tweedie's estimate, decoded), the
      discriminator's attention over its 8 x 8 patches, and, for guided runs, the size of the
      push per latent pixel.
Part B, through kinds of picture. Every training-set tile, including the ones never trained on
(hybrids, two of one kind, three animals, drawings and collages), noised to each of the first 20
sampler steps with fixed noise, scored by the check head (trained on seeds 1 to 6 only) and by the
final head. It shows whether "one cat and one dog apart" scores above the near misses at each step.

Writes, under MOUNT_ROOT/outputs/classifier_guidance_ladder/rung2/diagnostics/:
    filmstrip-seedNN.png, logit-through-time.png, push-through-time.png, kinds-through-time.png,
    diagnostics.json
"""
from __future__ import annotations

import csv
import json
import socket
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from poe_repair import paths  # noqa: E402
from poe_repair.composers._helpers import init_latents_for_cell  # noqa: E402
from poe_repair.methods._classifier_guided import run_poe_classifier_guided  # noqa: E402
from poe_repair.methods._sampling import run_cfg_poe  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import decode_latents, encode_prompt_sdxl, tweedie_mean  # noqa: E402
import classifier_guidance_from_the_paper as single  # noqa: E402
from discriminator import PairDiscriminator  # noqa: E402

RUNG2 = paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2"
OUT = RUNG2 / "diagnostics"
FINAL_HEAD = RUNG2 / "discriminator/final-all-seeds/head_step_001500.pt"
CHECK_HEAD = RUNG2 / "discriminator/check-holdout-7-8/head_step_001500.pt"
SEEDS = (1, 3, 9, 10)
SHOWN_STEPS = (0, 3, 6, 10, 15, 20, 30, 49)
RUNS = (("plain", None, None), ("guided s10", 10.0, None), ("guided s30", 30.0, None),
        ("random s10", 10.0, "random"))
CALLS = Path(__file__).resolve().parents[2] / \
    "artifacts/results/does-a-small-discriminator-steer-cat-x-dog/training-set/calls.csv"


def kind_of(row):
    if row["role"] == "positive":
        return "one cat and one dog apart (positive)"
    if row["column"] == "product":
        return "plain product, fused (negative)" if row["picture"] == "intact" else "damaged or drawn"
    if row["column"] in ("lone_cat", "lone_dog"):
        if row["animals"] == "many copies":
            return "one concept repeated"
        return "one animal (negative)" if row["picture"] == "intact" else "damaged or drawn"
    if row["picture"] == "damaged":
        return "damaged or drawn"
    if row["animals"] == "one fused":
        return "hybrid, never trained on"
    if row["animals"] == "one single":
        return "one animal, never trained on"
    return "two or three, wrong mix, never trained on"


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    print(f"{socket.gethostname()}: {torch.cuda.get_device_name(0)}", flush=True)
    single.disk_guard(OUT)
    ctx = make_ctx(output_root=OUT / "raw")
    enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = enc("")
    seq_a, pool_a = enc("a cat")
    seq_b, pool_b = enc("a dog")
    common = dict(models=ctx.models, scheduler=ctx.scheduler, seq_a=seq_a, pool_a=pool_a,
                  seq_b=seq_b, pool_b=pool_b, seq_e=seq_e, pool_e=pool_e,
                  guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
                  height=1024, width=1024, device=ctx.device, dtype=ctx.dtype)

    def load_disc(path):
        d = PairDiscriminator(ctx.models["unet"], seq_e, pool_e).to(ctx.device)
        d.head.load_state_dict(torch.load(path, map_location="cpu")["head"])
        d.head.float().eval()
        return d

    disc = load_disc(FINAL_HEAD)

    @torch.no_grad()
    def read(d, z, t):
        """The logit and the pool's attention over the 8 x 8 patches, for a batch of latents."""
        head = d.head
        h = head.down(d.features(z.to(ctx.dtype), t).float())
        tokens = h.flatten(2).transpose(1, 2)
        x = torch.cat([tokens.mean(dim=1, keepdim=True), tokens], dim=1) + head.pos
        pooled, w = head.attn(x[:, :1], x, x, need_weights=True, average_attn_weights=True)
        return head.out(pooled[:, 0])[:, 0].cpu(), w[:, 0, 1:].reshape(-1, 8, 8).cpu()

    def push_map(z, t):
        with torch.enable_grad():
            zz = z.detach().float().requires_grad_(True)
            g = torch.autograd.grad(disc.log_prob_both(zz, t), zz)[0]
        return g[0].norm(dim=0).cpu()                           # (128, 128)

    def picture(x0):
        with torch.no_grad():
            return decode_latents(ctx.models, x0.to(ctx.device, ctx.dtype))[0].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    ctx.scheduler.set_timesteps(ctx.num_inference_steps)
    record = {"seeds": SEEDS, "shown_steps": SHOWN_STEPS, "runs": {}}

    # Part A
    for seed in SEEDS:
        strip = {}
        for name, strength, mode in RUNS:
            init, sigma = init_latents_for_cell(single.cell_for(seed), ctx)
            if strength is None:
                o = run_cfg_poe(init_latents=init, euler_init_noise_sigma=sigma, **common)
            elif mode is None:
                o = run_poe_classifier_guided(init_latents=init, euler_init_noise_sigma=sigma, head=disc,
                                              strength=strength, **common)
            else:
                ref = run_poe_classifier_guided(init_latents=init, euler_init_noise_sigma=sigma, head=disc,
                                                strength=strength, **common)
                init, sigma = init_latents_for_cell(single.cell_for(seed), ctx)
                rg = torch.Generator(device="cpu").manual_seed(30_000 + 100 * seed + int(strength))
                o = run_poe_classifier_guided(init_latents=init, euler_init_noise_sigma=sigma,
                                              random_push_norms=ref.extras["applied_norm_per_step"],
                                              random_generator=rg, **common)
            tr = o.tracker
            logits = []
            for i in range(ctx.num_inference_steps):
                t = tr.timesteps[i].long().reshape(1).to(ctx.device)
                lg, _ = read(disc, tr.trajectories[i].to(ctx.device), t)
                logits.append(float(lg[0]))
            cells = []
            for i in SHOWN_STEPS:
                t = tr.timesteps[i].long().reshape(1).to(ctx.device)
                z = tr.trajectories[i].to(ctx.device)
                ab = ctx.scheduler.alphas_cumprod[int(tr.timesteps[i])].to(ctx.device).float()
                x0 = tweedie_mean(z.float(), ab, tr.velocities[i].to(ctx.device).float())
                lg, att = read(disc, z, t)
                pm = push_map(z, t) if (strength is not None and mode is None) else None
                cells.append({"step": i, "picture": picture(x0), "attention": att[0].numpy(),
                              "push": None if pm is None else pm.numpy(), "logit": float(lg[0])})
            strip[name] = cells
            extras = o.extras
            record["runs"][f"seed {seed}, {name}"] = {
                "logit_per_step": logits,
                "push_over_eps_per_step": [a / e for a, e in zip(extras["applied_norm_per_step"], extras["eps_norm_per_step"])],
                "capped_per_step": extras["capped_per_step"]}
            print(f"seed {seed}, {name}: logit step 0 {logits[0]:.2f}, step 10 {logits[10]:.2f}, "
                  f"step 49 {logits[-1]:.2f}", flush=True)

        n_rows = len(RUNS) + 1
        fig, axes = plt.subplots(n_rows, len(SHOWN_STEPS), figsize=(2.3 * len(SHOWN_STEPS), 2.45 * n_rows))
        for r, (name, _, _) in enumerate(RUNS):
            for c, cell in enumerate(strip[name]):
                ax = axes[r][c]
                ax.imshow(cell["picture"])
                att = np.kron(cell["attention"] / cell["attention"].max(), np.ones((128, 128)))
                ax.imshow(att, cmap="magma", alpha=0.35, extent=(0, 1024, 1024, 0))
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"step {cell['step']}  logit {cell['logit']:+.1f}", fontsize=8)
                if c == 0:
                    ax.set_ylabel(name, fontsize=10)
        for c, cell in enumerate(strip["guided s10"]):
            ax = axes[-1][c]
            ax.imshow(cell["push"], cmap="viridis")
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel("guided s10:\nwhere the push\nlands", fontsize=9)
        part = "training seed" if seed <= 8 else "held-out seed"
        fig.suptitle(f"Seed {seed} ({part}): the predicted finished picture at each step, the "
                     f"discriminator's attention in purple, its logit above", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(OUT / f"filmstrip-seed{seed:02d}.png", dpi=90)
        plt.close(fig)
        (OUT / "diagnostics.json").write_text(json.dumps(record, indent=1))

    colours = {"plain": "#555555", "guided s10": "#2b6cb0", "guided s30": "#1a365d", "random s10": "#c05621"}
    for key, fname, ylabel in (("logit_per_step", "logit-through-time.png", "discriminator logit"),
                               ("push_over_eps_per_step", "push-through-time.png", "push size over prediction size")):
        fig, axes = plt.subplots(1, len(SEEDS), figsize=(4.2 * len(SEEDS), 3.4), sharey=True)
        for ax, seed in zip(axes, SEEDS):
            for name, _, _ in RUNS:
                if key == "push_over_eps_per_step" and name == "plain":
                    continue
                ys = record["runs"][f"seed {seed}, {name}"][key]
                ax.plot(range(len(ys)), ys, color=colours[name], label=name, lw=1.6)
            if key == "logit_per_step":
                ax.axhline(0, color="grey", ls=":", lw=1)
            else:
                ax.axhline(0.1, color="grey", ls=":", lw=1)
                ax.text(49, 0.101, "cap 0.1", ha="right", va="bottom", fontsize=8, color="grey")
            ax.set_title(f"seed {seed} ({'training' if seed <= 8 else 'held out'})", fontsize=10)
            ax.set_xlabel("sampler step (0 = pure noise)")
        axes[0].set_ylabel(ylabel)
        axes[-1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=110)
        plt.close(fig)

    # Part B
    rows = list(csv.DictReader(open(CALLS)))
    lat_dir = RUNG2 / "training_set/latents"
    timesteps = ctx.scheduler.timesteps.clone().cpu()
    heads = {"check head (trained on seeds 1 to 6)": load_disc(CHECK_HEAD), "final head (all 8 seeds)": disc}
    kinds = sorted({kind_of(r) for r in rows})
    by_head = {}
    gen = torch.Generator().manual_seed(11)
    for hname, d in heads.items():
        per_kind = {k: [] for k in kinds}
        for r in rows:
            name = (f"seed{int(r['seed']):02d}__{r['column']}" if r["draw"] == "own"
                    else f"seed{int(r['seed']):02d}-d{r['draw']}__{r['column']}")
            z0 = torch.load(lat_dir / f"{name}.pt").float().to(ctx.device)
            line = []
            for s in range(20):
                t = timesteps[s].reshape(1).to(ctx.device)
                noisy = ctx.scheduler.add_noise(z0, torch.randn(z0.shape, generator=gen).to(ctx.device), t)
                line.append(float(read(d, noisy, t)[0][0]))
            per_kind[kind_of(r)].append({"tile": name, "seed": int(r["seed"]), "logits": line})
        by_head[hname] = per_kind
        print(f"{hname}: scored {len(rows)} tiles", flush=True)
    record["kinds"] = by_head
    (OUT / "diagnostics.json").write_text(json.dumps(record, indent=1))

    palette = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, (hname, per_kind) in zip(axes, by_head.items()):
        for k_i, k in enumerate(kinds):
            lines = np.array([x["logits"] for x in per_kind[k]])
            if len(lines) == 0:
                continue
            ax.plot(range(20), lines.mean(axis=0), color=palette(k_i), lw=2.2, label=f"{k} ({len(lines)})")
            for ln in lines:
                ax.plot(range(20), ln, color=palette(k_i), lw=0.5, alpha=0.3)
        ax.axhline(0, color="grey", ls=":", lw=1)
        ax.set_title(hname, fontsize=10)
        ax.set_xlabel("sampler step the tile is noised to (0 = pure noise)")
    axes[0].set_ylabel("discriminator logit (above 0: 'apart')")
    axes[1].legend(fontsize=7, loc="upper left")
    fig.suptitle("Every training-set tile, scored at each early step: thick line the mean per kind, "
                 "thin lines single tiles (count in brackets)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "kinds-through-time.png", dpi=110)
    plt.close(fig)
    print("DIAGNOSTICS DONE", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
