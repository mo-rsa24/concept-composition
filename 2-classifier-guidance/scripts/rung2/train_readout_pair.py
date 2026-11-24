#!/usr/bin/env python
"""A Readout-Guidance-style checker: is this half-drawn picture a clean pair of two different things?

The head follows Readout Guidance (Luo et al., CVPR 2024): it reads SDXL's frozen middle block and
first two upsampling blocks on the empty prompt (``train_count_head.Readout``), projects each to 128
channels at 32 x 32, sums them, and gives one logit after two convolutions and an attention pool. Its
gradient is the push once it passes the check.

Two ways to feed it:

--manifest training_set_v4.json (the main one). Every tile's finished latent is noised to a random
sampler step, the same way the check noises its tiles. Each batch takes 4 real and 4 fake; each side
first picks a source uniformly among the sources that have examples on that side, then a tile, so
no source's size or look can separate the sides.

--labels drawings.jsonl (plan 23 alone). Real = joint drawings the API called a faithful pair, fake =
product drawings, examples are the drawings' own saved trajectory states.

v4c switches (each off by default, so earlier runs reproduce):
    --pair-prompt   SDXL is read with each picture's own joint prompt ("a cat and a dog"), so the head
                    knows which two things were asked for and two cats can be wrong for cat x dog.
    --aug           each tile is drawn from three latents (plain, two changed copies from
                    augment_encode.py), so colour, blur, compression and layout cannot be memorised.
    --early-frac F  a share F of noise levels is drawn from the 20 noisiest sampler steps.
    --accum K       K batches of 8 per update (32 at K = 4).

The check, written before training, is plan 22's and plan 23's: on the held-out hand-called tiles
(7 whole pairs plus cat x dog seeds 7 and 8), each noised to the first 20 sampler steps 4 times,
balanced accuracy at least 0.75 on at least 10 of the 20 steps, good pairs against every negative.
Accuracy on singles and hybrids alone is reported beside it.

Writes, under MOUNT_ROOT/outputs/classifier_guidance_ladder/rung2/readout_pair/<run>/:
    head_step_NNNNNN.pt, check.json, split.json
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from poe_repair import paths  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import encode_prompt_sdxl  # noqa: E402
from train_count_head import GRID, Readout  # noqa: E402

RUNG2 = paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2"
HERE = Path(__file__).resolve().parent
BAR, MIN_STEPS, EARLY = 0.75, 10, 20


class PairHead(nn.Module):
    def __init__(self, channels=(1280, 1280, 640), width=128, n_out=1):
        super().__init__()
        self.n_out = n_out
        self.proj = nn.ModuleList([nn.Sequential(nn.Conv2d(c, width, 1), nn.GroupNorm(16, width), nn.SiLU())
                                   for c in channels])
        self.mix = nn.Sequential(nn.Conv2d(width, width, 3, stride=2, padding=1), nn.SiLU(),
                                 nn.Conv2d(width, width, 3, stride=2, padding=1), nn.SiLU())   # 32 -> 8
        self.pos = nn.Parameter(torch.randn(1, 65, width) / width ** 0.5)
        self.attn = nn.MultiheadAttention(width, 8, batch_first=True)
        self.out = nn.Linear(width, n_out)

    def forward(self, feats):
        h = sum(p(F.adaptive_avg_pool2d(f.float(), GRID)) for p, f in zip(self.proj, feats))
        tok = self.mix(h).flatten(2).transpose(1, 2)
        x = torch.cat([tok.mean(1, keepdim=True), tok], 1) + self.pos
        pooled, _ = self.attn(x[:, :1], x, x)
        o = self.out(pooled[:, 0])
        return o[:, 0] if self.n_out == 1 else o          # (B,) or (B, 3): clean pair, a cat present, a dog present


class ReadoutPairClassifier(nn.Module):
    """The trained head on SDXL's readout features, as the push code calls it: logits, and their sum."""

    def __init__(self, unet, seq, pool, head):
        super().__init__()
        self.reader, self.head = Readout(unet, seq, pool), head

    def forward(self, x_t, timestep):
        o = self.head(self.reader(x_t.to(self.reader.seq_e.dtype), timestep))
        return o if o.ndim == 1 else o.sum(-1)              # identity push: clean pair + cat present + dog present

    def log_prob_both(self, x_t, timestep):
        return self.forward(x_t, timestep).sum()


def joint_prompt(pair: str) -> str:
    """'a_cat__x__a_dog' -> 'a cat and a dog', the pair's joint prompt."""
    a, b = pair.split("__x__")
    return f"{a.replace('_', ' ')} and {b.replace('_', ' ')}"


def load_latent(t: dict) -> torch.Tensor:
    """A tile's finished latent as (1, 4, 128, 128) fp16."""
    path = paths.MOUNT_ROOT / t["latent"] if "/" in t["latent"] else RUNG2 / "training_set/latents" / f"{t['latent']}.pt"
    z = torch.load(path)
    if "state_index" in t:
        z = z[-1:] if t["state_index"] == -1 else z[t["state_index"]:t["state_index"] + 1]
    return z.to(torch.float16)


def manifest_sampler(args, sampler_t):
    data = json.loads(args.manifest.read_text())
    tiles = [t for t in data["tiles"] if not t.get("check")]
    if args.frac < 1.0:
        groups = collections.defaultdict(list)
        for t in tiles:
            groups[(t["source"], t["label"])].append(t)
        rng = random.Random(0)
        tiles = [t for g in groups.values() for t in rng.sample(g, max(1, round(args.frac * len(g))))]
    if any(t["pair"] == "a_cat__x__a_dog" and 9 <= t.get("seed", -1) <= 16 for t in tiles):
        raise SystemExit("guard: cat x dog seeds 9 to 16 are the result sheet's and may not be trained on")
    if any(t["pair"] in set(data["held_out_pairs"]) for t in tiles):
        raise SystemExit("guard: a held-out pair is in the training tiles")
    by = {0: collections.defaultdict(list), 1: collections.defaultdict(list)}
    missing = collections.Counter()
    aug = RUNG2 / "dataset-v4/aug"
    for t in tiles:
        try:
            t["zs"] = ([torch.load(aug / f"{t['id']}__a{k}.pt").to(torch.float16) for k in range(3)] if args.aug
                       else [load_latent(t)])
        except (FileNotFoundError, RuntimeError, EOFError):
            missing[t["source"]] += 1
            continue
        by[t["label"]][t["source"]].append(t)
    split = {f"{'real' if s else 'fake'}_{src}": len(v) for s in (1, 0) for src, v in sorted(by[s].items())}
    split["missing_latents"] = dict(missing)
    split["pairs_with_a_real_example"] = len({t["pair"] for v in by[1].values() for t in v})

    wrong = [t for v in by[0].values() for t in v if "wrong pair" in t.get("kind", "")]
    split["wrong_pair_rejects"] = len(wrong)

    def batch(scheduler, device, dtype):
        picks = [random.choice(by[1][random.choice(list(by[1]))]) for _ in range(4)] + \
                [random.choice(wrong) if wrong and random.random() < args.hard_frac
                 else random.choice(by[0][random.choice(list(by[0]))]) for _ in range(4)]
        z0 = torch.cat([random.choice(p["zs"]) for p in picks]).to(device, torch.float32)
        idx = [random.randrange(EARLY) if random.random() < args.early_frac else random.randrange(len(sampler_t))
               for _ in picks]
        t = sampler_t[torch.tensor(idx)].to(device)
        return (scheduler.add_noise(z0, torch.randn_like(z0), t).to(dtype), t, [p["pair"] for p in picks],
                [p["id"] for p in picks])

    return batch, split


TRAJ_CHECK_STEPS = (0, 2, 4, 6, 8, 10, 15, 20)


def load_trajectories(labels_path: Path) -> dict:
    """Plan 25's saved runs with a confident label, states held in memory: train {label: [stem]}, held_out [(stem, mode, label)]."""
    labels = json.loads(labels_path.read_text())
    root = labels_path.parent
    times = json.loads((root / "timesteps.json").read_text())
    keep = {s: r for s, r in labels.items() if r["label"] is not None}
    if any(r["seed"] <= 1250 for r in keep.values()):
        raise SystemExit("guard: a trajectory seed overlaps the held-out sheet or the cat x dog lists")
    train = {0: [], 1: []}
    held = []
    for s, r in sorted(keep.items()):
        (train[r["label"]].append(s) if r["split"] == "train" else held.append((s, r["mode"], r["label"])))
    states = {s: torch.load(root / "states" / f"{s}.pt") for s in keep}
    print(f"trajectories: {len(train[1])} clean and {len(train[0])} reject training runs, {len(held)} held-out runs",
          flush=True)
    return {"train": train, "held_out": held, "states": states,
            "kept_steps": times["kept_steps"], "timesteps": times["timesteps"]}


def mix_trajectories(batch, traj, args):
    """Replace each slot, with probability --traj-frac, by a saved state of a run with that slot's label."""
    early_idx = [k for k, s in enumerate(traj["kept_steps"]) if s < EARLY]
    any_idx = list(range(len(traj["kept_steps"]) - 1))   # the final latent is not a sampler state

    def mixed(scheduler, device, dtype):
        x, t, pairs, ids = batch(scheduler, device, dtype)
        x, t, pairs, ids = x.clone(), t.clone(), list(pairs), list(ids)
        for i in range(x.shape[0]):
            if random.random() >= args.traj_frac:
                continue
            label = 1 if i < 4 else 0
            stem = random.choice(traj["train"][label])
            k = random.choice(early_idx if random.random() < args.early_frac else any_idx)
            x[i] = traj["states"][stem][k].to(device, dtype)
            t[i] = traj["timesteps"][k]
            pairs[i], ids[i] = "a_cat__x__a_dog", f"traj-{stem}"
        return x, t, pairs, ids
    return mixed


def states_sampler(args):
    rows = [json.loads(l) for l in args.labels.read_text().splitlines() if l.strip()]
    real = [r["id"] for r in rows if r.get("method") == "joint" and r.get("verdict") == "faithful pair"]
    fake = [r["id"] for r in rows if r.get("method") == "product" and "verdict" in r]
    real = [s for s in real if (RUNG2 / "kim/states" / f"{s}.pt").exists()]
    fake = [s for s in fake if (RUNG2 / "kim/states" / f"{s}.pt").exists()]
    kept = json.loads((RUNG2 / "kim/timesteps.json").read_text())["timesteps"]
    split = {"real_clean_joint": len(real), "fake_product": len(fake),
             "joint_rejected_by_api": sum(1 for r in rows if r.get("method") == "joint" and r.get("verdict") != "faithful pair"),
             "pairs_with_a_clean_joint": len({s.split("__seed")[0] for s in real})}

    def batch(scheduler, device, dtype):
        names = random.choices(real, k=4) + random.choices(fake, k=4)
        ks = [random.randrange(len(kept)) for _ in names]
        x = torch.stack([torch.load(RUNG2 / "kim/states" / f"{s}.pt")[k] for s, k in zip(names, ks)]).to(device, dtype)
        return x, torch.tensor([max(kept[k], 1) for k in ks], device=device), [s.split("__seed")[0] for s in names], None

    return batch, split


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="readout-pair-v1")
    ap.add_argument("--manifest", type=Path, help="a tile list such as training_set_v4.json")
    ap.add_argument("--labels", type=Path, help="the API labels JSONL for plan 23's drawings")
    ap.add_argument("--eval", type=Path, default=HERE / "training_set_v3.json",
                    help="evaluation tile list (its 'check' tiles); the verdict is always on v3's check tiles")
    ap.add_argument("--pair-prompt", action="store_true")
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--early-frac", type=float, default=0.0)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--score", type=Path, default=None,
                    help="a saved head: no training, only score it on --eval over the first 20 steps (writes score-<eval>.json)")
    ap.add_argument("--presence", type=Path, default=None,
                    help="presence_catdog.json: adds two outputs, a cat present and a dog present, trained beside "
                         "clean pair; the push then follows the sum of all three (the identity push)")
    ap.add_argument("--presence-only", action="store_true",
                    help="with --presence: a presence-only checker (MNIST's separate-label control): two outputs, a cat "
                         "present and a dog present, and no clean-pair output; its push is their sum")
    ap.add_argument("--hard-frac", type=float, default=0.0,
                    help="share of each batch's rejects drawn from wrong-pair pictures (two cats, two dogs), "
                         "the mistake the push keeps making; the rest are drawn as before")
    ap.add_argument("--frac", type=float, default=1.0,
                    help="train on this share of the manifest's tiles, kept in proportion by source and label (learning curve)")
    ap.add_argument("--traj-labels", type=Path, default=None,
                    help="plan 25's trajectories/labels.json: adds the reading check on held-out joint runs' saved states "
                         "(clean against reject per sampler step) and, with --traj-frac, trains on the training runs' states")
    ap.add_argument("--traj-frac", type=float, default=0.0,
                    help="share of each batch's slots filled with a saved state of a run whose finished picture has that "
                         "slot's label; the state goes in as saved, at its own step, with no noise added")
    ap.add_argument("--trunk-lora", type=int, default=0,
                    help="rank of LoRA adapters on a separate copy of SDXL's down and middle blocks (attention q, k, v, out), "
                         "trained with the head so the features the checker reads can adapt; 0 keeps SDXL frozen")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--out-root", type=Path, default=None,
                    help="the folder the run's folder goes in; the default is this ladder's readout_pair folder")
    args = ap.parse_args()
    if not (args.manifest or args.labels):
        raise SystemExit("give --manifest or --labels")
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    out = (args.out_root or RUNG2 / "readout_pair") / args.run
    out.mkdir(parents=True, exist_ok=True)

    ctx = make_ctx(output_root=out / "raw")
    for n in ("vae", "unet", "text_encoder", "text_encoder_2"):
        ctx.models[n].requires_grad_(False)
        ctx.models[n].eval()
    ctx.scheduler.set_timesteps(ctx.num_inference_steps)
    sampler_t = ctx.scheduler.timesteps.clone().cpu()
    batch, split = manifest_sampler(args, sampler_t) if args.manifest else states_sampler(args)
    traj = load_trajectories(args.traj_labels) if args.traj_labels else None
    if traj and args.traj_frac > 0:
        batch = mix_trajectories(batch, traj, args)
        split["trajectory_runs_for_training"] = {"clean": len(traj["train"][1]), "reject": len(traj["train"][0])}
    if traj:
        split["trajectory_runs_held_out"] = {m: {"clean": sum(1 for r in traj["held_out"] if r[1] == m and r[2] == 1),
                                                 "reject": sum(1 for r in traj["held_out"] if r[1] == m and r[2] == 0)}
                                             for m in ("joint", "product")}
    (out / "split.json").write_text(json.dumps(split, indent=1))
    print(json.dumps(split), flush=True)

    seq_e, pool_e = encode_prompt_sdxl("", models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    lora_params, clf_unet = [], None
    if args.trunk_lora:
        from diffusers import UNet2DConditionModel
        from poe_repair.config import RunConfig
        from poe_repair.experiments.presence_head.dog_classifier import attach_trunk_adapters
        clf_unet = UNet2DConditionModel.from_pretrained(
            RunConfig().model_id, subfolder="unet", torch_dtype=ctx.dtype, use_safetensors=True).to(ctx.device)
        clf_unet.enable_gradient_checkpointing()
        clf_unet, lora_params = attach_trunk_adapters(clf_unet, rank=args.trunk_lora, alpha=args.trunk_lora)
        print(f"trunk LoRA rank {args.trunk_lora}: {sum(p.numel() for p in lora_params):,} adapter weights", flush=True)
    reader = Readout(clf_unet if clf_unet is not None else ctx.models["unet"], seq_e, pool_e).to(ctx.device)
    prompts = {}

    def read(x, t, pairs):
        if not args.pair_prompt:
            return reader(x, t)
        for p in pairs:
            if p not in prompts:
                prompts[p] = encode_prompt_sdxl(joint_prompt(p), models=ctx.models, device=ctx.device, dtype=ctx.dtype)
        return reader(x, t, torch.cat([prompts[p][0] for p in pairs]), torch.cat([prompts[p][1] for p in pairs]))
    presence = json.loads(args.presence.read_text()) if args.presence else None
    if args.presence_only and not presence:
        raise SystemExit("--presence-only needs --presence")
    head = PairHead(n_out=2 if args.presence_only else 3 if presence else 1).to(ctx.device)
    first = 0 if args.presence_only else 1          # where the presence outputs start
    groups = [{"params": head.parameters(), "lr": 3e-4}]
    if lora_params:
        groups.append({"params": lora_params, "lr": 1e-4})
    opt = torch.optim.AdamW(groups, weight_decay=1e-2)

    tiles = [t for t in json.loads(args.eval.read_text())["tiles"] if t["check"]]
    z_check = torch.cat([load_latent(t).float() for t in tiles])
    kind = [t["kind"] if t["label"] == 0 else "positive" for t in tiles]
    strict_pos = [t.get("strict_label", t["label"]) == 1 for t in tiles]
    source = [t.get("source", "v3") for t in tiles]
    is_v3 = [src == "v3" for src in source]

    @torch.no_grad()
    def check(steps):
        res, gen = [], torch.Generator().manual_seed(7)
        for s in steps:
            t_val, logits, extra = sampler_t[s], [[] for _ in tiles], [[] for _ in tiles]
            for _ in range(4):
                for i in range(0, len(tiles), 8):
                    z0 = z_check[i:i + 8].to(ctx.device)
                    t = t_val.repeat(z0.shape[0]).to(ctx.device)
                    noisy = ctx.scheduler.add_noise(z0, torch.randn(z0.shape, generator=gen).to(ctx.device), t)
                    pairs = [tt["pair"] for tt in tiles[i:i + 8]]
                    o = head(read(noisy.to(ctx.dtype), t, pairs)).float().cpu()
                    for j in range(o.shape[0]):
                        if o.ndim == 1:
                            logits[i + j].append(float(o[j]))
                        else:
                            logits[i + j].append(float(o[j].sum()) if args.presence_only else float(o[j, 0]))
                            extra[i + j].append(o[j, first:].tolist())
            yes = [sum(l) / len(l) > 0 for l in logits]

            def bal(neg_kinds, only_v3=True, only=None):
                use = [(v or not only_v3) if only is None else (src == only) for v, src in zip(is_v3, source)]
                pos = [p for p, k, u in zip(yes, kind, use) if k == "positive" and u]
                neg = [not p for p, k, u in zip(yes, kind, use) if k in neg_kinds and u]
                return (sum(pos) / len(pos) + sum(neg) / len(neg)) / 2 if pos and neg else float("nan")
            row = {}
            if presence and extra[0]:
                cat = [sum(e[0] for e in ex) / len(ex) > 0 for ex in extra]
                dog = [sum(e[1] for e in ex) / len(ex) > 0 for ex in extra]
                truth = [presence.get(tt["id"], [-1, -1]) for tt in tiles]

                def bal_presence(pred, k):
                    pos = [p for p, tr in zip(pred, truth) if tr[k] == 1]
                    neg = [not p for p, tr in zip(pred, truth) if tr[k] == 0]
                    return (sum(pos) / len(pos) + sum(neg) / len(neg)) / 2 if pos and neg else float("nan")
                kinds = [i for i, k in enumerate(kind) if "wrong pair" in k]
                row = {"cat_present_balanced_accuracy": bal_presence(cat, 0),
                       "dog_present_balanced_accuracy": bal_presence(dog, 1),
                       "wrong_pairs_missing_an_animal": (sum(1 for i in kinds if not (cat[i] and dog[i])) / len(kinds))
                       if kinds else float("nan")}
            res.append(row | {"sampler_step": int(s), "balanced_accuracy": bal(set(kind) - {"positive"}),
                        "singles_and_hybrids_balanced_accuracy": bal({"single", "hybrid"}),
                        "whole_evaluation_set_balanced_accuracy": bal(set(kind) - {"positive"}, only_v3=False),
                        "drawn_held_out_balanced_accuracy": bal(set(kind) - {"positive"}, only="more_heldout"),
                        "stature_rule_balanced_accuracy": (sum(p for p, sp in zip(yes, strict_pos) if sp) / max(1, sum(strict_pos))
                            + sum(not p for p, k in zip(yes, kind) if k != "positive") / max(1, sum(k != "positive" for k in kind))) / 2})
        return res

    @torch.no_grad()
    def traj_check():
        """Balanced accuracy, clean against reject, on held-out joint runs' saved states per sampler step."""
        res = {}
        for k, step_ in enumerate(traj["kept_steps"]):
            if step_ not in TRAJ_CHECK_STEPS:
                continue
            runs = [(stem, label) for stem, mode, label in traj["held_out"] if mode == "joint"]
            yes, lab = [], []
            for i in range(0, len(runs), 8):
                chunk = runs[i:i + 8]
                x = torch.stack([traj["states"][stem][k] for stem, _ in chunk]).to(ctx.device, ctx.dtype)
                t = torch.full((len(chunk),), traj["timesteps"][k], device=ctx.device)
                o = head(read(x, t, ["a_cat__x__a_dog"] * len(chunk))).float()
                o = o if o.ndim == 1 else o[:, 0]
                yes += [float(v) > 0 for v in o]
                lab += [label for _, label in chunk]
            pos = [y for y, l in zip(yes, lab) if l == 1]
            neg = [not y for y, l in zip(yes, lab) if l == 0]
            res[step_] = (sum(pos) / len(pos) + sum(neg) / len(neg)) / 2 if pos and neg else float("nan")
        early = [v for s_, v in res.items() if s_ <= 10]
        return {"per_step": res, "mean_steps_0_to_10": sum(early) / len(early)}

    def show_traj(tc):
        return ("trajectory check, held-out joint runs: " + "  ".join(f"{s_}:{v:.3f}" for s_, v in tc["per_step"].items())
                + f"  mean steps 0 to 10 {tc['mean_steps_0_to_10']:.3f}")

    if args.score:
        head.load_state_dict(torch.load(args.score, map_location="cpu")["head"])
        head.eval()
        if traj:
            tc = traj_check()
            (out / "score-trajectories.json").write_text(json.dumps({"head": str(args.score), **tc}, indent=1))
            print("SCORE " + show_traj(tc), flush=True)
            return 0
        rows_s = check(range(EARLY))
        mean = lambda k: sum(r[k] for r in rows_s) / len(rows_s)
        res = {"head": str(args.score), "eval": str(args.eval), "per_step": rows_s,
               "whole_evaluation_set_mean": mean("whole_evaluation_set_balanced_accuracy"),
               "v3_tiles_mean": mean("balanced_accuracy")}
        (out / f"score-{args.eval.stem}.json").write_text(json.dumps(res, indent=1))
        print(f"SCORE {args.score.parent.name}/{args.score.name} on {args.eval.name}: whole evaluation set "
              f"{res['whole_evaluation_set_mean']:.3f}", flush=True)
        return 0

    started = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            head.eval()
            rows_c = check(range(EARLY) if step == args.steps else range(0, EARLY, 4))
            mean = sum(r["balanced_accuracy"] for r in rows_c) / len(rows_c)
            whole = sum(r["whole_evaluation_set_balanced_accuracy"] for r in rows_c) / len(rows_c)
            strict_m = sum(r["stature_rule_balanced_accuracy"] for r in rows_c) / len(rows_c)
            drawn = [r["drawn_held_out_balanced_accuracy"] for r in rows_c if r["drawn_held_out_balanced_accuracy"] == r["drawn_held_out_balanced_accuracy"]]
            drawn_m = sum(drawn) / len(drawn) if drawn else float("nan")
            if presence:
                pm = lambda k: sum(r[k] for r in rows_c) / len(rows_c)
                print(f"step {step:5d}  presence outputs: cat present {pm('cat_present_balanced_accuracy'):.3f}, "
                      f"dog present {pm('dog_present_balanced_accuracy'):.3f}; held-out wrong pairs read as missing an "
                      f"animal {pm('wrong_pairs_missing_an_animal'):.3f}", flush=True)
            print(f"step {step:5d}  held-out balanced accuracy {mean:.3f} (v3 check tiles), {whole:.3f} (whole "
                  f"evaluation set), {strict_m:.3f} (stature-rule clean pairs against every reject), {drawn_m:.3f} "
                  f"(420 drawn held-out pictures)  "
                  f"({time.time() - started:.0f} s)", flush=True)
            if traj:
                tc = traj_check()
                print(f"step {step:5d}  " + show_traj(tc), flush=True)
            torch.save({"head": head.state_dict(), "step": step, "split": split, "pair_prompt": args.pair_prompt,
                        "heads": head.n_out, "trunk_lora": args.trunk_lora, "presence_only": args.presence_only,
                        "lora": {n: q.detach().cpu() for n, q in clf_unet.named_parameters() if "lora_" in n} if clf_unet is not None else None},
                       out / f"head_step_{step:06d}.pt")
            head.train()
            if step == args.steps:
                passing = [r["sampler_step"] for r in rows_c if r["balanced_accuracy"] >= BAR]
                verdict = {"rule": f"good pairs against every negative, balanced accuracy >= {BAR} on >= {MIN_STEPS} "
                                   f"of the first {EARLY} steps", "per_step": rows_c,
                           "passing_steps": passing, "passed": len(passing) >= MIN_STEPS,
                           "trajectory_check": traj_check() if traj else None}
                (out / "check.json").write_text(json.dumps(verdict, indent=1))
                print(f"READOUT CHECK {'PASSED' if verdict['passed'] else 'FAILED'}: {len(passing)} of {EARLY}", flush=True)
                break
        opt.zero_grad(set_to_none=True)
        y = torch.tensor([1.0] * 4 + [0.0] * 4, device=ctx.device)
        for _ in range(args.accum):
            x, t, pairs, ids = batch(ctx.scheduler, ctx.device, ctx.dtype)
            with (torch.enable_grad() if lora_params else torch.no_grad()):
                feats = read(x, t, pairs)
            o = head(feats)
            if o.ndim == 1:
                loss = F.binary_cross_entropy_with_logits(o, y)
            else:
                pres = torch.tensor([presence.get(i, [-1, -1]) for i in ids], dtype=torch.float32, device=ctx.device)
                mask = (pres >= 0).float()
                pres_loss = (F.binary_cross_entropy_with_logits(o[:, first:], pres.clamp(min=0), reduction="none")
                             * mask).sum() / mask.sum().clamp(min=1)
                loss = pres_loss * 2 if args.presence_only else F.binary_cross_entropy_with_logits(o[:, 0], y) + \
                    (F.binary_cross_entropy_with_logits(o[:, 1:], pres.clamp(min=0), reduction="none") * mask).sum() \
                    / mask.sum().clamp(min=1) * 2
            (loss / args.accum).backward()
        opt.step()
        if step % 100 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}  ({time.time() - started:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
