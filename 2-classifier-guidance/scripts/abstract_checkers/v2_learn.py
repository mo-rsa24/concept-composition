"""Step V2: does a checker learn plurality on digits it never trained on?

The map reader sees two maps, one per asked digit, of where that label changes the frozen model's noise prediction, and no digit
label. The raw reader sees the noisy canvas and the asked pair. Both train on the pairs among digits 0 to 4.

Stages, in order:
    smoke    20 updates of each reader on the 1% data
    overfit  each reader memorises 16 fixed examples
    small    1,000 updates of each reader on the 1% data, scored on the 1% seen test split
    inner    train on pairs among 0 to 3, test on pairs holding 4, and freeze a configuration; never reads digits 5 to 9
    final    retrain the frozen configuration on 0 to 4 and read unseen_test once

Usage:
    python v2_learn.py <stage> --model <unet.pt> --data <folder with dataset.npz> --out <folder> [--wandb-name <name>]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mnist"))
from canvas_unet import ABAR, NULL, H, W, ddim_timesteps  # noqa: E402
from checker import load_unet  # noqa: E402
from pixel_checker import PixelHead, to_x  # noqa: E402

# The bars, fixed before any run (abstract-checkers/03-design.md and board question 55).
MIN_BA = 0.75
MAX_GAP = 0.05
MIN_APART_WRONG = 0.75
OVERFIT_RIGHT = 15
FLOOR_BA = 0.55

DATA_HASHES = {"f70b0fb362e13e255ad332ba14ca93dfb5cf051a5dfd3c86eb30482931d5a5f3",
               "9b70760ad5bed2f8d1b9a0dc6714bbf37f256a9d3486f498b4e4b601278b1b0b"}
ARRAYS = ("x", "asked", "holds", "corner", "src", "kind", "target", "n_digits", "apart")
KINDS = ("apart", "overlaid", "same_half", "single", "wrong")
STEPS16 = ddim_timesteps(50)[:16]
# The toy timesteps whose signal share (abar) equals SDXL's at its first 16 sampler steps, t = 981, 961, ..., 681
# (scaled-linear betas 0.00085 to 0.012, leading spacing). Board question 64.
SDXL_MATCHED = torch.tensor([712, 696, 680, 664, 648, 632, 616, 601, 585, 570, 555, 539, 524, 510, 495, 480])
WINDOWS = {"early": STEPS16, "sdxl": SDXL_MATCHED}
PIXELS = H * W
EVAL_SEED = 880_000
BASE = {"reader": "map", "input": "base", "updates": 4000}
VARIATIONS = [{"reader": "map", "input": "totals", "updates": 4000},
              {"reader": "map", "input": "signed", "updates": 4000},
              {"reader": "map", "input": "base", "updates": 12000}]


def load_data(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["data_hash"] not in DATA_HASHES:
        raise SystemExit(f"unexpected data hash {manifest['data_hash']}")
    with np.load(folder / "dataset.npz") as z:
        data = {s: {n: z[f"{s}__{n}"] for n in ARRAYS} for s in ("train", "seen_test", "unseen_test")}
    return data, manifest["data_hash"]


def take(split, mask):
    return {k: v[mask] for k, v in split.items()}


def inner_splits(data):
    """Pairs among 0 to 3 train, pairs holding 4 test. No inner-training canvas holds a 4."""
    no_four = lambda s: (s["asked"] <= 3).all(1) & (s["holds"] <= 3).all(1)
    has_four = (data["seen_test"]["asked"] == 4).any(1)
    return (take(data["train"], no_four(data["train"])), take(data["seen_test"], no_four(data["seen_test"])),
            take(data["seen_test"], has_four))


def make_head(cfg, dev):
    if cfg["reader"] == "raw":
        return PixelHead(1, pair_aware=True).to(dev)
    head = PixelHead(1, pair_aware=False)
    head.stem[0] = nn.Conv2d(4 if cfg["input"] == "totals" else 2, 64, 3, padding=1)
    return head.to(dev)


def maps(model, xt, t, pair, variant):
    """The two maps of where each asked label changes the noise prediction, each adding to 1, then scaled by the pixel count."""
    n = xt.shape[0]
    y = torch.cat([torch.full((n,), NULL, device=xt.device), pair[:, 0], pair[:, 1]])
    e0, ea, eb = model(xt.repeat(3, 1, 1, 1), t.repeat(3), y).chunk(3)
    channels, totals = [], []
    for d in (ea - e0, eb - e0):
        total = d.abs().flatten(1).sum(1).clamp_min(1e-12)[:, None, None, None]
        channels.append((d if variant == "signed" else d.abs()) / total * PIXELS)
        totals.append(total.log())
    out = torch.cat(channels, 1)
    if variant == "totals":
        out = torch.cat([out, torch.cat(totals, 1).expand(-1, -1, H, W) / 5], 1)
    return out


def inputs(cfg, model, x0, t, pair, noise):
    a = ABAR.to(x0.device)[t][:, None, None, None]
    xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
    if cfg["reader"] == "raw":
        return xt
    with torch.no_grad():
        return maps(model, xt, t, pair, cfg["input"])


def train(cfg, model, split, dev, updates=None, seed=0, batch=256, log_every=500):
    updates = updates or cfg["updates"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    head = make_head(cfg, dev).train()
    opt = torch.optim.Adam(head.parameters(), 1e-3)
    pos, neg = np.flatnonzero(split["target"] == 1), np.flatnonzero(split["target"] == 0)
    start, loss = time.time(), None
    for step in range(1, updates + 1):
        i = np.concatenate([rng.choice(pos, batch // 2), rng.choice(neg, batch - batch // 2)])
        x0 = to_x(split["x"][i], dev)
        pair = torch.from_numpy(split["asked"][i].astype(np.int64)).to(dev)
        y = torch.from_numpy(split["target"][i].astype(np.float32)).to(dev)
        t = STEPS16[torch.randint(0, len(STEPS16), (len(i),), generator=gen)].to(dev)
        noise = torch.randn(x0.shape, generator=gen).to(dev)
        loss = F.binary_cross_entropy_with_logits(head(inputs(cfg, model, x0, t, pair, noise), t, pair)[:, 0], y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % log_every == 0:
            print(json.dumps({"cfg": cfg, "update": step, "loss": round(loss.item(), 4),
                              "ms_per_update": round(1000 * (time.time() - start) / step, 1)}), flush=True)
    return head.eval(), {"final_loss": round(loss.item(), 4), "seconds": round(time.time() - start, 1)}


def balanced(pred, y):
    if not (y == 1).any() or not (y == 0).any():
        return None
    return 0.5 * (float(pred[y == 1].mean()) + float((~pred[y == 0]).mean()))


@torch.no_grad()
def evaluate(head, cfg, model, split, dev, draws=4, batch=512):
    """Read every canvas at all 16 steps with fixed noise draws; a logit above 0 counts as label 1."""
    head.eval()
    n = len(split["x"])
    pred = np.zeros((n, len(STEPS16), draws), bool)
    for s, step in enumerate(STEPS16.tolist()):
        for d in range(draws):
            gen = torch.Generator().manual_seed(EVAL_SEED + 100 * s + d)
            out = []
            for i in range(0, n, batch):
                x0 = to_x(split["x"][i:i + batch], dev)
                pair = torch.from_numpy(split["asked"][i:i + batch].astype(np.int64)).to(dev)
                t = torch.full((len(x0),), step, device=dev, dtype=torch.long)
                noise = torch.randn(x0.shape, generator=gen).to(dev)
                out.append((head(inputs(cfg, model, x0, t, pair, noise), t, pair)[:, 0] > 0).cpu().numpy())
            pred[:, s, d] = np.concatenate(out)
    y, kind = split["target"], split["kind"]
    versus = np.isin(kind, [KINDS.index("apart"), KINDS.index("wrong")])
    pairs = sorted({(int(a), int(b)) for a, b in split["asked"]})
    return {
        "canvases": n, "reads": int(pred.size),
        "share_called_1": float(pred.mean()),
        "balanced_accuracy": balanced(pred, y),
        "apart_vs_wrong": balanced(pred[versus], y[versus]),
        "per_step": [balanced(pred[:, s], y) for s in range(len(STEPS16))],
        "apart_vs_wrong_per_step": [balanced(pred[versus][:, s], y[versus]) for s in range(len(STEPS16))],
        "timesteps": STEPS16.tolist(),
        "accuracy_by_kind": {k: float((pred[kind == j] == (y[kind == j][:, None, None] == 1)).mean())
                             for j, k in enumerate(KINDS) if (kind == j).any()},
        "by_pair": {f"{a}-{b}": balanced(pred[m], y[m]) for a, b in pairs
                    for m in [(split["asked"][:, 0] == a) & (split["asked"][:, 1] == b)]},
    }


def meets_bars(seen, unseen):
    gap = abs(seen["balanced_accuracy"] - unseen["balanced_accuracy"])
    checks = {"unseen_ba_at_least_0.75": unseen["balanced_accuracy"] >= MIN_BA, "gap_at_most_0.05": gap <= MAX_GAP,
              "apart_vs_wrong_at_least_0.75": unseen["apart_vs_wrong"] >= MIN_APART_WRONG}
    return {"gap": gap, "checks": checks, "all": all(checks.values())}


def overfit_one(cfg, model, split, dev, updates=800):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    pos, neg = np.flatnonzero(split["target"] == 1), np.flatnonzero(split["target"] == 0)
    i = np.concatenate([rng.choice(pos, 8, replace=False), rng.choice(neg, 8, replace=False)])
    x0 = to_x(split["x"][i], dev)
    pair = torch.from_numpy(split["asked"][i].astype(np.int64)).to(dev)
    y = torch.from_numpy(split["target"][i].astype(np.float32)).to(dev)
    t = STEPS16[torch.randperm(16, generator=torch.Generator().manual_seed(0))].to(dev)
    noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(1)).to(dev)
    inp = inputs(cfg, model, x0, t, pair, noise).detach()
    head = make_head(cfg, dev).train()
    opt = torch.optim.Adam(head.parameters(), 1e-3)
    for _ in range(updates):
        loss = F.binary_cross_entropy_with_logits(head(inp, t, pair)[:, 0], y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        right = int(((head(inp, t, pair)[:, 0] > 0).float() == y).sum())
    return {"right_of_16": right, "pass": right >= OVERFIT_RIGHT, "final_loss": round(loss.item(), 5)}


def log_wandb(name, config, summary):
    try:
        import wandb
        run = wandb.init(project="poe-repair-animals-compose", entity="prime_lab", name=name, group="loop-8-v2",
                         config=config, settings=wandb.Settings(init_timeout=60))
        run.summary.update(summary)
        run_id = run.id
        run.finish()
        return run_id
    except Exception as error:  # the numbers are on disk either way
        print(f"W&B logging failed: {error}", flush=True)
        return None


def save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_step", "by_pair")})[:1800], flush=True)


READERS = {"map": BASE, "raw": {"reader": "raw", "input": "base", "updates": 4000}}


def stage_smoke(args, model, data, dev):
    small = take(data["seen_test"], np.arange(min(40, len(data["seen_test"]["x"]))))
    result = {}
    for name, cfg in READERS.items():
        head, info = train(cfg, model, data["train"], dev, updates=20, log_every=20)
        score = evaluate(head, cfg, model, small, dev, draws=1)
        result[name] = {"ms_per_update": round(1000 * info["seconds"] / 20, 1), "loss": info["final_loss"],
                        "balanced_accuracy_on_40": score["balanced_accuracy"], "weights": sum(p.numel() for p in head.parameters())}
    save(args.out / "smoke.json", result)


def stage_overfit(args, model, data, dev):
    result = {name: overfit_one(cfg, model, data["train"], dev) for name, cfg in READERS.items()}
    result["pass"] = all(r["pass"] for r in result.values())
    save(args.out / "overfit.json", result)
    return 0 if result["pass"] else 1


def stage_small(args, model, data, dev):
    result = {}
    for name, cfg in READERS.items():
        head, info = train(cfg, model, data["train"], dev, updates=1000)
        result[name] = {**info, "seen_test": evaluate(head, cfg, model, data["seen_test"], dev)}
    save(args.out / "small.json", result)


def stage_inner(args, model, data, dev):
    tr, seen, unseen = inner_splits(data)
    if set(np.unique(tr["holds"]).tolist()) - {-1, 0, 1, 2, 3}:
        raise SystemExit("inner training data holds a digit above 3")
    rows = []
    for cfg in [BASE] + VARIATIONS:
        head, info = train(cfg, model, tr, dev)
        s, u = evaluate(head, cfg, model, seen, dev), evaluate(head, cfg, model, unseen, dev)
        rows.append({"cfg": cfg, **info, "inner_seen": s, "inner_unseen": u, "bars": meets_bars(s, u),
                     "rank_value": min(u["balanced_accuracy"], u["apart_vs_wrong"])})
        print(json.dumps({"cfg": cfg, "inner_unseen_ba": u["balanced_accuracy"], "apart_vs_wrong": u["apart_vs_wrong"],
                          "all_bars": rows[-1]["bars"]["all"]}), flush=True)
        if cfg is BASE and rows[-1]["bars"]["all"]:
            break
    chosen = rows[0] if rows[0]["bars"]["all"] else max(rows, key=lambda r: r["rank_value"])
    save(args.out / "inner.json", {"chosen": chosen["cfg"], "base_passed": rows[0]["bars"]["all"], "rows": rows})
    (args.out / "chosen.json").write_text(json.dumps(chosen["cfg"], indent=1))


def stage_rehearse(args, model, data, dev):
    """Board question 64: the base map reader and the raw reader on the rehearsal split at one noise window. Never reads
    digits 5 to 9. A reader that calls label 1 on under 1% or over 99% of reads is flagged as not trained."""
    tr, seen, unseen = inner_splits(data)
    if set(np.unique(tr["holds"]).tolist()) - {-1, 0, 1, 2, 3}:
        raise SystemExit("inner training data holds a digit above 3")
    result = {"window": args.window, "timesteps": STEPS16.tolist(), "overfit": {}}
    for name, cfg in READERS.items():
        result["overfit"][name] = overfit_one(cfg, model, tr, dev)
        head, info = train(cfg, model, tr, dev)
        s, u = evaluate(head, cfg, model, seen, dev), evaluate(head, cfg, model, unseen, dev)
        result[name] = {"cfg": cfg, **info, "inner_seen": s, "inner_unseen": u, "bars": meets_bars(s, u),
                        "trained": 0.01 < u["share_called_1"] < 0.99}
        print(json.dumps({"reader": name, "overfit": result["overfit"][name]["right_of_16"],
                          "inner_unseen_ba": u["balanced_accuracy"], "apart_vs_wrong": u["apart_vs_wrong"],
                          "share_called_1": u["share_called_1"], "all_bars": result[name]["bars"]["all"]}), flush=True)
    m = result["map"]
    result["verdict"] = ("inconclusive" if not result["overfit"]["map"]["pass"] else "pass" if m["bars"]["all"] else "miss")
    save(args.out / "rehearse.json", result)


def relabel(split, task):
    """count: two digits apart, so apart and wrong are label 1. identity: apart (1) against wrong (0) only."""
    if task == "count":
        out = dict(split)
        out["target"] = split["apart"].copy()
        return out
    if task == "identity":
        return take(split, np.isin(split["kind"], [KINDS.index("apart"), KINDS.index("wrong")]))
    raise ValueError(task)


def stage_split(args, model, data, dev):
    """Board question 67: a count checker and a digit-identity checker, each trained and scored on the rehearsal split.
    Never reads digits 5 to 9. The count bar is V2's first two bars; the identity bar is board question 55's."""
    tr0, seen0, unseen0 = inner_splits(data)
    if set(np.unique(tr0["holds"]).tolist()) - {-1, 0, 1, 2, 3}:
        raise SystemExit("inner training data holds a digit above 3")
    result = {"window": args.window, "timesteps": STEPS16.tolist()}
    for task in ("count", "identity"):
        tr, seen, unseen = (relabel(s, task) for s in (tr0, seen0, unseen0))
        res = {"train_canvases": len(tr["x"]), "train_label_1_share": float(tr["target"].mean()),
               "test_canvases_holding_4": len(unseen["x"])}
        for name, cfg in READERS.items():
            fit = overfit_one(cfg, model, tr, dev)
            head, info = train(cfg, model, tr, dev)
            s, u = evaluate(head, cfg, model, seen, dev), evaluate(head, cfg, model, unseen, dev)
            gap = abs(s["balanced_accuracy"] - u["balanced_accuracy"])
            checks = {"holding_4_ba_at_least_0.75": u["balanced_accuracy"] >= MIN_BA}
            if task == "count":
                checks["gap_at_most_0.05"] = gap <= MAX_GAP
            res[name] = {"cfg": cfg, "overfit": fit, **info, "inner_seen": s, "inner_unseen": u, "gap": gap,
                         "checks": checks, "all": all(checks.values()), "trained": 0.01 < u["share_called_1"] < 0.99}
            print(json.dumps({"task": task, "reader": name, "overfit": fit["right_of_16"], "seen_ba": s["balanced_accuracy"],
                              "holding_4_ba": u["balanced_accuracy"], "gap": gap, "share_called_1": u["share_called_1"],
                              "all": res[name]["all"]}), flush=True)
        m = res["map"]
        res["verdict"] = "inconclusive" if not m["overfit"]["pass"] else "pass" if m["all"] else "miss"
        result[task] = res
    save(args.out / "split.json", result)


EXAMPLES = ("apart_1_3", "wrong_1_3", "apart_3_4", "wrong_3_4")
EXPLAIN_TS = (712, 616, 539, 480)


def explain(head, model, folder, dev):
    """Real maps, logits and push gradients for the zoom ladder's example canvases, with the ladder's own noise draw."""
    noise = torch.from_numpy(np.load(folder / "display-noise.npy")).float().to(dev)[None, None]
    out = {"names": list(EXAMPLES), "timesteps": list(EXPLAIN_TS), "x_t": [], "maps": [], "logit": [], "grad": []}
    for name in EXAMPLES:
        a, b = (int(v) for v in name.split("_")[-2:])
        x0 = torch.from_numpy(np.load(folder / f"{name}-clean.npy")).float().to(dev)[None, None]
        pair = torch.tensor([[a, b]], device=dev)
        rows = {k: [] for k in ("x_t", "maps", "logit", "grad")}
        for step in EXPLAIN_TS:
            t = torch.tensor([step], device=dev)
            ab = ABAR.to(dev)[t][:, None, None, None]
            with torch.enable_grad():
                xt = (ab.sqrt() * x0 + (1 - ab).sqrt() * noise).requires_grad_(True)
                m = maps(model, xt, t, pair, "base")
                logit = head(m, t, pair)[:, 0]
                g, = torch.autograd.grad(logit.sum(), xt)
            for k, v in (("x_t", xt), ("maps", m), ("logit", logit), ("grad", g)):
                rows[k].append(v.detach().cpu().numpy()[0])
        for k in rows:
            out[k].append(np.stack(rows[k]))
    return {k: (np.stack(v) if k not in ("names", "timesteps") else np.array(v)) for k, v in out.items()}


def stage_final_count(args, model, data, dev, data_hash):
    """Board question 69: retrain the count checker on digits 0 to 4 at the given window, save it, and read digits 5 to 9 once."""
    out = args.out / "final-count.json"
    if out.exists():
        raise SystemExit(f"{out} exists: digits 5 to 9 are read once")
    split = {s: relabel(data[s], "count") for s in ("train", "seen_test", "unseen_test")}
    result = {"task": "count", "window": args.window, "timesteps": STEPS16.tolist(), "data_hash": data_hash, "model": str(args.model)}
    for name, cfg in READERS.items():
        fit = overfit_one(cfg, model, split["train"], dev)
        head, info = train(cfg, model, split["train"], dev)
        torch.save({"head": head.state_dict(), "cfg": cfg, "task": "count", "window": args.window,
                    "timesteps": STEPS16.tolist(), "data_hash": data_hash}, args.out / f"count-head-{name}.pt")
        s, u = evaluate(head, cfg, model, split["seen_test"], dev), evaluate(head, cfg, model, split["unseen_test"], dev)
        gap = abs(s["balanced_accuracy"] - u["balanced_accuracy"])
        checks = {"unseen_ba_at_least_0.75": u["balanced_accuracy"] >= MIN_BA, "gap_at_most_0.05": gap <= MAX_GAP}
        result[name] = {"cfg": cfg, "overfit": fit, **info, "seen_test": s, "unseen_test": u, "gap": gap, "checks": checks,
                        "all": all(checks.values()), "trained": 0.01 < u["share_called_1"] < 0.99}
        print(json.dumps({"reader": name, "overfit": fit["right_of_16"], "seen_ba": s["balanced_accuracy"],
                          "unseen_ba": u["balanced_accuracy"], "gap": gap, "all": result[name]["all"]}), flush=True)
        if name == "map" and args.examples:
            np.savez_compressed(args.out / "count-explained.npz", **explain(head, model, args.examples, dev))
    m = result["map"]
    result["verdict"] = "inconclusive" if not m["overfit"]["pass"] else "support" if m["all"] else "null"
    if args.wandb_name:
        result["wandb_run"] = log_wandb(args.wandb_name, {"task": "count", "window": args.window, "data_hash": data_hash}, {
            "verdict": result["verdict"], "map_seen_ba": m["seen_test"]["balanced_accuracy"],
            "map_unseen_ba": m["unseen_test"]["balanced_accuracy"], "map_gap": m["gap"],
            "raw_unseen_ba": result["raw"]["unseen_test"]["balanced_accuracy"]})
    save(out, result)


def stage_final(args, model, data, dev, data_hash):
    out = args.out / "final.json"
    if out.exists():
        raise SystemExit(f"{out} exists: unseen_test is read once")
    overfit = json.loads((args.ladder / "overfit.json").read_text())
    chosen = json.loads((args.out / "chosen.json").read_text())
    configs = {"map": chosen, "raw": {"reader": "raw", "input": "base", "updates": chosen["updates"]}}
    result = {"chosen": chosen, "data_hash": data_hash, "model": str(args.model), "overfit_passed": overfit["pass"]}
    for name, cfg in configs.items():
        head, info = train(cfg, model, data["train"], dev)
        torch.save({"head": head.state_dict(), "cfg": cfg, "data_hash": data_hash}, args.out / f"head-{name}.pt")
        s, u = evaluate(head, cfg, model, data["seen_test"], dev), evaluate(head, cfg, model, data["unseen_test"], dev)
        result[name] = {**info, "seen_test": s, "unseen_test": u, "bars": meets_bars(s, u)}
    m = result["map"]
    below_floor = all(v < FLOOR_BA for v in m["seen_test"]["per_step"])
    result["verdict"] = ("inconclusive" if not overfit["pass"] or below_floor else "support" if m["bars"]["all"] else "null")
    result["raw_matches_map_on_unseen"] = (result["raw"]["unseen_test"]["balanced_accuracy"]
                                           >= m["unseen_test"]["balanced_accuracy"] - MAX_GAP)
    if args.wandb_name:
        result["wandb_run"] = log_wandb(args.wandb_name, {"chosen": chosen, "data_hash": data_hash, "model": str(args.model)}, {
            "verdict": result["verdict"], "map_seen_ba": m["seen_test"]["balanced_accuracy"],
            "map_unseen_ba": m["unseen_test"]["balanced_accuracy"], "map_unseen_apart_vs_wrong": m["unseen_test"]["apart_vs_wrong"],
            "raw_unseen_ba": result["raw"]["unseen_test"]["balanced_accuracy"],
            "raw_unseen_apart_vs_wrong": result["raw"]["unseen_test"]["apart_vs_wrong"]})
    save(out, result)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["smoke", "overfit", "small", "inner", "final", "rehearse", "split", "final-count"])
    ap.add_argument("--examples", type=Path, help="the zoom ladder's example canvases, for the final-count stage")
    ap.add_argument("--window", choices=sorted(WINDOWS), default="early",
                    help="early: the toy's sampler steps 0 to 15; sdxl: the toy timesteps matching SDXL's steps 0 to 15")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ladder", type=Path, help="the folder holding overfit.json, for the final stage")
    ap.add_argument("--wandb-name", default=None)
    args = ap.parse_args()
    global STEPS16
    STEPS16 = WINDOWS[args.window]
    args.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    model = load_unet(args.model, dev)
    data, data_hash = load_data(args.data)
    print(json.dumps({"stage": args.stage, "data_hash": data_hash, "train": len(data["train"]["x"]),
                      "device": torch.cuda.get_device_name()}), flush=True)
    if args.stage == "final":
        if args.ladder is None:
            raise SystemExit("final needs --ladder")
        stage_final(args, model, data, dev, data_hash)
        return 0
    if args.stage == "final-count":
        stage_final_count(args, model, data, dev, data_hash)
        return 0
    return {"smoke": stage_smoke, "overfit": stage_overfit, "small": stage_small, "inner": stage_inner,
            "rehearse": stage_rehearse, "split": stage_split}[args.stage](
        args, model, data, dev) or 0


if __name__ == "__main__":
    sys.exit(main())
