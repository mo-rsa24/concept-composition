"""Retrain V6's identity checker on the sampler's own states and read fresh digits 5 to 9
(abstract-checkers/10-reading-identity/design.md, "Retraining on the sampler's own states"; board question 94).

Stages:
    smoke    one seen pair at 8 seeds collected, 20 updates, one unseen pair at 8 fresh seeds read
    collect  the 10 pairs among digits 0 to 4 at seeds 2000 to 2959: states at sampler steps 27 to 36, judge labels
    train    the overfit check, the training on the collected states, the read of digits 5 to 9 at seeds 3000 to 3095,
             and the read of V1's pasted digits 5 to 9 reported beside it

Usage:
    python v6_resample.py <stage> --model <unet.pt> --data <v1 data> --v6 <v6 folder> --count-head <count-head-map.pt>
                          --judge <judge.pt> --out <folder>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "mnist")]
import identity_grid as ig  # noqa: E402
import v2_learn as v2  # noqa: E402
import v3_compose as v3  # noqa: E402
import v6_baselines as vb  # noqa: E402
from canvas_unet import sample  # noqa: E402
from checker import load_unet  # noqa: E402
from judge import load as load_judge  # noqa: E402
from v3_compose import to_uint8  # noqa: E402

# Fixed before the run (10-reading-identity/design.md, "Retraining on the sampler's own states").
TRAIN_SEEDS = list(range(2000, 2960))
READ_SEEDS = list(range(3000, 3096))
SMOKE_READ_SEEDS = list(range(2960, 2968))   # never used by any step, and outside both ranges above
MIN_BA = vb.SAMPLER_MIN_BA          # 0.75
MIN_CLASS = vb.MIN_CLASS            # 40
UPDATES, BATCH = 4000, 128
OVERFIT_RIGHT = 15
STEPS = vb.IDENTITY_STEPS           # sampler steps 27 to 36
RENDER_BATCH = 96
CFG = ig.CELLS["signature-later"]


def collect(model, chead, net, pairs, seeds, dev):
    """Render with V3's count push, keep each labelled canvas's states at STEPS."""
    xs, ys, prs, calls = [], [], [], {}
    for a, bb in pairs:
        for i in range(0, len(seeds), RENDER_BATCH):
            ss = seeds[i:i + RENDER_BATCH]
            store = {}
            inner = v3.make_push("push", vb.SAMPLER_B, chead, model, a, bb, 4242 + 10 * a + bb)

            def record(x, t, eps):
                if int(t[0]) in STEPS:
                    store[int(t[0])] = x.detach().half().cpu()
                return inner(x, t, eps)
            x, _ = sample(model, ss, [a, bb], g=1.0, steps=50, cap=1.0, push_fn=record)
            canv = to_uint8(x)
            states = torch.stack([store[t] for t in STEPS], 1)          # n, 10, 1, H, W
            for k in range(len(ss)):
                label, call = vb.judge_label(net, canv[k], (a, bb), dev)
                calls[call] = calls.get(call, 0) + 1
                if label is not None:
                    xs.append(states[k])
                    ys.append(label)
                    prs.append((a, bb))
        print(json.dumps({"pair": f"{a}-{bb}", "kept": len(ys), "label_1": int(sum(ys)), "calls": calls}), flush=True)
    x = torch.stack(xs).numpy() if xs else np.zeros((0, len(STEPS), 1, 32, 64), np.float16)
    return {"x": x, "y": np.array(ys, np.uint8), "pair": np.array(prs, np.int64).reshape(-1, 2)}, calls


def batch_inputs(model, data, idx, steps_idx, dev):
    x = torch.from_numpy(data["x"][idx, steps_idx]).float().to(dev)
    t = torch.tensor([STEPS[j] for j in steps_idx], device=dev, dtype=torch.long)
    pair = torch.from_numpy(data["pair"][idx]).to(dev)
    with torch.no_grad():
        sig = ig.signature(model, x, t, pair)
    return sig, t, pair


def fit(model, data, dev, idx_pos, idx_neg, updates, batch, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    head = ig.make_head(CFG, dev).train()
    opt = torch.optim.Adam(head.parameters(), 1e-3)
    start, loss = time.time(), None
    for step in range(1, updates + 1):
        i = np.concatenate([rng.choice(idx_pos, batch // 2), rng.choice(idx_neg, batch - batch // 2)])
        j = rng.integers(len(STEPS), size=len(i))
        sig, t, pair = batch_inputs(model, data, i, j, dev)
        y = torch.from_numpy(data["y"][i].astype(np.float32)).to(dev)
        loss = F.binary_cross_entropy_with_logits(head(sig, t, pair)[:, 0], y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(json.dumps({"update": step, "loss": round(loss.item(), 4),
                              "ms_per_update": round(1000 * (time.time() - start) / step, 1)}), flush=True)
    return head.eval(), {"final_loss": round(loss.item(), 4), "seconds": round(time.time() - start, 1)}


@torch.no_grad()
def overfit(model, data, dev):
    rng = np.random.default_rng(0)
    pos, neg = np.flatnonzero(data["y"] == 1), np.flatnonzero(data["y"] == 0)
    i = np.concatenate([rng.choice(pos, 8, replace=False), rng.choice(neg, 8, replace=False)])
    j = np.full(16, len(STEPS) // 2)
    with torch.enable_grad():
        torch.manual_seed(0)
        head = ig.make_head(CFG, dev).train()
        opt = torch.optim.Adam(head.parameters(), 1e-3)
        sig, t, pair = batch_inputs(model, data, i, j, dev)
        y = torch.from_numpy(data["y"][i].astype(np.float32)).to(dev)
        for _ in range(800):
            loss = F.binary_cross_entropy_with_logits(head(sig, t, pair)[:, 0], y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    right = int(((head.eval()(sig, t, pair)[:, 0] > 0).float() == y).sum())
    return {"right_of_16": right, "pass": right >= OVERFIT_RIGHT, "final_loss": round(loss.item(), 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["smoke", "collect", "train"])
    for flag in ("model", "data", "v6", "count_head", "judge", "out"):
        ap.add_argument(f"--{flag.replace('_', '-')}", dest=flag, type=Path, required=True)
    args = ap.parse_args()
    dev = "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    model = load_unet(args.model, dev)
    chead, _ = v3.load_head(args.count_head, dev)
    net = load_judge(args.judge, dev)
    print(json.dumps({"stage": args.stage, "steps": STEPS, "train_seeds": [TRAIN_SEEDS[0], TRAIN_SEEDS[-1]],
                      "read_seeds": [READ_SEEDS[0], READ_SEEDS[-1]], "device": torch.cuda.get_device_name()}), flush=True)
    start = time.time()

    if args.stage in ("smoke", "collect"):
        pairs, seeds = (v3.SEEN[:2], TRAIN_SEEDS[:32]) if args.stage == "smoke" else (v3.SEEN, TRAIN_SEEDS)
        data, calls = collect(model, chead, net, pairs, seeds, dev)
        name = "smoke-states.npz" if args.stage == "smoke" else "train-states.npz"
        np.savez_compressed(args.out / name, **data)
        info = {"stage": args.stage, "kept": int(len(data["y"])), "label_1": int(data["y"].sum()),
                "label_0": int((data["y"] == 0).sum()), "calls": calls, "seconds": round(time.time() - start, 1)}
        (args.out / f"{args.stage}-collect.json").write_text(json.dumps(info, indent=1))
        print(json.dumps(info), flush=True)
        if args.stage == "collect":
            return 0

    name = "smoke-states.npz" if args.stage == "smoke" else "train-states.npz"
    with np.load(args.out / name) as z:
        data = {k: z[k] for k in ("x", "y", "pair")}
    pos, neg = np.flatnonzero(data["y"] == 1), np.flatnonzero(data["y"] == 0)
    result = {"stage": args.stage, "train_label_1": int(len(pos)), "train_label_0": int(len(neg))}
    if args.stage == "train" and min(len(pos), len(neg)) < MIN_CLASS:
        result["verdict"] = "inconclusive: fewer than 40 canvases in a training label"
        (args.out / "train-read.json").write_text(json.dumps(result, indent=1))
        print(json.dumps(result), flush=True)
        return 0
    if min(len(pos), len(neg)) >= 8:
        result["overfit"] = overfit(model, data, dev)
    if min(len(pos), len(neg)) == 0:     # smoke only: nothing to train on, so the read runs on an untrained head
        head, info = ig.make_head(CFG, dev).eval(), {"note": "a label was empty; head untrained"}
    else:
        head, info = fit(model, data, dev, pos, neg, 20 if args.stage == "smoke" else UPDATES, BATCH, seed=0)
    result.update(info)
    if args.stage == "train":
        torch.save({"head": head.state_dict(), "cfg": CFG, "task": "identity", "trained_on": "sampler states",
                    "steps": STEPS, "train_seeds": [TRAIN_SEEDS[0], TRAIN_SEEDS[-1]]}, args.out / "identity-head-sampler.pt")
    for p in head.parameters():
        p.requires_grad_(False)
    pairs, seeds = (v3.UNSEEN[:1], SMOKE_READ_SEEDS) if args.stage == "smoke" else (v3.UNSEEN, READ_SEEDS)
    read = vb.stage_sampler(model, head, chead, net, pairs, seeds, dev)
    overfit_ok = result.get("overfit", {}).get("pass", args.stage == "smoke")
    if args.stage == "train" and not overfit_ok:
        read["verdict"] = "inconclusive: the overfit check failed"
    result["read"] = {k: v for k, v in read.items() if k != "rows"}
    result["read_rows"] = read["rows"]
    if args.stage == "train":
        data_v1, _ = v2.load_data(args.data)
        v2.STEPS16 = ig.WINDOWS["later"]
        result["pasted_5_to_9"] = v2.evaluate(head, CFG, model, ig.apart_wrong(data_v1["unseen_test"]), dev)
    result["seconds_total"] = round(time.time() - start, 1)
    (args.out / f"{args.stage}-read.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({"verdict": read["verdict"], "mean_ba": read["mean_ba"], "label_1": read["labelled_1"],
                      "label_0": read["labelled_0"], "overfit": result.get("overfit"),
                      "pasted": result.get("pasted_5_to_9", {}).get("balanced_accuracy")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
