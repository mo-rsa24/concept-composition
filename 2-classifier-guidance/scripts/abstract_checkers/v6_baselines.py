"""Two checks on V6's identity checker before V7 (abstract-checkers/10-reading-identity/design.md, "Two checks before V7").

Stages:
    untrained  the rule that needs no training, on V6's 1,250 canvases of digits 5 to 9, against the checker per pair
    sampler    V6's checker read on the sampler's states at steps 27 to 36 of V3's pushed renders, labelled by the judge
    smoke      both on a few canvases and one pair at 4 seeds

Usage:
    python v6_baselines.py <stage> --model <unet.pt> --data <v1 data> --v6 <v6 folder> --count-head <count-head-map.pt>
                           --judge <judge.pt> --out <folder>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "mnist")]
import identity_grid as ig  # noqa: E402  (also installs the signature input into v2_learn)
import v2_learn as v2  # noqa: E402
import v3_compose as v3  # noqa: E402
import two_experts as te  # noqa: E402
from canvas_unet import NULL, ddim_timesteps, sample  # noqa: E402
from checker import load_unet  # noqa: E402
from judge import load as load_judge  # noqa: E402

# Fixed before either run (10-reading-identity/design.md, "Two checks before V7").
MIN_LEAD_SE = 1.96
SAMPLER_MIN_BA = 0.75
MIN_CLASS = 40
SAMPLER_SEEDS = list(range(1000, 1096))
SAMPLER_B = 0.4
IDENTITY_STEPS = [int(t) for t in ddim_timesteps(50) if 263 <= int(t) <= 466]   # sampler steps 27 to 36
CHUNK = 128


def fits(model, xt, t, noise):
    """Mean squared difference between each digit's noise prediction and the added noise: (n, 10)."""
    out = []
    for i in range(0, len(xt), CHUNK):
        x, tt, nz = xt[i:i + CHUNK], t[i:i + CHUNK], noise[i:i + CHUNK]
        n = len(x)
        y = torch.cat([torch.full((n,), c, device=x.device) for c in range(10)])
        eps = model(x.repeat(10, 1, 1, 1), tt.repeat(10), y).view(10, n, *x.shape[1:])
        out.append(((eps - nz[None]) ** 2).flatten(2).mean(2).T)
    return torch.cat(out)


def rule(err, pair):
    """1 when both asked digits fit better (smaller error) than the best of the other eight."""
    idx = torch.arange(len(err), device=err.device)
    worst_asked = torch.maximum(err[idx, pair[:, 0]], err[idx, pair[:, 1]])
    others = err.clone()
    others[idx, pair[:, 0]] = float("inf")
    others[idx, pair[:, 1]] = float("inf")
    return worst_asked < others.min(1).values


def balanced(pred, y):
    return v2.balanced(np.asarray(pred, bool), np.asarray(y))


def by_pair(pred, split):
    pairs = sorted({(int(a), int(b)) for a, b in split["asked"]})
    return {f"{a}-{b}": balanced(pred[m], split["target"][m]) for a, b in pairs
            for m in [(split["asked"][:, 0] == a) & (split["asked"][:, 1] == b)]}


def paired_lead(checker_pp, rule_pp):
    keys = sorted(checker_pp)
    d = np.array([checker_pp[k] - rule_pp[k] for k in keys])
    rng = np.random.default_rng(8604)
    means = d[rng.integers(len(d), size=(10000, len(d)))].mean(1)
    return float(d.mean()), float(means.std(ddof=1))


@torch.no_grad()
def stage_untrained(model, split, dev, steps, draws=4, batch=512):
    """The same reads as v2_learn.evaluate: per step and draw, one generator, noise drawn batch by batch."""
    n = len(split["x"])
    per_read = np.zeros((n, len(steps), draws), bool)
    summed = torch.zeros(n, 10, device=dev)
    for s, step in enumerate(steps):
        for d in range(draws):
            gen = torch.Generator().manual_seed(v2.EVAL_SEED + 100 * s + d)
            for i in range(0, n, batch):
                x0 = v2.to_x(split["x"][i:i + batch], dev)
                pair = torch.from_numpy(split["asked"][i:i + batch].astype(np.int64)).to(dev)
                t = torch.full((len(x0),), step, device=dev, dtype=torch.long)
                noise = torch.randn(x0.shape, generator=gen).to(dev)
                a = v2.ABAR.to(dev)[t][:, None, None, None]
                err = fits(model, a.sqrt() * x0 + (1 - a).sqrt() * noise, t, noise)
                per_read[i:i + batch, s, d] = rule(err, pair).cpu().numpy()
                summed[i:i + batch] += err
    pair_all = torch.from_numpy(split["asked"].astype(np.int64)).to(dev)
    return per_read, rule(summed, pair_all).cpu().numpy()


def judge_label(net, canvas, pair, dev):
    rd = te.reads(net, canvas, dev)
    call = te.call_two(rd, pair)
    if call == v3.CLEAN:
        return 1, call
    if len(rd) == 2 and sorted(int(x >= 32) for x, _ in rd) == [0, 1]:
        d = [p[:10] / p[:10].sum() for _, p in rd]
        if all(q.max() >= 0.9 for q in d) and sorted(int(q.argmax()) for q in d) != sorted(pair):
            return 0, call
    return None, call


def stage_sampler(model, ihead, chead, net, pairs, seeds, dev):
    rows = []
    for a, bb in pairs:
        store = {}
        inner = v3.make_push("push", SAMPLER_B, chead, model, a, bb, 4242 + 10 * a + bb)

        def record(x, t, eps):
            if int(t[0]) in IDENTITY_STEPS:
                store[int(t[0])] = x.detach().clone()
            return inner(x, t, eps)
        x, _ = sample(model, seeds, [a, bb], g=1.0, steps=50, cap=1.0, push_fn=record)
        pair = torch.tensor([[a, bb]], device=dev).expand(len(seeds), 2)
        logits = {}
        with torch.no_grad():
            for t in IDENTITY_STEPS:
                tt = torch.full((len(seeds),), t, device=dev, dtype=torch.long)
                logits[t] = ihead(ig.signature(model, store[t], tt, pair), tt, pair)[:, 0].cpu().numpy()
        for k, s in enumerate(seeds):
            label, call = judge_label(net, x[k], (a, bb), dev)
            rows.append({"pair": f"{a}-{bb}", "seed": s, "call": call, "label": label,
                         "logits": [float(logits[t][k]) for t in IDENTITY_STEPS]})
        print(json.dumps({"pair": f"{a}-{bb}", "labelled_1": sum(r["label"] == 1 for r in rows if r["pair"] == f"{a}-{bb}"),
                          "labelled_0": sum(r["label"] == 0 for r in rows if r["pair"] == f"{a}-{bb}")}), flush=True)
    kept = [r for r in rows if r["label"] is not None]
    y = np.array([r["label"] for r in kept], np.uint8)
    pred = np.array([r["logits"] for r in kept]) > 0
    per_step = [balanced(pred[:, j], y) for j in range(len(IDENTITY_STEPS))]
    ok = [v for v in per_step if v is not None]
    mean = float(np.mean(ok)) if ok else None
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    verdict = ("inconclusive: fewer than 40 canvases in a label" if min(n1, n0) < MIN_CLASS
               else "reads sampler states" if mean >= SAMPLER_MIN_BA else "does not read sampler states")
    return {"identity_steps": IDENTITY_STEPS, "seeds": [seeds[0], seeds[-1]], "canvases": len(rows), "labelled_1": n1,
            "labelled_0": n0, "per_step": per_step, "mean_ba": mean, "verdict": verdict, "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["smoke", "untrained", "sampler"])
    for flag in ("model", "data", "v6", "count_head", "judge", "out"):
        ap.add_argument(f"--{flag.replace('_', '-')}", dest=flag, type=Path, required=True)
    args = ap.parse_args()
    dev = "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    model = load_unet(args.model, dev)
    data, data_hash = v2.load_data(args.data)
    ck = torch.load(args.v6 / "identity-head.pt", map_location=dev, weights_only=False)
    ihead = ig.make_head(ck["cfg"], dev)
    ihead.load_state_dict(ck["head"])
    ihead.eval()
    for p in ihead.parameters():
        p.requires_grad_(False)
    steps = torch.tensor(ck["timesteps"])
    v2.STEPS16 = steps
    print(json.dumps({"stage": args.stage, "v1_hash": data_hash, "checker_window": ck["window"], "identity_steps": IDENTITY_STEPS,
                      "device": torch.cuda.get_device_name()}), flush=True)
    start = time.time()
    unseen = ig.apart_wrong(data["unseen_test"])
    if args.stage in ("smoke", "untrained"):
        split = v2.take(unseen, np.arange(48)) if args.stage == "smoke" else unseen
        per_read, summed = stage_untrained(model, split, dev, steps.tolist())
        y = split["target"]
        rpp = by_pair(per_read, split)
        r = {"canvases": len(y), "per_read_ba": balanced(per_read, y), "summed_ba": balanced(summed, y),
             "per_read_by_pair": rpp, "summed_by_pair": by_pair(summed, split),
             "per_read_per_step": [balanced(per_read[:, s], y) for s in range(len(steps))]}
        if args.stage == "untrained":
            final = json.loads((args.v6 / "final.json").read_text())
            cpp = final["unseen_test"]["by_pair"]
            lead, se = paired_lead(cpp, rpp)
            r.update({"checker_ba": final["unseen_test"]["balanced_accuracy"], "checker_by_pair": cpp, "lead": lead,
                      "lead_se": se, "checker_adds": lead > MIN_LEAD_SE * se})
        r["seconds"] = round(time.time() - start, 1)
        (args.out / f"{args.stage}-untrained.json").write_text(json.dumps(r, indent=1))
        print(json.dumps({k: v for k, v in r.items() if not isinstance(v, (dict, list))}), flush=True)
    if args.stage in ("smoke", "sampler"):
        chead, _ = v3.load_head(args.count_head, dev)
        net = load_judge(args.judge, dev)
        pairs, seeds = (v3.UNSEEN[:1], SAMPLER_SEEDS[:4]) if args.stage == "smoke" else (v3.UNSEEN, SAMPLER_SEEDS)
        r = stage_sampler(model, ihead, chead, net, pairs, seeds, dev)
        r["seconds"] = round(time.time() - start, 1)
        (args.out / f"{args.stage}-sampler.json").write_text(json.dumps(r, indent=1))
        print(json.dumps({k: v for k, v in r.items() if k != "rows"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
