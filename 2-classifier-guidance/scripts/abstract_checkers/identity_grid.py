"""Step V6: can a checker read which digits are present? A 2 by 2 grid of input (two maps or signature) by window
(early or later), trained on `apart` against `wrong` with a larger wrong-digit pool, then one read of digits 5 to 9.

Stages:
    smoke   every cell on 1% of the pool with 20 updates
    grid    the cells named by --cells (all four by default), on the pairs among digits 0 to 3, scored on pairs holding 4
    final   the best passing cell retrained on digits 0 to 4, saved, and read once on digits 5 to 9

Usage:
    python identity_grid.py <stage> --model <unet.pt> --mnist <raw dir> --data <v1 data> --out <folder> [--cells base-early,...]
"""
import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "mnist")]
import build_env as be  # noqa: E402
import v2_learn as v2  # noqa: E402
from canvas_unet import NULL, H, W  # noqa: E402
from checker import load_unet  # noqa: E402
from pixel_checker import PixelHead  # noqa: E402

# Fixed before any run (abstract-checkers/10-reading-identity/design.md).
MIN_BA = 0.75
UPDATES, BATCH = 4000, 128
APART_PER_PAIR, WRONG_PER_PAIR = 1250, 1250
POOL_SEED = 8601
LATER = torch.tensor([466, 451, 437, 423, 409, 395, 381, 368, 354, 341, 327, 314, 301, 288, 275, 263])
WINDOWS = {"early": v2.SDXL_MATCHED, "later": LATER}
CELLS = {f"{i}-{w}": {"reader": "map", "input": i, "window": w, "updates": UPDATES}
         for i in ("base", "signature") for w in ("early", "later")}
APART, WRONG = be.KINDS.index("apart"), be.KINDS.index("wrong")


def build_pool(mnist, digits, scale=1.0):
    """apart and wrong canvases for every pair among `digits`, from the MNIST training file, with a hash."""
    source = be.load_source(Path(mnist), "train")
    rows = []
    for a, b in itertools.combinations(digits, 2):
        rng = np.random.default_rng(np.random.SeedSequence([POOL_SEED, len(digits), a, b]))
        for kind, n in (("apart", APART_PER_PAIR), ("wrong", WRONG_PER_PAIR)):
            rows += [be.make_canvas(kind, a, b, digits, source, rng) for _ in range(max(1, round(n * scale)))]
    order = np.random.default_rng(np.random.SeedSequence([POOL_SEED, 99])).permutation(len(rows))
    rows = [rows[i] for i in order]
    types = {"x": np.uint8, "asked": np.int8, "holds": np.int8, "corner": np.int8, "src": np.int32, "kind": np.uint8,
             "target": np.uint8, "n_digits": np.uint8, "apart": np.uint8}
    pool = {k: np.array([r[k] for r in rows], types[k]) for k in be.ARRAYS}
    if set(np.unique(pool["holds"]).tolist()) - set(digits):
        raise SystemExit("a pasted digit lies outside the pool's range")
    h = hashlib.sha256(json.dumps({"seed": POOL_SEED, "digits": list(digits), "scale": scale}).encode())
    for k in be.ARRAYS:
        h.update(pool[k].tobytes())
    return pool, h.hexdigest()


def apart_wrong(split, mask=None):
    keep = np.isin(split["kind"], [APART, WRONG])
    return v2.take(split, keep if mask is None else keep & mask)


def signature(model, xt, t, pair):
    """d_a, d_b, the mean of the other eight differences, and their largest absolute value, divided by the canvas's
    mean absolute difference over all ten labels. Scoring passes batches of 512, so the eleven U-Net calls run in
    pieces of 128, the size training uses."""
    if xt.shape[0] > 128:
        return torch.cat([signature(model, xt[i:i + 128], t[i:i + 128], pair[i:i + 128])
                          for i in range(0, xt.shape[0], 128)])
    n = xt.shape[0]
    y = torch.cat([torch.full((n,), NULL, device=xt.device)] + [torch.full((n,), c, device=xt.device) for c in range(10)])
    out = model(xt.repeat(11, 1, 1, 1), t.repeat(11), y).chunk(11)
    d = torch.stack([o - out[0] for o in out[1:]], 1)[:, :, 0]          # n, 10, H, W
    idx = torch.arange(n, device=xt.device)
    da, db = d[idx, pair[:, 0]], d[idx, pair[:, 1]]
    keep = torch.ones(n, 10, dtype=torch.bool, device=xt.device)
    keep[idx, pair[:, 0]] = False
    keep[idx, pair[:, 1]] = False
    others = d[keep].view(n, 8, H, W)
    scale = d.abs().mean((1, 2, 3)).clamp_min(1e-12)[:, None, None, None]
    return torch.stack([da, db, others.mean(1), others.abs().amax(1)], 1) / scale


_base_inputs = v2.inputs


def inputs(cfg, model, x0, t, pair, noise):
    if cfg["input"] != "signature":
        return _base_inputs(cfg, model, x0, t, pair, noise)
    a = v2.ABAR.to(x0.device)[t][:, None, None, None]
    xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
    with torch.no_grad():
        return signature(model, xt, t, pair)


def make_head(cfg, dev):
    head = PixelHead(1, pair_aware=False)
    head.stem[0] = nn.Conv2d(4 if cfg["input"] == "signature" else 2, 64, 3, padding=1)
    return head.to(dev)


v2.inputs, v2.make_head = inputs, make_head


def run_cell(name, cfg, model, train, tests, dev, updates=None):
    v2.STEPS16 = WINDOWS[cfg["window"]]
    fit = v2.overfit_one(cfg, model, train, dev)
    head, info = v2.train(cfg, model, train, dev, updates=updates or cfg["updates"], batch=BATCH)
    scores = {k: v2.evaluate(head, cfg, model, s, dev) for k, s in tests.items()}
    return head, {"cfg": cfg, "timesteps": v2.STEPS16.tolist(), "overfit": fit, **info, **scores}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["smoke", "grid", "final"])
    for flag in ("model", "mnist", "data", "out"):
        ap.add_argument(f"--{flag}", type=Path, required=True)
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--wandb-name", default=None)
    args = ap.parse_args()
    dev = "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    model = load_unet(args.model, dev)
    data, data_hash = v2.load_data(args.data)
    seen = data["seen_test"]
    holds4 = apart_wrong(seen, (seen["asked"] == 4).any(1))
    trained = apart_wrong(seen, (seen["asked"] <= 3).all(1))
    print(json.dumps({"stage": args.stage, "v1_hash": data_hash, "holds4": len(holds4["x"]), "trained": len(trained["x"]),
                      "device": torch.cuda.get_device_name()}), flush=True)

    if args.stage in ("smoke", "grid"):
        scale = 0.01 if args.stage == "smoke" else 1.0
        pool, pool_hash = build_pool(args.mnist, (0, 1, 2, 3), scale)
        print(json.dumps({"pool": len(pool["x"]), "wrong": int((pool["kind"] == WRONG).sum()), "pool_hash": pool_hash}), flush=True)
        tests = {"holds4": holds4, "trained": trained}
        if args.stage == "smoke":
            tests = {k: v2.take(v, np.arange(min(40, len(v["x"])))) for k, v in tests.items()}
        result = {"stage": args.stage, "pool_hash": pool_hash, "pool": len(pool["x"]), "cells": {}}
        for name in args.cells.split(","):
            _, r = run_cell(name, CELLS[name], model, pool, tests, dev, updates=20 if args.stage == "smoke" else None)
            r["passes"] = r["holds4"]["balanced_accuracy"] >= MIN_BA
            r["gap"] = abs(r["trained"]["balanced_accuracy"] - r["holds4"]["balanced_accuracy"])
            result["cells"][name] = r
            print(json.dumps({"cell": name, "overfit": r["overfit"]["right_of_16"], "holds4": r["holds4"]["balanced_accuracy"],
                              "trained": r["trained"]["balanced_accuracy"], "passes": r["passes"], "seconds": r["seconds"]}), flush=True)
        tag = "smoke" if args.stage == "smoke" else "grid-" + args.cells.replace(",", "+")
        (args.out / f"{tag}.json").write_text(json.dumps(result, indent=1))
        return 0

    out = args.out / "final.json"
    if out.exists():
        raise SystemExit(f"{out} exists: digits 5 to 9 are read once")
    cells = {}
    for f in sorted(args.out.glob("grid-*.json")):
        cells.update(json.loads(f.read_text())["cells"])
    passing = {k: v for k, v in cells.items() if v["passes"] and v["overfit"]["pass"]}
    if not passing:
        raise SystemExit("no cell passed the grid; digits 5 to 9 are not read")
    name = max(passing, key=lambda k: passing[k]["holds4"]["balanced_accuracy"])
    pool, pool_hash = build_pool(args.mnist, (0, 1, 2, 3, 4))
    tests = {"seen_test": apart_wrong(seen), "unseen_test": apart_wrong(data["unseen_test"])}
    head, r = run_cell(name, CELLS[name], model, pool, tests, dev)
    torch.save({"head": head.state_dict(), "cfg": CELLS[name], "task": "identity", "window": CELLS[name]["window"],
                "timesteps": WINDOWS[CELLS[name]["window"]].tolist(), "pool_hash": pool_hash}, args.out / "identity-head.pt")
    u, s = r["unseen_test"]["balanced_accuracy"], r["seen_test"]["balanced_accuracy"]
    r.update({"chosen_cell": name, "pool_hash": pool_hash, "verdict": "support" if u >= MIN_BA else "null",
              "gap": abs(u - s), "overfit_passed": r["overfit"]["pass"]})
    if args.wandb_name:
        r["wandb_run"] = v2.log_wandb(args.wandb_name, {"cell": name, "pool_hash": pool_hash},
                                      {"verdict": r["verdict"], "unseen_ba": u, "seen_ba": s})
    out.write_text(json.dumps(r, indent=1))
    print(json.dumps({k: r[k] for k in ("chosen_cell", "verdict", "gap")} | {"unseen": u, "seen": s}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
