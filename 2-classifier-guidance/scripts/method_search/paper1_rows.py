#!/usr/bin/env python3
"""Step M1, Paper 1: the method against sampling each expert alone and pasting the results.

Four rows on the same pairs and seeds, scored by the digit judge, as the design file
artifacts/notes/progress/method-search/01-beat-the-baseline/design.md describes. The zero-push check runs in this process,
on this card, before any row is counted.

Usage: python paper1_rows.py --a <unet> --b <unet> --pair-head <pt> --sep-head <pt> --judge <pt> --out <dir>
       [--pairs 1-9,3-6,4-7,2-8,0-5] [--seeds 64] [--wandb-name <name>]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "mnist"))
import beat_baseline_bars as bars  # noqa: E402
import two_experts as te  # noqa: E402
import pixel_checker as pc  # noqa: E402
from canvas_unet import sample, to_uint8  # noqa: E402
from checker import load_unet  # noqa: E402
from judge import load as load_judge  # noqa: E402

CLEAN = te.CALLS[0]
ROWS = ("paste", "no_push", "push", "random")
PASTE_SEED_OFFSET = 100_000
PUSH_STRENGTH = 10.0
PUSH_CAP = 0.1
LOCK_T = 800
ETA = 1.0


def inked_half(canvas):
    left, right = canvas[:, :32], canvas[:, 32:]
    return left if (left > 128).sum() >= (right > 128).sum() else right


def paste(canvas_a, canvas_b, seed):
    """Expert A's half goes left when the seed's coin reads 0, and expert B's half takes the other side."""
    half_a, half_b = inked_half(canvas_a), inked_half(canvas_b)
    a_left = int(np.random.default_rng(1000 + seed).integers(2)) == 0
    return np.concatenate([half_a, half_b] if a_left else [half_b, half_a], axis=1)


def method_eps(A, B, a, b):
    return te.inferred_eps(A, B, a, b, True, LOCK_T)


def render(row, A, B, a, b, seeds, heads, device):
    if row == "paste":
        xa = to_uint8(sample(None, seeds, [], eps_fn=te.one_eps(A, a), device=device, eta=ETA)[0])
        xb = to_uint8(sample(None, [s + PASTE_SEED_OFFSET for s in seeds], [], eps_fn=te.one_eps(B, b), device=device, eta=ETA)[0])
        return np.stack([paste(ca, cb, s) for ca, cb, s in zip(xa, xb, seeds)])
    kind = {"no_push": "none", "push": "both", "random": "random"}[row]
    push = te.make_push(kind, a, b, heads, PUSH_STRENGTH)
    x = sample(None, seeds, [], eps_fn=method_eps(A, B, a, b), push_fn=push, cap=PUSH_CAP, device=device, eta=ETA)[0]
    te.LAYOUT_LOG.clear()
    return to_uint8(x)


def zero_push_check(A, B, a, b, seeds, heads, device):
    plain = sample(None, seeds, [], eps_fn=method_eps(A, B, a, b), push_fn=None, cap=PUSH_CAP, device=device, eta=ETA)[0]
    zero = sample(None, seeds, [], eps_fn=method_eps(A, B, a, b), push_fn=te.make_push("zero", a, b, heads),
                  cap=PUSH_CAP, device=device, eta=ETA)[0]
    te.LAYOUT_LOG.clear()
    return float((plain - zero).abs().max())


def log_wandb(name, config, summary):
    try:
        import wandb
        run = wandb.init(project="poe-repair-animals-compose", entity="prime_lab", name=name, group="loop-6-m1-paper1",
                         config=config, settings=wandb.Settings(init_timeout=60))
        run.summary.update(summary)
        run_id = run.id
        run.finish()
        return run_id
    except Exception as error:  # the run's numbers are on disk either way
        print(f"W&B logging failed: {error}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--pair-head", type=Path, required=True)
    ap.add_argument("--sep-head", type=Path, required=True)
    ap.add_argument("--judge", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs", default="1-9,3-6,4-7,2-8,0-5")
    ap.add_argument("--seeds", type=int, default=64)
    ap.add_argument("--wandb-name", default=None)
    args = ap.parse_args()
    device = "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    A, B = load_unet(args.a, device), load_unet(args.b, device)
    net = load_judge(args.judge, device)
    heads = {"pair": pc.load_head(args.pair_head, device), "sep": pc.load_head(args.sep_head, device)}
    pairs = te.parse_pairs(args.pairs)
    seeds = list(range(args.seeds))
    te.EVAL_SEEDS[:] = seeds

    zero_max = max(zero_push_check(A, B, a, b, seeds[:8], heads, device) for a, b in pairs)
    print(json.dumps({"zero_push_max_abs": zero_max, "bar": bars.ZERO_PUSH_MAX_ABS}), flush=True)
    if zero_max >= bars.ZERO_PUSH_MAX_ABS:
        (args.out / "zero_push_failed.json").write_text(json.dumps({"zero_push_max_abs": zero_max}))
        raise SystemExit("the zero push differs from the method without a push")

    clean = {row: {} for row in ROWS}
    tallies = {row: {} for row in ROWS}
    canvases = {}
    seconds = {}
    for row in ROWS:
        start = time.time()
        for a, b in pairs:
            x = render(row, A, B, a, b, seeds, heads, device)
            calls = [te.call_two(te.reads(net, c, device), (a, b)) for c in x]
            key = f"{a}-{b}"
            clean[row][key] = [int(c == CLEAN) for c in calls]
            tallies[row][key] = {k: calls.count(k) for k in te.CALLS}
            canvases[f"{row}/{key}"] = x
            (args.out / "sheets" / row).mkdir(parents=True, exist_ok=True)
            if len(x) == 64:
                te.sheet(x, calls, f"{row}, asked {a} and {b}: {sum(clean[row][key])} of 64 clean", args.out / "sheets" / row / f"digits-{a}-and-{b}.png")
        seconds[row] = round(time.time() - start, 1)
        print(json.dumps({"row": row, "clean": sum(sum(v) for v in clean[row].values()),
                          "of": len(pairs) * len(seeds), "seconds": seconds[row]}), flush=True)
    np.savez_compressed(args.out / "canvases.npz", **canvases)

    total = {row: sum(sum(v) for v in clean[row].values()) for row in ROWS}
    keys = [f"{a}-{b}" for a, b in pairs]
    diff = lambda r, s: [np.array(clean[r][k]) - np.array(clean[s][k]) for k in keys]
    se_method_paste = bars.se_within_groups(diff("push", "paste"))
    se_random_nopush = bars.se_within_groups(diff("random", "no_push"))
    verdict = bars.paper1_verdict(total["push"], total["paste"], total["no_push"], total["random"],
                                  se_method_paste, se_random_nopush, zero_max)
    result = {"pairs": keys, "seeds": [seeds[0], seeds[-1]], "canvases_per_row": len(keys) * len(seeds), "clean": total,
              "per_pair": {row: {k: sum(v) for k, v in clean[row].items()} for row in ROWS}, "tallies": tallies,
              "lead_push_over_paste": total["push"] - total["paste"], "se_push_minus_paste": se_method_paste,
              "lead_random_over_no_push": total["random"] - total["no_push"], "se_random_minus_no_push": se_random_nopush,
              "zero_push_max_abs": zero_max, "verdict": verdict, "seconds": seconds,
              "settings": {"strength": PUSH_STRENGTH, "cap": PUSH_CAP, "lock_t": LOCK_T, "eta": ETA},
              "checkers": [str(args.pair_head), str(args.sep_head)], "experts": [str(args.a), str(args.b)]}
    if args.wandb_name:
        result["wandb_run"] = log_wandb(args.wandb_name, result["settings"] | {"pairs": keys, "seeds": args.seeds},
                                        {"clean/" + k: v for k, v in total.items()} | {"verdict": verdict})
    (args.out / "result.json").write_text(json.dumps(result, indent=1))
    (args.out / "calls.json").write_text(json.dumps(clean))
    print(json.dumps({k: result[k] for k in ("clean", "lead_push_over_paste", "se_push_minus_paste", "verdict")}), flush=True)


if __name__ == "__main__":
    main()
