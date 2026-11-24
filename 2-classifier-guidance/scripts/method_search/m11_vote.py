#!/usr/bin/env python3
"""Step M11, Paper 1 on digits: does the experts' own vote place the digits, or would any held side do?

Five arms on the five tuning pairs at fresh seeds 192 to 255, scored by the digit judge, as
artifacts/notes/progress/method-search/06-vote-or-placement-on-digits/design.md describes. The arms come from Loop 10's design,
artifacts/notes/progress/score-space/10-paper-1-placement-or-sampler.md.

  vote          the side chosen by the experts' vote in steps 0 to 9 of 50, then held, the other half hidden; 5 corrector steps
  coin          the same, with the side chosen by a coin flip per seed
  coin_no_corr  the coin arm with no corrector steps
  shared        one shared canvas, no halves: the plain product over the blank-canvas base; 5 corrector steps
  paste         each expert samples alone, and the two inked halves are joined (M1's paste, unchanged)

Every arm but paste uses both checker pushes from M1's Paper 1 checkers, strength 10, capped at 0.1.

Usage: python m11_vote.py --a <unet> --b <unet> --pair-head <pt> --sep-head <pt> --judge <pt> --out <dir>
       [--pairs 1-9,3-6,4-7,2-8,0-5] [--seed0 192] [--seeds 64] [--arms vote,coin,...] [--wandb-name <name>]
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
import paper1_rows as p1  # noqa: E402
import pixel_checker as pc  # noqa: E402
import two_experts as te  # noqa: E402
from canvas_unet import ABAR, sample, to_uint8  # noqa: E402
from checker import load_unet  # noqa: E402
from judge import load as load_judge  # noqa: E402

ARMS = ("vote", "coin", "coin_no_corr", "shared", "paste")
CORRECTOR = (5, 3.0)            # 5 Langevin steps at step size 3 x beta_t, the setting behind 298 of 320
REAL, NONE = 64, 32             # Loop 10's bars, from plans 32 and 33: a difference of at least 64 of 320 is real, within 32 is none
CLEAN = te.CALLS[0]


def coin_side(seed):
    """True when expert A takes the left half; the same coin M1's paste and plan 33's region runs use."""
    return int(np.random.default_rng(1000 + seed).integers(2)) == 0


def held_eps(A, B, a, b, a_left):
    """The vote arm's composition with the side given in advance: the other half is hidden from each expert, as in
    two_experts.inferred_eps with occlusion, and each expert supplies the prediction on its own half."""
    left_t = torch.tensor(a_left)

    def f(x, t):
        dev = x.device
        ya = torch.full((x.shape[0],), a, device=dev)
        yb = torch.full((x.shape[0],), b, device=dev)
        left = left_t.to(dev)[:, None, None, None]
        ab = ABAR.to(dev)[t][:, None, None, None]
        g = torch.Generator(device=dev).manual_seed(12345 + int(t[0]))
        blank = -ab.sqrt() + (1 - ab).sqrt() * torch.randn(x.shape, generator=g, device=dev)
        m = torch.zeros_like(x)
        m[..., :32] = 1
        own_a = torch.where(left, m, 1 - m)
        own_b = 1 - own_a
        eA = A(own_a * x + (1 - own_a) * blank, t, ya)
        eB = B(own_b * x + (1 - own_b) * blank, t, yb)
        return own_a * eA + own_b * eB
    return f


def held_matches_vote(A, B, a, b, device):
    """Control: with the side set to what the vote picks at one call, held_eps must equal the vote's own prediction exactly."""
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(8, 1, 32, 64, generator=g, device=device)
    t = torch.full((8,), 900, device=device, dtype=torch.long)
    with torch.no_grad():
        vote = te.inferred_eps(A, B, a, b, True, None)(x, t)
        picked = te.LAYOUT_LOG[-1].tolist()
        held = held_eps(A, B, a, b, picked)(x, t)
    te.LAYOUT_LOG.clear()
    return float((vote - held).abs().max())


def render(arm, A, B, a, b, seeds, push, device):
    if arm == "paste":
        return p1.render("paste", A, B, a, b, seeds, None, device)
    if arm == "vote":
        eps = te.inferred_eps(A, B, a, b, True, p1.LOCK_T)
    elif arm == "shared":
        eps = te.two_eps(A, B, a, b, "blank")
    else:
        eps = held_eps(A, B, a, b, [coin_side(s) for s in seeds])
    langevin = None if arm == "coin_no_corr" else CORRECTOR
    x = sample(None, seeds, [], eps_fn=eps, push_fn=push, cap=p1.PUSH_CAP, device=device, eta=p1.ETA, langevin=langevin)[0]
    te.LAYOUT_LOG.clear()
    return to_uint8(x)


def log_wandb(name, config, summary):
    try:
        import wandb
        run = wandb.init(project="poe-repair-animals-compose", entity="prime_lab", name=name, group="loop-6-m11-vote",
                         config=config, settings=wandb.Settings(init_timeout=60))
        run.summary.update(summary)
        run_id = run.id
        run.finish()
        return run_id
    except Exception as error:  # the numbers are on disk either way
        print(f"W&B logging failed: {error}", flush=True)
        return None


def verdicts(total):
    def gap(x, y):
        d = total[x] - total[y]
        return {"lead": d, "reading": "real" if d >= REAL else "none" if abs(d) <= NONE else "inconclusive"}
    return {"vote_over_coin": gap("vote", "coin"), "coin_over_coin_no_corr": gap("coin", "coin_no_corr"),
            "coin_over_shared": gap("coin", "shared"), "vote_over_paste": gap("vote", "paste")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    for name in ("--a", "--b", "--pair-head", "--sep-head", "--judge", "--out"):
        ap.add_argument(name, type=Path, required=True)
    ap.add_argument("--pairs", default="1-9,3-6,4-7,2-8,0-5")
    ap.add_argument("--seed0", type=int, default=192)
    ap.add_argument("--seeds", type=int, default=64)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--wandb-name", default=None)
    args = ap.parse_args()
    device = "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    A, B = load_unet(args.a, device), load_unet(args.b, device)
    net = load_judge(args.judge, device)
    heads = {"pair": pc.load_head(args.pair_head, device), "sep": pc.load_head(args.sep_head, device)}
    pairs = te.parse_pairs(args.pairs)
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    if seeds[0] < 192:
        raise SystemExit("seeds below 192 were used for tuning or the final test")
    te.EVAL_SEEDS[:] = seeds
    arms = args.arms.split(",")

    control = max(held_matches_vote(A, B, a, b, device) for a, b in pairs)
    print(json.dumps({"held_side_matches_vote_max_abs": control}), flush=True)
    if control >= 1e-5:
        (args.out / "control_failed.json").write_text(json.dumps({"held_side_matches_vote_max_abs": control}))
        raise SystemExit("the held-side composition differs from the vote's own when given the vote's side")

    clean, tallies, canvases, seconds = {}, {}, {}, {}
    for arm in arms:
        start = time.time()
        clean[arm], tallies[arm] = {}, {}
        for a, b in pairs:
            push = None if arm == "paste" else te.make_push("both", a, b, heads, p1.PUSH_STRENGTH)
            x = render(arm, A, B, a, b, seeds, push, device)
            calls = [te.call_two(te.reads(net, c, device), (a, b)) for c in x]
            key = f"{a}-{b}"
            clean[arm][key] = sum(c == CLEAN for c in calls)
            tallies[arm][key] = {k: calls.count(k) for k in te.CALLS}
            canvases[f"{arm}/{key}"] = x
            if len(x) == 64:
                (args.out / "sheets" / arm).mkdir(parents=True, exist_ok=True)
                te.sheet(x, calls, f"{arm}, asked {a} and {b}: {clean[arm][key]} of 64 clean",
                         args.out / "sheets" / arm / f"digits-{a}-and-{b}.png")
        seconds[arm] = round(time.time() - start, 1)
        print(json.dumps({"arm": arm, "clean": sum(clean[arm].values()), "of": len(pairs) * len(seeds),
                          "seconds": seconds[arm]}), flush=True)
    np.savez_compressed(args.out / "canvases.npz", **canvases)
    total = {arm: sum(clean[arm].values()) for arm in arms}
    result = {"pairs": args.pairs, "seeds": [seeds[0], seeds[-1]], "corrector": CORRECTOR, "push_strength": p1.PUSH_STRENGTH,
              "push_cap": p1.PUSH_CAP, "lock_t": p1.LOCK_T, "eta": p1.ETA, "bars": {"real": REAL, "none": NONE},
              "held_side_matches_vote_max_abs": control, "clean": total, "clean_per_pair": clean, "calls": tallies,
              "seconds": seconds, "comparisons": verdicts(total) if set(ARMS) <= set(arms) else None}
    if args.wandb_name:
        result["wandb_run"] = log_wandb(args.wandb_name, {k: result[k] for k in ("pairs", "seeds", "corrector", "bars")},
                                        {f"clean/{k}": v for k, v in total.items()})
    (args.out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"clean": total, "comparisons": result["comparisons"]}), flush=True)


if __name__ == "__main__":
    main()
