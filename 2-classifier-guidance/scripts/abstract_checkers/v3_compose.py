"""Step V3: does the count checker's push make the plain product draw two separate, right digits on unseen pairs?

Stages, in order:
    smoke   pair 0 and 1 at 4 seeds, every row, and the zero-push check
    tune    the 10 seen pairs among digits 0 to 4 at seeds 0 to 31, push lengths 0.1, 0.2 and 0.4; freezes b
    final   the 10 unseen pairs among digits 5 to 9 at seeds 1000 to 1031, once

Usage:
    python v3_compose.py <stage> --model <unet.pt> --head <count-head-map.pt> --judge <judge.pt> --out <folder> [--wandb-name <name>]
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "mnist"), str(HERE.parent / "method_search")]
import beat_baseline_bars as bars  # noqa: E402
import two_experts as te  # noqa: E402
from canvas_unet import ABAR, ddim_timesteps, sample, to_uint8  # noqa: E402
from checker import load_unet  # noqa: E402
from judge import load as load_judge  # noqa: E402
from v2_learn import make_head, maps  # noqa: E402

# The bars and fixed settings, written before any run (abstract-checkers/06-does-it-compose/design.md).
ZERO_PUSH_MAX_ABS = bars.ZERO_PUSH_MAX_ABS
B_CHOICES = (0.1, 0.2, 0.4)
WINDOW = {int(t) for t in ddim_timesteps(50) if 480 <= int(t) <= 712}
CLEAN = "two clear, right, apart"
SEPARATE = {CLEAN, "two, not both clear and right"}
SEEN = list(itertools.combinations(range(5), 2))
UNSEEN = list(itertools.combinations(range(5, 10), 2))
SELECT_T, SELECT_SEED = 480, 777


def load_head(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    head = make_head(ck["cfg"], dev)
    head.load_state_dict(ck["head"])
    for p in head.parameters():
        p.requires_grad_(False)
    return head.eval(), ck


def make_push(kind, b, head, model, a, bb, seed):
    gen = torch.Generator().manual_seed(seed)

    def push(x, t, eps):
        if int(t[0]) not in WINDOW:
            return torch.zeros_like(eps)
        if kind == "random":
            d = torch.randn(x.shape, generator=gen).to(x.device)
        else:
            with torch.enable_grad():
                xx = x.detach().requires_grad_(True)
                pair = torch.tensor([[a, bb]], device=x.device).expand(x.shape[0], 2)
                logit = head(maps(model, xx, t, pair, "base"), t, pair)[:, 0]
                d, = torch.autograd.grad(logit.sum(), xx)
        size = eps.flatten(1).norm(dim=1)[:, None, None, None]
        return -size * b * d / d.flatten(1).norm(dim=1).clamp_min(1e-12)[:, None, None, None]
    return push


@torch.no_grad()
def select_logit(head, model, x, a, bb):
    """The checker's logit on finished canvases noised to timestep 480 with one fixed noise draw."""
    t = torch.full((x.shape[0],), SELECT_T, device=x.device, dtype=torch.long)
    ab = ABAR.to(x.device)[t][:, None, None, None]
    noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(SELECT_SEED)).to(x.device)
    pair = torch.tensor([[a, bb]], device=x.device).expand(x.shape[0], 2)
    return head(maps(model, ab.sqrt() * x.clamp(-1, 1) + (1 - ab).sqrt() * noise, t, pair, "base"), t, pair)[:, 0]


def render(row, b, model, head, a, bb, seeds):
    if row == "plain":
        x, _ = sample(model, seeds, [a, bb], g=1.0, steps=50, push_fn=None, cap=1.0)
    elif row in ("push", "zero"):
        x, _ = sample(model, seeds, [a, bb], g=1.0, steps=50, cap=1.0,
                      push_fn=make_push("checker", 0.0 if row == "zero" else b, head, model, a, bb, 0))
    elif row == "random":
        x, _ = sample(model, seeds, [a, bb], g=1.0, steps=50, cap=1.0,
                      push_fn=make_push("random", b, head, model, a, bb, 4242 + 10 * a + bb))
    elif row == "best4":
        tries = [sample(model, [s + 100_000 * k for s in seeds], [a, bb], g=1.0, steps=50, push_fn=None, cap=1.0)[0]
                 for k in range(4)]
        scores = torch.stack([select_logit(head, model, xk, a, bb) for xk in tries])
        pick = scores.argmax(0)
        x = torch.stack([tries[int(k)][i] for i, k in enumerate(pick)])
    else:
        raise ValueError(row)
    return x


def score(net, canvases, pair, dev):
    return [te.call_two(te.reads(net, c, dev), pair) for c in canvases]


def grid(rows, title, path, per_pair=8):
    """One sheet per row: each line is one pair, its first canvases framed green when clean."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = list(rows)
    per_pair = min(per_pair, min(len(rows[k][0]) for k in keys))
    fig, axs = plt.subplots(len(keys), per_pair, figsize=(per_pair * 1.6, len(keys) * 0.95 + 0.6), squeeze=False)
    for r, key in enumerate(keys):
        canv, calls = rows[key]
        for c in range(per_pair):
            ax = axs[r, c]
            ax.imshow(canv[c], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#1d7a3e" if calls[c] == CLEAN else "#c0262d"); sp.set_linewidth(2)
        axs[r, 0].set_ylabel(f"{key}\n{sum(k == CLEAN for k in calls)}/{len(calls)}", rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def zero_check(model, head, a, bb, seeds):
    plain = render("plain", 0.0, model, head, a, bb, seeds)
    zero = render("zero", 0.0, model, head, a, bb, seeds)
    return float((plain - zero).abs().max())


def run_rows(rows, pairs, seeds, b, model, head, net, dev, out, tag):
    counts, separate, per_pair, seconds = {}, {}, {}, {}
    for row in rows:
        start, sheet = time.time(), {}
        for a, bb in pairs:
            x = render(row, b, model, head, a, bb, seeds)
            canv = to_uint8(x)
            calls = score(net, canv, (a, bb), dev)
            key = f"{a}-{bb}"
            per_pair.setdefault(row, {})[key] = {"clean": sum(c == CLEAN for c in calls),
                                                 "separate": sum(c in SEPARATE for c in calls),
                                                 "calls": {k: calls.count(k) for k in te.CALLS}}
            sheet[key] = (canv, calls)
        counts[row] = sum(v["clean"] for v in per_pair[row].values())
        separate[row] = sum(v["separate"] for v in per_pair[row].values())
        seconds[row] = round(time.time() - start, 1)
        grid(sheet, f"{tag}, {row}: {counts[row]} clean pairs of {len(pairs) * len(seeds)}", out / "sheets" / f"{tag}-{row}.png")
        print(json.dumps({"stage": tag, "row": row, "b": b, "clean": counts[row], "separate": separate[row],
                          "of": len(pairs) * len(seeds), "seconds": seconds[row]}), flush=True)
    return counts, separate, per_pair, seconds


def log_wandb(name, config, summary):
    try:
        import wandb
        run = wandb.init(project="poe-repair-animals-compose", entity="prime_lab", name=name, group="loop-8-v3",
                         config=config, settings=wandb.Settings(init_timeout=60))
        run.summary.update(summary)
        run_id = run.id
        run.finish()
        return run_id
    except Exception as error:  # the numbers are on disk either way
        print(f"W&B logging failed: {error}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["smoke", "tune", "final"])
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--judge", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--wandb-name", default=None)
    args = ap.parse_args()
    dev = "cuda"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.out.mkdir(parents=True, exist_ok=True)
    model = load_unet(args.model, dev)
    head, ck = load_head(args.head, dev)
    net = load_judge(args.judge, dev)
    info = {"stage": args.stage, "model": str(args.model), "head": str(args.head), "head_window": ck.get("window"),
            "push_steps": sorted(WINDOW, reverse=True), "device": torch.cuda.get_device_name()}
    print(json.dumps(info), flush=True)

    if args.stage == "smoke":
        seeds = list(range(4))
        zero = zero_check(model, head, 0, 1, seeds)
        counts, separate, per_pair, seconds = run_rows(("plain", "push", "random", "best4"), [(0, 1)], seeds, 0.4,
                                                       model, head, net, dev, args.out, "smoke")
        result = {**info, "zero_push_max_abs": zero, "clean": counts, "separate": separate, "seconds": seconds}
        (args.out / "smoke.json").write_text(json.dumps(result, indent=1))
        print(json.dumps(result), flush=True)
        return 0 if zero < ZERO_PUSH_MAX_ABS else 1

    if args.stage == "tune":
        seeds = list(range(32))
        result = {**info, "seeds": [0, 31], "pairs": "seen", "rows": {}}
        base, base_sep, base_pp, _ = run_rows(("plain",), SEEN, seeds, 0.0, model, head, net, dev, args.out, "tune")
        result["rows"]["plain"] = {"clean": base["plain"], "separate": base_sep["plain"], "per_pair": base_pp["plain"]}
        for b in B_CHOICES:
            c, s, pp, _ = run_rows(("push",), SEEN, seeds, b, model, head, net, dev, args.out, f"tune-b{b}")
            result["rows"][f"push_b{b}"] = {"b": b, "clean": c["push"], "separate": s["push"], "per_pair": pp["push"]}
        best = max(B_CHOICES, key=lambda b: (result["rows"][f"push_b{b}"]["clean"], -b))
        result["chosen_b"] = best
        (args.out / "tune.json").write_text(json.dumps(result, indent=1))
        print(json.dumps({"chosen_b": best, **{k: v["clean"] for k, v in result["rows"].items()}}), flush=True)
        return 0

    out = args.out / "final.json"
    if out.exists():
        raise SystemExit(f"{out} exists: the unseen pairs are read once")
    b = json.loads((args.out / "tune.json").read_text())["chosen_b"]
    seeds = list(range(1000, 1032))
    zero = zero_check(model, head, 5, 6, seeds[:8])
    print(json.dumps({"zero_push_max_abs": zero, "bar": ZERO_PUSH_MAX_ABS}), flush=True)
    counts, separate, per_pair, seconds = run_rows(("plain", "push", "random", "best4"), UNSEEN, seeds, b,
                                                   model, head, net, dev, args.out, "final")
    keys = [f"{a}-{bb}" for a, bb in UNSEEN]
    diff = lambda r, s: [per_pair[r][k]["clean"] - per_pair[s][k]["clean"] for k in keys]
    se_push = bars.se_over_groups(diff("push", "plain"))
    se_random = bars.se_over_groups(diff("random", "plain"))
    se_best = bars.se_over_groups(diff("push", "best4"))
    lead, random_lead = counts["push"] - counts["plain"], counts["random"] - counts["plain"]
    if zero >= ZERO_PUSH_MAX_ABS:
        verdict = "inconclusive: the zero push differs from the plain product"
    elif bars.clears(random_lead, se_random):
        verdict = "inconclusive: the random push also beats the plain product"
    elif bars.clears(lead, se_push):
        verdict = "support"
    else:
        verdict = "null"
    result = {**info, "b": b, "seeds": [1000, 1031], "pairs": "unseen", "zero_push_max_abs": zero,
              "clean": counts, "separate": separate, "of": len(keys) * len(seeds), "seconds": seconds,
              "lead_push_over_plain": lead, "se_push_over_plain": se_push, "lead_random_over_plain": random_lead,
              "se_random_over_plain": se_random, "lead_push_over_best4": counts["push"] - counts["best4"],
              "se_push_over_best4": se_best, "verdict": verdict, "per_pair": per_pair}
    if args.wandb_name:
        result["wandb_run"] = log_wandb(args.wandb_name, {"b": b, "model": str(args.model), "head": str(args.head)},
                                        {"verdict": verdict, **{f"clean_{k}": v for k, v in counts.items()},
                                         **{f"separate_{k}": v for k, v in separate.items()},
                                         "se_push_over_plain": se_push, "zero_push_max_abs": zero})
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "per_pair"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
