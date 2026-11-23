"""Check that a zero push leaves the canvas sampler unchanged (step V1).

For each asked pair and seed batch this runs the plain sampler twice and the sampler with a zero push once, in one process on
one card and dtype. The plain repeat measures the card's own repeatability, and the zero push is compared with the first plain
run. Pass: the largest absolute pixel difference is under 1e-5.

Usage:
    python zero_push.py --model <unet.pt> --out <result.json> [--pairs all|5-6,7-9] [--seeds 8] [--wandb-name <name>]
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mnist"))
from canvas_unet import CanvasUNet, sample  # noqa: E402

BAR = 1e-5
ALL_PAIRS = list(itertools.combinations(range(5), 2)) + list(itertools.combinations(range(5, 10), 2))


def zero(x, t, eps):
    return torch.zeros_like(eps)


def log_wandb(name, config, summary):
    try:
        import wandb
        run = wandb.init(project="poe-repair-animals-compose", entity="prime_lab", name=name, group="loop-8-v1",
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
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs", default="all")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--wandb-name", default=None)
    args = ap.parse_args()

    pairs = ALL_PAIRS if args.pairs == "all" else [tuple(int(d) for d in p.split("-")) for p in args.pairs.split(",")]
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    ck = torch.load(args.model, map_location="cuda", weights_only=False)
    model = CanvasUNet(ck["args"].get("ch", 64)).cuda().eval()
    model.load_state_dict(ck["ema"])
    seeds = list(range(args.seeds))

    rows, start = [], time.time()
    with torch.no_grad():
        for a, b in pairs:
            plain, _ = sample(model, seeds, [a, b], g=1.0, steps=50, push_fn=None)
            repeat, _ = sample(model, seeds, [a, b], g=1.0, steps=50, push_fn=None)
            pushed, _ = sample(model, seeds, [a, b], g=1.0, steps=50, push_fn=zero)
            rows.append({"pair": f"{a}-{b}", "plain_vs_plain": float((plain - repeat).abs().max()),
                         "plain_vs_zero_push": float((plain - pushed).abs().max())})
            print(json.dumps(rows[-1]), flush=True)

    worst_repeat = max(r["plain_vs_plain"] for r in rows)
    worst_zero = max(r["plain_vs_zero_push"] for r in rows)
    verdict = "inconclusive" if worst_repeat >= BAR else ("pass" if worst_zero < BAR else "fail")
    result = {"verdict": verdict, "bar": BAR, "largest_plain_vs_plain": worst_repeat, "largest_plain_vs_zero_push": worst_zero,
              "pairs": len(pairs), "seeds": args.seeds, "canvases_per_run": len(pairs) * args.seeds, "steps": 50,
              "device": torch.cuda.get_device_name(), "dtype": str(next(model.parameters()).dtype),
              "seconds": round(time.time() - start, 1), "model": str(args.model), "rows": rows}
    if args.wandb_name:
        result["wandb_run"] = log_wandb(args.wandb_name, {k: result[k] for k in ("pairs", "seeds", "steps", "device", "dtype", "model")},
                                        {k: result[k] for k in ("verdict", "largest_plain_vs_plain", "largest_plain_vs_zero_push")})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)
    sys.exit(0 if verdict == "pass" else 1)


if __name__ == "__main__":
    main()
