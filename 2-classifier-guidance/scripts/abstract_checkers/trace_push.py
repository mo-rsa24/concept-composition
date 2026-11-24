"""How the count checker's push turns one fused glyph into two digits: plain and pushed sampling, recorded step by step.

For the 10 unseen pairs at V3's seeds 1000 to 1031, both rows run with V3's settings (b = 0.4, sampler steps 15 to 25).
At every sampler step it records, for every canvas, where the two maps sit (their centres of mass across the canvas and
their overlap) and the checker's logit. For a few example canvases it also keeps the full pictures at every step: the
noisy canvas, the model's current guess of the clean canvas, the two maps, the gradient g (on the plain row too, as the
push that would have been applied) and the push itself. The pushed row's clean counts are checked against V3's.

Usage:
    python trace_push.py --model <unet.pt> --head <count-head-map.pt> --judge <judge.pt> --v3 <final.json> --out <folder>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "mnist"), str(HERE.parent / "method_search")]
import two_experts as te  # noqa: E402
import v2_learn as v2  # noqa: E402
from canvas_unet import ABAR, sample, to_uint8, W  # noqa: E402
from checker import load_unet  # noqa: E402
from judge import load as load_judge  # noqa: E402
from v3_compose import CLEAN, UNSEEN, WINDOW, load_head  # noqa: E402

B = 0.4
SEEDS = list(range(1000, 1032))
EXAMPLE_PAIRS = ((5, 6), (6, 8), (7, 9))
XS = torch.arange(W, dtype=torch.float32)


def run(row, model, head, a, bb, dev):
    pair = torch.tensor([[a, bb]], device=dev).expand(len(SEEDS), 2)
    rec = {k: [] for k in ("t", "x", "x0", "maps", "g", "push", "logit", "com_a", "com_b", "overlap")}

    def recorder(x, t, eps):
        step = int(t[0])
        ab = ABAR.to(x.device)[t][:, None, None, None]
        with torch.enable_grad():
            xx = x.detach().requires_grad_(True)
            m = v2.maps(model, xx, t, pair, "base")
            logit = head(m, t, pair)[:, 0]
            g, = torch.autograd.grad(logit.sum(), xx)
        size = eps.flatten(1).norm(dim=1)[:, None, None, None]
        unit = g / g.flatten(1).norm(dim=1).clamp_min(1e-12)[:, None, None, None]
        push = -size * B * unit if (row == "push" and step in WINDOW) else torch.zeros_like(eps)
        md = m.detach()
        col = md.sum(2)                                    # (n, 2, W): each map's mass per column
        com = (col * XS.to(x.device)).sum(2) / col.sum(2).clamp_min(1e-12)
        rec["t"].append(step)
        rec["com_a"].append(com[:, 0].cpu()); rec["com_b"].append(com[:, 1].cpu())
        rec["overlap"].append((torch.minimum(md[:, 0], md[:, 1]).flatten(1).sum(1) / v2.PIXELS).cpu())
        rec["logit"].append(logit.detach().cpu())
        rec["x"].append(x.detach().cpu().half()); rec["x0"].append(((x - (1 - ab).sqrt() * eps) / ab.sqrt()).detach().cpu().half())
        rec["maps"].append(md.cpu().half()); rec["g"].append(g.detach().cpu().half()); rec["push"].append(push.detach().cpu().half())
        return push

    x, _ = sample(model, SEEDS, [a, bb], g=1.0, steps=50, push_fn=recorder, cap=1.0)
    out = {"t": np.array(rec["t"])}
    for k in ("com_a", "com_b", "overlap", "logit"):
        out[k] = torch.stack(rec[k], 1).numpy()            # (seeds, steps)
    for k in ("x", "x0", "maps", "g", "push"):
        out[k] = torch.stack(rec[k], 1).numpy()            # (seeds, steps, channels, H, W)
    out["final"] = to_uint8(x)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    for flag in ("model", "head", "judge", "v3", "out"):
        ap.add_argument(f"--{flag}", type=Path, required=True)
    args = ap.parse_args()
    dev = "cuda"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.out.mkdir(parents=True, exist_ok=True)
    model = load_unet(args.model, dev)
    head, _ = load_head(args.head, dev)
    net = load_judge(args.judge, dev)
    v3 = json.loads(args.v3.read_text())["per_pair"]
    summary, stats, examples = {"pairs": {}}, {}, {}
    for a, bb in UNSEEN:
        key = f"{a}-{bb}"
        rows = {row: run(row, model, head, a, bb, dev) for row in ("plain", "push")}
        calls = {row: [te.call_two(te.reads(net, c, dev), (a, bb)) for c in rows[row]["final"]] for row in rows}
        clean = {row: sum(c == CLEAN for c in calls[row]) for row in rows}
        summary["pairs"][key] = {"clean": clean, "v3_clean": {r: v3[r][key]["clean"] for r in ("plain", "push")},
                                 "matches_v3": all(clean[r] == v3[r][key]["clean"] for r in ("plain", "push"))}
        for row in rows:
            for k in ("com_a", "com_b", "overlap", "logit"):
                stats[f"{key}__{row}__{k}"] = rows[row][k]
            stats[f"{key}__{row}__clean"] = np.array([c == CLEAN for c in calls[row]])
        if (a, bb) in EXAMPLE_PAIRS:
            won = [i for i in range(len(SEEDS)) if calls["push"][i] == CLEAN and calls["plain"][i] == "one blob"][:2]
            lost = [i for i in range(len(SEEDS)) if calls["push"][i] != CLEAN][:1]
            for i in won + lost:
                tag = f"{key}_seed{SEEDS[i]}"
                for row in rows:
                    for k in ("x", "x0", "maps", "g", "push", "logit", "com_a", "com_b", "overlap"):
                        examples[f"{tag}__{row}__{k}"] = rows[row][k][i]
                    examples[f"{tag}__{row}__call"] = np.array(calls[row][i])
                    examples[f"{tag}__{row}__final"] = rows[row]["final"][i]
            summary["pairs"][key]["examples"] = {"won": [SEEDS[i] for i in won], "lost": [SEEDS[i] for i in lost]}
        print(json.dumps({"pair": key, **summary["pairs"][key]}), flush=True)
    summary["t"] = rows["plain"]["t"].tolist()
    summary["push_steps"] = sorted(WINDOW, reverse=True)
    summary["all_match_v3"] = all(p["matches_v3"] for p in summary["pairs"].values())
    np.savez_compressed(args.out / "trace-stats.npz", t=rows["plain"]["t"], **stats)
    np.savez_compressed(args.out / "trace-examples.npz", t=rows["plain"]["t"], **examples)
    (args.out / "trace.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"all_match_v3": summary["all_match_v3"]}), flush=True)


if __name__ == "__main__":
    main()
