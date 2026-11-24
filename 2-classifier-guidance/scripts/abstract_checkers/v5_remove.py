"""Step V5, digit rehearsal: remove one of three checkers (count, right digits, order) at a time.

Stages, in order:
    smoke      every stage on one pair at 4 seeds, with 50-update checkers, into <out>/smoke
    train      the right-digits and order checkers on digits 0 to 4 at SDXL's noise, saved with their accuracies
    calibrate  the push length b from the checkers' gradient cosines on the seen pairs, and the zero-push check
    final      eleven rows on the unseen pairs, once

Usage:
    python v5_remove.py <stage> --model <unet.pt> --data <v1 data> --count-head <pt> --judge <pt> --out <folder>
"""
import argparse
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
import v2_learn as v2  # noqa: E402
from canvas_unet import sample, to_uint8  # noqa: E402
from checker import load_unet  # noqa: E402
from judge import load as load_judge  # noqa: E402
from v3_compose import CLEAN, SEEN, SEPARATE, UNSEEN, WINDOW, grid, load_head  # noqa: E402

# The bars, fixed before any run (abstract-checkers/08-remove-one-checker/design.md).
OWN_DROP = 6.25          # percentage points
OTHER_MOVE = 4.2         # percentage points
CELL_MIN = 40            # canvases
VOID_TOTAL, VOID_SHARE = 0.6, 0.05
ZERO_PUSH_MAX_ABS = bars.ZERO_PUSH_MAX_ABS
NAMES = ("count", "identity", "order")
COLUMNS = ("count", "right", "order", "full")
OWN_COLUMN = {"count": "count", "identity": "right", "order": "order"}
ROWS = {"plain": ((), ()), "all": (NAMES, ())}
for n in NAMES:
    others = tuple(m for m in NAMES if m != n)
    ROWS[f"minus_{n}"] = (others, ())
    ROWS[f"minus_{n}_random"] = (others, (n,))
    ROWS[f"only_{n}"] = ((n,), ())

v2.STEPS16 = v2.SDXL_MATCHED   # every checker here trains and reads at the noise levels matching SDXL's


def order_split(split):
    """apart canvases twice: read in their true left-right order (1) and flipped (0)."""
    s = v2.take(split, split["kind"] == v2.KINDS.index("apart"))
    true = s["holds"][:, :2].astype(np.int8)
    out = {k: np.concatenate([v, v]) for k, v in s.items()}
    out["asked"] = np.concatenate([true, true[:, ::-1]])
    out["target"] = np.concatenate([np.ones(len(true), np.uint8), np.zeros(len(true), np.uint8)])
    return out


MAKERS = {"identity": lambda s: v2.relabel(s, "identity"), "order": order_split}


def unit(d):
    return d / d.flatten(1).norm(dim=1).clamp_min(1e-12)[:, None, None, None]


def checker_grad(head, model, x, t, pair):
    with torch.enable_grad():
        xx = x.detach().requires_grad_(True)
        logit = head(v2.maps(model, xx, t, pair, "base"), t, pair)[:, 0]
        g, = torch.autograd.grad(logit.sum(), xx)
    return g


def pairs_for(a, bb, seeds, dev):
    """count and identity read (a, b) as asked; order reads (left, right), lower digit left on even seeds."""
    n = len(seeds)
    plain = torch.tensor([[a, bb]], device=dev).expand(n, 2)
    order = torch.tensor([[a, bb] if s % 2 == 0 else [bb, a] for s in seeds], device=dev)
    return {"count": plain, "identity": plain, "order": order}


def make_push(active, randoms, b, heads, model, pairs, log, seed):
    gen = torch.Generator().manual_seed(seed)

    def push(x, t, eps):
        if int(t[0]) not in WINDOW:
            return torch.zeros_like(eps)
        size = eps.flatten(1).norm(dim=1)[:, None, None, None]
        total = torch.zeros_like(eps)
        for name in active:
            total = total + unit(checker_grad(heads[name], model, x, t, pairs[name]))
        for _ in randoms:
            total = total + unit(torch.randn(x.shape, generator=gen).to(x.device))
        p = -size * b * total
        log.append((p.flatten(1).norm(dim=1) / size.flatten()).cpu())
        return p
    return push


def probe(heads, model, pairs, cos_log):
    """Records the three checkers' pairwise gradient cosines on the pushed steps and pushes nothing."""
    def push(x, t, eps):
        if int(t[0]) in WINDOW:
            g = {n: unit(checker_grad(heads[n], model, x, t, pairs[n])).flatten(1) for n in NAMES}
            for i, j in (("count", "identity"), ("count", "order"), ("identity", "order")):
                cos_log.append((g[i] * g[j]).sum(1).cpu())
        return torch.zeros_like(eps)
    return push


def render(row, b, heads, model, a, bb, seeds, dev, log):
    pairs = pairs_for(a, bb, seeds, dev)
    active, randoms = ROWS[row] if row != "zero" else (NAMES, ())
    if row == "plain":
        x, _ = sample(model, seeds, [a, bb], g=1.0, steps=50, push_fn=None, cap=1.0)
    else:
        x, _ = sample(model, seeds, [a, bb], g=1.0, steps=50, cap=1.0,
                      push_fn=make_push(active, randoms, 0.0 if row == "zero" else b, heads, model, pairs, log,
                                        5151 + 10 * a + bb))
    return x


def score(net, canv, a, bb, seeds, dev):
    out = []
    for c, s in zip(canv, seeds):
        rd = te.reads(net, c, dev)
        call = te.call_two(rd, (a, bb))
        left = a if s % 2 == 0 else bb
        ok_order = None
        if call == CLEAN:
            ok_order = int(np.argmax(min(rd, key=lambda r: r[0])[1])) == left
        out.append((call, call in SEPARATE, call == CLEAN, ok_order))
    return out


def tallies(scored):
    n = len(scored)
    count = sum(s[1] for s in scored)
    right = sum(s[2] for s in scored)
    order = sum(1 for s in scored if s[3])
    return {"n": n, "count": count, "right": right, "order": order}


def column(t, col):
    """(numerator, denominator) of a column: count of all, right of count, order of right, fully right of all."""
    return {"count": (t["count"], t["n"]), "right": (t["right"], t["count"]),
            "order": (t["order"], t["right"]), "full": (t["order"], t["n"])}[col]


def share(per_pair, col, idx=None):
    keys = list(per_pair) if idx is None else [list(per_pair)[i] for i in idx]
    num = sum(column(per_pair[k], col)[0] for k in keys)
    den = sum(column(per_pair[k], col)[1] for k in keys)
    return 100.0 * num / den if den else float("nan")


def diff_se(pp_r, pp_s, col):
    rng = np.random.default_rng(bars.BOOTSTRAP_SEED)
    k = len(pp_r)
    picks = rng.integers(k, size=(bars.BOOTSTRAP_DRAWS, k))
    d = np.array([share(pp_r, col, p) - share(pp_s, col, p) for p in picks])
    d = d[np.isfinite(d)]
    return float(d.std(ddof=1)) if len(d) > 1 else float("nan")


def verdict(per_row, void, zero):
    if zero >= ZERO_PUSH_MAX_ABS or any(void.values()):
        return "inconclusive: the zero push or the total-length rule failed", {}
    detail, passes = {}, {}
    for n in NAMES:
        own = OWN_COLUMN[n]
        cells = [column(sum_pp(per_row[r]), c)[1] for r in ("all", f"minus_{n}", f"minus_{n}_random") for c in COLUMNS]
        if min(cells) < CELL_MIN:
            detail[n] = {"inconclusive": f"a cell holds {min(cells)} canvases, under {CELL_MIN}"}
            passes[n] = None
            continue
        drop = share(per_row["all"], own) - share(per_row[f"minus_{n}"], own)
        se = diff_se(per_row["all"], per_row[f"minus_{n}"], own)
        rdrop = share(per_row["all"], own) - share(per_row[f"minus_{n}_random"], own)
        rse = diff_se(per_row["all"], per_row[f"minus_{n}_random"], own)
        moved = {}
        for c in COLUMNS[:3]:
            if c == own:
                continue
            m = share(per_row["all"], c) - share(per_row[f"minus_{n}"], c)
            mse = diff_se(per_row["all"], per_row[f"minus_{n}"], c)
            moved[c] = {"move": m, "se": mse, "too_much": abs(m) > OTHER_MOVE and abs(m) > bars.Z * mse}
        ok = (drop >= OWN_DROP and drop > bars.Z * se and rdrop >= OWN_DROP and rdrop > bars.Z * rse
              and not any(v["too_much"] for v in moved.values()))
        detail[n] = {"own_column": own, "drop": drop, "se": se, "random_drop": rdrop, "random_se": rse, "others": moved,
                     "passes": ok}
        passes[n] = ok
    if any(v is None for v in passes.values()):
        return "inconclusive: a cell the verdict needs holds fewer than 40 canvases", detail
    full = share(per_row["all"], "full") - share(per_row["plain"], "full")
    full_se = diff_se(per_row["all"], per_row["plain"], "full")
    detail["all_over_plain_full"] = {"lead": full, "se": full_se, "clears": full > bars.Z * full_se}
    n_pass = sum(passes.values())
    if n_pass == len(NAMES) and detail["all_over_plain_full"]["clears"]:
        return "support", detail
    if n_pass >= 1:
        return "partial", detail
    return "null", detail


def sum_pp(per_pair):
    t = {"n": 0, "count": 0, "right": 0, "order": 0}
    for v in per_pair.values():
        for k in t:
            t[k] += v[k]
    return t


def run_rows(rows, pairs, seeds, b, heads, model, net, dev, out, tag):
    per_row, void, lengths = {}, {}, {}
    for row in rows:
        start, log, sheet = time.time(), [], {}
        per_row[row] = {}
        for a, bb in pairs:
            canv = to_uint8(render(row, b, heads, model, a, bb, seeds, dev, log))
            sc = score(net, canv, a, bb, seeds, dev)
            per_row[row][f"{a}-{bb}"] = tallies(sc)
            sheet[f"{a}-{bb}"] = (canv, [s[0] for s in sc])
        total = torch.cat(log) if log else torch.zeros(1)
        lengths[row] = {"mean": float(total.mean()), "share_over_0.6": float((total > VOID_TOTAL).float().mean())}
        void[row] = lengths[row]["share_over_0.6"] > VOID_SHARE
        t = sum_pp(per_row[row])
        grid(sheet, f"{tag}, {row}: {t['order']} fully right of {t['n']}", out / "sheets" / f"{tag}-{row}.png")
        print(json.dumps({"stage": tag, "row": row, **t, "length": lengths[row], "seconds": round(time.time() - start, 1)}),
              flush=True)
    return per_row, void, lengths


def heads_from(folder, count_head, dev):
    heads = {"count": load_head(count_head, dev)[0]}
    for n in ("identity", "order"):
        heads[n] = load_head(folder / f"{n}-head-map.pt", dev)[0]
    return heads


def stage_train(args, model, data, dev, out, updates):
    res = {}
    for name, make in MAKERS.items():
        sp = {k: make(data[k]) for k in ("train", "seen_test", "unseen_test")}
        fit = v2.overfit_one(v2.BASE, model, sp["train"], dev)
        head, info = v2.train(v2.BASE, model, sp["train"], dev, updates=updates)
        torch.save({"head": head.state_dict(), "cfg": v2.BASE, "task": name, "window": "sdxl",
                    "timesteps": v2.STEPS16.tolist()}, out / f"{name}-head-map.pt")
        s, u = v2.evaluate(head, v2.BASE, model, sp["seen_test"], dev), v2.evaluate(head, v2.BASE, model, sp["unseen_test"], dev)
        res[name] = {"overfit": fit, **info, "train_canvases": len(sp["train"]["x"]), "seen_ba": s["balanced_accuracy"],
                     "unseen_ba": u["balanced_accuracy"], "unseen_per_step": u["per_step"], "seen_per_step": s["per_step"]}
        print(json.dumps({"checker": name, "overfit": fit["right_of_16"], "seen_ba": s["balanced_accuracy"],
                          "unseen_ba": u["balanced_accuracy"]}), flush=True)
    (out / "train.json").write_text(json.dumps(res, indent=1))


def stage_calibrate(args, model, dev, out, heads, pairs, seeds):
    cos = []
    for a, bb in pairs:
        sample(model, seeds, [a, bb], g=1.0, steps=50, cap=1.0, push_fn=probe(heads, model, pairs_for(a, bb, seeds, dev), cos))
    allc = torch.cat(cos)
    rho = max(0.0, float(allc.mean()))
    b = 0.4 / (3 * (1 + 2 * rho)) ** 0.5
    plain = render("plain", 0.0, heads, model, pairs[0][0], pairs[0][1], seeds[:8], dev, [])
    zero = render("zero", 0.0, heads, model, pairs[0][0], pairs[0][1], seeds[:8], dev, [])
    res = {"mean_cosine": float(allc.mean()), "rho_clipped": rho, "b": b, "cosines": len(allc),
           "zero_push_max_abs": float((plain - zero).abs().max())}
    (out / "calibrate.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)
    return res


def log_wandb(name, config, summary):
    try:
        import wandb
        run = wandb.init(project="poe-repair-animals-compose", entity="prime_lab", name=name, group="loop-8-v5",
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
    ap.add_argument("stage", choices=["smoke", "train", "calibrate", "final"])
    for flag in ("model", "data", "count_head", "judge", "out"):
        ap.add_argument(f"--{flag.replace('_', '-')}", type=Path, required=True)
    ap.add_argument("--wandb-name", default=None)
    args = ap.parse_args()
    dev = "cuda"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = load_unet(args.model, dev)
    net = load_judge(args.judge, dev)
    data, data_hash = v2.load_data(args.data)
    print(json.dumps({"stage": args.stage, "data_hash": data_hash, "push_steps": sorted(WINDOW, reverse=True),
                      "device": torch.cuda.get_device_name()}), flush=True)

    if args.stage == "smoke":
        out = args.out / "smoke"
        out.mkdir(parents=True, exist_ok=True)
        stage_train(args, model, data, dev, out, updates=50)
        heads = heads_from(out, args.count_head, dev)
        cal = stage_calibrate(args, model, dev, out, heads, [(0, 1)], list(range(4)))
        per_row, void, lengths = run_rows(list(ROWS), [(0, 1)], list(range(4)), cal["b"], heads, model, net, dev, out, "smoke")
        (out / "smoke.json").write_text(json.dumps({"calibrate": cal, "rows": per_row, "void": void, "lengths": lengths}, indent=1))
        return 0 if cal["zero_push_max_abs"] < ZERO_PUSH_MAX_ABS else 1

    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "train":
        stage_train(args, model, data, dev, args.out, updates=v2.BASE["updates"])
        return 0
    heads = heads_from(args.out, args.count_head, dev)
    if args.stage == "calibrate":
        cal = stage_calibrate(args, model, dev, args.out, heads, SEEN, list(range(32)))
        return 0 if cal["zero_push_max_abs"] < ZERO_PUSH_MAX_ABS else 1

    out = args.out / "final.json"
    if out.exists():
        raise SystemExit(f"{out} exists: the unseen pairs are read once")
    cal = json.loads((args.out / "calibrate.json").read_text())
    per_row, void, lengths = run_rows(list(ROWS), UNSEEN, list(range(1000, 1032)), cal["b"], heads, model, net, dev,
                                      args.out, "final")
    v, detail = verdict(per_row, void, cal["zero_push_max_abs"])
    table = {r: {c: share(per_row[r], c) for c in COLUMNS} for r in ROWS}
    result = {"verdict": v, "b": cal["b"], "mean_cosine": cal["mean_cosine"], "zero_push_max_abs": cal["zero_push_max_abs"],
              "table_percent": table, "totals": {r: sum_pp(per_row[r]) for r in ROWS}, "void": void, "lengths": lengths,
              "detail": detail, "per_pair": per_row}
    if args.wandb_name:
        result["wandb_run"] = log_wandb(args.wandb_name, {"b": cal["b"]}, {"verdict": v, **{f"{r}_{c}": table[r][c]
                                                                           for r in ROWS for c in COLUMNS}})
    out.write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps({k: result[k] for k in ("verdict", "b", "table_percent")}, default=float), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
