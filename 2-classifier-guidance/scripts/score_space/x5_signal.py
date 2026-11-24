"""Loop 10, step X5: does a digit canvas's signature carry which digits are present?

The signature at one step is the ten maps d_t(c) = eps(x_t, t, c) - eps(x_t, t, null) for c = 0..9 on the same noisy
canvas, so the digits reach it only through the canvas (score-space/03-ideas.md, A1). A linear probe per digit reads
presence from the signature, and the same probe reads the noisy canvas, the raw noise, the per-label error readout and
the path length. Two windows: the brief's early window (timesteps 999 to 693) and the SDXL-matched window (712 to 480).
The design is artifacts/notes/progress/score-space/04-design.md.

Stages, in order:
    smoke    one batch through the signature at one step: shapes, finite values, seconds per forward pass
    control  at timestep 100 the signature and the noisy canvas must each reach presence accuracy 0.9
    overfit  the probe fitted and read on 16 canvases must call at least 15 of them fully right
    small    40 fitting and 20 reading canvases, both windows, every row; checks the code, carries no verdict
    full     4,000 fitting canvases from train and the 2,000 of seen_test, both windows, every row; writes the verdict

Usage:
    python x5_signal.py <stage> --model <unet.pt> --data <folder with dataset.npz> --out <folder> [--wandb-name <name>]
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
sys.path.insert(0, str(HERE.parent / "mnist"))
sys.path.insert(0, str(HERE.parent / "abstract_checkers"))
from canvas_unet import ABAR, NULL  # noqa: E402
from checker import load_unet, to_x  # noqa: E402
from v2_learn import KINDS, SDXL_MATCHED, STEPS16, load_data, take  # noqa: E402

# The bars, fixed before any run (score-space/04-design.md, run group 1).
MIN_PRESENCE = 0.75          # presence accuracy on the signature at some step of a window
MIN_OVER_NOISE = 0.10        # and at least this much above the raw-noise probe at that step
CONTROL_T = 100              # positive control timestep, sqrt(abar) = 0.946
MIN_CONTROL = 0.90
OVERFIT_N, OVERFIT_RIGHT = 16, 15
FIT_N, READ_N = 4000, 2000
SMALL_FIT, SMALL_READ = 40, 20
DIGITS = list(range(5))      # train and seen_test hold digits 0 to 4 only
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1)
FOLDS = 5
FIT_SEED, READ_SEED = 1_010_000, 1_020_000   # different noise draws for fitting and reading canvases
WINDOWS = {"early": [int(t) for t in STEPS16], "sdxl": [int(t) for t in SDXL_MATCHED]}
ROWS = ("signature", "canvas_pooled", "canvas_full", "raw_noise", "error_readout", "path_length", "shuffled")
# With --probe conv (board question 72): the grid inputs go through one small convolutional probe; the 10-number rows stay linear.
CONV_ROWS = {"signature": "signature_grid", "canvas": "canvas_grid", "raw_noise": "raw_noise_grid",
             "shuffled": "signature_grid", "error_readout": None, "path_length": None}
CONV_UPDATES = 1500
PROBE = {"kind": "linear"}


# ---------------------------------------------------------------- features

def noise_for(n, seed, dev):
    """One noise draw per canvas, the same at every step, so a window is one path."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 1, 32, 64, generator=g).to(dev)


@torch.no_grad()
def step_features(model, x0, eps, t):
    """Everything one step gives, from one forward pass over the null label and the ten digit labels."""
    n = x0.shape[0]
    a = ABAR.to(x0.device)[t]
    xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
    y = torch.cat([torch.full((n,), NULL, device=x0.device)] + [torch.full((n,), c, device=x0.device) for c in range(10)])
    tt = torch.full((11 * n,), t, device=x0.device, dtype=torch.long)
    out = model(xt.repeat(11, 1, 1, 1), tt, y).view(11, n, 1, 32, 64)
    e_null, e_c = out[0], out[1:].transpose(0, 1).squeeze(2)            # (n,1,32,64), (n,10,32,64)
    d = e_c - e_null                                                    # the signature, signed and unnormalised
    err_null = (eps - e_null).pow(2).flatten(1).sum(1, keepdim=True)
    err_c = (eps - e_c).pow(2).flatten(2).sum(2)                        # (n,10)
    return {
        "signature": F.avg_pool2d(d, 4).flatten(1),                     # 10 x 8 x 16 = 1,280
        "signature_coarse": F.avg_pool2d(d, 16).flatten(1),             # 10 x 2 x 4 = 80, for the stacked window
        "canvas_pooled": F.avg_pool2d(xt, 4).flatten(1),                # 128
        "canvas_full": xt.flatten(1),                                   # 2,048
        "raw_noise": F.avg_pool2d(eps, 4).flatten(1),                   # 128
        "error_readout": err_c - err_null,                              # 10
        "path_length": d.pow(2).flatten(2).sum(2),                      # 10
        "signature_grid": F.avg_pool2d(d, 2),                           # 10 x 16 x 32, for the convolutional probe
        "canvas_grid": F.avg_pool2d(xt, 2),                             # 1 x 16 x 32
        "raw_noise_grid": F.avg_pool2d(eps, 2),                         # 1 x 16 x 32
    }


@torch.no_grad()
def build(model, split, seed, ts, dev, batch=128):
    """Features for every canvas of a split at every timestep in ts, kept on the CPU in float16."""
    n = len(split["x"])
    eps_all = noise_for(n, seed, "cpu")
    feats = {t: {} for t in ts}
    for i in range(0, n, batch):
        x0 = to_x(split["x"][i:i + batch], dev)
        eps = eps_all[i:i + batch].to(dev)
        for t in ts:
            for k, v in step_features(model, x0, eps, t).items():
                feats[t].setdefault(k, []).append(v.half().cpu())
    return {t: {k: torch.cat(v) for k, v in f.items()} for t, f in feats.items()}


def presence(split):
    holds = torch.from_numpy(split["holds"].astype(np.int64))
    return torch.stack([(holds == k).any(1) for k in DIGITS], 1).float()  # (n,5)


# ---------------------------------------------------------------- the probe

def balanced_accuracy(pred, y):
    """Mean over columns of the mean of the true-positive and true-negative rates; nan for a column with one class."""
    out = []
    for j in range(y.shape[1]):
        p, t = pred[:, j], y[:, j].bool()
        if t.all() or (~t).all():
            continue
        out.append(0.5 * ((p[t]).float().mean() + (~p[~t]).float().mean()).item())
    return float(np.mean(out)) if out else float("nan")


def fit_linear(x, y, lam, dev, iters=200):
    """Logistic regression, one output per digit, L2 strength lam, full batch L-BFGS."""
    x, y = x.to(dev).float(), y.to(dev)
    w = torch.zeros(x.shape[1], y.shape[1], device=dev, requires_grad=True)
    b = torch.zeros(y.shape[1], device=dev, requires_grad=True)
    pos = y.mean(0).clamp(1e-3, 1 - 1e-3)
    weight = (pos / (1 - pos))                                            # class-balanced loss
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = x @ w + b
        loss = F.binary_cross_entropy_with_logits(z, y, pos_weight=1 / weight) + lam * w.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach(), b.detach()


class ConvProbe(torch.nn.Module):
    """One small convolutional probe, the same for every grid input; only the input channel count differs."""
    def __init__(self, cin, out=len(DIGITS)):
        super().__init__()
        blk = lambda a, b, s: [torch.nn.Conv2d(a, b, 3, stride=s, padding=1), torch.nn.GroupNorm(8, b), torch.nn.SiLU()]
        self.body = torch.nn.Sequential(*blk(cin, 32, 1), *blk(32, 64, 2), *blk(64, 64, 2))
        self.head = torch.nn.Linear(128, out)

    def forward(self, x):
        h = self.body(x)
        return self.head(torch.cat([h.amax((2, 3)), h.mean((2, 3))], 1))


def fit_conv(fit_x, fit_y, read_x, dev, updates=CONV_UPDATES, seed=0):
    """Fixed recipe, chosen before any run: Adam 1e-3, weight decay 1e-4, batch 256, class-balanced BCE."""
    torch.manual_seed(seed)
    m, s = fit_x.float().mean((0, 2, 3), keepdim=True), fit_x.float().std((0, 2, 3), keepdim=True).clamp_min(1e-6)
    norm = lambda x: ((x.float() - m) / s)
    net = ConvProbe(fit_x.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pos = fit_y.mean(0).clamp(1e-3, 1 - 1e-3).to(dev)
    g = torch.Generator().manual_seed(seed)
    for _ in range(updates):
        i = torch.randint(len(fit_x), (min(256, len(fit_x)),), generator=g)
        z = net(norm(fit_x[i]).to(dev))
        loss = F.binary_cross_entropy_with_logits(z, fit_y[i].to(dev), pos_weight=(1 - pos) / pos)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        run = lambda x: torch.cat([net(norm(x[j:j + 512]).to(dev)).cpu() for j in range(0, len(x), 512)])
        return run(fit_x), run(read_x)


def probe_conv(fit_x, fit_y, read_x, read_y, dev):
    lf, lr = fit_conv(fit_x, fit_y, read_x, dev)
    return {"presence": balanced_accuracy(lr > 0, read_y), "fit_presence": balanced_accuracy(lf > 0, fit_y)}, lf, lr


def standardise(fit, read):
    m, s = fit.float().mean(0), fit.float().std(0).clamp_min(1e-6)
    return (fit.float() - m) / s, (read.float() - m) / s


def probe(fit_x, fit_y, read_x, read_y, dev, seed=0):
    """Choose lam by 5-fold cross-validation on the fitting canvases, refit on all of them, read once."""
    fx, rx = standardise(fit_x, read_x)
    folds = torch.randperm(len(fx), generator=torch.Generator().manual_seed(seed)) % FOLDS
    best, best_lam = -1.0, LAMBDAS[0]
    for lam in LAMBDAS:
        scores = []
        for f in range(FOLDS):
            tr, va = folds != f, folds == f
            w, b = fit_linear(fx[tr], fit_y[tr], lam, dev)
            scores.append(balanced_accuracy(((fx[va].to(dev) @ w + b) > 0).cpu(), fit_y[va]))
        if np.nanmean(scores) > best:
            best, best_lam = float(np.nanmean(scores)), lam
    w, b = fit_linear(fx, fit_y, best_lam, dev)
    logits_fit = (fx.to(dev) @ w + b).cpu()
    logits = (rx.to(dev) @ w + b).cpu()
    return {"presence": balanced_accuracy(logits > 0, read_y), "lam": best_lam, "cv": best}, logits_fit, logits


def apart_vs_wrong(logits_fit, fit_split, logits_read, read_split):
    """On apart and wrong canvases: score the asked pair by the smaller of its two digit logits, threshold set on the
    fitting canvases, balanced accuracy on the reading canvases. Layout alone scores 0.5."""
    def scores(logits, split):
        keep = np.isin(split["kind"], [KINDS.index("apart"), KINDS.index("wrong")])
        asked = torch.from_numpy(split["asked"][keep].astype(np.int64))
        lg = logits[torch.from_numpy(keep)]
        s = torch.minimum(lg.gather(1, asked[:, :1]), lg.gather(1, asked[:, 1:])).squeeze(1)
        return s, torch.from_numpy(split["kind"][keep] == KINDS.index("apart"))
    sf, yf = scores(logits_fit, fit_split)
    sr, yr = scores(logits_read, read_split)
    if yf.all() or (~yf).all() or yr.all() or (~yr).all():
        return float("nan")
    cands = torch.unique(sf)
    accs = [0.5 * ((sf[yf] > c).float().mean() + (sf[~yf] <= c).float().mean()) for c in cands]
    c = cands[int(torch.stack(accs).argmax())]
    return 0.5 * ((sr[yr] > c).float().mean() + (sr[~yr] <= c).float().mean()).item()


def row_input(feats, t, row):
    if PROBE["kind"] == "conv":
        return feats[t][CONV_ROWS[row] or row]
    return feats[t]["signature" if row == "shuffled" else row]


def any_probe(row, fx, fy, rx, ry, dev):
    if PROBE["kind"] == "conv" and CONV_ROWS.get(row):
        return probe_conv(fx, fy, rx, ry, dev)
    return probe(fx.flatten(1) if fx.dim() > 2 else fx, fy, rx.flatten(1) if rx.dim() > 2 else rx, ry, dev)


def read_window(fit_f, read_f, fit_split, read_split, ts, dev, rows=None):
    rows = rows or (tuple(CONV_ROWS) if PROBE["kind"] == "conv" else ROWS)
    fy, ry = presence(fit_split), presence(read_split)
    shuffled = fy[torch.randperm(len(fy), generator=torch.Generator().manual_seed(7))]
    out = {r: [] for r in rows}
    for t in ts:
        for r in rows:
            res, lf, lr = any_probe(r, row_input(fit_f, t, r), shuffled if r == "shuffled" else fy,
                                    row_input(read_f, t, r), ry, dev)
            res["apart_vs_wrong"] = apart_vs_wrong(lf, fit_split, lr, read_split)
            res["t"] = t
            out[r].append(res)
    # the whole window stacked: coarse signature (80 per step) and summed path length (10)
    stack = lambda f, k: torch.cat([f[t][k] for t in ts], 1)
    for name, fx, rx in (("signature_window", stack(fit_f, "signature_coarse"), stack(read_f, "signature_coarse")),
                         ("path_length_window", sum(fit_f[t]["path_length"].float() for t in ts),
                          sum(read_f[t]["path_length"].float() for t in ts))):
        res, lf, lr = probe(fx, fy, rx, ry, dev)
        res["apart_vs_wrong"] = apart_vs_wrong(lf, fit_split, lr, read_split)
        out[name] = res
    return out


def verdict_for(window):
    """The bar: signature presence at least 0.75 at some step, and at least 0.1 above the raw-noise probe there."""
    sig, noise = window["signature"], window["raw_noise"]  # same row names under both probes
    hits = [s["t"] for s, z in zip(sig, noise) if s["presence"] >= MIN_PRESENCE and s["presence"] - z["presence"] >= MIN_OVER_NOISE]
    best = max(sig, key=lambda s: s["presence"])
    return {"passes": bool(hits), "steps_passing": hits, "best_presence": best["presence"], "best_t": best["t"]}


# ---------------------------------------------------------------- stages

def subset(split, n, seed):
    idx = np.sort(np.random.default_rng(seed).choice(len(split["x"]), n, replace=False))
    return take(split, idx)


def stage_smoke(model, data, dev):
    split = subset(data["train"], 128, 0)
    x0, eps = to_x(split["x"], dev), noise_for(128, FIT_SEED, dev)
    step_features(model, x0, eps, 600)
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    f = step_features(model, x0, eps, 600)
    if dev == "cuda":
        torch.cuda.synchronize()
    sec = time.time() - t0
    shapes = {k: list(v.shape) for k, v in f.items()}
    finite = all(torch.isfinite(v).all().item() for v in f.values())
    per_forward = sec / (11 * 128)
    return {"shapes": shapes, "finite": finite, "seconds_per_batch_of_128": sec,
            "canvas_forwards_per_second": 1 / per_forward, "pass": finite}


def stage_control(model, data, dev, n_fit=1000, n_read=500, gate=None):
    """gate: the rows that must reach MIN_CONTROL. Board question 74: in the full read only the canvas row gates."""
    fit = subset(data["train"], n_fit, 1)
    read = data["seen_test"] if n_read >= len(data["seen_test"]["x"]) else subset(data["seen_test"], n_read, 2)
    ff, rf = build(model, fit, FIT_SEED, [CONTROL_T], dev), build(model, read, READ_SEED, [CONTROL_T], dev)
    pair = ("signature", "canvas") if PROBE["kind"] == "conv" else ("signature", "canvas_full")
    res = {r: any_probe(r, row_input(ff, CONTROL_T, r), presence(fit), row_input(rf, CONTROL_T, r), presence(read), dev)[0]
           for r in pair}
    gate = gate or pair
    res["gate"] = list(gate)
    res["pass"] = all(res[r]["presence"] >= MIN_CONTROL for r in gate)
    return res


def stage_overfit(model, data, dev):
    split = subset(data["train"], OVERFIT_N, 3)
    f = build(model, split, FIT_SEED, [600], dev)
    x, y = standardise(f[600]["signature"], f[600]["signature"])[0], presence(split)
    w, b = fit_linear(x, y, 1e-6, dev, iters=500)
    right = int(((((x.to(dev) @ w + b) > 0).cpu().float()) == y).all(1).sum())
    return {"right": right, "of": OVERFIT_N, "pass": right >= OVERFIT_RIGHT}


def stage_read(model, data, dev, n_fit, n_read):
    fit = subset(data["train"], n_fit, 4)
    read = data["seen_test"] if n_read >= len(data["seen_test"]["x"]) else subset(data["seen_test"], n_read, 5)
    out = {}
    for name, ts in WINDOWS.items():
        ff, rf = build(model, fit, FIT_SEED, ts, dev), build(model, read, READ_SEED, ts, dev)
        out[name] = read_window(ff, rf, fit, read, ts, dev)
        out[name]["verdict"] = verdict_for(out[name])
        print(name, json.dumps(out[name]["verdict"]), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["smoke", "control", "overfit", "small", "full"])
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--probe", choices=["linear", "conv"], default="linear")
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = args.device
    PROBE["kind"] = args.probe
    if args.probe == "conv":
        args.out = args.out / "conv"
    data, data_hash = load_data(args.data)
    model = load_unet(args.model, dev)
    t0 = time.time()
    if args.stage == "smoke":
        res = stage_smoke(model, data, dev)
    elif args.stage == "control":
        res = stage_control(model, data, dev)
    elif args.stage == "overfit":
        res = stage_overfit(model, data, dev)
    elif args.stage == "small":
        res = stage_read(model, data, dev, SMALL_FIT, SMALL_READ)
    else:
        canvas_row = "canvas" if PROBE["kind"] == "conv" else "canvas_full"
        control = stage_control(model, data, dev, FIT_N, READ_N, gate=(canvas_row,))
        print("control at full size", json.dumps({k: v for k, v in control.items() if k != "gate"}, default=float), flush=True)
        if not control["pass"]:
            res = {"control": control, "verdict": "inconclusive: the canvas row missed the control at full size"}
        else:
            res = stage_read(model, data, dev, FIT_N, READ_N)
            res["control"] = control
            e, s = res["early"]["verdict"], res["sdxl"]["verdict"]
            res["verdict"] = ("support" if e["passes"] else "support at SDXL's noise only" if s["passes"] else "null")
    res.update({"probe": args.probe, "stage": args.stage, "data_hash": data_hash, "model": str(args.model), "seconds": time.time() - t0,
                "bars": {"min_presence": MIN_PRESENCE, "min_over_noise": MIN_OVER_NOISE, "min_control": MIN_CONTROL,
                         "overfit_right": OVERFIT_RIGHT}, "windows": WINDOWS, "fit_seed": FIT_SEED, "read_seed": READ_SEED})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.stage}.json").write_text(json.dumps(res, indent=1, default=float))
    print(args.stage, "done in", round(time.time() - t0, 1), "s", flush=True)
    if args.wandb_name and args.stage == "full" and "early" in res:
        import wandb
        run = wandb.init(project="poe-repair-animals-compose", entity="prime_lab", name=args.wandb_name,
                         config={"stage": args.stage, "windows": WINDOWS, "data_hash": data_hash})
        for w in WINDOWS:
            for r in (CONV_ROWS if args.probe == "conv" else ROWS):
                for s in res[w][r]:
                    run.log({f"{w}/{r}/presence": s["presence"], f"{w}/{r}/apart_vs_wrong": s["apart_vs_wrong"], "t": s["t"]})
        run.summary["verdict"] = res["verdict"]
        res["wandb_id"] = run.id
        (args.out / f"{args.stage}.json").write_text(json.dumps(res, indent=1, default=float))
        run.finish()


if __name__ == "__main__":
    main()
