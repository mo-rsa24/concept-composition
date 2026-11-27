#!/usr/bin/env python3
"""Paper 2's toy rung (plans/paper-2-dials/plans/04-full-ladder.md, rung 1) in miniature: minutes on the laptop instead of 26 GPU-hours.

Four dials (rank-4 adapters on a frozen conditional denoiser) are each trained alone by the plan's recipe for their kind, read alone,
stacked in pairs, and stacked with LoRA.rar's solo-behaviour correction. Every measurement is exact, from pixels (toy_canvas.py).

  identity   by gradient through the soft shape reading of the decoded estimate, on two-object square-and-diamond conditions
  binding    by gradient through the soft colour reading inside each object's own pixels, trained on green diamond, tested on yellow
             square (both never in a training canvas)
  count      on kept draws: the frozen model asked for one more blue object, draws whose count hit kept, fine-tuned on them with the
             original condition at strength 1
  relation   on kept draws: the frozen model's own draws binned by the horizontal gap between the two named objects, fine-tuned at
             strength 0.5 on the middle bin and at strength 1 on the far bin

Subcommands, run in order; each writes its section of results.json, its picture grids and its seconds:
  shapes   stage 1a: template margins, the reader on 2,000 renders, the 200,000 training canvases
  base     stage 1b: train the base model
  stage1   stage 1c: what the base draws, and where decoded estimates agree with the finished picture
  dials    train the four dials (closeness weight chosen from a 25-update smoke), read each on tuning seeds
  stage2   each dial alone on test conditions at strengths 0, 0.5, 1, 1.5, beside a random dial of the same norm
  stage3   every pair stacked at strength 1: interference on matched seeds
  corr     train the solo-behaviour correction on the pairs' training conditions
  stage4   the plain stack against the solo-behaviour correction on the pairs' test conditions
  stage5   the deciding test: the count dial stacked with the identity dial

  /home/molef/micromamba/envs/voice-agent/bin/python 2-classifier-guidance/scripts/paper2/toy/dials_toy.py shapes
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toy_canvas as tc  # noqa: E402
import toy_model as tm  # noqa: E402

st = tm.st
DEV = tm.DEV
REPO = Path(__file__).resolve().parents[4]
# PAPER2_TOY_OUT and PAPER2_TOY_RESULTS move checkpoints and results (on the cluster, under /datasets/mmolefe/poe_repair_min/toys/paper2/)
OUT = Path(os.environ.get("PAPER2_TOY_OUT", REPO / "outputs" / "paper2-toy"))
RESULTS = Path(os.environ.get("PAPER2_TOY_RESULTS", REPO / "artifacts" / "results" / "can-dials-be-trained-alone-and-stacked-on-a-small-toy"))

# ---- the bars (README "The bars"), fixed before stage 1 ran --------------------------------------------------------------------------
R1_TEMPLATE_MARGIN_MIN = 0.15
R1_READER_RENDER_MIN = 1.0
R1_BASE_COUNT_MIN = 0.95
R1_BASE_LOOKALIKE_MAX = 0.70
DECODED_AGREEMENT_MIN = 0.90
R1_DIAL_HIT_MIN = 0.85
R1_DIAL_MONOTONE = True
R1_DIAL_VS_RANDOM_MIN = 0.20
R1_ID_LEAD_MIN = 0.20
P_MAX = 0.05
R1_INTERFERENCE_SHOWN = 0.10
SOLO_VS_PLAIN_MIN = 0.15
DECIDE_INTERFERENCE_MIN = 0.10
DECIDE_SOLO_LEAD_MIN = 0.10
GPU_MINUTES_STOP = 240

N_EVAL = 1024
STRENGTHS = (0.0, 0.5, 1.0, 1.5)
TRAINED_STRENGTHS = {"identity": (1.0,), "binding": (1.0,), "count": (1.0,), "relation": (0.5, 1.0)}
GRID_COLS = 8
WINDOW_STEPS = 6
BOOT = 10_000


# -------------------------------------------------------------------------------------------------------------------------------------
# Bookkeeping


def sync():
    if DEV.type == "cuda":
        torch.cuda.synchronize()


def load_results():
    p = RESULTS / "results.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_results(key, value):
    RESULTS.mkdir(parents=True, exist_ok=True)
    res = load_results()
    res[key] = value
    res["device"] = str(DEV) + (f" ({torch.cuda.get_device_name()})" if DEV.type == "cuda" else "")
    (RESULTS / "results.json").write_text(json.dumps(res, indent=1))


def grid(rows, labels, path, col_labels=None, calls=None, scale=8, gap=2):
    """One row per entry of rows (each (n, 3, 16, 16)), GRID_COLS columns, 8x nearest-neighbour; optional reader call under each."""
    from PIL import Image, ImageDraw
    lw, tile = 170, tc.H * scale
    sub = 12 if calls else 0
    img = Image.new("RGB", (lw + GRID_COLS * (tile + gap), len(rows) * (tile + gap + sub) + 14), "white")
    dr = ImageDraw.Draw(img)
    for c in range(GRID_COLS):
        if col_labels:
            dr.text((lw + c * (tile + gap) + 2, 1), col_labels[c], fill="black")
    for r, (x, lab) in enumerate(zip(rows, labels)):
        y = 14 + r * (tile + gap + sub)
        for li, line in enumerate(lab.split("\n")):
            dr.text((4, y + tile // 2 - 12 + 12 * li), line, fill="black")
        arr = ((x[:GRID_COLS].clamp(-1, 1).cpu().numpy().transpose(0, 2, 3, 1) + 1) * 127.5).round().astype(np.uint8)
        for c in range(min(GRID_COLS, len(arr))):
            img.paste(Image.fromarray(arr[c]).resize((tile, tile), Image.NEAREST), (lw + c * (tile + gap), y))
            if calls:
                dr.text((lw + c * (tile + gap) + 2, y + tile), calls[r][c], fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def call_text(reading):
    s = " ".join(("S", "D", "b", "o")[sh] + "rgby"[c] for sh, c, *_ in reading["objects"])
    return s + (f" +{reading['specks']}sp" if reading["specks"] else "")


def mcnemar(a, b):
    """One-sided exact McNemar: a better than b on matched canvases. a, b boolean arrays."""
    from scipy.stats import binomtest
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(n10, n10 + n01, 0.5, alternative="greater").pvalue if n10 + n01 else 1.0
    return {"a_only": n10, "b_only": n01, "p": float(p)}


def eval_set(conds, n=N_EVAL, seed0=None):
    """(conditions per canvas, seeds): conditions cycled over n consecutive seeds."""
    seed0 = tc.SEEDS["test"] if seed0 is None else seed0
    return [conds[i % len(conds)] for i in range(n)], list(range(seed0, seed0 + n))


def gpu_seconds_so_far():
    res = load_results()
    return sum(v.get("seconds", 0) for k, v in res.items() if isinstance(v, dict))


# -------------------------------------------------------------------------------------------------------------------------------------
# Data


def canvases():
    d = np.load(OUT / "train_canvases.npz")
    return d["idx"], d["cond"]


def outside_purpose_mask(dial, cond):
    """Training canvases outside a dial's purpose, for its closeness term (cond: (N, 14) numpy)."""
    n_obj = cond[:, 8:10].sum(1)
    if dial == "count":
        return cond[:, 10 + tc.BLUE] == 0
    if dial == "relation":
        return n_obj != 2
    return np.ones(len(cond), bool)


def base_path():
    return OUT / "base.pt"


def dial_path(d):
    return OUT / f"dial-{d}.pt"


def load_dial_states(names):
    return {d: torch.load(dial_path(d)) for d in names}


# -------------------------------------------------------------------------------------------------------------------------------------
# Stage 1a: shapes, reader, training canvases


def cmd_shapes(a):
    t0 = time.time()
    margins = {}
    for s, nm in enumerate(("square", "diamond")):
        m = np.zeros((tc.H, tc.H), bool)
        m[5:10, 5:10] = tc.TEMPLATES[s]
        ious = tc.shape_ious(m)
        margins[nm] = {"iou_square": round(float(ious[0]), 3), "iou_diamond": round(float(ious[1]), 3),
                       "margin": round(float(ious[s] - ious[1 - s]), 3)}
    margin_ok = all(v["margin"] >= R1_TEMPLATE_MARGIN_MIN for v in margins.values())
    # the reader on 2,000 renders over every combination, mixed and held-out included
    rng = np.random.default_rng(tc.SEEDS["reader_check"])
    conds = [tuple(sorted(rng.integers(0, 8, size=rng.integers(1, 4)).tolist())) for _ in range(2000)]
    pos = tc.sample_positions([len(c) for c in conds], rng)
    idx = tc.render(conds, pos)
    x = tc.to_rgb(idx)
    R = tc.read(x)
    right = 0
    for r, c, p in zip(R, conds, pos):
        want = sorted((tc.shape_of(k), tc.colour_of(k)) for k in c)
        got = sorted((s, col) for s, col, *_ in r["objects"])
        # centres: a template's centre is its top-left + 2 in both coordinates
        ok = want == got and r["specks"] == 0
        cents = sorted((float(pp[0] + 2), float(pp[1] + 2)) for pp in p)
        gotc = sorted((round(cy, 6), round(cx, 6)) for _, _, cy, cx, _ in r["objects"])
        ok = ok and all(abs(a1 - b1) < 1e-6 and abs(a2 - b2) < 1e-6 for (a1, a2), (b1, b2) in zip(cents, gotc))
        right += ok
    reader_share = right / len(R)
    # training canvases
    tidx, tconds = tc.training_canvases(a.n)
    mixed = sum(1 for c in tconds if len({tc.shape_of(k) for k in c}) > 1)
    held = sum(1 for c in tconds if any(k in tc.HELD_OUT for k in c))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "train_canvases.npz", idx=tidx, cond=tc.cond_vector(tconds).numpy().astype(np.int8))
    grid([x[i * 8:(i + 1) * 8] for i in range(8)], [f"renders {i * 8}-{i * 8 + 7}" for i in range(8)],
         RESULTS / "stage1a-renders-with-reader-calls.png", calls=[[call_text(R[i * 8 + j]) for j in range(8)] for i in range(8)])
    info = {"template_margins": margins, "template_margin_bar": R1_TEMPLATE_MARGIN_MIN, "template_margin_pass": margin_ok,
            "reader_renders": len(R), "reader_right_share": reader_share, "reader_pass": reader_share >= R1_READER_RENDER_MIN,
            "training_canvases": a.n, "training_with_square_and_diamond": mixed, "training_with_held_out_combination": held,
            "training_conditions": len(tc.training_conditions()), "seconds": 0.0, "cpu_seconds": round(time.time() - t0, 1)}
    print(json.dumps(info, indent=1))
    save_results("stage1a", info)


# -------------------------------------------------------------------------------------------------------------------------------------
# Stage 1b: the base model


def cmd_base(a):
    sync()
    t0 = time.time()
    idx, cond = canvases()
    X = torch.from_numpy(idx).to(DEV)
    C = torch.from_numpy(cond).float().to(DEV)
    pal = tc.PALETTE.to(DEV)
    torch.manual_seed(0)
    model = tm.Denoiser().to(DEV)
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.99))
    nparams = sum(p.numel() for p in model.parameters())
    curve = []
    for u in range(a.updates):
        lr = a.lr * min(1.0, (a.updates - u) / (0.2 * a.updates))          # last fifth decays linearly to zero
        for g in opt.param_groups:
            g["lr"] = lr
        k = torch.randint(len(X), (a.batch,), device=DEV)
        x0 = pal[X[k].long()].permute(0, 3, 1, 2)
        s = st.draw_sigma(a.batch)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
            D = model(x0 + s.view(-1, 1, 1, 1) * torch.randn_like(x0), s, C[k])
        loss = st.edm_loss(D.float(), x0, s, model.net.logvar(s).float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - a.ema)
        if u % 500 == 0 or u == a.updates - 1:
            curve.append((u, round(float(loss.detach()), 4)))
            print(f"  base update {u} loss {float(loss.detach()):.4f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save(ema.state_dict(), base_path())
    sync()
    info = {"updates": a.updates, "batch": a.batch, "lr": a.lr, "ema": a.ema, "images_seen": a.updates * a.batch,
            "parameters": nparams, "loss_curve": curve, "seconds": round(time.time() - t0, 1)}
    save_results("stage1b", info)
    print(json.dumps({k: v for k, v in info.items() if k != "loss_curve"}))


# -------------------------------------------------------------------------------------------------------------------------------------
# Stage 1c: what the base draws


@torch.no_grad()
def decoded_agreement(model, conds, seeds):
    """Per Heun step: share of canvases whose decoded estimate (the step's first denoiser output) reads the same objects as the
    finished picture, (shape, colour) multiset and specks both equal."""
    cv = tc.cond_vector(conds).to(DEV)
    est = {}

    def den(x, s, i, second):
        D = model(x, s, cv[:len(x)])
        if not second:
            est[i] = D.cpu()
        return D
    final = tm.heun_range(den, st.starting_noise(seeds)).cpu()
    fr = tc.read(final)
    out = []
    for i in range(st.STEPS):
        er = tc.read(est[i])
        out.append(float(np.mean([tc.pairs_of(e) == tc.pairs_of(f) and e["specks"] == f["specks"] for e, f in zip(er, fr)])))
    return out


def cmd_stage1(a):
    sync()
    t0 = time.time()
    model = tm.load_base(base_path())
    res = {}
    # non-look-alike conditions: every training condition, test seeds
    conds, seeds = eval_set(tc.training_conditions(), seed0=a.test_seed0)
    x = tm.sample(model, conds, seeds)
    R = tc.read(x)
    exact = np.array([tc.asked_exactly(r, c) for r, c in zip(R, conds)])
    count_ok = np.array([len(r["objects"]) == len(c) and not r["specks"] for r, c in zip(R, conds)])
    by_n = {n: float(exact[[len(c) == n for c in conds]].mean()) for n in (1, 2, 3)}
    res["base_non_lookalike"] = {"canvases": len(R), "exactly_asked_share": float(exact.mean()), "count_right_share": float(count_ok.mean()),
                                 "exactly_asked_by_object_count": by_n, "bar": R1_BASE_COUNT_MIN, "pass": bool(exact.mean() >= R1_BASE_COUNT_MIN)}
    rows, labels, calls = [], [], []
    for n in (1, 2, 3):
        sel = [i for i, c in enumerate(conds) if len(c) == n][:GRID_COLS]
        rows.append(x[sel]); labels.append(f"base, {n} object(s)\n" + ", ".join(tc.name(conds[i]) for i in sel[:2]) + " ...")
        calls.append([call_text(R[i]) for i in sel])
    # look-alike: identity test conditions
    lc, ls = eval_set(tc.DIAL_CONDS["identity"][1], seed0=a.test_seed0)
    xl = tm.sample(model, lc, ls)
    Rl = tc.read(xl)
    both = np.array([tc.asked_exactly(r, c) for r, c in zip(Rl, lc)])
    shape_ok = np.array([tc.m_shape(r, c) for r, c in zip(Rl, lc)])
    what = Counter(" ".join(sorted(tc.NAMES[k] if k < 100 else ("b", "o")[(k - 100) // 10 - 2] + "RGBY"[k % 10]
                                   for k in tc.pairs_of(r).elements())) for r in Rl)
    res["base_lookalike"] = {"canvases": len(Rl), "exactly_asked_share": float(both.mean()), "shape_measure_share": float(shape_ok.mean()),
                             "per_condition": {tc.name(c): float(both[[cc == c for cc in lc]].mean()) for c in tc.DIAL_CONDS["identity"][1]},
                             "most_common_drawings": what.most_common(8), "bar_max": R1_BASE_LOOKALIKE_MAX,
                             "pass": bool(both.mean() <= R1_BASE_LOOKALIKE_MAX)}
    rows.append(xl[:GRID_COLS]); labels.append("base, square + diamond\n(identity test conditions)")
    calls.append([call_text(r) for r in Rl[:GRID_COLS]])
    # binding: held-out combinations, and every dial's training conditions on tuning seeds (what each dial starts from)
    for d in tc.DIALS:
        for side, cs, s0 in (("train_tuning_seeds", tc.DIAL_CONDS[d][0], a.tune_seed0), ("test", tc.DIAL_CONDS[d][1], a.test_seed0)):
            c2, s2 = eval_set(cs, n=512, seed0=s0)
            x2 = tm.sample(model, c2, s2)
            R2 = tc.read(x2)
            res[f"base_{d}_{side}"] = {"exactly_asked_share": float(np.mean([tc.asked_exactly(r, c) for r, c in zip(R2, c2)])),
                                       "canvases": len(R2)}
            if d == "binding" and side == "test":
                rows.append(x2[:GRID_COLS]); labels.append("base, held-out yellow square\n(binding test conditions)")
                calls.append([call_text(r) for r in R2[:GRID_COLS]])
                res["base_binding_test"]["most_common_drawings"] = Counter(
                    " ".join(sorted(tc.NAMES[k] if k < 100 else "?" for k in tc.pairs_of(r).elements())) for r in R2).most_common(6)
    # decoded estimates against the finished picture, on tuning seeds of the two gradient dials' training conditions
    ac, as_ = eval_set(tc.DIAL_CONDS["identity"][0] + tc.DIAL_CONDS["binding"][0], n=512, seed0=a.tune_seed0)
    agree = decoded_agreement(model, ac, as_)
    t = st.sigmas("cpu")
    first = next((i for i, v in enumerate(agree) if v >= DECODED_AGREEMENT_MIN), None)
    window = list(range(first, min(first + WINDOW_STEPS, st.STEPS))) if first is not None else []
    res["decoded_agreement"] = {"per_step": [{"step": i, "sigma": round(float(t[i]), 4), "agreement": round(v, 3)} for i, v in enumerate(agree)],
                                "bar": DECODED_AGREEMENT_MIN, "gradient_window_steps": window,
                                "gradient_window_sigmas": [round(float(t[i]), 4) for i in window]}
    res["test_seeds"] = [a.test_seed0, a.test_seed0 + N_EVAL - 1]
    res["tuning_seeds"] = [a.tune_seed0, a.tune_seed0 + 511]
    grid(rows, labels, RESULTS / f"stage1-base-samples{a.suffix}.png", calls=calls)
    sync()
    res["seconds"] = round(time.time() - t0, 1)
    save_results(a.key, res)
    print(json.dumps({k: v for k, v in res.items() if k != "decoded_agreement"}, indent=1))
    print("agreement", [(r["step"], r["sigma"], r["agreement"]) for r in res["decoded_agreement"]["per_step"]])
    print("window", window)


def cmd_misses(a):
    """Sort the base model's misses on the non-look-alike canvases (the same seeds as stage 1) by kind, and draw eight of them."""
    sync()
    t0 = time.time()
    model = tm.load_base(base_path())
    conds, seeds = eval_set(tc.training_conditions(), seed0=a.test_seed0)
    x = tm.sample(model, conds, seeds)
    R = tc.read(x)
    kinds, miss = Counter(), []
    for i, (r, c) in enumerate(zip(R, conds)):
        if tc.asked_exactly(r, c):
            continue
        miss.append(i)
        n = len(r["objects"])
        if r["specks"]:
            k = "specks"
        elif n < len(c):
            k = "too few objects"
        elif n > len(c):
            k = "too many objects"
        elif any(s >= 2 for s, *_ in r["objects"]):
            k = "blend or other shape"
        elif Counter(col for _, col, *_ in r["objects"]) != Counter(tc.colour_of(q) for q in c):
            k = "wrong colours"
        else:
            k = "wrong shape"
        kinds[k] += 1

    def close(r):
        o = r["objects"]
        return any(abs(p[2] - q[2]) < tc.OBJ + tc.GAP_MIN and abs(p[3] - q[3]) < tc.OBJ + tc.GAP_MIN for j, p in enumerate(o) for q in o[j + 1:])
    sel = miss[:GRID_COLS]
    grid([x[sel]], ["base misses,\nnon-look-alike"], RESULTS / f"stage1-base-misses{a.suffix}.png", col_labels=[tc.name(conds[i]) for i in sel],
         calls=[[call_text(R[i]) for i in sel]])
    sync()
    info = {"misses": len(miss), "kinds": dict(kinds), "misses_by_asked_object_count": dict(Counter(len(conds[i]) for i in miss)),
            "misses_with_two_objects_closer_than_training_gap": sum(close(R[i]) for i in miss), "seconds": round(time.time() - t0, 1)}
    info["test_seeds"] = [a.test_seed0, a.test_seed0 + N_EVAL - 1]
    save_results(a.key, info)
    print(json.dumps(info, indent=1))


# -------------------------------------------------------------------------------------------------------------------------------------
# Dials


def closeness(model, dial, X, C, mask_idx, batch, pal):
    k = mask_idx[torch.randint(len(mask_idx), (batch,), device=DEV)]
    x0 = pal[X[k].long()].permute(0, 3, 1, 2)
    s = st.draw_sigma(batch)
    x = x0 + s.view(-1, 1, 1, 1) * torch.randn_like(x0)
    tm.set_active(model, {})
    with torch.no_grad():
        Df = model(x, s, C[k])
    tm.set_active(model, {dial: 1.0})
    Dd = model(x, s, C[k])
    return (st.loss_weight(s.view(-1, 1, 1, 1)) * (Dd - Df) ** 2).mean()


def kept_draws(model, dial, n_per):
    """Count: the frozen model asked for one more blue object; kept when the object count and blue count hit. Relation: the frozen
    model's own draws with a defined gap, binned. Returns (pictures, conditions to train with, strengths, info)."""
    tm.set_active(model, {})
    conds = tc.DIAL_CONDS[dial][0]
    xs, cs, ss, info = [], [], [], {}
    for j, c in enumerate(conds):
        seeds = list(range(tc.SEEDS["kept"] + j * 100_000, tc.SEEDS["kept"] + j * 100_000 + n_per))
        if dial == "count":
            ask = tuple(sorted(c + (tc.asked_blue(c),)))
            x = tm.sample(model, [ask] * n_per, seeds)
            R = tc.read(x)
            keep = [i for i, r in enumerate(R) if not r["specks"] and len(r["objects"]) == len(ask)
                    and sum(1 for o in r["objects"] if o[1] == tc.BLUE) == sum(1 for k in ask if tc.colour_of(k) == tc.BLUE)]
            right_shape = float(np.mean([tc.asked_exactly(R[i], ask) for i in keep])) if keep else 0.0
            info[tc.name(c)] = {"asked": tc.name(ask), "drawn": n_per, "kept": len(keep), "kept_with_every_shape_right": right_shape}
            xs.append(x[keep]); cs += [c] * len(keep); ss += [1.0] * len(keep)
        else:
            x = tm.sample(model, [c] * n_per, seeds)
            R = tc.read(x)
            g = [tc.m_gap(r, c) for r in R]
            per = {}
            for s, (lo, hi) in tc.RELATION_BINS.items():
                keep = [i for i, v in enumerate(g) if v is not None and lo <= v < hi]
                xs.append(x[keep]); cs += [c] * len(keep); ss += [s] * len(keep)
                per[str(s)] = len(keep)
            info[tc.name(c)] = {"drawn": n_per, "gap_defined": sum(v is not None for v in g), "kept_per_strength": per}
    return torch.cat(xs), cs, torch.tensor(ss), info


def train_kept(model, dial, data, updates, w_close, batch, lr, X, C, close_idx, pal, log_every=250):
    xk, ck, sk = data
    xk, cvk, sk = xk.to(DEV), tc.cond_vector(ck).to(DEV), sk.to(DEV)
    params = tm.dial_params(model, dial)
    opt = torch.optim.Adam(params, lr=lr)
    levels = sorted(set(sk.tolist()))
    curve = []
    for u in range(updates):
        s_now = levels[u % len(levels)]
        pool = (sk == s_now).nonzero()[:, 0]
        k = pool[torch.randint(len(pool), (batch,), device=DEV)]
        x0 = xk[k]
        s = st.draw_sigma(batch)
        tm.set_active(model, {dial: s_now})
        D = model(x0 + s.view(-1, 1, 1, 1) * torch.randn_like(x0), s, cvk[k])
        main = (st.loss_weight(s.view(-1, 1, 1, 1)) * (D - x0) ** 2).mean()
        cl = closeness(model, dial, X, C, close_idx, batch, pal)
        loss = main + w_close * cl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if u % log_every == 0 or u == updates - 1:
            curve.append({"update": u, "main": round(float(main), 5), "closeness_raw": float(cl),
                          "closeness_share": round(float(w_close * cl / loss), 4)})
    return curve


def train_gradient(model, dial, updates, w_close, batch, lr, window, X, C, close_idx, pal, log_every=50):
    conds = tc.DIAL_CONDS[dial][0]
    kind = "identity" if dial == "identity" else "binding"
    params = tm.dial_params(model, dial)
    opt = torch.optim.Adam(params, lr=lr)
    t = st.sigmas(DEV)
    rng = random.Random(0)
    curve = []
    for u in range(updates):
        cb = [conds[rng.randrange(len(conds))] for _ in range(batch)]
        cv = tc.cond_vector(cb).to(DEV)
        step = window[rng.randrange(len(window))]
        x = torch.randn(batch, 3, tc.H, tc.H, device=DEV) * st.SIGMA_MAX
        tm.set_active(model, {dial: 1.0})
        with torch.no_grad():
            x = tm.heun_range(tm.denoiser_for(model, cv), x, 0, step)
        x0hat = model(x, t[step], cv)
        tgt, win, col, used = tm.targets_from(x0hat, cb, kind)
        main = tm.soft_loss(x0hat, tgt, win, col, kind)
        cl = closeness(model, dial, X, C, close_idx, batch, pal)
        loss = main + w_close * cl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if u % log_every == 0 or u == updates - 1:
            curve.append({"update": u, "step": step, "main": round(float(main), 5), "canvases_with_loss": used,
                          "closeness_raw": float(cl), "closeness_share": round(float(w_close * cl / loss.clamp_min(1e-12)), 4)})
    return curve


def choose_close_weight(smoke_curve):
    """The plan: the closeness term is at least a tenth of the loss at the first logged step. With raw closeness C and main loss L at
    the smoke's last update, w C >= (L + w C) / 10 gives w >= L / (9 C); rounded up to two significant figures, at least 1."""
    last = smoke_curve[-1]
    L, Cc = last["main"], last["closeness_raw"]
    if Cc <= 0:
        return 1.0
    w = L / (9 * Cc)
    mag = 10 ** math.floor(math.log10(w)) if w > 0 else 1
    return max(1.0, math.ceil(w / mag * 10) / 10 * mag)


def read_dial(model, dial, conds, seeds, strengths, base_R=None):
    """Sample and read a dial at each strength on the given canvases; returns per-strength readings and pictures."""
    out = {}
    for s in strengths:
        tm.set_active(model, {dial: s} if s else {})
        x = tm.sample(model, conds, seeds)
        out[s] = (x, tc.read(x))
    tm.set_active(model, {})
    return out


def cmd_dials(a):
    res = load_results().get("dials", {})
    window = load_results()["stage1"]["decoded_agreement"]["gradient_window_steps"]
    idx, cond = canvases()
    X = torch.from_numpy(idx).to(DEV)
    C = torch.from_numpy(cond).float().to(DEV)
    pal = tc.PALETTE.to(DEV)
    for dial in a.which:
        sync()
        t0 = time.time()
        close_idx = torch.from_numpy(np.nonzero(outside_purpose_mask(dial, cond))[0]).to(DEV)
        model = tm.load_base(base_path())
        info = {}
        if dial in ("count", "relation"):
            data = kept_draws(model, dial, a.kept_per)
            info["kept_draws"] = data[3]
            info["kept_total"] = len(data[0])
            data = data[:3]
            runner = lambda m, ups, w: train_kept(m, dial, data, ups, w, a.batch, a.lr, X, C, close_idx, pal)  # noqa: E731
            ups = a.kept_updates
        else:
            runner = lambda m, ups, w: train_gradient(m, dial, ups, w, a.grad_batch, a.lr, window, X, C, close_idx, pal)  # noqa: E731
            ups = a.grad_updates
        # smoke: 25 updates at closeness weight 1, to choose the weight
        tm.add_dial(model, dial, seed=1)
        smoke = runner(model, 25, 1.0)
        w = choose_close_weight(smoke)
        info["closeness_smoke"] = smoke[-1]
        info["closeness_weight"] = w
        model = tm.load_base(base_path())
        tm.add_dial(model, dial, seed=1)
        curve = runner(model, ups, w)
        info["updates"] = ups
        info["curve"] = curve
        torch.save(tm.dial_state(model, dial), dial_path(dial))
        # tuning read: training conditions, tuning seeds, strength 1
        conds, seeds = eval_set(tc.DIAL_CONDS[dial][0], n=512, seed0=tc.SEEDS["tuning"])
        rd = read_dial(model, dial, conds, seeds, (0.0, 1.0))
        hits = [tc.meets(r, b, c, {dial: 1.0}) for r, b, c in zip(rd[1.0][1], rd[0.0][1], conds)]
        info["tuning_meets_every_limit_s1"] = float(np.mean([bool(h) for h in hits]))
        info["tuning_base_exactly_asked"] = float(np.mean([tc.asked_exactly(r, c) for r, c in zip(rd[0.0][1], conds)]))
        sync()
        info["seconds"] = round(time.time() - t0, 1)
        res[dial] = info
        save_results("dials", res)
        print(dial, json.dumps({k: v for k, v in info.items() if k not in ("curve",)}), flush=True)
        print("  curve", [(c["update"], c["main"], c["closeness_share"]) for c in curve][:: max(1, len(curve) // 8)], flush=True)


# -------------------------------------------------------------------------------------------------------------------------------------
# Stage 2: each dial alone


def own_measure(dial, R, conds):
    if dial == "identity":
        return [tc.m_shape(r, c) for r, c in zip(R, conds)]
    if dial == "binding":
        return [tc.m_binding(r, c) for r, c in zip(R, conds)]
    if dial == "count":
        return [tc.m_count(r, c) for r, c in zip(R, conds)]
    return [tc.m_gap(r, c) for r, c in zip(R, conds)]


def measures(R, conds):
    g = [tc.m_gap(r, c) for r, c in zip(R, conds)]
    gd = [v for v in g if v is not None]
    return {"count (blue drawn minus asked, mean)": float(np.mean([tc.m_count(r, c) for r, c in zip(R, conds)])),
            "shape (share every object its asked shape)": float(np.mean([tc.m_shape(r, c) for r, c in zip(R, conds)])),
            "binding (share every asked combination drawn)": float(np.mean([tc.m_binding(r, c) for r, c in zip(R, conds)])),
            "gap (mean, canvas widths)": float(np.mean(gd)) if gd else None,
            "gap defined share": len(gd) / len(g)}


def unchanged(R, B, conds):
    """Per measurement, the share of canvases where it equals the frozen model's value on the same seed."""
    def eq(f):
        return float(np.mean([f(r, c) == f(b, c) for r, b, c in zip(R, B, conds)]))
    return {"count": eq(tc.m_count), "shape": eq(tc.m_shape), "binding": eq(tc.m_binding),
            "positions (every base object within 0.8 px)": float(np.mean([tc.positions_kept(r, b) for r, b in zip(R, B)])),
            "gap within 0.05": float(np.mean([(tc.m_gap(r, c) is not None and tc.m_gap(b, c) is not None and
                                               abs(tc.m_gap(r, c) - tc.m_gap(b, c)) <= 0.05) for r, b, c in zip(R, B, conds)]))}


def cmd_stage2(a):
    sync()
    t0 = time.time()
    states = load_dial_states(tc.DIALS)
    res = {}
    for dial in tc.DIALS:
        conds, seeds = eval_set(tc.DIAL_CONDS[dial][1])
        frozen = tm.load_base(base_path())
        xf = tm.sample(frozen, conds, seeds)
        Rf = tc.read(xf)
        model = tm.load_base(base_path(), dials=[dial], dial_states=states)
        rd = read_dial(model, dial, conds, seeds, STRENGTHS)
        rnd = tm.load_base(base_path(), dials=[dial], dial_states={dial: tm.random_like(states[dial], seed=7)})
        rr = read_dial(rnd, dial, conds, seeds, (1.0,))[1.0]
        info = {"s0_byte_identical_to_frozen": bool(torch.equal(rd[0.0][0], xf)), "rows": {}}
        own_means = []
        for s in STRENGTHS:
            x, R = rd[s]
            hit = [tc.meets(r, b, c, {dial: s}) for r, b, c in zip(R, Rf, conds)]
            om = [v for v in own_measure(dial, R, conds) if v is not None]
            own_means.append(float(np.mean(om)) if om else float("nan"))
            info["rows"][f"s={s}"] = {"meets_every_limit": None if any(h is None for h in hit) else float(np.mean(hit)),
                                      "own_measure_mean": own_means[-1], "measures": measures(R, conds),
                                      "unchanged_vs_frozen": unchanged(R, Rf, conds)}
        hits_by_s = {}
        for s in TRAINED_STRENGTHS[dial]:
            hits_by_s[s] = np.array([bool(tc.meets(r, b, c, {dial: s})) for r, b, c in zip(rd[s][1], Rf, conds)])
        rand_hit = np.array([bool(tc.meets(r, b, c, {dial: 1.0})) for r, b, c in zip(rr[1], Rf, conds)])
        om_r = [v for v in own_measure(dial, rr[1], conds) if v is not None]
        info["rows"]["random dial, s=1.0"] = {"meets_every_limit": float(rand_hit.mean()), "own_measure_mean": float(np.mean(om_r)) if om_r else None,
                                              "measures": measures(rr[1], conds), "unchanged_vs_frozen": unchanged(rr[1], Rf, conds)}
        mono = all(own_means[i + 1] > own_means[i] for i in range(2))       # 0 -> 0.5 -> 1
        vs_rand = mcnemar(hits_by_s[1.0], rand_hit)
        bars = {"hit": {str(s): {"share": float(h.mean()), "pass": bool(h.mean() >= R1_DIAL_HIT_MIN)} for s, h in hits_by_s.items()},
                "monotone_0_to_1": {"own_means": own_means, "pass": bool(mono)},
                "vs_random": {"lead": float(hits_by_s[1.0].mean() - rand_hit.mean()), **vs_rand,
                              "pass": bool(hits_by_s[1.0].mean() - rand_hit.mean() >= R1_DIAL_VS_RANDOM_MIN and vs_rand["p"] < P_MAX)}}
        if dial == "identity":
            both_d = np.array([tc.asked_exactly(r, c) for r, c in zip(rd[1.0][1], conds)])
            both_f = np.array([tc.asked_exactly(r, c) for r, c in zip(Rf, conds)])
            mc = mcnemar(both_d, both_f)
            bars["id_lead"] = {"dial_both_right": float(both_d.mean()), "frozen_both_right": float(both_f.mean()),
                               "lead": float(both_d.mean() - both_f.mean()), **mc,
                               "pass": bool(both_d.mean() - both_f.mean() >= R1_ID_LEAD_MIN and mc["p"] < P_MAX)}
        info["bars"] = bars
        info["pass"] = all(v["pass"] for k, v in bars.items() if k != "hit") and all(v["pass"] for v in bars["hit"].values())
        res[dial] = info
        rows = [xf] + [rd[s][0] for s in STRENGTHS] + [rr[0]]
        Rs = [Rf] + [rd[s][1] for s in STRENGTHS] + [rr[1]]
        grid(rows, ["frozen model"] + [f"{dial} dial, s={s}" for s in STRENGTHS] + ["random dial, s=1"],
             RESULTS / f"stage2-{dial}-alone.png", col_labels=[tc.name(c) for c in conds[:GRID_COLS]],
             calls=[[call_text(r) for r in R[:GRID_COLS]] for R in Rs])
        print(dial, json.dumps(bars), flush=True)
    sync()
    res["seconds"] = round(time.time() - t0, 1)
    res["pass"] = all(res[d]["pass"] for d in tc.DIALS)
    save_results("stage2", res)


# -------------------------------------------------------------------------------------------------------------------------------------
# Stage 3 and 4: pairs


def boot_mean_interference(per_pair, seed=0):
    """per_pair: list of (alone_both bool array, stack bool array, condition index array). Resample seeds within each condition of
    each pair, 10,000 times; the mean over pairs of (share both alone met - share stack met)."""
    rng = np.random.default_rng(seed)
    stats = np.zeros(BOOT)
    for ab, sk, ci in per_pair:
        groups = [np.nonzero(ci == g)[0] for g in np.unique(ci)]
        d = ab.astype(float) - sk.astype(float)
        tot = np.zeros(BOOT)
        for g in groups:
            draws = rng.integers(0, len(g), size=(BOOT, len(g)))
            tot += d[g][draws].sum(1)
        stats += tot / len(ab)
    stats /= len(per_pair)
    return [float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))]


def pair_name(p):
    return f"{p[0]} + {p[1]}"


def run_pair_rows(pair, conds, seeds, states, corr=None, static=None, layer_index=None):
    """Readings for one pair: frozen, each dial alone (a process loading only that dial), the plain stack, and the corrected stack."""
    out = {}
    frozen = tm.load_base(base_path())
    out["frozen"] = tm.sample(frozen, conds, seeds)
    for d in pair:
        m = tm.load_base(base_path(), dials=[d], dial_states=states)
        tm.set_active(m, {d: 1.0})
        out[d] = tm.sample(m, conds, seeds)
    m = tm.load_base(base_path(), dials=list(pair), dial_states=states)
    tm.set_active(m, {d: 1.0 for d in pair})
    out["plain stack"] = tm.sample(m, conds, seeds)
    if corr is not None:
        with torch.no_grad():
            colw = corr(static, {d: 1.0 for d in pair}, layer_index)
        tm.set_colw(m, colw)
        out["solo-behaviour correction"] = tm.sample(m, conds, seeds)
        tm.set_colw(m, {})
    return out


def cmd_stage3(a):
    sync()
    t0 = time.time()
    states = load_dial_states(tc.DIALS)
    res, per_pair = {"pairs": {}}, []
    for pair, (_, test) in tc.PAIRS.items():
        conds, seeds = eval_set(test)
        rows = run_pair_rows(pair, conds, seeds, states)
        R = {k: tc.read(v) for k, v in rows.items()}
        A = np.array([bool(tc.meets(r, b, c, {pair[0]: 1.0})) for r, b, c in zip(R[pair[0]], R["frozen"], conds)])
        B = np.array([bool(tc.meets(r, b, c, {pair[1]: 1.0})) for r, b, c in zip(R[pair[1]], R["frozen"], conds)])
        S = np.array([bool(tc.meets(r, b, c, {pair[0]: 1.0, pair[1]: 1.0})) for r, b, c in zip(R["plain stack"], R["frozen"], conds)])
        ci = np.array([test.index(c) for c in conds])
        per_pair.append((A & B, S, ci))
        res["pairs"][pair_name(pair)] = {
            "test_conditions": [tc.name(c) for c in test], "canvases": len(conds),
            "a_alone_meets": float(A.mean()), "b_alone_meets": float(B.mean()), "both_alone_meet_same_seed": float((A & B).mean()),
            "stack_meets": float(S.mean()), "interference": float((A & B).mean() - S.mean()),
            "interference_interval": boot_mean_interference([(A & B, S, ci)]),
            "product_form_printed_not_read": float(A.mean() * B.mean() - S.mean()),
            "stack_measures": measures(R["plain stack"], conds)}
        grid([rows[k] for k in ("frozen", pair[0], pair[1], "plain stack")], ["frozen model", f"{pair[0]} alone", f"{pair[1]} alone", "plain stack"],
             RESULTS / f"stage3-{pair[0]}-{pair[1]}.png", col_labels=[tc.name(c) for c in conds[:GRID_COLS]],
             calls=[[call_text(r) for r in R[k][:GRID_COLS]] for k in ("frozen", pair[0], pair[1], "plain stack")])
        torch.save({k: v for k, v in rows.items()}, OUT / f"stage3-{pair[0]}-{pair[1]}.pt")
        print(pair_name(pair), json.dumps({k: v for k, v in res["pairs"][pair_name(pair)].items() if k != "stack_measures"}), flush=True)
    mean_i = float(np.mean([v["interference"] for v in res["pairs"].values()]))
    ci_ = boot_mean_interference(per_pair)
    res["mean_interference"] = mean_i
    res["mean_interference_interval"] = ci_
    res["bar"] = R1_INTERFERENCE_SHOWN
    res["pass"] = bool(mean_i >= R1_INTERFERENCE_SHOWN and ci_[0] > 0)
    sync()
    res["seconds"] = round(time.time() - t0, 1)
    save_results("stage3", res)
    print("mean interference", mean_i, ci_, "pass", res["pass"])


def corr_setup(states):
    model = tm.load_base(base_path(), dials=list(tc.DIALS), dial_states=states)
    names = [n for n, _ in tm.layers(model)]
    layer_index = {n: i for i, n in enumerate(names)}
    static = {p: tm.Correction.static_features(model, list(p)) for p in tc.PAIRS}
    return model, layer_index, static


@torch.no_grad()
def state_bank(model, pair, conds, n_traj, seed0):
    """States the plain stack passes through: x before every Heun step and its sigma, on the pair's training conditions."""
    cs, ss = eval_set(conds, n=n_traj, seed0=seed0)
    cv = tc.cond_vector(cs).to(DEV)
    tm.set_active(model, {d: 1.0 for d in pair})
    xs, sig, cc = [], [], []
    t = st.sigmas(DEV)
    x = st.starting_noise(ss)
    for i in range(st.STEPS):
        xs.append(x.clone()); sig.append(torch.full((len(x),), float(t[i]), device=DEV)); cc.append(cv)
        x = tm.heun_range(tm.denoiser_for(model, cv), x, i, i + 1)
    tm.set_active(model, {})
    return torch.cat(xs), torch.cat(sig), torch.cat(cc)


def solo_loss(model, corr, static, layer_index, pair, x, s, cv, strengths, lam):
    with torch.no_grad():
        tm.set_colw(model, {})
        tm.set_active(model, {})
        D0 = model(x, s, cv)
        Ds = []
        for d in pair:
            tm.set_active(model, {d: strengths[d]})
            Ds.append(model(x, s, cv))
    colw = corr(static, strengths, layer_index) if corr is not None else {}
    tm.set_colw(model, colw)
    tm.set_active(model, strengths)
    Dc = model(x, s, cv)
    tm.set_colw(model, {})
    target = sum(Ds) - (len(Ds) - 1) * D0
    l = (st.loss_weight(s.view(-1, 1, 1, 1)) * (Dc - target) ** 2).mean()
    ov = tm.overlap_penalty(colw) if colw else torch.zeros((), device=DEV)
    return l, ov


def cmd_corr(a):
    sync()
    t0 = time.time()
    states = load_dial_states(tc.DIALS)
    model, layer_index, static = corr_setup(states)
    banks = {p: state_bank(model, p, tc.PAIRS[p][0], a.traj, tc.SEEDS["corr_states"]) for p in tc.PAIRS}
    tune = {p: state_bank(model, p, tc.PAIRS[p][0], 32, tc.SEEDS["corr_tuning"]) for p in tc.PAIRS}
    torch.manual_seed(0)
    corr = tm.Correction(len(layer_index)).to(DEV)
    opt = torch.optim.Adam(corr.parameters(), lr=a.lr)
    rng = random.Random(0)
    pairs = list(tc.PAIRS)

    def tuning_loss(c, weights=None):
        tot = 0.0
        for p in pairs:
            X, S, CV = tune[p]
            g = torch.Generator(device=DEV).manual_seed(1)
            k = torch.randint(len(X), (512,), device=DEV, generator=g)
            if weights is None:
                l, _ = solo_loss(model, c, static[p], layer_index, p, X[k], S[k], CV[k], {d: 1.0 for d in p}, 0.0)
            else:          # every column weight set to a constant (1 is the plain stack, 0 switches every dial off)
                cw = {n: {d: torch.full((m.in_dim,), float(weights), device=DEV) for d in p} for n, m in tm.layers(model)}
                with torch.no_grad():
                    tm.set_colw(model, {}); tm.set_active(model, {})
                    D0 = model(X[k], S[k], CV[k])
                    Ds = []
                    for d in p:
                        tm.set_active(model, {d: 1.0}); Ds.append(model(X[k], S[k], CV[k]))
                    tm.set_colw(model, cw); tm.set_active(model, {d: 1.0 for d in p})
                    Dc = model(X[k], S[k], CV[k]); tm.set_colw(model, {})
                l = (st.loss_weight(S[k].view(-1, 1, 1, 1)) * (Dc - (sum(Ds) - D0)) ** 2).mean()
            tot += float(l)
        return tot / len(pairs)
    smoke = {"plain_stack_weights_1": tuning_loss(None, 1.0), "weights_0": tuning_loss(None, 0.0), "start": tuning_loss(corr)}
    curve, best = [], (smoke["start"], 0, copy.deepcopy(corr.state_dict()))
    for u in range(1, a.updates + 1):
        p = pairs[rng.randrange(len(pairs))]
        X, S, CV = banks[p]
        k = torch.randint(len(X), (a.batch,), device=DEV)
        strengths = {d: rng.uniform(0.5, 1.0) for d in p}
        l, ov = solo_loss(model, corr, static[p], layer_index, p, X[k], S[k], CV[k], strengths, a.lam)
        loss = l + a.lam * ov
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if u % 250 == 0 or u == a.updates:
            tl = tuning_loss(corr)
            curve.append({"update": u, "train_solo": round(float(l), 6), "overlap": round(float(ov), 4), "tuning_solo": round(tl, 6)})
            print(curve[-1], flush=True)
            if tl < best[0]:
                best = (tl, u, copy.deepcopy(corr.state_dict()))
    corr.load_state_dict(best[2])
    torch.save(corr.state_dict(), OUT / "correction-solo.pt")
    smoke["after_training_chosen"] = best[0]
    sync()
    info = {"updates": a.updates, "batch": a.batch, "lr": a.lr, "lambda_overlap": a.lam, "trajectories_per_pair": a.traj,
            "chosen_update": best[1], "smoke_tuning_solo_loss": smoke, "curve": curve,
            "parameters": sum(p.numel() for p in corr.parameters()), "seconds": round(time.time() - t0, 1)}
    save_results("corr", info)
    print(json.dumps({k: v for k, v in info.items() if k != "curve"}))


def load_corr(states):
    model, layer_index, static = corr_setup(states)
    corr = tm.Correction(len(layer_index)).to(DEV)
    corr.load_state_dict(torch.load(OUT / "correction-solo.pt", map_location=DEV))
    corr.eval()
    return corr, layer_index, static


def cmd_stage4(a):
    sync()
    t0 = time.time()
    states = load_dial_states(tc.DIALS)
    corr, layer_index, static = load_corr(states)
    res = {"pairs": {}}
    P, Cc, picks = [], [], []
    for pair, (_, test) in tc.PAIRS.items():
        conds, seeds = eval_set(test)
        old = torch.load(OUT / f"stage3-{pair[0]}-{pair[1]}.pt")
        m = tm.load_base(base_path(), dials=list(pair), dial_states=states)
        tm.set_active(m, {d: 1.0 for d in pair})
        with torch.no_grad():
            colw = corr(static[pair], {d: 1.0 for d in pair}, layer_index)
        tm.set_colw(m, colw)
        xc = tm.sample(m, conds, seeds)
        mvals = torch.cat([w for res_ in colw.values() for w in res_.values()])
        Rf, Rp, Rc = tc.read(old["frozen"]), tc.read(old["plain stack"]), tc.read(xc)
        st_ = {d: 1.0 for d in pair}
        plain = np.array([bool(tc.meets(r, b, c, st_)) for r, b, c in zip(Rp, Rf, conds)])
        corr_ = np.array([bool(tc.meets(r, b, c, st_)) for r, b, c in zip(Rc, Rf, conds)])
        P.append(plain); Cc.append(corr_)
        res["pairs"][pair_name(pair)] = {"plain_stack_meets": float(plain.mean()), "solo_correction_meets": float(corr_.mean()),
                                         "lead": float(corr_.mean() - plain.mean()), **mcnemar(corr_, plain),
                                         "column_weights": {"mean": float(mvals.mean()), "min": float(mvals.min()), "max": float(mvals.max())},
                                         "correction_measures": measures(Rc, conds)}
        torch.save(xc, OUT / f"stage4-{pair[0]}-{pair[1]}-solo.pt")
        grid([old["frozen"], old[pair[0]], old[pair[1]], old["plain stack"], xc],
             ["frozen model", f"{pair[0]} alone", f"{pair[1]} alone", "plain stack", "solo-behaviour\ncorrection"],
             RESULTS / f"stage4-{pair[0]}-{pair[1]}.png", col_labels=[tc.name(c) for c in conds[:GRID_COLS]],
             calls=[[call_text(r) for r in R[:GRID_COLS]] for R in (Rf, tc.read(old[pair[0]]), tc.read(old[pair[1]]), Rp, Rc)])
        print(pair_name(pair), json.dumps({k: v for k, v in res["pairs"][pair_name(pair)].items() if k != "correction_measures"}), flush=True)
    P, Cc = np.concatenate(P), np.concatenate(Cc)
    mc = mcnemar(Cc, P)
    res["pooled"] = {"plain_stack_meets": float(P.mean()), "solo_correction_meets": float(Cc.mean()), "lead": float(Cc.mean() - P.mean()), **mc,
                     "pairs_ahead": int(sum(v["lead"] > 0 for v in res["pairs"].values())), "bar": SOLO_VS_PLAIN_MIN,
                     "pass": bool(Cc.mean() - P.mean() >= SOLO_VS_PLAIN_MIN and mc["p"] < P_MAX)}
    sync()
    res["seconds"] = round(time.time() - t0, 1)
    save_results("stage4", res)
    print(json.dumps(res["pooled"]))


def cmd_stage5(a):
    sync()
    t0 = time.time()
    pair = ("identity", "count")
    conds, seeds = eval_set(tc.DECIDING)
    assert tc.DECIDING == tc.PAIRS[pair][1]
    old = torch.load(OUT / "stage3-identity-count.pt")
    xc = torch.load(OUT / "stage4-identity-count-solo.pt")
    rows = {"frozen model": old["frozen"], "count dial alone": old["count"], "identity dial alone": old["identity"],
            "plain stack": old["plain stack"], "solo-behaviour correction": xc}
    R = {k: tc.read(v) for k, v in rows.items()}
    res = {"conditions": [tc.name(c) for c in tc.DECIDING], "canvases": len(conds), "rows": {}}
    ok = {}
    for k, Rk in R.items():
        succ = np.array([tc.asked_exactly(r, c, (tc.asked_blue(c),)) for r, c in zip(Rk, conds)])
        n3 = [r for r in Rk if len(r["objects"]) == 3 and not r["specks"]]
        blues_diamond = float(np.mean([all(s == 1 for s, col, *_ in r["objects"] if col == tc.BLUE) and
                                       sum(col == tc.BLUE for _, col, *_ in r["objects"]) == 2 for r in n3])) if n3 else None
        square_right = float(np.mean([any(s == 0 and col == tc.colour_of(c[0]) for s, col, *_ in r["objects"])
                                      for r, c in zip(Rk, conds) if len(r["objects"]) == 3 and not r["specks"]])) if n3 else None
        ok[k] = succ
        res["rows"][k] = {"deciding_share (exactly asked + one blue diamond, every object right)": float(succ.mean()),
                          "share_three_objects": len(n3) / len(Rk),
                          "of_three_object_canvases: both blue objects diamonds": blues_diamond,
                          "of_three_object_canvases: the asked square is a square": square_right,
                          "asked_exactly_without_added (two objects, both right)": float(np.mean([tc.asked_exactly(r, c) for r, c in zip(Rk, conds)]))}
    s3 = load_results()["stage3"]["pairs"]["identity + count"]
    mc = mcnemar(ok["solo-behaviour correction"], ok["plain stack"])
    res["bars"] = {"interference_on_the_deciding_pair": {"value": s3["interference"], "interval": s3["interference_interval"],
                                                        "bar": DECIDE_INTERFERENCE_MIN,
                                                        "pass": bool(s3["interference"] >= DECIDE_INTERFERENCE_MIN and s3["interference_interval"][0] > 0)},
                   "solo_correction_over_plain_stack": {"lead": float(ok["solo-behaviour correction"].mean() - ok["plain stack"].mean()), **mc,
                                                       "bar": DECIDE_SOLO_LEAD_MIN,
                                                       "pass": bool(ok["solo-behaviour correction"].mean() - ok["plain stack"].mean() >= DECIDE_SOLO_LEAD_MIN and mc["p"] < P_MAX)}}
    grid(list(rows.values()), list(rows.keys()), RESULTS / "stage5-deciding-test.png", col_labels=[tc.name(c) for c in conds[:GRID_COLS]],
         calls=[[call_text(r) for r in Rk[:GRID_COLS]] for Rk in R.values()])
    sync()
    res["seconds"] = round(time.time() - t0, 1)
    save_results("stage5", res)
    print(json.dumps(res, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("shapes")
    q.add_argument("--n", type=int, default=200_000)
    q = sub.add_parser("base")
    q.add_argument("--updates", type=int, default=6_000)
    q.add_argument("--batch", type=int, default=256)
    q.add_argument("--lr", type=float, default=1e-3)
    q.add_argument("--ema", type=float, default=0.999)
    for nm, key in (("stage1", "stage1"), ("misses", "stage1_misses")):
        q = sub.add_parser(nm)
        q.add_argument("--test-seed0", type=int, default=tc.SEEDS["test"])
        q.add_argument("--tune-seed0", type=int, default=tc.SEEDS["tuning"])
        q.add_argument("--key", default=key, help="results.json section to write")
        q.add_argument("--suffix", default="", help="appended to the picture file name")
    q = sub.add_parser("dials")
    q.add_argument("--which", nargs="+", default=list(tc.DIALS))
    q.add_argument("--kept-per", type=int, default=2048)
    q.add_argument("--kept-updates", type=int, default=1500)
    q.add_argument("--grad-updates", type=int, default=300)
    q.add_argument("--batch", type=int, default=128)
    q.add_argument("--grad-batch", type=int, default=64)
    q.add_argument("--lr", type=float, default=1e-3)
    sub.add_parser("stage2")
    sub.add_parser("stage3")
    q = sub.add_parser("corr")
    q.add_argument("--updates", type=int, default=1500)
    q.add_argument("--batch", type=int, default=128)
    q.add_argument("--lr", type=float, default=1e-3)
    q.add_argument("--lam", type=float, default=0.01)
    q.add_argument("--traj", type=int, default=128)
    sub.add_parser("stage4")
    sub.add_parser("stage5")
    a = p.parse_args()
    if a.cmd not in ("shapes",) and gpu_seconds_so_far() > GPU_MINUTES_STOP * 60:
        sys.exit(f"stopped: {gpu_seconds_so_far() / 60:.1f} GPU-minutes already used, past the {GPU_MINUTES_STOP}-minute stop")
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
