"""Every number the MNIST-stage figures print, read from the stage's code, checkpoints, samples and
result files, never typed. The figures (make_mnist_stage_figures.py) read mnist_stage_numbers.json
and the arrays in mnist_stage_arrays.npz.

Sources, read only:
  poe_repair/experiments/inclusive_captions/mnist/{canvases,model,train,sample,score,bar,judge_checks}.py
  data/mnist/checkpoints/dial-*-seed-*.pt and trainings.jsonl
  data/mnist/samples/dial-*-seed-*.npz
  data/mnist/judge/judge.pt and src/judge.py (the pinned judge)
  4-inclusive-captions/artifacts/results/do-inclusive-labels-help-the-product-compose-on-mnist/{counts.json,scores.csv}

    CUDA_VISIBLE_DEVICES= /home/molef/micromamba/envs/sdxl-local/bin/python -I make_mnist_stage_numbers.py
"""
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
MN = REPO / "poe_repair/experiments/inclusive_captions/mnist"
RES = REPO / "4-inclusive-captions/artifacts/results/do-inclusive-labels-help-the-product-compose-on-mnist"
sys.path.insert(0, str(MN))

import numpy as np
import torch
import bar
import canvases as cv
import model as md
import sample as sm
import score as sc
import train as tr

out = {"sources": {"code": str(MN.relative_to(REPO)), "results": str(RES.relative_to(REPO))}}
arrays = {}

# 1. The canvases and the label rule (canvases.py)
_, y = cv.load_mnist("train")
p = np.bincount(y, minlength=10) / len(y)
two_with = {d: [pr for pr in cv.SEEN if d in pr] for d in range(10)}
share_per_other = cv.TWO_SHARE * (1 - cv.SEVEN_EIGHT_SHARE) / len(cv.OTHER_SEEN)


def two_digit_share(d, dial):
    """Of the non-blank training draws labelled with the single digit d, the share whose canvas holds two digits."""
    one = (1 - cv.TWO_SHARE) * p[d]
    two = sum(cv.TWO_SHARE * cv.SEVEN_EIGHT_SHARE if pr == (7, 8) else share_per_other for pr in two_with[d])
    w = dial / (1 + 2 * dial)
    return float(two * w / (one + two * w))


out["canvases"] = {
    "size": [cv.H, cv.W], "digit box": cv.D, "pixel range": [-1, 1],
    "share of canvases with two digits": cv.TWO_SHARE,
    "share of two-digit canvases holding 7 and 8": cv.SEVEN_EIGHT_SHARE,
    "other seen pairs": len(cv.OTHER_SEEN), "held-out pairs, never on one canvas": cv.HELD_OUT,
    "two-digit placement": "one digit per half (columns 0 and 32, plus 0 to 4 pixels of jitter), order swapped half the time",
    "one-digit placement": "uniformly random place; digit drawn at MNIST's own frequency",
    "labels": cv.N_LABELS, "blank id": cv.BLANK, "blank rate": cv.BLANK_RATE,
    "label rule for a two-digit canvas holding a and b": ("blank on 10% of draws; otherwise one label per draw, "
                                                          "'a and b' : 'a' : 'b' in weights 1 : dial : dial"),
    "MNIST train frequency of 7": round(float(p[7]), 4), "MNIST train frequency of 8": round(float(p[8]), 4),
    "by dial": {f"{d:g}": {
        "joint share of a 7-and-8 canvas's non-blank draws": round(cv.design_shares(d)["joint"], 4),
        "'7' share of a 7-and-8 canvas's non-blank draws": round(d / (1 + 2 * d), 4),
        "share of '7' draws holding two digits": round(two_digit_share(7, d), 4),
        "share of '8' draws holding two digits": round(two_digit_share(8, d), 4),
        "score.py training line for '7' (percent)": sc.TRAINING_LINE["7"][i],
    } for i, d in enumerate(cv.DIALS)},
}
out["canvases"]["analytic '7' one-digit share (percent) against score.py's training line"] = {
    f"{d:g}": [round(100 * (1 - two_digit_share(7, d)), 2), sc.TRAINING_LINE["7"][i]] for i, d in enumerate(cv.DIALS)}

# 2. The network (model.py) and its training (train.py, trainings.jsonl)
rows = [json.loads(l) for l in open(REPO / "data/mnist/checkpoints/trainings.jsonl")]
width = rows[0]["width"]
assert all(r["width"] == width and r["steps"] == rows[0]["steps"] for r in rows)
net = md.UNet(width)
n_res = sum(isinstance(m, md.ResBlock) for m in net.modules())
out["network"] = {
    "width c": width, "levels": [[cv.H, cv.W, width], [cv.H // 2, cv.W // 2, 2 * width], [cv.H // 4, cv.W // 4, 4 * width]],
    "residual blocks": n_res, "attention blocks": sum(isinstance(m, md.Attention) for m in net.modules()),
    "parameters": int(sum(q.numel() for q in net.parameters())),
    "label table": list(net.lemb.weight.shape), "label table parameters": int(net.lemb.weight.numel()),
    "step embedding width": int(net.temb[-1].out_features),
    "how the label enters": "e = temb(step) + label_table[label]; every residual block adds Linear(e) to its channels after its first conv",
    "predicts": "the noise (train.py: loss = mean (pred - eps)^2)",
    "schedule": {"steps": md.T, "kind": "cosine, s = 0.008, clipped to [1e-5, 0.9999]"},
    "training": {"steps": rows[0]["steps"], "batch": tr.BATCH, "learning rate": tr.LR, "EMA": tr.EMA,
                 "trainings": len(rows), "final losses": [r["final_loss"] for r in rows],
                 "train.py default steps (not the one used)": tr.STEPS},
}

# 3. The sampler (sample.py)
src = (MN / "sample.py").read_text().splitlines()
line = next(i + 1 for i, l in enumerate(src) if "GUIDE * (e[1:] - e[0]).sum(0)" in l)
out["sampler"] = {"guidance": sm.GUIDE, "seeds": sm.N_SEEDS, "corrector steps per level": sm.CORR_STEPS,
                  "corrector step size": f"{sm.CORR_SIZE} * (1 - abar)", "product line": f"sample.py:{line}",
                  "product line text": src[line - 1].strip(), "rows": [r[0] for r in sm.ROWS]}

# 4. The bar (bar.py) and the verdict (counts.json)
cj = json.load(open(RES / "counts.json"))
V = cj["verdict"]
out["bar"] = {"purity floor": {k: {f"{d:g}": v for d, v in f.items()} for k, f in bar.PURITY_FLOOR.items()},
              "support min (judge, of 256)": bar.SUPPORT_MIN, "dial-0 max": bar.DIAL0_MAX,
              "null if mean below (judge, of 256)": bar.NULL_MAX_MEAN, "unseen min": bar.UNSEEN_MIN,
              "eye seeds": bar.EYE_SEEDS, "eye support min": bar.EYE_SUPPORT_MIN, "eye null mean": bar.EYE_NULL_MAX_MEAN}
out["verdict"] = {k: V[k] for k in ("verdict", "models_scored", "complete", "supported_settings", "judge")}
out["verdict"]["settings"] = V["settings"]
out["verdict"]["counting settings"] = [d for d, s in V["settings"].items() if s["counts"]]
out["verdict"]["mean corrected 7 x 8 at counting settings"] = {
    d: round(float(np.mean(V["settings"][d]["corrected_7x8_two"])), 2) for d in out["verdict"]["counting settings"]}

# 5. Per-cell counts and a tile picked by rule: the first seed whose call is the cell's most common call
calls = {}
with open(RES / "scores.csv") as f:
    for r in csv.DictReader(f):
        calls.setdefault((float(r["dial"]), int(r["training_seed"]), r["row"]), []).append(r["call"])
cells = {}
for (d, s, row), cs in calls.items():
    kind = sc.ROWS[row][0]
    target = sc.ALONE if kind == "single" else sc.TWO
    cnt = Counter(cs)
    major = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    key = f"{d:g}|{s}|{row}"
    assert dict(cnt) == cj["counts"][key], key
    cells[key] = {"count": cnt.get(target, 0), "counted call": target, "majority call": major,
                  "tile seed": cs.index(major), "calls": dict(cnt)}
out["cells"] = cells

# 6. Eye calls on the corrected 7 x 8 row (the plan's verdict reads these)
EYE = REPO / "4-inclusive-captions/artifacts/drips/inclusive-captions-mnist-stage/eye-calls"
eye = {}
for f_ in sorted(EYE.glob("dial-*-seed-*-7x8-corrected.csv")):
    rs = list(csv.DictReader(open(f_)))
    eye[f_.stem] = sum(r["eye_call"] == sc.TWO for r in rs)
out["eye calls, corrected 7 x 8, first 64 seeds"] = eye

# 7. One sampler step from the dial-1, training-1 model on CPU, for the step figure
ck = torch.load(REPO / "data/mnist/checkpoints/dial-1-seed-1.pt", map_location="cpu", weights_only=False)
net = md.UNet(ck["width"]).eval(); net.load_state_dict(ck["state"])
abar = md.make_abar()
K = 100
cell = cells["1|1|7 x 8 corrected"]
x0 = torch.tensor(np.load(REPO / "data/mnist/samples/dial-1-seed-1.npz")["7 x 8 corrected"][cell["tile seed"]],
                  dtype=torch.float32)[None, None] / 127.5 - 1
z = torch.randn(x0.shape, generator=torch.Generator().manual_seed(0))
xk = abar[K].sqrt() * x0 + (1 - abar[K]).sqrt() * z
with torch.no_grad():
    kk = torch.full((3,), K, dtype=torch.long)
    e = net(xk.repeat(3, 1, 1, 1), kk, torch.tensor([cv.BLANK, 7, 8]))
    eb, e7, e8 = e[0:1], e[1:2], e[2:3]
    ep = eb + sm.GUIDE * ((e7 - eb) + (e8 - eb))
    a_t, a_p = abar[K], abar[K - 1]
    al = a_t / a_p; be = 1 - al
    mean = (xk - be / (1 - a_t).sqrt() * ep) / al.sqrt()
    x_next = mean + (be * (1 - a_p) / (1 - a_t)).sqrt() * torch.randn(xk.shape, generator=torch.Generator().manual_seed(1))
    j = K - 1
    eta = sm.CORR_SIZE * (1 - abar[j])
    xc, g = x_next.clone(), torch.Generator().manual_seed(2)
    for _ in range(sm.CORR_STEPS):
        ej = net(xc.repeat(3, 1, 1, 1), torch.full((3,), j, dtype=torch.long), torch.tensor([cv.BLANK, 7, 8]))
        epj = ej[0:1] + sm.GUIDE * ((ej[1:2] - ej[0:1]) + (ej[2:3] - ej[0:1]))
        xc = xc + eta * (-epj / (1 - abar[j]).sqrt()) + (2 * eta).sqrt() * torch.randn(xc.shape, generator=g)


def u8(t):
    return ((t.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)[0, 0].numpy()


def u8e(t):
    v = t[0, 0].numpy()
    return np.clip(127.5 + 127.5 * v / (np.abs(v).max() or 1), 0, 255).astype(np.uint8)


for k_, v in (("x0", x0), ("x_k", xk), ("x_k-1", x_next), ("x_k-1 corrected", xc)):
    arrays[f"step/{k_}"] = u8(v)
for k_, v in (("eps blank", eb), ("eps 7", e7), ("eps 8", e8), ("eps product", ep)):
    arrays[f"step/{k_}"] = u8e(v)
out["one step (dial 1, training 1, CPU)"] = {
    "k": K, "abar": round(float(abar[K]), 4), "start": "the corrected 7 x 8 tile picked in r1 (seed "
                                                       f"{cell['tile seed']}), noised to step {K} with seed 0",
    "norms": {"eps blank": round(float(eb.norm()), 2), "eps 7 - eps blank": round(float((e7 - eb).norm()), 2),
              "eps 8 - eps blank": round(float((e8 - eb).norm()), 2), "eps product": round(float(ep.norm()), 2)},
    "cosine between the two label differences": round(float(((e7 - eb) * (e8 - eb)).sum() / (e7 - eb).norm() / (e8 - eb).norm()), 3),
    "corrector eta at step 99": round(float(eta), 5),
    "illustrative": "the tensors are real; the canvas starts from a finished tile noised by hand, not from a chain",
}

# 8. The judge (pinned copy, data/mnist/judge)
JD = REPO / "data/mnist/judge"
sys.path.insert(0, str(JD / "src"))
import judge as jg
jn = jg.judge_net()
jn.load_state_dict(torch.load(JD / "judge.pt", map_location="cpu"))
sizes, x = [], torch.zeros(1, 1, 28, 28)
with torch.no_grad():
    for layer in jn:
        x = layer(x)
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear, torch.nn.Flatten)):
            sizes.append({"layer": layer.__class__.__name__, "output": list(x.shape[1:])})
import judge_checks as jc
out["judge"] = {"sha256": hashlib.sha256((JD / "judge.pt").read_bytes()).hexdigest(),
                "matches counts.json": hashlib.sha256((JD / "judge.pt").read_bytes()).hexdigest() == V["judge"]["weights_sha256"],
                "parameters": int(sum(q.numel() for q in jn.parameters())), "layers": sizes,
                "classes": 11, "two digits on one spot": jg.TWO_ON_ONE_SPOT, "confident": jg.CONFIDENT,
                "split width": jg.SPLIT_WIDTH, "min blob": 25, "crop": 28,
                "pair calls (ladder)": jc.PAIR_CALLS, "single calls": jc.SINGLE_CALLS,
                "the stage's change": V["judge"]["change"],
                "single-label rows read ten classes only": True}

# 9. The share each setting is named by, from canvases.py: of the "7"-labelled (non-blank) training draws,
#    the share whose canvas also holds an 8, and the share holding any second digit
def share_with(d, dial, other=None):
    one = (1 - cv.TWO_SHARE) * p[d]
    w = dial / (1 + 2 * dial)
    two_all = sum(cv.TWO_SHARE * cv.SEVEN_EIGHT_SHARE if pr == (7, 8) else share_per_other for pr in two_with[d])
    two_sel = two_all if other is None else sum(
        cv.TWO_SHARE * cv.SEVEN_EIGHT_SHARE if pr == (7, 8) else share_per_other for pr in two_with[d] if other in pr)
    return float(two_sel * w / (one + two_all * w))


shares = {}
for d in cv.DIALS:
    s8 = share_with(7, d, 8)
    shares[f"{d:g}"] = {"share of '7' pictures that also show an 8": round(s8, 4),
                        "share of '7' pictures that also show any second digit": round(share_with(7, d), 4),
                        "outlined of 40": int(round(s8 * 40)),
                        "joint share of a 7-and-8 canvas's labelled draws": round(1 / (1 + 2 * d), 4),
                        "'7' share of a 7-and-8 canvas's labelled draws": round(d / (1 + 2 * d), 4)}
out["shares"] = shares

# 9b. What training saw, per model: expected label counts over the whole schedule, from canvases.py's rule
#     (canvases are built fresh every batch, so a model sees steps x batch canvases, none twice)
NDRAWS = rows[0]["steps"] * tr.BATCH
for d in cv.DIALS:
    w = d / (1 + 2 * d)
    nb = 1 - cv.BLANK_RATE
    two7 = sum(cv.TWO_SHARE * cv.SEVEN_EIGHT_SHARE if pr == (7, 8) else share_per_other for pr in two_with[7])
    two8 = sum(cv.TWO_SHARE * cv.SEVEN_EIGHT_SHARE if pr == (7, 8) else share_per_other for pr in two_with[8])
    s78 = cv.TWO_SHARE * cv.SEVEN_EIGHT_SHARE
    shares[f"{d:g}"]["expected draws per model"] = {
        "canvases": NDRAWS,
        '"7"': round(NDRAWS * nb * ((1 - cv.TWO_SHARE) * p[7] + two7 * w)),
        '"7" whose canvas also holds an 8': round(NDRAWS * nb * s78 * w),
        '"8"': round(NDRAWS * nb * ((1 - cv.TWO_SHARE) * p[8] + two8 * w)),
        '"7 and 8"': round(NDRAWS * nb * s78 / (1 + 2 * d)),
        "blank": round(NDRAWS * cv.BLANK_RATE)}
out["what training saw"] = {"canvases per model": NDRAWS, "built": "fresh every batch (canvases.py, CanvasMaker.batch)",
                            "from": f"MNIST's {len(y):,} training digits", "counts": "expected, from canvases.py's rule"}

# 10. r1's header grids: 40 real "7"-labelled canvases per setting, drawn with canvases.py; the first
#     `outlined` of them hold a 7 and an 8, the rest do not (the grid is composed to match the share)
grids = {}
for d in cv.DIALS:
    cm = cv.CanvasMaker(d, "cpu", seed=7)
    with_8, without = [], []
    while len(without) < 40 or (d > 0 and len(with_8) < 40):
        x, lab, holds = cm.batch(4000)
        is7 = lab == 7
        has8 = ((holds[:, 0] == 8) | (holds[:, 1] == 8)) & (holds[:, 1] >= 0)
        for i in torch.nonzero(is7 & has8).flatten().tolist():
            with_8.append(x[i, 0])
        for i in torch.nonzero(is7 & (holds[:, 1] < 0)).flatten().tolist():
            without.append(x[i, 0])
        if d == 0:
            break
    k = shares[f"{d:g}"]["outlined of 40"]
    pics = with_8[:k] + without[:40 - k]
    order = np.random.default_rng(0).permutation(40)
    arrays[f"grid/{d:g}"] = np.stack([((pics[i].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).numpy() for i in order])
    grids[f"{d:g}"] = {"outlined positions": [int(j) for j, i in enumerate(order) if i < k],
                       "'7' canvases holding an 8 seen while drawing": len(with_8)}
out["r1 header grids"] = grids

# 11. r6: per setting, the plain and corrected 7 x 8 calls of training 1, and the first seed with each call
r6 = {}
for d in cv.DIALS:
    row = {}
    for kind in ("7 x 8 plain", "7 x 8 corrected"):
        cs = calls[(float(d), 1, kind)]
        row[kind] = {c: {"count": cs.count(c), "first seed": cs.index(c) if c in cs else None} for c in jc.PAIR_CALLS}
    row["7 and 8, three trainings"] = [cells[f"{d:g}|{s}|7 and 8"]["count"] for s in (1, 2, 3)]
    pl = calls[(float(d), 1, "7 x 8 plain")]
    row["first 8 seeds, plain"] = pl[:8]
    row["first three seeds per call, plain"] = {c: [i for i, x in enumerate(pl) if x == c][:3] for c in jc.PAIR_CALLS}
    row["7 x 8 plain, three trainings"] = [cells[f"{d:g}|{s}|7 x 8 plain"]["count"] for s in (1, 2, 3)]
    r6[f"{d:g}"] = row
out["r6 calls, training 1"] = r6

# 12. r5's trajectories (written by make_r5_trajectories.py on the GPU)
R5 = HERE / "r5_trajectories.json"
if R5.exists():
    out["r5 trajectories"] = json.load(open(R5))
    sd = out["r5 trajectories"]["seed"]
    out["r5 calls"] = {f"{d:g}": calls[(float(d), 1, "7 x 8 plain")][sd] for d in cv.DIALS}
    # the recaptured chain's own finished canvas, called by the same judge (it differs from the stored sample)
    R5A = np.load(HERE / "r5_trajectories.npz")
    jj, jnet, _ = jc.pinned_judge()
    out["r5 calls of the recaptured canvas"] = {f"{d:g}": jc.pair_call(jj, jnet, R5A[f"{d:g}/k0"], (7, 8)) for d in cv.DIALS}

np.savez_compressed(HERE / "mnist_stage_arrays.npz", **arrays)
json.dump(out, open(HERE / "mnist_stage_numbers.json", "w"), indent=1)
print("wrote", HERE / "mnist_stage_numbers.json")
