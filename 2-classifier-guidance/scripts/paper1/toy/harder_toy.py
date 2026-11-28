#!/usr/bin/env python3
"""A harder squares toy, built from squares_toy.py's code, where Method A is not saturated, for the loss ladder (loss_ladder.py).

What changes from the toy. 32 by 32 canvases (16 by 16 there), squares of side 6 (4), 27 by 27 = 729 positions per colour (169),
and deeper occlusion: touching pairs either share an edge with no gap, or overlap by 1, 2 or 3 pixels (1 or 2 there) with the depth
order drawn both ways (red in front, its pixels win the overlap, or blue in front). 131,724 joint scenes (12,588 there). New red, blue
and empty experts are trained here with the toy's recipe; the networks are the toy's (the U-Net's coarsest map is now 8 by 8, so
one 3 by 3 convolution no longer sees the whole picture). Everything else (sampler, seeds, metrics, Method A, the corrector) is the
toy's own code, run with squares_toy's H and SIDE set to 32 and 6.

The bars, committed here before the first evaluation run on this toy (same form as the toy's):
  WORKS    clean touching share on the 256 shared seeds at least WORKS_MIN and at least WORKS_ABOVE above the neural plain sum's
  PARTIAL  at least PARTIAL_ABOVE above the neural plain sum's, short of WORKS
  FAILS    otherwise
  Rung 0 must pass before any later step runs: the exact joint's clean share at least RUNG0_EXACT_JOINT_MIN (the instrument can
  see a clean pair) and the exact plain sum FAILS against the bar measured from 0 (clean share below PARTIAL_ABOVE), so the toy
  still tests composition.

Outputs under /datasets/mmolefe/poe_repair_min/toys/paper1-harder/ (results.json, checkpoints, samples, pictures).

  harder_toy.py sets | rung0 | experts | rung1 | curve --ns 256 1024 4096
  then loss_ladder.py ladder --world harder --n <N> --runs ...
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("SQUARES_TOY_SRC", str(HERE)))
import squares_toy as st  # noqa: E402
import loss_ladder as ll  # noqa: E402

ROOT = Path(os.environ.get("HARDER_TOY_ROOT", "/datasets/mmolefe/poe_repair_min/toys/paper1-harder"))
H, SIDE, OVERLAPS = 32, 6, (1, 2, 3)

# --- the bars (constants; any change shows in a diff) ---
WORKS_MIN = 0.8
WORKS_ABOVE = 0.5
PARTIAL_ABOVE = 0.2
RUNG0_EXACT_JOINT_MIN = 0.95
BARS = (f"Harder toy (32 by 32, side 6, overlaps 1 to 3 with depth order). On 256 shared seeds, a method WORKS if its clean-pair "
        f"share is at least {WORKS_MIN} and at least {WORKS_ABOVE} above the neural plain sum's; PARTIAL if at least {PARTIAL_ABOVE} "
        f"above it but short of WORKS; FAILS otherwise. Rung 0 passes only if the exact joint reads at least {RUNG0_EXACT_JOINT_MIN} "
        f"and the exact plain sum FAILS (below {PARTIAL_ABOVE}).")

EXPERT_UPDATES = 6000              # the toy's experts: 6,000 updates for red and blue, a quarter for empty

st.H, st.SIDE = H, SIDE
st.OUT = st.RESULTS = ROOT


def verdict(clean, base):
    if clean >= WORKS_MIN and clean - base >= WORKS_ABOVE:
        return "WORKS"
    if clean - base >= PARTIAL_ABOVE:
        return "PARTIAL"
    return "FAILS"


def joint_set():
    """(pictures, metadata) in squares_toy.joint_set's layout; kind is the overlap depth (0 for a shared edge)."""
    pics, meta = [], []
    for rr, rc in st.positions():
        for br, bc in st.positions():
            dr, dc = abs(rr - br), abs(rc - bc)
            hi, lo = max(dr, dc), min(dr, dc)
            if hi == SIDE and lo < SIDE:
                img = st._blank(1)[0]
                st._paint(img, rr, rc, st.RED)
                st._paint(img, br, bc, st.BLUE)
                pics.append(img)
                meta.append((rr, rc, br, bc, 0, 0))
            elif SIDE - hi in OVERLAPS:
                for front in (1, 2):
                    img = st._blank(1)[0]
                    if front == 1:
                        st._paint(img, br, bc, st.BLUE)
                        st._paint(img, rr, rc, st.RED)
                    else:
                        st._paint(img, rr, rc, st.RED)
                        st._paint(img, br, bc, st.BLUE)
                    pics.append(img)
                    meta.append((rr, rc, br, bc, SIDE - hi, front))
    pics = torch.stack(pics)
    assert len(torch.unique(pics.flatten(1), dim=0)) == len(pics), "two touching configurations drew the same picture"
    return pics, np.array(meta)


def sets():
    """The four sets, cached at ROOT/sets.pt after the first build."""
    p = ROOT / "sets.pt"
    if p.exists():
        d = torch.load(p, weights_only=False)   # this script's own file; meta is a numpy array
        return d["e"], d["r"], d["b"], d["j"], d["meta"]
    e, r, b = st.empty_set(), st.single_set(st.RED), st.single_set(st.BLUE)
    j, meta = joint_set()
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.save({"e": e, "r": r, "b": b, "j": j, "meta": meta}, p)
    return e, r, b, j, meta


def split(n):
    """squares_toy.joint_split's rule: a fixed permutation (seed 0), every tenth scene held out."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).tolist()
    return [perm[i] for i in range(n) if i % st.VAL_NTH != 0], [perm[i] for i in range(0, n, st.VAL_NTH)]


def exact_all():
    e, r, b, j, meta = sets()
    return (e, r, b, j, meta), (st.Exact(e), st.Exact(r), st.Exact(b), st.Exact(j))


def results():
    p = ROOT / "results.json"
    return json.loads(p.read_text()) if p.exists() else {}


def w_ns():
    return SimpleNamespace(results=ROOT / "results.json", bars=BARS)


def save(key, value):
    ll.save_result(w_ns(), key, value)


def require(key, paths):
    ll.require_new(w_ns(), key, paths)


def rung0_passed():
    r = results().get("rung0")
    if not r or not r.get("passed"):
        raise SystemExit("rung 0 has not passed on the harder toy; run rung0 first")


def make_world():
    rung0_passed()
    (e, r, b, j, meta), ex = exact_all()
    tr, va = split(len(j))
    return SimpleNamespace(name="harder", h=H, side=SIDE, e=e, r=r, b=b, j=j, meta=meta, ex=ex, tr=tr, va=va,
                           nets=st.load_experts(), root=ROOT, results=ROOT / "results.json", ckpt=ROOT / "loss-ladder",
                           bars=BARS, verdict=verdict, grid_scale=4, eval_seeds=st.EVAL_SEEDS, tuning_seeds=ll.TUNING_SEEDS,
                           base_x=st.load_samples("neural-plainsum"), exact_joint_x=st.load_samples("exact-joint"))


# -------------------------------------------------------------------------------------------------------------------------------------


def cmd_sets(a):
    e, r, b, j, meta = sets()
    kinds = {f"overlap {k}" if k else "edge": int((meta[:, 4] == k).sum()) for k in (0,) + OVERLAPS}
    tr, va = split(len(j))
    info = {"empty": len(e), "red": len(r), "blue": len(b), "joint": len(j), "joint_by_kind": kinds, "joint_train_pool": len(tr),
            "joint_validation": len(va), "side": SIDE, "size": H}
    print(json.dumps(info, indent=1))
    if "sets" not in results():
        save("sets", info)


def cmd_rung0(a):
    t0 = time.time()
    require("rung0", [ROOT / "samples-exact-joint.pt", ROOT / "samples-exact-plainsum.pt", ROOT / "rung0-exact.png"])
    (e, r, b, j, meta), (Ee, Er, Eb, Ej) = exact_all()
    xj = st.sample(st.joint_denoiser(Ej))
    xp = st.sample(st.plainsum_denoiser(Er, Eb, Ee))
    st.save_samples("exact-joint", xj)
    st.save_samples("exact-plainsum", xp)
    mj, mp = st.metrics(xj, j), st.metrics(xp, j)
    mp["verdict_from_0"] = verdict(mp["clean_share"], 0.0)
    passed = mj["clean_share"] >= RUNG0_EXACT_JOINT_MIN and mp["verdict_from_0"] == "FAILS"
    st.grid([xj, xp], ["exact joint", "exact plain sum"], ROOT / "rung0-exact.png", scale=4)
    print(st.table([("exact joint", mj), ("exact plain sum", mp)]), "\nrung 0 passed:", passed, flush=True)
    save("rung0", {"rows": {"exact joint": mj, "exact plain sum": mp}, "passed": passed,
                   "check": f"exact joint >= {RUNG0_EXACT_JOINT_MIN} and exact plain sum FAILS from 0",
                   "picture": str(ROOT / "rung0-exact.png"), "seconds": round(time.time() - t0, 1)})


def cmd_experts(a):
    rung0_passed()
    t0 = time.time()
    require("experts", [ROOT / f"expert-{c}.pt" for c in ("red", "blue", "empty")])
    (e, r, b, j, meta), (Ee, Er, Eb, Ej) = exact_all()
    info = {}
    for name, pics, ex, ups in (("red", r, Er, a.updates), ("blue", b, Eb, a.updates), ("empty", e, Ee, a.updates // 4)):
        t1 = time.time()
        model, curve = st.train_expert(pics, ups, seed={"red": 1, "blue": 2, "empty": 3}[name], log=name)
        torch.save(model.state_dict(), ROOT / f"expert-{name}.pt")
        err = st.band_error(model, ex, pics)
        info[name] = {"updates": ups, "seconds": round(time.time() - t1, 1), "loss_curve": curve, "error_vs_exact_by_band": err}
        print(name, json.dumps(err), flush=True)
    info["seconds"] = round(time.time() - t0, 1)
    save("experts", info)


def cmd_rung1(a):
    rung0_passed()
    t0 = time.time()
    require("rung1", [ROOT / "samples-neural-plainsum.pt", ROOT / "rung1-neural-plain-sum.png"])
    (e, r, b, j, meta), (Ee, Er, Eb, Ej) = exact_all()
    red, blue, empty = st.load_experts()
    with torch.no_grad():
        xn = st.sample(st.plainsum_denoiser(red, blue, empty))
        xnc = st.sample_corrected(st.plainsum_denoiser(red, blue, empty), ll.CORR_STEPS)
    st.save_samples("neural-plainsum", xn)
    st.save_samples("neural-plainsum-corrected", xnc)
    err = st.band_error(lambda x, s: st.plainsum_of(red(x, s), blue(x, s), empty(x, s)),
                        lambda x, s: st.plainsum_of(Er(x, s), Eb(x, s), Ee(x, s)), j)
    mn, mnc = st.metrics(xn, j), st.metrics(xnc, j)
    for m, x in ((mn, xn), (mnc, xnc)):
        m.update(st.picture_extras(x))
        m["verdict"] = verdict(m["clean_share"], mn["clean_share"])
    rows = [("exact joint", st.metrics(st.load_samples("exact-joint"), j)), ("exact plain sum", st.metrics(st.load_samples("exact-plainsum"), j)),
            ("neural plain sum", mn), ("neural plain sum, corrector 20 steps (0.05 sigma^2, untuned for 32 by 32)", mnc)]
    print(st.table(rows), flush=True)
    st.grid([st.load_samples("exact-joint"), st.load_samples("exact-plainsum"), xn, xnc],
            ["exact joint", "exact plain sum", "neural plain sum", "plain sum +corr"], ROOT / "rung1-neural-plain-sum.png", scale=4)
    save("rung1", {"rows": dict(rows), "plainsum_error_vs_exact_plainsum_by_band": err, "seconds": round(time.time() - t0, 1)})


def cmd_curve(a):
    """Method A's data curve: the ladder's parent run (Method A's recipe, kept delta judged, with the corrector read beside it) at
    each N; the keys are the ladder's parent keys, so the ladder at any of these N reuses its parent."""
    w = ll.finish_world(make_world())
    for n in a.ns:
        ll.run_trained(w, n, a.updates, "parent-A", "none: Method A's recipe")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sets")
    sub.add_parser("rung0")
    q = sub.add_parser("experts")
    q.add_argument("--updates", type=int, default=EXPERT_UPDATES)
    sub.add_parser("rung1")
    q = sub.add_parser("curve")
    q.add_argument("--ns", type=int, nargs="+", default=[256, 1024, 4096])
    q.add_argument("--updates", type=int, default=ll.LADDER_UPDATES)
    q = sub.add_parser("all", help="sets, rung0, experts, rung1 and curve in one job, stopping at the first failure")
    q.add_argument("--ns", type=int, nargs="+", default=[256, 1024, 4096])
    q.add_argument("--updates", type=int, default=ll.LADDER_UPDATES)
    q.add_argument("--expert-updates", type=int, default=EXPERT_UPDATES)
    a = p.parse_args()
    if a.cmd == "all":
        cmd_sets(a)
        if "rung0" not in results():
            cmd_rung0(a)
        rung0_passed()
        if "experts" not in results():
            cmd_experts(argparse.Namespace(updates=a.expert_updates))
        if "rung1" not in results():
            cmd_rung1(a)
        cmd_curve(a)
        return
    {"sets": cmd_sets, "rung0": cmd_rung0, "experts": cmd_experts, "rung1": cmd_rung1, "curve": cmd_curve}[a.cmd](a)


if __name__ == "__main__":
    main()
