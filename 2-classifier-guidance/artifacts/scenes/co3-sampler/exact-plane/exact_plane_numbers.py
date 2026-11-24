"""Every number the exact-plane rung prints, from one real CO3 run (Slurm job 94450, two pairs, seed 6001).

Reads, per pair, meta.json, passes.npz and plain-joint-path.npy as written by
2-classifier-guidance/scripts/leakage/co3_inside.py (design: artifacts/notes/progress/leakage/co3-inside/design.md),
and writes exact_plane_numbers.json beside this file. Never edit the JSON by hand.

  python3 exact_plane_numbers.py [--data <folder holding <pair>/seed-6001/>]

What it computes (float64 from the float16 records):
  * per corrected step, at its last pass: the three guided Tweedie means
    x_hat[c] = x_t - sqrt(1 - abar) * (eps_empty + lambda * (eps_c - eps_empty)) for c = joint, A, B,
    an orthonormal basis of the flat sheet through them (origin at the joint's, first axis toward A's),
    and each point's in-sheet coordinates and distance from the sheet;
  * for corrector steps, the composed mean 2 x_hat[joint] + w1 x_hat[A] + w2 x_hat[B] (Algorithm 3 line 4) and the
    same after the code's rescale (attn_utils.py line 255, which matches largest entries);
  * for resampler steps, the composed noise term -sqrt(1 - abar) * lambda * (eps_joint + w1 eps_A + w2 eps_B)
    (attn_utils.py lines 194 to 196), a difference of two points of the sheet, so a direction within it;
  * the distance between the CO3 path and the plain path at each of the 50 steps, over the plain latent's length;
  * one corrector pass worked end to end (cheetah x cougar, step 3, first pass).
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = Path("/home/molef/.claude/jobs/0071c318/tmp/co3-inside")
PAIRS = {"a_cheetah__x__a_cougar": "borrowed", "an_eagle__x__a_fox": "own_parts"}
SEED = 6001
TRACE = dict(pair="a_cheetah__x__a_cougar", step=3, which="first")


def norm(v):
    return float(np.linalg.norm(v))


def guesses(x, eps, abar, lam):
    """The four Tweedie means the code forms from one pass (no division by sqrt(abar))."""
    s = (1 - abar) ** 0.5
    eu, em, e1, e2 = eps
    g = lambda e: x - s * (eu + lam * (e - eu))
    return dict(empty=x - s * eu, joint=g(em), a=g(e1), b=g(e2))


def sheet(gC, gA, gB):
    u = gA - gC
    b1 = u / np.linalg.norm(u)
    v = gB - gC
    v = v - (v @ b1) * b1
    b2 = v / np.linalg.norm(v)

    def project(p, point=True):
        q = p - gC if point else p
        c = np.array([q @ b1, q @ b2])
        return [float(c[0]), float(c[1])], norm(q - c[0] * b1 - c[1] * b2)
    return project


def pair_numbers(folder):
    meta = json.loads((folder / "meta.json").read_text())
    d = np.load(folder / "passes.npz")
    X, E, steps = d["x"].astype(np.float64), d["eps"].astype(np.float64), d["step"]
    lam = float(meta["lambda"])
    out = dict(prompt=meta["prompt"], single_prompts=meta["single_prompts"], lambda_=lam,
               worst_rel_diff_in_run=meta["worst_rel_diff"], control_passes=meta["control_passes"], steps=[])
    for s in range(6):
        idx = np.where(steps == s)[0]
        first, last = int(idx[0]), int(idx[-1])
        abar = float(meta["alphas"][str(s)])
        kind = meta["checks"][last]["kind"]
        x = X[last].ravel()
        eps = [e.ravel() for e in E[last]]
        g = guesses(x, eps, abar, lam)
        w1, w2 = meta["weights_log"][last]["w"]
        project = sheet(g["joint"], g["a"], g["b"])
        avg = -(w1 * g["a"] + w2 * g["b"])          # w1 + w2 = -1, so this is the singles' weighted average
        pts = {k: project(g[k]) for k in ("joint", "a", "b")}
        row = dict(step=s, t=981 - 20 * s, kind=kind, passes=len(idx), abar=abar, sqrt_one_minus_abar=(1 - abar) ** 0.5,
                   beta=meta["weights_log"][last]["beta"],
                   first_pass=dict(index=first, dists=meta["weights_log"][first]["dists"], w=meta["weights_log"][first]["w"]),
                   last_pass=dict(index=last, dists=meta["weights_log"][last]["dists"], w=[w1, w2],
                                  dists_recomputed=[norm(eps[2] - eps[1]), norm(eps[3] - eps[1])],
                                  rel_diff=meta["checks"][last]["rel_diff"]),
                   length_joint_guess=norm(g["joint"]),
                   side_joint_a=norm(g["a"] - g["joint"]), side_joint_b=norm(g["b"] - g["joint"]), side_a_b=norm(g["a"] - g["b"]),
                   sheet={k: dict(xy=v[0], off_sheet=v[1]) for k, v in pts.items()})
        row["sheet"]["singles_average"] = dict(zip(("xy", "off_sheet"), project(avg)))
        row["sheet"]["empty_off_sheet"] = project(g["empty"])[1]
        if kind == "correct":
            comb = 2 * g["joint"] + w1 * g["a"] + w2 * g["b"]
            scale = g["joint"].max() / comb.max()
            xy, off = project(comb)
            row["sheet"]["combination"] = dict(xy=xy, off_sheet=off, off_sheet_relative=off / norm(comb),
                                               move_from_joint=norm(comb - g["joint"]))
            xy_r, off_r = project(comb * scale)
            row["sheet"]["after_code_rescale"] = dict(scale=float(scale), xy=xy_r, off_sheet=off_r,
                                                      move_from_combination=norm(comb * scale - comb))
        else:
            term = -((1 - abar) ** 0.5) * lam * (eps[1] + w1 * eps[2] + w2 * eps[3])
            xy, off = project(term, point=False)
            row["sheet"]["resampler_term"] = dict(xy=xy, off_sheet=off, off_sheet_relative=off / norm(term), length=norm(term),
                                                  equals_joint_minus_average=norm(term - (g["joint"] - avg)) / norm(term))
        out["steps"].append(row)
    co3 = d["co3_path"].astype(np.float64).reshape(50, -1)
    plain = np.load(folder / "plain-joint-path.npy").astype(np.float64).reshape(50, -1)
    out["path"] = dict(relative_distance=[float(v) for v in np.linalg.norm(co3 - plain, axis=1) / np.linalg.norm(plain, axis=1)],
                       plain_length=[float(v) for v in np.linalg.norm(plain, axis=1)],
                       co3_length=[float(v) for v in np.linalg.norm(co3, axis=1)])
    return out


def trace(folder):
    meta = json.loads((folder / "meta.json").read_text())
    d = np.load(folder / "passes.npz")
    idx = np.where(d["step"] == TRACE["step"])[0]
    i = int(idx[0] if TRACE["which"] == "first" else idx[-1])
    x = d["x"][i].astype(np.float64).ravel()
    eu, em, e1, e2 = [e.ravel() for e in d["eps"][i].astype(np.float64)]
    nxt = d["x"][i + 1].astype(np.float64).ravel()
    abar, lam = float(meta["alphas"][str(TRACE["step"])]), float(meta["lambda"])
    beta = meta["weights_log"][i]["beta"]
    s = (1 - abar) ** 0.5
    dk = np.array([norm(e1 - em), norm(e2 - em)])
    a = np.exp(-beta * dk)
    w = -a / a.sum()
    w_log = np.array(meta["weights_log"][i]["w"])
    g = guesses(x, [eu, em, e1, e2], abar, lam)
    comb = 2 * g["joint"] + w_log[0] * g["a"] + w_log[1] * g["b"]
    r_code = float(g["joint"].max())
    r_paper = norm(g["joint"]) / norm(comb)
    comb_scaled = comb / comb.max() * r_code
    x2 = comb_scaled + s * eu
    x2_final = x2 * (norm(x) / norm(x2))
    return dict(pair=TRACE["pair"], step=TRACE["step"], t=981 - 20 * TRACE["step"], pass_index=i, pass_in_step=1,
                passes_in_step=len(idx), abar=abar, sqrt_one_minus_abar=s, lambda_=lam, beta=beta,
                dists_logged=meta["weights_log"][i]["dists"], dists_recomputed=dk.tolist(),
                affinities=a.tolist(), weights_recomputed=w.tolist(), weights_logged=w_log.tolist(),
                length_x=norm(x), length_joint_guess=norm(g["joint"]), length_a_guess=norm(g["a"]), length_b_guess=norm(g["b"]),
                length_combination=norm(comb), max_combination=float(comb.max()), max_joint_guess=r_code,
                r_code=r_code, r_paper=r_paper, code_scale_factor=r_code / float(comb.max()),
                length_after_rescale=norm(comb_scaled), length_x2_before_norm_match=norm(x2),
                norm_match_factor=norm(x) / norm(x2), length_x2=norm(x2_final), length_next_recorded=norm(nxt),
                rel_diff_here=norm(x2_final - nxt) / norm(nxt), rel_diff_in_run=meta["checks"][i]["rel_diff"],
                change_of_latent_relative=norm(x2_final - x) / norm(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    args = ap.parse_args()
    out = dict(about="Every number the exact-plane rung prints. Written by exact_plane_numbers.py; never edit by hand.",
               source=dict(local=str(args.data), cluster="/datasets/mmolefe/poe_repair_min/loop-12/co3-inside/",
                           slurm_job=94450, code_snapshot="44b51af4", seed=SEED,
                           runner="2-classifier-guidance/scripts/leakage/co3_inside.py",
                           design="artifacts/notes/progress/leakage/co3-inside/design.md",
                           labels={k: v for k, v in PAIRS.items()},
                           borrowed_note="cougar has cheetah spots and tear marks"),
               units="lengths are L2 norms over the 65,536 numbers of a 4 x 128 x 128 SDXL latent, in Tweedie-mean units "
                     "(x_t - sqrt(1 - abar) eps, before any division by sqrt(abar))",
               pairs={}, trace=None)
    for pair in PAIRS:
        out["pairs"][pair] = pair_numbers(args.data / pair / f"seed-{SEED}")
        out["pairs"][pair]["label"] = PAIRS[pair]
    out["trace"] = trace(args.data / TRACE["pair"] / f"seed-{SEED}")
    t = out["trace"]
    assert abs(t["rel_diff_here"]) < 0.01, "the worked pass does not reproduce the next recorded latent"
    assert np.allclose(t["weights_recomputed"], t["weights_logged"], atol=1e-3), "weights do not follow from the distances"
    offs = [r["sheet"][k]["off_sheet_relative"] for p in out["pairs"].values() for r in p["steps"]
            for k in ("combination", "resampler_term") if k in r["sheet"]]
    out["check"] = dict(worst_relative_off_sheet_of_combination_or_term=max(offs),
                        worst_off_sheet_of_the_three_guesses=max(r["sheet"][k]["off_sheet"] for p in out["pairs"].values()
                                                                 for r in p["steps"] for k in ("joint", "a", "b")))
    assert out["check"]["worst_relative_off_sheet_of_combination_or_term"] < 1e-4, "a composition left the sheet"
    (HERE / "exact_plane_numbers.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["check"]), json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in t.items()
                                                if not isinstance(v, list)}))


if __name__ == "__main__":
    main()
