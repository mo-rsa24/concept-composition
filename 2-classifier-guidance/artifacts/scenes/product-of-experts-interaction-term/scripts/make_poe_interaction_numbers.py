"""Every number on every page of this scene comes from here. Writes data/poe_interaction_numbers.json.
Run from the scene folder with the sdxl-local python:  python scripts/make_poe_interaction_numbers.py
All values are exact scores of the toy in poe_toy.py (no network), sampled with a fixed seed. Nothing here is measured on SDXL or on the
MNIST experts; those numbers are quoted on the pages with their source and are labelled as measured elsewhere.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import poe_toy as T

OUT = Path(__file__).parent.parent / "data" / "poe_interaction_numbers.json"
rng = np.random.default_rng(0)
N = 4000
res = {"toy": {"sigma0": T.SIGMA0, "steps": T.K, "abar_first": float(T.ABAR[0]), "abar_last": float(T.ABAR[-1])}}

# 1. the interaction term and the commit of the joint, along the noise levels, at typical states of the joint's own noised marginal
rows = []
for k in range(T.K):
    ab = T.ABAR[k]; a = np.sqrt(ab); v = ab * T.SIGMA0 ** 2 + (1 - ab)
    comp = rng.integers(2, size=N)
    x = a * T.JOINT_MEANS[comp] + np.sqrt(v) * rng.standard_normal((N, 4))
    sJ, r = T.score(x, ab, T.JOINT_MEANS)
    I = sJ - T.poe_score(x, ab)
    r_t = np.sqrt(1 - ab) * np.linalg.norm(I, axis=1)               # |r_t| = sigma_t |I_t|, the noise-prediction gap
    rows.append(dict(step=k, abar=float(ab), sigma=float(np.sqrt(1 - ab)), rms_r_t=float(np.sqrt((r_t ** 2).mean())),
                     mean_I=float(np.linalg.norm(I, axis=1).mean()), decided=float(r.max(1).mean()),
                     joint_posterior_of_true_component=float(r[np.arange(N), comp].mean())))
res["profile"] = rows
dec = np.array([r["decided"] for r in rows]); rr = np.array([r["rms_r_t"] for r in rows])
res["commit_step_decided_90"] = int(np.argmax(dec >= 0.9))
res["peak_r_t_step"] = int(np.argmax(rr))
res["r_t_share_of_total_in_steps_0_to_9"] = float((rr[:10] ** 2).sum() / (rr ** 2).sum())

# 2. what each composition draws, same starting noise for all
init = rng.standard_normal((N, 4)); noise = rng.standard_normal((T.K, N, 4))
poe = lambda x, k: T.eps_of(T.poe_score(x, T.ABAR[k]), T.ABAR[k])
jnt = lambda x, k: T.eps_of(T.joint_score(x, T.ABAR[k]), T.ABAR[k])


def window(lo, hi, lam=1.0):
    return lambda x, k: poe(x, k) + (lam * (jnt(x, k) - poe(x, k)) if lo <= k < hi else 0.0)


def tally(x):
    c = T.classify(x); names = ["two apart", "both in one cell", "one digit", "other"]
    return {n: int((c == n).sum()) for n in names}


methods = {
    "product of the two experts": poe,
    "product plus the interaction term at every step (the joint's own score)": jnt,
    "product plus the interaction term, steps 0 to 9 only": window(0, 10),
    "product plus the interaction term, steps 10 to 49 only": window(10, T.K),
    "product plus the interaction term, steps 0 to 4 only": window(0, 5),
    "product plus the interaction term, steps 5 to 9 only": window(5, 10),
    "each expert keeps its own cell, whole canvas visible, coin layout held (no hiding)": T.hide_eps("coin", False),
    "each expert keeps its own cell, other cell hidden, coin layout held": T.hide_eps("coin", True),
    "each expert keeps its own cell, other cell hidden, layout voted over steps 0 to 9 then held (E0e)": T.hide_eps("vote", True),
    "each expert keeps its own cell, other cell hidden, layout re-voted at every step (E0d-b)": T.hide_eps("vote-each-step", True),
    "each expert keeps its own cell, whole canvas visible, layout voted then held": T.hide_eps("vote", False),
}
res["outcomes_of_4000"] = {}
for name, fn in methods.items():
    res["outcomes_of_4000"][name] = tally(T.sample(fn, noise, init))

# 2b. where the interaction term has to act: the joint's score added only inside a window of steps, same noise
sweep = {"ten-step windows": {}, "from step s to the end": {}, "from step 0 to step e": {}}
for lo in range(0, T.K, 10):
    sweep["ten-step windows"][f"steps {lo} to {lo + 9}"] = tally(T.sample(window(lo, lo + 10), noise, init))["two apart"]
for lo in range(0, T.K, 5):
    sweep["from step s to the end"][str(lo)] = tally(T.sample(window(lo, T.K), noise, init))["two apart"]
for hi in range(5, T.K + 1, 5):
    sweep["from step 0 to step e"][str(hi)] = tally(T.sample(window(0, hi), noise, init))["two apart"]
res["two_apart_of_4000_by_window"] = sweep

# 3. the support argument in one table: the mass each density puts on the canvases that matter (at the cleanest level)
pts = {"4 in the left cell only": [1, 0, 0, 0], "7 in the right cell only": [0, 0, 0, 1], "4 left and 7 right (two apart)": [1, 0, 0, 1],
       "4 and 7 both in the left cell (fused)": [1, 1, 0, 0], "empty": [0, 0, 0, 0]}
ab = T.ABAR[-1]
tab = {}


def logdens(x, means):
    d = x[None] - means; return float(np.log(np.exp(-(d ** 2).sum(1) / (2 * T.SIGMA0 ** 2)).mean() + 1e-300))


for n, p in pts.items():
    p = np.array(p, float)
    lA, lB, l0, lJ = (logdens(p, m) for m in (T.A_MEANS, T.B_MEANS, T.BASE_MEANS, T.JOINT_MEANS))
    tab[n] = dict(log_p_A=lA, log_p_B=lB, log_p_base=l0, log_product_A_B_over_base=lA + lB - l0, log_p_joint=lJ)
res["log_density_table_at_clean_canvas"] = tab

# 4. a vector field on the slice (4 in left, 7 in right), at two noise levels, for the figure
grid = np.linspace(-0.4, 1.4, 13)
fields = {}
for k in (5, 25):
    ab = T.ABAR[k]; X, Y = np.meshgrid(grid, grid)
    x = np.zeros((X.size, 4)); x[:, 0] = X.ravel(); x[:, 3] = Y.ravel()
    sP, sJ = T.poe_score(x, ab), T.joint_score(x, ab)
    fields[str(k)] = dict(abar=float(ab), grid=grid.tolist(), poe=sP[:, [0, 3]].tolist(), joint=sJ[:, [0, 3]].tolist())
res["slice_fields"] = fields
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps(res["two_apart_of_4000_by_window"]))
print(json.dumps({k: res[k] for k in ("commit_step_decided_90", "peak_r_t_step", "r_t_share_of_total_in_steps_0_to_9")}))
for n, v in res["outcomes_of_4000"].items():
    print(f"{v['two apart']:5d} two apart  {v['both in one cell']:5d} one cell  {v['one digit']:5d} one digit  {v['other']:5d} other  | {n}")
