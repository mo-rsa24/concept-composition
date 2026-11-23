#!/usr/bin/env python
"""The word-map checker on a 2 x 2 latent: every value the miniature's pages print comes from this key.

The miniature is Learn-to-Steer's spatial head shrunk until one person can work it on paper, and set on
our product of experts. Its weights are chosen by hand, so it shows what the parts compute and says
nothing about how well a trained head performs on SDXL.

  latent   x: 2 x 2 patches, 2 channels (cat-ness, dog-ness) = 8 numbers
  maps     each word reads the latent through 2 cross-attention layers, keys k = 2 and k = 1
  pool     renormalise each map over the grid, then a fixed 3/4 and 1/4 mix (stands for the learned query)
  head     two hidden units, the difference in horizontal and vertical centre of mass, then 5 logits
  loss     L = -log p[left of], for "a cat to the left of a dog"
  push     Learn-to-Steer's z <- z - alpha grad L, or our capped push on the product's noise prediction

Run:
  PYTHONDONTWRITEBYTECODE=1 uv run --no-project --with numpy python -I miniature.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PATCHES = ["top left", "top right", "bottom left", "bottom right"]
RELATIONS = ["above", "below", "left of", "right of", "neutral"]
XI = np.array([-1.0, 1.0, -1.0, 1.0])       # horizontal position of each patch
ETA = np.array([1.0, 1.0, -1.0, -1.0])      # vertical position of each patch
KEYS = np.array([2.0, 1.0])                 # the word's key in layer 1 and layer 2
POOL = np.array([0.75, 0.25])               # stands for the learned query's weights
KAPPA_TABLE = {5: 4.0, 10: 4.0}             # the head's scale for the two step rows the miniature uses
KAPPA = KAPPA_TABLE[5]
LEFT = RELATIONS.index("left of")
X_START = np.array([2.0, 2.0, -1.0, -1.0, 2.0, 2.0, -1.0, -1.0])  # one fused body along the top row

ALPHA_BAR_T, ALPHA_BAR_PREV = 0.36, 0.64
EPS_CAT = np.array([0.25, 0.25, -0.25, -0.25, 0.25, 0.25, -0.25, -0.25])
EPS_DOG = EPS_CAT.copy()
EPS_NULL = np.zeros(8)
STRENGTH = 2.0
CAP = 0.4
LTS_ALPHA = 5.0
LTS_PASSES = 16                             # range(15 + 1) in the authors' refine_latents
LTS_TARGET = 0.00007                        # their loss target from sampler step 3 on


def r(a, n=4):
    """Rounded for the key: plain lists of floats, with -0.0 written as 0.0."""
    return (np.round(np.asarray(a, dtype=float), n) + 0.0).tolist()


def word_maps(channel):
    """Two prompts: at each patch one word competes only with the background token, whose key is 0."""
    return [np.exp(k * channel) / (1.0 + np.exp(k * channel)) for k in KEYS]


def pooled(channel):
    maps = word_maps(channel)
    return sum(w * (a / a.sum()) for w, a in zip(POOL, maps))


def forward(x, kappa=KAPPA):
    a_s, a_o = pooled(x[:4]), pooled(x[4:])
    hx, hy = float(XI @ (a_s - a_o)), float(ETA @ (a_s - a_o))
    logits = np.array([kappa * hy, -kappa * hy, -kappa * hx, kappa * hx, 0.0])
    p = np.exp(logits) / np.exp(logits).sum()
    return dict(a_s=a_s, a_o=a_o, hx=hx, hy=hy, logits=logits, p=p, loss=float(-np.log(p[LEFT])))


def loss(x, kappa=KAPPA):
    return forward(x, kappa)["loss"]


def d_pool_d_channel(channel):
    """The 4 x 4 Jacobian of the pooled map with respect to one word's channel."""
    jac = np.zeros((4, 4))
    for w, k, a in zip(POOL, KEYS, word_maps(channel)):
        s = a.sum()
        d_a = k * a * (1 - a)
        for p in range(4):
            for q in range(4):
                jac[p, q] += w * ((p == q) / s - a[p] / s ** 2) * d_a[q]
    return jac


def gradient(x, kappa=KAPPA):
    """The chain rule, written the way the page works it: loss to hidden units to pooled maps to latent."""
    f = forward(x, kappa)
    d_logits = np.array([[0, kappa], [0, -kappa], [-kappa, 0], [kappa, 0], [0, 0]], dtype=float)  # logits by (hx, hy)
    d_loss_d_logits = f["p"].copy()
    d_loss_d_logits[LEFT] -= 1.0
    d_loss_d_h = d_logits.T @ d_loss_d_logits
    d_loss_d_as = d_loss_d_h[0] * XI + d_loss_d_h[1] * ETA
    d_loss_d_ao = -d_loss_d_as
    g_s = d_pool_d_channel(x[:4]).T @ d_loss_d_as
    g_o = d_pool_d_channel(x[4:]).T @ d_loss_d_ao
    return np.concatenate([g_s, g_o]), dict(d_loss_d_h=d_loss_d_h, d_loss_d_as=d_loss_d_as, d_loss_d_ao=d_loss_d_ao)


def finite_difference(fn, x, e=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        d = np.zeros_like(x)
        d[i] = e
        g[i] = (fn(x + d) - fn(x - d)) / (2 * e)
    return g


def lts_loop(x, alpha=LTS_ALPHA, passes=LTS_PASSES, target=LTS_TARGET):
    rows, y = [], x.copy()
    for i in range(passes):
        f = forward(y)
        if not f["loss"] > target:
            rows.append(dict(i=i, loss=round(f["loss"], 4), p_left=round(float(f["p"][LEFT]), 4), stopped=True))
            break
        g, _ = gradient(y)
        rows.append(dict(i=i, loss=round(f["loss"], 4), p_left=round(float(f["p"][LEFT]), 4),
                         grad_norm=round(float(np.linalg.norm(g)), 4)))
        y = y - alpha * g
    return y, rows


def ddim(x, eps):
    x0 = (x - np.sqrt(1 - ALPHA_BAR_T) * eps) / np.sqrt(ALPHA_BAR_T)
    return x0, np.sqrt(ALPHA_BAR_PREV) * x0 + np.sqrt(1 - ALPHA_BAR_PREV) * eps


def correlation(x):
    a, b = pooled(x[:4]), pooled(x[4:])
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def train_logistic(data, steps=400, lr=1.0):
    """One logit u = a h + b tau; p(right of) = sigmoid(u). Full-batch gradient descent from zero."""
    w = np.zeros(2)
    X, y = np.array([d[:2] for d in data], dtype=float), np.array([d[2] for d in data], dtype=float)
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ w)))
        w -= lr * X.T @ (p - y) / len(y)
    return w


def sigmoid(u):
    return float(1 / (1 + np.exp(-u)))


def main():
    global KEYS
    key = {"_about": "Every value the miniature's pages and figures print. Written by miniature.py; never edited by hand.",
           "patches": PATCHES, "relations": RELATIONS, "xi": r(XI), "eta": r(ETA), "keys": r(KEYS), "pool_weights": r(POOL),
           "kappa": KAPPA, "kappa_table": {str(k): v for k, v in KAPPA_TABLE.items()}, "prompt": "a cat to the left of a dog"}

    # 1. the latent and the word maps, two prompts against one joint prompt
    x = X_START.copy()
    maps_s, maps_o = word_maps(x[:4]), word_maps(x[4:])
    joint = [np.exp(k * x[:4]) / (1 + np.exp(k * x[:4]) + np.exp(k * x[4:])) for k in KEYS]
    key["latent_start"] = {"cat": r(x[:4]), "dog": r(x[4:])}
    key["maps"] = {
        "cat_layer1": r(maps_s[0]), "cat_layer2": r(maps_s[1]), "dog_layer1": r(maps_o[0]), "dog_layer2": r(maps_o[1]),
        "joint_prompt_cat_layer1": r(joint[0]), "joint_prompt_cat_layer2": r(joint[1]),
        "layer1_sum": round(float(maps_s[0].sum()), 4), "layer2_sum": round(float(maps_s[1].sum()), 4),
        "cat_layer1_renormalised": r(maps_s[0] / maps_s[0].sum()), "cat_layer2_renormalised": r(maps_s[1] / maps_s[1].sum()),
    }

    # 2. the head's forward pass on the fused body
    f0 = forward(x)
    key["forward_fused"] = {"pooled_cat": r(f0["a_s"]), "pooled_dog": r(f0["a_o"]), "hx": round(f0["hx"], 4), "hy": round(f0["hy"], 4),
                            "logits": r(f0["logits"]), "p": r(f0["p"]), "loss": round(f0["loss"], 4)}

    # 3. the gradient, and the finite-difference check
    g, parts = gradient(x)
    fd = finite_difference(loss, x)
    assert np.max(np.abs(g - fd)) < 1e-7, "the chain rule must match the finite difference"
    key["gradient_fused"] = {"d_loss_d_hx_hy": r(parts["d_loss_d_h"]), "d_loss_d_pooled_cat": r(parts["d_loss_d_as"]),
                             "d_loss_d_pooled_dog": r(parts["d_loss_d_ao"]), "cat": r(g[:4]), "dog": r(g[4:]),
                             "norm": round(float(np.linalg.norm(g)), 4), "largest_finite_difference_gap": float(np.max(np.abs(g - fd)))}
    a1 = maps_s[0]
    key["gradient_fused"]["slope_of_map_layer1"] = r(KEYS[0] * a1 * (1 - a1))

    # the bottom-left entry of the cat's gradient, worked term by term on the rebuild page
    terms = []
    for w_l, k_l, a_l in zip(POOL, KEYS, maps_s):
        slope = float(k_l * a_l[2] * (1 - a_l[2]))
        terms.append(dict(pool_weight=float(w_l), slope=round(slope, 6), map_sum=round(float(a_l.sum()), 6),
                          term=round(float(w_l) * slope / float(a_l.sum()), 6)))
    key["gradient_fused"]["worked_bottom_left_cat"] = {"d_loss_d_pooled": -4.0, "terms": terms}
    assert abs(-4.0 * sum(q["term"] for q in terms) - g[2]) < 1e-4

    # 4. Learn-to-Steer's update on the latent, up to 16 passes at alpha 5
    y, rows = lts_loop(x)
    fy = forward(y)
    key["lts_loop"] = {"alpha": LTS_ALPHA, "passes_allowed": LTS_PASSES, "rows": rows, "updates_made": sum(1 for q in rows if "grad_norm" in q),
                       "final_latent": {"cat": r(y[:4]), "dog": r(y[4:])}, "final_pooled_cat": r(fy["a_s"]), "final_pooled_dog": r(fy["a_o"]),
                       "final_p": r(fy["p"]), "final_loss": round(fy["loss"], 4), "final_hx": round(fy["hx"], 4), "final_hy": round(fy["hy"], 4)}
    one = x - LTS_ALPHA * g
    fo = forward(one)
    key["lts_first_update"] = {"latent": {"cat": r(one[:4]), "dog": r(one[4:])}, "pooled_cat": r(fo["a_s"]), "pooled_dog": r(fo["a_o"]),
                               "hx": round(fo["hx"], 4), "p_left": round(float(fo["p"][LEFT]), 4), "loss": round(fo["loss"], 4)}

    # 5. our route: the product's prediction, a capped push, one DDIM step
    eps = EPS_CAT + EPS_DOG - EPS_NULL
    sigma = float(np.sqrt(1 - ALPHA_BAR_T))
    grad_log_p = -g
    raw = STRENGTH * sigma * grad_log_p
    limit = CAP * float(np.linalg.norm(eps))
    fired = float(np.linalg.norm(raw)) > limit
    push = raw * (limit / np.linalg.norm(raw)) if fired else raw
    eps_pushed = eps - push
    x0_plain, x_plain = ddim(x, eps)
    x0_push, x_push = ddim(x, eps_pushed)
    fp, fplain = forward(x_push), forward(x_plain)
    key["product_step"] = {
        "alpha_bar_t": ALPHA_BAR_T, "alpha_bar_prev": ALPHA_BAR_PREV, "sigma_t": round(sigma, 4), "strength": STRENGTH, "cap": CAP,
        "eps_cat": r(EPS_CAT), "eps_dog": r(EPS_DOG), "eps_null": r(EPS_NULL), "eps_product": r(eps), "eps_norm": round(float(np.linalg.norm(eps)), 4),
        "raw_push": r(raw), "raw_norm": round(float(np.linalg.norm(raw)), 4), "limit": round(limit, 4), "cap_fired": bool(fired),
        "scale_applied": round(limit / float(np.linalg.norm(raw)), 4) if fired else 1.0, "push": r(push), "push_norm": round(float(np.linalg.norm(push)), 4),
        "eps_pushed": r(eps_pushed),
        "x0_plain": r(x0_plain), "x_next_plain": r(x_plain), "x0_pushed": r(x0_push), "x_next_pushed": r(x_push),
        "plain_after": {"hx": round(fplain["hx"], 4), "p_left": round(float(fplain["p"][LEFT]), 4), "loss": round(fplain["loss"], 4)},
        "pushed_after": {"hx": round(fp["hx"], 4), "p_left": round(float(fp["p"][LEFT]), 4), "loss": round(fp["loss"], 4),
                         "pooled_cat": r(fp["a_s"]), "pooled_dog": r(fp["a_o"])},
    }

    # 6. what breaks
    # 6a. relation leakage: a reading h that is weak and a trace tau of the prompt's word
    naive = [(0.4, 1, 1), (0.2, 1, 1), (-0.4, -1, 0), (-0.2, -1, 0)]
    both = naive + [(0.4, -1, 1), (0.2, -1, 1), (-0.4, 1, 0), (-0.2, 1, 0)]
    wn, wb = train_logistic(naive), train_logistic(both)
    test = (-0.4, 1.0)                       # the cat is truly on the left, and the inversion prompt said "right of"
    key["broken_leakage"] = {
        "features": "h: the map's spatial reading, tau: the trace of the prompt's relation word (+1 right of, -1 left of)",
        "training_steps": 400, "learning_rate": 1.0,
        "captions_only_data": [list(d) for d in naive], "both_prompts_data": [list(d) for d in both],
        "captions_only_weights": {"a_on_h": round(float(wn[0]), 4), "b_on_tau": round(float(wn[1]), 4)},
        "both_prompts_weights": {"a_on_h": round(float(wb[0]), 4), "b_on_tau": round(float(wb[1]), 4)},
        "test_truly_left_prompt_says_right": {"h": test[0], "tau": test[1],
                                              "captions_only_p_right": round(sigmoid(wn @ np.array(test)), 4),
                                              "both_prompts_p_right": round(sigmoid(wb @ np.array(test)), 4)},
    }
    assert sigmoid(wn @ np.array(test)) > 0.5 > sigmoid(wb @ np.array(test)), "the naive head must follow the word, the fixed one the picture"
    assert abs(wb[1]) < 1e-9, "with both prompts the trace carries no information"

    # 6b. the hand-written correlation loss our own push used, on the same fused body
    g_corr = finite_difference(lambda v: correlation(v), x)
    nudged = x.copy()
    nudged[0] += 0.1                         # a tiny difference between the two maps
    g_corr_nudged = finite_difference(lambda v: correlation(v), nudged)
    key["broken_correlation"] = {
        "correlation_fused": round(correlation(x), 4), "gradient_norm_fused": round(float(np.linalg.norm(g_corr)), 6),
        "after_nudge_of_0.1_on_cat_top_left": {"correlation": round(correlation(nudged), 6),
                                               "gradient_norm": round(float(np.linalg.norm(g_corr_nudged)), 6)},
        "head_gradient_norm_fused": round(float(np.linalg.norm(g)), 4),
    }

    # 6c. no answer for "fused"
    key["broken_no_fused_class"] = {"p_on_identical_maps": r(f0["p"]), "note": "identical maps give 0.2 on all five classes, and no class means fused"}

    # 6d. the dose: one capped push against the full loop, measured on the same start
    key["dose"] = {
        "one_capped_push_then_ddim": {"p_left": round(float(fp["p"][LEFT]), 4), "latent_change_norm": round(float(np.linalg.norm(x_push - x_plain)), 4)},
        "lts_16_passes": {"p_left": round(float(fy["p"][LEFT]), 4), "latent_change_norm": round(float(np.linalg.norm(y - x)), 4)},
    }

    # 6e. the exercise on the page: both layers given the same key
    saved = KEYS
    KEYS = np.array([1.0, 1.0])
    g_equal, _ = gradient(x)
    y_equal, _ = lts_loop(x)
    key["exercise_equal_keys"] = {"keys": [1.0, 1.0], "gradient_norm": round(float(np.linalg.norm(g_equal)), 4),
                                  "p_left_after_16_updates": round(float(forward(y_equal)["p"][LEFT]), 4)}
    KEYS = saved

    # how much one unit of the prediction moves the next latent in the DDIM step
    key["product_step"]["ddim_coefficient_on_eps"] = round(
        float(np.sqrt(1 - ALPHA_BAR_PREV) - np.sqrt(1 - ALPHA_BAR_T) * np.sqrt(ALPHA_BAR_PREV) / np.sqrt(ALPHA_BAR_T)), 4)

    assert abs(f0["p"][LEFT] - 0.2) < 1e-12 and abs(f0["hx"]) < 1e-12
    assert fy["p"][LEFT] > 0.9, "the full loop must reach the relation on the miniature"
    assert fired, "the cap must fire in the worked step"
    (HERE / "miniature_answer_key.json").write_text(json.dumps(key, indent=1) + "\n")
    print(json.dumps({k: key[k] for k in ("maps", "forward_fused", "gradient_fused", "lts_first_update")}, indent=0)[:2600])
    print("LOOP", key["lts_loop"]["updates_made"], key["lts_loop"]["final_latent"], key["lts_loop"]["final_p"], [q["loss"] for q in rows][:6])
    print("PRODUCT", {k: key["product_step"][k] for k in ("raw_norm", "limit", "scale_applied", "push_norm", "plain_after", "pushed_after")})
    print("LEAK", key["broken_leakage"]["captions_only_weights"], key["broken_leakage"]["both_prompts_weights"], key["broken_leakage"]["test_truly_left_prompt_says_right"])
    print("CORR", key["broken_correlation"])
    print("DOSE", key["dose"])


if __name__ == "__main__":
    main()
