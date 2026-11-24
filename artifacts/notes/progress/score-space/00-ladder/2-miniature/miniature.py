"""The score-path miniature: 4 pixels, 2 sampler steps, integer weights set by hand. A toy that predicts nothing.

The canvas is 4 pixels, two on the left and two on the right. The toy noise predictor is

    eps(x, c) = x + u_c + v_c * x          (elementwise; u_null = v_null = 0, so eps(x, null) = x)

so d(c) = eps(x, c) - eps(x, null) = u_c + v_c * x depends on where the canvas is. Label 3 acts on the left pixels and
label 7 on the right. The toy sampler step subtracts the product's prediction: x_next = x - eps_PoE. The score path is d
at steps 0 and 1, read along the product's own trajectory: 2 x 4 = 8 numbers.

The planned score-path encoder is drawn as one integer 2 x 8 matrix E (illustrative; step X4 has not designed it). Caption
vectors are hand-set 2-number integers. The contrastive score is cos(z, e) / tau with W4's real tau = 0.07.

The joint path uses a pair label "3 and 7" that the real digit models do not have: its d is d(3) + d(7) + r, with r a
hand-set interaction term. It exists here only to show where the gap test reads.

Writes miniature_answer_key.json beside this file. Run: python3 miniature.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TAU = json.loads((HERE.parent / "numbers" / "numbers.json").read_text())["w4_text_with_latent"]["temperature"]

U = {"null": [0, 0, 0, 0], "3": [-2, -1, 0, 0], "7": [0, 0, -1, -2]}
V = {"null": [0, 0, 0, 0], "3": [1, 0, 0, 0], "7": [0, 0, 0, 1]}
R = [0, 1, 1, 1]                      # the toy interaction term, the joint label's only addition
X0 = [1, -1, 2, 0]                    # the starting noise, 4 integers
E = [[-1, -1, 0, 0, -1, -1, 0, 0],    # the toy encoder's two rows: minus the left d, minus the right d, both steps
     [0, 0, -1, -1, 0, 0, -1, -1]]
CAPTIONS = {"one 3": [3, 0], "one 7": [0, 3], "a 3 and a 7, apart": [3, 2], "a 3 on top of a 7": [4, 1]}


def eps(x, c):
    return np.array(x) + np.array(U[c]) + np.array(V[c]) * np.array(x)


def cos(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


steps, x = [], np.array(X0)
for k in range(2):
    e0, e3, e7 = eps(x, "null"), eps(x, "3"), eps(x, "7")
    d3, d7 = e3 - e0, e7 - e0
    poe = e0 + d3 + d7
    joint_d = d3 + d7 + np.array(R)
    x_next = x - poe
    steps.append({"step": k, "x": x.tolist(), "eps_null": e0.tolist(), "eps_3": e3.tolist(), "eps_7": e7.tolist(),
                  "d_3": d3.tolist(), "d_7": d7.tolist(), "eps_poe": poe.tolist(), "d_joint": joint_d.tolist(),
                  "x_next": x_next.tolist()})
    x = x_next

paths = {name: sum((s[key] for s in steps), []) for name, key in
         (("3", "d_3"), ("7", "d_7"), ("joint", "d_joint"))}
paths["poe"] = (np.array(paths["3"]) + np.array(paths["7"])).tolist()
Emat = np.array(E)
z = {k: (Emat @ np.array(v)).tolist() for k, v in paths.items()}
assert z["poe"] == (np.array(z["3"]) + np.array(z["7"])).tolist()  # a linear encoder adds exactly

# The contrastive matrix: rows are the three training paths, columns the three captions they match.
rows, cols = ["3", "7", "poe"], ["one 3", "one 7", "a 3 and a 7, apart"]
logits = [[cos(z[r], CAPTIONS[c]) / TAU for c in cols] for r in rows]
L = np.array(logits)
p_rows = np.exp(L - L.max(1, keepdims=True)); p_rows /= p_rows.sum(1, keepdims=True)
p_cols = np.exp(L - L.max(0, keepdims=True)); p_cols /= p_cols.sum(0, keepdims=True)
ce_rows = [-float(np.log(p_rows[i, i])) for i in range(3)]
ce_cols = [-float(np.log(p_cols[i, i])) for i in range(3)]
loss = 0.5 * (np.mean(ce_rows) + np.mean(ce_cols))

# Test 1: the product's path, ranked against all four captions.
rank = sorted(((cos(z["poe"], v), k) for k, v in CAPTIONS.items()), reverse=True)
# Test 2: the path gap against the text gap.
path_gap = (np.array(z["joint"]) - np.array(z["poe"])).tolist()
text_gap = (np.array(CAPTIONS["a 3 and a 7, apart"]) - np.array(CAPTIONS["one 3"]) - np.array(CAPTIONS["one 7"])).tolist()

key = {
    "what_this_is": "a hand-set toy; it shows where each term sits and predicts nothing about the digit models",
    "weights": {"u": U, "v": V, "r_interaction": R, "x0": X0, "encoder_E": E, "captions": CAPTIONS, "tau": TAU},
    "steps": steps,
    "score_paths": paths,
    "embeddings": z,
    "contrastive": {"rows": rows, "cols": cols,
                    "cos": [[round(cos(z[r], CAPTIONS[c]), 4) for c in cols] for r in rows],
                    "logits": [[round(v, 3) for v in row] for row in logits],
                    "softmax_rows": np.round(p_rows, 4).tolist(), "softmax_cols": np.round(p_cols, 4).tolist(),
                    "ce_rows": [round(v, 4) for v in ce_rows], "ce_cols": [round(v, 4) for v in ce_cols],
                    "loss": round(float(loss), 4)},
    "test_sum_lands_on_joint": {"ranking": [[k, round(c, 4)] for c, k in rank], "top1": rank[0][1]},
    "test_gap_predicts_term": {"path_gap": path_gap, "text_gap": text_gap, "cos": round(cos(path_gap, text_gap), 4)},
}
(HERE / "miniature_answer_key.json").write_text(json.dumps(key, indent=1))
print(json.dumps({k: key[k] for k in ("score_paths", "embeddings", "test_sum_lands_on_joint", "test_gap_predicts_term")}, indent=0))
print(json.dumps(key["contrastive"]))
for s in steps:
    print(s)
