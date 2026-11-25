"""Every value printed on any page or figure of Paper 1's ladder comes from here. Writes data/paper1_numbers.json.

Run from anywhere:  ~/.claude/venvs/figures/bin/python scripts/make_paper1_numbers.py

Four kinds of value, kept apart in the JSON:
  sizes    read from the design and code files they come from; each carries the file and the line it was read on, and the
           script stops if the quoted text is no longer on that line.
  results  read only from report pages (or, where marked, a review or design file), with file and line, checked the same way.
  toy      the product-of-experts toy of the PoE ladder (product-of-experts-interaction-term/scripts/poe_toy.py), imported
           read-only. Counts already in that ladder's JSON are copied from it, never recomputed under a new name.
  push     what this ladder adds to the toy: the exact checker on the toy and its push, on the product and on the region merge.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True                      # importing the PoE toy must leave its folder untouched
HERE = Path(__file__).resolve().parent
SCENE = HERE.parent
REPO = SCENE.parents[3]                              # .../concept-composition
POE = SCENE.parent / "product-of-experts-interaction-term"
sys.path.insert(0, str(POE / "scripts"))
import poe_toy as T  # noqa: E402

OUT = SCENE / "data" / "paper1_numbers.json"


def cite(rel, needle):
    """Return 'path:line' for the first line of rel holding needle; stop if it is not there."""
    lines = (REPO / rel).read_text().splitlines()
    for i, ln in enumerate(lines, 1):
        if needle in ln:
            return f"{rel}:{i}"
    raise SystemExit(f"cite failed: {needle!r} not found in {rel}")


def constant(rel, name):
    m = re.search(rf"^{name}\s*=\s*([0-9_.]+)", (REPO / rel).read_text(), re.M)
    if not m:
        raise SystemExit(f"constant {name} not found in {rel}")
    return float(m.group(1).replace("_", "")), cite(rel, f"{name} =")


res = {}

# ---------------------------------------------------------------- sizes, from the designs and the code
DES05 = "artifacts/notes/progress/method-search/05-two-experts-by-colour/design.md"
REV01C = "artifacts/notes/progress/method-search/01c-clevr-paste-first/review.md"
ROWS = "2-classifier-guidance/scripts/method_search/paper1_rows.py"
PC = "2-classifier-guidance/scripts/mnist/pixel_checker.py"
CU = "2-classifier-guidance/scripts/mnist/canvas_unet.py"
sizes = {
    "clevr_side_px": {"value": 64, "unit": "pixels, square", "source": cite(DES05, "squashed to 64 by 64")},
    "clevr_expert_model": {"value": "edm2-img64-xs", "source": cite(DES05, "`edm2-img64-xs`")},
    "clevr_expert_images_seen": {"value": 2000896, "unit": "training images seen per expert",
                                 "source": cite(DES05, "trained for 2,000,896 images")},
    "clevr_expert_minutes": {"value": 77, "unit": "minutes on one RTX PRO 6000 Blackwell card (position 1 expert, step M7)",
                             "source": cite(REV01C, "trained on 2,000,896 images in 77 minutes")},
    "clevr_renders_per_count": {"value": 5000, "unit": "renders per object count, 1 to 3, per colour",
                                "source": cite(DES05, "5,000 per count, 15,000 in all")},
    "clevr_counts_per_expert": {"value": [1, 2, 3], "source": cite(DES05, "takes a count label k, from 1 to 3")},
    "digit_canvas": {"value": [1, 32, 64], "unit": "channels, height, width in pixels",
                     "source": cite("2-classifier-guidance/artifacts/scenes/mnist-guidance-explained/4-the-derivation.md",
                                    "The 1 x 32 x 64 image")},
    "digit_experts": {"value": ["e04-either: digits 0 to 4, either side", "e59-either: digits 5 to 9, either side"],
                      "source": cite("2-classifier-guidance/report/the-method-search/does-composing-two-digit-models-beat-sampling-and-pasting.md",
                                     "`e04-either` draws digits 0 to 4")},
    "sampler_steps": {"value": 50, "unit": "DDIM steps", "source": cite(CU, "def sample(")},
}
for key, name in (("push_strength", "PUSH_STRENGTH"), ("push_cap", "PUSH_CAP"), ("layout_lock_t", "LOCK_T"), ("eta", "ETA")):
    v, src = constant(ROWS, name)
    sizes[key] = {"value": v, "source": src}
sizes["push_strength"]["unit"] = "lambda, multiplies sigma_t times the checker gradient"
sizes["push_cap"]["unit"] = "largest push as a share of the noise prediction's length, per canvas"
sizes["layout_lock_t"]["unit"] = "noise step t (of 1000) after which each expert's side is held"
sizes["eta"]["unit"] = "DDIM noise setting, 1 adds fresh noise every step"


def conv(cin, cout, k=3):
    return cin * cout * k * k + cout


def lin(cin, cout):
    return cin * cout + cout


def head_weights(out, pair_aware):
    """PixelHead in pixel_checker.py: stem of three convolutions, two 128-channel convolutions, FiLM from 16 step features
    (plus a 64-number pair vector when pair-aware), mean and max pooling, a two-layer MLP."""
    extra = 64 if pair_aware else 0
    n = conv(1, 64) + conv(64, 64) + conv(64, 128) + conv(128, 128) + conv(128, 128)
    n += (10 * 64 if pair_aware else 0) + lin(16 + extra, 256) + lin(256 + 16 + extra, 256) + lin(256, out)
    return n


sizes["checker_layers"] = {"value": "1 -> 64 -> 64 (stride 2) -> 128 (stride 2) -> 128 -> 128 channels; the step scales and shifts the "
                                    "middle maps; mean and max pooled to 256 numbers; MLP 256 -> answers",
                           "source": cite(PC, "self.stem = nn.Sequential")}
sizes["checker_grid_after_stem"] = {"value": [8, 16], "unit": "rows, columns after two stride-2 convolutions on 32 x 64",
                                    "source": cite(PC, "nn.Conv2d(64, 128, 3, stride=2")}
sizes["pair_checker_weights"] = {"value": head_weights(1, True), "unit": "trained weights, counted from the layers",
                                 "source": cite(PC, "self.film = nn.Linear")}
sizes["separate_checker_weights"] = {"value": head_weights(10, False), "unit": "trained weights, counted from the layers",
                                     "source": cite(PC, "self.mlp = nn.Sequential")}
sizes["pair_push_line"] = {"value": "return -s * _rate(t, x) * g * inside", "source": cite(PC, "return -s * _rate(t, x) * g * inside")}
sizes["cap_line"] = {"value": "per_sample_cap", "source": cite(CU, "def per_sample_cap")}
res["sizes"] = sizes

# ---------------------------------------------------------------- results, from report pages only (with two marked exceptions)
PASTE_DIG = "2-classifier-guidance/report/the-method-search/does-composing-two-digit-models-beat-sampling-and-pasting.md"
PASTE_CLEVR = "2-classifier-guidance/report/the-method-search/does-sample-and-paste-already-solve-small-clevr.md"
TWO_EXP = "2-classifier-guidance/report/classifier-guided-poe/32-mnist-two-experts.md"
FINAL = "2-classifier-guidance/report/classifier-guided-poe/can-two-separately-trained-digit-models-compose-while-sampling.md"
PS = "artifacts/notes/progress/problem-statements.md"
REV05 = "artifacts/notes/progress/method-search/05-two-experts-by-colour/review.md"
res["results"] = {
    "digits_halves_vs_paste": {
        "what": "clean digit pairs of 320 canvases (5 tuning pairs x 64 seeds), digit judge judge-v3, W&B vekm3sfh; null by the written rule",
        "paste": 287, "method_no_push": 259, "method_checker_push": 295, "method_random_push": 258,
        "lead_push_over_paste": 8, "bar": 13.54, "lift_push_over_no_push": 36,
        "source": cite(PASTE_DIG, "| Method with the checker push | 295 |"), "verdict_source": cite(PASTE_DIG, "lead of 8 is under the bar of 13.54")},
    "clevr_fixed_places_paste": {
        "what": "pasted images with every asked object in place, fixed separated places, step M7", "two_experts": [1024, 1024],
        "three_experts": [1024, 1024], "paste_rate": 1.0, "bar": 0.906, "each_expert_alone": [256, 256],
        "source": cite(PASTE_CLEVR, "| Two experts | 1,024 | 1.0 | 1.0 | not asked |")},
    "digits_either_side_plain_product": {
        "what": "two clear right digits of 320 seeds, experts trained apart drawing on either side, no regions",
        "plain_product": 0, "both_pushes": 4, "source": cite(TWO_EXP, "0 of 320 seeds when each draws its digit on either side")},
    "digits_shared_canvas_best": {
        "what": "clean pairs of 320 on one shared canvas with no halves, five tuning pairs; bar 288", "value": 121, "bar": 288,
        "source": cite(PS, "121 of 320 (38%)"), "note": "problem statement, not a report page; the finding below holds the same number"},
    "digits_final_void": {
        "what": "clean pairs of 1,280 on 20 unseen pairs, fixed halves; ON THE VOID LIST, out of any claim", "value": 1200, "of": 1280,
        "source": cite(PS, "The strongest count is on the void list"), "finding": cite(FINAL, "1,200")},
    "two_colour_clevr": {
        "what": "two-colour CLEVR experts (step M10): renders under way, no expert trained, no sample exists",
        "source": cite(REV05, "no model has trained"), "note": "a review file, the only record of this step"},
}

# ---------------------------------------------------------------- toy values already in the PoE ladder's JSON (copied, not recomputed)
poe_json = json.loads((POE / "data" / "poe_interaction_numbers.json").read_text())
oc = poe_json["outcomes_of_4000"]
copied = {
    "product": oc["product of the two experts"]["two apart"],
    "joint_every_step": oc["product plus the interaction term at every step (the joint's own score)"]["two apart"],
    "regions_visible_coin_held": oc["each expert keeps its own cell, whole canvas visible, coin layout held (no hiding)"]["two apart"],
    "regions_hidden_coin_held": oc["each expert keeps its own cell, other cell hidden, coin layout held"]["two apart"],
    "regions_hidden_vote_held": oc["each expert keeps its own cell, other cell hidden, layout voted over steps 0 to 9 then held (E0e)"]["two apart"],
    "regions_hidden_revoted": oc["each expert keeps its own cell, other cell hidden, layout re-voted at every step (E0d-b)"]["two apart"],
    "product_breakdown": oc["product of the two experts"],
}
res["toy_from_poe_ladder"] = {"source": "product-of-experts-interaction-term/data/poe_interaction_numbers.json, key outcomes_of_4000",
                              "two_apart_of_4000": copied}

# ---------------------------------------------------------------- the push on the toy (new)
# The checker is trained on pastes of the experts' own samples. On the toy, expert A draws a 4 in one cell, expert B a 7 in one
# cell. Pasting them gives three kinds of canvas: the two digits in different cells (labelled "both present, apart"), the two
# in the same cell (fused), and one expert's sample alone. The checker answers "both present, apart" with half the training
# canvases positive and half negative. Its exact (Bayes-optimal) answer at every noise level is computable, so the push is exact.
POS = T.JOINT_MEANS                                                        # (4L,7R), (7L,4R)
FUSED = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], float)                     # both in one cell
NEG = np.vstack([T.A_MEANS, T.B_MEANS, FUSED])                             # singles and fused, equal weight
MIX = np.vstack([POS, NEG])
MIX_W = np.concatenate([np.full(len(POS), 0.5 / len(POS)), np.full(len(NEG), 0.5 / len(NEG))])
res["toy_checker"] = {"positives": "4 and 7 in different cells (2 canvases), half the training set",
                      "negatives": "one digit alone (4 canvases) and both in one cell (2 canvases), half the training set",
                      "built_from": "pastes of the toy experts' own samples; no canvas the joint was trained on is used beyond what pasting makes"}


def wscore(x, ab, means, w):
    """Exact score of a weighted mixture of narrow Gaussians, noised to signal level ab."""
    a = np.sqrt(ab); v = ab * T.SIGMA0 ** 2 + (1 - ab)
    d = x[:, None, :] - a * means[None]
    logit = -(d ** 2).sum(-1) / (2 * v) + np.log(w)[None]
    logit -= logit.max(1, keepdims=True)
    r = np.exp(logit); r /= r.sum(1, keepdims=True)
    return -(r[..., None] * d).sum(1) / v


def grad_log_checker(x, ab):
    """grad log C_t(both apart | x_t) = score of the positives minus score of the checker's whole training mixture."""
    return T.score(x, ab, POS)[0] - wscore(x, ab, MIX, MIX_W)


def checker_prob(x, ab):
    a = np.sqrt(ab); v = ab * T.SIGMA0 ** 2 + (1 - ab)
    def logp(m, w):
        d = x[:, None, :] - a * m[None]
        lg = -(d ** 2).sum(-1) / (2 * v) + np.log(w)[None]
        mx = lg.max(1, keepdims=True)
        return (mx[:, 0] + np.log(np.exp(lg - mx).sum(1)))
    lp, lm = logp(POS, np.full(len(POS), 0.5 / len(POS))), logp(MIX, MIX_W)
    return np.exp(lp - lm)


def with_push(base_fn, lam, lo=0, hi=T.K):
    def f(x, k):
        ab = T.ABAR[k]
        e = base_fn(x, k)
        if lo <= k < hi and lam != 0:
            e = e - lam * np.sqrt(1 - ab) * grad_log_checker(x, ab)
        return e
    return f


N = 4000
# Same starting noise as the PoE ladder: its script draws the profile states first, then init and noise. Reproduce that order.
rng_poe = np.random.default_rng(0)
for k in range(T.K):
    rng_poe.integers(2, size=N); rng_poe.standard_normal((N, 4))
init = rng_poe.standard_normal((N, 4)); noise = rng_poe.standard_normal((T.K, N, 4))
poe = lambda x, k: T.eps_of(T.poe_score(x, T.ABAR[k]), T.ABAR[k])
mix_base = lambda x, k: T.eps_of(wscore(x, T.ABAR[k], MIX, MIX_W), T.ABAR[k])


def two_apart(fn):
    return int((T.classify(T.sample(fn, noise, init)) == "two apart").sum())


def tally(fn):
    c = T.classify(T.sample(fn, noise, init))
    return {n: int((c == n).sum()) for n in ["two apart", "both in one cell", "one digit", "other"]}


check_product = two_apart(poe)
check_regions = two_apart(T.hide_eps("coin", True))
assert check_product == copied["product"], (check_product, copied["product"])
assert check_regions == copied["regions_hidden_coin_held"], (check_regions, copied["regions_hidden_coin_held"])

lams = [0, 0.5, 1, 2, 4, 10]
push = {"lambdas": lams, "two_apart_of_4000": {}, "same_noise_as_poe_ladder": True,
        "reproduces_poe_ladder": {"product": check_product, "regions_hidden_coin_held": check_regions}}
bases = {"product of the two experts, plus the push": poe,
         "each expert on its own cell (other cell hidden, coin layout held), plus the push": T.hide_eps("coin", True)}
for name, base in bases.items():
    push["two_apart_of_4000"][name] = [two_apart(with_push(base, l)) for l in lams]
push["outcomes_at_lambda_1"] = {name: tally(with_push(base, 1.0)) for name, base in bases.items()}
push["bayes_check"] = {"what": "the checker's own training mixture as the base, plus the push at lambda 1; Bayes says this is the "
                               "positives' own score, so it should draw two apart every time",
                       "two_apart_of_4000": two_apart(with_push(mix_base, 1.0))}
push["windows_product_lambda_4"] = {"steps 0 to 9 only": two_apart(with_push(poe, 4.0, 0, 10)),
                                    "steps 10 to 49 only": two_apart(with_push(poe, 4.0, 10, T.K)),
                                    "every step": two_apart(with_push(poe, 4.0))}
res["toy_push"] = push

# How close the checker's gradient is to the interaction term, along the run, at states of the joint's own noised marginal
prof = []
rng2 = np.random.default_rng(1)
for k in range(0, T.K, 1):
    ab = T.ABAR[k]; a = np.sqrt(ab); v = ab * T.SIGMA0 ** 2 + (1 - ab)
    comp = rng2.integers(2, size=2000)
    x = a * POS[comp] + np.sqrt(v) * rng2.standard_normal((2000, 4))
    I = T.joint_score(x, ab) - T.poe_score(x, ab)
    g = grad_log_checker(x, ab)
    cos = (I * g).sum(1) / (np.linalg.norm(I, axis=1) * np.linalg.norm(g, axis=1) + 1e-12)
    prof.append({"step": k, "abar": float(ab), "median_cosine_I_vs_checker_gradient": float(np.median(cos)),
                 "median_norm_ratio_checker_over_I": float(np.median(np.linalg.norm(g, axis=1) / (np.linalg.norm(I, axis=1) + 1e-12)))})
res["toy_push"]["gradient_vs_interaction_term"] = prof

# ---------------------------------------------------------------- one worked state, for the maths page
k = 26
ab = float(T.ABAR[k]); a = np.sqrt(ab); sig = float(np.sqrt(1 - ab)); v = ab * T.SIGMA0 ** 2 + (1 - ab)
x = (a * POS[0] + np.sqrt(v) * np.random.default_rng(7).standard_normal(4))[None]
sA, sB, s0 = T.score(x, ab, T.A_MEANS)[0], T.score(x, ab, T.B_MEANS)[0], T.score(x, ab, T.BASE_MEANS)[0]
eA, eB, e0 = (T.eps_of(s, ab) for s in (sA, sB, s0))
ePoE = e0 + (eA - e0) + (eB - e0)
sJ = T.joint_score(x, ab)
I = sJ - (sA + sB - s0)
merge = np.empty_like(x)                                           # layout: expert A on the left cell, expert B on the right
merge[:, T.LEFT] = eA[:, T.LEFT]; merge[:, T.RIGHT] = eB[:, T.RIGHT]
g = grad_log_checker(x, ab)
lam = 1.0
ehat_prod = ePoE - lam * sig * g
ehat_merge = merge - lam * sig * g
x0_hat = (x - sig * ehat_merge) / a
abp = float(T.ABAR[k + 1])
r = lambda z: [round(float(t), 3) for t in z.ravel()]
res["worked_state"] = {
    "step": k, "abar": round(ab, 4), "sigma_t": round(sig, 4), "next_abar": round(abp, 4), "lambda": lam,
    "order": "(4 in left, 7 in left, 4 in right, 7 in right)",
    "x_t": r(x), "eps_A": r(eA), "eps_B": r(eB), "eps_empty": r(e0), "eps_PoE": r(ePoE),
    "score_A": r(sA), "score_B": r(sB), "score_empty": r(s0), "score_joint": r(sJ), "I_t_score_units": r(I),
    "r_t_noise_units": r(-sig * I), "eps_joint": r(T.eps_of(sJ, ab)),
    "merge_A_left_B_right": r(merge), "checker_prob_both_apart": round(float(checker_prob(x, ab)[0]), 4),
    "grad_log_checker": r(g), "push_noise_units": r(-lam * sig * g),
    "eps_hat_product_plus_push": r(ehat_prod), "eps_hat_merge_plus_push": r(ehat_merge),
    "x0_estimate_from_merge_plus_push": r(x0_hat),
    "x_next_mean_part": r(np.sqrt(abp) * x0_hat),
    "eta_sigma": round(float(np.sqrt((1 - abp) / (1 - ab)) * np.sqrt(1 - ab / abp)), 4),
    "x_next_before_fresh_noise": r(np.sqrt(abp) * x0_hat + np.sqrt(max(1 - abp - (1 - abp) / (1 - ab) * (1 - ab / abp), 0)) * ehat_merge),
    "eps_direction_coefficient": round(float(np.sqrt(max(1 - abp - (1 - abp) / (1 - ab) * (1 - ab / abp), 0))), 4),
    "cosine_I_vs_grad_log_checker": round(float((I * g).sum() / (np.linalg.norm(I) * np.linalg.norm(g))), 3),
    "eps_A_check_minus_sigma_times_score_A": r(-sig * sA),
    "note": "x_t is the joint's 4-left-7-right canvas noised to step 26 with a fixed draw (seed 7); every vector is exact on the toy",
}

OUT.write_text(json.dumps(res, indent=1))
print("pair checker weights", sizes["pair_checker_weights"]["value"], "separate", sizes["separate_checker_weights"]["value"])
for n, vals in push["two_apart_of_4000"].items():
    print(vals, n)
print("bayes check", push["bayes_check"]["two_apart_of_4000"], "windows", push["windows_product_lambda_4"])
print(json.dumps(res["worked_state"], indent=0)[:1500])
print([ (p["step"], round(p["median_cosine_I_vs_checker_gradient"], 2), round(p["median_norm_ratio_checker_over_I"], 2)) for p in prof[::5]])
