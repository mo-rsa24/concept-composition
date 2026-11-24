"""Every full-scale number the digit-map checker ladder prints, each with its source.

Reads the canvases rebuilt by build_env.py into numbers/env/ (data hash f70b0fb3...), and copies the noise schedule,
the 16 timesteps, the time features and the head's layer sizes from the code they live in. Writes numbers.json and the
example canvases (clean and noised) as .npy files under numbers/examples/.

Every value is one of: counted (computed here from the data or the code's constants), printed (copied from a results
file named beside it), or illustrative (a noise draw made here for display; the checker's own draws come from torch).

Run: ~/.claude/venvs/figures/bin/python -I make_numbers.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ENV = HERE / "env"
EXAMPLES = HERE / "examples"
DATA_HASH = "f70b0fb362e13e255ad332ba14ca93dfb5cf051a5dfd3c86eb30482931d5a5f3"
KINDS = ("apart", "overlaid", "same_half", "single", "wrong")
ARRAYS = ("x", "asked", "holds", "corner", "src", "kind", "target", "n_digits", "apart")

# 2-classifier-guidance/scripts/mnist/canvas_unet.py lines 35-38: linear betas over T = 1000, cumprod in float64
T = 1000
BETAS = np.linspace(1e-4, 0.02, T, dtype=np.float64)
ABAR = np.cumprod(1 - BETAS).astype(np.float32)
# 2-classifier-guidance/scripts/abstract_checkers/v2_learn.py line 46
SDXL_MATCHED = [712, 696, 680, 664, 648, 632, 616, 601, 585, 570, 555, 539, 524, 510, 495, 480]
SHOWN_STEPS = [712, 616, 539, 480]          # four of the sixteen, for the noise strip
H, W = 32, 64
PIXELS = H * W
DISPLAY_NOISE_SEED = 0                       # illustrative draw; the checker draws its own noise with torch


def time_features(t):
    """2-classifier-guidance/scripts/mnist/checker.py lines 69-73, in numpy."""
    a = ABAR[np.asarray(t)].astype(np.float32)
    lsnr = np.log(a / (1 - a))[:, None]
    f = np.exp(np.linspace(0, 3, 8, dtype=np.float32))[None] * lsnr / 10
    return np.concatenate([np.sin(f), np.cos(f)], 1), lsnr[:, 0]


def conv_out(n, k=3, s=1, p=1):
    return (n + 2 * p - k) // s + 1


def layers():
    """PixelHead (pixel_checker.py lines 33-56) with the first conv swapped to 2 input channels (v2_learn.py lines 77-82)."""
    rows, h, w = [], H, W
    spec = [("stem conv 1", 2, 64, 1), ("stem conv 2", 64, 64, 2), ("stem conv 3", 64, 128, 2),
            ("c1", 128, 128, 1)]
    for name, cin, cout, s in spec:
        h, w = conv_out(h, s=s), conv_out(w, s=s)
        rows.append({"layer": name, "kind": "conv 3x3", "stride": s, "in_channels": cin, "out_channels": cout,
                     "out_shape": [cout, h, w], "weights": cin * cout * 9 + cout})
    rows.append({"layer": "film", "kind": "linear 16 -> 256 (scale 128, shift 128)", "out_shape": [256],
                 "weights": 16 * 256 + 256})
    rows.append({"layer": "c2", "kind": "conv 3x3", "stride": 1, "in_channels": 128, "out_channels": 128,
                 "out_shape": [128, h, w], "weights": 128 * 128 * 9 + 128})
    rows.append({"layer": "pool", "kind": "mean and max over the 8 x 16 = 128 positions", "out_shape": [256], "weights": 0})
    rows.append({"layer": "mlp 1", "kind": "linear 272 -> 256 (256 pooled + 16 time features)", "out_shape": [256],
                 "weights": 272 * 256 + 256})
    rows.append({"layer": "mlp 2", "kind": "linear 256 -> 1", "out_shape": [1], "weights": 256 + 1})
    total = sum(r["weights"] for r in rows)
    assert total == 481_665, total                                     # the count measured on the cluster
    return rows, total


def load():
    manifest = json.loads((ENV / "manifest.json").read_text())
    assert manifest["data_hash"] == DATA_HASH, manifest["data_hash"]
    with np.load(ENV / "dataset.npz") as z:
        return {s: {n: z[f"{s}__{n}"] for n in ARRAYS} for s in ("train", "seen_test", "unseen_test")}, manifest


def take(split, mask):
    return {k: v[mask] for k, v in split.items()}


def inner_splits(data):
    """v2_learn.py lines 69-74."""
    no_four = lambda s: (s["asked"] <= 3).all(1) & (s["holds"] <= 3).all(1)
    has_four = (data["seen_test"]["asked"] == 4).any(1)
    return (take(data["train"], no_four(data["train"])), take(data["seen_test"], no_four(data["seen_test"])),
            take(data["seen_test"], has_four))


def relabel(split, task):
    """v2_learn.py lines 296-304."""
    if task == "count":
        out = dict(split)
        out["target"] = split["apart"].copy()
        return out
    return take(split, np.isin(split["kind"], [KINDS.index("apart"), KINDS.index("wrong")]))


def describe(split):
    return {"canvases": int(len(split["x"])), "label_1": int(split["target"].sum()),
            "label_0": int((split["target"] == 0).sum()),
            "by_kind": {k: int((split["kind"] == i).sum()) for i, k in enumerate(KINDS) if (split["kind"] == i).any()},
            "asked_pairs": sorted({f"{a}-{b}" for a, b in split["asked"].tolist()})}


def first(split, asked, kind):
    hit = np.flatnonzero((split["asked"][:, 0] == asked[0]) & (split["asked"][:, 1] == asked[1])
                         & (split["kind"] == KINDS.index(kind)))
    return int(hit[0])


def example(split_name, split, i, xt_noise):
    x0 = split["x"][i].astype(np.float32) / 127.5 - 1            # pixel_checker.py line 152, to_x
    noised = {}
    for t in SHOWN_STEPS:
        a = float(ABAR[t])
        noised[str(t)] = (np.sqrt(a) * x0 + np.sqrt(1 - a) * xt_noise).astype(np.float32)
    return x0, noised, {"split": split_name, "index": i, "asked": split["asked"][i].tolist(),
                        "holds": split["holds"][i].tolist(), "kind": KINDS[int(split["kind"][i])],
                        "corner": split["corner"][i].tolist(), "mnist_source_index": split["src"][i].tolist(),
                        "count_label": int(split["apart"][i]), "identity_label": (int(split["target"][i])
                        if KINDS[int(split["kind"][i])] in ("apart", "wrong") else None),
                        "ink_pixels": int((split["x"][i] > 0).sum())}


REVIEWS = "artifacts/notes/progress/abstract-checkers/"


def final_results():
    """Loop 8's digit results. The count checker's scores are read from real/count-final.json (job 94108); every other
    number is quoted from the review named beside it, unchanged."""
    cf = json.loads((HERE / "real" / "count-final.json").read_text())
    reader = {r: {"digits_0_to_4": round(cf[r]["seen_test"]["balanced_accuracy"], 3),
                  "digits_5_to_9": round(cf[r]["unseen_test"]["balanced_accuracy"], 3),
                  "gap": round(cf[r]["gap"], 3), "overfit_right_of_16": cf[r]["overfit"]["right_of_16"]}
              for r in ("map", "raw")}
    return {
        "count_checker": {"job": 94108, "wandb_run": cf["wandb_run"], "verdict": cf["verdict"],
                          "file": "numbers/real/count-final.json, copied from /datasets/mmolefe/poe_repair_min/loop-8/v2/count-final/",
                          "trained_on": "all 10 pairs among digits 0 to 4", "scored_on": "the 10 pairs among digits 5 to 9, read once",
                          "bars": {"digits_5_to_9_at_least": 0.75, "gap_at_most": 0.05}, **reader,
                          "digits_5_to_9_per_step": [round(v, 4) for v in cf["map"]["unseen_test"]["per_step"]],
                          "review": REVIEWS + "05-does-it-learn/review.md"},
        "identity_checker": {"job": 94085, "pairs_holding_4": 0.555, "bar": 0.75, "verdict": "miss",
                             "review": REVIEWS + "05-does-it-learn/review.md"},
        "right_digits_checker": {"job": 94153, "digits_5_to_9": 0.556, "digits_0_to_4": 0.601,
                                 "review": REVIEWS + "08-remove-one-checker/review.md"},
        "push": {"job": 94134, "wandb_run": "qhusod91", "verdict": "support", "pairs": "the 10 pairs among digits 5 to 9",
                 "seeds": "1000 to 1031", "canvases": 320, "b": 0.4, "b_chosen_from": [0.1, 0.2, 0.4],
                 "b_chosen_on": "the 10 seen pairs among digits 0 to 4 at seeds 0 to 31 (job 94133): 8, 20 and 40 clean pairs of 320",
                 "sampler_steps": "15 to 25 of 50", "timesteps": "693 to 489",
                 "clean_pairs": {"plain": 0, "push": 43, "random_push_same_length": 1, "best_of_4_seeds": 0},
                 "two_separate_digits": {"plain": 3, "push": 161, "random_push_same_length": 4, "best_of_4_seeds": 2},
                 "standard_error_of_lead": 7.58, "rerun_job": 94161, "rerun_clean_pairs_push": 41,
                 "review": REVIEWS + "06-does-it-compose/review.md"},
        "remove_one": {"job": 94186, "wandb_run": "j4anxfgy", "verdict": "null by the written rule (board question 75)",
                       "canvases_per_row": 960, "own_bar_points": 6.25, "other_column_limit_points": 4.2,
                       "rows": {"count": {"own_fall": 28.1, "random_fall": 25.8, "largest_other_move": "order rises 6.8"},
                                "right digits": {"own_fall": 2.6, "random_fall": 10.8, "largest_other_move": "order rises 3.0"},
                                "order": {"own_fall": 40.4, "random_fall": 45.9, "largest_other_move": "right digits falls 6.9"}},
                       "review": REVIEWS + "08-remove-one-checker/review.md"},
        "summary": REVIEWS + "09-summary/executive-summary.md",
    }


def main():
    data, manifest = load()
    tr, seen, hold4 = inner_splits(data)
    assert set(np.unique(tr["holds"]).tolist()) <= {-1, 0, 1, 2, 3}
    branches = {}
    for task in ("count", "identity"):
        branches[task] = {"train_pairs_0_to_3": describe(relabel(tr, task)),
                          "test_trained_pairs": describe(relabel(seen, task)),
                          "test_pairs_holding_4": describe(relabel(hold4, task))}

    feats, lsnr = time_features(SDXL_MATCHED)
    sched = [{"t": t, "abar": round(float(ABAR[t]), 6), "sqrt_abar": round(float(np.sqrt(ABAR[t])), 4),
              "sqrt_one_minus_abar": round(float(np.sqrt(1 - ABAR[t])), 4), "log_snr": round(float(l), 4)}
             for t, l in zip(SDXL_MATCHED, lsnr)]
    rows, total = layers()

    EXAMPLES.mkdir(exist_ok=True)
    rng = np.random.default_rng(DISPLAY_NOISE_SEED)
    noise = rng.standard_normal((H, W)).astype(np.float32)
    np.save(EXAMPLES / "display-noise.npy", noise)
    picks = {"apart_1_3": ("train_inner", tr, first(tr, (1, 3), "apart")),
             "wrong_1_3": ("train_inner", tr, first(tr, (1, 3), "wrong")),
             "overlaid_1_3": ("train_inner", tr, first(tr, (1, 3), "overlaid")),
             "same_half_1_3": ("train_inner", tr, first(tr, (1, 3), "same_half")),
             "single_1_3": ("train_inner", tr, first(tr, (1, 3), "single")),
             "apart_3_4": ("seen_test_holding_4", hold4, first(hold4, (3, 4), "apart")),
             "wrong_3_4": ("seen_test_holding_4", hold4, first(hold4, (3, 4), "wrong"))}
    examples = {}
    for name, (split_name, split, i) in picks.items():
        x0, noised, info = example(split_name, split, i, noise)
        np.save(EXAMPLES / f"{name}-clean.npy", x0)
        for t, xt in noised.items():
            np.save(EXAMPLES / f"{name}-t{t}.npy", xt)
        examples[name] = info

    out = {
        "data": {"hash": manifest["data_hash"], "source": "numbers/env/manifest.json, rebuilt by build_env.py",
                 "canvas": [H, W], "pixels": PIXELS, "pixel_range": "-1 (background) to +1 (ink), to_x in pixel_checker.py",
                 "counts": manifest["counts"]},
        "branches": branches,
        "branches_note": "counted from the rebuilt canvases with v2_learn.py's inner_splits and relabel; the inner training "
                         "set drops every canvas holding a 4, so wrong canvases for pairs among 0 to 3 keep only a 0 to 3 "
                         "digit as the stranger",
        "noise": {"T": T, "betas": "linear 1e-4 to 0.02", "timesteps": SDXL_MATCHED, "schedule": sched,
                  "shown_in_strip": SHOWN_STEPS,
                  "source": "canvas_unet.py lines 35-38 and v2_learn.py line 46, counted here"},
        "time_features": {"values": np.round(feats, 4).tolist(), "frequencies": np.round(np.exp(np.linspace(0, 3, 8)), 4).tolist(),
                          "source": "checker.py lines 69-73, counted here"},
        "head": {"layers": rows, "weights": total, "source": "pixel_checker.py lines 33-56 and v2_learn.py lines 77-82, "
                 "counted here and equal to the 481,665 measured by the smoke stage"},
        "training": {"loss": "binary cross-entropy on the logit", "optimiser": "Adam, learning rate 0.001",
                     "batch": 256, "label_1_per_batch": 128, "updates": 4000,
                     "timestep": "one of the 16 drawn per example", "source": "v2_learn.py lines 110-133"},
        "push": {"b": 0.4, "rule": "eps_hat = eps_product - ||eps_product|| * b * grad / ||grad||",
                 "source": "artifacts/notes/progress/abstract-checkers/03-design.md, section 'How the push is computed'"},
        "examples": examples,
        "examples_note": "clean canvases are real; the noised copies use one numpy draw (seed 0) shared across the four "
                         "timesteps, so only the noise level changes along the strip; the checker's own draws differ",
        "results_printed": {
            "combined_checker_sdxl_noise": {
                "job": 94080, "file": "/datasets/mmolefe/poe_repair_min/loop-8/v2/sdxl-window/rehearse.json",
                "review": "artifacts/notes/progress/abstract-checkers/05-does-it-learn/review.md",
                "map_reader": {"overfit_right_of_16": 16, "balanced_accuracy_pairs_holding_4": 0.704,
                               "gap_to_trained_pairs": 0.029, "apart_vs_wrong": 0.511, "share_called_1": 0.636,
                               "least_noisy_step": 0.81, "noisiest_step": 0.59},
                "read_afterwards_as_count": {"map_reader": 0.766, "raw_reader": 0.776,
                                             "note": "not planned before the run; supports no claim"}},
            "split_run": {"job": 94085, "file": "/datasets/mmolefe/poe_repair_min/loop-8/v2/sdxl-split/split.json",
                          "review": "artifacts/notes/progress/abstract-checkers/05-does-it-learn/review.md, 'Two checkers at SDXL's noise level'",
                          "state": "split.json lives on the cluster; the values below are quoted from the review",
                          "quoted": {"count": {"map": {"trained_pairs": 0.781, "holding_4_ba": 0.774, "gap": 0.007, "overfit_right_of_16": 16},
                                               "raw": {"trained_pairs": 0.793, "holding_4_ba": 0.775, "gap": 0.018, "overfit_right_of_16": 16},
                                               "verdict": "pass"},
                                     "identity": {"map": {"trained_pairs": 0.603, "holding_4_ba": 0.555, "gap": 0.048, "overfit_right_of_16": 16},
                                                  "raw": {"trained_pairs": 0.606, "holding_4_ba": 0.556, "gap": 0.049, "overfit_right_of_16": 16},
                                                  "verdict": "miss"}}},
        },
        "results_final": final_results(),
    }
    log_lines = HERE / "split-94085-log-lines.jsonl"
    if log_lines.exists():
        rows = [json.loads(l) for l in log_lines.read_text().splitlines() if l.strip()]
        out["results_printed"]["split_run"]["from_job_log"] = {
            "file": "/datasets/mmolefe/poe_repair_min/loop-8/logs/l8-v2-94085.log, the lines starting {\"task\"; "
                    "copied verbatim to numbers/split-94085-log-lines.jsonl at 07:19 on 2026-11-24 while the job ran",
            "rows": rows,
            "missing": sorted({"count", "identity"} - {r["task"] for r in rows})}
    split_file = HERE / "split.json"
    if split_file.exists():
        s = json.loads(split_file.read_text())
        out["results_printed"]["split_run"]["state"] = "read from numbers/split.json"
        out["results_printed"]["split_run"]["summary"] = {
            task: {reader: {k: s[task][reader][k] if k in s[task][reader] else None for k in ("gap", "all", "trained")}
                   | {"seen_ba": s[task][reader]["inner_seen"]["balanced_accuracy"],
                      "holding_4_ba": s[task][reader]["inner_unseen"]["balanced_accuracy"],
                      "holding_4_per_step": s[task][reader]["inner_unseen"]["per_step"],
                      "overfit_right_of_16": s[task][reader]["overfit"]["right_of_16"]}
                   for reader in ("map", "raw")} | {"verdict": s[task]["verdict"]}
            for task in ("count", "identity")}
    (HERE / "numbers.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"hash": out["data"]["hash"], "weights": total,
                      "count_train": branches["count"]["train_pairs_0_to_3"]["canvases"],
                      "identity_train": branches["identity"]["train_pairs_0_to_3"]["canvases"],
                      "abar_range": [sched[0]["abar"], sched[-1]["abar"]],
                      "split_run": out["results_printed"]["split_run"]["state"][:40]}))


if __name__ == "__main__":
    sys.exit(main())
