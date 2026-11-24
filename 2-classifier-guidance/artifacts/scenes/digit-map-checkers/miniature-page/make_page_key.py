"""The answer key for the interactive miniature page: the ladder's miniature at all 16 timesteps, plus the real canvases.

Reuses ../2-miniature/miniature.py unchanged (its weights, expert, head, hand backprop and finite-difference check) and
extends it where the page needs a step that key does not hold:

    noise level   the miniature run at each of the 16 real timesteps (712 to 480), with that timestep's real abar and
                  its real first-frequency time features, so the noise slider moves real numbers through hand-set weights
    push length   the push, the pushed noise guess and the clean-guess shift for b = 0.00, 0.05, ..., 1.00
    broken run    the push with the maps built inside torch.no_grad (as v2_learn.py's inputs() builds them for training):
                  the gradient reaching x_t is exactly zero, so g / ||g|| is 0 / 0 and no push exists
    real canvases apart_1_3 and wrong_1_3 at all 16 timesteps, made from the clean canvases and the display noise in
                  ../numbers/examples/ with x_t = sqrt(abar) x0 + sqrt(1 - abar) eps, checked against the four saved levels
    real stages   the saved count checker's every in-between output on four canvases at four timesteps, from
                  ../numbers/real/count-stages.npz (job 94116): canvases as grey bytes, signed tensors as bytes on a scale
                  set by the 99th percentile of |value|, 8 channels per layer chosen by largest mean, pooled numbers,
                  time features, the scale and shift, logits and the push gradient
    real scores   the saved count checker's balanced accuracy at each of the 16 timesteps, from
                  ../numbers/real/count-final.json (job 94108), cross-checked against the V2 review's prose

Asserts that its t = 480 values equal ../2-miniature/miniature_answer_key.json, every gradient against central
differences, that each real map adds to 2,048, that the stored pooled numbers are c2's mean and max, and that the real
logits equal the ones job 94116 printed. Writes page_key.json (read by build_page.py).
Run: PYTHONDONTWRITEBYTECODE=1 ~/.claude/venvs/figures/bin/python -I make_page_key.py
"""
import base64
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LADDER = HERE.parent
spec = importlib.util.spec_from_file_location("miniature", LADDER / "2-miniature" / "miniature.py")
mini = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mini)

NUMBERS = json.loads((LADDER / "numbers" / "numbers.json").read_text())
OLD_KEY = json.loads((LADDER / "2-miniature" / "miniature_answer_key.json").read_text())
SCHEDULE = NUMBERS["noise"]["schedule"]                     # real: 16 timesteps, abar from canvas_unet.py
B_GRID = [round(0.05 * i, 2) for i in range(21)]
EXAMPLES = LADDER / "numbers" / "examples"
r = mini.r


def set_noise_level(abar):
    """Point the miniature module at one timestep: its abar and its two real first-frequency time features."""
    lsnr = np.log(abar / (1 - abar))
    mini.PHI = np.array([np.sin(lsnr / 10), np.cos(lsnr / 10)])
    return np.sqrt(abar), np.sqrt(1 - abar), float(lsnr)


def one_timestep(row):
    sa, s1, lsnr = set_noise_level(row["abar"])
    P0 = mini.P0
    out = {"t": row["t"], "abar": row["abar"], "sqrt_abar": round(float(sa), 4), "sqrt_one_minus_abar": round(float(s1), 4),
           "log_snr": round(lsnr, 4), "time_features": r(mini.PHI), "canvases": {}, "branches": {}, "push": {}, "checks": {}}
    gamma, beta = P0["Wg"] @ mini.PHI, P0["Wb"] @ mini.PHI
    out["gamma"], out["beta"] = r(gamma), r(beta)
    xt = {}
    for name, x0 in mini.CANVASES.items():
        x = sa * x0 + s1 * mini.EPS
        xt[name] = x
        e0, ey, d = mini.expert(x)
        m = mini.maps_of(x)
        logit, c = mini.head(m, P0)
        prod = e0 + d[1] + d[3]
        out["canvases"][name] = {"x0": x0.tolist(), "xt": r(x), "eps_null": r(e0), "eps_a": r(ey[1]), "eps_b": r(ey[3]),
                                 "d_a": r(d[1]), "d_b": r(d[3]), "m_a": r(m[0]), "m_b": r(m[1]), "pre": r(c["pre"]),
                                 "act": r(c["act"]), "h": r(c["h"]), "mean": r(c["mean"]), "max": r(c["max"]),
                                 "argmax": c["argmax"].tolist(), "logit": round(logit, 4), "p": round(float(mini.sig(logit)), 4),
                                 "eps_product": r(prod)}
    for branch, ys in mini.LABELS.items():
        grads = {k: np.zeros_like(v, dtype=float) for k, v in P0.items()}
        rows = {}
        for name, x in xt.items():
            m = mini.maps_of(x)
            logit, _ = mini.head(m, P0)
            dl = float(mini.sig(logit) - ys[name])
            g, _, dpre = mini.head_backward(m, P0, dl)
            for k in grads:
                grads[k] += g[k] / len(xt)
            rows[name] = {"y": ys[name], "logit": round(logit, 4), "p": round(float(mini.sig(logit)), 4),
                          "loss": round(float(mini.loss(logit, ys[name])), 4), "dloss_dlogit": round(dl, 4), "dpre": r(dpre),
                          "dfeat": r(dl * P0["v"])}
        Pn = {k: v.copy().astype(float) for k, v in P0.items()}

        def batch_loss():
            return float(np.mean([mini.loss(mini.head(mini.maps_of(x), Pn)[0], ys[n]) for n, x in xt.items()]))
        for k in ("K", "Wg", "Wb", "v", "u", "bias"):
            out["checks"][f"{branch}: dL/d{k}"] = mini.check(f"{branch} {k}", grads[k], mini.numeric(batch_loss, Pn[k]))
        P1 = {k: P0[k] - mini.LR * grads[k] for k in P0}
        for name, x in xt.items():
            after = mini.head(mini.maps_of(x), P1)[0]
            rows[name]["logit_after_one_step"] = round(after, 4)
            rows[name]["p_after_one_step"] = round(float(mini.sig(after)), 4)
        out["branches"][branch] = {
            "rows": rows, "batch_loss_before": round(float(np.mean([v["loss"] for v in rows.values()])), 4),
            "batch_loss_after": round(float(np.mean([mini.loss(mini.head(mini.maps_of(x), P1)[0], ys[n]) for n, x in xt.items()])), 4),
            "gap_before": round(rows["apart"]["logit"] - rows["wrong"]["logit"], 4),
            "gap_after": round(rows["apart"]["logit_after_one_step"] - rows["wrong"]["logit_after_one_step"], 4),
            "grad": {k: r(v) for k, v in grads.items()}, "K_after": r(P1["K"])}
    for name, x in xt.items():
        m = mini.maps_of(x)
        _, dm, dpre = mini.head_backward(m, P0, 1.0)
        gx = mini.maps_backward(x, dm)
        xx = x.copy()
        out["checks"][f"push {name}: dlogit/dx"] = mini.check(f"push {name}", gx, mini.numeric(lambda: mini.logit_of_x(xx, P0), xx))
        # the broken run: maps detached from x_t, so the logit's gradient into x_t is identically zero
        detached = mini.maps_of(x).copy()
        xd = x.copy()
        g_detached = mini.numeric(lambda: mini.head(detached, P0)[0], xd)
        assert np.all(g_detached == 0.0), g_detached
        prod = np.array(out["canvases"][name]["eps_product"])
        unit = gx / np.linalg.norm(gx)
        per_b = []
        for b in B_GRID:
            push = np.linalg.norm(prod) * b * unit
            per_b.append({"b": b, "push": r(push), "eps_hat": r(prod - push), "x0_hat_shift": r(s1 / sa * push),
                          "push_length": round(float(np.linalg.norm(push)), 4)})
        out["push"][name] = {"dlogit_dfeat": r(P0["v"] * 1.0), "dlogit_dpre": r(dpre), "dlogit_dmaps": r(dm), "g": r(gx), "norm_g": round(float(np.linalg.norm(gx)), 4),
                             "unit_g": r(unit), "norm_eps_product": round(float(np.linalg.norm(prod)), 4),
                             "clean_guess_gain": round(float(s1 / sa), 4), "g_maps_inside_no_grad": r(g_detached),
                             "logit_after_small_step_along_g": round(mini.logit_of_x(x + 0.01 * unit, P0), 4), "by_b": per_b}
    return out


def same_as_old_key(page480):
    """Every value the old key holds at t = 480 must come back identical here."""
    n = 0
    for name in ("apart", "wrong"):
        for k, v in OLD_KEY["canvases"][name].items():
            if k in page480["canvases"][name]:
                assert page480["canvases"][name][k] == v, (name, k, v, page480["canvases"][name][k]); n += 1
        for k in ("g", "norm_g", "norm_eps_product", "dlogit_dmaps"):
            assert page480["push"][name][k] == OLD_KEY["push"][name][k], (name, k); n += 1
        row = [x for x in page480["push"][name]["by_b"] if x["b"] == 0.4][0]
        for k in ("push", "eps_hat", "x0_hat_shift"):
            assert row[k] == OLD_KEY["push"][name][k], (name, k, row[k], OLD_KEY["push"][name][k]); n += 1
    for branch in ("count", "identity"):
        ob, nb = OLD_KEY["branches"][branch], page480["branches"][branch]
        for k in ("batch_loss_before", "batch_loss_after"):
            assert ob[k] == nb[k], (branch, k); n += 1
        for name in ("apart", "wrong"):
            for k, v in ob["rows"][name].items():
                assert nb["rows"][name][k] == v, (branch, name, k); n += 1
        for k, v in ob["grad"].items():
            assert nb["grad"][k] == v, (branch, k); n += 1
    return n


def real_canvases():
    noise = np.load(EXAMPLES / "display-noise.npy")
    out = {}
    for name in ("apart_1_3", "wrong_1_3"):
        x0 = np.load(EXAMPLES / f"{name}-clean.npy")
        frames = {}
        for row in SCHEDULE:
            a = np.float32(row["abar"])
            xt = (np.sqrt(a) * x0 + np.sqrt(1 - a) * noise).astype(np.float32)
            saved = EXAMPLES / f"{name}-t{row['t']}.npy"
            if saved.exists():
                assert np.allclose(np.load(saved), xt, atol=2e-4), (name, row["t"])
            # grey from -1 (black) to +1 (white), as figure_kit.image draws the ladder's canvases
            frames[str(row["t"])] = base64.b64encode(np.round((np.clip(xt, -1, 1) + 1) * 127.5).astype(np.uint8).tobytes()).decode()
        frames["clean"] = base64.b64encode(np.round((x0 + 1) * 127.5).astype(np.uint8).tobytes()).decode()
        out[name] = {"shape": list(x0.shape), "frames": frames, "about": NUMBERS["examples"][name]}
    return out


REAL_STAGES = LADDER / "numbers" / "real" / "count-stages.npz"
REVIEW = LADDER.parents[3] / "artifacts" / "notes" / "progress" / "abstract-checkers" / "05-does-it-learn" / "review.md"
REAL_CANVASES = ["apart_1_3", "wrong_1_3", "apart_3_4", "wrong_3_4"]
REAL_TS = [712, 616, 539, 480]
N_CHANNELS = 8
# each convolution stage: (name on the page, tensor before SiLU, tensor after it)
CONV_STAGES = [("conv1", "stem0_conv2d", "stem1_silu"), ("conv2", "stem2_conv2d", "stem3_silu"),
               ("conv3", "stem4_conv2d", "stem5_silu"), ("c1", "c1_conv", "c1_silu"), ("c2", "c2_conv", "c2_silu")]


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode()


def signed(a, scale):
    """Values on one shared scale, -scale..+scale to bytes 0..255 (127.5 is zero); the page colours them blue, white, red."""
    return b64(np.round(127.5 * (1 + np.clip(a / scale, -1, 1))))


def grey(a):
    """A canvas from -1 (black) to +1 (white), clipped there, as figure_kit.image and the noise frame draw it."""
    return b64(np.round((np.clip(a, -1, 1) + 1) * 127.5))


def p99(*arrays):
    """The colour scale: the 99th percentile of |value| over the arrays drawn together, so one hot pixel cannot wash out the rest."""
    return float(np.percentile(np.abs(np.concatenate([np.ravel(a) for a in arrays])), 99))


def top_channels(a):
    """The rule printed on the page: the 8 channels with the largest mean value, on this canvas at this timestep."""
    means = a.reshape(a.shape[0], -1).mean(1)
    return [int(i) for i in np.argsort(-means, kind="stable")[:N_CHANNELS]]


def real_stages():
    """Every in-between output of the saved count checker (share-0.0 U-Net, job 94116) for the overview frame."""
    z = np.load(REAL_STAGES)
    f32 = lambda k: z[k].astype(np.float32)
    out = {"canvases": REAL_CANVASES, "timesteps": REAL_TS, "rule": "the 8 channels with the largest mean value, "
           "on this canvas at this timestep", "colour_rule": "each picture saturates at the 99th percentile of |value| over what is drawn with it", "source": "../numbers/real/count-stages.npz, written by "
           "2-classifier-guidance/scripts/abstract_checkers/explain_stages.py (job 94116)", "cells": {}, "logits": {}}
    for name in REAL_CANVASES:
        about = NUMBERS["examples"][name]
        out["logits"][name] = {str(t): round(float(z[f"{name}_t{t}__logit"]), 4) for t in REAL_TS}
        out.setdefault("about", {})[name] = {k: about[k] for k in ("asked", "holds", "kind", "count_label", "identity_label", "split")}
        for t in REAL_TS:
            k = f"{name}_t{t}__"
            x0, xt = f32(k + "x0")[0], f32(k + "x_t")[0]
            eps = np.stack([f32(k + e)[0] for e in ("eps_null", "eps_a", "eps_b")])
            maps, grad = f32(k + "maps"), f32(k + "grad")[0]
            s_eps, s_maps, s_grad = p99(eps), p99(maps), p99(grad)
            cell = {"x0": grey(x0), "xt": grey(xt), "xt_range": [round(float(xt.min()), 2), round(float(xt.max()), 2)],
                    "eps": signed(eps, s_eps), "eps_scale": round(s_eps, 3),
                    "maps": signed(maps, s_maps), "maps_scale": round(s_maps, 3),
                    "maps_sum": [round(float(m.sum()), 1) for m in maps], "maps_max": [round(float(m.max()), 2) for m in maps],
                    "grad": signed(grad, s_grad), "grad_scale": float(f"{s_grad:.3g}"), "grad_norm": float(f"{np.linalg.norm(grad):.3g}"),
                    "logit": round(float(z[k + "logit"]), 4), "p": round(float(1 / (1 + np.exp(-z[k + "logit"]))), 4),
                    "time_features": [round(float(v), 3) for v in f32(k + "time_features")],
                    "pooled": [round(float(v), 3) for v in f32(k + "pooled")], "stages": {}}
            for page_name, pre_k, post_k in CONV_STAGES:
                pre, post = f32(k + pre_k), f32(k + post_k)
                ch = top_channels(post)
                s = p99(pre[ch], post[ch])
                cell["stages"][page_name] = {"shape": list(post.shape), "channels": ch, "scale": round(s, 2),
                                             "pre": signed(pre[ch], s), "post": signed(post[ch], s),
                                             "post_mean": [round(float(post[c].mean()), 2) for c in ch],
                                             "post_min": round(float(post.min()), 3),
                                             "pre_negative_share": round(float((pre < 0).mean()), 3)}
            c1, film = f32(k + "c1_silu"), f32(k + "film_out")
            gam, bet = f32(k + "film_gamma"), f32(k + "film_beta")
            ch = top_channels(film)
            s = p99(c1[ch], film[ch])
            cell["stages"]["film"] = {"shape": list(film.shape), "channels": ch, "scale": round(s, 2),
                                      "pre": signed(c1[ch], s), "post": signed(film[ch], s),
                                      "post_mean": [round(float(film[c].mean()), 2) for c in ch],
                                      "gamma": [round(float(v), 3) for v in gam], "beta": [round(float(v), 3) for v in bet]}
            # the stored pooled numbers must equal the mean and max of the stored c2 output (float16, so to 1e-2)
            c2 = f32(k + "c2_silu")
            again = np.concatenate([c2.mean((1, 2)), c2.max((1, 2))])
            assert np.allclose(again, f32(k + "pooled"), atol=2e-2), (name, t, float(np.abs(again - f32(k + "pooled")).max()))
            assert all(abs(v - 2048) < 4 for v in cell["maps_sum"]), cell["maps_sum"]
            out["cells"][f"{name}|{t}"] = cell
    return out


COUNT_FINAL = LADDER / "numbers" / "real" / "count-final.json"


def real_scores():
    """The saved count checker's balanced accuracy at each of the 16 timesteps (job 94108, W&B tyhrpyov).

    Read from count-final.json (V2's final-count.json). The review's own figures are a cross-check, never the source.
    """
    import re
    d = json.loads(COUNT_FINAL.read_text())
    ts = [row["t"] for row in SCHEDULE]
    assert d["wandb_run"] == "tyhrpyov" or d["wandb_run"].endswith("tyhrpyov"), d["wandb_run"]
    assert d["timesteps"] == ts, (d["timesteps"], ts)
    out = {}
    for split, want in (("unseen_test", 0.791), ("seen_test", 0.782)):
        sp = d["map"][split]
        ps = [float(v) for v in sp["per_step"]]
        assert len(ps) == 16 and sp.get("timesteps", ts) == ts, split
        mean = sum(ps) / 16
        assert round(mean, 3) == want, (split, mean)
        assert abs(mean - sp["balanced_accuracy"]) < 1e-9, (split, mean, sp["balanced_accuracy"])
        out[split] = {"per_step": [round(v, 4) for v in ps], "mean": round(mean, 3)}
    # cross-check against the review's prose: the unseen curve's two ends
    ends = re.search(r"score on digits 5 to 9 rises from ([0-9.]+) at the noisiest step to ([0-9.]+) at the least noisy",
                     REVIEW.read_text())
    u = out["unseen_test"]["per_step"]
    assert ends and (float(ends.group(1)), float(ends.group(2))) == (round(u[0], 2), round(u[-1], 2)), ends
    return {"source": "../numbers/real/count-final.json (V2's final-count.json, job 94108, W&B run tyhrpyov), map reader",
            "timesteps": ts, "unseen_per_step": out["unseen_test"]["per_step"], "seen_per_step": out["seen_test"]["per_step"],
            "unseen_5_to_9": out["unseen_test"]["mean"], "seen_0_to_4": out["seen_test"]["mean"],
            "unseen_noisiest": round(u[0], 2), "unseen_least_noisy": round(u[-1], 2),
            "largest_gap": round(max(abs(a - b) for a, b in zip(u, out["seen_test"]["per_step"])), 3),
            "noisiest_t": ts[0], "least_noisy_t": ts[-1], "bar": 0.75, "chance": 0.5}


def main():
    steps = [one_timestep(row) for row in SCHEDULE]
    by_t = {s["t"]: s for s in steps}
    n_same = same_as_old_key(by_t[480])
    worst = max(v for s in steps for v in s["checks"].values())
    key = {
        "source": "make_page_key.py, reusing ../2-miniature/miniature.py; real values from ../numbers/numbers.json",
        "illustrative": OLD_KEY["illustrative"] + "; the noise draw eps on the four pixels is illustrative too",
        "weights": OLD_KEY["weights"], "eps": OLD_KEY["eps"], "lr": mini.LR, "b_grid": B_GRID, "b_design": mini.B,
        "labels": mini.LABELS, "templates": {str(k): v.tolist() for k, v in mini.TEMPLATE.items()},
        "expert": {"k": mini.K_EXPERT, "g": mini.G_EXPERT, "null_scale": 0.9},
        "timesteps": [s["t"] for s in steps], "steps": steps,
        "largest_relative_mismatch": worst, "values_equal_to_old_key_at_480": n_same,
        "full_scale": {"layers": NUMBERS["head"]["layers"], "weights": NUMBERS["head"]["weights"],
                       "training": NUMBERS["training"], "push": NUMBERS["push"],
                       "counts": {b: {k: v["canvases"] for k, v in NUMBERS["branches"][b].items()} for b in ("count", "identity")}},
        "results": {
            "source": "artifacts/notes/progress/abstract-checkers/05-does-it-learn/review.md (the identity row and the count row) "
                      "and ../numbers/numbers.json results_printed (job 94080 and job 94085's log lines)",
            "bar": 0.75, "chance": 0.5,
            "count_map_reader": {"pairs_holding_4": 0.774, "trained_pairs": 0.781, "job": 94085},
            "identity_map_reader": {"pairs_holding_4": 0.555, "trained_pairs": 0.603, "job": 94085,
                                    "noisiest_to_least_noisy": [0.48, 0.65]},
            "combined_checker": {"all_kinds": 0.704, "apart_vs_wrong": 0.511, "job": 94080, "noisiest_to_least_noisy": [0.59, 0.81]}},
        "real_canvases": real_canvases(),
        "real_stages": real_stages(),
        "real_scores": real_scores(),
    }
    # the logits the job printed (job 94116's log) must be the ones the page shows
    printed = {"apart_1_3": [0.543, 1.567, 3.806, 6.308], "wrong_1_3": [0.076, -1.115, -1.281, -1.019],
               "apart_3_4": [0.015, 0.081, 1.297, 2.999], "wrong_3_4": [0.064, -0.346, 1.271, 4.088]}
    for name, vals in printed.items():
        got = [key["real_stages"]["logits"][name][str(t)] for t in REAL_TS]
        assert all(abs(a - b) < 6e-4 for a, b in zip(got, vals)), (name, got, vals)
    key["results"]["count_final"] = {"seen_0_to_4": key["real_scores"]["seen_0_to_4"],
                                     "unseen_5_to_9": key["real_scores"]["unseen_5_to_9"], "job": 94108, "wandb": "tyhrpyov"}
    (HERE / "page_key.json").write_text(json.dumps(key, separators=(",", ":")))
    print(f"16 timesteps, {len(B_GRID)} push lengths; {n_same} values equal to the old key at t = 480; "
          f"largest relative mismatch, hand against finite differences: {worst:.1e}")


if __name__ == "__main__":
    main()
