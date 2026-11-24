"""The miniature: the digit-map checker shrunk to a 4-pixel canvas, worked by hand in numpy.

Everything here is illustrative except the noise level, which is the real toy's t = 480 (abar 0.094869, the least
noisy of the 16 training timesteps), and the two time features, which are the real first frequency's sin and cos.

    canvas      4 pixels, left half [p1, p2], right half [p3, p4], -1 background and +1 ink
    digits      "1" = [-1, +1], "3" = [+1, -1], "2" = [+1, +1] inside one half
    frozen      eps_null(x) = 0.9 x;  eps_y(x) = eps_null(x) - k * T_y * sigmoid(g x), T_y the digit's stroke tiled
                on both halves (the model may put the digit in either half), k = 0.5, g = 3
    maps        m_y = 4 * |eps_y - eps_null| / sum |eps_y - eps_null|   (the real code scales by 2,048)
    head        one conv (2 maps in, 2 channels out, one window per half, stride 2), SiLU, scale and shift from the
                two time features, mean and max over the 2 halves, one linear readout to the logit
    loss        binary cross-entropy;  count labels apart 1, wrong 1;  identity labels apart 1, wrong 0
    update      one plain gradient step of size 0.05 on the two-canvas batch (the real code uses Adam at 0.001)
    push        g = d logit / d x_t through the head, the maps and the frozen expert; eps_hat = eps_product -
                ||eps_product|| b g / ||g||, b = 0.4

Every hand-derived gradient is checked against central differences; a mismatch raises. Writes miniature_answer_key.json.
Run: ~/.claude/venvs/figures/bin/python -I miniature.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ABAR = 0.094869                                   # real: toy t = 480, numbers.json
SA, S1 = np.sqrt(ABAR), np.sqrt(1 - ABAR)
LSNR = np.log(ABAR / (1 - ABAR))
PHI = np.array([np.sin(LSNR / 10), np.cos(LSNR / 10)])   # real: the first frequency's two features at t = 480
STROKE = {1: np.array([-1., 1.]), 3: np.array([1., -1.]), 2: np.array([1., 1.])}
TEMPLATE = {1: np.array([.2, 1., .2, 1.]), 3: np.array([1., .2, 1., .2])}
K_EXPERT, G_EXPERT = 0.5, 3.0
EPS = np.array([0.5, -0.3, 0.2, -0.6])            # illustrative noise draw, shared by both canvases
ASKED = (1, 3)
CANVASES = {"apart": np.concatenate([STROKE[1], STROKE[3]]), "wrong": np.concatenate([STROKE[3], STROKE[2]])}
LABELS = {"count": {"apart": 1, "wrong": 1}, "identity": {"apart": 1, "wrong": 0}}
LR, B = 0.05, 0.4

P0 = {"K": np.array([[[1.0, 0.5], [-0.5, 0.0]],      # out channel 1: [map a taps, map b taps]
                     [[-0.5, 0.0], [1.0, 0.5]]]),     # out channel 2
      "bias": np.array([-0.5, -0.5]),
      "Wg": np.array([[0.2, 0.1], [0.1, 0.2]]),       # scale gamma = Wg phi
      "Wb": np.array([[0.0, 0.1], [0.1, 0.0]]),       # shift beta = Wb phi
      "v": np.array([0.5, 0.5, 1.0, 1.0]),            # readout on [mean1, mean2, max1, max2]
      "u": np.array([0.3, -0.2]),                     # readout on the two time features
      "c": np.array(-1.0)}


def sig(z):
    return 1 / (1 + np.exp(-z))


def silu(z):
    return z * sig(z)


def dsilu(z):
    s = sig(z)
    return s * (1 + z * (1 - s))


def expert(x):
    """The frozen 'U-Net': eps_null, eps_a, eps_b and the two differences."""
    e0 = 0.9 * x
    d = {y: -K_EXPERT * TEMPLATE[y] * sig(G_EXPERT * x) for y in ASKED}
    return e0, {y: e0 + d[y] for y in ASKED}, d


def maps_of(x):
    _, _, d = expert(x)
    return np.stack([4 * np.abs(d[y]) / np.abs(d[y]).sum() for y in ASKED])   # 2 x 4


def head(m, P):
    """Forward through the head; returns the logit and every intermediate."""
    win = m.reshape(2, 2, 2).transpose(1, 0, 2)          # [half, map, tap]
    pre = np.einsum("cmk,hmk->hc", P["K"], win) + P["bias"]  # [half, channel]
    act = silu(pre)
    gamma, beta = P["Wg"] @ PHI, P["Wb"] @ PHI
    h = act * (1 + gamma) + beta
    mean, mx = h.mean(0), h.max(0)
    feat = np.concatenate([mean, mx])
    logit = float(P["v"] @ feat + P["u"] @ PHI + P["c"])
    return logit, {"windows": win, "pre": pre, "act": act, "gamma": gamma, "beta": beta, "h": h, "mean": mean,
                   "max": mx, "argmax": h.argmax(0), "features": feat}


def head_backward(m, P, dlogit):
    """Hand backprop of the head: gradients for every weight and for the maps."""
    _, c = head(m, P)
    g = {"v": dlogit * c["features"], "u": dlogit * PHI, "c": np.array(dlogit)}
    dfeat = dlogit * P["v"]
    dh = np.zeros((2, 2))
    dh += dfeat[:2] / 2                                   # mean over the 2 halves
    for ch in range(2):
        dh[c["argmax"][ch], ch] += dfeat[2 + ch]          # max picks one half
    dgamma = (dh * c["act"]).sum(0)
    dbeta = dh.sum(0)
    g["Wg"], g["Wb"] = np.outer(dgamma, PHI), np.outer(dbeta, PHI)
    dpre = dh * (1 + c["gamma"]) * dsilu(c["pre"])
    g["K"] = np.einsum("hc,hmk->cmk", dpre, c["windows"])
    g["bias"] = dpre.sum(0)
    dwin = np.einsum("hc,cmk->hmk", dpre, P["K"])
    dm = dwin.transpose(1, 0, 2).reshape(2, 4)
    return g, dm, dpre


def maps_backward(x, dm):
    """d logit / d x through the maps' normalisation, the absolute value and the expert's sigmoid."""
    _, _, d = expert(x)
    dx = np.zeros(4)
    for j, y in enumerate(ASKED):
        a = np.abs(d[y])
        S = a.sum()
        da = 4 * (dm[j] / S - (dm[j] @ a) / S ** 2)      # quotient rule on 4 a / sum(a)
        dd = da * np.sign(d[y])
        s = sig(G_EXPERT * x)
        dx += dd * (-K_EXPERT * TEMPLATE[y] * G_EXPERT * s * (1 - s))
    return dx


def logit_of_x(x, P):
    return head(maps_of(x), P)[0]


def loss(logit, y):
    return -(y * np.log(sig(logit)) + (1 - y) * np.log(1 - sig(logit)))


def numeric(f, arr, h=1e-6):
    out = np.zeros_like(arr, dtype=float)
    it = np.nditer(arr, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        a0 = arr[i]
        arr[i] = a0 + h
        up = f()
        arr[i] = a0 - h
        dn = f()
        arr[i] = a0
        out[i] = (up - dn) / (2 * h)
    return out


def check(name, hand, num, tol=1e-6):
    err = np.max(np.abs(hand - num)) / max(1e-8, np.max(np.abs(num)))
    if err > tol:
        raise AssertionError(f"{name}: hand backprop differs from finite differences by {err:.2e}")
    return float(err)


def r(a, n=4):
    return np.round(np.asarray(a, dtype=float), n).tolist()


def main():
    key = {"illustrative": "every number except abar and the two time features is set by hand for teaching",
           "real": {"t": 480, "abar": ABAR, "sqrt_abar": round(SA, 4), "sqrt_one_minus_abar": round(S1, 4),
                    "log_snr": round(float(LSNR), 4), "time_features": r(PHI)},
           "eps": EPS.tolist(), "weights": {k: r(v) for k, v in P0.items()}, "canvases": {}, "checks": {}}
    xt = {}
    for name, x0 in CANVASES.items():
        x = SA * x0 + S1 * EPS
        xt[name] = x
        e0, ey, d = expert(x)
        m = maps_of(x)
        logit, c = head(m, P0)
        prod = e0 + d[ASKED[0]] + d[ASKED[1]]
        key["canvases"][name] = {
            "x0": x0.tolist(), "xt": r(x), "eps_null": r(e0), "eps_a": r(ey[1]), "eps_b": r(ey[3]),
            "d_a": r(d[1]), "d_b": r(d[3]), "m_a": r(m[0]), "m_b": r(m[1]), "pre": r(c["pre"]), "act": r(c["act"]),
            "gamma": r(c["gamma"]), "beta": r(c["beta"]), "h": r(c["h"]), "mean": r(c["mean"]), "max": r(c["max"]),
            "logit": round(logit, 4), "p": round(float(sig(logit)), 4), "eps_product": r(prod)}

    # training: per branch, losses, weight gradients (checked), one step, logits after
    key["branches"] = {}
    for branch, ys in LABELS.items():
        grads = {k: np.zeros_like(v, dtype=float) for k, v in P0.items()}
        rows = {}
        for name, x in xt.items():
            m = maps_of(x)
            logit, _ = head(m, P0)
            dl = float(sig(logit) - ys[name])                  # d loss / d logit
            g, _, _ = head_backward(m, P0, dl)
            for k in grads:
                grads[k] += g[k] / len(xt)
            rows[name] = {"y": ys[name], "logit": round(logit, 4), "p": round(float(sig(logit)), 4),
                          "loss": round(float(loss(logit, ys[name])), 4), "dloss_dlogit": round(dl, 4)}
        Pn = {k: v.copy().astype(float) for k, v in P0.items()}

        def batch_loss():
            return float(np.mean([loss(head(maps_of(x), Pn)[0], ys[n]) for n, x in xt.items()]))
        for k in ("K", "Wg", "v", "u", "bias"):
            key["checks"][f"{branch}: dL/d{k}"] = check(f"{branch} {k}", grads[k], numeric(batch_loss, Pn[k]))
        P1 = {k: P0[k] - LR * grads[k] for k in P0}
        for name, x in xt.items():
            after = head(maps_of(x), P1)[0]
            rows[name]["logit_after_one_step"] = round(after, 4)
            rows[name]["p_after_one_step"] = round(float(sig(after)), 4)
        key["branches"][branch] = {"rows": rows, "batch_loss_before": round(float(np.mean([v["loss"] for v in rows.values()])), 4),
                                   "batch_loss_after": round(float(np.mean([loss(head(maps_of(x), P1)[0], ys[n]) for n, x in xt.items()])), 4),
                                   "grad": {k: r(v) for k, v in grads.items()}}

    # the push, with the starting weights
    key["push"] = {"b": B}
    for name, x in xt.items():
        m = maps_of(x)
        _, dm, _ = head_backward(m, P0, 1.0)
        gx = maps_backward(x, dm)
        xx = x.copy()
        key["checks"][f"push {name}: dlogit/dx"] = check(f"push {name}", gx, numeric(lambda: logit_of_x(xx, P0), xx))
        prod = np.array(key["canvases"][name]["eps_product"])
        push = np.linalg.norm(prod) * B * gx / np.linalg.norm(gx)
        eps_hat = prod - push
        # the clean-canvas guess moves by +S1/SA times the push; first-order change in the logit at x_t is g . (step)
        x0_shift = S1 / SA * push
        key["push"][name] = {"dlogit_dmaps": r(dm), "g": r(gx), "norm_g": round(float(np.linalg.norm(gx)), 4),
                             "norm_eps_product": round(float(np.linalg.norm(prod)), 4), "push": r(push),
                             "push_over_product": round(float(np.linalg.norm(push) / np.linalg.norm(prod)), 4),
                             "eps_hat": r(eps_hat), "x0_hat_shift": r(x0_shift),
                             "logit_at_xt_plus_small_step_along_g": round(logit_of_x(x + 0.01 * gx / np.linalg.norm(gx), P0), 4),
                             "logit_at_xt": round(logit_of_x(x, P0), 4)}
    (HERE / "miniature_answer_key.json").write_text(json.dumps(key, indent=1))
    print(json.dumps({b: {n: (v["logit"], v["logit_after_one_step"]) for n, v in key["branches"][b]["rows"].items()}
                      for b in key["branches"]}))
    print("largest relative mismatch, hand against finite differences:", f"{max(key['checks'].values()):.1e}")


if __name__ == "__main__":
    main()
