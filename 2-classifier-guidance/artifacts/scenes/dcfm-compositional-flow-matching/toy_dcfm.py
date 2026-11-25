"""DCFM's own two-dimensional toy (Figs 2 and 3, p. 4), rebuilt on the CPU with numpy only.

Four Gaussian cells sit at (+-1, +-1). Attribute y1 is left (1, x = -1) or right (2, x = +1); attribute y2 is top
(1, y = +1) or bottom (2, y = -1). The combination (2, 1), right and top, is missing from every silo.

Two kinds of expert are used:

* exact experts: the velocity a perfectly trained flow-matching network would learn from its silo's data, computed in
  closed form for a Gaussian mixture under the straight path x_t = (1 - t) x_0 + t x_1 (t = 0 noise, t = 1 data);
* trained experts: a small numpy MLP trained with Eq. 9's masked flow-matching loss, with or without the penalty
  (Eq. 10 to 11 locally, Eq. 12 to 14 across peers for DCFM-A).

Choices the paper does not state are named in TOY_CHOICES and copied into dcfm_numbers.json.
"""
import numpy as np

# The four cells: (y1, y2) -> centre. y1: 1 left, 2 right. y2: 1 top, 2 bottom.
CENTRE = {(1, 1): (-1.0, 1.0), (1, 2): (-1.0, -1.0), (2, 2): (1.0, -1.0), (2, 1): (1.0, 1.0)}
MISSING = (2, 1)
SPLITS = {
    "iid": {1: [(1, 1), (1, 2), (2, 2)], 2: [(1, 1), (1, 2), (2, 2)]},
    "non_iid": {1: [(1, 2), (2, 2)], 2: [(1, 1), (1, 2)]},
}
TOY_CHOICES = {
    "cell_std": 0.15,
    "cell_std_source": "read by eye from Fig. 2 (p. 4): each cloud spans about +-0.4 around its centre; illustrative",
    "cells_equal_weight": True,
    "masking_weights_pi_full_marg_uncond": [1 / 3, 1 / 3, 1 / 3],
    "masking_weights_source": "not stated in the paper (Eq. 8, p. 5); the toy uses equal weights",
    "lambda_ci": 1.0,
    "lambda_source": "not stated in the paper (Eq. 11, 14, 17); the toy uses 1.0 and also sweeps it on the IID split",
    "penalty_norm": "squared Euclidean norm, as Eq. 7 (p. 3) and Eq. 16 (p. 6) print it; Eq. 10 and Eq. 12 (p. 5) print no norm",
    "ode_solver": "Euler, 200 equal steps from t = 0 to t = 1 (the paper's solver and step count are not stated)",
    "network": "MLP 15 -> 128 -> 128 -> 128 -> 2, SiLU, Adam lr 2e-3, batch 512, 4000 steps (the paper's toy network is not stated)",
    "routing_r": "not stated how r is drawn (Eq. 12, p. 5); the toy draws each r_i uniformly among silos whose data hold y_i, r_0 uniformly, and redraws until some entry is the silo being trained",
    "dcfm_a_start": "DCFM-A retraining starts from the silo's Stage I weights (the paper does not say)",
    "sampling_routing_missing": "for (2, 1) across silos: right from silo 1, top from silo 2, unconditional from silo 1",
}
S = TOY_CHOICES["cell_std"]


# ---------------------------------------------------------------- exact velocities
def tau2(t):
    """Variance of x_t inside one cell under the straight path: (1 - t)^2 + t^2 s^2."""
    return (1 - t) ** 2 + (t * S) ** 2


def exact_velocity(x, t, cells):
    """E[x_1 - x_0 | x_t = x] for equal-weight Gaussian cells (std S) under x_t = (1 - t) x_0 + t x_1.

    Inside cell k the answer is mu_k + c_t (x - t mu_k), with c_t = (t S^2 - (1 - t)) / tau^2; across cells it is the
    posterior-weighted average. x has shape (..., 2); t is a float in [0, 1].
    """
    x = np.asarray(x, float)
    mus = np.array([CENTRE[c] for c in cells])  # (K, 2)
    T2 = tau2(t)
    c_t = (t * S ** 2 - (1 - t)) / T2
    d = x[..., None, :] - t * mus  # (..., K, 2)
    logw = -0.5 * (d ** 2).sum(-1) / T2
    logw -= logw.max(-1, keepdims=True)
    w = np.exp(logw)
    w /= w.sum(-1, keepdims=True)
    vk = mus + c_t * d  # (..., K, 2)
    return (w[..., None] * vk).sum(-2)


def cells_for(cells, y1=None, y2=None):
    """The silo's cells that match a (possibly masked) label: y1 or y2 None means that attribute is left out."""
    return [c for c in cells if (y1 is None or c[0] == y1) and (y2 is None or c[1] == y2)]


def exact_expert(cells):
    """A perfectly trained silo expert: label (y1, y2) with None for the null token -> velocity function."""
    def v(x, t, y1=None, y2=None):
        sub = cells_for(cells, y1, y2)
        if not sub:
            raise ValueError(f"label ({y1}, {y2}) never seen in this silo: a network's answer there is unconstrained")
        return exact_velocity(x, t, sub)
    return v


def sum_rule(v_u, v_1, v_2):
    """Eq. 5 with w_i = 1: v(x) + (v(x|y1) - v(x)) + (v(x|y2) - v(x)) = v(x|y1) + v(x|y2) - v(x)."""
    return lambda x, t: v_1(x, t) + v_2(x, t) - v_u(x, t)


def ode_sample(field, n=2000, steps=200, seed=0):
    """Euler integration from x_0 ~ N(0, I) at t = 0 to t = 1 (DCFM's direction: noise at 0, data at 1)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2))
    dt = 1.0 / steps
    for k in range(steps):
        x = x + dt * field(x, k * dt)
    return x


def fraction_near(x, centre, radius=0.5):
    return float((np.linalg.norm(x - np.array(centre), axis=1) < radius).mean())


def nearest_cell_shares(x, radius=0.5):
    out = {f"{c[0]},{c[1]}": fraction_near(x, CENTRE[c], radius) for c in CENTRE}
    out["none"] = 1.0 - sum(out.values())
    return out


# ---------------------------------------------------------------- the numpy MLP
def label_features(y1, y2):
    """One-hot for y1 over (null, left, right) and y2 over (null, top, bottom); y = 0 is the null token."""
    n = len(y1)
    f = np.zeros((n, 6))
    f[np.arange(n), y1] = 1.0
    f[np.arange(n), 3 + y2] = 1.0
    return f


def time_features(t):
    t = t[:, None]
    ks = np.arange(1, 4)[None, :]
    return np.concatenate([t, np.sin(2 * np.pi * ks * t), np.cos(2 * np.pi * ks * t)], 1)  # 7


def features(x, t, y1, y2):
    return np.concatenate([x, time_features(t), label_features(y1, y2)], 1)  # 2 + 7 + 6 = 15


def silu(z):
    return z / (1 + np.exp(-z))


def dsilu(z):
    s = 1 / (1 + np.exp(-z))
    return s * (1 + z * (1 - s))


class MLP:
    def __init__(self, sizes=(15, 128, 128, 128, 2), seed=0):
        rng = np.random.default_rng(seed)
        self.W = [rng.standard_normal((a, b)) * np.sqrt(2.0 / a) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.W[-1] *= 0.1
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.step_count = 0

    def copy(self):
        o = MLP.__new__(MLP)
        o.W = [w.copy() for w in self.W]
        o.b = [b.copy() for b in self.b]
        o.m = [np.zeros_like(p) for p in o.W + o.b]
        o.v = [np.zeros_like(p) for p in o.W + o.b]
        o.step_count = 0
        return o

    def forward(self, f, keep=False):
        hs, zs = [f], []
        h = f
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            zs.append(z)
            h = silu(z) if i < len(self.W) - 1 else z
            hs.append(h)
        if keep:
            self._cache = (hs, zs)
        return h

    def backward_and_step(self, g_out, lr=2e-3, b1=0.9, b2=0.999):
        hs, zs = self._cache
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        g = g_out
        for i in reversed(range(len(self.W))):
            if i < len(self.W) - 1:
                g = g * dsilu(zs[i])
            gW[i] = hs[i].T @ g
            gb[i] = g.sum(0)
            g = g @ self.W[i].T
        self.step_count += 1
        params = self.W + self.b
        grads = gW + gb
        for j, (p, gr) in enumerate(zip(params, grads)):
            self.m[j] = b1 * self.m[j] + (1 - b1) * gr
            self.v[j] = b2 * self.v[j] + (1 - b2) * gr ** 2
            mh = self.m[j] / (1 - b1 ** self.step_count)
            vh = self.v[j] / (1 - b2 ** self.step_count)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)

    def velocity(self, x, t, y1=None, y2=None):
        x = np.atleast_2d(x)
        n = len(x)
        tt = np.full(n, float(t)) if np.isscalar(t) else t
        f = features(x, tt, np.full(n, y1 or 0, int), np.full(n, y2 or 0, int))
        return self.forward(f)


def draw_batch(rng, cells, n):
    """Data from the silo: a cell label, x_1 in the cell, x_0 noise, t uniform, x_t on the straight path."""
    idx = rng.integers(len(cells), size=n)
    lab = np.array([cells[i] for i in idx])
    mus = np.array([CENTRE[tuple(c)] for c in lab])
    x1 = mus + S * rng.standard_normal((n, 2))
    x0 = rng.standard_normal((n, 2))
    t = rng.uniform(0, 1, n)
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    return lab, xt, t, x1 - x0


def train(cells, lam=0.0, mode="none", steps=4000, batch=512, seed=0, init=None, peers=None, me=None, log_every=0):
    """Eq. 9 plus lam times the penalty.

    mode "none": Eq. 9 only. mode "local": Eq. 11, the penalty of Eq. 10 on the silo's own model.
    mode "peer": Eq. 14, the penalty of Eq. 12 with routing r over this model (index me) and frozen peers (dict).
    """
    rng = np.random.default_rng(seed)
    net = init.copy() if init is not None else MLP(seed=seed)
    pi = TOY_CHOICES["masking_weights_pi_full_marg_uncond"]
    seen_by = {}
    if mode == "peer":
        everyone = dict(peers)
        everyone[me] = None
        for i in (0, 1):
            for val in (1, 2):
                seen_by[(i, val)] = [a for a, cs in SPLITS_CURRENT.items() if any(c[i] == val for c in cs)]
    for step in range(steps):
        lab, xt, t, target = draw_batch(rng, cells, batch)
        # Eq. 8: mask m drawn as full (pi_full), one attribute kept (pi_marg / k each), or none (pi_uncond).
        u = rng.uniform(size=batch)
        keep1 = np.ones(batch, bool)
        keep2 = np.ones(batch, bool)
        marg = (u >= pi[0]) & (u < pi[0] + pi[1])
        unc = u >= pi[0] + pi[1]
        which = rng.integers(2, size=batch)
        keep1[marg & (which == 1)] = False
        keep2[marg & (which == 0)] = False
        keep1[unc] = False
        keep2[unc] = False
        y1m = np.where(keep1, lab[:, 0], 0)
        y2m = np.where(keep2, lab[:, 1], 0)
        f_fm = features(xt, t, y1m, y2m)
        if lam == 0 or mode == "none":
            out = net.forward(f_fm, keep=True)
            g = 2 * (out - target) / batch
            net.backward_and_step(g)
            continue
        nb = batch // 2
        lab_c, xt_c, t_c = lab[:nb], xt[:nb], t[:nb]
        y1, y2 = lab_c[:, 0], lab_c[:, 1]
        zero = np.zeros(nb, int)
        if mode == "local":
            f_ci = np.concatenate([features(xt_c, t_c, y1, y2), features(xt_c, t_c, y1, zero),
                                   features(xt_c, t_c, zero, y2), features(xt_c, t_c, zero, zero)])
            out = net.forward(np.concatenate([f_fm, f_ci]), keep=True)
            o_fm, o = out[:batch], out[batch:]
            J, M1, M2, U = o[:nb], o[nb:2 * nb], o[2 * nb:3 * nb], o[3 * nb:]
            R = J - (M1 + M2 - U)
            gR = 2 * lam * R / nb
            g = np.concatenate([2 * (o_fm - target) / batch, gR, -gR, -gR, gR])
            net.backward_and_step(g)
        else:  # peer, Eq. 12 to 14
            r = draw_routing(rng, y1, y2, seen_by, me)
            # Own-model terms carry gradient; peer terms are frozen (Eq. 13's StopGrad).
            parts = [("J", features(xt_c, t_c, y1, y2), np.full(nb, me))]
            parts += [("M1", features(xt_c, t_c, y1, zero), r[:, 1]), ("M2", features(xt_c, t_c, zero, y2), r[:, 2]),
                      ("U", features(xt_c, t_c, zero, zero), r[:, 0])]
            own_feats, own_slots = [f_fm], []
            vals = {}
            for name, f, who in parts:
                val = np.zeros((nb, 2))
                mine = who == me
                for a, pnet in peers.items():
                    sel = who == a
                    if sel.any():
                        val[sel] = pnet.forward(f[sel])
                vals[name] = val
                own_slots.append((name, np.where(mine)[0]))
                own_feats.append(f[mine])
            out = net.forward(np.concatenate(own_feats), keep=True)
            o_fm = out[:batch]
            pos = batch
            for name, idx in own_slots:
                vals[name][idx] = out[pos:pos + len(idx)]
                pos += len(idx)
            R = vals["J"] - (vals["M1"] + vals["M2"] - vals["U"])
            gR = 2 * lam * R / nb
            sign = {"J": 1.0, "M1": -1.0, "M2": -1.0, "U": 1.0}
            g_parts = [2 * (o_fm - target) / batch]
            for name, idx in own_slots:
                g_parts.append(sign[name] * gR[idx])
            net.backward_and_step(np.concatenate(g_parts))
        if log_every and step % log_every == 0:
            print(step, float(((o_fm - target) ** 2).sum(1).mean()), float((R ** 2).sum(1).mean()))
    return net


SPLITS_CURRENT = SPLITS["non_iid"]


def draw_routing(rng, y1, y2, seen_by, me):
    """r = (r_0, r_1, r_2) per row: r_0 uniform over the silos, r_i uniform over the silos whose data hold y_i,
    redrawn until some entry is `me` (Eq. 12 asks that at least one r_i = a)."""
    n = len(y1)
    r = np.zeros((n, 3), int)
    todo = np.arange(n)
    while len(todo):
        r[todo, 0] = rng.integers(1, 3, size=len(todo))
        for col, (attr, ys) in ((1, (0, y1)), (2, (1, y2))):
            opts = [seen_by[(attr, int(v))] for v in ys[todo]]
            pick = rng.integers(0, 2, size=len(todo))
            r[todo, col] = [o[p % len(o)] for o, p in zip(opts, pick)]
        ok = (r[todo] == me).any(1)
        todo = todo[~ok]
    return r


def interaction_norm(vj, vs, cells, ts=np.linspace(0.05, 0.95, 19), n=400, seed=1):
    """Mean |I_t| = |joint - sum rule| over x_t drawn on the path of each seen cell, per t; returns (ts, curve).

    vj(x, t, y) gives the joint velocity for label y; vs(x, t, y) gives the sum rule for y.
    """
    rng = np.random.default_rng(seed)
    curve = []
    for t in ts:
        acc = []
        for c in cells:
            x1 = np.array(CENTRE[c]) + S * rng.standard_normal((n, 2))
            x0 = rng.standard_normal((n, 2))
            xt = (1 - t) * x0 + t * x1
            acc.append(np.linalg.norm(vj(xt, t, c) - vs(xt, t, c), axis=1).mean())
        curve.append(float(np.mean(acc)))
    return ts, np.array(curve)
