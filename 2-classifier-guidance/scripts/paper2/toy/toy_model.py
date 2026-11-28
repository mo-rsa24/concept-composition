"""The base model, the rank-4 dials, the sampler, the soft reader and the stacking correction for Paper 2's miniature toy rung.

The base model is Paper 1's squares-toy U-Net (16 -> 8 -> 4 -> 8 -> 16, EDM preconditioning, sigma_data 0.5) with the condition
added to the noise embedding through a linear map whose weight starts at zero, so a condition column the training canvases never set
(a held-out combination's own count) stays exactly zero and the model reads that combination only through its shape and colour
counts.

A dial is a rank-4 adapter on every convolution and linear layer of the frozen network: y = W x + s * up(down(x)), up starting at
zero. An adapter that is not active is never computed, so a model with a dial loaded but switched off is the frozen model exactly.
The stacking correction scales each input column of each dial's down map by a weight m (LoRA.rar and ZipLoRA's merger coefficients
on the columns of the dial's weight change B A), m = 1 + a small network's output that starts at zero, so before training the
corrected stack is the plain stack exactly.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "paper1" / "toy"))
import squares_toy as st  # noqa: E402  (EDM pieces, the U-Net block, the sigma schedule, starting noise)

import toy_canvas as tc  # noqa: E402

DEV = st.DEV
RANK = 4
TAU = 0.25                       # soft reader temperature: softmax(-||x - palette||^2 / TAU)


class Attn(nn.Module):
    """Self-attention over a feature map's pixels, the residual's output layer starting at zero so an attention base begins as the
    plain U-Net. Linear layers only, so a dial's adapters wrap the query, key, value and output maps like any other layer."""

    def __init__(self, c, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(st.GROUPS, c)
        self.qkv = nn.Linear(c, 3 * c)
        self.proj = nn.Linear(c, c)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h):
        b, c, hh, ww = h.shape
        t = self.norm(h).flatten(2).transpose(1, 2)                              # (b, pixels, c)
        q, k, v = self.qkv(t).chunk(3, -1)
        sh = lambda z: z.view(b, -1, self.heads, c // self.heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(sh(q), sh(k), sh(v)).transpose(1, 2).reshape(b, -1, c)
        return h + self.proj(y).transpose(1, 2).view(b, c, hh, ww)


class CondUNet(nn.Module):
    def __init__(self, w=64, cond_dim=tc.COND_DIM, attn=False):
        super().__init__()
        self.attn = attn
        if attn:
            self.a2, self.a3 = Attn(2 * w), Attn(2 * w)            # at 8 by 8 after e2, at 4 by 4 after m1
        E = st.EMB
        self.cond = nn.Sequential(nn.Linear(st.FOURIER, E), nn.SiLU(), nn.Linear(E, E), nn.SiLU())
        self.label = nn.Linear(cond_dim, E, bias=False)
        nn.init.zeros_(self.label.weight)
        self.conv_in = nn.Conv2d(3, w, 3, padding=1)
        self.e1, self.e2 = st.Block(w, w), st.Block(w, 2 * w)
        self.m1, self.m2 = st.Block(2 * w, 2 * w), st.Block(2 * w, 2 * w)
        self.d2, self.d1 = st.Block(4 * w, 2 * w), st.Block(3 * w, w)
        self.conv_out = nn.Conv2d(w, 3, 3, padding=1)
        self.logvar_linear = nn.Linear(st.FOURIER, 1)
        nn.init.zeros_(self.logvar_linear.weight)
        nn.init.zeros_(self.logvar_linear.bias)

    def forward(self, h, sigma, cond):
        e = self.cond(st.fourier(sigma)) + self.label(cond)
        h1 = self.e1(self.conv_in(h), e)
        h2 = self.e2(F.avg_pool2d(h1, 2), e)
        if self.attn:
            h2 = self.a2(h2)
        h3 = self.m1(F.avg_pool2d(h2, 2), e)
        if self.attn:
            h3 = self.a3(h3)
        h3 = self.m2(h3, e)
        h = self.d2(torch.cat([F.interpolate(h3, scale_factor=2, mode="nearest"), h2], 1), e)
        h = self.d1(torch.cat([F.interpolate(h, scale_factor=2, mode="nearest"), h1], 1), e)
        return self.conv_out(h)

    def logvar(self, sigma):
        return self.logvar_linear(st.fourier(sigma)).view(-1, 1, 1, 1)


class Denoiser(nn.Module):
    """D(x, sigma, cond) = c_skip x + c_out F(c_in x, c_noise, cond)."""

    def __init__(self, w=64, attn=False):
        super().__init__()
        self.net = CondUNet(w, attn=attn)

    def forward(self, x, sigma, cond):
        sigma = torch.as_tensor(sigma, device=x.device, dtype=torch.float32).reshape(-1).expand(len(x))
        s4 = sigma.view(-1, 1, 1, 1)
        return st.c_skip(s4) * x + st.c_out(s4) * self.net(st.c_in(s4) * x, sigma, cond)


# -------------------------------------------------------------------------------------------------------------------------------------
# Dials


class Adapter(nn.Module):
    def __init__(self, base, r=RANK):
        super().__init__()
        if isinstance(base, nn.Conv2d):
            self.down = nn.Conv2d(base.in_channels, r, base.kernel_size, base.stride, base.padding, bias=False)
            self.up = nn.Conv2d(r, base.out_channels, 1, bias=False)
        else:
            self.down = nn.Linear(base.in_features, r, bias=False)
            self.up = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def delta_w(self):
        """The weight change up @ down as an (out, in * k * k) matrix."""
        d = self.down.weight.flatten(1)
        u = self.up.weight.flatten(1)
        return u @ d


class LoRALayer(nn.Module):
    """A frozen conv or linear layer with any number of named adapters; only the active ones run."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.adapters = nn.ModuleDict()
        self.active = {}             # name -> strength
        self.colw = {}               # name -> tensor (in_channels,) or absent

    @property
    def conv(self):
        return isinstance(self.base, nn.Conv2d)

    @property
    def in_dim(self):
        return self.base.in_channels if self.conv else self.base.in_features

    def forward(self, x):
        y = self.base(x)
        for n, s in self.active.items():
            if s == 0:
                continue
            a = self.adapters[n]
            w = self.colw.get(n)
            xi = x if w is None else x * (w.view(1, -1, 1, 1) if self.conv else w.view(1, -1))
            y = y + s * a.up(a.down(xi))
        return y


def wrap(model):
    """Replace every conv and linear layer of model.net (except the training-only uncertainty head) with a LoRALayer, in place."""
    def rec(mod, prefix):
        for k, ch in list(mod.named_children()):
            path = f"{prefix}.{k}" if prefix else k
            if path == "logvar_linear":
                continue
            if isinstance(ch, (nn.Conv2d, nn.Linear)):
                setattr(mod, k, LoRALayer(ch))
            else:
                rec(ch, path)
    rec(model.net, "")
    return model


def layers(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALayer)]


def add_dial(model, name, seed=0):
    torch.manual_seed(seed)
    for _, m in layers(model):
        m.adapters[name] = Adapter(m.base).to(next(m.base.parameters()).device)


def dial_params(model, name):
    return [p for _, m in layers(model) for p in m.adapters[name].parameters()]


def set_active(model, strengths):
    for _, m in layers(model):
        m.active = {n: s for n, s in strengths.items() if n in m.adapters}


def set_colw(model, colw):
    """colw: {layer name: {dial: tensor}} or {} to clear."""
    for n, m in layers(model):
        m.colw = colw.get(n, {})


def dial_state(model, name):
    return {n: {k: v.detach().cpu() for k, v in m.adapters[name].state_dict().items()} for n, m in layers(model)}


def load_dial(model, name, state):
    for n, m in layers(model):
        m.adapters[name] = Adapter(m.base).to(next(m.base.parameters()).device)
        m.adapters[name].load_state_dict(state[n])


def random_like(state, seed):
    """A random dial with the same Frobenius norm of up @ down as the given dial in every layer."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for n, sd in state.items():
        d = torch.randn(sd["down.weight"].shape, generator=g)
        u = torch.randn(sd["up.weight"].shape, generator=g)
        want = (sd["up.weight"].flatten(1) @ sd["down.weight"].flatten(1)).norm()
        have = (u.flatten(1) @ d.flatten(1)).norm()
        out[n] = {"down.weight": d, "up.weight": u * (want / have if have > 0 else 0.0)}
    return out


def freeze_base(model):
    for p in model.parameters():
        p.requires_grad_(False)


def base_config(path):
    """The base's shape from its sidecar (<base>.json: width and attention); a base without one is the original width-64 U-Net."""
    import json
    side = Path(path).with_suffix(".json")
    if not side.exists():
        return {}
    cfg = json.loads(side.read_text())
    return {"w": cfg.get("width", 64), "attn": cfg.get("attn", False)}


def load_base(path, dials=(), dial_states=None):
    """A fresh frozen base model, wrapped, with only the named dials loaded (a dial-alone row loads nothing else)."""
    m = Denoiser(**base_config(path)).to(DEV)
    m.load_state_dict(torch.load(path, map_location=DEV))
    freeze_base(m)
    m.eval()
    wrap(m)
    for d in dials:
        load_dial(m, d, dial_states[d])
        for p in dial_params(m, d):
            p.requires_grad_(False)
    return m


# -------------------------------------------------------------------------------------------------------------------------------------
# Sampler: EDM2's deterministic Heun, the same arithmetic as squares_toy.heun, with a start and stop step


def heun_range(denoise, x, start=0, stop=st.STEPS, on_step=None):
    """denoise(x, sigma, step, second) -> D. Runs steps start..stop-1 and returns the state at sigma t[stop]."""
    t = st.sigmas(x.device)
    for i in range(start, stop):
        t_cur, t_next = t[i], t[i + 1]
        d_cur = (x - denoise(x, t_cur, i, False)) / t_cur
        x_next = x + (t_next - t_cur) * d_cur
        if i < st.STEPS - 1:
            d_eval = denoise(x_next, t_next, i, True)
            x_next = x + (t_next - t_cur) * 0.5 * (d_cur + (x_next - d_eval) / t_next)
        x = x_next
        if on_step is not None:
            on_step(i, x, t_next)
    return x


def denoiser_for(model, cond):
    def denoise(x, s, step, second):
        return model(x, s, cond[:len(x)])
    return denoise


@torch.no_grad()
def sample(model, conds, seeds, batch=1024):
    """Final pictures for one condition list and matching seeds; starting noise per seed is squares_toy's, so it is shared by rows."""
    out = []
    cv = tc.cond_vector(conds).to(DEV)
    for i in range(0, len(seeds), batch):
        x = st.starting_noise(seeds[i:i + batch])
        out.append(heun_range(denoiser_for(model, cv[i:i + batch]), x).cpu())
    return torch.cat(out)


# -------------------------------------------------------------------------------------------------------------------------------------
# Soft reader and the two gradient losses


def soft_probs(x):
    """(N, 5, H, W): softmax over palette entries of -||x - p||^2 / TAU."""
    d = ((x[:, None] - tc.PALETTE.to(x.device).view(1, 5, 3, 1, 1)) ** 2).sum(2)
    return torch.softmax(-d / TAU, 1)


def _assign(objs, asked):
    """Best assignment of drawn objects to asked combinations (equal numbers), by colour match then template IoU, brute force."""
    import itertools
    best, arg = None, None
    for perm in itertools.permutations(range(len(asked))):
        cost = 0.0
        for o, j in zip(objs, perm):
            k = asked[j]
            cost += (o["colour"] != tc.colour_of(k)) * 1.0 + (1 - o["ious"][tc.shape_of(k)]) * 0.5
        if best is None or cost < best:
            best, arg = cost, perm
    return arg


def _objects_with_ious(reading):
    out = []
    for (s, c, cy, cx, n), m in zip(reading["objects"], reading["masks"]):
        out.append({"colour": c, "cy": cy, "cx": cx, "mask": m, "ious": tc.shape_ious(m)})
    return out


def targets_from(x0hat, conds, kind):
    """Read the decoded estimates (no gradient) and build per-canvas targets for the soft loss.

    identity: a target occupancy map with each object's asked template placed at the object's centre (held fixed), and a 7 by 7
              window around each object where the loss is read.
    binding:  a target colour index per pixel of each object's own pixels (held fixed), and that pixel mask.
    Canvases whose object count differs from the asked count get an all-zero mask (no loss)."""
    import numpy as np
    R = tc.read(x0hat.detach(), keep_masks=True)
    N = len(R)
    tgt = np.zeros((N, tc.H, tc.H), np.float32)
    win = np.zeros((N, tc.H, tc.H), np.float32)
    col = np.zeros((N, tc.H, tc.H), np.int64)
    used = 0
    for i, (r, c) in enumerate(zip(R, conds)):
        objs = _objects_with_ious(r)
        asked = list(c)
        if len(objs) != len(asked) or r["specks"]:
            continue
        used += 1
        perm = _assign(objs, asked)
        for o, j in zip(objs, perm):
            k = asked[j]
            if kind == "identity":
                r0 = min(max(int(round(o["cy"])) - 2, 0), tc.H - tc.OBJ)
                q0 = min(max(int(round(o["cx"])) - 2, 0), tc.H - tc.OBJ)
                tgt[i, r0:r0 + 5, q0:q0 + 5] = np.maximum(tgt[i, r0:r0 + 5, q0:q0 + 5], tc.TEMPLATES[tc.shape_of(k)])
                win[i, max(r0 - 1, 0):r0 + 6, max(q0 - 1, 0):q0 + 6] = 1
            else:
                win[i][o["mask"]] = 1
                col[i][o["mask"]] = tc.colour_of(k) + 1
    t = lambda a: torch.from_numpy(a).to(x0hat.device)  # noqa: E731
    return t(tgt), t(win), t(col), used


def soft_loss(x0hat, tgt, win, col, kind):
    p = soft_probs(x0hat)
    if kind == "identity":
        occ = 1 - p[:, 0]
        per = (win * (occ - tgt) ** 2).flatten(1).sum(1) / win.flatten(1).sum(1).clamp_min(1)
    else:
        lp = torch.log(p.clamp_min(1e-6)).gather(1, col[:, None])[:, 0]
        per = -(win * lp).flatten(1).sum(1) / win.flatten(1).sum(1).clamp_min(1)
    has = (win.flatten(1).sum(1) > 0).float()
    return (per * has).sum() / has.sum().clamp_min(1)


# -------------------------------------------------------------------------------------------------------------------------------------
# The stacking correction


class Correction(nn.Module):
    """For every layer, dial in the stack and input column c of that dial's weight change: features [log ||dW_i[:, c]||, the largest
    |cos| of dW_i[:, c] with the same column of any other dial in the stack, s_i, the mean strength of the others] plus a learned
    layer embedding -> an MLP -> m = 1 + output, the last layer starting at zero. Pooled over the other dials, so it takes pairs and
    triples alike."""

    def __init__(self, n_layers, hidden=64, emb=8):
        super().__init__()
        self.emb = nn.Embedding(n_layers, emb)
        self.mlp = nn.Sequential(nn.Linear(4 + emb, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def static_features(model, dials):
        """{layer: {dial: (in, 2) tensor}} from the loaded dials' weights: log column norm, max |cos| with other dials' columns."""
        out = {}
        for n, m in layers(model):
            cols = {}
            for d in dials:
                a = m.adapters[d]
                W = a.delta_w().detach()                                 # (out, in * k * k)
                k2 = W.shape[1] // m.in_dim
                cols[d] = W.view(W.shape[0], m.in_dim, k2).permute(1, 0, 2).flatten(1)    # (in, out * k * k)
            f = {}
            for d in dials:
                nrm = cols[d].norm(dim=1)
                cos = torch.zeros_like(nrm)
                for e in dials:
                    if e != d:
                        c = F.cosine_similarity(cols[d], cols[e], dim=1).abs()
                        cos = torch.maximum(cos, c)
                f[d] = torch.stack([torch.log(nrm + 1e-8), cos], 1)
            out[n] = f
        return out

    def forward(self, static, strengths, layer_index):
        """-> {layer: {dial: m (in,)}}."""
        dials = list(strengths)
        out = {}
        for n, f in static.items():
            e = self.emb.weight[layer_index[n]]
            res = {}
            for d in dials:
                others = [strengths[o] for o in dials if o != d]
                s_o = sum(others) / len(others) if others else 0.0
                x = torch.cat([f[d], torch.tensor([[strengths[d], s_o]], device=f[d].device).expand(len(f[d]), 2),
                               e[None].expand(len(f[d]), -1)], 1)
                res[d] = 1 + self.mlp(x)[:, 0]
            out[n] = res
        return out


def overlap_penalty(colw):
    """Mean over layers and dial pairs of |cos(m_i, m_j)| (ZipLoRA's term on merger coefficients; LoRA.rar Eq. 3)."""
    tot, n = 0.0, 0
    for res in colw.values():
        ds = list(res)
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                tot = tot + F.cosine_similarity(res[ds[i]], res[ds[j]], dim=0).abs()
                n += 1
    return tot / max(n, 1)
