"""The NoisyCLIP sentence-score push on a network small enough to check by hand.

The latent is 2 pixels with 4 channels. The linear map is the released one, with its real weights and biases. The image encoder
is a hand-set 2 by 6 linear layer, and the two sentence vectors are hand-set unit vectors. Nothing here was trained on data, and
no number here says anything about SDXL or NoisyCLIP's real behaviour. Every number on the miniature's pages comes from the key
this writes. Run: CUDA_VISIBLE_DEVICES= uv run --no-project --with torch python miniature.py
"""
import json
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)
HERE = Path(__file__).resolve().parent
W = torch.tensor([[60., -60, 25, -70], [60, -5, 15, -50], [60, 10, -5, -35]])
BIAS = torch.tensor([150., 140, 130])
M = torch.tensor([[2., 0, 0, 1, 0, 0], [0, 0, 2, 0, 1, 0]])
X = torch.tensor([[1., .5, 0, .5], [1, -1, 0, -.5]])
T_RIGHT = torch.tensor([.8, .6])
T_WRONG = torch.tensor([.6, .8])
BETA = 0.05
SIGMA, SIGNAL, SIGMA_NEXT, SIGNAL_NEXT = .8, .6, .6, .8
LAMBDA = 2.0
EPS = torch.tensor([[.5, 0, 0, -.5], [-.5, .5, 0, 0]])
TOY_SCALE, TOY_LR = 10.0, 0.5


def r(x, n=4):
    if isinstance(x, torch.Tensor):
        return [r(v, n) for v in x.detach().tolist()] if x.ndim else round(float(x.detach()), n)
    if isinstance(x, list):
        return [r(v, n) for v in x]
    return round(float(x), n)


def linear_map(z):
    return z @ W.T + BIAS


def hard(c):
    return c.clamp(0, 255)


def splus(x):
    return torch.nn.functional.softplus(x, beta=BETA)


def soft(c):
    return c - splus(c - 255) + splus(-c)


def encode(z, clamp=hard, m=M):
    p = clamp(linear_map(z)) / 255 - .5
    h = m @ p.reshape(-1)
    return h / h.norm()


def difference(z, clamp=hard, right=T_RIGHT, wrong=T_WRONG):
    v = encode(z, clamp)
    return v @ right - v @ wrong


def grad_of(z, **kw):
    z = z.clone().requires_grad_(True)
    return torch.autograd.grad(difference(z, **kw), z)[0]


# ------------------------------------------------------------------ forward
raw = linear_map(X)
clamped = hard(raw)
p = clamped / 255 - .5
h = M @ p.reshape(-1)
v = h / h.norm()
s_right, s_wrong = v @ T_RIGHT, v @ T_WRONG
d = s_right - s_wrong

# ------------------------------------------------------------------ backward, by hand and by autograd
delta = T_RIGHT - T_WRONG
dD_dh = (delta - (delta @ v) * v) / h.norm()
dD_dp = (M.T @ dD_dh).reshape(2, 3)
alive = ((raw >= 0) & (raw <= 255)).double()
dD_dc = dD_dp * alive / 255
by_hand = dD_dc @ W
g = grad_of(X)
assert (by_hand - g).abs().max() < 1e-12, 'the hand chain rule and autograd disagree'
fd = torch.zeros_like(X)
for i in range(2):
    for l in range(4):
        e = torch.zeros_like(X); e[i, l] = 1e-6
        fd[i, l] = (difference(X + e) - difference(X - e)) / 2e-6
assert (fd - g).abs().max() < 1e-7, 'finite differences disagree'

# ------------------------------------------------------------------ the push and one DDIM step
push = LAMBDA * SIGMA * g
eps_hat = EPS - push


def ddim(x, e):
    return SIGNAL_NEXT * (x - SIGMA * e) / SIGNAL + SIGMA_NEXT * e


x_plain, x_pushed = ddim(X, EPS), ddim(X, eps_hat)
d_plain, d_pushed = difference(x_plain), difference(x_pushed)
coefficient = SIGMA_NEXT - SIGNAL_NEXT * SIGMA / SIGNAL
assert (x_pushed - x_plain - (-coefficient) * push).abs().max() < 1e-12

# ------------------------------------------------------------------ three ways it breaks
c_grad = linear_map(X.clone().requires_grad_(True))
as_bytes = c_grad.clamp(0, 255).to(torch.uint8)
byte = {"has_gradient": as_bytes.requires_grad, "example_in": 127.9, "example_out": int(torch.tensor(127.9).to(torch.uint8))}
same = {"difference": r(difference(X, right=T_RIGHT, wrong=T_RIGHT)), "gradient_max": r(grad_of(X, right=T_RIGHT, wrong=T_RIGHT).abs().max())}
X_all = torch.tensor([[1., .5, 0, .5], [3, -3, 0, -3]])
c_all = linear_map(X_all)
g_all = grad_of(X_all)
assert (g_all[1] == 0).all()
g_soft = grad_of(X, clamp=soft)
slope_soft = 1 - torch.sigmoid(torch.tensor(BETA * (float(raw[1, 0]) - 255)))

# ------------------------------------------------------------------ one contrastive training step on the image layer
X2 = X.flip(0)
T = torch.stack([T_RIGHT, T_WRONG])


def contrastive(m):
    I = torch.stack([encode(X, m=m), encode(X2, m=m)])
    logits = TOY_SCALE * I @ T.T
    y = torch.arange(2)
    return I, logits, (torch.nn.functional.cross_entropy(logits, y) + torch.nn.functional.cross_entropy(logits.T, y)) / 2


m0 = M.clone().requires_grad_(True)
I0, logits0, loss0 = contrastive(m0)
grad_m = torch.autograd.grad(loss0, m0)[0]
m1 = M - TOY_LR * grad_m
I1, logits1, loss1 = contrastive(m1)

# ------------------------------------------------------------------ the score difference against a two-sentence classifier


def log_p_right(z):
    a, b_ = TOY_SCALE * (encode(z) @ T_RIGHT), TOY_SCALE * (encode(z) @ T_WRONG)
    return a - torch.logaddexp(a, b_)


z_lp = X.clone().requires_grad_(True)
lp = log_p_right(z_lp)
grad_lp = torch.autograd.grad(lp, z_lp)[0]
p_right = lp.exp()
factor = (1 - p_right) * TOY_SCALE
assert (grad_lp - factor * g).abs().max() < 1e-12, 'the two-sentence identity failed'

key = {
    "note": "A toy with hand-set weights. It shows what each part computes and predicts nothing about SDXL or NoisyCLIP.",
    "latent": r(X), "weights": r(W), "biases": r(BIAS), "image_layer": r(M), "t_right": r(T_RIGHT), "t_wrong": r(T_WRONG),
    "rgb_before_clamp": r(raw), "rgb_after_clamp": r(clamped), "scaled": r(p), "h": r(h), "h_length": r(h.norm()), "v": r(v),
    "s_right": r(s_right), "s_wrong": r(s_wrong), "difference": r(d),
    "dD_dh": r(dD_dh), "dD_dp": r(dD_dp), "alive": r(alive, 0), "dD_dc": r(dD_dc, 6), "g": r(g), "g_length": r(g.norm()),
    "checks": {"hand_vs_autograd_max": float((by_hand - g).abs().max()), "finite_difference_vs_autograd_max": float((fd - g).abs().max())},
    "sampler": {"sigma": SIGMA, "signal": SIGNAL, "sigma_next": SIGMA_NEXT, "signal_next": SIGNAL_NEXT, "lambda": LAMBDA,
                "eps_base": r(EPS), "push": r(push), "eps_hat": r(eps_hat), "x_plain": r(x_plain), "x_pushed": r(x_pushed),
                "coefficient_on_eps": r(coefficient), "difference_plain": r(d_plain), "difference_pushed": r(d_pushed)},
    "breaks": {"byte": byte, "same_sentence": same,
               "all_three_clamped": {"latent_pixel": r(X_all[1]), "rgb_before_clamp": r(c_all[1]), "g_row": r(g_all[1])},
               "hard_vs_soft": {"beta": BETA, "red_b_before_clamp": r(raw[1, 0]), "soft_slope_there": r(slope_soft), "g_hard": r(g), "g_soft": r(g_soft)}},
    "two_sentence_classifier": {"scale": TOY_SCALE, "p_right": r(p_right), "factor": r(factor), "identity_max_error": float((grad_lp - factor * g).abs().max())},
    "training": {"scale": TOY_SCALE, "learning_rate": TOY_LR, "image_vectors_before": r(I0), "logits_before": r(logits0), "loss_before": r(loss0),
                 "layer_gradient": r(grad_m), "layer_after": r(m1), "image_vectors_after": r(I1), "logits_after": r(logits1), "loss_after": r(loss1),
                 "text_vectors_before": r(T), "text_vectors_after": r(T)},
}
(HERE / "miniature_answer_key.json").write_text(json.dumps(key, indent=1))
print(json.dumps({k: key[k] for k in ("rgb_before_clamp", "rgb_after_clamp", "h", "v", "s_right", "s_wrong", "difference", "dD_dh", "g", "checks")}))
print(json.dumps(key["sampler"]))
print(json.dumps(key["breaks"]))
print(json.dumps(key["training"]))
