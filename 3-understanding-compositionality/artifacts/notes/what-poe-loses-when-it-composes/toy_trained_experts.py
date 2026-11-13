"""Two networks trained separately, then multiplied: does learning buy what the exact arithmetic does not?

Each expert is a small score network trained on its own pictures, with a controlled fraction of its
training set replaced by pictures holding both shapes. The same shared pictures go into both sets, so
the two experts agree on them the way condition D's hand-set weights did, except that here the
agreement has to survive training rather than being written down.

THE BAR, FIXED BEFORE THE FIRST RUN
  support      at some shared fraction, the product composes on at least 150 of 300 draws while each
               expert on its own still draws its own shape on at least 240 of 300, so the two are
               still distinct models
  null         the product's crisp count stays under 30 of 300 at every fraction where both experts
               are still distinct by that same 240-of-300 test
  inconclusive the crisp count only rises at fractions where expert purity has already fallen below
               240 of 300, which is the toy's degeneracy rather than a win

Run with the environment that has torch:
    /home/molef/micromamba/envs/sdxl-local/bin/python -I toy_trained_experts.py
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import toy_sampler_hybrids as t          # the picture bank, the schedule and the labelling library

OUT = HERE.parents[1] / "results" / "how-much-co-occurrence-the-product-needs"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
D, T, N = t.D, t.T, t.N
FRACTIONS = [0.0, 0.001, 0.01, 0.1, 0.3, 0.5]
TRAIN_IMAGES, STEPS, BATCH, LR = 4000, 6000, 256, 2e-4
PURE = 240                                # a draw count out of 300: the "still distinct" test
SEED = 20261022

CIRCLES = np.stack(list(t.CIRCLES.values()))
SQUARES = np.stack(list(t.SQUARES.values()))
PAIRS = np.stack([t.draw([("circle", t.XPOS["left"]), ("square", t.XPOS["right"])]),
                  t.draw([("square", t.XPOS["left"]), ("circle", t.XPOS["right"])])])
abar = torch.tensor(t.abar, dtype=torch.float32, device=DEV)


def build_set(own, frac, rng):
    n_pair = int(round(frac * TRAIN_IMAGES))
    idx_own = rng.integers(len(own), size=TRAIN_IMAGES - n_pair)
    idx_pair = rng.integers(len(PAIRS), size=n_pair)
    x = np.concatenate([own[idx_own], PAIRS[idx_pair]]) if n_pair else own[idx_own]
    x = x + t.SIGMA0 * rng.standard_normal(x.shape)       # the same spread the exact toy uses
    return torch.tensor(x, dtype=torch.float32, device=DEV)   # the toy's own 0-to-1 convention


class Net(nn.Module):
    def __init__(self, h=1024):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(D + 64, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                               nn.Linear(h, h), nn.SiLU(), nn.Linear(h, D))
        self.register_buffer("freq", torch.exp(torch.linspace(0, 8, 32)))

    def forward(self, x, k):
        a = k.float()[:, None] / T * self.freq[None]
        return self.f(torch.cat([x, torch.sin(a), torch.cos(a)], 1))


def train(data, seed):
    torch.manual_seed(seed)
    net = Net().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    for s in range(STEPS):
        i = torch.randint(len(data), (BATCH,), device=DEV)
        k = torch.randint(1, T + 1, (BATCH,), device=DEV)
        x0, eps = data[i], torch.randn(BATCH, D, device=DEV)
        a = abar[k][:, None]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
        loss = ((net(xt, k) - x0) ** 2).mean()     # predict the picture, not the noise:
        # with seven modes the picture is easy to learn and the reverse step stays stable
        opt.zero_grad(); loss.backward(); opt.step()
    return net.eval(), float(loss)


def eps_of(net):
    """Turn a picture-predicting network into the noise prediction the reverse step wants."""
    def f(x, k):
        a = abar[k][:, None]
        return (x - a.sqrt() * net(x, k)) / (1 - a).sqrt()
    return f


@torch.no_grad()
def sample(eps_fn, n=N):
    x = torch.randn(n, D, device=DEV)
    for k in range(T, 0, -1):
        a_t, a_prev = abar[k], abar[k - 1]
        alpha = a_t / a_prev; beta = 1 - alpha
        kk = torch.full((n,), k, device=DEV, dtype=torch.long)
        mean = (x - beta / (1 - a_t).sqrt() * eps_fn(x, kk)) / alpha.sqrt()
        sig = (beta * (1 - a_prev) / (1 - a_t)).sqrt() if k > 1 else 0.0
        x = mean + sig * torch.randn_like(x)
    return x.cpu().numpy()


def counts(x):
    labs = t.label(x)
    return {c: labs.count(c) for c in t.CATS}


def main():
    print(f"device {DEV}, {len(FRACTIONS)} fractions, {TRAIN_IMAGES} pictures and {STEPS} steps per expert",
          flush=True)
    results = {}
    for frac in FRACTIONS:
        t0 = time.time()
        rng = np.random.default_rng(SEED)
        A, lossA = train(build_set(CIRCLES, frac, rng), SEED)
        B, lossB = train(build_set(SQUARES, frac, rng), SEED + 1)
        torch.manual_seed(SEED)
        ea, eb = eps_of(A), eps_of(B)
        poe = counts(sample(lambda x, k: ea(x, k) + eb(x, k)))
        torch.manual_seed(SEED)
        alone_a = counts(sample(ea))
        torch.manual_seed(SEED)
        alone_b = counts(sample(eb))
        results[f"{frac:g}"] = {
            "product": poe, "circle expert alone": alone_a, "square expert alone": alone_b,
            "circle expert draws a circle": alone_a["one circle"],
            "square expert draws a square": alone_b["one square"],
            "still distinct by the 240 of 300 test":
                alone_a["one circle"] >= PURE and alone_b["one square"] >= PURE,
            "final training loss": [round(lossA, 5), round(lossB, 5)],
            "seconds": round(time.time() - t0, 1)}
        r = results[f"{frac:g}"]
        print(f"frac {frac:g} ({r['seconds']}s): product crisp {poe['two objects, crisp']}/300, "
              f"overlay {poe['hybrid: overlaid']}/300 | circle alone draws a circle "
              f"{r['circle expert draws a circle']}/300, square alone a square "
              f"{r['square expert draws a square']}/300 | still distinct: {r['still distinct by the 240 of 300 test']}",
              flush=True)

    ok = [f for f, r in results.items()
          if r["still distinct by the 240 of 300 test"] and r["product"]["two objects, crisp"] >= 150]
    hi = [f for f, r in results.items()
          if r["still distinct by the 240 of 300 test"] and r["product"]["two objects, crisp"] >= 30]
    verdict = ("support" if ok else ("null" if not hi else "inconclusive"))
    print("VERDICT:", verdict, "| fractions meeting the support bar:", ok or "none", flush=True)
    json.dump({"what": "two score networks trained separately on overlapping picture sets, then multiplied",
               "bar": {"support": "crisp >= 150 of 300 while both experts still pass 240 of 300 on their own shape",
                       "null": "crisp < 30 of 300 at every fraction where both experts still pass 240 of 300",
                       "inconclusive": "crisp rises only where expert purity has already fallen"},
               "verdict": verdict, "fractions_meeting_support": ok,
               "device": DEV, "train_images_per_expert": TRAIN_IMAGES, "train_steps": STEPS,
               "batch": BATCH, "lr": LR, "sigma0": t.SIGMA0, "reverse_steps": T, "draws_per_cell": N,
               "shared_pair_layouts": ["circle left square right", "square left circle right"],
               "results": results},
              open(OUT / "toy-trained-experts.json", "w"), indent=1)
    print("wrote", OUT / "toy-trained-experts.json")


if __name__ == "__main__":
    main()
