"""SDXL's shape at toy scale: a VAE, one conditional model, and a dial on the captioning policy.

A picture holding both shapes always carries the joint caption. The dial is how often that same
picture is ALSO trained under "a red circle" and under "a blue square". At 0 the two single captions
have never seen a picture containing the other shape, which is the binary captioning SDXL is trained
under. At 1 every both-shapes picture sits inside all three captions.

At inference the two single captions are combined the way this repository combines them, as summed
classifier-free guidance:
    eps = eps_null + g * ((eps_c1 - eps_null) + (eps_c2 - eps_null))

THE BAR, FIXED BEFORE THE FIRST RUN
  support       the compose rate rises to at least 150 of 300 at some dial setting while each single
                caption on its own still draws its own shape alone on at least 240 of 300
  null          the compose rate stays under 30 of 300 at every setting where both captions still
                pass that 240 of 300 test
  inconclusive  the compose rate only rises at settings where caption purity has already fallen

Pictures are read by toy_blob_scorer.py (red pixels, blue pixels, where they sit), because the
template library in toy_sampler_hybrids.py misreads decoded latents. "Compose" is the scorer's "red and
blue apart"; a caption "draws its own shape alone" when it scores "red only" or "blue only". A dial
setting counts only if its VAE reconstructs at most VAE_OK mean squared error and the scorer reads the
VAE's own reconstructions of every known picture right at least 95 times in 100.

    /home/molef/micromamba/envs/sdxl-local/bin/python -I toy_caption_dial_ldm.py
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import toy_sampler_hybrids as t
import toy_blob_scorer as b

OUT = HERE.parents[1] / "results" / "how-much-co-occurrence-the-product-needs"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
H, W, T, N = t.H, t.W, t.T, t.N
DIAL = [0.0, 0.1, 0.3, 0.6, 1.0]
IMAGES, VAE_STEPS, DIF_STEPS, BATCH, LR = 6000, 4000, 8000, 256, 2e-4
GUIDE, PURE, SEED = 3.0, 240, 20261022
VAE_OK, VAE_TRIES = 0.012, 5
CIRCLE, SQUARE, BOTH, NULL = 0, 1, 2, 3          # caption ids; NULL is the empty caption for guidance
abar = torch.tensor(t.abar, dtype=torch.float32, device=DEV)

SINGLES_C = np.stack(list(t.CIRCLES.values()))
SINGLES_S = np.stack(list(t.SQUARES.values()))
PAIRS = np.stack([t.draw([("circle", t.XPOS["left"]), ("square", t.XPOS["right"])]),
                  t.draw([("square", t.XPOS["left"]), ("circle", t.XPOS["right"])])])


def corpus(dial, rng):
    """Every picture with its caption. A both-picture is joint-captioned, and with probability
    `dial` it is additionally entered under each single caption."""
    imgs, caps = [], []
    per = IMAGES // 3
    for bank, cap in ((SINGLES_C, CIRCLE), (SINGLES_S, SQUARE)):
        imgs.append(bank[rng.integers(len(bank), size=per)]); caps += [cap] * per
    pair = PAIRS[rng.integers(len(PAIRS), size=per)]
    imgs.append(pair); caps += [BOTH] * per
    extra = rng.random(per) < dial
    if extra.any():
        imgs.append(pair[extra]); caps += [CIRCLE] * int(extra.sum())
        imgs.append(pair[extra]); caps += [SQUARE] * int(extra.sum())
    x = np.concatenate(imgs) + t.SIGMA0 * rng.standard_normal((len(caps), t.D))
    return (torch.tensor(x.reshape(-1, H, W, 3).transpose(0, 3, 1, 2), dtype=torch.float32, device=DEV),
            torch.tensor(caps, dtype=torch.long, device=DEV))


class VAE(nn.Module):
    """16 x 32 x 3 down to 4 x 8 x 4, which is 128 numbers, the same shape of move SDXL's VAE makes."""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.SiLU(),
                                 nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(), nn.Conv2d(64, 8, 3, 1, 1))
        self.dec = nn.Sequential(nn.Conv2d(4, 64, 3, 1, 1), nn.SiLU(), nn.Upsample(scale_factor=2),
                                 nn.Conv2d(64, 32, 3, 1, 1), nn.SiLU(), nn.Upsample(scale_factor=2),
                                 nn.Conv2d(32, 3, 3, 1, 1))

    def encode(self, x):
        mu, lv = self.enc(x).chunk(2, 1)
        return mu, lv.clamp(-8, 2)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + (0.5 * lv).exp() * torch.randn_like(mu)
        return self.dec(z), mu, lv


class Denoiser(nn.Module):
    def __init__(self, d=128, h=1024):
        super().__init__()
        self.cap = nn.Embedding(4, 64)
        self.f = nn.Sequential(nn.Linear(d + 64 + 64, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                               nn.Linear(h, h), nn.SiLU(), nn.Linear(h, d))
        self.register_buffer("freq", torch.exp(torch.linspace(0, 6, 32)))

    def forward(self, z, k, c):
        a = k.float()[:, None] / T * self.freq[None]
        return self.f(torch.cat([z, torch.sin(a), torch.cos(a), self.cap(c)], 1))


def train_vae_once(x, seed):
    torch.manual_seed(seed)
    v = VAE().to(DEV)
    opt = torch.optim.Adam(v.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, VAE_STEPS)
    for s in range(VAE_STEPS):
        i = torch.randint(len(x), (BATCH,), device=DEV)
        xb = x[i]
        rec, mu, lv = v(xb)
        loss = F.mse_loss(rec, xb) + 1e-4 * (-0.5 * (1 + lv - mu ** 2 - lv.exp())).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    v.eval()
    with torch.no_grad():
        mse = float(F.mse_loss(v.dec(v.encode(x[:1024])[0]), x[:1024]))
    return v, mse


def train_vae(x, seed):
    """Restart on a new seed until reconstruction error is at most VAE_OK; keep the best if none is."""
    best = None
    for k in range(VAE_TRIES):
        v, mse = train_vae_once(x, seed + 1000 * k)
        if best is None or mse < best[1]:
            best = (v, mse, k + 1)
        if mse <= VAE_OK:
            break
    return best


@torch.no_grad()
def scorer_reads_vae(vae):
    """The scorer's hit rate on the VAE's reconstructions of noisy copies of every known picture."""
    rng = np.random.default_rng(1)
    out = {}
    for name, (img, want) in b.KNOWN.items():
        xn = img + t.SIGMA0 * rng.standard_normal((100, img.size))
        xt = torch.tensor(xn.reshape(-1, H, W, 3).transpose(0, 3, 1, 2), dtype=torch.float32, device=DEV)
        rec = vae.dec(vae.encode(xt)[0]).permute(0, 2, 3, 1).reshape(100, -1).cpu().numpy()
        out[name] = sum(c == want for c in b.read(rec)[0]) / 100
    return out


def train_denoiser(z, c, seed):
    torch.manual_seed(seed)
    net = Denoiser(z.shape[1]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    for s in range(DIF_STEPS):
        i = torch.randint(len(z), (BATCH,), device=DEV)
        k = torch.randint(1, T + 1, (BATCH,), device=DEV)
        z0 = z[i]
        cc = torch.where(torch.rand(BATCH, device=DEV) < 0.1, torch.full_like(c[i], NULL), c[i])
        a = abar[k][:, None]
        zt = a.sqrt() * z0 + (1 - a).sqrt() * torch.randn_like(z0)
        loss = ((net(zt, k, cc) - z0) ** 2).mean()          # predict the latent, not the noise
        opt.zero_grad(); loss.backward(); opt.step()
    return net.eval(), float(loss)


@torch.no_grad()
def draw(net, vae, caps, d, n=N):
    """caps is a list of caption ids combined as summed classifier-free guidance."""
    z = torch.randn(n, d, device=DEV)
    nullc = torch.full((n,), NULL, device=DEV, dtype=torch.long)
    for k in range(T, 0, -1):
        a_t, a_p = abar[k], abar[k - 1]
        al = a_t / a_p; be = 1 - al
        kk = torch.full((n,), k, device=DEV, dtype=torch.long)
        e_null = (z - a_t.sqrt() * net(z, kk, nullc)) / (1 - a_t).sqrt()
        e = e_null.clone()
        for c in caps:
            cc = torch.full((n,), c, device=DEV, dtype=torch.long)
            e = e + GUIDE * ((z - a_t.sqrt() * net(z, kk, cc)) / (1 - a_t).sqrt() - e_null)
        mean = (z - be / (1 - a_t).sqrt() * e) / al.sqrt()
        z = mean + ((be * (1 - a_p) / (1 - a_t)).sqrt() if k > 1 else 0.0) * torch.randn_like(z)
    img = vae.dec(z.reshape(n, 4, 4, 8)).permute(0, 2, 3, 1).reshape(n, -1).cpu().numpy()
    labs = t.label(img)
    return ({"blob scorer": b.counts(img), "template library": {c: labs.count(c) for c in t.CATS}},
            np.clip(img[:8], 0, 1))


def main():
    print(f"device {DEV}, dial {DIAL}, guidance {GUIDE}", flush=True)
    results, sheets = {}, {}
    for dial in DIAL:
        t0 = time.time()
        rng = np.random.default_rng(SEED)
        x, c = corpus(dial, rng)
        vae, rec, tries = train_vae(x, SEED)
        reads = scorer_reads_vae(vae)
        readable = rec <= VAE_OK and min(reads.values()) >= 0.95
        with torch.no_grad():
            z = vae.encode(x)[0].reshape(len(x), -1)
        net, dloss = train_denoiser(z, c, SEED)
        poe, s_poe = draw(net, vae, [CIRCLE, SQUARE], z.shape[1])
        circ, s_c = draw(net, vae, [CIRCLE], z.shape[1])
        sq, s_s = draw(net, vae, [SQUARE], z.shape[1])
        joint, s_j = draw(net, vae, [BOTH], z.shape[1])
        pb, cb, qb, jb = (r["blob scorer"] for r in (poe, circ, sq, joint))
        results[f"{dial:g}"] = {
            "the two single captions multiplied": poe, "a red circle alone": circ,
            "a blue square alone": sq, "the joint caption": joint,
            "still distinct by the 240 of 300 test":
                cb["red only"] >= PURE and qb["blue only"] >= PURE,
            "readable": readable, "vae reconstruction mse": round(rec, 5), "vae tries": tries,
            "scorer hit rate on this vae's reconstructions": reads,
            "denoiser loss": round(dloss, 5), "seconds": round(time.time() - t0, 1)}
        sheets[f"{dial:g}"] = (s_c, s_s, s_poe, s_j)
        r = results[f"{dial:g}"]
        print(f"dial {dial:g} ({r['seconds']}s, vae mse {rec:.5f} after {tries} tries, scorer worst "
              f"{min(reads.values()):.2f}, readable {readable}) | product: apart {pb['red and blue apart']}, "
              f"purple {pb['purple or overlaid']}, red only {pb['red only']}, blue only {pb['blue only']} "
              f"of 300 | circle caption red only {cb['red only']}/300, square caption blue only "
              f"{qb['blue only']}/300 | joint caption apart {jb['red and blue apart']}/300 | distinct "
              f"{r['still distinct by the 240 of 300 test']}", flush=True)

    live = {d: r for d, r in results.items() if r["readable"] and r["still distinct by the 240 of 300 test"]}
    comp = {d: r["the two single captions multiplied"]["blob scorer"]["red and blue apart"] for d, r in live.items()}
    ok = [d for d, v in comp.items() if v >= 150]
    hi = [d for d, v in comp.items() if v >= 30]
    verdict = ("no readable setting" if not live else
               "support" if ok else ("null" if not hi else "inconclusive"))
    print("VERDICT:", verdict, "| settings meeting the support bar:", ok or "none", flush=True)
    np.savez_compressed(OUT / "toy-caption-dial-ldm-samples.npz",
                        **{f"{d}_{i}": s for d, ss in sheets.items() for i, s in enumerate(ss)})
    json.dump({"what": "a VAE plus one caption-conditional latent model; the dial is how often a "
                       "both-shapes picture is also trained under each single caption",
               "inference": "summed classifier-free guidance over the two single captions",
               "guidance": GUIDE, "bar": {"support": "compose >= 150 of 300 with both captions still "
               "passing 240 of 300", "null": "compose < 30 of 300 wherever both still pass"},
               "verdict": verdict, "settings_meeting_support": ok, "settings_judged": list(live),
               "scorer": "toy_blob_scorer.py; the template library's counts are kept beside it",
               "device": DEV,
               "images": IMAGES, "vae_steps": VAE_STEPS, "denoiser_steps": DIF_STEPS,
               "latent": "4 x 4 x 8, 128 numbers", "draws_per_cell": N, "results": results},
              open(OUT / "toy-caption-dial-ldm.json", "w"), indent=1)
    print("wrote", OUT / "toy-caption-dial-ldm.json")


if __name__ == "__main__":
    main()
