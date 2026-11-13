"""Does a Langevin corrector help PoE draw both shapes in the trained caption model?

Four trainings from the exact-count sweep (toy_caption_threshold.py) are retrained with the same
seeds: 2,000 relabelled pictures at seeds 20261022, 20261023, 20261024 and 200 pictures at seed
20261023. Each is sampled twice with the same starting noise: plain PoE (summed guidance 3, reverse
diffusion, 200 steps), and PoE with a corrector. 300 samples each, colour-blob scorer.

The corrector: after each reverse step lands at noise level j (j >= 1), CORR_STEPS Langevin steps
at that level, z <- z + eta * s + sqrt(2 eta) * noise, where s = -e / sqrt(1 - abar_j) is the
score implied by the same summed-guidance noise estimate e, and eta = CORR_SIZE * (1 - abar_j).
The single captions are drawn with the same corrector so their purity is checked under it.

Written before the first run:
  - Judged on the main setting (CORR_STEPS 5, CORR_SIZE 0.05, the setting the formula toy used).
  - Support for "the sampler limits the count": the median over the four trainings of the
    separate-pair count rises by at least 30 of 300 with the corrector, while "a red circle" and
    "a blue square" alone, under the corrector, each give at least 240 of 300 of their own colour.
  - Null: the median moves by less than 30 either way.
  - A fall of 30 or more is reported as the corrector hurting.
  - One stronger setting (CORR_STEPS 20, same size) is run and reported beside it, not judged.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
import toy_caption_dial_ldm as m
import toy_caption_threshold as th
import toy_blob_scorer as b

TAG = ""
RUNS = [(2000, 20261022), (2000, 20261023), (2000, 20261024), (200, 20261023)]
SETTINGS = {"main": (5, 0.05), "stronger": (20, 0.05)}
abar = m.abar


@torch.no_grad()
def draw(net, vae, caps, d, steps=0, size=0.0, seed=0, n=m.N, g=m.GUIDE):
    gen = torch.Generator(device=m.DEV).manual_seed(seed)
    z = torch.randn(n, d, device=m.DEV, generator=gen)
    nullc = torch.full((n,), m.NULL, device=m.DEV, dtype=torch.long)

    def eps(z, k):
        a = abar[k]; kk = torch.full((n,), k, device=m.DEV, dtype=torch.long)
        e0 = (z - a.sqrt() * net(z, kk, nullc)) / (1 - a).sqrt()
        e = e0.clone()
        for c in caps:
            cc = torch.full((n,), c, device=m.DEV, dtype=torch.long)
            e = e + g * ((z - a.sqrt() * net(z, kk, cc)) / (1 - a).sqrt() - e0)
        return e

    for k in range(m.T, 0, -1):
        a_t, a_p = abar[k], abar[k - 1]
        al = a_t / a_p; be = 1 - al
        e = eps(z, k)
        mean = (z - be / (1 - a_t).sqrt() * e) / al.sqrt()
        noise = torch.randn(z.shape, device=m.DEV, generator=gen)
        z = mean + ((be * (1 - a_p) / (1 - a_t)).sqrt() if k > 1 else 0.0) * noise
        j = k - 1
        if steps and j >= 1:
            eta = size * (1 - abar[j])
            for _ in range(steps):
                s = -eps(z, j) / (1 - abar[j]).sqrt()
                z = z + eta * s + (2 * eta).sqrt() * torch.randn(z.shape, device=m.DEV, generator=gen)
    img = vae.dec(z.reshape(n, 4, 4, 8)).permute(0, 2, 3, 1).reshape(n, -1).cpu().numpy()
    return b.counts(img), np.clip(img[:8], 0, 1)


def main():
    res, sheets = [], {}
    for k, s in RUNS:
        t0 = time.time()
        torch.manual_seed(s)
        x, c = th.corpus_k(k, np.random.default_rng(s))
        vae, rec, _ = m.train_vae(x, s)
        with torch.no_grad():
            z = vae.encode(x)[0].reshape(len(x), -1)
        net, _ = m.train_denoiser(z, c, s)
        row = {"pictures": k, "seed": s, "vae mse": round(rec, 5)}
        for name, (st, sz) in [("plain", (0, 0.0))] + list(SETTINGS.items()):
            out = {}
            for cap, caps in (("product", [m.CIRCLE, m.SQUARE]), ("circle", [m.CIRCLE]), ("square", [m.SQUARE])):
                cnt, tiles = draw(net, vae, caps, z.shape[1], st, sz, seed=s + 7)
                out[cap] = cnt
                if cap == "product": sheets[f"{k}_{s}_{name}"] = tiles
            row[name] = out
            print(f"{k} pictures seed {s} {name}: PoE apart {out['product']['red and blue apart']}, purple "
                  f"{out['product']['purple or overlaid']}, single {out['product']['red only'] + out['product']['blue only']}, "
                  f"none {out['product']['nothing readable']} | circle red {out['circle']['red only']}, square blue "
                  f"{out['square']['blue only']}", flush=True)
        row["seconds"] = round(time.time() - t0, 1)
        res.append(row)
    med = lambda key: float(np.median([r[key]["product"]["red and blue apart"] for r in res]))
    pure = all(r["main"]["circle"]["red only"] >= m.PURE and r["main"]["square"]["blue only"] >= m.PURE for r in res)
    d = med("main") - med("plain")
    verdict = ("support" if d >= 30 and pure else "null" if abs(d) < 30 else
               "the corrector hurts" if d <= -30 else "rise, but a single caption lost its purity")
    print(f"median plain {med('plain')}, main {med('main')}, stronger {med('stronger')} | purity under main {pure} | VERDICT {verdict}", flush=True)
    np.savez_compressed(m.OUT / f"toy-caption-corrector{TAG}-samples.npz", **sheets)
    json.dump({"what": "Langevin corrector on PoE in the trained caption model, four retrained models",
               "bar": __doc__.split("Written before the first run:")[1].strip(),
               "settings": {"plain": [0, 0.0], **SETTINGS}, "median plain": med("plain"),
               "median main": med("main"), "median stronger": med("stronger"),
               "single captions pure under main": pure, "verdict": verdict, "results": res},
              open(m.OUT / f"toy-caption-corrector{TAG}.json", "w"), indent=1)
    print("wrote", m.OUT / f"toy-caption-corrector{TAG}.json")


if __name__ == "__main__":
    # Added after the first run's result: the same corrector on models whose both-shapes pictures
    # carry only the joint caption. Reported, not judged against the bar above.
    if "--runs" in sys.argv:
        RUNS = [tuple(int(v) for v in r.split(":")) for r in sys.argv[sys.argv.index("--runs") + 1].split(",")]
        TAG = "-" + sys.argv[sys.argv.index("--tag") + 1]
    main()
