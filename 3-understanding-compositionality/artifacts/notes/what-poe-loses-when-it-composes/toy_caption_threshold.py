"""How many both-shapes training pictures must also carry each single caption before PoE of
"a red circle" and "a blue square" draws the two shapes apart?

Same VAE, denoiser, sampler and colour scorer as toy_caption_dial_ldm.py. The only change: instead
of relabelling a random share of the 2,000 both-shapes pictures, exactly `k` of them are added a
second time under "a red circle" and a third time under "a blue square". k = 0, 1, 5, 20, 50, 200.
Each k is trained three times with different seeds (data draw, VAE, denoiser and sampling noise).
300 PoE samples per training, guidance 3.

Written before the first run:
  - A training "draws the pair" when at least 30 of 300 PoE samples are a separate red circle and
    blue square, and the single captions alone still give at least 240 of 300 of their own colour.
  - The threshold is the smallest k at which all three trainings draw the pair.
  - "One relabelled picture is enough" is supported if k = 1 meets that in all three trainings,
    and contradicted if all three give under 30 of 300.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
import toy_caption_dial_ldm as m

COUNTS = [0, 1, 5, 20, 50, 200]
TAG = ""
SEEDS = [20261022, 20261023, 20261024]


def corpus_k(k, rng):
    imgs, caps = [], []
    per = m.IMAGES // 3
    for bank, cap in ((m.SINGLES_C, m.CIRCLE), (m.SINGLES_S, m.SQUARE)):
        imgs.append(bank[rng.integers(len(bank), size=per)]); caps += [cap] * per
    pair = m.PAIRS[rng.integers(len(m.PAIRS), size=per)]
    imgs.append(pair); caps += [m.BOTH] * per
    if k:
        extra = pair[rng.choice(per, size=k, replace=False)]
        imgs.append(extra); caps += [m.CIRCLE] * k
        imgs.append(extra); caps += [m.SQUARE] * k
    x = np.concatenate(imgs) + m.t.SIGMA0 * rng.standard_normal((len(caps), m.t.D))
    return (torch.tensor(x.reshape(-1, m.H, m.W, 3).transpose(0, 3, 1, 2), dtype=torch.float32, device=m.DEV),
            torch.tensor(caps, dtype=torch.long, device=m.DEV))


def main():
    res, sheets = {}, {}
    for k in COUNTS:
        res[str(k)] = []
        for s in SEEDS:
            t0 = time.time()
            torch.manual_seed(s)
            x, c = corpus_k(k, np.random.default_rng(s))
            vae, rec, tries = m.train_vae(x, s)
            with torch.no_grad():
                z = vae.encode(x)[0].reshape(len(x), -1)
            net, _ = m.train_denoiser(z, c, s)
            out = {}
            for name, caps in (("product", [m.CIRCLE, m.SQUARE]), ("circle", [m.CIRCLE]),
                               ("square", [m.SQUARE]), ("joint", [m.BOTH])):
                cnt, tiles = m.draw(net, vae, caps, z.shape[1])
                out[name] = cnt["blob scorer"]
                sheets[f"{k}_{s}_{name}"] = tiles
            apart = out["product"]["red and blue apart"]
            pure = out["circle"]["red only"] >= m.PURE and out["square"]["blue only"] >= m.PURE
            row = {"seed": s, "apart": apart, "single captions pure": pure,
                   "draws the pair": bool(apart >= 30 and pure), "vae mse": round(rec, 5),
                   "counts": out, "seconds": round(time.time() - t0, 1)}
            res[str(k)].append(row)
            print(f"k {k} seed {s}: PoE apart {apart}/300, purple {out['product']['purple or overlaid']}, "
                  f"none {out['product']['nothing readable']} | circle red {out['circle']['red only']}, "
                  f"square blue {out['square']['blue only']}, joint apart {out['joint']['red and blue apart']} "
                  f"| vae {rec:.4f} | {row['seconds']}s", flush=True)
    meets = [k for k in COUNTS if all(r["draws the pair"] for r in res[str(k)])]
    thr = meets[0] if meets else None
    one = res.get("1", [])
    one_verdict = ("not run" if not one else "supported" if all(r["draws the pair"] for r in one) else
                   "contradicted" if all(r["apart"] < 30 for r in one) else "mixed")
    print("THRESHOLD:", thr, "| one relabelled picture is enough:", one_verdict, flush=True)
    np.savez_compressed(m.OUT / f"toy-caption-threshold{TAG}-samples.npz", **sheets)
    json.dump({"what": "exactly k both-shapes pictures also captioned with each single shape; three "
                       "trainings per k; PoE at guidance 3; colour-blob scorer; 300 samples each",
               "bar": __doc__.split("Written before the first run:")[1].strip(),
               "threshold": thr, "one picture is enough": one_verdict, "results": res},
              open(m.OUT / f"toy-caption-threshold{TAG}.json", "w"), indent=1)
    print("wrote", m.OUT / f"toy-caption-threshold{TAG}.json")


if __name__ == "__main__":
    if "--counts" in sys.argv:          # extend the sweep; written to its own file
        COUNTS = [int(v) for v in sys.argv[sys.argv.index("--counts") + 1].split(",")]
        TAG = "-" + "-".join(map(str, COUNTS))
    main()
