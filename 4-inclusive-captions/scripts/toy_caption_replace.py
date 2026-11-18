"""The toy relabelling experiment with replacing instead of adding.

toy_caption_threshold.py ADDS k both-shapes copies under "a red circle" and under "a blue square",
so the training set grows from 6,000 to 6,000 + 2k pictures and each picture is seen fewer times.
Here each single caption keeps exactly 2,000 pictures: k of its lone-shape copies are REPLACED by
k both-shapes copies (the same k under both captions, as in the adding design). The joint caption's
2,000 are unchanged, so the set is 6,000 at every k and only the share k / 2,000 moves.

Replaced 200 (share 10%) and 1,000 (share 50%), seeds 20261022, 20261023, 20261024. Everything else
is the experiment's own code unchanged: VAE, denoiser, training length, sampler, guidance 3, the
colour-blob scorer, 300 samples per caption.

Written before the run, judged on the share-50% models only (the adding design at 9.1% drew the pair
in 1 of 3 trainings, so 10% cannot tell the designs apart and is reported, not judged):
  share drives it:  all three share-50% models draw at least BAR_APART of 300 separate pairs, as all
                    three adding-design models at 50% did (252, 74, 238)
  size was part of it: all three stay below BAR_APART
  mixed:            anything else
Single-caption purity (own colour only, of 300) is reported beside the verdict and does not decide
it, because one adding-design model at 50% fell to 223 against the usual 240 line.

    CUDA_VISIBLE_DEVICES= /home/molef/micromamba/envs/sdxl-local/bin/python scripts/toy_caption_replace.py

The full sweep at the adding design's shares (written to its own files, judged by nothing above):
    python scripts/toy_caption_replace.py --counts 1,5,20,49,182,462,750,1000 --tag=-sweep \
        --save-weights 1000:20261022
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "3-understanding-compositionality/artifacts/notes/what-poe-loses-when-it-composes"))
import toy_caption_dial_ldm as m

OUT = REPO / "4-inclusive-captions/artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose"
REPLACED = [200, 1000]
SEEDS = [20261022, 20261023, 20261024]
JUDGED = 1000
BAR_APART = 30
ADDING_AT_50 = {"seeds": SEEDS, "separate of 300": [252, 74, 238]}   # toy-caption-threshold-600-1200-2000.json


def corpus_replace(k, rng):
    """2,000 pictures under every caption; k of each single caption's lone shapes are replaced by the
    same k both-shapes pictures, each copy with its own pixel noise."""
    imgs, caps = [], []
    per = m.IMAGES // 3
    pair = m.PAIRS[rng.integers(len(m.PAIRS), size=per)]
    shared = pair[rng.choice(per, size=k, replace=False)]
    for bank, cap in ((m.SINGLES_C, m.CIRCLE), (m.SINGLES_S, m.SQUARE)):
        imgs.append(bank[rng.integers(len(bank), size=per - k)]); caps += [cap] * (per - k)
        imgs.append(shared); caps += [cap] * k
    imgs.append(pair); caps += [m.BOTH] * per
    x = np.concatenate(imgs) + m.t.SIGMA0 * rng.standard_normal((len(caps), m.t.D))
    return (torch.tensor(x.reshape(-1, m.H, m.W, 3).transpose(0, 3, 1, 2), dtype=torch.float32, device=m.DEV),
            torch.tensor(caps, dtype=torch.long, device=m.DEV))


WEIGHTS = REPO / "4-inclusive-captions/data/toy-relabelling-ladder"


def main(tag="", save=None):
    res, sheets = {}, {}
    for k in REPLACED:
        res[str(k)] = []
        for s in SEEDS:
            t0 = time.time()
            torch.manual_seed(s)
            x, c = corpus_replace(k, np.random.default_rng(s))
            per_cap = {int(i): int((c == i).sum()) for i in range(3)}
            vae, rec, tries = m.train_vae(x, s)
            with torch.no_grad():
                z = vae.encode(x)[0].reshape(len(x), -1)
            net, _ = m.train_denoiser(z, c, s)
            if save == (k, s):
                WEIGHTS.mkdir(parents=True, exist_ok=True)
                torch.save(vae.state_dict(), WEIGHTS / f"vae_replace{k}_seed{s}.pth")
                torch.save(net.state_dict(), WEIGHTS / f"denoiser_replace{k}_seed{s}.pth")
            out = {}
            for name, caps in (("product", [m.CIRCLE, m.SQUARE]), ("circle", [m.CIRCLE]),
                               ("square", [m.SQUARE]), ("joint", [m.BOTH])):
                cnt, tiles = m.draw(net, vae, caps, z.shape[1])
                out[name] = cnt["blob scorer"]
                sheets[f"{k}_{s}_{name}"] = tiles
            row = {"seed": s, "training pictures": int(len(c)), "pictures per caption id": per_cap,
                   "share holding both shapes": k / per_cap[m.CIRCLE],
                   "apart": out["product"]["red and blue apart"],
                   "'a red circle' drew only red": out["circle"]["red only"],
                   "'a blue square' drew only blue": out["square"]["blue only"],
                   "vae mse": round(rec, 5), "vae tries": tries, "counts": out,
                   "seconds": round(time.time() - t0, 1)}
            res[str(k)].append(row)
            print(f"replaced {k} seed {s}: {len(c)} pictures, apart {row['apart']}/300, "
                  f"purple {out['product']['purple or overlaid']}, none {out['product']['nothing readable']} | "
                  f"red only {row[chr(39) + 'a red circle' + chr(39) + ' drew only red']}, "
                  f"blue only {row[chr(39) + 'a blue square' + chr(39) + ' drew only blue']}, "
                  f"joint apart {out['joint']['red and blue apart']} | {row['seconds']}s", flush=True)
            json.dump({"partial": True, "results": res}, open(OUT / f"toy-caption-replace{tag}.json", "w"), indent=1)
    judged = [r["apart"] for r in res.get(str(JUDGED), [])]
    verdict = ("not judged: this run is the sweep" if tag else
               "share drives it" if all(a >= BAR_APART for a in judged) else
               "size was part of it" if all(a < BAR_APART for a in judged) else "mixed")
    print("VERDICT:", verdict, judged, flush=True)
    np.savez_compressed(OUT / f"toy-caption-replace{tag}-samples.npz", **sheets)
    json.dump({"what": "each single caption keeps 2,000 pictures; k of its lone shapes replaced by k both-shapes "
                       "pictures; three trainings per k; PoE at guidance 3; colour-blob scorer; 300 samples each",
               "bar": __doc__.split("Written before the run")[1].split("    CUDA_VISIBLE")[0].strip(),
               "bar constants": {"BAR_APART": BAR_APART, "JUDGED": JUDGED},
               "adding design at 50%": ADDING_AT_50,
               "verdict": verdict, "judged separate counts": judged, "results": res},
              open(OUT / f"toy-caption-replace{tag}.json", "w"), indent=1)
    print("wrote", OUT / f"toy-caption-replace{tag}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", help="comma-separated replaced counts; default 200,1000")
    ap.add_argument("--tag", default="", help="suffix for the output files, e.g. -sweep")
    ap.add_argument("--save-weights", help="k:seed of the one model whose VAE and denoiser are saved")
    a = ap.parse_args()
    if a.counts:
        REPLACED = [int(v) for v in a.counts.split(",")]
    save = tuple(int(v) for v in a.save_weights.split(":")) if a.save_weights else None
    print("device", m.DEV, "| counts", REPLACED, "| tag", a.tag or "(none)", "| save", save, flush=True)
    main(a.tag, save)
