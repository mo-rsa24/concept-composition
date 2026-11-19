"""The toy's corrector result in the replacing design, at evenly spread counts.

Each single caption keeps exactly 2,000 pictures; k of them show both shapes (corpus_replace in
toy_caption_replace.py). One model per (k, training seed). Each model draws 300 pictures of the product
"a red circle" x "a blue square" three ways from the same starting noise: plain, with 5 corrector steps
and with 20 corrector steps per noise level (toy_caption_corrector.draw, step 0.05 of the noise variance,
noise seed = training seed + 7 as in that script). "a red circle" and "a blue square" alone are drawn
plain, to show they still mean one shape.

Written before the run: with relabelled pictures (k >= 100), 20 corrector steps give at least 290 of 300
separate pairs in all three trainings; with none (k = 0) the corrector gives at most 3 of 300.

    python 4-inclusive-captions/scripts/toy_replace_corrector.py --counts 0,100 --tag -a
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "3-understanding-compositionality/artifacts/notes/what-poe-loses-when-it-composes"))
sys.path.insert(0, str(REPO / "4-inclusive-captions/scripts"))
import toy_caption_dial_ldm as m
import toy_caption_corrector as cr
from toy_caption_replace import corpus_replace

OUT = REPO / "4-inclusive-captions/artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose"
SEEDS = [20261022, 20261023, 20261024]
SETTINGS = {"plain": (0, 0.0), "5 corrector steps": (5, 0.05), "20 corrector steps": (20, 0.05)}
APART = "red and blue apart"


def main(counts, tag):
    res, tiles = {}, {}
    for k in counts:
        res[str(k)] = []
        for s in SEEDS:
            t0 = time.time()
            torch.manual_seed(s)
            x, c = corpus_replace(k, np.random.default_rng(s))
            vae, rec, tries = m.train_vae(x, s)
            with torch.no_grad():
                z = vae.encode(x)[0].reshape(len(x), -1)
            net, _ = m.train_denoiser(z, c, s)
            row = {"seed": s, "replaced": k, "of": m.IMAGES // 3, "vae mse": round(rec, 5)}
            for name, (st, sz) in SETTINGS.items():
                cnt, t = cr.draw(net, vae, [m.CIRCLE, m.SQUARE], z.shape[1], st, sz, seed=s + 7)
                row[name] = cnt
                tiles[f"{k}_{s}_{name}"] = t
            for cap, cid in (("a red circle alone", m.CIRCLE), ("a blue square alone", m.SQUARE)):
                cnt, t = cr.draw(net, vae, [cid], z.shape[1], 0, 0.0, seed=s + 7)
                row[cap] = cnt
                tiles[f"{k}_{s}_{cap}"] = t
            row["seconds"] = round(time.time() - t0, 1)
            res[str(k)].append(row)
            print(f"replaced {k} seed {s}: apart plain {row['plain'][APART]}, 5 steps {row['5 corrector steps'][APART]}, "
                  f"20 steps {row['20 corrector steps'][APART]} of 300 | circle alone red only "
                  f"{row['a red circle alone']['red only']}, square alone blue only {row['a blue square alone']['blue only']} "
                  f"| {row['seconds']}s", flush=True)
            json.dump({"partial": True, "results": res}, open(OUT / f"toy-replace-corrector{tag}.json", "w"), indent=1)
    np.savez_compressed(OUT / f"toy-replace-corrector{tag}-samples.npz", **tiles)
    json.dump({"what": "replacing design, k of 2,000 pictures under each single caption show both shapes; 300 product "
                       "pictures per model drawn plain, with 5 and with 20 corrector steps; colour-blob scorer",
               "bar": __doc__.split("Written before the run: ")[1].split("\n\n")[0].replace("\n", " "),
               "results": res}, open(OUT / f"toy-replace-corrector{tag}.json", "w"), indent=1)
    print("wrote", OUT / f"toy-replace-corrector{tag}.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    main([int(v) for v in a.counts.split(",")], a.tag)
