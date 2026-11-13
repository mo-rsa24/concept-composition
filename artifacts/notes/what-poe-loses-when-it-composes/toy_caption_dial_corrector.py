"""Figure 07b's data: the five share-of-pictures models of toy_caption_dial_ldm.py (0%, 10%, 30%,
60%, 100% of both-shapes pictures relabelled at random, seed 20261022), retrained, each drawn five
ways from the same retrained model: "a red circle", "a blue square", plain PoE, PoE with 20
Langevin corrector steps per noise level (step 0.05 of the noise variance, as in
toy_caption_corrector.py), and the joint caption. 300 samples each, colour-blob scorer.
The models are retrained, so counts differ from toy-caption-dial-ldm.json; every column of a row
comes from one model."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
import toy_caption_dial_ldm as m
import toy_caption_corrector as cr

res, keep = {}, {}
for dial in m.DIAL:
    t0 = time.time()
    torch.manual_seed(m.SEED)
    x, c = m.corpus(dial, np.random.default_rng(m.SEED))
    vae, rec, _ = m.train_vae(x, m.SEED)
    with torch.no_grad():
        z = vae.encode(x)[0].reshape(len(x), -1)
    net, _ = m.train_denoiser(z, c, m.SEED)
    row = {"vae mse": round(rec, 5)}
    for name, caps, steps in (("circle", [m.CIRCLE], 0), ("square", [m.SQUARE], 0),
                              ("poe", [m.CIRCLE, m.SQUARE], 0), ("poe_corrector20", [m.CIRCLE, m.SQUARE], 20),
                              ("joint", [m.BOTH], 0)):
        cnt, tiles = cr.draw(net, vae, caps, z.shape[1], steps, 0.05, seed=m.SEED + 7)
        row[name] = cnt; keep[f"{dial:g}_{name}"] = tiles
    res[f"{dial:g}"] = row
    print(f"share {dial:g} ({time.time() - t0:.0f}s): circle red {row['circle']['red only']}, square blue "
          f"{row['square']['blue only']}, PoE apart {row['poe']['red and blue apart']}, PoE+corrector apart "
          f"{row['poe_corrector20']['red and blue apart']}, joint apart {row['joint']['red and blue apart']}", flush=True)
np.savez_compressed(m.OUT / "toy-caption-dial-corrector-samples.npz", **keep)
json.dump({"what": __doc__, "results": res}, open(m.OUT / "toy-caption-dial-corrector.json", "w"), indent=1)
print("wrote", m.OUT / "toy-caption-dial-corrector.json")
