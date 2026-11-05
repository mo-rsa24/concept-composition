"""Why a latent-space PoE comes out purple in pixel space: decode the latent halfway between a red
circle and a blue square (same position, centre) and compare it with the pixel-space halfway
picture. One VAE trained as in toy_caption_dial_ldm.py (0 relabelled, seed 20261022)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
import toy_caption_dial_ldm as m
import toy_caption_threshold as th
t = m.t
S = m.SEED
torch.manual_seed(S)
x, _ = th.corpus_k(0, np.random.default_rng(S))
vae, rec, _ = m.train_vae(x, S)
toT = lambda im: torch.tensor(im.reshape(1, t.H, t.W, 3).transpose(0, 3, 1, 2), dtype=torch.float32, device=m.DEV)
toN = lambda tt: np.clip(tt.permute(0, 2, 3, 1).reshape(-1).cpu().numpy(), 0, 1)
out, pics = {"vae_mse": round(rec, 5)}, {}
for name, a, b in (("same position", t.CIRCLES["centre"], t.SQUARES["centre"]),
                   ("different positions", t.CIRCLES["left"], t.SQUARES["right"])):
    with torch.no_grad():
        za, zb = vae.encode(toT(a))[0], vae.encode(toT(b))[0]
        mid = vae.dec((za + zb) / 2)
    pm = (a + b) / 2
    key = name.replace(" ", "_")
    pics[f"{key}_circle"], pics[f"{key}_square"] = a, b
    pics[f"{key}_pixel_mid"], pics[f"{key}_latent_mid"] = pm, toN(mid)
    # colour at the centre pixel of the shape(s)
    def px(im, cx): return [round(float(v), 3) for v in im.reshape(t.H, t.W, 3)[8, int(cx)]]
    out[name] = {"circle centre pixel": px(a, t.XPOS["centre"] if name == "same position" else t.XPOS["left"]),
                 "pixel-space halfway, same pixel": px(pm, t.XPOS["centre"] if name == "same position" else t.XPOS["left"]),
                 "latent halfway decoded, same pixel": px(toN(mid), t.XPOS["centre"] if name == "same position" else t.XPOS["left"]),
                 "mean abs gap between the two halfway pictures": round(float(np.abs(toN(mid) - pm).mean()), 4)}
np.savez_compressed(m.OUT / "toy-latent-midpoint.npz", **pics)
json.dump(out, open(m.OUT / "toy-latent-midpoint.json", "w"), indent=1)
print(json.dumps(out, indent=1))
