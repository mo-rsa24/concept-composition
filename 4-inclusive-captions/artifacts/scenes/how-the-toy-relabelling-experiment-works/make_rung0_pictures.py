"""Every picture on rung 0 (the toy caption model's architecture diagram), drawn from the experiment's own
code and saved samples. Nothing here is typed by hand: pictures come from toy_sampler_hybrids.draw and
toy_caption_dial_ldm.PAIRS, sample strips from the saved npz files, and their categories from
toy_blob_scorer.read. The per-step pictures (latent, noised latents, the denoiser's guess, the three
calls of one PoE step) need a trained model, so one model is retrained here with the experiment's own
functions (0 relabelled, seed 20261022); it is not one of the reported trainings and every picture from
it says so in pictures.json.

    /home/molef/micromamba/envs/sdxl-local/bin/python -B make_rung0_pictures.py

Writes the-toy-caption-model-pictures/*.png and the-toy-caption-model-pictures/manifest.json.
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
NOTES = REPO / "3-understanding-compositionality/artifacts/notes/what-poe-loses-when-it-composes"
SAMPLES = REPO / "3-understanding-compositionality/artifacts/results/the-exact-toy/how-much-co-occurrence-the-product-needs"
OUT = HERE / "the-toy-caption-model-pictures"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(NOTES))
sys.dont_write_bytecode = True

import torch
import toy_sampler_hybrids as t
import toy_blob_scorer as b
import toy_caption_dial_ldm as m
import toy_caption_threshold as th

NUM = json.load(open(HERE / "toy_relabelling_numbers.json"))
SCALE = 4
manifest = {}


def save_rgb(flat, name, what):
    img = np.clip(np.asarray(flat, float).reshape(t.H, t.W, 3), 0, 1)
    im = Image.fromarray((img * 255).round().astype(np.uint8)).resize((t.W * SCALE, t.H * SCALE), Image.NEAREST)
    im.save(OUT / f"{name}.png")
    manifest[name] = what
    return name


def save_grey(arr, name, what, scale=8, lim=None):
    """A grid of numbers as grey squares: black low, white high. Never red or blue, so a heat map is
    never mistaken for one of the toy's shapes."""
    a = np.asarray(arr, float)
    lim = lim or float(np.abs(a).max()) or 1.0
    g = ((a / lim).clip(-1, 1) + 1) / 2
    im = Image.fromarray((g * 255).round().astype(np.uint8)).resize((a.shape[1] * scale, a.shape[0] * scale), Image.NEAREST)
    im.save(OUT / f"{name}.png")
    manifest[name] = what
    return name


def latent_maps(z, name, what, lim=None):
    """A 128-number latent as its 4 channels of 4 x 8, in a 2 x 2 block with a one-cell gap, grey scale."""
    z = np.asarray(z, float).reshape(4, 4, 8)
    lim = lim or float(np.abs(z).max())
    row = np.full((9, 17), np.nan)
    for ch in range(4):
        r, c = divmod(ch, 2)
        row[r * 5:r * 5 + 4, c * 9:c * 9 + 8] = z[ch]
    g = ((np.nan_to_num(row) / lim).clip(-1, 1) + 1) / 2
    rgb = np.stack([g] * 3, -1)
    rgb[np.isnan(row)] = 1.0
    im = Image.fromarray((rgb * 255).round().astype(np.uint8)).resize((row.shape[1] * 8, row.shape[0] * 8), Image.NEAREST)
    im.save(OUT / f"{name}.png")
    manifest[name] = what
    return name


# 1. Training pictures, drawn by the experiment's own draw()
for p in ("left", "centre", "right"):
    save_rgb(t.CIRCLES[p], f"clean-circle-{p}", f"training picture under id 0 'a red circle', circle centred at x = {t.XPOS[p]:g}")
    save_rgb(t.SQUARES[p], f"clean-square-{p}", f"training picture under id 1 'a blue square', square centred at x = {t.XPOS[p]:g}")
save_rgb(m.PAIRS[0], "clean-both-circle-left", "both-shapes layout 1: circle left, square right (id 2)")
save_rgb(m.PAIRS[1], "clean-both-square-left", "both-shapes layout 2: square left, circle right (id 2)")

# 2. The training set as corpus_k builds it, at k = 20, with its pixel noise
rng = np.random.default_rng(0)
x20, c20 = th.corpus_k(20, rng)
x20 = x20.cpu().permute(0, 2, 3, 1).reshape(len(c20), -1).numpy()
c20 = c20.cpu().numpy()
per = m.IMAGES // 3
picks = {"circle": 0, "square": per, "both": 2 * per, "relabelled-as-circle": 3 * per, "relabelled-as-square": 3 * per + 20}
ids = {}
for k, i in picks.items():
    save_rgb(x20[i], f"noisy-train-{k}", f"training copy number {i} of corpus_k(20, seed 0), caption id {int(c20[i])}, pixel noise sd {t.SIGMA0}")
    ids[k] = int(c20[i])
manifest["_ids"] = ids
# corpus_k draws from one generator in a fixed order: circle picks, square picks, pair picks, then the k
# relabelled indices. Replaying it finds the id-2 copy that is the same drawn picture as relabelled copy 0.
r = np.random.default_rng(0)
r.integers(3, size=per); r.integers(3, size=per); r.integers(2, size=per)
sel = r.choice(per, size=20, replace=False)
j = 2 * per + int(sel[0])
assert int(c20[j]) == m.BOTH and np.allclose(x20[j] - x20[3 * per], x20[j] - x20[3 * per])
same_clean = np.abs((x20[j] - x20[3 * per])).mean() < 0.2
save_rgb(x20[j], "noisy-train-relabelled-source", f"training copy number {j} of corpus_k(20, seed 0), caption id {int(c20[j])}: the same drawn picture as copies {3 * per} (id 0) and {3 * per + 20} (id 1), with its own pixel noise")
manifest["_relabel_same_picture"] = {"id 2 copy": j, "id 0 copy": 3 * per, "id 1 copy": 3 * per + 20, "same layout": bool(same_clean)}

# 3. Saved samples from the reported runs, each labelled by the scorer
def strip(key, src, prefix, n=8):
    tiles = np.load(src)[key][:n]
    cats, info = b.read(tiles)
    names = []
    for i, (tile, cat) in enumerate(zip(tiles, cats)):
        names.append(save_rgb(tile, f"{prefix}-{i}", f"saved sample {i} of '{key}' in {src.name}; scorer: {cat}"))
    manifest[f"_{prefix}"] = {"key": key, "file": src.name, "categories": cats, "info": info}
    return names, cats

TH = SAMPLES / "toy-caption-threshold-samples.npz"
DC = SAMPLES / "toy-caption-dial-corrector-samples.npz"
strip("0_20261023_product", TH, "product-k0-s20261023")
strip("0_20261022_product", TH, "product-k0-s20261022")
strip("0_20261023_circle", TH, "circle-k0-s20261023")
strip("0_20261023_square", TH, "square-k0-s20261023")
strip("1_poe", DC, "share1-poe")
strip("1_poe_corrector20", DC, "share1-poe-corrector20")
strip("0_poe_corrector20", DC, "share0-poe-corrector20")
strip("0_poe", DC, "share0-poe")
for kk in (1, 5, 20, 50, 200):
    strip(f"{kk}_20261023_product", TH, f"product-k{kk}-s20261023", n=1)

# 4. What the scorer sees: each pixel's nearest palette colour (grey where none is within MAX_DIST)
def scorer_map(flat, name, what):
    img = np.asarray(flat, float).reshape(t.H, t.W, 3)
    d = ((img[..., None, :] - b.PALETTE) ** 2).sum(-1) ** 0.5
    near = d.argmin(-1)
    near[d.min(-1) > b.MAX_DIST] = -1
    out = np.where(near[..., None] >= 0, b.PALETTE[np.clip(near, 0, 3)], np.array([0.55, 0.55, 0.55]))
    return save_rgb(out.reshape(-1), name, what)

for prefix, i in (("share1-poe-corrector20", 0), ("product-k0-s20261023", 0), ("product-k0-s20261022", 0)):
    tiles = np.load(DC if prefix.startswith("share") else TH)[manifest[f"_{prefix}"]["key"]]
    scorer_map(tiles[i], f"scorer-map-{prefix}-{i}", f"pixel colours the scorer assigns to {prefix}-{i}: white, red, blue, purple, grey for none within {b.MAX_DIST}")

# 5. One retrained model for the per-step pictures (0 relabelled, seed 20261022)
SEED = 20261022
torch.manual_seed(SEED)
x, c = th.corpus_k(0, np.random.default_rng(SEED))
vae, rec, tries = m.train_vae(x, SEED)
with torch.no_grad():
    z = vae.encode(x)[0].reshape(len(x), -1)
net, dloss = m.train_denoiser(z, c, SEED)
manifest["_retrain"] = {"relabelled": 0, "seed": SEED, "vae mse": round(rec, 5), "vae tries": tries, "device": m.DEV}
dev = m.DEV
dec = lambda zz: vae.dec(zz.reshape(-1, 4, 4, 8)).permute(0, 2, 3, 1).reshape(len(zz), -1).cpu().numpy()

with torch.no_grad():
    i_both = 2 * per                                    # the first both-shapes training copy
    z0 = z[i_both:i_both + 1]
    lim = float(z0.abs().max())
    save_rgb(x[i_both].permute(1, 2, 0).reshape(-1).cpu().numpy(), "vae-in", "the training copy fed to the VAE (retrained model)")
    latent_maps(z0.cpu().numpy(), "vae-latent", "its latent: 128 numbers as 4 channels of 4 x 8 in a 2 x 2 block, grey scale (retrained model)", lim)
    save_rgb(dec(z0)[0], "vae-out", "the VAE decoder's reconstruction of that latent (retrained model)")
    gen = torch.Generator(device=dev).manual_seed(5)
    eps = torch.randn(z0.shape, device=dev, generator=gen)
    for k in (0, 50, 100, 150, 200):
        a = m.abar[k]
        zt = a.sqrt() * z0 + (1 - a).sqrt() * eps
        save_rgb(dec(zt)[0], f"noised-k{k:03d}", f"the clean latent noised to level {k}, decoded so it can be seen (retrained model)")
        if k == 100:
            latent_maps(zt.cpu().numpy(), "noised-latent-k100", "the noisy latent at level 100 as 4 channels of 4 x 8 in a 2 x 2 block (retrained model)", lim)
            kk = torch.full((1,), k, device=dev, dtype=torch.long)
            for cname, cid in (("empty", m.NULL), ("circle", m.CIRCLE), ("square", m.SQUARE), ("both", m.BOTH)):
                g = net(zt, kk, torch.full((1,), cid, device=dev, dtype=torch.long))
                save_rgb(dec(g)[0], f"guess-k100-{cname}", f"denoiser's clean-latent guess at level 100 given caption id {cid}, decoded (retrained model)")
                if cname == "both":
                    latent_maps(g.cpu().numpy(), "guess-latent-k100-both", "that guess as 4 channels of 4 x 8 in a 2 x 2 block (retrained model)", lim)
            latent_maps(z0.cpu().numpy(), "clean-latent-target", "the clean latent the loss compares the guess with (retrained model)", lim)
            ang = kk.float()[:, None] / m.T * net.freq[None]
            feats = torch.cat([torch.sin(ang), torch.cos(ang)], 1).cpu().numpy()
            save_grey(feats, "time-features-k100", "the 64 time features of level 100: 32 sines then 32 cosines", scale=6, lim=1.0)
    table = net.cap.weight.detach().cpu().numpy()
    save_grey(table, "caption-table", "the trained 4 x 64 caption table: row 0 'a red circle', 1 'a blue square', 2 both, 3 empty", scale=6)

    # One PoE run, 8 starts, recording the three calls and the combination at level 100
    n = 8
    gen = torch.Generator(device=dev).manual_seed(11)
    zz = torch.randn(n, 128, device=dev, generator=gen)
    latent_maps(zz[0:1].cpu().numpy(), "start-latent", "a random start: 128 numbers drawn from a standard normal, as 4 channels of 4 x 8 in a 2 x 2 block", 3.0)
    save_rgb(dec(zz[0:1])[0], "start-decoded", "that random start passed through the decoder")
    nullc = torch.full((n,), m.NULL, device=dev, dtype=torch.long)
    frames = {}
    for k in range(m.T, 0, -1):
        a_t, a_p = m.abar[k], m.abar[k - 1]
        al = a_t / a_p; be = 1 - al
        kk = torch.full((n,), k, device=dev, dtype=torch.long)
        g_null = net(zz, kk, nullc)
        e_null = (zz - a_t.sqrt() * g_null) / (1 - a_t).sqrt()
        e = e_null.clone()
        gs = {"empty": g_null}
        for cname, cid in (("circle", m.CIRCLE), ("square", m.SQUARE)):
            g = net(zz, kk, torch.full((n,), cid, device=dev, dtype=torch.long))
            gs[cname] = g
            e = e + m.GUIDE * ((zz - a_t.sqrt() * g) / (1 - a_t).sqrt() - e_null)
        comb = (zz - (1 - a_t).sqrt() * e) / a_t.sqrt()
        if k in (200, 150, 100, 50, 1):
            frames[k] = dec(comb)
        if k == 100:
            for cname, g in gs.items():
                save_rgb(dec(g)[0], f"poe-call-k100-{cname}", f"PoE run start 0, level 100: the {cname} call's clean-latent guess, decoded (retrained model)")
            save_rgb(dec(comb)[0], "poe-combined-k100", "PoE run start 0, level 100: the clean latent implied by the combined estimate e, decoded (retrained model)")
        mean = (zz - be / (1 - a_t).sqrt() * e) / al.sqrt()
        zz = mean + ((be * (1 - a_p) / (1 - a_t)).sqrt() if k > 1 else 0.0) * torch.randn(zz.shape, device=dev, generator=gen)
    for k, fr in frames.items():
        save_rgb(fr[0], f"poe-run-k{k:03d}", f"PoE run start 0: the combined clean-latent guess at level {k}, decoded (retrained model)")
    final = dec(zz)
    cats, _ = b.read(np.clip(final, 0, 1))
    save_rgb(final[0], "poe-run-final", f"PoE run start 0 after the last step, decoded (retrained model); scorer: {cats[0]}")
    manifest["_retrain"]["final categories of the 8 starts"] = cats

json.dump(manifest, open(OUT / "manifest.json", "w"), indent=1)
print("wrote", len([k for k in manifest if not k.startswith("_")]), "pictures to", OUT)
print(json.dumps({k: v for k, v in manifest.items() if k.startswith("_") and k != "_retrain"}, default=str)[:1500])
print(manifest["_retrain"])
