"""Everything the "how the caption model works" page shows, computed once and written to scene_data.json
and page-images/.

Three sources, kept apart because they carry different weight:
  trace     one hooked forward pass of a freshly built VAE and Denoiser (batch 1): every layer's real
            input and output shape and parameter count. Weights do not matter for shapes.
  rebuild   one model retrained here on CPU with the repository's own code (toy_caption_dial_ldm.py,
            toy_caption_threshold.corpus_k), 2,000 relabelled pictures, seed 20261022. Used only to
            show a real latent, real denoiser inputs and outputs, and a real 200-step trajectory.
            It is not the saved run: CPU and GPU arithmetic differ, so its samples differ.
  saved     the samples the experiments saved (npz files in the results folder), shown untouched.

    CUDA_VISIBLE_DEVICES= /home/molef/micromamba/envs/sdxl-local/bin/python -I build_scene_data.py
"""
import json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CODE = ROOT / "artifacts" / "notes" / "what-poe-loses-when-it-composes"
RES = ROOT / "artifacts" / "results" / "how-much-co-occurrence-the-product-needs"
IMG = HERE / "page-images"
sys.path.insert(0, str(CODE))
import torch
import toy_caption_dial_ldm as m
import toy_caption_threshold as th

assert m.DEV == "cpu", "run with CUDA_VISIBLE_DEVICES= so the GPU is left alone"
H, W = m.H, m.W
out = {"made": time.strftime("%Y-%m-%d")}


def png(flat, path, scale=1):
    a = np.clip(np.asarray(flat, dtype=np.float32).reshape(H, W, 3), 0, 1)
    im = Image.fromarray((a * 255).round().astype(np.uint8))
    if scale > 1:
        im = im.resize((W * scale, H * scale), Image.NEAREST)
    im.save(path)
    return path.name


def strip(tiles, path, gap=2):
    """Pictures side by side on white, one pixel per pixel; the page scales them up."""
    n = len(tiles)
    canvas = np.ones((H, n * W + (n - 1) * gap, 3), dtype=np.float32)
    for i, f in enumerate(tiles):
        canvas[:, i * (W + gap):i * (W + gap) + W] = np.clip(np.asarray(f).reshape(H, W, 3), 0, 1)
    Image.fromarray((canvas * 255).round().astype(np.uint8)).save(path)
    return path.name


# ---------- trace ----------
def trace(model, args, names):
    rows, hooks = [], []
    for name, mod in model.named_modules():
        if name and not list(mod.children()):
            def hk(mod, inp, outp, name=name):
                rows.append({"name": name, "type": type(mod).__name__,
                             "in": list(inp[0].shape[1:]) if inp else None, "out": list(outp.shape[1:]),
                             "params": sum(p.numel() for p in mod.parameters())})
            hooks.append(mod.register_forward_hook(hk))
    with torch.no_grad():
        names(model, *args)
    for h in hooks:
        h.remove()
    return rows

torch.manual_seed(0)
vae0, den0 = m.VAE(), m.Denoiser()
x1 = torch.rand(1, 3, H, W)
out["trace"] = {
    "vae": trace(vae0, (x1,), lambda v, x: v.dec(v.encode(x)[0])),
    "denoiser": trace(den0, (torch.randn(1, 128), torch.tensor([100]), torch.tensor([2])),
                      lambda d, z, k, c: d(z, k, c)),
    "vae_params": sum(p.numel() for p in vae0.parameters()),
    "vae_enc_params": sum(p.numel() for p in vae0.enc.parameters()),
    "vae_dec_params": sum(p.numel() for p in vae0.dec.parameters()),
    "denoiser_params": sum(p.numel() for p in den0.parameters()),
    "caption_table_params": den0.cap.weight.numel(),
}
print("trace", out["trace"]["vae_params"], out["trace"]["denoiser_params"], flush=True)

# ---------- the training pictures (exact drawing code, no training needed) ----------
rng = np.random.default_rng(20261022)
t = m.t
clean = {"circle": t.CIRCLES["centre"], "square": t.SQUARES["left"], "both": m.PAIRS[0]}
noisy = {k: v + t.SIGMA0 * rng.standard_normal(v.size) for k, v in clean.items()}
out["train_pictures"] = {k: png(noisy[k], IMG / f"training-picture-{k}.png") for k in noisy}
out["train_pictures_clean"] = {k: png(clean[k], IMG / f"clean-picture-{k}.png") for k in clean}

# ---------- rebuild one model ----------
K, SEED = 2000, 20261022
t0 = time.time()
torch.manual_seed(SEED)
x, c = th.corpus_k(K, np.random.default_rng(SEED))
out["corpus_counts"] = {str(i): int((c == i).sum()) for i in range(4)}
vae, rec, tries = m.train_vae(x, SEED)
print(f"vae mse {rec:.4f} after {tries} tries, {time.time()-t0:.0f}s", flush=True)
with torch.no_grad():
    z_all = vae.encode(x)[0].reshape(len(x), -1)
net, dloss = m.train_denoiser(z_all, c, SEED)
print(f"denoiser loss {dloss:.4f}, {time.time()-t0:.0f}s", flush=True)
out["rebuild"] = {"relabelled": K, "seed": SEED, "vae_mse": round(rec, 5), "vae_tries": tries,
                  "denoiser_final_loss": round(dloss, 5), "seconds_cpu": round(time.time() - t0)}
abar = m.abar

with torch.no_grad():
    xb = torch.tensor(noisy["both"].reshape(1, H, W, 3).transpose(0, 3, 1, 2), dtype=torch.float32)
    mu, lv = vae.encode(xb)
    z0 = mu.reshape(1, -1)
    recon = vae.dec(mu).permute(0, 2, 3, 1).reshape(-1).numpy()
    out["vae_example"] = {
        "input": out["train_pictures"]["both"], "recon": png(recon, IMG / "vae-reconstruction-both.png"),
        "latent": [round(float(v), 3) for v in z0[0]],
        "logvar_mean": round(float(lv.mean()), 3),
        "recon_mse": round(float(((recon - noisy["both"]) ** 2).mean()), 5)}
    # what noise levels look like: mix the latent with noise, decode
    g = torch.Generator().manual_seed(7)
    nz = torch.randn(1, 128, generator=g)
    lev = []
    for k in [0, 40, 80, 120, 160, 200]:
        a = abar[k]
        zt = a.sqrt() * z0 + (1 - a).sqrt() * nz if k else z0
        im = vae.dec(zt.reshape(1, 4, 4, 8)).permute(0, 2, 3, 1).reshape(-1).numpy()
        lev.append({"k": k, "abar": round(float(a), 4), "img": png(im, IMG / f"noised-latent-k{k:03d}.png")})
    out["noise_levels"] = lev
    # one denoiser call, all inputs and the output, real numbers
    k = 100
    a = abar[k]
    zt = a.sqrt() * z0 + (1 - a).sqrt() * nz
    kk = torch.tensor([k]); cc = torch.tensor([m.BOTH])
    feats = kk.float()[:, None] / m.T * net.freq[None]
    z0hat = net(zt, kk, cc)
    out["denoiser_call"] = {
        "k": k, "abar": round(float(a), 4),
        "noisy": [round(float(v), 3) for v in zt[0]],
        "time": [round(float(v), 3) for v in torch.cat([torch.sin(feats), torch.cos(feats)], 1)[0]],
        "caption": [round(float(v), 3) for v in net.cap(cc)[0]],
        "out": [round(float(v), 3) for v in z0hat[0]],
        "target": [round(float(v), 3) for v in z0[0]],
        "err": round(float(((z0hat - z0) ** 2).mean()), 4),
        "out_img": png(vae.dec(z0hat.reshape(1, 4, 4, 8)).permute(0, 2, 3, 1).reshape(-1).numpy(),
                       IMG / "denoiser-guess-k100.png"),
        "noisy_img": png(vae.dec(zt.reshape(1, 4, 4, 8)).permute(0, 2, 3, 1).reshape(-1).numpy(),
                         IMG / "denoiser-input-k100.png")}
    out["caption_table"] = [[round(float(v), 3) for v in row] for row in net.cap.weight]
    out["schedule_abar"] = [round(float(v), 5) for v in abar]

# ---------- trajectories: same starting noise, one caption against PoE ----------
SHOW = [200, 150, 100, 70, 40, 20, 5]


@torch.no_grad()
def trajectory(caps, n=8, seed=11):
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(n, 128, generator=gen)
    nullc = torch.full((n,), m.NULL, dtype=torch.long)
    frames = {}
    for k in range(m.T, 0, -1):
        a_t, a_p = abar[k], abar[k - 1]
        al = a_t / a_p; be = 1 - al
        kk = torch.full((n,), k, dtype=torch.long)
        e_null = (z - a_t.sqrt() * net(z, kk, nullc)) / (1 - a_t).sqrt()
        e = e_null.clone()
        for cid in caps:
            ccc = torch.full((n,), cid, dtype=torch.long)
            e = e + m.GUIDE * ((z - a_t.sqrt() * net(z, kk, ccc)) / (1 - a_t).sqrt() - e_null)
        if k in SHOW:
            guess = (z - (1 - a_t).sqrt() * e) / a_t.sqrt()
            frames[k] = vae.dec(guess.reshape(n, 4, 4, 8)).permute(0, 2, 3, 1).reshape(n, -1).numpy()
        mean = (z - be / (1 - a_t).sqrt() * e) / al.sqrt()
        z = mean + ((be * (1 - a_p) / (1 - a_t)).sqrt() if k > 1 else 0.0) * torch.randn(z.shape, generator=gen)
    frames[0] = vae.dec(z.reshape(n, 4, 4, 8)).permute(0, 2, 3, 1).reshape(n, -1).numpy()
    return frames

traj = {}
for name, caps in (("circle", [m.CIRCLE]), ("poe", [m.CIRCLE, m.SQUARE])):
    fr = trajectory(caps)
    traj[name] = {}
    for k, imgs in fr.items():
        traj[name][str(k)] = [png(imgs[i], IMG / f"trajectory-{name}-s{i}-k{k:03d}.png") for i in range(len(imgs))]
out["trajectory"] = {"steps": SHOW + [0], "samples": 8, "seed": 11, "frames": traj}

# ---------- saved samples, untouched ----------
saved = {}
for f in ["toy-caption-threshold-samples.npz", "toy-caption-threshold-600-1200-2000-samples.npz"]:
    d = np.load(RES / f)
    for key in d.files:
        if key.endswith("_product") or key.endswith("_joint") or key.endswith("_circle") or key.endswith("_square"):
            saved[key] = strip(d[key], IMG / f"saved-threshold-{key.replace('_', '-')}.png")
for f, tag in [("toy-caption-corrector-samples.npz", "corrector"),
               ("toy-caption-corrector-no-single-captions-samples.npz", "corrector-no-single")]:
    d = np.load(RES / f)
    for key in d.files:
        saved["corr_" + key] = strip(d[key], IMG / f"saved-{tag}-{key.replace('_', '-')}.png")
out["saved"] = saved

counts = {}
for f in ["toy-caption-threshold.json", "toy-caption-threshold-600-1200-2000.json"]:
    for k, rows in json.load(open(RES / f))["results"].items():
        counts[k] = [{"seed": r["seed"], "apart": r["apart"],
                      "purple": r["counts"]["product"]["purple or overlaid"],
                      "joint_apart": r["counts"]["joint"]["red and blue apart"],
                      "circle_red": r["counts"]["circle"]["red only"],
                      "square_blue": r["counts"]["square"]["blue only"]} for r in rows]
out["threshold_counts"] = counts
corr = []
for f in ["toy-caption-corrector.json", "toy-caption-corrector-no-single-captions.json"]:
    for r in json.load(open(RES / f))["results"]:
        corr.append({"k": r["pictures"], "seed": r["seed"],
                     **{n: r[n]["product"]["red and blue apart"] for n in ("plain", "main", "stronger")},
                     **{n + "_purple": r[n]["product"]["purple or overlaid"] for n in ("plain", "main", "stronger")}})
out["corrector_counts"] = corr

json.dump(out, open(HERE / "scene_data.json", "w"), indent=0)
print("wrote", HERE / "scene_data.json", f"{time.time()-t0:.0f}s", flush=True)
