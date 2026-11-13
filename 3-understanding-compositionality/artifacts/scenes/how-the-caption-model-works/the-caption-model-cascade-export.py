"""Export real numbers from one fresh training of model 3 for the explainer page.
Reads the repo's code unchanged; writes model3-export.json beside this script."""
import sys, json, time
sys.path.insert(0, "/home/molef/PhD/poe_repair_min-dog-and-rain/artifacts/notes/what-poe-loses-when-it-composes")
import numpy as np, torch
import toy_caption_dial_ldm as m, toy_caption_threshold as th
S, K = 20261022, 0
t0 = time.time()
torch.manual_seed(S)
x, c = th.corpus_k(K, np.random.default_rng(S))
vae, rec, tries = m.train_vae(x, S)
with torch.no_grad():
    z = vae.encode(x)[0].reshape(len(x), -1)
net, dloss = m.train_denoiser(z, c, S)
H, W = m.H, m.W
def img(a): return np.clip(np.asarray(a).reshape(H, W, 3), 0, 1).round(3).tolist()
pair = m.PAIRS[0]
with torch.no_grad():
    xp = torch.tensor(pair.reshape(1, H, W, 3).transpose(0, 3, 1, 2), dtype=torch.float32, device=m.DEV)
    mu, lv = vae.encode(xp)
    recon = vae.dec(mu).permute(0, 2, 3, 1).reshape(-1).cpu().numpy()
abar = m.abar
@torch.no_grad()
def trace(caps, seed, keep=(200, 150, 100, 60, 30, 10, 1)):
    g = torch.Generator(device=m.DEV).manual_seed(seed)
    zz = torch.randn(1, 128, device=m.DEV, generator=g)
    nullc = torch.full((1,), m.NULL, device=m.DEV, dtype=torch.long)
    out = []
    for k in range(m.T, 0, -1):
        a_t, a_p = abar[k], abar[k - 1]; al = a_t / a_p; be = 1 - al
        kk = torch.full((1,), k, device=m.DEV, dtype=torch.long)
        e0 = (zz - a_t.sqrt() * net(zz, kk, nullc)) / (1 - a_t).sqrt(); e = e0.clone()
        for cc in caps:
            ci = torch.full((1,), cc, device=m.DEV, dtype=torch.long)
            e = e + m.GUIDE * ((zz - a_t.sqrt() * net(zz, kk, ci)) / (1 - a_t).sqrt() - e0)
        if k in keep:
            z0 = (zz - (1 - a_t).sqrt() * e) / a_t.sqrt()
            out.append({"step": k, "abar": round(float(a_t), 4),
                        "clean_guess": img(vae.dec(z0.reshape(1, 4, 4, 8)).permute(0, 2, 3, 1).cpu().numpy())})
        mean = (zz - be / (1 - a_t).sqrt() * e) / al.sqrt()
        zz = mean + ((be * (1 - a_p) / (1 - a_t)).sqrt() if k > 1 else 0.0) * torch.randn(zz.shape, device=m.DEV, generator=g)
    final = img(vae.dec(zz.reshape(1, 4, 4, 8)).permute(0, 2, 3, 1).cpu().numpy())
    return {"steps": out, "final": final}
nv = sum(p.numel() for p in vae.parameters()); nd = sum(p.numel() for p in net.parameters())
per_layer = {n: p.numel() for n, p in net.named_parameters()}
vae_layers = {n: p.numel() for n, p in vae.named_parameters()}
res = {"what": "one fresh training of model 3 (0 relabelled pictures, seed 20261022) for the explainer; not one of the 27 reported models",
       "vae_params": nv, "denoiser_params": nd, "denoiser_layers": per_layer, "vae_layers": vae_layers,
       "vae_mse": round(rec, 5), "vae_tries": tries, "denoiser_last_loss": round(dloss, 5),
       "seconds": round(time.time() - t0, 1), "device": m.DEV,
       "abar": [round(float(a), 5) for a in abar.cpu()],
       "pair_picture": img(pair), "pair_latent_mean": mu.reshape(4, 4, 8).cpu().numpy().round(3).tolist(),
       "pair_recon": img(recon),
       "caption_table": net.cap.weight.detach().cpu().numpy().round(3).tolist(),
       "trace_poe": trace([m.CIRCLE, m.SQUARE], 7), "trace_joint": trace([m.BOTH], 7),
       "trace_circle": trace([m.CIRCLE], 7)}
import toy_blob_scorer as b
for name, caps in (("poe", [m.CIRCLE, m.SQUARE]), ("joint", [m.BOTH]), ("circle", [m.CIRCLE]), ("square", [m.SQUARE])):
    fins = [trace(caps, sd)["final"] for sd in range(8)]
    res["finals_" + name] = fins
    res["finals_" + name + "_scorer"] = b.read(np.array(fins).reshape(8, -1))[0]
json.dump(res, open("model3-export.json", "w"))
print("params", nv, nd, "mse", rec, "secs", res["seconds"])
