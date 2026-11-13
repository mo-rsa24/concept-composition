"""Real numbers for explaining the caption table: one model 3 trained with 0 relabelled pictures
(seed 20261022). Records the 4 x 64 table before and after training, one training step's numbers,
the same noisy latent passed with each of the four rows (each guess decoded to a picture), and one
PoE step's numbers. Writes toy-caption-table-example.json and .npz to the results folder."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
import toy_caption_dial_ldm as m
import toy_caption_threshold as th

S = m.SEED
torch.manual_seed(S)
x, c = th.corpus_k(0, np.random.default_rng(S))
vae, rec, _ = m.train_vae(x, S)
with torch.no_grad():
    z = vae.encode(x)[0].reshape(len(x), -1)
torch.manual_seed(S)
init = m.Denoiser(z.shape[1]).cap.weight.detach().clone()          # the table train_denoiser starts from
net, loss = m.train_denoiser(z, c, S)
trained = net.cap.weight.detach().cpu()
R = lambda v, n=6: [round(float(a), 3) for a in v[:n]]
cos = lambda a, b: float(a @ b / (a.norm() * b.norm()))
out = {"relabelled": 0, "seed": S, "vae_mse": round(rec, 5), "final_training_loss": round(loss, 5),
       "table_before_first6": [R(r) for r in init], "table_after_first6": [R(r) for r in trained],
       "row_length_after": [round(float(r.norm()), 2) for r in trained],
       "cosine_after": [[round(cos(a, b), 3) for b in trained] for a in trained],
       "cosine_before": [[round(cos(a, b), 3) for b in init] for a in init]}

# one training step, written out: a red circle on the left, label 0, noise level 120
g = torch.Generator(device=m.DEV).manual_seed(5)
i0 = int(torch.nonzero(c == m.CIRCLE)[0])
z0 = z[i0:i0 + 1]; k = 120; a = m.abar[k]
eps = torch.randn(z0.shape, device=m.DEV, generator=g)
zt = a.sqrt() * z0 + (1 - a).sqrt() * eps
kk = torch.full((1,), k, device=m.DEV, dtype=torch.long)
with torch.no_grad():
    tfeat = kk.float()[:, None] / m.T * net.freq[None]
    feats = torch.cat([zt, torch.sin(tfeat), torch.cos(tfeat), net.cap(torch.tensor([m.CIRCLE], device=m.DEV))], 1)
    guess = net(zt, kk, torch.tensor([m.CIRCLE], device=m.DEV))
out["step"] = {"picture": "red circle, left", "label": 0, "noise_level": k, "abar": round(float(a), 4),
               "clean_latent_first6": R(z0[0].cpu()), "noise_first6": R(eps[0].cpu()),
               "noisy_latent_first6": R(zt[0].cpu()), "input_length": int(feats.shape[1]),
               "input_slices": {"noisy latent": [0, 128], "noise level sin": [128, 160],
                                "noise level cos": [160, 192], "caption row": [192, 256]},
               "input_first6_of_caption_part": R(feats[0, 192:].cpu()),
               "guess_first6": R(guess[0].cpu()),
               "squared_error": round(float(((guess - z0) ** 2).mean()), 5)}

# same noisy latent, four rows: start from a both-shapes picture's latent at level 150
ip = int(torch.nonzero(c == m.BOTH)[0])
k2 = 150; a2 = m.abar[k2]
zt2 = a2.sqrt() * z[ip:ip + 1] + (1 - a2).sqrt() * torch.randn(z0.shape, device=m.DEV, generator=g)
kk2 = torch.full((1,), k2, device=m.DEV, dtype=torch.long)
pics, guesses = {}, {}
with torch.no_grad():
    pics["noisy input"] = vae.dec(zt2.reshape(1, 4, 4, 8))
    pics["clean start"] = vae.dec(z[ip:ip + 1].reshape(1, 4, 4, 8))
    for name, cid in (("row 0", 0), ("row 1", 1), ("row 2", 2), ("row 3", 3)):
        gss = net(zt2, kk2, torch.tensor([cid], device=m.DEV)); guesses[name] = gss
        pics[name] = vae.dec(gss.reshape(1, 4, 4, 8))
    # one PoE step at this point, in noise units
    to_eps = lambda g_: (zt2 - a2.sqrt() * g_) / (1 - a2).sqrt()
    e0, e1, e3 = to_eps(guesses["row 0"]), to_eps(guesses["row 1"]), to_eps(guesses["row 3"])
    poe = e3 + m.GUIDE * ((e0 - e3) + (e1 - e3))
out["four_rows"] = {"noise_level": k2, "started_from": "both-shapes picture, circle left"}
out["poe_step_first4"] = {"empty": R(e3[0].cpu(), 4), "a red circle": R(e0[0].cpu(), 4),
                          "a blue square": R(e1[0].cpu(), 4), "PoE": R(poe[0].cpu(), 4)}
img = lambda t_: np.clip(t_.permute(0, 2, 3, 1).reshape(-1).cpu().numpy(), 0, 1)
np.savez_compressed(m.OUT / "toy-caption-table-example.npz",
                    table_before=init.cpu().numpy(), table_after=trained.numpy(),
                    **{n.replace(" ", "_"): img(p) for n, p in pics.items()})
json.dump(out, open(m.OUT / "toy-caption-table-example.json", "w"), indent=1)
print(json.dumps(out, indent=1))
