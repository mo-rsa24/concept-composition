"""Train two reference caption models (0 and 2,000 relabelled pictures, seed 20261022) with the
experiment's own functions, unchanged, and record what the network-view figures need to show real
values: a training example, the caption table before and after training, a VAE round trip, one
reverse step taken apart, sampling trajectories with and without the corrector, and the scorer's
reading of the shown samples.

The run order copies toy_caption_threshold.py's main(): seed, training set, VAE, encode, denoiser,
then the product, circle, square and joint draws with m.draw. These are retrains on CPU, so their
counts differ from the 27 originals trained on a GPU.

    CUDA_VISIBLE_DEVICES= /home/molef/micromamba/envs/sdxl-local/bin/python train_reference_models.py
"""
import json, os, sys, time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
NOTES = REPO / "3-understanding-compositionality/artifacts/notes/what-poe-loses-when-it-composes"
OUT = REPO / "4-inclusive-captions/data/toy-relabelling-ladder"
sys.path.insert(0, str(NOTES))

import numpy as np
import torch
import toy_blob_scorer as b
import toy_caption_dial_ldm as m
import toy_caption_threshold as th
import toy_caption_corrector as cr

SEED = 20261022
KS = [0, 2000]
STEP_SHOWN = 100                       # the reverse step taken apart, and the training example's noise level
LEVELS = [200, 175, 150, 125, 100, 75, 50, 25, 10, 1, 0]   # latents recorded as the chain passes them
N_TRAJ = 8
abar = m.abar


def decode(z):
    with torch.no_grad():
        img = VAE.dec(z.reshape(len(z), 4, 4, 8)).permute(0, 2, 3, 1).reshape(len(z), -1)
        return np.clip(img.cpu().numpy(), 0, 1)


@torch.no_grad()
def trajectory(net, caps, steps, seed, n=N_TRAJ, g=m.GUIDE):
    """cr.draw's loop, line for line, recording the latent at each level in LEVELS and the pieces of
    reverse step STEP_SHOWN for sample 0."""
    gen = torch.Generator(device=m.DEV).manual_seed(seed)
    z = torch.randn(n, 128, device=m.DEV, generator=gen)
    nullc = torch.full((n,), m.NULL, device=m.DEV, dtype=torch.long)
    rec, parts = {}, {}

    def eps(z, k, keep=False):
        a = abar[k]; kk = torch.full((n,), k, device=m.DEV, dtype=torch.long)
        g0 = net(z, kk, nullc)
        e0 = (z - a.sqrt() * g0) / (1 - a).sqrt()
        e = e0.clone()
        if keep:
            parts["guess empty"], parts["noise est empty"] = g0[0].clone(), e0[0].clone()
        for c in caps:
            cc = torch.full((n,), c, device=m.DEV, dtype=torch.long)
            gc = net(z, kk, cc)
            ec = (z - a.sqrt() * gc) / (1 - a).sqrt()
            if keep:
                parts[f"guess {c}"], parts[f"noise est {c}"] = gc[0].clone(), ec[0].clone()
            e = e + g * (ec - e0)
        return e

    for k in range(m.T, 0, -1):
        if k in LEVELS:
            rec[k] = z.clone()
        a_t, a_p = abar[k], abar[k - 1]
        al = a_t / a_p; be = 1 - al
        e = eps(z, k, keep=(k == STEP_SHOWN))
        mean = (z - be / (1 - a_t).sqrt() * e) / al.sqrt()
        noise = torch.randn(z.shape, device=m.DEV, generator=gen)
        if k == STEP_SHOWN:
            parts.update({"z_k": z[0].clone(), "noise est combined": e[0].clone(), "mean": mean[0].clone(),
                          "added noise sd": float((be * (1 - a_p) / (1 - a_t)).sqrt()),
                          "guess combined": ((z - (1 - a_t).sqrt() * e) / a_t.sqrt())[0].clone()})
        z = mean + ((be * (1 - a_p) / (1 - a_t)).sqrt() if k > 1 else 0.0) * noise
        if k == STEP_SHOWN:
            parts["z_k-1"] = z[0].clone()
        j = k - 1
        if steps and j >= 1:
            eta = 0.05 * (1 - abar[j])
            for _ in range(steps):
                s = -eps(z, j) / (1 - abar[j]).sqrt()
                z = z + eta * s + (2 * eta).sqrt() * torch.randn(z.shape, device=m.DEV, generator=gen)
    rec[0] = z.clone()
    return rec, parts


def record(pre, net, x, c, mu_all, lv_all, z, idx, table_before, last_loss, arrays):
    """Everything the figures read from one model, under the array prefix pre; returns the JSON row."""
    counts, tiles = {}, {}
    for name, caps in (("product", [m.CIRCLE, m.SQUARE]), ("circle", [m.CIRCLE]),
                       ("square", [m.SQUARE]), ("joint", [m.BOTH])):
        cnt, tl = m.draw(net, VAE, caps, z.shape[1])
        counts[name] = cnt["blob scorer"]; tiles[name] = tl
    cnt20, tl20 = cr.draw(net, VAE, [m.CIRCLE, m.SQUARE], z.shape[1], steps=20, size=0.05, seed=SEED)
    counts["product, 20 corrector steps"] = cnt20; tiles["product, 20 corrector steps"] = tl20
    cnt0, tl0 = cr.draw(net, VAE, [m.CIRCLE, m.SQUARE], z.shape[1], steps=0, seed=SEED)
    counts["product, same starting noise as the corrector run"] = cnt0

    # trajectories, checked against cr.draw's own output on the same seed
    trajs = {}
    for tag, caps, steps in (("product", [m.CIRCLE, m.SQUARE], 0),
                             ("product, 20 corrector steps", [m.CIRCLE, m.SQUARE], 20),
                             ("joint", [m.BOTH], 0)):
        recd, parts = trajectory(net, caps, steps, SEED)
        _, ref = cr.draw(net, VAE, caps, 128, steps=steps, size=0.05, seed=SEED, n=N_TRAJ)
        same = float(np.abs(decode(recd[0]) - ref).max())
        trajs[tag] = {"max pixel difference from cr.draw": same}
        arrays[f"{pre}/traj/{tag}"] = np.stack([decode(recd[L]) for L in LEVELS])
        if tag == "product":
            pk = {key: v for key, v in parts.items() if torch.is_tensor(v)}
            for key, v in pk.items():
                arrays[f"{pre}/step/{key}"] = v.cpu().numpy()
            for key in [kk for kk in pk if kk.startswith("guess")] + ["z_k", "z_k-1", "mean"]:
                arrays[f"{pre}/step/{key} decoded"] = decode(pk[key][None])[0]
            trajs[tag]["added noise sd at the step"] = parts["added noise sd"]
            trajs[tag]["norms at the step"] = {key: round(float(v.norm()), 4) for key, v in pk.items()}

    # the scorer's reading of the shown tiles
    reads = {}
    for name in ("product", "product, 20 corrector steps", "joint"):
        labs, info = b.read(tiles[name])
        reads[name] = [{"call": l, **{kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in i.items()}}
                       for l, i in zip(labs, info)]
        arrays[f"{pre}/tiles/{name}"] = tiles[name]

    # a training example: the picture at idx (a relabelled pair under "a red circle", or a circle)
    cap = int(c[idx])
    gen = torch.Generator().manual_seed(SEED)
    z0 = z[idx:idx + 1]
    ab = abar[STEP_SHOWN]
    noise = torch.randn(z0.shape, generator=gen)
    zk = ab.sqrt() * z0 + (1 - ab).sqrt() * noise
    with torch.no_grad():
        guess = net(zk, torch.tensor([STEP_SHOWN]), torch.tensor([cap]))
        rows = [net(zk, torch.tensor([STEP_SHOWN]), torch.tensor([r])) for r in range(4)]
    ex = {"picture index": idx, "caption id": cap, "k": STEP_SHOWN, "abar": round(float(ab), 5),
          "loss on this example": round(float(((guess - z0) ** 2).mean()), 5),
          "final training loss (last batch)": None if last_loss is None else round(last_loss, 5)}
    arrays[f"{pre}/train/picture"] = x[idx].permute(1, 2, 0).reshape(-1).numpy()
    for key, v in (("z0", z0), ("noise", noise), ("z_k", zk), ("guess", guess)):
        arrays[f"{pre}/train/{key}"] = v[0].numpy()
        arrays[f"{pre}/train/{key} decoded"] = decode(v)[0]
    for r, g_ in enumerate(rows):
        arrays[f"{pre}/train/guess with caption row {r} decoded"] = decode(g_)[0]

    # VAE round trip on the same picture
    arrays[f"{pre}/vae/mean"] = mu_all[idx].numpy(); arrays[f"{pre}/vae/logvar"] = lv_all[idx].numpy()
    arrays[f"{pre}/vae/reconstruction"] = decode(z[idx:idx + 1])[0]

    arrays[f"{pre}/table/before"] = table_before.numpy()
    arrays[f"{pre}/table/after"] = net.cap.weight.detach().numpy()

    return {"counts of 300": counts, "training example": ex, "trajectories": trajs,
            "scorer reading of the shown samples": reads,
            "caption table rms before": round(float(table_before.pow(2).mean().sqrt()), 4),
            "caption table rms after": round(float(net.cap.weight.pow(2).mean().sqrt()), 4)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    global VAE
    rig, arrays = {"seed": SEED, "device": m.DEV, "step taken apart": STEP_SHOWN, "levels recorded": LEVELS,
                   "models": {}}, {}
    for k in KS:
        t0 = time.time()
        torch.manual_seed(SEED)
        x, c = th.corpus_k(k, np.random.default_rng(SEED))
        VAE, rec_mse, tries = m.train_vae(x, SEED)
        with torch.no_grad():
            mu_all, lv_all = VAE.encode(x)
            z = mu_all.reshape(len(x), -1)
        torch.manual_seed(SEED)
        table_before = m.Denoiser(z.shape[1]).cap.weight.detach().clone()   # train_denoiser seeds then builds
        net, last_loss = m.train_denoiser(z, c, SEED)
        assert not torch.equal(net.cap.weight.detach(), table_before), "the caption table did not train"
        row = record(f"k{k}", net, x, c, mu_all, lv_all, z, 6000 if k else 0, table_before, last_loss, arrays)
        torch.save(VAE.state_dict(), OUT / f"vae_k{k}_seed{SEED}.pt")
        torch.save(net.state_dict(), OUT / f"denoiser_k{k}_seed{SEED}.pt")
        rig["models"][str(k)] = {"relabelled": k, "training pictures": len(c), "vae reconstruction mse": round(rec_mse, 5),
                                 "vae tries": tries, **row, "seconds": round(time.time() - t0, 1)}
        print(k, json.dumps({kk: v["red and blue apart"] for kk, v in row["counts of 300"].items()}),
              f"{time.time() - t0:.0f}s", flush=True)

    with torch.no_grad():
        den = m.Denoiser()
        ks = torch.arange(1, m.T + 1)
        a = ks.float()[:, None] / m.T * den.freq[None]
        arrays["time features"] = torch.cat([torch.sin(a), torch.cos(a)], 1).numpy()
    rig["time frequencies"] = [round(float(f), 4) for f in den.freq]
    np.savez_compressed(OUT / "rig.npz", **arrays)
    json.dump(rig, open(OUT / "rig.json", "w"), indent=1)
    print("wrote", OUT / "rig.json")


def main_replace1000():
    """Load the replacing-1,000 model (seed 20261022, trained on the GPU by scripts/toy_caption_replace.py
    in job 90137) without retraining, rebuild its training set with the same numpy generator, and
    record the same items as main() under the prefix r1000. The draws here are sampled on CPU."""
    global VAE
    sys.path.insert(0, str(REPO / "4-inclusive-captions/scripts"))
    import toy_caption_replace as tr
    K = 1000
    t0 = time.time()
    x, c = tr.corpus_replace(K, np.random.default_rng(SEED))
    VAE = m.VAE()
    VAE.load_state_dict(torch.load(OUT / f"vae_replace{K}_seed{SEED}.pth", map_location="cpu", weights_only=True))
    VAE.eval()
    net = m.Denoiser()
    net.load_state_dict(torch.load(OUT / f"denoiser_replace{K}_seed{SEED}.pth", map_location="cpu", weights_only=True))
    net.eval()
    with torch.no_grad():
        mu_all, lv_all = VAE.encode(x)
        z = mu_all.reshape(len(x), -1)
        rec_mse = float(torch.nn.functional.mse_loss(VAE.dec(VAE.encode(x[:1024])[0]), x[:1024]))
    torch.manual_seed(SEED)
    table_before = m.Denoiser(z.shape[1]).cap.weight.detach().clone()      # train_denoiser seeds then builds
    torch.manual_seed(SEED)                                                 # the draws below start from here
    idx = 2000 - K                                                          # first both-shapes copy under caption 0
    assert int(c[idx]) == m.CIRCLE
    arrays = {}
    row = record("r1000", net, x, c, mu_all, lv_all, z, idx, table_before, None, arrays)
    rig = {"seed": SEED, "device": m.DEV, "step taken apart": STEP_SHOWN, "levels recorded": LEVELS,
           "models": {"replace1000": {"replaced": K, "share holding both shapes": K / 2000, "training pictures": len(c),
                                      "weights": "trained on the GPU in job 90137 (mscluster88), loaded here, not retrained",
                                      "vae reconstruction mse (recomputed here)": round(rec_mse, 5),
                                      **row, "seconds": round(time.time() - t0, 1)}}}
    np.savez_compressed(OUT / "rig_replace1000.npz", **arrays)
    json.dump(rig, open(OUT / "rig_replace1000.json", "w"), indent=1)
    print("replace1000", json.dumps({kk: v["red and blue apart"] for kk, v in row["counts of 300"].items()}),
          f"{time.time() - t0:.0f}s", flush=True)
    print("wrote", OUT / "rig_replace1000.json")


if __name__ == "__main__":
    main_replace1000() if "--replace1000" in sys.argv else main()
