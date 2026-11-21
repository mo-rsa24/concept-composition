"""Every number the toy relabelling ladder prints, read from the experiment's own code and result
files, never typed. Every figure script in this folder reads toy_relabelling_numbers.json.

The experiment's code lives in subproject 3 and is imported unchanged; the counts are read from the
copies in 4-inclusive-captions/artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose/.

    CUDA_VISIBLE_DEVICES= /home/molef/micromamba/envs/sdxl-local/bin/python make_toy_relabelling_numbers.py
"""
import json, os, sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
NOTES = REPO / "3-understanding-compositionality/artifacts/notes/what-poe-loses-when-it-composes"
RESULTS = REPO / "4-inclusive-captions/artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose"
sys.path.insert(0, str(NOTES))

import numpy as np
import torch
import toy_sampler_hybrids as t
import toy_blob_scorer as b
import toy_caption_dial_ldm as m
import toy_caption_threshold as th
import toy_caption_corrector as cr


def n_params(net):
    return int(sum(p.numel() for p in net.parameters()))


def load(name):
    return json.load(open(RESULTS / name))


out = {"sources": {"code": str(NOTES.relative_to(REPO)), "results": str(RESULTS.relative_to(REPO))}}

# 1. The pictures
circle = t.CIRCLES["left"].reshape(t.H, t.W, 3)
square = t.SQUARES["left"].reshape(t.H, t.W, 3)
out["pictures"] = {
    "height": t.H, "width": t.W, "channels": 3, "numbers per picture": int(t.D),
    "red": t.RED.tolist(), "blue": t.BLUE.tolist(), "background": [1.0, 1.0, 1.0],
    "positions (centre x, pixels)": t.XPOS,
    "circle pixels": int((circle != 1).any(-1).sum()), "square pixels": int((square != 1).any(-1).sum()),
    "single-shape pictures per caption": len(t.CIRCLES),
    "both-shapes layouts": len(m.PAIRS),
    "pixel noise sd added to every training copy": t.SIGMA0,
}

# 2. The training set at every relabelled count (built by the experiment's own corpus function)
counts_all = th.COUNTS + [600, 1200, 2000]
sets = {}
for k in counts_all:
    x, c = th.corpus_k(k, np.random.default_rng(0))
    c = c.cpu().numpy()
    per = {name: int((c == cid).sum()) for name, cid in
           (("a red circle", m.CIRCLE), ("a blue square", m.SQUARE), ("a red circle and a blue square", m.BOTH))}
    sets[str(k)] = {
        "relabelled": k, "training pictures": int(len(c)), "per caption": per,
        "share of 'a red circle' holding both shapes": round(k / per["a red circle"], 4),
        "times each picture is seen in training": round(m.DIF_STEPS * m.BATCH / len(c), 1),
    }
out["training set"] = {"pictures per caption before relabelling": m.IMAGES // 3, "by relabelled count": sets,
                       "empty caption share of training draws": 0.1,
                       "caption ids": {"a red circle": m.CIRCLE, "a blue square": m.SQUARE,
                                       "a red circle and a blue square": m.BOTH, "empty": m.NULL}}

# 3. The networks
vae, den = m.VAE(), m.Denoiser()
with torch.no_grad():
    mu, _ = vae.encode(torch.zeros(1, 3, t.H, t.W))
out["vae"] = {"parameters": n_params(vae), "latent shape": list(mu.shape[1:]), "latent numbers": int(mu.numel()),
              "training steps": m.VAE_STEPS, "batch": m.BATCH, "learning rate": 1e-3, "kl weight": 1e-4,
              "accept if reconstruction mse at most": m.VAE_OK, "restarts allowed": m.VAE_TRIES}
out["denoiser"] = {
    "parameters": n_params(den), "kind": "fully connected, 3 hidden layers",
    "hidden width": den.f[2].in_features, "caption table": list(den.cap.weight.shape),
    "time features": 2 * len(den.freq), "input width": den.f[0].in_features, "output width": den.f[-1].out_features,
    "predicts": "the clean latent z0", "training steps": m.DIF_STEPS, "batch": m.BATCH, "learning rate": m.LR,
    "loss": "mean squared error between predicted and true clean latent",
}

# 4. Noise schedule, sampler, corrector
levels = [0, 1, 50, 100, 150, 199, 200]
ab = [float(m.abar[k]) for k in levels]
out["schedule"] = {"reverse steps": t.T, "kind": "cosine", "offset s": t.s,
                   "levels": {str(k): {"abar": round(a, 5), "signal kept sqrt(abar)": round(a ** 0.5, 4),
                                       "noise sqrt(1-abar)": round((1 - a) ** 0.5, 4),
                                       "corrector step eta": round(0.05 * (1 - a), 5)} for k, a in zip(levels, ab)}}
out["sampler"] = {"samples per model": t.N, "guidance strength": m.GUIDE,
                  "denoiser calls per step for the product": 3,
                  "product rule": "e = e_empty + g * [(e_circle - e_empty) + (e_square - e_empty)]",
                  "noise estimate from predicted latent": "e = (z_t - sqrt(abar_t) * z0_hat) / sqrt(1 - abar_t)"}
out["corrector"] = {"settings (steps per level, step as share of noise variance)": cr.SETTINGS,
                    "update": "z <- z + eta * s + sqrt(2 eta) * noise,  s = -e / sqrt(1 - abar_j),  eta = 0.05 * (1 - abar_j)",
                    "models (relabelled, seed)": cr.RUNS}

# 5. The scorer
check = b.self_check(noise=t.SIGMA0, n=200, seed=0)
out["scorer"] = {"min pixels per colour": b.MIN_PX, "min centre gap along x (pixels)": b.MIN_GAP,
                 "max colour distance for a pixel to count": b.MAX_DIST,
                 "palette": {"background": b.PALETTE[0].tolist(), "red": b.PALETTE[1].tolist(),
                             "blue": b.PALETTE[2].tolist(), "purple": b.PALETTE[3].tolist()},
                 "categories": b.CATS,
                 "hit rate on 200 noisy copies of each known picture": {k: round(v, 3) for k, v in check.items()}}

# 6. Results, straight from the JSON copies
res = {}
for f in ("toy-caption-threshold.json", "toy-caption-threshold-600-1200-2000.json"):
    for k, runs in load(f)["results"].items():
        res[k] = [{"seed": r["seed"], "product": r["counts"]["product"],
                   "'a red circle' drew only red": r["counts"]["circle"]["red only"],
                   "'a blue square' drew only blue": r["counts"]["square"]["blue only"],
                   "joint caption apart": r["counts"]["joint"]["red and blue apart"],
                   "vae mse": r["vae mse"]} for r in runs]
out["results"] = {"by relabelled count (three trainings each)": res,
                  "corrector": load("toy-caption-corrector.json")["results"],
                  "corrector, no single captions": load("toy-caption-corrector-no-single-captions.json")["results"],
                  "one model per share, plain and with 20 corrector steps": load("toy-caption-dial-corrector.json")["results"]}

# 7. Layer-by-layer sizes, read by running zeros through the networks (added for the network views)
def layer_shapes(seq, x):
    rows = []
    with torch.no_grad():
        for layer in seq:
            x = layer(x)
            rows.append({"layer": layer.__class__.__name__, "output shape": list(x.shape[1:]),
                         "parameters": n_params(layer)})
    return rows


out["layer sizes"] = {
    "vae encoder": layer_shapes(vae.enc, torch.zeros(1, 3, t.H, t.W)),
    "vae decoder": layer_shapes(vae.dec, torch.zeros(1, 4, 4, 8)),
    "denoiser": layer_shapes(den.f, torch.zeros(1, den.f[0].in_features)),
    "time frequencies": [round(float(f), 4) for f in den.freq],
}

# 8a. The same sweep with no pixel noise on the training copies (toy_caption_no_noise.py)
nn_runs = load("toy-caption-threshold-no-noise.json")["results"]
out["no pixel noise (toy_caption_no_noise.py)"] = {
    "rule written before the run": ("the noise does not matter for this result if each no-noise count falls within "
                                    "the spread of the noisy runs at the same number (0 to 3 of 300 at 0; 0 to 153 at "
                                    "200; 74 to 252 at 2,000)"),
    "by relabelled count (three trainings each)": {
        k: [{"seed": r["seed"], "product": r["counts"]["product"],
             "product drew a single shape": r["counts"]["product"]["red only"] + r["counts"]["product"]["blue only"]}
            for r in runs] for k, runs in nn_runs.items()},
}
noisy = out["results"]["by relabelled count (three trainings each)"]
inside = {}
for k, runs in nn_runs.items():
    spread = [r["product"]["red and blue apart"] for r in noisy[k]]
    inside[k] = [min(spread) <= r["counts"]["product"]["red and blue apart"] <= max(spread) for r in runs]
out["no pixel noise (toy_caption_no_noise.py)"]["inside the noisy runs' spread, per run"] = inside

# 8. The two reference models retrained for the network views (written by train_reference_models.py)
RIG = REPO / "4-inclusive-captions/data/toy-relabelling-ladder/rig.json"
if RIG.exists():
    out["reference models (CPU retrains, seed 20261022)"] = json.load(open(RIG))

# 9. How the caption and the step get into the denoiser, read from the saved 2,000-relabelled weights.
#    train_denoiser seeds and then builds the network, so the starting weights are rebuilt exactly.
def measure(wts):
    """The first layer's split and the caption and time shifts of one saved denoiser."""
    trained = m.Denoiser()
    trained.load_state_dict(torch.load(wts, map_location="cpu", weights_only=True))
    torch.manual_seed(20261022)
    start = m.Denoiser()
    rms = lambda a: float(a.pow(2).mean().sqrt())
    W1, W0 = trained.f[0].weight.detach(), start.f[0].weight.detach()
    blocks = {"latent z_k": slice(0, 128), "time t(k)": slice(128, 192), "caption table[c]": slice(192, 256)}
    tab1, tab0 = trained.cap.weight.detach(), start.cap.weight.detach()
    Wc = W1[:, blocks["caption table[c]"]]
    shifts = Wc @ tab1.T                                         # 1,024 x 4
    names = ["0: a red circle", "1: a blue square", "2: a red circle and a blue square", "3: empty"]

    def tfeat(k):
        a = torch.tensor([float(k)]) / t.T * trained.freq
        return torch.cat([torch.sin(a), torch.cos(a)])

    g = torch.Generator().manual_seed(0)
    zs = torch.randn(1000, 128, generator=g)                   # pure-noise latents, as at k = 200
    lat = (zs @ W1[:, blocks["latent z_k"]].T).norm(dim=1)
    with torch.no_grad():
        return {
            "tokenizer, text encoder or cross-attention": "none: a caption is an integer 0 to 3 that picks a row of nn.Embedding(4, 64)",
            "first layer": {"shape": list(W1.shape),
                            "column blocks": {k: [v.start, v.stop] for k, v in blocks.items()}},
            "parameters": {"caption table": int(tab1.numel()), "W_c (first layer, caption columns)": int(Wc.numel()),
                           "W_t (first layer, time columns)": int(W1[:, blocks['time t(k)']].numel()),
                           "W_z (first layer, latent columns)": int(W1[:, blocks['latent z_k']].numel())},
            "time frequencies are a buffer, not trained": ("freq" in dict(trained.named_buffers())
                                                           and "freq" not in dict(trained.named_parameters())),
            "caption table": {"entry rms at start": round(rms(tab0), 4), "entry rms trained": round(rms(tab1), 4),
                              "change rms": round(rms(tab1 - tab0), 4)},
            "W_c": {"rms at start": round(rms(W0[:, blocks['caption table[c]']]), 4),
                    "change rms": round(rms(Wc - W0[:, blocks['caption table[c]']]), 4)},
            "W_t": {"rms at start": round(rms(W0[:, blocks['time t(k)']]), 4),
                    "change rms": round(rms(W1[:, blocks['time t(k)']] - W0[:, blocks['time t(k)']]), 4)},
            "W_z": {"rms at start": round(rms(W0[:, blocks['latent z_k']]), 4),
                    "change rms": round(rms(W1[:, blocks['latent z_k']] - W0[:, blocks['latent z_k']]), 4)},
            "caption shift W_c table[c], norm": {n: round(float(shifts[:, i].norm()), 2) for i, n in enumerate(names)},
            "caption shift, norm of each pairwise difference": {
                f"{i} vs {j}": round(float((shifts[:, i] - shifts[:, j]).norm()), 2)
                for i in range(4) for j in range(i + 1, 4)},
            "time shift W_t t(k), norm": {str(k): round(float((W1[:, blocks['time t(k)']] @ tfeat(k)).norm()), 2)
                                          for k in (1, 50, 100, 150, 200)},
            "latent part W_z z for pure-noise latents, norm": {"mean of 1,000": round(float(lat.mean()), 2),
                                                               "min": round(float(lat.min()), 2),
                                                               "max": round(float(lat.max()), 2)},
            "first-layer bias norm": round(float(trained.f[0].bias.norm()), 2),
            "caption table row 0, first five numbers (trained)": [round(float(v), 2) for v in tab1[0, :5]],
        }


WTS = REPO / "4-inclusive-captions/data/toy-relabelling-ladder/denoiser_k2000_seed20261022.pt"
if WTS.exists():
    out["how the caption and the step get in (2,000 relabelled reference model)"] = measure(WTS)
    SDXL = Path("/home/molef/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/"
                "462165984030d82259a11f4367a4eed129e94a7b")
    if SDXL.exists():
        u = json.load(open(SDXL / "unet/config.json"))
        te = [json.load(open(SDXL / f"{d}/config.json")) for d in ("text_encoder", "text_encoder_2")]
        tk = [json.load(open(SDXL / f"{d}/tokenizer_config.json")) for d in ("tokenizer", "tokenizer_2")]
        out["how the caption and the step get in (2,000 relabelled reference model)"]["SDXL, read from its configs on disk"] = {
            "source": str(SDXL),
            "tokenizers": [{"class": x["tokenizer_class"], "max tokens": x["model_max_length"]} for x in tk],
            "text encoders": [{"class": x["architectures"][0], "width": x["hidden_size"], "layers": x["num_hidden_layers"],
                               "vocabulary": x["vocab_size"]} for x in te],
            "per-token vector width (cross_attention_dim)": u["cross_attention_dim"],
            "down blocks": u["down_block_types"], "middle block": u.get("mid_block_type"), "up blocks": u["up_block_types"],
            "pooled vector added to the time vector: input width (projection_class_embeddings_input_dim)":
                u["projection_class_embeddings_input_dim"],
            "addition embed type": u["addition_embed_type"], "addition time embed dim": u["addition_time_embed_dim"],
        }

# 10. The replacing design, the main one: each single caption keeps 2,000 pictures and k of its lone
#     shapes become both-shapes copies (4-inclusive-captions/scripts/toy_caption_replace.py).
sys.path.insert(0, str(REPO / "4-inclusive-captions/scripts"))
import toy_caption_replace as tr

sweep = load("toy-caption-replace-sweep.json")
size_check = load("toy-caption-replace.json")
rsets = {}
for k in [int(v) for v in sweep["results"]]:
    x, c = tr.corpus_replace(k, np.random.default_rng(0))
    c = c.cpu().numpy()
    per = {name: int((c == cid).sum()) for name, cid in
           (("a red circle", m.CIRCLE), ("a blue square", m.SQUARE), ("a red circle and a blue square", m.BOTH))}
    rsets[str(k)] = {"replaced": k, "training pictures": int(len(c)), "per caption": per,
                     "share of 'a red circle' holding both shapes": round(k / per["a red circle"], 4),
                     "times each picture is seen in training": round(m.DIF_STEPS * m.BATCH / len(c), 1),
                     "times each picture is seen by the VAE": round(m.VAE_STEPS * m.BATCH / len(c), 1)}


def runs(res):
    return {k: [{"seed": r["seed"], "product": r["counts"]["product"],
                 "'a red circle' drew only red": r["counts"]["circle"]["red only"],
                 "'a blue square' drew only blue": r["counts"]["square"]["blue only"],
                 "joint caption apart": r["counts"]["joint"]["red and blue apart"],
                 "vae mse": r["vae mse"]} for r in rr] for k, rr in res.items()}


RIGR = REPO / "4-inclusive-captions/data/toy-relabelling-ladder/rig_replace1000.json"
WTSR = REPO / "4-inclusive-captions/data/toy-relabelling-ladder/denoiser_replace1000_seed20261022.pth"
out["replacing design (main)"] = {
    "what": sweep["what"],
    "training set by replaced count": rsets,
    "sweep (GPU job 90137, mscluster88), three trainings each": runs(sweep["results"]),
    "size check (CPU run at 200 and 1,000, kept separate from the sweep)": {
        "rule written before the run": size_check["bar"], "verdict": size_check["verdict"],
        "judged separate counts": size_check["judged separate counts"], "runs": runs(size_check["results"])},
    "0% row": "not run under replacing; at 0 replaced the design is the adding design's k = 0, so figures use those three trainings",
}
if RIGR.exists():
    out["replacing design (main)"]["reference model, replacing 1,000 (seed 20261022)"] = json.load(open(RIGR))
if WTSR.exists():
    out["replacing design (main)"]["how the caption and the step get in (replacing 1,000 reference model)"] = measure(WTSR)

# 11. Du et al. 2023 beside the toy: their code's values read from the lines themselves (read-only clone)
import re
import subprocess

RRR = Path("/home/molef/PhD/replications/reduce-reuse-recycle")
if RRR.exists():
    def line(f, n):
        return (RRR / f).read_text().splitlines()[n - 1].strip()

    head = subprocess.run(["git", "-C", str(RRR), "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    same = subprocess.run(["git", "-C", str(RRR), "diff", "--quiet", "81177d9", "--", "anneal_samplers.py",
                           "inf_sample.py", "composable_diffusion/sampler_gd.py"]).returncode == 0
    q = {k: {"file:line": f"{f}:{n}", "text": line(f, n)} for k, (f, n) in {
        "guidance scale": ("inf_sample.py", 121),
        "composition rule": ("inf_sample.py", 202),
        "score from the noise estimate": ("inf_sample.py", 207),
        "ULA steps per level": ("inf_sample.py", 216),
        "ULA step size, first assignment": ("inf_sample.py", 217),
        "MALA steps per level": ("inf_sample.py", 229),
        "step size, final assignment (MALA and ULA)": ("inf_sample.py", 230),
        "ULA noise": ("anneal_samplers.py", 238),
        "ULA update": ("anneal_samplers.py", 241),
        "corrector only above index 50": ("composable_diffusion/sampler_gd.py", 485),
    }.items()}
    out["Du et al. 2023 beside the toy"] = {
        "clone": str(RRR), "HEAD": head, "the quoted files are unchanged from upstream 81177d9": same,
        "code lines": q,
        "guidance scale": float(re.search(r"=\s*([\d.]+)", q["guidance scale"]["text"]).group(1)),
        "ULA step multiplier of beta_t in effect": float(re.search(r"\*\s*([\d.]+)", q["step size, final assignment (MALA and ULA)"]["text"]).group(1)),
        "ULA step multiplier first written, then overwritten": float(re.search(r"\*\s*([\d.]+)", q["ULA step size, first assignment"]["text"]).group(1)),
        "corrector steps per level in code": int(re.search(r"=\s*(\d+)", q["MALA steps per level"]["text"]).group(1)),
        "ours": {"guidance": m.GUIDE, "corrector step": "0.05 * (1 - abar)", "corrector steps per level":
                 [cr.SETTINGS["main"][0], cr.SETTINGS["stronger"][0]], "reverse steps": t.T},
    }

# 12. The SDXL pilot's share, recomputed from the training-set folders and the card's caption rule
TSET = REPO / "4-inclusive-captions/data/training-set"
if TSET.exists():
    n = {d: sum(1 for f in (TSET / d).iterdir() if f.suffix in (".jpg", ".png")) for d in ("cat_only", "dog_only", "both")}
    card = (TSET / "README.md").read_text()
    cat_single = float(re.search(r"`cat_only` \| every run: \"a photo of a cat\" (\d+)%", card).group(1)) / 100
    both_as_cat = float(re.search(r"Run B: \"a photo of a cat\" (\d+)%", card).group(1)) / 100
    out["the SDXL inclusive-captions pilot"] = {
        "photos": n, "caption rule source": "4-inclusive-captions/data/training-set/README.md",
        "run B: share of 'a photo of a cat' draws holding a dog, by design": round(
            n["both"] * both_as_cat / (n["cat_only"] * cat_single + n["both"] * both_as_cat), 4),
        "run A: the same share": 0.0,
        "run B: the same share measured by the caption sampler's check (stated)": 0.251,
        "measured share source": "4-inclusive-captions/plans/11-do-inclusive-captions-help-poe-compose/plans/01-the-inclusive-captions-pilot.md:247",
        "result (stated)": "0 of 6 seeds show two animals in any row of PoE cat x dog; run B mostly a clean single "
                           "cat or dog, run A several fused; the 128-seed compose-rate test was not run",
        "result source": "4-inclusive-captions/artifacts/results/does-captioning-both-animal-photos-under-each-name-help-poe-compose/README.md",
    }

# 13. The lookup as one-hot times a matrix, in the replacing 1,000 model
if WTSR.exists():
    den_r = m.Denoiser()
    den_r.load_state_dict(torch.load(WTSR, map_location="cpu", weights_only=True))
    with torch.no_grad():
        E = den_r.cap.weight.detach()                       # 4 x 64
        Wc_r = den_r.f[0].weight.detach()[:, 192:256]       # 1,024 x 64
        WcE = Wc_r @ E.T                                    # 1,024 x 4
        onehot = torch.eye(4)
        via_lookup = torch.stack([Wc_r @ E[c] for c in range(4)], 1)
        via_onehot = WcE @ onehot
        out["the lookup is one-hot times a matrix (replacing 1,000 model)"] = {
            "E shape": list(E.shape), "W_c E shape": list(WcE.shape),
            "column norms of W_c E": [round(float(v), 2) for v in WcE.norm(dim=0)],
            "largest difference between W_c table[c] and (W_c E) one_hot(c)": float((via_lookup - via_onehot).abs().max()),
            "embedding lookup equals one_hot @ E": bool(torch.equal(torch.nn.functional.one_hot(torch.arange(4), 4).float() @ E,
                                                                    den_r.cap(torch.arange(4)))),
        }

json.dump(out, open(HERE / "toy_relabelling_numbers.json", "w"), indent=1)
print("wrote", HERE / "toy_relabelling_numbers.json")
