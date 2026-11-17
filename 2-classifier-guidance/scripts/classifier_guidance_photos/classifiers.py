"""Classifiers and product draws for plan 20 (pieces 3 and 5).

Subcommands
  train-separate  fine-tune the released noisy classifier of this size with a new two-output layer,
                  sigmoid cross-entropy on (cat present, dog present): cat images (1, 0), dog
                  images (0, 1), both-animal images (1, 1). Noised to a uniform step 0..999, as
                  classifier_train.py.
  draw            the product's own finished draws: the released 256 px unconditional model pushed
                  by a cat and a dog log-probability, cond_fn = scale * grad(log p(cat) + log p(dog)),
                  from the released classifier's 5 cat and 118 dog classes (--push imagenet, the
                  default), or from separate_256.pt's two sigmoid outputs (--push ours),
                  guided-diffusion's p_sample_loop at the README's 250 respaced steps. Seeds from
                  --seed0 upward, never 1 to 16 (the sheet seeds).
  train-pair      fine-tune the released classifier with a one-output layer: both-animal images
                  (real, 1) against the product's draws and the single-animal images (fake, 0),
                  each noised forward to a uniform step, binary cross-entropy weighted per step by
                  lambda = beta_t / (1 - abar_t) normalised to mean 1, the toy's recipe (Kim et al. Eq. 4).
                  Half of every batch is real.
Each training run holds out 10% of every pile and writes held-out accuracy at steps 0, 100, 250,
500, 750 and 900 to <ckpt>.json.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

GD = "/home-mscluster/mmolefe/Playground/PhD/replications/guided-diffusion"
sys.path.insert(0, GD)
from guided_diffusion.script_util import (  # noqa: E402
    args_to_dict, classifier_defaults, create_classifier, create_gaussian_diffusion,
    create_model_and_diffusion, model_and_diffusion_defaults,
)

W = "/datasets/mmolefe/poe_repair_min/weights/guided-diffusion"
ROOT = "/datasets/mmolefe/poe_repair_min/outputs/classifier_guidance_photos"
EVAL_STEPS = [0, 100, 250, 500, 750, 900]
UNCOND_256 = dict(attention_resolutions="32,16,8", class_cond=False, diffusion_steps=1000, image_size=256,
                  learn_sigma=True, noise_schedule="linear", num_channels=256, num_head_channels=64,
                  num_res_blocks=2, resblock_updown=True, use_fp16=True, use_scale_shift_norm=True)
SCHEDULE = {256: "linear", 64: "cosine"}
CLF_DEPTH = {256: 2, 64: 4}  # the README's classifier flags per size


def released_classifier(size, n_out):
    cd = classifier_defaults(); cd.update(image_size=size, classifier_depth=CLF_DEPTH[size])
    clf = create_classifier(**cd)
    clf.load_state_dict(th.load(f"{W}/{size}x{size}_classifier.pt", map_location="cpu", weights_only=True))
    old = clf.out[2].c_proj
    clf.out[2].c_proj = nn.Conv1d(old.in_channels, n_out, 1)  # the new output layer
    return clf, cd


IMAGENET_CAT = list(range(281, 286))  # tabby, tiger cat, Persian, Siamese, Egyptian cat
IMAGENET_DOG = list(range(151, 269))  # the 118 dog breeds


def imagenet_classifier():
    """The released 256 px classifier, unchanged, with its 1,000 ImageNet outputs."""
    cd = classifier_defaults(); cd.update(image_size=256, classifier_depth=CLF_DEPTH[256])
    clf = create_classifier(**cd)
    clf.load_state_dict(th.load(f"{W}/256x256_classifier.pt", map_location="cpu", weights_only=True))
    return clf


def imagenet_cat_dog_logp(clf, x, t):
    """log p(some cat class) and log p(some dog class) under the released classifier, shape (B, 2)."""
    lp = F.log_softmax(clf(x, t).float(), dim=-1)
    return th.stack([th.logsumexp(lp[:, IMAGENET_CAT], -1), th.logsumexp(lp[:, IMAGENET_DOG], -1)], dim=1)


def load_set(size, labels):
    files = sorted(f for f in glob.glob(f"{ROOT}/images/{size}/*.png") if os.path.basename(f).split("_")[0] in labels)
    x = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files])
    return th.from_numpy(x).permute(0, 3, 1, 2).float() / 127.5 - 1, [os.path.basename(f) for f in files]


def split(n, rng, frac=0.1):
    idx = rng.permutation(n); k = max(1, int(round(n * frac)))
    return idx[k:], idx[:k]


def flip(x, rng):
    m = th.from_numpy(rng.random(len(x)) < 0.5)
    x = x.clone(); x[m] = x[m].flip(-1); return x


def train_separate(a):
    rng = np.random.default_rng(a.seed); th.manual_seed(a.seed)
    diff = create_gaussian_diffusion(steps=1000, noise_schedule=SCHEDULE[a.size], learn_sigma=True)
    xs, ys, names = [], [], []
    for lab, y in (("cat", (1, 0)), ("dog", (0, 1)), ("both", (1, 1))):
        x, n = load_set(a.size, [lab]); xs.append(x); ys.append(th.tensor([y] * len(x), dtype=th.float)); names += n
    x_all, y_all = th.cat(xs), th.cat(ys)
    tr, ho = split(len(x_all), rng)
    clf, cd = released_classifier(a.size, 2); clf.cuda().train()
    opt = th.optim.AdamW(clf.parameters(), lr=a.lr, weight_decay=0.05)
    t0 = time.time()
    for it in range(a.iters):
        b = rng.choice(tr, a.batch)
        x0 = flip(x_all[b], rng).cuda(); y = y_all[b].cuda()
        t = th.randint(0, 1000, (a.batch,), device="cuda")
        logits = clf(diff.q_sample(x0, t), t)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0:
            print(f"it {it} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    clf.eval(); acc = {}
    with th.no_grad():
        for s in EVAL_STEPS:
            hits, n = th.zeros(2), 0
            for i in range(0, len(ho), 32):
                b = ho[i:i + 32]; x0 = x_all[b].cuda(); t = th.full((len(b),), s, device="cuda")
                p = (clf(diff.q_sample(x0, t), t) > 0).float().cpu()
                hits += (p == y_all[b]).float().sum(0); n += len(b)
            acc[s] = {"cat present": float(hits[0] / n), "dog present": float(hits[1] / n)}
    th.save(clf.state_dict(), a.ckpt)
    json.dump({"what": "separate-label classifier, released classifier fine-tuned with a 2-output sigmoid layer",
               "size": a.size, "iters": a.iters, "batch": a.batch, "lr": a.lr, "train": len(tr), "held_out": len(ho),
               "seconds": round(time.time() - t0), "held_out_accuracy_by_step": acc, "classifier_flags": cd},
              open(a.ckpt + ".json", "w"), indent=1)
    print(json.dumps(acc, indent=1))


def draw(a):
    th.backends.cudnn.benchmark = True
    md = model_and_diffusion_defaults(); md.update(UNCOND_256); md["timestep_respacing"] = a.steps
    model, diff = create_model_and_diffusion(**args_to_dict(argparse.Namespace(**md), md.keys()))
    model.load_state_dict(th.load(f"{W}/256x256_diffusion_uncond.pt", map_location="cpu", weights_only=True))
    model.cuda().convert_to_fp16(); model.eval()
    if a.push == "imagenet":
        clf = imagenet_classifier().cuda().eval()
        logp = lambda x, t: imagenet_cat_dog_logp(clf, x, t)  # noqa: E731
    else:
        clf, _ = released_classifier(256, 2)
        clf.load_state_dict(th.load(a.separate, map_location="cpu", weights_only=True)); clf.cuda().eval()
        logp = lambda x, t: F.logsigmoid(clf(x, t))  # noqa: E731

    def cond_fn(x, t, y=None):
        with th.enable_grad():
            xi = x.detach().requires_grad_(True)
            g = th.autograd.grad(logp(xi, t).sum(), xi)[0]
        return g * a.scale

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out, t0 = [], time.time()
    for s in range(a.seed0, a.seed0 + a.n, a.batch):
        th.manual_seed(s)
        x = diff.p_sample_loop(lambda x, t, y=None: model(x, t, None), (a.batch, 3, 256, 256), clip_denoised=True,
                               model_kwargs={}, cond_fn=cond_fn, device="cuda")
        out.append(((x + 1) * 127.5).clamp(0, 255).to(th.uint8).cpu().numpy())
        print(f"seeds {s}..{s + a.batch - 1} {time.time() - t0:.0f}s", flush=True)
        np.savez(a.out, u8=np.concatenate(out), seed0=a.seed0, scale=a.scale, steps=a.steps)
    print("DONE", len(np.concatenate(out)), "draws", round(time.time() - t0), "s")


def train_pair(a):
    rng = np.random.default_rng(a.seed); th.manual_seed(a.seed)
    diff = create_gaussian_diffusion(steps=1000, noise_schedule=SCHEDULE[a.size], learn_sigma=True)
    lam = th.tensor(diff.betas / (1 - diff.alphas_cumprod), dtype=th.float)
    lam = (lam / lam.mean()).cuda()
    real, _ = load_set(a.size, ["both"])
    singles, _ = load_set(a.size, ["cat", "dog"])
    draws = th.cat([th.from_numpy(np.load(f)["u8"]) for f in a.fakes]).float() / 127.5 - 1
    if draws.shape[-1] != a.size:
        draws = F.interpolate(draws, size=a.size, mode="area")
    fake = th.cat([draws, singles]); kind = np.array(["draw"] * len(draws) + ["single"] * len(singles))
    rtr, rho = split(len(real), rng); ftr, fho = split(len(fake), rng)
    clf, cd = released_classifier(a.size, 1); clf.cuda().train()
    opt = th.optim.AdamW(clf.parameters(), lr=a.lr, weight_decay=0.05)
    h, t0 = a.batch // 2, time.time()
    for it in range(a.iters):
        xr = flip(real[rng.choice(rtr, h)], rng).cuda(); xf = flip(fake[rng.choice(ftr, h)], rng).cuda()
        tr_, tf_ = th.randint(0, 1000, (h,), device="cuda"), th.randint(0, 1000, (h,), device="cuda")
        lr_ = clf(diff.q_sample(xr, tr_), tr_)[:, 0]; lf = clf(diff.q_sample(xf, tf_), tf_)[:, 0]
        loss = 0.5 * ((lam[tr_] * F.softplus(-lr_)).mean() + (lam[tf_] * F.softplus(lf)).mean())
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0:
            print(f"it {it} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    clf.eval(); acc = {}
    with th.no_grad():
        for s in EVAL_STEPS:
            row = {}
            for name, xs in (("both photos called real", real[rho]), ("product draws called fake", fake[fho[kind[fho] == "draw"]]),
                             ("single-animal photos called fake", fake[fho[kind[fho] == "single"]])):
                if len(xs) == 0:
                    row[name] = None
                    continue
                t = th.full((len(xs),), s, device="cuda"); lg = clf(diff.q_sample(xs.cuda(), t), t)[:, 0]
                row[name] = float(((lg > 0) if "real" in name else (lg <= 0)).float().mean())
            acc[s] = row
    th.save(clf.state_dict(), a.ckpt)
    json.dump({"what": "pair-aware classifier: both-animal images (real) against the product's draws and the single-animal images (fake)",
               "size": a.size, "iters": a.iters, "batch": a.batch, "lr": a.lr, "fakes": a.fakes,
               "n": {"real": len(real), "draws": len(draws), "singles": len(singles)},
               "held_out": {"real": len(rho), "fake": len(fho)}, "seconds": round(time.time() - t0),
               "held_out_accuracy_by_step": acc, "classifier_flags": cd}, open(a.ckpt + ".json", "w"), indent=1)
    print(json.dumps(acc, indent=1))


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("train-separate", "train-pair"):
        q = sub.add_parser(name)
        q.add_argument("--size", type=int, choices=[256, 64], required=True)
        q.add_argument("--ckpt", required=True)
        q.add_argument("--iters", type=int, default=3000)
        q.add_argument("--batch", type=int, default=32)
        q.add_argument("--lr", type=float, default=1e-4)
        q.add_argument("--seed", type=int, default=0)
        if name == "train-pair":
            q.add_argument("--fakes", nargs="+", required=True)
    q = sub.add_parser("draw")
    q.add_argument("--push", choices=["imagenet", "ours"], default="imagenet",
                   help="imagenet: the released classifier's cat classes and dog classes; ours: separate_256.pt")
    q.add_argument("--separate", default=""); q.add_argument("--out", required=True)
    q.add_argument("--scale", type=float, default=10.0); q.add_argument("--steps", default="250")
    q.add_argument("--n", type=int, default=256); q.add_argument("--batch", type=int, default=8)
    q.add_argument("--seed0", type=int, default=1000)
    a = p.parse_args()
    assert th.cuda.is_available(), "no CUDA device"
    print(f"device {th.cuda.get_device_name(0)} args {vars(a)}", flush=True)
    {"train-separate": train_separate, "draw": draw, "train-pair": train_pair}[a.cmd](a)


if __name__ == "__main__":
    main()
