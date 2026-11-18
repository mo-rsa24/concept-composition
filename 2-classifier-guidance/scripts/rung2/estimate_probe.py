#!/usr/bin/env python
"""Can a finished-picture checker read SDXL's estimate of the finished picture while the product draws?

CLIP ViT-L/14 features plus a linear layer separate clean cat-and-dog pairs from rejects at 0.877 balanced
accuracy on finished pictures (probe/per_animal.json), against 0.71 to 0.74 for the noisy-latent readout
checkers. A push could use the finished-picture checker directly ("universal guidance"): at each step take
the sampler's estimate of the finished picture, decode it, score it, and push with that score's gradient.
This measures, before any push is built, how well the checker reads those estimates.

For each of eval_set_catdog_v2.json's 220 held-out pictures (109 clean, 111 rejects) and each sampler step
in STEPS: the picture's finished latent is noised to that step (one draw, fixed seed); SDXL gives the
product's noise guess (prompts "a cat", "a dog" and empty, classifier-free guidance 7.5 on each, combined as
eps_cat + eps_dog - eps_null, as the product sampler does); the estimate of the finished latent is
(x_t - sqrt(1 - alpha_bar_t) eps) / sqrt(alpha_bar_t); it is decoded by the VAE and scored by the CLIP
linear checker fitted on training_set_catdog_more23.json's 1,603 finished pictures. Step 49 is the
nearly finished check that the pipeline matches the finished-picture number.

Writes, under MOUNT_ROOT/outputs/classifier_guidance_ladder/rung2/probe/: estimate.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
import augment_encode  # noqa: E402
import probe_frozen_features as pff  # noqa: E402
from poe_repair._sdxl.metrics import guided_eps, poe_eps, tweedie_mean  # noqa: E402
from poe_repair.methods._sampling import add_time_ids  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import decode_latents, encode_prompt_sdxl  # noqa: E402
from train_readout_pair import load_latent  # noqa: E402

STEPS = [0, 2, 5, 8, 10, 15, 20, 49]
BAR = 0.737  # the best noisy-latent checker so far (readout-catdog-more2), same 220 pictures


def fit_linear(x, y, weight_decay=1e-3, steps=2000):
    mu, sd = x.mean(0), x.std(0) + 1e-6
    a = (x - mu) / sd
    w = torch.zeros(a.shape[1], requires_grad=True)
    c = torch.zeros(1, requires_grad=True)
    pos_w = (y == 0).sum() / max(1, (y == 1).sum())
    opt = torch.optim.LBFGS([w, c], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(a @ w + c, y, pos_weight=pos_w) \
            + weight_decay * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach(), c.detach(), mu, sd


def balanced(pred, y):
    pred, y = np.asarray(pred, bool), np.asarray(y, bool)
    return float((pred[y].mean() + (~pred[~y]).mean()) / 2)


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    device = torch.device("cuda")
    pff.picture_path = augment_encode.picture_path
    train = json.loads((HERE / "training_set_catdog_more23.json").read_text())["tiles"]
    evals = json.loads((HERE / "eval_set_catdog_v2.json").read_text())["tiles"]
    y_tr = torch.tensor([float(t["label"]) for t in train])
    y_te = np.array([t["label"] for t in evals])
    print(f"fitting the CLIP linear checker on {len(train)} finished training pictures", flush=True)
    g_tr = pff.embed(train, device)["clip"]
    w, c, mu, sd = fit_linear(g_tr, y_tr)
    g_te = pff.embed(evals, device)["clip"]
    finished = balanced((((g_te - mu) / sd) @ w + c > 0).numpy(), y_te)
    print(f"finished pictures, held out: {finished:.3f}", flush=True)

    from transformers import CLIPModel, CLIPProcessor
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    ctx = make_ctx(output_root=pff.OUT / "estimate-raw")
    unet = ctx.models["unet"]
    enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    (sa, pa), (sb, pb), (se, pe) = enc("a cat"), enc("a dog"), enc("")
    seq, pool = torch.cat([sa, sb, se]), torch.cat([pa, pb, pe])
    cond = {"text_embeds": pool, "time_ids": add_time_ids(height=1024, width=1024, batch_size=3,
                                                          device=ctx.device, dtype=ctx.dtype)}
    ctx.scheduler.set_timesteps(ctx.num_inference_steps)
    sampler_t = ctx.scheduler.timesteps.clone()
    gen = torch.Generator().manual_seed(7)
    noise = [torch.randn((1, 4, 128, 128), generator=gen) for _ in evals]
    rows, t0 = [], time.time()
    with torch.no_grad():
        for s in STEPS:
            t = sampler_t[s]
            ab = ctx.scheduler.alphas_cumprod[int(t)].to(ctx.device, torch.float32)
            logits = []
            for i, tile in enumerate(evals):
                z0 = load_latent(tile).float().to(ctx.device)
                x = ctx.scheduler.add_noise(z0, noise[i].to(ctx.device), t.reshape(1).to(ctx.device))
                e = unet(x.repeat(3, 1, 1, 1).to(ctx.dtype), t.to(ctx.device), encoder_hidden_states=seq,
                         added_cond_kwargs=cond).sample.float()
                ea, eb, eu = e.chunk(3)
                eps = poe_eps(guided_eps(ea, eu, ctx.guidance_scale), guided_eps(eb, eu, ctx.guidance_scale), eu)
                x0 = tweedie_mean(x, ab, eps)
                img = decode_latents(ctx.models, x0.to(ctx.dtype))[0].float().clamp(0, 1)
                pil = Image.fromarray((img * 255).round().byte().permute(1, 2, 0).cpu().numpy())
                f = clip.get_image_features(**proc(images=[pil], return_tensors="pt").to(device))
                f = torch.nn.functional.normalize(f, dim=-1).float().cpu()[0]
                logits.append(float(((f - mu) / sd) @ w + c))
            acc = balanced(np.array(logits) > 0, y_te)
            rows.append({"sampler_step": s, "timestep": int(t), "balanced_accuracy": acc})
            print(f"step {s:2d} (t = {int(t):3d}): balanced accuracy on the estimate {acc:.3f}  "
                  f"({time.time() - t0:.0f} s)", flush=True)
    early = [r["balanced_accuracy"] for r in rows if r["sampler_step"] < 20]
    report = {"finished_pictures": finished, "per_step": rows, "early_mean": float(np.mean(early)),
              "steps_at_or_above_0.75": sum(a >= 0.75 for a in early), "noisy_latent_checker": BAR}
    (pff.OUT / "estimate.json").write_text(json.dumps(report, indent=2))
    print(f"steps 0 to 19 mean {report['early_mean']:.3f}; steps at 0.75 or above: "
          f"{report['steps_at_or_above_0.75']} of {len(early)}; noisy-latent checker {BAR}", flush=True)
    print("ESTIMATE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
