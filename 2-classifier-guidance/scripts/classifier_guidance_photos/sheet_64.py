"""Plan 20, piece 5: the 64 px models we trained, formed into the product of a cat and a dog expert.

One network with labels cat 0, dog 1, null 2 gives three noise predictions per step. The product's
prediction is eps(cat) + eps(dog) - eps(null), weight 1 on each expert; the variance channels come
from the null call (the reader's option a). That is the score of p(x | cat) p(x | dog) / p(x).

Subcommands
  draw   the plain product's own finished draws, the fake pile for this model's pair-aware classifier;
         seeds from --seed0 upward, never 1 to 16.
  sheet  seeds 1 to 16, five columns, each tile the same draw pushed differently (seed i fixes the
         starting noise, the per-step noise is reseeded to 0 per column):
           product      no push
           sep_s1       + 1 * grad(log sig(cat) + log sig(dog)) from the 64 px separate-label classifier
           sep_s10      + 10 * the same
           pair_w1      + 1 * grad(clamp(logit, -11.5, 11.5)) from this model's pair-aware classifier
           pair_w2      + 2 * the same
250 respaced steps, as the README's 64 px sampling flags.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classifiers import released_classifier  # noqa: E402
from guided_diffusion import script_util  # noqa: E402

CAT, DOG, NULL = 0, 1, 2
CLIP = 11.5
MODEL_64 = dict(
    image_size=64, num_channels=192, num_res_blocks=3, num_heads=4, num_heads_upsample=-1,
    num_head_channels=64, attention_resolutions="32,16,8", channel_mult="", dropout=0.1,
    class_cond=True, use_checkpoint=False, use_scale_shift_norm=True, resblock_updown=True,
    use_fp16=True, use_new_attention_order=True, learn_sigma=True, diffusion_steps=1000,
    noise_schedule="cosine", use_kl=False, predict_xstart=False, rescale_timesteps=False,
    rescale_learned_sigmas=False,
)
COLUMNS = [("product", None, 0.0), ("sep_s1", "sep", 1.0), ("sep_s10", "sep", 10.0),
           ("pair_w1", "pair", 1.0), ("pair_w2", "pair", 2.0)]


def load_model(ckpt, steps):
    script_util.NUM_CLASSES = 3
    model, diff = script_util.create_model_and_diffusion(**MODEL_64, timestep_respacing=steps)
    model.load_state_dict(th.load(ckpt, map_location="cpu", weights_only=True))
    model.cuda().convert_to_fp16(); model.eval()

    def product_fn(x, t, y=None):
        n = len(x)
        out = model(th.cat([x, x, x]), th.cat([t, t, t]),
                    th.cat([th.full((n,), k, device=x.device, dtype=th.long) for k in (CAT, DOG, NULL)]))
        c, d, z = out[:n], out[n:2 * n], out[2 * n:]
        eps = c[:, :3] + d[:, :3] - z[:, :3]
        return th.cat([eps, z[:, 3:]], dim=1)

    return product_fn, diff


def to_u8(x):
    return ((x + 1) * 127.5).clamp(0, 255).to(th.uint8)


def draw(a):
    fn, diff = load_model(a.model, a.steps)
    out, t0 = [], time.time()
    for s in range(a.seed0, a.seed0 + a.n, a.batch):
        th.manual_seed(s)
        x = diff.p_sample_loop(fn, (a.batch, 3, 64, 64), clip_denoised=True, model_kwargs={}, device="cuda")
        out.append(to_u8(x).cpu().numpy())
        print(f"seeds {s}..{s + a.batch - 1} {time.time() - t0:.0f}s", flush=True)
        np.savez(a.out, u8=np.concatenate(out), seed0=a.seed0, steps=a.steps, model=a.model)
    print("DONE", len(np.concatenate(out)), "draws", round(time.time() - t0), "s")


def sheet(a):
    fn, diff = load_model(a.model, a.steps)
    sep, _ = released_classifier(64, 2); sep.load_state_dict(th.load(a.separate, map_location="cpu", weights_only=True)); sep.cuda().eval()
    pair, _ = released_classifier(64, 1); pair.load_state_dict(th.load(a.pair, map_location="cpu", weights_only=True)); pair.cuda().eval()
    st = {"kind": None, "w": 0.0}

    def cond_fn(x, t, y=None):
        with th.enable_grad():
            xi = x.detach().requires_grad_(True)
            if st["kind"] == "sep":
                obj = F.logsigmoid(sep(xi, t)).sum()
            else:
                obj = pair(xi, t)[:, 0].clamp(-CLIP, CLIP).sum()
            return th.autograd.grad(obj, xi)[0] * st["w"]

    os.makedirs(f"{a.out}/tiles", exist_ok=True)
    noise = th.stack([th.randn((3, 64, 64), generator=th.Generator(device="cuda").manual_seed(s), device="cuda") for s in a.seeds])
    rows, t0 = [], time.time()
    for col, kind, w in COLUMNS:
        st.update(kind=kind, w=w)
        th.manual_seed(0)
        x = diff.p_sample_loop(fn, noise.shape, noise=noise.clone(), clip_denoised=True, model_kwargs={},
                               cond_fn=cond_fn if kind else None, device="cuda")
        with th.no_grad():
            tz = th.zeros(len(x), dtype=th.long, device="cuda")
            ps = th.sigmoid(sep(x, tz)).cpu(); lo = pair(x, tz)[:, 0].cpu()
        u8 = to_u8(x).permute(0, 2, 3, 1).cpu().numpy()
        for i, s in enumerate(a.seeds):
            name = f"{col}_seed{s:02d}.png"
            Image.fromarray(u8[i]).save(f"{a.out}/tiles/{name}")
            Image.fromarray(u8[i]).resize((256, 256), Image.NEAREST).save(f"{a.out}/tiles/{name[:-4]}_x4.png")
            rows.append({"seed": s, "column": col, "tile": name, "sep_p_cat": round(float(ps[i, 0]), 3),
                         "sep_p_dog": round(float(ps[i, 1]), 3), "pair_logit": round(float(lo[i]), 2), "call": ""})
        print(col, f"{time.time() - t0:.0f}s", flush=True)
    with open(f"{a.out}/calls.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    json.dump({"model": a.model, "separate": a.separate, "pair": a.pair, "steps": a.steps, "clip": CLIP,
               "columns": COLUMNS, "seeds": a.seeds}, open(f"{a.out}/sheet.json", "w"), indent=1)
    tile, pad, cap = 128, 6, 18
    sh = Image.new("RGB", (60 + pad + len(COLUMNS) * (tile + pad), cap + pad + len(a.seeds) * (tile + pad)), "white")
    d = ImageDraw.Draw(sh)
    for j, (col, _, _) in enumerate(COLUMNS):
        d.text((60 + pad + j * (tile + pad), 4), col, fill="black")
        for i, s in enumerate(a.seeds):
            if j == 0:
                d.text((6, cap + pad + i * (tile + pad) + tile // 2), f"seed {s}", fill="black")
            sh.paste(Image.open(f"{a.out}/tiles/{col}_seed{s:02d}.png").resize((tile, tile), Image.NEAREST),
                     (60 + pad + j * (tile + pad), cap + pad + i * (tile + pad)))
    sh.save(f"{a.out}/{a.sheet_name}")
    print("DONE", len(rows), "tiles", round(time.time() - t0), "s")


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("draw")
    q.add_argument("--model", required=True); q.add_argument("--out", required=True)
    q.add_argument("--n", type=int, default=512); q.add_argument("--batch", type=int, default=64)
    q.add_argument("--seed0", type=int, default=1000); q.add_argument("--steps", default="250")
    q = sub.add_parser("sheet")
    q.add_argument("--model", required=True); q.add_argument("--separate", required=True); q.add_argument("--pair", required=True)
    q.add_argument("--out", required=True); q.add_argument("--sheet-name", default="sheet.png")
    q.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 17))); q.add_argument("--steps", default="250")
    a = p.parse_args()
    assert th.cuda.is_available(), "no CUDA device"
    print(f"device {th.cuda.get_device_name(0)} args {vars(a)}", flush=True)
    {"draw": draw, "sheet": sheet}[a.cmd](a)


if __name__ == "__main__":
    main()
