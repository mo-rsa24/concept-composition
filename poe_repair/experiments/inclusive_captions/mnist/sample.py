"""Draw the MNIST stage's eight rows from every trained model, with the toy's sampler.

The sampler is the toy caption model's (toy_caption_dial_ldm.py and toy_caption_corrector.py):
reverse diffusion over the 200-step cosine schedule, guidance g = 3,

    eps = eps_blank + g * sum over labels of (eps_label - eps_blank)

and, on corrected rows, 20 Langevin steps after every reverse step at the new level j >= 1, each

    x <- x + eta * s + sqrt(2 * eta) * z,   s = -eps(x, j) / sqrt(1 - abar_j),   eta = 0.05 * (1 - abar_j)

Seeds 0 to 255. Seed i's starting noise comes from torch.Generator().manual_seed(i); the noise at reverse
step k and at corrector step m of level k comes from generators seeded by k and m alone, over the same
256-seed batch. So every row and every model starts from the same noise and sees the same noise.

Writes data/mnist/samples/dial-<d>-seed-<s>.npz: one uint8 array [256, 32, 64] per row, pixel 0 to 255.

Usage (from the repo root):
    python -I poe_repair/experiments/inclusive_captions/mnist/sample.py            # every checkpoint
    python -I poe_repair/experiments/inclusive_captions/mnist/sample.py --ckpt <path>
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canvases import BLANK, H, W, joint_id  # noqa: E402
from model import T, UNet, make_abar  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
CKPTS = REPO / "data" / "mnist" / "checkpoints"
OUT = REPO / "data" / "mnist" / "samples"
N_SEEDS, GUIDE, CORR_STEPS, CORR_SIZE = 256, 3.0, 20, 0.05

ROWS = [  # name, labels combined, corrector steps
    ("7 alone", [7], 0),
    ("8 alone", [8], 0),
    ("4 alone", [4], 0),
    ("7 and 8", [joint_id(7, 8)], 0),
    ("7 x 8 plain", [7, 8], 0),
    ("7 x 8 corrected", [7, 8], CORR_STEPS),
    ("4 x 7 plain", [4, 7], 0),
    ("4 x 7 corrected", [4, 7], CORR_STEPS),
]


def start_noise(dev):
    return torch.stack([torch.randn(1, H, W, generator=torch.Generator().manual_seed(i))
                        for i in range(N_SEEDS)]).to(dev)


def noise_like(x, seed):
    return torch.randn(x.shape, device=x.device, generator=torch.Generator(device=x.device).manual_seed(seed))


@torch.no_grad()
def draw(net, labels, steps, abar, dev):
    n, m = N_SEEDS, len(labels)
    labs = torch.tensor([BLANK] + labels, device=dev).repeat_interleave(n)

    def eps(x, k):
        kk = torch.full((n * (m + 1),), k, device=dev, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = net(x.repeat(m + 1, 1, 1, 1), kk, labs).float()
        e = e.reshape(m + 1, n, 1, H, W)
        return e[0] + GUIDE * (e[1:] - e[0]).sum(0)

    x = start_noise(dev)
    for k in range(T, 0, -1):
        a_t, a_p = abar[k], abar[k - 1]
        al = a_t / a_p
        be = 1 - al
        mean = (x - be / (1 - a_t).sqrt() * eps(x, k)) / al.sqrt()
        x = mean + (((be * (1 - a_p) / (1 - a_t)).sqrt() * noise_like(x, 1_000_000 + k)) if k > 1 else 0.0)
        j = k - 1
        if steps and j >= 1:
            eta = CORR_SIZE * (1 - abar[j])
            for s in range(steps):
                score = -eps(x, j) / (1 - abar[j]).sqrt()
                x = x + eta * score + (2 * eta).sqrt() * noise_like(x, 2_000_000 + 100 * k + s)
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)[:, 0].cpu().numpy()


def sample_checkpoint(path, out_dir=OUT, rows=None, dev="cuda"):
    out = out_dir / path.name.replace(".pt", ".npz")
    if out.exists():
        print(f"skip {out.name}: already drawn", flush=True)
        return
    ck = torch.load(path, map_location=dev)
    net = UNet(ck["width"]).to(dev).eval()
    net.load_state_dict(ck["state"])
    net = torch.compile(net, mode="max-autotune-no-cudagraphs")
    abar = make_abar(dev)
    arrays, t0 = {}, time.time()
    for name, labels, steps in [r for r in ROWS if rows is None or r[0] in rows]:
        t1 = time.time()
        arrays[name] = draw(net, labels, steps, abar, dev)
        print(f"  {path.stem} · {name}: {len(arrays[name])} pictures in {time.time() - t1:.0f}s", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    shown = out.resolve().relative_to(REPO) if out.resolve().is_relative_to(REPO) else out.resolve()
    print(f"wrote {shown} · {len(ROWS)} rows x {N_SEEDS} pictures · {time.time() - t0:.0f}s",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--rows", nargs="+", help='only these rows, e.g. --rows "7 alone" "7 x 8 plain"')
    ap.add_argument("--guide", type=float, default=GUIDE,
                    help="guidance strength g in eps_blank + g * sum(eps_label - eps_blank); 1 is the exact Bayes product")
    a = ap.parse_args()
    globals()["GUIDE"] = a.guide
    print(f"guidance {GUIDE:g}", flush=True)
    paths = [a.ckpt] if a.ckpt else sorted(CKPTS.glob("dial-*-seed-*.pt"))
    for p in paths:
        sample_checkpoint(p, a.out, a.rows)


if __name__ == "__main__":
    main()
