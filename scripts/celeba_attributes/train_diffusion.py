"""One attribute-conditioned diffusion model on CelebA at 64 x 64, with a chosen number of blond men kept.

Every training face carries both attributes; each is dropped to "unspecified" independently with
probability 0.1, so the one model also gives the male expert, the blond expert and the empty prompt.
Epsilon prediction, cosine schedule, 1000 steps, EMA 0.9995.

    python train_diffusion.py --blond-men 0 --steps 150000 --out runs/bm0
"""
import argparse, json, sys, time, copy
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from diffusers import UNet2DModel, DDPMScheduler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARRAYS, RUNS, RES, ABSENT, PRESENT, UNSPEC

DEV = "cuda"


class AttrUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet2DModel(sample_size=RES, in_channels=3, out_channels=3, layers_per_block=2,
                                block_out_channels=(128, 256, 256, 256),
                                down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
                                up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
                                class_embed_type="identity")
        self.male = nn.Embedding(3, 512); self.blond = nn.Embedding(3, 512)

    def forward(self, x, t, attrs):
        return self.unet(x, t, class_labels=self.male(attrs[:, 0]) + self.blond(attrs[:, 1])).sample


def scheduler():
    return DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blond-men", type=int, required=True, help="how many of the 1,387 training blond men to keep")
    ap.add_argument("--steps", type=int, default=150000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-every", type=int, default=25000)
    a = ap.parse_args()
    out = Path(a.out) if Path(a.out).is_absolute() else RUNS / a.out
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)

    d = np.load(ARRAYS / "train.npz")
    male, blond = d["male"], d["blond"]
    bm = np.flatnonzero(male & blond)
    keep_bm = rng.choice(bm, size=min(a.blond_men, len(bm)), replace=False)
    keep = np.sort(np.concatenate([np.flatnonzero(~(male & blond)), keep_bm]))
    x = torch.from_numpy(d["images"][keep])                     # uint8, stays on the CPU
    attrs = torch.tensor(np.stack([np.where(male[keep], PRESENT, ABSENT),
                                   np.where(blond[keep], PRESENT, ABSENT)], 1), dtype=torch.long)
    info = {"blond men kept": int(len(keep_bm)), "faces": int(len(keep)),
            "blond men share": float(len(keep_bm) / len(keep)),
            "share of blond faces that are men": float(len(keep_bm) / blond[keep].sum()), **vars(a)}
    json.dump(info, open(out / "data.json", "w"), indent=1)
    print(json.dumps(info), flush=True)

    net = AttrUNet().to(DEV); ema = copy.deepcopy(net).eval().requires_grad_(False)
    print(f"parameters {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=0.0)
    sch = scheduler(); abar = sch.alphas_cumprod.to(DEV)
    t0 = time.time(); log = open(out / "loss.log", "a")
    for s in range(1, a.steps + 1):
        i = torch.randint(len(x), (a.batch,))
        xb = x[i].to(DEV, non_blocking=True).permute(0, 3, 1, 2).float().div(127.5).sub(1)
        flip = torch.rand(a.batch, device=DEV) < 0.5
        xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)
        ab = attrs[i].to(DEV)
        ab = torch.where(torch.rand(ab.shape, device=DEV) < a.drop, torch.full_like(ab, UNSPEC), ab)
        t = torch.randint(0, 1000, (a.batch,), device=DEV)
        n = torch.randn_like(xb)
        at = abar[t][:, None, None, None]
        xt = at.sqrt() * xb + (1 - at).sqrt() * n
        with torch.autocast("cuda", torch.bfloat16):
            loss = ((net(xt, t, ab).float() - n) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        with torch.no_grad():
            dec = min(0.9995, (1 + s) / (10 + s))
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pn, 1 - dec)
        if s % 500 == 0:
            msg = f"step {s} loss {loss.item():.4f} {time.time() - t0:.0f}s {s / (time.time() - t0):.2f} it/s"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
        if s % a.save_every == 0 or s == a.steps:
            torch.save({"ema": ema.state_dict(), "step": s}, out / "ema.pt")
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "step": s}, out / "state.pt")
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
