"""Capture the plain 7 x 8 product's trajectory for one seed at every share (training 1), for r5.

The loop is sample.py's draw() line for line (same start noise, same step noise seeded by k alone over
the full 256-seed batch, same bf16 autocast on CUDA), with torch.compile as sample.py when --compile is given. It records, for the chosen
seed, the canvas at k = 200, 150, 100, 50, 25 and the finished canvas, and the three noise estimates at
k = 100. The finished canvas is compared with the stored sample for that seed.

The seed: the lowest seed whose call in the share-0 model (dial 0, training 1, 7 x 8 plain) is that
model's most common call, read from mnist_stage_numbers.json.

    /home/molef/micromamba/envs/sdxl-local/bin/python -I make_r5_trajectories.py      (needs the GPU)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
MN = REPO / "poe_repair/experiments/inclusive_captions/mnist"
sys.path.insert(0, str(MN))
import sample as sm
from canvases import BLANK, H, W
from model import T, UNet, make_abar

N = json.load(open(HERE / "mnist_stage_numbers.json"))
SEED = N["cells"]["0|1|7 x 8 plain"]["tile seed"]
LEVELS = [200, 150, 100, 50, 25]
DIALS = ["0", "0.1", "0.3", "0.6", "1"]
dev = "cuda"


@torch.no_grad()
def capture(net, abar):
    labels, n = [7, 8], sm.N_SEEDS
    m = len(labels)
    labs = torch.tensor([BLANK] + labels, device=dev).repeat_interleave(n)
    pieces = {}

    def eps(x, k, keep=False):
        kk = torch.full((n * (m + 1),), k, device=dev, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = net(x.repeat(m + 1, 1, 1, 1), kk, labs).float()
        e = e.reshape(m + 1, n, 1, H, W)
        if keep:
            pieces.update({"eps blank": e[0, SEED, 0].cpu().numpy(), "eps 7": e[1, SEED, 0].cpu().numpy(),
                           "eps 8": e[2, SEED, 0].cpu().numpy()})
        return e[0] + sm.GUIDE * (e[1:] - e[0]).sum(0)

    x = sm.start_noise(dev)
    traj = {}
    for k in range(T, 0, -1):
        if k in LEVELS:
            traj[k] = x[SEED, 0].float().cpu().numpy().copy()
        a_t, a_p = abar[k], abar[k - 1]
        al = a_t / a_p
        be = 1 - al
        mean = (x - be / (1 - a_t).sqrt() * eps(x, k, keep=(k == 100))) / al.sqrt()
        x = mean + (((be * (1 - a_p) / (1 - a_t)).sqrt() * sm.noise_like(x, 1_000_000 + k)) if k > 1 else 0.0)
    traj[0] = x[SEED, 0].float().cpu().numpy()
    final = ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)[:, 0].cpu().numpy()
    return traj, pieces, final


def u8(a):
    return np.clip((np.clip(a, -1, 1) + 1) * 127.5, 0, 255).round().astype(np.uint8)


def main():
    abar = make_abar(dev)
    out, arrays = {"seed": SEED, "rule": "lowest seed whose share-0 call (dial 0, training 1, 7 x 8 plain) is that "
                                         "model's most common call", "levels": LEVELS + [0], "by dial": {}}, {}
    s0 = sm.start_noise("cpu")
    out["start noise identical across models"] = "start_noise(i) uses torch.Generator().manual_seed(i); " \
                                                 "step noise uses manual_seed(1_000_000 + k) over the 256 batch"
    for d in DIALS:
        ck = torch.load(REPO / f"data/mnist/checkpoints/dial-{d}-seed-1.pt", map_location=dev, weights_only=False)
        net = UNet(ck["width"]).to(dev).eval(); net.load_state_dict(ck["state"])
        if "--compile" in sys.argv:
            net = torch.compile(net, mode="max-autotune-no-cudagraphs")      # as sample.py does
        traj, pc, final = capture(net, abar)
        stored = np.load(REPO / f"data/mnist/samples/dial-{d}-seed-1.npz")["7 x 8 plain"]
        diff = np.abs(final.astype(int) - stored.astype(int))
        eb, e7, e8 = pc["eps blank"], pc["eps 7"], pc["eps 8"]
        d7, d8 = e7 - eb, e8 - eb
        push = sm.GUIDE * (d7 + d8)
        out["by dial"][d] = {
            "start noise equals seed's start_noise": bool(np.allclose(traj[200], s0[SEED, 0].numpy())),
            "this seed: largest pixel difference from the stored sample": int(diff[SEED].max()),
            "this seed: pixels that differ": int((diff[SEED] > 0).sum()),
            "all 256 seeds: largest pixel difference": int(diff.max()),
            "at step 100": {"|eps blank|": round(float(np.linalg.norm(eb)), 2), "|eps 7 - eps blank|": round(float(np.linalg.norm(d7)), 3),
                            "|eps 8 - eps blank|": round(float(np.linalg.norm(d8)), 3),
                            "cosine": round(float((d7 * d8).sum() / np.linalg.norm(d7) / np.linalg.norm(d8)), 3),
                            "|3 (sum of the two differences)|": round(float(np.linalg.norm(push)), 3)}}
        for k, a in traj.items():
            arrays[f"{d}/k{k}"] = u8(a)
        arrays[f"{d}/final stored"] = stored[SEED]
        for name, a in (("d7", d7), ("d8", d8)):
            arrays[f"{d}/{name}"] = a.astype(np.float32)
        print(d, out["by dial"][d], flush=True)
    np.savez_compressed(HERE / "r5_trajectories.npz", **arrays)
    json.dump(out, open(HERE / "r5_trajectories.json", "w"), indent=1)
    print("wrote r5_trajectories.json")


if __name__ == "__main__":
    main()
