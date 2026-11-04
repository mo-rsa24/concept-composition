"""Draw from one trained model: each single attribute, the joint prompt, and the product of the two
single attributes, then score every draw with the classifier.

The product at guidance g is e_empty + g * ((e_male - e_empty) + (e_blond - e_empty)); at g = 1 that is
e_male + e_blond - e_empty, the composition with the shared prior divided out once. The joint prompt
at guidance g is e_empty + g * (e_joint - e_empty). DDIM, 100 steps, eta 0.

THE BAR, FIXED BEFORE THE FIRST RUN (the model with 0 blond men)
  support       the product puts >= 20% of 1,000 draws on male and blond at guidance 1 or 3, while the
                joint prompt is under 5% at the same guidance
  null          the product is under 5% at both guidance values
  inconclusive  anything else
  validity      each single attribute at guidance 3 must draw its attribute on >= 90% of 1,000
                (male expert: male; blond expert: blond), or the run says nothing

    python sample.py --run runs/bm0 --n 1000
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from diffusers import DDIMScheduler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS, RES, CONDITIONS
from train_diffusion import AttrUNet
import classifier

DEV = "cuda"


def attrs(name, n):
    return torch.tensor([CONDITIONS[name]] * n, device=DEV)


@torch.no_grad()
def draw(net, combo, g, n, seed, steps=100):
    sch = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon",
                        clip_sample=True)
    sch.set_timesteps(steps)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(n, 3, RES, RES, device=DEV, generator=gen)
    names = ["empty"] + combo
    for t in sch.timesteps:
        tt = torch.full((n,), int(t), device=DEV)
        xin = torch.cat([x] * len(names)); ain = torch.cat([attrs(c, n) for c in names])
        with torch.autocast("cuda", torch.bfloat16):
            e = net(xin, torch.cat([tt] * len(names)), ain).float().chunk(len(names))
        eps = e[0] + g * sum(ei - e[0] for ei in e[1:])
        x = sch.step(eps, t, x).prev_sample
    return x.clamp(-1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=250); ap.add_argument("--guidance", default="1,3")
    a = ap.parse_args()
    run = Path(a.run) if Path(a.run).is_absolute() else RUNS / a.run
    net = AttrUNet().to(DEV); net.load_state_dict(torch.load(run / "ema.pt", map_location=DEV)["ema"]); net.eval()
    clf = classifier.load()
    res, tiles = {}, {}
    for g in [float(v) for v in a.guidance.split(",")]:
        for label, combo in (("male expert", ["male"]), ("blond expert", ["blond"]),
                             ("joint prompt", ["joint"]), ("product of male and blond", ["male", "blond"])):
            xs = [draw(net, combo, g, a.batch, seed=1000 * b) for b in range(a.n // a.batch)]
            x = torch.cat(xs)
            pm, pb = classifier.predict(clf, x)
            r = {"male": int(pm.sum()), "blond": int(pb.sum()), "male and blond": int((pm & pb).sum()), "n": len(x)}
            res[f"{label}, guidance {g:g}"] = r
            tiles[f"{label}|{g:g}"] = ((x[:32].permute(0, 2, 3, 1).cpu().numpy() + 1) * 127.5).round().astype(np.uint8)
            print(f"g {g:g} {label}: male {r['male']}, blond {r['blond']}, male and blond {r['male and blond']} of {r['n']}",
                  flush=True)
    valid = (res["male expert, guidance 3"]["male"] >= 0.9 * a.n and res["blond expert, guidance 3"]["blond"] >= 0.9 * a.n)
    p = {g: res[f"product of male and blond, guidance {g}"]["male and blond"] / a.n for g in ("1", "3")}
    j = {g: res[f"joint prompt, guidance {g}"]["male and blond"] / a.n for g in ("1", "3")}
    verdict = ("invalid: a single attribute misses its own 90% check" if not valid else
               "support" if any(p[g] >= 0.2 and j[g] < 0.05 for g in p) else
               "null" if all(v < 0.05 for v in p.values()) else "inconclusive")
    print("VERDICT:", verdict, flush=True)
    np.savez_compressed(run / "samples.npz", **{k.replace(" ", "_"): v for k, v in tiles.items()})
    json.dump({"verdict": verdict, "results": res, "n": a.n, "sampler": "DDIM 100 steps, eta 0"},
              open(run / "samples.json", "w"), indent=1)


if __name__ == "__main__":
    main()
