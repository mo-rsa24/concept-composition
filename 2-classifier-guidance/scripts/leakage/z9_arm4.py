#!/usr/bin/env python3
"""Loop 12 step Z9, arm 4: a small model of the term the product drops, r_t = eps_k(x_t | d, c) - eps_product(x_t | d, c).

Design: artifacts/notes/progress/leakage/09-dcfm-colored-mnist/design.md, "Arm 4". Both experts stay frozen; nothing trains on
the holders' pictures.

  data    the product's own trajectories (arm 1's sampler) on the 40 training pairs and the 10 held-back pairs, seeds 2,000 to
          2,015, every step's x_t with its target r_t from the one expert that saw the pair, asked with both labels
  train   the model (digit and colour as separate inputs) and its twin trained on shuffled combinations
  held    the fit check and the gate on the 10 held-back pairs: arm 1, arm 4, the twin, 100 pictures each
  novel   the 50 novel combinations: arm 4, the twin, the ask-the-experts baseline; accuracy, McNemar, FID
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "mnist"))
from z9_push import mcnemar_one_sided, real_stats  # noqa: E402
from z9_sample import SEEDS, Experts, fid, fid_stats, inception, judge_right, load_judge, log_wandb, sample, to_uint8  # noqa: E402
from z9_train import ColourUNet  # noqa: E402

TRAJ_SEEDS = list(range(2000, 2016))


def owner(split, d, c):
    for k, h in enumerate(split["holders"]):
        if [d, c] in h:
            return k
    raise ValueError((d, c))


@torch.no_grad()
def trajectories(experts, split, pairs, dev):
    """Arm 1's sampler with eta 1 on each pair, recording x_t, t and r_t at every step."""
    xs, ts, ds, cs, rs = [], [], [], [], []
    for d, c in pairs:
        k = owner(split, d, c)

        def record(i, x, t):
            e_prod = experts.eps(x, t, d, c)
            e_k = experts.eps_one(k, x, t, d, c)
            xs.append(x.half().cpu()); ts.append(t.cpu()); rs.append((e_k - e_prod).half().cpu())
            ds.append(torch.full((x.shape[0],), d)); cs.append(torch.full((x.shape[0],), c))
        sample(lambda xx, tt: experts.eps(xx, tt, d, c), TRAJ_SEEDS, dev, record=record)
    return torch.cat(xs), torch.cat(ts), torch.cat(ds), torch.cat(cs), torch.cat(rs)


def data(args, split, dev):
    experts = Experts(args.experts, split, dev)
    held = [tuple(k) for k in split["ratio_held_back"]]
    train = [tuple(k) for k in split["seen"] if tuple(k) not in held]
    out = args.out / "arm4"; out.mkdir(parents=True, exist_ok=True)
    for name, pairs in (("train", train), ("held", held)):
        x, t, d, c, r = trajectories(experts, split, pairs, dev)
        torch.save({"x": x, "t": t, "d": d, "c": c, "r": r, "pairs": pairs}, out / f"traj-{name}.pt")
        print(json.dumps({name: int(len(x)), "pairs": len(pairs), "r_rms": float(r.float().pow(2).mean().sqrt())}), flush=True)


def train(args, split, dev, shuffled):
    blob = torch.load(args.out / "arm4" / "traj-train.pt", weights_only=False)
    x, t, d, c, r = blob["x"], blob["t"], blob["d"], blob["c"], blob["r"]
    if shuffled:  # every trajectory's target paired with another training pair's digit and colour, fixed once
        pairs = blob["pairs"]
        rng = np.random.default_rng(41)
        perm = {p: pairs[j] for p, j in zip(pairs, rng.permutation(len(pairs)))}
        while any(perm[p] == p for p in pairs):
            perm = {p: pairs[j] for p, j in zip(pairs, rng.permutation(len(pairs)))}
        new = [perm[(int(a), int(b))] for a, b in zip(d.tolist(), c.tolist())]
        d = torch.tensor([p[0] for p in new]); c = torch.tensor([p[1] for p in new])
    torch.manual_seed(42 if not shuffled else 43)
    model = ColourUNet(32).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    g = torch.Generator().manual_seed(44)
    for step in range(1, args.steps + 1):
        i = torch.randint(0, len(x), (256,), generator=g)
        pred = model(x[i].to(dev).float(), t[i].to(dev), d[i].to(dev), c[i].to(dev))
        loss = F.mse_loss(pred, r[i].to(dev).float())
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0:
            print(json.dumps({"shuffled": shuffled, "step": step, "loss": round(loss.item(), 6)}), flush=True)
    torch.save({"model": model.state_dict(), "shuffled": shuffled, "steps": args.steps}, args.out / "arm4" / ("twin.pt" if shuffled else "model.pt"))


def load(args, dev, name):
    m = ColourUNet(32).to(dev).eval()
    m.load_state_dict(torch.load(args.out / "arm4" / name, map_location=dev, weights_only=False)["model"]); m.requires_grad_(False)
    return m


def eps_with(experts, model, d, c):
    def f(x, t):
        b = x.shape[0]
        return experts.eps(x, t, d, c) + model(x, t, torch.full((b,), d, device=x.device), torch.full((b,), c, device=x.device))
    return f


def eps_ask(experts, d, c):
    def f(x, t):
        return 0.5 * (experts.eps_one(0, x, t, d, c) + experts.eps_one(1, x, t, d, c))
    return f


def draw(eps_for, pairs, dev):
    imgs, dd, cc = [], [], []
    for d, c in pairs:
        x, _ = sample(eps_for(d, c), SEEDS, dev)
        imgs.append(to_uint8(x)); dd += [d] * 100; cc += [c] * 100
    return np.concatenate(imgs), np.array(dd), np.array(cc)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["data", "train", "held", "novel"])
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--experts", type=Path, required=True)
    ap.add_argument("--judge", type=Path, required=True)
    ap.add_argument("--mnist", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=10_000)
    args = ap.parse_args()
    split = json.loads(args.split.read_text()); dev = "cuda"; t0 = time.time()
    if args.stage == "data":
        return data(args, split, dev)
    if args.stage == "train":
        train(args, split, dev, False); train(args, split, dev, True); return
    experts = Experts(args.experts, split, dev); judge = load_judge(args.judge, dev)
    model, twin = load(args, dev, "model.pt"), load(args, dev, "twin.pt")
    summary = {}
    if args.stage == "held":
        blob = torch.load(args.out / "arm4" / "traj-held.pt", weights_only=False)
        with torch.no_grad():
            num = den = 0.0
            for i in range(0, len(blob["x"]), 2000):
                pr = model(blob["x"][i:i + 2000].to(dev).float(), blob["t"][i:i + 2000].to(dev), blob["d"][i:i + 2000].to(dev), blob["c"][i:i + 2000].to(dev))
                tg = blob["r"][i:i + 2000].to(dev).float()
                num += float((pr - tg).pow(2).sum()); den += float(tg.pow(2).sum())
        explained = 1 - num / den
        held = [tuple(k) for k in split["ratio_held_back"]]
        i1, d, c = draw(lambda a, b: (lambda x, t: experts.eps(x, t, a, b)), held, dev)
        i4, _, _ = draw(lambda a, b: eps_with(experts, model, a, b), held, dev)
        it, _, _ = draw(lambda a, b: eps_with(experts, twin, a, b), held, dev)
        r1, r4, rt = (judge_right(judge, im, d, c, dev) for im in (i1, i4, it))
        v1, v2 = mcnemar_one_sided(r4, r1), mcnemar_one_sided(r4, rt)
        passes = explained >= 0.25 and 100 * (r4.mean() - r1.mean()) >= 5 and v1[2] < 0.05 and 100 * (r4.mean() - rt.mean()) >= 5 and v2[2] < 0.05
        summary["gate"] = {"explained_variance": explained, "arm_1": float(r1.mean()), "arm_4": float(r4.mean()), "twin": float(rt.mean()),
                           "against_arm_1": {"won": v1[0], "lost": v1[1], "p": v1[2]}, "against_twin": {"won": v2[0], "lost": v2[1], "p": v2[2]},
                           "passes": bool(passes)}
        np.savez_compressed(args.out / "arm4" / "held.npz", arm1=i1, arm4=i4, twin=it, digits=d, colours=c)
        print(json.dumps(summary["gate"]), flush=True)
    else:
        novel = [tuple(k) for k in split["novel"]]
        base = np.load(args.out / "plain" / "novel-right.npy")
        fnet = inception(dev); s_real = real_stats(args, split, "novel", fnet, dev)
        rights = {}
        for name, ef in (("arm_4", lambda a, b: eps_with(experts, model, a, b)), ("twin", lambda a, b: eps_with(experts, twin, a, b)),
                         ("ask_the_experts", lambda a, b: eps_ask(experts, a, b))):
            im, d, c = draw(ef, novel, dev)
            r = judge_right(judge, im, d, c, dev); rights[name] = r
            np.savez_compressed(args.out / "arm4" / f"{name}-novel.npz", images=im, digits=d, colours=c, right=r)
            summary[name] = {"accuracy": float(r.mean()), "fid": fid(fid_stats(fnet, im, dev), s_real)}
            print(json.dumps({name: summary[name]}), flush=True)
        for other, ref in (("arm_1", base), ("twin", rights["twin"]), ("ask_the_experts", rights["ask_the_experts"])):
            w, l, p = mcnemar_one_sided(rights["arm_4"], ref)
            summary[f"arm_4_against_{other}"] = {"lead_points": 100 * float(rights["arm_4"].mean() - ref.mean()), "won": w, "lost": l, "p": p}
        print(json.dumps({k: v for k, v in summary.items() if k.startswith("arm_4_against")}), flush=True)
    summary["seconds"] = round(time.time() - t0, 1)
    (args.out / "arm4" / f"{args.stage}.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    flat = {f"{k}/{kk}": vv for k, v in summary.items() if isinstance(v, dict) for kk, vv in v.items() if isinstance(vv, (int, float, bool))}
    print(json.dumps({"wandb_run": log_wandb(f"loop-12-z9-arm4-{args.stage}", {"stage": args.stage}, flat)}), flush=True)


if __name__ == "__main__":
    main()
