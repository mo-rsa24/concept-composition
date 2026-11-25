#!/usr/bin/env python3
"""Loop 12 step Z11: arm 2's checker trained with a gradient penalty toward the owning expert's score (Ganz, Kawar and Elad).

Design: artifacts/notes/progress/leakage/11-gradient-trained-classifier/design.md.

The push is -strength * sigma_t * grad_x f, so it moves the noise prediction along -grad_x f. The target at a seen pair (d, c) in
noise units is t = eps_k(x_t | d, c) - eps_none(x_t), the owning expert asked with both labels minus the mean of both experts with
no labels; the penalty is 1 - cos(grad_x f(x_t; d, c), -t), so the push moves the prediction toward the owning expert.

  train   --kind penalised | zero | shuffled    one checker on the 40 training pairs (3,000 steps, batch 128, seed 31)
  held    the three checkers on the 10 held-back pairs: cosines on arm 4's held-back trajectories (lambda 0 twin printed first),
          then 100 pictures per pair each, accuracy and McNemar; the gate
  novel   the penalised push, the lambda 0 twin, control a and the random push at the penalised push's length on the 50 novel
          combinations; already-right seen pictures broken by the penalised and the random push; the bar
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
from canvas_unet import ABAR, T, per_sample_cap  # noqa: E402
from z9_data import colourise, load_mnist  # noqa: E402
from z9_push import (CAP, STRENGTH, CheckerHead, SeparateHead, expert_features, matched_push, mcnemar_one_sided, random_direction,  # noqa: E402
                     real_stats, sigma)
from z9_sample import SEEDS, Experts, fid, fid_stats, inception, judge_right, load_judge, log_wandb, sample, to_uint8  # noqa: E402
from z9_train import NONE  # noqa: E402

LAMBDA, OTHER_PAIRS, PEN_BATCH = 1.0, 3, 32


def owner(split, d, c):
    for k, h in enumerate(split["holders"]):
        if [d, c] in h:
            return k
    raise ValueError((d, c))


def target(experts, split, x, t, d, c):
    """eps_k(x_t | d, c) - eps_none(x_t) per picture, for per-picture pairs d, c (lists): one batched call per expert."""
    with torch.no_grad():
        none = torch.full((x.shape[0],), NONE, device=x.device)
        e0 = sum(m(x, t, none, none) for m in experts.models) / 2
        d_t = torch.tensor([int(v) for v in d], device=x.device); c_t = torch.tensor([int(v) for v in c], device=x.device)
        own = torch.tensor([owner(split, int(a), int(b)) for a, b in zip(d, c)], device=x.device)
        out = torch.zeros_like(x)
        for k, m in enumerate(experts.models):
            sel = own == k
            if sel.any():
                out[sel] = m(x[sel], t[sel], d_t[sel], c_t[sel])
        return out - e0


def train(args, split, dev):
    experts = Experts(args.experts, split, dev)
    images, labels = load_mnist(args.mnist, "train")
    drawn = np.load(args.z9 / "checker" / "drawn.npz")
    held = {tuple(k) for k in split["ratio_held_back"]}
    pairs = [tuple(k) for k in split["seen"] if tuple(k) not in held]
    pa = np.array(pairs)
    keep = np.array([(int(a), int(b)) not in held for a, b in zip(drawn["digits"], drawn["colours"])])
    dpos = np.flatnonzero(drawn["right"] & keep); dneg = np.flatnonzero(~drawn["right"] & keep)
    by_digit = {d: np.flatnonzero(labels == d) for d in range(10)}
    rng = np.random.default_rng(31); torch.manual_seed(31)
    perm = None
    if args.kind == "shuffled":
        prng = np.random.default_rng(32)
        while True:
            j = prng.permutation(len(pairs))
            if all(j[i] != i for i in range(len(pairs))):
                break
        perm = {pairs[i]: pairs[j[i]] for i in range(len(pairs))}
    head = CheckerHead().to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    lam = 0.0 if args.kind == "zero" else LAMBDA
    hb = torch.load(args.z9 / "arm4" / "traj-held.pt", weights_only=False)
    hsub = torch.randperm(len(hb["x"]), generator=torch.Generator().manual_seed(7))[:1000]
    hb = {k: (v[hsub] if torch.is_tensor(v) else v) for k, v in hb.items()}
    line_file = args.out / "checkers" / "zero-held-cosine.json"
    to = lambda idx: drawn["images"][idx].astype(np.float32) / 127.5 - 1
    log = []
    for step in range(1, args.steps + 1):
        n = 32
        k = pa[rng.integers(0, len(pa), n)]
        xp = colourise(images[[rng.choice(by_digit[int(v)]) for v in k[:, 0]]], k[:, 1]); dp, cp = k[:, 0], k[:, 1]
        k1 = pa[rng.integers(0, len(pa), n // 2)]
        xw1 = colourise(images[[rng.choice(by_digit[int(v)]) for v in k1[:, 0]]], (k1[:, 1] + rng.integers(1, 10, n // 2)) % 10)
        k2 = pa[rng.integers(0, len(pa), n // 2)]
        xw2 = colourise(images[[rng.choice(by_digit[int((v + rng.integers(1, 10)) % 10)]) for v in k2[:, 0]]], k2[:, 1])
        ip, ineg = rng.choice(dpos, n), rng.choice(dneg, n)
        x = np.concatenate([xp, to(ip), xw1, xw2, to(ineg)])
        d = np.concatenate([dp, drawn["digits"][ip], k1[:, 0], k2[:, 0], drawn["digits"][ineg]])
        c = np.concatenate([cp, drawn["colours"][ip], k1[:, 1], k2[:, 1], drawn["colours"][ineg]])
        y = np.concatenate([np.ones(2 * n), np.zeros(2 * n)]).astype(np.float32)
        x = torch.from_numpy(x).to(dev)
        tt = torch.randint(0, T, (len(x),), device=dev)
        a = ABAR.to(dev)[tt][:, None, None, None]
        xt = (a.sqrt() * x + (1 - a).sqrt() * torch.randn_like(x)).detach()
        dt, ct = torch.from_numpy(d.astype(np.int64)).to(dev), torch.from_numpy(c.astype(np.int64)).to(dev)
        with torch.no_grad():
            feats = expert_features(experts, xt, tt)
        loss_bce = F.binary_cross_entropy_with_logits(head(feats, tt, dt, ct), torch.from_numpy(y).to(dev))
        loss_pen = torch.zeros((), device=dev)
        if lam > 0:
            sel = torch.from_numpy(rng.choice(len(x), PEN_BATCH, replace=False)).to(dev)
            xs, ts = xt[sel], tt[sel]
            own = [(int(dt[i]), int(ct[i])) for i in sel.tolist()]
            asked = [[p] + [pairs[j] for j in rng.choice(len(pairs), OTHER_PAIRS, replace=False)] for p in own]  # own pair is always a training pair
            terms = []
            for r in range(1 + OTHER_PAIRS):
                ad = torch.tensor([q[r][0] for q in asked], device=dev); ac = torch.tensor([q[r][1] for q in asked], device=dev)
                src = [perm[q[r]] if perm else q[r] for q in asked]
                tgt = target(experts, split, xs, ts, [s_[0] for s_ in src], [s_[1] for s_ in src])
                xx = xs.clone().requires_grad_(True)
                logit = head(expert_features(experts, xx, ts), ts, ad, ac)
                g, = torch.autograd.grad(logit.sum(), xx, create_graph=True)
                cos = F.cosine_similarity(g.flatten(1), (-tgt).flatten(1), dim=1)
                tn = tgt.flatten(1).norm(dim=1)
                bins = (ts // 100)
                w = torch.stack([tn[i] / tn[bins == bins[i]].mean().clamp_min(1e-8) for i in range(len(tn))])
                terms.append((w * (1 - cos)).mean())
            loss_pen = torch.stack(terms).mean()
        loss = loss_bce + lam * loss_pen
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 250 == 0:
            log.append({"step": step, "bce": round(loss_bce.item(), 4), "penalty": round(float(loss_pen), 4)})
            if step % 500 == 0:
                head.eval()
                log[-1]["held_cosine"] = cosines(experts, split, head, hb, dev)["median_cos_target"]
                head.train()
            print(json.dumps({args.kind: log[-1]}), flush=True)
            if args.kind == "penalised" and step == args.steps // 2 and line_file.exists():
                line = json.loads(line_file.read_text())["held_cosine"]
                if not log[-1]["held_cosine"] > line:
                    (args.out / "checkers").mkdir(parents=True, exist_ok=True)
                    (args.out / "checkers" / "futility-stop.json").write_text(json.dumps(
                        {"step": step, "penalised_held_cosine": log[-1]["held_cosine"], "line_zero_twin": line}, indent=1) + "\n")
                    print(json.dumps({"FUTILITY_STOP": {"penalised": log[-1]["held_cosine"], "line": line}}), flush=True)
                    raise SystemExit(3)
    out = args.out / "checkers"; out.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.state_dict(), "kind": args.kind, "lambda": lam, "steps": args.steps, "pairs": pairs,
                "perm": {f"{a}-{b}": list(v) for (a, b), v in perm.items()} if perm else None}, out / f"{args.kind}.pt")
    (out / f"{args.kind}-log.json").write_text(json.dumps(log, indent=1) + "\n")
    if args.kind == "zero":
        head.eval()
        line_file.write_text(json.dumps({"held_cosine": cosines(experts, split, head, hb, dev)["median_cos_target"], "step": args.steps}) + "\n")


def load_head(args, dev, kind):
    h = CheckerHead().to(dev).eval()
    h.load_state_dict(torch.load(args.out / "checkers" / f"{kind}.pt", map_location=dev, weights_only=False)["head"]); h.requires_grad_(False)
    return h


def push_fn(experts, head, d, c):
    def push(x, t, eps):
        with torch.enable_grad():
            xx = x.detach().requires_grad_(True)
            dd = torch.full((x.shape[0],), d, device=x.device); cc = torch.full((x.shape[0],), c, device=x.device)
            g, = torch.autograd.grad(head(expert_features(experts, xx, t), t, dd, cc).sum(), xx)
        return -STRENGTH * sigma(t) * g
    return push


def cosines(experts, split, head, blob, dev):
    """Median cosine between the push's change to the noise prediction (along -grad f) and the target, and with r_t."""
    out_t, out_r, steps = [], [], []
    for i in range(0, len(blob["x"]), 256):
        x = blob["x"][i:i + 256].to(dev).float(); t = blob["t"][i:i + 256].to(dev); d = blob["d"][i:i + 256]; c = blob["c"][i:i + 256]
        tgt = target(experts, split, x, t, d.tolist(), c.tolist())
        with torch.enable_grad():
            xx = x.clone().requires_grad_(True)
            g, = torch.autograd.grad(head(expert_features(experts, xx, t), t, d.to(dev), c.to(dev)).sum(), xx)
        change = -g
        out_t.append(F.cosine_similarity(change.flatten(1), tgt.flatten(1), dim=1).cpu())
        out_r.append(F.cosine_similarity(change.flatten(1), blob["r"][i:i + 256].to(dev).float().flatten(1), dim=1).cpu())
        steps.append(t.cpu())
    ct, cr, st = torch.cat(out_t), torch.cat(out_r), torch.cat(steps)
    per_step = {int(s): float(ct[st == s].median()) for s in sorted(set(st.tolist()))[::10]}
    return {"median_cos_target": float(ct.median()), "median_cos_dropped_term": float(cr.median()), "per_step_sample": per_step}


def draw(experts, pairs, dev, pf_for, per=100):
    imgs, dd, cc, lens = [], [], [], {}
    for d, c in pairs:
        x, applied = sample(lambda xx, tt: experts.eps(xx, tt, d, c), SEEDS[:per], dev, push_fn=pf_for(d, c), cap=CAP)
        imgs.append(to_uint8(x)); dd += [d] * per; cc += [c] * per
        if applied is not None:
            lens[(d, c)] = applied.numpy()
    return np.concatenate(imgs), np.array(dd), np.array(cc), lens


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["train", "held", "novel"])
    ap.add_argument("--kind", choices=["penalised", "zero", "shuffled"])
    ap.add_argument("--mnist", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--experts", type=Path, required=True)
    ap.add_argument("--judge", type=Path, required=True)
    ap.add_argument("--z9", type=Path, required=True, help="Z9's output folder (drawn pictures, arm 4's trajectories, plain results)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--plain-here", action="store_true", help="redraw the plain product's seen pictures on this card for the already-right count (on another card type than Z9's)")
    args = ap.parse_args()
    split = json.loads(args.split.read_text()); dev = "cuda"; t0 = time.time()
    if args.stage == "train":
        return train(args, split, dev)
    experts = Experts(args.experts, split, dev); judge = load_judge(args.judge, dev)
    summary = {}
    heads = {k: load_head(args, dev, k) for k in ("zero", "penalised", "shuffled")}
    if args.stage == "held":
        blob = torch.load(args.z9 / "arm4" / "traj-held.pt", weights_only=False)
        tr = torch.load(args.z9 / "arm4" / "traj-train.pt", weights_only=False)
        sub = torch.randperm(len(tr["x"]), generator=torch.Generator().manual_seed(5))[:4000]
        tr = {k: (v[sub] if torch.is_tensor(v) else v) for k, v in tr.items()}
        for k in ("zero", "penalised", "shuffled"):
            summary[f"cosine_{k}"] = {"held_back": cosines(experts, split, heads[k], blob, dev), "training_pairs": cosines(experts, split, heads[k], tr, dev)}
            print(json.dumps({f"cosine_{k}": {"held": summary[f'cosine_{k}']['held_back']['median_cos_target'],
                                              "train": summary[f'cosine_{k}']['training_pairs']['median_cos_target']}}), flush=True)
        held = [tuple(x) for x in split["ratio_held_back"]]
        rights = {}
        for k in ("zero", "penalised", "shuffled"):
            im, d, c, _ = draw(experts, held, dev, lambda a, b, k=k: push_fn(experts, heads[k], a, b))
            rights[k] = judge_right(judge, im, d, c, dev)
            np.savez_compressed(args.out / f"held-{k}.npz", images=im, digits=d, colours=c, right=rights[k])
        v0, vs = mcnemar_one_sided(rights["penalised"], rights["zero"]), mcnemar_one_sided(rights["penalised"], rights["shuffled"])
        dcos = summary["cosine_penalised"]["held_back"]["median_cos_target"] - summary["cosine_zero"]["held_back"]["median_cos_target"]
        lead0, leads = 100 * (rights["penalised"].mean() - rights["zero"].mean()), 100 * (rights["penalised"].mean() - rights["shuffled"].mean())
        summary["gate"] = {"accuracy": {k: float(v.mean()) for k, v in rights.items()}, "cosine_lead_over_zero": dcos,
                           "against_zero": {"lead_points": lead0, "won": v0[0], "lost": v0[1], "p": v0[2]},
                           "against_shuffled": {"lead_points": leads, "won": vs[0], "lost": vs[1], "p": vs[2]},
                           "passes": bool(dcos >= 0.10 and lead0 >= 5 and v0[2] < 0.05 and leads >= 5 and vs[2] < 0.05)}
        print(json.dumps(summary["gate"]), flush=True)
    else:
        novel = [tuple(x) for x in split["novel"]]
        fnet = inception(dev); s_real = real_stats(args, split, "novel", fnet, dev)
        sep = SeparateHead().to(dev).eval()
        sep.load_state_dict(torch.load(args.z9 / "checker" / "separate-head.pt", map_location=dev, weights_only=False)["head"]); sep.requires_grad_(False)

        def sep_direction(d, c):
            def f(x, t):
                with torch.enable_grad():
                    xx = x.detach().requires_grad_(True)
                    ld, lc = sep(expert_features(experts, xx, t), t)
                    g, = torch.autograd.grad((F.log_softmax(ld, 1)[:, d] + F.log_softmax(lc, 1)[:, c]).sum(), xx)
                return -g
            return f
        res = {}
        im, d, c, lens = draw(experts, novel, dev, lambda a, b: push_fn(experts, heads["penalised"], a, b))
        res["penalised"] = (im, judge_right(judge, im, d, c, dev))
        for name, pf in (("zero", lambda a, b: push_fn(experts, heads["zero"], a, b)),
                         ("control_a", lambda a, b: matched_push(sep_direction(a, b), lens, a, b)),
                         ("random", lambda a, b: matched_push(random_direction(11_000 + 100 * a + b), lens, a, b))):
            im2, _, _, _ = draw(experts, novel, dev, pf)
            res[name] = (im2, judge_right(judge, im2, d, c, dev))
        for name, (im2, r) in res.items():
            summary[name] = {"accuracy": float(r.mean()), "fid": fid(fid_stats(fnet, im2, dev), s_real)}
            np.savez_compressed(args.out / f"novel-{name}.npz", images=im2, digits=d, colours=c, right=r)
            print(json.dumps({name: summary[name]}), flush=True)
        for other in ("zero", "control_a", "random"):
            w, l, p = mcnemar_one_sided(res["penalised"][1], res[other][1])
            summary[f"penalised_against_{other}"] = {"lead_points": 100 * float(res["penalised"][1].mean() - res[other][1].mean()), "won": w, "lost": l, "p": p}
        seen = [tuple(x) for x in split["seen"]]
        if args.plain_here:  # the already-right baseline drawn on the same card type as the pushes
            imp, dp_, cp_, _ = draw(experts, seen, dev, lambda a, b: None)
            plain_seen = judge_right(judge, imp, dp_, cp_, dev)
            summary["plain_seen_redrawn_here"] = float(plain_seen.mean())
        else:
            plain_seen = np.load(args.z9 / "plain" / "seen-right.npy")
        ims, ds, cs, lens_s = draw(experts, seen, dev, lambda a, b: push_fn(experts, heads["penalised"], a, b))
        rs = judge_right(judge, ims, ds, cs, dev)
        imr, _, _, _ = draw(experts, seen, dev, lambda a, b: matched_push(random_direction(12_000 + 100 * a + b), lens_s, a, b))
        rr = judge_right(judge, imr, ds, cs, dev)
        summary["already_right_broken"] = {"penalised": int((plain_seen & ~rs).sum()), "random": int((plain_seen & ~rr).sum()), "plain_right": int(plain_seen.sum())}
        a0, aa = summary["penalised_against_zero"], summary["penalised_against_control_a"]
        summary["supports"] = bool(a0["lead_points"] >= 5 and a0["p"] < 0.05 and aa["lead_points"] >= 5 and aa["p"] < 0.05
                                   and summary["penalised"]["fid"] < summary["zero"]["fid"]
                                   and summary["already_right_broken"]["penalised"] <= summary["already_right_broken"]["random"] + 50)
        print(json.dumps({k: v for k, v in summary.items() if k.startswith("penalised_against") or k in ("already_right_broken", "supports")}), flush=True)
    summary["seconds"] = round(time.time() - t0, 1)
    (args.out / f"{args.stage}.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    flat = {f"{k}/{kk}": vv for k, v in summary.items() if isinstance(v, dict) for kk, vv in v.items() if isinstance(vv, (int, float, bool))}
    print(json.dumps({"wandb_run": log_wandb(f"loop-12-z11-{args.stage}", {"stage": args.stage}, flat)}), flush=True)


if __name__ == "__main__":
    main()
