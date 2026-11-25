#!/usr/bin/env python3
"""Step M13's sign check: does the push, as CO3's push code applies it, move the noise prediction along r_t or against it?

M13 measured the cosine between the kept checker's latent gradient and r_t = eps(joint) - (eps(A) + eps(B) - eps(empty)). The push code
(co3_push.py) subtracts strength * sigma * gradient from the noise prediction, so the change it makes is minus the gradient's direction.
This check runs the actual PushingUNet and Pusher on saved CO3 states, takes the noise prediction with and without the push, and reports
the cosine of (pushed minus unpushed) with r_t beside the gradient's own cosine.

  python m13_sign_check.py --head <pt> --latents-from <read folder> --out <dir> [--pairs-limit 3]
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--latents-from", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs-limit", type=int, default=3)
    args = ap.parse_args()
    import torch
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parent / "rung2"))
    sys.path.insert(0, str(HERE.parents[2]))
    from co3_common import load, load_parser
    from co3_push import Pusher, PushingUNet
    from paper2_rows import CAP, STRENGTH, WINDOW, build_checker, encode, point_checker
    from poe_repair.methods._sampling import add_time_ids
    from train_readout_pair import joint_prompt

    rows = [json.loads(l) for l in (args.latents_from / "reads.jsonl").read_text().splitlines()][: args.pairs_limit]
    model, config = load(joint_prompt(rows[0]["pair"]), load_parser(), args.out / "_config")
    unet = model.unet
    unet.requires_grad_(False)
    checker, _ = build_checker(model, unet, args.head, joint_prompt(rows[0]["pair"]))
    wrapper = PushingUNet(unet)
    abar = model.scheduler.alphas_cumprod.to(unet.device)
    cos = lambda u, v: float(torch.nn.functional.cosine_similarity(u.flatten().double(), v.flatten().double(), dim=0))
    out = []
    for r in rows:
        pair = r["pair"]
        a, b = (w.replace("_", " ") for w in pair.split("__x__"))
        seqs, pools = zip(*(encode(model, q) for q in ["", joint_prompt(pair), a, b]))
        point_checker(model, checker, joint_prompt(pair))
        saved = torch.load(args.latents_from / "latents" / pair / f"seed-{r['seed']}.pt", weights_only=False)
        for step in (0, 5, 10, 15):
            st = saved[step]
            x = st["x"].to(unet.device, torch.float16)
            t = torch.full((1,), st["t"], device=unet.device)
            cond4 = {"text_embeds": torch.cat(pools), "time_ids": add_time_ids(height=1024, width=1024, batch_size=4, device=unet.device,
                                                                                 dtype=torch.float16)}
            with torch.no_grad():
                empty, joint, ea, eb = unet(x.repeat(4, 1, 1, 1), t.repeat(4), encoder_hidden_states=torch.cat(seqs),
                                            added_cond_kwargs=cond4).sample.float()
            r_t = joint - (ea + eb - empty)
            # CO3's main call: a batch of two, unconditional then joint, through the push wrapper
            cond2 = {"text_embeds": torch.cat(pools[:2]), "time_ids": add_time_ids(height=1024, width=1024, batch_size=2,
                                                                                    device=unet.device, dtype=torch.float16)}
            sample = x.repeat(2, 1, 1, 1)
            wrapper.pusher = None
            with torch.no_grad():
                plain = wrapper(sample, t.repeat(2), encoder_hidden_states=torch.cat(seqs[:2]), added_cond_kwargs=cond2)["sample"].float()
            pusher = Pusher(checker, config.guidance_scale, STRENGTH, CAP, WINDOW, "push")
            pusher.step, pusher.alpha_bar = step, float(abar[st["t"]])
            wrapper.pusher = pusher
            pushed = wrapper(sample, t.repeat(2), encoder_hidden_states=torch.cat(seqs[:2]), added_cond_kwargs=cond2)["sample"].float()
            wrapper.pusher = None
            change = (pushed - plain)[1]                    # the joint entry; the unconditional entry moves by the same amount
            with torch.enable_grad():
                z = x.detach().float().requires_grad_(True)
                grad = torch.autograd.grad(checker.log_prob_both(z, t), z)[0][0]
            row = {"pair": pair, "seed": r["seed"], "step": step, "cos_change_with_r_t": cos(change, r_t),
                   "cos_gradient_with_r_t": cos(grad, r_t), "cos_change_with_gradient": cos(change, grad),
                   "change_norm": float(change.norm()), "pushed_minus_plain_equal_on_both_entries":
                   float(((pushed - plain)[0] - (pushed - plain)[1]).abs().max())}
            out.append(row)
            print(json.dumps(row), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sign-check.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
