#!/usr/bin/env python3
"""Record every pass of CO3's resampler and corrector, for the CO3 zoom ladder's exact-plane rung.

Design: artifacts/notes/progress/leakage/co3-inside/design.md.

For each pair at one seed: CO3 as M1 rendered it, recording each corrector pass's input latent, its four noise predictions
[empty, joint, A, B], alpha-bar and the weights CO3's weight function returns; then the same seed with CO3's correction off
for the joint prompt and for each animal alone. Each pass's output is recomputed from the records with the authors'
formulas and compared with the latent the next pass received.

  python co3_inside.py --splits <json> --out <dir> --pairs a,b --seed 6001
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "rung2"))
from co3_common import load, load_parser, set_prompt, start_noise  # noqa: E402
from composers.utils_custom import seed_everything  # noqa: E402
from disallowed_subjects import is_allowed  # noqa: E402
from train_readout_pair import joint_prompt  # noqa: E402

MAX_REL_DIFF = 0.01


class Recorder(torch.nn.Module):
    """The U-Net, recording corrector passes (batch of 4) and the latent at every sampler step (batch of 2)."""

    def __init__(self, unet):
        super().__init__()
        self.inner, self.step, self.on = unet, None, True
        self.passes, self.path = [], {}

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("inner"), name)

    def forward(self, sample, timestep, *args, **kwargs):
        out = self.inner(sample, timestep, *args, **kwargs)
        if self.on and sample.shape[0] == 4:
            self.passes.append({"step": self.step, "x": sample[0].float().cpu(), "eps": out["sample"].float().cpu()})
        if self.on and sample.shape[0] == 2 and self.step not in self.path:
            self.path[self.step] = sample[0].half().cpu()
        return out


def to_png(img, path):
    from PIL import Image
    arr = (img[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def recompute(p, nxt, at, lam, kind, w):
    """The authors' next latent from one pass's records (attn_utils.py 194-198 resampler, 251-257 corrector)."""
    x, (eu, em, e1, e2) = p["x"], p["eps"]
    s = (1 - at) ** 0.5
    if kind == "resample":
        comp = lam * (w[0] * em + w[1] * e1 + w[2] * e2)
        x2 = -s * comp + s * eu
    else:
        guess = lambda e: x - s * (eu + lam * (e - eu))
        comp = w[0] * guess(em) + w[1] * guess(e1) + w[2] * guess(e2)
        r = guess(em).max()
        comp = comp / comp.max() * r
        x2 = comp + s * eu
        x2 = x2 * (x.norm() / x2.norm())
    return float((x2 - nxt).norm() / nxt.norm())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--seed", type=int, default=6001)
    args = ap.parse_args()
    split = json.loads(args.splits.read_text())
    pairs = args.pairs.split(",")
    assert set(pairs) <= set(split["tuning_pairs"]) and args.seed in split["tuning_seeds"], "tuning data only"
    prompts = {p: joint_prompt(p) for p in pairs}
    assert all(is_allowed(p) for p in prompts.values()), "the subject guard refuses a prompt"

    parser_nlp = load_parser()
    model, config = load(prompts[pairs[0]], parser_nlp, args.out / "_config")
    model.unet.requires_grad_(False)
    rec = Recorder(model.unet)
    model.unet = rec
    original_step, original_weights = model.denoise_step, model.latentCorrector._get_concept_weights_from_dists
    weights_log, alphas = [], {}

    def denoise_step(x, t, step):
        rec.step = int(step)
        alphas[int(step)] = float(model.alpha(t))
        return original_step(x, t, step)

    def weights(dists, **kw):
        w = original_weights(dists, **kw)
        weights_log.append({"step": rec.step, "dists": [float(d) for d in dists], "w": [float(v) for v in w], "beta": kw.get("beta")})
        return w

    model.denoise_step = denoise_step
    model.latentCorrector._get_concept_weights_from_dists = weights
    lam, n_resample = float(model.latentCorrector.lmda), int(config.num_resampling_steps)

    for pair in pairs:
        folder = args.out / pair / f"seed-{args.seed}"
        folder.mkdir(parents=True, exist_ok=True)
        assert set_prompt(model, config, prompts[pair], parser_nlp) == 2
        corrected = list(model.config.latent_corrector_ts)
        rec.passes, rec.path, weights_log[:] = [], {}, []
        seed_everything(args.seed)
        model.sample_loop(start_noise(model, args.seed))[0].save(folder / "co3.png")
        passes, wl, co3_path = list(rec.passes), list(weights_log), dict(rec.path)

        # the control: each pass's output against the next pass's input (or the main call's input after the last pass)
        checks = []
        for i, p in enumerate(passes):
            nxt = passes[i + 1]["x"] if i + 1 < len(passes) and passes[i + 1]["step"] == p["step"] else co3_path[p["step"]].float()
            kind = "resample" if p["step"] < n_resample else "correct"
            w = [1.0 if kind == "resample" else 2.0] + wl[i]["w"][:2]
            checks.append({"pass": i, "step": p["step"], "kind": kind, "weights_used": w, "n_weights_returned": len(wl[i]["w"]),
                           "rel_diff": recompute(p, nxt, alphas[p["step"]], lam, kind, w)})

        # decoded clean-picture guesses at each step's first and last pass
        for i, p in enumerate(passes):
            first = i == 0 or passes[i - 1]["step"] != p["step"]
            last = i + 1 == len(passes) or passes[i + 1]["step"] != p["step"]
            if not (first or last):
                continue
            s = (1 - alphas[p["step"]]) ** 0.5
            x, (eu, em, e1, e2) = p["x"], p["eps"]
            for name, e in (("empty", eu), ("joint", eu + lam * (em - eu)), ("a", eu + lam * (e1 - eu)), ("b", eu + lam * (e2 - eu))):
                guess = (x - s * e) / alphas[p["step"]] ** 0.5
                to_png(model.decode_latent(guess[None].to(model.vae.device, model.vae.dtype)),
                       folder / f"guess-step{p['step']}-{'first' if first else 'last'}-{name}.png")

        np.savez_compressed(folder / "passes.npz", x=np.stack([p["x"].half().numpy() for p in passes]),
                            eps=np.stack([p["eps"].half().numpy() for p in passes]), step=np.array([p["step"] for p in passes]),
                            co3_path=np.stack([co3_path[k].numpy() for k in sorted(co3_path)]))

        # the same seed with the correction off: the joint prompt, then each animal alone
        model.config.latent_corrector_ts = []
        plain = {}
        for name, prompt in (("joint", prompts[pair]),) + tuple(zip(("a", "b"), model.prompts_single[:2])):
            set_prompt(model, config, prompt, parser_nlp)
            model.config.latent_corrector_ts = []
            rec.path = {}
            seed_everything(args.seed)
            model.sample_loop(start_noise(model, args.seed))[0].save(folder / f"plain-{name}.png")
            if name == "joint":
                plain = dict(rec.path)
        np.save(folder / "plain-joint-path.npy", np.stack([plain[k].numpy() for k in sorted(plain)]))
        set_prompt(model, config, prompts[pair], parser_nlp)
        model.config.latent_corrector_ts = corrected

        worst = max(c["rel_diff"] for c in checks)
        meta = {"pair": pair, "seed": args.seed, "prompt": prompts[pair], "single_prompts": list(model.prompts_single[:2]),
                "lambda": lam, "alphas": alphas, "corrected_timesteps": [int(t) for t in corrected], "passes": len(passes),
                "weights_log": wl, "checks": checks, "worst_rel_diff": worst, "control_passes": worst < MAX_REL_DIFF,
                "device": torch.cuda.get_device_name(0)}
        (folder / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        print(f"{pair} seed={args.seed} passes={len(passes)} worst_rel_diff={worst:.2e} "
              f"weights_returned={sorted({c['n_weights_returned'] for c in checks})}", flush=True)


if __name__ == "__main__":
    main()
