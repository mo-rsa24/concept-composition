#!/usr/bin/env python3
"""Loop 12 step Z3: CO3, plus one corrector pass per step at sampler steps 15 to 30, with whole-canvas or per-region weights.

Design: artifacts/notes/progress/leakage/03-one-fix/design.md. Bars: z3_bars.py.

The added pass follows the authors' corrector (attn_utils.py co3_corrector, lines 236 to 257): four predictions [empty, joint,
A, B] on the latent, clean-picture guesses with guidance lambda, a weighted sum with the joint's weight 2, the code's
largest-entry rescale r, the empty prompt's noise added back, and the rescale to the input latent's length. Row "whole" uses
CO3's equation-15 weights everywhere; row "boxes" sets, inside A's box, A's weight 0 and B's -1, and inside B's box, A's -1
and B's 0, with equation-15 weights where the boxes overlap and outside both.

  python z3_regions.py --splits <json> --boxes <boxes.json> --out <dir> --rows whole,boxes [--pairs a,b] [--seeds 6001]
  python z3_regions.py ... --control      the empty-box check and the step-15 check, on the first pair and seed
"""
import argparse
import json
import sys
import time
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
from z3_bars import STEPS  # noqa: E402

BETA = 0.9
GRID = 128
BORROWING = ["a_cheetah__x__a_cougar", "a_duck__x__a_goose", "a_puma__x__a_condor",
             "a_boar__x__a_pheasant", "a_bobcat__x__a_quail", "a_seal__x__a_gull"]


def box_mask(box, device):
    m = torch.zeros(GRID, GRID, device=device)
    if box:
        x0, y0, x1, y1 = [int(round(v / 8)) for v in box]
        m[max(y0, 0):min(y1, GRID), max(x0, 0):min(x1, GRID)] = 1
    return m


class Corrector:
    """One added corrector pass per step at STEPS; row is None, 'whole' or 'boxes'."""

    def __init__(self, model):
        self.model, self.row, self.boxes, self.log, self.entry = model, None, (None, None), [], {}

    @torch.no_grad()
    def pass_(self, x, t):
        m, lc = self.model, self.model.latentCorrector
        emb, pool = m.text_embeds[0][:4], m.text_embeds[1][:4]
        added = {"time_ids": m.add_time_ids.repeat(4, 1), "text_embeds": pool}
        eps = m.unet(x.repeat(4, 1, 1, 1), t, encoder_hidden_states=emb, added_cond_kwargs=added)["sample"].float()
        eu, em, ea, eb = eps[0:1], eps[1:2], eps[2:3], eps[3:4]
        xf, at, lam = x.float(), float(m.alpha(t)), float(lc.lmda)
        s = (1 - at) ** 0.5
        dists = torch.tensor([(ea - em).norm().item(), (eb - em).norm().item()])
        w = lc._get_concept_weights_from_dists(dists, method="exp", beta=BETA)
        guess = lambda e: xf - s * (eu + lam * (e - eu))
        wa = torch.full((1, 1, GRID, GRID), float(w[0]), device=x.device)
        wb = torch.full((1, 1, GRID, GRID), float(w[1]), device=x.device)
        if self.row == "boxes":
            ma, mb = box_mask(self.boxes[0], x.device), box_mask(self.boxes[1], x.device)
            only_a, only_b = (ma > 0) & (mb == 0), (mb > 0) & (ma == 0)
            wa[0, 0][only_a], wb[0, 0][only_a] = 0.0, -1.0
            wa[0, 0][only_b], wb[0, 0][only_b] = -1.0, 0.0
        composed = 2.0 * guess(em) + wa * guess(ea) + wb * guess(eb)
        r = guess(em).max()
        composed = composed / composed.max() * r
        x2 = composed + s * eu
        x2 = x2 * (xf.norm() / x2.norm())
        self.log.append({"t": int(t), "dists": [round(float(d), 4) for d in dists], "w": [round(float(v), 4) for v in w]})
        return x2.to(x.dtype)


def attach(model):
    corr = Corrector(model)
    original = model.denoise_step

    def denoise_step(x, t, step):
        if step == STEPS[0]:
            corr.entry[int(step)] = x.detach().float().cpu()
        if corr.row is not None and step in STEPS:
            x = corr.pass_(x, t)
        return original(x, t, step)

    model.denoise_step = denoise_step
    return corr


def render(model, corr, seed, row, boxes):
    corr.row, corr.boxes, corr.log, corr.entry = row, boxes, [], {}
    seed_everything(seed)
    begin = time.time()
    image = model.sample_loop(start_noise(model, seed))[0]
    return image, time.time() - begin


def log_wandb(name, config, summary):
    try:
        import wandb
        run = wandb.init(project="poe-repair-animals-compose", entity="prime_lab", name=name, group="loop-12-z3",
                         config=config, settings=wandb.Settings(init_timeout=60))
        run.summary.update(summary)
        run_id = run.id
        run.finish()
        return run_id
    except Exception as error:  # the pictures are on disk either way
        print(f"W&B logging failed: {error}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--boxes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", default="whole,boxes")
    ap.add_argument("--pairs", default=",".join(BORROWING))
    ap.add_argument("--seeds", default="")
    ap.add_argument("--rung", default="z3")
    ap.add_argument("--control", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("torch sees no CUDA card")
    split = json.loads(args.splits.read_text())
    pairs = args.pairs.split(",")
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else split["tuning_seeds"]
    assert set(pairs) <= set(BORROWING) and set(seeds) <= set(split["tuning_seeds"]), "the six borrowing tuning pairs only"
    boxes = json.loads(args.boxes.read_text())
    prompts = {p: joint_prompt(p) for p in pairs}
    assert all(is_allowed(p) for p in prompts.values()), "the subject guard refuses a prompt"
    parser_nlp = load_parser()
    model, config = load(prompts[pairs[0]], parser_nlp, args.out / "_config")
    model.unet.requires_grad_(False)
    corr = attach(model)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.control:
        pair, seed = pairs[0], seeds[0]
        assert set_prompt(model, config, prompts[pair], parser_nlp) == 2
        render(model, corr, seed, None, (None, None)); plain15 = corr.entry[STEPS[0]]
        img_w, _ = render(model, corr, seed, "whole", (None, None)); whole15 = corr.entry[STEPS[0]]
        b = boxes[pair][str(seed)]
        # empty boxes: the region row must reproduce the whole-canvas row
        img_e, _ = render(model, corr, seed, "boxes", (None, None))
        diff_img = float(np.abs(np.asarray(img_w, np.int16) - np.asarray(img_e, np.int16)).max())
        result = {"pair": pair, "seed": seed, "device": torch.cuda.get_device_name(0),
                  "step15_plain_vs_whole_max_abs": float((plain15 - whole15).abs().max()),
                  "empty_boxes_vs_whole_max_pixel_diff": diff_img, "boxes_for_seed": b}
        result["passes"] = result["step15_plain_vs_whole_max_abs"] < 1e-5 and diff_img == 0.0
        (args.out / "control.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result), flush=True)
        raise SystemExit(0 if result["passes"] else 1)

    done, seconds = 0, []
    for pair in pairs:
        assert set_prompt(model, config, prompts[pair], parser_nlp) == 2
        for seed in seeds:
            b = boxes[pair][str(seed)]
            for row in args.rows.split(","):
                folder = args.out / row / pair
                path, meta = folder / f"seed-{seed}.png", folder / f"seed-{seed}.json"
                if path.exists() and meta.exists():
                    continue
                img, sec = render(model, corr, seed, row, (b.get("A"), b.get("B")) if row == "boxes" else (None, None))
                folder.mkdir(parents=True, exist_ok=True)
                img.save(path)
                meta.write_text(json.dumps({"row": row, "pair": pair, "seed": seed, "seconds": round(sec, 2), "boxes": b,
                                            "steps": list(STEPS), "beta": BETA, "passes": corr.log}) + "\n")
                seconds.append(sec); done += 1
                print(f"{done} {row} {pair} seed={seed} {sec:.1f}s", flush=True)
    summary = {"rendered": done, "median_seconds": sorted(seconds)[len(seconds) // 2] if seconds else None}
    summary["wandb_run"] = log_wandb(f"loop-12-{args.rung}", {"pairs": pairs, "seeds": seeds, "rows": args.rows, "steps": list(STEPS)}, summary)
    (args.out / f"summary-{args.rows.replace(',', '-')}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
