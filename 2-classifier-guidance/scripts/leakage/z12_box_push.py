#!/usr/bin/env python3
"""Loop 12 step Z12, stage A: CO3 alone against CO3 with the box push, each animal's own prediction inside its box.

Design: artifacts/notes/progress/leakage/12-aligned-classifier-on-co3/design.md.

  boxes    --images <root> --pairs ... --seeds ... --out <boxes.json>
           Grounding DINO places the bodies (queried with the pair's two nouns), the two largest kept boxes are named by
           zero-shot CLIP ViT-B/32 on each padded crop, the names assigned by the permutation with the higher summed log-probability.
  lengths  --boxes <boxes.json> --out <dir>      CO3 alone on seeds 6001 and 6002 with T measured, not applied; writes lengths.json
  control  --boxes <boxes.json> --lengths <json> --out <dir>   empty boxes give CO3 alone byte for byte; the latent entering step 5
  render   --rows co3|box --boxes <json> --lengths <json> --out <dir> [--shard i --num-shards n]

T at step s in 5..29, in noise units against CO3's guided prediction e = uncond + g (cond - uncond):
  T = M_A (eps_A - e) + M_B (eps_B - e),   eps_A = eps_empty + lambda (eps(x | "a duck") - eps_empty)
with M_A the latent cells inside A's box and outside B's. The push the wrapper subtracts is p = -L_s T / |T|, so the prediction moves by
L_s along T, capped at CAP times |e|. Empty boxes give T = 0 and no push.
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
sys.path.insert(0, str(HERE.parent / "method_search"))
sys.path.insert(0, str(HERE.parent / "rung2"))

WINDOW = (5, 30)  # sampler steps 5 to 29
CAP = 0.4
GRID = 128
TRAIN_SEEDS = (6001, 6002)
TEST_SEEDS = tuple(range(6017, 6025))
CLIP_ID = "openai/clip-vit-base-patch32"
TEMPLATES = ("a photo of a {}", "a {}", "a close-up photo of a {}", "a photo of one {}")
BOX_PAD = 0.08
DET_CONF = 0.30
MIN_SIDE = 0.10  # a kept box's longer side, as a fraction of the picture's


def nouns(pair):
    a, b = pair.split("__x__")
    strip = lambda s: s.split("_", 1)[1].replace("_", " ")
    return strip(a), strip(b)


def splits(path):
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------- boxes
def boxes_stage(args):
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    from poe_repair.experiments.residual_between_mono_and_poe import metrics as vmetrics
    dev = torch.device("cuda")
    clip = CLIPModel.from_pretrained(CLIP_ID).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_ID)
    pairs = args.pairs.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    out = {"how": ("Grounding DINO-Tiny (poe_repair metrics.detect_boxes) queried once per noun of the pair, detections at confidence "
                   f">= {DET_CONF} and longer side >= {MIN_SIDE} of the picture, greedy NMS at IoU 0.5, the two largest kept; each box padded by "
                   f"{BOX_PAD} of its longer side and named by zero-shot {CLIP_ID} between the two nouns over {len(TEMPLATES)} templates; "
                   "the names assigned by the permutation with the higher summed log-probability; one kept box is named alone. "
                   "Boxes in picture pixels [x0, y0, x1, y1]; divide by 8 for the 128 by 128 latent. No box is drawn or corrected by eye.")}

    @torch.no_grad()
    def text_feats(a, b):
        t = proc(text=[tpl.format(c) for c in (a, b) for tpl in TEMPLATES], return_tensors="pt", padding=True).to(dev)
        f = clip.get_text_features(**t)
        f = f / f.norm(dim=-1, keepdim=True)
        f = f.reshape(2, len(TEMPLATES), -1).mean(1)
        return f / f.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def logprobs(img, box, tf):
        x0, y0, x1, y1 = box
        pad = BOX_PAD * max(x1 - x0, y1 - y0)
        c = img.crop((max(0, x0 - pad), max(0, y0 - pad), min(img.width, x1 + pad), min(img.height, y1 + pad)))
        f = clip.get_image_features(**proc(images=c, return_tensors="pt").to(dev))
        f = f / f.norm(dim=-1, keepdim=True)
        logits = 100.0 * f @ tf.T
        return torch.log_softmax(logits, -1)[0].tolist()

    for pair in pairs:
        a, b = nouns(pair)
        tf = text_feats(a, b)
        out[pair] = {}
        for seed in seeds:
            path = Path(args.images) / pair / f"seed-{seed}.png"
            img = Image.open(path).convert("RGB")
            dets = [d for d in vmetrics.detect_boxes(path, [a, b], device=dev) if d["confidence"] >= DET_CONF]
            dets = [d for d in dets if max(d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]) >= MIN_SIDE * max(img.size)]
            dets.sort(key=lambda d: -d["confidence"])
            keep = []
            for d in dets:
                if all(vmetrics.box_iou(d["box"], k["box"]) < 0.5 for k in keep):
                    keep.append(d)
            keep.sort(key=lambda d: -(d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))
            keep = keep[:2]
            rec = {"A": None, "B": None, "kept": len(keep), "detections": len(dets)}
            boxes = [[round(float(v), 1) for v in d["box"]] for d in keep]
            lps = [logprobs(img, bx, tf) for bx in boxes]
            if len(boxes) == 2:
                straight, swapped = lps[0][0] + lps[1][1], lps[0][1] + lps[1][0]
                rec["A"], rec["B"] = (boxes[0], boxes[1]) if straight >= swapped else (boxes[1], boxes[0])
                rec["margin"] = round(abs(straight - swapped), 3)
            elif len(boxes) == 1:
                rec["A" if lps[0][0] >= lps[0][1] else "B"] = boxes[0]
            rec["clip_logprobs"] = [[round(v, 3) for v in lp] for lp in lps]
            out[pair][str(seed)] = rec
            print(pair, seed, json.dumps({k: rec[k] for k in ("A", "B", "kept")}), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")


# ---------------------------------------------------------------- the box push
def box_mask(box, device):
    m = torch.zeros(GRID, GRID, device=device, dtype=torch.bool)
    if box:
        x0, y0, x1, y1 = [int(round(v / 8)) for v in box]
        m[max(y0, 0):min(y1, GRID), max(x0, 0):min(x1, GRID)] = True
    return m


class BoxPusher:
    """Called by co3_push.PushingUNet on CO3's one two-entry call per step. mode 'measure' records |T| and pushes nothing."""

    def __init__(self, model, inner, guidance, box_a, box_b, lengths=None, mode="push"):
        self.model, self.inner, self.guidance, self.mode, self.lengths = model, inner, float(guidance), mode, lengths
        dev = model.unet.device if hasattr(model.unet, "device") else "cuda"
        ma, mb = box_mask(box_a, dev), box_mask(box_b, dev)
        self.only_a, self.only_b = (ma & ~mb)[None, None], (mb & ~ma)[None, None]
        self.any = bool(self.only_a.any() or self.only_b.any())
        self.step, self.alpha_bar = None, None
        self.t_norms, self.applied, self.capped, self.main_calls = {}, [], [], {}

    def __call__(self, x, timestep, eps):
        self.main_calls[self.step] = self.main_calls.get(self.step, 0) + 1
        if self.main_calls[self.step] > 1:
            raise RuntimeError(f"CO3 made more than one two-entry UNet call on step {self.step}")
        if not (WINDOW[0] <= self.step < WINDOW[1]) or not self.any:
            self.applied.append(0.0); self.capped.append(False)
            return torch.zeros_like(x)
        m, lc = self.model, self.model.latentCorrector
        idx = [0, 2, 3]  # empty, A, B in CO3's text embeddings (Z3's Corrector reads [empty, joint, A, B])
        emb, pool = m.text_embeds[0][idx], m.text_embeds[1][idx]
        added = {"time_ids": m.add_time_ids.repeat(3, 1), "text_embeds": pool}
        with torch.no_grad():
            e = self.inner(x.repeat(3, 1, 1, 1), timestep, encoder_hidden_states=emb, added_cond_kwargs=added)["sample"].float()
        eu, ea, eb = e[0:1], e[1:2], e[2:3]
        lam = float(lc.lmda)
        own_a, own_b = eu + lam * (ea - eu), eu + lam * (eb - eu)
        uncond, cond = eps[:1].float(), eps[1:2].float()
        e_co3 = uncond + self.guidance * (cond - uncond)
        T = torch.where(self.only_a, own_a - e_co3, torch.zeros_like(e_co3)) + torch.where(self.only_b, own_b - e_co3, torch.zeros_like(e_co3))
        tn = float(T.norm())
        self.t_norms[int(self.step)] = tn
        if self.mode == "measure" or tn == 0.0:
            self.applied.append(0.0); self.capped.append(False)
            return torch.zeros_like(x)
        length = float(self.lengths[str(int(self.step))])
        limit = CAP * float(e_co3.norm())
        fired = length > limit
        length = min(length, limit)
        self.applied.append(length); self.capped.append(fired)
        return (-length / tn) * T


def setup(args, first_prompt):
    from co3_common import load, load_parser
    from co3_push import attach
    parser_nlp = load_parser()
    model, config = load(first_prompt, parser_nlp, args.out / "_config")
    model.unet.requires_grad_(False)
    wrapper = attach(model)
    entries = {}
    original = model.denoise_step

    def denoise_step(x, t, step):
        if int(step) == WINDOW[0]:
            entries["x5"] = x.detach().float().cpu()
        return original(x, t, step)
    model.denoise_step = denoise_step
    return model, config, parser_nlp, wrapper, entries


def run_one(model, wrapper, seed, pusher):
    from co3_common import start_noise
    from composers.utils_custom import seed_everything
    seed_everything(seed)
    wrapper.pusher = pusher
    t0 = time.time()
    image = model.sample_loop(start_noise(model, seed))[0]
    wrapper.pusher = None
    return image, time.time() - t0


def prompts_for(pairs):
    from disallowed_subjects import is_allowed
    from train_readout_pair import joint_prompt
    prompts = {p: joint_prompt(p) for p in pairs}
    refused = [p for p in prompts.values() if not is_allowed(p)]
    if refused:
        raise SystemExit(f"the subject guard refuses: {refused}")
    return prompts


def lengths_stage(args):
    from co3_common import set_prompt
    pairs = splits(args.splits)["tuning_pairs"]
    prompts = prompts_for(pairs)
    boxes = json.loads(Path(args.boxes).read_text())
    model, config, parser_nlp, wrapper, _ = setup(args, prompts[pairs[0]])
    per_step, records = {}, []
    for pair in pairs:
        assert set_prompt(model, config, prompts[pair], parser_nlp) == 2, pair
        for seed in TRAIN_SEEDS:
            b = boxes[pair][str(seed)]
            pusher = BoxPusher(model, wrapper.inner, config.guidance_scale, b.get("A"), b.get("B"), mode="measure")
            _, sec = run_one(model, wrapper, seed, pusher)
            for s, n in pusher.t_norms.items():
                per_step.setdefault(str(s), []).append(n)
            records.append({"pair": pair, "seed": seed, "t_norms": pusher.t_norms, "seconds": round(sec, 1)})
            print(f"lengths {pair} seed={seed} {sec:.1f}s steps={len(pusher.t_norms)}", flush=True)
    lengths = {s: float(np.median(v)) for s, v in sorted(per_step.items(), key=lambda kv: int(kv[0]))}
    res = {"rule": "median |T| per step over the tuning pairs at seeds 6001 and 6002 with instrument boxes; T measured, not applied",
           "trajectories": len(records), "trajectories_with_a_box": sum(1 for r in records if r["t_norms"]),
           "lengths": lengths, "records": records}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "lengths.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"lengths": lengths}), flush=True)


def control_stage(args):
    from co3_common import set_prompt
    pairs = splits(args.splits)["tuning_pairs"]
    prompts = prompts_for(pairs)
    boxes = json.loads(Path(args.boxes).read_text())
    lengths = json.loads(Path(args.lengths).read_text())["lengths"]
    model, config, parser_nlp, wrapper, entries = setup(args, prompts[pairs[0]])
    pair, seed = args.control_pair, TEST_SEEDS[0]
    assert set_prompt(model, config, prompts[pair], parser_nlp) == 2
    img1, _ = run_one(model, wrapper, seed, None); x5_plain = entries["x5"]
    img_e, _ = run_one(model, wrapper, seed, BoxPusher(model, wrapper.inner, config.guidance_scale, None, None, lengths))
    b = boxes[pair][str(seed)]
    p = BoxPusher(model, wrapper.inner, config.guidance_scale, b.get("A"), b.get("B"), lengths)
    img_b, _ = run_one(model, wrapper, seed, p); x5_box = entries["x5"]
    diff = lambda u, v: int(np.abs(np.asarray(u, np.int16) - np.asarray(v, np.int16)).max())
    res = {"pair": pair, "seed": seed, "device": torch.cuda.get_device_name(0), "empty_boxes_vs_co3_max_pixel": diff(img1, img_e),
           "entering_step5_max_abs": float((x5_plain - x5_box).abs().max()), "box_push_vs_co3_max_pixel": diff(img1, img_b),
           "pushed_steps": sum(1 for n in p.applied if n > 0), "cap_fired": sum(p.capped),
           "two_entry_calls_per_step": sorted(set(p.main_calls.values())), "boxes": b}
    res["passes"] = res["empty_boxes_vs_co3_max_pixel"] == 0 and res["entering_step5_max_abs"] == 0.0 and res["pushed_steps"] > 0
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "control.json").write_text(json.dumps(res, indent=1) + "\n")
    for name, img in (("co3", img1), ("box", img_b)):
        img.save(args.out / f"control-{name}.png")
    print(json.dumps(res), flush=True)
    raise SystemExit(0 if res["passes"] else 1)


def make_dir(folder):
    """mkdir that tolerates another job creating the same folder at the same moment on /datasets (job 95493 died on that race)."""
    for _ in range(5):
        try:
            folder.mkdir(parents=True, exist_ok=True)
            return
        except FileExistsError:
            time.sleep(2)
    if not folder.is_dir():
        raise SystemExit(f"cannot create {folder}")


def render_stage(args):
    from co3_common import set_prompt
    pairs = splits(args.splits)["tuning_pairs"][args.shard::args.num_shards]
    prompts = prompts_for(pairs)
    row = args.rows
    boxes = json.loads(Path(args.boxes).read_text()) if row == "box" else {}
    lengths = json.loads(Path(args.lengths).read_text())["lengths"] if row == "box" else None
    model, config, parser_nlp, wrapper, _ = setup(args, prompts[pairs[0]])
    done, secs = 0, []
    for pair in pairs:
        assert set_prompt(model, config, prompts[pair], parser_nlp) == 2, pair
        for seed in TEST_SEEDS:
            folder = args.out / row / pair
            path, meta = folder / f"seed-{seed}.png", folder / f"seed-{seed}.json"
            if path.exists() and meta.exists():
                continue
            pusher = None
            if row == "box":
                b = boxes[pair][str(seed)]
                pusher = BoxPusher(model, wrapper.inner, config.guidance_scale, b.get("A"), b.get("B"), lengths)
            img, sec = run_one(model, wrapper, seed, pusher)
            make_dir(folder)
            img.save(path)
            meta.write_text(json.dumps({"row": row, "pair": pair, "seed": seed, "seconds": round(sec, 2),
                                        "applied": pusher.applied if pusher else [], "cap_fired": sum(pusher.capped) if pusher else 0,
                                        "t_norms": pusher.t_norms if pusher else {}, "window": list(WINDOW), "cap": CAP}) + "\n")
            done += 1; secs.append(sec)
            print(f"{done} {row} {pair} seed={seed} {sec:.1f}s", flush=True)
    summary = {"row": row, "shard": args.shard, "rendered": done, "median_seconds": float(np.median(secs)) if secs else None}
    try:
        import wandb
        run = wandb.init(project="poe-repair-animals-compose", entity="prime_lab", name=f"loop-12-z12-{row}-{args.shard}",
                         group="loop-12-z12", config={"pairs": pairs, "seeds": TEST_SEEDS, "row": row, "window": WINDOW, "cap": CAP},
                         settings=wandb.Settings(init_timeout=60))
        run.summary.update(summary); summary["wandb_run"] = run.id; run.finish()
    except Exception as error:
        print(f"W&B logging failed: {error}", flush=True)
    (args.out / f"summary-{row}-{args.shard}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["boxes", "lengths", "control", "render"])
    ap.add_argument("--splits", type=Path)
    ap.add_argument("--images", type=Path)
    ap.add_argument("--pairs", default="")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--boxes", type=Path)
    ap.add_argument("--lengths", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", default="co3", choices=["co3", "box"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--control-pair", default="a_duck__x__a_goose")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("torch sees no CUDA card")
    {"boxes": boxes_stage, "lengths": lengths_stage, "control": control_stage, "render": render_stage}[args.stage](args)


if __name__ == "__main__":
    main()
