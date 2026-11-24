#!/usr/bin/env python3
"""Loop 12 step Z2: read three predictors of borrowing while CO3 renders M1's tuning pictures.

Design: artifacts/notes/progress/leakage/02-what-predicts-borrowing/design.md. Bars: z2_bars.py.

CO3 renders each pair and seed exactly as M1's CO3 row did (seed_everything, then sample_loop from start_noise). At each
read step (CO3's 6 corrected steps, then every 5th step to 45), when the main two-entry U-Net call arrives, one extra call with
the batch [empty, joint, A, B] runs first on that latent, under no_grad, with attention processors that keep the joint
prompt's cross-attention for the two nouns, the end-of-text token and the padding. Its output gives d_t(A) and d_t(B)
and is then discarded, and the sampler's own call runs with the original processors.

  python z2_predictors.py --splits <json> --m1 <M1 renders/co3 folder> --out <dir> [--pairs a,b] [--seeds 6001,6002]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "rung2"))
from co3_common import load, load_parser, set_prompt, start_noise  # noqa: E402
from composers.utils_custom import seed_everything  # noqa: E402
from disallowed_subjects import is_allowed  # noqa: E402
from train_readout_pair import joint_prompt  # noqa: E402
from z2_bars import DAAM_CUT, PICTURE_MATCH_MAX_MEAN_ABS, READ_STEPS, WINDOWS  # noqa: E402

GRID = 128  # SDXL latent side at 1024 pixels
COLUMNS = ("A", "B", "eot", "pad")


def noun_index(tokenizer, prompt, word):
    """Position of the last token of `word` inside `prompt`'s token ids (start token included)."""
    ids = tokenizer(prompt).input_ids
    target = tokenizer(word, add_special_tokens=False).input_ids
    for i in range(len(ids) - len(target), -1, -1):
        if ids[i:i + len(target)] == target:
            return i + len(target) - 1
    raise ValueError(f"{word!r} not found in {prompt!r}")


def nouns(pair):
    a, b = pair.split("__x__")
    return a.split("_", 1)[1].replace("_", " "), b.split("_", 1)[1].replace("_", " ")


class KeepCrossAttention:
    """Plain attention; for the batch entry `entry`, keeps the head-mean probabilities of the chosen key columns."""

    def __init__(self, store, name, entry, cols):
        self.store, self.name, self.entry, self.cols = store, name, entry, cols

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):
        batch, seq, _ = hidden_states.shape
        query = attn.head_to_batch_dim(attn.to_q(hidden_states))
        key = attn.head_to_batch_dim(attn.to_k(encoder_hidden_states))
        value = attn.head_to_batch_dim(attn.to_v(encoder_hidden_states))
        probs = attn.get_attention_scores(query, key, attention_mask)
        p = probs.view(batch, attn.heads, seq, -1)[self.entry].mean(0).float()  # seq x 77
        a, b, eot, pad_from = self.cols
        kept = torch.stack([p[:, a], p[:, b], p[:, eot], p[:, pad_from:].mean(1)], 0)  # 4 x seq
        self.store.append(kept)
        out = attn.batch_to_head_dim(torch.bmm(probs, value))
        return attn.to_out[1](attn.to_out[0](out))


class CaptureUNet(torch.nn.Module):
    """The U-Net; before CO3's main call at a corrected step, one discarded capture call on the same latent."""

    def __init__(self, unet, model):
        super().__init__()
        self.inner = unet
        self.model = model
        self.step = None
        self.cols = None
        self.records = {}

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("inner"), name)

    def forward(self, sample, timestep, *args, **kwargs):
        if self.cols is not None and sample.shape[0] == 2 and self.step in READ_STEPS and self.step not in self.records:
            self.records[self.step] = self.capture(sample[:1], timestep)
        return self.inner(sample, timestep, *args, **kwargs)

    @torch.no_grad()
    def capture(self, x, timestep):
        emb, pool = self.model.text_embeds[0][:4], self.model.text_embeds[1][:4]
        added = {"time_ids": self.model.add_time_ids.repeat(4, 1), "text_embeds": pool}
        saved = dict(self.inner.attn_processors)
        store, procs = [], {}
        for name, proc in saved.items():
            procs[name] = KeepCrossAttention(store, name, 1, self.cols) if name.endswith("attn2.processor") else proc
        self.inner.set_attn_processor(procs)
        try:
            eps = self.inner(x.repeat(4, 1, 1, 1), timestep, encoder_hidden_states=emb, added_cond_kwargs=added)["sample"].float()
        finally:
            self.inner.set_attn_processor(saved)
        maps = torch.zeros(len(COLUMNS), GRID, GRID, device=x.device)
        for kept in store:
            side = int(round(kept.shape[1] ** 0.5))
            maps += F.interpolate(kept.view(1, len(COLUMNS), side, side), size=(GRID, GRID), mode="bilinear",
                                  align_corners=False)[0]
        return {"maps": maps.cpu(), "dA": (eps[2] - eps[0]).cpu(), "dB": (eps[3] - eps[0]).cpu(), "layers": len(store)}


def scaled(m):
    """Each summed map rescaled to run from 0 to 1. On SDXL the raw sum over layers sits within about a quarter of its
    peak everywhere (smoke run, eagle at step 0: mean 10.97 against a maximum of 14.32), so DAAM's cut at 0.4 of the
    raw maximum keeps every cell and every overlap reads 1.0."""
    return (m - m.min()) / (m.max() - m.min() + 1e-8)


def iou(m1, m2):
    m1, m2 = scaled(m1), scaled(m2)
    a, b = m1 >= DAAM_CUT * m1.max(), m2 >= DAAM_CUT * m2.max()
    return float((a & b).sum() / max(int((a | b).sum()), 1))


def soft_overlap(m1, m2):
    m1, m2 = scaled(m1), scaled(m2)
    return float(torch.minimum(m1 / m1.sum(), m2 / m2.sum()).sum())


def cosine(u, v):
    return float(F.cosine_similarity(u.flatten().double(), v.flatten().double(), dim=0))


def step_numbers(r):
    m = r["maps"]
    return {"iou_AB": iou(m[0], m[1]), "soft_AB": soft_overlap(m[0], m[1]),
            "iou_A_eot": iou(m[0], m[2]), "iou_B_eot": iou(m[1], m[2]),
            "iou_A_pad": iou(m[0], m[3]), "iou_B_pad": iou(m[1], m[3]),
            "dt_cos": cosine(r["dA"], r["dB"]), "dt_norm_cos": cosine(r["dA"].norm(dim=0), r["dB"].norm(dim=0)),
            "self_iou_A": iou(m[0], m[0]), "layers": r["layers"]}


def text_numbers(model, pair, prompt):
    """Cosines between the two nouns' vectors as the U-Net receives them (2,048 numbers: CLIP-L 768 then bigG 1,280)."""
    a, b = nouns(pair)
    single = [s.lower() for s in model.prompts_single[:2]]
    if not (single[0].endswith(a) and single[1].endswith(b)):
        raise SystemExit(f"CO3 parsed {pair} as {model.prompts_single}; d_t(A) and d_t(B) would be mislabelled")
    emb, pool = model.text_embeds[0], model.text_embeds[1]  # [empty, joint, A alone, B alone, joint]
    tok = model.tokenizer
    ja, jb = noun_index(tok, prompt, a), noun_index(tok, prompt, b)
    sa, sb = noun_index(tok, model.prompts_single[0], a), noun_index(tok, model.prompts_single[1], b)
    joint_a, joint_b, alone_a, alone_b = emb[1, ja], emb[1, jb], emb[2, sa], emb[3, sb]
    return {"text_joint": cosine(joint_a, joint_b), "text_joint_clipL": cosine(joint_a[:768], joint_b[:768]),
            "text_joint_bigG": cosine(joint_a[768:], joint_b[768:]), "text_alone": cosine(alone_a, alone_b),
            "text_alone_clipL": cosine(alone_a[:768], alone_b[:768]), "text_alone_bigG": cosine(alone_a[768:], alone_b[768:]),
            "text_pooled": cosine(pool[2], pool[3]), "tokens": {"joint": [ja, jb], "alone": [sa, sb]},
            "single_prompts": list(model.prompts_single[:2])}


def token_columns(model, pair, prompt):
    a, b = nouns(pair)
    ids = model.tokenizer(prompt).input_ids
    eot = len(ids) - 1
    return (noun_index(model.tokenizer, prompt, a), noun_index(model.tokenizer, prompt, b), eot, eot + 1)


def log_wandb(name, config, summary):
    try:
        import wandb
        run = wandb.init(project="poe-repair-animals-compose", entity="prime_lab", name=name, group="loop-12-z2",
                         config=config, settings=wandb.Settings(init_timeout=60))
        run.summary.update(summary)
        run_id = run.id
        run.finish()
        return run_id
    except Exception as error:  # the numbers are on disk either way
        print(f"W&B logging failed: {error}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--m1", type=Path, required=True, help="M1's renders/co3 folder")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs", default="")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--rung", default="z2")
    ap.add_argument("--plain-check", action="store_true",
                    help="before the first picture, render it once with the capture off, to tell the capture's effect from the card's")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("torch sees no CUDA card")
    split = json.loads(args.splits.read_text())
    pairs = args.pairs.split(",") if args.pairs else split["tuning_pairs"]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else split["tuning_seeds"]
    assert set(pairs) <= set(split["tuning_pairs"]) and set(seeds) <= set(split["tuning_seeds"]), "tuning data only"
    prompts = {p: joint_prompt(p) for p in pairs}
    refused = [p for p in prompts.values() if not is_allowed(p)]
    if refused:
        raise SystemExit(f"the subject guard refuses: {refused}")

    parser_nlp = load_parser()
    model, config = load(prompts[pairs[0]], parser_nlp, args.out / "_config")
    model.unet.requires_grad_(False)
    wrapper = CaptureUNet(model.unet, model)
    model.unet = wrapper
    original = model.denoise_step

    def denoise_step(x, t, step):
        wrapper.step = int(step)
        return original(x, t, step)

    model.denoise_step = denoise_step
    rows, failed = [], []
    for pair in pairs:
        concepts = set_prompt(model, config, prompts[pair], parser_nlp)
        if concepts != 2:
            print(f"skipping {pair}: {concepts} concepts", flush=True)
            failed.append({"pair": pair, "reason": f"{concepts} concepts"})
            continue
        text = text_numbers(model, pair, prompts[pair])
        wrapper.cols = token_columns(model, pair, prompts[pair])
        folder = args.out / pair
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "text.json").write_text(json.dumps(text, indent=1) + "\n")
        if args.plain_check and not (folder / "plain-check.json").exists():
            seed, cols = seeds[0], wrapper.cols
            wrapper.cols = None
            seed_everything(seed)
            plain = model.sample_loop(start_noise(model, seed))[0]
            wrapper.cols, wrapper.records = cols, {}
            seed_everything(seed)
            captured = model.sample_loop(start_noise(model, seed))[0]
            reference = args.m1 / pair / f"seed-{seed}.png"
            arr = lambda im: np.asarray(im.convert("RGB") if hasattr(im, "convert") else im, np.int16)
            check = {"pair": pair, "seed": seed, "device": torch.cuda.get_device_name(0),
                     "plain_vs_captured_mean_abs": float(np.abs(arr(plain) - arr(captured)).mean()),
                     "plain_vs_m1_mean_abs": float(np.abs(arr(plain) - arr(Image.open(reference))).mean()),
                     "captured_vs_m1_mean_abs": float(np.abs(arr(captured) - arr(Image.open(reference))).mean())}
            plain.save(folder / f"plain-check-seed-{seed}.png")
            (folder / "plain-check.json").write_text(json.dumps(check, indent=1) + "\n")
            print("plain check", json.dumps(check), flush=True)
        for seed in seeds:
            meta_path = folder / f"seed-{seed}.json"
            if meta_path.exists():
                rows.append(json.loads(meta_path.read_text()))
                continue
            wrapper.records = {}
            begin = time.time()
            seed_everything(seed)
            image = model.sample_loop(start_noise(model, seed))[0]
            seconds = time.time() - begin
            image.save(folder / f"seed-{seed}.png")
            reference = args.m1 / pair / f"seed-{seed}.png"
            diff = (float(np.abs(np.asarray(image, np.int16) - np.asarray(Image.open(reference).convert("RGB"), np.int16)).mean())
                    if reference.exists() else None)
            steps = {s: step_numbers(r) for s, r in sorted(wrapper.records.items())}
            np.savez_compressed(folder / f"seed-{seed}-maps.npz",
                                maps=np.stack([wrapper.records[s]["maps"].half().numpy() for s in READ_STEPS]),
                                dA=np.stack([wrapper.records[s]["dA"].half().numpy() for s in READ_STEPS]),
                                dB=np.stack([wrapper.records[s]["dB"].half().numpy() for s in READ_STEPS]),
                                steps=np.array(READ_STEPS),
                                columns=np.array(COLUMNS))
            keys = next(iter(steps.values()))
            windows = {w: {k: float(np.mean([steps[s][k] for s in ss])) for k in keys} for w, ss in WINDOWS.items()}
            mean = windows["early"]
            row = {"pair": pair, "seed": seed, "prompt": prompts[pair], "seconds": round(seconds, 2),
                   "m1_mean_abs_diff": diff, "picture_matches": diff is not None and diff < PICTURE_MATCH_MAX_MEAN_ABS,
                   "captured_steps": sorted(wrapper.records), "steps": steps, "windows": windows,
                   "token_columns": list(wrapper.cols)}
            meta_path.write_text(json.dumps(row, indent=1) + "\n")
            rows.append(row)
            print(f"{pair} seed={seed} {seconds:.1f}s diff={diff} iou early/late={mean['iou_AB']:.3f}/{windows['late']['iou_AB']:.3f} dt_cos={mean['dt_cos']:.3f} "
                  f"text={text['text_joint']:.3f} self_iou={mean['self_iou_A']:.3f}", flush=True)
    summary = {"pictures": len(rows), "pictures_matching_m1": sum(r["picture_matches"] for r in rows),
               "median_seconds": float(np.median([r["seconds"] for r in rows])) if rows else None, "failed": failed}
    run_id = log_wandb(f"loop-12-{args.rung}", {"pairs": pairs, "seeds": seeds, "daam_cut": DAAM_CUT,
                                                "read_steps": list(READ_STEPS), "windows": {k: list(v) for k, v in WINDOWS.items()}}, summary)
    summary["wandb_run"] = run_id
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
