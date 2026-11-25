#!/usr/bin/env python3
"""Step M15: a score that names no subject, from SDXL's attention on CO3's saved states, judged within each pair.

Design: artifacts/notes/progress/method-search/11-a-score-that-names-no-subject/design.md. The cross-attention measure is the cat-and-dog
figure's own (2-classifier-guidance/scripts/rung2/attention_overlap.py): probabilities of SDXL's 32 x 32 cross-attention layers averaged over
heads and layers, the start token dropped and each position renormalised over the prompt's words, and the top-10% intersection over union.

  control  rerun the figure's 8 cat x dog joint-prompt trajectories with this code and compare the top-10% overlap at steps 5, 10 and 15
           with the figure's saved overlap.json
  read     on CO3's saved states (steps 5, 10, 15), one SDXL pass under the joint prompt each, keeping cross- and self-attention at 32 x 32:
           the two nouns' overlap and each noun's instance count
  score    the within-pair area under the curve per step, the shuffled control, and the descriptive splits

Usage: python m15_attention.py control --out <dir> --saved <overlap.json>
       python m15_attention.py read --latents-from <dir> ... --out <dir>
       python m15_attention.py score --out <dir> --labels <json of "pair|seed" -> true when two clear> [--kinds <json>]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
READ_STEPS = (5, 10, 15)
TOP, K, KMEANS_SEED, KMEANS_ITERS = 0.10, 6, 0, 50


class Store:
    def __init__(self):
        self.cross, self.self_ = [], []

    def clear(self):
        self.cross, self.self_ = [], []


def processors(unet, store, torch):
    class Keep:
        def __init__(self, kind):
            self.kind = kind

        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):
            batch, seq, _ = hidden_states.shape
            context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
            query = attn.head_to_batch_dim(attn.to_q(hidden_states))
            key = attn.head_to_batch_dim(attn.to_k(context))
            value = attn.head_to_batch_dim(attn.to_v(context))
            probs = attn.get_attention_scores(query, key, attention_mask)
            if seq == 32 * 32:
                kept = probs.view(batch, attn.heads, seq, -1).mean(1).float().cpu()
                (store.cross if self.kind == "cross" else store.self_).append(kept)
            out = attn.batch_to_head_dim(torch.bmm(probs, value))
            return attn.to_out[1](attn.to_out[0](out))
    return {n: (Keep("cross") if ".attn2." in n else Keep("self") if ".attn1." in n else p) for n, p in unet.attn_processors.items()}


def token_positions(tokenizer, prompt, word):
    """Every position of the word's tokens in the prompt; a word that splits into several tokens keeps them all (the figure's
    token_index allows one token, which cat and dog are)."""
    ids = tokenizer(prompt).input_ids
    target = tokenizer(word, add_special_tokens=False).input_ids
    for i in range(len(ids) - len(target) + 1):
        if ids[i:i + len(target)] == target:
            return list(range(i, i + len(target)))
    raise ValueError(f"{word!r} not found in {prompt!r}")


def noun_maps(cross, tokenizer, prompt, words, torch):
    """cross: list of (batch, 1024, 77) for one UNet call; the joint prompt's entry is the last. -> one 1024-vector per word: the figure's
    words_only renormalisation (start token dropped, each position renormalised over the prompt's word tokens), averaged over the word's
    tokens."""
    sys.path.insert(0, str(HERE.parent / "rung2"))
    from attention_overlap import words_only
    m = torch.stack(cross).mean(0)[-1]
    n_words = len(tokenizer(prompt).input_ids) - 2
    w = words_only(m, n_words)
    return [w[:, [i - 1 for i in token_positions(tokenizer, prompt, word)]].mean(1) for word in words]


def top_mask(v, torch):
    k = max(1, int(TOP * v.numel()))
    mask = torch.zeros_like(v, dtype=torch.bool)
    mask[torch.topk(v, k).indices] = True
    return mask


def instance_counts(self_attn, masks, torch):
    """k-means (k = 6, seed 0) on the rows of the averaged self-attention; a group belongs to the noun holding more than half its cells in
    the noun's top-10% mask; the noun's count is the number of 4-connected pieces of its groups' union."""
    from scipy.ndimage import label
    rows = self_attn
    g = torch.Generator().manual_seed(KMEANS_SEED)
    centres = rows[torch.randperm(rows.shape[0], generator=g)[:K]].clone()
    for _ in range(KMEANS_ITERS):
        assign = torch.cdist(rows, centres).argmin(1)
        for c in range(K):
            if (assign == c).any():
                centres[c] = rows[assign == c].mean(0)
    counts = []
    for mask in masks:
        owned = torch.zeros(rows.shape[0], dtype=torch.bool)
        for c in range(K):
            cells = assign == c
            if cells.any() and mask[cells].float().mean() > 0.5:
                owned |= cells
        _, n = label(owned.view(32, 32).numpy())
        counts.append(int(n))
    return counts


def top_iou(a, b, torch):
    ma, mb = top_mask(a, torch), top_mask(b, torch)
    return float((ma & mb).sum() / (ma | mb).sum())


def control(args):
    import torch
    sys.path.insert(0, str(HERE.parents[2]))
    sys.path.insert(0, str(HERE.parent))
    sys.path.insert(0, str(HERE.parent / "rung2"))
    from poe_repair.composers._helpers import init_latents_for_cell
    from poe_repair.methods._sampling import run_cfg
    from poe_repair.run import make_ctx
    from poe_repair.runtime import encode_prompt_sdxl
    import classifier_guidance_from_the_paper as single

    saved = json.loads(args.saved.read_text())["top10_iou"]
    ctx = make_ctx(output_root=args.out / "raw")
    unet = ctx.models["unet"]
    enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = enc("")
    prompt = "a cat and a dog"
    seq_j, pool_j = enc(prompt)
    store = Store()
    originals = dict(unet.attn_processors)
    steps_seen = []
    hook = unet.register_forward_pre_hook(lambda *_: steps_seen.append(list(store.cross)) or store.clear())   # cross only: memory
    rows = {}
    try:
        unet.set_attn_processor(processors(unet, store, torch))
        for seed in range(1, 9):
            steps_seen.clear()
            store.clear()
            init, sigma = init_latents_for_cell(single.cell_for(seed), ctx)
            run_cfg(init_latents=init, euler_init_noise_sigma=sigma, seq_cond=seq_j, pool_cond=pool_j, models=ctx.models,
                    scheduler=ctx.scheduler, seq_e=seq_e, pool_e=pool_e, guidance_scale=ctx.guidance_scale,
                    num_inference_steps=ctx.num_inference_steps, height=1024, width=1024, device=ctx.device, dtype=ctx.dtype)
            per_step = steps_seen[1:] + [list(store.cross)]
            ours, theirs = {}, saved[f"seed {seed}, joint"]
            for s in READ_STEPS:
                cross = [c[:1] for c in per_step[s]]             # the figure reads the first batch entry
                cat, dog = noun_maps(cross, ctx.models["tokenizer"], prompt, ["cat", "dog"], torch)
                ours[s] = round(top_iou(cat, dog, torch), 4)
            rows[seed] = {"ours": ours, "figure": {s: round(theirs[s], 4) for s in READ_STEPS},
                          "largest_gap": round(max(abs(ours[s] - theirs[s]) for s in READ_STEPS), 4)}
            print(seed, rows[seed], flush=True)
    finally:
        hook.remove()
        unet.set_attn_processor(originals)
    apart = (2, 3, 7)
    means = {s: {"apart": round(float(np.mean([rows[x]["ours"][s] for x in apart])), 4),
                 "rest": round(float(np.mean([rows[x]["ours"][s] for x in rows if x not in apart])), 4)} for s in READ_STEPS}
    result = {"per_seed": rows, "largest_gap": max(r["largest_gap"] for r in rows.values()), "reproduces_within_0.05":
              all(r["largest_gap"] <= 0.05 for r in rows.values()), "group_means": means}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "control.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("largest_gap", "reproduces_within_0.05", "group_means")}), flush=True)


def read(args):
    import torch
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parent / "rung2"))
    sys.path.insert(0, str(HERE.parents[2]))
    from co3_common import load, load_parser
    from paper2_rows import encode
    from poe_repair.methods._sampling import add_time_ids
    from train_readout_pair import joint_prompt

    source = []
    for folder in args.latents_from:
        for r in map(json.loads, (folder / "reads.jsonl").read_text().splitlines()):
            path = folder / "latents" / r["pair"] / f"seed-{r['seed']}.pt"
            if path.exists() and r["reproduced"]:
                source.append((r["pair"], int(r["seed"]), path))
    source = sorted(set(source))
    if args.limit:
        source = source[: args.limit]
    model, _ = load(joint_prompt(source[0][0]), load_parser(), args.out / "_config")
    unet = model.unet
    store = Store()
    originals = dict(unet.attn_processors)
    unet.set_attn_processor(processors(unet, store, torch))
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "attention.jsonl", "w") as log:
        for pair, seed, latent_path in source:
            prompt = joint_prompt(pair)
            words = [w.split("_", 1)[1].replace("_", " ") for w in pair.split("__x__")]
            seq, pool = encode(model, prompt)
            saved = torch.load(latent_path, weights_only=False)
            row = {"pair": pair, "seed": seed, "steps": {}}
            for step in READ_STEPS:
                store.clear()
                st = saved[step]
                x = st["x"].to(unet.device, torch.float16)
                t = torch.full((1,), st["t"], device=unet.device)
                cond = {"text_embeds": pool, "time_ids": add_time_ids(height=1024, width=1024, batch_size=1, device=unet.device,
                                                                         dtype=torch.float16)}
                with torch.no_grad():
                    unet(x, t, encoder_hidden_states=seq, added_cond_kwargs=cond)
                a, b = noun_maps(store.cross, model.tokenizer, prompt, words, torch)
                self_attn = torch.stack(store.self_).mean(0)[0]
                masks = [top_mask(a, torch), top_mask(b, torch)]
                counts = instance_counts(self_attn, masks, torch)
                row["steps"][str(step)] = {"overlap": top_iou(a, b, torch), "count_first": counts[0], "count_second": counts[1]}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(pair, seed, {s: (round(v["overlap"], 3), v["count_first"], v["count_second"]) for s, v in row["steps"].items()},
                  flush=True)
    unet.set_attn_processor(originals)


def within_pair_auc(rows, labels, value):
    num, den = 0.0, 0
    pairs = sorted({r["pair"] for r in rows})
    for p in pairs:
        pos = [value(r) for r in rows if r["pair"] == p and labels[f"{p}|{r['seed']}"]]
        neg = [value(r) for r in rows if r["pair"] == p and not labels[f"{p}|{r['seed']}"]]
        for x in pos:
            for y in neg:
                num += 1.0 if x > y else 0.5 if x == y else 0.0
                den += 1
    return num / den if den else float("nan"), den


def score(args):
    rows = [json.loads(l) for l in (args.out / "attention.jsonl").read_text().splitlines() if l.strip()]
    labels = json.loads(args.labels.read_text())
    rows = [r for r in rows if f"{r['pair']}|{r['seed']}" in labels]
    S = lambda st: (lambda r: -r["steps"][st]["overlap"] - abs(r["steps"][st]["count_first"] - 1) - abs(r["steps"][st]["count_second"] - 1))
    O = lambda st: (lambda r: -r["steps"][st]["overlap"])
    C = lambda st: (lambda r: -abs(r["steps"][st]["count_first"] - 1) - abs(r["steps"][st]["count_second"] - 1))
    rng = np.random.default_rng(0)
    result = {"pictures": len(rows), "per_step": {}}
    for st in map(str, READ_STEPS):
        auc, couples = within_pair_auc(rows, labels, S(st))
        shuffled = []
        for _ in range(2000):
            perm = {}
            for p in {r["pair"] for r in rows}:
                keys = [f"{p}|{r['seed']}" for r in rows if r["pair"] == p]
                vals = [labels[k] for k in keys]
                rng.shuffle(vals)
                perm.update(dict(zip(keys, vals)))
            shuffled.append(within_pair_auc(rows, perm, S(st))[0])
        p95, p99 = np.percentile(shuffled, [95, 99])
        result["per_step"][st] = {"within_pair_auc": round(auc, 4), "couples": couples, "shuffled_p95": round(float(p95), 4),
                                  "shuffled_p99": round(float(p99), 4), "overlap_alone": round(within_pair_auc(rows, labels, O(st))[0], 4),
                                  "counts_alone": round(within_pair_auc(rows, labels, C(st))[0], 4),
                                  "median_overlap_two_clear": round(float(np.median([r["steps"][st]["overlap"] for r in rows
                                                                                     if labels[f"{r['pair']}|{r['seed']}"]])), 4),
                                  "median_overlap_rest": round(float(np.median([r["steps"][st]["overlap"] for r in rows
                                                                               if not labels[f"{r['pair']}|{r['seed']}"]])), 4)}
    ps = result["per_step"]
    if any(v["within_pair_auc"] >= 0.70 and v["within_pair_auc"] > v["shuffled_p99"] for v in ps.values()):
        verdict = "a subject-free score ranks the pictures"
    elif all(v["within_pair_auc"] <= v["shuffled_p95"] for v in ps.values()):
        verdict = "it does not"
    else:
        verdict = "inconclusive"
    result["verdict"] = verdict
    if args.kinds:
        kinds = json.loads(args.kinds.read_text())
        result["by_failure_kind"] = {}
        for kind in sorted({k for k in kinds.values() if k != "own_parts"}):
            sub = [r for r in rows if kinds.get(f"{r['pair']}|{r['seed']}") in ("own_parts", kind)]
            sub_labels = {f"{r['pair']}|{r['seed']}": kinds[f"{r['pair']}|{r['seed']}"] == "own_parts" for r in sub}
            result["by_failure_kind"][f"own parts against {kind}"] = {
                st: {"within_pair_auc": round(within_pair_auc(sub, sub_labels, S(st))[0], 4),
                     "couples": within_pair_auc(sub, sub_labels, S(st))[1]} for st in map(str, READ_STEPS)}
    (args.out / "score.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"verdict": verdict, "per_step": {s: (v["within_pair_auc"], v["shuffled_p99"]) for s, v in ps.items()}}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["control", "read", "score"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--saved", type=Path)
    ap.add_argument("--latents-from", type=Path, nargs="*")
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--kinds", type=Path)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    {"control": control, "read": read, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
