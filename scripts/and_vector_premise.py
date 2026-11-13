#!/usr/bin/env python
"""Does text arithmetic recover a word in SDXL's encoders? The premise under every x variant.

Every "and" vector built so far assumes E("A and B") - E("A") lands near E("B"), the way
king - man + woman lands near queen in word2vec. Nobody has checked. This checks it, with no UNet
and no sampling: build the difference for many pairs, then rank every noun in a dictionary by how
close it sits to that difference, and report where the true B came out.

    rank 1 of N           the arithmetic recovers the held-out word; the x programme has a premise
    rank near N/2         the difference points nowhere in particular; subtraction is not meaningful here

Two readouts per pair, because SDXL gives two kinds of vector and they can disagree:

    pooled      the 1280-wide summary vector, one per prompt, no token positions to align
    content     the mean of the sequence's content positions, which is the closest single vector
                to what the 77-position arithmetic actually manipulates

Two controls, both needed to read the rank at all:

    no subtraction    rank B using E("A and B") alone. The joint prompt names B, so B should already
                      rank well. Subtraction has to beat this to have done anything.
    shuffled          rank a random other noun using the same difference. This is where chance sits.

    python scripts/and_vector_premise.py --pairs 200 --out artifacts/results/the-positional-and-vector
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

REPO = Path(os.environ.get("POE_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from and_vector_norms import load_text_encoders, tokens  # noqa: E402

# A dictionary of concrete single-token-ish nouns. The rank is "out of this many", so the list is
# the denominator of every number this script prints and it is written here rather than fetched.
NOUNS = """
cat dog horse cow sheep goat pig chicken duck goose rabbit mouse rat squirrel fox wolf bear deer
lion tiger leopard cheetah elephant giraffe zebra hippo rhino camel llama monkey gorilla panda koala
kangaroo penguin owl eagle hawk falcon crow raven sparrow robin pigeon parrot flamingo peacock swan
whale dolphin shark octopus crab lobster shrimp turtle frog snake lizard crocodile spider ant bee
butterfly moth beetle dragonfly grasshopper snail worm
car truck bus train plane boat ship bicycle motorcycle scooter helicopter rocket tractor van taxi
chair table desk bed sofa bench stool shelf cabinet drawer mirror lamp clock vase basket box bag
cup mug glass bottle plate bowl fork knife spoon kettle pan pot oven fridge toaster blender
phone laptop computer keyboard mouse monitor camera television radio speaker headphone printer
book magazine newspaper letter envelope pencil pen brush paint canvas paper notebook map poster
shirt dress coat jacket hat scarf glove sock shoe boot belt tie button zipper pocket
apple banana orange lemon grape cherry peach pear plum melon strawberry pineapple mango coconut
bread cheese butter egg milk rice pasta soup salad cake cookie pizza sandwich burger
tree flower grass leaf branch root seed bush fern moss cactus palm oak pine rose tulip daisy
mountain hill valley river lake ocean beach desert forest island cliff cave waterfall
house barn castle tower bridge fence gate door window roof wall stair chimney garden
sun moon star cloud rain snow storm lightning rainbow fog wind fire smoke ice
hammer screwdriver wrench saw drill nail screw rope chain ladder bucket broom shovel axe
ball bat racket glove net goal helmet skateboard surfboard kite balloon drum guitar piano violin
trumpet flute harp banjo
""".split()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO / "artifacts/results/the-positional-and-vector")
    args = ap.parse_args()

    import torch

    from poe_repair.config import DEFAULT_MODEL_ID
    from poe_repair.runtime import encode_prompt_sdxl

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_text_encoders(DEFAULT_MODEL_ID)
    for k in ("text_encoder", "text_encoder_2"):
        models[k].to(device)
    enc = lambda p: encode_prompt_sdxl(p, models=models, device=device, dtype=torch.float32)

    vocab = sorted(set(NOUNS))
    tk = models["tokenizer"]
    content_end = {w: tokens(tk, w).index("<|endoftext|>") for w in vocab}

    def two_views(prompt: str, end: int | None = None):
        """One pooled vector and one content-mean vector for a prompt."""
        seq, pooled = enc(prompt)
        end = end if end is not None else tokens(tk, prompt).index("<|endoftext|>")
        return pooled[0].cpu(), seq[0, 1:end].mean(dim=0).cpu()

    print(f"encoding {len(vocab)} nouns on {device}", flush=True)
    single = {w: two_views(w, content_end[w]) for w in vocab}
    P = torch.stack([single[w][0] for w in vocab])
    C = torch.stack([single[w][1] for w in vocab])
    P = P / P.norm(dim=-1, keepdim=True)
    C = C / C.norm(dim=-1, keepdim=True)
    index = {w: i for i, w in enumerate(vocab)}

    rng = random.Random(args.seed)
    pairs = []
    while len(pairs) < args.pairs:
        a, b = rng.sample(vocab, 2)
        if (a, b) not in pairs:
            pairs.append((a, b))

    def rank_of(query, bank, target_i):
        q = query / query.norm()
        sims = bank @ q
        return int((sims > sims[target_i]).sum()) + 1

    rows = []
    for n, (a, b) in enumerate(pairs):
        joint = f"{a} and {b}"
        jp, jc = two_views(joint)
        ap_, ac = single[a]
        decoy = rng.choice([w for w in vocab if w not in (a, b)])
        rows.append({
            "a": a, "b": b,
            "pooled_subtracted": rank_of(jp - ap_, P, index[b]),
            "pooled_no_subtraction": rank_of(jp, P, index[b]),
            "pooled_decoy": rank_of(jp - ap_, P, index[decoy]),
            "content_subtracted": rank_of(jc - ac, C, index[b]),
            "content_no_subtraction": rank_of(jc, C, index[b]),
            "content_decoy": rank_of(jc - ac, C, index[decoy]),
        })
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(pairs)} pairs", flush=True)

    import statistics
    keys = [k for k in rows[0] if k not in ("a", "b")]
    summary = {}
    for k in keys:
        v = [r[k] for r in rows]
        summary[k] = {"median_rank": statistics.median(v), "mean_rank": sum(v) / len(v),
                      "top_1": sum(x == 1 for x in v) / len(v),
                      "top_10": sum(x <= 10 for x in v) / len(v)}

    print(f"\nrank of the held-out word among {len(vocab)} nouns, over {len(pairs)} pairs")
    print(f"  chance is a median near {len(vocab) / 2:.0f}; perfect recovery is 1\n")
    print(f"  {'readout':<26} {'median':>7} {'mean':>8} {'top-1':>7} {'top-10':>7}")
    for k in keys:
        s = summary[k]
        print(f"  {k:<26} {s['median_rank']:>7.0f} {s['mean_rank']:>8.1f} "
              f"{s['top_1']:>6.1%} {s['top_10']:>6.1%}")

    args.out.mkdir(parents=True, exist_ok=True)
    rec = {"model_id": DEFAULT_MODEL_ID, "vocabulary_size": len(vocab), "pairs": len(pairs),
           "prompt_form": "'<a> and <b>' against '<a>', bare words, as the sheets use",
           "chance_median_rank": len(vocab) / 2, "summary": summary, "rows": rows}
    (args.out / "premise-word-recovery.json").write_text(json.dumps(rec, indent=1))
    print(f"\nwrote {args.out / 'premise-word-recovery.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
