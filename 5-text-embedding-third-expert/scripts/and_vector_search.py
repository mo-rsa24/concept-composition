#!/usr/bin/env python
"""Search for a third expert that puts two separate animals back in the frame.

Every third expert so far was hand-picked. This searches the space that is known to act, scores every
candidate automatically with the instance-count detector, and reports what beats the best phrase
anyone has written by hand.

Three kinds of candidate, all scored the same way:

    prompt    PoE(a, b, "<phrase>")         a written phrase as the third expert
    mix       PoE(a, b, (1-L)E("") + L E(joint))
                                            a prompt-shaped point between the empty prompt and the
                                            joint prompt, which is the line x_fix and x_pos sit near
    negative  PoE(a, b) with "<phrase>" in the unconditional branch instead of ""
                                            not a third expert at all: the branch that enters at
                                            (1 - N*w), the largest coefficient in the sum

The objective is CO-OCCURRENCE, not identity: two separate animals in the frame. That is exactly what
`count_instances` measures and exactly what it is reliable at, since it counts animals without naming
them. The same property is this search's main hazard: TWO DOGS SCORE AS A WIN. Nothing here can tell a
cat from a dog, so a high-scoring candidate is a candidate for round two, never a result.

    python 5-text-embedding-third-expert/scripts/and_vector_search.py --shard 0 --of 6 --out <dir>
    python 5-text-embedding-third-expert/scripts/and_vector_search.py --collect --out <dir>
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("POE_REPO", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from concept_pair_sheet import NOISE_PAIR, concept_slug  # noqa: E402

# The pairs the search runs over. Cat by dog is the one every earlier result is measured on; the other
# three give the learned map in step D something to generalise from, and elephant by penguin is held
# out of that fit entirely.
PAIRS = [("dog", "cat", "dog and cat"),
         ("tiger", "dog", "tiger and dog"),
         ("lion", "meerkat", "lion and meerkat"),
         ("elephant", "penguin", "elephant and penguin")]

# The bar to beat, measured on the same seeds: "a dog next to a cat" reached two separate animals on 5
# of the 7 judgeable seeds. A candidate is interesting only above that, and only at enough seeds for
# the difference to be readable, which round one is not. Round one shortlists; it decides nothing.
BEST_HAND_PICKED = {"phrase": "a dog next to a cat", "two_animals_of_7": 5}

PLURALITY = ["two animals", "a pair of animals", "two different animals", "two separate animals",
             "two creatures", "a duo of animals", "both animals", "two pets", "a group of two animals"]
SPATIAL = ["side by side", "next to each other", "beside one another", "one on the left and one on the right",
           "apart from each other", "a few feet apart", "at opposite ends of the frame",
           "not touching", "separated", "in different parts of the picture"]
SCENE = ["together in one photograph", "in the same room", "sharing a garden", "on a wide lawn",
         "a wide shot of two animals", "a full-body photograph of two animals",
         "a wildlife photograph of two animals", "two animals posing for a portrait"]
COUNTED = ["{a} and {b} together", "one {a} and one {b}", "a single {a} beside a single {b}",
           "{a} on the left, {b} on the right", "two animals: a {a} and a {b}",
           "a photograph of a {a} and a {b} apart from each other"]
# Phrases that name the failure, for the unconditional branch. The branch is subtracted, so a phrase
# here is a description of what the picture must NOT be.
FAILURES = ["a single animal", "one animal", "a hybrid animal", "a chimera", "a crossbreed",
            "one creature with features of two animals", "a blended animal",
            "a single animal with mixed features", "a close-up of one animal's face",
            "a portrait of one animal"]


def candidates(a: str, b: str) -> list[dict]:
    """The candidate list, built the same way every run so a shard index always means the same thing."""
    out = []
    for group, items in (("plurality", PLURALITY), ("spatial", SPATIAL), ("scene", SCENE)):
        out += [{"kind": "prompt", "group": group, "text": t} for t in items]
    out += [{"kind": "prompt", "group": "counted", "text": t.format(a=a, b=b)} for t in COUNTED]
    out += [{"kind": "mix", "group": "mix", "lam": round(l, 3)}
            for l in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]
    out += [{"kind": "negative", "group": "negative", "text": t} for t in FAILURES]
    # The anchors: what has already been measured, so every shard's numbers can be read against them
    # without trusting a count made in another session on another day.
    out += [{"kind": "prompt", "group": "anchor", "text": "a dog next to a cat"},
            {"kind": "prompt", "group": "anchor", "text": f"{a} and {b}"},
            {"kind": "none", "group": "anchor"}]
    for i, c in enumerate(out):
        c["id"] = i
    return out


def tile(root: Path, pair_slug: str, cid: int, seed: int) -> Path:
    return root / "tiles" / pair_slug / f"cand{cid:03d}" / f"seed{seed:02d}.png"


def render(args) -> None:
    import torch

    from poe_repair.composers._helpers import init_latents_for_cell
    from poe_repair.experiments.interaction_term.cell import cell_from_slug
    from poe_repair.methods._sampling import add_time_ids, write_decoded_image
    from poe_repair.run import make_ctx
    from poe_repair.runtime import (ddim_prev_from_x0_eps, decode_latents, encode_prompt_sdxl,
                                    tweedie_mean)

    ctx = make_ctx()
    assert torch.cuda.is_available(), "no CUDA device"
    unet, sched, w = ctx.models["unet"], ctx.scheduler, ctx.guidance_scale
    enc = lambda p: tuple(t.float() for t in encode_prompt_sdxl(p, models=ctx.models,
                                                                device=ctx.device, dtype=ctx.dtype))
    args.out.mkdir(parents=True, exist_ok=True)

    for a, b, joint in PAIRS:
        pair_slug = f"{concept_slug(a)}__x__{concept_slug(b)}"
        cands = candidates(a, b)
        mine = [c for c in cands if c["id"] % args.of == args.shard]
        e_a, e_b, e_j, e_0 = enc(a), enc(b), enc(joint), enc("")
        print(f"{pair_slug}: {len(mine)} of {len(cands)} candidates on shard {args.shard}", flush=True)

        for c in mine:
            # Every candidate is one choice of "what are the conditions" and "what is subtracted".
            if c["kind"] == "prompt":
                conds, null = [e_a, e_b, enc(c["text"])], e_0
            elif c["kind"] == "mix":
                lam = c["lam"]
                conds = [e_a, e_b, tuple((1 - lam) * z + lam * j for z, j in zip(e_0, e_j))]
                null = e_0
            elif c["kind"] == "negative":
                conds, null = [e_a, e_b], enc(c["text"])
            else:
                conds, null = [e_a, e_b], e_0

            for seed in args.seeds:
                out = tile(args.out, pair_slug, c["id"], seed)
                if out.exists():
                    continue
                cell = cell_from_slug(NOISE_PAIR, seed)
                init, sigma = init_latents_for_cell(cell, ctx)
                embs = conds + [null]
                seq = torch.cat([s for s, _ in embs], dim=0).to(ctx.dtype)
                pool = torch.cat([p for _, p in embs], dim=0).to(ctx.dtype)
                n = len(embs)
                cond = {"text_embeds": pool,
                        "time_ids": add_time_ids(height=cell.height, width=cell.width, batch_size=n,
                                                 device=ctx.device, dtype=ctx.dtype)}
                sched.set_timesteps(ctx.num_inference_steps)
                lat = (init / sigma).to(device=ctx.device, dtype=ctx.dtype)
                with torch.no_grad():
                    for i, t in enumerate(sched.timesteps):
                        eps = unet(sched.scale_model_input(lat.repeat(n, 1, 1, 1), t), t,
                                   encoder_hidden_states=seq, added_cond_kwargs=cond,
                                   timestep_cond=None).sample
                        e_u = eps[-1:]
                        eg = e_u + w * sum(eps[k:k + 1] - e_u for k in range(n - 1))
                        abar = sched.alphas_cumprod[int(t.item())].to(device=ctx.device, dtype=ctx.dtype)
                        lat = ddim_prev_from_x0_eps(scheduler=sched, timestep=t, step_index=i,
                                                    x0=tweedie_mean(lat, abar, eg), eps=eg)
                    img = decode_latents(ctx.models, lat).cpu()
                out.parent.mkdir(parents=True, exist_ok=True)
                write_decoded_image(img, out)
                gc.collect()
                torch.cuda.empty_cache()
            print(f"  cand {c['id']:03d} {c['group']:<9} {c.get('text', c.get('lam', ''))}", flush=True)

    del unet
    gc.collect()
    import torch as _t
    _t.cuda.empty_cache()
    score(args)


def score(args) -> None:
    """Count distinct animals in every tile this shard rendered."""
    from poe_repair.experiments.compose_scorer_validation.detection_scorer import count_instances

    rows = []
    for a, b, joint in PAIRS:
        pair_slug = f"{concept_slug(a)}__x__{concept_slug(b)}"
        cands = candidates(a, b)
        for c in [c for c in cands if c["id"] % args.of == args.shard]:
            counts = {}
            for seed in args.seeds:
                p = tile(args.out, pair_slug, c["id"], seed)
                if p.exists():
                    counts[seed] = count_instances(p)[0]
            if counts:
                rows.append({**c, "pair": pair_slug, "a": a, "b": b, "counts": counts,
                             "two_or_more": sum(v >= 2 for v in counts.values()),
                             "of": len(counts)})
                r = rows[-1]
                print(f"  {r['pair']:<22} cand {r['id']:03d} {r['group']:<9} "
                      f"two or more animals on {r['two_or_more']}/{r['of']}", flush=True)
    (args.out / f"scores-shard{args.shard}.json").write_text(
        json.dumps({"shard": args.shard, "of": args.of, "seeds": args.seeds,
                    "beat_this": BEST_HAND_PICKED, "rows": rows}, indent=1))
    print(f"wrote {args.out / f'scores-shard{args.shard}.json'}")


def collect(args) -> None:
    """Every shard's scores in one table, best first, with the anchors kept visible."""
    rows = []
    for f in sorted(args.out.glob("scores-shard*.json")):
        rows += json.loads(f.read_text())["rows"]
    rows.sort(key=lambda r: (-r["two_or_more"] / max(r["of"], 1), r["id"]))
    print(f"{'pair':<22} {'group':<9} {'two or more':>12}  candidate")
    for r in rows[:40]:
        print(f"{r['pair']:<22} {r['group']:<9} {r['two_or_more']:>5}/{r['of']:<6} "
              f"{r.get('text', r.get('lam', 'no third expert'))}")
    (args.out / "scores-all.json").write_text(json.dumps(
        {"beat_this": BEST_HAND_PICKED, "shards": len(list(args.out.glob('scores-shard*.json'))),
         "rows": rows}, indent=1))
    print(f"\nwrote {args.out / 'scores-all.json'} ({len(rows)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    if a.collect:
        collect(a)
    elif a.score_only:
        score(a)
    else:
        render(a)
