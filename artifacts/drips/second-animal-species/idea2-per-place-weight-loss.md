# Idea 2: a loss on where each expert is turned up or down

## Why it exists

On 8 cells (4 training pairs, seeds 9 and 10), injecting only the part of the true correction that re-weights the two experts per 4×4 place, over steps 0 to 29, gives the same outcome as the full correction on 8 of 8 (7 bind, 1 fails with it). The same amounts applied as one weight per step, as a fixed 0.5, or moved to shuffled places all fail; the rest of the correction scaled to the same size does not bind; swapping the prompts changes nothing. So binding is carried by the per-place pattern of expert weights. The current loss charges every part of the correction equally and never asks for that pattern by name.

## What the trainer already has

`poe_repair/experiments/one_pair_one_seed/trainer.py` (around line 594) already splits the error into an in-span part (what re-weighting the two experts could supply) and an out-of-span part, using a span basis of the two experts' directions, and charges the out-of-span part `orth_weight` times. That span is global: two directions for the whole image. A global re-weighting is exactly the version that failed in the controls (one weight per step).

## The change

Make the span per place, and up-weight the in-span part instead of the out-of-span part.

1. Per 4×4 latent block, build the two-column basis from that block's d_A and d_B (orthonormalised, the same Gram-Schmidt the global basis uses, with a ridge for nearly parallel directions).
2. Project the error per block onto its basis: `in_place` (the per-place re-weighting the adapter got wrong) and `rest`.
3. Loss per sample: `(in_place_weight * ||in_place||² + ||rest||²) / D`, then the same x0-space weight as today. `in_place_weight = 1` reproduces today's loss exactly (the check that the code path is right).
4. One new flag, `--in-place-weight` (default 1.0), and one new logged number, the share of the error energy that is in-place.

No new network and nothing at sampling time: the LoRA runs exactly as now, without the joint prompt.

## Run

v58-03-self recipe on `cells_v58_bound.json`, steps 0 to 40 (the same as v58-06-bound-w40, so the two differ in this one setting), `--in-place-weight 4.0`. About 23 hours on an RTX 8000, about 8 on a Blackwell.

## Bar, written before it runs

Judge on the four pairs at seeds 9 to 16, adapter on steps 0 to 39, labels by eye from full-resolution tiles (ladder 0 to 3).

- Support: at least 6 more level-3 tiles out of 32 than v58-06-bound-w40 at the same step, with no more than 2 tiles dropping below level 2.
- Null: within 2 tiles of v58-06-bound-w40.
- Inconclusive: anything between.

Held-out pairs are rendered once, after the checkpoint is chosen on training pairs.

## What would surprise me

`in_place_weight` 4.0 doing worse than 1.0 on binding: that would mean the in-place error is already small and up-weighting it only crowds out layout.
