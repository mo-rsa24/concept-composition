```text
/drip-execute-plan --request "Find out why the adapter's second animal takes on the first animal's
looks, and test the cheapest fix before any retraining"
Model: Opus, high effort. Box: 2 sittings, about 3 hours of your time. Filing: tied.

Background
The paper "Restoring plurality in compositional diffusion by training correction adapters" (ICLR
2027 submission 56173) trains a LoRA on SDXL so the product of two concept prompts draws both
things. The repo is /home/molef/PhD/poe_repair_min on the laptop. Runs happen on mscluster through
scripts/cluster.sh. Two animals now appear, but the second often looks like the first: lion and
meerkat at seed 1 gives a lion and a lion cub. Fidelity of each animal is the paper's weak point.
This walk tests two causes, cheapest first, and proposes a retrain only if both fail.

Objectives
1. Render 4 training pairs at seeds 9 to 16 with the adapter on for steps 0 to 9, 0 to 24 and 0 to 49.
2. Render the same 32 cells with v58-01-plain and v58-03-self, both at step 47,500, best window.
3. Label every tile by eye on the ladder below, from full-resolution tiles.
4. Say which cause the labels support, with the counts.
5. If neither fix reaches the target, write one retrain proposal and stop for a yes.

What to read
- artifacts/drips/answer-iclr-reviews/ground-check.md: the adapter of record and its training recipe.
- report/paper-review/prze-3a-pair-lists.md: the three runs behind Figure 5, with paths.
- report/what-the-adapter-ablations-show/does-adapting-self-attention-change-the-result.md: the
  earlier, mixed comparison of self-attention against cross-attention.
- report/paper-review/prze-8-protocol.md: why held-out pairs stay untouched here.
- scripts/figure_candidates.py: the renderer; --window N keeps the adapter on for steps 0 to N.
- poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py, around line 532: the step filter.
- environment/hpc/nodes.md and report/paper-review/prze-10-compute.md: nodes and pythons.
- /datasets/mmolefe/poe_repair_min/outputs/showcase/cohort3/v58-03-self/ and v58-01-plain/:
  config.json, pair_pool.json, dataset_meta.json, verdict.json, checkpoints/.

What is already known
- The adapter of record is v58-03-self, checkpoint lora_step_050000.pt, W&B run qtk5gi37
  (ground-check.md).
- It adapts self- and cross-attention query, key and value at rank 32, alpha 32 (its config.json).
- It trained 1,000 epochs of 50 optimiser steps on 48 pairs and 182 cells (config.json, verdict.json).
- It trained only on step indices 0 to 9 of 50 DDIM steps, timesteps 981 to 801 (config.json
  train_step_range [0, 10]; the cache meta.json).
- Every earlier render of it ran the adapter for all 50 steps or for 25 (the W&B sample panels).
- v58-01-plain (W&B qx5ffxzp) adapts attn2 query, key and value only. Its config.json matches
  v58-03-self on schedule, steps 0 to 9, x0 loss, orthogonality 1.0, EMA 0.9999 and contrast 0.
- v58-01-plain stopped at step 47,500 with no verdict.json; its last checkpoint is
  lora_step_047500.pt. v58-03-self also has lora_step_047500.pt, so compare the two at 47,500.
- The renderer takes each seed's starting noise from the a_cat__x__a_dog cache, which every pair
  shares per seed (scripts/figure_candidates.py line 40). The four pairs have no cache at seeds
  9 to 16 and need none.
- Training seeds are 1 to 8 and 17 to 58, so seeds 9 to 16 are unseen (dataset_meta.json).
- The held-out pairs are bison and horse, cat and dog, piano and candle, skateboard and helmet,
  elephant and penguin (the run's own pair_pool.json). Do not render or read them in this walk.
- Use these training pairs: a_lion__x__a_meerkat, a_camel__x__a_pelican, a_donkey__x__a_duck,
  a_chicken__x__a_rabbit (pair_pool_v58.yaml).
- No render with the adapter off after step 9 exists in /datasets/mmolefe/poe_repair_min/outputs/
  ablation_grid/ (checked 2026-11-10).
- The windowed sampler leaves the adapter attached after its window closes, so render the plain
  product in a separate process from any adapter render (memory, cost one re-render on 2026-09-05).
- GroundingDINO counts animals but cannot name the species; never use it to judge species.
- Labels made on downscaled contact sheets overcall clear pairs; label from full-resolution tiles.
- Laplacian variance measures edges, not blur; do not use it as a quality read.
- Usable GPU nodes on 2026-11-10: mscluster108 (two RTX 8000, co3 python), mscluster111 and
  mscluster112 (one RTX PRO 6000 Blackwell each, co3_bw python). mscluster110 is another user's
  until about 2026-11-12 11:30 (prze-10-compute.md).
- Every launcher must assert torch.cuda.is_available() after pinning the device (nodes.md).

The ladder for one tile, lowest to highest
0 one animal. 1 a hybrid of both. 2 two bodies, but one has the other's species look.
3 two animals, each its own species. Label "destroyed" for a broken image.

When something is missing
- If a render job fails on a node, read the log, try one other node, then report it.
- If the a_cat__x__a_dog cache lacks a seed from 9 to 16, drop that seed and say which.
- If the render time per tile is unknown, time the first pair and give the total before going on.

Rules
- Read freely; launch render jobs only after showing the command and getting a yes.
- Launch no training job. A retrain is a written proposal only.
- Write renders under /datasets/mmolefe/poe_repair_min/outputs/species-fidelity/ only.
- Write repo files under artifacts/results/why-the-second-animal-loses-its-species/ only.
- Commit nothing until the user says so.
- Stop and show the first pair's full row of tiles before rendering the other three.
- Stop and show the label table before writing any conclusion.

What to hand back
- artifacts/results/why-the-second-animal-loses-its-species/labels.csv: one row per tile, columns
  pair, seed, adapter, window, ladder_label, note.
- artifacts/results/why-the-second-animal-loses-its-species/README.md: how the tiles were made,
  one sheet per pair as thumbnails, the counts per setting, and the verdict.
- A one-line count command that reads labels.csv and prints the level-3 count per setting.
- New rows in "What we know so far" in plans/route-answer-the-iclr-reviews.md.
- If objective 5 runs: a retrain proposal with the one flag that changes and its cost in hours.

What you will see
Along the way: the first pair's row of tiles, then the label table, each with one question.
At the end: what was asked, what was done with paths, PASS or MISS, what is open, at most one
question. PASS means a setting reaches level 3 on at least 6 more of the 32 tiles than v58-03-self
at step 50,000 on all 50 steps, with no more than 2 tiles dropping below level 2. You judge from
labels.csv and the sheets, and it is done when every tile carries a label.
```
