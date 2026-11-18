# 🧪 Does the recipe work on other pairs

## Position in the plan tree

[Root](../../MASTER_PLAN.md) › [the classifier-guided PoE route](route-classifier-guided-poe-from-toy-to-animals.md) ›
this plan, after [plan 29](29-does-the-early-push-hold-on-new-seeds.md). Its pass marks are in
[its review file](review/30-does-the-recipe-work-on-other-pairs.md).

## Quick context

**Why this plan exists**

The recipe works on cat × dog: a separate cat and dog on 28 of 32 new seeds against 1 for plain PoE. One pair cannot
make it a method. Lion × meerkat is a second animal pair the project already uses; typewriter × cactus is an object pair.

**What it does**

Step 0 runs the recipe unchanged, with the checker trained only on cats and dogs, on both pairs. If it separates them,
part of what the checker learnt ("two things, not one fused thing") carries across pairs. If not, step 1 trains a
checker across many pairs, from plan 23's 80 pairs plus new joint-prompt runs, and is planned after step 0's answer.

**What was tried before**

Plan 22 trained a checker on 81 clean pictures across 38 pairs; it failed its accuracy check at every step. The cat × dog
checker worked only after about 500 clean examples of one pair.

## Environment Facts This Plan Depends On

- mscluster111 and mscluster112 are Blackwell cards driven only by `co3_bw` ([the node table](../../environment/hpc/nodes.md)).
- Outputs go under `POE_REPAIR_MOUNT=/home-mscluster/mmolefe/poe_repair_min_scratch` while `/datasets` is down
  ([storage](../../environment/storage.md)).
- The starting noise per seed is shared across pairs; seeds with no cached noise draw it from the seed.

## Tasks

- [x] **0.1 Code.** `render_results.py --prompts A B`, the checker reading with "A and B".
- [ ] **0.2 Draw.** `rung2/pairs-step0.sh`: lion × meerkat, then typewriter × cactus, seeds 3301 to 3316.
- [ ] **0.3 Call.** Every full-size tile, pushed, plain and random, into the review file's Q1.
- [ ] **1.1 Draw.** `rung2/pair-runs.sh` on both cards: `render_pair_runs.py`, 78 pairs × 40 joint-prompt runs (seeds 5001 to
  5040), states and pictures under `pair-runs/`; after it, list claims without `done` and rerun them.
- [ ] **1.2 Measure the labelling cost.** Label 50 pictures with `api_label.py --strict --real-only` on the laptop, report the
  cost, and wait for the user's yes before labelling the other 3,070.
- [ ] **1.3 Label, train, push.** Then the checker across pairs and the push on the two test pairs, read against the review
  file's Q2.

