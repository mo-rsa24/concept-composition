# Design: is the adapter's correction a dial or a switch on pairs it never saw? (step Z10)

## What this asks

The pooled adapter is a low-rank correction to the product of experts, trained on seen pairs. This step adds its correction at weights 0, 0.25, 0.5, 0.75, 1 and 1.25 on eight animal pairs whose animals it never trained on, and asks how the share of pictures with one subject, a hybrid, or two clear subjects moves with the weight. A gradual change would make co-occurrence strength a dial that a 1-to-10 strength can be mapped through; a jump within one small step would make it a switch, as SuperDiff's own term was between 0.25 and 0.75. It is Loop 15's experiment 3 ([the directions note](../../directions/04-limits-and-what-to-test.md), "Three experiments, ranked"), run for board step Z10 after the owner's answer to board question 112 (run 1, then 2, then 3; only 3 once 1 said the checker is wrong, which M14 did).

## Words this uses

- **The adapter's weight** $\lambda$: the sampled prediction is $\epsilon_{\text{off}} + \lambda(\epsilon_{\text{on}} - \epsilon_{\text{off}})$, the product of experts with the adapter switched off and on.
- **One, hybrid, two clear**: a blind reviewer's call on a finished picture, on [Loop 6's rubric](../../method-search/01-beat-the-baseline/blind-count-rubric.md), drawing style ignored (board question 92): one subject; one body mixing both; two separate subjects each clearly the named animal. Two bodies that are not both clear are called `two_bodies` and reported.

## The adapter

The paper's adapter, `v58-03-self` at step 30,000 (`/datasets/mmolefe/poe_repair_min/outputs/showcase/cohort3/v58-03-self/checkpoints/lora_step_030000.pt`), the one the frozen held-out set is scored with (`1-lora-correction/report/paper-review/held-out-set.md`). It adapts both cross-attention and self-attention at rank 32, so it is loaded with `poe_repair/experiments/_adapter_shape.attach`, which reads the shape from the checkpoint and refuses to load half an adapter; the scope-06 loader attaches cross-attention only and would drop the other half in silence. It was trained on the 48 pairs of `artifacts/_shared/cross_pair_pool_configs/pair_pool_v58.yaml`.

## The pairs

**None of the frozen held-out pairs is used.** Of the 53 held-out pairs, 20 are animal pairs, and only 3 of those share no animal with the adapter's training pairs (seal × walrus, baboon × gazelle, goose × swan). The directions note's count of 23 held-out pairs sharing none came from a summary and holds only when object pairs are counted, which share no animal by kind. So the owner chose, through the manager loop (2026-11-25), eight new animal pairs instead, which also keeps the frozen evaluation set unspent and Loop 12's rule against held-out pairs and benchmark sets.

**The rule.** From the stated list of common animals in [the pairs file](pairs.json), drop every animal named in `pair_pool_v58.yaml`; form every pair of the rest; keep a pair only if the subject guard (`scripts/shared/disallowed_subjects.py`) allows its joint prompt and it appears, in either order, in none of the held-out set, the Attend-and-Excite prompts (`data/prompt-sets/attend-and-excite/a.e_prompts.json`), Loop 12's splits (`2-classifier-guidance/scripts/checker_choice/splits.json`) or the pool file; order the joint prompts by SHA-256 and take the first eight in which no animal repeats. "Bat" and "mole" were removed from the list after the first draw and before any render, because each has a common meaning that is not an animal and would confuse the calls.

| Pair | Joint prompt |
|---|---|
| 1 | a fox and a llama |
| 2 | a butterfly and a crab |
| 3 | a lobster and a whale |
| 4 | a bee and an owl |
| 5 | a kangaroo and a squid |
| 6 | a flamingo and a sloth |
| 7 | a goat and a rhinoceros |
| 8 | a ladybug and an octopus |

**Seeds** 6001 to 6004, the per-seed starting noise every pair shares; none of the held-out set's seeds (9 to 16) or SDXL's held-out seeds (7001 to 7008).

## The run

Plain product of the two single-animal prompts, classifier-free guidance 7.5 per expert, 50 DDIM steps, 1,024 pixels, with the adapter's correction at each of the six weights. 8 pairs by 4 seeds by 6 weights, 192 pictures, in four `bigbatch` jobs of two pairs each (the owner lifted the one-job rule for this step), every job refusing any card that is not an RTX 3090 so all pictures come from one card type; control 3 runs in the first job, and the outputs merge before scoring.

**Two copies of the U-Net.** The adapter's off switch does not fully switch this checkpoint off: on fox × llama at seed 6001, the picture with the adapter attached and switched off differs from the plain product by a mean of 10.8 grey levels (largest 231), against 34.5 with it switched on, while the plain and the guided samplers agree exactly with no adapter (job 95428, `diagnose.json`). So $\epsilon_{\text{off}}$ comes from the untouched U-Net and $\epsilon_{\text{on}}$ from a copy carrying the adapter (`run_two_unet` in `2-classifier-guidance/scripts/leakage/z10_dial.py`); at weight 0 the copy is not called.

## Controls

1. **Weight 0 is the plain product.** On the first pair and seed, in one process: the picture at $\lambda = 0$ must equal `run_cfg_poe`'s byte for byte, and `run_cfg_poe` rendered again after the copy receives the adapter must too, so attaching to the copy leaves the original untouched.
2. **The whole adapter is loaded.** `_adapter_shape.attach` must match every saved tensor, or the run stops.
3. **The adapter still works where it trained.** One training pair from the pool file (the first under its `train:` key), seeds 6001 to 6004, at $\lambda$ 0 and 1: 8 pictures, called blind with the rest and reported beside the curve, with no bar.

## How pictures are counted

All 200 pictures under neutral ids, the key held apart; two blind reviewers call each one, hybrid, two bodies or two clear, with `broken` marked; a third blind call settles each picture on which the two differ in that call.

## The bar

Let $T(\lambda)$, $H(\lambda)$ and $O(\lambda)$ be the counts of two-clear, hybrid and one-subject pictures out of 32 at weight $\lambda$, and $\Delta = T(1) - T(0)$.

| Result | What it takes |
|---|---|
| No effect on new pairs | $\Delta < 6$, or an exact one-sided McNemar test on the 32 matched pictures at $\lambda$ 1 against 0 gives p ≥ 0.05 |
| A dial | the effect holds, no single step of 0.25 between 0 and 1 carries more than half of $\Delta$, and $T$ never falls by more than 2 from one weight to the next between 0 and 1 |
| A switch | the effect holds and one step of 0.25 between 0 and 1 carries at least three quarters of $\Delta$ |
| In between | the effect holds and neither dial nor switch applies |
| Inconclusive | control 1 or 2 fails |

$H(\lambda)$, $O(\lambda)$ and the weight 1.25 are reported on the same plot with no bar of their own.

## What it shows

One plot: the share of one, hybrid and two-clear pictures (out of 32) against the weight, one line each, from `calls` and the key, saved with the numbers file beside it; and one sheet per pair, one row per weight, four seeds across.

## Cost

192 pictures plus 8 for control 3. The smoke run measured 51 seconds per picture at a weight above 0 (both U-Nets) and 24 at weight 0 on an RTX 3090 (job 95430), 2.83 GPU-hours in all, which the owner approved (2026-11-25, through the manager loop). `bigbatch`, one job, excluding `mscluster[44-45,48,50-51,57,61,65,68,71-76,82-83]`; W&B project `prime_lab/poe-repair-animals-compose`. Two blind reviewers plus a third. No API tokens.

## What this cannot say

Eight pairs and four seeds: steps of 0.25 hold 32 pictures each, so the dial-against-switch call is coarse. One adapter checkpoint. New pairs here mean no animal shared with training; how near the new animals are to trained ones in SDXL's text space is not measured.
