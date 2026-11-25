# Design: does each animal's own prediction, applied as a push inside its box, beat CO3 alone? (step Z12, stage A)

## What this asks

Z11 trained a checker whose push points along a direction supplied by the expert that saw the pair, and it beat every control on combinations no expert saw ([Z11's review](../11-gradient-trained-classifier/review.md)). On CO3 no model has seen the pair, so the direction has to be built: inside each animal's box, that animal's own prediction minus CO3's. Before any classifier is trained to copy that direction, stage A asks whether the direction itself, applied as a push with the boxes given, draws more pictures with two clear subjects than CO3 alone. The reviewer's verdict and fixes are "The aligned classifier on CO3, 2026-11-25" in `reviewer/2026-11-24-review.md`; the owner's go and the manager's line are in `requests.md` (2026-11-25). Stage B, the classifier, waits for stage A to pass and for the owner's yes.

## Words this uses

- **CO3's prediction**, $\epsilon_{\text{CO3}}$: the guided noise prediction of CO3's one two-entry UNet call per step, `uncond + guidance * (cond - uncond)`, which the push changes (`2-classifier-guidance/scripts/method_search/co3_push.py`, `Pusher`).
- **An animal's own prediction**, $\epsilon_A = \epsilon_\varnothing + \lambda(\epsilon(x_t \mid \text{"a duck"}) - \epsilon_\varnothing)$ at CO3's own guidance $\lambda$, as Z7 formed it.
- **The box direction** at sampler step $s$ from 5 to 29:

$$T(x_t) = M_A \odot \big(\epsilon_A(x_t) - \epsilon_{\text{CO3}}(x_t)\big) + M_B \odot \big(\epsilon_B(x_t) - \epsilon_{\text{CO3}}(x_t)\big)$$

  $M_A$ is 1 on latent cells inside A's box and outside B's, 0 elsewhere; where the boxes overlap and outside both, $T$ is 0.
- **The box push** moves CO3's prediction by $L_s \cdot T / \lVert T \rVert$: toward each animal's own prediction inside its box, by a length $L_s$ fixed per step before any pushed render, capped at 0.4 of $\lVert\epsilon_{\text{CO3}}\rVert$ (the kept cap), with every firing of the cap counted.

## Rows

| Row | What is sampled | Why |
|---|---|---|
| 1. CO3 alone | released settings, no push | the baseline, and the pictures the boxes are read from |
| 2. The box push, boxes given | CO3 with the box push at steps 5 to 29 | whether the direction a classifier would copy helps at all |

**Pairs and seeds.** The 12 tuning pairs of `2-classifier-guidance/scripts/checker_choice/splits.json` at seeds 6017 to 6024: 96 pictures a row, none drawn before. Giraffe × tortoise is rendered and counted and never placed on a sheet. No held-out pair, held-out seed (7001 to 7008) or benchmark prompt is used.

**Card.** Both rows on `bigbatch` RTX 3090 cards, the jobs refusing any other card, whole nodes, the faulted nodes excluded.

## Where the boxes come from

**From instruments, never by eye.** For each finished picture, Grounding DINO (the repo's `count_instances`, queried with the pair's two nouns) places the bodies; the two largest kept boxes are named by zero-shot CLIP ViT-B/32 on each padded crop, as a two-way choice between the pair's two nouns, and the two names are assigned by the permutation with the higher summed log-probability. This is the protocol of `3-understanding-compositionality/scripts/layout_from_joint_render.py`, whose CLIP naming passed 64 of 64 crops there; Grounding DINO alone cannot name the animal (it called 64 cat renders "dog"). With one kept box, that box is named and the other is empty; with none, the push is zero for that picture, which is counted. The boxes come from row 1's finished picture of the same pair and seed, so they are a best case for any reader of regions, said wherever a result is quoted.

**Checked against the eye before use.** The same instrument runs on Z3's 48 CO3 pictures (6 borrowing pairs, seeds 6001 to 6008), whose boxes were drawn by eye; the median overlap (intersection over union) of matched boxes and the share named the same are reported beside the result, deciding nothing.

## The push length, fixed before any pushed render

$L_s$ is the median of $\lVert T \rVert$ at step $s$ over 24 training trajectories: CO3 alone on the 12 tuning pairs at seeds 6001 and 6002, with boxes from the same instrument on Z2's finished CO3 pictures of those seeds (`/datasets/mmolefe/poe_repair_min/loop-12/z2/full/`), $T$ computed at every step and not applied. The 24 lengths are written to `lengths.json` before row 2 starts.

## Controls

| Control | What it does | What comes back |
|---|---|---|
| Empty boxes | row 2's code with both boxes empty | row 1's picture byte for byte on the same card, or the job stops |
| Entering step 5 | the latent entering step 5 in row 2 against row 1 | identical, since the push starts at step 5 |
| The lengths come from training seeds | $L_s$ from seeds 6001 and 6002 only | no test seed informs the push's size |

## How pictures are counted

As Z7: two clear subjects with their own parts, drawing style ignored, on [Loop 6's rubric](../../method-search/01-beat-the-baseline/blind-count-rubric.md), with `broken`, `faithful`, `damaged`, `a_own` and `b_own` marked. Two blind reviewers call all 192 pictures in one shuffled batch under neutral ids, the key held apart; a third settles each two-clear disagreement.

## The bar, written before any render

**Support** when all three hold:

1. Row 2 has at least 5 more `two_clear` pictures than row 1, of 96.
2. The matched-seed lead is above 1.96 standard errors: with $d_i$ the per-picture difference in `two_clear` (row 2 minus row 1), $\bar d / (\text{sd}(d)/\sqrt{96}) > 1.96$ (Paper 2's bar).
3. No more than 2 of row 1's `two_clear` pictures are not `two_clear` in row 2.

Otherwise **stop**: no classifier is trained, and the box route on CO3 is closed with Z7 and Z12 as its record. Reported beside the bar and deciding nothing: per pair, because duck × goose carried Z7; the per-subject own-parts counts; and how often the cap fired.

## A rule that can only stop

The owner's decision, 2026-11-25, through the manager loop. **The first half** is the first 48 box-push pictures by each job's own order: the first two pairs of each of the three jobs at all eight seeds, so six pairs by seeds 6017 to 6024 (puma × condor, zebra × flamingo, seal × gull, hippopotamus × swallow, giraffe × tortoise, bobcat × quail). Those 48 and the 48 CO3-alone pictures of the same pairs and seeds are called blind in one shuffled batch, by the rule in "How pictures are counted". **If the box push's two-clear count on those 48 is not above CO3 alone's on the same 48, a lead of 0 or less, the remaining box-push jobs are cancelled and stage A is reported as stopped for futility**, with no full count and no stage B. A lead above 0 decides nothing: the bar above, on all 96, stays the judge. The rule was added while the box-push jobs were drawing (they started at 15:54 UTC), before any box-push picture was looked at or called.

## Cost

| Part | RTX 3090 hours |
|---|---:|
| Lengths: 24 CO3 trajectories with $T$ computed, about 60 s each | 0.4 |
| Row 1: 96 at about 43 s | 1.15 |
| Row 2: 96 at about 56 s | 1.5 |
| Boxes: Grounding DINO and CLIP on 216 pictures | under 0.1 |
| **In all** | **about 3.1** |

W&B project `prime_lab/poe-repair-animals-compose`. No API tokens.
