# Design: can the checker's push fix a known failure at all? (step Z8)

## What this asks

Paper 2's candidate correction is the kept early push: the multi-pair readout checker (`readout_pair/readout-multipair/head_step_002000.pt`) pushing the sampler at strength 30, the push capped at 0.4 of the prediction's length, on sampler steps 0 to 15 only. This step asks whether it turns a known failure into two clear subjects on each of three base samplers, more often than a random push of the same size does. It is the first test in the [four-way design](../../manager/paper2-four-way-design.md), section "First: can the correction fix a known failure at all?", run for board step Z8 at the owner's request through the manager loop. It shows possibility, not a rate: no number from it goes in a paper.

## Words this uses

- **The push**: the checker's gradient step above, applied to the base sampler's prediction at steps 0 to 15.
- **The random push**: a fresh random direction at each step, scaled to the length the push had at that step in the matching pushed run.
- **Two clear**: both named subjects present as two separate, clearly identifiable things, each with its own parts, none extra or missing ([Loop 6's rubric](../../method-search/01-beat-the-baseline/blind-count-rubric.md), drawing style ignored by board question 92).

## The three bases

| Base | What it is | Code |
|---|---|---|
| Plain PoE | $\epsilon_\varnothing + w(\epsilon_A-\epsilon_\varnothing) + w(\epsilon_B-\epsilon_\varnothing)$, $w = 7.5$, 50 DDIM steps | `run_poe_classifier_guided` (`poe_repair/methods/_classifier_guided.py`), the kept recipe's sampler |
| The joint prompt | $\epsilon_\varnothing + w(\epsilon_J-\epsilon_\varnothing)$ on "a {A} and a {B}", $w = 7.5$, 50 DDIM steps | the same sampler with the joint prompt as both experts at guidance $w/2$, which is the same sum; the identity check below proves it on the card |
| CO3 | the released sampler at its own settings | Loop 6's `paper2_rows.py` and `co3_push.py`, unchanged |

## The identity check, before any screening picture

The joint prompt drawn through the product sampler (joint prompt as both experts, guidance 3.75, strength 0) against `run_cfg` at guidance 7.5, both at the model's own precision, on the first pair and seed. **It passes when the mean absolute pixel difference is under 1 grey level of 255**; the largest difference is reported. A failure stops the run.

## The cases

Eleven, from the 11 tuning pairs other than giraffe × tortoise (`2-classifier-guidance/scripts/checker_choice/splits.json`), which lie outside the checker's 243 training pairs and the benchmark pool. Seeds are tuning seeds; no held-out seed (7001 to 7008) is used.

**CO3, five cases from W2d's labels** (`noise-text-critic/02d-binding/part-binding-calls.csv`, both labellers agreeing):

| Case | Pair | Seed | W2d's call |
|---|---|---|---|
| CO3 failure 1 | cheetah × cougar | 6001 | borrowed: the cougar has cheetah spots and tear lines |
| CO3 failure 2 | puma × condor | 6001 | borrowed: the condor has a puma's head |
| CO3 failure 3 | duck × goose | 6002 | borrowed: the goose has a mallard's green head |
| CO3 right 1 | zebra × flamingo | 6001 | own parts: clean zebra and flamingo |
| CO3 right 2 | eagle × fox | 6001 | own parts: clean eagle and fox |

**Plain PoE and the joint prompt, three failures each, picked from the screening pass.** The screening pass draws plain PoE and the joint prompt on the 11 pairs at seeds 6001 to 6004 (88 pictures) on mscluster107 card 1 under board question 103 (`2-classifier-guidance/scripts/leakage/z8_screen.py`). The pick rule is fixed here, before any screening picture is seen: for each base, walk the pairs in `splits.json` order and, within a pair, the seeds in order; the first picture that is not two clear is that pair's case; stop at three pairs. Each pick is recorded with what failed (one subject, a hybrid, two of a kind, borrowed parts). If fewer than three pairs fail, the base gets as many cases as there are, and its bar scales (below).

## The arms

Each case is drawn on its own base three ways, in one job: **without** (the base alone, once), **with the push** (four draws), **with the random push** (four draws). Four draws because pushed renders at one seed have differed between runs. The screening pictures and W2d's CO3 pictures are not reused as the "without" arm, because Z8 proper runs in `bigbatch` on a card type that may differ; each "without" is drawn again in the same job.

99 pictures: plain PoE and the joint prompt, 3 cases × 9 each (54 at about 30 seconds); CO3, 5 cases × 9 (45 at about 60 seconds, the "without" row included), about 75 GPU-minutes with the model loads.

## How pictures are counted

Blind, as in Z3 and Z7: all pictures under neutral ids, the key held apart, two reviewers calling each on the rubric's ladder (one, hybrid, two bodies, two clear) with `broken` marked, a third blind call settling each disagreement on two clear.

## The bar

| Result | What it takes |
|---|---|
| A base passes | the push gives at least 4 more two-clear pictures than the random push, out of that base's 12 (three cases, four draws); with fewer cases, at least a third of its draws more |
| Go to the four-way test | plain PoE and the joint prompt both pass, and on the two CO3 cases already right the push leaves no more pictures short of two clear (out of 8) than the random push does |
| Parked | otherwise; the eleven grids are the evidence |

CO3's three failures are reported beside M1's result on all 96 tuning pictures with one draw each: faithful 7 with the push, 12 with the random push, 11 for CO3 alone ([M1's review](../../method-search/01-beat-the-baseline/review.md)).

## What would surprise

The push fixing CO3's borrowing cases when M1 found it no better than the random push on CO3. The joint prompt failing to show any failure in 44 screening pictures, which would leave it with fewer than three cases.

## Cost and what it buys

**Cost:** the screening pass, about 35 GPU-minutes on mscluster107 card 1 (question 103); Z8 proper, about 75 GPU-minutes in `bigbatch`, excluding `mscluster[44-45,50-51,57,61,65,68,71-76,82-83]`. Two blind reviewers plus a third. No API tokens.

**Buys:** whether the correction can fix a failure at all on each base, before any larger test is sized. A pass on plain PoE and the joint prompt sends Paper 2 to the four-way test; anything else parks the correction with the grids as the record.

## What this cannot say

Three cases per base, chosen as failures, so no rate. Tuning pairs and seeds only. One push recipe (strength 30, cap 0.4, steps 0 to 15).
