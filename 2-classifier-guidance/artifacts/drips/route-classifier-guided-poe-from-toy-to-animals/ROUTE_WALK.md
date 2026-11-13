# Route walk: classifier-guided PoE, from the toy to SDXL animals

Builds [the route](../../../plans/route-classifier-guided-poe-from-toy-to-animals.md) one decision
per round, with `/drip-skill-route --research`. Opened 2026-11-12.

## Position

Closed: all rounds settled, route emitted.

## Round 1: the sentence and the disk

**The goal.** The paper can say whether a sampling-time correction (a small classifier's gradient
added at each product step, SDXL frozen) can stand in for the trained LoRA. Stated in the sentence.

**The objective.** At each rung of a six-rung ladder, a bar written before the rung runs is passed
or missed: on the toy, the guidance's cosine with the true correction and the crisp-pair count of
300; on SDXL, the count of two separate animals out of the held-out seeds, beside best-of-N at equal
compute and the joint prompt, and on rungs 5 and 6 beside the rank-32 pooled LoRA. Inferred from
the sentence's ladder.

**The outcome.** A comparison table and sheet for the paper (plain product, best-of-N reranking,
guided product, joint prompt, LoRA) on held-out pairs, plus a finding in `report/` per rung, or the
rung at which the climb stopped and its diagnosis. Inferred.

### What the disk holds

| Path | What it is | Bearing on the route |
|---|---|---|
| `plans/06-.../hypothesis/15-can-a-classifier-supply-the-missing-term.md` | plan 15, presence head | tasks 1.1 to 3.1 ticked; 3.2 (probe), 3.3 (launch) and piece 4 open; never trained |
| `poe_repair/methods/_classifier_guided.py:112` | plan 15's sampler | rescales the gradient to the product prediction's own norm, the rule dog-01 broke |
| branch `exec/can-a-classifier-supply-the-missing-term`, worktree in a stale scratchpad (marked prunable) | plans 17 and 18 | exist only on this branch, unmerged; not in `plans/` on any checked-out branch |
| plan 17 on that branch | the Algorithm 2 sampler with nothing rescaled | the sampler every rung should use |
| plan 18 on that branch | dog-01, W&B `z2k1bdmi` | accuracy check failed (7 of 20 steps at 0.75 or more); clean dog 2 of 4, a dog on all 4 |
| plan 18, commit `6be7d0c5` | cat × dog product pushed toward "dog present" | two animals in 0 of 16 tiles; the fused animal turned into a cat on 3 of 4 seeds |
| `2-classifier-guidance/artifacts/drips/can-a-classifier-supply-the-missing-term/EXECUTE_WALK.md` | plan 15's walk map | stale: says piece 1 in flight |
| `3-understanding-compositionality/artifacts/results/the-exact-toy/how-much-co-occurrence-the-product-needs/README.md` | the toy card | the true noised product's score is closed form, so rung 1 has an exact answer |
| `3-understanding-compositionality/scripts/census/crop_classifier_cat_vs_dog.py` | CLIP on the detector's padded crop | species instrument, 64 of 64 at median margin 0.990 (plan 15's quick context) |
| `report/00-INDEX.md:155-156` | the reranking rows | plain product best-of-16 composed 0 of 128 particles; with the adapter at λ 0.5, best of 16 is 8 of 8 |

### Where the sentence and the disk disagree

1. The brief says plan 15 "has started the first classifier". Its head is written and never
   trained, and its sampler still rescales to the prediction's norm.
2. The brief says plan 18 is absent and must be located. It is on the unmerged branch above, and its
   last result is already a version of rung 3: one species classifier on the cat × dog product, 0 of
   16 two animals.
3. The brief says nothing in the repo tells cat from dog. The CLIP crop classifier does, measured on
   64 renders; it has not been tested on overlapping boxes.
4. The brief says best-of-16 composes 8 of 8. That is with the adapter attached. On the plain product
   best-of-16 composed 0 of 128 particles, so on rungs 3 and 4 reranking is a weak baseline and the
   strong one is the adapter plus best-of-N.

## Decisions

| Round | What | Chosen | Options there were | Who chose | Why |
|---|---|---|---|---|---|
| 1 | the classifier branch | merge it first; plan 18's 0-of-16 product sheet is rung 3's one-species baseline | leave it and cite it; fold it into plan 15 | user | recommended option taken |
| 1 | the push's size | plan 17's Algorithm 2 push with nothing rescaled, capped at a fraction κ of the product's own step; plan 15 repaired to this before it trains | no cap until the controller rung; keep both and compare | user | recommended option taken |
| 1 | rung 1's classifier | a discriminator, joint pictures against the product's own draws, scored against the true correction; presence and instance heads come later as the research question 2 comparison | presence head first; both on rung 1 | user | recommended option taken |
| 1 | where the SDXL classifier reads | noisy latents; DINOv2 on the decoded estimate held as an ablation on rung 4 if the latent head misses | decoded estimate from rung 3; decide at rung 3 | user | recommended option taken |
| 2 | the ladder's structure | three phases: setting climbs with the architecture held (A1 to A4), parts added one per rung at A4's setting (B1 to B7), best configuration carried up (C1, C2) | phase B on the toy first; parts added where each first appears | user, `a` | "Do we scaffold the architecture properly and by adding one variable at a time holding others constant?" |
| 2 | classifier diagnosis | every classifier rung runs D1 minimal-pair gap against a shuffled-label floor, D2 single-head push decoded from 4 hybrid states, D3 cosine with the correction per step, D4 LDA picture; beside the guided sweep, never gating it | D1 gates the rung; diagnosis at A1, A3 and C2 only | user, `a` | "Do we have plans to diagnose the classifiers to ensure they are looking at the right thing?" |
| 2 | rungs between the toy and SDXL | A2b MNIST digits two to a canvas and A2c animal photos at 64×64 with pasted pairs, both keeping the both-share dial; phase B stays at A4 | MNIST only; animals only at 224 through a latent model; phase B at A2c | user, `a` | "we can even use MNIST, then animal images of size 224 or smaller?" |
| 2 | the outcome board | settled: eighteen rows, R0, A1, A2, A2b, A2c, A3, A4, B1 to B7, C1, C2, P1, X1 | | user, by picking `a` on its last open question | |
| 3 | how much planning | a new scope (number rechecked before writing, 12 free on 2026-11-12), master plan holding the ladder with provisional bars, plan files written one rung ahead (R0 and A1 now), each with a review file; plans 15, 17, 18 stay in scope 06 and are cited | no scope, each rung a request walk; a full tree now | user, `a` | |
| 4 | the chain, and step 1 | flow 20 spliced with the research spine's read layer, seven steps; step 1 a lean read of 2211.17091 | | user, `settle` | |
| 5 | step 2 | frame the hypothesis over the whole ladder, A1's bar firm and the rest provisional | | user, `settle all` | "you can settle all of them" |
| 6 | steps 3 to 7 | hypothesis-to-scope, populate-plans for R0 and A1, walk R0, walk A1, walk A1's retrospective | | user, `settle all` | |
| 7 | time and schedule | about 8 h to A1's close, 13 h budget at 2× on steps 1 and 6, 10 sittings at three 90-minute sittings a week (assumed) | | user, `settle all` | |
| 8 | check and emit | route and references written; board drawn | | user, `settle all` | |
