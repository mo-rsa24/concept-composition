# Route walk: classifier-guided PoE, from the toy to SDXL animals

Builds [the route](../../../plans/route-classifier-guided-poe-from-toy-to-animals.md) one decision
per round, with `/drip-skill-route --research`. Opened 2026-11-12.

## Position

Reopened 2026-11-17 with `--from` at round 2 to add the scale-up ladder. Closed again 2026-11-17: all rounds settled, route and board written. Reopened 2026-11-18 with `--from` to add the MNIST stage (step 7b): Closed again 2026-11-18: all eleven rounds settled, step 7b written. Queued after the close: the picture control in the drip controls (the user's `make it`).

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

## Reopened 2026-11-17: the scale-up ladder

**The ask.** An explicit ladder for fixing the product with a guidance push, each rung a proof of
concept for the next and changing one thing, each with a compute plan written before launch.

| Round | What | Chosen | Options there were | Who chose | Why |
|---|---|---|---|---|---|
| 1 | GPU budget and deadline | no hard cap, no date; each rung records what it spent; schedule stays in sittings | about 60, 100 or 150 GPU-hours; 2, 4 or 8 weeks | user | |
| 1 | the photo stage (A2r, A2c) | beside the ladder, gating nothing | rung 1b gating rung 2; dropped | user | recommended option taken |
| 1 | rung 3's prompts | colour words stripped so only the pair kind changes | as the benchmark wrote them, marked mixed; both | user | every object prompt carries a colour and no animal pair does |
| 2 | rung 2 | A3q: a pair-aware discriminator on 10 to 15 renders checked by eye from seeds 1 to 4, the product's renders from those seeds as negatives, judged on seeds 9 to 12 against plain product and best-of-N at equal compute; plan 15's head (A3) off the ladder | plan 15's head on training seeds 1 to 8 | user | "the fastest real test" |
| 2 | rung 2's bar | four readings on A3q's page; only "two animals on held-out seeds, pictures intact" lets rung 3 start; the sheet carries seeds 1 to 4 beside 9 to 12 | | user | the screenshot of the reading table |
| 2 | rung 3 and 4 rows | K1, K2, K3 (3 pairs per kind, seeds 9 to 12); 4a A4, 4b C1, 4c C2 | | user, `settle` | |
| 2 | compute | every rung row gains "compute plan written before launch"; quickest card wanted, Blackwell first if it can run the sampler | | user | "use GPU that would get results quickest" |
| 2 | card order | mscluster111 or 112 (Blackwell, `co3_bw`) first, the other Blackwell second, mscluster107 card 1 (RTX 8000, `co3`) third; every column of one rung's sheet on one card; rung 2's first piece checks the sampler runs under `co3_bw` | 107 card 1 first, keeping comparability with the RTX 8000 numbers on file | user, `settle` | both Blackwells passed a CUDA probe on 2026-11-17; about 9 times faster per product render |
| 3 | rung 1 (step 7) and the chain | chain `7 → 9 → 10 → rung 3 plan → rung 3 → rung 4 plan → 4a, 4b, 4c → 11`; every rung opens on a compute piece that stops for a yes; step 7 gains the compute piece, an axes block (classifier varies, pair weight, draws, schedule and s grid held), and a bar where separate-label stays at most 15 of 300 at every s | the axes planner for rung 1 too; the compute piece as its own route step | user, `settle` | |
| 4 | rung 2 (steps 9 and 10) | positives: 10 to 15 checked renders of cat and dog apart from seeds 1 to 8 (joint prompt plus two separating wordings); negatives: the product's renders and lone cat and lone dog; axis: how the discriminator is used (none, best-of-N it picks, guided at 3 strengths); four readings with counts out of 4; step 9 reframed for A3q, step 10 is A3q's walk; plan 15 re-routed after it | positives from seeds 1 to 4 including lone animals | user, `settle` | the joint prompt composes on at most 2 seeds below 9, and plan 18's push turned the fused animal into one cat |
| 5 | rung 3 (steps 10b, 10c) | an axes planner writes rung 3's plan, then a walk runs it; pairs drawn by rule (guard-clean "and" prompts, colours stripped, cat and dog excluded, 3 per kind with random.Random(0)); one discriminator per pair by rung 2's recipe; bar per kind: at least 4 more of 12 held-out tiles than plain product, more than best-of-N | one pooled discriminator; one kind per sitting; a hand-picked set | user, `settle` | |
| 6 | rung 4 (steps 10d, 10e) | three sub-rungs in order: 4a rung 2's discriminator on seeds 9 to 16, seeds 13 to 16 read apart; 4b one discriminator over the v58 pool's admitted targets, also scored on cat × dog beside 4a's to separate pooling from more pairs; 4c the same on the 53 held-out pairs, the 23 sharing no concept apart; automatic two-things count on every tile and by eye on one fixed seed per pair; bar within 10 points of the rank-32 LoRA and above best-of-N | per-pair discriminators on the v58 pool (too few admitted targets per pair) | Claude, under "settle until we are finished" | the user handed the remaining rounds over; veto stands |
| 7 | step 11 and the schedule | step 11 walks every rung's retrospective and re-routes to P1, B1 to B7 and plan 15; 22 two-hour sittings, about 18 h to finish and 32 h budget at 2×, about 15 GPU-hours on a Blackwell | | Claude, under the same | |
| 8 | check and emit | every row moved by a step except A2 and A2b (not on this route by an earlier decision); every walk step has a Filing line; every new step has its eight prompt sections; no pending line; the catalog is stale for hld-diagram and prompt-storyboard only, which steps 1 and 6 use and this reopening did not change | | Claude, under the same | |

## Reopened 2026-11-18: the MNIST stage

**The ask.** Add an MNIST stage (board row A2b) so the checker recipe is settled where labels are
exact and free. Rung 2's checker stalled below 0.75 held-out balanced accuracy on SDXL cat × dog
(`artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md`).

| Round | What | Chosen | Options there were | Who chose | Why |
|---|---|---|---|---|---|
| 1 | where the MNIST stage sits | step 7b, a lane beside rung 2 (step 10), gating nothing; its compose count against checker accuracy decides step 10's next move (more cat × dog data, or a per-animal checker) | in front of rung 2, pausing the checker loop; a third arm of step 8's photo stage | user, `settle` | the checker loop keeps its running work; MNIST is the first rung with learned experts and exact labels |
| 2 | the canvas and the two-digit share | 32 × 64 canvases of real MNIST digits; shares 0, 0.01, 0.1; 5 pairs held out by `random.Random(0)` (0 & 3, 1 & 9, 3 & 4, 3 & 6, 4 & 7), the other 40 on two-digit canvases; 7 & 8 seen; a support check (does expert 7 alone ever draw a held-out partner) added to round 4 | any of 45 pairs; only 7 & 8; all but 7 & 8; 9 held out; an empty-canvas background model | user, `settle` | the user asked for unseen pairs: the classifier should learn plurality, and Bradley et al. 2025 section 3.2 says the plain product is zero wherever either expert is, so clutter (the share) is what gives unseen pairs weight |
| 3 | the model, and how a two-digit canvas is labelled | one class-conditional U-Net per share, null label on 10% of canvases, DDIM 50 steps, laptop RTX 4070; a two-digit canvas enters training once per digit it holds | both digits as one label (a joint-prompt reference, addable later as one model); a latent model; ancestral sampling | user, `settle` | keeps each expert the cluttered conditional the share was designed to make |
| 4 | the first measurement, before any checker | plain product on 7 & 8 and the 5 held-out pairs, 64 seeds × 3 models; the support check (experts 7 and 4 alone, 1,000 seeds at share 0.1); the ladder with a strict headline (two clear, right digits); a single-digit judge gated at 58 of 64 against eye calls; a share at 32 of 64 or more for 7 & 8 leaves the bar | loose headline (two digits of any identity); other judges (canvas classifier, detector, eye only) | user, `settle` | the prototype judge ([judge sheet](figures/mnist-judge-ladder.png)) called 7 of 10 pasted canvases as built: a fused 7 and 8 can read as a confident 8, so the gate and a fused test are needed |
| 5 | the checker's recipe, against the toy's | a small head on the frozen U-Net's decoder features; the fourteen-row table in section 7b; plain product's result decides whether the checker is built (stops at 32 of 64 or more for 7 & 8 at every share) | its own small noise-aware CNN on pixels; both as a checker axis | user, `settle` | the accuracy-against-compose curve has to carry over to SDXL's checker; the user asked what happens if MNIST simply composes, since it has no text encoder or cross-attention |
| 6 | the comparisons on the same seeds | seeds 0 to 63 for evaluation, checker data from seed 1,000 up; plain product, pair-aware push, separate-label push, size-matched random push, best-of-N at matched compute with N = 16 beside it; byte-identical at s = 0 | matched N only; N = 16 only; the judge's own push as a ceiling | user, `settle` | the user asked for a simpler re-explanation first; the zoom ladder's [judge network view](figures/judge-network-view.png) landed the same round |
| 7 | the two sweeps for rung 2 | weaker checkers by less data (1%, 5%, 25%, 100%) and by flipped labels (10%, 20%), one knob at a time; compose count against held-out accuracy with 0.655 and 0.771 marked; crossing below 0.771 keeps step 10 fetching data, above it switches step 10 to a per-animal checker; cap at 0.1, 0.3, 1.0 | less data only; flipped labels only | user, `settle` | SDXL suffers from both: 167 clean cat × dog pairs and 79% label agreement. The user also checked that the push comes from a whole-canvas checker, not the blob judge; section 7b now names the three networks |
| 8 | the bar, and FashionMNIST | plain at most 8 of 64; pair-aware at least 32 of 64 at some s; random and separate-label within 6 of plain; best-of-N at least 8 below; held-out pairs at least 160 of 320; four readings; FashionMNIST after MNIST passes | three quarters, 48 of 64 | user, `settle` | the toy's bar was half (150 of 300); the [three-network picture](figures/three-networks-in-step-7b.png) landed this round |
| 9 | how much planning, and board row A2b | no plan tree; the walk writes its own short plan, and its first piece copies the bar into `plans/review/`; row A2b set to todo on step 7b, its page `report/classifier-guided-poe/24-mnist-one-model.md` with seven outcomes, its expected-outcome line filled; board redrawn (`board-2026-11-18-step-0.png`) | a plan file written by the axes planner first | user, `settle` | section 7b already holds the design and the bar. The user asked for a control that turns any reply or screenshot into a zoom-ladder picture, kept per round and consolidated at the walk's close: a change to the drip navigation contract, offered as its own step |
| 10 | time on both clocks | three two-hour sittings (M1 to M3) before rung 2's next sitting; about 6 h of the user's time, 3 to 4 h on the laptop's RTX 4070 (a guess until M1 times it); stop if M3 ends without the five-version sheet | alternating with rung 2 | user, `settle` | step 10's next move waits on the accuracy sweep. The user also said `make it`: the picture control is queued for after this walk closes |
| 11 | step 7b's prompt, the checks, and the edits to step 10 and the references | the prompt written into section 7b; step 10 gains the 0.771 rule under "When something is missing"; the references file puts MNIST on the route; four rows added to "What we know so far"; board redrawn | | Claude, under the user's "settle at your discretion" | every check passed except the catalog's staleness for drip's shared controls, which step 7b's walk reads live |

**Round 2, asked along the way.** The user asked to see the shares and whether an unseen pair is
possible at all. Filed: [the canvas sheet](figures/mnist-canvas-shares.png), real MNIST digits on
32 × 64 canvases, built by `2-classifier-guidance/scripts/mnist/canvas_sheet.py` from
`/home/molef/PhD/replications/original-superdiff/data/MNIST/raw` with seed 0 (0, 1 and 3
two-digit canvases in 30 draws at shares 0, 0.01 and 0.1). The rule `random.Random(0)` holds out
0 & 3, 1 & 9, 3 & 4, 3 & 6 and 4 & 7; 7 & 8 stays seen. The answer rests on Bradley et al. 2025
section 3.2 (the Bayes composition is zero wherever either conditional is) and section 5.3 (clutter
makes conditionals approximately factorized, which explains Liu et al. 2022's more-than-five objects).

## Pictures

| File | Round | What it answers |
|---|---|---|
| [mnist-canvas-shares.png](figures/mnist-canvas-shares.png) | 2 | what the two-digit share means, and what the 7 and 4 labels see in training |
| [mnist-judge-ladder.png](figures/mnist-judge-ladder.png) | 4 | the four calls on the ladder, and the prototype judge calling 7 of 10 pasted canvases as built |
| [judge-network-view.png](figures/judge-network-view.png) | 5 | the judge as one small network between two hand-written rules |
| [three-networks-in-step-7b.png](figures/three-networks-in-step-7b.png) | 7 | the checker that pushes, the control that cannot fix blending, and the judge that only scores |
