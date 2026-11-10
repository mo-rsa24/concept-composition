# Route walk: answering the ICLR 2027 reviews of submission 56173

The map of the drip-skill-route walk. Each round records the decision, who chose it, why, and what was not taken.

## Round 1: the sentence and the disk (settled 2026-11-10)

- **Goal:** Reviewer PrzE sees each claim backed by numbers, and a respectful reply goes up on OpenReview.
- **Objective:** scores for the adapter against PoE, CO3 and more corrected samplers on the same held-out pairs and seeds, the pair list, and a three-concept result.
- **Outcome:** a revised paper in a shared Overleaf, a second Overleaf for the replies, comment text ready to post.
- **Chosen by you:** all four of PrzE's points get new numbers.
- **Found on disk:** the manuscript at `paper/iclr/`, scorers for CLIP and ImageReward in `composition/focus/eval_metrics.py`, the Langevin corrector, Feynman-Kac and SMC code under `scripts/`, the pool files under `artifacts/_shared/cross_pair_pool_configs/`, a figure register at `paper/iclr/figures.md`.
- **Problems found:** the pair list and the paper's checkpoint are unproven; four held-out pairs cannot carry a rate; the adapter has no slot for a third concept; the local source differs from the submitted PDF.

## Round 2: the outcomes (settled 2026-11-10)

- **Board:** `report/paper-review/board.md`, 12 rows, PRZE-1 to PRZE-11 with PRZE-3 split into 3a and 3b.
- **Chosen by you:** design A (official Attend-and-Excite and CONFORM code unmodified on SD 1.5), a held-out set enlarged into three groups, a three-concept test with a written bar, training on SD 1.5 gated and revisited, placeholders for every figure and table, compute chosen deliberately, and UHz6's early-steps question added.
- **Not taken:** porting the baselines to SDXL (our own port invites an unfairness charge), training a three-concept adapter by default.
- **Open from this round:** which node the idle Blackwell screenshot came from; `sinfo` shows 107 and 110 allocated and 111 idle, against your list of 107, 108 and 112; the cluster ssh timed out once.

## Round 3: how much planning (settled 2026-11-10)

- **Chosen by you:** the short plan. The route file `plans/route-answer-the-iclr-reviews.md` is the plan; each experiment opens as its own request walk.
- **Not taken:** a full plan tree (too slow for 2.5 working days, and gated steps would force rewrites).
- **Added by you in this round:** every experiment ships qualitative grids, including the cheap early-steps test; mscluster112 may be probed; mscluster107 is your own job 83423 with two RTX 8000 cards.
- **Verified in this round:** mscluster112 is healthy (Blackwell, 97,887 MiB, torch matmul passes); `/datasets` has 212 TB free.

## Round 4: step 1, check the ground (settled 2026-11-10)

- **Settled as proposed:** `/drip-execute-plan --request`, 90 minute box, filing tied, written into `plans/route-answer-the-iclr-reviews.md` as step 1. The chain line was recorded as proposed; steps 2 to 13 settle one per round.
- **Added by you in this round:** the CO3 paper's Tables 1, 5, 6, 7 and a pointer to its many-method qualitative figures. They set the test protocol (Chefer et al. prompt sets, BLIP-VQA, ImageReward) and the layout for our own multi-method table and grid.

## Round 5: step 2, the protocol and the layout sheet (settled 2026-11-10)

- **Settled as proposed:** `/drip-execute-plan --request`, Opus medium, 60 minute box, filing tied, with `/plan-figures` as the side route at piece 2. Written into the route file.
- **Added by you in this round:** a gap step, a `/submerge` for the eleven evaluation and baseline words. The chain grows to 14 steps; the gap step is step 3.

## Round 6: step 3, submerge the gap words (settled 2026-11-10)

- **Settled as proposed:** `/submerge` in its default mode, gloss tier for all eleven words, 30 minute box for the open and the gate, the journey builds in the background. Written into the route file.
- **Why it is where it is:** it needs no run, so it fits a GPU wait; the paper-reading step 6 and the scores step 8 use its notes.

## Round 7: step 4, choose and freeze the evaluation set and the one checkpoint (settled 2026-11-10)

- **Settled as proposed:** a request walk, Sonnet medium, 60 minute box, filing on. Written into the route file.
- **Found on the way:** Figure 5's "ours" column comes from three different adapters (`artifacts/results/four-pairs-four-methods/README.md`), so one checkpoint is frozen before any held-out result is read, and Figure 5 is re-rendered from it.

## Round 8: step 5, the compute plan and the first GPU jobs (settled 2026-11-10)

- **Settled as proposed:** a request walk, Sonnet medium, 45 minute box, filing on, `/calibrate --predict` and `/run-experiment` as side routes at the launch piece. Written into the route file.
- **Found in this round:** you hold your one `biggpu` slot in job 83423 on mscluster107, so other jobs use the shared-device path on mscluster112; the 107 allocation ends in about 49 hours; the Blackwell is 3 to 4 times faster than the RTX 8000 cards.
- **Chain change:** steps 6 and 7 split into a pipeline diagram and a paper-reading walk, since both are walks and run one at a time. The chain is 15 steps.

## Round 9: step 6, draw the evaluation pipeline (settled 2026-11-10)

- **Settled as proposed:** `/drip-hld-diagram --from` the route file, seven pieces with a fairness mark on each, the hardware level for the machines. No page of this system existed in the registry (18 pages). Written into the route file.

## Round 10: step 7, read the benchmark papers together (settled 2026-11-10)

- **Settled as proposed:** `/drip-read --combine --lean` on one idea across four sources (CO3, T2I-CompBench++, ImageReward, GenEval), predict before each answer, two small-object images from our own census. The rival-methods reading stays a possible second run.

## Round 11: step 8, the corrector proof and the six-column figure (settled 2026-11-10)

- **Settled as proposed:** a request walk at `--depth objective`, Opus medium, 120 minute box, filing on. The bar is the exact stationary variance of the update on a Gaussian, so the test can fail.
- **Budget flag raised:** your time through step 8 is about 7 hours, the remaining steps about 10 more, against 10 to 15 hours. The cuts offered are `--depth objective`, the AI's lane only for compute steps, and gated trainings kept out of the attention budget. To be confirmed at the schedule round.

## Round 12: step 9, the scores (settled 2026-11-10)

- **Settled as proposed:** a request walk at `--depth objective`, Sonnet medium, three checkpoints, the bar (supports, null, inconclusive, reported as is) written into the protocol before any score. A scorer that cannot be installed in 30 minutes is reported as not run, never replaced under the same name.

## Round 13: step 10, the SD 1.5 set and the ceiling test (settled 2026-11-10)

- **Settled as proposed:** design A on SD 1.5 with the official code unmodified (Attend-and-Excite vendored in `composition/aae/`, CONFORM from its repository), the measured correction as the ceiling test, SD 1.5 numbers in their own table. The SD 1.5 adapter stays gated on a PASS and your yes.

## Round 14: step 11, the three-concept test (settled 2026-11-10)

- **Settled as proposed:** the frozen adapter inside a four-branch product, about 10 triples avoiding the training concepts, the bar committed first, the guidance weight tuned equally for both products on three tuning triples. No three-concept training.

## Round 15: step 12, the two gated trainings (settled 2026-11-10)

- **Settled as proposed:** a decision step. Each training opens only if its cheap test passes its written bar and you say yes to the exact command; each is matched to the paper's recipe in one axis only, with the same checkpoint-choice rule. A closed gate is reported as a finding.

## Round 16: step 13, revise the paper (settled 2026-11-10)

- **Settled as proposed:** a `/drip-write` walk for the evaluation section and the limitations paragraph, with the mechanical fixes and appendix tables as one AI diff for one yes. The Overleaf token never reaches Claude: you clone and push. Hours so far add to about 11.8 with this step, inside the 10 to 15.

## Round 17: step 14, draft the replies (settled 2026-11-10)

- **Settled as proposed:** a `/drip-write` walk for the reply to Reviewer PrzE, one markdown source built into LaTeX for the replies project and plain text for OpenReview, every number matched to the revised paper by a script, two disclosures stated plainly (Figure 5's three adapters; baselines that beat the adapter). You create the replies project and post.

## Round 18: step 15, explain it back (settled 2026-11-10)

- **Settled as proposed:** `/merge-sessions --retro` over the step sessions, then `/drip-retro` with a prediction before each chunk, then `/quiz-me` on the five things. `/pressure-test` left out: it vets ideas before experiments, and these have run.

## Round 19: time and schedule (settled 2026-11-10)

- **Settled as proposed:** your time about 14 h plus about 1 h posting, inside a 15 h box; windows as proposed (Tuesday evening to Tuesday 17 November), since none were given. Cuts held in reserve, cheapest first: step 7 on two sources (about 25 min), step 15 without the quiz (about 15 min), step 3's gate after the rebuttal (about 30 min).

## Round 20: checks and emit (2026-11-10)

- Every board row is moved by at least one step, and every step moves at least one row. All 12 rows are `todo` except PRZE-10, `in progress`.
- Fifteen steps, each with its Filing line and eight prompt sections (16 prompts, since step 15 has two).
- No `pending` line left in the route file or the references file.
- Prompt lines wrapped at 100 columns except three single-path lines that cannot be broken.
- The catalog check printed `catalog current`.
