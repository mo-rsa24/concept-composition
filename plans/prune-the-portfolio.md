# ✂️ Prune the portfolio

<!-- template: exempt. Reason: a decision list for the pruning pass, one box per candidate -->

Everything below is a candidate because nothing points at it, which means unreferenced, not
unwanted. Tick **Keep** or **Archive** on each row. Nothing moves until a whole section is decided;
then archived items move to `_archive/` with their sub-paths kept, finished plans move to
`artifacts/plans/completed/`, and nothing is deleted. Built 2026-11-13 by reading every tracked
markdown file for links and names, after the regroup into project folders.

| Section | Candidates | Proposed |
|---|---|---|
| [Finished plans](#1-finished-plans-still-in-plans) | 20 | move to `artifacts/plans/completed/` |
| [Results nothing links to](#2-results-folders-nothing-links-to) | 9 | look at each figure, then keep and link, or archive |
| [Markdown nothing links to](#3-markdown-nothing-links-to-or-names) | 111 | archive unless it is worth linking |
| [Modules nothing imports or names](#4-poe_repair-modules-nothing-imports-or-names) | 10 | archive |

## 1. Finished plans still in `plans/`

Every task ticked. Moving them keeps `plans/` to what is still open.

| Keep in plans/ | Plan | Project | Ticked tasks | Last change |
|---|---|---|---|---|
| [ ] | [🧪 Review: does more of the LoRA's own output give more composition?](../1-lora-correction/artifacts/plans/completed/01-showcase-the-trained-lora/review/03-the-correction-amount-series.md) | 1-lora-correction | 7 | 2026-11-13 |
| [ ] | [🔬 Review: does the extended tracking set log everything, before anythi](../1-lora-correction/plans/01-showcase-the-trained-lora/review/06-extend-the-tracking-set.md) | 1-lora-correction | 6 | 2026-11-13 |
| [ ] | [🧪 Review: does a correction confined to the early steps, followed by a](../1-lora-correction/plans/01-showcase-the-trained-lora/review/14-correct-early-then-clean-up.md) | 1-lora-correction | 11 | 2026-11-13 |
| [ ] | [🧪 Hypothesis 02: More correction, more composition](../1-lora-correction/artifacts/plans/completed/03-does-the-correction-cause-composition/plans/hypothesis/03-more-correction-more-composition.md) | 1-lora-correction | 8 | 2026-11-13 |
| [ ] | [📊 What the cached trajectories already tell us](../1-lora-correction/plans/03-does-the-correction-cause-composition/plans/hypothesis/04-what-the-cached-runs-already-show.md) | 1-lora-correction | 7 | 2026-11-13 |
| [ ] | [⏱️ When in the denoising run is the correction needed?](../1-lora-correction/artifacts/plans/completed/03-does-the-correction-cause-composition/plans/hypothesis/05-when-in-the-run-it-matters.md) | 1-lora-correction | 10 | 2026-11-13 |
| [ ] | [🔭 The same story from three independent sides](../1-lora-correction/plans/03-does-the-correction-cause-composition/plans/hypothesis/06-the-same-story-from-three-sides.md) | 1-lora-correction | 5 | 2026-11-13 |
| [ ] | [🔬 Does the fix change what a word paints, or where it looks?](../1-lora-correction/artifacts/plans/completed/03-does-the-correction-cause-composition/plans/hypothesis/07-what-the-fix-changes-inside-the-model.md) | 1-lora-correction | 6 | 2026-11-13 |
| [ ] | [🔧 Build and check the measuring scripts the rest of this scope uses](../1-lora-correction/artifacts/plans/completed/03-does-the-correction-cause-composition/plans/tools/01-build-the-measuring-scripts.md) | 1-lora-correction | 9 | 2026-11-13 |
| [ ] | [📏 Pre-register the correction-size measure](../1-lora-correction/artifacts/plans/completed/03-does-the-correction-cause-composition/plans/tools/02-fix-the-size-measure-before-any-result.md) | 1-lora-correction | 3 | 2026-11-13 |
| [ ] | [🧪 Review: does more correction give more composition?](../1-lora-correction/artifacts/plans/completed/03-does-the-correction-cause-composition/review/03-more-correction-more-composition.md) | 1-lora-correction | 10 | 2026-11-13 |
| [ ] | [🧪 Review: when in the run is the correction needed?](../1-lora-correction/artifacts/plans/completed/03-does-the-correction-cause-composition/review/05-when-in-the-run-it-matters.md) | 1-lora-correction | 9 | 2026-11-13 |
| [ ] | [🧪 Review: what does the fix change inside the model?](../1-lora-correction/artifacts/plans/completed/03-does-the-correction-cause-composition/review/07-what-the-fix-changes-inside-the-model.md) | 1-lora-correction | 6 | 2026-11-13 |
| [ ] | [🧬 A clean pool, built only once the scorer is trusted](../1-lora-correction/plans/04-does-the-fix-reach-unseen-pairs/plans/tools/01-the-clean-pair-pool.md) | 1-lora-correction | 4 | 2026-11-13 |
| [ ] | [🔌 Three live curves while training](../1-lora-correction/plans/04-does-the-fix-reach-unseen-pairs/plans/tools/02-three-live-curves-while-training.md) | 1-lora-correction | 5 | 2026-11-13 |
| [ ] | [🖼️ Review: where does each condition land, and when does each run comm](../1-lora-correction/artifacts/plans/completed/05-when-does-the-outcome-lock-in/review/06-where-each-condition-lands.md) | 1-lora-correction | 14 | 2026-11-13 |
| [ ] | [🔬 Review: does the corrector change nothing when switched off, and wha](../1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/review/02-the-corrector-and-the-step-size-it-runs-at.md) | 1-lora-correction | 8 | 2026-11-13 |
| [ ] | [🧪 Review: does the corrector's compose rate peak in the same window th](../1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/review/04-does-the-corrector-compose-in-the-same-window.md) | 1-lora-correction | 11 | 2026-11-13 |
| [ ] | [🧪 Review: does the same product still fuse a cat and a dog on a model ](../1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/review/14-does-a-clip-free-model-hybridise.md) | 1-lora-correction | 3 | 2026-11-13 |
| [ ] | [🧪 Review: does SDXL still fuse when it is sampled the way PixArt-alpha](../1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/review/16-does-sdxl-compose-at-pixarts-settings.md) | 1-lora-correction | 3 | 2026-11-13 |

## 2. Results folders nothing links to

No plan, card, walk or finding points at these. Open the figure, then keep and link it from the
plan or finding it answers, or archive it.

| Keep | Archive | Folder | Project | Size | Figure |
|---|---|---|---|---|---|
| [ ] | [ ] | [do-sampling-strategies-fix-the-adapter](../1-lora-correction/artifacts/results/against-other-methods/samplers-and-correctors/do-sampling-strategies-fix-the-adapter) ((no card)) | 1-lora-correction | 6 MB | [seven-strategies-cat-and-dog-seed09.png](../1-lora-correction/artifacts/results/against-other-methods/samplers-and-correctors/do-sampling-strategies-fix-the-adapter/seven-strategies-cat-and-dog-seed09.png) · [seven-strategies-cat-and-dog-seed10.png](../1-lora-correction/artifacts/results/against-other-methods/samplers-and-correctors/do-sampling-strategies-fix-the-adapter/seven-strategies-cat-and-dog-seed10.png) |
| [ ] | [ ] | [does-each-adapter-hold-across-ten-seeds](../1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/how-far-one-adapter-reaches/does-each-adapter-hold-across-ten-seeds) (Every adapter across ten seeds) | 1-lora-correction | 178 MB | [a-camel-and-a-forest.png](../1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/how-far-one-adapter-reaches/does-each-adapter-hold-across-ten-seeds/a-camel-and-a-forest.png) · [a-cat-and-a-dog.png](../1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/how-far-one-adapter-reaches/does-each-adapter-hold-across-ten-seeds/a-cat-and-a-dog.png) |
| [ ] | [ ] | [how-big-the-figure-text-prints](../_archive/1-lora-correction/artifacts/results/how-big-the-figure-text-prints) (How big the figure text prints) | 1-lora-correction | 0 MB | no image |
| [ ] | [ ] | [how-the-adapter-settings-change-cat-and-dog](../1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog) ((no card)) | 1-lora-correction | 14 MB | [fail-r0-c0.png](../1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/limitations-tiles/fail-r0-c0.png) · [fail-r0-c1.png](../1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/limitations-tiles/fail-r0-c1.png) |
| [ ] | [ ] | [which-new-pairs-compose-cleanly](../1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-transfer/which-new-pairs-compose-cleanly) ((no card)) | 1-lora-correction | 325 MB | [pair-audition.png](../1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-transfer/which-new-pairs-compose-cleanly/pair-audition.png) · [mono.png](../1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-transfer/which-new-pairs-compose-cleanly/tiles/a_badger__x__a_magpie/seed02/mono.png) |
| [ ] | [ ] | [what-the-fusion-is-not-caused-by](../3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/poe-against-the-joint/what-the-fusion-is-not-caused-by) ((no card)) | 3-understanding-compositionality | 3 MB | [summary-grid.png](../3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/poe-against-the-joint/what-the-fusion-is-not-caused-by/summary-grid.png) |
| [ ] | [ ] | [how-sdxl-turns-a-prompt-into-an-image](../artifacts/results/how-sdxl-works/how-sdxl-turns-a-prompt-into-an-image) ((no card)) | root | 8 MB | [bpe-1.png](../artifacts/results/how-sdxl-works/how-sdxl-turns-a-prompt-into-an-image/pictures/bpe-1.png) · [bpe-2.png](../artifacts/results/how-sdxl-works/how-sdxl-turns-a-prompt-into-an-image/pictures/bpe-2.png) |
| [ ] | [ ] | [how-the-sdxl-vae-is-built](../artifacts/results/how-sdxl-works/how-the-sdxl-vae-is-built) ((no card)) | root | 4 MB | [sdxl-vae-high-level-design-explainer__dec-dark.png](../artifacts/results/how-sdxl-works/how-the-sdxl-vae-is-built/vae-stills/sdxl-vae-high-level-design-explainer__dec-dark.png) · [sdxl-vae-high-level-design-explainer__dec-light.png](../artifacts/results/how-sdxl-works/how-the-sdxl-vae-is-built/vae-stills/sdxl-vae-high-level-design-explainer__dec-light.png) |
| [ ] | [ ] | [why-the-latent-is-scaled](../artifacts/results/how-sdxl-works/why-the-latent-is-scaled) (Why the latent is scaled) | root | 0 MB | no image |

## 3. Markdown nothing links to or names

Old walk pieces, notes and write-ups. Archive by default; keep any worth linking from its
project's README or index.


**1-lora-correction**

| Keep | File | Title | Last change |
|---|---|---|---|
| [ ] | [artifacts/drips/answer-iclr-reviews/lookup/step-01.md](../_archive/1-lora-correction/artifacts/drips/answer-iclr-reviews/lookup/step-01.md) | Lookup: step 1 | 2026-11-13 |
| [ ] | [artifacts/drips/answer-iclr-reviews/lookup/step-03.md](../_archive/1-lora-correction/artifacts/drips/answer-iclr-reviews/lookup/step-03.md) | Lookup: step 3 | 2026-11-13 |
| [ ] | [artifacts/drips/answer-iclr-reviews/lookup/step-04.md](../_archive/1-lora-correction/artifacts/drips/answer-iclr-reviews/lookup/step-04.md) | Lookup: step 4 | 2026-11-13 |
| [ ] | [artifacts/drips/answer-iclr-reviews/prompt-reviewers-k4nd-and-uhz6.md](../_archive/1-lora-correction/artifacts/drips/answer-iclr-reviews/prompt-reviewers-k4nd-and-uhz6.md) | Prompt: answer Reviewers k4nd and UHz6, fixing the paper where we can | 2026-11-13 |
| [ ] | [artifacts/drips/answer-iclr-reviews/prompt-route-step-10-sd15-and-ceiling.md](../_archive/1-lora-correction/artifacts/drips/answer-iclr-reviews/prompt-route-step-10-sd15-and-ceiling.md) | Prompt: route step 10, the SD 1.5 comparison set and the ceiling test | 2026-11-13 |
| [ ] | [artifacts/drips/answer-iclr-reviews/prompt-route-step-11-three-concepts.md](../_archive/1-lora-correction/artifacts/drips/answer-iclr-reviews/prompt-route-step-11-three-concepts.md) | Prompt: route step 11, the three-concept test | 2026-11-13 |
| [ ] | [artifacts/drips/answer-iclr-reviews/prompt-route-step-12-gated-trainings.md](../_archive/1-lora-correction/artifacts/drips/answer-iclr-reviews/prompt-route-step-12-gated-trainings.md) | Prompt: route step 12, decide and launch the two gated trainings | 2026-11-13 |
| [ ] | [artifacts/drips/answer-iclr-reviews/prompt-route-step-15-explain-it-back.md](../_archive/1-lora-correction/artifacts/drips/answer-iclr-reviews/prompt-route-step-15-explain-it-back.md) | Prompts: route step 15, explain it back | 2026-11-13 |
| [ ] | [artifacts/drips/answer-iclr-reviews/prompt-route-step-2-protocol-and-layout.md](../_archive/1-lora-correction/artifacts/drips/answer-iclr-reviews/prompt-route-step-2-protocol-and-layout.md) |  | 2026-11-13 |
| [ ] | [artifacts/drips/answer-iclr-reviews/prompt-route-step-5-compute-and-first-jobs.md](../_archive/1-lora-correction/artifacts/drips/answer-iclr-reviews/prompt-route-step-5-compute-and-first-jobs.md) | Prompt: route step 5, the compute plan and the first GPU jobs | 2026-11-13 |
| [ ] | [artifacts/drips/answer-iclr-reviews/prompt-route-step-6-evaluation-pipeline.md](../_archive/1-lora-correction/artifacts/drips/answer-iclr-reviews/prompt-route-step-6-evaluation-pipeline.md) | Prompt: route step 6, draw the evaluation pipeline | 2026-11-13 |
| [ ] | [artifacts/drips/answer-iclr-reviews/prompt-route-step-7-read-the-scorers.md](../_archive/1-lora-correction/artifacts/drips/answer-iclr-reviews/prompt-route-step-7-read-the-scorers.md) | Prompt: route step 7, read the scorers together | 2026-11-13 |
| [ ] | [artifacts/drips/answer-iclr-reviews/prompt-route-step-8-langevin-and-sampler-figure.md](../_archive/1-lora-correction/artifacts/drips/answer-iclr-reviews/prompt-route-step-8-langevin-and-sampler-figure.md) | Prompt: route step 8, prove the Langevin corrector and render the six- | 2026-11-13 |
| [ ] | [artifacts/drips/answer-iclr-reviews/prompt-route-step-9-score-and-tables.md](../_archive/1-lora-correction/artifacts/drips/answer-iclr-reviews/prompt-route-step-9-score-and-tables.md) | Prompt: route step 9, score every image and build the tables | 2026-11-13 |
| [ ] | [artifacts/drips/answer-iclr-reviews/prompt-why-the-second-animal-loses-its-species.md](../_archive/1-lora-correction/artifacts/drips/answer-iclr-reviews/prompt-why-the-second-animal-loses-its-species.md) |  | 2026-11-13 |
| [ ] | [artifacts/drips/answer-iclr-reviews/routes/from-k4nd-uhz6-walk-co3-renders-complete.md](../_archive/1-lora-correction/artifacts/drips/answer-iclr-reviews/routes/from-k4nd-uhz6-walk-co3-renders-complete.md) | For route step 9: CO3's renders are complete, its scoring can start | 2026-11-13 |
| [ ] | [artifacts/drips/answer-iclr-reviews/routes/from-k4nd-uhz6-walk-more-hard-pairs.md](../_archive/1-lora-correction/artifacts/drips/answer-iclr-reviews/routes/from-k4nd-uhz6-walk-more-hard-pairs.md) | For the route: grow the hard group of Table 2 to 66 pairs | 2026-11-13 |
| [ ] | [artifacts/drips/answer-iclr-reviews/routes/from-k4nd-uhz6-walk-prze-replies.md](../_archive/1-lora-correction/artifacts/drips/answer-iclr-reviews/routes/from-k4nd-uhz6-walk-prze-replies.md) | For route step 14: replies/prze.md exists | 2026-11-13 |
| [ ] | [artifacts/drips/answer-iclr-reviews/routes/from-k4nd-uhz6-walk-r11-filled.md](../_archive/1-lora-correction/artifacts/drips/answer-iclr-reviews/routes/from-k4nd-uhz6-walk-r11-filled.md) | Slot R11 is filled: do not re-render | 2026-11-13 |
| [ ] | [artifacts/drips/answer-iclr-reviews/routes/from-k4nd-uhz6-walk-r2-filled.md](../_archive/1-lora-correction/artifacts/drips/answer-iclr-reviews/routes/from-k4nd-uhz6-walk-r2-filled.md) | Slot R2 (Figure 5) is filled | 2026-11-13 |
| [ ] | [artifacts/drips/answer-iclr-reviews/routes/from-k4nd-uhz6-walk-results-paragraph-draft.md](../_archive/1-lora-correction/artifacts/drips/answer-iclr-reviews/routes/from-k4nd-uhz6-walk-results-paragraph-draft.md) | For route step 13: a draft results paragraph for Tables 1 and 2 | 2026-11-13 |
| [ ] | [artifacts/drips/answer-iclr-reviews/routes/from-k4nd-uhz6-walk-success-rate.md](../_archive/1-lora-correction/artifacts/drips/answer-iclr-reviews/routes/from-k4nd-uhz6-walk-success-rate.md) | For route step 9: UHz6 asks for a success rate | 2026-11-13 |
| [ ] | [artifacts/drips/answer-iclr-reviews/routes/from-step-8b-co3-complete.md](../_archive/1-lora-correction/artifacts/drips/answer-iclr-reviews/routes/from-step-8b-co3-complete.md) | To the step 9 walk: CO3 is complete on R1 and R3 | 2026-11-13 |
| [ ] | [artifacts/drips/answer-iclr-reviews/routes/from-step-8b-co3-rated.md](../_archive/1-lora-correction/artifacts/drips/answer-iclr-reviews/routes/from-step-8b-co3-rated.md) | To the step 9 walk: CO3 is rated, ready to merge | 2026-11-13 |
| [ ] | [artifacts/drips/answer-iclr-reviews/routes/plan-sd15-baselines-and-syngen-sdxl.md](../_archive/1-lora-correction/artifacts/drips/answer-iclr-reviews/routes/plan-sd15-baselines-and-syngen-sdxl.md) | Plan: the SD 1.5 baselines for Table 4, and SynGen ported to SDXL for  | 2026-11-13 |
| [ ] | [artifacts/drips/answer-iclr-reviews/routes/to-step-8-superdiff-out-co3-on-six-shards.md](../_archive/1-lora-correction/artifacts/drips/answer-iclr-reviews/routes/to-step-8-superdiff-out-co3-on-six-shards.md) | To the step 8 walk: SuperDiff is out, CO3 runs on six shards | 2026-11-13 |
| [ ] | [artifacts/drips/answer-reviewers-k4nd-and-uhz6/lookup/step-01.md](../_archive/1-lora-correction/artifacts/drips/answer-reviewers-k4nd-and-uhz6/lookup/step-01.md) | Lookup: step 1 | 2026-11-13 |
| [ ] | [artifacts/drips/answer-reviewers-k4nd-and-uhz6/lookup/step-02.md](../_archive/1-lora-correction/artifacts/drips/answer-reviewers-k4nd-and-uhz6/lookup/step-02.md) | Lookup: step 2 | 2026-11-13 |
| [ ] | [artifacts/drips/answer-reviewers-k4nd-and-uhz6/while-it-runs.md](../_archive/1-lora-correction/artifacts/drips/answer-reviewers-k4nd-and-uhz6/while-it-runs.md) | While it runs | 2026-11-13 |
| [ ] | [artifacts/drips/does-a-clip-free-model-hybridise/lookup/step-04.md](../_archive/1-lora-correction/artifacts/drips/does-a-clip-free-model-hybridise/lookup/step-04.md) | Lookup: step 4 | 2026-11-13 |
| [ ] | [artifacts/drips/does-a-clip-free-model-hybridise/while-it-runs.md](../_archive/1-lora-correction/artifacts/drips/does-a-clip-free-model-hybridise/while-it-runs.md) | While it runs | 2026-11-13 |
| [ ] | [artifacts/drips/second-animal-species/idea2-per-place-weight-loss.md](../_archive/1-lora-correction/artifacts/drips/second-animal-species/idea2-per-place-weight-loss.md) | Idea 2: a loss on where each expert is turned up or down | 2026-11-13 |
| [ ] | [artifacts/drips/second-animal-species/lookup/step-01.md](../_archive/1-lora-correction/artifacts/drips/second-animal-species/lookup/step-01.md) | Lookup: step 1 | 2026-11-13 |
| [ ] | [artifacts/drips/second-animal-species/while-it-runs.md](../_archive/1-lora-correction/artifacts/drips/second-animal-species/while-it-runs.md) | While it runs | 2026-11-13 |
| [ ] | [plans/01-showcase-the-trained-lora/diagrams/process-versions/01-2026-08-29.md](../_archive/1-lora-correction/plans/01-showcase-the-trained-lora/diagrams/process-versions/01-2026-08-29.md) | Process lane snapshot 01 (2026-08-29) | 2026-11-13 |
| [ ] | [plans/08-improve-the-rank-32-pooled-adapter/diagrams/process-versions/01-2026-09-08.md](../_archive/1-lora-correction/plans/08-improve-the-rank-32-pooled-adapter/diagrams/process-versions/01-2026-09-08.md) | Process lane, version 01: 2026-09-08 | 2026-11-13 |
| [ ] | [plans/08-improve-the-rank-32-pooled-adapter/launch-prompt.md](../_archive/1-lora-correction/plans/08-improve-the-rank-32-pooled-adapter/launch-prompt.md) | Handoff prompt: launch the parent and children on the v57 pool | 2026-11-13 |
| [ ] | [report/diagrams/process-versions/01-2026-09-07.md](../_archive/1-lora-correction/report/diagrams/process-versions/01-2026-09-07.md) | Process lane 01: 2026-09-07 | 2026-11-13 |
| [ ] | [report/diagrams/process-versions/02-2026-09-07.md](../_archive/1-lora-correction/report/diagrams/process-versions/02-2026-09-07.md) | Process lane 02: 2026-09-07 | 2026-11-13 |

**2-classifier-guidance**

| Keep | File | Title | Last change |
|---|---|---|---|
| [ ] | [artifacts/drips/route-classifier-guided-poe-from-toy-to-animals/lookup/step-03.md](../_archive/2-classifier-guidance/artifacts/drips/route-classifier-guided-poe-from-toy-to-animals/lookup/step-03.md) | Lookup: step 3 | 2026-11-13 |

**3-understanding-compositionality**

| Keep | File | Title | Last change |
|---|---|---|---|
| [ ] | [artifacts/drips/camel-and-dog-in-the-training-data/lookup/step-00.md](../_archive/3-understanding-compositionality/artifacts/drips/camel-and-dog-in-the-training-data/lookup/step-00.md) | Lookup: step 0 | 2026-11-13 |
| [ ] | [artifacts/drips/camel-and-dog-in-the-training-data/lookup/step-01.md](../_archive/3-understanding-compositionality/artifacts/drips/camel-and-dog-in-the-training-data/lookup/step-01.md) | Lookup: step 1 | 2026-11-13 |
| [ ] | [artifacts/drips/forward-process-and-the-score/lookup/step-01.md](../_archive/3-understanding-compositionality/artifacts/drips/forward-process-and-the-score/lookup/step-01.md) | Lookup: step 1 | 2026-11-13 |
| [ ] | [artifacts/drips/forward-process-and-the-score/while-it-runs.md](../_archive/3-understanding-compositionality/artifacts/drips/forward-process-and-the-score/while-it-runs.md) | While it runs | 2026-11-13 |
| [ ] | [artifacts/drips/how-much-co-occurrence-the-product-needs/lookup/step-03.md](../_archive/3-understanding-compositionality/artifacts/drips/how-much-co-occurrence-the-product-needs/lookup/step-03.md) | Lookup: step 3 | 2026-11-13 |
| [ ] | [artifacts/drips/how-much-co-occurrence-the-product-needs/lookup/step-05.md](../_archive/3-understanding-compositionality/artifacts/drips/how-much-co-occurrence-the-product-needs/lookup/step-05.md) | Lookup: step 5 | 2026-11-13 |
| [ ] | [artifacts/drips/how-much-co-occurrence-the-product-needs/routes/01-projective-composition-vocabulary.prompt.md](../_archive/3-understanding-compositionality/artifacts/drips/how-much-co-occurrence-the-product-needs/routes/01-projective-composition-vocabulary.prompt.md) | Route: three words the composition literature uses that no note covers | 2026-11-13 |
| [ ] | [artifacts/drips/how-much-co-occurrence-the-product-needs/routes/02-caption-posterior.prompt.md](../_archive/3-understanding-compositionality/artifacts/drips/how-much-co-occurrence-the-product-needs/routes/02-caption-posterior.prompt.md) | Route: deriving an expert from a caption posterior | 2026-11-13 |
| [ ] | [artifacts/drips/how-much-co-occurrence-the-product-needs/while-it-runs.md](../_archive/3-understanding-compositionality/artifacts/drips/how-much-co-occurrence-the-product-needs/while-it-runs.md) | While it runs | 2026-11-13 |
| [ ] | [artifacts/drips/is-the-layout-the-missing-piece/lookup/step-03.md](../_archive/3-understanding-compositionality/artifacts/drips/is-the-layout-the-missing-piece/lookup/step-03.md) | Lookup: step 3 | 2026-11-13 |
| [ ] | [artifacts/drips/is-the-layout-the-missing-piece/while-it-runs.md](../_archive/3-understanding-compositionality/artifacts/drips/is-the-layout-the-missing-piece/while-it-runs.md) | While it runs | 2026-11-13 |
| [ ] | [artifacts/drips/reduce-reuse-recycle-mscluster112/RESUME_PROMPT.md](../_archive/3-understanding-compositionality/artifacts/drips/reduce-reuse-recycle-mscluster112/RESUME_PROMPT.md) | Where things stand | 2026-11-13 |
| [ ] | [artifacts/drips/reduce-reuse-recycle/mscluster112-prompt.md](../_archive/3-understanding-compositionality/artifacts/drips/reduce-reuse-recycle/mscluster112-prompt.md) | The open, already answered | 2026-11-13 |
| [ ] | [artifacts/drips/simple-distributions-rebuild/SWEEP.md](../_archive/3-understanding-compositionality/artifacts/drips/simple-distributions-rebuild/SWEEP.md) | Sweep: bringing the rebuild walk up to the current formats | 2026-11-13 |
| [ ] | [artifacts/drips/simple-distributions-rebuild/ladder/prompts/INDEX-02-05.md](../_archive/3-understanding-compositionality/artifacts/drips/simple-distributions-rebuild/ladder/prompts/INDEX-02-05.md) | Prompts for layers 2 to 5 | 2026-11-13 |
| [ ] | [artifacts/drips/simple-distributions-rebuild/ladder/prompts/INDEX-05a-08.md](../_archive/3-understanding-compositionality/artifacts/drips/simple-distributions-rebuild/ladder/prompts/INDEX-05a-08.md) | Prompt files for layers 5a to 8 | 2026-11-13 |
| [ ] | [artifacts/drips/simple-distributions-rebuild/ladder/prompts/INDEX-08a-12.md](../_archive/3-understanding-compositionality/artifacts/drips/simple-distributions-rebuild/ladder/prompts/INDEX-08a-12.md) | Prompts for layers 8a to 12 | 2026-11-13 |
| [ ] | [artifacts/drips/toy-failure-six-checkpoints/lookup/step-01.md](../_archive/3-understanding-compositionality/artifacts/drips/toy-failure-six-checkpoints/lookup/step-01.md) | Lookup: step 1 | 2026-11-13 |
| [ ] | [artifacts/drips/toy-failure-six-checkpoints/lookup/step-06.md](../_archive/3-understanding-compositionality/artifacts/drips/toy-failure-six-checkpoints/lookup/step-06.md) | Lookup: step 6 | 2026-11-13 |
| [ ] | [artifacts/drips/toy-failure-six-checkpoints/while-it-runs.md](../_archive/3-understanding-compositionality/artifacts/drips/toy-failure-six-checkpoints/while-it-runs.md) | While it runs | 2026-11-13 |

**4-inclusive-captions**

| Keep | File | Title | Last change |
|---|---|---|---|
| [ ] | [artifacts/drips/inclusive-captions-pilot/lookup/step-03.md](../_archive/4-inclusive-captions/artifacts/drips/inclusive-captions-pilot/lookup/step-03.md) | Lookup: step 3 | 2026-11-13 |
| [ ] | [artifacts/drips/inclusive-captions-pilot/lookup/step-04.md](../_archive/4-inclusive-captions/artifacts/drips/inclusive-captions-pilot/lookup/step-04.md) | Lookup: step 4 | 2026-11-13 |
| [ ] | [artifacts/drips/inclusive-captions-pilot/lookup/step-05.md](../_archive/4-inclusive-captions/artifacts/drips/inclusive-captions-pilot/lookup/step-05.md) | Lookup: step 5 | 2026-11-13 |
| [ ] | [artifacts/drips/inclusive-captions-pilot/lookup/step-06.md](../_archive/4-inclusive-captions/artifacts/drips/inclusive-captions-pilot/lookup/step-06.md) | Lookup: step 6 | 2026-11-13 |
| [ ] | [artifacts/drips/inclusive-captions-pilot/lookup/submerge-rank-and-what-low-rank-buys.md](../_archive/4-inclusive-captions/artifacts/drips/inclusive-captions-pilot/lookup/submerge-rank-and-what-low-rank-buys.md) | The submerge prompt: rank, and what low-rank buys | 2026-11-13 |
| [ ] | [artifacts/ideas/inclusive-captions-axis/lookup/step-02.md](../_archive/4-inclusive-captions/artifacts/ideas/inclusive-captions-axis/lookup/step-02.md) | Lookup: step 2 | 2026-11-13 |
| [ ] | [artifacts/ideas/inclusive-captions-axis/lookup/step-03.md](../_archive/4-inclusive-captions/artifacts/ideas/inclusive-captions-axis/lookup/step-03.md) | Lookup: step 3 | 2026-11-13 |
| [ ] | [artifacts/ideas/inclusive-captions-axis/while-it-runs.md](../_archive/4-inclusive-captions/artifacts/ideas/inclusive-captions-axis/while-it-runs.md) | While it runs | 2026-11-13 |
| [ ] | [artifacts/scenes/how-the-inclusive-captions-pilot-works/the-inclusive-captions-pilot-high-level-design-explainer.md](../_archive/4-inclusive-captions/artifacts/scenes/how-the-inclusive-captions-pilot-works/the-inclusive-captions-pilot-high-level-design-explainer.md) | The inclusive-captions pilot: the high-level design, explained | 2026-11-13 |

**5-text-embedding-third-expert**

| Keep | File | Title | Last change |
|---|---|---|---|
| [ ] | [artifacts/drips/a-learned-and-token/lookup/step-04.md](../_archive/5-text-embedding-third-expert/artifacts/drips/a-learned-and-token/lookup/step-04.md) | Lookup: step 4 | 2026-11-13 |
| [ ] | [artifacts/drips/can-a-positional-and-vector-restore-composition/lookup/step-02.md](../_archive/5-text-embedding-third-expert/artifacts/drips/can-a-positional-and-vector-restore-composition/lookup/step-02.md) | Lookup: step 2 | 2026-11-13 |
| [ ] | [artifacts/drips/can-a-positional-and-vector-restore-composition/lookup/step-03.md](../_archive/5-text-embedding-third-expert/artifacts/drips/can-a-positional-and-vector-restore-composition/lookup/step-03.md) | Lookup: step 3 | 2026-11-13 |
| [ ] | [artifacts/drips/today-the-third-concept/lookup/step-01.md](../_archive/5-text-embedding-third-expert/artifacts/drips/today-the-third-concept/lookup/step-01.md) | Lookup: step 1 | 2026-11-13 |
| [ ] | [artifacts/drips/today-the-third-concept/lookup/step-02.md](../_archive/5-text-embedding-third-expert/artifacts/drips/today-the-third-concept/lookup/step-02.md) | Lookup: step 2 | 2026-11-13 |
| [ ] | [artifacts/drips/today-the-third-concept/lookup/step-03.md](../_archive/5-text-embedding-third-expert/artifacts/drips/today-the-third-concept/lookup/step-03.md) | Lookup: step 3 | 2026-11-13 |
| [ ] | [artifacts/drips/today-the-third-concept/lookup/step-05.md](../_archive/5-text-embedding-third-expert/artifacts/drips/today-the-third-concept/lookup/step-05.md) | Lookup: step 3 | 2026-11-13 |
| [ ] | [artifacts/drips/today-the-third-concept/while-it-runs.md](../_archive/5-text-embedding-third-expert/artifacts/drips/today-the-third-concept/while-it-runs.md) | While it runs | 2026-11-13 |
| [ ] | [artifacts/ideas/the-and-vector-as-a-third-expert/lookup/step-01.md](../_archive/5-text-embedding-third-expert/artifacts/ideas/the-and-vector-as-a-third-expert/lookup/step-01.md) | Lookup: step 1 | 2026-11-13 |
| [ ] | [artifacts/ideas/the-and-vector-as-a-third-expert/lookup/step-02.md](../_archive/5-text-embedding-third-expert/artifacts/ideas/the-and-vector-as-a-third-expert/lookup/step-02.md) | Lookup: step 2 | 2026-11-13 |
| [ ] | [artifacts/ideas/the-and-vector-as-a-third-expert/lookup/step-04.md](../_archive/5-text-embedding-third-expert/artifacts/ideas/the-and-vector-as-a-third-expert/lookup/step-04.md) | Lookup: step 4 | 2026-11-13 |

**root**

| Keep | File | Title | Last change |
|---|---|---|---|
| [ ] | [_archive/artifacts/_quarantine/learning-captures/data-inventory-artifact-catalogue-2026-07-21/MANIFEST.md](../_archive/artifacts/_quarantine/learning-captures/data-inventory-artifact-catalogue-2026-07-21/MANIFEST.md) | Coverage Manifest: data-inventory-artifact-catalogue-2026-07-21 | 2026-11-13 |
| [ ] | [_archive/artifacts/_quarantine/learning-captures/data-inventory-artifact-catalogue-2026-07-21/NEXT-STEP-data-integrity-check.md](../_archive/artifacts/_quarantine/learning-captures/data-inventory-artifact-catalogue-2026-07-21/NEXT-STEP-data-integrity-check.md) | Next step: /data-integrity-check (paste into the downstream session) | 2026-08-24 |
| [ ] | [_archive/artifacts/drips/present-revision-to-supervisor/reviews-transcribed.md](../_archive/artifacts/drips/present-revision-to-supervisor/reviews-transcribed.md) | The three reviews of ICLR 2027 submission 56173, transcribed | 2026-11-12 |
| [ ] | [_archive/artifacts/drips/silicotb-conversation/HANDOFF.md](../_archive/artifacts/drips/silicotb-conversation/HANDOFF.md) | Silicotuberculosis conversation: what a session picking this up needs | 2026-11-13 |
| [ ] | [_archive/artifacts/notes/session-merges/2026-11-03-cat-and-dog-captions-handoff.md](../_archive/artifacts/notes/session-merges/2026-11-03-cat-and-dog-captions-handoff.md) | Handoff to the laptop: cat and dog captions | 2026-11-13 |
| [ ] | [_archive/artifacts/plans/archived/phases/05-group-a-failure.md](../_archive/artifacts/plans/archived/phases/05-group-a-failure.md) | Phase 5 — Group-A external-corrector failures | 2026-08-30 |
| [ ] | [_archive/artifacts/plans/archived/phases/RUNBOOK-12-heldout-pair-eval.md](../_archive/artifacts/plans/archived/phases/RUNBOOK-12-heldout-pair-eval.md) | Runbook — Plan 12 — Held-out-pair eval of pueuo7bl | 2026-11-13 |
| [ ] | [_archive/artifacts/plans/parked/inspector-interaction-term/11-inspector.md](../_archive/artifacts/plans/parked/inspector-interaction-term/11-inspector.md) | 🎛️ Inspector tabs for this scope's results | 2026-11-13 |
| [ ] | [_archive/artifacts/plans/parked/mechanism-study/notes/poe-interaction-term-synthesis.md](../_archive/artifacts/plans/parked/mechanism-study/notes/poe-interaction-term-synthesis.md) | The mechanism, reconciled across 7 papers + your data | 2026-09-21 |
| [ ] | [_archive/artifacts/plans/parked/mechanism-study/plans/04-what-gets-painted.md](../_archive/artifacts/plans/parked/mechanism-study/plans/04-what-gets-painted.md) | 🔬 What gets painted: probe how the LoRA fixes composition without re-a | 2026-08-30 |
| [ ] | [_archive/artifacts/plans/parked/mechanism-study/todoist/BUILD-SPEC.md](../_archive/artifacts/plans/parked/mechanism-study/todoist/BUILD-SPEC.md) | Build spec — Mechanism Study → Todoist | 2026-11-13 |
| [ ] | [_archive/composition/p2p/contributing.md](../_archive/composition/p2p/contributing.md) | How to Contribute | 2026-05-14 |
| [ ] | [_archive/paper/iclr/figures/lora-architecture.prompt.md](../_archive/paper/iclr/figures/lora-architecture.prompt.md) | Figure specification: the LoRA adapter's architecture | 2026-09-17 |
| [ ] | [_archive/paper/iclr/figures/lora-inference.prompt.md](../_archive/paper/iclr/figures/lora-inference.prompt.md) | Figure specification: sampling with the compositional adapter | 2026-09-21 |
| [ ] | [_archive/paper/iclr/figures/lora-training-and-architecture.prompt.md](../_archive/paper/iclr/figures/lora-training-and-architecture.prompt.md) | Figure specification: training the compositional adapter | 2026-09-21 |
| [ ] | [_archive/paper/iclr/figures/lora-training.prompt.md](../_archive/paper/iclr/figures/lora-training.prompt.md) | Figure specification: training the composition adapter | 2026-09-21 |
| [ ] | [_archive/scripts/local/run_queue_status.md](../_archive/scripts/local/run_queue_status.md) | Run queue | 2026-11-13 |
| [ ] | [_archive/supervisor-meeting/0-update/what-the-training-captions-show/camel-any/sheets/sheet-01.md](../_archive/supervisor-meeting/0-update/what-the-training-captions-show/camel-any/sheets/sheet-01.md) | camel-any, sheet 01 | 2026-11-04 |
| [ ] | [_archive/supervisor-meeting/0-update/what-the-training-captions-show/camel-any/sheets/sheet-02.md](../_archive/supervisor-meeting/0-update/what-the-training-captions-show/camel-any/sheets/sheet-02.md) | camel-any, sheet 02 | 2026-11-04 |
| [ ] | [_archive/supervisor-meeting/0-update/what-the-training-captions-show/camel-any/sheets/sheet-03.md](../_archive/supervisor-meeting/0-update/what-the-training-captions-show/camel-any/sheets/sheet-03.md) | camel-any, sheet 03 | 2026-11-04 |
| [ ] | [_archive/supervisor-meeting/0-update/what-the-training-captions-show/camel-wet-words/sheets/sheet-01.md](../_archive/supervisor-meeting/0-update/what-the-training-captions-show/camel-wet-words/sheets/sheet-01.md) | camel-wet-words, sheet 01 | 2026-11-04 |
| [ ] | [_archive/supervisor-meeting/0-update/what-the-training-captions-show/camel-wet-words/sheets/sheet-02.md](../_archive/supervisor-meeting/0-update/what-the-training-captions-show/camel-wet-words/sheets/sheet-02.md) | camel-wet-words, sheet 02 | 2026-11-04 |
| [ ] | [_archive/supervisor-meeting/0-update/what-the-training-captions-show/camel-wet-words/sheets/sheet-03.md](../_archive/supervisor-meeting/0-update/what-the-training-captions-show/camel-wet-words/sheets/sheet-03.md) | camel-wet-words, sheet 03 | 2026-11-04 |
| [ ] | [_archive/supervisor-meeting/0-update/what-the-training-captions-show/camel-wet-words/sheets/sheet-04.md](../_archive/supervisor-meeting/0-update/what-the-training-captions-show/camel-wet-words/sheets/sheet-04.md) | camel-wet-words, sheet 04 | 2026-11-04 |
| [ ] | [_archive/supervisor-meeting/0-update/what-the-training-captions-show/camel-wet-words/sheets/sheet-05.md](../_archive/supervisor-meeting/0-update/what-the-training-captions-show/camel-wet-words/sheets/sheet-05.md) | camel-wet-words, sheet 05 | 2026-11-04 |
| [ ] | [_archive/supervisor-meeting/0-update/what-the-training-captions-show/dog-any/sheets/sheet-01.md](../_archive/supervisor-meeting/0-update/what-the-training-captions-show/dog-any/sheets/sheet-01.md) | dog-any, sheet 01 | 2026-11-04 |
| [ ] | [_archive/supervisor-meeting/0-update/what-the-training-captions-show/dog-any/sheets/sheet-02.md](../_archive/supervisor-meeting/0-update/what-the-training-captions-show/dog-any/sheets/sheet-02.md) | dog-any, sheet 02 | 2026-11-04 |
| [ ] | [_archive/supervisor-meeting/0-update/what-the-training-captions-show/dog-any/sheets/sheet-03.md](../_archive/supervisor-meeting/0-update/what-the-training-captions-show/dog-any/sheets/sheet-03.md) | dog-any, sheet 03 | 2026-11-04 |
| [ ] | [_archive/supervisor-meeting/0-update/what-the-training-captions-show/dog-wet-words/sheets/sheet-01.md](../_archive/supervisor-meeting/0-update/what-the-training-captions-show/dog-wet-words/sheets/sheet-01.md) | dog-wet-words, sheet 01 | 2026-11-04 |
| [ ] | [_archive/supervisor-meeting/0-update/what-the-training-captions-show/dog-wet-words/sheets/sheet-02.md](../_archive/supervisor-meeting/0-update/what-the-training-captions-show/dog-wet-words/sheets/sheet-02.md) | dog-wet-words, sheet 02 | 2026-11-04 |
| [ ] | [_archive/supervisor-meeting/0-update/what-the-training-captions-show/dog-wet-words/sheets/sheet-03.md](../_archive/supervisor-meeting/0-update/what-the-training-captions-show/dog-wet-words/sheets/sheet-03.md) | dog-wet-words, sheet 03 | 2026-11-04 |

## 4. `poe_repair` modules nothing imports or names

No code imports these and no markdown, shell script or job file names them. All ten date
from 2026-08-24 or earlier.

| Keep | Module | Last change |
|---|---|---|
| [ ] | [_archive/poe_repair/experiments/compose_scorer_validation/anchor_contact_sheet.py](../_archive/poe_repair/experiments/compose_scorer_validation/anchor_contact_sheet.py) | 2026-08-24 |
| [ ] | [_archive/poe_repair/experiments/compose_scorer_validation/diagnose_figure.py](../_archive/poe_repair/experiments/compose_scorer_validation/diagnose_figure.py) | 2026-08-24 |
| [ ] | [_archive/poe_repair/experiments/compose_scorer_validation/f1_scorer_works.py](../_archive/poe_repair/experiments/compose_scorer_validation/f1_scorer_works.py) | 2026-08-24 |
| [ ] | [_archive/poe_repair/experiments/compose_scorer_validation/gen_anchors.py](../_archive/poe_repair/experiments/compose_scorer_validation/gen_anchors.py) | 2026-08-24 |
| [ ] | [_archive/poe_repair/experiments/compose_scorer_validation/gen_validation_outputs.py](../_archive/poe_repair/experiments/compose_scorer_validation/gen_validation_outputs.py) | 2026-08-24 |
| [ ] | [_archive/poe_repair/experiments/mechanism_study/decode_samples.py](../_archive/poe_repair/experiments/mechanism_study/decode_samples.py) | 2026-08-24 |
| [ ] | [_archive/poe_repair/experiments/mechanism_study/export_sweep_data.py](../_archive/poe_repair/experiments/mechanism_study/export_sweep_data.py) | 2026-08-24 |
| [ ] | [_archive/poe_repair/experiments/mechanism_study/export_trace.py](../_archive/poe_repair/experiments/mechanism_study/export_trace.py) | 2026-08-24 |
| [ ] | [_archive/poe_repair/experiments/mechanism_study/export_viewer_data.py](../_archive/poe_repair/experiments/mechanism_study/export_viewer_data.py) | 2026-08-24 |
| [ ] | [_archive/poe_repair/experiments/mechanism_study/mechanism_across_training.py](../_archive/poe_repair/experiments/mechanism_study/mechanism_across_training.py) | 2026-08-24 |
