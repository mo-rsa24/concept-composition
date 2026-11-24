# 🗂️ Drain census: the loops' figures and results that no pillar page shows

This page lists every picture, scene page and review file the looping sessions made that no page under `report/`, `context/`, `runbook/` or `environment/` links or embeds, with the page each should land on. It was written on 2026-11-24 by `python3 artifacts/notes/progress/drain/census.py --markdown artifacts/notes/progress/drain/01-census.md`, run from the repository root, and the landing pages come from [the landing rules](landing.py). Rerun it to refresh the page.

## Words this page uses

**Placed** means a pillar page links or embeds the file's exact path. **Folder only** means a pillar page links a folder above the file but not the file. **Unplaced** means neither. **Made by a loop** means the file first entered git on or after 2026-11-21, when the looping sessions started, or is not in git yet. Results folders count as made before the loops even where a folder move on 2026-11-22 re-added them to git. **What it shows** is the caption the loop wrote where it embeds or links the picture, the review's own opening sentences for a review, and otherwise the file name in plain words, marked *(file name)*.

## Why the drain never found these

The drain's sweep (`/artifacts-drain . --dry-run`) found 8 new and 4 changed files on this run, all of them scene cards under the root `artifacts/scenes/`. None of the loops' work was among them, for two reasons:

1. The drain ledger, `report/.artifacts-drain-ledger.json`, holds `artifacts/notes/progress` out of every drain, and every loop folder (`targets/`, `reproductions/`, `method-search/`, `abstract-checkers/`, `noise-text-critic/`, `score-space/`) sits inside it.
2. The sweep reads only the root `artifacts/`. It never looks in `2-classifier-guidance/artifacts/` or `1-lora-correction/artifacts/`, where the loops' scenes and the older results live.

## The counts

| Where it came from | Pictures and scene pages, unplaced | Reviews holding a verdict | Reviews with no verdict yet |
|---|---|---|---|
| Loop 4 | 9 | 0 | 0 |
| Loop 5 | 8 | 0 | 1 |
| Loop 6 | 11 | 0 | 1 |
| Loop 7 | 12 | 0 | 0 |
| Loop 8 | 3 | 0 | 0 |
| Loop 9 | 7 | 0 | 0 |
| Loop 10 | 18 | 0 | 0 |
| scenes in `2-classifier-guidance/artifacts/scenes/` | 198 | 0 | 0 |
| older results and scenes, made before the loops | 406 | not counted | not counted |

## Results with a verdict, for step Y2

Each row is a review file whose verdict is recorded. The verdict column quotes the review's opening sentences; Y2 copies the full verdict, numbers, bar and run id from the review itself.

| File | Loop | What the review records | Lands on |
|---|---|---|---|

## Reviews with no verdict yet

These get no finding until their verdict lands. Y5 checks them on every tick.

| File | Loop | What the review records | Why it waits |
|---|---|---|---|
| [review.md](../../../../artifacts/notes/progress/reproductions/CoInD/review.md) `CoInD` | Loop 5 | This file holds what the runs in the CoInD design produced. CelebA is context by the owner's answer to board question 35, so this rerun stopped after the classifier and the smoke test | no finding: CelebA was made context by the owner's answer to board question 35, so no generated face was scored |
| [review.md](../../../../artifacts/notes/progress/method-search/01c-clevr-paste-first/review.md) `01c-clevr-paste-first` | Loop 6 | This file holds what the runs in the M7 design produced. M7 has no verdict yet. | waits for a verdict: the step is still running (M7) |

## Pictures and scene pages, by loop

Folders with more than ten files are folded; open one to see every file.

### Loop 4

**`artifacts/notes/progress/targets/`: 8 files**

| File | What it shows | Lands on |
|---|---|---|
| [how-one-rival-gets-reproduced.png](../../../../artifacts/notes/progress/targets/how-one-rival-gets-reproduced.png) | how-one-rival-gets-reproduced.png | none: a picture of the route's own plan, which stays beside the plan it explains |
| [how-the-clevr-target-gets-tested.png](../../../../artifacts/notes/progress/targets/how-the-clevr-target-gets-tested.png) | how-the-clevr-target-gets-tested.png | none: a picture of the route's own plan, which stays beside the plan it explains |
| [one-clevr-experiment-for-both-papers.png](../../../../artifacts/notes/progress/targets/one-clevr-experiment-for-both-papers.png) | One CLEVR experiment for both papers | none: a picture of the route's own plan, which stays beside the plan it explains |
| [one-network-against-separate-experts.png](../../../../artifacts/notes/progress/targets/one-network-against-separate-experts.png) | one-network-against-separate-experts.png | `context/methods/01-the-product-of-experts/01-overview.md` (to write) |
| [path-to-a-filled-scoreboard.png](../../../../artifacts/notes/progress/targets/path-to-a-filled-scoreboard.png) | path-to-a-filled-scoreboard.png | none: a picture of the route's own plan, which stays beside the plan it explains |
| [rivals-by-dataset.png](../../../../artifacts/notes/progress/targets/rivals-by-dataset.png) | rivals-by-dataset.png | `1-lora-correction/report/against-other-methods/do-the-rivals-reproduce-their-printed-scores.md`, which rival reports on which benchmark |
| [what-we-are-doing-and-the-three-detours.png](../../../../artifacts/notes/progress/targets/what-we-are-doing-and-the-three-detours.png) | what-we-are-doing-and-the-three-detours.png | none: a picture of the route's own plan, which stays beside the plan it explains |
| [where-clevr-sits-before-the-go-or-no-go.png](../../../../artifacts/notes/progress/targets/where-clevr-sits-before-the-go-or-no-go.png) | where-clevr-sits-before-the-go-or-no-go.png | none: a picture of the route's own plan, which stays beside the plan it explains |

**`artifacts/notes/progress/targets/reads/`: 1 file**

| File | What it shows | Lands on |
|---|---|---|
| [how-a-number-becomes-a-target.png](../../../../artifacts/notes/progress/targets/reads/how-a-number-becomes-a-target.png) | how-a-number-becomes-a-target.png | `1-lora-correction/report/against-other-methods/do-the-rivals-reproduce-their-printed-scores.md`, how a printed number becomes a bar |

### Loop 5

**`artifacts/notes/progress/reproductions/`: 2 files**

| File | What it shows | Lands on |
|---|---|---|
| [what-each-row-waits-on.png](../../../../artifacts/notes/progress/reproductions/what-each-row-waits-on.png) | what-each-row-waits-on.png | none: a picture of the route's own plan, which stays beside the plan it explains |
| [where-we-go-from-here.png](../../../../artifacts/notes/progress/reproductions/where-we-go-from-here.png) | where-we-go-from-here.png | none: a picture of the route's own plan, which stays beside the plan it explains |

**`artifacts/notes/progress/reproductions/CO3/`: 1 file**

| File | What it shows | Lands on |
|---|---|---|
| [paper-against-released-script.png](../../../../artifacts/notes/progress/reproductions/CO3/paper-against-released-script.png) | paper-against-released-script.png | waits for a verdict: the step is still running |

**`artifacts/notes/progress/reproductions/CoMat/`: 1 file**

| File | What it shows | Lands on |
|---|---|---|
| [comat-one-percent-seed42.png](../../../../artifacts/notes/progress/reproductions/CoMat/comat-one-percent-seed42.png) | sheet of 9 prompts at seed 42 | waits for a verdict: the step is still running |

**`artifacts/notes/progress/reproductions/RTD/`: 2 files**

| File | What it shows | Lands on |
|---|---|---|
| [where-the-rtd-run-stands.png](../../../../artifacts/notes/progress/reproductions/RTD/where-the-rtd-run-stands.png) | where-the-rtd-run-stands.png | `1-lora-correction/report/against-other-methods/do-the-rivals-reproduce-their-printed-scores.md` |
| [which-job-renders-which-prompts.png](../../../../artifacts/notes/progress/reproductions/RTD/which-job-renders-which-prompts.png) | which-job-renders-which-prompts.png | `1-lora-correction/report/against-other-methods/do-the-rivals-reproduce-their-printed-scores.md` |

**`artifacts/notes/progress/reproductions/ToMe/`: 1 file**

| File | What it shows | Lands on |
|---|---|---|
| [tome-one-percent-seed0.png](../../../../artifacts/notes/progress/reproductions/ToMe/tome-one-percent-seed0.png) | tome-one-percent-seed0.png | waits for a verdict: the step is still running |

**`artifacts/notes/progress/reproductions/inclusive-finetune-and-push/`: 1 file**

| File | What it shows | Lands on |
|---|---|---|
| [the-four-conditions.png](../../../../artifacts/notes/progress/reproductions/inclusive-finetune-and-push/the-four-conditions.png) | the-four-conditions.png | waits for a verdict: the step is still running (Loop 6's inclusive fine-tune) |

### Loop 6

**`artifacts/notes/progress/method-search/`: 5 files**

| File | What it shows | Lands on |
|---|---|---|
| [abstract-checker-idea.png](../../../../artifacts/notes/progress/method-search/abstract-checker-idea.png) | abstract-checker-idea.png | `context/methods/03-the-digit-map-checkers/01-overview.md` (to write), the idea before Loop 8 tested it |
| [how-loops-6-7-8-connect.png](../../../../artifacts/notes/progress/method-search/how-loops-6-7-8-connect.png) | how-loops-6-7-8-connect.png | none: a picture of the route's own plan, which stays beside the plan it explains |
| [how-the-checker-could-scale.png](../../../../artifacts/notes/progress/method-search/how-the-checker-could-scale.png) | how-the-checker-could-scale.png | `context/methods/02-the-readout-checker/01-overview.md` (to write) |
| [loop-6-route.png](../../../../artifacts/notes/progress/method-search/loop-6-route.png) | loop-6-route.png | none: a picture of the route's own plan, which stays beside the plan it explains |
| [loop-6-scoreboard.png](../../../../artifacts/notes/progress/method-search/loop-6-scoreboard.png) | loop-6-scoreboard.png | none: a picture of the route's own plan, which stays beside the plan it explains |

**`artifacts/notes/progress/method-search/01c-clevr-paste-first/`: 6 files**

| File | What it shows | Lands on |
|---|---|---|
| [full-p1-first-16-samples.png](../../../../artifacts/notes/progress/method-search/01c-clevr-paste-first/full-p1-first-16-samples.png) | Sixteen samples from the position 1 expert, each a single CLEVR object at the upper left on an empty floor | waits for a verdict: the step is still running (M7) |
| [overfit-batch-256-real-and-sampled.png](../../../../artifacts/notes/progress/method-search/01c-clevr-paste-first/overfit-batch-256-real-and-sampled.png) | Top two rows: the 16 real pictures. Bottom two rows: 16 samples showing the same scene with one clear object at the same position | waits for a verdict: the step is still running (M7) |
| [overfit-scaled-ramp-real-and-sampled.png](../../../../artifacts/notes/progress/method-search/01c-clevr-paste-first/overfit-scaled-ramp-real-and-sampled.png) | Top two rows: the 16 real pictures. Bottom two rows: 16 samples with the right floor and lighting and faint coloured blotches in place of objects | waits for a verdict: the step is still running (M7) |
| [overfit-unscaled-ramp-real-and-sampled.png](../../../../artifacts/notes/progress/method-search/01c-clevr-paste-first/overfit-unscaled-ramp-real-and-sampled.png) | Top two rows: the 16 real pictures of one object. Bottom two rows: 16 samples from the model, which are colour noise | waits for a verdict: the step is still running (M7) |
| [smoke-objects-at-their-positions.png](../../../../artifacts/notes/progress/method-search/01c-clevr-paste-first/smoke-objects-at-their-positions.png) | Twelve CLEVR renders at 64 pixels: four with one object at position 1, four empty scenes, and four with objects at all three positions, each asked position outlined by its 18-pixel paste window | waits for a verdict: the step is still running (M7) |
| [window-control-written-rule.png](../../../../artifacts/notes/progress/method-search/01c-clevr-paste-first/window-control-written-rule.png) | How often each pixel changes, for the three positions, with the 18 by 18 window drawn: position 3's object runs past the top and bottom of its window | waits for a verdict: the step is still running (M7) |

### Loop 7

<details><summary>`artifacts/notes/progress/abstract-checkers/01-reading/`: 12 files</summary>

| File | What it shows | Lands on |
|---|---|---|
| [adding-logs-error-by-noise.png](../../../../artifacts/notes/progress/abstract-checkers/01-reading/adding-logs-error-by-noise.png) | Error of the summed logs against noise level for three pairs of labels on a small example. Labels on separate pixels have zero error. Independent labels on the same pixels peak near 0.14 nats. Labels where one implies the other reach 0.69… | `context/methods/03-the-digit-map-checkers/01-overview.md` (to write), background from the reading |
| [checker-and-critic-compared.png](../../../../artifacts/notes/progress/abstract-checkers/01-reading/checker-and-critic-compared.png) | checker-and-critic-compared.png | `context/methods/03-the-digit-map-checkers/01-overview.md` (to write), background from the reading |
| [five-checkers-as-a-chain.png](../../../../artifacts/notes/progress/abstract-checkers/01-reading/five-checkers-as-a-chain.png) | Word maps feed a count checker trained on every picture. Spatial relation, position and size, binding and prompt match train only on pictures holding two separate subjects. Each checker pushes at its own fixed length, added to the product… | `context/methods/03-the-digit-map-checkers/01-overview.md` (to write), background from the reading |
| [glide-clip-score-against-people.png](../../../../artifacts/notes/progress/abstract-checkers/01-reading/glide-clip-score-against-people.png) | GLIDE's 16 random samples for CLIP guidance and classifier-free guidance on one prompt, beside its CLIP score plot and human Elo ratings. | `context/methods/04-the-sentence-score-push/01-overview.md` (to write), a figure reproduced from the GLIDE paper |
| [one-concept-as-one-score.png](../../../../artifacts/notes/progress/abstract-checkers/01-reading/one-concept-as-one-score.png) | one-concept-as-one-score.png | `context/methods/03-the-digit-map-checkers/01-overview.md` (to write), background from the reading |
| [removal-test.png](../../../../artifacts/notes/progress/abstract-checkers/01-reading/removal-test.png) | Switching off one of five checkers and scoring all five properties, with the prediction that each removal lowers only its own property. A second panel shows the shared cap handing a removed checker's share to the others, fixed by giving ea… | `context/methods/03-the-digit-map-checkers/01-overview.md` (to write), background from the reading |
| [style-annotated-explainer-sentence-score.png](../../../../artifacts/notes/progress/abstract-checkers/01-reading/style-annotated-explainer-sentence-score.png) | style-annotated-explainer-sentence-score.png | none: a style sample for drawing, not a result or a method |
| [three-ways-to-score-n-objects.png](../../../../artifacts/notes/progress/abstract-checkers/01-reading/three-ways-to-score-n-objects.png) | A prompt with N subjects feeds three readers of the noisy latent: word-map heads for where things are, a question head on SDXL features for what they are, and a sentence score for the whole picture. Each reader has its own fixed push lengt… | `context/methods/03-the-digit-map-checkers/01-overview.md` (to write), background from the reading |
| [what-score-addition-means.png](../../../../artifacts/notes/progress/abstract-checkers/01-reading/what-score-addition-means.png) | what-score-addition-means.png | `context/methods/03-the-digit-map-checkers/01-overview.md` (to write), background from the reading |
| [what-the-closer-papers-read.png](../../../../artifacts/notes/progress/abstract-checkers/01-reading/what-the-closer-papers-read.png) | The seven closer papers grouped by whether they read a noisy input or clean images, and by how each reaches a generator. None trains a checker on word maps from a noisy latent, tests it on held-out concepts and steers a sampler with it. | none: a picture of the route's own plan, which stays beside the plan it explains |
| [what-the-papers-cover.png](../../../../artifacts/notes/progress/abstract-checkers/01-reading/what-the-papers-cover.png) | Word attention maps feed five proposed checkers. Only the spatial relation checker has a trained precedent, Learn-to-Steer on FLUX.1-dev. Counting, binding, position and size, and prompt match are gaps. Their gradients are summed, capped a… | none: a picture of the route's own plan, which stays beside the plan it explains |
| [where-each-model-can-reach-the-latent.png](../../../../artifacts/notes/progress/abstract-checkers/01-reading/where-each-model-can-reach-the-latent.png) | where-each-model-can-reach-the-latent.png | `context/methods/03-the-digit-map-checkers/01-overview.md` (to write), background from the reading |

</details>

### Loop 8

**`artifacts/notes/progress/abstract-checkers/`: 1 file**

| File | What it shows | Lands on |
|---|---|---|
| [how-the-loops-fit.png](../../../../artifacts/notes/progress/abstract-checkers/how-the-loops-fit.png) | Three loops as boxes. Loop 7 reads papers, works out the score maths and writes the design. Loop 8 builds the digit world, asks whether a checker learns plurality on unseen digits, whether its gradient composes them, and whether it works o… | none: a picture of the route's own plan, which stays beside the plan it explains |

**`artifacts/notes/progress/abstract-checkers/09-summary/`: 2 files**

| File | What it shows | Lands on |
|---|---|---|
| [evidence-at-a-glance.png](../../../../artifacts/notes/progress/abstract-checkers/09-summary/evidence-at-a-glance.png) | Left: plain-product canvases for digit pairs 5 and 6 to 5 and 9, each a single fused glyph; below, the same seeds with the count checker's push, many now two separate digits, a few framed green as clean. Right: clean pairs of 320, plain 0,… | none: a picture of the route's own plan, which stays beside the plan it explains |
| [where-loop-8-stands.png](../../../../artifacts/notes/progress/abstract-checkers/09-summary/where-loop-8-stands.png) | Loop 8's steps in order with their verdicts, V4 not run; Loops 9 and 10 point to V2 as agreeing that identity is unreadable early and that SDXL's noise level carries more signal; Loop 6 points to V4 as the risk that counting is what the ri… | none: a picture of the route's own plan, which stays beside the plan it explains |

### Loop 9

**`artifacts/notes/progress/noise-text-critic/`: 5 files**

| File | What it shows | Lands on |
|---|---|---|
| [ideas-and-where-they-land.png](../../../../artifacts/notes/progress/noise-text-critic/ideas-and-where-they-land.png) | Six owner ideas, from using Latent CLIP directly to training our own, each with an arrow to the step that tests it: W2 and W3 for the zero-shot push, W2b for linear layers, W4 for training our own, and swapping SDXL's text encoder marked n… | none: a picture of the route's own plan, which stays beside the plan it explains |
| [three-way-space-idea.png](../../../../artifacts/notes/progress/noise-text-critic/three-way-space-idea.png) | Text through frozen CLIP, a half-drawn canvas through a latent encoder told the step, and the noise predictions along the path through a new score encoder all meet in one shared space, which is tested on digit pairs never seen together by… | `context/methods/07-the-score-path/01-overview.md` (to write) |
| [way-1-noisyclip-push.png](../../../../artifacts/notes/progress/noise-text-critic/way-1-noisyclip-push.png) | way-1-noisyclip-push.png | `context/methods/04-the-sentence-score-push/01-overview.md` (to write) |
| [way-2-latent-clip-push.png](../../../../artifacts/notes/progress/noise-text-critic/way-2-latent-clip-push.png) | way-2-latent-clip-push.png | `context/methods/05-latent-clip/01-overview.md` (to write) |
| [where-the-pieces-sit.png](../../../../artifacts/notes/progress/noise-text-critic/where-the-pieces-sit.png) | where-the-pieces-sit.png | none: a picture of the route's own plan, which stays beside the plan it explains |

**`artifacts/notes/progress/noise-text-critic/01-reading/`: 2 files**

| File | What it shows | Lands on |
|---|---|---|
| [latent-clip-beside-noisyclip.png](../../../../artifacts/notes/progress/noise-text-critic/01-reading/latent-clip-beside-noisyclip.png) | A half-drawn SDXL latent goes either through NoisyCLIP or through Latent CLIP. Both give a right-minus-wrong sentence score. W2 asks whether it reads at steps 0 to 15, and only steps that read reach the push on CO3 in W3. | `context/methods/05-latent-clip/01-overview.md` (to write) |
| [sentence-score-push.png](../../../../artifacts/notes/progress/noise-text-critic/01-reading/sentence-score-push.png) | A half-drawn SDXL latent passes through a fixed linear map into the NoisyCLIP image encoder. A right and a wrong sentence pass through the frozen CLIP text encoder. The two scores are subtracted, and the gradient of that difference is adde… | `context/methods/04-the-sentence-score-push/01-overview.md` (to write) |

### Loop 10

**`artifacts/notes/progress/score-space/`: 2 files**

| File | What it shows | Lands on |
|---|---|---|
| [03-score-path-defined-two-ways.png](../../../../artifacts/notes/progress/score-space/03-score-path-defined-two-ways.png) | Top: a canvas with a 3 and a 7 goes into the digit model together with the labels 3 and 7, marked in red as the answer; the path d_t(3), d_t(7) lets a reader recover 3 and 7 from the labels it was handed, also marked red. Bottom, in blue:… | `context/methods/07-the-score-path/01-overview.md` (to write) |
| [summary-where-loop-10-fits.png](../../../../artifacts/notes/progress/score-space/summary-where-loop-10-fits.png) | Loop 9's sentence scores and Loop 8's two-map checker feed into Loop 10's finding that all ten signed maps read which digits are present at 0.655. Dashed, untested arrows lead to Paper 1 (feed the checker all ten maps) and to Loop 6 and Pa… | `context/methods/07-the-score-path/01-overview.md` (to write) |

**`artifacts/notes/progress/score-space/00-ladder/`: 1 file**

| File | What it shows | Lands on |
|---|---|---|
| [style-sample-score-path.png](../../../../artifacts/notes/progress/score-space/00-ladder/style-sample-score-path.png) | style-sample-score-path.png | none: a style sample for drawing, not a result or a method |

**`artifacts/notes/progress/score-space/00-ladder/0.5-network-view/`: 5 files**

| File | What it shows | Lands on |
|---|---|---|
| [sp1-inside-the-digit-unet.png](../../../../artifacts/notes/progress/score-space/00-ladder/0.5-network-view/sp1-inside-the-digit-unet.png) | The digit U-Net as a U with real sizes | `context/methods/07-the-score-path/02-the-network-view.md` (to write) |
| [sp2-sixteen-steps-make-a-score-path.png](../../../../artifacts/notes/progress/score-space/00-ladder/0.5-network-view/sp2-sixteen-steps-make-a-score-path.png) | The real window: 16 steps with their timesteps and signal scales, empty d_t tiles, and three stacks of 16 x 32 x 64 | `context/methods/07-the-score-path/02-the-network-view.md` (to write) |
| [sp3-the-planned-score-path-encoder.png](../../../../artifacts/notes/progress/score-space/00-ladder/0.5-network-view/sp3-the-planned-score-path-encoder.png) | Three encoders into one width, the score-path one dashed and planned, beside X4's open questions | `context/methods/07-the-score-path/02-the-network-view.md` (to write) |
| [sp4-three-spaces-three-matrices.png](../../../../artifacts/notes/progress/score-space/00-ladder/0.5-network-view/sp4-three-spaces-three-matrices.png) | Three 4 by 4 similarity matrices with pale blue diagonals and one summed loss | `context/methods/07-the-score-path/02-the-network-view.md` (to write) |
| [sys0-where-the-score-path-sits.png](../../../../artifacts/notes/progress/score-space/00-ladder/0.5-network-view/sys0-where-the-score-path-sits.png) | System picture: canvas, frozen U-Net, d_t, score path, planned encoder and shared space, with the CLIP-L text lane and the W4 canvas lane | `context/methods/07-the-score-path/02-the-network-view.md` (to write) |

**`artifacts/notes/progress/score-space/00-ladder/2-miniature/`: 4 files**

| File | What it shows | Lands on |
|---|---|---|
| [mm1-one-step-three-calls.png](../../../../artifacts/notes/progress/score-space/00-ladder/2-miniature/mm1-one-step-three-calls.png) | One toy step: three calls, two differences, one sum, on 4 pixels | `context/methods/07-the-score-path/03-the-miniature.md` (to write) |
| [mm2-two-steps-make-a-score-path.png](../../../../artifacts/notes/progress/score-space/00-ladder/2-miniature/mm2-two-steps-make-a-score-path.png) | Two toy steps on 4 pixels with d(3) and d(7) at each, laid end to end into 8-number score paths | `context/methods/07-the-score-path/03-the-miniature.md` (to write) |
| [mm3-the-encoder-and-the-contrastive-score.png](../../../../artifacts/notes/progress/score-space/00-ladder/2-miniature/mm3-the-encoder-and-the-contrastive-score.png) | A 2 by 8 integer encoder, embeddings, and a 3 by 3 contrastive matrix with softmax and loss | `context/methods/07-the-score-path/03-the-miniature.md` (to write) |
| [mm4-the-two-tests-in-miniature.png](../../../../artifacts/notes/progress/score-space/00-ladder/2-miniature/mm4-the-two-tests-in-miniature.png) | The two tests in a 2-number space: caption ranking and gap cosine | `context/methods/07-the-score-path/03-the-miniature.md` (to write) |

**`artifacts/notes/progress/score-space/00-ladder/overview/`: 5 files**

| File | What it shows | Lands on |
|---|---|---|
| [canvas-3-7-native.png](../../../../artifacts/notes/progress/score-space/00-ladder/overview/canvas-3-7-native.png) | canvas-3-7-native.png | `context/methods/07-the-score-path/01-overview.md` (to write) |
| [canvas-3-7-step0-t999.png](../../../../artifacts/notes/progress/score-space/00-ladder/overview/canvas-3-7-step0-t999.png) | canvas-3-7-step0-t999.png | `context/methods/07-the-score-path/01-overview.md` (to write) |
| [canvas-3-7-step15-t693.png](../../../../artifacts/notes/progress/score-space/00-ladder/overview/canvas-3-7-step15-t693.png) | canvas-3-7-step15-t693.png | `context/methods/07-the-score-path/01-overview.md` (to write) |
| [canvas-3-7.png](../../../../artifacts/notes/progress/score-space/00-ladder/overview/canvas-3-7.png) | canvas-3-7.png | `context/methods/07-the-score-path/01-overview.md` (to write) |
| [ov0-the-score-path-as-a-third-space.png](../../../../artifacts/notes/progress/score-space/00-ladder/overview/ov0-the-score-path-as-a-third-space.png) | A real canvas of a 7 and a 3 noised to step 0, the frozen digit U-Net called with no digit, 3 and 7, the two d_t stacks over steps 0 to 15, three encoders (CLIP-L text, W4 canvas, planned score-path), a planned shared space with hand-place… | `context/methods/07-the-score-path/01-overview.md` (to write) |

**`artifacts/notes/progress/score-space/01-reading/`: 1 file**

| File | What it shows | Lands on |
|---|---|---|
| [three-spaces-and-the-closest-papers.png](../../../../artifacts/notes/progress/score-space/01-reading/three-spaces-and-the-closest-papers.png) | Three inputs, a caption, a noisy canvas and a score path, each with an arrow toward one shared space. The caption reaches it through a frozen text encoder and the canvas through a trained image encoder. The score path's arrow is dashed, la… | `context/methods/07-the-score-path/01-overview.md` (to write) |

### Scenes in `2-classifier-guidance/artifacts/scenes/`

**`2-classifier-guidance/artifacts/scenes/annotated-overviews/`: 5 files**

| File | What it shows | Lands on |
|---|---|---|
| [loop-4-question-transformer.png](../../../../2-classifier-guidance/artifacts/scenes/annotated-overviews/loop-4-question-transformer.png) | How BLIP's question reads the picture | `context/methods/08-the-blip-vqa-score/01-overview.md` (to write) |
| [loop-4.png](../../../../2-classifier-guidance/artifacts/scenes/annotated-overviews/loop-4.png) | Loop 4 overview | `context/methods/08-the-blip-vqa-score/01-overview.md` (to write) |
| [loop-6-kept-checker.png](../../../../2-classifier-guidance/artifacts/scenes/annotated-overviews/loop-6-kept-checker.png) | Inside the kept checker | `context/methods/02-the-readout-checker/01-overview.md` (to write) |
| [loop-6.png](../../../../2-classifier-guidance/artifacts/scenes/annotated-overviews/loop-6.png) | Loop 6 overview | `context/methods/02-the-readout-checker/01-overview.md` (to write) |
| [loop-8.png](../../../../2-classifier-guidance/artifacts/scenes/annotated-overviews/loop-8.png) | Loop 8 overview | `context/methods/03-the-digit-map-checkers/01-overview.md` (to write) |

**`2-classifier-guidance/artifacts/scenes/digit-map-checkers/0.5-network-view/`: 8 files**

| File | What it shows | Lands on |
|---|---|---|
| [dc0-where-the-checkers-sit.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/0.5-network-view/dc0-where-the-checkers-sit.png) | One sampler step on the digit canvas: three frozen U-Net calls give the product and the two maps, the checker turns the maps into one logit, and the logit's gradient becomes a push of fixed length 0.4 of the product's | `context/methods/03-the-digit-map-checkers/` (to write), one page per rung, overview first |
| [dc1-the-canvas-and-its-noise.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/0.5-network-view/dc1-the-canvas-and-its-noise.png) | An apart canvas showing 1 and 3 and a wrong canvas showing 3 and 2, clean and at timesteps 712, 616, 539 and 480; at all four noise levels the digits cannot be seen by eye | `context/methods/03-the-digit-map-checkers/` (to write), one page per rung, overview first |
| [dc2-where-the-two-maps-come-from.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/0.5-network-view/dc2-where-the-two-maps-come-from.png) | The noisy canvas through three frozen U-Net calls, the two differences from the no-digit call, and the scaling that makes each map add to 2,048; the map tiles are marked as owed | `context/methods/03-the-digit-map-checkers/` (to write), one page per rung, overview first |
| [dc3-the-checker-network.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/0.5-network-view/dc3-the-checker-network.png) | CNN volumes from 2 by 32 by 64 maps through three stem convolutions to 128 by 8 by 16, c1, scale and shift from the time features, c2, mean and max pooling to 256, and an MLP from 272 to one logit; 481,665 weights | `context/methods/03-the-digit-map-checkers/` (to write), one page per rung, overview first |
| [dc4-the-timestep-as-sixteen-numbers.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/0.5-network-view/dc4-the-timestep-as-sixteen-numbers.png) | A 16 by 16 heat map of the real time features, one row per training timestep, sin columns then cos columns, values from minus 1 to plus 1 | `context/methods/03-the-digit-map-checkers/` (to write), one page per rung, overview first |
| [dc5-two-checkers-two-label-sets.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/0.5-network-view/dc5-two-checkers-two-label-sets.png) | The five canvas kinds for the asked pair 1 and 3 with the count label and the identity label under each; count trains on 11,504 canvases and identity on 7,004, of which 1,004 are wrong | `context/methods/03-the-digit-map-checkers/` (to write), one page per rung, overview first |
| [dc6-training-backprop-into-the-weights.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/0.5-network-view/dc6-training-backprop-into-the-weights.png) | The forward pass from the maps through the checker to the logit and the loss, and the backward lane carrying sigma of the logit minus y into each trained layer's weights and stopping before the maps and the U-Net | `context/methods/03-the-digit-map-checkers/` (to write), one page per rung, overview first |
| [dc7-the-push-gradient-into-the-canvas.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/0.5-network-view/dc7-the-push-gradient-into-the-canvas.png) | The forward pass from the noisy canvas through the three U-Net calls, the maps and the checker to the logit, the backward lane all the way to the canvas, the push formula, and a table comparing the two backward passes | `context/methods/03-the-digit-map-checkers/` (to write), one page per rung, overview first |

**`2-classifier-guidance/artifacts/scenes/digit-map-checkers/2-miniature/`: 3 files**

| File | What it shows | Lands on |
|---|---|---|
| [mm1-the-forward-pass-with-every-number.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/2-miniature/mm1-the-forward-pass-with-every-number.png) | For the apart and the wrong canvas: the clean pixels, the noisy pixels, the two maps, the hidden values per half and channel, the pooled four numbers and the logit, 0.772 for apart and 0.352 for wrong | `context/methods/03-the-digit-map-checkers/` (to write), one page per rung, overview first |
| [mm2-one-training-step-per-branch.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/2-miniature/mm2-one-training-step-per-branch.png) | Logits before and after one step: count moves both canvases up, from 0.772 to 1.459 and from 0.352 to 0.997; identity moves both down, from 0.772 to 0.565 and from 0.352 to 0.146, and the gap between them goes from 0.4205 to 0.4192 | `context/methods/03-the-digit-map-checkers/` (to write), one page per rung, overview first |
| [mm3-the-push-on-the-four-pixels.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/2-miniature/mm3-the-push-on-the-four-pixels.png) | For each canvas: the noisy pixels, the gradient of the logit with respect to them, the product's noise guess, the push scaled to 0.4 of the product's length, and the pushed guess | `context/methods/03-the-digit-map-checkers/` (to write), one page per rung, overview first |

**`2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/`: 2 files**

| File | What it shows | Lands on |
|---|---|---|
| [digit-map-checkers-miniature.html](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/digit-map-checkers-miniature.html) | digit-map-checkers-miniature.html | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [page_template.html](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/page_template.html) | page_template.html | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |

**`2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/node_modules/katex/contrib/auto-render/`: 1 file**

| File | What it shows | Lands on |
|---|---|---|
| [index.html](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/node_modules/katex/contrib/auto-render/index.html) | index.html | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |

**`2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/node_modules/katex/contrib/copy-tex/`: 1 file**

| File | What it shows | Lands on |
|---|---|---|
| [index.html](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/node_modules/katex/contrib/copy-tex/index.html) | index.html | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |

<details><summary>`2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/`: 64 files</summary>

| File | What it shows | Lands on |
|---|---|---|
| [00a-real-input-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00a-real-input-dark.png) | 00a-real-input-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [00a-real-input-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00a-real-input-light.png) | 00a-real-input-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [00b-real-maps-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00b-real-maps-dark.png) | 00b-real-maps-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [00b-real-maps-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00b-real-maps-light.png) | 00b-real-maps-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [00c-real-conv1-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00c-real-conv1-dark.png) | 00c-real-conv1-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [00c-real-conv1-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00c-real-conv1-light.png) | 00c-real-conv1-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [00d-real-film-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00d-real-film-dark.png) | 00d-real-film-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [00d-real-film-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00d-real-film-light.png) | 00d-real-film-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [00e-real-pool-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00e-real-pool-dark.png) | 00e-real-pool-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [00e-real-pool-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00e-real-pool-light.png) | 00e-real-pool-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [00f-real-logit-wrong-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00f-real-logit-wrong-dark.png) | 00f-real-logit-wrong-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [00f-real-logit-wrong-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00f-real-logit-wrong-light.png) | 00f-real-logit-wrong-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [00g-real-push-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00g-real-push-dark.png) | 00g-real-push-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [00g-real-push-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00g-real-push-light.png) | 00g-real-push-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [00h-real-unet-712-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00h-real-unet-712-dark.png) | 00h-real-unet-712-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [00h-real-unet-712-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00h-real-unet-712-light.png) | 00h-real-unet-712-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [00i-real-readout-network-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00i-real-readout-network-dark.png) | 00i-real-readout-network-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [00i-real-readout-network-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00i-real-readout-network-light.png) | 00i-real-readout-network-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [01-forward-mid-convolution-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/01-forward-mid-convolution-dark.png) | 01-forward-mid-convolution-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [01-forward-mid-convolution-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/01-forward-mid-convolution-light.png) | 01-forward-mid-convolution-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [02-forward-at-rest-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/02-forward-at-rest-dark.png) | 02-forward-at-rest-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [02-forward-at-rest-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/02-forward-at-rest-light.png) | 02-forward-at-rest-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [03-training-into-the-kernel-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/03-training-into-the-kernel-dark.png) | 03-training-into-the-kernel-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [03-training-into-the-kernel-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/03-training-into-the-kernel-light.png) | 03-training-into-the-kernel-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [04-training-stops-at-maps-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/04-training-stops-at-maps-dark.png) | 04-training-stops-at-maps-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [04-training-stops-at-maps-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/04-training-stops-at-maps-light.png) | 04-training-stops-at-maps-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [05-push-through-the-unet-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/05-push-through-the-unet-dark.png) | 05-push-through-the-unet-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [05-push-through-the-unet-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/05-push-through-the-unet-light.png) | 05-push-through-the-unet-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [06-push-into-the-pixels-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/06-push-into-the-pixels-dark.png) | 06-push-into-the-pixels-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [06-push-into-the-pixels-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/06-push-into-the-pixels-light.png) | 06-push-into-the-pixels-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [07-push-broken-run-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/07-push-broken-run-dark.png) | 07-push-broken-run-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [07-push-broken-run-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/07-push-broken-run-light.png) | 07-push-broken-run-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [08-noise-sweep-at-712-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/08-noise-sweep-at-712-dark.png) | 08-noise-sweep-at-712-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [08-noise-sweep-at-712-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/08-noise-sweep-at-712-light.png) | 08-noise-sweep-at-712-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [09-push-length-at-1-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/09-push-length-at-1-dark.png) | 09-push-length-at-1-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [09-push-length-at-1-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/09-push-length-at-1-light.png) | 09-push-length-at-1-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [10-identity-training-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/10-identity-training-dark.png) | 10-identity-training-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [10-identity-training-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/10-identity-training-light.png) | 10-identity-training-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [11-noise-real-scores-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/11-noise-real-scores-dark.png) | 11-noise-real-scores-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [11-noise-real-scores-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/11-noise-real-scores-light.png) | 11-noise-real-scores-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [12-convolution-hover-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/12-convolution-hover-dark.png) | 12-convolution-hover-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [12-convolution-hover-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/12-convolution-hover-light.png) | 12-convolution-hover-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [13-push-score-card-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/13-push-score-card-dark.png) | 13-push-score-card-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [13-push-score-card-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/13-push-score-card-light.png) | 13-push-score-card-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [14-two-digits-before-push-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/14-two-digits-before-push-dark.png) | 14-two-digits-before-push-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [14-two-digits-before-push-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/14-two-digits-before-push-light.png) | 14-two-digits-before-push-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [15-two-digits-push-acting-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/15-two-digits-push-acting-dark.png) | 15-two-digits-push-acting-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [15-two-digits-push-acting-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/15-two-digits-push-acting-light.png) | 15-two-digits-push-acting-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [16-two-digits-end-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/16-two-digits-end-dark.png) | 16-two-digits-end-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [16-two-digits-end-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/16-two-digits-end-light.png) | 16-two-digits-end-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [17-two-digits-lost-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/17-two-digits-lost-dark.png) | 17-two-digits-lost-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [17-two-digits-lost-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/17-two-digits-lost-light.png) | 17-two-digits-lost-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [p1-real-conv1-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/p1-real-conv1-dark.png) | p1-real-conv1-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [p1-real-conv1-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/p1-real-conv1-light.png) | p1-real-conv1-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [p2-real-logit-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/p2-real-logit-dark.png) | p2-real-logit-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [p2-real-logit-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/p2-real-logit-light.png) | p2-real-logit-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [p3-network-forward-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/p3-network-forward-dark.png) | p3-network-forward-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [p3-network-forward-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/p3-network-forward-light.png) | p3-network-forward-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [p4-training-end-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/p4-training-end-dark.png) | p4-training-end-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [p4-training-end-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/p4-training-end-light.png) | p4-training-end-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [p5-real-readout-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/p5-real-readout-dark.png) | p5-real-readout-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [p5-real-readout-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/p5-real-readout-light.png) | p5-real-readout-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [p6-two-digits-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/p6-two-digits-dark.png) | p6-two-digits-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [p6-two-digits-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/p6-two-digits-light.png) | p6-two-digits-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |

</details>

**`2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills-codex/`: 4 files**

| File | What it shows | Lands on |
|---|---|---|
| [1-without-the-push-the-digits-fuse.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills-codex/1-without-the-push-the-digits-fuse.png) | the plain run's guess of the clean canvas at steps 10, 20, 30 and 49, then its final canvas, judged one blob | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [2-the-gradient-pulls-the-maps-apart.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills-codex/2-the-gradient-pulls-the-maps-apart.png) | step 20, plain above and pushed below: the two maps with their centres marked, and the gradient g | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [3-after-the-push-two-digits.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills-codex/3-after-the-push-two-digits.png) | the plain and pushed final canvases side by side, judged one blob and two clear digits | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [4-maps-separate-only-while-the-push-acts.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills-codex/4-maps-separate-only-while-the-push-acts.png) | three curves over 50 sampler steps: map centres apart, overlap and logit, plain against pushed, with the push window shaded | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |

<details><summary>`2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/`: 28 files</summary>

| File | What it shows | Lands on |
|---|---|---|
| [desktop-light-01.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/desktop-light-01.png) | desktop-light-01.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [desktop-light-02.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/desktop-light-02.png) | desktop-light-02.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [desktop-light-03.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/desktop-light-03.png) | desktop-light-03.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [desktop-light-04.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/desktop-light-04.png) | desktop-light-04.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [desktop-light-05.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/desktop-light-05.png) | desktop-light-05.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [desktop-light-06.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/desktop-light-06.png) | desktop-light-06.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [desktop-light-07.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/desktop-light-07.png) | desktop-light-07.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [desktop-light-08.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/desktop-light-08.png) | desktop-light-08.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [desktop-light-09.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/desktop-light-09.png) | desktop-light-09.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [desktop-light-10.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/desktop-light-10.png) | desktop-light-10.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [desktop-light-11.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/desktop-light-11.png) | desktop-light-11.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [desktop-light-12.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/desktop-light-12.png) | desktop-light-12.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [desktop-light-13.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/desktop-light-13.png) | desktop-light-13.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [desktop-light-14.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/desktop-light-14.png) | desktop-light-14.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [phone-dark-01.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/phone-dark-01.png) | phone-dark-01.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [phone-dark-02.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/phone-dark-02.png) | phone-dark-02.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [phone-dark-03.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/phone-dark-03.png) | phone-dark-03.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [phone-dark-04.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/phone-dark-04.png) | phone-dark-04.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [phone-dark-05.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/phone-dark-05.png) | phone-dark-05.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [phone-dark-06.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/phone-dark-06.png) | phone-dark-06.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [phone-dark-07.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/phone-dark-07.png) | phone-dark-07.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [phone-dark-08.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/phone-dark-08.png) | phone-dark-08.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [phone-dark-09.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/phone-dark-09.png) | phone-dark-09.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [phone-dark-10.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/phone-dark-10.png) | phone-dark-10.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [phone-dark-11.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/phone-dark-11.png) | phone-dark-11.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [phone-dark-12.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/phone-dark-12.png) | phone-dark-12.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [phone-dark-13.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/phone-dark-13.png) | phone-dark-13.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |
| [phone-dark-14.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/phone-dark-14.png) | phone-dark-14.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md` (to write), frames of one interactive page; the page is linked, not each frame |

</details>

**`2-classifier-guidance/artifacts/scenes/digit-map-checkers/numbers/`: 1 file**

| File | What it shows | Lands on |
|---|---|---|
| [results-so-far.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/numbers/results-so-far.png) | Balanced accuracy on pairs holding digit 4: one combined checker 0.704 on all kinds and 0.511 on apart against wrong; the count checker 0.774 with the map reader and 0.775 with the raw-canvas reader; the identity checker not written yet | `context/methods/03-the-digit-map-checkers/` (to write), one page per rung, overview first |

**`2-classifier-guidance/artifacts/scenes/digit-map-checkers/numbers/env/`: 1 file**

| File | What it shows | Lands on |
|---|---|---|
| [canvas-kinds-sheet.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/numbers/env/canvas-kinds-sheet.png) | canvas-kinds-sheet.png | `context/methods/03-the-digit-map-checkers/` (to write), one page per rung, overview first |

**`2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/`: 2 files**

| File | What it shows | Lands on |
|---|---|---|
| [noisyclip-ladder.html](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/noisyclip-ladder.html) | noisyclip-ladder.html | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [style-sample-clip-contrastive.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/style-sample-clip-contrastive.png) | style-sample-clip-contrastive.png | none: a style sample for drawing, not a result or a method |

<details><summary>`2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/0.5-network-view/`: 12 files</summary>

| File | What it shows | Lands on |
|---|---|---|
| [nc1-one-model-three-uses.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/0.5-network-view/nc1-one-model-three-uses.png) | Three lanes separated by dotted lines: training, reading and pushing, each through the same text and image encoders. | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [nc10-how-the-push-joins-co3.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/0.5-network-view/nc10-how-the-push-joins-co3.png) | The frozen U-Net gives unconditional and conditional predictions, the push is subtracted from both, and the guided prediction feeds CO3's next step. | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [nc11-the-text-only-word-check.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/0.5-network-view/nc11-the-text-only-word-check.png) | Four sentences of one swap set pass the frozen text encoder; the same-meaning pair and the swapped pair are each summed, and their difference w is what a sum of word slots cannot make. Result bars give cat and dog's ratios beside a faded r… | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [nc12-what-survives-for-six-noun-pairs.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/0.5-network-view/nc12-what-survives-for-six-noun-pairs.png) | The ratio from 0 to 1 on x, one dot per noun pair, in four rows: binding and order, trained and random weights, with each row's mean marked. | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [nc2-the-whole-read-and-push-path.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/0.5-network-view/nc2-the-whole-read-and-push-path.png) | The read and push path: a latent through a fixed map and a tuned image encoder, scored against a right and a wrong sentence, with the gradient taken at the latent and subtracted from the base noise prediction. | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [nc3-four-latent-channels-to-rgb.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/0.5-network-view/nc3-four-latent-channels-to-rgb.png) | Four grey noise tiles, a three by four weight table with biases, then red, green and blue channels and their colour image. | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [nc4-clamping-kills-the-gradient.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/0.5-network-view/nc4-clamping-kills-the-gradient.png) | Left, clamp output against map output with slope one inside 0 to 255. Right, bars of the share of clamped pixels per colour. | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [nc5-what-is-tuned-and-what-is-frozen.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/0.5-network-view/nc5-what-is-tuned-and-what-is-frozen.png) | The CLIP frame holds a faded text encoder, a saturated image encoder and two faded projections, with the image encoder opened into its blocks at the right. | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [nc6-training-on-noisy-latents.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/0.5-network-view/nc6-training-on-noisy-latents.png) | Captions and noisy latents enter a faded text encoder and a saturated image encoder, then a similarity matrix with a pale blue diagonal, a contrastive loss label and an attract and repel panel. | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [nc7-trained-noise-levels-and-pushed-steps.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/0.5-network-view/nc7-trained-noise-levels-and-pushed-steps.png) | Noise scale and signal scale against sampler step 0 to 49, with steps 0 to 15 shaded orange and steps 20 to 29 shaded green. | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [nc8-right-minus-wrong.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/0.5-network-view/nc8-right-minus-wrong.png) | An image strip meets two sentence columns, and a circle sketch shows the image vector with the two sentence vectors and the dashed line between them. | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [nc9-the-gradient-back-to-the-latent.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/0.5-network-view/nc9-the-gradient-back-to-the-latent.png) | Ten forward stages from the latent to the score difference, an orange backward row beneath them, and two boxes naming where the released map blocks the gradient. | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |

</details>

**`2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/1-illustrations/`: 6 files**

| File | What it shows | Lands on |
|---|---|---|
| [ni1-a-cheap-colour-view-of-the-latent.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/1-illustrations/ni1-a-cheap-colour-view-of-the-latent.png) | Four grey latent channels pass a fixed linear map into a rough colour view and then the image encoder, with the VAE decoder faded and skipped. | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [ni2-one-picture-against-two-sentences.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/1-illustrations/ni2-one-picture-against-two-sentences.png) | One noisy picture meets a right and a wrong sentence at two product nodes, giving two score chips and their difference, on toy numbers. | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [ni3-training-on-half-drawn-pictures.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/1-illustrations/ni3-training-on-half-drawn-pictures.png) | Noisy pictures and captions enter a tuned image encoder and a frozen text encoder, then a similarity matrix with a pale blue diagonal and an attract and repel panel. | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [ni4-the-push-bends-the-noise-prediction.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/1-illustrations/ni4-the-push-bends-the-noise-prediction.png) | A vector picture: the base noise prediction, an orange push from its tip, and their sum, with the gradient taken at the latent tile. | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [ni5-two-places-the-signal-is-cut.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/1-illustrations/ni5-two-places-the-signal-is-cut.png) | A pipeline from the latent to the image encoder, with the orange return lane cut at the byte conversion and ending at a clamped colour. | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [ni6-trained-later-than-we-push.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/1-illustrations/ni6-trained-later-than-we-push.png) | A row of an armchair emerging from noise, with this loop's steps 0 to 15 bracketed in orange and NoisyCLIP's training steps 20 to 29 bracketed in green. | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |

**`2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/1-illustrations/drop/`: 7 files**

| File | What it shows | Lands on |
|---|---|---|
| [ni1-a-cheap-colour-view-of-the-latent.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/1-illustrations/drop/ni1-a-cheap-colour-view-of-the-latent.png) | ni1-a-cheap-colour-view-of-the-latent.png | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [ni2-one-picture-against-two-sentences.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/1-illustrations/drop/ni2-one-picture-against-two-sentences.png) | ni2-one-picture-against-two-sentences.png | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [ni3-training-on-half-drawn-pictures.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/1-illustrations/drop/ni3-training-on-half-drawn-pictures.png) | ni3-training-on-half-drawn-pictures.png | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [ni4-the-push-bends-the-noise-prediction.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/1-illustrations/drop/ni4-the-push-bends-the-noise-prediction.png) | ni4-the-push-bends-the-noise-prediction.png | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [ni5-two-places-the-signal-is-cut-before-rev1.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/1-illustrations/drop/ni5-two-places-the-signal-is-cut-before-rev1.png) | ni5-two-places-the-signal-is-cut-before-rev1.png | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [ni5-two-places-the-signal-is-cut.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/1-illustrations/drop/ni5-two-places-the-signal-is-cut.png) | ni5-two-places-the-signal-is-cut.png | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [ni6-trained-later-than-we-push.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/1-illustrations/drop/ni6-trained-later-than-we-push.png) | ni6-trained-later-than-we-push.png | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |

**`2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/2-miniature/`: 9 files**

| File | What it shows | Lands on |
|---|---|---|
| [mm0-the-whole-miniature.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/2-miniature/mm0-the-whole-miniature.png) | Eight latent numbers feed six colours, a two-number image layer, a unit vector, two scores and their difference, with an orange lane back to the latent. | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [mm1-from-latent-to-colours.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/2-miniature/mm1-from-latent-to-colours.png) | A table computing red, green and blue for pixels A and B. Pixel B red is 305 before the clamp and 255 after. | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [mm2-forward-to-the-two-scores.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/2-miniature/mm2-forward-to-the-two-scores.png) | Four cards: scale the colours, apply the image layer, make the vector unit length, take two dot products, then the difference. | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [mm3-the-gradient-back-to-the-latent.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/2-miniature/mm3-the-gradient-back-to-the-latent.png) | Four orange cards apply the chain rule from the score difference back through the layer, the clamp mask and the map, ending in a table of eight gradient values. | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [mm4-the-push-joins-the-noise-prediction.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/2-miniature/mm4-the-push-joins-the-noise-prediction.png) | The gradient times two times 0.8 is subtracted from the base noise prediction. A DDIM step and two bars compare the score difference with and without the push. | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [mm5-four-ways-the-gradient-dies.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/2-miniature/mm5-four-ways-the-gradient-dies.png) | Four panels: byte conversion has no gradient, identical sentence vectors give zero difference, a pixel with all colours clamped has a zero gradient row, and a soft clamp passes a small slope. | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [mm6-one-training-step.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/2-miniature/mm6-one-training-step.png) | A two by two score matrix before and after one update of the image layer, with the loss falling from 0.9264 to 0.7929 and the text vectors unchanged. | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [mm7-a-sum-of-word-slots-cancels.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/2-miniature/mm7-a-sum-of-word-slots-cancels.png) | Word-slot vectors build four sentences as sums; the same-meaning pair and the swapped pair both sum to (4, 4, 5), so w is (0, 0, 0). | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [mm8-one-interaction-term-survives.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/2-miniature/mm8-one-interaction-term-survives.png) | The same sums with an interaction term of 0.5 times side times the noun difference; the pairs now sum to (5, 3, 5) and (3, 5, 5), w is (2, −2, 0), and the real ratios sit beside the toy's ratio. | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |

**`2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/overview/`: 3 files**

| File | What it shows | Lands on |
|---|---|---|
| [ov0-the-sentence-score-push.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/overview/ov0-the-sentence-score-push.png) | One half-drawn picture and two sentences followed through NoisyCLIP's text and vision encoders, two scores, the third score and the push, with the vision encoder opened in an inset. | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [ov1-latent-clip-push.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/overview/ov1-latent-clip-push.png) | The same picture as a half-drawn latent, divided by 0.13025 and pooled to 4 by 64 by 64 with no RGB map and no clamp, read by Latent CLIP ViT-B/4-plus and its own text encoder, with the same push. | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [w4-input-canvas.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/overview/w4-input-canvas.png) | w4-input-canvas.png | `context/methods/04-the-sentence-score-push/` (to write), one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |

**`2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/reference/`: 3 files**

| File | What it shows | Lands on |
|---|---|---|
| [clip-frame-beside-own-model.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/reference/clip-frame-beside-own-model.png) | clip-frame-beside-own-model.png | none: style references for drawing |
| [dual-contrastive-attract-repel.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/reference/dual-contrastive-attract-repel.png) | dual-contrastive-attract-repel.png | none: style references for drawing |
| [multi-view-contrastive-loss.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/reference/multi-view-contrastive-loss.png) | multi-view-contrastive-loss.png | none: style references for drawing |

**`2-classifier-guidance/artifacts/scenes/product-of-experts-interaction-term/figures/`: 5 files**

| File | What it shows | Lands on |
|---|---|---|
| [level0-one-sampler-step-of-a-product-of-experts.png](../../../../2-classifier-guidance/artifacts/scenes/product-of-experts-interaction-term/figures/level0-one-sampler-step-of-a-product-of-experts.png) | One sampler step of a product of two experts, with the dropped interaction term in orange | `context/methods/01-the-product-of-experts/` (to write), one page per rung, overview first |
| [level05-the-interaction-term-against-the-sampler-step.png](../../../../2-classifier-guidance/artifacts/scenes/product-of-experts-interaction-term/figures/level05-the-interaction-term-against-the-sampler-step.png) | Size of the interaction term and how settled the joint's layout is, against the sampler step | `context/methods/01-the-product-of-experts/` (to write), one page per rung, overview first |
| [level05-the-product-score-the-joint-score-and-the-term.png](../../../../2-classifier-guidance/artifacts/scenes/product-of-experts-interaction-term/figures/level05-the-product-score-the-joint-score-and-the-term.png) | Three arrow fields on the plane of 4 in the left cell against 7 in the right cell: the product's score, the joint's score, and the term between them, at two noise levels | `context/methods/01-the-product-of-experts/` (to write), one page per rung, overview first |
| [level2-what-each-composition-draws.png](../../../../2-classifier-guidance/artifacts/scenes/product-of-experts-interaction-term/figures/level2-what-each-composition-draws.png) | Stacked bars of what each composition draws out of 4000 | `context/methods/01-the-product-of-experts/` (to write), one page per rung, overview first |
| [level2-where-the-term-has-to-act.png](../../../../2-classifier-guidance/artifacts/scenes/product-of-experts-interaction-term/figures/level2-where-the-term-has-to-act.png) | Two apart out of 4000 against the step at which the term starts or stops | `context/methods/01-the-product-of-experts/` (to write), one page per rung, overview first |

**`2-classifier-guidance/artifacts/scenes/product-of-experts-interaction-term/figures/real-run/`: 5 files**

| File | What it shows | Lands on |
|---|---|---|
| [e0e-1-where-each-expert-wants-its-digit.png](../../../../2-classifier-guidance/artifacts/scenes/product-of-experts-interaction-term/figures/real-run/e0e-1-where-each-expert-wants-its-digit.png) | e0e-1-where-each-expert-wants-its-digit.png | `context/methods/01-the-product-of-experts/` (to write), one page per rung, overview first |
| [e0e-2-the-layout-flips-or-holds.png](../../../../2-classifier-guidance/artifacts/scenes/product-of-experts-interaction-term/figures/real-run/e0e-2-the-layout-flips-or-holds.png) | e0e-2-the-layout-flips-or-holds.png | `context/methods/01-the-product-of-experts/` (to write), one page per rung, overview first |
| [e0e-3-hide-the-other-half-and-join.png](../../../../2-classifier-guidance/artifacts/scenes/product-of-experts-interaction-term/figures/real-run/e0e-3-hide-the-other-half-and-join.png) | e0e-3-hide-the-other-half-and-join.png | `context/methods/01-the-product-of-experts/` (to write), one page per rung, overview first |
| [one-sampler-step.png](../../../../2-classifier-guidance/artifacts/scenes/product-of-experts-interaction-term/figures/real-run/one-sampler-step.png) | one real sampler step | `context/methods/01-the-product-of-experts/` (to write), one page per rung, overview first |
| [qualitative-ways-of-composing.png](../../../../2-classifier-guidance/artifacts/scenes/product-of-experts-interaction-term/figures/real-run/qualitative-ways-of-composing.png) | the ways of composing, three pairs | `context/methods/01-the-product-of-experts/` (to write), one page per rung, overview first |

**`2-classifier-guidance/artifacts/scenes/word-map-checker/0.5-network-view/`: 7 files**

| File | What it shows | Lands on |
|---|---|---|
| [wm0-where-the-checker-sits.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/0.5-network-view/wm0-where-the-checker-sits.png) | One sampler step: the latent through the frozen U-Net to the product's noise prediction, minus a capped push from a checker that reads only the word maps, then a DDIM step. | `context/methods/06-the-word-map-checker/` (to write), one page per rung, overview first |
| [wm1-three-inputs-and-where-the-maps-come-from.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/0.5-network-view/wm1-three-inputs-and-where-the-maps-come-from.png) | A caption, a timestep and the noisy latent enter a closed U-Net whose 64 by 64 and 32 by 32 cross-attention layers each give every word a stack of maps. | `context/methods/06-the-word-map-checker/` (to write), one page per rung, overview first |
| [wm2-from-one-layer-to-a-words-stack.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/0.5-network-view/wm2-from-one-layer-to-a-words-stack.png) | Each patch shares attention across tokens; the cat column is a map; it is resized to 16 by 16, renormalised and kept with every other layer and head. | `context/methods/06-the-word-map-checker/` (to write), one page per rung, overview first |
| [wm3-the-head-reads-the-two-stacks.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/0.5-network-view/wm3-the-head-reads-the-two-stacks.png) | Each word's maps pass a transformer and a learned query into one 16 by 16 map; the pair is flattened, projected, passed through a one-token transformer, joined with a step vector and read by an MLP into five logits. | `context/methods/06-the-word-map-checker/` (to write), one page per rung, overview first |
| [wm4-training-on-the-right-and-wrong-word.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/0.5-network-view/wm4-training-on-the-right-and-wrong-word.png) | A training picture is inverted under the right caption and a wrong one, and both sets of maps carry the true label. | `context/methods/06-the-word-map-checker/` (to write), one page per rung, overview first |
| [wm5-two-ways-to-apply-the-push.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/0.5-network-view/wm5-two-ways-to-apply-the-push.png) | Learn-to-Steer updates the latent up to 16 times per step; our sampler subtracts one capped push from the product's prediction. | `context/methods/06-the-word-map-checker/` (to write), one page per rung, overview first |
| [wm6-one-joint-prompt-against-two-prompts.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/0.5-network-view/wm6-one-joint-prompt-against-two-prompts.png) | Under one joint prompt the maps separate because they share a softmax; under two prompts each has its own and both fill one body. | `context/methods/06-the-word-map-checker/` (to write), one page per rung, overview first |

**`2-classifier-guidance/artifacts/scenes/word-map-checker/1-illustrations/`: 6 files**

| File | What it shows | Lands on |
|---|---|---|
| [wi1-where-each-word-landed.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/1-illustrations/wi1-where-each-word-landed.png) | A half-drawn fused cat-dog with the two word heat maps lighting the same place. | `context/methods/06-the-word-map-checker/` (to write), one page per rung, overview first |
| [wi2-a-small-judge-reads-only-the-two-maps.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/1-illustrations/wi2-a-small-judge-reads-only-the-two-maps.png) | Two maps go to a small judge who never sees the picture. | `context/methods/06-the-word-map-checker/` (to write), one page per rung, overview first |
| [wi3-it-learned-the-word-instead-of-the-picture.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/1-illustrations/wi3-it-learned-the-word-instead-of-the-picture.png) | A dog left of a cat captioned above: the naive judge says above, the judge trained on both captions says left of. | `context/methods/06-the-word-map-checker/` (to write), one page per rung, overview first |
| [wi4-the-judges-complaint-pushes-the-noise.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/1-illustrations/wi4-the-judges-complaint-pushes-the-noise.png) | The judge's error travels back to the noisy picture as an orange push. | `context/methods/06-the-word-map-checker/` (to write), one page per rung, overview first |
| [wi5-one-capped-nudge-against-sixteen-big-steps.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/1-illustrations/wi5-one-capped-nudge-against-sixteen-big-steps.png) | One small capped nudge against sixteen big steps. | `context/methods/06-the-word-map-checker/` (to write), one page per rung, overview first |
| [wi6-the-map-can-move-without-a-second-animal.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/1-illustrations/wi6-the-map-can-move-without-a-second-animal.png) | The maps separate while the picture still holds one animal. | `context/methods/06-the-word-map-checker/` (to write), one page per rung, overview first |

**`2-classifier-guidance/artifacts/scenes/word-map-checker/1-illustrations/drop/`: 9 files**

| File | What it shows | Lands on |
|---|---|---|
| [wi1-where-each-word-landed-before-rev1.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/1-illustrations/drop/wi1-where-each-word-landed-before-rev1.png) | wi1-where-each-word-landed-before-rev1.png | `context/methods/06-the-word-map-checker/` (to write), one page per rung, overview first |
| [wi1-where-each-word-landed.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/1-illustrations/drop/wi1-where-each-word-landed.png) | wi1-where-each-word-landed.png | `context/methods/06-the-word-map-checker/` (to write), one page per rung, overview first |
| [wi2-a-small-judge-reads-only-the-two-maps.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/1-illustrations/drop/wi2-a-small-judge-reads-only-the-two-maps.png) | wi2-a-small-judge-reads-only-the-two-maps.png | `context/methods/06-the-word-map-checker/` (to write), one page per rung, overview first |
| [wi3-it-learned-the-word-instead-of-the-picture-before-rev1.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/1-illustrations/drop/wi3-it-learned-the-word-instead-of-the-picture-before-rev1.png) | wi3-it-learned-the-word-instead-of-the-picture-before-rev1.png | `context/methods/06-the-word-map-checker/` (to write), one page per rung, overview first |
| [wi3-it-learned-the-word-instead-of-the-picture.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/1-illustrations/drop/wi3-it-learned-the-word-instead-of-the-picture.png) | wi3-it-learned-the-word-instead-of-the-picture.png | `context/methods/06-the-word-map-checker/` (to write), one page per rung, overview first |
| [wi4-the-judges-complaint-pushes-the-noise-before-rev1.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/1-illustrations/drop/wi4-the-judges-complaint-pushes-the-noise-before-rev1.png) | wi4-the-judges-complaint-pushes-the-noise-before-rev1.png | `context/methods/06-the-word-map-checker/` (to write), one page per rung, overview first |
| [wi4-the-judges-complaint-pushes-the-noise.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/1-illustrations/drop/wi4-the-judges-complaint-pushes-the-noise.png) | wi4-the-judges-complaint-pushes-the-noise.png | `context/methods/06-the-word-map-checker/` (to write), one page per rung, overview first |
| [wi5-one-capped-nudge-against-sixteen-big-steps.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/1-illustrations/drop/wi5-one-capped-nudge-against-sixteen-big-steps.png) | wi5-one-capped-nudge-against-sixteen-big-steps.png | `context/methods/06-the-word-map-checker/` (to write), one page per rung, overview first |
| [wi6-the-map-can-move-without-a-second-animal.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/1-illustrations/drop/wi6-the-map-can-move-without-a-second-animal.png) | wi6-the-map-can-move-without-a-second-animal.png | `context/methods/06-the-word-map-checker/` (to write), one page per rung, overview first |

**`2-classifier-guidance/artifacts/scenes/word-map-checker/2-miniature/`: 6 files**

| File | What it shows | Lands on |
|---|---|---|
| [mm0-the-whole-miniature.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/2-miniature/mm0-the-whole-miniature.png) | A fused body in a 2 by 2 latent gives identical pooled maps, zero centre-of-mass gaps, 0.2 on all five classes and a loss of 1.6094, with the gradient running back to the latent. | `context/methods/06-the-word-map-checker/` (to write), one page per rung, overview first |
| [mm1-from-latent-to-pooled-map.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/2-miniature/mm1-from-latent-to-pooled-map.png) | The cat channel through two layers' softmaxes, each map divided by its sum, then mixed three quarters and one quarter into one pooled map. | `context/methods/06-the-word-map-checker/` (to write), one page per rung, overview first |
| [mm2-the-head-reads-a-relation.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/2-miniature/mm2-the-head-reads-a-relation.png) | Pooled cat and dog maps feed two hidden units; these give five logits and probabilities. Identical maps give 0.2 on every class. | `context/methods/06-the-word-map-checker/` (to write), one page per rung, overview first |
| [mm3-the-gradient-back-to-the-latent.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/2-miniature/mm3-the-gradient-back-to-the-latent.png) | Three chain-rule steps from the loss to the latent, then the map's slope at each patch beside the gradient on the cat and dog channels. | `context/methods/06-the-word-map-checker/` (to write), one page per rung, overview first |
| [mm4-sixteen-updates-against-one-capped-push.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/2-miniature/mm4-sixteen-updates-against-one-capped-push.png) | The chance of left of across sixteen updates of step size 5, reaching 0.9936, and one push capped at 0.4 of the prediction's norm, reaching 0.2367. | `context/methods/06-the-word-map-checker/` (to write), one page per rung, overview first |
| [mm5-three-ways-it-breaks.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/2-miniature/mm5-three-ways-it-breaks.png) | Three panels: a head trained only on true captions follows the prompt's word; a correlation loss has zero gradient on identical maps; identical maps give 0.2 on all five classes and no class means fused. | `context/methods/06-the-word-map-checker/` (to write), one page per rung, overview first |

## Older results and scenes the census also found

These sit in the results and scenes folders but were made before the loops started, so this drain does not place them. They are listed so the count is complete; whether each has a finding was not checked here.

27 of them fail the subject guard (`scripts/shared/disallowed_subjects.py`, read by file name) or sit on the held-pictures list, and stay off every page. No picture a loop made fails either check.

| Folder | Unplaced files | Note |
|---|---|---|
| `1-lora-correction/artifacts/results/against-other-methods/samplers-and-correctors/` | 1 | made before the loops and moved into place on 2026-11-22; not part of this drain |
| `1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/` | 43 | made before the loops and moved into place on 2026-11-22; not part of this drain |
| `1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-transfer/` | 19 | made before the loops and moved into place on 2026-11-22; not part of this drain |
| `1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/` | 12 | made before the loops and moved into place on 2026-11-22; not part of this drain |
| `1-lora-correction/artifacts/results/measuring-composition/the-scorers/` | 25 | made before the loops and moved into place on 2026-11-22; not part of this drain |
| `1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/how-far-one-adapter-reaches/` | 3 | made before the loops and moved into place on 2026-11-22; not part of this drain |
| `1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/` | 156 | made before the loops and moved into place on 2026-11-22; not part of this drain |
| `1-lora-correction/artifacts/results/what-the-correction-is/what-it-looks-like/` | 9 | made before the loops and moved into place on 2026-11-22; not part of this drain |
| `1-lora-correction/artifacts/results/what-the-correction-is/when-it-acts/` | 36 | made before the loops and moved into place on 2026-11-22; not part of this drain |
| `2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/01-the-first-quick-test/` | 1 | made before the loops; not part of this drain |
| `2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/02-can-a-checker-read-a-clean-pair/` | 10 | made before the loops; not part of this drain |
| `2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/03-does-an-early-push-separate-them/` | 13 | made before the loops; not part of this drain |
| `2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/04-can-a-second-push-name-each-animal/` | 10 | made before the loops; not part of this drain |
| `2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/05-does-it-carry-to-other-pairs/` | 7 | made before the loops; not part of this drain |
| `2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/06-how-close-it-comes-to-the-joint-prompt/` | 15 | made before the loops; not part of this drain |
| `2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/pools-of-training-pictures/` | 2 | made before the loops; not part of this drain |
| `2-classifier-guidance/artifacts/results/does-classifier-guidance-draw-a-dog/dog-01-from-an-animal/` | 1 | made before the loops; not part of this drain |
| `2-classifier-guidance/artifacts/results/does-classifier-guidance-work-on-the-released-weights/` | 1 | made before the loops; not part of this drain |
| `2-classifier-guidance/artifacts/results/does-classifier-guidance-work-on-the-released-weights/guide/` | 1 | made before the loops; not part of this drain |
| `2-classifier-guidance/artifacts/results/does-classifier-guidance-work-on-the-released-weights/identity/` | 1 | made before the loops; not part of this drain |
| `2-classifier-guidance/artifacts/results/what-each-line-of-classifier-guidance-computes/pictures/` | 2 | made before the loops; not part of this drain |
| `2-classifier-guidance/artifacts/scenes/classifier-guidance-on-poe/` | 1 | made before the loops; not part of this drain |
| `2-classifier-guidance/artifacts/scenes/mnist-guidance-explained/1-the-sampler-step/` | 3 | made before the loops; not part of this drain |
| `2-classifier-guidance/artifacts/scenes/mnist-guidance-explained/2-the-networks/` | 9 | made before the loops; not part of this drain |
| `2-classifier-guidance/artifacts/scenes/mnist-guidance-explained/3-the-gradient/` | 3 | made before the loops; not part of this drain |
| `2-classifier-guidance/artifacts/scenes/mnist-guidance-explained/4-the-derivation/` | 7 | made before the loops; not part of this drain |
| `2-classifier-guidance/artifacts/scenes/mnist-guidance-explained/5-qualitative-grids/` | 3 | made before the loops; not part of this drain |
| `2-classifier-guidance/artifacts/scenes/mnist-guidance-explained/toys/` | 12 | made before the loops; not part of this drain |

## Folders the drain does not look in

| Folder | Why the sweep misses it | Who covers it now |
|---|---|---|
| `artifacts/notes/progress/` and every loop folder inside it | held in the drain ledger | this census, rerun by step Y5 |
| `2-classifier-guidance/artifacts/` | the sweep reads the root `artifacts/` only | this census, for `scenes/` and `results/` |
| `1-lora-correction/artifacts/` | the sweep reads the root `artifacts/` only | this census, for `results/` |
| `3-understanding-compositionality/artifacts/`, `4-inclusive-captions/artifacts/`, `5-text-embedding-third-expert/artifacts/` | the sweep reads the root `artifacts/` only | nobody: no loop writes there |
| `/datasets/mmolefe/poe_repair_min/loop-5/` to `loop-10/` | on the cluster, outside the repository | each loop's review cites what it needs from there |

