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
| Loop 4 | 8 | 0 | 0 |
| Loop 5 | 7 | 0 | 1 |
| Loop 6 | 13 | 1 | 1 |
| Loop 7 | 8 | 0 | 0 |
| Loop 8 | 5 | 1 | 0 |
| Loop 9 | 2 | 0 | 0 |
| Loop 10 | 8 | 0 | 0 |
| Loop 12 | 7 | 0 | 0 |
| scenes in `2-classifier-guidance/artifacts/scenes/` | 142 | 0 | 0 |
| older results and scenes, made before the loops | 405 | not counted | not counted |

## Results with a verdict, for step Y2

Each row is a review file whose verdict is recorded. The verdict column quotes the review's opening sentences; Y2 copies the full verdict, numbers, bar and run id from the review itself.

| File | Loop | What the review records | Lands on |
|---|---|---|---|
| [review.md](../../../../artifacts/notes/progress/method-search/05-two-experts-by-colour/review.md) `05-two-experts-by-colour` | Loop 6 | This file holds what the runs in the M10 design produced. The renders are under way, and no model has trained. |  |
| [review.md](../../../../artifacts/notes/progress/abstract-checkers/10-reading-identity/review.md) `10-reading-identity` | Loop 8 | Status: support. A checker trained on digits 0 to 4 tells two right digits from one right and one wrong digit at 0.861 balanced accuracy on digits 5 to 9 | `2-classifier-guidance/report/abstract-checkers-on-digits/can-a-checker-read-which-digits-are-present.md` (to write) |

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
| [path-to-a-filled-scoreboard.png](../../../../artifacts/notes/progress/targets/path-to-a-filled-scoreboard.png) | path-to-a-filled-scoreboard.png | none: a picture of the route's own plan, which stays beside the plan it explains |
| [rivals-by-dataset.png](../../../../artifacts/notes/progress/targets/rivals-by-dataset.png) | rivals-by-dataset.png | `1-lora-correction/report/against-other-methods/do-the-rivals-reproduce-their-printed-scores.md`, which rival reports on which benchmark |
| [what-we-are-doing-and-the-three-detours.png](../../../../artifacts/notes/progress/targets/what-we-are-doing-and-the-three-detours.png) | what-we-are-doing-and-the-three-detours.png | none: a picture of the route's own plan, which stays beside the plan it explains |
| [where-clevr-sits-before-the-go-or-no-go.png](../../../../artifacts/notes/progress/targets/where-clevr-sits-before-the-go-or-no-go.png) | where-clevr-sits-before-the-go-or-no-go.png | none: a picture of the route's own plan, which stays beside the plan it explains |
| [where-clevr-stage-2-sits-before-the-go-or-no-go.png](../../../../artifacts/notes/progress/targets/where-clevr-stage-2-sits-before-the-go-or-no-go.png) | where-clevr-stage-2-sits-before-the-go-or-no-go.png | none: a picture of the route's own plan, which stays beside the plan it explains |

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

**`artifacts/notes/progress/method-search/`: 3 files**

| File | What it shows | Lands on |
|---|---|---|
| [how-loops-6-7-8-connect.png](../../../../artifacts/notes/progress/method-search/how-loops-6-7-8-connect.png) | how-loops-6-7-8-connect.png | none: a picture of the route's own plan, which stays beside the plan it explains |
| [loop-6-route.png](../../../../artifacts/notes/progress/method-search/loop-6-route.png) | loop-6-route.png | none: a picture of the route's own plan, which stays beside the plan it explains |
| [loop-6-scoreboard.png](../../../../artifacts/notes/progress/method-search/loop-6-scoreboard.png) | loop-6-scoreboard.png | none: a picture of the route's own plan, which stays beside the plan it explains |

**`artifacts/notes/progress/method-search/01c-clevr-paste-first/`: 9 files**

| File | What it shows | Lands on |
|---|---|---|
| [experts-alone-first-16-samples.png](../../../../artifacts/notes/progress/method-search/01c-clevr-paste-first/experts-alone-first-16-samples.png) | Four blocks of 16 samples: one object at upper left, one at upper right, one larger object lower centre, and empty floors | waits for a verdict: the step is still running (M7) |
| [full-p1-first-16-samples.png](../../../../artifacts/notes/progress/method-search/01c-clevr-paste-first/full-p1-first-16-samples.png) | Sixteen samples from the position 1 expert, each a single CLEVR object at the upper left on an empty floor | waits for a verdict: the step is still running (M7) |
| [overfit-batch-256-real-and-sampled.png](../../../../artifacts/notes/progress/method-search/01c-clevr-paste-first/overfit-batch-256-real-and-sampled.png) | Top two rows: the 16 real pictures. Bottom two rows: 16 samples showing the same scene with one clear object at the same position | waits for a verdict: the step is still running (M7) |
| [overfit-scaled-ramp-real-and-sampled.png](../../../../artifacts/notes/progress/method-search/01c-clevr-paste-first/overfit-scaled-ramp-real-and-sampled.png) | Top two rows: the 16 real pictures. Bottom two rows: 16 samples with the right floor and lighting and faint coloured blotches in place of objects | waits for a verdict: the step is still running (M7) |
| [overfit-unscaled-ramp-real-and-sampled.png](../../../../artifacts/notes/progress/method-search/01c-clevr-paste-first/overfit-unscaled-ramp-real-and-sampled.png) | Top two rows: the 16 real pictures of one object. Bottom two rows: 16 samples from the model, which are colour noise | waits for a verdict: the step is still running (M7) |
| [paste-three-experts-first-32.png](../../../../artifacts/notes/progress/method-search/01c-clevr-paste-first/paste-three-experts-first-32.png) | Thirty-two pasted images, each with three objects at the three positions, several on a faintly lighter square around the object | waits for a verdict: the step is still running (M7) |
| [paste-two-experts-first-32.png](../../../../artifacts/notes/progress/method-search/01c-clevr-paste-first/paste-two-experts-first-32.png) | paste-two-experts-first-32.png | waits for a verdict: the step is still running (M7) |
| [smoke-objects-at-their-positions.png](../../../../artifacts/notes/progress/method-search/01c-clevr-paste-first/smoke-objects-at-their-positions.png) | Twelve CLEVR renders at 64 pixels: four with one object at position 1, four empty scenes, and four with objects at all three positions, each asked position outlined by its 18-pixel paste window | waits for a verdict: the step is still running (M7) |
| [window-control-written-rule.png](../../../../artifacts/notes/progress/method-search/01c-clevr-paste-first/window-control-written-rule.png) | How often each pixel changes, for the three positions, with the 18 by 18 window drawn: position 3's object runs past the top and bottom of its window | waits for a verdict: the step is still running (M7) |

**`artifacts/notes/progress/method-search/05-two-experts-by-colour/`: 1 file**

| File | What it shows | Lands on |
|---|---|---|
| [smoke-renders-by-colour-and-count.png](../../../../artifacts/notes/progress/method-search/05-two-experts-by-colour/smoke-renders-by-colour-and-count.png) | Five rows of 8 CLEVR renders: one red object, three red, three blue, two red with one blue, and three red with three blue, each labelled with its counts | none: a picture of the route's own plan, which stays beside the plan it explains |

### Loop 7

**`artifacts/notes/progress/abstract-checkers/01-reading/`: 8 files**

| File | What it shows | Lands on |
|---|---|---|
| [adding-logs-error-by-noise.png](../../../../artifacts/notes/progress/abstract-checkers/01-reading/adding-logs-error-by-noise.png) | Error of the summed logs against noise level for three pairs of labels on a small example. Labels on separate pixels have zero error. Independent labels on the same pixels peak near 0.14 nats. Labels where one implies the other reach 0.69… | `context/methods/03-the-digit-map-checkers/01-overview.md`, background from the reading |
| [checker-and-critic-compared.png](../../../../artifacts/notes/progress/abstract-checkers/01-reading/checker-and-critic-compared.png) | checker-and-critic-compared.png | `context/methods/03-the-digit-map-checkers/01-overview.md`, background from the reading |
| [glide-clip-score-against-people.png](../../../../artifacts/notes/progress/abstract-checkers/01-reading/glide-clip-score-against-people.png) | GLIDE's 16 random samples for CLIP guidance and classifier-free guidance on one prompt, beside its CLIP score plot and human Elo ratings. | `context/methods/04-the-sentence-score-push/01-overview.md`, a figure reproduced from the GLIDE paper |
| [one-concept-as-one-score.png](../../../../artifacts/notes/progress/abstract-checkers/01-reading/one-concept-as-one-score.png) | one-concept-as-one-score.png | `context/methods/03-the-digit-map-checkers/01-overview.md`, background from the reading |
| [style-annotated-explainer-sentence-score.png](../../../../artifacts/notes/progress/abstract-checkers/01-reading/style-annotated-explainer-sentence-score.png) | style-annotated-explainer-sentence-score.png | none: a style sample for drawing, not a result or a method |
| [what-score-addition-means.png](../../../../artifacts/notes/progress/abstract-checkers/01-reading/what-score-addition-means.png) | what-score-addition-means.png | `context/methods/03-the-digit-map-checkers/01-overview.md`, background from the reading |
| [what-the-closer-papers-read.png](../../../../artifacts/notes/progress/abstract-checkers/01-reading/what-the-closer-papers-read.png) | The seven closer papers grouped by whether they read a noisy input or clean images, and by how each reaches a generator. None trains a checker on word maps from a noisy latent, tests it on held-out concepts and steers a sampler with it. | none: a picture of the route's own plan, which stays beside the plan it explains |
| [what-the-papers-cover.png](../../../../artifacts/notes/progress/abstract-checkers/01-reading/what-the-papers-cover.png) | Word attention maps feed five proposed checkers. Only the spatial relation checker has a trained precedent, Learn-to-Steer on FLUX.1-dev. Counting, binding, position and size, and prompt match are gaps. Their gradients are summed, capped a… | none: a picture of the route's own plan, which stays beside the plan it explains |

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

**`artifacts/notes/progress/abstract-checkers/10-reading-identity/`: 2 files**

| File | What it shows | Lands on |
|---|---|---|
| [identity-grid.png](../../../../artifacts/notes/progress/abstract-checkers/10-reading-identity/identity-grid.png) | The 2 by 2 grid. V2's identity checker at 0.555 leads into the early window, where two maps score 0.563 and the ten-label signature 0.612, both red. The later window holds two maps at 0.702, red, and the ten-label signature at 0.888, green… | `2-classifier-guidance/report/abstract-checkers-on-digits/can-a-checker-read-which-digits-are-present.md` (to write), lands once step V6 is done on the board |
| [robustness-ladder.png](../../../../artifacts/notes/progress/abstract-checkers/10-reading-identity/robustness-ladder.png) | robustness-ladder.png | `2-classifier-guidance/report/abstract-checkers-on-digits/can-a-checker-read-which-digits-are-present.md` (to write), lands once step V6 is done on the board |

### Loop 9

**`artifacts/notes/progress/noise-text-critic/`: 2 files**

| File | What it shows | Lands on |
|---|---|---|
| [ideas-and-where-they-land.png](../../../../artifacts/notes/progress/noise-text-critic/ideas-and-where-they-land.png) | Six owner ideas, from using Latent CLIP directly to training our own, each with an arrow to the step that tests it: W2 and W3 for the zero-shot push, W2b for linear layers, W4 for training our own, and swapping SDXL's text encoder marked n… | none: a picture of the route's own plan, which stays beside the plan it explains |
| [where-the-pieces-sit.png](../../../../artifacts/notes/progress/noise-text-critic/where-the-pieces-sit.png) | where-the-pieces-sit.png | none: a picture of the route's own plan, which stays beside the plan it explains |

### Loop 10

**`artifacts/notes/progress/score-space/`: 3 files**

| File | What it shows | Lands on |
|---|---|---|
| [intended-chain-and-recommendations.png](../../../../artifacts/notes/progress/score-space/intended-chain-and-recommendations.png) | intended-chain-and-recommendations.png | `context/methods/07-the-score-path/01-overview.md` |
| [manager-tick.png](../../../../artifacts/notes/progress/score-space/manager-tick.png) | manager-tick.png **Held: fails the subject guard or is on the held-pictures list, so it stays off every page.** | `context/methods/07-the-score-path/01-overview.md` |
| [summary-where-loop-10-fits.png](../../../../artifacts/notes/progress/score-space/summary-where-loop-10-fits.png) | Loop 9's sentence scores and Loop 8's two-map checker feed into Loop 10's finding that all ten signed maps read which digits are present at 0.655. Dashed, untested arrows lead to Paper 1 (feed the checker all ten maps) and to Loop 6 and Pa… | `context/methods/07-the-score-path/01-overview.md` |

**`artifacts/notes/progress/score-space/00-ladder/`: 1 file**

| File | What it shows | Lands on |
|---|---|---|
| [style-sample-score-path.png](../../../../artifacts/notes/progress/score-space/00-ladder/style-sample-score-path.png) | style-sample-score-path.png | none: a style sample for drawing, not a result or a method |

**`artifacts/notes/progress/score-space/00-ladder/overview/`: 4 files**

| File | What it shows | Lands on |
|---|---|---|
| [canvas-3-7-native.png](../../../../artifacts/notes/progress/score-space/00-ladder/overview/canvas-3-7-native.png) | canvas-3-7-native.png | `context/methods/07-the-score-path/01-overview.md` |
| [canvas-3-7-step0-t999.png](../../../../artifacts/notes/progress/score-space/00-ladder/overview/canvas-3-7-step0-t999.png) | canvas-3-7-step0-t999.png | `context/methods/07-the-score-path/01-overview.md` |
| [canvas-3-7-step15-t693.png](../../../../artifacts/notes/progress/score-space/00-ladder/overview/canvas-3-7-step15-t693.png) | canvas-3-7-step15-t693.png | `context/methods/07-the-score-path/01-overview.md` |
| [canvas-3-7.png](../../../../artifacts/notes/progress/score-space/00-ladder/overview/canvas-3-7.png) | canvas-3-7.png | `context/methods/07-the-score-path/01-overview.md` |

### Loop 12

**`artifacts/notes/progress/leakage/`: 5 files**

| File | What it shows | Lands on |
|---|---|---|
| [gpu-line.png](../../../../artifacts/notes/progress/leakage/gpu-line.png) | gpu-line.png | Loop 12 started on 2026-11-24; its landing page is set when its first verdict lands |
| [loop-12-and-the-other-loops.png](../../../../artifacts/notes/progress/leakage/loop-12-and-the-other-loops.png) | loop-12-and-the-other-loops.png | Loop 12 started on 2026-11-24; its landing page is set when its first verdict lands |
| [loop-12-progress.png](../../../../artifacts/notes/progress/leakage/loop-12-progress.png) | Loop 12's six steps: Z1 read the leakage papers, done; Z2 what predicts borrowing, in progress; Z3 one fix; Z4 Paper 2's method; Z5 audit Loops 8 to 10; Z6 verdict. | Loop 12 started on 2026-11-24; its landing page is set when its first verdict lands |
| [mechanism-checker-idea.png](../../../../artifacts/notes/progress/leakage/mechanism-checker-idea.png) | A loop: a pretrained sampler (CO3, ToMe, plain SDXL or PoE) gives its own failure signal at steps 0 to 5, such as attention overlap, d_t maps or a blob count. A small checker reads only that signal and never the subjects' names. Its output… | Loop 12 started on 2026-11-24; its landing page is set when its first verdict lands |
| [where-leakage-could-enter.png](../../../../artifacts/notes/progress/leakage/where-leakage-could-enter.png) | A prompt flows through the text encoder into SDXL's U-Net, where the layout, cross-attention and self-attention sit in one step, then into CO3's sum of predictions and out to a picture with two bodies and borrowed parts. Each of the five p… | Loop 12 started on 2026-11-24; its landing page is set when its first verdict lands |

**`artifacts/notes/progress/leakage/02-what-predicts-borrowing/`: 1 file**

| File | What it shows | Lands on |
|---|---|---|
| [z2-what-and-when.png](../../../../artifacts/notes/progress/leakage/02-what-predicts-borrowing/z2-what-and-when.png) | what Z2 reads and when | Loop 12 started on 2026-11-24; its landing page is set when its first verdict lands |

**`artifacts/notes/progress/leakage/02-what-predicts-borrowing/small-read/`: 1 file**

| File | What it shows | Lands on |
|---|---|---|
| [small-read-two-pairs.png](../../../../artifacts/notes/progress/leakage/02-what-predicts-borrowing/small-read/small-read-two-pairs.png) | Two panels. Left: attention overlap by window, one dot per seed. Early, the two pairs overlap (0.66 against 0.71). Middle and late, cheetah and cougar sit higher (0.83 against 0.69; 0.75 against 0.41). Right: score-map similarity by window… | Loop 12 started on 2026-11-24; its landing page is set when its first verdict lands |

### Scenes in `2-classifier-guidance/artifacts/scenes/`

**`2-classifier-guidance/artifacts/scenes/co3-sampler/`: 8 files**

| File | What it shows | Lands on |
|---|---|---|
| [co3-corrector.png](../../../../2-classifier-guidance/artifacts/scenes/co3-sampler/co3-corrector.png) | Algorithm 3 as a small network, with the printed and coded rules boxed beneath | made before the loops; not part of this drain |
| [co3-geometry-sketch.png](../../../../2-classifier-guidance/artifacts/scenes/co3-sampler/co3-geometry-sketch.png) | co3-geometry-sketch.png | made before the loops; not part of this drain |
| [co3-overview.png](../../../../2-classifier-guidance/artifacts/scenes/co3-sampler/co3-overview.png) | the whole run, with this step in the small forward diagram | made before the loops; not part of this drain |
| [co3-resampler.png](../../../../2-classifier-guidance/artifacts/scenes/co3-sampler/co3-resampler.png) | Algorithm 2 drawn as a small network with its equations | made before the loops; not part of this drain |
| [co3-schedule.png](../../../../2-classifier-guidance/artifacts/scenes/co3-sampler/co3-schedule.png) | the 50-step schedule with the six CO3 steps zoomed and the branch's rules boxed | made before the loops; not part of this drain |
| [co3-toy-corrector.png](../../../../2-classifier-guidance/artifacts/scenes/co3-sampler/co3-toy-corrector.png) | co3-toy-corrector.png | made before the loops; not part of this drain |
| [co3-weights.png](../../../../2-classifier-guidance/artifacts/scenes/co3-sampler/co3-weights.png) | equations 14 and 15 as a small network beside equation 15's curve | made before the loops; not part of this drain |
| [plain-vs-co3.png](../../../../2-classifier-guidance/artifacts/scenes/co3-sampler/plain-vs-co3.png) | plain-vs-co3.png | made before the loops; not part of this drain |

**`2-classifier-guidance/artifacts/scenes/co3-sampler/exact-plane/`: 4 files**

| File | What it shows | Lands on |
|---|---|---|
| [co3-exact-plane.png](../../../../2-classifier-guidance/artifacts/scenes/co3-sampler/exact-plane/co3-exact-plane.png) | the flat sheet through the three Tweedie means at each of CO3's six steps, both pairs | made before the loops; not part of this drain |
| [co3-path-difference.png](../../../../2-classifier-guidance/artifacts/scenes/co3-sampler/exact-plane/co3-path-difference.png) | relative gap between the CO3 and plain latents over 50 steps, both pairs, CO3's steps shaded | made before the loops; not part of this drain |
| [co3-steps-cheetah-cougar.png](../../../../2-classifier-guidance/artifacts/scenes/co3-sampler/exact-plane/co3-steps-cheetah-cougar.png) | cheetah and cougar: four decoded guesses per CO3 step with distances and weights, and the finished pictures | made before the loops; not part of this drain |
| [co3-steps-eagle-fox.png](../../../../2-classifier-guidance/artifacts/scenes/co3-sampler/exact-plane/co3-steps-eagle-fox.png) | eagle and fox: the same grid, smaller | made before the loops; not part of this drain |

**`2-classifier-guidance/artifacts/scenes/co3-sampler/reference/`: 3 files**

| File | What it shows | Lands on |
|---|---|---|
| [owner-annotated-progress-with-code-boxes.png](../../../../2-classifier-guidance/artifacts/scenes/co3-sampler/reference/owner-annotated-progress-with-code-boxes.png) | owner-annotated-progress-with-code-boxes.png | made before the loops; not part of this drain |
| [owner-network-weights-and-equations.png](../../../../2-classifier-guidance/artifacts/scenes/co3-sampler/reference/owner-network-weights-and-equations.png) | owner-network-weights-and-equations.png | made before the loops; not part of this drain |
| [owner-network-with-rule-boxes.png](../../../../2-classifier-guidance/artifacts/scenes/co3-sampler/reference/owner-network-with-rule-boxes.png) | owner-network-with-rule-boxes.png | made before the loops; not part of this drain |

**`2-classifier-guidance/artifacts/scenes/co3-sampler/source/`: 3 files**

| File | What it shows | Lands on |
|---|---|---|
| [co3-algorithm-1-ddim-with-corrector.png](../../../../2-classifier-guidance/artifacts/scenes/co3-sampler/source/co3-algorithm-1-ddim-with-corrector.png) | co3-algorithm-1-ddim-with-corrector.png | made before the loops; not part of this drain |
| [co3-algorithms-2-and-3-and-equation-15.png](../../../../2-classifier-guidance/artifacts/scenes/co3-sampler/source/co3-algorithms-2-and-3-and-equation-15.png) | co3-algorithms-2-and-3-and-equation-15.png | made before the loops; not part of this drain |
| [co3-eagle-fox-seed-6001.png](../../../../2-classifier-guidance/artifacts/scenes/co3-sampler/source/co3-eagle-fox-seed-6001.png) | co3-eagle-fox-seed-6001.png | made before the loops; not part of this drain |

**`2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/`: 1 file**

| File | What it shows | Lands on |
|---|---|---|
| [page_template.html](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/page_template.html) | page_template.html | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |

**`2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/node_modules/katex/contrib/auto-render/`: 1 file**

| File | What it shows | Lands on |
|---|---|---|
| [index.html](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/node_modules/katex/contrib/auto-render/index.html) | index.html | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |

**`2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/node_modules/katex/contrib/copy-tex/`: 1 file**

| File | What it shows | Lands on |
|---|---|---|
| [index.html](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/node_modules/katex/contrib/copy-tex/index.html) | index.html | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |

<details><summary>`2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/`: 62 files</summary>

| File | What it shows | Lands on |
|---|---|---|
| [00a-real-input-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00a-real-input-dark.png) | 00a-real-input-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [00a-real-input-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00a-real-input-light.png) | 00a-real-input-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [00b-real-maps-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00b-real-maps-dark.png) | 00b-real-maps-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [00c-real-conv1-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00c-real-conv1-dark.png) | 00c-real-conv1-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [00c-real-conv1-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00c-real-conv1-light.png) | 00c-real-conv1-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [00d-real-film-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00d-real-film-dark.png) | 00d-real-film-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [00d-real-film-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00d-real-film-light.png) | 00d-real-film-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [00e-real-pool-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00e-real-pool-dark.png) | 00e-real-pool-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [00e-real-pool-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00e-real-pool-light.png) | 00e-real-pool-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [00f-real-logit-wrong-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00f-real-logit-wrong-dark.png) | 00f-real-logit-wrong-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [00f-real-logit-wrong-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00f-real-logit-wrong-light.png) | 00f-real-logit-wrong-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [00g-real-push-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00g-real-push-dark.png) | 00g-real-push-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [00g-real-push-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00g-real-push-light.png) | 00g-real-push-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [00h-real-unet-712-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00h-real-unet-712-dark.png) | 00h-real-unet-712-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [00h-real-unet-712-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00h-real-unet-712-light.png) | 00h-real-unet-712-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [00i-real-readout-network-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00i-real-readout-network-dark.png) | 00i-real-readout-network-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [00i-real-readout-network-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/00i-real-readout-network-light.png) | 00i-real-readout-network-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [01-forward-mid-convolution-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/01-forward-mid-convolution-dark.png) | 01-forward-mid-convolution-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [01-forward-mid-convolution-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/01-forward-mid-convolution-light.png) | 01-forward-mid-convolution-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [02-forward-at-rest-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/02-forward-at-rest-dark.png) | 02-forward-at-rest-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [02-forward-at-rest-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/02-forward-at-rest-light.png) | 02-forward-at-rest-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [03-training-into-the-kernel-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/03-training-into-the-kernel-dark.png) | 03-training-into-the-kernel-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [03-training-into-the-kernel-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/03-training-into-the-kernel-light.png) | 03-training-into-the-kernel-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [04-training-stops-at-maps-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/04-training-stops-at-maps-dark.png) | 04-training-stops-at-maps-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [04-training-stops-at-maps-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/04-training-stops-at-maps-light.png) | 04-training-stops-at-maps-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [05-push-through-the-unet-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/05-push-through-the-unet-dark.png) | 05-push-through-the-unet-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [05-push-through-the-unet-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/05-push-through-the-unet-light.png) | 05-push-through-the-unet-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [06-push-into-the-pixels-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/06-push-into-the-pixels-dark.png) | 06-push-into-the-pixels-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [06-push-into-the-pixels-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/06-push-into-the-pixels-light.png) | 06-push-into-the-pixels-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [07-push-broken-run-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/07-push-broken-run-dark.png) | 07-push-broken-run-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [07-push-broken-run-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/07-push-broken-run-light.png) | 07-push-broken-run-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [08-noise-sweep-at-712-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/08-noise-sweep-at-712-dark.png) | 08-noise-sweep-at-712-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [08-noise-sweep-at-712-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/08-noise-sweep-at-712-light.png) | 08-noise-sweep-at-712-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [09-push-length-at-1-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/09-push-length-at-1-dark.png) | 09-push-length-at-1-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [09-push-length-at-1-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/09-push-length-at-1-light.png) | 09-push-length-at-1-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [10-identity-training-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/10-identity-training-dark.png) | 10-identity-training-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [10-identity-training-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/10-identity-training-light.png) | 10-identity-training-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [11-noise-real-scores-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/11-noise-real-scores-dark.png) | 11-noise-real-scores-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [11-noise-real-scores-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/11-noise-real-scores-light.png) | 11-noise-real-scores-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [12-convolution-hover-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/12-convolution-hover-dark.png) | 12-convolution-hover-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [12-convolution-hover-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/12-convolution-hover-light.png) | 12-convolution-hover-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [13-push-score-card-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/13-push-score-card-dark.png) | 13-push-score-card-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [13-push-score-card-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/13-push-score-card-light.png) | 13-push-score-card-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [14-two-digits-before-push-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/14-two-digits-before-push-dark.png) | 14-two-digits-before-push-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [14-two-digits-before-push-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/14-two-digits-before-push-light.png) | 14-two-digits-before-push-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [15-two-digits-push-acting-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/15-two-digits-push-acting-dark.png) | 15-two-digits-push-acting-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [16-two-digits-end-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/16-two-digits-end-dark.png) | 16-two-digits-end-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [16-two-digits-end-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/16-two-digits-end-light.png) | 16-two-digits-end-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [17-two-digits-lost-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/17-two-digits-lost-dark.png) | 17-two-digits-lost-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [17-two-digits-lost-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/17-two-digits-lost-light.png) | 17-two-digits-lost-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [p1-real-conv1-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/p1-real-conv1-dark.png) | p1-real-conv1-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [p1-real-conv1-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/p1-real-conv1-light.png) | p1-real-conv1-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [p2-real-logit-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/p2-real-logit-dark.png) | p2-real-logit-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [p2-real-logit-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/p2-real-logit-light.png) | p2-real-logit-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [p3-network-forward-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/p3-network-forward-dark.png) | p3-network-forward-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [p3-network-forward-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/p3-network-forward-light.png) | p3-network-forward-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [p4-training-end-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/p4-training-end-dark.png) | p4-training-end-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [p4-training-end-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/p4-training-end-light.png) | p4-training-end-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [p5-real-readout-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/p5-real-readout-dark.png) | p5-real-readout-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [p5-real-readout-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/p5-real-readout-light.png) | p5-real-readout-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [p6-two-digits-dark.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/p6-two-digits-dark.png) | p6-two-digits-dark.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [p6-two-digits-light.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/p6-two-digits-light.png) | p6-two-digits-light.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |

</details>

**`2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills-codex/`: 4 files**

| File | What it shows | Lands on |
|---|---|---|
| [1-without-the-push-the-digits-fuse.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills-codex/1-without-the-push-the-digits-fuse.png) | the plain run's guess of the clean canvas at steps 10, 20, 30 and 49, then its final canvas, judged one blob | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [2-the-gradient-pulls-the-maps-apart.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills-codex/2-the-gradient-pulls-the-maps-apart.png) | step 20, plain above and pushed below: the two maps with their centres marked, and the gradient g | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [3-after-the-push-two-digits.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills-codex/3-after-the-push-two-digits.png) | the plain and pushed final canvases side by side, judged one blob and two clear digits | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [4-maps-separate-only-while-the-push-acts.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills-codex/4-maps-separate-only-while-the-push-acts.png) | three curves over 50 sampler steps: map centres apart, overlap and logit, plain against pushed, with the push window shaded | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |

<details><summary>`2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/`: 28 files</summary>

| File | What it shows | Lands on |
|---|---|---|
| [desktop-light-01.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/desktop-light-01.png) | desktop-light-01.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [desktop-light-02.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/desktop-light-02.png) | desktop-light-02.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [desktop-light-03.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/desktop-light-03.png) | desktop-light-03.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [desktop-light-04.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/desktop-light-04.png) | desktop-light-04.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [desktop-light-05.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/desktop-light-05.png) | desktop-light-05.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [desktop-light-06.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/desktop-light-06.png) | desktop-light-06.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [desktop-light-07.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/desktop-light-07.png) | desktop-light-07.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [desktop-light-08.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/desktop-light-08.png) | desktop-light-08.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [desktop-light-09.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/desktop-light-09.png) | desktop-light-09.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [desktop-light-10.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/desktop-light-10.png) | desktop-light-10.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [desktop-light-11.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/desktop-light-11.png) | desktop-light-11.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [desktop-light-12.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/desktop-light-12.png) | desktop-light-12.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [desktop-light-13.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/desktop-light-13.png) | desktop-light-13.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [desktop-light-14.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/desktop-light-14.png) | desktop-light-14.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [phone-dark-01.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/phone-dark-01.png) | phone-dark-01.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [phone-dark-02.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/phone-dark-02.png) | phone-dark-02.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [phone-dark-03.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/phone-dark-03.png) | phone-dark-03.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [phone-dark-04.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/phone-dark-04.png) | phone-dark-04.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [phone-dark-05.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/phone-dark-05.png) | phone-dark-05.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [phone-dark-06.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/phone-dark-06.png) | phone-dark-06.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [phone-dark-07.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/phone-dark-07.png) | phone-dark-07.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [phone-dark-08.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/phone-dark-08.png) | phone-dark-08.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [phone-dark-09.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/phone-dark-09.png) | phone-dark-09.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [phone-dark-10.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/phone-dark-10.png) | phone-dark-10.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [phone-dark-11.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/phone-dark-11.png) | phone-dark-11.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [phone-dark-12.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/phone-dark-12.png) | phone-dark-12.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [phone-dark-13.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/phone-dark-13.png) | phone-dark-13.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |
| [phone-dark-14.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/stills/gate/phone-dark-14.png) | phone-dark-14.png | `context/methods/03-the-digit-map-checkers/03-the-miniature.md`, frames of one interactive page; the page is linked, not each frame |

</details>

**`2-classifier-guidance/artifacts/scenes/digit-map-checkers/numbers/env/`: 1 file**

| File | What it shows | Lands on |
|---|---|---|
| [canvas-kinds-sheet.png](../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/numbers/env/canvas-kinds-sheet.png) | canvas-kinds-sheet.png | `context/methods/03-the-digit-map-checkers/`, one page per rung, overview first |

**`2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/`: 1 file**

| File | What it shows | Lands on |
|---|---|---|
| [style-sample-clip-contrastive.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/style-sample-clip-contrastive.png) | style-sample-clip-contrastive.png | none: a style sample for drawing, not a result or a method |

**`2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/1-illustrations/drop/`: 7 files**

| File | What it shows | Lands on |
|---|---|---|
| [ni1-a-cheap-colour-view-of-the-latent.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/1-illustrations/drop/ni1-a-cheap-colour-view-of-the-latent.png) | ni1-a-cheap-colour-view-of-the-latent.png | `context/methods/04-the-sentence-score-push/`, one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [ni2-one-picture-against-two-sentences.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/1-illustrations/drop/ni2-one-picture-against-two-sentences.png) | ni2-one-picture-against-two-sentences.png | `context/methods/04-the-sentence-score-push/`, one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [ni3-training-on-half-drawn-pictures.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/1-illustrations/drop/ni3-training-on-half-drawn-pictures.png) | ni3-training-on-half-drawn-pictures.png | `context/methods/04-the-sentence-score-push/`, one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [ni4-the-push-bends-the-noise-prediction.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/1-illustrations/drop/ni4-the-push-bends-the-noise-prediction.png) | ni4-the-push-bends-the-noise-prediction.png | `context/methods/04-the-sentence-score-push/`, one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [ni5-two-places-the-signal-is-cut-before-rev1.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/1-illustrations/drop/ni5-two-places-the-signal-is-cut-before-rev1.png) | ni5-two-places-the-signal-is-cut-before-rev1.png | `context/methods/04-the-sentence-score-push/`, one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [ni5-two-places-the-signal-is-cut.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/1-illustrations/drop/ni5-two-places-the-signal-is-cut.png) | ni5-two-places-the-signal-is-cut.png | `context/methods/04-the-sentence-score-push/`, one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |
| [ni6-trained-later-than-we-push.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/1-illustrations/drop/ni6-trained-later-than-we-push.png) | ni6-trained-later-than-we-push.png | `context/methods/04-the-sentence-score-push/`, one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |

**`2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/overview/`: 1 file**

| File | What it shows | Lands on |
|---|---|---|
| [w4-input-canvas.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/overview/w4-input-canvas.png) | w4-input-canvas.png | `context/methods/04-the-sentence-score-push/`, one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip |

**`2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/reference/`: 3 files**

| File | What it shows | Lands on |
|---|---|---|
| [clip-frame-beside-own-model.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/reference/clip-frame-beside-own-model.png) | clip-frame-beside-own-model.png | none: style references for drawing |
| [dual-contrastive-attract-repel.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/reference/dual-contrastive-attract-repel.png) | dual-contrastive-attract-repel.png | none: style references for drawing |
| [multi-view-contrastive-loss.png](../../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/reference/multi-view-contrastive-loss.png) | multi-view-contrastive-loss.png | none: style references for drawing |

**`2-classifier-guidance/artifacts/scenes/sdxl-down-the-layers/`: 5 files**

| File | What it shows | Lands on |
|---|---|---|
| [l0-sdxl-from-prompt-to-picture.png](../../../../2-classifier-guidance/artifacts/scenes/sdxl-down-the-layers/l0-sdxl-from-prompt-to-picture.png) | SDXL from the prompt through two text encoders, the U-Net 50 times and the VAE decoder to the eagle and fox picture, with CO3's box and Z2's three readings marked | made before the loops; not part of this drain |
| [l05a-inside-the-unet.png](../../../../2-classifier-guidance/artifacts/scenes/sdxl-down-the-layers/l05a-inside-the-unet.png) | The SDXL U-Net as a U: down 0 at 128 by 128, down 1 at 64, down 2 at 32, a middle, and up 0, 1 and 2 back to 128, with skip widths and the cross-attention layers Z2 reads marked in gold | made before the loops; not part of this drain |
| [l05b-one-transformer-block.png](../../../../2-classifier-guidance/artifacts/scenes/sdxl-down-the-layers/l05b-one-transformer-block.png) | One transformer block at 32 by 32 and 1280 channels: self-attention, cross-attention and feed-forward each with a LayerNorm before and its input added back, with cross-attention opened into to_q, to_k, to_v, the 1024 by 77 weight table and… | made before the loops; not part of this drain |
| [l05c-noise-guess-and-clean-guess.png](../../../../2-classifier-guidance/artifacts/scenes/sdxl-down-the-layers/l05c-noise-guess-and-clean-guess.png) | The eagle and fox picture noised to SDXL's alpha-bar at steps 0, 5, 20 and 45, the formula from the noise guess to the clean guess, and an empty slot for the decoded clean guesses at CO3's corrected steps | made before the loops; not part of this drain |
| [l2-one-attention-head-by-hand.png](../../../../2-classifier-guidance/artifacts/scenes/sdxl-down-the-layers/l2-one-attention-head-by-hand.png) | Left: a 2 by 2 canvas, 3 words, queries, keys, values, the score and softmax tables and each word's column laid back on the canvas. Right: the real eagle and fox columns at steps 3, 20 and 40 with the animals' outlines | made before the loops; not part of this drain |

**`2-classifier-guidance/artifacts/scenes/word-map-checker/1-illustrations/drop/`: 9 files**

| File | What it shows | Lands on |
|---|---|---|
| [wi1-where-each-word-landed-before-rev1.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/1-illustrations/drop/wi1-where-each-word-landed-before-rev1.png) | wi1-where-each-word-landed-before-rev1.png | `context/methods/06-the-word-map-checker/`, one page per rung, overview first |
| [wi1-where-each-word-landed.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/1-illustrations/drop/wi1-where-each-word-landed.png) | wi1-where-each-word-landed.png | `context/methods/06-the-word-map-checker/`, one page per rung, overview first |
| [wi2-a-small-judge-reads-only-the-two-maps.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/1-illustrations/drop/wi2-a-small-judge-reads-only-the-two-maps.png) | wi2-a-small-judge-reads-only-the-two-maps.png | `context/methods/06-the-word-map-checker/`, one page per rung, overview first |
| [wi3-it-learned-the-word-instead-of-the-picture-before-rev1.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/1-illustrations/drop/wi3-it-learned-the-word-instead-of-the-picture-before-rev1.png) | wi3-it-learned-the-word-instead-of-the-picture-before-rev1.png | `context/methods/06-the-word-map-checker/`, one page per rung, overview first |
| [wi3-it-learned-the-word-instead-of-the-picture.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/1-illustrations/drop/wi3-it-learned-the-word-instead-of-the-picture.png) | wi3-it-learned-the-word-instead-of-the-picture.png | `context/methods/06-the-word-map-checker/`, one page per rung, overview first |
| [wi4-the-judges-complaint-pushes-the-noise-before-rev1.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/1-illustrations/drop/wi4-the-judges-complaint-pushes-the-noise-before-rev1.png) | wi4-the-judges-complaint-pushes-the-noise-before-rev1.png | `context/methods/06-the-word-map-checker/`, one page per rung, overview first |
| [wi4-the-judges-complaint-pushes-the-noise.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/1-illustrations/drop/wi4-the-judges-complaint-pushes-the-noise.png) | wi4-the-judges-complaint-pushes-the-noise.png | `context/methods/06-the-word-map-checker/`, one page per rung, overview first |
| [wi5-one-capped-nudge-against-sixteen-big-steps.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/1-illustrations/drop/wi5-one-capped-nudge-against-sixteen-big-steps.png) | wi5-one-capped-nudge-against-sixteen-big-steps.png | `context/methods/06-the-word-map-checker/`, one page per rung, overview first |
| [wi6-the-map-can-move-without-a-second-animal.png](../../../../2-classifier-guidance/artifacts/scenes/word-map-checker/1-illustrations/drop/wi6-the-map-can-move-without-a-second-animal.png) | wi6-the-map-can-move-without-a-second-animal.png | `context/methods/06-the-word-map-checker/`, one page per rung, overview first |

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

