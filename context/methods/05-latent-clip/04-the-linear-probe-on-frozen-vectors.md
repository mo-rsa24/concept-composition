# 🌍 Context: the linear probe on frozen vectors

⬅️ Previous: [how its push differs from NoisyCLIP's](03-how-its-push-differs-from-noisyclips.md) | 📋 [The pages of this method](01-overview.md#the-pages-of-this-method) | Next: [the methods index](../00-INDEX.md) ➡️

Loop 9's step W2b trained the smallest thing it could on top of three frozen picture encoders (NoisyCLIP, frozen CLIP and Latent CLIP): one logistic regression per sampler step, reading the encoder's vector for a noised picture and answering yes or no. It asked whether those vectors hold counting and prompt match in a form a single weighted sum can read, on animal and object pairs the probe never trained on. Unlike the rest of this folder, it reads and never pushes. The verdict (null for all three encoders) belongs to [the linear probe finding](../../../2-classifier-guidance/report/the-sentence-score/does-a-linear-probe-on-frozen-image-vectors-read-unseen-pairs.md); this page explains the method, with Latent CLIP's numbers as the running case.

<a href="../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/overview/ov2-the-linear-probe-on-frozen-vectors.png"><img src="../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/overview/ov2-the-linear-probe-on-frozen-vectors.png" width="720" alt="The linear probe, left to right: labels from each picture's kind (counting, 1 two animals and 0 hybrid or single; prompt match, 1 the asked pair and 0 wrong pair), a noised cat and dog training picture through a grey frozen image encoder (NoisyCLIP, frozen CLIP or Latent CLIP) to one vector of 768 or 640 numbers, a green logistic regression with one weight per number plus a bias, 16 probes one per step, a box reading p equals sigma of w dot v plus b fitted on 119 pairs and read once on 7 unseen pairs with a control on random numbers, and hollow seen and unseen bars marked with a question mark beside a grey random-numbers bar near 0.5 under a dashed 0.75 bar; an inset multiplies the picture's vector by the learned weights; a dashed grey line says no push in this step"></a>

*Read left to right in five columns. What goes in: the two label sets, "counting, 1: two animals, 0: hybrid, single" and "prompt match, 1: the asked pair, 0: wrong pair", taken from each picture's kind, and a training picture noised with illustrative noise. Frozen image encoder: one grey box, "NoisyCLIP, frozen CLIP or Latent CLIP, frozen". Image vector: "768 or 640 numbers". Linear probe, per step: "logistic regression, one weight per number + a bias", "16 probes, one per step". Read seen and unseen pairs: the box "p = σ(w · v + b), fitted on 119 pairs, read once on 7 unseen pairs, control: a probe on random numbers", and three bars under the label "balanced accuracy, 0 to 1: not yet measured", with a dashed "0.75 bar": seen pairs and unseen pairs hollow with "?", random numbers solid "near 0.5". The inset multiplies "this picture's vector v" by "learned weights w" to "one number from 0 to 1". The dashed grey frame reads "no push in this step: it reads and does not steer".* ✅ *Read on the picture at full size. It was drawn before the run, so its bars are not results; the measured values are in [what the probes read](#what-the-probes-read).* ✍️ *([the ladder's card](../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/README.md#3-w2b-a-linear-probe-on-frozen-vectors), line 53)*

## Table of contents

- [Words this page uses](#words-this-page-uses)
- [What one probe does](#what-one-probe-does)
- [The names, taken apart](#the-names-taken-apart)
- [The probe, by what moves](#the-probe-by-what-moves)
- [What the probes read](#what-the-probes-read)
- [How it fails and how you know it worked](#how-it-fails-and-how-you-know-it-worked)
- [Where the sources disagree](#where-the-sources-disagree)
- [Where this came from](#where-this-came-from)
- [Read more](#read-more)

## Words this page uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-one-probe-does) ➡️

**Image vector v**: what an encoder's picture side gives for one noised latent, set to length 1: 768 numbers for NoisyCLIP and frozen CLIP, 640 for Latent CLIP.

**Kind**: the label each training picture already carried from an earlier labeller: `positive` (the asked pair, two bodies), `wrong pair`, `hybrid` or `single`.

**Seen and unseen pairs**: seen pairs appear in the probe's training pictures, and their seen test is 10% of their pictures held back from fitting. Unseen pairs appear in no training picture and are read once.

**Step**: one of the 16 sampler steps 0 to 15 of SDXL's 50, the window where a push would act. Each picture is noised to each step with one shared noise draw.

## What one probe does

Navigation: ⬅️ [Words this page uses](#words-this-page-uses) | 📋 [TOC](#table-of-contents) | [Next](#the-names-taken-apart) ➡️

**One probe takes one encoder's vector at one step, multiplies each number by its own learned weight, adds a bias, and says yes when the sum is above 0.** ✅ `fit_logistic` and `probe` in `2-classifier-guidance/scripts/noise_text_critic/w2b_probe.py`, lines 125 to 185; the yes rule is `X[m] @ w + b > 0`, line 176.

| Box on the picture | In the run, with Latent CLIP |
|---|---|
| a training picture, noised | a finished latent from the rung 2 training set, noised to step k with the draw seeded 700,000 + k, the same draw for every encoder ✅ lines 96 to 102 |
| the frozen encoder | Latent CLIP, read through the divide and pool of [page 2](02-reading-a-latent-with-no-vae.md#the-path-stage-by-stage), under 16-bit arithmetic ✅ lines 103 to 107 |
| the vector | 640 numbers of length 1, saved to disk once per picture and step ✅ lines 108 to 115 |
| the probe | 640 weights and 1 bias, one probe per property and step ✅ |
| the read | balanced accuracy on the seen test and on the 7 unseen pairs ✅ lines 172 to 177 |

**How many probes.** Three encoders and the random-number control, two properties, 16 steps: 4 × 2 × 16 = 128 probes, each fitted on CPU from the saved vectors. ✅ Computed from `STEPS`, `MODELS` and `LABELS`, lines 42 to 44, and the control in `stage_fit`. The run that extracted the vectors for all 4,039 pictures took 13 minutes. ✍️ [The linear probe review](../../../artifacts/notes/progress/noise-text-critic/02b-linear-probe/review.md), line 11.

| Property | Yes | No | Fit pictures (yes, no) | Unseen pictures (yes, no) |
|---|---|---|---|---|
| counting | `positive`, `wrong pair` | `hybrid`, `single` | 785, 2,483 | 195, 227 |
| prompt match | `positive` | `wrong pair` | 297, 488 | 109, 86 |

✅ Labels read in `LABELS`, line 44; counts read in `counts` in [the linear probe result file](../../../artifacts/notes/progress/noise-text-critic/02b-linear-probe/result.json). The 7 unseen pairs are bison and horse, donkey and duck, guitar and traffic cone, horse and pig, llama and alpaca, rhino and zebra, and windmill and tractor. ✍️ [The linear probe design](../../../artifacts/notes/progress/noise-text-critic/02b-linear-probe/design.md#the-pictures), line 29.

## The names, taken apart

Navigation: ⬅️ [What one probe does](#what-one-probe-does) | 📋 [TOC](#table-of-contents) | [Next](#the-probe-by-what-moves) ➡️

**Four names, each pointing at one box of the opening picture.**

🧸 **Toy:** a fruit stall sorts fruit into ripe and not ripe from two numbers per fruit, colour and softness, each from 0 to 1. A probe with weights (2, 1) and bias −1.5 reads fruit A at (0.9, 0.4) and fruit B at (0.3, 0.2). Expected before computing: A, redder and softer, comes out ripe and B does not.

| Fruit | Colour × 2 | Softness × 1 | Plus the bias | Sum | σ(sum) | Call |
|---|---|---|---|---|---|---|
| A (0.9, 0.4) | 1.8 | 0.4 | −1.5 | 0.7 | 0.668 | ripe |
| B (0.3, 0.2) | 0.6 | 0.2 | −1.5 | −0.7 | 0.332 | not ripe |

✅ Computed. The fruit and the weights are invented.

🔤 **The name:** linear probe
- **linear**: in a straight line, with no bends. On the toy, the score is a weighted sum along a row, and doubling a fruit's softness adds exactly one more weight's worth.
- **probe**: a slender tool pushed in to find out what is inside without changing it. On the picture, the green box after the frozen encoder: it learns where in the encoder's numbers a property sits and leaves the encoder as it was.
- **Formula, in those words:** p = σ(w · v + b), one weight per number in the vector, plus one bias.
- **Consequence:** a linear probe can only read a property that lies along one direction of the vector. Run back on this test: Latent CLIP's counting probe has 640 weights and a bias, and on the 7 unseen pairs it reads 0.527 balanced accuracy, near the 0.5 that guessing gives. ✅ `latent_clip.counting.unseen_mean` in the result file. The probe cannot say whether a bent reader, one with a hidden layer, would read more.

🔤 **The name:** frozen
- **frozen**: fixed in place, unable to change. On the picture, the grey encoder box.
- **Formula, in those words:** while the probe is fitted, the encoder's weights get no update; only w and b do.
- **Consequence:** because nothing upstream changes, each picture's vector at each step is computed once and saved, and all 128 probes are then fitted from the saved files. A probe that unfroze the encoder would be a different test, the one Loop 9's step W4 ran by training a whole encoder of its own ([its finding](../../../2-classifier-guidance/report/the-sentence-score/does-our-own-noise-aware-model-read-unseen-digit-pairs.md)). ✅ Read in `stage_fit`, which loads `vectors-<model>.npy`, lines 188 to 199.

🔤 **The name:** logistic regression
- **logistic**: from the logistic curve, σ(z) = 1 / (1 + e^(−z)), which squashes any number into 0 to 1. On the toy, the σ column: 0.7 becomes 0.668.
- **regression**: here a misleading word. A logistic regression is a classifier: it answers yes or no. The name is historical, because it fits a straight line (the sum) underneath the squashing.
- **Formula, in those words:** fit w and b so that σ(w · v + b) is near 1 for the yes pictures and near 0 for the no pictures, by making the cross-entropy loss small; call yes when σ is above 0.5, which is the same as the sum being above 0.
- **Consequence:** move the toy's bias from −1.5 to −0.5 and fruit B's sum rises from −0.7 to 0.3, σ to 0.574, and its call flips to ripe. The bias sets where the line between yes and no sits. ✅ Computed.

🔤 **The name:** balanced accuracy
- **accuracy**: the share of calls that are right.
- **balanced**: each class gets the same say, however many pictures it has.
- **Formula, in those words:** the share of yes pictures called yes, plus the share of no pictures called no, divided by 2. 0.5 is chance and 1 is perfect.
- **Consequence:** on the 422 unseen counting pictures (195 yes, 227 no), a probe that says yes to everything is right on 195, an accuracy of 0.462, and has a balanced accuracy of (1 + 0) / 2 = 0.5. One that says no to everything has an accuracy of 0.538 and the same 0.5. ✅ Computed from the result file's counts; the rule is `balanced`, line 121.

## The probe, by what moves

Navigation: ⬅️ [The names, taken apart](#the-names-taken-apart) | 📋 [TOC](#table-of-contents) | [Next](#what-the-probes-read) ➡️

**The fit makes a class-weighted cross-entropy small, with a penalty on large weights whose strength is chosen per step by cross-validation.** ✅ `fit_logistic`, lines 125 to 141, and `choose_l2`, lines 144 to 157. The colours: green for the probe's learned pieces, blue for the frozen vector, grey for what the fit sets.

$$
\underbrace{L}_{\text{what the fit makes small}}
= \frac{1}{n}\sum_{i}\textcolor{#6b7a88}{\underbrace{c_{y_i}}_{\text{class weight}}}\;\text{BCE}\big(\textcolor{#1d6b3a}{w}\cdot\textcolor{#1f5fd0}{v_i} + \textcolor{#1d6b3a}{b},\; y_i\big)
+ \textcolor{#6b7a88}{\underbrace{\lambda}_{\text{penalty strength}}}\,\frac{\lVert \textcolor{#1d6b3a}{w} \rVert^2}{n}
$$

BCE is the binary cross-entropy, −log σ(z) for a yes picture and −log(1 − σ(z)) for a no picture.

| Symbol | On the counting fit, every encoder | If it moved |
|---|---|---|
| c_yes = 0.5 / (share of yes) | 0.5 / (785 / 3,268) = 2.082 | the yes pictures' total weight, 785 × 2.082 = 1,634, equals the no pictures', 2,483 × 0.658 = 1,634; set both weights to 1 and a probe that always says no scores 0.760 plain accuracy on the fit set, at 0.5 balanced |
| c_no = 0.5 / (share of no) | 0.658 | |
| λ | one of 0.01, 0.1, 1 or 10, chosen per step by 5-fold cross-validation on the fit set, folds grouped by pair and seed | Latent CLIP's counting probe chose 0.01 at 13 of its 16 steps and 0.1 at 3; a larger λ pulls every weight toward 0 and the probe toward guessing |
| the call | yes when w · v + b > 0 | the threshold is fixed at σ = 0.5 and never tuned on the test pictures |

✅ The weights and totals computed from the result file's counts; the λ choices read in `latent_clip.counting.l2`; the grid and folds read in `L2_GRID` and `FOLDS`, lines 37 to 38, and `choose_l2`.

## What the probes read

Navigation: ⬅️ [The probe, by what moves](#the-probe-by-what-moves) | 📋 [TOC](#table-of-contents) | [Next](#how-it-fails-and-how-you-know-it-worked) ➡️

**Every encoder reads a little above chance on seen pairs and closer to chance on the 7 unseen pairs, and every one is null against the 0.75 bar.** ✍️ [The linear probe finding, section 1](../../../2-classifier-guidance/report/the-sentence-score/does-a-linear-probe-on-frozen-image-vectors-read-unseen-pairs.md#1-no-encoder-reads-the-unseen-pairs), whose table is quoted here; ✅ every mean rechecked in the result file.

| Encoder | Counting, seen | Counting, unseen | Prompt match, seen | Prompt match, unseen |
|---|---|---|---|---|
| NoisyCLIP | 0.637 | 0.560 | 0.644 | 0.586 |
| frozen CLIP | 0.579 | 0.526 | 0.614 | 0.552 |
| Latent CLIP | 0.575 | 0.527 | 0.579 | 0.527 |
| random numbers (control) | 0.502 | 0.506 | 0.505 | 0.497 |

Each cell is mean balanced accuracy over steps 0 to 15. The bar asked for at least 0.75 on unseen pairs and within 0.05 of the seen score. ✍️ [The design's bars](../../../artifacts/notes/progress/noise-text-critic/02b-linear-probe/design.md#the-bars); ✅ `MIN_BA` and `MAX_GAP`, lines 32 to 33.

**Latent CLIP, step by step.** Its counting probe reads from 0.486 (step 1) to 0.656 (step 15) on seen pairs and from 0.478 (step 6) to 0.571 (step 12) on unseen pairs. ✅ `latent_clip.counting` in the result file. The design wrote that a Latent CLIP probe passing at steps 0 to 5 would surprise, since it never trained on a noisy latent; its unseen counting there averages 0.512. ✅ Computed from the same entry; ✍️ the design, line 60.

> 🖼️ **Diagram wanted**: balanced accuracy against step 0 to 15, one panel per property, one line per reader, seen solid and unseen dashed, with the 0.75 bar and the 0.5 chance line. The slot and its data path are owned by [the linear probe finding, section 1](../../../2-classifier-guidance/report/the-sentence-score/does-a-linear-probe-on-frozen-image-vectors-read-unseen-pairs.md#1-no-encoder-reads-the-unseen-pairs); once rendered there, embed it here.

## How it fails and how you know it worked

Navigation: ⬅️ [What the probes read](#what-the-probes-read) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**The control is a probe fitted the same way on random numbers, and it stays at chance, so the encoders' small lead over 0.5 comes from the pictures.** ✅ `random_numbers` and `random_control_passes: true` in the result file; the control's vectors are 768 standard normal numbers per picture and step, set to length 1, seed 0 (lines 196 to 200).

💥 **Broken run (without):** labels that can be predicted from something other than the picture, such as one kind sitting in one part of the file order.
- **Predict:** what does the random-number probe read on unseen pairs?
- **Wrong:** above 0.6, which the design treats as a sign that the labels leak.
- **Right**, as measured: 0.506 on counting and 0.497 on prompt match.

✅ The ceiling 0.6 is `RANDOM_CEILING`, line 36; the reading of a breach is ✍️ the design, line 58.

**A run that fails loudly.** The first smoke test (job 94110) failed at the fit stage, because training tiles from saved trajectories hold every state in one file and the extractor took all of them; the second (job 94219) kept only the state each tile names, and the full run (job 94220) found 0 latents missing. ✍️ The review, lines 9 to 11; ✅ `missing_latents: 0` in the result file.

> [!NOTE]
> What a pass would not have shown. A probe that reads unseen pairs can still have a gradient that does not move a picture, and a finished picture noised to step 5 is not the sampler's own state at step 5. ✍️ The design, lines 71 and 73. The labels' error rate on these pictures was never measured (line 75).

## Where the sources disagree

Navigation: ⬅️ [How it fails and how you know it worked](#how-it-fails-and-how-you-know-it-worked) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**Two records about the probe say it never ran. [The linear probe finding](../../../2-classifier-guidance/report/the-sentence-score/does-a-linear-probe-on-frozen-image-vectors-read-unseen-pairs.md#where-the-sources-disagree) lists three more disagreements inside the run's own records, and this page resolves none of them.**

| What | One source | The other source |
|---|---|---|
| Whether W2b has run | [the ladder's card](../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/README.md#3-w2b-a-linear-probe-on-frozen-vectors), line 53: "The seen and unseen bars are hollow and marked "?", because W2b has not run"; the opening picture: "not yet measured" | [the linear probe review](../../../artifacts/notes/progress/noise-text-critic/02b-linear-probe/review.md), lines 3 and 11: job 94220 ran on 4,039 pictures and every encoder is null; step W2b is done on the progress board |

## Where this came from

Navigation: ⬅️ [Where the sources disagree](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#read-more) ➡️

| What | How it was established | When |
|---|---|---|
| The noise draw, extraction, fit, λ choice, threshold and control | Read in `2-classifier-guidance/scripts/noise_text_critic/w2b_probe.py`, lines 32 to 44 and 88 to 200 | 2026-11-24 |
| Counts, means, per-step values, λ choices, `missing_latents`, the control | Read in the result file; the class weights, totals, ranges and the steps 0 to 5 mean computed from it | 2026-11-24 |
| The 13-minute run, the job numbers, the smoke test failure | Stated by the review, lines 7 to 11 | 2026-11-24 |
| The unseen pairs, the bars' meaning, the surprise, what a pass cannot say | Stated by the design, lines 29, 50 to 60 and 71 to 75 | 2026-11-24 |
| The toy | Invented and computed this session | 2026-11-24 |
| The opening picture's labels | Read on the picture at full size; its path, its input render's path and every label passed `is_showable`, and it is not on the held pictures list | 2026-11-24 |

## Read more

Navigation: ⬅️ [Where this came from](#where-this-came-from) | 📋 [TOC](#table-of-contents)

| What | Where |
|---|---|
| A last layer that reads an answer out of a network's features, worked on one small network | [the readout guidance chapter, the readout head](../../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/textbook/06-readout-guidance-on-one-small-network.md#readout-head-the-last-layer-that-reads-the-answer-out) |
| Balanced accuracy, the hit rate on each class averaged | [the same chapter, balanced accuracy](../../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/textbook/06-readout-guidance-on-one-small-network.md#balanced-accuracy-the-hit-rate-on-each-class-averaged) |
| Giving each class the same say in a loss | [the same chapter, class-balanced](../../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/textbook/06-readout-guidance-on-one-small-network.md#class-balanced-each-group-gets-the-same-say) |
| The reader that does read early, on SDXL's own features | [the readout checker](../02-the-readout-checker/01-overview.md) |

⬅️ Previous: [how its push differs from NoisyCLIP's](03-how-its-push-differs-from-noisyclips.md) | 📋 [The pages of this method](01-overview.md#the-pages-of-this-method) | Next: [the methods index](../00-INDEX.md) ➡️

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
