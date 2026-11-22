# 📒 Report: which line of code computes which part of guided sampling

**Four pieces of code were read line by line, so each step of guided sampling can be traced to the
file and line that computes it.** These walks explain code and test no claim, so no verdict was
recorded for them.

| Code | What it does | Its line-by-line walk |
|---|---|---|
| Dhariwal and Nichol's classifier guidance, `guided-diffusion` at `22e0df8` | steers a 250-step image sampler with a classifier's gradient | held back, see the warning below; the [one-step stepper on the line toy](https://claude.ai/artifact/1g1Y617YgRmzxDxuPgRnQy) covers one step of it |
| this project's guided sampler, `poe_repair/methods/_classifier_guided.py` | adds that push to the product of a cat and a dog prediction on SDXL | [our guided sampler walk](https://claude.ai/artifact/WwzF3fXBFFb6arZySDA8qv) |
| Readout Guidance's released code, `readout_guidance` at `a8a699c` | trains a small head on SDXL's inner features, then steers with its error | [Readout Guidance loop walk](https://claude.ai/artifact/SHnMMo46yDvhj2GaXuEs7H) |
| this project's count head, `scripts/rung2/train_count_head.py` at `c5e61cec` | trains a head to read how many animals a noisy latent holds | [count head walk](https://claude.ai/artifact/2VgqMaBHqrXNoHdTWjxro1) |

> [!WARNING]
> `2-classifier-guidance/artifacts/results/what-each-line-of-classifier-guidance-computes/classifier-guidance-loop-codewalk.html`
> not shown: holds a subject the pair ledger keeps off sheets. One render from the seeded image run
> is embedded in that walk, so neither the file nor its published link appears on this page. The
> other three walks and the stepper each carry one link to it.

## Table of contents

- [The authors' classifier guidance loop](#the-authors-classifier-guidance-loop)
- [Our guided sampler](#our-guided-sampler)
- [Readout Guidance and our count head](#readout-guidance-and-our-count-head)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## The authors' classifier guidance loop

Navigation: 📋 [TOC](#table-of-contents) | [Next](#our-guided-sampler) ➡️

**On an exact toy, scale 1 samples the cat class and scale 10 ends 0.605 past the target the paper
describes.** The toy is a line whose cat values centre on +1 and dog values on -1, run through the
authors' own sampler with the network and classifier replaced by their exact answers.

| One step of the loop | Where the guided mean goes | What the lines did on images |
|---|---|---|
| <a href="../../artifacts/results/what-each-line-of-classifier-guidance-computes/classifier-guidance-loop-high-level-design-explainer-light.png"><img src="../../artifacts/results/what-each-line-of-classifier-guidance-computes/classifier-guidance-loop-high-level-design-explainer-light.png" width="200" alt="Eight boxes in three labelled rows: the ordinary sampler, the steering, and the scale s"></a> | <a href="../../artifacts/results/what-each-line-of-classifier-guidance-computes/pictures/three_ways_mean.png"><img src="../../artifacts/results/what-each-line-of-classifier-guidance-computes/pictures/three_ways_mean.png" width="200" alt="Mean of 4096 particles against sampler step for no guidance, scale 1 and scale 10, with dashed exact means"></a> | <a href="../../artifacts/results/what-each-line-of-classifier-guidance-computes/pictures/run_grad.png"><img src="../../artifacts/results/what-each-line-of-classifier-guidance-computes/pictures/run_grad.png" width="200" alt="Two log-scale line plots against step: the classifier gradient length and the push as a percentage of the mean, three classes at scales 1 and 10"></a> |
| The top row is the ordinary sampler, the middle row is the steering, and the bottom box is the scale at `classifier_sample.py` line 61 `verified` | y is the mean of 4,096 particles and x is the step, 249 down to 0. Scale 1 ends at 0.964 against an exact 1.000. Scale 10 ends at 2.448 against 1.843 for p(x)·p(cat given x)^10, from [the toy's numbers](../../artifacts/results/what-each-line-of-classifier-guidance-computes/pictures/rig.json) `verified` | y is a length on a log scale and x is the step, seed 0. The corgi's gradient at scale 10 falls from 0.515 at step 199 to 0.113 at step 49. At scale 1 it is 2.637 at step 149 and 1.853 at step 49, from [the image run's numbers](../../artifacts/results/what-each-line-of-classifier-guidance-computes/pictures/run_numbers.json) `verified` |

The 0.605 is 2.448 minus 1.843. The two spreads almost match (standard deviation 0.683 against
0.681), and the run does not show why the location differs. It is one seed on a one-dimensional toy
`stated` (the walk's card).

## Our guided sampler

Navigation: ⬅️ [Previous](#the-authors-classifier-guidance-loop) | 📋 [TOC](#table-of-contents) | [Next](#readout-guidance-and-our-count-head) ➡️

**Our sampler adds four things the authors' code has no line for: the product (line 130), the cap
(lines 46 to 61), the step window (lines 133 and 134) and the check that strength is not zero (line
136)** `stated` ([the sampler's card](../../artifacts/results/what-each-line-of-our-guided-sampler-computes/README.md)).

| One step of the sampler | The push on small numbers | Where a cap of 0.1 would bite |
|---|---|---|
| <a href="../../artifacts/results/what-each-line-of-our-guided-sampler-computes/our-guided-sampler-high-level-design-explainer-light.png"><img src="../../artifacts/results/what-each-line-of-our-guided-sampler-computes/our-guided-sampler-high-level-design-explainer-light.png" width="200" alt="Eleven boxes: the SDXL step on top, two checks and the classifier in the middle, the strength, noise level and cap at the bottom"></a> | <a href="../../artifacts/results/what-each-line-of-our-guided-sampler-computes/pictures/eq_push.png"><img src="../../artifacts/results/what-each-line-of-our-guided-sampler-computes/pictures/eq_push.png" width="200" alt="Four arrow panels: the gradient, the product prediction, the push, and the pushed prediction"></a> | <a href="../../artifacts/results/what-each-line-of-our-guided-sampler-computes/pictures/run_push_vs_cap.png"><img src="../../artifacts/results/what-each-line-of-our-guided-sampler-computes/pictures/run_push_vs_cap.png" width="200" alt="Two dot plots of push size over prediction size at strengths 10 to 300, with the 0.1 cap as a dashed line"></a> |
| One of 50 steps, each box labelled with its lines. Boxes marked OURS have no counterpart in the authors' code `verified` | Each panel is a two-number vector. The gradient (18, 10) times noise level 0.6 and weight 1 is the push (10.8, 6), and (2.1, -1.8) minus the push is (-8.7, -7.8) `verified` | y is push length over prediction length at one step, log scale. x is strength. One dot is one seed's largest step, one dash its median step, seeds 9 to 12. In the product run, which had no cap, 9 of 16 renders pass 0.1: 1 of 4 at strength 30 and all at 100 and 300 `verified` (counted, matches the card) |

## Readout Guidance and our count head

Navigation: ⬅️ [Previous](#our-guided-sampler) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**The released code gives its head a different time input at training and at guidance.** Training
passes the step index (`diffusion_extractor.py` line 148) and guidance passes the scheduler timestep
(`rg_pipeline.py` line 165). At the first guided step those are 8 and 991, and their two time vectors
have cosine -0.462 `verified` (the code lines and `pictures/eq_time.png`). Whether that hurts a render
was not tested `stated`.

| The released loop | Our count head | One guided step's error |
|---|---|---|
| <a href="../../artifacts/results/what-each-line-of-readout-guidance-computes/readout-guidance-loop-high-level-design-explainer-light.png"><img src="../../artifacts/results/what-each-line-of-readout-guidance-computes/readout-guidance-loop-high-level-design-explainer-light.png" width="200" alt="Twelve boxes in two rows: training the head on top, one guided sampling step below"></a> | <a href="../../artifacts/results/what-each-line-of-readout-guidance-computes/our-count-head-high-level-design-explainer-light.png"><img src="../../artifacts/results/what-each-line-of-readout-guidance-computes/our-count-head-high-level-design-explainer-light.png" width="200" alt="Eleven boxes: saved states into the frozen U-Net, three hooks, pooling and a density map, then the detector target, loss and held-out check"></a> | <a href="../../artifacts/results/what-each-line-of-readout-guidance-computes/pictures/eq_readout.png"><img src="../../artifacts/results/what-each-line-of-readout-guidance-computes/pictures/eq_readout.png" width="200" alt="A stick-figure pose, the head reading, their squared error, and where the error gradient lands"></a> |
| The top row trains a head on nine SDXL layers, 6,720 channels once stacked. The bottom row takes one AdamW step on the latent at each guided step `verified` | Three hooked layers are pooled to 32 × 32 and summed into one map whose total is the count. The script holds no push `stated` ([the Readout Guidance card](../../artifacts/results/what-each-line-of-readout-guidance-computes/README.md)) | Four panels at timestep 991: the target pose, the released head's reading (-0.498 to -0.188), their squared error per pixel, and the gradient that error sends back to the features `verified` |

The card's caveat table names each place the count head departs from the released head, line by line.

## Where the sources disagree

Navigation: ⬅️ [Previous](#readout-guidance-and-our-count-head) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Two recorded facts conflict, and the code explains both.**

| What | One source | The other source | What the code shows |
|---|---|---|---|
| the colours of three diagrams | the cards describe teal, green and red boxes by role | the three light PNGs hold 0 pixels of those colours, counted within 12 levels per channel of each colour's hex value | `gen_codewalk.py` writes the role colours into the SVG only (lines 512 to 519 for our sampler, 503 to 512 for Readout Guidance). The SVGs carry them. That the PNGs were exported from the uncoloured drawing is `inferred` |
| which optimizer `rg_numbers.json` names | the Readout Guidance card says it names `torch.optim.AdamW` | the file today names `bnb.optim.AdamW8bit` for float16 latents and `torch.optim.AdamW` otherwise | `rg_pipeline.py` lines 159 and 264 pick the 8-bit optimizer on float16, so the file matches the code. That the card predates an edit to the file is `inferred` |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**How often the cap fires on real capped renders**

The picture above is an uncapped run. The capped logs sit on `/datasets`, which the walk could not
read. Plan 21 records the cap firing on 2 to 6 percent of steps at strength 30 `stated`.

**Anything about the count head's trained behaviour**

No training run or `check.json` existed when its walk was built, so the walk reads the training code
without running it `stated`.

**The size of the real Readout Guidance push**

The backward pass through SDXL did not fit on the laptop, so the walk reports the step rule on
stand-in gradients only `stated`.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. The project numbers are from the sampler walk's miniature, in
[its results file](../../artifacts/results/what-each-line-of-our-guided-sampler-computes/pictures/miniature.json)
and the term figures beside it.

**The product of two experts**

A coffee scores 5 before anyone adjusts it. One taster's recipe brings it to 7 and another's to 6.
To keep both changes, add the two scores and take the shared 5 away once: 7 + 6 - 5 = 8. "Experts"
are the two tasters, and "product" is the name for multiplying their probabilities, which adds their
noise predictions in this way. The formula is ε_PoE = ε_cat + ε_dog - ε_uncond. On the miniature,
(2.1, -0.6) + (0.9, -1.8) - (0.9, -0.6) = (2.1, -1.8). The [product of experts note](../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Product%20Of%20Experts.md#where-to-learn-this)
and [the derivation from Bayes](../../../../../goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/plans/17-poe-rule-from-the-fork.md#1--drip-it-on-the-toy) teach it.

**The push**

Suppose a checker says "move by (2, 1)", the strength is 3 and the noise level is 0.5. The push is
3 × 0.5 × (2, 1) = (3, 1.5), and a prediction of (4, 4) becomes (4, 4) - (3, 1.5) = (1, 2.5). "Push"
is the amount taken off the prediction, and "strength" is the dial that scales it. On the miniature,
1 × 0.6 × (18, 10) = (10.8, 6), and (2.1, -1.8) - (10.8, 6) = (-8.7, -7.8). The term index has no note
for classifier guidance itself. The [classifier note](../../../../../Obsidian/Artificial%20Intelligence/Machine%20Learning/Classifier.md#where-to-learn-this)
covers the part that produces the gradient.

**The cap**

If a prediction has length 5 and the cap is 0.1, the push may be at most 0.1 × 5 = 0.5 long. A push of
length 2 is scaled down by 0.5 / 2 = 0.25 and keeps its direction. On the miniature, the prediction's
length is 2.766, so the limit is 0.277, and the push of length 12.35 is shortened about 45 times.

**A readout head**

A readout head is a small network that reads an answer from a frozen model's inner layers. The toy is
a thermometer clipped to a running engine: it reads the engine and changes nothing inside it. The
[readout head section of chapter 6](../../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/textbook/06-readout-guidance-on-one-small-network.md#readout-head-the-last-layer-that-reads-the-answer-out)
works it on a small network.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| the toy's means and spreads | [the toy's numbers](../../artifacts/results/what-each-line-of-classifier-guidance-computes/pictures/rig.json), read 2026-11-22 | `verified` |
| the image run's gradient lengths | [the image run's numbers](../../artifacts/results/what-each-line-of-classifier-guidance-computes/pictures/run_numbers.json), read 2026-11-22 | `verified` |
| the miniature's vectors | [the miniature's results](../../artifacts/results/what-each-line-of-our-guided-sampler-computes/pictures/miniature.json), read 2026-11-22 | `verified` |
| the released code's lines | the byte-for-byte copy in [the Readout Guidance code file](../../artifacts/results/what-each-line-of-readout-guidance-computes/readout-guidance-loop-code.json) | `verified` |
| how the authors' walk was made | `2-classifier-guidance/artifacts/results/what-each-line-of-classifier-guidance-computes/README.md` | `stated` |
| how the other two walks were made | [the sampler's card](../../artifacts/results/what-each-line-of-our-guided-sampler-computes/README.md) and [the Readout Guidance card](../../artifacts/results/what-each-line-of-readout-guidance-computes/README.md) | `stated` |
| 46 pictures held out of the drain | the other term figures, frames, dark versions and SVG copies in the three code-walk result folders. The classifier-guidance walk page is kept off because it embeds a render of a subject the [pair ledger](../../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md) keeps off sheets | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Decide what happens to the authors' walk, which embeds a held-back render in both its local and published copies. The owner decides.
- [ ] Re-export the three diagram PNGs from their coloured SVGs, or change the cards' colour descriptions.
- [ ] Correct the Readout Guidance card's sentence about which optimizer `rg_numbers.json` names.
- [ ] Measure how often the cap fires on capped renders, from the logs on `/datasets`.
- [ ] Test whether the different time inputs at training and at guidance change a render.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
