# 📒 Report: can a classifier supply the term the product drops?

**No verdict has been recorded, because the eight-seed sweep that would judge this has no result.**
The product of a cat model and a dog model tends to draw one fused animal. [Plan 15](../../plans/15-can-a-classifier-supply-the-missing-term.md)
asks whether adding a classifier's gradient at each sampling step makes it draw two separate animals.
What exists so far is a test that can see a fused animal, and early pushed renders whose push was too
large to judge the idea.

## Table of contents

- [What would count](#what-would-count)
- [Which test can see a fused animal](#which-test-can-see-a-fused-animal)
- [The first pushed renders](#the-first-pushed-renders)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## What would count

Navigation: 📋 [TOC](#table-of-contents) | [Next](#which-test-can-see-a-fused-animal) ➡️

**The bar asks for two clear animals on at least 4 of 8 held-out seeds at some classifier weight.**
Seeds 9 to 16 are called by eye at full resolution `stated` ([the review file](../../plans/review/15-can-a-classifier-supply-the-missing-term.md#the-question-written-before-the-run)).
The damped product below is the same sum with each model's guidance held at 3.0 in place of 7.5.

| Result | Reading |
|---|---|
| at least 4 of 8 at some weight, and the damped product no higher | support |
| 0 or 1 of 8 at every weight | no support |
| 2 or 3 of 8 at the best weight, or a count above plain product and not above the damped product | inconclusive |
| the count rises while the pictures are destroyed | its own outcome, neither support nor null |

## Which test can see a fused animal

Navigation: ⬅️ [Previous](#what-would-count) | 📋 [TOC](#table-of-contents) | [Next](#the-first-pushed-renders) ➡️

**Asking "cat?" and "dog?" separately sees both animals in 8 of 14 fused renders, against 1 of 14 when
the two species compete for one answer.** The labels a classifier trains on depend on this, because a
test that names one species would teach it that a fused cat-dog is a dog.

| Setting the floor | The floor on fused renders | CLIP ranking finished pictures |
|---|---|---|
| <a href="../../artifacts/results/can-a-classifier-supply-the-missing-term/species-test-calibration-sweep.png"><img src="../../artifacts/results/can-a-classifier-supply-the-missing-term/species-test-calibration-sweep.png" width="200" alt="Four falling curves of photo pass rate against the similarity floor, with the chosen floor at 0.245 and the bars at 0.9 and 0.1"></a> | <a href="../../artifacts/results/can-a-classifier-supply-the-missing-term/hybrid-check-cosine-sheet.png"><img src="../../artifacts/results/can-a-classifier-supply-the-missing-term/hybrid-check-cosine-sheet.png" width="200" alt="Fourteen fused cat and dog renders, each titled with its two similarity scores, eight in green where both tests fire and six in red"></a> | <a href="../../artifacts/results/can-a-classifier-supply-the-missing-term/clip-pair-vs-smoosh-ranked.png"><img src="../../artifacts/results/can-a-classifier-supply-the-missing-term/clip-pair-vs-smoosh-ranked.png" width="200" alt="Two rows of eight renders ranked by score: fused single animals on top, pictures of a cat and a dog together below"></a> |
| y is the share of 200 photographs per species that pass, and x is the floor on the similarity score. One curve per test and photo set. At the chosen floor of 0.245, own-species pass rates are 0.925 and 0.950 (bar: at least 0.90) and cross-species rates are 0.070 and 0.035 (bar: at most 0.10) `stated` (the card); the picture draws that floor and both bars `verified` | One tile is one fused render with its similarity to cat and to dog. Green means both cleared 0.245: 8 of 14 `verified` (counted). The two-way test saw both on 1 of 14 `stated` (the card) | Each score is the picture's similarity to "a cat and a dog" minus its similarity to "one animal that is half cat and half dog". The 8 fused renders score -0.034 to +0.017 and the 8 joint-prompt renders +0.018 to +0.050, so every joint render ranks above every fused one `verified` (read off the picture) |

**The call can flip between two renders of one seed.** The 14 renders come from 11 seeds. Seeds 4, 5
and 42 each appear twice, and in each pair one render clears both floors and the other does not
`verified` (read off the titles).

## The first pushed renders

Navigation: ⬅️ [Previous](#which-test-can-see-a-fused-animal) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**A presence classifier's push left one animal when gentle and destroyed the picture when strong.**
These renders date from 2026-11-04, when the push was rescaled to a fraction of the whole prediction.
That made it roughly 44,000 times what the classifier asked for `stated` (`2-classifier-guidance/plans/17-classifier-guidance-from-the-paper.md`,
Quick context), so they say nothing yet about the classifier.

| One seed, seven ways | The classifier after 7,500 training steps |
|---|---|
| <a href="../../artifacts/results/can-a-classifier-supply-the-missing-term/seed9-seven-ways.png"><img src="../../artifacts/results/can-a-classifier-supply-the-missing-term/seed9-seven-ways.png" width="200" alt="Seven renders of seed 9 in a row: a fused animal, a softer fused animal, a cat beside a dog, two pushed single animals, and two broken color patterns"></a> | <a href="../../artifacts/results/can-a-classifier-supply-the-missing-term/probe-step-007500.png"><img src="../../artifacts/results/can-a-classifier-supply-the-missing-term/probe-step-007500.png" width="200" alt="Two by two grid for seeds 9 and 10: push 0.5 shows one animal, damaged on seed 10, and push 2.0 shows broken color patterns"></a> |
| Left to right: plain product, damped product, the joint prompt, then four pushes labelled very gentle to strong. Pushes 1 and 2 show one animal, and pushes 3 and 4 are broken patterns `verified`. The second and fourth pushes look the same as the seed 9 tiles beside it, so these are taken to be the same classifier `inferred` | Rows are seeds 9 and 10, and columns are push 0.5 and 2.0 as fractions of the prediction's length. Push 0.5 leaves one animal on seed 9 and a damaged one on seed 10, and push 2.0 breaks both `verified` |

## Where the sources disagree

Navigation: ⬅️ [Previous](#the-first-pushed-renders) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Three recorded facts conflict, and the code explains each.**

| What | One source | The other source | What the code shows |
|---|---|---|---|
| whether a classifier was trained and pushed | the card: "No head is trained and no render has been made through a guided sampler"; the review file lists the training run as not started | the folder holds a picture titled "the product with the presence head pushing it, after 7500 training steps" | [the trainer](../../scripts/train_presence_head.py) defaults to 7,500 steps (line 151), probes every 250 (line 64) on seeds 9 and 10 (line 65), and carried probe weights 0.5 and 2.0 until commit `8e68c451`. That this picture came from it is `inferred` |
| what the floor's unit is | the first picture's x-axis says "probability" and the second's titles say p(cat) | the card says raw cosine similarity | [the calibration script](../../scripts/presence_head_calibrate_species.py) defaults to cosine (line 168), reads it with no softmax (lines 129 to 135) and sweeps 0.150 to 0.300 (line 67). The axis label is fixed text at line 239 |
| where the sweep's numbers live | the card: `sweep.json` "beside" the picture | the folder holds no json | the script writes `sweep.json` to its own output folder on `/datasets` (line 263), so the exact rates were not re-read here |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**Whether a classifier composes**

The sweep has no result, and the pushed renders above used the oversized push.

**Whether CLIP can steer**

The ranking reads finished pictures. Reading a picture and pushing a noisy one are separate abilities.

**Whether a classifier trained on these fused animals generalises**

The fused animals come from the model being corrected, so a classifier may learn this sampler's
failure and still not know what two animals look like `stated` (the card).

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. The product, the push and the cap are taught on
[the code walk page](which-line-computes-which-part-of-guided-sampling.md#concepts-this-uses).

**Two questions that compete, against two that do not**

A picture scores 0.30 for "cat" and 0.31 for "dog". A [softmax](../../../../../Obsidian/Artificial%20Intelligence/Deep%20Learning/Softmax.md#where-to-learn-this)
makes them share one total. Multiply each score by 100 and the shares are e^30 / (e^30 + e^31) = 0.27
for cat and 0.73 for dog, so a gap of 0.01 becomes a confident "dog". A floor asks each question alone:
with a floor of 0.25, 0.30 passes and 0.31 passes, so both animals can be present.

On this project's 14 fused renders, the median [cosine similarity](../../../../../Obsidian/Mathematics/Linear%20Algebra/Cosine%20Similarity.md#where-to-learn-this)
is 0.2618 to cat and 0.2683 to dog, and both clear the 0.245 floor. A softmax turned that 0.0065 gap
into 0.003 against 0.979 `stated` (the card and lines 131 to 134 of the calibration script).

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| the counts and scores read off the pictures | the five pictures above, opened 2026-11-22 | `verified` |
| the calibration rates and the 1 of 14 | `2-classifier-guidance/artifacts/results/can-a-classifier-supply-the-missing-term/README.md` | `stated` |
| the bar | [the review file](../../plans/review/15-can-a-classifier-supply-the-missing-term.md) | `stated` |
| how the CLIP scores were made | [the CLIP comparison script](../../scripts/does_clip_tell_a_pair_from_a_smoosh.py) | `stated` |
| 38 pictures held out of the drain | the other tiles, sheets and probes in this result's folder. They repeat a picture described above, or have not been cleared against the subjects the [pair ledger](../../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md) keeps off sheets, so the folder is not linked here | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Run the eight-seed sweep with the unrescaled push, then record a verdict against the bar.
- [ ] Update the card and the review file, which say no classifier was trained.
- [ ] Bring `sweep.json` back beside its picture, or point the card at where it lives.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
