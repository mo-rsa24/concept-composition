# 🎯 How much co-occurrence does the product need?
<!-- template: exempt. Reason: a one-sitting CPU walk on the existing toy, not a plan -->

> [!NOTE]
> **You will hold:** the weight the side-by-side pair image needs inside both experts before the exact
> product stops returning hybrids and starts returning crisp pairs, as a number, with the same count for
> the two samplers that are actually used; and whether a competing-caption model, where that weight is
> derived rather than set, lands above or below it. · **Time:** ~2 h, CPU only ·
> **Stop point:** the card states the threshold weight as a number and the competing-caption counts out
> of 300

Walk it with:

```text
/drip-execute-plan plans/10-what-poe-loses-when-it-composes/today-how-much-co-occurrence-the-product-needs.md
```

## Contents

- [Why this exists](#why-this-exists)
- [The test, written before any run](#the-test-written-before-any-run)
- [Where things land](#where-things-land)
- [Steps](#steps)
- [What it cannot claim](#what-it-cannot-claim)

## Why this exists

Navigation: 📋 [Contents](#contents) | [Next](#the-test-written-before-any-run) ➡️

**Condition B already showed that the product's target can be right.** Putting the side-by-side pair
image inside both experts moves the exact product's peak onto the pair: 300 of 300 crisp pairs from the
exact sampler, against 300 of 300 overlaid hybrids in condition A where each expert holds only its own
shape ([the counts](../../artifacts/notes/what-poe-loses-when-it-composes/toy-sampler-hybrids.json)).

**What B does not say is how little of the pair image is enough.** B gives the pair image weight 0.1
inside each expert, one component in four. A caption set scraped from the web gives a two-animal
photograph far less than that: a photo of a cat and a dog is captioned "a cat and a dog", and the
captions "a cat" and "a dog" each reach it rarely. So the useful question is where between zero and 0.1
the product's answer changes, and whether a realistic caption share clears it.

**The hand-set weight also misses the thing that makes captions hard.** Captions compete. A photo
holding both shapes is most likely to be captioned with the name for both, which takes probability away
from each single name. A mixture with weights chosen by hand cannot represent that competition, so the
second half of this walk derives the weight from three captions instead of setting it.

## The test, written before any run

Navigation: ⬅️ [Previous](#why-this-exists) | 📋 [Contents](#contents) | [Next](#where-things-land) ➡️

**The threshold.** The smallest pair-image weight w in `{0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5}` at
which the exact sampler's "two objects, crisp" count exceeds 150 of 300. Its closed-form companion is the
w at which the exact product's pair-with-itself component first holds more than half the total mixture
weight, computed from the component weights rather than sampled.

| Result | Reading |
|---|---|
| the grid threshold is 0.003 or above, and the closed form agrees within one grid step | **the product needs a measurable share**: the number is a real threshold and a caption share can be compared against it |
| the grid threshold is 0.001, the smallest non-zero point | **the grid bottoms out**: the closed form carries the answer and the grid only bounds it from above |
| the closed form and the grid disagree by more than one grid step | **the sampled and exact readings disagree**: report both and say which the figure draws |
| the exact count never exceeds 150 at any w | **no threshold in range**: the pair needs more weight than B gives it, and B's result needs re-checking |

**The two competing-caption forms.** Both define three captions ("circle", "square", "circle and
square") and read each expert off the caption posterior by Bayes against a uniform prior over the seven
library images. They differ in how a caption's fit to an image is scored.

| Form | What the score counts | What it predicts |
|---|---|---|
| names present | how many of the caption's named shapes appear in the image | the pair image keeps appreciable weight inside each single-name expert |
| names present, omissions penalised | the same, minus the shapes the image holds that the caption does not name | the pair image's weight inside each single-name expert is crushed |

| Result | Reading |
|---|---|
| a form's implied pair weight is above the threshold and its exact count is above 150 | **competing captions clear the bar**: competition alone does not break the product |
| a form's implied pair weight is below the threshold and its exact count is at or below 150 | **competition is enough to break it**: the product hybridises on a caption set no hand-set weight would have predicted |
| the implied weight is above the threshold but the count is not, or the reverse | **the derived weight does not predict the outcome**: the caption posterior changes more than one weight, and the comparison is mixed |

**Every count in this plan is out of 300 draws and names the sampler that produced it.** The five
categories are the labelling library's own and are not redefined: `one circle`, `one square`,
`two objects, crisp`, `two faint objects`, `hybrid: overlaid`.

## Where things land

Navigation: ⬅️ [Previous](#the-test-written-before-any-run) | 📋 [Contents](#contents) | [Next](#steps) ➡️

| What | Where |
|---|---|
| the two new conditions | [the toy script](../../artifacts/notes/what-poe-loses-when-it-composes/toy_sampler_hybrids.py) |
| the counts at every w, all three samplers | `artifacts/notes/what-poe-loses-when-it-composes/toy-sampler-cooccurrence-sweep.json` |
| the sweep figure | `artifacts/results/how-much-co-occurrence-the-product-needs/weight-sweep.png` |
| the sample sheets | `artifacts/results/how-much-co-occurrence-the-product-needs/samples-*.png` |
| the card | `artifacts/results/how-much-co-occurrence-the-product-needs/README.md` |
| one line under Runs | [the idea map](../../artifacts/ideas/is-poe-ill-posed-for-composition/IDEA_MAP.md#runs) |

Conditions A, B and C and `toy-sampler-hybrids.json` are not touched. The new conditions run under their
own tag so the existing counts stay comparable.

## Steps

Navigation: ⬅️ [Previous](#where-things-land) | 📋 [Contents](#contents) | [Next](#what-it-cannot-claim) ➡️

Every command runs from `/home/molef/PhD/poe_repair_min-dog-and-rain/artifacts/notes/what-poe-loses-when-it-composes`
with `PY=/home/molef/miniconda3/bin/python3`. CPU only.

### 1. 🔧 The hand-weighted condition, with w as a flag (AI, ~15 min)

- [x] **Add condition D** ✅ verified (`--pair-weight`, 2026-11-04) to the toy script: the pair image inside both experts at weight w, the
  remaining 1 − w split equally across that expert's three single shapes. `--pair-weight` sets w.
- [x] **Handle w = 0** ✅ verified (the pair component is dropped, 2026-11-04) by dropping the pair component rather than taking `log(0)`.
- [x] **📸 You should see** at w = 0 the same counts condition A already holds: 300 hybrids from the
  exact sampler, 298 from reverse diffusion, 300 from annealed MALA, give or take the draw.
  ✅ verified (byte-identical to A, 2026-11-04; D at 0.1 is B's density exactly, and the exact
  sampler agrees at 300 of 300 while the two sampled ones differ by draw alone)

▶ Next: step 2.

### 2. 🔧 The sweep, and the closed-form crossing (AI, ~15 min)

- [ ] **Run D at all eight weights** for all three samplers, 300 draws each, writing the five counts per
  weight per sampler to `toy-sampler-cooccurrence-sweep.json`.
- [ ] **Compute the closed-form crossing** from the exact product's component weights, over a fine grid,
  and write it into the same file.
- [ ] **📸 You should see** the exact sampler's crisp count rising with w and the two sampled ones
  staying low, matching B's split at w = 0.1.

◀ Needs: step 1. ▶ Next: step 3.

### 3. 🔧 The figure and the threshold (AI, ~15 min)

- [ ] **Draw the sweep** as `weight-sweep.png`: w on a log x axis, the fraction of 300 draws called
  `two objects, crisp` on y, one curve per sampler, condition B's pair weight marked at 0.1, and a
  vertical line at the grid threshold with the closed-form crossing beside it.
- [ ] **Build a sample sheet** showing eight draws per sampler at three weights, so the counts can be
  checked against the pictures.

◀ Needs: step 2. ▶ Next: instruction 4.

### 4. 👀 Read the figure and say whether the threshold stands (you, ~10 min)

- [ ] **Open the figure and the sample sheet** at the paths in [Where things land](#where-things-land).
- [ ] **Check the threshold line** against the exact curve, and the pictures against the counts.
- [ ] **Read it against [the table](#the-test-written-before-any-run)** and say which row it is.

◀ Needs: step 3. ▶ Next: step 5.

### 5. 🔧 The competing-caption condition, both forms (AI, ~25 min)

- [ ] **Add condition E**: three captions, a softmax over each caption's fit to each image, each expert
  read off the posterior by Bayes against a uniform image prior, then the product of the two
  single-caption experts. Both scoring forms, the softmax temperature a flag.
- [ ] **Report the five counts** for all three samplers, and the pair weight each form implies inside
  each expert.
- [ ] **Run D at each form's implied weight** so the comparison holds the weight fixed and changes only
  whether the captions competed.
- [ ] **📸 You should see** two count sets, and for each one the implied weight placed on the sweep
  figure's x axis.

◀ Needs: instruction 4. ▶ Next: step 6.

### 6. ✍️ The card, the idea-map line and the filing (AI with you, ~20 min)

- [ ] **Write the card** at `artifacts/results/how-much-co-occurrence-the-product-needs/README.md`:
  both axes named, the data behind each curve, the threshold as a number, the competing-caption counts
  out of 300 per sampler, and what the toy cannot say about SDXL.
- [ ] **Append one line** under [the idea map's Runs](../../artifacts/ideas/is-poe-ill-posed-for-composition/IDEA_MAP.md#runs)
  giving the threshold and whether the competing-caption version hybridises.
- [ ] **Add the new figures** to the note's own README table.

◀ Needs: step 5.

## What it cannot claim

Navigation: ⬅️ [Previous](#steps) | 📋 [Contents](#contents)

**Seven pictures, not a dataset.** Each expert is a mixture over three single-shape images and the pair.
A real expert is trained on millions of photographs whose captions overlap in ways three hand-written
captions cannot represent, so the weight this walk finds is the weight *this* toy needs.

**The threshold is a property of the exact product, not of a trained model.** It falls out of the
distance between a circle image and a square image at σ = 0.1. Change the spread, the image size or how
far apart the shapes sit and the number moves. What transfers is the shape of the curve and the gap
between the exact sampler and the two that are used in practice, not the number itself.

**It says nothing about why SDXL's hybrid is a plausible animal.** The toy's hybrid is a pixel average
because the toy has no learned density to rule one out. That gap is
[the latent-averaging walk's](today-averaging-in-the-sdxl-latent.md) question, not this one's.

**A caption posterior is not a caption distribution.** Form and temperature are both chosen here, and the
implied weight moves with them. The walk reports the weight each choice implies rather than claiming a
caption share for any real corpus.
