# How much co-occurrence the product needs

Three figures and two count files, asking one question on an exact toy: how many training images
holding *both* concepts does a product of experts need before it prefers a picture holding both?

The toy is `artifacts/notes/what-poe-loses-when-it-composes/toy_sampler_hybrids.py`. Each expert is
an exact mixture of Gaussians, spread σ = 0.1, over 16 × 32 RGB pictures of red circles and blue
squares. Every composed density and score is closed form, so a sampler's draws can be set against
the distribution it was aiming at rather than against another sampler. Every draw is labelled by its
nearest picture in a fixed library of the single shapes, the side-by-side pair, and every
circle-with-square pixel average, so an overlay is named as one. 300 draws per cell throughout.

## The answer

**None that a grid can measure.** The exact product goes from 0 crisp pairs of 300 at zero
co-occurrence to 300 of 300 at a pair-image weight of 0.001, which is the smallest non-zero weight
the sweep used. A closed-form curve over 400 points down to 1e-5 is already above half at its own
smallest point, so 0.001 is the grid's floor and not a crossing.

The reason is in the product's own arithmetic. A product of two Gaussian mixtures is a mixture over
every pairing of one component from each side, and the pairing of components at means `mi` and `mj`
carries weight `exp(−‖mi − mj‖² / 4σ²)`. The pair image paired with itself has distance zero and
pays nothing. A circle paired with a square has `‖mi − mj‖² = 104.4`, so at σ = 0.1 it pays
`exp(−2610)`.

**Its two samplers need about half the mass.** Reverse diffusion and annealed MALA both cross half
of 300 between a pair weight of 0.3 and 0.5. That is the finding: the gap between the co-occurrence
a product needs to prefer the composition and the co-occurrence its samplers need to find it spans
about three orders of magnitude in that weight.

## Files

| File | What it shows |
|---|---|
| `how-far-the-dial-has-to-go.png` | the first dial over its whole range, 0.001 to 0.99 (x, log scale), against the fraction of 300 draws labelled a crisp pair (y), one curve per sampler, with the band past 0.5 where each expert keeps less than half its mass on its own shapes and the weight at which both samplers first reach 300 of 300 |
| `the-leak-dial.png` | the second dial: each expert's weight on each one of the other's three shapes (x, log scale, 1e-4 to 0.2) against the share of the exact product that is a clean pair and the share that is a single shape (y, 0 to 1), with the four sampled settings as markers and the half-share crossing at 0.00567 marked. The pair image's weight is held at 0.1 throughout |
| `four-samplers-across-the-overlap-dial.png` | the same four situations, now with four rows: the exact product, plain reverse diffusion, the Langevin corrector and a Hamiltonian corrector, each with its crisp-pair count and the corrector's acceptance rate |
| `can-it-make-a-third-arrangement.png` | both experts hold two pair layouts (circle left and circle right, each at half the pair weight of 0.1); the counts say which of four layouts each draw lands nearest, two of them trained on and two never seen |
| `what-changes-as-overlap-rises.png` | four situations across (no overlap, one picture in a thousand, one in ten, and the overlap three competing captions imply), two rows down (what the exact product calls most likely, and what reverse diffusion draws). Each cell carries four draws, the label most of its 300 were given, and a green or red tint for whether that label is a clean pair |
| `the-chain-of-results.png` | the four earlier results in date order ending on this sweep, one row each: the year and the claim, the paper's own figure where one is on disk, what it showed and what it left open. Liu et al. 2022 and Bradley 2025b have no figure fetched, so their panels state the result in words rather than standing in a picture |
| `weight-sweep.png` | the fraction of 300 draws labelled a crisp pair (y, 0 to 1) against the pair image's weight inside each expert (x, log scale, 0.001 to 0.5), one curve per sampler, with condition B's weight of 0.1 marked and the grid's floor at 0.001 labelled as a floor |
| `draws-across-weights.png` | eight draws per sampler at pair weights 0.003, 0.03, 0.1, 0.3 and 0.5, fifteen rows, each with its crisp count out of 300 |
| `competing-captions.png` | eight draws per sampler for both competing-caption forms and for the hand-weighted condition re-run at each form's implied weight, twelve rows with counts |
| `../../notes/what-poe-loses-when-it-composes/toy-sampler-cooccurrence-sweep.json` | the five category counts per weight per sampler, the MALA acceptance rate per weight, and the closed-form category shares over the 400-point grid |
| `toy-caption-competition.json` | the same counts for the four competing-caption conditions, both forms' implied pair weights, and each form's implied weight as a function of its softmax temperature |
| `what-the-caption-model-draws.png` | one VAE plus one caption-conditional latent model trained per dial setting (0, 0.1, 0.3, 0.6, 1: how often a both-shapes picture is also captioned with each single name), four rows per setting (each single caption, the two single captions multiplied, the joint caption), eight draws each titled with the colour-blob scorer's call and the 300-draw counts beside them |
| `toy-caption-dial-ldm.json` | those counts from both scorers, each setting's VAE reconstruction error, and the colour-blob scorer's hit rate on that VAE's reconstructions of every known picture |
| `toy-trained-experts.json` | two separately trained score networks combined by summed scores, crisp pairs and single-shape purity out of 300 at six shared-picture fractions |
| `can-the-product-draw-a-layout-nobody-trained-on.png` | white background (top four rows) and grey background (bottom four): the joint caption trained on two layouts, then circle x square / background drawn exactly, by reverse diffusion with its true noised score, and by reverse diffusion with summed noised scores; eight draws per row titled with their layout, NEW marking the four layouts no training picture holds, and counts of 300 |
| `toy-factorized.json` | those counts per layout, the composition's pairing weights on each background, and the scorer's hit rate on known pictures on each background |
| `summed-scores-against-the-true-noised-product.png` | left: the share of the mixture on the pair image paired with itself (y, 0 to 1) against the reverse-diffusion step (x, 200 at pure noise to 0 at the clean picture), one blue curve per pair weight for the summed noised expert scores and one green line for the true noised product; right: eight reverse-diffusion draws per pair weight with each score, and the crisp-pair count of 300 |
| `toy-noised-product.json` | those shares by step and the five category counts for both scores at pair weights 0.001, 0.01, 0.1 and 0.3 |
| `caption-model-across-guidance.png` | the two single captions multiplied in the caption model, dial setting down (0 to 1) and guidance strength across (1, 2, 3, 5, 7), four draws per cell titled with the colour-blob scorer's call, the cell's crisp-pair count out of 300 and the lower of the two single captions' purity counts |
| `toy-caption-dial-ldm-guidance.json` | those counts for every caption at every dial and guidance strength, one training per dial shared across strengths |
| `toy-caption-threshold.json` | the caption model trained with exactly 0, 1, 5, 20, 50 or 200 both-shapes pictures also given each single caption, three trainings each: PoE's separate-shapes count of 300, both single captions' own-colour counts and the joint caption's count per training, with the rule written before the run. Up to 50 no training draws the pair; at 200 one of three does (153, 1 and 0) |
| `toy-caption-threshold-600-1200-2000.json` | the same exact-count sweep at 600, 1,200 and 2,000 pictures, three trainings each: separate-pair counts 157, 17 and 134; 183, 29 and 57; 252, 74 and 238 of 300 |
| `toy-caption-corrector.json` | a Langevin corrector (5 or 20 steps per noise level, step 0.05 of the noise variance) on PoE in four retrained caption models with the single captions (2,000 pictures at three seeds, 200 at one): median separate-pair count 260.5 of 300 plain, 300 with 5 steps, 300 with 20; the rule written before the run is met |
| `toy-caption-corrector-no-single-captions.json` | the same corrector on three 0-picture models and one 50-picture model, added after the first result: 0 of 300 separate in every case, the samples tidied into purple hybrids |
| `plain-figures/` | the supervisor write-up (its `README.md`) and its ten simple figures, drawn by `plain_figures.py` from the counts and samples in this folder |

## The sweep, weight by weight

Crisp pairs out of 300. The pair image's weight inside each expert is `w`; the remaining `1 − w` is
split equally over that expert's three single shapes.

| w | exact | reverse diffusion | annealed MALA |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 0.001 | 300 | 0 | 0 |
| 0.003 | 300 | 0 | 0 |
| 0.01 | 300 | 0 | 0 |
| 0.03 | 300 | 1 | 1 |
| 0.1 | 300 | 10 | 30 |
| 0.3 | 300 | 122 | 140 |
| 0.5 | 300 | 238 | 249 |

At `w = 0` the pair component is dropped rather than given `log(0)`, which makes the condition
identical to the singles-only product and reproduces its counts exactly. At `w = 0.1` the density is
identical to the pair-captioned product's, and the exact sampler agrees at 300 of 300 while the two
sampled ones give 10 and 30 against that run's 14. One density at one weight produces crisp counts
of 10, 14 and 30 across three runs, so the sampled curves carry draw noise of about that width and
the threshold is read off the exact curve.

Annealing buys 140 against 122 at `w = 0.3`, and 30 against 10 at `w = 0.1`.

## How far the dial has to go

Carried past 0.5, crisp pairs out of 300:

| w | mass each expert keeps on its own shapes | exact | reverse diffusion | annealed MALA |
|---|---|---|---|---|
| 0.5 | 0.50 | 300 | 238 | 249 |
| 0.65 | 0.35 | 300 | 281 | 286 |
| 0.8 | 0.20 | 300 | 298 | 299 |
| 0.9 | 0.10 | 300 | 300 | 300 |
| 0.95 | 0.05 | 300 | 300 | 300 |
| 0.99 | 0.01 | 300 | 300 | 300 |

**Both samplers reach 300 of 300 at a pair weight of 0.9, where each expert has a tenth of its mass
left on its own single shapes.** The weight that makes the sampler reliable is the weight at which
the two experts have stopped being a circle expert and a square expert and have become two copies of
the same thing. As a recipe that reads: a product of experts draws the composition dependably once
the experts are no longer distinct, which is the point at which there is nothing left to compose.

## Competing captions

Three captions ("a circle", "a square", "a circle and a square") score every picture in a bank of
seven (three circles, three squares, one pair). A softmax over the captions gives p(caption|picture),
and Bayes against a uniform picture prior turns each caption into an expert. The two single-caption
experts are then multiplied.

Two defensible ways to score a caption against a picture, both run:

**Form 1, distance.** The caption scores a picture by log-sum-exp of `−‖x − m‖² / 2τ²` over the
pictures that caption describes. An extra object costs its pixels. Default τ = 5.

**Form 2, shape counting.** The caption scores a picture by (named shapes present) − (named shapes
missing) − (present shapes the caption does not name), divided by τ. An extra object costs one unit.
Default τ = 1.

Both imply almost the same pair weight. The joint caption takes 0.700 of the posterior on the pair
picture under form 1 and 0.787 under form 2, leaving the pair picture a weight of 0.0461 inside the
"a circle" expert under form 1 and 0.0458 under form 2.

Counts out of 300, with the hand-weighted condition re-run at each implied weight so that the only
thing differing between the two halves of the table is whether the captions competed:

| condition | sampler | one circle | one square | crisp pair | two faint | overlaid |
|---|---|---|---|---|---|---|
| competing captions, form 1, τ = 5 | exact | 146 | 150 | 4 | 0 | 0 |
| competing captions, form 1, τ = 5 | reverse diffusion | 144 | 154 | 2 | 0 | 0 |
| competing captions, form 1, τ = 5 | annealed MALA | 143 | 155 | 2 | 0 | 0 |
| competing captions, form 2, τ = 1 | exact | 137 | 144 | 19 | 0 | 0 |
| competing captions, form 2, τ = 1 | reverse diffusion | 148 | 150 | 2 | 0 | 0 |
| competing captions, form 2, τ = 1 | annealed MALA | 149 | 150 | 1 | 0 | 0 |
| hand-weighted at w = 0.0461 | exact | 0 | 0 | 300 | 0 | 0 |
| hand-weighted at w = 0.0461 | reverse diffusion | 0 | 0 | 2 | 2 | 296 |
| hand-weighted at w = 0.0461 | annealed MALA | 0 | 0 | 4 | 0 | 296 |
| hand-weighted at w = 0.0458 | exact | 0 | 0 | 300 | 0 | 0 |
| hand-weighted at w = 0.0458 | reverse diffusion | 0 | 0 | 2 | 2 | 296 |
| hand-weighted at w = 0.0458 | annealed MALA | 0 | 0 | 4 | 0 | 296 |

**The competing-caption product does not hybridise where the hand-weighted one does. It produces one
object.** At the same pair weight the hand-weighted product's exact sampler is right 300 times out of
300 and its two sampled ones produce 296 overlays each. The competing-caption product's exact sampler
produces 4 or 19 crisp pairs and about 296 single objects, and no overlay at all. The failure moved
from the sampler to the target.

The mechanism is leakage. Both experts are now mixtures over the same seven pictures, so a picture
paired with itself survives in the product whatever its weights, while every pairing of unlike
pictures still pays `exp(−2610)`. The "a circle" expert gives a lone square a weight of 0.107 under
form 1 and 0.015 under form 2, and the "a square" expert gives that same picture about 0.205. Their
product beats the pair picture's `0.046 × 0.052`, so the surviving components are the single shapes
and the product behaves like a mixture.

## The second dial: how much the experts may disagree

The sweep above turns up how much the two experts agree. This one holds that agreement at a pair
weight of 0.1, where the exact product gives 300 of 300, and turns up how much they disagree: the
circle expert is given weight `leak` on each of the three single squares and the square expert the
same on each of the three single circles.

**A leak of 0.00567 destroys an agreement of 0.1.** That is the point where the exact product's crisp
share falls below half, computed in closed form over a 240-point grid. Each expert needs to put only
half a percent of its mass on each of the other's shapes to undo an overlap twenty times its size.

| leak | exact: crisp | exact: one shape | reverse diffusion: crisp |
|---|---|---|---|
| 0 | 300 | 0 | 10 |
| 0.003 | 201 | 99 | 9 |
| 0.01 | 113 | 187 | 9 |
| 0.1 | 20 | 280 | 8 |

The arithmetic behind it: the pair image paired with itself carries `w²`, so 0.01, while each of the
six single shapes paired with itself carries the leaking expert's `leak` times the owning expert's
own weight of about 0.3. Six of those pass one of the pair at a leak a little over half a percent,
and nothing else survives, because every pairing of unlike pictures still pays `exp(−2610)`.

This is the axis the competing-caption condition landed on without anyone choosing it. Its "a circle"
expert gives a lone square 0.107 under form 1 and 0.015 under form 2, both far above 0.00567, which is
why that condition produces single objects.

**What it means for a real model.** A text-conditioned "a cat" expert assigns non-zero probability to
pictures of dogs. The question this dial poses is whether that quantity is above or below the toy's
threshold, which is a measurement on the real model with its reading written in advance.

## Where the samplers lose the pair

**The loss is in adding noised scores, and reverse diffusion itself is fine.** On condition D the
product at step 0 is a Gaussian mixture over pairings, so the true score of the noised product has a
closed form at every step. Reverse diffusion with that true score draws 300 crisp pairs of 300 at pair
weights 0.001, 0.01, 0.1 and 0.3. The same sampler with the two experts' noised scores added draws 0,
0, 15 and 109 of 300.

The sum is the score of the product of the two noised experts, whose pairing weights use the noised
distance. At pure noise that distance penalty vanishes, so the pair image's self-pairing holds w² of
the mixture (0.0001 at w = 0.01) where the true noised product holds 1.000 at every step. The summed
score reaches 1.000 only after step 95 to 160, depending on w, by which time the draw has already
committed to one position and lands on a single purple hybrid there.

## A layout nobody trained on

**Divided by the background, the product composes with no co-occurrence at all, but only on grey.**
Here the experts are a circle at any of three positions, a square at any of three, and the plain
background, with no both-shapes picture in any of them. Composed as circle x square / background,
which is e_circle + e_square − e_background at guidance 1, the step-0 target is a mixture over
pairings with mean m_circle + m_square − m_background and weight exp(⟨m_circle − m_b, m_square − m_b⟩ / σ²).
Shapes at different positions share no pixel, so that dot product is zero and all six apart layouts
weigh the same.

On a grey background the exact draws land on the four layouts no training picture holds 202 of 300
times, against 0 of 300 for a joint caption trained on the other two. Reverse diffusion gives 197 of
300 with the true noised score and 210 of 300 with plain summed noised scores, so here the sampler
loses nothing. This meets the bar written before the run.

On white, red and blue are both darker than the background in every channel, so the dot product on
the 36 overlapping pixels is positive and the same-position pairing takes the whole mixture. Every
exact draw is the two shapes stacked and subtracted to near-black, 0 of 300 on unseen layouts.

The summed scores lose nothing on grey because the noise only rescales a pairing's weight
exponent by abar / (abar σ² + 1 − abar), and a weight exponent of zero stays zero. In the
co-occurrence sweep the exponents were large and negative at step 0 and the noise flattened them
toward zero, which is where the pair image's share fell to w².

## Trained models

**Two separately trained networks never compose.** Each is a small network trained on its own shapes,
with a shared fraction of both-shapes pictures in both training sets, combined by summing their
scores and drawn by reverse diffusion. Crisp pairs are 0 of 300 at every fraction from 0 to 0.5, and
the networks blur into each other first: single-shape purity falls from 300 of 300 to 151 of 300 at a
fraction of 0.5. Numbers in `toy-trained-experts.json`.

**One caption-conditional latent model, built the way SDXL is, composes partly once pair pictures
carry the single captions.** A VAE compresses each 16 × 32 picture to 128 numbers, one network
denoises those under a caption, and the two single captions are combined as summed
classifier-free guidance at 3.0 and drawn by reverse diffusion. Counts are the colour-blob scorer's
"red and blue apart" out of 300, for the two single captions multiplied:

| Dial (pair pictures also under each single name) | 0 | 0.1 | 0.3 | 0.6 | 1 |
|---|---|---|---|---|---|
| red circle and blue square apart | 0 | 118 | 114 | 55 | 174 |
| purple or overlaid | 22 | 146 | 186 | 230 | 122 |
| nothing readable (a faint smear) | 278 | 36 | 0 | 15 | 4 |
| "a red circle" alone draws only red | 300 | 300 | 299 | 285 | 250 |
| "a blue square" alone draws only blue | 300 | 300 | 298 | 295 | 234 |
| the joint caption draws the pair apart | 299 | 300 | 300 | 300 | 300 |

At dial 0 the product draws a faint grey-blue smear over all three positions, the purple-hybrid
failure seen on SDXL. Any overlap at all moves it up: at 0.1, 118 of 300 draws are a crisp circle
beside a crisp square, and most of the rest are two purple blobs at the two pair positions, two
things in frame with their colours mixed. The rate does not climb with the dial between 0.1 and 0.6,
and at 1 the square caption itself draws the pair on 66 of 300, so its purity fails the 240 test.

**The result falls between the bar's outcomes.** Support needed 150 of 300 apart while both
captions still pass 240 of 300; the best such setting reaches 118. Null needed under 30 everywhere
those captions pass; three settings exceed it. The written inconclusive case was a rise only where
purity had already fallen, and this rise happens where purity holds, so none of the three describes
it. The script prints "inconclusive" for anything between support and null. Every setting's VAE reconstructs
at a mean squared error of 0.010 to 0.0103 and the scorer reads that VAE's reconstructions of every
known picture right 100 times in 100. The template library's counts are kept beside the scorer's in
the JSON and are not used, because it calls the joint caption's crisp pairs "two faint objects".

**Guidance strength does not rescue dial 0, and it moves the rest around without a trend.** The same
models drawn at guidance 1, 2, 3, 5 and 7: at dial 0 the product gives at most 1 crisp pair of 300 at
any strength, a single purple blob at guidance 1, 5 and 7 and a faint smear at 2 and 3. At every
non-zero dial it gives between 1 and 264 of 300, with no strength best across dials. Two cells reach
150 of 300 with both captions above 240 (dial 0.1 at guidance 1: 179, purity 274; dial 0.3 at
guidance 7: 180, purity 271), but they were found by scanning 25 cells under a bar written for
guidance 3 alone, and retraining the same dial 0.1 model at guidance 3 gave 118 in one run and 149
in the next, so a single cell moves by about 30 of 300 between trainings.

**The "purple or overlaid" call covers two different pictures.** At dial 0 it is one purple blob at
one position, a true hybrid. At non-zero dials it is mostly a bluish square beside a reddish circle,
two separate shapes whose colours have drifted toward each other. The scorer counts the second kind
as not composed, so the crisp counts understate how often two things sit in frame.

## What this cannot claim

**The product returns arrangements it was trained on and no others.** With two pair layouts inside
both experts, the circle on the left in one and on the right in the other, each at half the pair
weight of 0.1, the exact product splits 145 and 155 of 300 between those two and puts 0 of 300 within
a squared distance of 30 of either of two layouts nobody trained on. An exact draw sits about 7.6
from its own layout, so 0 is not a thresholding artefact. The sweep measures when the product prefers
a two-object picture; it does not show the product arranging two objects.

**Momentum does not rescue the sampler.** At a pair weight of 0.1 the plain sampler gives 10 crisp
pairs of 300, the Langevin corrector 30 and a Hamiltonian corrector 31, the last at four times the
gradient cost and with 93% of its moves accepted against the Langevin corrector's 38%. A corrector
treble the plain rate and leaves nine draws in ten wrong. A single tuning run at 100 draws read 17,
which scaled to 51 of 300 and was wrong; the 300-draw figure is 31.

**Every crisp pair is the one picture both experts were given, not a new arrangement.** The
side-by-side pair holds the circle on the left at x = 10 and the square on the right at x = 22, and
that is the only two-object picture in either expert. Of 300 exact draws at a pair weight of 0.001
and again at 0.1, all 300 sit nearer that layout than its mirror image, median squared distance 7.63
against 97.21, and the mirrored layout appears nowhere in the labelling library. So the product is
retrieving the picture its two experts agree on rather than arranging two objects it has only seen
apart. The sweep answers how much co-occurrence it takes before the product prefers a two-object
picture. It does not show the product generalising to an arrangement nobody trained on, and a test
of that needs the pair pictures to differ from the one being asked for.


**The caption model's support is a modelling choice nobody has settled.** Reading each expert off a
posterior over the whole picture bank gives every expert non-zero weight on every picture. Restricting
each expert to the pictures its caption describes would remove the leakage and collapse the
competing-caption condition back onto the hand-weighted one. Which of the two models a text encoder
better is not something this toy can answer.

**The corrector is Metropolis-adjusted Langevin, not Hamiltonian Monte Carlo.** Five steps per reverse
step at a step size of 0.05 of the diffused variance. Du et al.'s own demonstration uses HMC, so these
counts are not a reproduction of their method and the annealed column should not be read as a verdict
on it.

**Two shapes, one pair image, one noise level.** σ = 0.1 fixes the penalty exponent at 2610. A larger
spread would soften it and move every number in this card, and nothing here says where a real model's
equivalent sits.

**A `divide by zero in log` warning fires whenever a mixture is built with a component of weight
exactly zero,** which the leak sweep does at `leak = 0`. The component is then correctly dropped. The
same warning appeared once in the competing-caption run, where no weight in any of the four conditions
is zero and the warning does not reproduce under warnings-raised-as-errors; that instance is
unexplained, and nothing was dropped there.

## How it was made

```bash
cd /home/molef/PhD/poe_repair_min-dog-and-rain/artifacts/notes/what-poe-loses-when-it-composes
/home/molef/miniconda3/bin/python3 -I toy_cooccurrence_sweep.py    # 8 weights, about 9 minutes
/home/molef/miniconda3/bin/python3 -I toy_cooccurrence_figure.py   # weight-sweep.png
/home/molef/miniconda3/bin/python3 -I toy_cooccurrence_sheet.py    # draws-across-weights.png
/home/molef/miniconda3/bin/python3 -I toy_caption_competition.py   # competing-captions.png
/home/molef/miniconda3/bin/python3 -I toy_chain_of_results.py      # the-chain-of-results.png
/home/molef/miniconda3/bin/python3 -I toy_comparison_grid.py       # four-samplers-across-the-overlap-dial.png
/home/molef/miniconda3/bin/python3 -I toy_leak_sweep.py            # toy-leak-sweep.json, about 1 minute
/home/molef/miniconda3/bin/python3 -I toy_leak_figure.py           # the-leak-dial.png
/home/molef/miniconda3/bin/python3 -I toy_layout_generalisation.py # can-it-make-a-third-arrangement.png
/home/molef/miniconda3/bin/python3 -I toy_high_overlap.py          # toy-high-overlap-sweep.json, about 6 minutes
/home/molef/miniconda3/bin/python3 -I toy_full_range_figure.py     # how-far-the-dial-has-to-go.png
```

One condition on its own, at a chosen weight or caption form:

```bash
/home/molef/miniconda3/bin/python3 -I toy_sampler_hybrids.py --only D --pair-weight 0.03 --tag=-mine
/home/molef/miniconda3/bin/python3 -I toy_sampler_hybrids.py --only E --caption-form 2 --tag=-mine
```

The random stream is reset to seed 20261022 at every weight and every condition, so draw noise is
shared along each curve and the zero-weight point reproduces the singles-only product exactly.
