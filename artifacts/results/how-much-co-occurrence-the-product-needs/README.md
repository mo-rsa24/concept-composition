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
| `the-leak-dial.png` | the second dial: each expert's weight on each one of the other's three shapes (x, log scale, 1e-4 to 0.2) against the share of the exact product that is a clean pair and the share that is a single shape (y, 0 to 1), with the four sampled settings as markers and the half-share crossing at 0.00567 marked. The pair image's weight is held at 0.1 throughout |
| `what-changes-as-overlap-rises.png` | four situations across (no overlap, one picture in a thousand, one in ten, and the overlap three competing captions imply), two rows down (what the exact product calls most likely, and what reverse diffusion draws). Each cell carries four draws, the label most of its 300 were given, and a green or red tint for whether that label is a clean pair |
| `the-chain-of-results.png` | the four earlier results in date order ending on this sweep, one row each: the year and the claim, the paper's own figure where one is on disk, what it showed and what it left open. Liu et al. 2022 and Bradley 2025b have no figure fetched, so their panels state the result in words rather than standing in a picture |
| `weight-sweep.png` | the fraction of 300 draws labelled a crisp pair (y, 0 to 1) against the pair image's weight inside each expert (x, log scale, 0.001 to 0.5), one curve per sampler, with condition B's weight of 0.1 marked and the grid's floor at 0.001 labelled as a floor |
| `draws-across-weights.png` | eight draws per sampler at pair weights 0.003, 0.03, 0.1, 0.3 and 0.5, fifteen rows, each with its crisp count out of 300 |
| `competing-captions.png` | eight draws per sampler for both competing-caption forms and for the hand-weighted condition re-run at each form's implied weight, twelve rows with counts |
| `../../notes/what-poe-loses-when-it-composes/toy-sampler-cooccurrence-sweep.json` | the five category counts per weight per sampler, the MALA acceptance rate per weight, and the closed-form category shares over the 400-point grid |
| `toy-caption-competition.json` | the same counts for the four competing-caption conditions, both forms' implied pair weights, and each form's implied weight as a function of its softmax temperature |

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

## What this cannot claim

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
/home/molef/miniconda3/bin/python3 -I toy_comparison_grid.py       # what-changes-as-overlap-rises.png
/home/molef/miniconda3/bin/python3 -I toy_leak_sweep.py            # toy-leak-sweep.json, about 40 seconds
/home/molef/miniconda3/bin/python3 -I toy_leak_figure.py           # the-leak-dial.png
```

One condition on its own, at a chosen weight or caption form:

```bash
/home/molef/miniconda3/bin/python3 -I toy_sampler_hybrids.py --only D --pair-weight 0.03 --tag=-mine
/home/molef/miniconda3/bin/python3 -I toy_sampler_hybrids.py --only E --caption-form 2 --tag=-mine
```

The random stream is reset to seed 20261022 at every weight and every condition, so draw noise is
shared along each curve and the zero-weight point reproduces the singles-only product exactly.
