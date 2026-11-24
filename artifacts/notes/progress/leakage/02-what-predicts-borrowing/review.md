# Review: what predicts which pairs borrow parts under CO3? (step Z2)

**None of the three readings predicts which pairs borrow, in any window.** All seven tests in [the design](design.md) are null against their bar of a Spearman $\rho$ of at least 0.706 between the reading and W2d's borrowed count over the 12 tuning pairs. The design's "what would surprise" case happened: crow × raven, which borrows on 0 of 8 seeds, reads highest of the 12 pairs on all three readings. The readings follow how alike the two subjects are; borrowing does not.

## Runs

| Rung | Slurm job | W&B | Pictures | Outcome |
|---|---|---|---|---|
| Smoke, raw cut | 94366 | `lmm75rgv` | 1 | every overlap read 1.0; the cut was fixed (design, "The three predictors") |
| Smoke, rescaled cut and the capture-off render | 94384 | `pn7tsanv` | 1 | capture on against off on one card: 0.0 of 255 |
| [Small read](small-read/README.md) | 94534 | `27u2m8sh` | 16 | eagle × fox against cheetah × cougar |
| Full | 94591 | `llct9xp0` | 96 | the seven tests below |

Code snapshot `44b51af4` for the small and full rungs. Records under `/datasets/mmolefe/poe_repair_min/loop-12/z2/`. The tests are computed by `2-classifier-guidance/scripts/leakage/z2_analyse.py`, committed before the full run landed, with the bars in `z2_bars.py`. Median 49.6 seconds per picture on the card used; about 1.8 GPU-hours across the four rungs (114 pictures and four model loads).

## The seven tests

Spearman $\rho$ between each reading's per-pair mean (over 8 seeds and the window's read steps) and the borrowed count; one-sided permutation p from 100,000 permutations; 95% interval from 10,000 bootstrap resamples of the pairs.

| Reading | Window | $\rho$ | One-sided p | 95% interval | Call |
|---|---|---|---|---|---|
| Text similarity of the two nouns inside the joint prompt | none | +0.137 | 0.341 | −0.51 to 0.75 | null |
| Attention overlap | early, steps 0 to 5 | −0.284 | 0.824 | −0.85 to 0.37 | null |
| | middle, steps 10 to 20 | +0.228 | 0.239 | −0.43 to 0.78 | null |
| | late, steps 25 to 45 | +0.489 | 0.056 | −0.17 to 0.86 | null |
| Score-map similarity, cosine of $d_t(A)$ and $d_t(B)$ | early | +0.228 | 0.241 | −0.47 to 0.91 | null |
| | middle | −0.137 | 0.675 | −0.72 to 0.54 | null |
| | late | −0.432 | 0.923 | −0.84 to 0.23 | null |

The bar was one-sided p below 0.05 / 7 = 0.0071. No test is called wrong way. Without hippopotamus × swallow, which never shows two subjects, the largest $\rho$ is 0.409 (late overlap). Against the borrowed share of two-subject pictures it is 0.391 (late overlap).

## Per pair

Means over 8 seeds. The text reading has no step.

| Pair | Borrowed of 8 | Text | Overlap early | Overlap middle | Overlap late | Score maps early | Score maps middle | Score maps late |
|---|---|---|---|---|---|---|---|---|
| cheetah × cougar | 7 | 0.25 | 0.71 | 0.83 | 0.75 | 0.65 | 0.06 | −0.07 |
| duck × goose | 7 | 0.43 | 0.61 | 0.61 | 0.71 | 0.84 | 0.39 | 0.38 |
| puma × condor | 7 | 0.14 | 0.44 | 0.50 | 0.58 | −0.25 | −0.09 | −0.15 |
| boar × pheasant | 4 | 0.21 | 0.55 | 0.63 | 0.62 | 0.31 | 0.02 | 0.02 |
| bobcat × quail | 4 | 0.17 | 0.51 | 0.82 | 0.82 | 0.49 | 0.12 | −0.08 |
| seal × gull | 4 | 0.28 | 0.48 | 0.89 | 0.89 | 0.41 | −0.07 | −0.15 |
| **crow × raven** | **0** | **0.52** | **0.80** | **0.92** | **0.89** | **0.91** | **0.82** | **0.73** |
| giraffe × tortoise | 0 | 0.25 | 0.61 | 0.57 | 0.57 | 0.15 | −0.04 | −0.05 |
| hippopotamus × swallow | 0 | 0.07 | 0.73 | 0.39 | 0.26 | 0.16 | −0.09 | 0.00 |
| lighthouse × sailboat | 0 | 0.17 | 0.41 | 0.49 | 0.55 | 0.43 | 0.10 | 0.26 |
| zebra × flamingo | 0 | 0.26 | 0.56 | 0.50 | 0.36 | 0.01 | 0.07 | 0.06 |
| eagle × fox | 0 | 0.16 | 0.66 | 0.69 | 0.41 | 0.25 | 0.08 | 0.05 |

**Crow × raven and puma × condor break every reading.** Crow and raven are the most alike pair by every reading and never borrow; puma and condor are among the least alike and borrow on 7 of 8 seeds. The two-pair contrast in the small read (eagle × fox against cheetah × cougar) separated on early score maps and late overlap, and that separation does not hold across the 12 pairs.

## The sharper test, within pairs

The three pairs that hold both labels show no consistent difference between the seeds that borrowed and the seeds that kept their own parts. Late overlap: bobcat × quail 0.816 borrowed against 0.839 own parts; seal × gull 0.912 against 0.868; puma × condor 0.578 against 0.574. Early score maps: 0.469 against 0.486; 0.378 against 0.454; −0.288 against −0.027. Twenty-two pictures carry no bar.

## Secondary readings, reported only

Of 20 secondary readings, the largest is the overlap between the second subject's map and the end-of-text token's map in the early window, $\rho$ = 0.740; the first subject's overlap with the end-of-text token in the late window gives 0.603. These were not tested and 20 were looked at, so the largest is expected to be high by chance. Magnet and Bounded Attention both name the end-of-text and padding tokens as a route for leakage ([Magnet's note](../01-reading/magnet.md), [Bounded Attention's note](../01-reading/bounded-attention.md) p. 14), so this reading is the one candidate for a pre-registered follow-up on new pairs (inferred).

## Controls

**The capture leaves the picture unchanged:** 0.0 of 255 between capture on and off on one card (smoke rung).

**Pictures over 10 of 255 from M1's were checked by eye:** 8 of 96 (eagle × fox seeds 6002 to 6007, puma × condor seed 6004, giraffe × tortoise seed 6006). Each matches M1's composition and keeps its W2d call; all 96 readings are used.

**Each noun's map overlaps itself at 1.0** on every picture, and CO3 parsed every prompt into its two subject nouns.

## What this means for Z3

The design's rule for this case: "If none has support, Z3 keeps its default and the review says the place is unknown." **The place is unknown.** No reading at the text encoder, the attention layers or the sum of predictions, at any of the three windows, separates the pairs that borrow from those that do not.

Three findings from this loop bear on Z3's choice and are for the owner to weigh (inferred):

- Borrowing does not follow how alike the two subjects are, so a fix that keeps similar subjects apart (Bounded Attention, Conform) is not aimed at a measured cause.
- On cheetah × cougar at seed 6001, plain SDXL draws two cheetahs and CO3's push away from the cheetah at steps 4 and 5 leaves a cougar-shaped animal with cheetah spots ([the capture note](../co3-inside/design.md)). On that seed the borrowing looks like an unfinished change of animal, a place-by-place failure that CO3's one weight per subject cannot address.
- [Loop 4's bar](../04-paper-2-method.md) is CO3 alone on the same pictures, and Bounded Attention, the default fix, is also the rival Paper 2 must beat.

## What this cannot say

Twelve pairs are few, and the 95% intervals span more than a unit of $\rho$ for every test, so a modest relationship cannot be ruled out. The outcome is W2d's labels of one picture per seed. The readings are means over a window; a reading that predicts borrowing only at one step, or only in one region of the canvas, would be averaged away.
