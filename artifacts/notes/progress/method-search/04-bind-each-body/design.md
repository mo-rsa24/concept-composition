# Does giving each subject its own region fix identity? (step M9)

## What this asks

CO3 drew two separate bodies in all 96 tuning pictures, and only 11 were faithful, because bodies borrow parts, take the wrong species or
gain an extra subject. This stage tests whether letting each subject's prediction act only inside its own region fixes that. Rung 1 hands
the sampler the right regions on cat and dog. Rung 2 runs only if rung 1 passes, and asks whether CO3 can find its own regions. The source
is the [binding design](../../targets/binding-each-body-design.md), approved on board question 61.

## Words this uses

**Mask.** A soft weight per cell of SDXL's 128 by 128 latent grid. The two subjects' masks sum to 1 at every cell.

**Regional sampler.** It predicts $\epsilon_u + g\,[m_A(\epsilon_A-\epsilon_u) + m_B(\epsilon_B-\epsilon_u)]$, so each subject acts mainly inside its own mask.

**Oracle boxes.** The two animals' boxes, detected on the joint prompt's own render of the same seed. No deployed method would have them.

**Clear pair.** A picture on the `two_clear` rung of the [blind count rubric](../01-beat-the-baseline/blind-count-rubric.md): exactly one clear A and one clear B as
separate bodies, with no borrowed features.

## Rung 1: do the right regions fix identity at all?

**Group.** Runs that try ideas (group 2). This rung is an upper bound and supports no claim.

**Varied axis.** Where the two subjects' regions come from and which prompt each carries. Model, seeds, sampler and guidance stay fixed.

**What runs.** The [oracle-layout plan](../../../../../3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/is-the-layout-the-missing-piece.md), on its saved boxes and masks:
SDXL with 50 DDIM steps at guidance 7.5, the masks applied from step 0 as its bar was written, the even overlap reading and no bootstrapping.
The even reading is the composer's default, because the plan's choice of reading was never recorded.

**Seeds.** 9, 12, 16, 19, 22, 25, 27 and 30, from `boxes.json`, the file the saved masks were built from. The plan's task line lists seed 20
where this file lists seed 30.

| Row | What it does | What it separates |
|---|---|---|
| Joint prompt | "a cat and a dog" | the ceiling, and the source of the boxes |
| Plain product | no regions | the floor |
| Right boxes | each subject in its own detected box | the experiment |
| Random boxes | two boxes of the same sizes, placed at random | a layout from the right layout |
| Swapped prompts | the right boxes with the two prompts exchanged | two regions from the right region for each animal |
| Same prompt | the right boxes with "a cat" in both | regions that make two objects from regions that make the two named objects |

**Controls.** Masks of 0.5 everywhere at guidance 7.5 are the plain product at guidance 3.75, so the two renders must differ by at most 3 of
255 in mean absolute pixel value, the project's reproduction limit. Each joint render must also match the render its boxes were detected
on by at most 3 of 255.

**Counting.** All 48 pictures are shuffled under neutral names, each shown with the prompt "a cat and a dog". Two reviewers each call every
picture at full size under the rubric, without the key, and also record how many separate, clear animals it holds. Where they disagree on
`two_clear` or on whether there are at least two separate, clear animals, a third blind call at full size decides.

The same-prompt row asks for a cat in both regions, so it cannot make a clear cat and dog. Its count in the bar is the pictures with at least
two separate, clear animals, which is the oracle-layout plan's own call. Every other row counts clear pairs.

**The bar, written on 2026-11-04.**

| Result | What it takes | What happens |
|---|---|---|
| Support | right boxes reach at least 4 clear pairs of 8, and random boxes reach fewer | rung 2 runs |
| Layout content irrelevant | right boxes reach at least 4, and random boxes are within 1 | rung 2 runs, and the review says the regions' place did not matter |
| Regions without names | right boxes reach at least 4, and swapped or same prompt is within 1 | rung 2 runs, and the review says which prompt a region carries did not matter |
| Inconclusive | right boxes reach 2 or 3 | the idea stops at this rung |
| Null | right boxes reach 0 or 1 | held until the two sampler checks below have run, because a null from this composer alone cannot separate "regions are not enough" from "the composer is wrong" |

**The two sampler checks behind a null.** The first turns on bootstrapping, the one part of MultiDiffusion the plain fusion leaves out. For
the first 15 of 50 steps, the value the plan's overlap figure used, each region is denoised over a constant colour and the latents are
fused. The right, random, swapped and same-prompt rows rerun on the same 8 seeds, and their 32 pictures are counted blind with the 8 joint
renders mixed in, under the same bar. The second check reproduces MultiDiffusion's own code (arXiv 2302.08113) on the same eight layouts. It
runs only if bootstrapping also reaches 0 or 1, and its cost goes to the owner before it starts.

**Cost:** about 1 GPU-hour for 48 renders, the controls and the bootstrapping rerun. **Buys:** whether knowing where each body is can fix identity on SDXL at all.

## Rung 2: can CO3 find its own regions?

Rung 2 runs only if rung 1 reaches 4 clear pairs of 8. Its code and its zero-binding check are written after that result.

**Group.** Runs that try ideas (group 2).

**Varied axis.** Where the masks come from.

**What runs.** Steps 0 to 9 stay CO3's own. At step 10, each subject's word attention map becomes a soft mask through
`poe_repair/composers/_regional_masks.py`, and from step 10 each subject's prediction acts only inside its mask.

**Pairs and seeds.** M1's 12 tuning pairs at seeds 6001 to 6008, so 96 pictures per row.

| Row | What it does |
|---|---|
| CO3 alone | M1's 96 pictures, reused and counted again with the other rows |
| CO3 with its own masks | the method |
| CO3 with swapped masks | each subject in the other's region |
| CO3 with random masks | a control on whether the regions matter |

**Control.** A zero-binding run reproduces CO3's own loop to under 1e-5, on one card and dtype.

**The bar, on faithful pairs of 96.** Standard errors come from 10,000 bootstrap resamples of the 12 pairs.

| Result | What it takes |
|---|---|
| Support | own masks lead CO3 alone by at least 5 and by more than 1.96 standard errors, and random masks do not |
| Null | own masks have no such lead |
| Regions do nothing | random masks lead CO3 alone by as much as own masks do |
| Inconclusive | the zero-binding check fails |

**Cost:** about 5.3 GPU-hours for 288 renders at a budgeted 60 seconds and a smoke test. **Buys:** whether Paper 2 can draw on regions the
sampler finds for itself, with no outside layout.

## What this cannot say

Tuning pairs and seeds support no paper claim. Rung 1 covers one pair and uses boxes from renders that already succeeded. Word attention can
miss a subject at step 10, so rung 2 may fail because of its masks while regions still work, and rung 1 is what separates those readings.
