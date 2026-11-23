# Does balancing the checker's training data help? (step M5)

## What this asks

The checker's training batches are about 60% cat and dog, so it may have learned those two animals more than it learned what a separate
pair looks like. This stage trains the same checker on the current list and on a list balanced across pairs, then compares the pictures
each one produces on pairs neither has seen. The better list goes to step D3.

## Words this uses

**Checker.** A small classifier on SDXL's frozen middle and first two up-block features, read under the pair's joint prompt.

**Push.** The checker's gradient added to the noise prediction during the early sampling steps.

**Product of experts (PoE).** The sampler that adds SDXL's noise predictions for the two single-subject prompts.

**Faithful pair.** A picture with exactly one clear subject A and one clear subject B as separate bodies, no borrowed features and no
damage, called under the [blind count rubric](../01-beat-the-baseline/blind-count-rubric.md).

## The run

**Group.** Runs that try ideas (group 2). The result picks a training list and makes no claim in either paper.

**Varied axis.** How the training pictures are chosen. Architecture, labels, loss, training length, push settings, pairs, seeds and scoring
stay the same.

## The two lists

`2-classifier-guidance/scripts/method_search/build_balanced_list.py` builds both and rebuilds them byte for byte. Each starts from
`training_set_multipair.json`, removes the pairs the subject guard refuses, and holds out every pair whose name's SHA-256 is 0 modulo 4,
the owner's 25% rule from board question 37. That holds out 58 of 216 pairs. Cat and dog stays in training. The counts are in
[the composition file](composition.json).

| | Current list | Balanced list |
|---|---:|---:|
| How a batch draws a picture | one of 10 sources, then a picture | one pair, then a picture |
| Pictures | 2,464 | 877 |
| Clean pictures | 999 | 431 |
| Pairs | 158 | 158 |
| Pairs with a clean picture | 131 | 131 |
| Cat and dog pictures | 1,603 | 16 |
| Cat and dog share of clean draws | 60% | 0.76% |

The balanced list keeps 8 clean and 8 reject cat and dog pictures, because 16 is the size of the largest other pair.

## What stays fixed

**One output.** Both checkers train the clean-pair output only. The presence outputs have labels for cat and dog alone, so keeping them
would change the labels along with the list. Step M2 tests the labels.

**Training.** 2,000 updates with the kept checker's settings: the joint prompt, the three augmented copies of each picture, half of the
timesteps drawn from the first 20 sampler steps, and 4 accumulated batches per update.

**Push.** Strength 30, capped at 0.4 of PoE's prediction, on steps 0 to 15, then plain PoE. D3 compares its candidates on plain PoE, so
this stage does too.

## Pairs, seeds and rows

The 12 tuning pairs in `2-classifier-guidance/scripts/checker_choice/splits.json` at seeds 6001 to 6008, so 96 pictures per row. Neither
list holds any of those pairs. The 12 held-out pairs and seeds 7001 to 7008 are not touched.

| Row | What it does |
|---|---|
| 1. Plain PoE | no push, and D3 reuses these 96 pictures |
| 2. Current list | the push from the checker trained on the current list |
| 3. Balanced list | the push from the checker trained on the balanced list |

All 288 pictures are shuffled under neutral names and called by eye at full size, with no API tokens. Giraffe and tortoise is counted and
stays off every sheet, page and message.

**Control.** A zero push reproduces plain PoE with a largest absolute latent difference under 1e-5, in one process on one card and dtype.

## The bar, on faithful pairs of 96

The standard error comes from 10,000 bootstrap resamples of the 12 pairs. Repeating a pushed render has moved its count by about 2 of 96,
so a lead under 5 is a tie.

| Result | What it takes | Which list goes to D3 |
|---|---|---|
| Balanced wins | row 3 leads row 2 by at least 5 and by more than 1.96 standard errors | balanced |
| Current wins | row 2 leads row 3 by at least 5 and by more than 1.96 standard errors | current |
| Tie | neither lead holds | balanced, because it depends least on cat and dog, unless it has at least 5 more damaged pictures than current |
| Inconclusive | the zero-push check fails | none until the check passes |

**What would surprise.** A balanced checker that loses clearly would mean the extra cat and dog pictures teach something that carries to
other subjects, which is the opposite of the concern behind this stage.

## Cost and what it buys

**Cost:** about 8 GPU-hours of this loop's 79: two trainings budgeted at 2.5 GPU-hours each, and 288 pictures at a budgeted 40 seconds.
No API tokens. The smoke rung replaces these budgets with measured times.

**Buys:** step M5, the training list D3 uses, and a first measured answer to whether the checker's dependence on cat and dog matters.

## The ladder

Each rung saves its config, numbers, sheet and commit under `/datasets/mmolefe/poe_repair_min/loop-6/m5/<rung>/`.

1. **Smoke test.** One batch from each list, one checkpoint saved and reloaded, one pushed render. Records seconds per update and picture.
2. **Overfit.** Each checker memorises 16 pictures, reaching at least 15 of 16 right.
3. **1% of the data.** Each list trains on 1% of its pictures, and the logged loss falls.
4. **Controls.** The zero-push check.
5. **Comparison.** Both full trainings, the three rows and the blind count.

## What this cannot say

One training per list leaves out the variation between trainings. The balanced list is also smaller, so a difference could come from
having fewer pictures as well as from their balance. Eleven of the 12 test pairs are animals, so D3's animals-to-objects test is the
stronger check of whether a checker carries across categories.
