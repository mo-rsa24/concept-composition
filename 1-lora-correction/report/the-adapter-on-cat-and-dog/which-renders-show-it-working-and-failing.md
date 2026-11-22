# 📒 Report: which renders show the adapter working and failing

**No verdict has been recorded, and this folder has no card.** In September 2026 the project looked
for pictures for the paper's main figure by rendering held-out pairs with several adapters and
render-time changes. These sheets are that search, in the order it happened, and the calls on them
are by eye.

## Table of contents

- [A training snapshot at step 45,000](#a-training-snapshot-at-step-45000)
- [Candidates on held-out pairs](#candidates-on-held-out-pairs)
- [Switching the adapter off early](#switching-the-adapter-off-early)
- [Two render-time changes](#two-render-time-changes)
- [Sheets kept off this page](#sheets-kept-off-this-page)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## A training snapshot at step 45,000

Navigation: 📋 [TOC](#table-of-contents) | [Next](#candidates-on-held-out-pairs) ➡️

<a href="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/which-renders-show-it-working-and-failing/existing/x0-seeds-step-45000/out_out__a_cat__x__a_dog__seed09.png"><img src="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/which-renders-show-it-working-and-failing/existing/x0-seeds-step-45000/out_out__a_cat__x__a_dog__seed09.png" width="200" alt="A tabby cat sitting beside a white dog on a grey background, held-out cat and dog at seed 9."></a>

**The renders the trainer saved at step 45,000 draw clean pairs on some cells and a hybrid on another.**
The picture is held-out cat and dog at seed 9, a clean tabby and a white dog. Cat and dog at seed 10
also shows both animals, and elephant and penguin at seed 9 shows a tall elephant-penguin hybrid
beside an elephant `verified`. The folder name and the [training recipe notes](../../../3-understanding-compositionality/artifacts/ideas/one-training-recipe/OPENING_PROMPT.md)
point to the run `v57-09-x0-seeds` `inferred`.

## Candidates on held-out pairs

Navigation: ⬅️ [Previous](#a-training-snapshot-at-step-45000) | 📋 [TOC](#table-of-contents) | [Next](#switching-the-adapter-off-early) ➡️

<a href="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/which-renders-show-it-working-and-failing/held-out-pairs-draft-grid.png"><img src="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/which-renders-show-it-working-and-failing/held-out-pairs-draft-grid.png" width="140" alt="Five pairs by three columns: joint prompt, plain product and the adapter, for cat and dog, tiger and parrot, zebra and monkey, goat and cat, cow and chicken."></a>

**The plain product draws one animal on all five pairs of this draft sheet, and the adapter draws both
animals apart on four, two of them with an extra head.** Columns are the joint prompt, the plain
product and the adapter. The sheet does not name
its adapter or seed, and no file beside it does.

| Pair | Plain product | Adapter `verified` |
|---|---|---|
| cat and dog | one fused face | a clean cat and dog |
| tiger and parrot | one zebra-striped parrot | both |
| zebra and monkey | one zebra | a monkey and two zebra heads |
| goat and cat | a goat with a cat's whiskers | a cat and a dark goat-like body, with a faint extra head behind |
| cow and chicken | a chicken on cow legs | a chicken's head on a cow's body, beside a calf fused with a chick |

## Switching the adapter off early

Navigation: ⬅️ [Previous](#candidates-on-held-out-pairs) | 📋 [TOC](#table-of-contents) | [Next](#two-render-time-changes) ➡️

<a href="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/which-renders-show-it-working-and-failing/adapter-switch-off/a_tiger__x__a_parrot.png"><img src="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/which-renders-show-it-working-and-failing/adapter-switch-off/a_tiger__x__a_parrot.png" width="200" alt="Tiger and parrot at seeds 9 to 12: joint prompt, plain product, then two adapters each on for 10, 20 and all 50 steps."></a>

**With the adapter labelled run 9, tiger and parrot stay fused when it is on for 10 steps, and only
seed 9 draws a clean pair at 20 or 50.** Rows are seeds 9 to 12. The joint prompt draws both on all
four and the plain product fuses all four `verified`.

| Seed | Run 9, on for 10 steps | On for 20 steps | On for all 50 |
|---|---|---|---|
| 9 | fused | tiger and parrot | tiger and parrot |
| 10 | fused | fused heads | fused heads |
| 11 | fused | parrot with a hybrid | parrot, tiger, extra small tiger |
| 12 | fused | two hybrids, invented hands | two parrot-tiger hybrids |

The columns labelled "original" draw two bodies at 20 and 50 steps with parrot and tiger patterns mixed
across them. No file defines "run 9" or "original".

## Two render-time changes

Navigation: ⬅️ [Previous](#switching-the-adapter-off-early) | 📋 [TOC](#table-of-contents) | [Next](#sheets-kept-off-this-page) ➡️

| Spread match and handover | The contrast corrector |
|---|---|
| <a href="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/which-renders-show-it-working-and-failing/spread-and-handover/a_tiger__x__a_parrot.png"><img src="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/which-renders-show-it-working-and-failing/spread-and-handover/a_tiger__x__a_parrot.png" width="200" alt="Tiger and parrot at seeds 9 to 12: joint prompt, plain product, run 9, run 9 with spread match, and run 9 handing over to the original adapter at step 20."></a> | <a href="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/which-renders-show-it-working-and-failing/contrast-corrector/a_goat__x__a_cat/seed11/ours_run9c6_w50.png"><img src="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/which-renders-show-it-working-and-failing/contrast-corrector/a_goat__x__a_cat/seed11/ours_run9c6_w50.png" width="200" alt="A flat black and white poster of a large dog-like figure, a horned goat and a small seated cat-like animal."></a> |
| neither change alters what each seed draws: the same seeds stay clean or fused | goat and cat at seed 11, with the corrector added to run 9: flat poster art |

**Spread match and the handover leave each seed's layout where run 9 put it, and the corrector turns
every render into poster art.** The first two were opened on the four tiger and parrot seeds. The
corrector was opened on the 8 tiger and parrot and goat and cat seeds, and all 8 come out flat or
saturated `verified`.

## Sheets kept off this page

Navigation: ⬅️ [Previous](#two-render-time-changes) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

> [!WARNING]
> Each file below is not shown, because a small shape in one tile could not be identified against
> the [pair ledger](../../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md). Paths are
> relative to this result's folder, `1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/which-renders-show-it-working-and-failing/`.

| File | What could not be told |
|---|---|
| `elephant-and-penguin-seeds-and-checkpoints.png` | a fuzzy round-eared creature at seed 16, and a small splayed silhouette at seed 12 |
| `existing/pool43-early25-step-40000/contact_sheet.png` | a segmented ring and a small winged shape at the animals' feet, cat and dog at seed 9 |
| `existing/x0-seeds-step-45000/contact_sheet.png` | a blurred winding shape in the background of the lion and meerkat tile |
| `contrast-corrector/a_tiger__x__a_parrot.png` | a brown lump with a snout inside a shell shape at seed 12 |
| `contrast-corrector/a_goat__x__a_cat.png` | a brown winged shape in the air at seed 12 |

## What this cannot tell you

Navigation: ⬅️ [Previous](#sheets-kept-off-this-page) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**These sheets were made to find good pictures, so they cannot give a success rate.** Every call is
by eye on a small number of seeds, and no [compose rate](../../../context/world/compose-rate.md#what-a-compose-rate-is) is quoted here. The two training snapshot folders
also hold `__cmp` comparison strips, whose target panel always shows two animals, so a score taken
over those folders by the compose-rate script before commit `0811f029` would count them.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers are invented. The plain product, the joint prompt, the same starting noise and the
adapter are taught in [the four-pairs page](does-one-adapter-compose-four-pairs.md#concepts-this-uses).

**Spread match**

Suppose a prediction's numbers spread about their mean by 4, and the two single prompts spread by 2
on average. Scaling by 2 / 4 = 0.5 brings the spread back to 2. With a blend weight of 0.7, a value of
10 becomes 0.7 × (10 × 0.5) + 0.3 × 10 = 6.5. That is the rule at lines 375 to 379 of
[the sampler](../../../poe_repair/methods/_poe_langevin.py), with the spread measured as a standard
deviation. The weight used for this sheet is not recorded.

**Handover**

Two painters can share a canvas: the first paints up to step 20 and the second finishes. Here the
second painter is another adapter, set by `--late-checkpoint` and `--switch-at` in
[the candidate script](../../scripts/figure_candidates.py).

**The contrast corrector**

Suppose the product's predicted clean picture reads 10 at one point, and the two concepts read 6 and
2. The corrector doubles the product to 20, then subtracts the concepts with weights that sum to 1,
say 0.75 for the concept closest to the product and 0.25 for the other: 20 − (0.75 × 6 + 0.25 × 2) = 15.
So whichever concept is taking over gets pushed away hardest. The result is then rescaled to the size
the product alone had, as lines 323 to 331 of [the sampler](../../../poe_repair/methods/_poe_langevin.py)
describe. The tile's name, `run9c6`, suggests it ran over the first 6 steps `inferred`.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| how candidates were rendered | [the candidate script](../../scripts/figure_candidates.py), from commit `fe089023`, 2026-09-22 | `stated` |
| the contrast corrector | commit `4c51b0bd`, 2026-09-23 | `stated` |
| the order of the sheets | file modified times, 2026-09-22 to 2026-09-23 | `inferred` |
| what each placed sheet holds | each opened 2026-11-22 | `verified` |
| 104 pictures held out of the drain | the other tiles and layouts in this result's folder. They repeat a picture placed above, or could not be cleared against the subjects the [pair ledger](../../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md) keeps off sheets, so the folder is not linked here | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Write this folder's card, naming the adapters "run 9" and "original", and the draft grid's adapter and seed.
- [ ] Look at the five sheets kept off this page and decide whether any may be shown.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
