# 📒 Report: is the layout the piece the product of experts is missing

**This question is unanswered: the boxes and masks exist, one preview was rendered, and the comparison the verdict depends on never ran.** The idea is to give each prompt its own region of the picture, taken from a render where the joint prompt "a cat and a dog" already drew both animals, and ask whether that alone gives two clear animals. [The plan](../../plans/10-what-poe-loses-when-it-composes/is-the-layout-the-missing-piece.md) set the bar at 4 clear pairs of 8 seeds before anything ran. No verdict is recorded.

## Table of contents

- [The boxes come from the joint prompt's own render](#the-boxes-come-from-the-joint-prompts-own-render)
- [Each box becomes a soft mask](#each-box-becomes-a-soft-mask)
- [The one preview that was rendered](#the-one-preview-that-was-rendered)
- [What would count](#what-would-count)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## The boxes come from the joint prompt's own render

Navigation: 📋 [TOC](#table-of-contents) | [Next](#each-box-becomes-a-soft-mask) ➡️

| The eight layouts | Every candidate box |
|---|---|
| <a href="../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/is-the-layout-the-missing-piece/detected-layouts.png"><img src="../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/is-the-layout-the-missing-piece/detected-layouts.png" width="200" alt="Eight renders of a cat and a dog, each with a blue box on the cat and a red box on the dog"></a> | <a href="../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/is-the-layout-the-missing-piece/candidate-crops.png"><img src="../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/is-the-layout-the-missing-piece/candidate-crops.png" width="200" alt="A grid of animal crops from 24 seeds, each titled with the name CLIP gave it, its margin and its share of the frame"></a> |
| Seeds 9, 12, 16, 19, 22, 25, 27 and 30. Blue is the box called cat and red the box called dog `verified` (read on the picture; matches the boxes file). | All 49 boxes from seeds 9 to 32, cropped as the classifier received them. Orange titles are crops it left uncertain `verified`. |

**Eight of the 24 candidate seeds were usable, and four of those eight needed a name corrected by hand.** A seed had to give exactly two boxes, neither covering more than 70% of the frame nor holding more than half of the other, named one cat and one dog `stated` ([the plan, task 1.2](../../plans/10-what-poe-loses-when-it-composes/is-the-layout-the-missing-piece.md#1--find-the-layout-in-the-joint-renders)).

| The 16 boxes on the eight selected seeds | Boxes | Mark |
|---|---|---|
| CLIP's name is the final name | 12 | `verified` |
| CLIP named the wrong animal, corrected by hand (seeds 9 and 19) | 2 | `verified` |
| CLIP was uncertain, named by hand (seeds 12 and 25) | 2 | `verified` |

Counts are from [the boxes file](../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/is-the-layout-the-missing-piece/boxes.json), comparing each box's `clip_call` with its final `call`.

## Each box becomes a soft mask

Navigation: ⬅️ [Previous](#the-boxes-come-from-the-joint-prompts-own-render) | 📋 [TOC](#table-of-contents) | [Next](#the-one-preview-that-was-rendered) ➡️

<a href="../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/is-the-layout-the-missing-piece/mask-overlays.png"><img src="../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/is-the-layout-the-missing-piece/mask-overlays.png" width="140" alt="Eight rows of three panels: the joint render, then blue and red masks over it under two ways of splitting the overlap"></a>

**The two masks add to exactly 1 at every cell of the 128 by 128 grid, on all 8 seeds and both ways of splitting the overlap.** Each row is one seed: the joint render, then the masks with the overlap split evenly, then with the overlap given to the smaller box. Blue is where the cat prompt acts and red the dog prompt. The largest deviation from 1 is 0.0 against an allowed 0.00001 `verified` ([the masks file](../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/is-the-layout-the-missing-piece/masks.json)).

## The one preview that was rendered

Navigation: ⬅️ [Previous](#each-box-becomes-a-soft-mask) | 📋 [TOC](#table-of-contents) | [Next](#what-would-count) ➡️

| Two seeds, three columns | Flat masks | Plain product at half guidance |
|---|---|---|
| <a href="../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/is-the-layout-the-missing-piece/overlap-readings.png"><img src="../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/is-the-layout-the-missing-piece/overlap-readings.png" width="200" alt="Two rows: the joint render, the layout without bootstrapping, and the layout with 15 bootstrapping steps"></a> | <a href="../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/is-the-layout-the-missing-piece/identity/identity_regional_flat_masks.png"><img src="../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/is-the-layout-the-missing-piece/identity/identity_regional_flat_masks.png" width="200" alt="One fused orange and white cat-dog face"></a> | <a href="../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/is-the-layout-the-missing-piece/identity/identity_plain_poe_half_guidance.png"><img src="../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/is-the-layout-the-missing-piece/identity/identity_plain_poe_half_guidance.png" width="200" alt="Almost the same fused cat-dog face"></a> |
| Seeds 9 and 19. The layout alone draws one animal on both. With 15 opening steps denoised against a flat colour, a cat and a dog appear as cut-outs on grey `inferred` (by eye, two seeds). | The regional sampler with both masks at 0.5 everywhere. | The plain product at guidance 3.75, which the flat-mask sampler should reproduce. |

**The regional sampler reproduces the plain product to 1.531 grey levels of 255 on average, against the script's limit of 3.0.** That figure is the mean absolute difference over all 1024 by 1024 pixels and three colour channels of the two identity pictures, the calculation in `mean_abs_diff` at line 94 of [the preview script](../../../_archive/scripts/regional_preview.py), with its limit at line 63. The largest single difference is 121 of 255, and the two pictures differ visibly at the right eye and the whiskers `inferred`. The results file holds no recorded outcome for this check, so 1.531 is recomputed here `verified`.

## What would count

Navigation: ⬅️ [Previous](#the-one-preview-that-was-rendered) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**The verdict needs six columns on eight seeds, and only the boxes, masks and preview exist.** The plan's outcomes, written before the run:

| Outcome | What it takes |
|---|---|
| support | the layout reaches 4 or more clear pairs of 8, and random boxes reach fewer |
| null | the layout reaches 1 or 0, reported only after MultiDiffusion's own code is run on the same eight layouts |
| inconclusive | the layout reaches 2 or 3 |
| layout content irrelevant | 4 or more, with random boxes within 1 of it |
| regions without names | swapped prompts or the same prompt in both boxes come within 1 of the layout |

> [!IMPORTANT]
> The plan's rule for a usable seed can be read two ways, and the choice is the owner's. Its pass criteria ask for one cat box and one dog box "above the 0.20 margin floor", and task 1.2 lets the eye override the classifier.

| How the rule is read | Usable seeds | That reading would give |
|---|---|---|
| CLIP's names alone, each above the 0.20 margin | 16, 22, 27, 30 (4) | fewer than eight, so the plan restates the bar out of 4 before any call is made |
| CLIP's names with hand corrections | 9, 12, 16, 19, 22, 25, 27, 30 (8) | the bar stays 4 of 8. Two hand-named boxes sit under the margin: seed 12 at 0.077 and seed 25 at 0.069 |

## Where the sources disagree

Navigation: ⬅️ [Previous](#what-would-count) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Four recorded facts conflict, and the code explains part of each.**

| What | One source | The other source | What the code shows |
|---|---|---|---|
| the eighth seed | the plan's task 1.2 lists seed 20 and no seed 30 | the boxes file and both figures use seed 30 and reject seed 20 | [the layout script](../../scripts/layout_from_joint_render.py), lines 408 to 411, takes the first eight usable seeds. Seed 20's second box is uncertain at -0.007 and has no hand name. An earlier hand name for it would explain the plan's list `inferred` |
| whether the regional sampler ran | the plan leaves tasks 3.1 and 3.2 unticked, and [the stuck-points note](../../../artifacts/ideas/beyond-the-product-of-experts/where-we-are-stuck.md#what-has-never-been-run) says it never ran | the sampler is committed, and the preview and identity renders exist | [the preview script](../../../_archive/scripts/regional_preview.py) calls it. The six-column sweep has no output |
| what `overlap-readings.png` compares | the plan says two overlap readings on two seeds | the picture and its file hold one reading, "even", at 0 and 15 bootstrapping steps | the script's `--reading` help, lines 152 to 154, says the two readings "were indistinguishable on the first preview, so only one is rendered now". That first preview has no file here |
| the identity check | the stuck-points note says the flat-mask image has no record of how it was made | the results file holds `identity_check: null` | the script, lines 102 to 137, makes both identity images. It writes null when run with `--skip-identity`, lines 159 and 179 |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**The preview covers two seeds with no controls,** so it cannot separate the right layout from any layout.

**The layout comes from a render that already succeeded.** No deployed method would have these boxes, so a pass would show that a correct layout is enough and leave open where a layout comes from.

**The seeds are chosen for clean boxes,** 8 of 24, so a count on them describes easy layouts.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. How hard each prompt pushes is taught on [the cats and dogs page](does-poe-of-cats-and-dogs-match-the-joint.md#concepts-this-uses).

**A soft mask**

Two people paint a wall of 4 tiles. A mask says what share of each tile belongs to whom: person A holds 1, 1, 0.5, 0 and person B holds 0, 0, 0.5, 1. Each column adds to 1, so every tile is fully covered and nobody paints twice. Here each cell of the 128 by 128 grid has a cat share and a dog share that add to 1, and the blurred edge is where a cell is shared.

**The check that flat masks equal half guidance**

Say the two prompts push by 2 and 4, and the guidance is 6. With both masks at 0.5, the regional rule gives 6 × (0.5 × 2 + 0.5 × 4) = 18. The plain product at half guidance gives 3 × (2 + 4) = 18. The two are the same sum, so any difference between the renders comes from the code or from rounding. Here the guidance is 7.5 against 3.75.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| the boxes, names and rejections | [boxes file](../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/is-the-layout-the-missing-piece/boxes.json) and [hand names](../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/is-the-layout-the-missing-piece/hand-names.json), read 2026-11-22 | `verified` |
| the masks | [masks file](../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/is-the-layout-the-missing-piece/masks.json), read 2026-11-22 | `verified` |
| the preview's settings | [overlap readings file](../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/is-the-layout-the-missing-piece/overlap-readings.json) | `verified` |
| how boxes were found | [the layout script](../../scripts/layout_from_joint_render.py) | `stated` |
| how the preview was made | [the preview script](../../../_archive/scripts/regional_preview.py) | `stated` |
| the bar and outcomes | [the plan](../../plans/10-what-poe-loses-when-it-composes/is-the-layout-the-missing-piece.md#what-has-to-pass-before-this-runs) | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Choose which reading of the usable-seed rule applies.
- [ ] Run the six-column sheet: joint prompt, plain product, layout, random boxes, swapped prompts and the same prompt in both boxes.
- [ ] Call all 48 tiles at full size and write the card. This folder has none.
- [ ] Bring the plan's seed list and unticked tasks in line with what exists on disk.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
