# 📒 Report: does "a photo of a cat" already draw dogs in base SDXL

**By eye none of the 64 renders holds a dog, the detector reports a dog on all 64, and a classifier on the cropped animal calls all 64 a cat.** The count matters because the [inclusive-captions idea](../../../4-inclusive-captions/artifacts/ideas/inclusive-captions-axis/IDEA_MAP.md) measures its effect as a rise in dogs drawn from the cat prompt, and a rise only means something if the untrained model starts near zero. The rule written before the run counts "dog hits", and it gives opposite answers depending on which count it reads, so no single verdict is recorded for the floor.

## Table of contents

- [The same 64 renders, read two ways](#the-same-64-renders-read-two-ways)
- [What would count](#what-would-count)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## The same 64 renders, read two ways

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-would-count) ➡️

| What the detector says | What the classifier says |
|---|---|
| <a href="../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/does-the-cat-prompt-already-draw-dogs/contact_sheet.png"><img src="../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/does-the-cat-prompt-already-draw-dogs/contact_sheet.png" width="200" alt="64 renders of a cat in an 8 by 8 grid, every tile titled dog 1 cat 1"></a> | <a href="../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/does-the-cat-prompt-already-draw-dogs/crop_classifier_sheet.png"><img src="../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/does-the-cat-prompt-already-draw-dogs/crop_classifier_sheet.png" width="200" alt="The same 64 cats cropped to the animal box, every tile titled cat with a margin near 0.99"></a> |
| One render per seed, 0 to 63. Every tile shows one cat and is titled "dog 1 cat 1" `verified` (counted on the picture; matches the card). | The same renders cropped to the animal. Every tile is titled cat, with a margin of +0.98 to +1.00 `verified` (read on the picture; the summary file gives 0.982 to 0.995). |

**The detector cannot tell these cats from dogs, and the crop classifier can.** All renders use base SDXL at 1024 px, 50 DDIM steps and guidance 7.5 `verified` ([run summary](../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/does-the-cat-prompt-already-draw-dogs/summary.json)).

| Read | Result on the 64 renders | Mark |
|---|---|---|
| by eye | 0 hold a dog; each is a single cat | `verified` (contact sheet) |
| detector, asked for "dog" | 64 report one dog | `verified` ([run summary](../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/does-the-cat-prompt-already-draw-dogs/summary.json)) |
| detector confidence, 0 to 1 | median 0.889 for "dog", 0.943 for "cat" | `verified` ([per-seed file](../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/does-the-cat-prompt-already-draw-dogs/per_seed.json)) |
| detector margin, cat confidence minus dog confidence | median 0.054; 54 renders under 0.10; "dog" scores higher on 3 | `verified` (recomputed from the per-seed file) |
| classifier on the cropped animal | 64 cat, 0 dog, 0 uncertain; margin median 0.990 | `verified` ([classifier summary](../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/does-the-cat-prompt-already-draw-dogs/crop_classifier_summary.json)) |
| classifier on the whole image | 64 cat; margin median 0.990 | `verified` (same file) |

The classifier's three bars are constants in [its script](../../scripts/census/crop_classifier_cat_vs_dog.py), lines 81 to 83: at least 62 of 64 crops called cat, a median margin of at least 0.50, and anything under a 0.20 margin reported uncertain. The run cleared all three, and its summary file records "the instrument works" `verified`.

## What would count

Navigation: ⬅️ [Previous](#the-same-64-renders-read-two-ways) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**The rule written before the run gives a different answer for each count of "dog hits".** [The census script](../../scripts/census/cat_condition_dog_rate.py), lines 17 to 23, set three outcomes: 0 or 1 hits is a clean floor, 2 to 4 is a weak floor, and 5 or more means the base model already leaks dogs into "cat".

| How "dog hits" is read | Hits of 64 | That reading would give |
|---|---|---|
| the per-class detector, the count the script computes | 64 | 5 or more: the base already leaks dogs, and the pilot needs a held-out denoising-loss read |
| the crop classifier, which the card adopts for naming the animal | 0 | 0 or 1: a clean floor |
| by eye | 0 | 0 or 1: a clean floor |

> [!IMPORTANT]
> Which reading settles the floor is the owner's choice. The card argues for the classifier because the detector puts a "dog" box on a cat at a median confidence of 0.889.

## Where the sources disagree

Navigation: ⬅️ [Previous](#what-would-count) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Three recorded facts conflict or need a qualifier, and each is shown here.**

| What | One source | The other source |
|---|---|---|
| whether the renders hold dogs | the run summary records `fraction_with_a_dog` 1.0 | the card and the contact sheet show a single cat in all 64 |
| whether the crop classifier has been tried on two animals | [the card](../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/does-the-cat-prompt-already-draw-dogs/README.md) says it has not | [the layout experiment](is-the-layout-the-missing-piece.md#the-boxes-come-from-the-joint-prompts-own-render) ran the same classifier on 16 boxes in 8 two-animal renders: 12 match the final name, 2 name the wrong animal and 2 are uncertain |
| the size of the detector margin | the card gives a median of 0.054 and 54 renders under 0.10 | both hold for cat minus dog with its sign. Ignoring the sign gives a median of 0.057 and 52 renders, because a render where "dog" wins by more than 0.10 counts as under 0.10 only when the sign is kept |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**This is one prompt on one model over 64 seeds.** It says nothing about other wordings of the cat prompt, about other animals, or about a model after training.

**The reference is one reader's eye.** The classifier summary records its ground truth as "every render is a single cat, read by eye off the base-rate contact sheet".

**On one-animal pictures the crop adds nothing measurable.** The cropped and whole-image reads both give 64 cat calls at a median margin of 0.990.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. The project numbers are from the result files named above.

**A confidence margin**

A taster scores one coffee 0.94 for "dark roast" and 0.89 for "medium roast". The margin is the gap between the two names, 0.94 - 0.89 = 0.05, and a small gap says the taster barely told them apart. The sign records who won: taken as dark minus medium, a coffee scored the other way round gets a negative margin. On seed 0 here the detector gives "cat" 0.9438 and "dog" 0.9207, a margin of +0.0232, so a confident-looking "dog" box sits almost level with the right answer.

**A zero-shot classifier's margin**

[Zero-shot](../../../../../Obsidian/Artificial%20Intelligence/Machine%20Learning/Zero-Shot.md#where-to-learn-this) means the classifier was never trained on these labels: it compares the picture with the words "cat" and "dog" and turns the two match scores into shares that add to 1. Match scores of 3 and 1 become e³ / (e³ + e¹) = 0.88 and 0.12, a margin of 0.88 - 0.12 = 0.76. Because the shares add to 1, a margin m means the winner holds (1 + m) / 2. The median margin here of 0.990 is a cat share of 0.995 and a dog share of 0.005.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| detector counts and settings | [run summary](../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/does-the-cat-prompt-already-draw-dogs/summary.json) and [per-seed file](../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/does-the-cat-prompt-already-draw-dogs/per_seed.json), read 2026-11-22 | `verified` |
| classifier counts and verdict | [classifier summary](../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/does-the-cat-prompt-already-draw-dogs/crop_classifier_summary.json), read 2026-11-22 | `verified` |
| how the renders were made | [the census script](../../scripts/census/cat_condition_dog_rate.py) | `stated` |
| how the crops were scored | [the crop classifier script](../../scripts/census/crop_classifier_cat_vs_dog.py) | `stated` |
| the by-eye call and what it changes | [the folder's card](../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/does-the-cat-prompt-already-draw-dogs/README.md) | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Choose which count of "dog hits" settles the floor.
- [ ] Measure the crop classifier on two-animal renders with a labelled set. The 16 boxes in the layout experiment come from seeds picked for that experiment, four of them with names corrected by hand.
- [ ] Repair the card's link to the idea map, which still points at its old folder.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
