# 📒 Report: can an automatic score see which species a render drew

**The best automatic judge agrees with the eye on 13 of 19 confident renders, below the bar of 27 of 30 written for it.** The judge counts the bodies in a render, then names each body with CLIP, so that a cat beside a dog scores differently from two dogs. The [recipe walk](../../artifacts/ideas/one-training-recipe/IDEA_MAP.md#3--a-per-checkpoint-judge-that-sees-species-can-be-fixed-before-the-run) records the result as usable to rank checkpoints and short of stopping a run on its own. This folder has no card and no finding verdict.

## Table of contents

- [What the judge calls on three renders](#what-the-judge-calls-on-three-renders)
- [How often it agrees with the eye](#how-often-it-agrees-with-the-eye)
- [Whether each word looks at its own animal](#whether-each-word-looks-at-its-own-animal)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## What the judge calls on three renders

Navigation: 📋 [TOC](#table-of-contents) | [Next](#how-often-it-agrees-with-the-eye) ➡️

| The eye and the judge agree | The judge sees a pair, the eye two dogs | The judge sees two tigers, the eye a hybrid |
|---|---|---|
| `a_cat__x__a_dog_s09_v58-05-self-contrast_10000.png` is not shown because it failed the subject check | <a href="../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/can-a-score-see-species/run-02/annotated/a_cat__x__a_dog_s42_v58-05-self-contrast_20000.png"><img src="../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/can-a-score-see-species/run-02/annotated/a_cat__x__a_dog_s42_v58-05-self-contrast_20000.png" width="200" alt="Two white animals on a sofa, boxed and labelled cat and dog"></a> | <a href="../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/can-a-score-see-species/run-02/annotated/a_tiger__x__a_dog_s42_v58-05-self-contrast_10000.png"><img src="../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/can-a-score-see-species/run-02/annotated/a_tiger__x__a_dog_s42_v58-05-self-contrast_10000.png" width="200" alt="A tiger beside a white dog with tiger stripes, both boxes labelled tiger"></a> |
| Cat and dog, seed 9. Both call it composed. The judge finds 3 bodies `verified`. | Cat and dog, seed 42. The eye says two of one species, the judge says composed `verified`. | Tiger and dog, seed 42. The eye says hybrid, because the striped body has a dog's face. The judge says two of one `verified`. |

**Each picture is one render from the adapter checkpoint `v58-05-self-contrast`, with a box per body and CLIP's three shares for that body.** The shares are for the first animal, the second animal and "half of each". The calls are from [the second pass's verdicts](../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/can-a-score-see-species/run-02/verdicts.json).

## How often it agrees with the eye

Navigation: ⬅️ [Previous](#what-the-judge-calls-on-three-renders) | 📋 [TOC](#table-of-contents) | [Next](#whether-each-word-looks-at-its-own-animal) ➡️

**The second pass agrees on 13 of 19 confident renders, up from 10 of 23 on the first pass, and neither reaches the bar.** Both passes judged the same 29 renders.

| | First pass, `run-01` | Second pass, `run-02` |
|---|---|---|
| CLIP model | ViT-B/32 | ViT-L/14 |
| answers for each body | two names | two names and "half of each" |
| by-eye labels | first pass, at thumbnail size | redone at full size |
| confident renders | 23 | 19 |
| agreement | 10 of 23 | 13 of 19 |
| cat and dog | 8 of 13 | 9 of 12 |
| tiger and dog | 0 of 6 | 4 of 7 |
| cat and fox | 2 of 4 | no confident render |

Every count was recomputed from the two verdict files `verified`. The walk's bar is agreement on 27 or more of 30 labelled cells, and it records this result as below that bar `stated` ([the walk's check table](../../artifacts/ideas/one-training-recipe/IDEA_MAP.md#checks-outstanding)).

## Whether each word looks at its own animal

Navigation: ⬅️ [Previous](#how-often-it-agrees-with-the-eye) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

| Where each word attends, by step | The boxes it is measured against | The same boxes on one body |
|---|---|---|
| <a href="../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/can-a-score-see-species/attention-probe/catdog_s9_lora/read/maps_seed_9.png"><img src="../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/can-a-score-see-species/attention-probe/catdog_s9_lora/read/maps_seed_9.png" width="200" alt="Two rows of five grey attention maps; both rows brighten on the same two animal shapes"></a> | <a href="../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/can-a-score-see-species/attention-probe/catdog_s9_lora/read/boxes_seed_9.png"><img src="../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/can-a-score-see-species/attention-probe/catdog_s9_lora/read/boxes_seed_9.png" width="200" alt="A cat and a dog in a field, with a yellow box on the cat and a cyan box on the dog"></a> | <a href="../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/can-a-score-see-species/attention-probe/tigerdog_s42_plain/read/boxes_seed_42.png"><img src="../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/can-a-score-see-species/attention-probe/tigerdog_s42_plain/read/boxes_seed_42.png" width="200" alt="One fused striped animal filling the frame, with both boxes around the whole picture"></a> |
| Cat and dog, seed 9, with the adapter. Top row is the "cat" word and bottom row the "dog" word, at steps 0, 5, 10, 25 and 49 `stated` ([script](../../scripts/attention_probe_read.py), line 105). | The boxes GroundingDINO found for "a cat" and "a dog" on the same render `verified`. | Tiger and dog, seed 42, plain product. Both words' boxes cover 97% of the frame `verified`. |

**Each word puts about as much attention inside a box as that box's share of the frame, so the numbers show no preference for its own animal `stated` (the walk's reading of run 5).** One row is one capture. A share is the fraction of a word's attention map inside a box, averaged over all 50 steps, computed from each capture's `attention_shares_seed_*.json`. A uniform map would score the box's share of the frame.

| Capture | Overlap of the two boxes (intersection over union, 0 to 1) | First box's share of the frame | First word inside it | Second box's share of the frame | First word inside it |
|---|---|---|---|---|---|
| cat and dog, seed 9, adapter | 0.002 | 0.270 | 0.289 | 0.439 | 0.451 |
| cat and dog, seed 9, plain product | 0.996 | 0.527 | 0.604 | 0.527 | 0.604 |
| tiger and dog, seed 42, adapter | 0.992 | 0.508 | 0.520 | 0.508 | 0.520 |
| tiger and dog, seed 42, plain product | 0.996 | 0.969 | 0.976 | 0.969 | 0.976 |

## Where the sources disagree

Navigation: ⬅️ [Previous](#whether-each-word-looks-at-its-own-animal) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Four recorded facts conflict or need a qualifier, and each is shown here.**

| What | One source | The other source |
|---|---|---|
| how many renders were labelled | the walk's bar counts 30 by-eye labels | the labels file holds 29 |
| which renders were labelled confidently | the first pass counts 23 | the second counts 19. The labels file says the first, thumbnail-size labels were replaced at full size, and 16 of the 29 rows changed label or confidence |
| whether attention matches the frame share in every capture | the walk says each share "equals" the box's share of the frame | the plain cat and dog capture puts 0.604 in a box covering 0.527. The walk's 0.58 for this capture is the average over steps 0 to 9 alone |
| which boxes landed on one body | the walk says the "tiger" and "dog" boxes did, in three of four images | the three captures with overlapping boxes are two tiger and dog captures and one cat and dog capture |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**The second pass has no confident cat and fox render,** so it says nothing about that pair.

**Only one attention capture can compare a word's own animal with the other animal.** In the other three, the two boxes are the same box, so "inside its own" and "inside the other" are one number.

**The agreement is against one reader's labels** on 29 renders from two seeds per pair.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. The project numbers are from the two verdict files. The confidence margin and the zero-shot classifier are taught on [the cat prompt page](does-the-cat-prompt-already-draw-dogs.md#concepts-this-uses).

**Agreement between a judge and the eye**

Two people taste 10 coffees and each calls them strong or weak. Put one person across the top and the other down the side, and the diagonal is where they agree:

| | second says strong | second says weak |
|---|---|---|
| first says strong | 4 | 1 |
| first says weak | 2 | 3 |

Agreement is the diagonal over the total, (4 + 3) / 10 = 0.7. Here the judge runs across the top and the eye down the side, on the second pass's 19 confident renders:

| | judge: composed | judge: two of one | judge: hybrid |
|---|---|---|---|
| eye: composed | 9 | 1 | 1 |
| eye: two of one | 2 | 1 | 1 |
| eye: hybrid | 0 | 1 | 3 |

The diagonal is 9 + 1 + 3 = 13, and 13 / 19 is 0.68. The bar of 27 of 30 is 0.90.

**A share against chance**

Spill paint evenly over a board of 100 squares, and a frame covering 30 squares catches 30% of it. A spill that favours the frame would put more than 30% there. In the first capture the cat's box covers 0.270 of the picture and holds 0.289 of the "cat" word's attention, which is 1.07 times what an even spread would put there (0.289 / 0.270).

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| by-eye labels | [the labels file](../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/can-a-score-see-species/labels.json), read 2026-11-22 | `verified` |
| the first pass's calls | [first verdicts](../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/can-a-score-see-species/run-01/verdicts.json), recounted 2026-11-22 | `verified` |
| the second pass's calls | [second verdicts](../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/can-a-score-see-species/run-02/verdicts.json), recounted 2026-11-22 | `verified` |
| how the judge works | [the species judge](../../scripts/species_judge.py); the hybrid margin of 0.10 is at line 39 | `stated` |
| how attention was read | [the attention reader](../../scripts/attention_probe_read.py) | `stated` |
| the runs, the bar and the walk's reading | [the recipe walk](../../artifacts/ideas/one-training-recipe/IDEA_MAP.md#runs), runs 2, 4 and 5 | `stated` |
| 57 repeat pictures held out of the drain | 26 annotated captures in [run 1](../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/can-a-score-see-species/run-01/annotated), 26 in [run 2](../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/can-a-score-see-species/run-02/annotated) and 5 [attention-probe reads](../../artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/can-a-score-see-species/attention-probe), in the same layouts as the examples placed above | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Write this folder's card. Its index row says "no card yet".
- [ ] Decide whether the bar stays 27 of 30 now that 29 renders are labelled and 19 are confident.
- [ ] Label cat and fox renders confidently, or state that the judge does not cover that pair.
- [ ] Repeat the attention read with boxes that land on separate bodies. The walk proposes instance boxes plus the judge's per-body label.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
