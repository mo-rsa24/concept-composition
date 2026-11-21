# 📒 Report: does the product of "cat" and "dog" match the joint prompt

**On seeds 1 to 7, the product of "cat" and "dog" draws one fused animal on six seeds, and one wording that adds "multiple animals" draws separate animals on three.** The sheet compares eight ways of asking SDXL for cats and dogs, starting every row from the same noise. This folder has no card, no pass rule and no per-tile calls, so no verdict is recorded and the counts here are one by-eye reading.

## Table of contents

- [Eight ways of asking, seven seeds](#eight-ways-of-asking-seven-seeds)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## Eight ways of asking, seven seeds

Navigation: 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

| Seeds 1 to 7 | Seeds 4 to 7 again |
|---|---|
| <a href="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-of-cats-and-dogs-match-the-joint/cats-and-dogs-sheet.jpg"><img src="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-of-cats-and-dogs-match-the-joint/cats-and-dogs-sheet.jpg" width="200" alt="Seven rows by eight columns of cat and dog renders, three one-prompt columns then five product columns"></a> | <a href="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-of-cats-and-dogs-match-the-joint/cats-and-dogs-sheet-seeds-4-to-7.jpg"><img src="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-of-cats-and-dogs-match-the-joint/cats-and-dogs-sheet-seeds-4-to-7.jpg" width="200" alt="Four rows by eight columns, the same columns for seeds 4 to 7"></a> |
| Rows are seeds and columns are prompts `verified`. | The same eight columns for the last four seeds `verified`. |

| Fused animal, "cat" × "dog" | Crowd, the joint "cats and dogs" | Separate animals, "dog and cat" × "multiple animals" |
|---|---|---|
| <a href="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-of-cats-and-dogs-match-the-joint/tiles/cat__x__dog/seed01/poe.png"><img src="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-of-cats-and-dogs-match-the-joint/tiles/cat__x__dog/seed01/poe.png" width="200" alt="One animal on a pavement with a cat's ears and a dog's muzzle"></a> | <a href="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-of-cats-and-dogs-match-the-joint/tiles/cats__x__dogs/seed06/mono.png"><img src="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-of-cats-and-dogs-match-the-joint/tiles/cats__x__dogs/seed06/mono.png" width="200" alt="An illustrated group of about eight cats with one dog at the top right"></a> | <a href="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-of-cats-and-dogs-match-the-joint/tiles/dog_and_cat__x__multiple_animals/seed02/poe.png"><img src="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-of-cats-and-dogs-match-the-joint/tiles/dog_and_cat__x__multiple_animals/seed02/poe.png" width="200" alt="A golden dog, a white cat and a white kitten side by side"></a> |
| Seed 1 at full size `inferred`. | Seed 6 at full size `inferred`. | Seed 2 at full size `inferred`. |

**The two-word product fuses, and the plural prompts draw illustrated crowds.** Each count is out of the seven seeds, read by eye at sheet scale, where a fused animal can pass for a pair.

| Column | What is asked | What it draws, by eye `inferred` |
|---|---|---|
| 1 and 2 | one prompt, "cats" and "dogs" | mostly illustrated crowds; seed 5 is one cat's face and seed 6 of "dogs" is two dogs |
| 3 | joint prompt, "cats and dogs" | illustrated crowds of cats and dogs on all 7 |
| 4 | product of "cats" and "dogs" | illustrated crowds on all 7 |
| 5 | product of "cat" and "dog" | one fused animal on 6; seed 4 is a repeated pattern of animals |
| 6 | product of "dog", "cat" and "single" | one fused animal on 7 |
| 7 | product of "dog", "cat" and "multiple animals" | crowds on 4 (seeds 1, 2, 4, 7); a large fused face, with or without small animals, on 3 |
| 8 | product of "dog and cat" and "multiple animals" | separate animals in a photo on 3 (seeds 2, 6, 7); illustrated crowds on 4 |

## Where the sources disagree

Navigation: ⬅️ [Previous](#eight-ways-of-asking-seven-seeds) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Two recorded descriptions differ from the files.**

| What | One source | The other source |
|---|---|---|
| column 5 on every seed | [the stuck-points note](../../../artifacts/ideas/beyond-the-product-of-experts/where-we-are-stuck.md#1-the-product) says one animal with a cat's ears and a dog's muzzle "on every seed" | seed 4 of column 5 on the sheet is a repeated pattern of many grey animals `inferred` |
| how many seeds exist | the sheets show seeds 1 to 7 | the tiles folder also holds seed 8 for "cats", "dogs" and the joint prompt, with no product tile at seed 8 `verified` |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**Column 8 is a mixed comparison.** Its first prompt, "dog and cat", already names both animals, so its separate animals cannot be credited to the product alone.

**The columns differ in how hard they push.** One prompt pushes at guidance 7.5, two at a total of 15 and three at 22.5, per [the concept below](#how-hard-a-product-pushes). A column-to-column gap mixes wording with push strength.

**Seven seeds and no scorer.** These counts are descriptions of one sheet. The project has seen sheet-scale reads overcall composition, so a claim from them needs a full-size call per tile.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented.

### How hard a product pushes

[Classifier-free guidance](../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Classifier-Free%20Guidance.md#where-to-learn-this) starts from the model's prediction with an empty prompt and moves it toward the prompt's prediction, scaled by a weight. If the empty prompt predicts 10, the prompt predicts 12 and the weight is 3, the result is 10 + 3 × (12 - 10) = 16.

A [product of experts](../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Product%20Of%20Experts.md#where-to-learn-this) adds one such move per prompt. With a second prompt predicting 11, the result is 10 + 3 × 2 + 3 × 1 = 19. The two moves carry a total weight of 6, twice what one prompt carries.

The project's code does the same at weight 7.5: `poe_eps` adds the two guided predictions and subtracts the empty one, at lines 16 and 17 of [the sampling metrics](../../../poe_repair/_sdxl/metrics.py), and the three-prompt version sums over every prompt at line 67 of [the multi-prompt script](../../../1-lora-correction/scripts/concept_multi_poe.py). So the total weight is 7.5 for one prompt, 7.5 + 7.5 = 15 for two, and 22.5 for three.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| settings of each tile: 50 steps, guidance 7.5, 1024 px, noise from the cat and dog cache | the JSON file beside [each tile](../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-of-cats-and-dogs-match-the-joint/tiles/cat__x__dog/seed01/poe.json), read 2026-11-22 | `verified` |
| one- and two-prompt tiles | [the concept pair script](../../../scripts/shared/concept_pair_sheet.py) | `stated` |
| three-prompt tiles | [the multi-prompt script](../../../1-lora-correction/scripts/concept_multi_poe.py) | `stated` |
| what assembled the two sheets | no record found | owed |
| 56 full-size tiles held out of the drain | the [tiles folder](../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-of-cats-and-dogs-match-the-joint/tiles), whose members make up the two sheets placed above | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Write this folder's card. Its index row says "no card yet".
- [ ] Call every product tile at full size if these columns are to support a claim.
- [ ] Rerun the comparison at matched total guidance to separate wording from push strength.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
