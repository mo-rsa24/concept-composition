# 📒 Report: does the product of experts match the joint prompt for each kind of composition

**The product matches the joint prompt on some kinds of composition and misses on others, and these sheets show examples without measuring a rate.** On Stable Diffusion v1-4, the product fuses a cat and a dog on 4 of 4 seeds, paints rain over a dry dog and gives near-identical pictures for a dog and an oil painting. No pass rule was written for this comparison and no verdict is recorded. The 42 pairs in the seven SD3.5 grids were selected to show the product failing, so those grids cannot say how often it fails.

## Table of contents

- [Eleven generated pairs](#eleven-generated-pairs)
- [The same pairs on Stable Diffusion 3.5](#the-same-pairs-on-stable-diffusion-35)
- [Seven kinds, six pairs each](#seven-kinds-six-pairs-each)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## Eleven generated pairs

Navigation: 📋 [TOC](#table-of-contents) | [Next](#the-same-pairs-on-stable-diffusion-35) ➡️

| All eleven pairs | Two objects at two positions | "Red cube" and not "blue sphere" |
|---|---|---|
| <a href="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/pairs-by-seed.png"><img src="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/pairs-by-seed.png" width="140" alt="Eleven rows of generated pairs by four seed columns, each cell a joint sample beside a product sample"></a> | <a href="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/tiles/spatial-poe-seeds0to3.png"><img src="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/tiles/spatial-poe-seeds0to3.png" width="200" alt="Four renders, each with two separate objects at the same two positions"></a> | <a href="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/tiles/negation-poe-seeds0to3.png"><img src="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/tiles/negation-poe-seeds0to3.png" width="200" alt="Four renders of stacked red cubes with some other objects and no blue sphere"></a> |
| Each row is a pair and each column a seed, 0 to 3. A cell holds the joint sample on the left and the product on the right, from the same seed `verified`. | The product on CLEVR, a synthetic scene of simple objects. Two objects land at the two positions on 4 of 4 seeds `verified`. | The product with a negative weight on "blue sphere". No blue sphere appears, and red cubes stack with extra objects on all 4 `verified`. |

**Each kind of composition gets a different answer from the product.** Ten rows use Stable Diffusion v1-4 and the red cube row uses Liu et al.'s CLEVR relation model. Counts are out of four seeds, as written on [the five-kinds page](../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/five-interactions.md) `stated`.

| Kind | Pair | Joint prompt | Product of experts |
|---|---|---|---|
| structural | cat, dog | a dog in 1; two cats in 3 | one fused cat-dog in 4 |
| structural | dog, oil painting | a painted dog in 4 | a near-identical painted dog in 4 |
| relational | a cat chasing a dog | animals in motion in 4 | one fused animal in 4 |
| relational | a cat behind a tree | trunk in front of the cat in 3; no cat in 1 | no occlusion in any |
| relational | red cube left of blue sphere | several objects | one object; attributes crossed in 3 |
| causal | wet dog, rain | wet fur in 4 | rain streaks painted over the image |
| causal | camel, sunset shadow | a long shadow in 2 | a long shadow in none |
| shared | butterfly, meadow | among plants in 4 | among plants in 4 |
| shared | dolphin, desert | "a dolphin" alone is in water in 4 | leaps from sand that splashes like water |
| shared | camel, forest | a camel in a forest in 4 | face and trees fused in 3; camel dropped in 1 |
| temporal | burning log, smoke | smoke in 1 | smoke in 3 |

## The same pairs on Stable Diffusion 3.5

Navigation: ⬅️ [Previous](#eleven-generated-pairs) | 📋 [TOC](#table-of-contents) | [Next](#seven-kinds-six-pairs-each) ➡️

| Joint prompt uses the same words | Joint prompt says what a person would write |
|---|---|
| <a href="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/pairs-by-seed-sd35.png"><img src="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/pairs-by-seed-sd35.png" width="140" alt="Eleven rows by four seeds on SD3.5, the joint prompt being the two product prompts joined with and"></a> | <a href="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/pairs-by-seed-sd35-intended-prompt.png"><img src="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/pairs-by-seed-sd35-intended-prompt.png" width="140" alt="Eleven rows by four seeds on SD3.5, the joint prompt adding words the product never sees"></a> |
| "A and B" at guidance 4.5, against the product at 2.25 per prompt `verified` (the picture's subtitle). | The prompt a person would write, such as "a wet dog in the rain", against the same product `verified`. |

**The second sheet separates what the joint gains from extra words.** The product column shows the same pictures in both sheets `verified` (compared by eye), so any change on the left comes from the wording. These two sheets have no entry on the card, and their settings are read from their own titles.

## Seven kinds, six pairs each

Navigation: ⬅️ [Previous](#the-same-pairs-on-stable-diffusion-35) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

| All seven kinds | Two objects keep their identity | Cause and effect | When fusion is the goal |
|---|---|---|---|
| <a href="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/interaction-grids/joint-only-atlas.png"><img src="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/interaction-grids/joint-only-atlas.png" width="140" alt="Seven rows of six joint-prompt images, one row per kind of interaction"></a> | <a href="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/interaction-grids/structural-objects.png"><img src="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/interaction-grids/structural-objects.png" width="200" alt="Six object pairs by four seeds; the joint keeps two objects and the product merges them"></a> | <a href="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/interaction-grids/causal.png"><img src="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/interaction-grids/causal.png" width="200" alt="Six cause and object pairs by four seeds, joint beside product"></a> | `when-fusion-is-the-target.png` is not shown because it failed the subject check |
| The joint prompt alone, seed 0 `stated` ([composition terms](../../artifacts/ideas/poe-overlap-dial/composition-terms.md)). | Violin and pineapple, backpack and toaster and four more. The product merges most of them `inferred`. | Flame and paper, rain and a box and four more, seeds 0 to 3. | Style and material pairs. The product does well where one concept is a surface over the other `stated` (composition terms). |

**Each grid shows one kind of interaction on six pairs and four seeds.** Every grid is SD3.5 medium with the joint prompt "A and B" at guidance 4.5 and the product at 2.25 per prompt, from the same seed. The [out-of-context grid](../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/interaction-grids/associative-out-of-context.png) is a further example of the same layout.

## Where the sources disagree

Navigation: ⬅️ [Previous](#seven-kinds-six-pairs-each) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Four recorded facts differ from the files.**

| What | [The card](../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/README.md) says | The files show |
|---|---|---|
| which selection was drawn | three rounds, with `round3.json` the selection drawn | a [fourth round](../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/pair-selection/round4.json) replaces all six temporal pairs. The other six grids match round 3, and the drawn temporal row is round 4's `verified` |
| how many grids | seven | nine files. The two extra are the joint-only overview and the fusion grid, described in [the composition terms](../../artifacts/ideas/poe-overlap-dial/composition-terms.md) |
| how many cropped panels | 12 | 17 files. The five-kinds page says five more came from screenshots, and 12 + 5 = 17 `inferred` |
| name of the fourth kind | "associative" on the grid | "shared" in the selection files for the same six pairs `verified`, and "shared" as the name of the kind on the five-kinds page |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**The SD3.5 grids are selected.** The third selection round kept rows where the joint shows the interaction and the product fails, and replaced the rest, per [its prompt](../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/pair-selection/round3-prompt.md). They show that these failures exist and say nothing about how common they are.

**The two model families are a mixed comparison.** SD v1-4 runs the joint at 7.5 and the product at 7.5 per prompt, a total of 15. SD3.5 runs 4.5 against 2.25 per prompt, a total of 4.5. The two also differ in architecture.

**A joint prompt that adds words carries more information than the product.** "A wet dog in the rain" names the wet fur that "a dog" and "rain" leave out.

> [!NOTE]
> The chest radiographs and the panels cropped from published papers are left off this page, because they are patient images or figures fetched from outside the project. They stay in their folders and are described by [the card](../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/README.md).

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. How hard each prompt pushes is taught on [the cats and dogs page](does-poe-of-cats-and-dogs-match-the-joint.md#how-hard-a-product-pushes).

**A row kept because it failed**

Flip 10 coins and keep only the ones that landed heads. Every coin you kept shows heads, and that says nothing about how often a coin lands heads. Selecting pairs works the same way. Here 6 pairs per grid were kept from 8 candidates and further replacements, using the rule that the product should fail, so "the product fails on 6 of 6 rows" is what the selection asked for.

**Kinds of composition**

Take "a dog" and "rain". Asking for both to appear is structural, asking for the dog to be under the rain is relational, and asking for the rain to wet the dog is causal. The same two words can be judged against different kinds, so the table above names the kind each pair was chosen to show.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| per-pair counts | [the five-kinds page](../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/five-interactions.md), read 2026-11-22 | `stated` |
| which model made each row | [the build sheet](../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/MANIFEST.md) | `stated` |
| why each model was chosen | scouting for [the CLEVR rows](../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/scouting/clevr-rows.md) and [the text-to-image rows](../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/scouting/text-to-image-rows.md) | `stated` |
| how pairs were chosen | [the selection brief](../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/pair-selection/brief.md) and [review rules](../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/pair-selection/review.md) | `stated` |
| 35 repeat pictures held out of the drain | 23 [tiles](../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/tiles), 8 [pairs](../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/pairs) and 4 [interaction grids](../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/interaction-grids), members of or matching the sheets placed above | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Find the generation scripts. The card names `scripts/joint_vs_poe_grid/`, which is in neither this repository nor its git history, and its `generated/` folder is absent.
- [ ] Add card entries for the two SD3.5 sheets, the two extra grids and the fourth selection round.
- [ ] Measure a failure rate on pairs chosen without looking at the product's output, if this comparison is to support a claim about how often each kind fails.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
