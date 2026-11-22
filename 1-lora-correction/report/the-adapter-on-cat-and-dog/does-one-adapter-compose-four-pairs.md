# 📒 Report: does one adapter compose four pairs at one seed?

**No verdict has been recorded for this picture.** It is a review sheet built while the paper's
Figure 5 was being rebuilt: four pairs the adapter never trained on, five ways of drawing them, one
seed. It asks whether a single adapter, the one [the review protocol](../paper-review/protocol.md)
named, separates all four.

## Table of contents

- [What each method draws at seed 10](#what-each-method-draws-at-seed-10)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## What each method draws at seed 10

Navigation: 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

<a href="../../artifacts/results/the-adapter-on-cat-and-dog/how-far-one-adapter-reaches/does-one-adapter-compose-four-pairs/figure5-seed10-five-columns.png"><img src="../../artifacts/results/the-adapter-on-cat-and-dog/how-far-one-adapter-reaches/does-one-adapter-compose-four-pairs/figure5-seed10-five-columns.png" width="200" alt="Four rows of pairs by five columns of methods at seed 10: joint prompt, plain product, CO3, product with Langevin, and the adapter."></a>

**At seed 10 the adapter and CO3 show both things on all four rows, and the adapter adds a third
animal on one of them.** Rows are the pairs and columns are the methods. Each cell below is a call by
eye from this sheet, whose tiles are about 296 pixels wide `verified`.

| Pair | Joint prompt | Plain product | CO3 | Product + Langevin | Adapter |
|---|---|---|---|---|---|
| cat and dog | two dogs, no cat | one fused face | both, apart | one dog | a cat and a dog, touching |
| elephant and penguin | elephant, penguin-shaped bird, small bird | one fused body | both, apart | elephant with penguin markings | elephant, penguin, extra small elephant |
| tent and snowy mountain | both | a tent-shaped mountain | both | snowy peak above a dark dome | both |
| lighthouse and stormy sea | both | both | both | both | both |

The adapter is v58-03-self at training step 30,000, switched on for all 50 drawing steps. The
Langevin column uses step size c = 3 and k = 20 corrector steps at all 50 noise levels. CO3 runs at
its published default, and its later resampling noise comes from the seed number, so its column
shares only the first latent with the others `stated` (the sheet's footer and its card).

## Where the sources disagree

Navigation: ⬅️ [Previous](#what-each-method-draws-at-seed-10) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Four records describe this figure differently.** The first is explained by a commit. The other
three are shown as they stand.

| What | One source | The other source |
|---|---|---|
| which animal pairs Figure 5 prints | this sheet and the [layout sheet](../paper-review/layout-sheet.md) name cat and dog, elephant and penguin | the paper's Figure 5 file prints cat and lion, elephant and turtle. Commit `1a891ba7` (2026-11-11) changed those rows, and the layout sheet was not updated |
| how many methods | this sheet and the layout sheet have five columns, and the card says ToMe left the figure | the paper's caption and its Figure 5 file have six, with ToMe. The [figure register](../../../writing/restoring-plurality-in-compositional-diffusion_overleaf/figures.md) says four methods |
| whether the plain product fails on all four | the figure register says the adapter composes all four "where the plain product does not" | on this sheet the plain product draws both the lighthouse and the sea |
| the Langevin column on the tent row | the card says it "keeps one of them on the first three" rows | the tent row shows a snowy peak above a dark blue dome that may be the tent |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**One seed per pair gives no rate.** The rate over the 53 held-out pairs is in
[how every SDXL method compares](../against-other-methods/how-does-every-sdxl-method-compare-on-the-held-out-pairs.md).
The joint prompt at this seed draws two dogs and no cat, so it is a poor reference on that row, as
[the finding on the joint prompt](../is-the-adapter-just-copying-the-joint-prompt/does-the-joint-prompt-always-draw-both-concepts.md)
records for other cells. The four pairs are absent from the training list of the
[v58 pair pool](../../../artifacts/_shared/cross_pair_pool_configs/pair_pool_v58.yaml), though the
lighthouse appears there in another pair.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers are invented. The sibling pages link here for these ideas.

**The plain product**

A diffusion model predicts a number for every point of the picture. Take one point: with an empty
prompt the model says u = 1, with "a cat" it says 3, with "a dog" it says 2. Each prompt's push is its
difference from the empty prompt, here 2 and 1. With guidance weight w = 2 the
[product of experts](../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Product%20Of%20Experts.md#where-to-learn-this)
gives 1 + 2 × (2 + 1) = 7. The word "product" comes from multiplying the two models' probabilities,
which adds their predictions. This project runs the same sum at w = 7.5, and the
[context entry](../../../context/world/poe-composition.md#what-poe-composition-is) explains why it tends to draw one fused animal.

**The joint prompt**

The joint prompt is one sentence naming both things, "a cat and a dog", given to the model once. In
the toy it would be a single prediction, say 5, with no sum. The adapter never sees this prompt while
drawing, so a column of joint-prompt renders is a reference for what the model can do when asked directly.

**The same starting noise**

A picture starts as random noise, and the seed picks which noise. Baking two loaves from one lump of
dough with different recipes means any difference comes from the recipe. Here a row shares the noise
of `a_cat__x__a_dog` at seed 10, so a difference along the row comes from the method, apart from
CO3's later resampling noise.

**The adapter**

A [LoRA](../../../../../Obsidian/Artificial%20Intelligence/Large%20Language%20Models/LoRA.md#where-to-learn-this)
is a small set of extra weights added to the model. Here it learns the correction the plain product
leaves out, as the [context entry on the corrector](../../../context/world/lora-corrector.md#what-the-lora-corrector-is) describes.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| what each cell holds | the sheet, opened 2026-11-22 | `verified` |
| settings and how the sheet was assembled, 2026-11-11 | `does-one-adapter-compose-four-pairs/README.md` in this result's folder | `stated` |
| the adapter choice | [review protocol](../paper-review/protocol.md), rule 1 | `stated` |
| the change to Figure 5's rows | commit `1a891ba7`, 2026-11-11 | `verified` |
| 14 pictures held out of the drain | the other tiles and layouts in this result's folder. They repeat a picture placed above, or could not be cleared against the subjects the [pair ledger](../../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md) keeps off sheets, so the folder is not linked here | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Update the layout sheet's and the figure register's rows for Figure 5 to what the paper prints.
- [ ] Decide whether the sentence "where the plain product does not" stays, given the lighthouse row at seed 10.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
