# 📒 Report: does the two-concept adapter compose three concepts?

**No verdict has been recorded.** The adapter is trained on pairs only, and this experiment asks
whether it still separates things when the product is given three prompts. It was run on one triple
of each of four kinds, at one seed. The pass bar [the review step](../paper-review/prze-4-three-concepts.md)
asks for has not been written, so the renders below are a description.

## Table of contents

- [What the adapter draws for three prompts](#what-the-adapter-draws-for-three-prompts)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## What the adapter draws for three prompts

Navigation: 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**On the three triples shown, the plain product draws one fused animal or drops a concept, and the
adapter draws separate bodies without reliably drawing all three things.** Every render starts from
the same noise at seed 10, with 50 steps and guidance 7.5 `stated` ([the render script](../../scripts/review_three_concepts.py)).

| | cat, dog, penguin | dog, chair, cat | cat, dog, umbrella |
|---|---|---|---|
| Plain product | `product.png` is on the cluster at `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/how-far-one-adapter-reaches/does-the-adapter-compose-three-concepts/tiles/animal-animal-animal/product.png` | `product.png` is on the cluster at `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/how-far-one-adapter-reaches/does-the-adapter-compose-three-concepts/tiles/animal-object-animal/product.png` | <a href="../../artifacts/results/the-adapter-on-cat-and-dog/how-far-one-adapter-reaches/does-the-adapter-compose-three-concepts/tiles/animal-animal-object/product.png"><img src="../../artifacts/results/the-adapter-on-cat-and-dog/how-far-one-adapter-reaches/does-the-adapter-compose-three-concepts/tiles/animal-animal-object/product.png" width="200" alt="One grey animal with a dog face and cat ears under a red umbrella."></a> |
| What it holds `verified` | one fused body | one toy-like animal, no chair | one animal and the umbrella |
| Adapter | <a href="../../artifacts/results/the-adapter-on-cat-and-dog/how-far-one-adapter-reaches/does-the-adapter-compose-three-concepts/tiles/animal-animal-animal/ours.png"><img src="../../artifacts/results/the-adapter-on-cat-and-dog/how-far-one-adapter-reaches/does-the-adapter-compose-three-concepts/tiles/animal-animal-animal/ours.png" width="200" alt="A large white dog, a small grey and white cat and a fluffy white shape. No penguin."></a> | <a href="../../artifacts/results/the-adapter-on-cat-and-dog/how-far-one-adapter-reaches/does-the-adapter-compose-three-concepts/tiles/animal-object-animal/ours.png"><img src="../../artifacts/results/the-adapter-on-cat-and-dog/how-far-one-adapter-reaches/does-the-adapter-compose-three-concepts/tiles/animal-object-animal/ours.png" width="200" alt="Two dogs lying together on a cushion. No cat and no chair."></a> | `ours.png` is not shown because it failed the subject check |
| What it holds `verified` | dog and cat, no penguin | two dogs, no cat, no chair | dog, two cats, umbrella |

> [!WARNING]
> The fourth kind of triple (teapot, lamp, book) and the whole sheet are off this page.
> `1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/how-far-one-adapter-reaches/does-the-adapter-compose-three-concepts/tiles/object-object-object/ours.png`
> and the three sheets built from it, `three_concepts.png`, `three_concepts-contact-sheet.png` and
> `tiles/three_concepts.png`, are not shown: holds a subject the pair ledger keeps off sheets.
> The paper's three-concepts figure file is byte-identical to `three_concepts.png`.

## Where the sources disagree

Navigation: ⬅️ [Previous](#what-the-adapter-draws-for-three-prompts) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Three statements about this experiment conflict, and each stands here unresolved.** The two paper
statements are at lines 889 and 1386 of [the paper source](../../../writing/restoring-plurality-in-compositional-diffusion_overleaf/iclr2027_conference.tex).

| What | One source | The other source |
|---|---|---|
| triples that include an object | the main text says the adapter "draws all three named things when the triple includes objects" | the figure caption says that for dog, chair and cat "it draws two dogs, without the cat or the chair", and the render shows that |
| how many animals it stops at | the main text says that with two or three animals "it stops at two" | the cat, dog and umbrella render holds three animals, which the caption calls "an extra small animal" |
| whether a bar came first | the [figure register](../../../writing/restoring-plurality-in-compositional-diffusion_overleaf/figures.md) describes this figure as judged "against a bar written first" | the [review step](../paper-review/prze-4-three-concepts.md) leaves its "pass bar written before the run" box unticked, and no bar is recorded there |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**One seed and one triple per kind cannot give a rate.** The instance counts the review step asks
for are absent, so no [compose rate](../../../context/world/compose-rate.md#what-a-compose-rate-is) exists for three concepts. The calls in the table are by eye from
1,024-pixel renders. The adapter was never trained on a triple, so these renders say nothing about
what a three-concept adapter would do.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers are invented. The two-prompt version is taught in
[the plain product on the four-pairs page](does-one-adapter-compose-four-pairs.md#concepts-this-uses).

**The product with three prompts**

Take one number from each prediction: empty prompt u = 1, "cat" = 3, "dog" = 2, "penguin" = 2, with
guidance weight w = 2. Each prompt adds its own difference from the empty prompt: 2, 1 and 1. The
product is 1 + 2 × (2 + 1 + 1) = 9. The rule in [the render script](../../scripts/review_three_concepts.py)
is the same sum, u + w(c1 + c2 + c3 − 3u), run on every number of the prediction at w = 7.5.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| triples, seed, adapter (v58-03-self at step 30,000) | the JSON file beside each tile, read 2026-11-22 | `verified` |
| how the renders were made | [the render script](../../scripts/review_three_concepts.py) | `stated` |
| what each tile holds | each tile opened at full size, 2026-11-22 | `verified` |
| the card | this result has no card | owed |
| 5 pictures held out of the drain | the other tiles and layouts in this result's folder. They repeat a picture placed above, or could not be cleared against the subjects the [pair ledger](../../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md) keeps off sheets, so the folder is not linked here | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Decide what happens to the paper's three-concepts figure, whose teapot, lamp and book column holds the render kept off this page.
- [ ] Write the pass bar the review step asks for, then count instances on the same seeds.
- [ ] Reconcile the main text with the caption on the two conflicts above.
- [ ] Write the card for this result.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
