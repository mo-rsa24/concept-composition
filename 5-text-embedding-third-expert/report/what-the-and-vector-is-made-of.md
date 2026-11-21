# 📒 Report: what the "and" vector is made of, and whether it works as a third expert

**No single verdict is recorded for this question.** The "and" vector is the difference between the
text embedding of a joint prompt such as "dog and cat" and the embeddings of its two halves. This page
asks whether that difference, or something built to replace it, can join
[product of experts](../../context/world/poe-composition.md#what-poe-composition-is) (PoE, which adds
the noise predictions of separate prompts during drawing) as a third expert and bring back one cat
beside one dog. Each test below has its own pass mark and result. The learned token's picture test has
no recorded calls. The verdict across the whole series is in
[the text-space third expert finding](can-a-text-space-third-expert-restore-composition.md).

## Table of contents

- [What the vector holds at each position](#what-the-vector-holds-at-each-position)
- [The vector and written phrases as a third expert](#the-vector-and-written-phrases-as-a-third-expert)
- [Which seeds the joint prompt passes](#which-seeds-the-joint-prompt-passes)
- [Steering the starting noise](#steering-the-starting-noise)
- [A learned token in place of the vector](#a-learned-token-in-place-of-the-vector)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## What the vector holds at each position

Navigation: 📋 [TOC](#table-of-contents) | [Next](#the-vector-and-written-phrases-as-a-third-expert) ➡️

**The plain difference is dominated by the start token, and the repair that cancels it leaves a
sequence unlike a real prompt.** E is a prompt's embedding, 77 token positions long. The plain
difference is x_raw = E("a cat and a dog") − E("a cat") − E("a dog"), and the repair is
x_fix = x_raw + E("").

| Size at each position | Which word sits at each position |
|---|---|
| <a href="../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/a-cat-and-a-dog-norms.png"><img src="../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/a-cat-and-a-dog-norms.png" width="200" alt="Two bar charts of vector length against token position 0 to 76, with bars for x_raw, x_fix and the joint prompt in each text encoder."></a> | <a href="../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/token-alignment-matrix.png"><img src="../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/token-alignment-matrix.png" width="200" alt="Table of the token each prompt holds at positions 0 to 7, beside the length of x_raw, x_fix and the joint prompt at that position."></a> |
| The y-axis is a position's length (its norm), the x-axis is the token position, and each group of bars is one position. Numbers are from [the norms file](../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/a-cat-and-a-dog-norms.json) `verified`. | Each row is one position. "dog" sits at position 5 of the joint prompt and position 2 of "a dog", so the subtraction there mixes different words `verified` (read on the picture). |

| Where, for "a cat and a dog" | x_raw length | x_fix length | Joint prompt length |
|---|---|---|---|
| start token, smaller text encoder (CLIP ViT-L) | 1190.0 | 0.0 | 1190.0 |
| median padding position, smaller encoder | 23.9 | 15.1 | 19.8 |
| median padding position, larger encoder (OpenCLIP bigG) | 22.6 | 9.7 | 19.8 |

**The repair missed its own mark.** It was meant to bring x_fix's padding under 0.1 of the joint
prompt's. The ratio is 0.766 in the smaller encoder and 0.487 in the larger `verified` (norms file,
`x_fix_over_joint_pad_median`). For the prompt "dog and cat" the same ratios are 0.572 and 0.555
`verified` ([its norms file](../artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/token-norms-x-pos.json)).

## The vector and written phrases as a third expert

Navigation: ⬅️ [Previous](#what-the-vector-holds-at-each-position) | 📋 [TOC](#table-of-contents) | [Next](#which-seeds-the-joint-prompt-passes) ➡️

**No arithmetic vector or phrase reached the pass mark of one clear cat beside one clear dog on 4 of
the 7 counted seeds.** Seeds 1 to 7 count because the joint prompt draws two dogs on seed 8. On all
three sheets each row is one seed and every column starts from the same noise.

| The arithmetic vectors | Written phrases | x_fix at rising weight |
|---|---|---|
| <a href="../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/dog-and-cat-sheet.png"><img src="../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/dog-and-cat-sheet.png" width="200" alt="Twelve columns by eight seeds: the references, then x_raw, x_fix and x_aff alone and in the product, then two phrases."></a> | <a href="../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/dog-and-cat-phrases-sheet.png"><img src="../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/dog-and-cat-phrases-sheet.png" width="200" alt="Thirteen columns by eight seeds: the references, seven phrases as the third expert and x_fix at two lower weights."></a> | <a href="../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/dog-and-cat-weight-sweep-sheet.png"><img src="../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/dog-and-cat-weight-sweep-sheet.png" width="200" alt="Nine columns by eight seeds: the references, then PoE with x_fix at weights 0.1, 0.25, 0.5, 0.75 and 1."></a> |
| x_raw alone and in the product is a flat grey tile on all 8 seeds `verified` (seen on the sheet). | "a dog next to a cat" and the joint prompt give the most two-animal tiles `stated` ([the idea map](../artifacts/ideas/the-and-vector-as-a-third-expert/IDEA_MAP.md)). | Seed 1 is wallpaper from weight 0.25 upward and seed 8 only at weight 1 `verified` (seen on the sheet). |

| Third expert beside "dog" and "cat" | Clear cat and dog, of 7 | What the tiles show | Called by |
|---|---|---|---|
| x_raw | 0 | a flat grey tile on all 8 seeds | Claude |
| x_fix | 1 or 0 ([disagreement](#where-the-sources-disagree)) | a kitten and a puppy on seed 3, wallpaper on seeds 1, 4 and 8, one hybrid on the rest | Claude |
| x_aff, which is x_raw + 2E("") | 0 | no seed with one of each | Claude |
| "single animal" | 0 | one animal on all 8 seeds | Claude |
| "two animals" | 0 or 1 ([disagreement](#where-the-sources-disagree)) | a dog and a cat side by side on seed 4, two animals on seeds 2 and 8, one on the rest | Claude |
| "a dog next to a cat" | 0 | two hybrid animals on seeds 1, 3, 4, 6, 7 and 8 | the owner |
| the joint prompt "dog and cat" | 0 | two hybrid animals on seeds 1, 2, 3, 6 and 8 | the owner |

All calls are `stated` from claim 2 of the idea map. It records the other four phrases and x_fix at
weights 0.5 and 0.25 as at most one clear pair each, on calls nobody has checked.

## Which seeds the joint prompt passes

Navigation: ⬅️ [Previous](#the-vector-and-written-phrases-as-a-third-expert) | 📋 [TOC](#table-of-contents) | [Next](#steering-the-starting-noise) ➡️

<a href="../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/joint-seeds-to-pass.png"><img src="../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/joint-seeds-to-pass.png" width="200" alt="Contact sheet of 32 renders of the joint prompt dog and cat, four rows of eight seeds."></a>

**The joint prompt draws one whole dog and one whole cat on 28 of 32 seeds.** Each tile is one seed.
Claude made these calls from the sheet, and they are open to the owner's veto `verified` (counted in
[the seed calls file](../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/joint-seed-calls.json)).

| Call | Seeds | What is there |
|---|---|---|
| pass | 28 seeds: 1 to 7, 9 to 12, 15 to 21, 23 to 32 | one dog and one cat, each whole |
| fail | 8 and 14 | two dogs |
| left for the owner | 13 | a dog, a cat and a small kitten |
| left for the owner | 22 | a dog and an animal that reads as either a kitten or a puppy |

## Steering the starting noise

Navigation: ⬅️ [Previous](#which-seeds-the-joint-prompt-passes) | 📋 [TOC](#table-of-contents) | [Next](#a-learned-token-in-place-of-the-vector) ➡️

<a href="../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/separation-steering-sheet.png"><img src="../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/separation-steering-sheet.png" width="200" alt="Five columns by eight seeds: plain PoE, PoE steered at small, medium and large step sizes, and the joint prompt."></a>

**Steering gave a clear dog and cat on 0 of 7 seeds at every step size, against a mark of 4, so the
question is not supported** `stated` ([the results card](../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/README.md)).
At steps 0 to 9, a gradient step moves the image latent to reduce the overlap between where the dog
expert and the cat expert attend. Rows are seeds, and each column is one step size.

| Step size, with its share of the latent's length | Clear pair, of 7 | What the card records |
|---|---|---|
| none | 0 | plain PoE |
| small, 2.5e4 (1%) | 0 | one animal, a wallpaper (seeds 1 and 4) or a hybrid |
| medium, 1.2e5 (5%) | 0 | flat graphics or saturated blocks |
| large, 5e5 (20%) | 0, plus one borderline tile on seed 6 | flat graphics or saturated blocks |

The attention maps are almost flat at these steps, with a largest value of 0.0012 against 0.00098 for
a flat map. About 98% of each update comes from the terms that sharpen each map, so the step barely
separates them `stated` (the card). The run is recorded in [the steering walk](../artifacts/drips/separation-steering/EXECUTE_WALK.md).

## A learned token in place of the vector

Navigation: ⬅️ [Previous](#steering-the-starting-noise) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**One fitted token removed under 10% of the gap at every checked step, so the run's stop rule fired.
The picture test that would give its verdict has no recorded calls.** The token `<and>` has one
trainable row of numbers in each text encoder. It was fitted on 182 cached examples over 48 animal and
object pairs, with cat and dog and seeds 9 to 16 held out `stated`
([the learned-token plan](../../3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/today-a-learned-and-token.md)).

| Before training | After 8,000 updates |
|---|---|
| <a href="../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/learned-and-token-sheet-iter00000.png"><img src="../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/learned-and-token-sheet-iter00000.png" width="200" alt="Five columns by held-out seeds 9 to 16: dog alone, cat alone, the joint prompt, plain PoE, and PoE with the untrained token."></a> | <a href="../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/learned-and-token-sheet-iter08000.png"><img src="../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/learned-and-token-sheet-iter08000.png" width="200" alt="The same five columns and seeds, with the last column using the token after 8,000 updates."></a> |
| Rows are held-out seeds 9 to 16. The last column is PoE with the untrained token. | The plan records one animal mixing dog and cat features on every seed in both PoE columns `stated`. |

| Share of the gap removed, mean of seeds 9 to 16 | Step 0 | Step 10 | Step 25 | Step 40 |
|---|---|---|---|---|
| before training | −0.142 | −0.064 | −0.086 | −0.052 |
| after 8,000 updates | −0.070 | +0.013 | −0.005 | −0.008 |
| needed at any one step to continue | above 0.10 | above 0.10 | above 0.10 | above 0.10 |

A share of 1 closes the whole gap between plain PoE and the joint prompt, and 0 closes none. These are
`stated` from the plan's record of W&B run `c6f1pmmq`. [The supervisor deck](../../artifacts/decks/supervisor-meeting/README.md#2-training-fit-the-third-expert-instead-of-building-it)
lists the other token variants, and all of them stop on the same rule.

| Lion and meerkat, a training pair | Typewriter and cactus, a training pair | Elephant and penguin, held out |
|---|---|---|
| <a href="../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/learned-and-token-a_lion__x__a_meerkat-over-time.png"><img src="../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/learned-and-token-a_lion__x__a_meerkat-over-time.png" width="200" alt="Six seeds down, with columns for the joint prompt, plain PoE and the token at five training checkpoints, for a lion and a meerkat."></a> | <a href="../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/learned-and-token-a_typewriter__x__a_cactus-over-time.png"><img src="../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/learned-and-token-a_typewriter__x__a_cactus-over-time.png" width="200" alt="Six seeds down, with columns for the joint prompt, plain PoE and the token at five training checkpoints, for a typewriter and a cactus."></a> | <a href="../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/learned-and-token-an_elephant__x__a_penguin-over-time.png"><img src="../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/learned-and-token-an_elephant__x__a_penguin-over-time.png" width="200" alt="Six seeds down, with columns for the joint prompt, plain PoE and the token at five training checkpoints, for an elephant and a penguin."></a> |
| Rows are seeds 1 to 6. Columns run from the joint prompt and plain PoE through 300, 1,000, 2,000, 4,000 and 8,000 updates. | Same layout, seeds 1 to 6. The deck reads these objects as still fused `stated`. | Same layout, seeds 9 to 14. The deck calls this creature an elephant-penguin `stated`. |

**Three more sheets at 2,000 updates are owed a caption, because no record says what separates their
runs.** Their column headers give the third prompt as `a cat <and> a dog` on run6a and run6c2, and as
`<and>` on run6b `verified`.

| run6a | run6b | run6c2 |
|---|---|---|
| <a href="../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/learned-and-token-run6a-iter02000.png"><img src="../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/learned-and-token-run6a-iter02000.png" width="200" alt="Held-out seeds 9 to 16 with the token inside the sentence at 2,000 updates. Caption owed."></a> | <a href="../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/learned-and-token-run6b-iter02000.png"><img src="../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/learned-and-token-run6b-iter02000.png" width="200" alt="Held-out seeds 9 to 16 with the bare token at 2,000 updates. Caption owed."></a> | <a href="../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/learned-and-token-run6c2-iter02000.png"><img src="../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/learned-and-token-run6c2-iter02000.png" width="200" alt="Held-out seeds 9 to 16 with the token inside the sentence at 2,000 updates. Caption owed."></a> |

## Where the sources disagree

Navigation: ⬅️ [Previous](#a-learned-token-in-place-of-the-vector) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Seven recorded facts conflict, and each is shown here unresolved.**

| What | One source | The other source |
|---|---|---|
| the x_fix tile on seed 3 | the idea map's first call is a kitten and a puppy, and [the finding](can-a-text-space-third-expert-restore-composition.md#2-co-occurrence-comes-back-identity-does-not) and the deck count 1 clear pair | the idea map's phrase search calls the same tile two animals with at least one hybrid, and its summary row records 0 clear pairs for x_fix |
| the "two animals" tile on seed 4 | the idea map's first call is a dog and a cat side by side | the finding counts 0 clear pairs for "two animals" |
| x_fix at low weight | the idea map says weights 0.1 to 0.5 return PoE's hybrid on every seed, and 0.75 and 1 turn seeds 1, 4 and 8 to wallpaper | the weight sheet shows seed 1 as wallpaper from 0.25, seed 4 as wallpaper at every weight including plain PoE, and seed 8 as wallpaper only at 1 |
| small steering steps | the card says they keep the photograph style | the sheet shows seeds 5, 6, 7 and 8 as drawings or framed illustrations at small steps |
| whether the learned token ran | the idea map's route table says "not run" | the plan records 2,000-update and 8,000-update runs, and their sheets exist |
| whether the weight sweep ran | the idea map's runs table says "queued" | its sheet exists, and the same map quotes calls on it |
| the size of the steering gradient | the card gives 6.6e-5 to 1.8e-4 for the whole loss | the idea map gives 4e-5 to 2e-4 for the two sharpening terms, a wider range than the whole loss |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

> [!WARNING]
> Every clear-pair count is a by-eye call. Claude's first calls on the two strongest phrases overcalled
> from a downscaled sheet, and the idea map says its other calls lean the same way. Only the two rows
> marked "the owner" were checked.

**One pair carries the third-expert tests.** The sheets cover dog and cat on 7 counted seeds, so they
say nothing about other pairs.

**The steering test used one loss.** A loss on the overlap term alone was not run.

**Share removed reads noise predictions at four steps.** It does not show whether a picture composes,
which is what the uncalled picture test would measure.

**Plain PoE does not reproduce pixel for pixel here.** The original script, rendered again, differed
from its own earlier seed 1 tile by a mean of 13.9 grey levels on a 0 to 255 scale `stated` (the card).

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. The project numbers are from the files named on this page.

**A vector's length (its norm)**

The list (3, 4) has length √(3² + 4²) = 5. You square each number, add them and take the root. Each
token position here is one such list, so "length at a position" is this sum over all its numbers. The
start token's length is 1190 and the word "cat" in the joint prompt is 16.3. More on the
[norm](../../../../Obsidian/Mathematics/Linear%20Algebra/Norm.md#where-to-learn-this).

**The guided sum**

Say the empty prompt predicts 1, "dog" predicts 3, "cat" predicts 2 and the guidance weight is 2. The
sum is 1 + 2 × (3 − 1) + 2 × (2 − 1) = 7. A third expert predicting 1.1 adds only 2 × 0.1 = 0.2, and
one predicting 5 adds 8. Each expert adds weight × (its prediction − the empty prompt's). The weight
here is 7.5, so any error in a third expert is multiplied by 7.5. The
[guided sum and its error](../../../../goal-setting/learning/deep-learning/diffusion-models/text-embedding-arithmetic-in-sdxl/plans/12-guided-sum-and-its-error.md)
walks this, and [classifier-free guidance](../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Classifier-Free%20Guidance.md#where-to-learn-this)
is the rule it extends.

**Attention overlap**

Take two maps of two pixels each. If both are (0.5, 0.5), the overlap is 0.5 × 0.5 + 0.5 × 0.5 = 0.5.
If one is (1, 0) and the other (0, 1), the overlap is 0. Overlap is each pixel's two values multiplied,
then summed. The maps here have 32 × 32 = 1,024 cells, so a flat map holds 1 / 1,024 = 0.00098 in each,
and the measured peak of 0.0012 is barely above that. The maps come from
[cross-attention](../../../../goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/plans/04-cross-attention-asking-for-a-seven.md).

**Share of the gap removed**

If the gap to the target has length 10 and a third expert leaves 8 of it, the share is 1 − 8 / 10 = 0.2.
If it leaves 11, the share is 1 − 11 / 10 = −0.1, which is further away than where it started. The
share is 1 minus the length left over, divided by the length of the gap. At step 0 after 8,000 updates
the mean share of −0.070 means the leftover averages 1.070 times the gap. The learning tree has no
page on this measure, and the [learned-token plan](../../3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/today-a-learned-and-token.md#the-test-written-before-any-render)
holds its definition.

**A learned token**

Suppose a model outputs 2 × x and the target is 7. Training nudges only x until it reaches 3.5, and
everything else stays frozen. Fitting one embedding this way is called textual inversion. Here x is
the `<and>` token's row in each text encoder, updated 8,000 times. More in
[what textual inversion optimises](../../../../goal-setting/learning/deep-learning/diffusion-models/text-embedding-arithmetic-in-sdxl/plans/15-what-textual-inversion-optimises.md).

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| position lengths | [the norms file](../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/a-cat-and-a-dog-norms.json), read 2026-11-21 | `verified` |
| calls on the third-expert sheets | [the "and" vector idea map](../artifacts/ideas/the-and-vector-as-a-third-expert/IDEA_MAP.md), claim 2 | `stated` |
| joint prompt seed calls | [the seed calls file](../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/joint-seed-calls.json), read 2026-11-21 | `verified` |
| steering pass mark and result | [the results card](../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/README.md) and [the steering walk](../artifacts/drips/separation-steering/EXECUTE_WALK.md) | `stated` |
| learned token runs | [the learned-token plan](../../3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/today-a-learned-and-token.md) and [the supervisor deck](../../artifacts/decks/supervisor-meeting/README.md) | `stated` |
| a related question: does a third word bring a wet camel back? | [the third-concept walk](../artifacts/drips/today-the-third-concept/EXECUTE_WALK.md) and [its plan](../../3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/today-the-third-concept.md); the tiles are rendered, the calls and verdict are not recorded | `stated` |
| a related question: does the LoRA's correction have a text embedding? | [the residual idea's card](../artifacts/ideas/text-embedding-of-the-residual/README.md) and [its idea map](../artifacts/ideas/text-embedding-of-the-residual/IDEA_MAP.md); parked with no verdict | `stated` |
| 19 learned-token checkpoint sheets held out of the drain | 15 per-pair sheets, whose checkpoints the three over-time strips above repeat, and 4 sheets at 300 to 4,000 updates, which sit between the before and after sheets. All stay in [the results folder](../artifacts/results/the-and-vector/what-the-and-vector-is-made-of) | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Settle the x_fix tile on seed 3 and the "two animals" tile on seed 4 from their full-resolution renders. Both stay under the pass mark of 4 either way.
- [ ] Call joint prompt seeds 13 and 22, which are left for the owner.
- [ ] Run the learned-token plan's picture test. Its step 6 has no card, renders or calls.
- [ ] Record what separates the run6a, run6b and run6c2 sheets.
- [ ] Add card entries for the phrase sheet, the weight sheet, the joint seed sheet, the token matrix and every learned-token sheet. The [results card](../artifacts/results/the-and-vector/what-the-and-vector-is-made-of/README.md) covers only the norms, the first sheet and steering.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
