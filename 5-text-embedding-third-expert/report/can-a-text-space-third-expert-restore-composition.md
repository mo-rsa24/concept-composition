# ⚪ A text-space third expert restores co-occurrence but never identity, and a plain phrase does it just as well

Eight ways of building a third expert beside "dog" and "cat" have now been rendered on the same 8
seeds. The best of them puts two separate animals in the frame on 6 of the 7 judgeable seeds, which
is real movement. **None of them, on any seed, draws one whole cat beside one whole dog.** The
pass mark, written before the first tile rendered, was 4 of 7.

## Table of contents

- [The claim and where it came from](#the-claim-and-where-it-came-from)
- [The bar, written before the renders](#the-bar-written-before-the-renders)
- [What was tried, in order](#what-was-tried-in-order)
- [1. The arithmetic recovers the word](#1-the-arithmetic-recovers-the-word)
- [2. Co-occurrence comes back; identity does not](#2-co-occurrence-comes-back-identity-does-not)
- [3. The product is worse than the third expert alone](#3-the-product-is-worse-than-the-third-expert-alone)
- [What this cannot say](#what-this-cannot-say)
- [Provenance](#provenance)
- [Still open](#still-open)
- [Cross-references](#cross-references)

## The claim and where it came from

Navigation: 📋 [TOC](#table-of-contents) | [Next](#the-bar-written-before-the-renders) ➡️

**The claim under test.** [Product-of-experts](../../context/world/poe-composition.md#what-poe-composition-is) draws one fused animal where the joint prompt "dog and
cat" draws two. If the difference between the joint prompt's embedding and its two halves carries
what the joint adds, feeding that difference in as a third expert should restore the pair.

**Where it came from.** Claim 2 of
[the "and" vector as a third expert](../artifacts/ideas/the-and-vector-as-a-third-expert/IDEA_MAP.md),
in the spirit of word-vector analogies: king − man + woman ≈ queen.

## The bar, written before the renders

Navigation: ⬅️ [Previous](#the-claim-and-where-it-came-from) | 📋 [TOC](#table-of-contents) | [Next](#what-was-tried-in-order) ➡️

One whole cat beside one whole dog on at least **4 of the 7 judgeable seeds**. Seeds 1 to 8 from the
same cached starting noise; seed 8 is not judgeable because the joint prompt itself draws two dogs
there. The mark was not changed at any point, including after the counts came in.

## What was tried, in order

Navigation: ⬅️ [Previous](#the-bar-written-before-the-renders) | 📋 [TOC](#table-of-contents) | [Next](#1-the-arithmetic-recovers-the-word) ➡️

| Third expert | What it is |
|---|---|
| x_raw | E(joint) − E(a) − E(b), across all 77 token slots |
| x_fix | x_raw + E(""), which cancels the start token everywhere |
| x_aff | x_raw + 2E(""), whose coefficients sum to 1 |
| seven written phrases | "a dog next to a cat", "side by side", "together", "two different animals", "a pair of animals", "two animals", and the joint prompt itself |
| latent steering | not a third expert: a gradient step on the starting noise |
| x_pos | the joint prompt's own sequence with each animal's word subtracted only at the slot holding that word |
| x_and | the empty prompt's scaffolding with only what the joint adds written into the word slots |

## 1. The arithmetic recovers the word

Navigation: ⬅️ [Previous](#what-was-tried-in-order) | 📋 [TOC](#table-of-contents) | [Next](#2-co-occurrence-comes-back-identity-does-not) ➡️

The premise under every variant above is that E("A and B") − E("A") carries B. It does. Over 200
random pairs against a 276-noun dictionary where chance is a median rank of 138, the held-out word
comes back at **median rank 1**, top-1 75.0% on the pooled summary vector and 98.5% on the content
mean, with a decoy word at chance in both. Numbers in
[premise-word-recovery.json](../artifacts/results/can-a-positional-and-vector-restore-composition/premise-word-recovery.json),
read in [the card](../artifacts/results/can-a-positional-and-vector-restore-composition/README.md).

🖼️ **Figure wanted:** the rank of the held-out word as a histogram, one bar per rank bucket, the
subtracted and unsubtracted readouts side by side with the decoy control behind them, from
`premise-word-recovery.json`. Save as
`5-text-embedding-third-expert/artifacts/results/can-a-positional-and-vector-restore-composition/premise-word-recovery.png`.

**So the failures below are not caused by text arithmetic being meaningless.** Read the content row
with care: it averages the word slots, so subtracting A is partly bound by arithmetic to land near B.
The pooled row is the one that is not.

## 2. Co-occurrence comes back; identity does not

Navigation: ⬅️ [Previous](#1-the-arithmetic-recovers-the-word) | 📋 [TOC](#table-of-contents) | [Next](#3-the-product-is-worse-than-the-third-expert-alone) ➡️

Scored on the ladder (one animal, one fused animal, two separate animals, two whole species) rather
than on clear pairs alone, over the 7 judgeable seeds:

| Third expert | Two separate animals | One whole cat beside one whole dog |
|---|---|---|
| none, plain PoE | 0 | 0 |
| x_raw | 0, every tile flat grey | 0 |
| x_fix | 1 | 1 |
| "two animals" | 2 | 0 |
| the joint prompt itself | 4 | 0 |
| "a dog next to a cat" | 5 | 0 |
| x_pos | 6 | 0 |
| x_and | 0 | 0 |

**The second column never moves.** The first column moves a long way, and the largest single jump is
from writing an English sentence, not from any arithmetic: "a dog next to a cat" reaches 5 of 7, and
every construction added since buys at most one more seed.

**Why x_raw and x_fix produced no picture at all.** x_raw's start-token slot reaches a norm of 1190
where a word sits at about 15, and that error is multiplied by the guidance weight of 7.5. x_fix
cancels that slot exactly but leaves the padding at 0.57 of a real prompt's and the word slots above
a real prompt's. Per-slot norms for all four sequences in both encoders. Position 0 is off the scale for x_raw at
1190; the green bars sit exactly on the grey ones at every padding slot, which is x_pos matching the
joint prompt there by construction:

<a href="../artifacts/results/the-positional-and-vector/token-norms-x-pos.png"><img src="../artifacts/results/the-positional-and-vector/token-norms-x-pos.png" width="320" alt="per-position norm, four sequences, both encoder slices"></a>

The whole series in one picture, three seeds down and eight columns across from the same noise.
Read left to right: a grey tile, then wallpaper or one hybrid, then two animals on some seeds, then
two animals on seven of eight, and no column after the first with one whole cat beside one whole dog:

<a href="../artifacts/results/the-positional-and-vector/how-the-third-expert-changed.png"><img src="../artifacts/results/the-positional-and-vector/how-the-third-expert-changed.png" width="320" alt="every third expert tried, three seeds down, eight columns across"></a>

At 300 px a tile that picture shows the trajectory only; every identity call was made from the
full-resolution tiles.

## 3. The product is worse than the third expert alone

Navigation: ⬅️ [Previous](#2-co-occurrence-comes-back-identity-does-not) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-say) ➡️

x_pos edits 2 of the joint prompt's 77 slots and leaves the other 75 untouched, so it is close to the
joint prompt and renders like it. That makes it an accidental but sharp instrument. Mean absolute
pixel difference from the joint prompt's own render, 0 to 255, averaged over the 8 seeds:

| | distance from the joint prompt's image |
|---|---|
| x_pos used alone as the only condition | **23.43** |
| PoE(dog, cat, x_pos) | 39.16 |
| plain PoE(dog, cat) | 47.05 |

**Adding the two single-animal experts moves the picture away from the target.** The same ordering
appears in the by-eye calls: x_pos alone gives a clear pair on 2 of 7 seeds, and 0 once the dog and
cat experts are added. Per-seed comparisons:
[per-seed/](../artifacts/results/can-a-positional-and-vector-restore-composition/per-seed).

This agrees with the sampler check already on file, where the exact target placed in the third slot
at every step renders the joint image to within 2.23. The third slot is not the limitation.

## What this cannot say

Navigation: ⬅️ [Previous](#3-the-product-is-worse-than-the-third-expert-alone) | 📋 [TOC](#table-of-contents) | [Next](#provenance) ➡️

**One pair and seven judgeable seeds.** A count of 2 or 3 would have been inconclusive. The zeros are
firm only because they are far from the mark.

**One weight.** Every third expert enters at the run's guidance scale. Only x_fix had a weight sweep,
and it returned plain PoE's hybrid at low weights and wallpaper at high ones.

**By-eye calls.** Identity was called from full-resolution tiles by Claude and confirmed by the user
after three calls were queried and re-checked against the joint prompt's own cat on the same seed.
No automatic scorer was used, because the project's [instance-count scorer](../../context/world/compose-rate.md#words-this-file-uses) cannot tell a cat from a
dog.

**Nothing about a correction inside the denoiser.** This is an inference-only result.

## Provenance

Navigation: ⬅️ [Previous](#what-this-cannot-say) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

Renders on mscluster108 outside Slurm, 2026-11-04, from `5-text-embedding-third-expert/scripts/and_vector_sheet.py`; earlier
columns from runs 2 to 7 of the idea map on mscluster106 and mscluster109, 2026-10-27 and 2026-10-28.
Token norms from `5-text-embedding-third-expert/scripts/and_vector_norms.py`, premise test from `5-text-embedding-third-expert/scripts/and_vector_premise.py`,
both text encoders only. Tiles under
`/datasets/mmolefe/poe_repair_min/outputs/concept_pair_sheet/and_vector/dog__x__cat/`. Calls in
[calls.json](../artifacts/results/can-a-positional-and-vector-restore-composition/calls.json). Walked in
[today-the-positional-and-vector.md](../../artifacts/plans/completed/today-the-positional-and-vector.md).

## Still open

Navigation: ⬅️ [Previous](#provenance) | 📋 [TOC](#table-of-contents) | [Next](#cross-references) ➡️

**Whether any embedding can hit the target at all.** Running as of 2026-11-04: an optimiser searching
for x directly at every sampling step, against the cached target, with a fixed-x control. Bars in the
source of `5-text-embedding-third-expert/scripts/and_vector_per_step_x.py`. That result decides whether a learned x is worth
pursuing or whether the correction has to live inside the denoiser.

**x_and is settled and it is a null.** The difference written onto the empty prompt's scaffolding, so
only what the joint adds is carried: 0 of 8 tiles show two separate animals and 0 of 7 judgeable seeds
a clear pair, with PoE(dog, cat, x_and) at 48.49 mean absolute pixel difference from the joint prompt
against plain PoE's 47.05. A condition enters the guided sum as w·(ε_x − ε_∅), so a third expert built
faithfully as a difference sits near the empty prompt and contributes nothing. See
[the card](../artifacts/results/can-a-positional-and-vector-restore-composition/README.md).

**The pooled channel alone**, which scored 75% top-1 on the premise test and has never been rendered
as a condition on its own.

## Cross-references

Navigation: ⬅️ [Previous](#still-open) | 📋 [TOC](#table-of-contents)

- The mention of **a text-space third expert**, and of why one cannot reach inside the two experts' forward passes, in [the plan that puts the LoRA on the text encoders instead](../../3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/today-a-lora-on-the-text-encoders.md#why-this-exists), which is the follow-on this result leaves open.
- The mention of **the text-space finding**, placed in the text-embedding project, in [the project table for regrouping the core repo](../../plans/split-into-idea-repos.md).
