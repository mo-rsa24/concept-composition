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
- [Where the sources disagree](#where-the-sources-disagree)
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
[premise-word-recovery.json](../artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/premise-word-recovery.json),
read in [the card](../artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/README.md).

🖼️ **Figure wanted:** the rank of the held-out word as a histogram, one bar per rank bucket, the
subtracted and unsubtracted readouts side by side with the decoy control behind them, from
`premise-word-recovery.json`. Save as
`5-text-embedding-third-expert/artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/premise-word-recovery.png`.

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

<a href="../artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/token-norms-x-pos.png"><img src="../artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/token-norms-x-pos.png" width="320" alt="per-position norm, four sequences, both encoder slices"></a>

The whole series in one picture, three seeds down and eight columns across from the same noise.
Read left to right: a grey tile, then wallpaper or one hybrid, then two animals on some seeds, then
two animals on seven of eight, and no column after the first with one whole cat beside one whole dog:

<a href="../artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/how-the-third-expert-changed.png"><img src="../artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/how-the-third-expert-changed.png" width="320" alt="every third expert tried, three seeds down, eight columns across"></a>

At 300 px a tile that picture shows the trajectory only; every identity call was made from the
full-resolution tiles.

The three sheets behind the earlier columns have seeds 1 to 8 down the rows, with every column from
the same starting noise. The tests behind them are on
[the page about what the "and" vector is made of](what-the-and-vector-is-made-of.md).

| x_fix beside the references | Written phrases | x_fix at rising weight |
|---|---|---|
| `supervisor-sheet-1-before-x-pos.png` is not shown because it failed the subject check | `supervisor-3-real-phrases.png` is not shown because it failed the subject check | <a href="../artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/supervisor-4-x-at-rising-weight.png"><img src="../artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/supervisor-4-x-at-rising-weight.png" width="200" alt="Ten columns by eight seeds: the references, PoE with x_fix at five weights from 0.1 to 1, and PoE with x_pos."></a> |
| x_fix alone draws wallpaper or texture on all 8 seeds `verified` (seen on the sheet). | The last column is x_pos, beside the seven phrases and two lower weights of x_fix `verified`. | Weights run 0.1, 0.25, 0.5, 0.75 and 1, with x_pos in the last column `verified`. |

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
per-seed/ (on the cluster at `/datasets/mmolefe/poe_repair_min/artifacts-archive/5-text-embedding-third-expert/artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/per-seed`).

| x_pos alone and in the product | The target, plain PoE and the product on every seed |
|---|---|
| `positional-sheet.png` is not shown because it failed the subject check | <a href="../artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/all-seeds-target-poe-xpos.png"><img src="../artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/all-seeds-target-poe-xpos.png" width="200" alt="Eight rows of three whole tiles: the joint prompt, plain PoE and PoE with x_pos, for seeds 1 to 8."></a> |
| Rows are seeds 1 to 8. The two right-hand columns are x_pos alone and PoE(dog, cat, x_pos) `verified` (seen on the sheet). Their distances from the joint prompt's image are the 23.43 and 39.16 above. | The product column is called two animals with at least one hybrid on 7 of 8 tiles, which is 6 of the 7 judgeable seeds, and one animal on seed 5 `stated` ([the tile calls](../artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/calls.json)). |

This agrees with the sampler check already on file, where the exact target placed in the third slot
at every step renders the joint image to within 2.23. The third slot is not the limitation.

## What this cannot say

Navigation: ⬅️ [Previous](#3-the-product-is-worse-than-the-third-expert-alone) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**One pair and seven judgeable seeds.** A count of 2 or 3 would have been inconclusive. The zeros are
firm only because they are far from the mark.

**One weight.** Every third expert enters at the run's guidance scale. Only x_fix had a weight sweep,
and it returned plain PoE's hybrid at low weights and wallpaper at high ones.

**By-eye calls.** Identity was called from full-resolution tiles by Claude and confirmed by the user
after three calls were queried and re-checked against the joint prompt's own cat on the same seed.
No automatic scorer was used, because the project's [instance-count scorer](../../context/world/compose-rate.md#words-this-file-uses) cannot tell a cat from a
dog.

**Nothing about a correction inside the denoiser.** This is an inference-only result.

## Where the sources disagree

Navigation: ⬅️ [Previous](#what-this-cannot-say) | 📋 [TOC](#table-of-contents) | [Next](#provenance) ➡️

**Six recorded facts conflict, and each is shown here unresolved.**

| What | One source | The other source |
|---|---|---|
| whether any third expert drew a clear pair | this page's opening says none did on any seed, and [the idea map](../artifacts/ideas/the-and-vector-as-a-third-expert/IDEA_MAP.md) records 0 clear pairs for x_fix | this page's table, [the positional card](../artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/README.md) and [the supervisor deck](../../artifacts/decks/supervisor-meeting/README.md) count 1 for x_fix, on seed 3; the idea map also calls that tile two animals with at least one hybrid |
| tiles with two animals for the best phrase, of 8 | the positional card and the deck give 5 | the idea map lists 6 for "a dog next to a cat": seeds 1, 3, 4, 6, 7 and 8, of which 5 are judgeable |
| how many third experts were tried | this page's opening and the idea map say eight | [the learned-token plan](../../3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/today-a-learned-and-token.md) says nine had failed before x_pos, and [the walk's retrospective](../../artifacts/notes/session-merges/2026-11-04-the-positional-and-vector-retro.md) calls x_pos the seventh |
| plain PoE's distance from the joint prompt's image, 0 to 255 | 47.05 on this page, averaged over seeds 1 to 8 | 34.69 in the deck's sampler check, the source of the 2.23 quoted above; the deck does not name its seeds |
| the staging copy of the first supervisor sheet | the positional card says it is byte-identical to its installed copy | by MD5 checksum on 2026-11-21, [the staging copy](../artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/supervisor-1-arithmetic-vector-x.png) differs; the installed sheet matches `supervisor-sheet-1-before-x-pos.png` |
| whether the x_pos render finished | [the walk](../artifacts/drips/can-a-positional-and-vector-restore-composition/EXECUTE_WALK.md) says it is running, with its closing pass owed | the card, the tile calls and this page hold the finished render |

## Provenance

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

Renders on mscluster108 outside Slurm, 2026-11-04, from `5-text-embedding-third-expert/scripts/and_vector_sheet.py`; earlier
columns from runs 2 to 7 of the idea map on mscluster106 and mscluster109, 2026-10-27 and 2026-10-28.
Token norms from `5-text-embedding-third-expert/scripts/and_vector_norms.py`, premise test from `5-text-embedding-third-expert/scripts/and_vector_premise.py`,
both text encoders only. Tiles under
`/datasets/mmolefe/poe_repair_min/outputs/concept_pair_sheet/and_vector/dog__x__cat/`. Calls in
[calls.json](../artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/calls.json). Walked in
[today-the-positional-and-vector.md](../../artifacts/plans/completed/today-the-positional-and-vector.md).

The eight tiles in the per-seed folder (on the cluster at `/datasets/mmolefe/poe_repair_min/artifacts-archive/5-text-embedding-third-expert/artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/per-seed`)
are held out of the artifact drain, because
[the all-seeds sheet](../artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/all-seeds-target-poe-xpos.png)
holds the same three columns for all 8 seeds.

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
[the card](../artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/README.md).

<a href="../artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/poe-x-and-all-seeds.png"><img src="../artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/poe-x-and-all-seeds.png" width="200" alt="Eight tiles of PoE with x_and, seeds 1 to 8, each showing one fused cat-dog animal."></a>

PoE(dog, cat, x_and) on seeds 1 to 8 draws one fused animal in every tile, with seeds 4 and 5 as flat
graphics `verified` (seen on the picture; matches the card).

**The pooled channel alone**, which scored 75% top-1 on the premise test and has never been rendered
as a condition on its own.

**The x_fix tile on seed 3.** Whether it is a clear pair or two animals with a hybrid is the owner's
call from the full-resolution tile. The count stays under the pass mark of 4 either way.

## Cross-references

Navigation: ⬅️ [Previous](#still-open) | 📋 [TOC](#table-of-contents)

- The mention of **a text-space third expert**, and of why one cannot reach inside the two experts' forward passes, in [the plan that puts the LoRA on the text encoders instead](../../3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/today-a-lora-on-the-text-encoders.md#why-this-exists), which is the follow-on this result leaves open.
- The mention of **the text-space finding**, placed in the text-embedding project, in [the project table for regrouping the core repo](../../plans/split-into-idea-repos.md).

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
