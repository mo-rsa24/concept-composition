# 🧪 Review: does a small pair-aware discriminator's push make SDXL's cat × dog product draw two animals on seeds it never saw?

**Nothing has run yet.** This file judges rung 2 of the [scale-up ladder](../../report/classifier-guided-poe/board.md#the-scale-up-ladder), run as step 10 of
[the route from the toy to the animals](../route-classifier-guided-poe-from-toy-to-animals.md);
its page is [A3q](../../report/classifier-guided-poe/a3q-cat-x-dog-quick-discriminator-test.md).
The readings below were written on 2026-11-17, before any SDXL render for this rung, and step 10
reads them.

- [x] 📝 bar written before the run: seven readings, each with its count out of 4 seeds, in
      [The question written before the run](#the-question-written-before-the-run)

## Recommended prompt (when the run lands)

```
/analyze-run @plans/review/a3q-cat-x-dog-quick-discriminator-test.md — <the per-tile calls CSV from step 10>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [the route, step 10](../route-classifier-guided-poe-from-toy-to-animals.md) | the design: training set, discriminator, columns, compute piece |
| [A3q's page](../../report/classifier-guided-poe/a3q-cat-x-dog-quick-discriminator-test.md) | the rung's outcomes and its reading table, for a reader of the report |
| **this file** | **the bar, and later the verdict** |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

**The product (plain product)**

SDXL's empty-prompt noise prediction plus the guidance scale times the sum, over "a cat" and
"a dog", of each prompt's difference from the empty prediction. On cat × dog
it draws one fused animal. Best-of-16 composed 0 of 128 particles on this pair.

**The pair-aware discriminator**

A small network that reads a noisy SDXL latent and its timestep and outputs one log-odds: "this
is a cat and a dog apart" against "this is the product's render, or a lone cat, or a lone dog".
It trains on 10 to 15 renders of a cat and a dog apart, each checked by eye, from seeds 1 to 8
only, with every image noised the same way.

**The push, and its strength s**

The discriminator's log-odds gradient, added to the product's prediction at every step as
strength × sqrt(1 − ᾱ_t) × gradient, unrescaled. The guided column renders at three strengths,
**s = 10, 30 and 100**, fixed here. The toy composed at 1 and 2, but its units do not carry to
SDXL. [Plan 18](../18-a-dog-classifier-trained-the-papers-way.md)'s dog classifier, pushed the same unrescaled way on SDXL, first changed the cat × dog
product's picture at 30 (the fused animal became one cat) and broke it at 300; from the plain
prompt it drew clean dogs at 30 and 100. No strength is changed after any seed 9 to 12 render.

**The cap**

The push is clipped to 0.1 of the product prediction's norm (`PUSH_CAP_FRACTION` in
`poe_repair/methods/_classifier_guided.py`). The sampler logs, every step, whether it fired.

**The random push**

At every step, a fresh random direction in the latent, drawn from a fixed generator and scaled to
the norm the real push had at that step, on the same seed and strength, after the cap. It is read
from the guided render's per-step log. It shows whether any push this size separates the animals,
whatever its direction. It renders at all three strengths, so no strength is picked after seeing
seeds 9 to 12.

**Best-of-N at equal compute**

N plain-product renders per seed, with the discriminator picking the one it scores highest. N is
set by step 10's compute piece so that the N renders cost the same GPU time as one guided render.
The picked tile is the one called.

**Training seeds and held-out seeds**

Seeds 1 to 4 had renders in the discriminator's training set. Seeds 9 to 12 had none, and no
render from seeds 9 to 16 enters training.

**The two calls on every tile**

Each tile is called twice, by eye, at full resolution. The animal call is **one single animal**,
**one fused animal** (a hybrid, or two heads on one body), or **two separate animals** (two
bodies, of either species or unclear species). The picture call is **intact** or **damaged**:
damaged means the photograph breaks at full size (smeared or melted fur, noise or banding across
the frame, a body dissolving into the background, or a drift from photo to drawing). The fused
animal plain product draws is not damage. Species (a cat and a dog, two of one kind, unclear) is
recorded as a third column and never decides a reading.

**Composed, and composed intact**

A tile is composed when its animal call is two separate animals. It is composed intact when its
picture call is also intact.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim.** Only support lets rung 3 start; every other reading stops the ladder at rung
2 and names the one change worth trying next. Between plain product and the guided column one
mechanism differs (the push is present at a nonzero strength). Between best-of-N and guided, the
same discriminator is used two ways at equal compute, so that comparison is clean on how it is
used. Between the random push and guided only the push's direction differs; its size is matched
step by step.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| Compute piece: 4 guided and 4 plain renders timed on the chosen card | setup | not yet | minutes | the GPU-hours, card, fallback and N, written into route step 10 | ⚠️ not started |
| The discriminator, trained on the checked renders from seeds 1 to 8 | builds the push | not yet | written by the compute piece | `/datasets/mmolefe/poe_repair_min/outputs/classifier_guidance_ladder/rung2/` | ⚠️ not started |
| The sheet: seeds 1 to 4 and 9 to 12; plain product, best-of-N, guided and the random push at s = 10, 30, 100 | Tests the claim | not yet | about 0.3 GPU-hours on a Blackwell (inferred until timed) | the same folder, and the card under `2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/` | ⚠️ not started |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ **Does the push draw two separate, intact animals on held-out seeds 9 to 12 more often
      than plain product and than best-of-N picked by the same discriminator?**

      Counts are out of 4 seeds per seed set, per strength. The readings are checked in this
      order, and the first that holds is the verdict.

      **1. Support.** At some s, composed intact on seeds 9 to 12 is **at least 2 of 4**, and
      more than best-of-N's composed intact on seeds 9 to 12, and the random push at the same s
      composes intact on at most 1 of 4 of seeds 9 to 12. The push generalises across seeds, and
      its direction is what does it. Rung 3 may start.

      **2. The discriminator picks, the push adds nothing.** At some s, composed intact on seeds
      9 to 12 is at least 2 of 4, but at every such s best-of-N's composed intact count on seeds
      9 to 12 is equal or higher. The discriminator is useful as a picker and its gradient does
      no better. Rung 3 does not start on the push; the next step is a choice between scaling
      the picker and diagnosing the push.

      **3. Any push this size does it.** At some s, composed intact on seeds 9 to 12 is at least
      2 of 4 and more than best-of-N's, but at every such s the random push also composes intact
      on at least 2 of 4. Disturbing the sampler separates the animals, and the discriminator's
      direction adds nothing measurable. Rung 3 does not start; the finding goes to the board
      as a property of the product's sampler.

      **4. Inconclusive: composed, but wrecked.** None of the above, and at some s composed on
      seeds 9 to 12 (intact or damaged) is at least 2 of 4. The push can separate the animals
      but breaks the picture doing it. Next: rerun seeds 9 to 12 at s = 3 and at the lowest s
      that composed, with the cap's firing read per step.

      **5. Memorised.** At every s composed on seeds 9 to 12 is at most 1 of 4, and at some s
      composed on seeds 1 to 4 is at least 2 of 4. The push works only on the noise it trained
      beside. The idea needs more checked renders over more seeds, not more compute.

      **6. Null.** At every s composed is at most 1 of 4 on both seed sets, and both of these
      hold at s = 100: the guided tile differs visibly from plain product's tile of the same seed
      (pose, layout or number of animals) on at least 2 of the 8 seeds, and the cap fired on
      fewer than half the sampler's steps on every seed. A pair-aware push this small cannot
      fix cat × dog. Stop before scaling. On the toy the product's own target composed and the
      push repaired the sampler; on SDXL the gap may also be the model's, so this null does not
      contradict A1.

      **7. Too weak to read.** The null's counts hold but either check at s = 100 fails. If the
      picture barely moved and the cap did not fire, rerun seeds 1 to 4 and 9 to 12 at s = 300.
      If the cap fired on half the steps or more, the push was clipped, so rerun at s = 100 with
      the cap at 0.3. No null is called until one of those reruns lands.

      **Why 2 of 4 is the line.** Plain product composed 0 of 128 renders on this pair, so its
      rate is at most 2.3% (the 95% upper bound, 3 / 128). At that rate the chance of 2 or more
      composed tiles in 4 by luck is under 0.4%.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] ⚠️ **Did the cap fire, and on which steps, at each strength?** On the toy the push stayed
      under a tenth of the summed score's size, so the prediction is that it never fires at
      s = 10 or 30. Report the fraction of steps it fired, per seed and strength, either way.

- [ ] ⚠️ **When at least one of the N best-of-N renders composed, did the discriminator pick
      one of them?** Counted over seeds 9 to 12. A discriminator that passes over composed
      renders it was handed is not reading what it was trained to read, and that weakens every
      reading above.

- [ ] ⚠️ **Does the push turn the fused animal into one lone animal?** Plan 18's push turned it
      into one cat on 3 of 4 seeds, which is why lone animals are negatives here. Count guided
      tiles called one single animal against plain product's, per strength.

- [ ] ⚠️ **Which species do the composed tiles show?** A cat and a dog, two of one kind, or
      unclear, from the third column. Reported beside the verdict, never as it.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Nothing yet.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?** Every column of the sheet renders on one card in one loop,
      plain product and guided from the same starting noise per seed, the random push included. Print the random push's norm over the real
      push's at every step; 1 on every step is the only pass. Best-of-N's N comes from
      timing on that card, and the realised GPU time of both columns is printed beside the
      sheet.

- [ ] ⚠️ **Was the measuring tool sound?** The calls are by eye at full resolution, because the
      instance-count detector cannot tell a cat from a dog and two hybrids read as a clear pair
      at sheet scale. Claude's calls are drafts until the reader checks each tile.

- [ ] ⚠️ **Did the run respect the environment?** The training set holds no seed 9 to 16 render
      (print the seed list it read), strength 0 matched plain product byte for byte at step 5's
      canary, and the outputs landed under `/datasets`.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| a pair-aware push composes cat × dog on held-out seeds | 4 seeds, one pair; the strength it was read at; best-of-N's and the random push's counts on the same seeds; the damaged count beside the intact one |
| a pair-aware push does not compose cat × dog | that the push visibly moved the picture and the cap did not bind, and that on SDXL the gap may be the model's |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| N for best-of-N | step 10's compute piece timing guided and plain renders on one card | the best-of-N column |
| A3q's outcome line trains on "the product's renders from seeds 1 to 4"; route step 10 trains on seeds 1 to 8. Seeds 1 to 4 are training seeds either way, so the bar holds under both | the training-set sheet in step 10, piece 1 | nothing in this bar |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Route step 10's compute piece, started with the `/drip-execute-plan --request` prompt in
[the route](../route-classifier-guided-poe-from-toy-to-animals.md).
