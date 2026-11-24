# 📒 Report: how close the kept cat and dog push comes to the joint prompt and CO3

**The kept push drew a clean cat and dog on 35 of 128 seeds by the automatic scorer and 54 by eye, below
the joint prompt (57 and 69) and CO3 (89 and 101).** Plain product drew 0 on both counts. The kept push
is the [early push](does-an-early-push-separate-the-cat-and-dog.md) with a checker trained on cat and
dog plus labelled pictures of other pairs, at strength 30 on steps 0 to 15, capped at 0.4 of the
product's prediction. No overall verdict was recorded when the improvement rounds closed on 2026-11-19.

## Table of contents

- [The four methods on 128 seeds](#the-four-methods-on-128-seeds)
- [Every improvement round](#every-improvement-round)
- [How much of a difference is noise](#how-much-of-a-difference-is-noise)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## The four methods on 128 seeds

Navigation: 📋 [TOC](#table-of-contents) | [Next](#every-improvement-round) ➡️

**CO3 leads on both counts, and the kept push sits between plain product and the joint prompt.** Each
method drew one picture per seed on seeds 6001 to 6128. Scorer counts and eye counts were recounted
from each folder's `scores.jsonl` and `eye-calls.csv` `verified`. Plain product's row is
from `result-sheet-catdog-big`.

| Method | Folder | Faithful pairs, of 128 (scorer) | Not an intact photograph, of 128 (scorer) | Clean or near-clean, of 128 (eye) | Clean (eye) |
|---|---|---|---|---|---|
| plain product | `result-sheet-catdog-big` | 0 | 72 | 0 `stated` | 0 `stated` |
| kept push | `result-sheet-catdog-big-multipair-cap0.4` | 35 | 65 | 54 | 32 |
| joint prompt "a cat and a dog" | `joint-catdog-eval` | 57 | 42 | 69 | 64 |
| CO3 | `co3-catdog-eval` | 89 | 9 | 101 | 97 |

A faithful pair is the scorer's strictest verdict: exactly two bodies, one clear cat, one clear dog,
nothing borrowed and an intact photograph. "Not an intact photograph" counts tiles whose
`photo_intact` answer in `scores.jsonl` is false, which includes drawings and melted or broken parts.
The joint prompt also drew 14 cat-and-dog pictures as
drawings, which the eye count leaves out. CO3's pictures are mostly a cat and a dog side by side on a
plain studio backdrop, so they vary less than the other two columns `stated`.

## Every improvement round

Navigation: ⬅️ [Previous](#the-four-methods-on-128-seeds) | 📋 [TOC](#table-of-contents) | [Next](#how-much-of-a-difference-is-noise) ➡️

**Two of the changes were kept: the wider training set and the larger cap.** A change was kept when
faithful pairs rose by at least 4 of 128 and two bodies did not fall, and rounds on 32 seeds carried
their own marks `stated` ([the loop's close](../../artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md#loop-closed-on-2026-11-19)).
All counts below were recounted from each folder's `scores.jsonl` `verified`.

| Folder | What changed | Seeds | Two bodies | Faithful | Damaged verdict | Not intact | Outcome |
|---|---|---|---|---|---|---|---|
| `result-sheet-catdog-big` | starting point: cat and dog checker, cap 0.3 | 128 | 114 | 25 | 13 | 80 | baseline |
| `result-sheet-catdog-big-rerun` | drawn again unchanged | 128 | 116 | 25 | 11 | 83 | noise measure |
| `result-sheet-catdog-big-traj` | checker trained on the sampler's own states | 128 | 101 | 14 | 7 | 70 | dropped |
| `result-sheet-catdog-eta1` | partly random sampler, DDIM eta 1.0 | 32 | 28 | 2 | 7 | 27 | dropped |
| `result-sheet-catdog-general-iter4` | "cat present" and "dog present" outputs removed | 32 | 26 | 3 | 3 | 24 | outputs kept |
| `result-sheet-catdog-general-rerun` | early push drawn again unchanged | 32 | 30 | 7 | 3 | 19 | noise measure |
| `result-sheet-catdog-presence-added` | a separate presence checker's push added | 32 | 27 | 4 | 9 | 21 | dropped |
| `result-sheet-catdog-big-window25` | push on steps 0 to 24 | 128 | 122 | 24 | 12 | 89 | dropped |
| `result-sheet-catdog-big-multipair` | checker trained on 243 pairs | 128 | 116 | 29 | 10 | 60 | kept |
| `result-sheet-catdog-big-multipair2` | 78 more pairs' joint-prompt pictures added | 128 | 115 | 23 | 19 | 81 | dropped |
| `result-sheet-catdog-big-multipair-long` | trained 4,000 steps in place of 2,000 | 128 | 117 | 13 | 23 | 89 | dropped |
| `result-sheet-catdog-big-multipair-cap0.4` | cap raised from 0.3 to 0.4 | 128 | 121 | 35 | 11 | 65 | kept |
| `result-sheet-catdog-big-multipair-presence` | presence outputs across every pair, cap 0.4 | 128 | 119 | 24 | 20 | 94 | dropped |

A rank-16 adapter inside the checker was stopped at training step 500, still at chance (0.522), and has
no sheet `stated`. The random push of the same size gave 0 or 1 two-body tiles and 0 faithful pairs on
every 128-seed sheet that scored it `verified`.

## How much of a difference is noise

Navigation: ⬅️ [Previous](#every-improvement-round) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**Drawing the same 128 seeds twice kept the faithful count at 25, while only 15 seeds were faithful both
times.** The two runs gave the same verdict on 81 of 128 seeds `verified` (compared seed by seed from
the two `scores.jsonl` files). So totals can be compared between pushed runs and single seeds cannot.
On 32 seeds the unchanged early push moved from 5 to 7 faithful pairs `verified`.

## Where the sources disagree

Navigation: ⬅️ [Previous](#how-much-of-a-difference-is-noise) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Five recorded statements do not match the saved files or each other.**

| What | One source | The other source | What the code or data shows |
|---|---|---|---|
| how often the cap cut the multi-pair checker's push at cap 0.3 | the loop: "on most steps" | `capped_per_step` in that sheet's `result.json`: a mean of 35.5% of steps 0 to 15 over 128 seeds (54.0% for the starting checker, 18.9% at cap 0.4) | the flag is set when the push exceeds the limit (`poe_repair/methods/_classifier_guided.py` lines 58 to 60). "Most" holds for the starting checker only |
| whether seeds 6001 to 6128 were used for tuning | the loop: "never used for training or tuning" | the same loop keeps or drops each round on these seeds | no training list holds them. Choosing among rounds on them is a form of tuning `inferred` |
| how many pairs the multi-pair checker trained on | the loop: "over 200 pairs" for the labelled pool | the loop: "243 pairs" | `training_set_multipair.json` holds 242 pool pairs plus cat and dog, and 200 of the 242 have a clean picture `verified`. That the "200" means these is `inferred` |
| damaged pictures under the early push on seeds 3301 to 3332 | the loop: 3 | `scorer-check/plan29.jsonl`: 4; the re-render's `scores.jsonl`: 3 | the loop's single figure matches the re-render and leaves out the first run |
| how many pictures are damaged | the loop's damaged counts, such as 11 of 128 for the kept push | the same files' `photo_intact` answer is false on 65 of 128 | the labeller gives the verdict damaged only when hybrid, single and wrong pair do not apply first (`api_label.py` lines 73 to 76), so a broken hybrid counts as a hybrid |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**The 35 was chosen as the best of several rounds on these same 128 seeds, so it may flatter the kept
push `inferred`.** No fresh set of seeds has tested the final setting. The eye counts were made at 330
to 420 px, and a smaller view has overcounted clean pairs before: plan 29 counted 17 at half size and 14
at full size `stated`. Every count here is one pair.

> [!IMPORTANT]
> The cap 0.4 round's limit on damage can be read two ways, and the choice is the owner's. The round
> was recorded as kept, and that record stands.

| How "damaged" is read | Cap 0.3 | Cap 0.4 | Rise | Limit | That reading would give |
|---|---|---|---|---|---|
| the verdict damaged, the count the mark quotes ("10 now") | 10 | 11 | 1 | at most 4 | met |
| `photo_intact` false | 60 | 65 | 5 | at most 4 | missed |

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. The project numbers are from plan 29's calls and scorer file.

**The automatic scorer against the eye**

Ten photos, called by a person and by a stricter machine:

| | Machine says faithful | Machine says not |
|---|---|---|
| person says clean | 2 | 3 |
| person says not clean | 0 | 5 |

The diagonal (2 + 5) is agreement, 7 of 10. The empty bottom-left cell means the machine never passed a
photo the person rejected, so it is stricter. On plan 29's 32 pushed tiles the same table is 5, 9, 0 and
18, which is agreement on 23 of 32 `stated`. These counts come from `scripts/rung2/score_sheet.py`, which
sends each tile to the Claude API labeller. The older `compose_rate.py` and its comparison-strip fault
play no part: no rung 2 script imports it and this grouping holds no `__cmp` file `verified`.

**Noise from drawing twice**

Flip 10 coins twice and you might get 5 heads both times with only 3 coins heads in both runs. The total
held and the individual coins moved. The cat and dog sheet did the same: 25 and 25 faithful pairs, 15
seeds in both.

**Picking the best of several tries**

Ten equally good students sit one test, and the top score is 8 of 10. Her true level may be 6, because
picking the highest score also picks the luckiest day. A second test tells you. Here seven
changes were compared on one set of 128 seeds, and the kept setting has had no second test.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| the rounds, marks and reasons | [the checker loop](../../artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md) | `stated` |
| scorer and eye counts | `scores.jsonl` and `eye-calls.csv` in each folder, read 2026-11-22 | `verified` |
| how often the cap fired | `capped_per_step` in each sheet's `result.json`: for each seed, the share of steps 0 to 15 with the flag set, then the mean over the seeds | `verified` |
| how the scorer works | [the sheet scorer](../../scripts/rung2/score_sheet.py) | `verified` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Draw the kept push, the joint prompt and CO3 once on seeds no round has used, and score them the same way.
- [ ] Repeat the eye calls for all three methods at full size.
- [ ] Decide whether the kept push's gap to the joint prompt (35 against 57 faithful pairs) is worth more rounds.

## Cross-references

- The mention of **the kept push finding** in [the context page on the readout checker, one pushed sampling step, section "The names, taken apart"](../../../context/methods/02-the-readout-checker/01-overview.md#the-names-taken-apart).
- The mention of **the kept push finding** in [the context page on the readout checker, the push and its cap, section "The cap, worked"](../../../context/methods/02-the-readout-checker/03-the-push-and-its-cap.md#the-cap-worked).

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
