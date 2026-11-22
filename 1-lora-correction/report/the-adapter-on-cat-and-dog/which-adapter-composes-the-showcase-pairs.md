# 📒 Report: which adapter composes the two showcase pairs?

**The showcase rule picked `v58-03-self` at training step 30,000, with seed 10 as the shared seed**
`verified` (the [recorded choice](../../../artifacts/_shared/cross_pair_pool_configs/adapter_of_record.yaml),
and a read-only rerank of the calls gives the same result). The review response needed one adapter for
all its numbers and images. Nine candidate checkpoints each drew a cat and a dog, and an elephant and a
penguin, at seeds 9 to 16. Each of the 144 pictures was called clear or not clear by eye.

## Table of contents

- [What the chosen adapter draws](#what-the-chosen-adapter-draws)
- [How the nine candidates ranked](#how-the-nine-candidates-ranked)
- [Pictures kept off this page](#pictures-kept-off-this-page)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## What the chosen adapter draws

Navigation: 📋 [TOC](#table-of-contents) | [Next](#how-the-nine-candidates-ranked) ➡️

**Each picture below shows the two named animals apart, and the calls file marks both clear**
`verified`. They come from different seeds because the shared seed's cat and dog tile is
[kept off this page](#pictures-kept-off-this-page).

| Cat and dog, seed 12 | Elephant and penguin, seed 11 |
|---|---|
| <a href="../../artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/which-adapter-composes-the-showcase-pairs/tiles/v58-03-self_s030000__a_cat__x__a_dog__seed12.png"><img src="../../artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/which-adapter-composes-the-showcase-pairs/tiles/v58-03-self_s030000__a_cat__x__a_dog__seed12.png" width="200" alt="A shaggy white dog lying beside a white and tabby cat with a red collar on dry grass."></a> | <a href="../../artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/which-adapter-composes-the-showcase-pairs/tiles/v58-03-self_s030000__an_elephant__x__a_penguin__seed11.png"><img src="../../artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/which-adapter-composes-the-showcase-pairs/tiles/v58-03-self_s030000__an_elephant__x__a_penguin__seed11.png" width="200" alt="A grey elephant standing beside a penguin on a plain grey background."></a> |
| a dog and a cat, side by side | an elephant and a penguin, side by side |

## How the nine candidates ranked

Navigation: ⬅️ [Previous](#what-the-chosen-adapter-draws) | 📋 [TOC](#table-of-contents) | [Next](#pictures-kept-off-this-page) ➡️

**The chosen checkpoint is clear on both pairs at five of the eight seeds, and the next two reach
four.** A candidate scores a seed only when both of its pictures at that seed are called clear. Ties
go to the most clear pictures out of 16, then to the most training steps
([the ranking script](../../scripts/review_showcase_pick.py), lines 45 to 60).

| Candidate | Training step | Seeds clear on both pairs | How many | Clear pictures of 16 |
|---|---|---|---|---|
| `v58-03-self` | 30,000 | 10, 11, 12, 13, 16 | 5 | 12 |
| `v58-03-self` | 27,500 | 10, 12, 13, 16 | 4 | 11 |
| `v58-02-contrast` | 27,500 | 10, 11, 15, 16 | 4 | 10 |
| `v58-03-self` | 50,000 | 11, 13, 16 | 3 | 10 |
| `v57-11b-x0-self-decay` | 50,000 | 11, 12, 16 | 3 | 10 |
| `v58-01-plain` | 47,500 | 10, 15, 16 | 3 | 10 |
| `v58-03-self` | 25,000 | 11, 13, 16 | 3 | 10 |
| `v58f-05-self-contrast` | 25,000 | 10, 13, 16 | 3 | 10 |
| `v58-05-self-contrast` | 15,000 | 10, 13, 16 | 3 | 9 |

Computed by running that script with `--dry-run`, which reads [the calls file](../paper-review/showcase-calls.csv) and writes nothing `verified`.

## Pictures kept off this page

Navigation: ⬅️ [Previous](#how-the-nine-candidates-ranked) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**Four sheets and the shared seed's cat and dog tile are not shown.** Paths are relative to
`1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/which-adapter-composes-the-showcase-pairs/`.

| File | What could not be told |
|---|---|
| `tiles/v58-03-self_s030000__a_cat__x__a_dog__seed10.png` | a small form with an eye and a pointed end on the large dog's head |
| `cat-and-dog-nine-candidates.png` | holds that tile |
| `showcase_calls.png` | holds that tile |
| `elephant-and-penguin-nine-candidates.png` | its tiles were not all cleared, and the chosen adapter's seed 12 tile has small creatures on the beach |
| `contested-seeds-larger.png` | its tiles were not all cleared |

The other tiles in that folder are one series and were not opened one by one, so the folder is not
linked here.

## Where the sources disagree

Navigation: ⬅️ [Previous](#pictures-kept-off-this-page) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Two records need the owner's attention, and neither changes the recorded choice.**

| What | One source | The other source | What the code or files show |
|---|---|---|---|
| the shared seed's cat and dog picture | [the calls file](../paper-review/showcase-calls.csv) calls `v58-03-self` at step 30,000, seed 10, clear | at the stored 256 pixels, the smaller animal's face could be read as a cat or a small dog, and a form on the large dog's head cannot be named `inferred` | the full-size render is on the cluster. The call stands as recorded, and this reading does not change it |
| how to change a call | [the card](../../artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/which-adapter-composes-the-showcase-pairs/README.md) says to edit the calls file and rerun the script | the calls file's `picture` column still points at `artifacts/results/which-adapter-composes-the-showcase-pairs/tiles/`, which no longer exists `verified` | [the script](../../scripts/review_showcase_pick.py) rewrites the choice file at lines 133 to 144, then opens those paths at line 107 to redraw the figure. A full rerun would therefore change the choice file and then stop with an error `inferred` (not run) |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**This ranking is a choice among nine checkpoints on two pairs, so it gives no success rate.** Every
call is by eye, 16 per candidate. The [protocol](../paper-review/protocol.md) keeps these two pairs out
of every held-out rate because the adapter was picked on them. No [compose rate](../../../context/world/compose-rate.md#what-a-compose-rate-is) is quoted here.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy is invented. The adapter and the joint prompt are taught in
[the four-pairs page](does-one-adapter-compose-four-pairs.md#concepts-this-uses).

**The showcase rule**

Two candidates each draw two pairs at three seeds:

| Seed | A, pair 1 | A, pair 2 | B, pair 1 | B, pair 2 |
|---|---|---|---|---|
| 1 | clear | clear | clear | not |
| 2 | clear | not | clear | clear |
| 3 | clear | clear | not | clear |

A seed counts only when both of a candidate's pictures are clear. A counts seeds 1 and 3, so 2. B counts
seed 2, so 1. A wins even though each candidate has clear pictures on every seed. Run on this project's
calls, the same count gives 5 seeds for `v58-03-self` at step 30,000 and 4 for the next two. The shared
seed is the lowest seed the winner counts, here seed 10.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| the rule and the candidates | [the protocol](../paper-review/protocol.md), rule 1 | `stated` |
| the 144 calls | [the calls file](../paper-review/showcase-calls.csv), made by eye on 2026-11-10 | `stated` |
| the recorded choice | [the choice file](../../../artifacts/_shared/cross_pair_pool_configs/adapter_of_record.yaml) | `verified` |
| how the pictures were made | [the card](../../artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/which-adapter-composes-the-showcase-pairs/README.md) | `stated` |
| the two placed pictures | each opened and enlarged 2026-11-22 | `verified` |
| 141 pictures held out of the drain | the raw tiles, build history and repeat layouts in this result's folder. They repeat a picture or sheet described above, or have not been cleared against the subjects the [pair ledger](../../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md) keeps off sheets, so the folder is not linked here | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Look at the full-size cat and dog render at seed 10 and confirm or change its call.
- [ ] Repoint the `picture` column of the calls file before anyone reruns the ranking script in full.
- [ ] Look at the sheets kept off this page and decide whether any may be shown.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
