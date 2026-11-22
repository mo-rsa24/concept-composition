# 📒 Report: does the early push carry from cat and dog to other pairs?

**It carried to one object pair and only partly to a second [animal pair](../../../context/world/animal-pair.md).** The
[early push](does-an-early-push-separate-the-cat-and-dog.md), with a checker trained only on cats and
dogs, separated a typewriter and a cactus on 14 of 16 seeds and a lion and a meerkat on 6 of 16. Later
checkers trained across many pairs never produced a lion and meerkat the automatic scorer accepted.

## Table of contents

- [The cat and dog checker on two new pairs](#the-cat-and-dog-checker-on-two-new-pairs)
- [Checkers trained across many pairs](#checkers-trained-across-many-pairs)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## The cat and dog checker on two new pairs

Navigation: 📋 [TOC](#table-of-contents) | [Next](#checkers-trained-across-many-pairs) ➡️

**The object pair met the support mark, and the animal pair landed in between.** The mark for each pair
was both named things separate on at least 10 of 16 seeds, with plain product and a random push of the
same size each at most 2. Seeds are 3301 to 3316, called by eye from full-size tiles `stated`
([plan 30's review](../../plans/review/30-does-the-recipe-work-on-other-pairs.md#answers)).

| Folder | Pair | Plain product | Random push | Early push | Clean, early push | Verdict |
|---|---|---|---|---|---|---|
| `result-sheet-lion-meerkat` | lion and meerkat | 1 of 16 | 1 of 16 | 6 of 16 | 3 of 16 | in between |
| `result-sheet-typewriter-cactus` | typewriter and cactus | 1 of 16 | 1 of 16 | 14 of 16 | 6 of 16 | support |

On lion and meerkat the push drew two animals on 10 of 16 seeds, and 4 of those were two meerkats, so
they do not count as the pair.

`not shown: not cleared against the subjects the pair ledger keeps off sheets`: the `views` sheets in
both folders.

## Checkers trained across many pairs

Navigation: ⬅️ [Previous](#the-cat-and-dog-checker-on-two-new-pairs) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**Every lion and meerkat sheet scored 0 faithful pairs of 16, whichever checker or cap was used.** A
faithful pair is the automatic scorer's strictest verdict: exactly two bodies, one clear lion, one clear
meerkat, nothing borrowed and an intact photograph. Lion and meerkat stayed out of these checkers'
training. The written stop condition was 6 or more faithful pairs of 16 `stated`
([the loop's close](../../artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md#loop-closed-on-2026-11-19)).
Scorer counts were recounted from each folder's `scores.jsonl` `verified`.

| Folder | Checker and cap | Two bodies, of 16 | Faithful pairs, of 16 | By eye `stated` |
|---|---|---|---|---|
| `result-sheet-lion-meerkat-multipair` | trained on cat and dog plus the labelled benchmark pictures, cap 0.3 | 12 | 0 | a lion beside a meerkat on about 4 seeds |
| `result-sheet-lion-meerkat-multipair-cap0.4` | the same checker, cap 0.4 | 15 | 0 | about 4 seeds, the rest mostly two meerkats |
| `result-sheet-lion-meerkat-multipair2` `inferred` | 78 more pairs' joint-prompt pictures added, cap 0.3 | 11 | 0 | 1 seed |
| `result-sheet-lion-meerkat-multipair-long` | trained twice as long, cap 0.3 | 11 | 0 | none recorded |
| `result-sheet-lion-meerkat-multipair-presence` | "first animal present" and "second animal present" outputs, cap 0.4 | 11 | 0 | pictures break into posters and collages |

The loop's record gives the reason: the push makes two bodies and leaves which animal each becomes to
the product, where the meerkat expert wins on most seeds `stated`.

## Where the sources disagree

Navigation: ⬅️ [Previous](#checkers-trained-across-many-pairs) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**The scorer counted different totals on identical pictures, and it counts objects differently from
the eye.**

| What | One source | The other source | What explains it |
|---|---|---|---|
| plain product, lion and meerkat, two bodies of 16 | `scorer-check/lion-meerkat.jsonl` and the `multipair2` sheet: 3 | the `multipair` sheet: 2 | the 16 plain tiles are byte-identical across the three sheets (checked by file hash). Seed 3314 got a different answer, so the labeller itself varies between runs `verified` |
| plain product, typewriter and cactus | eye: 1 of 16 separate | scorer: 6 of 16 with two bodies | the loop records that the scorer counts loose cacti as extra bodies `stated` |
| the early push, lion and meerkat | eye: 3 clean or near-clean | scorer: 0 faithful pairs | the scorer's faithful pair is stricter than the eye's near-clean, which allows one small borrowed feature `stated` |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**Two pairs cannot show how far the push travels.** Plan 30's second question asked for a checker
trained on 78 pairs plus cat and dog, tested on both pairs. The nearest run, the `multipair2` checker,
also holds the benchmark pictures, and it has no typewriter and cactus sheet. So that question has no
recorded verdict.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. The project numbers are from plan 30's review and the scorer files.

**Two things against the right two things**

Ask for an apple and a pear on 10 plates. Seven plates hold two pieces of fruit, and 3 of those are two
pears. Two things appear on 7 plates and the right two on 4. Plan 30 counts the second. For lion and
meerkat, two animals appear on 10 of 16 seeds, 4 are two meerkats, and the pair is right on 6.

**A pair held out of training**

A cook who has only practised omelettes is tested on a pancake, which shows what carried over. A cook
who practised pancakes proves nothing on one. Lion and meerkat and typewriter and cactus are written as
held out in `scripts/rung2/training_set_multipair.json`, so no training tile comes from them `verified`.

**A judge that can disagree with itself**

Show one judge the same 16 photos on two days. If she counts 3 on Monday and 2 on Tuesday, a gap of 1
between two methods means little. The scorer did this on byte-identical plain tiles, counting 3, 2 and
3 across three runs.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| the first test | [plan 30](../../plans/30-does-the-recipe-work-on-other-pairs.md) and [its review](../../plans/review/30-does-the-recipe-work-on-other-pairs.md) | `stated` |
| the later checkers | [the checker loop](../../artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md) | `stated` |
| scorer counts | `scores.jsonl` in each folder and `scorer-check/`, read 2026-11-22 | `verified` |
| the `inferred` folder | `result-sheet-lion-meerkat-multipair2` is matched to its checker by the head named in its `result.json`; the loop reports its numbers without naming the folder | `inferred` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Run or retire plan 30's second question. Its tasks 1.1 to 1.3 are unticked, though the 78-pair runs were drawn and labelled.
- [ ] Try the walk's next idea, giving each detected body its own prompt after the layout settles. No plan holds it yet.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
