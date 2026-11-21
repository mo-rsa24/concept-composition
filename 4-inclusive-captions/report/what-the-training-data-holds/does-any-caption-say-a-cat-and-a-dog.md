# 📒 Report: does any caption say a cat and a dog, and does the picture show both? No verdict is recorded

**The claim**

1,102,469 captions in Re-LAION-2B-en name a cat word and a dog word. Whether their pictures show both
animals has no answer on record. A [check page](../../artifacts/results/does-any-caption-say-a-cat-and-a-dog/caption-check.md)
was built so a person could mark whether each sampled picture really shows both animals, and all of its
marking cells are empty.

## Table of contents

- [What would have counted](#what-would-have-counted)
- [1. How the captions name the two animals](#1-how-the-captions-name-the-two-animals)
- [2. Whether the pictures show both](#2-whether-the-pictures-show-both)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## What would have counted

Navigation: 📋 [TOC](#table-of-contents) | [Next](#1-how-the-captions-name-the-two-animals) ➡️

**No pass rule was written for this question.** The check page asks for one letter per picture: `B` both
a cat and a dog, `O` only one of them, `N` neither or not a photo. It sets no share of `B` that would
count as an answer.

## 1. How the captions name the two animals

Navigation: ⬅️ [Previous](#what-would-have-counted) | 📋 [TOC](#table-of-contents) | [Next](#2-whether-the-pictures-show-both) ➡️

> 🖼️ **Figure wanted**: one bar per caption form (bare, short, mid, long) on y, the number of captions on
> x on a log scale with true counts printed on the bars, from `counts.json`. Render into
> `artifacts/results/does-any-caption-say-a-cat-and-a-dog/`, update that grouping's README card, then embed
> here.

**Most captions naming both animals are 13 to 29 words long, and few are only the two animals.** Counts
are of the 1,102,469 captions, from the [counts file](../../artifacts/results/does-any-caption-say-a-cat-and-a-dog/counts.json)
`verified`. The median caption is 14 words.

| Caption form | What it holds | Captions | Share |
|---|---|---|---|
| bare | only the two animals joined by a word | 4,971 | 0.45% |
| short | 12 words or fewer, not bare | 423,432 | 38.41% |
| mid | 13 to 29 words | 588,332 | 53.37% |
| long | 30 words or more | 85,734 | 7.78% |

**In 821,378 captions the two animal words sit next to each other.** "And" joins them in 261,475, and
368,780 have no joining word. The other joiners are in the counts file.

## 2. Whether the pictures show both

Navigation: ⬅️ [Previous](#1-how-the-captions-name-the-two-animals) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

> [!WARNING]
> The 155 pictures in this folder were fetched from the open web through LAION, and their licences and
> contents are unchecked. None is embedded here. Opening the check page displays them, beside captions
> that name people and products. They are tracked in git and held out of the artifact drain until the
> owner decides whether they stay.

**0 of the 117 rows on the check page carry a mark.** The script drew 60 captions in each of four
classes, 240 in all. Of those, 155 pictures still loaded, and the page lists at most 30 per class
`verified` (the [fetch record](../../artifacts/results/does-any-caption-say-a-cat-and-a-dog/fetched.json) and the page).

| Class on the check page | Captions in the class | Drawn | Pictures that loaded | Rows on the page | Rows marked |
|---|---|---|---|---|---|
| bare | 4,971 | 60 | 27 | 27 | 0 |
| short | 423,432 | 60 | 41 | 30 | 0 |
| random, from every caption naming both | 1,102,469 | 60 | 41 | 30 | 0 |
| long | 85,734 | 60 | 46 | 30 | 0 |

## Where the sources disagree

Navigation: ⬅️ [Previous](#2-whether-the-pictures-show-both) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Two recorded facts conflict, and each is shown here unresolved.**

| What | One source | The other source |
|---|---|---|
| pictures that loaded in the short, random and long classes | the check page prints 31 for each | the fetch record holds 41, 41 and 46. The [page script](../../../3-understanding-compositionality/scripts/census/cat_and_dog_captions.py) stops counting one past its limit of 30 rows, which is where 31 comes from |
| what the results index says about this folder | the [results index](../../artifacts/results/README.md) says "no card yet" | the folder has [a card](../../artifacts/results/does-any-caption-say-a-cat-and-a-dog/README.md), which describes the files and states no finding |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**A caption naming both animals says nothing yet about the picture.** "Cats and dogs" can caption one
animal, a product or a drawing, and the check that would measure this is unmarked.

**The sample is limited to pictures that still load.** 85 of the 240 drawn pictures no longer load, so any
share of `B` would describe the 155 that remain.

**Word search has limits.** The search looks for cat, cats, kitten, kittens, dog, dogs, puppy and puppies,
so a caption using another word for either animal is not counted.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. The project numbers are from the files named above.

**A share of a very large set**

If 5 of 10,000 letters mention both a cat and a dog, the share is 5 / 10,000 = 0.05%. Here, 1,102,469 of
the 2,097,693,557 captions searched is 1,102,469 / 2,097,693,557 = 0.053%.

**Words next to each other, and what joins them**

In "a cat and a dog" the animal words have one joiner, "and", once the small word "a" is set aside. In
"dog food for cats" they are apart. Here 821,378 of 1,102,469 captions (74.5%) have the two words
adjacent, and "and" joins 261,475 of those 821,378 (31.8%).

**A random draw from each class**

To check a big pile, pull a few at random: 60 of 4,971 bare captions, 60 of 85,734 long ones. Every caption
in a class has the same chance, so what you find describes the class. The draw used seed 20261103 and is
saved in the [sample file](../../artifacts/results/does-any-caption-say-a-cat-and-a-dog/sample.json).

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| caption counts | [counts file](../../artifacts/results/does-any-caption-say-a-cat-and-a-dog/counts.json), read 2026-11-21 | `verified` |
| total captions searched | the [check page](../../artifacts/results/does-any-caption-say-a-cat-and-a-dog/caption-check.md), read 2026-11-21 | `stated` |
| pictures loaded per class | [fetch record](../../artifacts/results/does-any-caption-say-a-cat-and-a-dog/fetched.json) and [sample file](../../artifacts/results/does-any-caption-say-a-cat-and-a-dog/sample.json), counted 2026-11-21 | `verified` |
| how the search was run | [results card](../../artifacts/results/does-any-caption-say-a-cat-and-a-dog/README.md) and [the page script](../../../3-understanding-compositionality/scripts/census/cat_and_dog_captions.py) | `stated` |
| the question this serves | [do inclusive captions help PoE compose](../../plans/11-do-inclusive-captions-help-poe-compose/MASTER_PLAN.md) | `inferred` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Mark the 117 rows on the check page `B`, `O` or `N`, which is the only way to answer the second half of the question.
- [ ] Decide what share of `B` would count as an answer, before marking.
- [ ] Decide whether the 155 web pictures stay in the folder, given their unchecked licences.
- [ ] Render the caption-form figure named above.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
