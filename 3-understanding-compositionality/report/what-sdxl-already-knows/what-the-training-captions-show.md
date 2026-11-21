# 📒 Report: what the training captions say about wet camels and wet dogs

**Dog captions use a wet word about 6 times as often as camel captions (0.346% against 0.057%), and no picture behind them has been called wet or dry, so no verdict is recorded.** The [walk that made these counts](../../plans/10-what-poe-loses-when-it-composes/today-camel-and-dog-in-the-training-data.md#why-this-exists) tests one idea: the product of "a camel" and "rain" can only draw a wet camel if camel training pictures already show wet camels. SDXL's training data is unpublished, so the counts come from 2 of the 128 files of Re-LAION-2B-en, its public stand-in.

## Table of contents

- [How often each animal's captions use a wet word](#how-often-each-animals-captions-use-a-wet-word)
- [Whether two files stand in for all 128](#whether-two-files-stand-in-for-all-128)
- [How many pictures could still be fetched](#how-many-pictures-could-still-be-fetched)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## How often each animal's captions use a wet word

Navigation: 📋 [TOC](#table-of-contents) | [Next](#whether-two-files-stand-in-for-all-128) ➡️

> 🖼️ **Figure wanted**: a grouped bar chart drawn from `counts.json`. The x-axis is the word family (wet, weather, water, any of them), the y-axis is the percentage of that animal's captions, one bar per animal, with the raw count printed on each bar. It holds no photographs. Render into `artifacts/results/what-sdxl-already-knows/what-the-captions-say/what-the-training-captions-show/`, add it to a card there, then embed it here.

**The wet words favour dogs, and the water words lean slightly toward camels.** Each share is the row's caption count divided by all captions naming that animal in the two files: 12,291 for camel and 225,544 for dog `verified` ([saved counts](../../artifacts/results/what-sdxl-already-knows/what-the-captions-say/what-the-training-captions-show/counts.json)).

| Words in the caption | Camel captions | Share of camel captions | Dog captions | Share of dog captions | Dog share ÷ camel share |
|---|---|---|---|---|---|
| wet: wet, soaked, drenched, soaking, dripping, damp | 7 | 0.057% | 781 | 0.346% | 6.08 |
| weather: rain, raining, rainy, storm, downpour, puddle | 22 | 0.179% | 651 | 0.289% | 1.61 |
| water: swimming, bath, river, lake, splash, beach | 213 | 1.733% | 3,456 | 1.532% | 0.88 |
| any of the 18 words | 239 | 1.945% | 4,822 | 2.138% | 1.10 |

The camel wet count is only 7 captions, so its share is uncertain. The 95% Wilson ranges are 0.028% to 0.118% for camel and 0.323% to 0.371% for dog, and they do not overlap. They were computed from the counts above with the formula in [the slice script](../../scripts/census/laptop_slice.py), lines 184 to 189, at z = 1.96.

## Whether two files stand in for all 128

Navigation: ⬅️ [Previous](#how-often-each-animals-captions-use-a-wet-word) | 📋 [TOC](#table-of-contents) | [Next](#how-many-pictures-could-still-be-fetched) ➡️

**On the three counts that exist for all 128 files, the two files land within 0.5% of an even split.** The expected column is the full count × 2 / 128 `verified` (computed from the saved counts).

| Count | In the 2 files | In all 128 files | Expected in 2 files |
|---|---|---|---|
| captions | 32,776,409 | 2,097,693,557 | 32,776,462 |
| captions naming a camel | 12,291 | 784,797 | 12,262 |
| captions naming a dog | 225,544 | 14,396,194 | 224,941 |
| dog captions per camel caption | 18.35 | 18.34 | |

The wet, weather and water counts exist for the two files only, so this check covers the animal names and leaves the word families unchecked.

## How many pictures could still be fetched

Navigation: ⬅️ [Previous](#whether-two-files-stand-in-for-all-128) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**Between 43% and 69% of the links in each group still gave a picture, and none of those pictures has been called by eye.** The pictures are web photos whose licences and contents are unchecked, so they stay off this page. A caption can sit in more than one group. Counts are from each group's `fetch.json` `verified`.

| Group | Which captions | Captions tried | Pictures loaded | Share loaded |
|---|---|---|---|---|
| `camel-any` | 120 drawn at random from camel captions | 120 | 71 | 59.2% |
| `camel-wet-words` | every camel caption with any of the 18 words | 239 | 149 | 62.3% |
| `dog-any` | 120 drawn at random from dog captions | 120 | 76 | 63.3% |
| `dog-wet-words` | 120 drawn at random from the 4,822 with any of the 18 words | 120 | 81 | 67.5% |
| `camel-wet` | every camel caption with a wet word | 7 | 3 | 42.9% |
| `camel-weather` | every camel caption with a weather word | 22 | 13 | 59.1% |
| `dog-wet` | every dog caption with a wet word | 781 | 490 | 62.7% |
| `dog-weather` | every dog caption with a weather word | 651 | 448 | 68.8% |

> 🖼️ **Figure wanted**: the by-eye tally the walk planned, a stacked bar per group showing wet animal, dry animal, and no animal or cannot tell. It needs the by-eye calls first, and [the slice script](../../scripts/census/laptop_slice.py) draws it from them as `tally.png`.

## Where the sources disagree

Navigation: ⬅️ [Previous](#how-many-pictures-could-still-be-fetched) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Four recorded facts conflict, and each is shown here unresolved.** [The camel page](../a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet.md#where-the-sources-disagree) records a fifth, on where the all-file counts come from.

| What | One source | The other source |
|---|---|---|
| how many files were searched | the walk's opening and limits say 4 of 128, about 3% of the captions | the walk's step 1 records stopping at 2 files, and the saved counts hold 2, which is 1.56% of the captions (32,776,409 / 2,097,693,557) |
| how many captions pair a wet word with camel or dog | [the write-up, section 7.2](../../plans/10-what-poe-loses-when-it-composes/what-poe-loses-when-it-composes.md#72-the-census-what-the-training-captions-hold), gives 2,574 in 4 files | the saved counts give 788 (7 + 781) in 2 files, so twice the files holds 3.27 times the hits. The six wet words are at line 41 of the slice script. The 2,574 is typed into [the figure script](../../artifacts/notes/what-poe-loses-when-it-composes/make_figures.py) at line 249, and no file read records its word list |
| whether the by-eye tally exists | the walk's layout lists `calls.json`, `tally.md` and `tally.png` | none of the three is on disk, steps 3, 5 and 6 are unticked, and the `wet-look` folder is empty |
| whether the pictures were fetched | the walk's step 4 is unticked | the pictures and contact sheets for all four of its groups are on disk |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**A caption count does not count wet animals.** A [whole-word match](../a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet.md#concepts-this-uses) finds the word and leaves open what the picture shows. In the write-up's 4-file search, 1,078 of 2,574 wet-word captions mention food `stated`.

**The stand-in is not SDXL's training set.** Re-LAION-2B-en is the cleaned re-release of the set one of SDXL's two text encoders was trained on `stated` ([write-up, section 7.2](../../plans/10-what-poe-loses-when-it-composes/what-poe-loses-when-it-composes.md#72-the-census-what-the-training-captions-hold)).

**The pictures that loaded may differ from the ones that did not.** Nothing recorded shows the dead links are a random part of each group.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. The project numbers are from the tables above. The whole-word match is taught on [the camel page](../a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet.md#concepts-this-uses).

**A share within one animal's captions**

A cafe has 200 tea reviews, 10 of them saying "hot", and 4,000 coffee reviews, 400 of them saying "hot". The raw count favours coffee 40 to 1. The share is the reviews saying the word divided by all reviews of that drink: 10 / 200 = 5% for tea and 400 / 4,000 = 10% for coffee, so coffee leads only 2 to 1. Here the raw wet count favours dogs 781 to 7, about 112 to 1, and the shares (0.346% and 0.057%) put the lead at 6.08 to 1.

**A range around a small count**

Toss a coin 10 times and see 2 heads. The share is 20%, but so few tosses leave it loose: the Wilson 95% range runs from 5.7% to 51%. Wilson's range is centred a little toward one half and narrows as the tosses grow. Read it as the shares that would not be surprised by this count. For 7 wet camel captions out of 12,291 it runs from 0.028% to 0.118%. The [confidence interval note](../../../../../Obsidian/Mathematics/Statistics/Hypothesis%20Testing/Confidence%20Interval.md#where-to-learn-this) teaches the idea, and the [rule of three](../../../../../Obsidian/Mathematics/Statistics/Hypothesis%20Testing/Rule%20Of%20Three.md#where-to-learn-this) is the bound the walk plans to quote if a by-eye count finds no wet camels.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| caption counts | [saved counts](../../artifacts/results/what-sdxl-already-knows/what-the-captions-say/what-the-training-captions-show/counts.json), read 2026-11-22 | `verified` |
| word lists, sampling and fetching | [the slice script](../../scripts/census/laptop_slice.py) and [the wet and weather script](../../scripts/census/wet_weather_download.py), read 2026-11-22 | `verified` |
| pictures loaded per group | each group's `fetch.json`, counted 2026-11-22; the group counts match [the wet and weather card](../../artifacts/results/what-sdxl-already-knows/what-the-captions-say/what-the-training-captions-show/wet-and-weather/README.md) | `verified` |
| what the walk set out to do | [the walk on camels and dogs in the training data](../../plans/10-what-poe-loses-when-it-composes/today-camel-and-dog-in-the-training-data.md) | `stated` |
| the food count and the stand-in | [write-up, section 7.2](../../plans/10-what-poe-loses-when-it-composes/what-poe-loses-when-it-composes.md#72-the-census-what-the-training-captions-hold) | `stated` |
| 1,345 web photos held out of the drain | the `images/` and `sheets/` photos under [the caption results folder](../../artifacts/results/what-sdxl-already-knows/what-the-captions-say/what-the-training-captions-show). They were fetched from the open web with unchecked licences and contents, so none was opened or placed here. They are git-ignored | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Decide whether the fetched web photos may be kept and looked at, since their licences and contents are unchecked.
- [ ] Call the pictures by eye, which the [route on whether an effect needs support](../../plans/10-what-poe-loses-when-it-composes/route-causal-does-an-effect-need-support.md) needs before its test can be read.
- [ ] Draw the caption-share chart from the saved counts.
- [ ] Find the word list behind the write-up's 2,574.
- [ ] Count the wet, weather and water words over all 128 files.
- [ ] Write the card for this folder. The [grouping card](../../artifacts/results/what-sdxl-already-knows/what-the-captions-say/README.md) says "no card yet".

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
