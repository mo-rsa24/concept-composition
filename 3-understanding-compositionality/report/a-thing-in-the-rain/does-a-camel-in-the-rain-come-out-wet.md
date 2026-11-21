# 📒 Report: does a camel in the rain come out wet

**SDXL draws a wet camel on 4 of 5 seeds when the prompt says "a wet camel", and the reader called every product of experts render of "a camel" and "rain" dry.** No verdict has been recorded for the hypothesis these sheets serve, because one of its inputs, a by-eye count of wet camels in training pictures, does not exist yet. [Product of experts](../../../context/world/poe-composition.md#what-poe-composition-is) (PoE) adds the pull of each prompt at every drawing step.

## Table of contents

- [What SDXL draws for a camel and rain](#what-sdxl-draws-for-a-camel-and-rain)
- [What the hypothesis test still needs](#what-the-hypothesis-test-still-needs)
- [What the public captions hold](#what-the-public-captions-hold)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## What SDXL draws for a camel and rain

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-the-hypothesis-test-still-needs) ➡️

| Each prompt asked directly | The camel with rain | PoE on seed 9, full size |
|---|---|---|
| <a href="../../artifacts/results/a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet/asked-directly.png"><img src="../../artifacts/results/a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet/asked-directly.png" width="200" alt="Five seeds down the page and four prompts across: a wet camel, a camel in the rain, a wet dog, a dog in the rain."></a> | <a href="../../artifacts/results/a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet/a-camel-and-rain.png"><img src="../../artifacts/results/a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet/a-camel-and-rain.png" width="200" alt="Five seeds down the page and five columns across: a camel alone, rain alone, the joint prompt, product of experts and SuperDiff."></a> | <a href="../../artifacts/results/a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet/tiles/a_camel__x__rain/seed09/poe.png"><img src="../../artifacts/results/a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet/tiles/a_camel__x__rain/seed09/poe.png" width="200" alt="An illustrated camel head on grey with white drops hanging from its chin and neck."></a> |
| Seeds 1, 2, 4, 9 and 10 down the page, one prompt per column `verified` | The same seeds. Columns: camel, rain, joint prompt, PoE, SuperDiff `verified` | White drops hang from the chin and neck. The recorded call is dry `verified` |

**Only the prompt that ties "wet" to the camel gets there.** The joint prompt and PoE are dry on 5 of 5 seeds, and "a camel in the rain" is dry on 4 with 1 half. The dog column is here for comparison, and its own sheet is on the [dog page](does-a-dog-in-the-rain-come-out-wet.md).

| How it was asked | Camel, over 5 seeds | Dog, over 5 seeds |
|---|---|---|
| "a wet ..." | wet 4, half 1 `verified` | wet 5 `verified` |
| "a ... in the rain" | dry 4, half 1 `verified` | wet 4, half 1 `verified` |
| joint prompt "a ... and rain" | dry 5 `verified` | wet 2, half 1 `stated` |
| PoE of "a ..." and "rain" | dry 5 `verified` | wet 2 `stated` |
| SuperDiff at 200 steps | broken 3, half 1, cannot tell 1 `verified` | no call recorded |

Camel counts and the direct dog counts are tallied from the reader's two call files, [direct prompts](../../artifacts/results/a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet/asked-directly-calls.json) and [camel sheet](../../artifacts/results/a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet/a-camel-and-rain-calls.json). The two dog rows marked `stated` come from [the walk](../../artifacts/drips/probing-sdxl/a-dog-and-rain/EXECUTE_WALK.md#the-cut).

## What the hypothesis test still needs

Navigation: ⬅️ [Previous](#what-sdxl-draws-for-a-camel-and-rain) | 📋 [TOC](#table-of-contents) | [Next](#what-the-public-captions-hold) ➡️

**The first half of [the written test](../../artifacts/drips/probing-sdxl/a-dog-and-rain/EXECUTE_WALK.md#hypothesis-test-poe-draws-only-an-overlap-sdxl-saw) can be read today, and the second half has no input.** The hypothesis is that PoE can only draw an overlap of two concepts that SDXL saw in training. A half call counts as dry.

| Part of the test | Dog | Camel | State |
|---|---|---|---|
| PoE's wet count is no more than the wet count of "a ... in the rain" | 2 against 4 | 0 against 0 | holds for both on the recorded counts |
| the animal with fewer by-eye wet caption pictures comes out wet on fewer PoE seeds | no count | no count | the table it reads is missing |

> [!IMPORTANT]
> The test says "inconclusive" when "a caption count is not reachable". Whether a count nobody has made yet is unreachable is the owner's choice, so no verdict is written here.

| How the missing count is read | What that reading gives |
|---|---|
| the count cannot be reached | inconclusive |
| the count is still to be made | the test stays open until the by-eye calls exist |

## What the public captions hold

Navigation: ⬅️ [Previous](#what-the-hypothesis-test-still-needs) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**In the public stand-in for SDXL's training data, dog captions outnumber camel captions 18 to 1 (14,396,194 / 784,797 = 18.3), and few captions of either kind mention wetness.** SDXL's own training data is unpublished. The stand-in is [Re-LAION-2B-en research-safe](../../artifacts/results/a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet/caption-counts.md#dataset-searched), the cleaned re-release of the set one of SDXL's text encoders was trained on. A match is a whole-word, case-ignoring hit on camel, camels or dromedary, or on dog, dogs or puppy.

| Captions counted | Camel | Dog | Searched |
|---|---|---|---|
| naming the animal | 784,797 | 14,396,194 | all 128 files, 2,097,693,557 captions |
| naming the animal | 12,291 | 225,544 | 2 files, 32,776,409 captions |
| with a wet word | 7 | 781 | the same 2 files |
| with a weather word | 22 | 651 | the same 2 files |
| with a water word | 213 | 3,456 | the same 2 files |

Counts are from [the saved caption counts](../../artifacts/results/what-sdxl-already-knows/what-the-captions-say/what-the-training-captions-show/counts.json) `verified`. A caption hit does not show a wet animal: in 4 files, 1,078 of 2,574 captions pairing a wet word with camel or dog mention food `stated` ([write-up, section 7.2](../../plans/10-what-poe-loses-when-it-composes/what-poe-loses-when-it-composes.md#72-the-census-what-the-training-captions-hold)). The [wet and weather caption pages](../../artifacts/results/what-sdxl-already-knows/what-the-captions-say/what-the-training-captions-show/wet-and-weather/README.md) hold the pictures behind these hits, with no calls on them yet.

**A first attempt to count wet animals with CLIP failed its own check: 0 of the 7 pictures it called wet showed a wet animal by eye, against a pass mark of 80%.** CLIP is a model that scores how well a picture fits a sentence. The eye check was Claude's `stated` (write-up, section 7.2). The two contact sheets from this pilot are web photos, so they stay off this page.

| Pilot stage | Camel captions | Dog captions |
|---|---|---|
| sampled | 100 | 100 |
| picture still loads | 63 | 59 |
| CLIP says the animal is there | 43 | 47 |
| CLIP says it is wet | 4 | 3 |
| of those, wet by eye | 0 | 0 |

The stages are from [the figure script](../../artifacts/notes/what-poe-loses-when-it-composes/make_figures.py), lines 251 and 252 `verified`.

## Where the sources disagree

Navigation: ⬅️ [Previous](#what-the-public-captions-hold) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Four recorded facts conflict, and each is shown here unresolved.**

| What | One source | The other source |
|---|---|---|
| PoE on seed 9 | the reader's call is dry | the full-size render shows white drops hanging from the chin and neck, on a plain grey background |
| what CLIP's 7 wet calls are out of | the write-up says 7 of 122 fetched pictures | [the pilot script, line 185](../../scripts/caption_wetness.py), and the figure count wet calls among the 90 pictures (43 camel + 47 dog) CLIP accepted as showing the animal. The saved scores on the cluster would settle it |
| where the by-eye wet count lives | the walk's test reads a "Checked by eye" table in the caption counts file | [that file](../../artifacts/results/a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet/caption-counts.md) has no such table |
| where the full caption counts come from | the saved counts name the caption counts file as the source of 784,797 and 14,396,194 | that file holds neither number. It records a first search of 8 files with 48,786 camel captions |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**A dry PoE camel does not show that missing training pictures caused it.** PoE on SDXL also carries sampler error, and the [exact toy](../the-exact-toy/what-poe-needs-from-training-to-draw-two-things-at-once.md) is where training content is varied directly.

**The counts describe a stand-in.** Nothing here measures what SDXL saw, and 39% of the pilot's links no longer load: (100 - 63) + (100 - 59) = 78 of 200.

**SuperDiff is a mixed comparison twice over.** On this sheet it differs from its row in starting noise, sampler and step count. Against the dog sheet it also ran on a different graphics card, an RTX 3090 in place of the card used for the dog's SuperDiff `stated` ([the card](../../artifacts/results/a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet/README.md)).

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. The project numbers are from the tables above.

**Joint prompt and product of experts**

Say a camel's pull on the picture is (3, 0) and rain's is (0, 4). PoE moves along their sum, (3, 4), so each prompt only ever contributes its own direction. A joint prompt is one sentence, read once by the text encoder, so its pull can be something else, such as (2, 2, 5). Here PoE of "a camel" and "rain" is dry on 5 of 5 seeds. The [product of experts note](../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Product%20Of%20Experts.md#where-to-learn-this) gives the probability behind the sum.

**Precision of a wet call**

A coffee taster calls 10 cups "strong", and 8 of those really are. Precision is the share of what was called that was right: 8 / 10 = 80%. "Called" is the denominator and "right" is the numerator. CLIP called 7 pictures wet and 0 were wet by eye, so its precision is 0 / 7 = 0%, against the 80% the [pilot script](../../scripts/caption_wetness.py) required at line 53.

**A whole-word match**

Take three captions: "camels at dusk", "a camel coat" and "camelback mountain". A whole-word search for camel or camels matches the first two, and skips the third because "camel" is only part of a longer word. The second is a colour, so a match says the word is present and leaves open what the picture shows. So the 784,797 counts captions using the word, and some of them describe a colour or a product.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| how the two sheets were rendered | [the card](../../artifacts/results/a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet/README.md), jobs 72389 and 72470, 2026-10-20 | `stated` |
| the reader's calls | [direct prompts](../../artifacts/results/a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet/asked-directly-calls.json) and [camel sheet](../../artifacts/results/a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet/a-camel-and-rain-calls.json), read 2026-11-22 | `verified` |
| the written test | [the dog and rain walk](../../artifacts/drips/probing-sdxl/a-dog-and-rain/EXECUTE_WALK.md), read 2026-11-22 | `stated` |
| caption counts | [saved counts](../../artifacts/results/what-sdxl-already-knows/what-the-captions-say/what-the-training-captions-show/counts.json), read 2026-11-22 | `verified` |
| the pilot's method | [the pilot script](../../scripts/caption_wetness.py), read 2026-11-22 | `verified` |
| the pilot's results | [write-up, section 7.2](../../plans/10-what-poe-loses-when-it-composes/what-poe-loses-when-it-composes.md#72-the-census-what-the-training-captions-hold) and [the figure script](../../artifacts/notes/what-poe-loses-when-it-composes/make_figures.py) | `stated` |
| 39 full-size tiles held out of the drain | 12 tiles in [the camel with rain folder](../../artifacts/results/a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet/tiles/a_camel__x__rain) and 27 in [the single-prompt folder](../../artifacts/results/a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet/tiles/single), each a cell of a sheet placed above. They stay as the source for any new call | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Call the training pictures by eye, which is step 4 of the [route on whether an effect needs support](../../plans/10-what-poe-loses-when-it-composes/route-causal-does-an-effect-need-support.md). The hypothesis test needs this count.
- [ ] Decide whether a count nobody has made yet is "not reachable" under the written test.
- [ ] Decide whether the call on PoE at seed 9 stays dry, given the drops on the camel's chin and neck.
- [ ] Read the pilot's saved scores on the cluster to settle what CLIP's 7 wet calls are out of.
- [ ] Render and call the walk's second test on these seeds and this starting noise: PoE of "a camel" with "wet" and with "in the rain". The write-up lists 90 camel renders on the cluster that are not called.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
