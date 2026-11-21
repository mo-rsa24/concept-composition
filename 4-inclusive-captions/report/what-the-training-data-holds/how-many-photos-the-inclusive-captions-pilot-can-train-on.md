# 📒 Report: how many photos can the inclusive-captions pilot train on? Recorded verdict: "131 usable both-animal photos" (counts file)

**The claim**

The pilot's design wanted 200 COCO photos holding both a cat and a dog. COCO train2017 has 208, and 131
clear the size filter, which is the number the
[counts file](../../artifacts/results/how-many-photos-the-inclusive-captions-pilot-can-train-on/counts.json)
records as its verdict. Looking at all 131 then kept 91. The pool finally trained on holds 150
both-animal photos, of which 38 are from COCO and 112 were generated.

## Table of contents

- [What would have counted](#what-would-have-counted)
- [1. From COCO to the size filter](#1-from-coco-to-the-size-filter)
- [2. From the size filter to the photos trained on](#2-from-the-size-filter-to-the-photos-trained-on)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## What would have counted

Navigation: 📋 [TOC](#table-of-contents) | [Next](#1-from-coco-to-the-size-filter) ➡️

**The target was 300 cat-only, 300 dog-only and 200 both-animal photos.** No pass rule was written for
the count itself. [Plan 01's audit step](../../plans/11-do-inclusive-captions-help-poe-compose/plans/01-the-inclusive-captions-pilot.md#8--audit-the-training-photos)
names one failure: more than about 20 rejected both-animal photos means the single-name caption shares
must rise before anything trains. The audit rejected 33.

## 1. From COCO to the size filter

Navigation: ⬅️ [Previous](#what-would-have-counted) | 📋 [TOC](#table-of-contents) | [Next](#2-from-the-size-filter-to-the-photos-trained-on) ➡️

<a href="../../artifacts/results/how-many-photos-the-inclusive-captions-pilot-can-train-on/both_sheet_00.png"><img src="../../artifacts/results/how-many-photos-the-inclusive-captions-pilot-can-train-on/both_sheet_00.png" width="200" alt="Contact sheet of 48 COCO photos holding a cat and a dog, each tile labelled with its audit number, the two animals' box sizes and their overlap"></a>

**The both-animal set is the only one short of its target, by 69 photos.** The sheet is the first 48 of
the 131, each tile labelled with its audit number, each animal's box as a fraction of the image's longer
side, and how much the two boxes overlap. No person's face shows on this page. The other two pages are
linked and left unembedded because they do: [page 2](../../artifacts/results/how-many-photos-the-inclusive-captions-pilot-can-train-on/both_sheet_01.png)
has people's faces on tiles 48, 73 and 88, and [page 3](../../artifacts/results/how-many-photos-the-inclusive-captions-pilot-can-train-on/both_sheet_02.png)
has a person's face on a television in tile 103 `verified` (each page opened).

Counts of images in COCO train2017, from the counts file `verified`:

| Set | In COCO | After the size filter (box at least 0.25) | Wanted | Short by |
|---|---|---|---|---|
| cat only | 3,906 | 3,335 | 300 | 0 |
| dog only | 4,177 | 2,708 | 300 | 0 |
| both animals | 208 (169 with exactly one of each) | 131 | 200 | 69 |
| neither animal | 108,975 | 108,975 | 200 | 0 |

## 2. From the size filter to the photos trained on

Navigation: ⬅️ [Previous](#1-from-coco-to-the-size-filter) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**Each later step changed the both-animal count.** The audit called each of the 131 tiles against "one
live cat and one live dog, both really in the photograph".

| Step | Both-animal photos | Source | Mark |
|---|---|---|---|
| hold both labels in COCO | 208 | counts file | `verified` |
| clear the 0.25 size filter | 131 | counts file | `verified` |
| kept by the eye audit | 91 (33 rejected, 7 uncertain) | [audit verdicts](../../artifacts/results/how-many-photos-the-inclusive-captions-pilot-can-train-on/audit-verdicts.json) | `verified` |
| second pool: COCO kept by the vision-language judge | 38 of 64 candidates | [plan 01 task 2.6](../../plans/11-do-inclusive-captions-help-poe-compose/plans/01-the-inclusive-captions-pilot.md#2--finish-the-data-build) | `stated` |
| second pool: generated photos used | 112 (243 of 306 passed the judge) | plan 01 task 2.6 | `stated` |
| trained on | 150, beside 200 cat and 200 dog, 550 in all | plan 01 task 2.6 and [the pilot's card](../../artifacts/results/does-captioning-both-animal-photos-under-each-name-help-poe-compose/README.md) | `stated` |

The audit was made by the assistant at the owner's request, from 448-pixel tiles `verified` (the
`judged_by` and `judged_from` fields).

## Where the sources disagree

Navigation: ⬅️ [Previous](#2-from-the-size-filter-to-the-photos-trained-on) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Four recorded facts conflict, and each is shown here unresolved.**

| What | One source | The other source |
|---|---|---|
| usable both-animal photos | the counts file and [its card](../../artifacts/results/how-many-photos-the-inclusive-captions-pilot-can-train-on/README.md): 131 | the audit: 91 clear. The pool trained on: 150, of which 38 are COCO |
| what the 131 photos are | the card: "domestic photographs of a cat and a dog in one frame, and no hybrids" | the audit rejects 33, including drawings, collages and photos with one animal |
| what the size filter measures | the card: an animal "smaller than a quarter of the frame" | the sheet's title and plan 01: the box as a fraction of the image's longer side |
| the photos in the plan's headline | the [master plan](../../plans/11-do-inclusive-captions-help-poe-compose/MASTER_PLAN.md#depends-on): 200, 200 and 91 | plan 01 task 2.6: 200, 200 and 150 |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**These counts say what is available and nothing about whether it is enough.** Whether 150 both-animal
photos move the product is the [pilot's question](../do-inclusive-captions-help-the-product-compose/does-captioning-both-animal-photos-under-each-name-help-poe-compose.md),
which has no compose rate yet.

**The audit has one reader.** Seven tiles were left uncertain, and the card's single-animal sets have no
recorded audit in this folder.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. The project numbers are from the files named above.

**A box fraction of the longer side**

A photo is 400 pixels wide and 300 tall, so its longer side is 400. A dog's box 120 pixels across scores
120 / 400 = 0.30 and stays. One 80 across scores 0.20 and goes. Here the floor is 0.25, and it removes 77
of the 208 both-animal photos (208 − 131).

**The share of "cat" draws that hold a dog**

A jar with 9 plain sweets and 1 striped one gives a striped sweet 1 draw in 10. With 300 cat-only photos
captioned "cat" on 90% of draws, and 131 both-animal photos on 40%, the share holding a dog is
52.4 / (270 + 52.4) = 16.3%, the figure plan 01 opens with. That is why the count of both-animal photos
sets the size of the effect.

**An eye audit**

A label says what an annotator marked, and a person checks what is really there. Of 10 labelled apples, 2
might be wax. Here 131 labelled photos became 91 clear, 33 rejected and 7 uncertain, and 91 + 33 + 7 = 131.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| counts and the recorded verdict | [counts file](../../artifacts/results/how-many-photos-the-inclusive-captions-pilot-can-train-on/counts.json), read 2026-11-21 | `verified` |
| the audit | [audit verdicts](../../artifacts/results/how-many-photos-the-inclusive-captions-pilot-can-train-on/audit-verdicts.json), read 2026-11-21 | `verified` |
| how the sheets were made | [results card](../../artifacts/results/how-many-photos-the-inclusive-captions-pilot-can-train-on/README.md) | `stated` |
| the second pool | [plan 01 task 2.6](../../plans/11-do-inclusive-captions-help-poe-compose/plans/01-the-inclusive-captions-pilot.md#2--finish-the-data-build) | `stated` |
| what an animal pair is here | [animal pair](../../../context/world/animal-pair.md) | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Update the card so its answer of 131 sits beside the audit's 91 and the 150 trained on.
- [ ] Update the master plan and the review file, which still describe 491 photos.
- [ ] Settle the seven uncertain tiles, or record that they stay out.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
