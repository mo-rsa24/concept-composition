# 🔬 Report: do inclusive captions help the product of experts compose?

What this project found, one page per question, each claim beside a figure and the statistic that backs
it. A verdict appears only where a card, a results file or a plan records one, and each row names that
source. The training captions are the one thing this project changes: a picture holding two things is
also trained under each thing's own name.

## Table of contents

- [What was found](#what-was-found)
- [What the training data holds](#what-the-training-data-holds)
- [Decisions waiting on the owner](#decisions-waiting-on-the-owner)
- [Figures still missing](#figures-still-missing)
- [Still open](#still-open)
- [Studied](#studied)

## What was found

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-the-training-data-holds) ➡️

**Read this table before asserting any result: only the toy has a recorded yes, and the MNIST and SDXL
rows carry open conditions.** Read a page when you need the evidence behind its row.

| The question | Recorded verdict | The number | The finding |
|---|---|---|---|
| Does relabelling both-shapes pictures help the toy product compose? | "Yes, once about 9%" (card) | [separate pairs](../../3-understanding-compositionality/report/the-exact-toy/what-poe-needs-from-training-to-draw-two-things-at-once.md) of 300 at a 9.1% share: 13, 5, 72; at 23.1%: 66, 64, 146 | [the toy relabelling finding](do-inclusive-captions-help-the-product-compose/does-relabelling-both-shapes-pictures-help-the-toy-product-compose.md) |
| Do inclusive labels help the product compose on MNIST? | ⚪ null (plan 02) | highest counting mean 15.33 of 256 against a line of 39; dial 0.6 misses its floor by 2 seeds and would read 52.33 | [the MNIST labels finding](do-inclusive-captions-help-the-product-compose/do-inclusive-labels-help-the-product-compose-on-mnist.md) |
| Does the share of two-digit pictures help the product compose on MNIST? | "NULL" (card), plan at 0 of 9 tasks ticked, judge not checked by eye | corrected 7 × 8 at the 10% share: mean 23 of 256 against a line of 39 | [the MNIST share finding](do-inclusive-captions-help-the-product-compose/does-the-share-of-two-digit-pictures-help-the-product-compose-on-mnist.md) |
| Does captioning both animal photos under each name help PoE compose? | none: "a sheet and not a verdict" (card) | 0 of 6 seeds show two animals in plain SDXL, run A or run B; no [compose rate](../../context/world/compose-rate.md#what-a-compose-rate-is) exists | [the SDXL pilot finding](do-inclusive-captions-help-the-product-compose/does-captioning-both-animal-photos-under-each-name-help-poe-compose.md) |

## What the training data holds

Navigation: ⬅️ [Previous](#what-was-found) | 📋 [TOC](#table-of-contents) | [Next](#decisions-waiting-on-the-owner) ➡️

**Read these before relying on a claim about what the models were trained on.**

| The question | Recorded verdict | The number | The finding |
|---|---|---|---|
| How many photos can the pilot train on? | "131 usable both-animal photos" (counts file) | 208 in COCO, 131 after the size filter, 91 after the eye audit, 150 trained on (38 from COCO) | [the photo pool finding](what-the-training-data-holds/how-many-photos-the-inclusive-captions-pilot-can-train-on.md) |
| Does any caption say a cat and a dog, and does the picture show both? | none recorded | 1,102,469 captions name both; 0 of 117 sampled rows have been marked by eye | [the caption search finding](what-the-training-data-holds/does-any-caption-say-a-cat-and-a-dog.md) |

## Decisions waiting on the owner

Navigation: ⬅️ [Previous](#what-the-training-data-holds) | 📋 [TOC](#table-of-contents) | [Next](#figures-still-missing) ➡️

**Four choices change what a page may claim, and none has been made.**

- [ ] [Whether dial 0.6 counts on MNIST](do-inclusive-captions-help-the-product-compose/do-inclusive-labels-help-the-product-compose-on-mnist.md#2-the-answer-under-each-reading-of-the-rule), which moves that reading from null to inconclusive.
- [ ] [Whether the toy's "Yes" and its corrector sentence stand](do-inclusive-captions-help-the-product-compose/does-relabelling-both-shapes-pictures-help-the-toy-product-compose.md#where-the-sources-disagree), given a replacing-design run that misses its written rule at four of six shares.
- [ ] [Whether the MNIST share NULL stands before its judge is checked by eye](do-inclusive-captions-help-the-product-compose/does-the-share-of-two-digit-pictures-help-the-product-compose-on-mnist.md#where-the-sources-disagree).
- [ ] [Whether the SDXL criterion, written for 491 photos, applies to the 550 trained](do-inclusive-captions-help-the-product-compose/does-captioning-both-animal-photos-under-each-name-help-poe-compose.md#where-the-sources-disagree).

## Figures still missing

Navigation: ⬅️ [Previous](#decisions-waiting-on-the-owner) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

**Two evidence pairs are waiting on a render.**

- [ ] 🖼️ [The SDXL compose-rate figure](do-inclusive-captions-help-the-product-compose/does-captioning-both-animal-photos-under-each-name-help-poe-compose.md#2-the-planned-test): compose rate for base SDXL, run A and run B on 128 seeds. Render into `artifacts/results/do-inclusive-captions-help-poe-compose/`, then embed.
- [ ] 🖼️ [The caption-form bars](what-the-training-data-holds/does-any-caption-say-a-cat-and-a-dog.md#1-how-the-captions-name-the-two-animals): captions per form from `counts.json`. Render into `artifacts/results/does-any-caption-say-a-cat-and-a-dog/`, then embed.

## Still open

Navigation: ⬅️ [Previous](#figures-still-missing) | 📋 [TOC](#table-of-contents) | [Next](#studied) ➡️

- [ ] Run the 128-seed scoring pass for the SDXL pilot.
- [ ] Run the MNIST share run's judge-against-eye check, plan 03 task 4.2.
- [ ] Mark the 117 rows on the caption check page.
- [ ] Write a card for the MNIST labels results folder, which has none.
- [ ] Bring the toy and MNIST share cards up to date with the result files added after they were written.

## Studied

Navigation: ⬅️ [Previous](#still-open) | 📋 [TOC](#table-of-contents)

| Item | Studied |
|---|---|
| [Does relabelling both-shapes pictures help the toy product compose](do-inclusive-captions-help-the-product-compose/does-relabelling-both-shapes-pictures-help-the-toy-product-compose.md) | - [ ] |
| [Do inclusive labels help the product compose on MNIST](do-inclusive-captions-help-the-product-compose/do-inclusive-labels-help-the-product-compose-on-mnist.md) | - [ ] |
| [Does the share of two-digit pictures help the product compose on MNIST](do-inclusive-captions-help-the-product-compose/does-the-share-of-two-digit-pictures-help-the-product-compose-on-mnist.md) | - [ ] |
| [Does captioning both animal photos under each name help PoE compose](do-inclusive-captions-help-the-product-compose/does-captioning-both-animal-photos-under-each-name-help-poe-compose.md) | - [ ] |
| [How many photos the pilot can train on](what-the-training-data-holds/how-many-photos-the-inclusive-captions-pilot-can-train-on.md) | - [ ] |
| [Does any caption say a cat and a dog](what-the-training-data-holds/does-any-caption-say-a-cat-and-a-dog.md) | - [ ] |

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
