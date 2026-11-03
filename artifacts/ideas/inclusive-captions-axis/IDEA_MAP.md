# 💡 Inclusive captions for the cat and dog experts

## Position in the idea

| Claim | Mark | Settled by |
|---|---|---|
| **1 (current). ⭐ Training both-animal photos under each single name makes "a photo of a cat" cover dog-present images** | **needs a check** | **render "a photo of a cat" under LoRAs A and B on the same seeds and count renders with a dog in them** |
| 2. That wider cat condition makes PoE compose a cat and a dog more often | open | |
| 3. LoRA A against LoRA B changes one variable | open | |
| 4. LoRA C, as written, is a distinct ablation | open | |
| 5. The LoRA with no both-animal photos is rung 11b's share-zero point | open | |
| 6. The pilot runs as specified on COCO, the scorer and one day of GPU | open | |

Load-bearing: claim 1. The idea dies without it. Claim 2 is what makes it worth a paragraph in the
paper.

## Table of contents

- [Position in the idea](#position-in-the-idea)
- [Quick context: where you are](#quick-context-where-you-are)
- [The idea, as it stands](#the-idea-as-it-stands)
- [The claims](#the-claims)
- [Design table](#design-table)
- [What the words are](#what-the-words-are)
- [Held claims](#held-claims)
- [Dead ends](#dead-ends)
- [Checks outstanding](#checks-outstanding)
- [Runs](#runs)
- [Routes](#routes)
- [Sources](#sources)
- [Next step](#next-step)

## Quick context: where you are

Navigation: ⬅️ [Position](#position-in-the-idea) | 📋 [TOC](#table-of-contents) | [Next](#the-idea-as-it-stands) ➡️

**What the idea is**

If SDXL is fine-tuned so that a photo holding both a cat and a dog is also trained under "a photo
of a cat" and under "a photo of a dog", each single-animal condition learns cat-present instead of
cat-only, and the product of the two conditions then holds more pictures with both animals in them.

**Where the walk is**

Claim 1, round 1.

**What compile would produce today**

Destination A, with claims 1 and 2 needing checks the pilot itself runs. The design table below is
not complete yet: claims 3, 4 and 6 fill its open cells.

**Bar and time box**

Every claim carries a mark that is not `open`, and the design table has no cell reading `tbd`.
Box: 90 minutes, started 2026-11-03.

## The idea, as it stands

Navigation: ⬅️ [Quick context](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claims) ➡️

A condition in a text-to-image model learns the distribution of the images it was trained under.
When every photo holding both animals is captioned only "a photo of a cat and a dog", the cat
condition sees cats without dogs, and fine-tuning pulls p(x | "a photo of a cat") toward cat-only.
Sampling each both-animal photo's caption from the subsets of its true labels (multi-label
conditioning with condition dropout) keeps dog-present photos inside the cat condition. The
product of experts multiplies the two conditions, so the place where both are high is the place
both animals appear, and that place should hold more mass when each condition already includes
the other animal.

## The claims

Navigation: ⬅️ [The idea](#the-idea-as-it-stands) | 📋 [TOC](#table-of-contents) | [Next](#design-table) ➡️

### 1. 🔍 ⭐ Training both-animal photos under each single name makes "a photo of a cat" cover dog-present images  ← current

Mark: needs a check (round 1 delivered, not settled).

**Said plainly.** With 300 cat-only photos and 200 both-animal photos drawn under "cat" 25% of the
time, about `50 / (300 + 50) = 14%` of LoRA B's "a photo of a cat" training draws hold a dog. In
LoRA A the share is 0%.

**What it commits the idea to.** Base SDXL's "a photo of a cat" rarely draws a dog, so most of the
effect has to come from B adding dog-present mass, with little from A removing it. In layer 5's
terms this is an overlap dial at about 14%; that toy put `0.30` of the product's mass deep in the
overlap at 5%. It also commits a rank-8 LoRA to learning from about
`2,000 × 200/800 × 0.25 ≈ 125` cat-captioned dog-present draws. Okawa's onset near 10 images says
that may be enough; it is not shown at this scale.

**The check.** Render "a photo of a cat" from base SDXL, A and B on the same 64 seeds and count
renders a dog detector fires on. Holds if base and A sit near 0 and B rises clearly toward 14%.

**The way round it, if B shows no dogs.** (a) Hold 8 of the 208 both-animal photos out of
training and compare the denoising loss of "a photo of a cat" on them under A and B; lower under
B means the condition widened even if renders hide it. (b) Raise the both-animal photos' cat and
dog shares to 40% each, or train longer; this moves claim 3's balance. (c) Accept a null on this
axis and move the idea to rung 11b's from-scratch model.

**Offered next:** `how`, `break`, `known`, `fix`, `settle` (marks it needs a check with both
checks named and moves to claim 2).

### 2. ⬜ That wider cat condition makes PoE compose a cat and a dog more often

Mark: open.

### 3. ⬜ LoRA A against LoRA B changes one variable

Mark: open.

### 4. ⬜ LoRA C, as written, is a distinct ablation

Mark: open.

### 5. ⬜ The LoRA with no both-animal photos is rung 11b's share-zero point

Mark: open.

### 6. ⬜ The pilot runs as specified on COCO, the scorer and one day of GPU

Mark: open.

## Design table

Navigation: ⬅️ [The claims](#the-claims) | 📋 [TOC](#table-of-contents) | [Next](#what-the-words-are) ➡️

The table the `--request` walk trains from. A cell reading `tbd` is filled by the claim named in
it.

| Arm | Cat-only photos | Dog-only photos | Both-animal photos, captions per draw | Empty-caption rate, all photos | Steps | Seeds rendered |
|---|---|---|---|---|---|---|
| base SDXL | none | none | none | none | 0 | tbd (claim 6) |
| A | 300, "a photo of a cat" | 300, "a photo of a dog" | 200, joint only | tbd (claim 3) | 2,000 | tbd (claim 6) |
| B | same 300 | same 300 | same 200: cat 25%, dog 25%, joint 40%, empty 10% | tbd (claim 3) | 2,000 | tbd (claim 6) |
| C | same 300 | same 300 | tbd (claim 4) | tbd (claim 3) | 2,000 | tbd (claim 6) |
| no-both | same 300 | same 300 | none | tbd (claim 3) | 2,000 | tbd (claim 6) |

## What the words are

Navigation: ⬅️ [Design table](#design-table) | 📋 [TOC](#table-of-contents) | [Next](#held-claims) ➡️

**Multi-label conditioning with condition dropout.** Training an image that carries several true
labels under a random subset of them each time it is drawn. The unconditional branch of
classifier-free guidance is the case where the subset is empty. Confident on the practice; the
exact phrase varies by paper.

**Inherited support.** The images a fine-tuned condition can still produce because pretraining put
mass there. Layer 5 of the simple-distributions ladder measured it eroding under fine-tuning.

## Held claims

Navigation: ⬅️ [What the words are](#what-the-words-are) | 📋 [TOC](#table-of-contents) | [Next](#dead-ends) ➡️

None yet.

## Dead ends

Navigation: ⬅️ [Held claims](#held-claims) | 📋 [TOC](#table-of-contents) | [Next](#checks-outstanding) ➡️

None yet.

## Checks outstanding

Navigation: ⬅️ [Dead ends](#dead-ends) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

None settled yet; each claim's check is named in its round.

## Runs

Navigation: ⬅️ [Checks outstanding](#checks-outstanding) | 📋 [TOC](#table-of-contents) | [Next](#routes) ➡️

None yet.

## Routes

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#sources) ➡️

None yet. Returns land in `routes/` beside this file.

## Sources

Navigation: ⬅️ [Routes](#routes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

- [The source design](source-design.md), from a ChatGPT conversation.
- [The rung 11b design](../../notes/what-poe-loses-when-it-composes/evidence-poe-fails-at-scale.md#what-this-means-for-rung-11b),
  with Okawa et al. 2310.09336, Liang et al. 2408.13256, Jeong et al. 2605.00273, Zarei et al.
  2406.07844, Kamb and Ganguli 2412.20292.
- [Layer 5's fine-tune figure](../../drips/simple-distributions-rebuild/figures/walk05-step7-finetune-decay.png).
- The project's PoE combine, `poe_repair/training_cache.py:11`.
- [The route this walk serves](../../../plans/route-inclusive-captions-pilot.md).

## Next step

Navigation: ⬅️ [Sources](#sources) | 📋 [TOC](#table-of-contents)

Walk claim 1.
