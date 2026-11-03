# 💡 Inclusive captions for the cat and dog experts

## Position in the idea

| Claim | Mark | Settled by |
|---|---|---|
| 1. ⭐ Training both-animal photos under each single name makes "a photo of a cat" cover dog-present images | needs a check | three reads, in order: the base rate of dogs under "a photo of a cat" (running, mscluster106 device 1); the real dog-present share from COCO's own counts; the held-out denoising loss of "a photo of a cat" on 8 unseen both-animal photos under A against B |
| **2 (current). That wider cat condition makes PoE compose a cat and a dog more often** | **open** | |
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

Claim 2, opening. Claim 1 settled after two rounds.

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

### 1. ✅ ⭐ Training both-animal photos under each single name makes "a photo of a cat" cover dog-present images

Mark: needs a check, with three reads named below.

**Said plainly.** With 300 cat-only photos and 200 both-animal photos drawn under "cat" 25% of
the time, about `50 / (300 + 50) = 14%` of LoRA B's "a photo of a cat" training draws hold a dog.
In LoRA A the share is 0%.

**What it commits the idea to.** Base SDXL's "a photo of a cat" rarely draws a dog, so most of the
effect has to come from B adding dog-present mass, with little from A removing it. In layer 5's
terms this is an overlap dial at about 14%; that toy put `0.30` of the product's mass deep in the
overlap at 5%. It also commits a rank-8 LoRA to learning from about
`2,000 × 200/800 × 0.25 ≈ 125` cat-captioned dog-present draws. Okawa's onset near 10 images says
that may be enough; it is not shown at this scale.

**Where the field already measured this, with the sign flipped.** The customization literature
measures the same quantity and calls it a defect: a fine-tuned concept pulling the other concept
in unprompted is concept bleeding, and MultiCompose (2608.03708) reports leakage, confusion and
missing-concept errors as three things to minimise, with Custom Diffusion (2212.04488) as the
earlier anchor. DreamBooth-family training carries a prior-preservation term precisely to stop a
class condition collapsing onto the fine-tuning images, which is the narrowing this claim asserts,
taken as given and never named as a change to the condition's support. SynGen (2306.08877) defines
Proper Binding, Improper Binding and Entity Neglect over generated images; Improper Binding is the
count this claim wants, read as a positive. Nobody has run the experiment that deliberately induces
the effect, because in every setting where it was measured it was a bug.

**The strongest argument against it, and what it changes.** The denoising objective under a fixed
prompt is minimised by the conditional mean of its targets given the noisy latent and the time
step. At high noise the latent carries almost nothing about layout, so the "cat" condition's best
prediction there averages its 86% cat-only draws with its 14% dog-present ones, and that average
is one animal with a smeared second region rather than a choice between two modes. The
disambiguating signal arrives late, and this project's own early-window result puts the composition
decision in the first ten steps. A trained diffusion model is a sampler rather than a regressor, so
14% of the mass should appear as 14% of renders, but that is not guaranteed. The argument does not
kill the claim; it adds a third outcome to the check:

| What the renders show | What it means |
|---|---|
| base and A near zero, B rising toward 14% | the condition widened; claim 1 holds |
| all three near zero, B's cats look like A's cats | the condition did not widen; a clean null |
| all three near zero, B's cats visibly worse than A's, blurred or carrying a second indistinct animal shape | the condition absorbed the dog-present draws as averaging; go after mode selection, not the caption share |

Reading the third row as a null would retire the idea for the wrong reason.

**The three reads that settle it, in order.**

1. **The base rate, running now.** "a photo of a cat" from base SDXL on 64 seeds at 50 DDIM steps
   and guidance 7.5, with per-class detector counts for "dog" and "cat" read apart, since the
   compose scorer's generic `"animal"` query cannot tell them apart.
   [The script](../../../scripts/census/cat_condition_dog_rate.py) carries the three outcomes and
   their meanings in its own docstring. Zero or one hit in 64 gives claim 1 a clean floor; five or
   more means the base already leaks dogs into "cat", 64 seeds is too few, and the read below
   replaces the render count.
2. **The real dog-present share, no GPU, about 2 minutes.** The 14% comes from the illustrative
   300 and 200. Compute it from COCO's own counts after the pilot's filtering of the 169
   one-of-each photos. A usable both-animal count near 100 rather than 200 halves the share to
   about 7% and halves the effect size with it.
3. **The held-out denoising loss, minutes on one GPU.** Hold 8 both-animal photos out, then
   compare the loss of "a photo of a cat" on them under A and under B. Lower under B says the
   condition widened even where renders hide it, and it reads the density rather than the samples,
   so it is immune to the averaging failure above.

**The ways round a null, accepted in order.** (a) the held-out loss read, which is read 3 above.
(b) Raise the single-name shares to 40% each: doubles the effect size, and costs claim 3's
one-variable comparison unless the no-both arm absorbs the difference. (c) Train the cross-attention
key and value projections only, putting all 125 dog-present draws into the parameters that carry
the text: sharper per step, cheaper, holds the 2,000-step budget, and costs generality, since the
result is then about the text pathway alone. (d) Accept the null and move the idea to rung 11b's
from-scratch model, which has no inherited support to erode, so the captioning axis is the only
thing setting the condition's support: weeks rather than days.

**What blocks the rest.** Every A-against-B read needs an image-denoising LoRA trainer, and none
exists in this repo. That is a claim-6 problem and it is the pilot's critical path.

### 2. 🔍 That wider cat condition makes PoE compose a cat and a dog more often  ← current

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

**Claim 1, the base rate.** Running since 2026-11-03 on mscluster106 device 1, pid 1946628,
log at `/datasets/mmolefe/poe_repair_min/outputs/cat_condition_dog_rate/logs/base-sdxl.log`,
results at `.../cat_condition_dog_rate/base-sdxl/summary.json`. Harvest with `pgrep`, not
`squeue`: it runs outside Slurm.

**Claim 1, the real dog-present share.** Not started. Needs COCO's annotations on the cluster and
the pilot's filtering rule, which claim 6 has to fix first.

**Claim 1, the held-out denoising loss.** Blocked on the missing image-denoising LoRA trainer.

## Runs

Navigation: ⬅️ [Checks outstanding](#checks-outstanding) | 📋 [TOC](#table-of-contents) | [Next](#routes) ➡️

| # | What | Where | State |
|---|---|---|---|
| 1 | base SDXL "a photo of a cat" on 64 seeds, per-class dog and cat instance counts, the floor for claim 1 | mscluster106 device 1, outside Slurm, pid 1946628 | launched 2026-11-03 |

## Routes

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#sources) ➡️

**Whether PoE can use any of the cross-attention repairs.** Emitted from claim 1's `known` round.
The binding and omission literature repairs composition by acting on cross-attention maps inside
one forward pass, and PoE adds two experts' scores after each pass, so none of those repairs has a
place to act. A separate `/drip-idea` walk asks whether that makes PoE ill-posed for compositional
generation, or whether it is the one method that sidesteps the text encoder's own limits. Returns
land in `routes/` beside this file.

## Sources

Navigation: ⬅️ [Routes](#routes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

- [The source design](source-design.md), from a ChatGPT conversation.
- [The rung 11b design](../../notes/what-poe-loses-when-it-composes/evidence-poe-fails-at-scale.md#what-this-means-for-rung-11b),
  with Okawa et al. 2310.09336, Liang et al. 2408.13256, Jeong et al. 2605.00273, Zarei et al.
  2406.07844, Kamb and Ganguli 2412.20292.
- [Layer 5's fine-tune figure](../../drips/simple-distributions-rebuild/figures/walk05-step7-finetune-decay.png).
- The project's PoE combine, `poe_repair/training_cache.py:11`.
- The customization line that measures this claim's quantity as a defect: Kumari et al. 2212.04488
  (Custom Diffusion), MultiCompose 2608.03708, and SynGen 2306.08877 for the Proper Binding,
  Improper Binding and Entity Neglect counts.
- [The base-rate script](../../../scripts/census/cat_condition_dog_rate.py), whose docstring
  carries the three outcomes and what each means.
- [The route this walk serves](../../../plans/route-inclusive-captions-pilot.md).

## Next step

Navigation: ⬅️ [Sources](#sources) | 📋 [TOC](#table-of-contents)

Walk claim 2.
