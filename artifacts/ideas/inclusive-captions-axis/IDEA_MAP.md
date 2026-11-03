# 💡 Inclusive captions for the cat and dog experts

## Position in the idea

| Claim | Mark | Settled by |
|---|---|---|
| 1. ⭐ Training both-animal photos under each single name makes "a photo of a cat" cover dog-present images | needs a check | the held-out denoising loss of "a photo of a cat" on 8 unseen both-animal photos under A against B, now the primary read: the base rate came back at 0 of 64 by eye (a clean floor) and the per-class detector read is unusable. Plus the real dog-present share from COCO's own counts |
| 2. That wider cat condition makes PoE compose a cat and a dog more often | needs a check | PoE compose rate under A against B on the same seeds, scored in three categories, pair against chimera against single animal, from full-resolution tiles |
| **3 (current). LoRA A against LoRA B changes one variable** | **wrong as stated** | **repaired: every arm gets the same 10% empty-caption rate on every photo, so the both-animal caption split is the only cell that differs** |
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

Claim 3. Claims 1 and 2 settled.

**What compile would produce today**

Destination A, with claims 1 and 2 needing checks the pilot itself runs. The design table's
empty-caption column is now fixed; claims 4 and 6 fill the cells that remain.

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

1. **The base rate, done 2026-11-03, and it answered two questions.** By eye, all 64 renders of
   "a photo of a cat" from base SDXL are a single cat and none holds a dog, so the floor is clean
   and any rise under B is attributable to the captioning. By detector, all 64 report one dog:
   GroundingDINO grounds "dog" on a cat at a median confidence of 0.889 against 0.943 for "cat",
   the median margin is 0.054, 54 of 64 sit under a 0.10 margin, and the sign flips on 3. No
   threshold or margin rule separates the classes, so the render-and-count check is dead as
   designed and read 3 below becomes the primary one. Any future class read needs a classifier on
   the cropped box rather than a second detector query. Card and sheet at
   `artifacts/results/does-the-cat-prompt-already-draw-dogs/`.

   The original specification, kept because the three outcomes it names are still the right ones
   for a working instrument: "a photo of a cat" from base SDXL on 64 seeds at 50 DDIM steps
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

### 2. ✅ That wider cat condition makes PoE compose a cat and a dog more often

Mark: needs a check, by the PoE compose rate under A against B, scored in three categories.

**Said plainly.** If the cat condition's support includes dog-present images and the dog
condition's includes cat-present ones, their product has more mass where both animals appear, so
PoE sampling under B composes more often than under A.

**What it commits the idea to.** That the added mass lands on co-occurrence rather than on blends.
A chimera is the argmax of the product in this project's own finding, because a blended cat-dog
animal scores high under both conditions at once. Widening both conditions adds mass and the
captioning scheme says nothing about where it lands, so B could compose no more often than A and
produce better-looking chimeras instead.

**Why it survives.** COCO's 169 one-of-each photos are photographs of a cat and a dog in one
frame. They are co-occurrence images and hold no hybrids, so the mass B adds is co-occurrence mass
by construction. The condition attached is about the scoring rather than the training: a chimera
and a genuine pair both satisfy "a cat is present and a dog is present" at the whole-image level,
so the instance count has to be read per class, since it cannot tell a cat and a dog from two dogs.

**The check.** PoE compose rate under A against B, same seeds, same guidance, scored in three
categories rather than two: pair, chimera, single animal. Calls come from full-resolution tiles,
because composition calls overcall on downscaled sheets, where two hybrids read as a clear cat and
dog at sheet scale.

**Why the third category is not optional.** If B moves renders from single animal to chimera
without reaching pair, that is a result about where the added mass goes. Folded into a two-category
score it reads as a null and is lost.

### 3. ❌ LoRA A against LoRA B changes one variable  ← current

Mark: wrong as stated. Repaired below, and the repair fills the design table's empty-caption column.

**Said plainly.** A and B were meant to differ only in how the 200 both-animal photos are
captioned, so any difference between them is attributable to the caption assignment alone.

**Why it fails as written.** The source design gives LoRA B a 10% non-specific caption and says
nothing about one for A. So A and B differ in two things at once: the both-animal caption split,
and the share of training draws that carry an empty caption. The empty-caption share is the
unconditional branch's training share, which is what classifier-free guidance subtracts at every
step, in every condition, for every prompt. Moving it moves the guidance behaviour of the whole
model rather than the cat condition alone. A difference between A and B then has two candidate
causes and the comparison cannot separate them.

**The repair.** One empty-caption rate, 10%, applied to every photo in every arm. The both-animal
caption split then absorbs the whole difference:

| Arm | Both-animal photos, captions per draw |
|---|---|
| A | joint 90%, empty 10% |
| B | cat 25%, dog 25%, joint 40%, empty 10% |

Cat-only and dog-only photos carry their own name 90% of the time and an empty caption 10% in
every arm. Every arm's shares sum to 1, every arm's unconditional share is identical, and the
only cell that differs between A and B is the both-animal split. That is the one-variable
comparison the claim wanted.

**What the repair costs.** A's joint share drops from 100% to 90% of its both-animal draws, so A
is no longer the pure "joint caption only" arm the source described. The no-both arm is what
carries the zero-both-animal-photos case, so nothing is lost by it.

**The risk the repair does not remove.** A 10% empty-caption rate on a 2,000-step run is about 200
unconditional draws, which may be too few to hold the unconditional branch steady against the base
model's own. That is a risk shared identically by every arm, so it cannot confound A against B; it
can only weaken all four together. Worth recording, not worth a design change.

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
| A | 300, "a photo of a cat" | 300, "a photo of a dog" | 200: joint 90%, empty 10% | 10% | 2,000 | tbd (claim 6) |
| B | same 300 | same 300 | same 200: cat 25%, dog 25%, joint 40%, empty 10% | 10% | 2,000 | tbd (claim 6) |
| C | same 300 | same 300 | tbd (claim 4) | 10% | 2,000 | tbd (claim 6) |
| no-both | same 300 | same 300 | none | 10% | 2,000 | tbd (claim 6) |

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

**Claim 1, the base rate.** Done. 0 of 64 by eye, 64 of 64 by detector; see Runs.

**The instrument.** Whether a classifier on the cropped box (CLIP zero-shot, or an ImageNet
classifier) separates cat from dog on these 64 renders well enough to measure claim 1 and to score
claim 2's three categories. Nothing in the repo does this yet, and every class-level read in the
pilot now depends on it.

**Claim 1, the real dog-present share.** Not started. Needs COCO's annotations on the cluster and
the pilot's filtering rule, which claim 6 has to fix first.

**Claim 1, the held-out denoising loss.** Blocked on the missing image-denoising LoRA trainer.

## Runs

Navigation: ⬅️ [Checks outstanding](#checks-outstanding) | 📋 [TOC](#table-of-contents) | [Next](#routes) ➡️

| # | What | Where | State |
|---|---|---|---|
| 1 | base SDXL "a photo of a cat" on 64 seeds, per-class dog and cat instance counts, the floor for claim 1 | mscluster106 device 1, outside Slurm | done 2026-11-03, 1,316s. By eye 0 of 64 renders hold a dog, which is the clean floor. By detector 64 of 64 report one dog, which kills the per-class read. Card and sheet at `artifacts/results/does-the-cat-prompt-already-draw-dogs/` (gitignored, on disk only) |

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

Walk claim 3.
