# 💡 Inclusive captions for the cat and dog experts

## Position in the idea

| Claim | Mark | Settled by |
|---|---|---|
| 1. ⭐ Training both-animal photos under each single name makes "a photo of a cat" cover dog-present images | needs a check | two reads, both live again: the dog-present render count under A against B, scored by the CLIP crop call rather than by a detector query, and the held-out denoising loss on 8 unseen both-animal photos. Plus the real dog-present share from COCO's own counts |
| 2. That wider cat condition makes PoE compose a cat and a dog more often | needs a check | PoE compose rate under A against B on the same seeds, scored in three categories, pair against chimera against single animal, from full-resolution tiles |
| 3. LoRA A against LoRA B changes one variable | wrong as stated, settled | **repaired: every arm gets the same 10% empty-caption rate on every photo, so the both-animal caption split is the only cell that differs. The repair held under the drift argument; what it leaves is a noise floor, not a confound** |
| 4. LoRA C, as written, is a distinct ablation | wrong as stated, settled | **no assignment rule makes C differ from B in one thing: matching the draw counts leaves the cat condition seeing 50 photos against B's 200. Recommendation is to drop C and buy a second training seed for A and B** |
| 5. The LoRA with no both-animal photos is rung 11b's share-zero point | wrong as stated, settled | **repaired: 200 photos holding neither animal replace the removed both-animal photos, so every arm trains on 800 and the cat-only photos are seen equally often. Even repaired it is one anchor, not the sweep** |
| 6. The pilot runs as specified on COCO, the scorer and one day of GPU | wrong as stated, settled | **repaired: about 9 GPU hours once the pieces exist, but the trainer the pilot needs is not in this repo and writing it is the real cost. Seeds fixed at 128 for A and B, 48 for the rest** |

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

Closed. All six claims carry a mark and the design table has no blank cell. `compile` is the
next move.

**What compile would produce today**

Destination A, with claims 1 and 2 needing checks the pilot itself runs. The design table's
empty-caption column and LoRA C's row are now fixed; claim 6 fills the seed counts that remain.

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
   threshold or margin rule separates the classes. Naming the animal was then handed to a
   classifier on the cropped box, and that works: zero-shot CLIP ViT-B/32 calls cat on 64 of 64
   crops with a probability margin from 0.982 to 0.995. So the render-and-count check stands, with
   the count read through the CLIP crop call and the detector kept only for how many animals a
   render holds. Card and both sheets at
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

### 3. ❌ LoRA A against LoRA B changes one variable

Mark: wrong as stated, settled. Repaired below, and the repair fills the design table's
empty-caption column. The repair was then attacked on its own terms and survived; see the last
block.

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

**The argument that would reopen the repair, and why it does not.** PoE subtracts the
unconditional prediction twice. In `eps_uncond + g (eps_a + eps_b - 2 eps_uncond)` the
unconditional term carries the coefficient `1 - 2g`, which at guidance 7.5 is -14, against -6.5
in an ordinary one-condition render. So any movement in the unconditional branch is amplified
fourteenfold in a PoE sample, and a design that shares the empty-caption rate across arms is not
the same thing as a design where the unconditional branch is identical across arms.

It does not reopen the repair, because the empty-caption draws come from the same photo pool at
the same rate in every arm, so the unconditional branch's training distribution is identical by
construction. What differs between A and B is the realised draws, not the distribution they come
from. The amplification therefore does not create a cause; it sets a noise floor. The cost is
real and belongs in claim 6: an A-against-B difference in PoE compose rate has to clear a
run-to-run term that enters at -14, which argues for matched training seeds across arms and for
reading the single-expert renders beside the PoE ones, since the single-expert read carries the
same drift at -6.5 and so separates drift from composition.

**Why 0% dropout is not the cleaner variant.** Dropping the empty caption entirely would leave
the unconditional branch untrained, but the adapter is still attached during the unconditional
forward pass at sampling time, so `eps_uncond` moves anyway and moves without ever having been
fit. Training it on 10% of draws is the lesser of the two. The only design that truly freezes it
detaches the adapter on the unconditional pass, which is a sampler change rather than a training
one, and this repo has already paid for forgetting that an attached adapter runs on every pass.

### 4. ❌ LoRA C, as written, is a distinct ablation

Mark: wrong as stated. Repaired below, and the repair fills the design table's last caption cell.

**Said plainly.** The source gives three training schemes and calls C an ablation: inclusive
captioning "without dropping individual conditions". A third arm is worth a third of the GPU
budget only if it isolates something A and B do not already isolate between them.

**Why it fails as written.** The phrase is never defined, and both natural readings confound.

Read as *never dropping a single name*, every both-animal photo is trained under cat and dog and
joint on every pass instead of one sampled caption per draw. The both-animal photos then
contribute three times as many draws as they do in A and B, so C against B changes the caption
assignment and the effective amount of both-animal data at once.

Read as *no dropout*, the empty caption goes to zero and the shares become cat, dog and joint at a
third each. That moves the unconditional branch's training share, which is precisely the cell
claim 3 found confounding and fixed at 10% for every arm.

Either way C is not a distinct ablation; it is a second variable bolted to the one B already
tests.

**The repair.** C keeps B's shares and B's 10% empty rate, and assigns each both-animal photo one
caption once, fixed for the whole run, rather than resampling a caption each time the photo is
drawn. Across the 200 photos the assignment is drawn so the pool-level shares match B's exactly:
50 photos fixed at "cat", 50 at "dog", 80 at joint, 20 at empty.

**What C then isolates.** B and C put the same number of dog-present draws into the cat condition
from the same photos. They differ in whether a single photo is ever seen under more than one name.
If B beats C, the gain comes from one image being bound to two conditions, which is the mechanism
the idea actually claims. If B and C match, the gain is the share of dog-present draws alone and
the resampling is bookkeeping, which is a cheaper recipe and a weaker story. A null here does not
touch claim 1 or claim 2; it narrows the explanation.

**The repair does not hold either, and this is the finding.** Fixing one caption per photo
matches B's number of cat-captioned dog-present draws, and it cannot match how many different
photos those draws come from. Over a run, B shows the cat condition all 200 both-animal photos,
each one occasionally. C shows it the same 50 photos over and over. So C against B is still two
things at once: whether a photo is ever seen under two names, and whether the cat condition meets
200 dog-present scenes or 50. The two cannot be separated by any assignment rule, because matching
the variety means captioning every photo "cat" at some point, which is the resampling the arm
exists to remove.

**What is left, and the recommendation.** Three options. Keep C and state the confound, which buys
a result about fixed captioning as a whole rather than about multi-naming. Keep C and match
variety instead of draws by giving all 200 photos a fixed caption, which changes the shares and so
breaks the comparison in the other direction. Or drop C and spend its quarter of the budget on a
second training seed for A and B, which buys the run-to-run band that claim 3's amplification
argument says the comparison needs. The third is the recommendation: a confounded third arm is
worth less than knowing how big the noise is on the two arms that carry the claims.

**Where the decision belongs.** Claim 6, with the seed counts. The design table keeps C's row so
the option stays costed, marked as the arm the pilot may drop.

### 5. ❌ The LoRA with no both-animal photos is rung 11b's share-zero point

Mark: wrong as stated. Repaired below by a control the source proposed and the design table had
dropped. One limitation survives the repair and belongs in how the pilot is written up.

**Said plainly.** The source's phase B trains on the cat-only and dog-only photos with the
both-animal photos removed. Rung 11b sweeps the share of both-animal photos in the training set
from zero upward and reads the compose rate along it. If the two are the same configuration, the
pilot pays for 11b's leftmost point while answering its own question.

**What it commits the idea to.** That one of this pilot's arms can be lifted into a different
experiment's sweep without being retrained, which is what makes the pilot cheap for the paper
rather than a detour.

**Why it fails as written.** Rung 11b's rule is that the single-concept photos stay identical
while only the both-animal set moves. Removing 200 photos from a pool of 800 and training the same
2,000 steps breaks that: each cat-only photo is drawn a third more often in the no-both arm than
in A, B or C, so the arm differs in the both-animal share and in how hard the cat-only photos are
trained. A lower compose rate then has two causes.

**The repair.** Put 200 photos holding neither a cat nor a dog in the removed photos' place, so
every arm trains on 800 photos for 2,000 steps and every cat-only photo is drawn equally often in
all five. The source already proposed a background set of 100 to 200 neither-animal photos and the
design table had dropped it; this is where it earns its place. The cost is that the no-both arm now
also learns from scenes with no animal in them, which widens nothing about cats and dogs and is
shared by no other arm. That is a smaller asymmetry than the one it removes, and it is the price
of holding the draw counts equal.

**The limitation the repair does not remove.** Rung 11b wants shares at 0, 1 in 1,000, 1 in 100,
1 in 20 and 1 in 5, spaced that way because Okawa's onset near 10 images and Liang's percolation
threshold both put the bend at small counts. The pilot gives two points, 0 and 200 in 800, and the
second sits just above 11b's densest point. So the pilot anchors the sweep's ends and says nothing
about where the curve bends, which is the question 11b exists to answer. Written up as "the zero
anchor, measured" it is honest; written up as "11b's share axis, started" it is not.

**What rung 11b asks for that the pilot already does.** The text encoders stay frozen, so the
encoder is not a second variable. The scorer now labels one animal, two animals and a hybrid
apart, which is the failure-kind read 11b asks for. What the pilot does not match is 11b's
scene-complexity control: two-animal photos are busier than single-animal ones, and nothing in the
design table matches them for size and clutter. That one is recorded, not solved.

### 6. ❌ The pilot runs as specified on COCO, the scorer and one day of GPU  ← current (settled)

Mark: wrong as stated. The GPU arithmetic is roughly right and it is not the cost that matters.

**Said plainly.** The source costs the pilot as one day of GPU on COCO with this project's
existing scorer. If that is true the idea is cheap enough to try before deciding whether it is
right.

**Why it fails as written.** Three of the five pieces do not exist here.

The trainer is the blocker. This repo's LoRA trainer
(`poe_repair/experiments/one_pair_one_seed/trainer.py`) learns from cached noise tensors toward a
residual target: no images, no VAE encode, no captions, no text encoder in the loop. The pilot
needs an ordinary text-to-image denoising trainer, which is a different program. The libraries
for one are installed on the cluster (diffusers 0.29.2, peft 0.19.1, accelerate 1.13.0) and no
copy of the diffusers SDXL LoRA example is on disk, so the example is fetched and adapted rather
than written from nothing. That still means a caption sampler, the data pipeline below, and a
first run that is debugged rather than trusted.

The data build does not exist: selecting the cat-only, dog-only, both-animal and neither-animal
sets from COCO's annotations, auditing the both-animal ones by hand because annotations miss
animals, and recording each photo's assignment so claim 4's fixed-caption arm is reproducible.

The caption sampler does not exist: the per-draw die roll for arm B, the once-and-for-all
assignment for arm C, and the shared 10% empty rate claim 3 pinned.

What does exist is the scoring end: the instance count for how many animals, and now the CLIP
crop call for which animal, both already run on this project's renders.

**The GPU arithmetic, with the seed counts it forces.** Two numbers set everything else. Training
2,000 steps at 512 square, rank 8, is roughly an hour a run on one of the 48 GB cards. Rendering
one image at 1024 square, 50 steps, measured on this project's own base-rate run, is 20 seconds.

The seed count comes from the size of the move worth seeing. Comparing two rates at 80% power and
the usual 5% two-sided test, 48 seeds an arm resolves a move of about 20 points (5% against 25%)
and 128 seeds an arm resolves about 10 points (5% against 15%). A 14% caption share is not
expected to move the compose rate 20 points, so the two arms carrying claims 1 and 2 get 128 and
the rest get 48.

| Piece | Cost |
|---|---|
| training, 4 arms plus a second seed for A and B | about 6 hours |
| the watching sheets during training, see below | about 1 hour |
| rendering, 128 seeds each for A and B, 48 for base, C and no-both | about 2.5 hours |
| scoring, detector plus CLIP crop call on roughly 500 renders | under 1 hour |
| **total GPU** | **about 10 hours, which is one day** |

**What the run shows while it is training.** Every run reports to Weights and Biases, project
`prime_lab/poe-repair-inclusive-captions`, one W&B run per arm, grouped so the arms line up on
one axis. Per optimiser step: the loss, the learning rate and the gradient norm. At steps 0, 250,
500, 1,000 and 2,000 the run pauses and renders a sheet.

The sheet is eight fixed seeds down the rows and four columns across: "a photo of a dog", "a photo
of a cat", "a photo of a cat and a dog", and the product of the cat and dog conditions. The same
eight seeds at every checkpoint and in every arm, so the sheets stack into a film of one picture
changing as training proceeds, and the arms are comparable tile for tile.

The product column costs nothing extra to train, because combining two conditions happens at
sampling time and needs no training of its own: three predictions per step (cat, dog, empty)
against two for an ordinary render. That is why it can sit beside the three prompt columns from
step 0 onward, and it is the column the idea lives or dies on.

The sheets render at the training resolution of 512 square with 25 sampling steps, roughly four
seconds a tile, which is why an hour covers every checkpoint of every arm. The scored renders
stay at 1,024 square with 50 steps, because the composition calls are made on those and this
project has already been caught calling pairs off downscaled sheets.

**What the sheets may and may not be used for.** Eight seeds is a picture, not a rate. The sheet
says whether anything is happening and when, for instance whether the product column changes by
step 250 or only by step 2,000, which is free information about how fast the caption share bites.
The compose rate that answers claims 1 and 2 comes from the 128-seed renders at the end and from
nowhere else. A sheet that looks good at step 500 is not a result and must not be reported as
one.

**The repair.** The one-day figure stands for GPU and is wrong for the pilot. Written honestly:
one day of GPU after two or three days of writing the trainer, the data build and the caption
sampler, with the first training run treated as a debug run rather than an arm.

**What that changes about whether to run it.** The pilot stops being a cheap side bet and becomes
a week. The cheapest thing that still tests claim 1 is a single pair of runs, A and B, at 128
seeds, with C dropped per claim 4 and the no-both arm deferred: roughly 3 GPU hours and the same
trainer. That is the version to build first, and the rest of the table follows only if A and B
separate.

## Design table

Navigation: ⬅️ [The claims](#the-claims) | 📋 [TOC](#table-of-contents) | [Next](#what-the-words-are) ➡️

The table the `--request` walk trains from. A cell reading `tbd` is filled by the claim named in
it.

| Arm | Cat-only photos | Dog-only photos | Both-animal photos, captions per draw | Empty-caption rate, all photos | Steps | Seeds rendered |
|---|---|---|---|---|---|---|
| base SDXL | none | none | none | none | 0 | 48 |
| A | 300, "a photo of a cat" | 300, "a photo of a dog" | 200: joint 90%, empty 10% | 10% | 2,000 | 128 |
| B | same 300 | same 300 | same 200: cat 25%, dog 25%, joint 40%, empty 10% | 10% | 2,000 | 128 |
| C (droppable, see claim 4) | same 300 | same 300 | same 200, one caption fixed per photo for the whole run: 50 cat, 50 dog, 80 joint, 20 empty. Matches B's draw counts, not B's photo variety | 10% | 2,000 | 48 |
| no-both | same 300 | same 300 | none; 200 photos holding neither animal take their place, so every arm trains on 800 | 10% | 2,000 | 48 |

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

**The instrument.** Answered for one animal in a frame: zero-shot CLIP on the cropped box calls
cat on 64 of 64, margins 0.982 to 0.995, and
[the script](../../../scripts/census/crop_classifier_cat_vs_dog.py) is in the repo. Still open for
two animals in a frame, where the boxes overlap and the earlier detector reads failed. Claim 2's
three-category score cannot be trusted until a both-animal frame has been put through it, and the
first real pair renders the pilot produces are the natural place to check.

**Claim 1, the real dog-present share.** Not started. Needs COCO's annotations on the cluster and
the pilot's filtering rule, which claim 6 has to fix first.

**Claim 1, the held-out denoising loss.** Blocked on the missing image-denoising LoRA trainer.

## Runs

Navigation: ⬅️ [Checks outstanding](#checks-outstanding) | 📋 [TOC](#table-of-contents) | [Next](#routes) ➡️

| # | What | Where | State |
|---|---|---|---|
| 1 | base SDXL "a photo of a cat" on 64 seeds, per-class dog and cat instance counts, the floor for claim 1 | mscluster106 device 1, outside Slurm | done 2026-11-03, 1,316s. By eye 0 of 64 renders hold a dog, which is the clean floor. By detector 64 of 64 report one dog, which kills the per-class read. Card and sheet at `artifacts/results/does-the-cat-prompt-already-draw-dogs/` (gitignored, on disk only) |
| 2 | the same 64 renders, cropped to the detector's animal box and classified cat against dog by zero-shot CLIP ViT-B/32, to test the instrument run 1 killed | mscluster106 device 1, outside Slurm | done 2026-11-03, 41s. 64 of 64 called cat, none uncertain, margin 0.982 to 0.995 against a bar of 0.50. The uncropped image gives the same calls, so cropping is unproven rather than useless; the two-animal case is untested |

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

The walk is closed and compiled to [the standing write-up](COMPILED.md). Claim 2 goes to the
route file's step 2, the falsification criterion, because the comparison that step names is the
product's compose rate for B against A, which is claim 2's check. Claim 1's two reads run beside
it as the mechanism question and do not set the bar.
