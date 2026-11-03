# Inclusive captions for the cat and dog experts: the design, checked

What the idea is, which parts of it stand, what has to be built before any of it can run, and
what the pilot would have to show to be worth a paragraph in the paper. The claim-by-claim
reasoning and the rounds that produced it are in [the idea map](IDEA_MAP.md); this file is the
standing version.

## The idea in one paragraph

A condition in a text-to-image model learns the distribution of the images trained under it. When
every photo holding both a cat and a dog is captioned only "a photo of a cat and a dog", the cat
condition sees cats without dogs, and fine-tuning pulls it toward cat-only. If each both-animal
photo is instead sometimes trained under "a photo of a cat" and sometimes under "a photo of a
dog", both conditions keep dog-present and cat-present images inside them. The product of the two
conditions is high where both are high, so it should hold more pictures containing both animals.

## What the five training runs are

Every run trains on the same 731 photos: 300 cat-only, 300 dog-only, 131 holding both. COCO train2017 holds 208 both-animal photos and 131 of them survive the 0.25 box-fraction floor the compose scorer uses; the single-animal pools are ten times what is needed (3,335 and 2,708). Counts and the audit sheets are at `artifacts/results/the-inclusive-captions-photo-pool/`. The only
thing that changes is the caption each both-animal photo is trained under. Every run shares one
10% blank-caption rate on every photo, which is what makes the comparison one-variable.

| Run | Cat-only | Dog-only | Both-animal, captions per draw | Blank rate | Steps | Seeds scored |
|---|---|---|---|---|---|---|
| base SDXL | none | none | none | none | 0 | 48 |
| A | 300, "a photo of a cat" | 300, "a photo of a dog" | 131: joint 90%, blank 10% | 10% | 2,000 | 128 |
| B | same 300 | same 300 | same 131: cat 40%, dog 40%, joint 10%, blank 10% | 10% | 2,000 | 128 |
| C (droppable) | same 300 | same 300 | same 131, one caption fixed per photo for the whole run, at B's shares | 10% | 2,000 | 48 |
| no-both | same 300 | same 300 | none; 131 photos holding neither animal take their place | 10% | 2,000 | 48 |

B against A is the experiment. Everything else is a control or an anchor.

**The share, and why the single-name rates are 40% rather than the source's 25%.** The effect
size is the fraction of B's "a photo of a cat" training draws that hold a dog. At 131 both-animal
photos and a 25% cat share that fraction is 11%, below the 15% the design assumed at 200 photos
and no larger than the move 128 seeds can resolve. At 40% it is 15% again. The change costs
nothing, because the blank rate stays at 10% in every arm and that is the only cell the
one-variable comparison depends on; it only moves how often a both-animal photo is called by one
animal's name rather than by both.

## Where each claim landed

| Claim | Mark | What settles or repairs it |
|---|---|---|
| 1. Both-animal photos under each single name widen the cat condition | needs a check | the dog-present render count under A against B, scored by the CLIP crop call, and the held-out denoising loss on 8 unseen both-animal photos |
| 2. The wider condition makes the product compose more often | needs a check | the product's compose rate under A against B on 128 shared seeds, scored in three categories: pair, hybrid, single animal |
| 3. A against B changes one variable | wrong as stated, repaired | the source gave only B a blank caption; one shared 10% rate across every run fixes it, and the repair survived the argument that the product's double subtraction of the unconditional amplifies any residual drift |
| 4. C is a distinct third run | wrong as stated, no clean repair | matching B's draw counts leaves the cat condition seeing 50 photos against B's 200; drop C and buy a second training seed for A and B instead |
| 5. The no-both run is the share-zero point of the planned share sweep | wrong as stated, repaired | 131 neither-animal photos replace the removed ones, so every run trains on 731 and every cat photo is drawn equally often. It is still one anchor, not the sweep |
| 6. One day of GPU | wrong as stated | about ten GPU hours, after two or three days building the trainer, the COCO data build and the caption machinery |

## What does not exist yet, and is the real cost

**The trainer.** This repo's LoRA trainer learns from cached noise tensors toward a correction
target, with no photographs, no captions and no text encoder in the loop. The pilot needs an
ordinary image-and-caption denoising trainer. diffusers 0.29.2, peft 0.19.1 and accelerate 1.13.0
are installed in `co3` and no copy of the diffusers SDXL LoRA example is on disk, so the example
is fetched and adapted.

**The data build.** Selecting the cat-only, dog-only, both-animal and neither-animal sets from
COCO's annotations, auditing the both-animal photos by eye because annotations miss animals, and
recording each photo's assignment so run C is reproducible.

**The caption machinery.** The per-draw die roll for B, the decided-once assignment for C, and
the shared 10% blank rate.

What does exist is the scoring end: GroundingDINO's instance count for how many animals a render
holds, and the CLIP crop call for which animal each one is.

## What the runs show while they train

Weights and Biases, project `prime_lab/poe-repair-inclusive-captions`, one run per arm, grouped
so the arms share an axis. Loss, learning rate and gradient norm every step.

At steps 0, 250, 500, 1,000 and 2,000 each run renders a sheet: eight fixed seeds down the rows,
four columns across, "a photo of a dog", "a photo of a cat", "a photo of a cat and a dog", and
the product of the cat and dog conditions. The same eight seeds at every checkpoint and in every
run, so the sheets stack into a film and any two runs compare tile for tile.

The product column costs nothing to train, because combining two conditions happens at sampling
time: three predictions a step against two for an ordinary render. It is the column the idea
lives or dies on, and having it from step 0 shows whether the product moves while the
single-prompt columns stay put.

The sheets render at the training resolution, 512 square at 25 steps, about four seconds a tile,
so every checkpoint of every run costs under an hour in total. The scored renders stay at 1,024
square at 50 steps, because composition calls are made on those and this project has been caught
calling clear pairs off downscaled sheets.

**Eight seeds is a picture, not a rate.** The sheets say whether anything is happening and when.
The compose rate that answers claims 1 and 2 comes from the 128-seed renders and from nowhere
else.

## The cost

| Piece | Cost |
|---|---|
| training, 4 runs plus a second seed for A and B | about 6 hours |
| the watching sheets | about 1 hour |
| rendering, 128 seeds each for A and B, 48 for base, C and no-both | about 2.5 hours |
| scoring about 500 renders | under 1 hour |
| **total GPU** | **about 10 hours** |
| **before any of it** | **two or three days building the trainer, the data build and the caption machinery** |

The cheapest version that still tests the idea is A and B alone at 128 seeds, with C dropped and
the no-both run deferred: about three GPU hours on the same trainer. The rest of the table is
worth buying only if A and B separate.

## What would make this worth a paragraph, and what would kill it

Worth it: B's product compose rate is above A's by more than the band the bar fixes, and the
held-out denoising loss on unseen both-animal photos is lower under B. The caption assignment
then has a measurable effect on what a single-animal condition covers, and composition follows it.

Dead: the rates match and the losses match. The captioning does not widen the condition at the
scale a rank-8 adapter on 2,000 steps can reach, and the idea moves to the from-scratch model,
which has no inherited support to erode.

The outcome to be careful with: all three rates near zero, with B's cats visibly worse than A's,
blurred or carrying a second indistinct animal shape. That is the condition absorbing the
dog-present draws as an average rather than as a second mode, and reading it as a plain null
would retire the idea for the wrong reason.

## Where this sits in the literature

Bradley et al. 2502.04549 state the condition under which adding two experts' scores is exact:
the two conditions are independent given the image. CoInD (2503.01145) attacks that condition
from the training side with a loss term that penalises the divergence between a model's joint
conditional and the product of its single-concept conditionals. This idea attacks the same
quantity from the data side, with captions instead of a loss, which is cheaper and weaker. Their
divergence is also a measurement, not only a loss, so computing it under A and under B is a third
read of claim 1 that neither the render count nor the held-out loss gives.

FactorDiff (2607.11758) routes the product's weighting per factor instead of one global scalar,
in discrete diffusion. It is an inference-time weighting scheme and this pilot changes training
data, so it bears on the discussion rather than on the design.

## What the pilot does not settle

Whether the product's failure is about co-occurrence at all, which is the question the planned
share sweep answers and this pilot only anchors. Whether the CLIP crop call survives two animals
sharing a frame with overlapping boxes, which is untested and which claim 2's three-category
score depends on. And whether any of the cross-attention repairs in the binding literature can
reach a method that adds two experts' scores after each forward pass, which is a separate walk
named in the map's Routes section.
