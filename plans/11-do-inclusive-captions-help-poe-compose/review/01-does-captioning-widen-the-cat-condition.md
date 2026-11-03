# 🔍 Review: does captioning both-animal photos by single name widen the cat condition?

The criterion for [the pilot](../plans/01-the-inclusive-captions-pilot.md), written before either
run trained. Nothing in the question section may change once a training step has run; the
sections after it are filled from the result.

## Position in the plan tree

[Root](../../../MASTER_PLAN.md) › [do inclusive captions help PoE compose](../MASTER_PLAN.md) ›
[the pilot](../plans/01-the-inclusive-captions-pilot.md) › this review.

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

**Run A.** A LoRA trained on 731 COCO photos, where every photo holding both a cat and a dog is
captioned "a photo of a cat and a dog".

**Run B.** The same LoRA on the same 731 photos, where a both-animal photo is captioned "a photo
of a cat" 40% of the time, "a photo of a dog" 40%, jointly 10%, redrawn every time the photo is
seen. Both runs leave the caption blank 10% of the time, on every photo in every set.

**The product.** `eps_uncond + g * (eps_a + eps_b - 2 * eps_uncond)` with the cat prompt as `a`,
the dog prompt as `b`, and guidance 7.5.

**Compose rate.** The fraction of a run's 128 product renders holding exactly one cat and one
dog, with neither animal a hybrid of the two. Counted by GroundingDINO, named by the CLIP crop
classifier, judged off 1,024-square tiles at full resolution.

**The band.** Two standard errors of the difference between the two measured compose rates:

    band = 2 * sqrt( pA*(1-pA)/128 + pB*(1-pB)/128 ) * 100   percentage points

where `pA` and `pB` are the rates the renders actually produce. The formula is fixed here and the
number is not, because the number depends on where the rates land: it is 12.5 points at rates
near 0.5, 7.5 at rates near 0.1, and 12.5 is the widest it can ever be. Fixing 12.5 outright
would be the most conservative choice and also the wrong one, since plain product composition on
this pair has come in near 0.1 before, and a band nearly twice as wide as the data warrants
pushes a real effect into the inconclusive region. What a fixed formula and a free number cannot
do is move after the answer is seen, which is the property that matters.

Any difference smaller than the band is a difference this experiment cannot see.

## Run kind

Experiment. One axis differs between run A and run B: the caption a both-animal photo is trained
under. Everything else is held, including the photographs themselves, the blank-caption rate, the
step count, the sampler, the guidance scale and the 128 scoring seeds.

## Runs

| Run | W&B id | Captions on the 131 both-animal photos | Steps | Scoring seeds |
|---|---|---|---|---|
| base SDXL | none | none, untrained | 0 | 48 |
| A | to be filled | joint 90%, blank 10% | 2,000 | 128 |
| B | to be filled | cat 40%, dog 40%, joint 10%, blank 10% | 2,000 | 128 |

## The question written before the run

**The comparison.** Run B's compose rate minus run A's compose rate, each measured on the same
128 seeds, same sampler, 50 DDIM steps, guidance 7.5, 1,024 square. Base SDXL's rate on 48 seeds
is the reference row and is not part of the comparison.

**What counts as support.** Run B's compose rate exceeds run A's by **the band or more**, the
band computed from the measured rates by the formula above. Secondary, reported apart and not
part of the bar: run B's held-out denoising loss on 8 unseen both-animal photos is lower than run
A's.

**What counts as no support.** Run B's compose rate is **at or below** run A's. Reported as no
support rather than as a null, because 128 seeds cannot tell a true zero from a true gain of 6
points that happened to measure below zero. The claim this outcome licenses is that the
captioning did not widen the condition by an amount this experiment could detect, and nothing
stronger.

**The inconclusive band.** Run B's rate is above run A's by **more than 0 and less than the
band**. The direction is right and the experiment cannot see it. This outcome is reported as
inconclusive with the cost of resolving it stated: quadrupling the seeds to 512 per run halves
the band, which is roughly 8 further GPU hours of rendering.

**The counts.** 128 product renders per run for A and B, on the same 128 seeds. 48 for base SDXL.
8 held-out both-animal photos for the secondary loss read, none of them in the 731.

**What would surprise.** Run B's joint-prompt compose rate moves while its product rate does not.
That would mean the captioning changed what the model can draw when asked for both animals
together, without changing what the product of the two single conditions reaches, which the
design has no account for.

**The fourth outcome, which is not a null.** All three compose rates below 0.10, with run B's cat
column judged worse than run A's by eye on 5 or more of the 8 sheet seeds: blurred, or carrying a
second indistinct animal shape. That is the cat condition absorbing its dog-present draws as an
average rather than holding two modes. It is reported as its own result, and the idea moves to
the from-scratch model rather than being retired.

## Written before the run, answered after

| Question | Written before | The answer |
|---|---|---|
| Does run B's product compose more often than run A's? | support at a gap of the band or more, no support at a gap of 0 or less, inconclusive between | to be filled |
| Is run B's held-out denoising loss on unseen both-animal photos lower? | reported apart from the bar | to be filled |
| How many of run B's renders did the classifier call uncertain? | unknown; it has never been read on two animals in one frame | to be filled |
| At which checkpoint can the two runs be told apart by eye? | unknown | to be filled |

## Asked after the result

To be filled once the figure exists.

## Could the answer be an artefact

Three ways it could be, each with what rules it out.

**The caption shares drifted from the design.** Ruled out before training by the caption
sampler's own check, which measures the realised shares off 200,000 draws against three bars held
as constants in its source. It already caught one real bug: run C's blank caption was counted
twice, putting its blank rate 1.6 points off A's and B's.

**The photos do not hold what COCO says they hold.** COCO labels drawings and statues as animals
and misses animals that are present. Ruled out by looking at all 131 both-animal photos by eye
and writing down the rejections, which is instruction 8 of the plan.

**The scorer cannot tell a cat from a dog in a frame holding both.** Not ruled out. The
GroundingDINO per-class read is known dead here: it called a dog in 64 of 64 single-cat renders
at a median margin of 0.054. The CLIP crop classifier replaces it at 64 of 64 correct with
margins above 0.98, but only on frames holding one animal. The uncertain count is reported with
the result, and a high one invalidates the three-category split rather than the compose rate
itself.

## What the write-up owes

The compose-rate figure with its sidecar json, the realised caption shares from the sampler's
check, the count of both-animal photos rejected in the audit with reasons, and the uncertain count
from the classifier. The numbers live in the figure and its sidecar, never restated in prose.

## Still open

- Whether the crop classifier survives two animals sharing a frame with overlapping boxes. This
  is the one untested piece of the instrument and the three-category split depends on it.
- Whether the result transfers past cat and dog. One pair is trained.
- Whether the product's failure is about co-occurrence at all, which is the share sweep's
  question. This review anchors one point of it.

## Next step

Instruction 7 of [the pilot](../plans/01-the-inclusive-captions-pilot.md): read this and approve
it, before any code is written.
