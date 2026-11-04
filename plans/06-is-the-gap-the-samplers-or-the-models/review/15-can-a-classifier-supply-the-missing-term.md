# 🧪 Review: does a presence classifier's gradient make the product draw two animals?

The verdict file for
[hypothesis-15](../plans/hypothesis/15-can-a-classifier-supply-the-missing-term.md). The design is
there; what the runs cost, what came back and what it licenses are here.

## Recommended prompt (when the run lands)

```
/analyze-run @plans/06-is-the-gap-the-samplers-or-the-models/review/15-can-a-classifier-supply-the-missing-term.md — <the best weight's count out of eight, and the damped product's>
```

## Position in the plan tree

**Step 78 of 78.** Pairs with
[hypothesis-15](../plans/hypothesis/15-can-a-classifier-supply-the-missing-term.md). The one order
is the `## Running order` table in the [repo root MASTER_PLAN.md](../../../MASTER_PLAN.md).

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
- [Cross-references](#cross-references)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

- **The product**: the empty prompt's noise prediction plus the guidance scale times the sum, over
  "a photo of a cat" and "a photo of a dog", of each prompt's difference from the empty
  prediction. Guidance 7.5, 50 DDIM steps.
- **The damped product**: the same sum with each expert's own guidance held at a constant 3.0. It
  already puts both animals in frame on 8 of 8 cells and ruins the picture doing it, so it is the
  column the new method is read against, not plain product.
- **The joint prompt**: "a photo of a cat and a dog" put through the model the ordinary way, with
  no product at all. It composes on 8 of 8 held-out seeds and is the ceiling.
- **The presence head**: a conv net on the noisy latent and its timestep with two independent
  sigmoid outputs, "a cat instance is present" and "a dog instance is present". SDXL is frozen and
  only the head trains.
- **The classifier weight**: how much of the head's gradient enters each step, in units of the
  product prediction's own norm, because the gradient is normalised per step. A weight of 1 means
  a step as large as the prediction it is added to.
- **Two clear animals**: a cat that reads as a cat and a dog that reads as a dog, both whole, in
  one picture, judged at full resolution.
- **One hybrid animal**: a single animal carrying features of both. Its own category, not a failed
  version of either other one, because two renders already read as a clear pair at sheet scale and
  as hybrids at full resolution.
- **One single animal**: one animal of one species, with no feature of the other.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim.** Between the four classifier weights exactly one thing differs, so that axis
is clean. Against plain product two things differ (the head's gradient is present, and the weight
is nonzero), which is the comparison the claim is about and is still one mechanism. Against the
damped product the comparison is mixed: a learned gradient against a changed guidance scale. It is
kept anyway because it is the only existing fix that already puts two animals in frame, and it is
read as "does the new method beat the crude one", never as cause and effect.

What would make the run worthless is a composer that does not reproduce plain product at weight 0,
since then every column would measure the plumbing. The zero-weight canary at task 2.3 is the
gate and it runs before any weight is swept.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| The labelled set and its latent cache | setup | not yet | one VAE pass over the chosen photos and the mined hybrids | `/datasets/mmolefe/poe_repair_min/outputs/presence_head/labelled_set/` | ⚠️ not started |
| The zero-weight canary, the toy seed through the new composer | gates the claim | not yet | 1 render | the checksum against plain product's render of the same seed | ⚠️ not started |
| The head, trained with a probe every 250 steps | Tests the claim | not yet | about 2 GPU-hours on one Quadro RTX 8000; roughly 30 probes at 4 renders each | checkpoints and probe sheets under `/datasets/mmolefe/poe_repair_min/outputs/presence_head/`, time series in W&B `prime_lab/poe-repair-animals-compose` | ⚠️ not started |
| The weight sweep with its three reference columns | Tests the claim | not yet | 56 renders at 1024², 50 steps each, about 50 minutes, one loop one device | `artifacts/results/can-a-classifier-supply-the-missing-term/` | ⚠️ not started |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] **Does adding the presence head's gradient to plain product sampling make the cat-by-dog cell
      draw two separate animals?** Eight held-out seeds, 9 to 16, at guidance 7.5 and 50 DDIM
      steps, at four classifier weights, with plain product, the damped product and the joint
      prompt rendered in the same loop on the same device. Every call is made by eye from the
      full-resolution tile in the three categories named above, because the instance-count scorer
      counts animals without naming them.

      **Support** if at least 4 of 8 seeds hold two clear animals at some weight, and the damped
      product's count on the same seeds is no higher. The gradient composes, and it does so without
      the smearing the crude fix pays for.

      **No support** if 0 or 1 of 8 at every weight. The gradient did not move composition.

      **Inconclusive** at 2 or 3 of 8 at the best weight, or whenever the count beats plain product
      but not the damped product. Eight seeds cannot resolve a move that small.

      **The fourth outcome, which is not a null and not a support**: the count rises while the
      pictures are destroyed. The Langevin corrector's norm guard reported healthy numbers through
      exactly that, so this line exists to keep it from being read as either of the other three.

      Plain product's 0 of 8 and the joint prompt's 8 of 8 are not quoted from an earlier run: both
      render in the same loop as the sweep, because a float32 against float16 comparison of the
      same combine gives about three grey levels over 50 steps.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] **Does the composer reproduce plain product at weight 0?** The checksum of the zero-weight
      render against plain product's render of the same seed. Identical is the only pass. Anything
      else is a plumbing fault and the sweep does not run until it is fixed.

- [ ] **Does composition emerge as the head trains, or arrive all at once?** The compose count on
      the probe's two seeds against the training step, one series per probe weight. Two seeds
      cannot carry a rate, so this is read as a shape, and the step it first moves at is what
      picks the checkpoint for the sweep.

- [ ] **What does the head actually learn to see?** Balanced accuracy per label on held-out images
      at every sampling step from 0 to 49. A curve that sits at chance past step 10 means the head
      cannot read a noisy latent, and the gradient it supplies at those steps is noise.

- [ ] **How many hybrid negatives were used, and from which cells?** The count and the pair list.
      These come from the model being corrected, which is a circularity: the head is taught to
      dislike exactly what this model does, so a gain may not transfer to another model's failure
      mode.

## Asked after the result

Navigation: ⬅️ [Written before the run, answered after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

- [ ] Which weight was best, and is the best weight at an edge of the swept range? An optimum at
      the edge means the range was wrong rather than the method optimal.
- [ ] Did the gradient's size need the per-step normalisation, or was it already in range?

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

**The negatives come from this model.** The hybrids the head is trained to call "neither" are this
sampler's own product renders. A head that learns to recognise this particular failure is not the
same as a head that knows what two animals look like, and nothing here separates the two.

**Nothing has tested whether the missing term is a gradient.** A classifier can only ever supply a
term that is the gradient of some scalar field. `scripts/is_the_correction_a_gradient.py` does not
exist in this repo, so a no-support verdict cannot distinguish a wrong instrument from an
impossible one. The closing report says so rather than reading the null as the stronger claim.

**The species instrument has never been tested on overlapping boxes.** It was measured on 64
single-cat renders, where it scored 64 of 64 at median margin 0.990. A frame with a cat and a dog
whose boxes overlap is a case it has never seen, which is why the calls here are made by eye and
the labeller's uncertain count is reported.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- The weight-sweep sheet, with the three reference columns, as the figure.
- The labelling rule, quoted from the source constants, because it is where the meaning of "both"
  is defined and nothing in the captions carries it.
- The hybrid-negative count and its circularity, stated and not buried.
- The probe time series, as the thing that shows whether composition emerged or appeared.

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

- **Whether the correction the product drops is the gradient of any scalar.** Untested here, and it
  bounds what this plan's null could mean.
- **Whether a head trained on this model's hybrids transfers to another model's.** Out of scope and
  worth naming before the result is generalised.
- **Whether the four proposed outcome lines are the right ones.** They are proposed by the walk,
  not set by the user, and instruction 8.2 is where they are ticked or rewritten.

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents) | [Next](#cross-references) ➡️

Read the four outcome lines above and either tick the box on the question or rewrite the bars. Then
[task 1.1](../plans/hypothesis/15-can-a-classifier-supply-the-missing-term.md#1--the-labelled-set-and-the-rule-in-the-source).

## Cross-references

Navigation: ⬅️ [Next step](#next-step) | 📋 [TOC](#table-of-contents)

- [the plan this answers](../plans/hypothesis/15-can-a-classifier-supply-the-missing-term.md)
- [why a corrector alone produced nothing](../../../report/is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md)
- [what the crude damping costs](../../../report/is-the-gap-the-samplers-or-the-models/does-any-render-time-fix-recover-picture-quality.md)
- [why steering on the scorer found nothing](../../../report/is-the-gap-the-samplers-or-the-models/does-steering-on-the-scorer-find-a-composing-proposal.md)
- [the species instrument](../../../scripts/census/crop_classifier_cat_vs_dog.py)
- [the compose rate's definition](../../../context/world/compose-rate.md)
