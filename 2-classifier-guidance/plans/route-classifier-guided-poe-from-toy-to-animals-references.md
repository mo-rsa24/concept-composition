# References: can a classifier's push stand in for the LoRA

The background for [the route](route-classifier-guided-poe-from-toy-to-animals.md): what was found
on disk, the plan weight, who the route is for, the rationale and the sources. The decisions and
who made them are in
[the route walk](../artifacts/drips/route-classifier-guided-poe-from-toy-to-animals/ROUTE_WALK.md).

## What was found on disk

| Path | What it is | Bearing on the route |
|---|---|---|
| `1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis/15-can-a-classifier-supply-the-missing-term.md` | plan 15, a two-output presence head | tasks 1.1 to 3.1 ticked; the probe, the launch and the sweep open; never trained |
| `poe_repair/methods/_classifier_guided.py:112` | plan 15's sampler | rescales the push to the product prediction's own norm |
| branch `exec/can-a-classifier-supply-the-missing-term` | plans 15, 17 and 18 with their results | unmerged; its worktree sits in a stale scratchpad marked prunable |
| plan 17 on that branch | classifier guidance following Algorithm 2, nothing rescaled | the sampler every rung reuses |
| plan 18 on that branch | dog-01, a dog classifier trained through LoRA adapters in SDXL's down and middle blocks | accuracy check failed 7 of 20; product pushed to "dog present" gave 0 of 16 two-animal tiles |
| `2-classifier-guidance/artifacts/drips/can-a-classifier-supply-the-missing-term/EXECUTE_WALK.md` | plan 15's walk map | stale, says piece 1 in flight |
| `3-understanding-compositionality/artifacts/results/the-exact-toy/how-much-co-occurrence-the-product-needs/README.md` | the toy card | closed-form product, the 0.01 row, summed scores against the true noised product |
| `3-understanding-compositionality/scripts/census/crop_classifier_cat_vs_dog.py` | CLIP on the detector's padded crop | the species read for the independent evaluator |
| `5-text-embedding-third-expert/scripts/one_animal_axis.py` | "a cat" against "two cats" with a random-direction floor | the minimal-pair pattern D1 copies |
| `1-lora-correction/report/when-does-the-outcome-lock-in/where-does-each-condition-land.md` | the both-ness axis in DINOv2 space | failed as a cross-pair tool, so D4 carries no bar |
| `report/00-INDEX.md` lines 155 and 156 | Feynman-Kac steering and best-of-16 | the reranking baseline is weak on the plain product, strong with the adapter |
| `3-understanding-compositionality/artifacts/notes/coind-conditional-independence-loss/README.md` | CoInD on colored MNIST | the literature contact for the MNIST rung (binding there, co-occurrence here) |
| the ChatGPT session "Classifier Guidance vs CFG", read whole in Chrome on 2026-11-12 | the design | the discriminator's log-odds as the density-ratio score; the four heads; the controller's cap, schedule, agreement and per-head stop; the nine-row baseline progression; the three research questions |

## The plan weight

This repo is the scope, and its master plan holds the ladder, with plan files written one rung
ahead. The work is fifteen rungs over many weeks and the paper depends on it, which calls for a
full tree. A research ladder re-plans at every rung's close, which calls for writing only the
next rung. The master plan carries the ladder and the bars; the plan files follow the climb.

## Who this route is for

The person who runs the experiments and writes the paper, at Opus on high effort, steering each
walk. The depth is "understand it well enough to rebuild it" (2× budget) for the method read and
the toy rung, and "gloss" for the repair and the planning steps.

## The rationale

**Two axes, climbed separately.** The setting (toy to SDXL to held-out pairs) and the
architecture (one discriminator, then heads, then controller parts) are separate axes. Holding the
architecture while the setting climbs, then holding the setting while parts are added, means each
rung's gain is attributable to one change.

**The MNIST and small-animal rungs.** They split the step from the caption toy to SDXL, and both
keep the dial for how many training pictures hold both things, which no pretrained model has.

**Diagnosis beside the sweep.** A classifier can compose for the wrong reason, for instance by
learning the joint prompt's style or a pasting seam. The minimal-pair read separates plurality
from those, and running it beside the sweep keeps the earlier choice not to gate on accuracy.

**Time estimates.** Every time in the route is an estimate: pieces times 30 minutes times 1.5 for
the walks, a learning multiplier of 2× on the read and the toy rung, and labelled guesses for the
one-pass skills. The windows are assumed at three 90-minute sittings a week.

## Later

- A non-animal pair rung after C2 (a far-transfer test), offered and not taken.
- Phase B at the 64-pixel animal rung instead of A4, offered and not taken.
- Retraining the discriminator on guided draws, named as its own rung after phase A.
- A `video-cadence` sweep for classifier guidance and density ratio, once a journey exists.

## Sources

- Kim et al., "Refining Generative Process with Discriminator Guidance in Score-based Diffusion
  Models", ICML 2023, arXiv 2211.17091 (id from memory; step 1 checks it).
- Dhariwal and Nichol, arXiv 2105.05233, section 4 and Algorithm 2 (reproduced by plan 17).
- CoInD, arXiv 2503.01145 (judged already, in the notes).
- The shelf's little book of generative AI foundations, p. 162, the discriminator.
