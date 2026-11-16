# References: can a classifier's push fix the product

Navigation: [the route](route-classifier-guided-poe-from-toy-to-animals.md)

The background for the route: what was found on disk, the plan weight, who the route is for, the
rationale and the sources. The earlier decisions and who made them are in
[the route walk](../artifacts/drips/route-classifier-guided-poe-from-toy-to-animals/ROUTE_WALK.md).

## Found on disk

| Path | What it is | Bearing on the route |
|---|---|---|
| `2-classifier-guidance/plans/15-can-a-classifier-supply-the-missing-term.md` | plan 15, a two-output presence head with pair-aware labels | tasks 1.1 to 3.1 ticked; the probe, the launch and the sweep open; step 10 resumes it |
| `2-classifier-guidance/artifacts/drips/can-a-classifier-supply-the-missing-term/EXECUTE_WALK.md` | plan 15's walk map | says piece 1 in flight; the run finished on 2026-11-04 |
| `2-classifier-guidance/plans/17-classifier-guidance-from-the-paper.md` | classifier guidance on SDXL in the paper's form | tasks 1 to 3 done; 4.1 becomes a control column in step 10 |
| `2-classifier-guidance/plans/18-a-dog-classifier-trained-the-papers-way.md` | dog-01, a dog classifier trained through LoRA adapters | accuracy check failed; 0 of 16 two-animal tiles; 4.1 becomes a control column |
| `poe_repair/methods/_classifier_guided.py:112` | plan 15's product sampler | still rescales the push; step 5 repairs it |
| `2-classifier-guidance/artifacts/drips/read-refining-generative-process-discriminator/READ_MAP.md` | the Kim et al. lean read | open at the abstract; step 3 resumes it |
| `3-understanding-compositionality/artifacts/results/the-exact-toy/how-much-co-occurrence-the-product-needs/README.md` | the toy card | closed-form product, the 0.01 row, summed scores against the true noised product |
| `4-inclusive-captions/artifacts/results/how-many-photos-the-inclusive-captions-pilot-can-train-on/README.md` | the photo pool | 550 photographs, 200 cat-only, 200 dog-only, 150 both; step 8's data |
| `3-understanding-compositionality/scripts/census/crop_classifier_cat_vs_dog.py` | CLIP on the detector's padded crop | the species read for the independent evaluator |
| `environment/reduce-reuse-recycle-code.md` line 47 | a broken copy of the guided-diffusion U-Net | step 4 uses the upstream repository |
| the HLD registry entry "FKC on our PoE sampler", https://claude.ai/artifact/RZynLjqAMebuiV3nTFC5CN | where each method touches the product's prediction | step 1 adds the classifier push to it |
| `report/00-INDEX.md` lines 155 and 156 | Feynman-Kac steering and best-of-16 | the reranking baseline for the later stages |
| `/home/molef/PhD/replications/` | six reproductions, none of classifier guidance | step 4 adds `guided-diffusion` here |

Two Codex jobs were waiting for their check at routing time (20261013-082911-405afd and
20261116-155418-b16461). Neither belongs to this route.

## Plan weight

Short, because each stage is one walk that writes its own plan at its open, and the research spine
re-plans at every stage's close. Plans 15, 17 and 18 stay as they are; step 10 walks plan 15.

## Who you are for this route

Own the maths of classifier guidance by hand (a submerge journey, started in step 2). Build tier on
the exact toy, where you type the guided step and the discriminator. Gloss for the sampler repair
and the SDXL stage. Opus on high for the steps that derive or design (2, 3, 7, 9, 11); Sonnet on
medium for the runs (1, 4, 5, 6, 8, 10). Sittings of two hours, no dates yet.

## Later

- The held-out seeds 9 to 16 against best-of-N at equal compute, re-routed at step 11.
- The head ablations (presence, instance count, spatial mask, hybrid and quality; the weak,
  strong, fading schedule; gradient agreement; per-head early stopping), re-routed at step 11.
- The best configuration over the v58 pool and on held-out pairs, beside the rank-32 LoRA.
- The paper's comparison table and its paragraph.
- MNIST digits two to a canvas, and the toy's trained caption model: dropped from this climb,
  because the photo stage covers the step from the toy to real images.
- A product of two ImageNet classes on the released class-conditional model, which would need an
  unconditional prediction it does not have.
- FID and the paper's tables for step 4, at 50,000 samples per number.
- A `video-cadence` sweep for classifier guidance and density ratio, once step 2's journey exists.
- Retraining the pair-aware classifier on guided draws.

## Why this chain

**The order.** Laid on the research spine: read (step 3), reproduce (step 4), rebuild small (steps 7
and 8), state the test (step 9), test it at scale (step 10), explain (step 11). The released-weights
reproduction comes first because every later stage ports its code, and a push that fails on the
authors' own weights would make every later miss uninterpretable.

**The separate-label control.** By Bayes, a classifier trained on cat and dog separately supplies
the same gradient the product already carries, so it can only act like a heavier product. Running
it beside the pair-aware classifier at every stage turns that prediction into a check, and plan
18's 0 of 16 is the first data point.

**Both arms in step 8.** The released model asks whether the push works on a model trained on data
we do not control. The model we train asks it where we control the data, including the case where
the model never saw a both-animal photo while the classifier did. The cluster can train a 64 px
model, so both are affordable.

**Times.** Every time is an estimate unless a run count is given: pieces times 30 minutes times
1.5 for the walks, then 2× for learning on the teaching walks (1.5× on the sampler repair). The
measured medians are from `~/.claude/skills/skill-route/references/measured-times.md`: submerge 9.3
min over 4 runs, prompt-storyboard 16 min over 5, render-diagrams 11 min over 17, picture-speak 15
min over 9; the lean read and the reproduce mode have fewer than 3 runs each. Completion to the SDXL
stage is about 20 hours; the budget about 38 hours, 19 sittings.

**Parallel and sequential.** Steps 1, 2 and 3 run in sitting 1, with step 4's downloads and step 2's
journey build in the background. Steps 4 and 5 run side by side in sittings 3 and 4, since neither
reads the other's output. Step 6 runs in the background. From step 7 on, each step reads the one
before it, so they run one after another, with the 64 px trainings in the background of step 8.

## References

- Dhariwal and Nichol, "Diffusion Models Beat GANs on Image Synthesis", NeurIPS 2021, arXiv
  2105.05233; code and checkpoints at github.com/openai/guided-diffusion.
- Kim et al., "Refining Generative Process with Discriminator Guidance in Score-based Diffusion
  Models", ICML 2023, arXiv 2211.17091.
- The shelf's little book of generative AI foundations, p. 162, the discriminator.
- The product-of-experts rule from Bayes:
  [the sampling-from-a-graph plan](/home/molef/goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/plans/17-poe-rule-from-the-fork.md),
  with its textbook chapter and the video segment "Graphical Building Blocks; Chains and Forks",
  18:34 to 26:42.
- Score additivity and the product density:
  [the speciation-dynamics plan](/home/molef/goal-setting/learning/deep-learning/diffusion-models/06-diffusion-speciation-dynamics/plans/05-composition-and-the-experiment/plans/01-score-additivity-product-density.md).
- CoInD, arXiv 2503.01145, judged earlier in the notes.
