# Reproducing Feynman-Kac Correctors on GenEval

## What this asks

Feynman-Kac Correctors (FKC) run several images side by side during sampling, weight each one, and resample them so the batch follows the guided
distribution more exactly. The question is whether FKC on SDXL gives the two-object GenEval score its paper prints. GenEval's two-object score is the
share of images in which a detector finds both named objects.

## Status

The row is `blocked`, and nothing has run. Board question 23 asks the owner whether to run the closest version or leave it blocked.

## The row

| Row | Method | Benchmark | Published | Source |
|---|---|---|---|---|
| `FKC-TA4-059` | SDXL + FKC at guidance 7.5 | GenEval, 1,000 prompts, two objects | 0.80 | FKC Table A4, p39 |

## Why it cannot run as printed

| Needed | What exists |
|---|---|
| SDXL sampler | the authors' repository has an EDM2 image sampler and no SDXL code |
| The 1,000 prompts | GenEval's released metadata holds 553 prompts, of which 99 are two-object. Its generator takes a seed and a count per task, and the paper states neither for its 1,000 |
| Particle count | the paper gives batch sizes for EDM2 (32 and 8) and none for SDXL |
| Sampler, steps, size | not stated for SDXL |

The hub's `scripts/methods/feynman-kac-correctors.py` ports FKC to our product of experts on cat × dog. It is a different target from classifier-free
guidance on GenEval, so it does not stand in for the authors' run.

## What the printed table says

Table A4 prints plain SDXL ("our run") at 0.79 on the same 1,000 prompts and SDXL + FKC at 0.80. The authors write that they "do not observe any
significant increase" from FKC on SDXL (appendix F.5). The Paper 2 target of 0.81 on this row therefore sits one step above a number that is about
equal to plain SDXL.

## The closest version, if the owner approves it

| Choice | Value | Reason |
|---|---|---|
| Published row | `FKC-TA4-038`, SDXL + FKC at guidance 7.5 on the original 553 prompts, two objects, 0.78 | the only printed FKC number whose prompts are public |
| Plain SDXL check | `FKC-TA4-024`, 0.80, the authors' own run on the same prompts | checks our sampler and scorer |
| Sampler | a port of the weight and systematic resampling in `applications/images/edm2/generate_images.py` lines 340 to 356, clamped to ±2 as there | the authors' only image implementation |
| Particles | 4, GenEval's images per prompt | the paper states none for SDXL |
| Scorer | GenEval's Mask2Former evaluation at commit `af4902f` | the benchmark's own |

**Cost.** 99 two-object prompts at 4 images is 396 images per method, under 2 GPU-hours with the detector.

**The test.** GenEval's score is per image, so the standard error comes from resampling prompts with replacement 10,000 times. The row counts as
`reproduced` when |ours − published| ≤ 1.96 × √(SE_ours² + SE_published²). Because the sampler is our port, its note says the result tests the port as
much as the method.
