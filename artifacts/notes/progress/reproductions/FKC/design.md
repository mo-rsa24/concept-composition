# Reproducing Feynman-Kac Correctors on GenEval

## What this asks

Feynman-Kac Correctors (FKC) run several images side by side during sampling, weight each one, and resample them so the batch follows the guided
distribution more exactly. The question is whether FKC on SDXL gives the two-object GenEval score its paper prints. GenEval's two-object score is the
share of images in which a detector finds both named objects.

## Status

The row is waiting on its run. By the owner's answer to board question 23, FKC is tested on GenEval's released 553 prompts against the 0.78
printed for that set, using a port of the authors' sampler.

## The row

| Row | Method | Benchmark | Published | Source |
|---|---|---|---|---|
| `FKC-TA4-038` | SDXL + FKC at guidance 7.5 | GenEval, the released 553 prompts, two objects | 0.78 | FKC Table A4, p39 |

## Why the 1,000-prompt row cannot run

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
significant increase" from FKC on SDXL (appendix F.5). By the owner's answer to board question 24, the Paper 2 target moves to 0.79 on the 553 prompts,
and a win must also beat plain SDXL's printed 0.80 on the same prompts and seeds.

## The run

| Choice | Value | Reason |
|---|---|---|
| Published row | `FKC-TA4-038`, SDXL + FKC at guidance 7.5 on the original 553 prompts, two objects, 0.78 | the only printed FKC number whose prompts are public |
| Plain SDXL check | `FKC-TA4-024`, 0.80, the authors' own run on the same prompts | checks our sampler and scorer |
| Sampler | a port of the weight and systematic resampling in `applications/images/edm2/generate_images.py` lines 340 to 356, clamped to ±2 as there | the authors' only image implementation |
| Particles | 4, GenEval's images per prompt | the paper states none for SDXL |
| Scorer | GenEval's Mask2Former evaluation at commit `af4902f` | the benchmark's own |

**The scorer's environment.** `/datasets/mmolefe/poe_repair_min/loop-5/envs/geneval` holds the authors' pins: Python 3.9, torch 1.12.1 for CUDA 11.3,
mmcv-full 1.7.1, open-clip 2.20.0, and mmdet 2.28.2 run from its checkout at `e9cae2d0`. Three changes were needed to make it load. `fsspec` is unpinned,
because the authors' file pins a version older than their `huggingface_hub` requires. The executable-stack flag is cleared on torch's libraries,
because this system refuses to load them otherwise. mmdet runs from `PYTHONPATH`, because its setup script no longer installs in editable mode.

**The detector's weights.** The Mask2Former checkpoint matches the SHA-256 prefix in its file name (`743b7d99`). It is a pickle file, so it was opened with
`weights_only=True` and re-saved as 649 plain tensors and the 80 class names, and only that clean copy is loaded.

**Cost.** 99 two-object prompts at 4 images is 396 images per method, under 2 GPU-hours with the detector.

**The test.** GenEval's score is per image, so the standard error comes from resampling prompts with replacement 10,000 times. The row counts as
`reproduced` when |ours − published| ≤ 1.96 × √(SE_ours² + SE_published²). Because the sampler is our port, its note says the result tests the port as
much as the method.
