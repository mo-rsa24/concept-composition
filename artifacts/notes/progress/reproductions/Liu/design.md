# Reproducing Composable Diffusion (Liu et al.)

## What this asks

Composable Diffusion splits a prompt into parts and adds each part's guidance to one sampling step. The question is whether the official code reaches
the numbers printed for it on three benchmarks.

## Status

| Row | Benchmark | Published | State |
|---|---|---|---|
| `CoMat-T1-007` | T2I-CompBench Color, BLIP-VQA | 0.4063 | `not comparable` until board question 25 is answered |
| `CoInD-T6-004` | CelebA, male and smiling held out | 14.19% | runs with CoInD, in the [CoInD design](../CoInD/design.md) |
| `CoInD-T4-026` | Shapes3D, partial support | 51.56% | no released weights, so it needs a costed question with CoInD's Shapes3D row |

CMNIST, Shape and Texture are `not comparable` by the owner's answer to board question 13.

## T2I-CompBench Color

CoMat Table 1 quotes 0.4063, and T2I-CompBench++ Table VIII, p9, prints it as "Composable + SD v2". Everything the benchmark settles, the check on plain
SD2 (0.5065), the 10 images per prompt on seeds 42 to 51, the scorer, and the missing official model are the same as in the
[Attend-and-Excite design](../AttendExcite/design.md). Plain SD2 runs once and serves both rivals.

**What nothing settles.** The benchmark's authors did not release their Composable run, and the paper does not say how a prompt was split.

**The rule for splitting, fixed before any run.** Split at the first " and ", and keep both sides as written, so "a red book and a yellow vase" becomes
"a red book | a yellow vase". Of the 288 prompts the subject guard allows, 266 contain " and ". The other 22 render as one prompt through the same
pipeline, and their count goes in the row's note.

**Sampler.** `ComposableStableDiffusionPipeline` at `77af82b` with its defaults: DDIM, 50 steps, weight 7.5 per part.

**Cost.** 2,880 images, about 3 GPU-hours, unmeasured until a smoke test.

**Buys.** `CoMat-T1-007`, the Composable Diffusion number on the Color target.

**The test.** The row counts as `reproduced` when |ours − 0.4063| ≤ 1.96 × √(SE_ours² + SE_ours²), with the standard error from resampling prompts with
replacement 10,000 times. Its note says it is scored on 288 of 300 prompts and that it tests a rebuild of an unreleased run.
