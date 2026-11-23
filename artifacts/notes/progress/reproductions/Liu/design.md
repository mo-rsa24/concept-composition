# Reproducing Composable Diffusion (Liu et al.)

## What this asks

Composable Diffusion splits a prompt into parts and adds each part's guidance to one sampling step. The question is whether the official code reaches
the numbers printed for it on three benchmarks.

## Status

| Row | Benchmark | Published | State |
|---|---|---|---|
| `CoMat-T1-007` | T2I-CompBench Color, BLIP-VQA | 0.4063 | `not comparable` by the owner's answer to board question 25 |
| `CoInD-T6-004` | CelebA, male and smiling held out | 14.19% | runs with CoInD, in the [CoInD design](../CoInD/design.md) |
| `CoInD-T4-026` | Shapes3D, partial support | 51.56% | no released weights, so it needs a costed question with CoInD's Shapes3D row |

CMNIST, Shape and Texture are `not comparable` by the owner's answer to board question 13.

## T2I-CompBench Color

CoMat Table 1 quotes 0.4063, and T2I-CompBench++ Table VIII, p9, prints it as "Composable + SD v2". The benchmark's authors ran it on
Stable Diffusion 2 base with a rebuild they did not release, so it cannot change an SDXL target. Nothing is downloaded and no GPU time is
spent on it. The [Attend-and-Excite design](../AttendExcite/design.md) sets out the same facts.

## Cross-references

- The mention of **Composable Diffusion**, a paper of reading sitting 1, in [the route to two papers, under The steps](../../../../../plans/route-two-papers.md#the-steps).
