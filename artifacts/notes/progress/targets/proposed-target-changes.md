# Proposed changes to the accepted targets

The owner accepted 10 targets on 2026-11-22. Reading the field table afterwards turned up printed numbers above three of
them and a benchmark that fits Paper 1 better. This page sets out what would change. The accepted targets stay on
[the scoreboard](../scoreboard.json) until the owner says yes on the board.

## What would change

| Paper | Benchmark | Accepted target | Proposed target | Number it rests on |
|---|---|---|---|---|
| 1 | CLEVR, 5 objects | 45.5% (Domain Expansion, 44.5) | **99%** | Projective Composition, 98 of 100 hand-counted images, Table 1, p17 |
| 2 | T2I-CompBench Complex, ImageReward | 0.4506 (CO3, 0.4406) | **0.4761** | Rectify Then Diffuse, 0.4661, Table 2, p6 |
| 2 | RareBench Concat, ImageReward | 0.1710 (CO3, 0.1610) | **0.4564** | Rectify Then Diffuse, 0.4464, Table 2, p6 |
| 2 | GenEval two objects, 1,000 prompts | none | **0.81**, added as a sixth target | FKC, 0.80, Table A4, p39 |

Shapes3D, Colored MNIST, CelebA, the CIFAR-10 split and T2I-CompBench Color, Shape and Texture keep their accepted targets.

## Why each one

**CLEVR.** Domain Expansion trains one base model on all 30,000 multi-object images (its p11), so it does not work in Paper
1's setting. Projective Composition composes models that saw only single-object images. Its appendix describes a model
conditioned on coordinates (p15), so the composed parts may be one network. It is still the closest printed setting.

**Complex and RareBench Concat.** Rectify Then Diffuse states the same SDXL setup as CO3 and prints higher scores on both.
Its baseline ImageReward cells match CO3's table on 19 of 20 values, so CO3 was probably not rerun.

**GenEval two objects.** None of the 10 accepted targets counts two separate objects, which is what Paper 2 claims to
restore. On this task FKC prints 0.80 and the same authors' plain SDXL prints 0.79, so the target sits one step above both.

## The scope this rests on

**Paper 2 targets are on SDXL.** Newer base models score higher with no composition method at all. Plain SD3 prints 0.8132
on Color and 0.7334 on Texture in T2I-CompBench++ (Table XIII, p12). Those belong in the paper as context.

**Methods that need an outside language model are reported beside the targets and do not set them.** RPG on SDXL prints
Color 0.8335, Shape 0.6801 and Texture 0.8129 (Table 1, p9). It needs GPT-4, hand-written examples and planned regions, which
Paper 2's stated setting of one model, the joint prompt and a checker leaves out. It names no scorer, prompt count or
images per prompt. If RPG counted, those three targets would be 0.8435, 0.6901 and 0.8229.

## What no target can cover

No published row combines models trained apart with several separate objects, so Paper 1's central claim has no rival
number. It needs its own table, with a control that samples each model alone and pastes the two results side by side.
