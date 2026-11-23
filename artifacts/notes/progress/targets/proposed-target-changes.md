# Why four targets rest on these numbers

Four of the 11 targets on [the scoreboard](../scoreboard.json) needed a judgement about which printed number counts. This
page gives the number each one rests on, the reason, and the scope behind them.

| Paper | Benchmark | Target | Number it rests on |
|---|---|---|---|
| 1 | CLEVR, 5 objects | **99%** | Projective Composition, 98 of 100 hand-counted images, Table 1, p17 |
| 2 | T2I-CompBench Complex, ImageReward | **0.4761** | Rectify Then Diffuse, 0.4661, Table 2, p6 |
| 2 | RareBench Concat, ImageReward | **0.4564** | Rectify Then Diffuse, 0.4464, Table 2, p6 |
| 2 | GenEval two objects, 553 released prompts | **0.79**, and a win must also beat plain SDXL's 0.80 | FKC, 0.78, Table A4, p39 |

## Why each one

**CLEVR.** Projective Composition composes models that saw only single-object images, the closest printed setting to Paper
1. Its appendix describes a model conditioned on coordinates (p15), so the composed parts may be one network. Domain
Expansion's 44.5 at 5 positions is a different benchmark, because it trains one base model on all 30,000 multi-object images
(its p11). With 100 hand-counted images, one image is the whole margin between the rival and the target.

**Complex and RareBench Concat.** Rectify Then Diffuse states the same SDXL setup as CO3 and prints higher scores on both.
Its baseline ImageReward cells match CO3's table on 19 of 20 values, so CO3 was probably not rerun.

**GenEval two objects.** This is the one headline that counts two separate objects, which is what Paper 2 claims to restore.
FKC released no list of its 1,000 prompts, so the target uses GenEval's released 553. There FKC prints 0.78 and the same authors' plain SDXL
prints 0.80, so the scoreboard target is 0.79 and a win must also beat plain SDXL's 0.80 on the same prompts and seeds.

## The scope behind them

**Paper 2 targets are on SDXL.** Newer base models score higher with no composition method at all. Plain SD3 prints 0.8132
on Color and 0.7334 on Texture in T2I-CompBench++ (Table XIII, p12). Those belong in the paper as context.

**Methods that need an outside language model are reported beside the targets and do not set them.** RPG on SDXL prints
Color 0.8335, Shape 0.6801 and Texture 0.8129 (Table 1, p9). It needs GPT-4, hand-written examples and planned regions, which
Paper 2's stated setting of one model, the joint prompt and a checker leaves out. It names no scorer, prompt count or
images per prompt.

## What no target can cover

No published row combines models trained apart with several separate objects, so Paper 1's central claim has no rival
number. It needs its own table, with a control that samples each model alone and pastes the two results side by side.
