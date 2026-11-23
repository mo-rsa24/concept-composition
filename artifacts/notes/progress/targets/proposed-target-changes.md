# Why four targets rest on these numbers

Four of the 7 targets on [the scoreboard](../scoreboard.json) needed a judgement about which printed number counts. This
page gives the number each one rests on, the reason, and the scope behind them.

| Paper | Benchmark | Target | Number it rests on |
|---|---|---|---|
| 1 | CLEVR, models trained on one object | **7 objects** | Bradley's local patch model, 5 objects, Table 1, p3, and Projective Composition's 97 of 100 at 6 objects, Table 1, p17 |
| 2 | T2I-CompBench Complex, ImageReward | **0.4761** | Rectify Then Diffuse, 0.4661, Table 2, p6 |
| 2 | CLEVR, models trained on one object | **7 objects** | The same row as Paper 1, with one network plus our checker |
| 2 | GenEval two objects, 553 released prompts | **0.79** printed, and a win must beat our plain SDXL rerun, 0.7399 | FKC, 0.78, Table A4, p39 |

## Why each one

**CLEVR.** Paper 1's models each see one concept, so the target is the most objects that succeed for models trained on single
objects. A count K succeeds when K objects appear in at least 50% of images and at least K-2 appear in at least 90%. Bradley's local
patch model reaches 5 on 1,024 images (Local Mechanisms, Table 1, p3). Projective Composition places all 6 objects in 97 of 100
hand-counted images (Table 1, p17) and first clearly fails at 7 (Figure 3, p3), so the target is 7. Both are a single network asked at
each position, and both train on 134 million images against our 8 million, so printed numbers are context and a win is judged against
the same rule on our models. Domain Expansion's 44.5 at 5 positions is a different benchmark, because it trains one base model on all
30,000 multi-object images (its p11).

**Complex.** Rectify Then Diffuse states the same SDXL setup as CO3 and prints higher scores on Complex and RareBench Concat.
Its baseline ImageReward cells match CO3's table on 19 of 20 values, so CO3 was probably not rerun.

**Our reruns sit beside those two printed numbers**, in the [reproduction table](../reproductions/table.csv). On T2I-CompBench
Complex, Rectify Then Diffuse scores ImageReward 0.2423 on 988 of the 1,000 prompts at 3 seeds, against 0.4661 printed, so it is
not reproduced (W&B `5i9dt7by`). On the same pictures it sits 0.0214 below plain SDXL, with a standard error of 0.0153. The printed bar stays 0.4761, and a win on Complex is judged against the best finished rerun, CO3's 0.3326. On RareBench Concat it scores 0.4166 on 33 of 40 prompts at 3 seeds, against
0.4464 printed (W&B `tc75fx0i`). That passes, but with 33 prompts the pass band is plus or minus 0.443, so the test says little, and
RareBench Concat is a reported row that sets no target.

**ToMe and plain SDXL reruns.** On Shape, ToMe scores 0.5502 on 293 of 300 prompts at seed 0, against 0.6051 printed, so it is not
reproduced (W&B `8cowoyea`). On Texture it scores 0.6503 on all 300 prompts at seed 0, against 0.6894 printed, which is reproduced
(W&B `ljypsxnd`). Plain SDXL on GenEval's 99 two-object prompts scores 0.7399 over 396 images, against 0.80 printed, which is
reproduced (W&B `qcdz35cj`). Each win reports the printed number beside it as reached or not.

**CO3 reruns.** CO3's released script scores 0.3326 on T2I-CompBench Complex over 988 prompts at 3 seeds, against 0.4406 printed, so
it is not reproduced (W&B `02fl0ia0`). On the same pictures plain SDXL scores 0.2637 and Rectify Then Diffuse 0.2423, so CO3 ranks
first here, while Rectify Then Diffuse's paper prints itself first. The released script's settings differ from the paper's section
4.1, whose corrector guidance of 5.0 sits outside the 0 to 1 the released code allows, so the miss could come from the settings. On
RareBench Concat CO3 scores 0.4374 on 33 prompts at 3 seeds, against 0.1610 printed, which passes on a weak test (W&B `i11r81ox`).

These reruns stay out of the scoreboard's `reproduced` field, because their prompt counts differ from the published ones or the
published page leaves their image count and seeds unstated.

**GenEval two objects.** This is Paper 2's one text-to-image headline that asks for two named objects. Its evaluator checks each class alone, so one fused object can pass.
FKC released no list of its 1,000 prompts, so the target uses GenEval's released 553. There FKC prints 0.78 and the same authors' plain SDXL
prints 0.80, so the printed bar is 0.79, and a win must beat our plain SDXL rerun on the same pictures by the noise margin. FKC is not
rerun, so this row is judged against our plain SDXL rerun and says FKC's 0.78 is printed only.

## The scope behind them

**Paper 2's text-to-image targets are on SDXL, and its CLEVR target tests the same method on a controlled count.** Newer base models score higher with no composition method at all. Plain SD3 prints 0.8132
on Color and 0.7334 on Texture in T2I-CompBench++ (Table XIII, p12). Those belong in the paper as context.

**Methods that need an outside language model are reported beside the targets and do not set them.** RPG on SDXL prints
Color 0.8335, Shape 0.6801 and Texture 0.8129 (Table 1, p9). It needs GPT-4, hand-written examples and planned regions, which
Paper 2's stated setting of one model, the joint prompt and a checker leaves out. It names no scorer, prompt count or
images per prompt.

**Shapes3D, Colored MNIST and CelebA are context.** They put several attributes on one object, from one network, so Paper 1's
method of giving each model its own region cannot run on them.

**CIFAR-10 halves are context.** SuperDiff's FID of 4.11 comes from the OR setting (p7), which samples from either model to
recover the full dataset. Paper 1 claims AND, both things in one image, so this number sets no target.

## What no target can cover

No published row combines models trained apart with several separate objects, so Paper 1's central claim has no rival
number. It needs its own table, with a control that samples each model alone and pastes the two results side by side. The
[CLEVR scaling plan](clevr-scaling-plan.md) holds that table.
