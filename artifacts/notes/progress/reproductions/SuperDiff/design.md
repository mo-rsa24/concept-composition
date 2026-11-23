# Reproducing SuperDiff on CIFAR-10 disjoint halves

## What this asks

SuperDiff samples from two diffusion models at once, weighting them by how likely the current sample is under each. Its CIFAR-10 test trains one
model on the first five labels and one on the last five, then generates from both. The question is whether the authors' code, trained from scratch,
gives their printed Fréchet Inception Distance (FID) of 4.11, where lower is better.

## Status

The row is `blocked` by the owner's answer to board question 22, and nothing has run. The question goes back to the owner once the cheaper
rivals have landed.

## The row

| Row | Method | Benchmark | Published FID | Source |
|---|---|---|---|---|
| `SuperDiff-T1-031` | SuperDiff, T = 100, OR, ODE inference | CIFAR-10 disjoint halves | 4.11 | SuperDiff Table 1, p8 |

SuperDiff's Colored MNIST row is `not comparable` by the owner's answer to board question 13.

## What exists

| Piece | State |
|---|---|
| Code | official, `github.com/necludov/super-diffusion` at commit `e859107`, with `configs/sm/cifar/vpsde_less_5.py` and `vpsde_more_5.py` for the two halves |
| Weights | none released for CIFAR-10; the authors' Hugging Face account holds Stable Diffusion models only |
| Environment | the cluster's `superdiff` environment has jax 0.4.28 and flax 0.9.0; its GPU access is unchecked |
| Earlier runs here | `original-superdiff` logs cover MNIST experiments, and no trained CIFAR-10 checkpoint was found |

## The protocol

| Setting | Value | Where it comes from |
|---|---|---|
| Training | 500,000 iterations, batch 128, learning rate 2e-4, per half | the two config files |
| Inference | ODE with density estimation, 100 steps | Table 1's T = 100 row |
| Images | 50,000 per FID | `eval.num_samples` in the configs; the paper does not state a count |

## The test, if the owner approves it

FID is computed over a whole set, and this rival is retrained, so the variation between trainings has to be measured. Three independent trainings of
both halves each give one FID on 50,000 images. SuperDiff counts as `reproduced` when the published 4.11 lies within
mean ± 4.303 × sd × √(1 + 1/3) of the three values, where 4.303 is the two-sided 95% t value for 2 degrees of freedom. Otherwise it is `not reproduced`.

## The cost, if the owner approves it

**Cost.** Six trainings of 500,000 steps at an estimated 8 to 12 steps per second on an RTX 3090 are about 70 to 105 GPU-hours. Three joint sampling runs
of 50,000 images add about 5 GPU-hours. The total is about 75 to 110 GPU-hours, an estimate until a smoke test measures the step time.

**Buys.** `SuperDiff-T1-031`, the number behind Paper 1's CIFAR-10 target.

**A cheaper version does not qualify.** One training per half is about 25 to 35 GPU-hours, but the brief marks a single training `not comparable`, because
it leaves out the variation between trainings.
