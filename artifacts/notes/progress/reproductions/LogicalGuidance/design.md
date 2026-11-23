# Reproducing Logical Guidance on Colored MNIST

> **Status.** By the owner's answer to board question 35, Colored MNIST is context and sets no target, because Paper 1 gives each model its own region of the picture and this benchmark asks for several attributes on one object. This run does not go ahead, and its printed numbers stay published.

## What this asks

Does the authors' released code, trained from scratch, give the 83.6% conformity score printed for Logical Guidance with repulsive
guidance on the Colored MNIST "AND" task? The task asks for one digit in one colour. The conformity score is the percentage of generated
images that a separate classifier reads as having both.

## The run

**Group.** Reproduction (group 4). The number freezes as a baseline (group 5) when it lands.

**Varied axis.** None. Three independent trainings measure how much the score moves between trainings.

**Buys.** `LogicalGuidance-T4-009` in `table.csv`.

**Cost.** Each of 3 trainings is one 50,000-step diffusion model and one composition classifier, budgeted at 1.5 GPU-hours. One judge
classifier adds about 0.2 GPU-hours. Sampling 10,000 images per training is about 0.3 GPU-hours. The total is about 6 GPU-hours, and the
smoke rung measures it.

## The code and weights

The official repository is `github.com/TanjaBien/LogDiff`, linked from the paper's ICML 2026 slides and exported at commit `94ef35b`. It
ships no trained weights, so every model is trained here.

## The protocol

| Setting | Value | Where it comes from |
|---|---|---|
| Data | Colored MNIST at 28 by 28, `cmnist_uniform` | `configs/dm_cmnist.yaml` and the checkpoint path in `configs/cmnist_inference.yaml` |
| Diffusion model | U-Net, channels 56, 112, 168, 50,000 steps, learning rate 1e-3 | paper Table 9 and `configs/diffusion/cmnist_dm.yaml` |
| Sampler | 50 steps, guidance 1.0 | paper Table 9 and `evaluate_cmnist_guidance.py` |
| Repulsive guidance | on, with `use_neg_guidance: true` and weight `not: 1.0` | Table 4 prints no weight, so this is the default in `configs/cmnist_inference.yaml` |
| Queries | 100 random digit-and-colour queries of 100 samples each, 10,000 samples | paper appendix E.2 |
| Judge | a ResNet-18 with digit and colour outputs, from `configs/cs_cmnist.yaml` | paper appendix E.2 |
| Composition classifier | trained on noisy images, from `configs/cs_cmnist_composition.yaml` | README |
| Evaluation seed | 42 | `evaluation_utils.py` |

## Where the paper and the code differ

**Sample count.** The paper states 10,000 samples from 100 queries. `evaluate_cmnist_guidance.py` sets `EVAL_TOTAL_SAMPLES = 1000`. This run
uses the paper's count, because the brief asks for the published sample count.

**Python version.** The five `evaluate_*.py` entry scripts use a Python 3.12 f-string, and `co3` is Python 3.10. All the library code parses
under 3.10. A new entry script in the replications hub makes the same calls with the quoting changed, and no upstream file is edited.

**Libraries.** The authors pin torch 2.9.0, while `co3` holds 2.5.1. Their pinned hydra-core and pytorch-lightning are added as a separate
package folder under `/datasets/mmolefe/poe_repair_min/loop-5/envs/`.

## The test, fixed before the first run

This rival is retrained, so sample sets from one model would leave out the variation between trainings. Three independent trainings each
give one conformity score. Logical Guidance counts as `reproduced` when the published 83.6 lies within
mean ± 4.303 × sd × √(1 + 1/3) of the three scores, where 4.303 is the two-sided 95% t value for 2 degrees of freedom. Otherwise it is
`not reproduced`, and all three scores are recorded.

## Seeds

Training seeds 1, 2 and 3 are declared here. Nothing is tuned, so this run has no tuning seeds, and no setting is chosen on any seed.

## The ladder

Each rung saves its config, metrics, image sheet and commit under `/datasets/mmolefe/poe_repair_min/loop-5/renders/LogicalGuidance/<rung>/`.

1. **Smoke test.** One short training of each model and one batch of samples through the entry script, recording the time per step.
2. **Overfit.** The diffusion model and the classifiers each fit 16 images.
3. **1% of the data.** 500 diffusion steps and one query of 100 samples, with the image sheet opened at full size.
4. **Controls.** The judge reaches at least 95% on digit and on colour on held-out images. With every guidance term at zero, the sampler must
   match the same pipeline's unconditional samples to under 1e-5 (largest absolute difference).
5. **Full protocol.** Three trainings and 10,000 samples from each.
