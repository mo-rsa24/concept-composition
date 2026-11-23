# Reproducing CoInD and its composed baseline on CelebA

## What this asks

Do the authors' released SD1.5 weights reproduce the conformity scores in CoInD's Table 6? The conformity score is the percentage of
generated faces that a classifier reads as both smiling and male, when the two ideas come from separate prompts. The model never saw a
smiling man during fine-tuning, so this is a composition it has to invent.

CoInD fine-tunes SD1.5 with an extra loss that keeps the two attributes independent. The baseline, printed as Composed GLIDE, is the
same fine-tune without that loss, and it stands in for Liu's composition on this row.

## The run

**Group.** Reproduction (group 4). Each number freezes as a baseline (group 5) when it lands.

**Varied axis.** Which released U-Net is loaded. The prompts, seed, sampler and classifier are the same for both.

**Buys.** Two rows of `table.csv`.

| Row | Method | Published conformity score | Source |
|---|---|---|---|
| `CoInD-T6-011` | CoInD, λ = 1.0 | 49.15% | CoInD Table 6, p27, and the authors' results sheet |
| `CoInD-T6-004` | Composed GLIDE, the Liu row | 14.19% | CoInD Table 6, p27, and the authors' results sheet |

**Cost.** 10,000 images per method is 20,000 images. At 6.1 seconds per image, measured on an RTX 3090 in the smoke rung, that is about
34 GPU-hours, over the 24 GPU-hour limit for one run. By the owner's answer to board question 27, each method renders the first 2,500
images, about 8.5 GPU-hours for both, and each row is labelled with that count. The 30,000-step classifier adds about 0.6 GPU-hours.

## The weights

`sd_celeba.zip` (25.3 GB) from the Google Drive link in the authors' results sheet holds both methods. Only the U-Net weights and configs
were unpacked, 13.8 GB in `safetensors` format, with checksums in
`/datasets/mmolefe/poe_repair_min/loop-5/third_party/coind-sd-celeba/unpacked/manifest.json`. The archive's 8 pickle-format
training-state files stay sealed, because loading them could run code.

## The protocol

| Setting | Value | Where it comes from |
|---|---|---|
| Prompts | "Photo of a smiling celebrity" and "Photo of a male celebrity" | CoInD appendix E.3 |
| U-Net | the `unet/` folder of each checkpoint | `test_text_to_image.py` loads this folder, not `unet_ema/` |
| Base model | `runwayml/stable-diffusion-v1-5` | `test_text_to_image.py` |
| Composition | `ComposableStableDiffusionPipeline`, equal weights of 7.5 and an empty negative prompt | the code's default when `--weights` is not passed |
| Steps | 100 | `test_text_to_image.py` |
| Size | 512 by 512 | the U-Net's `sample_size` of 64 |
| Seed | 42 plus the batch number, one generator per batch of 5 | `test_text_to_image.py` sets seed 42 once for all batches; a seed per batch lets separate jobs render separate batches, so individual images differ from any the authors drew |
| Images | the first 2,500 per method | the owner's answer to board question 27; the paper's stated CelebA count (p22) and the default of `generate_celeba_male_smile_samples.py` are 10,000, and Table 6 does not state its own |

## The conformity classifier

The authors release no classifier, so one is trained from their config, `configs/cs_celeba.yaml`: a ResNet-18 with two outputs (male,
smiling), AdamW at a learning rate of 3e-4, on CelebA's training split, which holds all four combinations. Its input is 128 by 128, as in
`evaluate_celeba_gender_smile.py`.

By the owner's answer to board question 30, a classifier trained from the authors' config counts when its overall accuracy on CelebA's
test split is within one percentage point of both accuracies CoInD prints, 98.2% for gender and 92.1% for smile (Figure 10, p24). The bar
is `MAX_ACCURACY_GAP` in the hub's `coind_celeba.py`.

| Classifier | Gender | Smile | Largest gap from print | Lowest per-group accuracy | Role |
|---|---|---|---|---|---|
| 10,000 steps, the authors' config | 97.93% | 92.96% | 0.86 points | 88.75%, smiling on smiling men | sets the verdict |
| 30,000 steps | 97.98% | 92.38% | 0.28 points | 89.01%, smiling on smiling men | scores the same images, printed beside it |

Every row prints the per-group accuracy, because the paper reports none per group. The existing classifier in
`/datasets/mmolefe/celeba/classifier/` reads male and blond, so it does not apply.

## The test, fixed before the first run

The conformity score is per image. The standard error comes from resampling the 2,500 images with replacement 10,000 times. Table 6
prints no standard error, so the published one is taken as equal to ours. A row counts as `reproduced` when
|ours − published| ≤ 1.96 × √(SE_ours² + SE_published²), and otherwise as `not reproduced`, with both numbers recorded.

A row stays `not comparable` if the 10,000-step classifier's score and the 30,000-step classifier's score
differ by more than the row's pass band, 1.96 × √(SE_ours² + SE_published²).

Our classifier is a retraining of the authors' scorer, so a miss could come from the classifier as well as from the weights. The row's
note says so.

## The data

CelebA is at `/datasets/mmolefe/celeba/flwrlabs-celeba/` under a non-commercial research licence that forbids further distribution.
Generated and real faces stay on the cluster. No CelebA sheet is committed, published or filed on a page.

## The ladder

Each rung saves its config, metrics and commit under `/datasets/mmolefe/poe_repair_min/loop-5/renders/CoInD/<rung>/`.

1. **Smoke test.** One batch of 5 images per method, recording the time per image.
2. **Overfit.** The classifier memorises 16 images, reaching at least 15 of 16 right on both outputs.
3. **1% of the data.** 100 images per method, looked at on the cluster.
4. **Controls.** The classifier passes its check against the printed accuracy. With the same seed, the CoInD and baseline U-Nets produce different images, which
   shows the two checkpoints were both loaded.
5. **Full protocol.** The first 2,500 images per method, scored with both classifiers.

## Shapes3D

The authors' results sheet has an empty Shapes3D tab, so those two rows (CoInD 91.10%, Liu 51.56%) have no released weights. They need
at least 3 independent trainings per method, and they get their own cost and question before anything runs.
