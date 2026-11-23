# Reproducing CoInD and its baseline on Shapes3D

> **Status.** By the owner's answer to board question 35, Shapes3D is context and sets no target, because Paper 1 gives each model its own region of the picture and this benchmark asks for several attributes on one object. This run does not go ahead, and its printed numbers stay published. Board question 34 skips this rerun.

## What this asks

Does the authors' code, trained from scratch, give the conformity scores printed for Shapes3D? Each Shapes3D image has six attributes, such as
floor colour and object shape. The models never see some combinations during training, and they are asked to generate exactly those by composing
one guidance term per attribute. The conformity score is the percentage of generated images in which a classifier reads all six attributes as
requested.

CoInD adds a loss that keeps the attributes independent. Its baseline, printed as Composed GLIDE, is the same model without that loss, and it
stands in for Liu's composition on this row.

## The run

**Group.** Reproduction (group 4). Each number freezes as a baseline (group 5) when it lands.

**Varied axis.** The weight of the independence loss, λ = 1.0 for CoInD and λ = 0 for the baseline. Three trainings per method measure how much
the score moves between trainings.

**Buys.** Two rows of `table.csv`.

| Row | Method | Published conformity score | Source |
|---|---|---|---|
| `CoInD-T4-036` | CoInD, λ = 1.0 | 91.10% | CoInD Table 4, p20 |
| `CoInD-T4-026` | Composed GLIDE, the Liu row | 51.56% | CoInD Table 4, p20 |

**Cost.** Six trainings of 100,000 steps and six evaluations, inside the owner's budget of 18 to 36 GPU-hours (board question 30). The smoke rung
measures the step time. If the measured total passes 36 GPU-hours, the run stops for a new question.

## The code and data

The authors' code is `coind` at commit `99056cc`, and they released no Shapes3D weights. The data is the composition split of Shapes3D from
[Zenodo record 4898937](https://zenodo.org/records/4898937), CC BY 4.0, which the authors' loader names. Its files are about 11.2 GB and go to
`/datasets/mmolefe/poe_repair_min/loop-5/data/shapes3d/`, each checked against the MD5 Zenodo lists.

The authors' loader calls `np.load` with `allow_pickle=True`. Before it runs, every downloaded file is opened with `allow_pickle=False`, which
refuses pickled objects, so only plain numeric arrays reach it.

## The protocol

| Setting | Value | Where it comes from |
|---|---|---|
| Training data | `shapes3d_composition_train` | `configs/dataset/shapes3d_orthogonal.yaml`, paper appendix D |
| Model | U-Net, channels 56, 112, 168, 224, 17.2M parameters, 64 pixels | paper Table 5, `configs/diffusion/shapes3d_dm.yaml` |
| Training | 100,000 steps, batch 64, AdamW at 2e-4, 5,000 warm-up steps | paper Table 5 and the config |
| Null-label probability | 0.3 | the code's `p_null_mask`; the paper's text says 0.2, and the code settles it |
| Independence loss | λ = 1.0 with `coind_masking: one`; λ = 0 for the baseline | Table 4's row names, and the config |
| Sampler | DDIM, 100 steps | paper Table 5 and the row's protocol; `evaluate_shapes3d.py` sets 150 |
| Guidance | 7.5 per attribute | `evaluate_shapes3d.py` |
| Training seeds | 1, 2, 3 | declared here; nothing is tuned, so no seed chooses a setting |

**What the release does not settle.** The paper evaluates AND on unseen compositions. The released `evaluate_shapes3d.py` has that query commented
out for the unseen split, and it keeps 4 images of each batch of 64, about 628 images. The owner's answer to board question 34 skips this rerun, so neither
evaluation runs.

## The conformity classifier

The classifier is a ResNet-18 with six outputs from `configs/cs_shapes3d.yaml`, trained for its 10,000 steps on the full dataset. By the rule the
owner set on board question 30, it counts when it is within one percentage point of the accuracy the paper prints. Figure 10 (p24) prints 100% for all
six attributes, so each attribute must reach at least 99% on the held-out random test split.

## The test, fixed before the first run

This rival is retrained, so three independent trainings each give one conformity score. A row counts as `reproduced` when the published value
lies within mean ± 4.303 × sd × √(1 + 1/3) of the three scores, where 4.303 is the two-sided 95% t value for 2 degrees of freedom. Otherwise it is
`not reproduced`, and all three scores are recorded.

## The ladder

Each rung saves its config, metrics, image sheet and commit under `/datasets/mmolefe/poe_repair_min/loop-5/renders/CoInD-Shapes3D/<rung>/`.

1. **Smoke test.** 200 training steps of each method and one batch of samples, recording the time per step and per image.
2. **Overfit.** The diffusion model and the classifier each fit 16 images.
3. **1% of the data.** 1,000 training steps of each method and 100 samples, with the sheet opened at full size.
4. **Controls.** The classifier passes its 99% check. With every guidance term at zero, the composed sampler matches the same loop's unconditional
   samples to under 1e-5 (largest absolute difference).
5. **Full protocol.** Three trainings per method, each evaluated as question 34 settles.
