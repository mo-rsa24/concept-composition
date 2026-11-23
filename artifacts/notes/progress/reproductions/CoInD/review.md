# Review: reproducing CoInD on CelebA

This file holds what the runs in [the CoInD design](design.md) produced. CoInD has no verdict. The conformity classifier missed the
90% bar set before any run, so no generated face has been scored, and both CelebA rows count as `not comparable` until board question
28 is answered. Board question 27 asks how many images to render once the classifier is settled.

## Runs

Outputs are on the cluster under `/datasets/mmolefe/poe_repair_min/loop-5/renders/CoInD/`, and logs are in
`/datasets/mmolefe/poe_repair_min/loop-5/logs/`. Real and generated faces stay on the cluster. A commit pair is the repository commit,
then the replications hub commit.

| Rung | Slurm jobs | Commits | Output | State |
|---|---|---|---|---|
| Classifier overfit | 93120 | `d297871d-b2f1320` | `overfit/overfit.json` | passed |
| Smoke test, one batch of 5 per checkpoint | 93123 (CoInD), 93130 (baseline), after 93121 and 93122 failed | `d75bad9f-f28dd6c` | `smoke/` | passed |
| Classifier, 10,000 steps | 93131 | `d75bad9f-f28dd6c` | `classifier/` | **missed its bar** |
| 1% of the data | none | | | waits on question 28 |
| Full protocol | none | | | waits on questions 27 and 28 |

Job 93121 failed because the cached SD1.5 holds no safety-checker weights, and the authors' script switches that checker off after
loading. Job 93122 failed because the authors' pipeline calls an undefined `logger` when the checker is absent. The render now loads
the pipeline with the checker off, which leaves the images unchanged.

## Questions

**Can the classifier memorise 16 images?** ✅ Yes. It reads 16 of 16 right on both male and smiling, against a bar of 15.

**Do both released checkpoints render?** ✅ Yes. Each wrote 5 images at 6.1 seconds per image on an RTX 3090, and the first image from
each differs by checksum, which shows two different U-Nets were loaded. Nobody has looked at these images.

**Does the classifier reach 90% on both attributes in all four groups of CelebA's test split?** ❌ No. Its worst cell is 88.75%, reading
smiling on smiling men (W&B `zolb4ltw`).

| Test group | Faces | Male read correctly | Smiling read correctly |
|---|---|---|---|
| Women, not smiling | 5,354 | 97.8% | 93.9% |
| Women, smiling | 6,893 | 98.8% | 93.1% |
| Men, not smiling | 4,621 | 97.0% | 94.5% |
| Men, smiling | 3,094 | 97.5% | **88.8%** |

The miss matters for this row in particular, because the conformity score counts images read as both smiling and male. A classifier
that misses smiling on about 11% of smiling men can pull that score down, so a verdict from it could reflect the classifier.

The training matches `configs/cs_celeba.yaml`: ResNet-18, 10,000 steps, batch 64, AdamW at 3e-4 with weight decay 1e-3, cosine schedule,
gradient clip 1.0, mixed precision, 128-pixel images with a random flip. The bar has not moved. Whether the paper prints its own
classifier's accuracy is unchecked.

**Does CoInD reproduce its published conformity score?** Unknown. No generated image has been scored.
