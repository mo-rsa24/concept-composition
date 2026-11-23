# Review: reproducing CoInD on CelebA

This file holds what the runs in [the CoInD design](design.md) produced. CelebA is context by the owner's answer to board question 35, so this rerun
stopped after the classifier and the smoke test. No generated face was scored, and 49.15% and 14.19% stay published numbers.

## Runs

Outputs are on the cluster under `/datasets/mmolefe/poe_repair_min/loop-5/renders/CoInD/`, and logs are in
`/datasets/mmolefe/poe_repair_min/loop-5/logs/`. Real and generated faces stay on the cluster. A commit pair is the repository commit,
then the replications hub commit.

| Rung | Slurm jobs | Commits | Output | State |
|---|---|---|---|---|
| Classifier overfit | 93120 | `d297871d-b2f1320` | `overfit/overfit.json` | passed |
| Smoke test, one batch of 5 per checkpoint | 93123 (CoInD), 93130 (baseline), after 93121 and 93122 failed | `d75bad9f-f28dd6c` | `smoke/` | passed |
| Classifier, 10,000 steps | 93131 | `d75bad9f-f28dd6c` | `classifier/` | passed the printed-accuracy bar, sets the verdict |
| Classifier, 30,000 steps | 93210 | `f96659e8-3bc1415` | `classifier-30k/` | passed the printed-accuracy bar, printed beside it |
| 1% of the data | none | | | not run, because CelebA became context |
| Full protocol, 2,500 images per method | none | | | not run, because CelebA became context |

Job 93121 failed because the cached SD1.5 holds no safety-checker weights, and the authors' script switches that checker off after
loading. Job 93122 failed because the authors' pipeline calls an undefined `logger` when the checker is absent. The render now loads
the pipeline with the checker off, which leaves the images unchanged.

## Questions

**Can the classifier memorise 16 images?** ✅ Yes. It reads 16 of 16 right on both male and smiling, against a bar of 15.

**Do both released checkpoints render?** ✅ Yes. Each wrote 5 images at 6.1 seconds per image on an RTX 3090, and the first image from
each differs by checksum, which shows two different U-Nets were loaded. Nobody has looked at these images.

**Is the classifier within one percentage point of the accuracy CoInD prints?** ✅ Yes, at both lengths. CoInD prints 98.2% for gender and
92.1% for smile (Figure 10, p24). The 10,000-step classifier reads 97.93% and 92.96% (W&B `zolb4ltw`), and the 30,000-step classifier reads
97.98% and 92.38% (W&B `uo78zzgu`), both on CelebA's 19,962 test faces.

**How accurate is it in each group?** The weakest cell is reading smiling on smiling men, at 88.75% and 89.01%. That is under the 90% per-group
bar this design first set. The owner replaced that bar on board question 30, because the paper reports no per-group accuracy to compare with.

| Test group | Faces | Male right, 10,000 steps | Smiling right, 10,000 steps | Male right, 30,000 steps | Smiling right, 30,000 steps |
|---|---|---|---|---|---|
| Women, not smiling | 5,354 | 97.8% | 93.9% | 97.7% | 92.9% |
| Women, smiling | 6,893 | 98.8% | 93.1% | 98.7% | 93.0% |
| Men, not smiling | 4,621 | 97.0% | 94.5% | 97.3% | 93.1% |
| Men, smiling | 3,094 | 97.5% | **88.8%** | 97.7% | **89.0%** |

The weak cell matters for this row in particular, because the conformity score counts images read as both smiling and male. A classifier
that misses smiling on about 11% of smiling men can pull that score down, so every CelebA number is printed with this table.

Tripling the training moved the worst cell by 0.26 percentage points. Both runs follow `configs/cs_celeba.yaml` apart from the step count: ResNet-18, batch 64, AdamW at 3e-4 with weight decay 1e-3, cosine schedule,
gradient clip 1.0, mixed precision, 128-pixel images with a random flip.

**Does CoInD reproduce its published conformity score?** Not tested. CelebA is context, so 49.15% (CoInD) and 14.19% (Composed GLIDE, the Liu
row) stay published numbers.
