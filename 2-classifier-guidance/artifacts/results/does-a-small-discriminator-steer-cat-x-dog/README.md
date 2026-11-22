# Does a small pair-aware discriminator steer SDXL's cat × dog product?

Rung 2 of the classifier-guided PoE ladder ([A3q's page](../../../report/classifier-guided-poe/a3q-cat-x-dog-quick-discriminator-test.md),
[plan 21](../../../plans/21-rung2-a-small-discriminator-steers-cat-x-dog.md),
[plan 22](../../../plans/22-rung2-a-discriminator-trained-on-the-benchmark-pairs.md)). Everything here
was made on 2026-11-17.

## training-set/

**What it is.** The 73 renders rung 2's first discriminator trained on: seeds 1 to 8 × six prompts,
plus 25 fresh-noise draws that replaced damaged lone-animal and product tiles and added pairs.
`calls.csv` holds every call (17 positives, 24 intact negatives); `problem-seeds.csv` lists the 23
pair-prompt tiles that did not give one cat and one dog apart.

**How it was made.** `scripts/rung2/render_training_set.py` on `mscluster107` GPU 0; the reader
called the first 48 tiles, and delegated the rest to Claude with veto after.

`kept-positives.png` is on the cluster at `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/01-the-first-quick-test/training-set/strips/kept-positives.png`
`kept-negatives.png` is on the cluster at `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/01-the-first-quick-test/training-set/strips/kept-negatives.png`

## result-sheet/

**What it is.** Seeds 1 to 4 and 9 to 12 by plain product, best of 2, guided at s = 10, 30, 100 and
a same-sized random push at each, rendered with the first discriminator. `calls-draft.csv` holds
Claude's draft calls (the reader has not confirmed them): 0 of 4 composed on seeds 9 to 12 in every
column. `result.json` holds the per-step push sizes, cap firing and checker scores.

`result-sheet.png` is not shown because it failed the subject check

## diagnostics/

**What it is.** What the first discriminator reads: filmstrips of the predicted picture with its
attention per step (seeds 1, 3, 9, 10), its score through all 50 steps per run, and every training
tile kind scored per step. Made by `scripts/rung2/diagnose.py` on `mscluster107` GPU 1.

<a href="01-the-first-quick-test/diagnostics.png"><img src="01-the-first-quick-test/diagnostics.png" width="200" alt="Logit against step per tile kind: hybrids and wrong pairs from the pair prompts score with the positives"></a>

## discriminator/

**What it is.** Held-out balanced accuracy per sampler step for every discriminator trained:
v1 (17 positives, easy negatives), v2 (pair-prompt failures added as negatives), v3 head only and
v3 with adapters (81 positives over 38 benchmark pairs). Only v1 passes, on a check that held no
hard negatives.

<a href="01-the-first-quick-test/discriminator.png"><img src="01-the-first-quick-test/discriminator.png" width="200" alt="Held-out accuracy against step for four discriminators; only v1 clears 0.75"></a>

## dataset-v3/

**What it is.** Plan 22's training set from about 1,000 local 1024 px pair pictures: 543 called by
two independent passes of helper agents (91% agreement), the 37 disagreements in the 87 two-things
pairs settled by Claude at full size. `calls-final.csv` records how each call was decided;
`spot-check-sheet.png` is the reader's sample. The latents and the pictures for training live in
`/home-mscluster/mmolefe/poe_repair_min_scratch/outputs/classifier_guidance_ladder/rung2/` while
`/datasets` is down.

`spot-check-sheet.png` is not shown because it failed the subject check

<!-- cluster:start -->

## On the cluster

The other 11 files of this result (44.8 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `comparison-sheets` | 11 | 44.8 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/comparison-sheets/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/<run>/ <run>/
```

<!-- cluster:end -->
