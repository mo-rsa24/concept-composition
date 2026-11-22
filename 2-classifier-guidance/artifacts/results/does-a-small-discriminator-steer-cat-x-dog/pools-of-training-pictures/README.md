# The pools of training pictures

This folder answers which labelled pictures, beyond cat and dog, the multi-pair checkers trained on. No report page owns it: the [128-seed comparison](../../../../report/classifier-guided-poe/how-close-the-kept-push-comes-to-the-joint-prompt.md) reports what those checkers did, and the [checker loop](../../../drips/rung2-pair-aware-discriminator/checker-loop.md) records how each pool was collected and labelled.

## What each run tried

Rows follow the time each run's first file was written `inferred`.

| Run | What changed | What came out | Picture |
|---|---|---|---|
| `benchmark-pool` | labelled pictures of other pairs added to the checker's training: the joint prompt, the adapter, CO3 and ToMe at seeds 9 to 12 on the Attend-and-Excite benchmark. The checker loop records 4,400 pictures across 275 pairs. This folder holds 3,680 over 230 pairs, because the subject guard keeps the other 720 pictures of 45 pairs out | kept for training: 611 clean and 645 rejects over 200 pairs (checker loop). `training_set_multipair.json` holds 242 pool pairs plus cat and dog, and 200 of the 242 have a clean picture. The checker trained on it, `result-sheet-catdog-big-multipair`: two bodies 116 and faithful 29 of 128, kept | <a href="benchmark-pool.jpg"><img src="benchmark-pool.jpg" width="120" alt="An elephant wearing a crown and a red cape, drawn in a heraldic style"></a> |
| `pair-runs-pool` | 78 more pairs' joint-prompt pictures added to the multi-pair checker's training: 3,120 pictures, seeds 5001 to 5040 (checker loop) | labelled faithful pair 393, hybrid 1,381, wrong pair 725, single 55, damaged 566; for training clean 393, reject 778, left out 1,949 (checker loop). The checker with this pool added, `result-sheet-catdog-big-multipair2`: two bodies 115 and faithful 23 of 128, dropped | <a href="pair-runs-pool.jpg"><img src="pair-runs-pool.jpg" width="120" alt="A brown bear standing in a river with a salmon beside it, in an inked style"></a> |

Plans 21 and 22 name neither pool. The 48 benchmark pairs in [plan 22](../../../../plans/22-rung2-a-discriminator-trained-on-the-benchmark-pairs.md) are pictures rendered afresh for that plan, and they are in `dataset-v3`.

<!-- cluster:start -->

## On the cluster

The other 6,809 files of this result (812.4 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/pools-of-training-pictures/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `benchmark-pool` | 3,687 | 434.0 MB | [benchmark-pool.jpg](benchmark-pool.jpg) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/pools-of-training-pictures/benchmark-pool/` |
| `pair-runs-pool` | 3,122 | 378.4 MB | [pair-runs-pool.jpg](pair-runs-pool.jpg) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/pools-of-training-pictures/pair-runs-pool/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/pools-of-training-pictures/<run>/ <run>/
```

<!-- cluster:end -->
