# The pools of training pictures

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
