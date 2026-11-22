# The first quick test of a small discriminator on cat and dog

<!-- cluster:start -->

## On the cluster

The other 192 files of this result (409.6 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/01-the-first-quick-test/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `diagnostics` | 7 | 9.4 MB | [diagnostics.png](diagnostics.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/01-the-first-quick-test/diagnostics/` |
| `discriminator` | 2 | 0.1 MB | [discriminator.png](discriminator.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/01-the-first-quick-test/discriminator/` |
| `result-sheet` | 83 | 189.9 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/01-the-first-quick-test/result-sheet/` |
| `training-set` | 100 | 210.1 MB | [training-set.png](training-set.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/01-the-first-quick-test/training-set/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/01-the-first-quick-test/<run>/ <run>/
```

<!-- cluster:end -->
