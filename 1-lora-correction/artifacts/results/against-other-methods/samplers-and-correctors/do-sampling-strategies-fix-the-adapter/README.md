# Do sampling strategies fix the adapter?

Seven sampling strategies run with the adapter on cat × dog, one sheet per seed:
seed 9 (`seven-strategies-cat-and-dog-seed09.png`, not linked because it failed the subject check) and seed 10 (`seven-strategies-cat-and-dog-seed10.png`, not linked because it failed the subject check).
Rows are three adapters (v58 plain, contrast, self-attention); columns are the joint prompt, the plain
product, the adapter alone, a clean tail from step 20, Langevin on the tail, the CO3 corrector and best
of 8, read from the sheet's own headers. Cited by [the render-time fixes finding](../../../../../report/is-the-gap-the-samplers-or-the-models/does-any-render-time-fix-recover-picture-quality.md#1-what-each-fix-does-to-the-picture).

<!-- cluster:start -->

## On the cluster

The other 2 files of this result (5.9 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/against-other-methods/samplers-and-correctors/do-sampling-strategies-fix-the-adapter/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `(files at this level)` | 2 | 5.9 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/against-other-methods/samplers-and-correctors/do-sampling-strategies-fix-the-adapter/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/against-other-methods/samplers-and-correctors/do-sampling-strategies-fix-the-adapter/<run>/ <run>/
```

<!-- cluster:end -->
