# How the adapter's settings change cat and dog

The paper's limitations figure (figures 6 and 7): the layout specs (`figure6*-spec.json`,
`figure7-limitations*-spec.json`), the rendered versions as PDFs (`adapter-limitations-*.pdf`),
and the tiles they are built from in `tiles/` and `limitations-tiles/`. Stated from the file
names and the figure numbers in them.

<!-- cluster:start -->

## On the cluster

The other 7 files of this result (4.3 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `(files at this level)` | 7 | 4.3 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/<run>/ <run>/
```

<!-- cluster:end -->
