# What the adapter ablations show

The two grids behind [Appendix: adapter ablations](../../../paper/overleaf-iclr/iclr2027_conference.tex)
(`app:adapter-ablations`), one per axis, with a sidecar JSON beside each recording every plotted
value, the run that logged it, and what else moved in that comparison.

Both read the DINOv2 drift, which each training run logs at the end of an evaluation pass: the
corrected render's cosine distance to the joint-prompt render minus its cosine distance to the
plain product render. It is dimensionless, runs from -2 to +2, and is negative when the render
sits nearer the joint-prompt target, so lower is better. The compose rate is not used because it
reads 1.0 for every configuration here and cannot separate them.

## which-layers-carry-the-adapter.png

**What it shows**

DINOv2 drift per tracking cell for four rank-32 adapters on the same pool: three that adapt the
cross-attention query, key and value projections, and one that also adapts the self-attention
projections. Two cells are pairs the adapters trained on, four are held out.

**What it says**

Covering self-attention as well is not systematically better or worse. It is the best of the four
on cat and dog seed 9 (-0.527) and the worst on elephant and penguin seed 9 (+0.174, the only
held-out cell in the set where an adapter leaves the render nearer the plain product).

**How it was made**

`scripts/adapter_ablation_figures.py`, from `measures.json`, which `scripts/local/extract_measures.py`
builds out of each run's `history.json` on the cluster. No value is recomputed here.

**What it cannot say**

The comparison is mixed: the cross-and-self run also decays its learning rate and resumed from a
step-5,000 checkpoint. A self-attention-only adapter does not exist, so that cell of the axis is
empty rather than negative.

## how-much-rank-the-correction-needs.png

**What it shows**

DINOv2 drift against adapter rank (8, 16, 32) at fixed layers, on cat and dog seeds 9 and 10, the
only two cells the old and current tracking sets share.

**What it says**

No trend in rank on either cell, and seed 10 stays above zero at every rank.

**How it was made**

The same script and the same extraction.

**What it cannot say**

The three runs differ in loss space and in training length as well as in rank, so the differences
are not the effect of rank. Rank 64 has no evaluation pass logged and appears only as a parameter
count.

<!-- cluster:start -->

## On the cluster

The other 22 files of this result (37.2 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/what-the-adapter-ablations-show/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `(files at this level)` | 22 | 37.2 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/what-the-adapter-ablations-show/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/what-the-adapter-ablations-show/<run>/ <run>/
```

<!-- cluster:end -->
