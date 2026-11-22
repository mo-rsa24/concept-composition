# What the product of two experts draws, and what samplers lose

All pictures and scripts here were made on 2026-10-29 in the 2D toy, on mscluster110, and copied into the repo because that machine's `/tmp` is not shared. The explanation of every column is in [the note on how the experiment works](../../notes/how-the-poe-toy-experiment-works/how-the-experiment-works.md).

## [The 14-column table, ideal experts and trained ResNets](interaction-table-ideal-and-trained.png)

**What it shows.** Five rows (same coordinates, separate coordinates, disjoint supports, a joint point added to both experts, partial overlap). Per row: the two datasets, then the exact product and four samplers for ideal experts, then the two learned experts, then the exact product and four samplers for trained experts. Sampler settings are Du et al.'s notebook settings for both groups.

**How we got it.** `table_trained.py` after `train_one.py` trained eight ResNets (15,001 steps each, 2271 s on CPU for the slowest). The four networks for the standard panels were trained again here because the repo's checkpoints are not on this machine.

**What it cannot say.** One data draw, one seed and 1500 draws per cell. No total variation numbers are computed for this table.

## [The same table with a sixth row, ring times bar](interaction-table-ideal-and-trained-six-rows.png)

**What it shows.** The five rows above plus Du et al.'s Figure 1 pair: eight Gaussians on a ring of radius 0.5 times a vertical bar of half-width 0.2. The exact product is two modes. In the ideal group reverse diffusion draws the whole ring and UHA draws two modes. In the trained group the learned ring has unequal mode heights, so the exact product of the trained experts is lopsided.

**How we got it.** Two more networks (`ring03`, `bar02`, 15,001 steps) and `table_trained.py` with `ROWS=5 TAG=_ring`; the row picture alone is [here](interaction-table-ideal-and-trained_ring.png).

**What it cannot say.** The ring's own spread is 0.03, smaller than the ideal expert's smoothing of 0.05, so in the ideal group the ring modes are wider (about 0.058) than the data's. Every other ideal row uses data std 0.056 so the expert's total is 0.075.

## Earlier pictures in this folder

**[The four-row interaction table](interaction-table.png)**: exact products only, for four support relations.

**[The same table with four samplers](interaction-table-with-samplers.png)**: ideal experts, older step sizes.

**[The log-scale overlap](log-overlap.png)**: the tails of the two experts do overlap at std 0.075 and 0.03.

**[The clean-share sweep](clean-share-sweep.png)**: the share of the product that is the joint draw against mode width and joint weight.

**[The sampler run](sampler-run.png)** and **[its numbers](sampler-results.json)**: three joint weights, five samplers, with joint share and total variation.

**[Twelve compositions with three backgrounds](interaction-table-backgrounds.png)**: ideal experts, older step sizes. The cluttered background makes the exact product jump to a corner where that background is faint.

**[Ten training samples](ten-samples.png)**: from the earlier circle and square toy.

## Other files

`hld/` holds the scene's spec, code copies, maths files and pictures. The published scene is at [the artifact link](https://claude.ai/artifact/CuxEQhwCRBiJ7ESwtdmwYw). `ckpt/` holds the eight trained networks and their loss logs.

<!-- cluster:start -->

## On the cluster

The other 48 files of this result (2.2 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/3-understanding-compositionality/artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `(files at this level)` | 27 | 2.2 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/3-understanding-compositionality/artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/` |
| `ckpt` | 20 | 0.1 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/3-understanding-compositionality/artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/ckpt/` |
| `hld` | 1 | 0.0 MB | [preview.png](hld/preview.png), [product-of-experts-toy-experiment-high-level-design-explainer.svg](hld/product-of-experts-toy-experiment-high-level-design-explainer.svg), [data.png](hld/img/data.png), [exact-product.png](hld/img/exact-product.png), [expert-levels.png](hld/img/expert-levels.png), [perfect-draws.png](hld/img/perfect-draws.png), [samplers-table.png](hld/img/samplers-table.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/3-understanding-compositionality/artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/hld/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/3-understanding-compositionality/artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/<run>/ <run>/
```

<!-- cluster:end -->
