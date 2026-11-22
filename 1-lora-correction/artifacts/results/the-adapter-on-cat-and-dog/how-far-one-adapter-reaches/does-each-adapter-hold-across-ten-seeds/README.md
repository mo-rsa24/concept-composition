# Every adapter across ten seeds

Seven pairs, one grid each, so a method can be read across ten seeds rather than on one lucky cell.

## an-elephant-and-a-penguin.png, a-cat-and-a-dog.png, a-tiger-and-a-dog.png, a-cat-and-a-teapot.png, a-horse-and-a-bicycle.png, a-picnic-basket-and-a-watermelon.png, a-camel-and-a-forest.png

**What they show**

Rows are seeds 1, 2, 4 and 9 to 15. Columns are the joint prompt, the plain product, CO3, then
fourteen adapters to the right of the dotted rule: every run from v58 on except v58-04-freezenull,
plus v57-11b-x0-self-decay. The heading under each adapter names the checkpoint step. Pairs: three
animal pairs (elephant and penguin, cat and dog, tiger and dog), two animal-and-object pairs (cat
and teapot, horse and bicycle), two objects (picnic basket and watermelon), and an animal in a place
(camel and forest). None of the seven is in the v58 training pool. Seeds 1 to 8 were training seeds
for other pairs; seeds 9 to 16 were never trained on.

**How to read a row**

Every column except CO3 starts a row from the same cached noise (the `a_cat__x__a_dog` starting
latent at that seed, which the cache shares across pairs), so a difference along a row is the
method. CO3 draws its own noise from the same seed number and is not pixel-comparable with the rest.

**Which checkpoint each adapter is**

v58-03-self at 30k and v58-05-self-contrast at 15k are the steps the training-recipe notes name as
good; every other run is its last checkpoint. R1 was trained only on the part of the correction
that re-weighting the two experts cannot supply, so it renders with the experts weighted 3.0 and
3.0, the constant the render-time-fix finding used. The column list is `ADAPTERS` in
`scripts/seed_method_grid.py`.

**How they were made**

`scripts/seed_method_grid.sbatch`, rendered 2026-09-26: Slurm array job 60266 on `batch` (one
column per task, joint prompt resubmitted as 60280 after a folder race), two shared-device workers
on mscluster107 and mscluster109 device 1, and CO3 through `scripts/showcase/run_co3_targets.sh`
(job 60339 plus one job per pair, 60355 to 60358). Tiles are under
`/datasets/mmolefe/poe_repair_min/outputs/ablation_grid/<pair>/seed<NN>/`, CO3 under
`/datasets/mmolefe/poe_repair_min/outputs/showcase/co3_seed_grid/`. Grids assembled on the laptop
by `scripts/seed_method_grid.py <tiles> <co3> <out_dir>`; the JSON beside each image lists the
source path of every tile.

## five-picked-pairs.png

**What it shows**

One hand-picked seed per pair, five rows, no row labels: horse and bicycle at seed 2, camel and
forest at seed 12, cat and teapot at seed 10, tiger and dog at seed 13, picnic basket and watermelon
at seed 10. Columns are the joint prompt, the plain product and CO3, then one adapter right of the
dotted rule: reward KL K0 at 2,000 steps (`refl-kl-K0_s002000`), headed "Adapter". The tiles are the
same renders as the ten-seed grids above, so a row reads the same way. The row list is `PICKED` in
`scripts/seed_method_grid.py`, built with `--picked`; the sidecar names each row's pair and seed.

<!-- cluster:start -->

## On the cluster

The other 16 files of this result (177.6 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/how-far-one-adapter-reaches/does-each-adapter-hold-across-ten-seeds/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `(files at this level)` | 16 | 177.6 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/how-far-one-adapter-reaches/does-each-adapter-hold-across-ten-seeds/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/how-far-one-adapter-reaches/does-each-adapter-hold-across-ten-seeds/<run>/ <run>/
```

<!-- cluster:end -->
