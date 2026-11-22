# The paper's main grid: four pairs, four methods

**four-pairs-four-methods.png**

The figure the results section opens with, in `paper/overleaf-iclr/figures/`. Columns: the
joint prompt, the plain product, CO3, the product through the adapter. Rows: cat and dog
(seed 9), elephant and penguin (seed 10), tent and snowy mountain (seed 9), lighthouse and
stormy sea (seed 9). The first, second and fourth tiles of a row start from the same cached
noise; CO3 draws its own.

**How it was made**

`python scripts/paper_main_grid.py tiles tiles/co3 four-pairs-four-methods.png`, with `tiles/`
the copies kept here. The "ours" tile is picked per row, and the pick is the `ROWS` list in that
script: the cross+self adapter for cat and dog, v58-05 at step 15,000 (raw weights) for elephant
and penguin, the v58 self adapter for the two object-and-place rows. Every tile was rendered by
`scripts/figure_candidates.py` into the ablation grid on the cluster. The CO3 tiles were
rendered on 2026-09-25 by `scripts/showcase/run_co3_targets.sh` (Slurm job 59714) into
`outputs/showcase/co3_grid/`, seeds 9 and 10 as CO3's own seed index.

**What it lets you say**

On these four cells the adapter draws both concepts as separate things from the two concept
prompts and the null token alone; the plain product fuses or drops one; CO3 also composes all
four but keeps the joint prompt at inference and renders the elephant row as clip art.

**What it cannot say**

Four hand-picked cells from three adapters, so nothing about rates or about one checkpoint.
The held-out compose rate and the species read live in
`artifacts/results/can-a-score-see-species/` and the idea map at
`artifacts/ideas/one-training-recipe/IDEA_MAP.md`.

<!-- cluster:start -->

## On the cluster

The other 16 files of this result (22.5 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/how-four-methods-compare-on-four-pairs/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `(files at this level)` | 1 | 2.0 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/how-four-methods-compare-on-four-pairs/` |
| `tiles` | 15 | 20.4 MB | [mono.png](tiles/a_cat__x__a_dog/seed09/mono.png), [poe.png](tiles/a_lighthouse__x__a_stormy_sea/seed09/poe.png), [ours_v58self_w50.png](tiles/a_tent__x__a_snowy_mountain/seed09/ours_v58self_w50.png), [ours_v58-05-self-contrast_s015000_rawA_w50.png](tiles/an_elephant__x__a_penguin/seed10/ours_v58-05-self-contrast_s015000_rawA_w50.png), [a_lighthouse_and_a_stormy_sea.png](tiles/co3/a_lighthouse__x__a_stormy_sea/algo-co3-hybrid_/a_lighthouse_and_a_stormy_sea/seed-9/a_lighthouse_and_a_stormy_sea.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/how-four-methods-compare-on-four-pairs/tiles/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/how-four-methods-compare-on-four-pairs/<run>/ <run>/
```

<!-- cluster:end -->
