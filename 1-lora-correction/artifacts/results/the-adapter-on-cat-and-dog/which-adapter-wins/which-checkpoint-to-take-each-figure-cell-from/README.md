# Which checkpoint to take each figure cell from

Running `scripts/local/sweep_watch.sh` again downloads this result's tiles back to the laptop.

<!-- cluster:start -->

## On the cluster

The other 6,052 files of this result (7.24 GB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/which-checkpoint-to-take-each-figure-cell-from/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `history` | 2,896 | 4.74 GB | [which-checkpoint-v58-01-plain.png](history/20260924-1455/which-checkpoint-v58-01-plain.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/which-checkpoint-to-take-each-figure-cell-from/history/` |
| `latest` | 70 | 108.6 MB | [which-checkpoint-refl-03-self-contrast-15k_seeds_13_15.png](latest/which-checkpoint-refl-03-self-contrast-15k_seeds_13_15.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/which-checkpoint-to-take-each-figure-cell-from/latest/` |
| `tiles` | 3,086 | 2.39 GB | [ours_v58self_w50.png](tiles/a_bicycle__x__a_brick_wall/seed12/ours_v58self_w50.png), [ours_v58-05-self-contrast_s010000_w50.png](tiles/a_canoe__x__a_misty_lake/seed42/ours_v58-05-self-contrast_s010000_w50.png), [ours_v58-04-freezenull_s010000_w50.png](tiles/a_cat__x__a_cat/seed05/ours_v58-04-freezenull_s010000_w50.png), [ours_v58-02-contrast_s010000_w50.png](tiles/a_dog__x__a_dog/seed15/ours_v58-02-contrast_s010000_w50.png), [ours_v58-02-contrast_s020000_w50.png](tiles/a_leopard__x__a_cheetah/seed05/ours_v58-02-contrast_s020000_w50.png), [ours_v58-05-self-contrast_s010000_w50.png](tiles/a_lighthouse__x__a_stormy_sea/seed05/ours_v58-05-self-contrast_s010000_w50.png), [ours_crossself_w50.png](tiles/a_picnic_basket__x__a_watermelon/seed12/ours_crossself_w50.png), [ours_v58-03-self_s030000_w50.png](tiles/a_tent__x__a_snowy_mountain/seed42/ours_v58-03-self_s030000_w50.png), [ours_v58self_w50.png](tiles/a_tiger__x__a_tiger/seed09/ours_v58self_w50.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/which-checkpoint-to-take-each-figure-cell-from/tiles/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/which-checkpoint-to-take-each-figure-cell-from/<run>/ <run>/
```

<!-- cluster:end -->
