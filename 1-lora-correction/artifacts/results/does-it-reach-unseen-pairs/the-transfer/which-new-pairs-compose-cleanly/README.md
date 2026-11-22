# Which new pairs compose cleanly?

An audition of candidate pairs the adapter never trained on: one row per pair in
the audition sheet (`pair-audition.png`, not linked because it failed the subject check) (1,484 × 22,368 pixels), with each pair's tiles in
`tiles/<pair>/`, for example `a_lion__x__a_hyena` and `a_microscope__x__a_test_tube`. Stated from
the file names; which pairs passed is not recorded beside the sheet.

Running `scripts/local/conveyor.sh` again downloads this result's tiles back to the laptop.

<!-- cluster:start -->

## On the cluster

The other 291 files of this result (293.5 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-transfer/which-new-pairs-compose-cleanly/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `(files at this level)` | 1 | 30.9 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-transfer/which-new-pairs-compose-cleanly/` |
| `tiles` | 290 | 262.6 MB | [mono.png](tiles/a_badger__x__a_magpie/seed11/mono.png), [mono.png](tiles/a_bear__x__a_salmon/seed09/mono.png), [mono.png](tiles/a_bicycle__x__a_birdhouse/seed42/mono.png), [mono.png](tiles/a_camel__x__a_pelican/seed02/mono.png), [mono.png](tiles/a_cow__x__a_milk_churn/seed09/mono.png), [mono.png](tiles/a_deer__x__a_woodpecker/seed02/mono.png), [mono.png](tiles/a_duck__x__a_goose/seed42/mono.png), [mono.png](tiles/a_goat__x__a_sheep/seed02/mono.png), [mono.png](tiles/a_guitar__x__a_traffic_cone/seed42/mono.png), [mono.png](tiles/a_hedgehog__x__a_robin/seed09/mono.png), [mono.png](tiles/a_horse__x__a_water_trough/seed02/mono.png), [mono.png](tiles/a_microscope__x__a_test_tube/seed42/mono.png), [mono.png](tiles/a_pony__x__a_red_barn/seed02/mono.png), [mono.png](tiles/a_reindeer__x__a_sled/seed02/mono.png), [mono.png](tiles/a_soccer_ball__x__a_park_bench/seed02/mono.png), [mono.png](tiles/a_suitcase__x__a_globe/seed42/mono.png), [mono.png](tiles/a_swan__x__a_heron/seed42/mono.png), [mono.png](tiles/a_teddy_bear__x__a_birthday_cake/seed42/mono.png), [mono.png](tiles/a_tiger__x__a_parrot/seed02/mono.png), [mono.png](tiles/a_windmill__x__a_tractor/seed42/mono.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-transfer/which-new-pairs-compose-cleanly/tiles/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-transfer/which-new-pairs-compose-cleanly/<run>/ <run>/
```

<!-- cluster:end -->
