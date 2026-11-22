# What co3 does on our pairs

Running `scripts/local/reap_co3.sh` again downloads this result's tiles back to the laptop.

<!-- cluster:start -->

## On the cluster

The other 14 files of this result (19.5 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/what-co3-does-on-our-pairs/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `(files at this level)` | 1 | 3.4 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/what-co3-does-on-our-pairs/` |
| `tiles` | 13 | 16.1 MB | [a_bison_and_a_horse.png](tiles/a_bison_and_a_horse/seed-42/a_bison_and_a_horse.png), [a_bobcat_and_a_quail.png](tiles/a_bobcat_and_a_quail/seed-42/a_bobcat_and_a_quail.png), [a_camel_and_a_llama.png](tiles/a_camel_and_a_llama/seed-2/a_camel_and_a_llama.png), [a_cow_and_a_chicken.png](tiles/a_cow_and_a_chicken/seed-2/a_cow_and_a_chicken.png), [a_cow_and_a_milk_churn.png](tiles/a_cow_and_a_milk_churn/seed-9/a_cow_and_a_milk_churn.png), [a_goat_and_a_sheep.png](tiles/a_goat_and_a_sheep/seed-2/a_goat_and_a_sheep.png), [a_leopard_and_a_cheetah.png](tiles/a_leopard_and_a_cheetah/seed-42/a_leopard_and_a_cheetah.png), [a_rabbit_and_a_hare.png](tiles/a_rabbit_and_a_hare/seed-42/a_rabbit_and_a_hare.png), [a_soccer_ball_and_a_park_bench.png](tiles/a_soccer_ball_and_a_park_bench/seed-9/a_soccer_ball_and_a_park_bench.png), [a_suitcase_and_a_globe.png](tiles/a_suitcase_and_a_globe/seed-11/a_suitcase_and_a_globe.png), [a_tiger_and_a_parrot.png](tiles/a_tiger_and_a_parrot/seed-2/a_tiger_and_a_parrot.png), [a_wolf_and_a_coyote.png](tiles/a_wolf_and_a_coyote/seed-42/a_wolf_and_a_coyote.png), [a_wolf_and_a_raven.png](tiles/a_wolf_and_a_raven/seed-9/a_wolf_and_a_raven.png), [a_zebra_and_a_horse.png](tiles/a_zebra_and_a_horse/seed-2/a_zebra_and_a_horse.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/what-co3-does-on-our-pairs/tiles/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/what-co3-does-on-our-pairs/<run>/ <run>/
```

<!-- cluster:end -->
