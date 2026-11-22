# Does a camel in the rain come out wet?

Whether SDXL draws a wet camel and a wet dog when asked for one directly, set beside what plain
product-of-experts (PoE) draws for each animal with rain. The hypothesis test this folder serves is
written in the [dog-and-rain walk map](../../drips/a-dog-and-rain/EXECUTE_WALK.md#hypothesis-test-poe-draws-only-an-overlap-sdxl-saw).

## asked-directly.png

Five seeds down the page (1, 2, 4, 9, 10). Columns: "a wet camel", "a camel in the rain", then past
the dotted rule "a wet dog" and "a dog in the rain", each prompt rendered alone.

**How it was made**

`scripts/concept_pair_sheet.py --column single`, rendered by `scripts/concept_pair_sheet.sbatch`
(job 72389, 2026-10-20) on `mscluster66`, an RTX 3090, the same card model as the dog sheet's job
65969, on branch `exec/dog-and-rain`. 50 DDIM steps at guidance 7.5, each row starting from the
cached starting latent of `a_cat__x__a_dog` at that seed, the same latent the dog sheet's rows use.
The sheet was built on the laptop with `--direct-sheet` from the tiles copied into `tiles/single/`.
Its sidecar `asked-directly.json` lists every tile's cluster path under
`/datasets/mmolefe/poe_repair_min/outputs/concept_pair_sheet/single/` beside its local copy.

## a-camel-and-rain.png

The dog sheet's shape for the camel. Five seeds down the page. Columns: "a camel" alone, "rain"
alone, then the joint prompt "a camel and rain", plain PoE and SuperDiff.

**How it was made**

`scripts/concept_pair_sheet.py`, rendered by `scripts/concept_pair_sheet.sbatch` (job 72470,
2026-10-20) on `mscluster78`, an RTX 3090, on branch `exec/dog-and-rain`. The first four columns
use 50 DDIM steps at guidance 7.5 and start each row from the cached starting latent of
`a_cat__x__a_dog` at that seed. "rain" alone is the dog sheet's own tile, reused unchanged.
SuperDiff draws its own noise from the seed number and runs at its own 200 steps; it ran on the
3090 because all three Blackwell cards on `biggpu` were down, so its card differs from the dog
sheet's 200-step SuperDiff (job 65970, `mscluster110`). The sheet was built on the laptop with
`--sheet`; its sidecar `a-camel-and-rain.json` lists every tile's cluster path beside its local copy.

## Cross-references

- The mention of **the camel-in-the-rain results**, placed in the understanding-compositionality project, in [the project table for regrouping the core repo](../../../plans/split-into-idea-repos.md).

<!-- cluster:start -->

## On the cluster

The other 30 files of this result (0.0 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/3-understanding-compositionality/artifacts/results/a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `tiles` | 30 | 0.0 MB | [mono.png](tiles/a_camel__x__rain/seed01/mono.png), [poe.png](tiles/a_camel__x__rain/seed01/poe.png), [superdiff_200.png](tiles/a_camel__x__rain/seed01/superdiff_200.png), [mono.png](tiles/a_camel__x__rain/seed02/mono.png), [poe.png](tiles/a_camel__x__rain/seed02/poe.png), [superdiff_200.png](tiles/a_camel__x__rain/seed02/superdiff_200.png), [mono.png](tiles/a_camel__x__rain/seed04/mono.png), [poe.png](tiles/a_camel__x__rain/seed04/poe.png), [superdiff_200.png](tiles/a_camel__x__rain/seed04/superdiff_200.png), [mono.png](tiles/a_camel__x__rain/seed09/mono.png), [poe.png](tiles/a_camel__x__rain/seed09/poe.png), [superdiff_200.png](tiles/a_camel__x__rain/seed09/superdiff_200.png), [mono.png](tiles/a_camel__x__rain/seed10/mono.png), [poe.png](tiles/a_camel__x__rain/seed10/poe.png), [superdiff_200.png](tiles/a_camel__x__rain/seed10/superdiff_200.png), [seed01.png](tiles/single/a_camel/seed01.png), [seed02.png](tiles/single/a_camel/seed02.png), [seed04.png](tiles/single/a_camel/seed04.png), [seed09.png](tiles/single/a_camel/seed09.png), [seed10.png](tiles/single/a_camel/seed10.png), [seed01.png](tiles/single/a_camel_in_the_rain/seed01.png), [seed02.png](tiles/single/a_camel_in_the_rain/seed02.png), [seed04.png](tiles/single/a_camel_in_the_rain/seed04.png), [seed09.png](tiles/single/a_camel_in_the_rain/seed09.png), [seed10.png](tiles/single/a_camel_in_the_rain/seed10.png), [seed01.png](tiles/single/a_dog_in_the_rain/seed01.png), [seed02.png](tiles/single/a_dog_in_the_rain/seed02.png), [seed04.png](tiles/single/a_dog_in_the_rain/seed04.png), [seed09.png](tiles/single/a_dog_in_the_rain/seed09.png), [seed10.png](tiles/single/a_dog_in_the_rain/seed10.png), [seed01.png](tiles/single/a_wet_camel/seed01.png), [seed02.png](tiles/single/a_wet_camel/seed02.png), [seed04.png](tiles/single/a_wet_camel/seed04.png), [seed09.png](tiles/single/a_wet_camel/seed09.png), [seed10.png](tiles/single/a_wet_camel/seed10.png), [seed01.png](tiles/single/a_wet_dog/seed01.png), [seed02.png](tiles/single/a_wet_dog/seed02.png), [seed04.png](tiles/single/a_wet_dog/seed04.png), [seed09.png](tiles/single/a_wet_dog/seed09.png), [seed10.png](tiles/single/a_wet_dog/seed10.png), [seed01.png](tiles/single/rain/seed01.png), [seed02.png](tiles/single/rain/seed02.png), [seed04.png](tiles/single/rain/seed04.png), [seed09.png](tiles/single/rain/seed09.png), [seed10.png](tiles/single/rain/seed10.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/3-understanding-compositionality/artifacts/results/a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet/tiles/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/3-understanding-compositionality/artifacts/results/a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet/<run>/ <run>/
```

<!-- cluster:end -->
