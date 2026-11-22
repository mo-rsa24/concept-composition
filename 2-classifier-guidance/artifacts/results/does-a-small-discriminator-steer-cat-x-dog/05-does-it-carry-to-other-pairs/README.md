# Does the early push carry to other pairs

This folder answers whether the early push, built on cat and dog, also separates other pairs. The pair-by-pair verdicts are on the [report page for the other pairs](../../../../report/classifier-guided-poe/does-the-early-push-carry-to-other-pairs.md).

## What each run tried

Rows follow the time each run's first file was written `inferred`.

| Run | What changed | What came out | Picture |
|---|---|---|---|
| `result-sheet-lion-meerkat` | the cat and dog checker's early push on lion and meerkat, seeds 3301 to 3316 | both separate on 6 of 16 (plain product 1 of 16, random push 1 of 16), clean 3 of 16: in between. The push drew two animals on 10 of 16 seeds, and 4 of those were two meerkats | <a href="result-sheet-lion-meerkat.png"><img src="result-sheet-lion-meerkat.png" width="120" alt="Seed 3314 at strength 30: a meerkat standing in front of a maned lion seen from behind"></a> |
| `result-sheet-typewriter-cactus` | the cat and dog checker's early push on typewriter and cactus, seeds 3301 to 3316 | both separate on 14 of 16 (plain product 1 of 16, random push 1 of 16), clean 6 of 16: support | <a href="result-sheet-typewriter-cactus.png"><img src="result-sheet-typewriter-cactus.png" width="120" alt="Seed 3302 at strength 30: a tall potted cactus beside a typewriter with small cacti growing from its top"></a> |
| `scorer-check` | not stated on the report page | plain product, lion and meerkat: 3 of 16 with two bodies, against 2 on the `multipair` sheet for byte-identical tiles. Seed 3314 got a different answer, so the labeller itself varies between runs | none kept |
| `result-sheet-lion-meerkat-multipair` | trained on cat and dog plus the labelled benchmark pictures, cap 0.3 | two bodies 12 of 16, faithful pairs 0 of 16; by eye a lion beside a meerkat on about 4 seeds | <a href="result-sheet-lion-meerkat-multipair.png"><img src="result-sheet-lion-meerkat-multipair.png" width="120" alt="Seed 3312 at strength 30: a small meerkat facing a large animal with a meerkat face and a lion-like mane"></a> |
| `result-sheet-lion-meerkat-multipair2` | 78 more pairs' joint-prompt pictures added, cap 0.3 `inferred` | two bodies 11 of 16, faithful pairs 0 of 16; by eye 1 seed | <a href="result-sheet-lion-meerkat-multipair2.png"><img src="result-sheet-lion-meerkat-multipair2.png" width="120" alt="Seed 3302 at strength 30: one animal lying down with a lion mane and a meerkat face"></a> |
| `result-sheet-lion-meerkat-multipair-cap0.4` | the same checker, cap 0.4 | two bodies 15 of 16, faithful pairs 0 of 16; by eye about 4 seeds, the rest mostly two meerkats | <a href="result-sheet-lion-meerkat-multipair-cap0.4.png"><img src="result-sheet-lion-meerkat-multipair-cap0.4.png" width="120" alt="Seed 3302 at strength 30: a large fused animal with a meerkat face and a mane beside a small meerkat"></a> |
| `result-sheet-lion-meerkat-multipair-presence` | "first animal present" and "second animal present" outputs, cap 0.4 | two bodies 11 of 16, faithful pairs 0 of 16; by eye pictures break into posters and collages | <a href="result-sheet-lion-meerkat-multipair-presence.png"><img src="result-sheet-lion-meerkat-multipair-presence.png" width="120" alt="Seed 3301 at strength 30: a small meerkat beside a larger meerkat-faced animal with a shaggy mane"></a> |
| `result-sheet-lion-meerkat-multipair-long` | trained twice as long, cap 0.3 | two bodies 11 of 16, faithful pairs 0 of 16; by eye none recorded | <a href="result-sheet-lion-meerkat-multipair-long.png"><img src="result-sheet-lion-meerkat-multipair-long.png" width="120" alt="Seed 3303 at strength 30: two meerkats on a stone ledge and no lion"></a> |

<!-- cluster:start -->

## On the cluster

The other 328 files of this result (474.2 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/05-does-it-carry-to-other-pairs/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `result-sheet-lion-meerkat` | 52 | 79.3 MB | [result-sheet-lion-meerkat.png](result-sheet-lion-meerkat.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/05-does-it-carry-to-other-pairs/result-sheet-lion-meerkat/` |
| `result-sheet-lion-meerkat-multipair` | 51 | 81.5 MB | [result-sheet-lion-meerkat-multipair.png](result-sheet-lion-meerkat-multipair.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/05-does-it-carry-to-other-pairs/result-sheet-lion-meerkat-multipair/` |
| `result-sheet-lion-meerkat-multipair-cap0.4` | 34 | 50.5 MB | [result-sheet-lion-meerkat-multipair-cap0.4.png](result-sheet-lion-meerkat-multipair-cap0.4.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/05-does-it-carry-to-other-pairs/result-sheet-lion-meerkat-multipair-cap0.4/` |
| `result-sheet-lion-meerkat-multipair-long` | 33 | 49.0 MB | [result-sheet-lion-meerkat-multipair-long.png](result-sheet-lion-meerkat-multipair-long.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/05-does-it-carry-to-other-pairs/result-sheet-lion-meerkat-multipair-long/` |
| `result-sheet-lion-meerkat-multipair-presence` | 50 | 75.8 MB | [result-sheet-lion-meerkat-multipair-presence.png](result-sheet-lion-meerkat-multipair-presence.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/05-does-it-carry-to-other-pairs/result-sheet-lion-meerkat-multipair-presence/` |
| `result-sheet-lion-meerkat-multipair2` | 50 | 75.1 MB | [result-sheet-lion-meerkat-multipair2.png](result-sheet-lion-meerkat-multipair2.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/05-does-it-carry-to-other-pairs/result-sheet-lion-meerkat-multipair2/` |
| `result-sheet-typewriter-cactus` | 52 | 62.9 MB | [result-sheet-typewriter-cactus.png](result-sheet-typewriter-cactus.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/05-does-it-carry-to-other-pairs/result-sheet-typewriter-cactus/` |
| `scorer-check` | 6 | 0.1 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/05-does-it-carry-to-other-pairs/scorer-check/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/05-does-it-carry-to-other-pairs/<run>/ <run>/
```

<!-- cluster:end -->
