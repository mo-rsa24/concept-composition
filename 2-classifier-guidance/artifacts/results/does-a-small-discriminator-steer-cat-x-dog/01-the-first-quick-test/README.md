# The first quick test of a small discriminator on cat and dog

This folder answers whether a small discriminator, trained on a few checked renders of a cat and a dog apart, can steer the product toward two separate animals on seeds it never saw. The verdict is on the [report page for the first quick test](../../../../report/classifier-guided-poe/a3q-cat-x-dog-quick-discriminator-test.md).

## What each run tried

Rows follow the time each run's first file was written `inferred`.

| Run | What changed | What came out | Picture |
|---|---|---|---|
| `training-set` | renders of a cat and a dog apart, each checked by eye, from seeds 1 to 8 (the joint prompt plus "a cat on the left and a dog on the right" and "a cat and a dog sitting apart"); negatives are the product's own renders from the same seeds and renders of a lone cat and a lone dog | 17 checked positives and 24 negatives from seeds 1 to 8 | <a href="training-set.png"><img src="training-set.png" width="120" alt="Sheet of 16 extra training draws on seeds 5 to 8 under two pair prompts, most showing a cat beside a dog"></a> |
| `discriminator` | the discriminator trained on the checked renders, then trained again with hybrids and wrong pairs among its negatives: 17 positives, then 81 positives over 38 benchmark pairs, with and without adapters on SDXL's trunk | first discriminator: held-out check 13 of 20 sampler steps. Every discriminator with hybrids and wrong pairs among its negatives failed: all 0 of 20 sampler steps at the 0.75 bar | <a href="discriminator.png"><img src="discriminator.png" width="120" alt="Held-out balanced accuracy against sampler step for four discriminators; only v1 rises above the 0.75 line, from step 7"></a> |
| `result-sheet` | one sheet, seeds 1 to 4 and 9 to 12 side by side, columns plain product, best-of-N at equal compute and guided, with the size-matched random push at 10, 30 and 100 | Draft reading 7, too weak to read: no tile on seeds 9 to 12 shows two separate animals, and the size cap clipped the push on every step at s = 100. Composed is 0 of 4 on the held-out seeds in every column, against 1 of 4 on the training seeds (seed 4, a drawing). The reader has not yet confirmed the 64 calls | none kept |
| `diagnostics` | discriminator logit against sampler step per kind of training tile | The discriminator did not read the concept. It scored pair-prompt renders positive whatever they showed, hybrids and two dogs included, and looked at one fixed patch of every picture | <a href="diagnostics.png"><img src="diagnostics.png" width="120" alt="Two panels of discriminator score against sampler step, one thick line per tile kind; positives, hybrids and wrong mixes rise above zero while trained lone animals and plain products fall below it"></a> |

<!-- cluster:start -->

## On the cluster

The other 192 files of this result (409.6 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/01-the-first-quick-test/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `diagnostics` | 7 | 9.4 MB | [diagnostics.png](diagnostics.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/01-the-first-quick-test/diagnostics/` |
| `discriminator` | 2 | 0.1 MB | [discriminator.png](discriminator.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/01-the-first-quick-test/discriminator/` |
| `result-sheet` | 83 | 189.9 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/01-the-first-quick-test/result-sheet/` |
| `training-set` | 100 | 210.1 MB | [training-set.png](training-set.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/01-the-first-quick-test/training-set/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/01-the-first-quick-test/<run>/ <run>/
```

<!-- cluster:end -->
