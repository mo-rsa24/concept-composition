# does-a-pair-aware-push-compose-our-photos

**What this asks**

Can a classifier trained on our own cat and dog images push the product of a cat model and a dog model into drawing two separate animals, on more seeds than a push built from separate "cat present" and "dog present" answers? No verdict is recorded, because the runs that would judge it stopped in a storage outage `stated`.

**What is here**

<a href="imagenet-product-check-256.png"><img src="imagenet-product-check-256.png" width="200" alt="Four draws side by side, each showing one furry animal"></a>

These are four draws from the released 256 px model, pushed by the released classifier's cat and dog classes at scale 10 on seeds 1 to 4 `stated`. The walk called draws 2 and 4 cat-dog hybrids and draws 1 and 3 one dog-like animal, by eye at grid scale `stated`. No pair-aware push acted on these draws.

[The image counts](photo-set-counts.json) describe the training set: 550 images (200 cat, 200 dog, 150 both), of which 312 are SD3.5 renders and 238 are COCO photos `verified`.

**How it was made**

[Plan 20](../../../plans/20-our-photos-both-ways.md) holds the design, and [the walk on our photos](../../drips/our-photos-both-ways/EXECUTE_WALK.md) records the runs up to the outage.

**Where it is used**

[The report page on this question](../../../report/classifier-guided-poe/does-a-pair-aware-push-compose-our-photos.md) gives the bar and how far the work got.

<!-- cluster:start -->

## On the cluster

The other 5 files of this result (8.0 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-pair-aware-push-compose-our-photos/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `(files at this level)` | 5 | 8.0 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-pair-aware-push-compose-our-photos/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-pair-aware-push-compose-our-photos/<run>/ <run>/
```

<!-- cluster:end -->
