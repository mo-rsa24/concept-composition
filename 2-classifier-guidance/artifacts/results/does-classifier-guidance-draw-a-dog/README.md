# does-classifier-guidance-draw-a-dog

**What this asks**

Does SDXL draw a dog from an empty prompt when a trained dog classifier's gradient pushes each sampling step? The recorded call is bar not met: a clean dog on 2 of 4 seeds, against a bar of 3 of 4 `stated`. The check before steering also failed, because balanced accuracy reached 0.75 on 7 of the first 20 sampler steps and the check needed 10 `stated`.

**What is here**

<a href="dog-01-from-an-animal/tiles/seed11__pushs30.png"><img src="dog-01-from-an-animal/tiles/seed11__pushs30.png" width="200" alt="A grey wolf-like animal face seen from the front, with orange eyes, on a pale background with orange stripes"></a>

This is seed 11 at strength 30, from the same push started at the prompt "an animal" instead of an empty prompt `stated`. Plan 18 calls it a clean wolf-like canine `stated`, and the picture shows one grey wolf-like face `verified`.

**How it was made**

The run is `dog-01` (W&B `z2k1bdmi`) `stated`. [The empty-prompt script](../../../scripts/classifier_guidance_from_the_paper.py) and [the product script](../../../scripts/classifier_guidance_on_the_product.py) drew the pushed sheets at strengths 10, 30, 100 and 300 `stated`. [Plan 18](../../../plans/18-a-dog-classifier-trained-the-papers-way.md) holds the design and every call.

**Where it is used**

[The report page on this question](../../../report/classifier-guided-poe/does-classifier-guidance-draw-a-dog.md) gives the calls seed by seed and the two places where its sources disagree.

<!-- cluster:start -->

## On the cluster

The other 102 files of this result (168.9 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-classifier-guidance-draw-a-dog/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `(files at this level)` | 5 | 10.6 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-classifier-guidance-draw-a-dog/` |
| `cat-piles` | 2 | 5.1 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-classifier-guidance-draw-a-dog/cat-piles/` |
| `dog-01-from-a-cat` | 30 | 44.4 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-classifier-guidance-draw-a-dog/dog-01-from-a-cat/` |
| `dog-01-from-an-animal` | 29 | 47.7 MB | [seed11__pushs30.png](dog-01-from-an-animal/tiles/seed11__pushs30.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-classifier-guidance-draw-a-dog/dog-01-from-an-animal/` |
| `dog-01-probes` | 7 | 11.9 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-classifier-guidance-draw-a-dog/dog-01-probes/` |
| `dog-01-sheet` | 29 | 49.1 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-classifier-guidance-draw-a-dog/dog-01-sheet/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-classifier-guidance-draw-a-dog/<run>/ <run>/
```

<!-- cluster:end -->
