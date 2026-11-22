# Can a checker read a clean cat and dog pair

This folder answers whether a small checker can tell a clean cat and dog from a fused animal while the picture is still being drawn. The readings and their pass marks are on the [report page for reading a clean cat and dog](../../../../report/classifier-guided-poe/can-a-checker-read-a-clean-cat-and-dog.md).

## What each run tried

Rows follow the time each run's first file was written `inferred`.

| Run | What changed | What came out | Picture |
|---|---|---|---|
| `dataset-v3` | head only: 81 clean pictures over 38 pairs. Head and adapters: the same data, adapters on SDXL's trunk | head only: 0 of 20 steps, 0.53 to 0.63, failed. Head and adapters: 0 of 20 steps, 0.55 to 0.69, failed | none kept |
| `api-labels` | plan 23's 1,920 drawings labelled, then a readout checker built on them | no held-out reading is written in any plan, walk or card; no verdict recorded | none kept |
| `dataset-v4` | plan 23's 1,920 drawings labelled, then a readout checker built on them | no held-out reading is written in any plan, walk or card; no verdict recorded | <a href="dataset-v4.jpg"><img src="dataset-v4.jpg" width="120" alt="Labelled evaluation set in five rows: 13 clean pairs, 7 clean cat and dog, 35 hybrids, 13 wrong pairs and 6 singles"></a> |
| `result-sheet-v4c` | plan 23's 1,920 drawings labelled, then a readout checker built on them | no held-out reading is written in any plan, walk or card; no verdict recorded | <a href="result-sheet-v4c.png"><img src="result-sheet-v4c.png" width="120" alt="Brown dog facing a white fluffy animal with a dark strap across its muzzle"></a> |
| `result-sheet-v4c-early` | plan 23's 1,920 drawings labelled, then a readout checker built on them | no held-out reading is written in any plan, walk or card; no verdict recorded | <a href="result-sheet-v4c-early.png"><img src="result-sheet-v4c-early.png" width="120" alt="Black and white dog in front of a black cat, with a gold ring where the cat's eye should be"></a> |
| `result-sheet-catdog` | first version, 753 tiles | balanced accuracy 0.639; two bodies on 3 of seeds 9 to 16 (draft calls) | none kept |
| `result-sheet-catdog-coco` | COCO photos added | balanced accuracy 0.600, dropped; two bodies on 3 of seeds 9 to 16 (draft calls) | none kept |
| `result-sheet-catdog-more2` | 350 new joint drawings added, COCO left out | balanced accuracy 0.737, kept; two bodies on 5 of seeds 9 to 16 (draft calls) | <a href="result-sheet-catdog-more2.png"><img src="result-sheet-catdog-more2.png" width="120" alt="White animal with a collar and a patchy face sitting over a golden animal that lies with its face half hidden"></a> |
| `result-sheet-catdog-more3` | 500 more joint drawings | balanced accuracy 0.741, kept; two bodies on 4 of seeds 9 to 16 (draft calls) | <a href="result-sheet-catdog-more3.png"><img src="result-sheet-catdog-more3.png" width="120" alt="Cream animal with a part cat, part dog face and a pink tag, with a blurred golden dog behind it"></a> |
| `result-sheet-catdog-more3-hard` | half of each batch's rejects are wrong pairs | balanced accuracy 0.756, kept as the loop's result; two bodies on 3 of seeds 9 to 16 (draft calls) | none kept |
| `result-sheet-catdog-more3-long` | the same data, 2,000 training steps in place of 750 | balanced accuracy 0.755, kept; two bodies on about 7 of seeds 9 to 16 (draft calls) | <a href="result-sheet-catdog-more3-long.png"><img src="result-sheet-catdog-more3-long.png" width="120" alt="Small fluffy white and tan animal beside a large cream animal with a collar, the two apart"></a> |
| `finetune-misses` | networks fine-tuned on 1,603 finished pictures, read on the 220 held-out ones | DINOv2 ViT-B/14 0.846 and 0.847, CLIP ViT-L/14 0.865 to 0.900, CLIP ViT-bigG/14 0.873 and 0.909. Two seeds of CLIP ViT-L/14 differ by 0.035, as large as any change tried. This folder holds the misses | <a href="finetune-misses.png"><img src="finetune-misses.png" width="120" alt="Eleven pictures of a clear cat and dog labelled positive, each with its DINOv2 and CLIP score, every CLIP score below zero"></a> |
| `half-drawn-guesses` | CLIP trained on finished pictures, then trained on 4,966 guesses | balanced accuracy at steps 0, 5, 10, 15, 20, 30 and finished: 0.50, 0.50, 0.51, 0.58, 0.62, 0.81, 0.90 before; 0.51, 0.54, 0.72, 0.82, 0.84, 0.87, 0.90 after | <a href="half-drawn-guesses.png"><img src="half-drawn-guesses.png" width="120" alt="One held-out clean pair, seed 605, at seven sampler steps, with checker accuracy before and after 4 training epochs under each"></a> |
| `result-sheet-catdog-identity` | the 2,000-step checker plus "cat present" and "dog present" outputs `inferred` | balanced accuracy 0.721, kept as a push; two bodies on 5 of seeds 9 to 16 (draft calls) | <a href="result-sheet-catdog-identity.png"><img src="result-sheet-catdog-identity.png" width="120" alt="Plain unpushed render of seed 11: one line-drawn fused animal head on a grey background"></a> |
| `plan25-labels` | label sample for training on 1,972 labelled runs' own states, labelled by the 0.900 CLIP checker | mean balanced accuracy over steps 0 to 10 rose from 0.677 to 0.730, +0.053, against a mark of at least 0.75 and at least 0.05 above the earlier checker: in between | <a href="plan25-labels.png"><img src="plan25-labels.png" width="120" alt="Label sample in four rows of eight: two rows called clean show a cat beside a dog, and two rows of rejects show fused animals, drawings and a pattern"></a> |

<!-- cluster:start -->

## On the cluster

The other 722 files of this result (937.8 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/02-can-a-checker-read-a-clean-pair/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `api-labels` | 15 | 2.7 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/02-can-a-checker-read-a-clean-pair/api-labels/` |
| `dataset-v3` | 17 | 6.7 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/02-can-a-checker-read-a-clean-pair/dataset-v3/` |
| `dataset-v4` | 1 | 1.1 MB | [dataset-v4.jpg](dataset-v4.jpg) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/02-can-a-checker-read-a-clean-pair/dataset-v4/` |
| `finetune-misses` | 47 | 42.4 MB | [finetune-misses.png](finetune-misses.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/02-can-a-checker-read-a-clean-pair/finetune-misses/` |
| `half-drawn-guesses` | 8 | 0.4 MB | [half-drawn-guesses.png](half-drawn-guesses.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/02-can-a-checker-read-a-clean-pair/half-drawn-guesses/` |
| `result-sheet-catdog` | 76 | 107.7 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/02-can-a-checker-read-a-clean-pair/result-sheet-catdog/` |
| `result-sheet-catdog-coco` | 76 | 105.3 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/02-can-a-checker-read-a-clean-pair/result-sheet-catdog-coco/` |
| `result-sheet-catdog-identity` | 31 | 41.3 MB | [result-sheet-catdog-identity.png](result-sheet-catdog-identity.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/02-can-a-checker-read-a-clean-pair/result-sheet-catdog-identity/` |
| `result-sheet-catdog-more2` | 75 | 106.3 MB | [result-sheet-catdog-more2.png](result-sheet-catdog-more2.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/02-can-a-checker-read-a-clean-pair/result-sheet-catdog-more2/` |
| `result-sheet-catdog-more3` | 75 | 106.3 MB | [result-sheet-catdog-more3.png](result-sheet-catdog-more3.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/02-can-a-checker-read-a-clean-pair/result-sheet-catdog-more3/` |
| `result-sheet-catdog-more3-hard` | 76 | 106.3 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/02-can-a-checker-read-a-clean-pair/result-sheet-catdog-more3-hard/` |
| `result-sheet-catdog-more3-long` | 75 | 105.4 MB | [result-sheet-catdog-more3-long.png](result-sheet-catdog-more3-long.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/02-can-a-checker-read-a-clean-pair/result-sheet-catdog-more3-long/` |
| `result-sheet-v4c` | 75 | 102.4 MB | [result-sheet-v4c.png](result-sheet-v4c.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/02-can-a-checker-read-a-clean-pair/result-sheet-v4c/` |
| `result-sheet-v4c-early` | 75 | 103.5 MB | [result-sheet-v4c-early.png](result-sheet-v4c-early.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/02-can-a-checker-read-a-clean-pair/result-sheet-v4c-early/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/02-can-a-checker-read-a-clean-pair/<run>/ <run>/
```

<!-- cluster:end -->
