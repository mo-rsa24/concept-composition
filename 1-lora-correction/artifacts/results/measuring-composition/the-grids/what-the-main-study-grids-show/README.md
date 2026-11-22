# What the main study's grids show

Comparison grids for the paper's main study, one image per question. Every tile is a real render,
and a row always starts from the same noise, so a difference across a row is the method rather
than the seed. Each grid has a sidecar JSON listing every tile's path, the run behind each column
and its W&B id.

The adapter column is `v58-03-self` at step 50,000 unless a sidecar says otherwise: the v58 pool
with the adapter on both cross-attention and self-attention, correction on all fifty steps.

## the-same-concept-twice.png

**What it shows**

Five cells naming one concept twice (dog and dog at seeds 1, 2 and 9, cat and cat at seed 9,
tiger and tiger at seed 9), under the joint prompt, the plain product and the adapter.

**What it says**

The plain product returns one animal in all five, and the adapter returns two in four of five.
Filed as [does the adapter return two when both prompts name the same
concept](../../../report/is-the-adapter-just-copying-the-joint-prompt/does-the-adapter-return-two-when-both-prompts-name-the-same-concept.md).

**How it was made**

`scripts/main_study_grids.py`, from renders by `scripts/figure_candidates.py` on the cluster.

## where-the-joint-prompt-itself-fails.png

**What it shows**

Six cells where the joint prompt drops a concept or adds one, under the same three columns.

**What it says**

The target is wrong on these cells: no hourglass, no cat, a person beside one dog. On the chess
cell at seed 10 the plain product holds both objects where the target holds one. Filed as
[does the joint prompt always draw both
concepts](../../../report/is-the-adapter-just-copying-the-joint-prompt/does-the-joint-prompt-always-draw-both-concepts.md).

**What it cannot say**

Five of its six adapter tiles are not rendered, so it cannot say whether the adapter beats the
target on those cells.

## the-fix-is-not-about-animals.png

**What it shows**

Six cells across three kinds of pair, two animals, two objects, and an object in a place, under
the same three columns.

**How it was made**

The same script and renders. No finding leans on it yet.

## what-the-adapter-costs-on-a-joint-prompt.png

**What it shows**

Four cells rendered twice from one noise: the joint prompt on the base model, and the same joint
prompt through the adapter. The adapter is never meant to touch a joint prompt, so a difference
here is damage outside what it was trained to repair.

**How it was made**

The same script, using the renderer's `monolora` column.

## published-models-then-ours.png

**What it shows**

Cat and dog, butterfly and flower meadow, and camel and forest at seed 42, through SD 1.4, SD 2.1,
SD 3.5 and SDXL on the joint prompt, SDXL's plain product, CO3, then a dotted rule and the
adapter.

**What it cannot say**

CO3 starts from its own noise, so its tile is not pixel-comparable with the rest of its row. Some
adapter tiles were still rendering when the grid was last built.

**How it was made**

The four-model renders come from `scripts/same_rule_four_models.py`, CO3 from the published
repository at `/home-mscluster/mmolefe/Playground/co3`, the rest from `figure_candidates.py`.

<!-- cluster:start -->

## On the cluster

The other 2 files of this result (0.0 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/measuring-composition/the-grids/what-the-main-study-grids-show/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `(files at this level)` | 2 | 0.0 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/measuring-composition/the-grids/what-the-main-study-grids-show/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/measuring-composition/the-grids/what-the-main-study-grids-show/<run>/ <run>/
```

<!-- cluster:end -->
