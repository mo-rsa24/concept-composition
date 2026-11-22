# Which adapter composes cat and dog

Every adapter this project has trained, rendering the same held-out cell, so they can be read
against each other rather than one run at a time.

## every-adapter-cat-and-dog-seed-09.png, -seed-11.png, -seed-12.png

**What they show**

Twenty-four tiles each: the no-adapter control, then twenty-three trained adapters, all rendering
`a_cat__x__a_dog` on one held-out seed. Each tile is labelled with the adapter, its rank and the
checkpoint step. Seeds 9, 11 and 12 were never trained on by any adapter here.

**What they say**

The seed decides whether a render is a photograph or a drawing, and nearly every adapter keeps its
seed's look. Seed 12 draws a photograph with no adapter at all and seed 11 draws a pencil sketch,
and the adapters follow. The two adapters trained against filtered renders ("target is a picture"
and "picture + cached empty") are the exceptions: both draw a photograph in a green field on all
three seeds. The looks each failure takes are named in the
[context entry on how a render fails](../../../context/world/how-a-render-fails.md).

**Where they came from**

`scripts/across_adapters.py --seed <n>`, one process per adapter because attaching a second LoRA
to the same U-Net stacks it on the first, then `--sheet` to assemble. Rendered 2026-09-21 on
mscluster109 (seed 9) and the batch nodes mscluster122 (seed 11) and mscluster140 (seed 12), fifty
DDIM steps, guidance 7.5, 1024 px. The list of adapters and the checkpoint each was read at is the
`ROWS` table at the top of that script.

## every-adapter-cat-and-dog-by-seed.png

**What it shows**

The same 72 tiles as the three per-seed sheets, on one page. Rows are seeds 9, 11 and 12. Columns
are the 24 adapters, grouped under a heading naming what the run changed: no adapter, the original
recipe's rank and regularisation, a different loss, which cells it trains on, what the target is,
and extra loss terms. The two filtered-picture adapters are outlined in red.

**What it says**

Along a row, most tiles keep the look the seed has with no adapter: a sketch on seed 11, a
photograph in dry grass on seed 12. The two outlined columns break that on every seed.

**How it was made**

`scripts/across_adapters_by_family.py`, from the tiles `scripts/across_adapters.py --render` wrote
under `/datasets/mmolefe/poe_repair_min/outputs/across_adapters_seed{09,11,12}/`, pulled to the
laptop. The column groups and plain labels are the `GROUPS` table in that script.

## every-adapter-cat-and-dog-by-seed-v2.png

**What it shows**

The paper version of the sheet above: no title or subtitle, and 10 adapter columns plus the
no-adapter control. Groups: the original recipe at ranks 8, 16 and 32, the loss on the
predicted picture, which denoising steps the correction trains on, and what the target is. The
step-window group names only the steps, all three at step 40,000. All 50 and first 25 share one
training pool of 43 cells (a cell is one pair at one seed) and the ordinary loss. The headings
carry no rank: the step-window runs are rank 32 and the target runs are rank 16. The first-10-steps, 40k column is `v57-09-x0-seeds` at
step 40,000, and it is not a like-for-like match: it trains with the loss on the predicted picture
and on the same pairs at every cached seed. The like-for-like run, `v57w10-01-base`, stops at
10,000 steps. The choice rests on the author's read that like-for-like training made no significant
difference; the comparison behind that read is not linked from this card yet.

**How it was made**

`scripts/across_adapters_by_family.py <tiles> <out.png> --v2`, the `GROUPS_V2` table. The
`v57-09-x0-seeds` tiles are row 25 of `scripts/across_adapters.py`, rendered 2026-09-26 on the batch
partition (job 60319).

## x0-seeds-against-x0-contrast-step-45000.png

**What it shows**

Four held-out cells as rows (cat and dog seeds 9 and 10, elephant and penguin seeds 9 and 10).
Columns: the joint-prompt render, plain product-of-experts, `v57-09-x0-seeds` (W&B `ek4w6lbu`) and
`v57-10-x0-contrast` (W&B `boammtzo`), both at step 45,000.

**What it says**

The run trained on every cached seed draws a sharp tabby cat and white dog on cat and dog seed 9,
close to the joint-prompt render, and a cat and dog on seed 10. The contrast run matches it
everywhere except cat and dog seed 9, where it invents two people. Elephant and penguin seed 9 is
messy in both. The joint-prompt render is itself wrong on two of the four rows: two dogs on cat and
dog seed 10, and an elephant with no penguin on elephant and penguin seed 9.

**Where it came from**

Cropped from the training runs' own step-45,000 preview sheets (`*__cmp.png`, top row) under
`/datasets/mmolefe/poe_repair_min/outputs/showcase/cohort2/<run>/samples/per_epoch/epoch_0900_step_045000/`.

## how-a-render-fails-gallery.png

**What it shows**

Six ways a render goes wrong, one panel each, with the telling patch enlarged in the corner of its
own panel and a red box on the render marking where that patch came from: a fused cat-dog face,
invented people, grey mush, flat line art, a third animal in the corner, and a horse standing where
the dog was asked for.

**What it says**

The looks named in [how a render fails](../../../context/world/how-a-render-fails.md) are visible
rather than asserted. Five of the eleven looks have no panel yet.

**How it was made**

`scripts/failure_gallery.py`, from renders already on the cluster: the per-adapter tiles of seeds 9
and 11 written by `scripts/across_adapters.py`, and two per-epoch samples of the `v57w10` runs. The
sidecar beside the image lists every panel's source file and crop box.

## how-a-render-fails-paper.png

**What it shows**

The same six panels as `how-a-render-fails-gallery.png`, laid out for the manuscript: panels and
their bold look labels only, no title, no cause or provenance lines, gaps closed. The paper's
caption carries the causes.

**How it was made**

`python scripts/failure_gallery.py --paper how-a-render-fails-gallery.png how-a-render-fails-paper.png`,
which crops the panels out of the rendered gallery, so it needs none of the source renders. A copy
sits at `paper/overleaf-iclr/figures/how-a-render-fails.png`.

<!-- cluster:start -->

## On the cluster

The other 9 files of this result (22.2 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/which-adapter-composes-cat-and-dog/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `(files at this level)` | 9 | 22.2 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/which-adapter-composes-cat-and-dog/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/which-adapter-composes-cat-and-dog/<run>/ <run>/
```

<!-- cluster:end -->
