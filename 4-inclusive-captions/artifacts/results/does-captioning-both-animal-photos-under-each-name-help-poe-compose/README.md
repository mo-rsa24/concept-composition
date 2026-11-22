# Does captioning both-animal photos under each name help PoE compose?

Not on the seeds looked at so far. Both fine-tunes still draw one animal in every product picture.
What run B changes is which animal: mostly a clean single cat or dog, where run A gives fused
cat-dogs. The planned test (PoE compose rate on 128 seeds per run, read against the pilot's
pre-written criterion) has not been run, so this is a sheet and not a verdict.

## poe-cat-x-dog-plain-run-a-run-b-six-seeds.png

<a href="poe-cat-x-dog-plain-run-a-run-b-six-seeds.png"><img src="poe-cat-x-dog-plain-run-a-run-b-six-seeds.png" width="600" alt="SDXL PoE of a cat and a dog on seeds 0, 1, 3, 5, 6 and 7: plain SDXL, run A, run B; one animal in every picture, fused in the middle row and mostly clean single animals in the bottom row"></a>

PoE of "a cat" × "a dog" on SDXL, one column per seed (0, 1, 3, 5, 6, 7), same starting noise
down each column.

| Row | What was trained |
|---|---|
| plain SDXL | nothing |
| run A | LoRA on 550 photos (200 cat, 200 dog, 150 both); the 150 both-animal photos captioned "a cat and a dog" 90%, blank 10% |
| run B | same photos; each draw of a both-animal photo captioned "cat" 40%, "dog" 40%, "a cat and a dog" 10%, blank 10% |

**Read by eye:** 0 of 6 seeds show two animals in any row. Run A's row has several fused
cat-dogs; run B's row is mostly a single clean cat or dog. `verified` by looking at the sheet.

**How it was made.** `plain_figures.py`'s `fig_pilot()` in
`3-understanding-compositionality/artifacts/notes/what-poe-loses-when-it-composes/` tiles the
full-size product renders `seed_NNNN_product.png` from three folders, `plain`, `v4-run-A` and
`v4-run-B`, under a pilot output directory passed on the command line. Those renders were made by
`poe_repair/experiments/inclusive_captions/product_renders.py` on the cluster, under
`/datasets/mmolefe/poe_repair_min/outputs/inclusive_captions/`; they are not on the laptop. The
same picture sits in the toy's results as
`3-understanding-compositionality/artifacts/results/the-exact-toy/how-much-co-occurrence-the-product-needs/plain-figures/10-sdxl-inclusive-captions-pilot.png`,
where it closes the toy's argument; this copy is the pilot's own record of it.

**What the walk recorded at each checkpoint** (eight sheet seeds, by eye, from
[the pilot walk](../../drips/inclusive-captions-pilot/EXECUTE_WALK.md)):

| Checkpoint | Product: two animals in frame | Joint prompt |
|---|---|---|
| run A, step 0 | 0 of 8 | one cat beside one dog on 8 of 8 |
| run A, step 500 (rank 32 and rank 16 alike) | 0 of 8 | drifts: two cats on 2 to 3 seeds, a third animal on 2 |
| run B, step 500 | 0 of 8, close to run A seed for seed | one cat and one dog where run A drew a third cat on seed 7 |
| run A, step 2,000 | 0 of 8, three read as a fused head | recovers: one cat beside one dog on 8 of 8 |

**What it cannot say.** Six seeds, called by eye. No compose rate exists yet for either run. Run
A's first 128 scoring renders were discarded at 52 seeds after the contrast bug in both renderers
was found, and run B's 128 were never started. Every sheet before that fix was rendered half a
stop light. That squeeze applies equally to every sheet, so calls about composition across sheets
stand.
