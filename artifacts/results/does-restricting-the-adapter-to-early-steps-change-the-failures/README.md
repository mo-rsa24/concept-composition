# Does restricting the adapter to early steps change the failures?

The renders behind slots R9 and R10, from wave 1's early-steps job (report/paper-review/prze-10-compute.md, job 2): the adapter of record, v58-03-self at step 30,000, switched on for the first 10 or the first 25 of 50 steps and off after, on the 53 held-out pairs at seeds 9 to 16, 848 pictures in `/datasets/mmolefe/poe_repair_min/outputs/paper_review/early_steps/`. The all-50-steps column is R3's renders in `paper_review/r3_ours/`.

## early-steps-seed10-sheet.png

<a href="early-steps-seed10-sheet.png"><img src="early-steps-seed10-sheet.png" width="200" alt="Six held-out pairs at seed 10: joint, PoE, adapter on steps 0 to 9, 0 to 24 and all 50"></a>

**What it shows.** Two pairs from each held-out group at seed 10. By eye, the 25-step window separated the two things more often than the 10-step window (seal and walrus, cat and horse, bathtub and streetlamp), and on cow and horse every window drew two cows and no horse. One seed; R9's failure counts over 8 seeds decide. `seen`

**How it was made.** Drawn on mscluster84 on 2026-11-11 from the tiles named above, joint and PoE from step 4's census. Two cells read "rendering" because the session node's view of `/datasets` lagged the launch node; the tiles exist.

## cow-horse-seeds-9-12-first-look.png

<a href="cow-horse-seeds-9-12-first-look.png"><img src="cow-horse-seeds-9-12-first-look.png" width="200" alt="Cow and horse at seeds 9 to 12: joint, PoE, adapter on steps 0 to 9"></a>

**What it shows.** The hard pair cow and horse at seeds 9 to 12: PoE draws one animal on every seed, the 10-step adapter draws two on all four, and none of the four shows a clear horse. `seen`

**How it was made.** The first 10-step tiles of the early-steps job beside step 4's census joint and PoE, drawn on mscluster84 on 2026-11-11.

## early_steps_grid.pdf (slot R10)

<a href="early_steps_grid.png"><img src="early_steps_grid.png" width="200" alt="Six held-out rows by the LoRA's window: off, steps 0 to 10, 0 to 25, all 50, and a grey column for the all-steps-trained LoRA"></a>

**What it shows.** The early-steps renders on R4's rows (rule 5: per held-out group, ours' highest- and lowest-BLIP-VQA cell). Columns are the LoRA off (plain PoE), v58-03-self at step 30,000 on steps 0 to 10, on steps 0 to 25 and on all 50 steps; the fifth, a LoRA trained on all steps, is grey because gated training job 12 has not run. By eye, steps 0 to 10 already carry most of the change from PoE to the all-steps picture, on the working rows and the failing rows alike. `seen`

**How it was made.** Window tiles from `/datasets/mmolefe/poe_repair_min/outputs/paper_review/early_steps/` (wave 1's job 2), PoE from `review_step4/census/`, all-steps from `paper_review/r3_ours/`; resized to 512 px into `tiles-rule5/` and drawn at 5.5 in by `scripts/paper_tile_figure.py` from `early_steps_grid-spec.json`, 2026-11-11. The PNG is a 110 dpi preview of the PDF.