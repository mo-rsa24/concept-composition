# How the adapter scores against other methods

The renders and tables for slots R1 and R12: every SDXL method on the 276 Attend-and-Excite prompts.

## methods_grid.pdf (slot R12)

<a href="methods_grid.png"><img src="methods_grid.png" width="200" alt="Six benchmark prompts by five methods at seed 9: joint prompt, PoE, CO3, PoE plus Langevin, ours"></a>

**What it shows.** The renders behind R1, picked by protocol rule 5 from `scores/scores_long.csv`: per benchmark set, the cell where ours scores highest on BLIP-VQA and the one where it scores lowest at seeds 9 to 12 (animals: dog and frog at seed 10, 0.97, and horse and frog at seed 10, 0.00; animals with objects: frog and red suitcase at seed 12, 0.99, and dog with glasses at seed 9, 0.00; objects: yellow bow and pink bowl at seed 10, 0.98, and black backpack and green bow at seed 10, 0.00). Rows carry only the pair and seed; the rule and the values sit here. Columns are R1's methods in R1's order. PoE + Langevin runs on the animals set only. By eye, the lowest rows lose an object or merge the two; dog with glasses has no glasses. `seen`

**How it was made.** Picks by `scripts/review_rule5_picks.py`; tiles from `/datasets/mmolefe/poe_repair_min/outputs/paper_review/` (`r1_joint/`, `r1_poe/`, `co3_default/r1/`, `langevin_subset/`, `r1_ours/`), every row from its seed's shared latent, resized to 512 px into `tiles-rule5/` and drawn at 5.5 in by `scripts/paper_tile_figure.py` from `methods_grid-spec.json`, 2026-11-11. The PNG is a 110 dpi preview of the PDF.
## ours-best-cases.png (a review sheet, not a paper figure)

<a href="ours-best-cases.png"><img src="ours-best-cases.png" width="200" alt="Thirty-three cells of ours, its best seed on the top six pairs per held-out group and benchmark set by BLIP-VQA"></a>

**What it shows.** Ours (v58-03-self at step 30,000 on all 50 steps) at its best seed on the top six pairs of each held-out group (R3) and benchmark set (R1) by BLIP-VQA; the hard group holds three pairs. By eye, most show both things cleanly (giraffe and lion, mug and wine glass, dolphin and wave, monkey and orange suitcase, elephant and green balloon). Some high-scoring cells are still wrong: cat and lion draws a lion and its cub, bird and turtle has no clear bird, and cow and horse fails even at its best seed (0.01). So BLIP-VQA near 0.97 does not by itself mean two correct things. Picked for its best cases, so it says nothing about how often ours works; R3's rates answer that. `seen`

**How it was made.** One cell per pair, its highest-BLIP-VQA seed, from `scores/scores_long.csv` on the cluster; tiles from `paper_review/r3_ours/` and `r1_ours/`, laid out with matplotlib, 2026-11-11.

## best-cases-held-out.pdf, best-cases-benchmark.pdf, worst-cases-held-out.pdf, worst-cases-benchmark.pdf (appendix)

<a href="best-cases-held-out.png"><img src="best-cases-held-out.png" width="140" alt="Six held-out cells where ours does best, by joint prompt, PoE, CO3 and ours"></a> <a href="best-cases-benchmark.png"><img src="best-cases-benchmark.png" width="140" alt="Six benchmark cells where ours does best"></a> <a href="worst-cases-held-out.png"><img src="worst-cases-held-out.png" width="140" alt="Six held-out cells where ours does worst"></a> <a href="worst-cases-benchmark.png"><img src="worst-cases-benchmark.png" width="140" alt="Six benchmark cells where ours does worst"></a>

**What they show.** By protocol rule 5: per held-out group (R3) and benchmark set (R1), the two pairs where ours' best seed scores highest on BLIP-VQA, each at that seed, and the two where ours' worst seed scores lowest. Columns are the joint prompt, PoE, CO3 at its published default, PoE + Langevin and ours (v58-03-self at step 30,000 on all 50 steps). PoE + Langevin comes from rule 6's subset where a cell is in it, and from `paper_review/langevin_extra/` (the same call, c = 3 and k = 20) for the rest. By eye, it turns most cells into a flat poster or line drawing of one object. By eye, on the best cells ours draws both things where PoE often drops or fuses one (suitcase and desk fan, giraffe and lion, drum set and snowman); on the worst cells ours loses a thing or swaps a species (two cows for cow and horse, no glacier, no glasses), while CO3 keeps both on most of the same cells. The rows carry only pair and seed; the captions say how the cells were chosen. `seen`

**How they were made.** Picks from `scores/scores_long.csv` on the cluster (the rule above, 24 cells); tiles from `review_step4/census/`, `paper_review/r1_joint/`, `r1_poe/`, `co3_default/`, `r3_ours/` and `r1_ours/`, resized to 512 px into `tiles-best-worst/` and drawn at 5.5 in by `scripts/paper_tile_figure.py` from each `*-spec.json`, 2026-11-11.
