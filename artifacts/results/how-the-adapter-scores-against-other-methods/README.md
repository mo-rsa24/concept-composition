# How the adapter scores against other methods

The renders and tables for slots R1 and R12: every SDXL method on the 276 Attend-and-Excite prompts.

## methods_grid.pdf (slot R12)

<a href="methods_grid.png"><img src="methods_grid.png" width="200" alt="Six benchmark prompts by five methods at seed 9: joint prompt, PoE, CO3, PoE plus Langevin, ours"></a>

**What it shows.** The renders behind R1, picked by protocol rule 5 from `scores/scores_long.csv`: per benchmark set, the cell where ours scores highest on BLIP-VQA and the one where it scores lowest at seeds 9 to 12 (animals: dog and frog at seed 10, 0.97, and horse and frog at seed 10, 0.00; animals with objects: frog and red suitcase at seed 12, 0.99, and dog with glasses at seed 9, 0.00; objects: yellow bow and pink bowl at seed 10, 0.98, and black backpack and green bow at seed 10, 0.00). Rows carry only the pair and seed; the rule and the values sit here. Columns are R1's methods in R1's order. PoE + Langevin runs on the animals set only. By eye, the lowest rows lose an object or merge the two; dog with glasses has no glasses. `seen`

**How it was made.** Picks by `scripts/review_rule5_picks.py`; tiles from `/datasets/mmolefe/poe_repair_min/outputs/paper_review/` (`r1_joint/`, `r1_poe/`, `co3_default/r1/`, `langevin_subset/`, `r1_ours/`), every row from its seed's shared latent, resized to 512 px into `tiles-rule5/` and drawn at 5.5 in by `scripts/paper_tile_figure.py` from `methods_grid-spec.json`, 2026-11-11. The PNG is a 110 dpi preview of the PDF.