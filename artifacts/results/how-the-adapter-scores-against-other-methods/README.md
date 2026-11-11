# How the adapter scores against other methods

The renders and tables for slots R1 and R12: every SDXL method on the 276 Attend-and-Excite prompts.

## methods_grid.pdf (slot R12)

<a href="methods_grid.png"><img src="methods_grid.png" width="200" alt="Six benchmark prompts by five methods at seed 9: joint prompt, PoE, CO3, PoE plus Langevin, ours"></a>

**What it shows.** The renders behind R1: the first two prompts of each Attend-and-Excite set in the order of `scripts/review_benchmark_pairs.txt` (animals: elephant and rabbit, dog and frog; animals with objects: horse with glasses, bear with glasses; objects: pink crown and purple bow, blue clock and blue apple) at seed 9, by protocol rule 5. Columns are R1's methods in R1's order. PoE + Langevin runs on the animals set only, so its other four tiles are grey. By eye, its dog and frog tile is destroyed, and ours draws two horses for "horse with glasses" and a pink creature for "pink crown and purple bow". One seed. `seen`

**How it was made.** Tiles from `/datasets/mmolefe/poe_repair_min/outputs/paper_review/`: `r1_joint/`, `r1_poe/`, `co3_default/r1/`, `langevin_subset/` and `r1_ours/`, every row from its seed's shared latent; resized to 512 px into `tiles-r12/` and drawn at 5.5 in by `scripts/paper_tile_figure.py` from `methods_grid-spec.json`, 2026-11-11. The PNG is a 110 dpi preview of the PDF.
