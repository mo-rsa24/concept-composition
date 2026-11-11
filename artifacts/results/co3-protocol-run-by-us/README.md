# Every SDXL method side by side on the held-out pairs (slot R4)

## methods_side_by_side.pdf

<a href="methods_side_by_side.png"><img src="methods_side_by_side.png" width="200" alt="Six rows of held-out pairs by five methods: joint prompt, PoE, CO3, PoE plus Langevin, ours"></a>

**What it shows.** The renders behind R3: the first pair of each held-out group in the list order of `report/paper-review/held-out-set.md` (cow and horse, hard; baboon and gazelle, easy but unseen; bookcase and glacier, not competing) at seeds 9 and 10, by protocol rule 5. Columns are the joint prompt, PoE, CO3 at its published default, PoE + Langevin and ours (v58-03-self at step 30,000 on all 50 steps). The Langevin column is grey: its subset covers the hard group only, and those tiles are still rendering. `seen`

**How it was made.** Tiles from `/datasets/mmolefe/poe_repair_min/outputs/`: joint and PoE from step 4's census (`review_step4/census/`), CO3 from `paper_review/co3_default/r3/`, ours from `paper_review/r3_ours/`; every row starts from its seed's shared latent. Resized to 512 px into `tiles-r4/` and drawn at 5.5 in by `scripts/paper_tile_figure.py` from `methods_side_by_side-spec.json`, 2026-11-11. The PNG is a 110 dpi preview of the PDF.
