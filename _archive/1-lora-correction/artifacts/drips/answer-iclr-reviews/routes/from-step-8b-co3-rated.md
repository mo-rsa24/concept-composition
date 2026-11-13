# To the step 9 walk: CO3 is rated, ready to merge

From the step 8b walk, 2026-11-11 16:25, on the author's word.

- **Rows.** `scores/rows_co3.csv`: 1,528 rows (1,104 R1, 424 R3), built by `1-lora-correction/scripts/review_score_rows.py --folders co3_default/r1 co3_default/r3` from mscluster107.
- **Ratings,** by this walk's copies of your own scripts, unchanged: `clip_ir_co3.csv` (CLIP score and ImageReward on CPU, Slurm 85115, 1,528 rows), `blip_co3.csv` (BLIP-VQA on an RTX 3090, Slurm 85116), `comp_co3.csv` (the animal count on CPU, Slurm 85117, 176 animal rows). Same hardware as batches 1 to 3, per protocol rule's one-GPU-model line.
- **To do on your side:** merge the three into `scores_long.csv` and rebuild Tables 1 and 2 with `review_build_tables.py`. CO3's sidecars carry `method co3`, `setting default`, `nfe 260`.
- **PoE + Langevin's subset** finishes about 16:30; four helpers on mscluster107, 108 and 112 filled its tail into `langevin_subset/`.
