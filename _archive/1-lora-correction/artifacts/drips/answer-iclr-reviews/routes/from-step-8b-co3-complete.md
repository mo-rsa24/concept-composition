# To the step 9 walk: CO3 is complete on R1 and R3

From the step 8b walk (the baselines), 2026-11-11 14:30.

- **CO3 at its published default has every cell.** 1,104 R1 tiles and 424 R3 tiles, each with a rule 9 sidecar carrying its counted `nfe` (260), counted from mscluster107. mscluster84's view of `/datasets` undercounted by 35, so count from another node.
- **Folders.** `/datasets/mmolefe/poe_repair_min/outputs/paper_review/co3_default/r1/<pair>/seedNN/co3.png` and `.../r3/...`; Figure 5's four pairs at seed 10 are under `.../r2/`.
- **Rows.** `1-lora-correction/scripts/review_score_rows.py --folders co3_default/r1 co3_default/r3` reads them (commit `2f838931`); wave 1's rows come out byte-identical.
- **Same noise.** CO3 starts from the seed latent the joint prompt, PoE and ours use, so its column carries no dagger for noise. Its guidance is CFG++ 0.8, its own scale.
