# Four pairs, one adapter: Figure 5 rebuilt (slot R2)

## figure5-seed10-five-columns.png

<a href="figure5-seed10-five-columns.png"><img src="figure5-seed10-five-columns.png" width="200" alt="Four pairs by five methods at seed 10: joint, PoE, CO3, PoE plus Langevin, ours"></a>

**What it shows.** Rows are cat and dog, elephant and penguin, tent and snowy mountain, lighthouse and stormy sea, all at seed 10; columns are R2's five: the joint prompt, PoE, CO3 at its published default, PoE + Langevin (c = 3, k = 20, all 50 levels) and ours (v58-03-self at step 30,000 on all 50 steps). By eye, CO3 and ours show both things on all four rows and PoE + Langevin keeps one of them on the first three. One seed. `seen`

**How it was made.** Assembled on 2026-11-11 from tiles already on `/datasets/mmolefe/poe_repair_min/outputs/`: joint and PoE from step 4's census (`review_step4/census/`) and wave 1's job 1 (`paper_review/figure5/`), ours from step 4's showcase renders, R3 ours and job 1, CO3 from `paper_review/co3_default/r6/` and PoE + Langevin from `paper_review/r6_correctors/` (the PRZE-2 walk). Every row starts from one latent. A review sheet only; the paper's figure is drawn by `scripts/paper_tile_figure.py`.

## figure5-layout-seed10-wave1.png

<a href="figure5-layout-seed10-wave1.png"><img src="figure5-layout-seed10-wave1.png" width="200" alt="Figure 5 in its earlier seven-column layout with wave 1's tiles and grey placeholders"></a>

**What it shows.** The same four rows in the seven-column layout R2 had in the morning of 2026-11-11, with wave 1's joint, PoE and ours tiles and grey placeholders for the baselines. Superseded by the five-column sheet above once ToMe and SuperDiff left the figure. `seen`

## figure5-standin-submitted-cells.png

<a href="figure5-standin-submitted-cells.png"><img src="figure5-standin-submitted-cells.png" width="200" alt="The submitted Figure 5 tiles placed in the rebuilt layout as a stand-in"></a>

**What it shows.** The submitted paper's own Figure 5 tiles (three adapters, seeds 9 and 10 picked per row, CO3 on its own noise) placed in the rebuilt layout, as a stand-in while wave 1 rendered. Not under the protocol. Built from `artifacts/results/four-pairs-four-methods/tiles/`. `seen`
