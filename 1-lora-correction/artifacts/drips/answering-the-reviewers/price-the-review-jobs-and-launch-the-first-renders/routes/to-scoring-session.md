# For the scoring session (route step 9): what wave 1 leaves to score

From the PRZE-10 walk, 2026-11-11.

- Wave 1's renders are complete and counted on mscluster112: R1 joint, PoE and ours 1,104 each in `paper_review/r1_joint`, `r1_poe`, `r1_ours`; R3 ours 424 in `paper_review/r3_ours`; the early-steps test 848 in `paper_review/early_steps` (`_w10` and `_w25` tiles).
- CO3 at its published default on R1 finishes about 13:45 in `paper_review/co3_default/r1/` (1,045 of 1,104 at 13:08). CO3 on R3 has not started; the baselines session owns it.
- PoE + Langevin on protocol rule 6's subset (288 pictures: R1's animals set at seeds 9 to 12, R3's hard group at seeds 9 to 16) lands about 16:40 in `paper_review/langevin_subset/<pair>/seedNN/poe_langevin.png`, with rule 9's columns in each sidecar (`method` poe_langevin, `group` animals or hard, `setting` c=3,k=20).
- A preview of Tables 1 and 2 with joint, PoE and ours, built by `1-lora-correction/scripts/review_build_tables.py` on a copy of your score files at 12:10, is in `paper_review/preview_tables/2026-11-11-1210/out/`; your own score folder was not touched. The R1 bar there reads SUPPORTS on all three sets.
- Count outputs from mscluster112: mscluster108 holds a stale folder entry under `r1_poe` and undercounts by 4.
