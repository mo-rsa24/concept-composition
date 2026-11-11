# History of the paper-review board

## Drawings

| Date | Step | open | todo | in progress | done | issue | Picture |
|---|---|---|---|---|---|---|---|
| 2026-11-10 | 0 | 11 | 0 | 0 | 0 | 0 | [board](board-2026-11-10-step-0.png) |
| 2026-11-10 | 0 | 12 | 0 | 0 | 0 | 0 | [board](board-2026-11-10-step-0.png) |
| 2026-11-10 | 0 | 0 | 11 | 1 | 0 | 0 | [board](board-2026-11-10-step-0.png) |
| 2026-11-10 | 1 | 0 | 10 | 2 | 0 | 0 | [board](board-2026-11-10-step-1.png) |
| 2026-11-10 | 1 | 0 | 9 | 3 | 0 | 0 | [board](board-2026-11-10-step-1.png) |
| 2026-11-10 | 2 | 0 | 7 | 5 | 0 | 0 | [board](board-2026-11-10-step-2.png) |
| 2026-11-10 | 4 | 0 | 6 | 6 | 0 | 0 | [board](board-2026-11-10-step-4.png) |
| 2026-11-11 | 8 | 0 | 5 | 7 | 0 | 0 | [board](board-2026-11-11-step-8.png) |
| 2026-11-11 | 8 | 0 | 5 | 7 | 0 | 0 | [board](board-2026-11-11-step-8.png) |

## Row changes

| Date | Row | Change | Why |
|---|---|---|---|
| 2026-11-10 | PRZE-3 | split into PRZE-3a and PRZE-3b | the pair lists and the size of the held-out set are separate problems with separate checks |
| 2026-11-10 | PRZE-8, PRZE-9, PRZE-10 | added | a comparison protocol, figure and table slots that refill, and a compute plan were asked for before any run |
| 2026-11-10 | PRZE-11 | added | UHz6's early-steps question uses the same compute and the same adapter |

## Protocol changes

| Date | Rule | Change | Why |
|---|---|---|---|
| 2026-11-10 | 2, 3 | ToMe added on SDXL; SynGen, Divide and Bind and InitNO added on SD 1.5; FK steering moved from the main table to the CO3-protocol table as a noise-search control | the CO3 paper's Table 1 compares against these methods, and FK steering tests noise search rather than binding; changed before any evaluation run |
| 2026-11-10 | 4, 10 | the main table, R1, scores the 276 Attend-and-Excite prompts at seeds 9 to 12; the 53 held-out census pairs move to an ablation, R3, at seeds 9 to 16; frog and mouse allowed for those 45 benchmark prompts only (`artifacts/_shared/cross_pair_pool_configs/pair_ledger.md`) | the same prompts and metrics as the CO3 paper's Table 1 let the two tables stand side by side; decided before any method was run on them |
| 2026-11-11 | 3 | the Langevin corrector runs once at c = 3, k = 20, with no tuning grid; unadjusted HMC is not a baseline | PrzE asked for one MCMC corrector baseline, not a tuned grid; your call (`fix langevin only`, `fix no grid`), to keep to what the reviewer asked and save about 3.3 GPU-hours; decided before any corrector image of the review |
| 2026-11-11 | 2, 3 | ToMe dropped from SDXL; the starting latent for seeds 9 to 16 named as the seeded CPU draw wave 1 used; CO3 and SuperDiff grids, the winning rule and the tuning seeds 9 to 12 written; SuperDiff's knob changed from its mixing rule to step count and κ clamp | ToMe's official code did not import within its 30 minutes (your call, `drop tome`); the cache has no entry for seeds 9 to 16; our SuperDiff has only AND; 16 cells by your call to halve the grid's cost; written before any tuning image |
| 2026-11-11 | 3 | CO3 and SuperDiff run at their published defaults only; their tuning grids and winning rule removed | your call, to spend the compute on the evaluation rather than on tuning; the reply to PrzE says these baselines ran at their authors' settings; decided before any tuning or evaluation image |
| 2026-11-11 | 6, and the R1 bar | no table carries an NFE column; the ours-minus-PoE line is not printed in R1 (the bar is still computed and its verdict reported in the finding) | your call: reviewers will not look for either in the tables; decided before any R1 table was built |
| 2026-11-11 | 2, 3 | SuperDiff removed from every table and figure | your call: it does not claim to fix hybrids, so it is not a baseline for composition; decided before any SuperDiff image was scored |
| 2026-11-11 | the scorers, R3 | R3 (Table 2) reports BLIP-VQA ± 95% interval per held-out group instead of the instance-count rate | your call: the count is validated on animals only, and the held-out groups hold object and place pairs (the not-competing group has no animal pair); decided before any R3 table was built |
| 2026-11-11 | 6 | the corrector's labelled subset named: R1's animals set at seeds 9 to 12 (264) and R3's hard group at seeds 9 to 16 (24) | your call (option a of three): the animals set is where two things compete for one space and the hard group is where a corrector would matter most; decided before any corrector picture on either |
