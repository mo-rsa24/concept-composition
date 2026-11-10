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
