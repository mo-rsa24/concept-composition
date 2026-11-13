# Walk map: route step 2, the protocol, the layout sheet and the slots

Plan: [route step 2](../../../../../plans/route-answer-the-iclr-reviews.md#2--write-the-comparison-protocol-and-the-figure-and-table-layout-sheet), walked with `/drip-execute-plan --request`.
Board: [the paper review board](../../../../report/paper-review/board.md).

## Stop point

Bar: the export compiles with every slot as a placeholder, one slot refills from a result file, `python3 1-lora-correction/scripts/refresh_paper_slots.py --check` prints PASS with the slot count, the board redraws with PRZE-8 and PRZE-9 ticked, and you agree the protocol.
Time box: 60 minutes of your time.
Steps at the open: 4 (pieces 0 to 3, depth full).

| Sitting | Start | End |
|---|---|---|
| 1 | 2026-11-10 15:52 | 2026-11-10 18:26 |

Toy: elephant and penguin, seed 10.

## Position

Closed 2026-11-10 18:26: bar met, PASS (refill tested from a test file). Retrospective: [the retrospective](2026-11-10-protocol-and-slots-retro.md).

| Piece | What | Evidence |
|---|---|---|
| 3 | `1-lora-correction/scripts/refresh_paper_slots.py`; R1 refilled from a test CSV in a scratch copy; first run overflowed by 247.97 pt, fixed by shrinking an over-wide table to the text width; real paper built with every slot a placeholder; PRZE-9 lines 1 to 4 ticked; board redrawn at step 2 | `--check` PASS: 13 slots; both builds 22 pages, exit 0, floats on the same pages; `1-lora-correction/report/paper-review/board-2026-11-10-step-2.png` |

| Piece | What | Evidence |
|---|---|---|
| 1 | `1-lora-correction/report/paper-review/protocol.md`, option B, agreed; PRZE-8 lines 1, 3, 4 ticked | your "Yes", 2026-11-10 16:15 |
| 2 | `1-lora-correction/report/paper-review/layout-sheet.md` (13 slots), 13 reserved rows in `paper/iclr-56173/figures.md`, `\reviewslot` and 13 slots in the tex | Tectonic build 2026-11-10 16:28: 21 pages, no error, no overfull box; Figures 1 to 8 on their piece 0 pages; References moved from page 10 to 11 (R1) |

## Ticks

| Piece | What | Evidence |
|---|---|---|
| 0 | export unpacked to `paper/iclr-56173/` (62 files, tex 89,856 bytes), rsynced, built with Tectonic on mscluster | build wrote `iclr2027_conference.pdf`, 33.33 MiB, 14 pages, 2026-11-10 15:57; the submitted PDF is 13 pages |

## Facts found on the way

- The Tectonic build sets the body in Latin Modern (`LMRoman10`), while the submitted PDF (pdfTeX 1.40.29) uses Nimbus Roman (Times). The `times` package does not take effect under Tectonic's XeTeX engine, so line breaks differ and the build runs 14 pages against 13.
- The PDF inside the export, `paper/iclr-56173/iclr2027_conference.pdf`, is a 16-page build from 2026-09-25 00:13, older than the submission. It is not the submitted PDF.
- Figure 5's elephant and penguin "ours" tile equals `ablation_grid/an_elephant__x__a_penguin/seed10/ours_v58-05-self-contrast_s015000_rawA_w50.png` pixel for pixel (mean absolute difference 0.00 on a 0 to 255 scale, 300 by 300 crop). The reward fine-tunes `refl-kl-K0`, `K1`, `K2` (W&B t4wiwy5p, u73uvqyg, yj7l40h5) differ by 12.86, 6.32 and 14.10. `refl-kl-K3` (ybjd4trt) has no tile at that seed.
- The four `refl-kl-K*` runs start from v58-05-self-contrast step 15,000 and train on cat and dog and elephant and penguin at seeds 1 to 8 (W&B config `pairs`, `seeds`). Any adapter from them has seen both held-out pairs.

## Decisions

| Decision | Options | Who chose | Why |
|---|---|---|---|
| Toy pair | elephant and penguin seed 10 · cat and dog seed 9 | you, by `defaults` | held out from v58 training, seed never trained on |
| Committing `paper/iclr-56173/` later | tex, bib, sty, `figures.md` with `figures/` and PDF ignored · everything | you, by `defaults` | 119 MB of figures |
| What "no page shift" means | Tectonic before and after, 14 pages and every section on its page, true count checked in Overleaf at upload · TinyTeX pdfLaTeX on mscluster | you: progress (a) | the comparison should not hold up the work |
| How the one adapter is picked | A: on the tuning set, all five held-out pairs stay held out · B: on cat and dog and elephant and penguin at one shared seed, those two become showcase pairs out of held-out rates | you: B | the best adapter, clean on both showcase pairs on one seed, with the choice stated |
| Improvements | rule-to-reviewer column · `--check` fails rows missing axis values | you, by `defaults` (both kept) | |

## Runs, threads, handoffs

None.
