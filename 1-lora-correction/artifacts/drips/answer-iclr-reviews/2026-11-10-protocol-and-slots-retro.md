# Retrospective: route step 2, the protocol, the layout sheet and the slots

Walk: [the walk map](EXECUTE_WALK-step-2.md). Results: [the comparison protocol](../../../report/paper-review/protocol.md), [the layout sheet](../../../report/paper-review/layout-sheet.md), [the refill script](../../../scripts/refresh_paper_slots.py). Board after the step: [board picture](../../../report/paper-review/board-2026-11-10-step-2.png).

## 0. Compile the untouched export

**The request.** Unpack `_ICLR2027__LoRA_Plurality.zip` into `paper/iclr-56173/` and confirm it compiles unchanged.

**The problem.** No LaTeX on the laptop; every build runs with Tectonic on mscluster.

**What was done.** `unzip` into `paper/iclr-56173/` (62 files), `rsync` to the cluster checkout, Tectonic build through `scripts/cluster.sh sh`; `pdfinfo` and `pdffonts` against the submitted PDF. The four W&B runs you linked for elephant and penguin were read and Figure 5's tile was matched against each candidate by mean absolute pixel difference.

**Where the results live.** The facts are rows in the route's "What we know so far" and in [the walk map](EXECUTE_WALK-step-2.md#facts-found-on-the-way).

**Before and after.** Before: the export had never been built. After: it builds, 14 pages against the submitted 13, because Tectonic's XeTeX engine ignores the `times` package and sets Latin Modern. Figure 5's elephant and penguin tile equals `ours_v58-05-self-contrast_s015000_rawA_w50.png` (difference 0.00 on a 0 to 255 scale); the `refl-kl-K*` runs differ by 6.32 to 14.10 and trained on both showcase pairs.

**Depends on.** Step 1's finding that the export is the source to edit.

## 1. The comparison protocol

**The request.** Write the rules every result must obey and agree them before any run.

**The problem.** Figure 5 mixed three adapters picked per row, and you wanted the best adapter in every row without being open to an unfair-comparison attack.

**What was done.** Ten rules, each with the reviewer point it defends against, and a worked row on elephant and penguin seed 10. Rule 1 was rewritten from a fixed adapter to the showcase rule: 9 fixed candidates render cat and dog and elephant and penguin at seeds 9 to 16, and the adapter with the most seeds where both are two clear animals wins. ToMe, SynGen, Divide and Bind and InitNO were added and FK steering moved to the CO3-protocol table, dated in [the board history](../../../report/paper-review/history.md#protocol-changes).

**Where the results live.** [The protocol](../../../report/paper-review/protocol.md); PRZE-8 lines 1, 3 and 4 ticked on [its page](../../../report/paper-review/prze-8-protocol.md).

**Before and after.** Before: no written rules, one adapter fixed by description. After: agreed 2026-11-10; the adapter is chosen by a procedure fixed before any held-out result; cat and dog and elephant and penguin leave every held-out rate.

**Depends on.** Piece 0's tile match, which showed which runs had seen the showcase pairs.

## 2. The layout sheet and the slots

**The request.** One row per new figure and table, a register row per slot, and a placeholder at final size in the tex.

**The problem.** The main text already ends on page 9, so every main-text slot costs room.

**What was done.** 13 slots, R1 to R13: two in the main text (R1, the SDXL score table; R2, Figure 5 rebuilt in its own wrapped slot) and eleven in the appendix. One macro, `\reviewslot`, wraps the existing `\figurepanel` and picks up a slot file when it exists. 13 reserved rows in `paper/iclr-56173/figures.md`. R1 went through three shapes on your look: full columns, one two-block table, then SDXL in R1 and SD 1.5 in R5 in the appendix, centered, with an "ours (SD 1.5 adapter)" row and a caption sentence on official SD 1.x code.

**Where the results live.** [The layout sheet](../../../report/paper-review/layout-sheet.md); the register's section "Slots for the review response to submission 56173"; the tex.

**Before and after.** Before: no slot. After: the paper builds with every slot a placeholder (21 pages, exit 0, no overfull box); the References move from page 10 to 11, paid for by moving Figure 6 to the appendix in step 13 (your decision a).

**Depends on.** Piece 1's rules, which set R1's rows, R5's split and the mixed-comparison marking.

## 3. The refill script

**The request.** A script that rebuilds a slot from its result file and compiles.

**The problem.** No result file exists yet.

**What was done.** `1-lora-correction/scripts/refresh_paper_slots.py` reads the layout sheet, copies a PNG or PDF, turns a CSV into a centered table, refuses a CSV without protocol rule 9's twelve axis columns, builds on mscluster with `--compile`, and sheets every slot's page with `--gallery`. Tested by refilling R1 and R5 from a test CSV of zeros in a scratch copy of the paper. Four faults found and fixed on the way: a table 247.97 pt too wide (short labels and a `reviewslot <id> shrunk` log line), the table set flush left (centered), a stale remote slot file (the rsync now deletes), and the gallery calling the submitted Figure 5 a placeholder.

**Where the results live.** [The refill script](../../../scripts/refresh_paper_slots.py); the build log at `paper/iclr-56173/build/iclr2027_conference.log`; [the gallery](../../../../paper/iclr-56173/build/slot-gallery.png); PRZE-9 lines 1 to 4 ticked on [its page](../../../report/paper-review/prze-9-slots.md).

**Before and after.** Before: no refill path. After: `--check` prints `PASS: 13 slots, 0 with a result file, 13 placeholders`; a refilled R1 fits the text width at 8 pt and moves no float.

**Depends on.** Piece 2's slots and sizes.

## The stop point

Bar: the export compiles with every slot as a placeholder, one slot refills from a result file, `--check` prints PASS with the slot count, the board redraws with PRZE-8 and PRZE-9 ticked, and you agree the protocol. Time box: 60 minutes. Used: one sitting, 15:52 to 18:26 on the clock, with your gaps between replies inside it. The bar came first, with one caveat: the refill ran from a test file in a scratch copy, because no result exists yet. PASS. Way on: ship, then route step 3.

## Decisions

| Decision | Chosen | Options | Who chose | Why |
|---|---|---|---|---|
| Toy pair | elephant and penguin, seed 10 | cat and dog, seed 9 | you, `defaults` | held out at training, seed never trained on |
| Committing the export later | tex, bib, sty, `figures.md`; figures and PDF ignored | everything, 119 MB | you, `defaults` | size |
| What "no page shift" means | Tectonic before and after; true count in Overleaf at upload | TinyTeX pdfLaTeX on mscluster | you, "shouldn't we progress" | progress over an exact match |
| How the one adapter is picked | B: on the two showcase pairs at one shared seed | A: on the tuning set | you, `B` | the best adapter, clean on both pairs, choice stated |
| Room for R1 | Figure 6 to the appendix | R1 in the appendix; cut text | you, `a` | Figure 6 is an ablation |
| Baselines | ToMe on SDXL, five binding methods on SD 1.5, FK steering as a noise-search control | FK steering in the main table | you, `A` | CO3 Table 1's methods; FK tests search, not binding |
| R1's shape | SDXL in R1 (main), SD 1.5 in R5 (appendix), centered | one two-block table | you, on the render | the white space and the page limit |
| SD 2.x and SD 3.5 | left out; SD 1.5 adapter row waits on step 12 | add either model | you, `caption and fix R5` | cost against an 18 November deadline |
| Overleaf | download the export and upload by hand | clone and push | you, "I don't want to git push anything" | no git to Overleaf |

## Dead ends and not done

- PRZE-8's "adapter named" line waits on the 144 showcase renders (route step 4 or 5).
- PRZE-9's legibility line stays open until each image slot's result lands.
- Whether ToMe's official code loads SDXL and SynGen, Divide and Bind and InitNO load SD 1.x is recalled; route step 3's prompt now asks for the check.
- The reviews PDF was not reread in this walk; the reviewer-to-slot map came from the board pages.
- Nothing was committed: the brief said to commit nothing until you say so.

## The family table

| What | Id | State |
|---|---|---|
| this walk's session | 2aaf06ce-a69d-473c-a89c-cb6cd4e4ae1d | closed |
| runs, threads, handoffs | none | |

---

Validation: every repository path above was listed on disk on 2026-11-10; the numbers were copied from this walk's command outputs.
