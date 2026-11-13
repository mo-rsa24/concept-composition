# Retrospective: route step 4, the frozen evaluation set and the one adapter

Walk: [the walk map](EXECUTE_WALK-step-4.md). Result: [the frozen set](../../../../report/paper-review/held-out-set.md), commit `089abd6e`. Choice: `artifacts/_shared/cross_pair_pool_configs/adapter_of_record.yaml`. Board after the step: [board picture](../../../../report/paper-review/board-2026-11-10-step-4.png).

## 1. The candidates

**The request.** List candidate pairs from the pair files and the Attend-and-Excite prompt sets, minus the 48 training pairs.

**The problem.** Steps 1 and 2 were untracked, so the cluster checkout could not see the protocol; the Attend-and-Excite prompts are not in that project's repository.

**What was done.** Steps 1 and 2 committed (`084ad92e`) after checking the repo's only remote is GitHub. 62 candidates sorted by kind from `pair_prompts*.yaml`; the prompt sets fetched from the attachment the README links and saved byte for byte to `data/prompt-sets/attend-and-excite/a.e_prompts.json` with a card.

**Where the results live.** [The walk map](EXECUTE_WALK-step-4.md#ticks); the prompt file and its card.

**Before and after.** Before: no candidate list. After: 30 competing, 21 object and place (tent and snowy mountain added from Figure 5), 10 style, 2 showcase; 276 benchmark prompts, none naming a training pair.

**Depends on.** Step 2's protocol, rule 4.

## 2. The census

**The request.** Render the joint prompt and the plain product at seeds 9 to 16 on the cluster.

**The problem.** `biggpu` allows one Slurm job per user and that slot was held; the existing launcher always rendered the adapter.

**What was done.** `1-lora-correction/scripts/review_step4_renders.sh` launched with `nohup` on mscluster111 and 112 behind disk, device and CUDA guards, reusing `concept_pair_sheet.py` (no adapter attached) and `figure_candidates.py` (the 9 candidates). The first sync failed: an earlier rsync had left untracked copies in the cluster checkout. 25 files were moved to a backup, a leftover stashed, and the checkout reset to `6a070f84` with your yes. The paper build now runs outside the checkout, and both faults are catalogued as `poe-sync-001` and `poe-launch-003`.

**Where the results live.** `/datasets/mmolefe/poe_repair_min/outputs/review_step4/` (census, showcase, logs).

**Before and after.** Before: no joint or product picture at seeds 9 to 16 for these pairs. After: 848 census pictures, 144 adapter pictures, 64 look-alike pictures, 0 errors, at about 5 to 6 s a picture against the 12 s estimated.

**Depends on.** Piece 1's candidate list.

## 3. The groups

**The request.** Sort the pairs into hard, easy but unseen and not competing.

**The problem.** The validated instance count covers animals only and cannot tell species apart.

**What was done.** 288 animal pictures counted (`1-lora-correction/scripts/review_step4_count.py`); object pairs called by eye from full resolution; then a species check by eye on every animal pair, which turned cow × horse and seal × walrus from easy to hard; five look-alike pairs added, two of them left out as species-unjudgeable and frog × toad held by the subject guard.

**Where the results live.** [The frozen set](../../../../report/paper-review/held-out-set.md), table "The held-out groups"; R7's list in `1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-pairs/which-pairs-trained-and-which-were-held-out/pair_list.csv`.

**Before and after.** Before: the count alone said no animal pair was hard. After: hard 3 (suitcase × desk fan 3 of 8, cow × horse 2, seal × walrus 0), easy but unseen 29, not competing 21. The plain product composes on 0 to 2 of 8 seeds for every animal pair.

**Depends on.** Piece 2's census.

## 4. The one adapter

**The request.** Run the showcase rule and name the adapter and the shared seed.

**The problem.** Calls from shrunken sheets run generous, and the choice had to stay easy to change.

**What was done.** The 144 pictures sheeted, the contested seeds checked larger, every call written to `1-lora-correction/report/paper-review/showcase-calls.csv`, and `1-lora-correction/scripts/review_showcase_pick.py` ranks the candidates, writes the choice file and can take an override (tested with seed 11, then restored).

**Where the results live.** The choice file; [the showcase results folder](../../../results/the-adapter-on-cat-and-dog/which-adapter-wins/which-adapter-composes-the-showcase-pairs/README.md).

**Before and after.** Before: v58-03-self 50k by description. After: v58-03-self 30k, clear on both pairs at 5 seeds (10, 11, 12, 13, 16) against 4 for the next two; shared seed 10. The selection grid (R8) was drawn and then removed from the paper on your judgement that it answers no reviewer question.

**Depends on.** Piece 2's showcase renders.

## 5. The frozen file and the page

**The request.** Commit the frozen file before any adapter result on the evaluation pairs; fill R7 and R8; redraw the board.

**The problem.** The main table's design changed mid-step.

**What was done.** On your call, R1 became the 276 Attend-and-Excite prompts at seeds 9 to 12 with a CLIP column, R3 the 53-pair ablation, R14 GenEval and T2I-CompBench on their own prompts; frog and mouse allowed for the 45 benchmark prompts in a new pair ledger. Frozen file committed; R7 filled with a list mode; PRZE-3b and PRZE-8 lines ticked; board redrawn. Codex reviewed the stand-ins for presentation; its findings are a checklist on the layout sheet.

**Where the results live.** [The frozen set](../../../../report/paper-review/held-out-set.md); [the protocol](../../../../report/paper-review/protocol.md); [the layout sheet](../../../../report/paper-review/layout-sheet.md); [the Codex review](routes/codex-presentation-review.md).

**Before and after.** Before: 12 placeholder slots, no frozen set. After: 13 slots (R7 filled), `--check` PASS, frozen set at `089abd6e`, board 6 in progress and 6 to do.

**Depends on.** Pieces 3 and 4.

## The stop point

Bar: the frozen file committed with at least 8 pairs per group or a smaller count you accept, the adapter named by the showcase rule, and no adapter result on any evaluation pair. Time box: 60 minutes of your time, cluster renders not counted. Used: one sitting, 19:04 to 21:13 on the clock, with your gaps between replies. The bar is met with the hard group at 3, accepted under option a. PASS. Way on: ship.

## Decisions

| Decision | Chosen | Options | Who chose | Why |
|---|---|---|---|---|
| Commit steps 1 and 2 | committed | launch without | you, "only if it won't make it to Overleaf" | the only remote is GitHub |
| Style pairs | left out | a fourth group | you, `defaults` | fit no group |
| Census scope | all 50 | 30 | you, `defaults` | |
| Attend-and-Excite sets in the census | no | add 66 animal prompts | you, `a` | census time |
| Joint prompt template | "a and b" for every pair | recorded "a in b" for 10 | you, `yes a` | one template |
| Commit the linking agents' lines | committed separately | wait | you | unblock the sync |
| Cluster checkout repair | `git reset --hard 6a070f84` | `/reconcile-machines` | you, `yes reset` | every leftover equalled the commit or was backed up |
| Object pairs judged | by eye | per-name detector, left out | you, `a` | the count covers animals only |
| Hard group | 3 pairs, under 8, stated | cut at 4 or fewer for 7 | default a, carried | no rule changed after the data |
| More hard candidates | look-alikes plus a species check | accept 1, widen to 66 prompts | you, `a` | |
| Main table | 276 benchmark prompts at 4 seeds, census as R3 | 55 pairs as R1, 1 seed | you, `yes 4, frog mouse yes` | the same prompts and metrics as CO3's Table 1 |
| Benchmarks | CLIP in R1, new R14 | CLIP only | you, `yes R14` | PrzE named GenEval and T2I-CompBench |
| The adapter | v58-03-self 30k, seed 10 | seed 11 | you, agree; keep it swappable | |
| R8 in the paper | removed | kept | you | answers no reviewer question; coloured frames rejected |

## Dead ends and not done

- PRZE-3b's per-seed cat-and-dog line and PRZE-3a's appendix table (training pairs and a CO3 filter column) remain open.
- R2 still shows the submitted three-adapter Figure 5 until it is rebuilt from v58-03-self 30k at seed 10, with labels at 7 pt.
- The Codex presentation checklist is not yet applied to the generators (booktabs, grouped headers, `siunitx`, no shrinking, captions above tables, reader-facing captions).
- eagle × hawk and leopard × jaguar are out (species unjudgeable); frog × toad was not rendered.
- The paper text still names step 50,000 as the adapter's checkpoint; step 13 changes it to 30,000 (PRZE-5).

## The family table

| What | Id | State |
|---|---|---|
| this walk's session | 2aaf06ce-a69d-473c-a89c-cb6cd4e4ae1d | closed |
| census-a, mscluster111 | PID 25435 | finished 18:14 UTC |
| census-b and the adapter test, mscluster112 | PID 31439 | finished 18:39 UTC |
| look-alikes and recount, mscluster111 | nohup chain | finished 18:34 UTC |
| Codex presentation review | `codex exec`, read-only | returned, filed in `routes/` |

---

Validation: every repository path above was listed on disk on 2026-11-10; the counts were copied from this walk's command outputs and `census_counts.csv`.
