# Walk map: route step 1, check the ground

Plan: [route step 1](../../../plans/route-answer-the-iclr-reviews.md), walked with `/drip-execute-plan --request`.
Board: [the paper review board](../../../report/paper-review/board.md).

## Stop point

Bar: all three objectives answered with evidence, judged from [the ground check](ground-check.md).
Time box: 90 minutes.
Steps at the open: 3 (depth full).

| Sitting | Start | End |
|---|---|---|
| 1 | 2026-11-10 15:06 | 2026-11-10 15:32 |

## Position

Closed 2026-11-10 15:32: bar met, PASS. Retrospective: [the retrospective](2026-11-10-ground-check-retro.md).

## Ticks

| Piece | What | Evidence |
|---|---|---|
| 1 | node table in PRZE-10, its probe line ticked | `nvidia-smi -L` and a 1000 by 1000 matmul on 108, 110, 111 through `scripts/cluster.sh sh`; `sinfo -N`, `squeue -w`, 2026-11-10 15:07 |
| 2 | PRZE-3a lines 1 and 2 ticked; adapter and step findings in PRZE-3a, PRZE-11, the ground check | each Figure 5 tile's record under `ablation_grid/`, each run's `config.json`, `pair_pool.json` and `dataset_meta.json`, `train_pooled.py` line 532, the cache `meta.json` and `step_000.pt`, 2026-11-10 |
| 3 | source differences in the ground check, section 3 (partial) | `pdfinfo`, `git log` and `git status` in `paper/overleaf-iclr/`, `pdftotext` headings, `compare_pdf_to_tex.py` sentence match, 2026-11-10 |

## Decisions

| Decision | Options | Who chose | Why |
|---|---|---|---|
| Overleaf | compare local tex with PDF only and mark partial, ask for token or zip at piece 3 · ask now | you, by `defaults` | no token in hand |
| PRZE-11 training test | train an adapter on all 50 steps against the paper's steps-0-to-9 adapter · keep as written | you: all 50 | the paper's adapter already is the early-only one |
| Adapter of record | v58-03-self step 50,000 · all three Figure 5 runs side by side | you: v58-03 as the record, all three listed | one recipe for scores; the figure's mix stays visible |
| Node plan | 107, 108, 111, 112, with 110 busy until about 2026-11-12 11:30 · your own order | you, `yes` | 110 is held by avogt job 83315 |

## Owed

- `environment/hpc/nodes.md` still says mscluster111's card is unusable from torch (2026-09-05). The 2026-11-10 probe found it working. Proposed diff shown in chat; this walk may not write `environment/`, so it waits for your yes in a session that may.

## Runs, threads, handoffs

None.
