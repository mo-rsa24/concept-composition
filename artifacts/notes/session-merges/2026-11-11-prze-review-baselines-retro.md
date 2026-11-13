# CO3 and the other baselines for the review: retrospective, 2026-11-11

The walk that rendered the baseline methods for the ICLR 56173 review response, route step 8b. Plan: [the baselines plan](../../../1-lora-correction/plans/the-baselines-the-reviewers-asked-for.md). Walk map: [the walk map](../../../1-lora-correction/artifacts/drips/review-baselines/EXECUTE_WALK.md).

## 1. One cell per method

**The request.** CO3, ToMe and SuperDiff each render lion and meerkat at seed 9 on SDXL, from our starting latent where the code accepts one.

**The problem.** CO3 draws its own noise inside `run_sampling`; ToMe needed its own environment; the cache has no entry for seeds 9 to 16.

**What was done.** A wrapper hands CO3's `sample_loop` our latent without editing the vendored code. The seed 9 to 16 latents are saved through the same function wave 1 calls, which falls back to a seeded fp16 CPU draw, byte-identical under torch 2.5.1 and 2.11. ToMe was cloned at `f256a49a` and its environment built in 22 minutes, then failed to import (`cached_download` missing from `huggingface_hub` 0.36).

**Where it lives.** `outputs/paper_review/co3_default/a_lion__x__a_meerkat/seed09/` and `superdiff_default/…`; `1-lora-correction/scripts/review_seed_latents.py`, `1-lora-correction/scripts/review_baseline_cell.py`, `scripts/review_tome_cell.py`.

**Before and after.** No baseline rendered from our noise → CO3 79.7 s, two clear animals; SuperDiff 165.6 s, one hybrid; ToMe dropped.

**Link.** Chunk 2 writes the defaults these cells used.

## 2. The published defaults in the protocol

**The request.** A grid of at most six settings per method and a winning rule, written before any tuning image.

**The problem.** The grid cost about 5 card-hours per method, and no reviewer asked for tuned baselines.

**What was done.** The grids and the rule were written (`028d9d69`), then removed on your call, so rule 3 says CO3 runs at its published default only (`9055bb59`).

**Where it lives.** Rule 3 of `1-lora-correction/report/paper-review/protocol.md`; two dated rows in `1-lora-correction/report/paper-review/history.md`.

**Before and after.** Default plus a tuned row per method → one default row per method.

**Link.** Chunk 3 renders at these defaults.

## 3. The evaluation renders

**The request.** Each method at its default on R1 (1,104 cells) and R3 (424 cells).

**The problem.** A dead node (mscluster88), SuperDiff dropped by the step 9 walk, a slow card (mscluster48 at 83 s), and a stale NFS view on mscluster84.

**What was done.** Six `bigbatch` jobs, then four more CO3 shards from the step 9 walk, then two tail helpers on mscluster108 with whole pairs and a retrying mkdir (`cdd58a18`). Figure 5's four pairs at seed 10 rendered on mscluster85. The row builder `1-lora-correction/scripts/review_score_rows.py` was taught to read baseline sidecars (`2f838931`).

**Where it lives.** `outputs/paper_review/co3_default/{r1,r3,r2}/`, logs `eval_eval-co3-*.log` and `co3_tail2way_*.log`.

**Before and after.** 0 → 1,104 R1 and 424 R3 CO3 tiles, counted from mscluster107; 1,548 renders for 1,528 cells.

**Link.** Chunk 4 counts and records them.

## 4. Count, file and record

**The request.** Counts against 1,104 and 424, jobs table rows 7 and 8, the route step.

**What was done.** Counted from mscluster107; rows 7 and 8 filled; route step 8b and five findings rows added (`ebc0e5ba`).

**Where it lives.** `1-lora-correction/report/paper-review/prze-10-compute.md`, `plans/route-answer-the-iclr-reviews.md`.

## The stop point

Bar: every evaluation job finished with its count matching. Met for CO3: 1,104 and 424. Box: 90 minutes; used about 2 h 40 min of your time over 07:24 to 14:20, most of it waiting on renders. Way on: ship.

## Decisions

| Decision | Chosen | Options | Who | Why |
|---|---|---|---|---|
| CO3 noise | our latent through `sample_loop` | ours · its own, mixed | default you accepted | an unmixed column without editing CO3 |
| ToMe | dropped | pin the hub library · drop | you | missed its 30-minute limit |
| Tuning | none, defaults only | 32 cells · 16 cells · none | you | spend the compute on the evaluation |
| SuperDiff | out of every table | mixed column · out | you, in the step 9 walk | it does not claim to fix hybrids |
| Shard order | R1 first, R3 last | leave · reorder | you | restarting cost more than it bought |
| Tail helpers | two on mscluster108 | three · two | you | leave mscluster107 alone |

## Dead ends and not done

- ToMe: environment built, import failed; not retried.
- SuperDiff: about 373 tiles under `superdiff_default/`, unused.
- The tuning grid launcher `1-lora-correction/scripts/review_baselines_grid.sh` is committed and unused.
- Instruction 1 (your call on the one-cell tiles) and instruction 2 (PASS or MISS) are open.
- No contact sheet was filed: the grids that would have made one were not run.

## The family

| Session or job | What | State |
|---|---|---|
| this session `ece09d30` | the walk | closed |
| `47281f64` | route step 9, scoring; cancelled SuperDiff, added CO3 shards 2 to 5 | separate session |
| `a4d17e2c` | wave 1 (PRZE-10); the lessons pasted at 13:12 | separate session |
| Slurm 84574, 84580 | one-cell checks | done; ToMe failed |
| Slurm 84590, 84591, 84757, 84759, 84760, 84761 | CO3 evaluation | done |
| Slurm 84592 to 84597 | SuperDiff evaluation | cancelled |
| mscluster108 pids 127193, 127483; mscluster85 pid 20830 | CO3 tail and Figure 5 | done |

---

Validation: every repository path above exists (checked 2026-11-11 14:22); the counts come from mscluster107's `ls` at 14:10.
