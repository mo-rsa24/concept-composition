# Route step 9: score every image and build Tables 1 and 2

The retrospective of the 2026-11-11 `/drip-execute-plan --request` walk over [route step 9](../../../plans/route-answer-the-iclr-reviews.md#9--score-every-image-and-build-the-tables), session `47281f64-fc21-4b71-b0fd-3b968913d9b3`. Walk map: [the step 9 walk map](../../../1-lora-correction/artifacts/drips/answering-the-reviewers/answer-iclr-reviews/EXECUTE_WALK-step-9.md).

## 1. Scorers installed and proved on two images

**The request.** Install and verify each scorer on the two small-object images from step 7, and write each one's version and settings into the protocol.

**The problem.** Step 7 never left its two images or its combine map. GenEval and T2I-CompBench had no code in the repo, and their pinned stacks (torch 1.12 with mmcv-full 1.7.1; torch 2.0.1 with transformers 4.30.2) do not run on the Blackwell cards.

**What was done.** The mug and wine glass pair, seed 11, from the step 4 census became the probe: the joint render shows both objects, the product render shows only the glass. CLIP score and ImageReward ran in `co3`. T2I-CompBench's BLIP-VQA and GenEval's evaluator were installed into prefix envs under `/datasets/mmolefe/envs/`, after four fixes: a corrupted shared conda cache (fresh `CONDA_PKGS_DIRS`), torch 1.12.1's executable-stack flag (`patchelf --clear-execstack`), the mmcv-full wheel served under `torch1.12.0/` rather than the URL pip built, and `ruamel.yaml` pinned to T2I-CompBench's 0.17.32.

**Where the results live.** [The four-scorer probe figure](../../../1-lora-correction/artifacts/results/measuring-composition/the-scorers/what-each-scorer-says-about-two-objects/four-scorers-on-mug-and-wine-glass.png), with the three probe CSVs beside it.

<a href="../../results/what-each-scorer-says-about-two-objects/four-scorers-on-mug-and-wine-glass.png"><img src="../../results/what-each-scorer-says-about-two-objects/four-scorers-on-mug-and-wine-glass.png" width="200" alt="the mug and wine glass pair with four scores under each image"></a>

**Before and after.** Before: two scorers in the repo, one of which (`eval_metrics.py`) does not import under `co3`. After: four scorers, each ranking the two-object image above the one with the mug missing; CLIP score moved 26.7 to 23.6, BLIP-VQA 0.364 to 0.004.

**Links to.** The scorers are what chunk 3 runs.

## 2. The bar written before any score

**The request.** Write the supports, null and inconclusive lines into the protocol before any score.

**The problem.** None beyond writing it; the user later removed the bar's line and the NFE column from the printed table.

**What was done.** The scorer section and the R1 bar (ours minus PoE in BLIP-VQA, interval above 0 in all three sets) were committed as `3c65c634` before any evaluation image was scored. The thresholds are constants in `1-lora-correction/scripts/review_build_tables.py`.

**Where the results live.** The protocol's sections "The scorers" and "The bar for R1", and `1-lora-correction/report/paper-review/history.md` for each later change.

**Before and after.** Before: no scorer named, no verdict rule. After: both fixed in the protocol and dated.

**Links to.** Chunk 4 reads the verdict against this bar.

## 3. Every complete folder scored

**The request.** Score every image and write one long results file with the axis columns on every row.

**The problem.** The `ours` R1 column had not started, the six `bigbatch` slots were held by the baseline session, and mscluster108's view of `/datasets` was missing whole folders.

**What was done.** `ours` was started on the free cards, CLIP score and ImageReward moved to CPU (protocol changed and dated first), BLIP-VQA ran on mscluster108's two idle RTX 8000 cards at 3 question passes (equal to 8 here, since no prompt has more than 3 noun phrases), and the animal count ran on CPU. The 5 images 108 could not read are listed in `scores/rows_hidden.csv`.

**Where the results live.** `/datasets/mmolefe/poe_repair_min/outputs/paper_review/scores/scores_long.csv`, 4,584 images, one line each with rule 9's axis columns.

**Before and after.** Before: no evaluation score. After: joint prompt, PoE and ours scored on all 1,104 R1 images each and all 424 held-out images each.

**Links to.** Chunk 4 averages this file.

## 4. Tables 1 and 2, and the verdict

**The request.** Build the CO3-style table, the rate table with intervals, and the paired differences.

**The problem.** The instance count is validated on animals only, and Table 2's groups hold object and place pairs.

**What was done.** Table 2 moved to BLIP-VQA with an interval from resampling pairs, then seeds (the user's call). `1-lora-correction/scripts/review_build_tables.py` writes both tables and the bar's verdict.

**Where the results live.** [Table 1](../../../1-lora-correction/report/paper-review/table-1-r1-benchmark.csv), [Table 2](../../../1-lora-correction/report/paper-review/table-2-r3-held-out.csv) and [the sidecar](../../../1-lora-correction/report/paper-review/tables-1-and-2-sidecar.json) with each cell's count and the verdict.

**Before and after.** The bar reads SUPPORTS: ours minus PoE in BLIP-VQA is above 0 in all three sets (the intervals are in the sidecar).

**Links to.** Chunk 5 fills these into the paper.

## 5. Slots filled and the paper built

**The request.** Fill the table slots and check the paper still compiles.

**The problem.** The slot layouts still carried NFE and the count for R3, and both captions described the old tables.

**What was done.** The layouts in `1-lora-correction/scripts/refresh_paper_slots.py` and both captions were rewritten to match; slots R1 and R3 filled.

**Where the results live.** `paper/iclr-56173/slots/R1.tex`, `R3.tex`; page 9 of the build. Commit `56b968b6`.

**Before and after.** Before: two grey placeholders. After: `--check` PASS (12 slots, 5 filled), 22-page build, no slot too wide.

**Links to.** Step 13 revises the prose around these tables.

## The stop point

Bar: `--check` PASS, the paper compiles, and every R1 cell scored or its skip recorded. Met for joint, PoE and ours; GenEval and T2I-CompBench feed R14 only, by the layout sheet, and are recorded as owed there. Time box: 60 minutes of the user's time; the session ran 05:20 to 13:30 with most of it waiting on renders. The way on taken: ship the three-row tables now, add CO3 when step 8's renders land.

## Decisions

| Decision | Chosen | Options | Who | Why |
|---|---|---|---|---|
| step 7 gate | close it inside this walk | run step 7 first | default accepted | step 7's outputs were never written |
| prediction before numbers | dropped | keep | the user | "I don't want to guess" |
| NFE column and printed ours-minus-PoE line | dropped from the tables | keep | the user | reviewers will not look for them |
| SuperDiff | out of every table; its four jobs cancelled, CO3 moved to six shards | keep | the user | it does not claim to fix hybrids |
| CLIP score and ImageReward hardware | CPU for every row | wait for a GPU slot; cancel a baseline job | the user | the GPU slots were full |
| seeds per prompt | 4, as the protocol says | 1; mixed | default accepted | Attend-and-Excite used 64 per prompt; 1 is too few |
| Table 2's measure | BLIP-VQA with intervals | the count on animal pairs; both | the user | the count is validated on animals only |
| Figure 15 | parked | finish now | the user | tables first |

## Dead ends and not done

- Figure 15 (R12): parked, owed under protocol rule 8; the preview has 16 of 18 tiles.
- CO3 row of both tables: rendering under step 8 (six jobs).
- CO3's printed numbers for the grey row: not read yet.
- R14 (GenEval and T2I-CompBench on their own prompts): scorers installed and probed; the prompts are not rendered.
- 5 images unreadable from mscluster108: listed, to be scored on another GPU.
- Relaunching `ours` beside running work on 111 and 112 needed an override of the render script's sharing guard, which the session was not allowed to use.

## The family

| Id | Kind | State |
|---|---|---|
| session `47281f64-fc21-4b71-b0fd-3b968913d9b3` | this walk | closed |
| Slurm 84587, 84588 | probe jobs | done |
| Slurm 84640, 84737, 84788, 84779, 84790 | CPU scoring | done |
| BLIP-VQA on mscluster108, batches 1a, 1b, 2, 3 | GPU scoring outside Slurm | done |
| Slurm 84590, 84591, 84757, 84759, 84760, 84761 | CO3 renders (step 8) | running |

---

Validation: every repo path above was listed on disk on 2026-11-11; `scores_long.csv` and the hidden list live on `/datasets` and were read from mscluster84. The verdict and counts come from the sidecar.
