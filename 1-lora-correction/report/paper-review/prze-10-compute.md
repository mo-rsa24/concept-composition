# PRZE-10: the compute plan

Every `biggpu` node was probed on 2026-11-10, so jobs are placed from the table below and not from memory. Jobs go to 107 (your interactive job), 108, 111 and 112. Node 110 holds another user's job until about 2026-11-12 11:30 and is not planned on.

## The nodes, probed 2026-11-10

Each node was reached over `ssh` from mscluster84 through `scripts/cluster.sh sh`, read with `nvidia-smi -L` and `--query-gpu`, and given a 1000 by 1000 ones matmul on every card under the absolute path of the named python. A matmul sum of 1e9 means the card computed on the GPU. State and jobs come from `sinfo -N` and `squeue -w` the same hour.

| Node | Cards | Python | `torch.cuda.is_available()` | Matmul sum | Slurm state, who holds it | Usable for this work |
|---|---|---|---|---|---|---|
| mscluster107 | 2× Quadro RTX 8000, 49,152 MiB each | `co3` | True (earlier probe today) | | alloc, your job 83423 (interactive, 3-day limit) | yes, inside your job |
| mscluster108 | 2× Quadro RTX 8000, 49,152 MiB each, 1 MiB used | `co3` (torch 2.5.1+cu118) | True | 1e9 on both cards | idle | yes |
| mscluster110 | 1× RTX PRO 6000 Blackwell, 97,887 MiB, 17,624 MiB used, 100% busy | `co3_bw` (torch 2.11.0+cu128) | True | 1e9 | alloc, avogt job 83315, started 1 day 3 h before the probe, 3-day limit | no, until about 2026-11-12 11:30 |
| mscluster111 | 1× RTX PRO 6000 Blackwell, 97,887 MiB, 2 MiB used | `co3_bw` (torch 2.11.0+cu128) | True | 1e9 | idle | yes |
| mscluster112 | 1× RTX PRO 6000 Blackwell, 97,887 MiB, 2 MiB used | `co3_bw` | True (earlier probe today) | 8e9 on a 2000 by 2000 matmul | idle | yes |

No node failed to answer, so no retry was needed.

## Where each card stands, 2026-11-11 06:30

Re-read with `nvidia-smi --query-gpu` and `pgrep -af python` on each node, through `scripts/cluster.sh sh`.

| Node, card | Memory used, load | What holds it | Free for wave 1 |
|---|---|---|---|
| mscluster107, cards 0 and 1 | 29,556 MiB each, 100% | your trainings `v58-06-bound-w40` and `v58-07-bound-w40-inplace4` (job 83423, 1 day 8 h 43 min left) | no |
| mscluster111, card 0 | 32,694 MiB, 87% | your training `v58-08-bound-w40-attnsep005`, started with `nohup` | no |
| mscluster108, card 0 | 1 MiB, 0% at 06:34 | nothing; the CO3 selection run wrote its 9 of 9 pictures and exited at 06:33 | yes |
| mscluster108, card 1 | 1 MiB, 0% | nothing; matmul sum 1e9 under `co3`, xformers 0.0.28.post3 | yes |
| mscluster112, card 0 | 2 MiB, 0% | nothing | yes |

## How long one image takes, measured

From step 4's logs in `/datasets/mmolefe/poe_repair_min/outputs/review_step4/logs/`: the time between one column's start line and the next, over the images that column wrote. SDXL, 1024 by 1024, 50 DDIM steps, guidance 7.5, fp16, on the RTX PRO 6000 Blackwell.

| Column | Log | Images | Wall time | Seconds per image |
|---|---|---|---|---|
| joint | `census-a.log` (mscluster111) | 240 | 20 min 56 s | 5.2 |
| joint | `census-b.log` (mscluster112) | 184 | 15 min 28 s | 5.0 |
| PoE | `census-a.log` (mscluster111) | 240 | 23 min 26 s | 5.9 |
| PoE | `census-b.log` (mscluster112) | 184 | 17 min 22 s | 5.7 |
| ours, adapter on all 50 steps | `census-b.log`, nine showcase candidates (mscluster112) | 16 per candidate | 3 min 13 s to 3 min 48 s | 12.1 to 14.3, adapter load included |

No column of this kind has been timed on an RTX 8000. The one RTX 8000 measurement, an eval pass at 51.5 s per cell against 11.2 to 16.0 s per cell on a Blackwell (`environment/hpc/throughput.md`), puts an RTX 8000 at 3.2 to 4.6 times slower, so a joint image there is estimated at 16 to 24 s.

## The jobs

Priority 1 is wave 1, which no gate blocks. Hours are on the node named, one card, run one job after another in the order listed. Memory is an estimate until each job's prediction, below, is checked against `nvidia-smi` at launch. Images count every render the job writes; a job resumes from the tiles already on disk.

| # | Job | Slot | Node, card | Memory | Images | Hours | Basis | Priority | Waits on |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Figure 5: joint, PoE and ours for lighthouse × stormy sea at seed 10 (tent × snowy mountain is a held-out pair: its joint and PoE tiles are in step 4's census, its ours tile comes from job 3) | R2 | 112, 0 | 10 to 16 GB | 3 | 0.02 | measured rates above | 1 | nothing; runs first as the launch check |
| 2 | early-steps test: ours with the adapter on steps 0 to 10 and on steps 0 to 25, the product alone after, 53 held-out pairs, seeds 9 to 16 | R9, R10 | 112, 0 | 10 to 16 GB | 848 | 2.9 to 3.4 | ours rate, measured | 1 | job 1 |
| 3 | ours on the 53 held-out pairs, seeds 9 to 16 | R3, R4 | 112, 0 | 10 to 16 GB | 424 | 1.4 to 1.7 | ours rate, measured | 1 | job 2 |
| 4 | PoE on the 276 benchmark prompts, seeds 9 to 12 | R1, R12 | 108, 1 | 10 to 16 GB | 1,104 | 5.6 to 8.4 | estimate, the PoE rate scaled from the RTX 8000 eval pass | 1 | the launcher's joint-text and ledger changes |
| 5 | ours on the 276 benchmark prompts, seeds 9 to 12 | R1, R12 | 111, 0 | 10 to 16 GB | 1,104 | 3.7 to 4.4 | ours rate, measured | 1 | `v58-08-bound-w40-attnsep005` finishing on 111, about 09:30 |
| 6 | joint on the 276 benchmark prompts as written, seeds 9 to 12 | R1, R12 | 108, 0 | 10 to 16 GB | 1,104 | 4.9 to 7.4 | estimate, scaled from the RTX 8000 eval pass | 1 | the launcher's joint-text and ledger changes |
| 7 | One cell per baseline at its published default (lion and meerkat, seed 9) in place of a tuning grid, which protocol rule 3 dropped; ToMe's official code did not import and SuperDiff left every table | rule 3 | `bigbatch` RTX 3090, mscluster48 (Slurm 84574) | 8.4 to 8.9 GB, read by the job's memory log | 2 | CO3 79.7 s and SuperDiff 165.6 s per image, measured | `co3_default/` and `superdiff_default/a_lion__x__a_meerkat/seed09/`, 2026-11-11 | 2 | done |
| 8 | CO3 at its published default on R1 and R3, and on Figure 5's four pairs at seed 10 | R1, R3, R2 | `bigbatch` RTX 3090 on mscluster48, 67, 60, 42, 86 and 89 (Slurm 84590, 84591, 84757, 84759, 84760, 84761); mscluster108 cards 0 and 1 (RTX 8000) for the last cells; Figure 5 on mscluster85 (RTX 3090) | about 9 GB | 1,104 R1 and 424 R3, counted from mscluster107; 8 for Figure 5 | 39 to 42 s per image on an RTX 3090, 83 s on mscluster48's slower card, 44 to 64 s on an RTX 8000, measured; 1,548 renders for 1,528 cells, 20 drawn twice where jobs met | the logs `eval_eval-co3-*.log` and `co3_tail2way_*.log` under `outputs/paper_review/logs/` | 2 | done, 2026-11-11 08:43 to 14:07 |
| 9 | PoE with a Langevin corrector on a labelled subset | R1†, R2 | 112 or 111 | 10 to 16 GB | subset not fixed | about 20 times a PoE image | estimate (protocol rule 6) | 3 | the two-Gaussian proof (route step 8) |
| 10 | SD 1.5: PoE, [Attend-and-Excite](../../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/06-attend-and-excite-bringing-a-missing-subject-back.md), [SynGen](../../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/09-syngen-binding-by-the-prompts-grammar.md), [Divide and Bind](../../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/10-divide-and-bind-attendance-plus-binding.md), [InitNO](../../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/11-initno-choosing-the-starting-noise.md), [CONFORM](../../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/07-conform-pulling-look-alike-subjects-apart.md) on the benchmark | R5, R13 | an RTX 8000 | not measured | 1,104 per method | not measured | none yet | 4 | route step 10 |
| 11 | three concepts: joint, product and ours on 4 triples at seed 10 | R11 | 112, 0 | 10 to 16 GB | 12 | under 0.1 | ours rate, measured | 4 | route step 11 |
| 12 | early-only training, if job 2 opens it | R9 | a whole card | about 25 GB (rank 32) | none | not set | waits on job 2's result | 5 | job 2 and route step 12's gate |
| 13 | every R1 method on [GenEval](../../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/05-geneval-a-detector-checks-the-objects.md)'s two-object and counting prompts and [T2I-CompBench](../../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/04-t2i-compbench-a-fixed-prompt-set-per-category.md)'s spatial and complex prompts, at each benchmark's own seeds and images per prompt | R14 | 112, 111 or 108 | 8 to 9 GB for joint, PoE and ours (measured in wave 1) | set by each benchmark's prompt files, counted when they are fetched | joint 5 s, PoE 6 s, ours 6.7 s per image on a Blackwell, times the count | rates measured in wave 1; counts not yet read | 2 | the benchmark prompt files fetched; PoE and ours run the two-object prompts only, and read not applicable on counting, spatial and complex (your decision, 2026-11-11) |

Wave 1 runs on four cards and none of the three trainings is stopped: 112 takes jobs 1 to 3 (4.3 to 5.1 h), 108 takes jobs 6 and 4 on its two cards, and 111 takes job 5 once its training ends. From a 07:00 start every wave 1 job is expected to finish between about 13:30 and 15:30 on 2026-11-11.

## Outcomes

- [x] ✅ check passed: `nvidia-smi -L` and a torch CUDA probe on each candidate node (`co3` on the RTX 8000 nodes, `co3_bw` on the Blackwell nodes), with [faulted cards](../../environment/known-failures.md#entry-id-poe-launch-002) excluded and the result written down. ✓ verified 2026-11-10: all five nodes answer with a working card (table above); 110 is excluded because another user holds it, not for a fault. `/datasets` is 44% used with 212 TB free `verified`
- [x] 📝 markdown created: a compute plan table, one row per job: node, GPU memory needed, expected hours (measured where a measurement exists, estimate otherwise), priority, and what it waits on. ✓ verified 2026-11-11: [the jobs table](#the-jobs), 13 rows, 5 of 6 wave 1 rows timed from step 4's logs `verified`
- [x] 💻 code ran: jobs queued in that order; biggpu's [one-job-per-user rule](../../runbook/running-things-on-the-cluster/launching-and-harvesting-a-run.md#3-harvest-what-is-running) is respected by starting the extra long jobs with `nohup` over SSH and checking them with `pgrep`. ✓ verified 2026-11-11: wave 1 ran on mscluster112, 111, 108 and 107 by the shared-device path, every launch after a yes; all six counts matched at 11:14 `verified`
- [x] ✅ check passed: every job's output lands on `/datasets`, with the [disk guard](../../environment/storage.md#the-disk-guard-rule-and-the-failure-that-motivates-it) reading that filesystem. ✓ verified 2026-11-11: every launcher's first check is `df` on `/datasets/mmolefe/poe_repair_min/outputs/paper_review`, and every output folder sits under it `verified`
