# Prompt: route step 5, the compute plan and the first GPU jobs

Paste into a fresh session in `/home/molef/PhD/poe_repair_min`. It replaces the prompt block under step 5 of `plans/route-answer-the-iclr-reviews.md`, whose job counts predate the main-table change in step 4.

```text
/drip-execute-plan --request "Write the compute plan for the ICLR review response and launch the
first wave of GPU jobs on the cluster, after the user says yes to each launch command"
Model: Sonnet, medium effort. Box: 45 minutes of your time. Filing: on, because job logs and
contact sheets are the record of what ran.
Controls: piece 1 example (one job's row with its measured basis), piece 3 debug (a launch that
does not start), piece 4 test (that output counts match the plan).
Side routes: piece 3 /calibrate --predict <the launch command> (an 80% interval for runtime and
memory before each job).

Background
Submission 56173 to ICLR 2027 is being revised to answer three reviews; reviewer PrzE asked for
standard benchmarks, more baselines, the pair lists and more than two concepts. Steps 2 and 4
froze the protocol, the evaluation set and one adapter. This step places every GPU job on a node
with a priority and a priced estimate, then launches the jobs no gate blocks. The repo is
/home/molef/PhD/poe_repair_min and every cluster action goes through scripts/cluster.sh.

Objectives
1. Write the compute plan as a table in 1-lora-correction/report/paper-review/prze-10-compute.md: job, node,
   memory, images, hours with its basis (measured or estimate), priority, what it waits on.
2. Record an 80% runtime and memory prediction for each wave 1 job before it starts.
3. Launch wave 1 on mscluster112 and mscluster107 by the shared-device path, each after the user
   says yes to the exact command.
4. Poll each job, count its outputs against the plan, and report any job that is stuck.

What to read
- 1-lora-correction/report/paper-review/protocol.md (rules 1 to 10), held-out-set.md, layout-sheet.md (the slots
  R1 to R14 and the result file each one reads).
- artifacts/_shared/cross_pair_pool_configs/adapter_of_record.yaml: the one adapter and its seed.
- environment/hpc/execution-protocol.md (the shared-device path and its safety rules, all
  mandatory), environment/hpc/throughput.md, environment/hpc/nodes.md, environment/storage.md.
- scripts/cluster.sh and the launchers in scripts/ (review_step4_renders.sh,
  concept_pair_sheet.py, figure_candidates.py, joint_poe_adapter_grid.py): reuse, do not rewrite.
- 1-lora-correction/report/paper-review/prze-1, prze-2, prze-4, prze-11: the outcome lines each job feeds.

What is already known
- The adapter of record is v58-03-self at step 30,000, shared seed 10
  (adapter_of_record.yaml; protocol rule 1).
- The main table R1 is every SDXL method on the 276 Attend-and-Excite prompts
  (data/prompt-sets/attend-and-excite/a.e_prompts.json: animals 66, animals with objects 144,
  objects 66) at seeds 9 to 12: 1,104 images per method. Its methods are joint, PoE, CO3, ToMe,
  SuperDiff, PoE with a Langevin corrector (a labelled subset, protocol rule 6) and ours
  (layout-sheet.md, row R1). Prompts split into two concepts at " and " or " with " (rule 4).
- The ablation R3 is the same methods on the 53 held-out pairs (hard 3, easy but unseen 29, not
  competing 21) at seeds 9 to 16: 424 images per method (held-out-set.md).
- Step 4's census already rendered the joint prompt and the product for all 53 held-out pairs at
  seeds 9 to 16 (848 pictures) under protocol rule 2's settings, in
  /datasets/mmolefe/poe_repair_min/outputs/review_step4/census/. Reuse them; do not render again.
- Every method gets the same starting latent per seed (cache noise is per seed and shared across
  pairs; protocol rule 2), 1024 x 1024, 50 DDIM steps, guidance 7.5, fp16.
- Baselines get a tuning grid of at most six settings on the tuning pairs (lion and meerkat, camel
  and pelican, donkey and duck, chicken and rabbit) at seeds 9 to 16 before any evaluation run
  (rule 3). CO3, ToMe and SuperDiff need their official code set up first; that is not wave 1.
- One SDXL image takes about 11 to 16 s on the RTX PRO 6000 Blackwell in mscluster112 and about
  51 s on an RTX 8000 (environment/hpc/throughput.md). The Langevin corrector costs about 20 times
  a product image (an estimate until counted; rule 6).
- The user holds one biggpu slot: interactive job 83423 on mscluster107, two RTX 8000 of 49,152 MiB,
  1 day 13 hours elapsed at 2026-11-11 of a 3-day limit, so it ends around midday Thursday
  12 November. biggpu allows one job per user, so other jobs use the shared-device path.
- mscluster112 has one healthy RTX PRO 6000 Blackwell of 97,887 MiB; a torch matmul passes under
  co3_bw; /datasets has 212 TB free (verified 2026-11-10).
- Outputs go under /datasets/mmolefe/poe_repair_min/ only, never /home-mscluster.
- Dates: draft to supervisors Thursday 12 or Friday 13 November; OpenReview closes Wednesday 18.

Wave 1 (no gate blocks these)
1. Ours on R1: 1,104 images. Joint and PoE on R1: 1,104 each.
2. Ours on R3: 424 images (joint and PoE already exist from the census).
3. Ours, joint and PoE on Figure 5's four rows at seed 10 where not already rendered.
4. Ours over steps 0 to 10 and over steps 0 to 25 only (the product alone after that) on the 53
   held-out pairs at seeds 9 to 16: 424 images each, 848 in all. This is the early-steps test that
   feeds the failure-type figure R9 and opens or closes step 12's early-only training.
Later waves: baselines after their code and tuning grid (steps 6 and 7), the samplers after the
two-Gaussian proof (step 8), SD 1.5 (step 10), three concepts (step 11), gated trainings (step 12).

When something is missing
- If mscluster112 is taken by another user's job, run the Blackwell jobs on 110 or 111 only after
  probing them with nvidia-smi -L and a torch check, and ask the user first.
- If a job's launcher does not exist, write the smallest one beside the existing launchers and
  show it before running it.
- If the 107 allocation would end before a job finishes, put the job on 112 and say so.

Rules
- Show each launch command and wait for a yes. Submit nothing, kill nothing and pkill nothing by
  pattern. Never run find on /. Follow every safety rule in the execution protocol.
- Do not launch a gated training in this step.
- Commit before scripts/cluster.sh run, which refuses uncommitted tracked changes. Nothing is
  pushed to Overleaf.
- Every result row carries protocol rule 9's columns: model, method, adapter, sampler, steps,
  guidance, resolution, pair, seed, group, setting, nfe.

What to hand back
- The compute plan table with one prediction per wave 1 job.
- Run folders under /datasets/mmolefe/poe_repair_min/outputs/paper_review/<job>/ with logs beside
  them.
- New rows in "What we know so far" in plans/route-answer-the-iclr-reviews.md.
- Proof: for each wave 1 job, pgrep on the node shows it running, nvidia-smi shows its memory, and
  the output count matches the plan's expected count by the time it finishes (1,104, 1,104, 1,104,
  424, Figure 5's missing tiles, and 848 for the early-steps test).

What you will see
Along the way: the plan table, then each job's prediction and launch command, then each job's first
outputs, each followed by one question. At the end: what was asked, what was launched with paths,
PASS or MISS, what is open, and at most one question. PASS means every wave 1 job is running or
finished with its output count matching the plan. You judge from the table and the logs.
```

- `added:` the main-table and ablation image counts (from `protocol.md` rule 4 and `layout-sheet.md` R1 and R3) · `added:` the census renders to reuse (from the step 4 retro) · `added:` the adapter of record (from `adapter_of_record.yaml`) · `added:` the allocation's end time (from `squeue`, 2026-11-11) · `added:` protocol rule 9's columns
- `added:` the early-steps test as wave 1 job 4 (the route's job 2; step 12's gate reads it)
- `dropped:` the old job 1 of 960 images and the SD 1.5, sampler and three-concept rows from wave 1 (why: each waits on a later step's gate; they stay in the compute plan as later waves)
