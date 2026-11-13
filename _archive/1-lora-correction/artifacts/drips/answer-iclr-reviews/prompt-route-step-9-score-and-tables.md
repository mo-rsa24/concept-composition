# Prompt: route step 9, score every image and build the tables

Paste into a fresh session in `/home/molef/PhD/concept-composition`. It replaces the prompt block under step 9 of `plans/route-answer-the-iclr-reviews.md`, written before the main table moved to the 276 benchmark prompts.

```text
/drip-execute-plan --request --depth objective "Score every image from the evaluation, early-steps,
baseline and SD 1.5 jobs with CLIP score, ImageReward, BLIP-VQA, GenEval and T2I-CompBench, and
build
the main table, the held-out ablation, the standard-benchmark table, the paired differences
and the picture grids beside them, with the bar written first"
Model: Sonnet, medium effort. Box: 60 minutes of your time, at three checkpoints. Filing: on.
Controls: piece 1 prove (each scorer on the two small-object images), piece 3 example (one cell's
row traced from the table to its image), piece 4 break (an image a scorer should fail on).
Side routes: piece 5 /analyze-run <the W&B run, if any> only for a training run, none for sampling.

Background
Reviewer PrzE found no quantitative metric in the paper and asked for T2I-CompBench, GenEval, CLIP
score and ImageReward against standard baselines. The images exist from steps 5 and 8 on the frozen
evaluation set. This step scores them under the protocol, builds the tables the paper needs and
fills
the paper's table slots. The repo is /home/molef/PhD/concept-composition.

Objectives
1. Install and verify each scorer on the two small-object images from step 7, and write each one's
   version and settings into the protocol.
2. Write the supports, null, inconclusive and reported-as-is bar into the protocol before any score.
3. Score every image and write one long results file with the axis columns on every row.
4. Build the tables into their slots' result files (layout-sheet.md): the main table R1
   (main_table.csv, model SDXL), the held-out ablation R3 with 95% intervals
   (held_out_ablation.csv), the standard benchmarks R14 (standard_benchmarks.csv), and the paired
   differences.
5. Build the picture grid beside each table from the same result file (protocol rule 8): R12
   beside R1 (two prompts per set at seed 9, every method) and R4 beside R3 (one pair per group at
   seeds 9 and 10), drawn with 1-lora-correction/scripts/paper_tile_figure.py at 5.5 in, labels 8 pt, each within
   its slot's height on the layout sheet.
6. Fill the slots through python3 1-lora-correction/scripts/refresh_paper_slots.py --compile --gallery and check the
   paper still compiles with no slot too wide.

What to read
- 1-lora-correction/artifacts/drips/answering-the-reviewers/answer-iclr-reviews/read-the-scorers/COMBINE_MAP.md: what each metric measures.
- 1-lora-correction/report/paper-review/protocol.md, held-out-set.md, layout-sheet.md and prze-10-compute.md.
- composition/focus/eval_metrics.py: the CLIP and ImageReward scorers already written here.
- The run folders under /datasets/mmolefe/poe_repair_min/outputs/paper_review/.
- environment/hpc/nodes.md: the co3 and co3_bw interpreters and the PyPI speed on the nodes.
- The CO3 paper, arXiv 2509.25940, Table 1, for the table's shape.

What is already known
- The main table R1 is every SDXL method on the 276 Attend-and-Excite prompts at seeds 9 to 12; its
  column groups are animals (66 prompts: Comp., CLIP, BLIP, IR), animals with objects (144: CLIP,
  BLIP, IR) and objects (66: CLIP, BLIP, IR), then NFE; CO3's printed Table 1 sits under it as a
  grey cited row, never mixed into ours (protocol rules 4 and 10; layout-sheet.md R1).
- The ablation R3 is the same methods on the 53 held-out pairs at seeds 9 to 16, Comp. with a 95%
  interval per group (hard 3, easy but unseen 29, not competing 21).
- R14 is GenEval two-object and counting and T2I-CompBench spatial and complex, scored by their
  official evaluators on each benchmark's own prompts and seeds. Those prompts are not in step 5's
  wave 1: add their renders to 1-lora-correction/report/paper-review/prze-10-compute.md as a job, with its image
  count, and launch it after the user's yes.
- Every method shares one starting latent per seed; one frozen adapter, v58-03-self at step 30,000
  (adapter_of_record.yaml). Result rows carry protocol rule 9's columns.
- Our compose count (Comp.) is a GroundingDINO instance count, count of "animal" at least 2; it
  counts animals only and cannot tell a cat from a dog, so it runs on the animals set and the
  held-out animal pairs only.
- The cat and dog joint prompt is wrong on 4 of 8 held-out seeds, so those seeds are unjudgeable
  for any score that uses the joint render.
- GenEval and T2I-CompBench have no code in this repo today.
- CO3 does not state prompts per set, seeds, resolution or its metric code; do not copy its numbers.
- Seeds within a pair are not independent; intervals resample pairs, then seeds.

When something is missing
- If a scorer will not install in its time box (30 minutes each), skip it, record the reason in
  the protocol, and ask the user whether to try a second approach.
- If an image is missing for a cell, list the cell, do not impute it, and report the count.
- If the frozen evaluation set or checkpoint is not there, stop and say steps 4 and 5 are owed.

Rules
- The bar is in the protocol before the first score is computed.
- Never substitute a different metric under a scorer's name.
- Show the user each checkpoint's result before the next stage: the scorer checks, the CO3-style
  table, then the paired differences.
- Launch scoring and render jobs as rows of the step 5 compute plan, each after the user's yes.

What to hand back
- The results file under /datasets/mmolefe/poe_repair_min/outputs/paper_review/scores/ and a copy
  of the small tables under 1-lora-correction/report/paper-review/.
- The three tables and the paired differences, in their slots, with sidecar files.
- Rows added to "What we know so far" in plans/route-answer-the-iclr-reviews.md.
- Proof: python3 1-lora-correction/scripts/refresh_paper_slots.py --check prints PASS, the paper compiles, and the
  board redraws with --step 9 ticking the PRZE-1 and PRZE-3b lines.

What you will see
Along the way: each scorer's result on the two small-object images, then the CO3-style table, then
the paired differences, each followed by one question. At the end: what was asked, what was done
with paths, SUPPORTS, NULL or INCONCLUSIVE against the written bar, what is open, and at most one
question. You judge from the tables and the grids.
```

- `added:` the three tables by slot with their result files and column groups (`layout-sheet.md` R1, R3, R14) · `added:` the R14 render job missing from step 5's wave 1 · `added:` the picture grids beside each table (protocol rule 8) · `added:` the adapter of record and rule 9's columns
- `dropped:` "seeds 9 to 16" as the seeds of every image (why: the main table runs at seeds 9 to 12)
