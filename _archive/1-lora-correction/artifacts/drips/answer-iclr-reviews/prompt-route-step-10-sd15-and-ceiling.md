# Prompt: route step 10, the SD 1.5 comparison set and the ceiling test

Paste into a fresh session in `/home/molef/PhD/concept-composition`. It replaces the prompt block under step 10 of `plans/route-answer-the-iclr-reviews.md`.

```text
/drip-execute-plan --request --depth objective "Build the SD 1.5 comparison set with the official
Attend-and-Excite, SynGen, Divide and Bind, InitNO and CONFORM code unmodified, joint and plain
PoE, and the measured correction, and run the ceiling test that decides whether an SD 1.5 adapter
is trained"
Model: Sonnet, medium effort. Box: 45 minutes of your time. Filing: on.
Controls: piece 1 example (one pair's token indices), piece 2 prove (the measured correction on a
cell where PoE fails), piece 4 test (that every column used the same starting noise).
Side routes: none; no installed mode fits better than the walk itself.

Background
Reviewer UHz6 asked about Attend-and-Excite and CONFORM. Both are written for Stable Diffusion 1.x
and 2.x, and our adapter is on SDXL, so a fair comparison puts them on the same weights as a PoE
baseline and the measured correction, not on SDXL. This step builds that set and runs a test that
decides whether training an SD 1.5 adapter is worth the hours. The repo is
/home/molef/PhD/concept-composition.

Objectives
1. Add SD 1.5 to 1-lora-correction/scripts/same_rule_four_models.py, or a copy of it, for joint and plain PoE.
2. Write the measured-correction sampler for SD 1.x: joint minus PoE added back at full strength.
3. Run Attend-and-Excite from composition/aae/, and SynGen, Divide and Bind, InitNO and CONFORM
   from their official repositories, all unmodified at their defaults, on the 276 Attend-and-Excite
   prompts at seeds 9 to 12, 512 x 512, from the same starting noise (layout-sheet.md R5;
   protocol rule 2).
4. Write the ceiling bar into the protocol, run the test, and report PASS or MISS.
5. Build the picture grid R13 beside Table 4 from the same result file: R12's six prompts at seed
   9, columns R5's methods in R5's order, drawn with 1-lora-correction/scripts/paper_tile_figure.py at 5.5 in.

What to read
- composition/aae/: the official Attend-and-Excite code (README, run.py, the pipeline).
- 1-lora-correction/scripts/same_rule_four_models.py: joint prompt and PoE on four Stable Diffusion models.
- The CONFORM repository, github.com/gemlab-vt/conform, and arXiv 2312.06059.
- 1-lora-correction/report/paper-review/protocol.md, held-out-set.md and prze-1-scores.md.
- poe_repair/methods/_sampling.py: the repo's own PoE rule, to copy the rule exactly.
- environment/hpc/nodes.md: co3 and co3_bw interpreters; mscluster107 has two RTX 8000 of 49 GB.

What is already known
- Attend-and-Excite's official code is already in composition/aae/ and takes a seeded generator.
- The four-models script draws SD 1.x noise from a CUDA generator seeded with the seed, so one seed
  is one tensor within a column.
- CONFORM's official code loads Stable Diffusion v1.5 and runs from a notebook (searched
  2026-11-10); whether it accepts external latents is not yet known.
- The SD 1.5 numbers are a separate table (R5, appendix C.3), whose caption says SD 1.5 methods run
  from their official code and a number compares only within its table. Its rows are PoE, PoE with
  the measured correction, Attend-and-Excite, SynGen, Divide and Bind, InitNO, CONFORM, and ours
  only if step 12 trains the SD 1.5 adapter (otherwise the row is dropped and the limit stated).
- mscluster107's allocation (biggpu job 83423) ends around midday Thursday 12 November; a job that
  would not finish before then runs on mscluster112 instead.
- The SD 1.5 adapter training is gated on this test and would take 7 to 10 hours on the Blackwell.
- The tuning pairs and evaluation set are frozen by step 4 (held-out-set.md); never tune on the 276
  benchmark prompts or the held-out pairs.

When something is missing
- If SD 1.5 weights are not on the cluster, search the Hugging Face hub for the maintained mirror,
  show the user the repository, and download only after the user says yes.
- If a method's official code cannot take external latents, run it on its own noise from the same
  integer seed and mark its row mixed with a dagger (protocol rules 2 and 7).
- If the evaluation pairs are not frozen, stop and say step 4 is owed.

Rules
- Do not edit Attend-and-Excite or CONFORM code. Call them as released, from a thin script.
- Token indices are computed by one function and saved, never hand-picked.
- Add this step's GPU jobs to 1-lora-correction/report/paper-review/prze-10-compute.md and launch each after the
  user's yes.
- The ceiling bar is in the protocol before the test runs.

What to hand back
- The three new scripts and the saved token-index file.
- The SD 1.5 images under /datasets/mmolefe/poe_repair_min/outputs/paper_review/sd15/ and a contact
  sheet with a card entry.
- The ceiling test's PASS or MISS line, and the decision it implies for the SD 1.5 adapter.
- Rows added to "What we know so far" in plans/route-answer-the-iclr-reviews.md.
- Proof: the test script prints PASS or MISS with the rate, and the board redraws with --step 10.

What you will see
Along the way: one prompt's token indices, then a contact sheet of the SD 1.5 methods, then the
ceiling test's rate, each followed by one question. At the end: what was asked, what was done with
paths, PASS or MISS against the written bar, what is open, and at most one question. You judge from
the contact sheet and the rate.
```

- `added:` SynGen, Divide and Bind and InitNO (`protocol.md` rule 2, `layout-sheet.md` R5) · `added:` the 276 benchmark prompts at seeds 9 to 12 as the inputs · `added:` the R13 grid · `added:` the 107 allocation's end time
- `dropped:` "the evaluation pairs" as the inputs and "not matched" as the mark (why: the table reuses R1's prompts; the protocol's word is mixed)
