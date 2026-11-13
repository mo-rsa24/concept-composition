```text
/drip-execute-plan --request "Write the comparison protocol and the figure and table layout sheet
for the ICLR review response, and set up placeholder slots that refill from result files"
Model: Opus, medium effort. Box: 60 minutes of your time. Filing: tied.
Controls: piece 1 example (one worked row of the protocol on a real pair), piece 2 draw (the table
and grid layouts), piece 3 debug (the refill script's first run).
Side routes: piece 2 /plan-figures on the figure register; piece 3 none.

Background
Reviewers of ICLR 2027 submission 56173 asked for scores against standard benchmarks, more
baselines and aggregate rates over held-out pairs. Before any run, the comparison rules and the
shape of every new figure and table are fixed, so no result can be called an unfair comparison.
The repo is /home/molef/PhD/concept-composition on the laptop. This is step 2 of the route in
plans/route-answer-the-iclr-reviews.md. Step 1 found that the paper's source is the Overleaf
export, not either local copy, and that Figure 5 mixes three adapters.

Objectives
1. Unpack _ICLR2027__LoRA_Plurality.zip into paper/iclr-56173/ and confirm it compiles unchanged.
2. Write 1-lora-correction/report/paper-review/protocol.md with the rules every result must obey.
3. Write 1-lora-correction/report/paper-review/layout-sheet.md with one row per new figure and table.
4. Add a row per slot to paper/iclr-56173/figures.md and a placeholder at final size in its tex.
5. Write 1-lora-correction/scripts/refresh_paper_slots.py, which rebuilds a slot from its result file and compiles.

What to read
- 1-lora-correction/artifacts/drips/answering-the-reviewers/answer-iclr-reviews/ground-check.md: step 1's findings on nodes, the adapter
  and the paper source.
- 1-lora-correction/report/paper-review/board.md and the pages prze-1, prze-2, prze-3a, prze-5, prze-8, prze-9 and
  prze-11 beside it: the outcome lines this step and later steps must meet.
- plans/route-answer-the-iclr-reviews.md: the table "What we know so far", read whole.
- environment/paper.md: how the paper builds here.
- paper/iclr-56173/iclr2027_conference.tex lines 30 to 90 after unpacking: the placeholder colours
  and the \figurepanel macro already in the source.
- paper/iclr-56173/figures.md: the figure register that ships with the export.
- 56173_Restoring_plurality_in_c.pdf and Reviews.pdf at the repo root: the paper and the reviews.

What is already known
- The submitted source is the Overleaf export _ICLR2027__LoRA_Plurality.zip at the repo root. Its
  14 section titles equal the PDF's, and every sentence checked against the PDF was found in it
  (ground-check.md, section 3).
- paper/iclr/ is an older draft from 2026-09-21. paper/overleaf-iclr/ is an Overleaf clone last
  pulled before submission, with uncommitted edits. Edit neither (ground-check.md).
- The adapter of record is v58-03-self at step 50,000, W&B run qtk5gi37: 48 training pairs,
  1,000 epochs, trained on step indices 0 to 9 of 50 DDIM steps (prze-3a-pair-lists.md).
- Figure 5's adapter column uses three runs: v57-11b-x0-self-decay at step 50,000 for cat and dog,
  v58-05-self-contrast at step 15,000 for elephant and penguin, and v58-03-self at step 50,000 for
  tent and lighthouse (prze-3a-pair-lists.md). The protocol must fix one adapter for every figure.
- The v58 runs held out five pairs at training time: bison and horse, cat and dog, piano and
  candle, skateboard and helmet, elephant and penguin. pair_pool_v58.yaml now lists four
  (prze-3a-pair-lists.md).
- Tent and snowy mountain, and lighthouse and stormy sea, are in neither the training list nor any
  held-out list (prze-3a-pair-lists.md).
- Seeds 1 to 8 and 17 to 58 were trained on; seeds 9 to 16 never were (the run's dataset_meta.json).
- A separate session is testing species fidelity on four training pairs at seeds 9 to 16. Its
  pairs (lion and meerkat, camel and pelican, donkey and duck, chicken and rabbit) are tuning data.
  The protocol must keep them out of the evaluation set.
- The PDF says the adapter sits on cross-attention only (pages 1 and 7) and acts at every
  denoising step (page 1). The run adapted self- and cross-attention on steps 0 to 9 only.
- UHz6's early-steps question is now answered by comparing an adapter trained on all 50 steps with
  the paper's steps-0-to-9 adapter (prze-11-early-training.md).
- The CO3 paper (arXiv 2509.25940) scores the Chefer et al. prompt sets animals, animals-objects and
  objects with BLIP-VQA and ImageReward at 50-step DDIM and guidance 5.0. It does not state prompts
  per set, seeds, resolution or metric code. We run every method ourselves and never copy numbers.
- Some prompts in those sets may match the 48 training pairs. The main table excludes them.
- The user's rules: same starting noise per pair and seed, tuning on a set that excludes every
  evaluation pair, shown images chosen by a fixed rule, network evaluations per image as a column,
  comparisons that differ in more than one thing marked mixed, and every number paired with a
  qualitative grid from the same run.
- Attend-and-Excite and CONFORM run on SD 1.5 with their official code, unmodified.
- The Langevin corrector costs about 20 times a product image at 20 steps (an estimate). Correctors
  run on a labelled subset, and the product and the adapter are reported on that subset too.
- The paper builds with Tectonic, never pdfLaTeX. On mscluster it is
  /home-mscluster/mmolefe/.local/bin/tectonic, and the first build on a node needs network
  (environment/paper.md). The laptop has no tectonic (which tectonic, checked 2026-11-10).
- So every build runs on mscluster. First copy the folder there:
  rsync -az paper/iclr-56173/ mscluster84:/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/paper/iclr-56173/
- Then build there:
  scripts/cluster.sh sh "cd paper/iclr-56173 && /home-mscluster/mmolefe/.local/bin/tectonic iclr2027_conference.tex"

When something is missing
- If the rsync or the build on mscluster fails, show the error and ask the user before trying
  another route.
- If the export does not compile before any edit, stop, show the error, and ask whether to fix the
  build or write the layout sheet only.
- If a placeholder cannot hold a slot at the required size, mark the slot unresolved in the sheet
  and ask the user how wide it should be.

Rules
- Run no GPU job on the cluster. The only cluster actions are the rsync and the Tectonic build.
- Change no result file.
- Edit no body text of the paper. Touch only slots, placeholders and the figure register.
- Write the paper source only under paper/iclr-56173/. Do not touch paper/iclr/ or
  paper/overleaf-iclr/.
- Push nothing to Overleaf. The user uploads the revised source when the paper is revised.
- Show protocol.md and wait for a yes before the layout sheet.
- Commit nothing until the user says so.

What to hand back
- 1-lora-correction/report/paper-review/protocol.md and 1-lora-correction/report/paper-review/layout-sheet.md.
- New rows in paper/iclr-56173/figures.md, status reserved, each with its claim, layout and result
  file.
- 1-lora-correction/scripts/refresh_paper_slots.py and the compile log showing the paper builds with every slot as
  a placeholder.
- New rows in "What we know so far" in plans/route-answer-the-iclr-reviews.md, and a diff for any
  later step whose prompt names paper/iclr/ as the source, applied after the user's yes.
- Proof: python3 1-lora-correction/scripts/refresh_paper_slots.py --check prints PASS and the slot count, and
  python3 ~/.claude/skills/skill-route/scripts/board.py 1-lora-correction/report/paper-review/board.md --step 2
  redraws the board with the PRZE-8 and PRZE-9 lines ticked.

What you will see
Along the way: the compile of the untouched export, the protocol page, the layout sheet with a
drawn sketch of each table and grid, then the first refill of one slot, each followed by one
question. At the end: what was asked, what was done with paths, PASS or MISS, what is open, at most
one question. PASS means the export compiles with every slot as a placeholder and one slot refills
from a result file. You judge by opening the compiled PDF, and it is done when you agree the
protocol.
```
