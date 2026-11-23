# Session 6: find the best method for each paper

This runs as a background session, in the same permission mode as the other loops. To start it by hand from `/home/molef/PhD/concept-composition`:

```text
claude --bg --permission-mode auto --model claude-opus-5-5 --effort high --name "🧪 PhD Loop 6: Find The Best Method" "/loop Follow artifacts/notes/progress/prompts/session-6-find-the-best-method.md to its end. Read artifacts/notes/progress/prompts/shared-rules.md on every tick."
```

## Background

The two papers ask whether a small trained classifier, called the checker, can make diffusion models draw two separate objects where the product of experts fuses them. Paper 1
composes models trained apart, and Paper 2 works inside one pretrained SDXL. The good MNIST and toy results came from settings nobody searched systematically, and the plan
went from one narrow checker comparison straight to the benchmarks. This loop finds the best method, checker, labels and settings for each paper on tuning data, then freezes one
recipe per paper before any benchmark run. Your owner code on the board is `s6`, and your steps are M1, M7, M5, D3, M2, D5, M3, M6, M8 and M4. Session 5 owns D2, the rival reruns.

## Objectives

1. Step M1: write `artifacts/notes/progress/method-search/01-beat-the-baseline/design.md`, then test whether each paper's method beats its simple baseline. Paper 1 is compared with sampling each MNIST expert alone and pasting the two results. Paper 2 is compared with CO3 alone, by adding the push to CO3 on SDXL.
2. Step M7: write `artifacts/notes/progress/method-search/01c-clevr-paste-first/design.md`, then train the small CLEVR experts M6 needs (2 and 3 experts at 64 pixels, 5,000 images per model, tuning positions only) and score sample-and-paste on them before any tuning, because Paper 1 tied paste on MNIST. M6 reuses these models. If paste reaches 7 objects, bring the owner a replacement test where pasting fails, as board question 54 says.
3. Step M5: write `artifacts/notes/progress/method-search/01b-balance-the-checker-data/design.md`, then train the current checker architecture on the current list and on a list that caps cat and dog and samples evenly per pair, compare them on the 12 tuning pairs at seeds 6001 to 6008, and pass the better list to D3.
4. Step D3: apply the owner's data rule from board question 37, then finish the comparison in `artifacts/notes/progress/reproductions/checker/design.md` and write the verdict to `reproductions/checker/review.md`.
5. Step M2: write `artifacts/notes/progress/method-search/02-pair-aware-or-separate-labels/design.md`, then compare a pair-aware checker with a checker that has one separate label per subject, on the architecture D3 chose.
6. Step D5: run the test in `artifacts/notes/progress/reproductions/inclusive-finetune-and-push/design.md` with the checker M2 chose, and write the outcome to `review.md` in that folder.
7. Step M3: write `artifacts/notes/progress/method-search/03-tune-the-winner/design.md`, then tune push strength, the steps it acts on, its cap and the checker's training-set size for the winning method, on tuning pairs and seeds only.
8. Step M6: write `artifacts/notes/progress/method-search/03b-small-clevr-test/design.md`, then test each paper's method on CLEVR with 2 and 3 experts at 64 pixels, 5,000 images per model, on tuning positions only, because neither the MNIST halves nor the SDXL-feature checker can run on CLEVR as written.
9. Step M8: write `artifacts/notes/progress/method-search/03c-head-to-head/design.md`, then compare this loop's best Paper 2 checker with Loop 8's count checker on the six unseen object pairs at seeds 6001 to 6008, 48 pictures each, reporting all six pairs and the five without lighthouse and sailboat, with CO3 alone beside both. Reuse V4's pictures where the settings match. A winner needs a lead of at least 3 of 48 and more than 1.96 standard errors. A tie goes to fewer damaged pictures, then to this loop's checker. If Loop 8 stops on a miss before V4, record that and this loop's checker stands.
10. Step M4: freeze one recipe per paper in `artifacts/notes/progress/method-search/recipe-paper-1.json` and `recipe-paper-2.json`, each naming the code commit, checker checkpoint, settings and the evidence behind them, including the small CLEVR test.

## What to read

- `artifacts/notes/progress/prompts/shared-rules.md`: the rules, read on every tick.
- `artifacts/notes/progress/problem-statements.md`: what each paper claims, what it may use and the evidence today.
- `artifacts/notes/progress/what-we-know.md` and `artifacts/notes/progress/void-list.md`: every result so far, and the ones that may be false. Nothing on the void list may justify a choice.
- `artifacts/notes/progress/scoreboard.json` and `artifacts/notes/progress/targets/`: the targets, the [fairness audit](../targets/fairness-audit.md) and the comparison protocol.
- `artifacts/notes/progress/reproductions/table.csv` and each rival's `review.md`: what the rival reruns measured.
- `artifacts/notes/progress/reproductions/checker/design.md`, `2-classifier-guidance/scripts/checker_choice/` and `2-classifier-guidance/scripts/rung2/`: the checker comparison, its selection rule and the existing checker code.
- `2-classifier-guidance/artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md`: how the current checker was improved, and what failed.
- `2-classifier-guidance/plans/33-experts-that-compose-at-inference.md`, `2-classifier-guidance/scripts/mnist/` and `2-classifier-guidance/artifacts/results/can-two-digits-compose-on-mnist/`: Paper 1's MNIST method, models and results.
- `/home/molef/PhD/replications/published-methods-on-our-pairs/scripts/reproductions/`: the CO3 port (`co3_common.py`, `co3_generate.py`), scorers and job scripts.
- `environment/overview.md`, `environment/hpc/nodes.md`, `environment/hpc/execution-protocol.md`, `environment/hpc/throughput.md`, `environment/storage.md` and `environment/known-failures.md`: how anything runs on this cluster.
- `runbook/running-things-on-the-cluster/launching-and-harvesting-a-run.md`: the launch and harvest recipes.
- `~/.claude/EXPERIMENT_CONVENTIONS.md`, `~/.claude/RESEARCH_PRACTICE.md` and `context/research-guidelines.md`: run groups, design and verdict files, cost lines, and the mistakes this project has already made.

## What is already known

- The diffusion models stay frozen. The owner ruled out fine-tuning or training a LoRA for the main results (board questions 44 and 45). Only classifiers are trained for composition.
- Inclusive captions appear only as one appendix paragraph with a comment on scale. Step D5 uses the pilot's two saved LoRAs and trains nothing (board questions 45 and 46).
- CLEVR stays at its planned scale, and its models are trained once from scratch on single objects (board question 45). The full CLEVR ladder belongs to steps E1 and E2 after the go or no-go. This loop trains only the small CLEVR models in step M6 (board question 48), budgeted at about 37 CPU node-hours and 9 to 12 GPU-hours.
- `artifacts/notes/progress/reproductions/CLEVR/design.md` and `targets/clevr-scaling-plan.md` hold the CLEVR protocol. Blender 2.78c and the generator are installed under `/datasets/mmolefe/poe_repair_min/loop-5/third_party/`, and 32-pixel render tiles take 5.46 seconds per single-object image and 4.50 per empty scene on 28 threads. Bradley's 12 printed positions are the evaluation positions, so M6 tunes on other positions and leaves those 12 held out.
- A CLEVR win at 7 objects needs 7 on both the automatic scorer and a hand count of 100 images (board question 42). Bradley's local patch model is not rebuilt, and its 5 stays a printed number (board question 49).
- T2I-CompBench and GenEval wins are judged on our own pictures against the best finished rerun, with the printed bar reported beside it. The stopped rival reruns do not come back (board question 47).
- The checker's training list holds 2,805 pictures over 216 pairs and 114 concepts. Cat and dog is 1,603 of them (57%) and about 60% of each batch, because the trainer samples its 10 sources evenly and 6 of them hold only cat and dog. The median pair has 5 pictures, 188 pairs have 10 or fewer, and 33 have no clean example. Keep and drop calls in the checker loop were judged mostly on cat and dog seeds.
- Paper 1's checker trains on expert samples only. Paper 2 holds out 25% of the Attend-and-Excite pairs by name hash (board question 37, about 6 GPU-hours).
- Paper 1's best result is 1,200 of 1,280 clean digit pairs with fixed halves, against 1,043 for the same method without pushes. It is on the void list, and the exact sample-and-paste control has not been run. On one shared canvas the best run draws 121 of 320 against a bar of 288.
- Paper 2's kept push draws 54 of 128 cat and dog pictures by eye, against 0 for the plain product and 101 for CO3. It was selected on seeds 6001 to 6128, so it is on the void list and those seeds are tuning seeds. The push has never been added to CO3.
- On T2I-CompBench Complex, at 988 prompts and 3 seeds, CO3 scores 0.3326, plain SDXL 0.2637 and RTD 0.2423 in our runs. RTD prints 0.4661 and CO3 prints 0.4406.
- The checker comparison uses 12 tuning pairs at seeds 6001 to 6008. The 12 held-out pairs at seeds 7001 to 7008 stay unused until a recipe is frozen (`2-classifier-guidance/scripts/checker_choice/splits.json`).
- The 20 unseen MNIST digit pairs were used in the void-listed final test. Tune Paper 1 on the five tuning pairs from plan 33, and choose nothing on the 20.
- With the push at zero, the sampler must reproduce the plain sampler to under 1e-5, in the same loop on the same card and dtype. Two separate loops can differ by about 3 grey levels with no bug.
- A pushed SDXL render is not bit-reproducible, and re-rendering one recipe moved its faithful count by about 2 of 96. Compare counts, and treat a lead under 5 of 96 as a tie.
- A guiding classifier needs negatives holding neither subject, because one trained without them called a fridge a cat.
- GroundingDINO counts animals but cannot name them, and calls made on downscaled sheets overcount composition. Check counts at full size.
- `scripts/cluster.sh run` and `sync` push to GitHub, so this loop never uses them. `scripts/cluster.sh sh` and `pull` are allowed.
- Code reaches the cluster as a snapshot. Commit by named paths, export with `git archive <commit>`, and `rsync` to `/datasets/mmolefe/poe_repair_min/loop-6/code/<repo commit>-<hub commit>/` through `mscluster84`. Rsync into a folder and check the job file exists there before submitting.
- `biggpu` allows one job per user and `bigbatch` allows six, and Loop 5 shares those caps. As of 2026-11-23, mscluster61's RTX 3090 is thermally throttled to about a quarter of normal speed. Exclude it along with the faulted nodes in Loop 5's job scripts, and read per-picture times in the log after every launch.
- The laptop's SSH refuses mscluster61 because its host key changed. Leave `known_hosts` alone. Read a running job's GPU with `srun --jobid=<id> --overlap -n1 nvidia-smi` from `mscluster84`.

## When something is missing

- **A method loses to its simple baseline.** Record the result, file an `ask` naming what it means for that paper with continuing as the default only if a concrete fix exists, and carry on with the other paper.
- **A run would pass 24 GPU-hours, or this loop's total would pass 79 GPU-hours.** File an `ask` with the cost, what it buys and a cheaper default, and carry on with other work.
- **Scoring needs Claude API tokens beyond D3's approved 1.46 million input and 67,000 output.** Count by eye at full size, or file an `ask` with the token cost.
- **A design choice is not settled by these sources.** Use the setting from the kept recipe, record it, and test it in M3 if it matters.
- **A job fails.** Read its log and classify it per the execution protocol. Retry at most twice, then record it as blocked.
- **The cluster is unreachable.** Record it in that stage's `review.md`, and wake again in 1,800 seconds.

## Rules

- Run only the runs this brief names. Every run has a design first, with its group, its one varied axis, a `Cost:` line, a `Buys:` line and the result that would count as support, null or inconclusive.
- Work the stages in order, because each one can change what the next should test. Keep each stage's number of settings small and named before it runs. Never run a full grid across axes.
- Climb the ladder at each stage: smoke test, overfit 16 images if something trains, 1% of the data, controls, then the comparison.
- Search only on tuning pairs and seeds. Touch no benchmark prompt set, and no held-out pair or seed, until a recipe is frozen. Never choose anything after seeing a held-out result.
- Rivals stay frozen at their authors' settings. Never tune a rival, and never change a bar. Session 4 owns the targets.
- Launch only through `sbatch`, inside the caps of one `biggpu` and six `bigbatch` jobs shared with Loop 5. If a slot is not free, wait. Never cancel a job this loop did not start.
- Write code to `2-classifier-guidance/scripts/method_search/` and `2-classifier-guidance/scripts/checker_choice/`, outputs to `/datasets/mmolefe/poe_repair_min/loop-6/`, and notes to `artifacts/notes/progress/method-search/`. D3's and D5's notes stay in their folders under `reproductions/`. Read Loop 5's models and data under `/datasets/mmolefe/poe_repair_min/loop-5/` and never write there.
- Check every render, score and sheet against `scripts/shared/disallowed_subjects.py`, and check any picture placed on a page at full resolution. CelebA images stay on the cluster.
- Log every run to W&B project `prime_lab/poe-repair-animals-compose`, and copy only its verdict, run id and bar into the review file.
- Send picture sheets as they land, with a number beside each sheet.

## What to hand back

- A `design.md` and `review.md` for each stage, in the folders named in the objectives.
- `artifacts/notes/progress/method-search/recipe-paper-1.json` and `recipe-paper-2.json`.
- One line per result in `artifacts/notes/progress/requests.md`, addressed to `s4`.
- The proof for each step is the proof printed by `python3 scripts/progress_board.py show`.

## What you will see

Along the way, each stage's verdict and picture sheet land in chat, and any decision appears on the waiting picture. At the end the session reports what was asked, what was done
with paths, the frozen recipe for each paper with its evidence, what was ruled out, GPU-hours spent against the 79-hour limit, what is still open, and at most one question.

A useful result is a recipe per paper that beats its simple baseline on tuning data, or a clear statement that a paper's method does not. The owner judges it from the recipes and stage
reviews, and it is done when M1, M7, M5, D3, M2, D5, M3, M6, M8 and M4 are green.
