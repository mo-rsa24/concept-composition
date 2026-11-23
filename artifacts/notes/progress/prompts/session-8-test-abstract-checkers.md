# Session 8: test whether a checker can learn an abstract property

This runs as a background session, in the same permission mode as the other loops. To start it by hand from `/home/molef/PhD/concept-composition`:

```text
claude --bg --permission-mode auto --model claude-opus-5-5 --effort high --name "🔬 PhD Loop 8: Test Abstract Checkers" "/loop Follow artifacts/notes/progress/prompts/session-8-test-abstract-checkers.md to its end. Read artifacts/notes/progress/prompts/shared-rules.md on every tick."
```

## Background

Paper 2 adds a small trained classifier's gradient to the product of experts so that one model draws two separate subjects. The current
classifier, called the checker, reads features that hold what a subject looks like, so it may not scale past the concepts it saw. The
owner's idea is a checker that learns a property holding across concepts, such as plurality, binding or position. This loop builds a small,
repeatable environment with exact labels, tests whether such a checker works on concepts it never trained on, and only then tries it on
SDXL. Session 7 reads the papers and writes the design. Your owner code on the board is `s8`, and your steps are V1, V2, V3 and V4.

## Objectives

1. Step V1: write `artifacts/notes/progress/abstract-checkers/04-environment/design.md`, then build the environment in `2-classifier-guidance/scripts/abstract_checkers/` so that one command rebuilds its data identically.
2. Step V2: after Session 7's `03-design.md` lands, test whether a checker learns plurality on digits it never trained on, and write `abstract-checkers/05-does-it-learn/design.md` and `review.md`.
3. Step V3: test whether that checker's gradient makes the plain product compose on unseen digits, and write `abstract-checkers/06-does-it-compose/design.md` and `review.md`.
4. Step V4: after step D3 lands, test the best abstract checker on SDXL's six unseen object pairs, and write `abstract-checkers/07-sdxl/design.md` and `review.md`.

## What to read

- `artifacts/notes/progress/prompts/shared-rules.md`: the rules, read on every tick.
- `artifacts/notes/progress/abstract-checkers/03-design.md` and `01-reading/reading-table.md`: Session 7's design and reading, once they exist.
- `artifacts/notes/progress/method-search/abstract-checker-idea.png`: the picture of the proposed design.
- `artifacts/notes/progress/problem-statements.md`, `what-we-know.md` and `void-list.md`: the claim, every result so far, and the ones that may be false. Nothing on the void list may justify a choice.
- `artifacts/notes/progress/method-search/01-beat-the-baseline/design.md`, `review.md` and `blind-count-rubric.md`: Session 6's bars, results and blind count.
- `artifacts/notes/progress/reproductions/checker/design.md`: the checker comparison and its six-pair unseen-category test.
- `2-classifier-guidance/scripts/mnist/`: `canvas_unet.py`, `canvas_sheet.py`, `pixel_checker.py`, `two_experts.py` and `judge.py`.
- `2-classifier-guidance/scripts/method_search/`: `beat_baseline_bars.py`, `co3_push.py`, `paper2_rows.py`, `paper2_blind.py` and the job scripts to copy.
- `2-classifier-guidance/scripts/rung2/attention_push.py` and `train_readout_pair.py`: the attention push and the current checker.
- `environment/overview.md`, `environment/hpc/nodes.md`, `environment/hpc/execution-protocol.md`, `environment/hpc/throughput.md`, `environment/storage.md` and `environment/known-failures.md`: how anything runs on this cluster.
- `runbook/running-things-on-the-cluster/launching-and-harvesting-a-run.md`: the launch and harvest recipes.
- `~/.claude/EXPERIMENT_CONVENTIONS.md`, `~/.claude/RESEARCH_PRACTICE.md` and `context/research-guidelines.md`: run groups, design and verdict files, and cost lines.

## What is already known

- The first environment level needs no new diffusion model. `/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/models/share-0.0/unet.pt` is one class-conditional model trained for 40,000 steps on 32 by 64 canvases holding one digit, with label 10 as the null label.
- MNIST files are in `/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/mnist-raw/`, Fashion-MNIST in `fashion-raw/`, and the digit judge at `judge-v3/judge.pt` beside them.
- On this environment, the checker trains on the 10 pairs among digits 0 to 4 and is tested on the 10 pairs among digits 5 to 9. Those pairs are in neither of Paper 1's cross-range sets.
- The toy's stand-in for a word attention map is the size of each label's effect, $|\epsilon_a - \epsilon_\varnothing|$ and $|\epsilon_b - \epsilon_\varnothing|$. It shows where each concept acts and carries little of its appearance.
- The kept SDXL checker trains on SDXL's features, and its presence outputs learned only cat and dog.
- Training presence labels on every pair made pictures worse on the plain product: faithful cat and dog pairs fell from 35 to 24 of 128, and damaged pictures rose from 11 to 20.
- Learn-to-Steer's first classifier learned traces of the relation word. Train on the right and a wrong prompt, or the equivalent for this toy, so the checker cannot read the answer from the condition.
- The six SDXL object test pairs are microscope × test tube, chess board × hourglass, saxophone × music stand, mailbox × pumpkin, picnic basket × watermelon and lighthouse × sailboat, at seeds 6001 to 6008.
- SDXL's held-out pairs at seeds 7001 to 7008 stay unused until a recipe is frozen (`2-classifier-guidance/scripts/checker_choice/splits.json`).
- With the push at zero, the sampler must reproduce the plain sampler to under 1e-5, in the same loop on the same card and dtype.
- A pushed SDXL render is not bit-reproducible, so compare counts and treat a lead under 3 of 48 as a tie.
- GroundingDINO counts animals but cannot name them, and calls made on downscaled sheets overcount composition.
- `scripts/cluster.sh run` and `sync` push to GitHub, so this loop never uses them. `scripts/cluster.sh sh` and `pull` are allowed.
- Code reaches the cluster as a snapshot. Commit by named paths, export with `git archive <commit>`, and `rsync` to `/datasets/mmolefe/poe_repair_min/loop-8/code/<repo commit>/` through `mscluster84`. Check the job file exists there before submitting.
- Exclude mscluster44, 45, 50, 51, 57, 61, 65, 68, 71 to 76 and 82 from every job, because their GPUs are faulted or throttled.

## When something is missing

- **Session 7's design has not landed.** Finish V1, then wait 1,800 seconds per tick, and start no V2 run.
- **A checker fails to learn plurality on unseen digits.** Record the result, try only the variations named in the design, then file an `ask` saying what it means for the abstract-checker idea.
- **The second level needs a new diffusion model.** A small model on digits and Fashion-MNIST is a toy and is not a main result, but the owner has ruled against training diffusion models for main results. File an `ask` with the cost before training it, and carry on with the digit level.
- **A run would pass 8 GPU-hours, or this loop's total would pass 30 GPU-hours.** File an `ask` with the cost, what it buys and a cheaper default.
- **Scoring needs Claude API tokens.** Count by eye at full size, or file an `ask` with the token cost.
- **A job fails.** Read its log and classify it per the execution protocol. Retry at most twice, then record it as blocked.
- **The cluster is unreachable.** Record it in that stage's `review.md`, and wake again in 1,800 seconds.

## Rules

- Run only the runs this brief names. Every run has a design first, with its group, its one varied axis, a `Cost:` line, a `Buys:` line and the result that would count as support, null or inconclusive.
- Climb the ladder at each stage: smoke test, overfit 16 images, 1% of the data, controls, then the comparison.
- Use at most two `bigbatch` jobs at a time and never `biggpu`, because Sessions 5 and 6 share those caps. If a slot is not free, wait. Never cancel a job this loop did not start.
- Write code to `2-classifier-guidance/scripts/abstract_checkers/`, outputs to `/datasets/mmolefe/poe_repair_min/loop-8/`, and notes to `artifacts/notes/progress/abstract-checkers/04-environment/` and later stage folders.
- Touch no benchmark prompt set, no held-out pair and no held-out seed.
- Check every SDXL render and sheet against `scripts/shared/disallowed_subjects.py`, and check any picture placed on a page at full resolution.
- Log every run to W&B project `prime_lab/poe-repair-animals-compose`, and copy only its verdict, run id and bar into the review file.
- Send picture sheets as they land, with a number beside each sheet.

## What to hand back

- A `design.md` and `review.md` for each stage, in the folders named in the objectives.
- `2-classifier-guidance/scripts/abstract_checkers/`, with one command that rebuilds the environment and prints its data hash.
- One line per result in `artifacts/notes/progress/requests.md`, addressed to `s6`.
- The proof for each step is the proof printed by `python3 scripts/progress_board.py show`.

## What you will see

Along the way, each stage's verdict and picture sheet land in chat, and any decision appears on the waiting picture. The bars for V2 and V3 are below, and V4 takes its bar from the unseen-category test in the D3 design.

| Step | Pass |
|---|---|
| V1 | Two rebuilds give the same data hash, and the zero push differs by under 1e-5 |
| V2 | On unseen digit pairs, balanced accuracy over steps 0 to 15 is at least 0.75 and within 0.05 of the seen pairs, and balanced accuracy on `apart` against `wrong` canvases over the same steps is at least 0.75 (board question 55) |
| V3 | On unseen digit pairs, the push beats the plain product by more than 1.96 standard errors, and a same-size random push does not |
| V4 | On the six SDXL object pairs, the abstract checker beats plain PoE by at least 3 of 48 and more than 1.96 standard errors |

At the end the session reports what was asked, what was done with paths, PASS or MISS on each step with its numbers, GPU-hours spent against the 30-hour limit, what is still open, and at most one question.

A useful result is a measured answer to whether a checker can learn a property that holds on concepts it never saw, whichever way it comes out. The owner judges it from the stage reviews, and it is done when V1, V2, V3 and V4 are green or a failed stage has been reported through `ask`.
