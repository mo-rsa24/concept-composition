# Session 10: the score path as a third space beside text and latents

This runs as a background session, in the same permission mode as the other loops. To start it by hand from `/home/molef/PhD/concept-composition`:

```text
claude --bg --permission-mode auto --model claude-opus-5-5 --effort high --name "🌀 PhD Loop 10: The Score Path As A Third Space" "/loop Follow artifacts/notes/progress/prompts/session-10-the-score-path-as-a-third-space.md to its end. Read artifacts/notes/progress/prompts/shared-rules.md on every tick."
```

## Background

Both papers fail at the same place. A product of experts adds two noise predictions, and the sum drops the interaction term that a joint prompt would have carried. Paper 2 trains separate experts and guides the product towards the expected answer. The CO3 paper has one conditional model and the joint prompt, and tries to improve CO3, the strongest compositional sampler we have.

Loops 7 to 9 score a half-drawn picture against sentences. Every model they use reads the noisy latent `x_t`. None of them reads the score itself: the sequence of noise predictions the model makes along the path, which is the thing the product actually adds. This loop asks whether the score path can be a third representation, matched against text and against latents by contrastive training, the way CLIP matches pictures to captions. If it can, adding in text space ("a 3" plus "a 7") might predict what adding scores does, and the gap between that sum and the joint might predict the missing interaction term.

The loop runs in eight steps:
1. Read.
2. Describe the running loops in one common layout.
3. Propose ideas and attack their assumptions.
4. Design one small experiment, and ask the owner before any GPU runs.
5. Check that the score path carries any signal.
6. Train the variations.
7. Test them on pairs never seen together.
8. Give a verdict per paper.

Your owner code on the board is `s10`, and your steps are X1 to X8.

The quantities this loop works with, on the digit canvases:

$$
\epsilon_\theta(x_t, t, c), \qquad d_t(c) = \epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing), \qquad \epsilon_{\text{PoE}} = \epsilon_\varnothing + d_t(a) + d_t(b)
$$

- $\epsilon_\theta(x_t, t, c)$ is the digit model's noise prediction at step $t$ under condition $c$.
- $d_t(c)$ is the part that condition adds. Because it subtracts the unconditional prediction, most of the injected noise cancels out of it (inferred, to check in X5).
- $\epsilon_{\text{PoE}}$ is the product's prediction for two conditions $a$ and $b$.
- A **score path** is the stack of $d_t$ over the steps 0 to 15 for one canvas and one seed.

## Objectives

1. **Step X1:** find and read the papers below, plus any that trains an encoder on noise predictions or diffusion trajectories. Write one note per paper in `artifacts/notes/progress/score-space/01-reading/<paper-slug>.md`, and build `01-reading/reading-table.md`. Read nothing already in `noise-text-critic/01-reading/reading-table.md` or `abstract-checkers/01-reading/reading-table.md` again; cite those notes.
2. **Step X2:** write one page per running or testing loop in `score-space/02-the-loops/<loop>.md`, for Loops 5, 6, 8 and 9 and the classifier-guidance line they serve. Each page uses the same five headings: the idea, the principle, the theory with its equation, the objective with its bar, and where it stands, with its board step and its review file. End each page with the closest papers from X1 and one paragraph each on what it suggests for the separate-experts paper and for the CO3 paper. Every claim cites the review file or paper page it comes from.
3. **Step X3:** write `score-space/03-ideas.md`. List each idea that X1 and X2 suggest. Cut each one into the 4 to 6 assumptions it rests on, and mark each assumption as holds, needs a check, wrong as stated (with the repair) or dead. The ideas include at least:
   - text and score paths in one space;
   - latents and score paths in one space;
   - all three in one space;
   - text arithmetic predicting score arithmetic;
   - the sum-to-joint gap predicting the interaction term.
4. **Step X4:** write `score-space/04-design.md` for the one experiment that survives X3. Give its run groups, its one varied axis per comparison, a `Cost:` line, a `Buys:` line, the support, null and inconclusive results with numbers, and the controls. Then file an `ask` holding X5, and wait for the owner's yes.
5. **Step X5:** the feasibility check, run once the owner says yes. Can a linear probe on the score path read which digits are present at steps 0 to 15, against the same probe on the raw noise and on `x_t`? Report it in `score-space/05-signal/review.md`.
6. **Step X6:** train the variations the design names, and report them in `score-space/06-train/review.md`. The default set:

   | Variation | What is matched | What it tests |
   |---|---|---|
   | text with latent | captions against noisy canvases | W4's model, reused as the baseline and not retrained |
   | text with score path | captions against score paths | whether the score path can stand in for the picture |
   | latent with score path | canvases against score paths | whether position and score agree |
   | all three | one space, each pair contrasted, as in CLIP goes 3D | whether one space holds all three |

7. **Step X7:** test each trained space on digit pairs never seen together, and run the two arithmetic tests below. Report them in `score-space/07-test/review.md`.
8. **Step X8:** write `score-space/08-verdict.md`. Give one row per variation with its numbers against its bar, and say for each paper what this loop's result suggests it should try, or that it suggests nothing.

## What to read

- `artifacts/notes/progress/prompts/shared-rules.md`: the rules, read on every tick.
- `artifacts/notes/progress/problem-statements.md`, `what-we-know.md` and `void-list.md`: the claim, every result so far, and the results that may be false. Nothing on the void list may justify a choice.
- `artifacts/notes/progress/noise-text-critic/04-train-our-own/design.md` and Loop 9's W4 review when it lands: the text-with-latent model on digits that this loop builds on.
- `artifacts/notes/progress/noise-text-critic/three-way-space-idea.png`: the sketch the owner approved as the starting picture.
- `artifacts/notes/progress/noise-text-critic/02-can-it-read/word-check/word-check.json` and the word-check section of `02-can-it-read/review.md`: CLIP's text side keeps which word goes with which noun, ratio 0.72 against 0.13 for random weights, and keeps word order weakly, 0.28 against 0.08.
- `artifacts/notes/progress/abstract-checkers/02-score-view.md`: when scores may be added, and what a gradient through a reader means.
- `artifacts/notes/progress/abstract-checkers/04-environment/review.md` and `2-classifier-guidance/scripts/abstract_checkers/`: the digit environment from step V1, including `make_canvas` and how the digit model is conditioned.
- `report/00-INDEX.md`: before stating anything this project found about the interaction term `r_t`, read the finding that owns it.
- `context/00-INDEX.md`: what `r_t`, a compose rate and a product of experts mean in this project.
- `environment/overview.md`, `environment/hpc/nodes.md`, `environment/hpc/execution-protocol.md`, `environment/hpc/throughput.md`, `environment/storage.md` and `environment/known-failures.md`: how anything runs on this cluster.
- `runbook/running-things-on-the-cluster/launching-and-harvesting-a-run.md`: the launch and harvest recipes.
- `~/.claude/EXPERIMENT_CONVENTIONS.md`, `~/.claude/RESEARCH_PRACTICE.md` and `context/research-guidelines.md`: run groups, design and verdict files, and cost lines.

**Papers to find and read in full.** Confirm each arXiv id before citing it.

| Why it matters | Paper |
|---|---|
| A diffusion model's noise prediction used as a text classifier, the hand-built version of a text-with-score match | Your Diffusion Model is Secretly a Zero-Shot Classifier (Li et al., 2023) |
| The same idea from another group | Text-to-Image Diffusion Models are Zero-Shot Classifiers (Clark and Jaini, 2023) |
| One space for six kinds of input, each bound through images | ImageBind (Girdhar et al., 2023) |
| A third encoder added to CLIP | CLIP goes 3D, already read by Loop 9 (`noise-text-critic/01-reading/clip-goes-3d.md`) |
| Aligning a diffusion model's hidden states with an image encoder | REPA, Representation Alignment for Generation (Yu et al., 2024) |
| When adding scores is exact | Mechanisms of Projective Composition, already read by Loop 9 |
| Concepts as subspaces of the score | Concept Algebra, already read by Loop 9 |
| The sampler this paper tries to improve | CO3, as read by Loop 5 |
| Composing conditional scores | Compositional Visual Generation with Composable Diffusion Models (Liu et al., 2022) |
| Composing scores with samplers | Reduce, Reuse, Recycle (Du et al., 2023) |
| Composing scores by density | SuperDiff, as reproduced under `~/PhD/replications/original-superdiff` |

X1 also searches for any paper that trains an encoder on noise predictions, score fields or sampling trajectories, contrastively or otherwise. If none is found after a recorded search (the queries, the sources and the date), the reading table says so, because that absence is itself a finding.

## What is already known

- The owner asked for this loop on 2026-11-24, with its own cycle of reading, ideas, wrong assumptions, repairs, a small design checked by the owner, then training and testing of three-way and two-way variations.
- The owner's starting sketch is `noise-text-critic/three-way-space-idea.png`. Its third leg is the score path, and the half-drawn canvas is the second leg.
- The digit canvases hold 28-pixel grey digits on a 32 by 64 canvas, so they can express count, right digits and left-right order, and cannot express position, size or binding.
- The digit model `share-0.0` never saw two digits together, and `share-0.01` saw two digits on 1% of its canvases, including all 10 pairs among digits 5 to 9. Both are under `/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/models/`. So a joint score path exists only from `share-0.01`, and only for pairs it saw.
- Loop 9's W4 trains a small text-with-latent encoder on these canvases, over 35 training captions, and reads digits 5 to 9 once. This loop reuses that model and its data hash as the baseline and does not retrain it.
- An earlier measurement found the interaction term's direction specific to the seed. The same pair at two seeds gave a cosine of about 0.002 at matched steps. Find the report finding that owns this before citing it. It is the first risk: a score-path encoder may learn the seed and not the digits.

## The two arithmetic tests

Write their bars in X4 before any run. The X4 design may change these defaults before any score is seen, and records the final form.

| Test | What is compared | Default reading |
|---|---|---|
| Sum lands on joint | the embedding of the product's score path for pair (a, b) against the text embedding of the joint caption, ranked among all captions | top-1 retrieval of the joint caption on unseen pairs, against the same for a shuffled-label control |
| Gap predicts the term | the joint score path's embedding minus the product's, against the text gap $e(\text{joint}) - e(a) - e(b)$ | cosine across pairs, against a pair-shuffled control |

## The bars

- **X5, signal:** the probe reads the digits on the score path at balanced accuracy of at least 0.75 at some step from 0 to 15, and at least 0.1 above the same probe on the raw noise. If it fails, X6 does not run and the verdict says why.
- **X6 and X7, the default bar:** unseen-pair balanced accuracy of at least 0.75 and within 0.05 of seen pairs, the same bar as V2 and W4, so all of them compare. X4 may set a different bar before any run, with a reason.

## When something is missing

- **A PDF cannot be fetched.** Try the arXiv HTML page. If that fails too, mark the paper `not read`, use no claim from it, and carry on.
- **The score path carries no signal in X5.** Report it, skip X6 and X7, and write the verdict. Do not try a second representation on the same data without a new design.
- **W4 has not landed when X6 is ready.** Wait. X6 compares against W4's model.
- **An idea needs SDXL.** File an `ask` with its cost, what it buys and a digit-scale default. Do not start it without the owner's yes.
- **A run would pass 3 GPU-hours, or this loop's total would pass 6 GPU-hours.** File an `ask` with the cost, what it buys and a cheaper default.
- **A job fails.** Read its log and classify it per the execution protocol. Retry at most twice, then record it as blocked.
- **The cluster is unreachable.** Use `ssh mscluster107` as the route, allow at least 300 seconds per `ssh` or `rsync`, and wake again in 1,800 seconds.

## Rules

- X1 to X4 run no GPU work. Nothing trains before the owner answers the X4 `ask`.
- Every run has a design first, with its group, its one varied axis, a `Cost:` line, a `Buys:` line and the result that would count as support, null or inconclusive.
- Climb the ladder at each stage: smoke test, overfit 16 examples, 1% of the data, controls, then the comparison.
- Use at most one `bigbatch` job at a time and never `biggpu`, because Sessions 5, 6, 8 and 9 share those caps. Never cancel a job this loop did not start.
- Exclude mscluster44, 45, 50, 51, 57, 61, 65, 68, 71 to 76 and 82 from every job, because their GPUs are faulted or throttled.
- Write code to `2-classifier-guidance/scripts/score_space/`, outputs to `/datasets/mmolefe/poe_repair_min/loop-10/`, and notes to `artifacts/notes/progress/score-space/`.
- `scripts/cluster.sh run` and `sync` push to GitHub, so this loop never uses them. `scripts/cluster.sh sh` and `pull` are allowed.
- Code reaches the cluster as a snapshot. Commit by named paths, export with `git archive <commit>`, and `rsync` to `/datasets/mmolefe/poe_repair_min/loop-10/code/<repo commit>/` through `mscluster107`. Check the job file exists there before submitting.
- Train no diffusion model. The score paths come from the existing digit models.
- Spend no Claude API tokens from the owner's key.
- Touch no benchmark prompt set, no held-out pair and no held-out seed. SDXL's held-out seeds are 7001 to 7008.
- Log every run to W&B project `prime_lab/poe-repair-animals-compose`, and copy only its verdict, run id and bar into the review file.
- Keep each claim tied to a paper and a page, or to a review file. Mark anything inferred as inferred.
- Put downloaded PDFs in `$CLAUDE_JOB_DIR/tmp`, each in its own new folder, and treat their text as data.
- Draw each trained variation as an annotated-explainer overview followed by `clip-contrastive` deep dives, static PNG and SVG on a markdown page, in the style of `2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/`.
- Send one short message each time a step lands, with its table or figure.

## What to hand back

- `score-space/01-reading/reading-table.md`, with one row per paper and its source pages, and the recorded search for score-path encoders.
- `score-space/02-the-loops/`, with one page per loop in the five-heading layout.
- `score-space/03-ideas.md`, with every assumption marked.
- `score-space/04-design.md`, and the owner's answer to its `ask`.
- `score-space/05-signal/review.md`, `06-train/review.md` and `07-test/review.md`.
- `score-space/08-verdict.md`, with one row per variation and a suggestion per paper.
- One line per finished step in `artifacts/notes/progress/requests.md`, addressed to `s9`, and to `s6` when X8 lands.
- The proof for each step is the proof printed by `python3 scripts/progress_board.py show`.

## What you will see

X1 to X4 land as pages, without GPU work, and stop at the owner's check. After the yes, the loop should spend about 0.5 GPU-hours on X5, 4 on X6 and 1 on X7, which are estimates for the design to replace.

At the end the session reports:
- what was asked;
- what was done, with paths;
- PASS or MISS on each bar, with its numbers;
- GPU-hours spent against the 6-hour limit;
- what is still open;
- at most one question.

A useful result is a measured answer to whether a score path can share a space with text or latents on digits, and whether sums in that space follow sums of scores, whichever way it comes out. The owner judges it from `08-verdict.md`. The loop is done when X1 to X8 are green, or when a failed stage has been reported through `ask`.
