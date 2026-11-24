# Session 12: where CO3's subjects borrow each other's parts, and what stops it

This runs as a background session, in the same permission mode as the other loops. To start it by hand from `/home/molef/PhD/concept-composition`:

```text
claude --bg --permission-mode auto --model claude-opus-5-5 --effort high --name "🩹 PhD Loop 12: Where CO3 Leaks" "/loop Follow artifacts/notes/progress/prompts/session-12-where-co3-leaks.md to its end. Read artifacts/notes/progress/prompts/shared-rules.md on every tick."
```

## Background

Paper 2 adds something to CO3, the strongest compositional sampler this project has measured, so that one SDXL model draws two subjects faithfully. CO3 already draws two separate bodies in all 96 of Loop 6's tuning pictures. What it gets wrong is which body carries which parts: one subject borrows the other's face, fur, beak or markings. The literature calls this semantic leakage.

Every Paper 2 attempt so far aimed at the outcome without knowing the mechanism. A checker's gradient push lowered CO3's faithful pairs from 11 to 7 of 96. Sentence scores read nothing at SDXL's steps 0 to 15. A score-path reader cannot hold more than the canvas it is computed from. The published methods that make a noise-aware score help mostly use it to choose or to train, not to push: Dynamic CFG chooses the guidance scale per step and scores 47.2 Gecko against 45.3 with the same evaluators' gradients.

This loop asks where CO3's leakage happens and whether a fix aimed at that place removes it. Then it goes back over Loops 8, 9 and 10 to find the small design issues that, once fixed, could change one of their verdicts.

The loop runs in six steps:
1. Read the leakage papers.
2. Measure what predicts borrowing on CO3's 12 tuning pairs.
3. Test one fix on the pairs that leak.
4. Redesign Paper 2's method from what steps 2 and 3 found.
5. Audit the designs of Loops 8, 9 and 10, and propose the reruns worth making.
6. Give the verdict and hand off.

Your owner code on the board is `s12`, and your steps are Z1 to Z6.

## Objectives

1. **Step Z1:** read in full the papers on leakage between subjects, and write one note per paper in `artifacts/notes/progress/leakage/01-reading/<paper-slug>.md`, plus `01-reading/mechanism-table.md`. The table has one row per paper: where it says leakage enters (text encoder, cross-attention, self-attention, layout, or the sum of scores), what it changes there, whether it needs training, which model it runs on, and what it measured on similar subjects, with pages. Use Loop 10's reader brief at `score-space/01-reading/tools/reader-brief.md` as the note template, with the questions changed to these. Cite and do not reread anything already in `score-space/01-reading/reading-table.md`, `noise-text-critic/01-reading/reading-table.md` or `abstract-checkers/01-reading/reading-table.md`.
2. **Step Z2:** measure what predicts borrowing on CO3. For each of the 12 tuning pairs, compute three candidate predictors and set each beside the pair's borrowed count from W2d: the cosine of the two subject names under CLIP's text encoder; the overlap of the two subjects' cross-attention maps at CO3's corrected steps; and the cosine of the two subjects' $d_t$ maps at those steps. Report it in `leakage/02-what-predicts-borrowing/review.md`. File an `ask` with the cost before the SDXL pass.
3. **Step Z3:** test one fix aimed at the mechanism Z2 points to. The default is Bounded Attention, a training-free method that keeps each subject's attention inside its own region, run on CO3's leaking pairs and scored on W2d's rubric against CO3 alone. Report it in `leakage/03-one-fix/review.md`. File an `ask` with the cost before any run.
4. **Step Z4:** write `leakage/04-paper-2-method.md`. Choose Paper 2's method from what Z2 and Z3 found and what the choosing test on board question 80 found: attention bounds, the checker choosing CO3's settings, an identity reward LoRA, or a combination. Give each candidate its evidence, its cost and the rival it must beat.
5. **Step Z5:** audit Loops 8, 9 and 10. Write `leakage/05-design-audit.md` with one row per design issue: which loop and step, what the issue is, its source, whether fixing it could change that step's verdict, the rerun that would test it, and the rerun's cost. Mark each issue as small (a fix within the design) or fundamental (the question itself needs changing). Start from the issues listed under "What is already known". File one `ask` listing the reruns worth making, with their total cost.
6. **Step Z6:** write `leakage/06-verdict.md`. Say for Paper 2 what the mechanism is, whether the fix worked with its numbers against its bar, the method Z4 chose, and the reruns Z5 proposes. Say in one paragraph what it suggests for Paper 1.

## What to read

- `artifacts/notes/progress/prompts/shared-rules.md`: the rules, read on every tick.
- `artifacts/notes/progress/problem-statements.md`, `what-we-know.md` and `void-list.md`. Nothing on the void list may justify a choice.
- `artifacts/notes/progress/noise-text-critic/02d-binding/design.md` and `part-binding-calls.csv` beside it: W2d's hand labels of CO3's 96 tuning pictures, with the borrowed counts per pair.
- `artifacts/notes/progress/method-search/01-beat-the-baseline/design.md`, `review.md` and `blind-count-rubric.md`: Loop 6's M1, the CO3 run, its pairs, seeds, settings and scoring.
- `artifacts/notes/progress/reproductions/CO3/design.md`: CO3's paper settings against the released script.
- `2-classifier-guidance/artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md`: what each checker reads at each step, including on the predicted clean image.
- `artifacts/notes/progress/score-space/09-verdict.md`, `11-the-checker-chooses-on-co3.md` and `../../ideas/which-directions-to-pursue/README.md`: Loop 10's verdict, the choosing test, and the directions map.
- `artifacts/ideas/beyond-the-product-of-experts/where-we-are-stuck.md`: what each line of work found, and that no push has restored identity.
- `artifacts/notes/progress/field-table.md`, rows for Be Decisive and Be Yourself, and `targets/rivals.md`.
- `report/00-INDEX.md` and `context/00-INDEX.md` before stating what this project found or what a term means.
- `environment/overview.md`, `environment/hpc/nodes.md`, `environment/hpc/execution-protocol.md`, `environment/hpc/throughput.md`, `environment/storage.md`, `environment/known-failures.md` and `runbook/running-things-on-the-cluster/launching-and-harvesting-a-run.md`.
- `~/.claude/EXPERIMENT_CONVENTIONS.md`, `~/.claude/RESEARCH_PRACTICE.md` and `context/research-guidelines.md`.

**Papers to find and read in full.** Confirm each arXiv id before citing it.

| Why it matters | Paper |
|---|---|
| Bounds each subject's cross-attention and self-attention so features do not leak; a partial rival in the field table | Be Yourself: Bounded Attention for Multi-Subject Text-to-Image Generation (Dahary et al., arXiv 2403.16990) |
| Predicts a layout from the starting noise so each subject stays apart; a direct rival in the field table | Be Decisive: Noise-Induced Layouts for Multi-Subject Generation (arXiv 2505.21488) |
| Contrasts attention maps to bind attributes and avoid neglect | Conform: Contrast is All You Need for High-Fidelity Text-to-Image Diffusion Models |
| Separates subjects and binds attributes with attention losses | Divide & Bind Your Attention for Improved Generative Semantic Nursing |
| Training-free attribute binding through structured text | Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis |

Z1 also searches for any paper that measures which subject pairs leak and why. If none is found after a recorded search, the mechanism table says so.

## What is already known

- CO3 runs at the released settings: CFG++ at guidance 0.8, 50 steps, 6 corrected steps, weight sharpness $\beta$ 0.9. It is deterministic once its starting noise is set: its corrector re-noises with the unconditional prediction (`composers/utils/attn_utils.py` lines 198 and 256 in the replications hub's `co3`), and the only random draw is the starting noise (`composers/Co3.py` line 373).
- On Loop 6's 12 tuning pairs at seeds 6001 to 6008, CO3 alone gives 11 faithful pairs of 96, and CO3 with the kept push gives 7 (M1's review).
- W2d's labels show borrowing is a property of the pair. Eagle and fox borrow on 0 of 8 seeds, crow and raven on 0, cheetah and cougar on 7, duck and goose on 7, puma and condor on 7. Similar-looking pairs do not simply leak, so name similarity alone cannot be the whole answer.
- On SDXL, the readout checker on U-Net features reads a clean pair at 0.63, 0.74, 0.78 and 0.79 at steps 0, 5, 10 and 15. A CLIP checker on the decoded predicted clean image reads 0.50 to 0.54 before step 10. Pushing through the predicted clean image moved texture and not layout (the checker loop).
- Two bodies can be pushed for: the kept push gives two bodies on 121 of 128 plain-product seeds. No push, third expert or checker has restored which body is which.
- **Design issues already seen, the starting list for Z5:**
  - Loop 8, V2 and the identity checker: the map reader read the asked pair's two maps, made positive and normalised to sum to 1, which drops the other labels, the sign and the size. On digits, all ten signed maps read the right pair against a wrong digit at 0.655, where the normalised maps read 0.511 (Loop 10's X5 review).
  - Loop 8, V2: the early window was the digit model's timesteps 999 to 693, where the canvas holds at most 8.7% of its signal; SDXL's steps 0 to 15 match the digit model's steps 14 to 25 (the V2 review).
  - Loop 9, W2 and W4: the sentence scores and W4's encoder read the noisy latent at steps where it holds little of the picture. W4 has since moved to steps 14 to 25 first.
  - Loop 10, X5: the probe read canvases noised from known pictures with one noise draw shared across steps, not the sampler's own path; the stacked-window read may only cancel that shared draw. The error readout used the true injected noise.
  - All three: each probe or push was designed before anyone checked how much of the picture the input still held at those steps.

## Rules

- Z1, Z4, Z5 and Z6 run no GPU work. Z2 and Z3 run only after the owner answers their `ask`.
- Every run has a design first, with its group, its one varied axis, a `Cost:` line, a `Buys:` line, and the support, null and inconclusive results with numbers.
- Climb the ladder at each stage: smoke test, a small read, controls, then the comparison.
- Use at most one GPU job at a time, `bigbatch` with a 48 GB card or `biggpu`, inside the shared caps. Never cancel a job this loop did not start. Exclude mscluster44, 45, 50, 51, 57, 61, 65, 68, 71 to 76 and 82.
- Write notes to `artifacts/notes/progress/leakage/`, code to `2-classifier-guidance/scripts/leakage/`, and outputs to `/datasets/mmolefe/poe_repair_min/loop-12/`.
- `scripts/cluster.sh run` and `sync` push to GitHub, so this loop never uses them. Code reaches the cluster as a snapshot: commit by named paths, export with `git archive <commit>`, `rsync` to `/datasets/mmolefe/poe_repair_min/loop-12/code/<commit>/` through `mscluster107`, and check the job file exists there before submitting.
- Reuse Loop 6's CO3 runner and M1's pairs, seeds and rubric without changing them; never edit another loop's files.
- Score pictures by eye at full size, blind, on W2d's rubric. Spend no Claude API tokens from the owner's key.
- Giraffe and tortoise is a tuning pair the subject guard holds back: count it, and keep it off every sheet, page and message.
- Touch no benchmark prompt set, no held-out pair and no held-out seed. SDXL's held-out seeds are 7001 to 7008.
- Log every run to W&B project `prime_lab/poe-repair-animals-compose`, and copy only its verdict, run id and bar into the review file.
- Before any probe or push, write down how much of the picture its input holds at the steps it acts on.
- Figures are static PNG and SVG on markdown pages. Send one short message each time a step lands, with its table or figure.

## What to hand back

- `leakage/01-reading/mechanism-table.md`, with one row per paper and the recorded search.
- `leakage/02-what-predicts-borrowing/review.md` and `03-one-fix/review.md`.
- `leakage/04-paper-2-method.md` and `05-design-audit.md`, with the owner's answers to their `ask`s.
- `leakage/06-verdict.md`.
- One line per finished step in `artifacts/notes/progress/requests.md`, addressed to `s6`, and to `s8` and `s9` for Z5's findings about their loops.

At the end the session reports what was asked, what was done with paths, PASS or MISS on each bar with its numbers, GPU-hours spent, what is still open, and at most one question. The loop is done when Z1 to Z6 are green, or when a failed stage has been reported through `ask`.
