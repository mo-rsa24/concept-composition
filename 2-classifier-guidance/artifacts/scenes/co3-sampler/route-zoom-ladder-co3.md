# 📚 CO3 from the whole run down to the code
<!-- template: exempt. Reason: a skill-route runbook built to route-file-template, not a plan -->

> [!NOTE]
> **You will hold:** the CO3 ladder in this folder, read from [the reading page](index.md) · **Time:** levels 0, 0.5
> and 3 are built; the worked case waits on the Z2 smoke run · **Stop point:** every rung the owner chose is built or
> marked `skipped` in `ladder.json` · **Outcome board:** none for this ladder · **Background:** [the folder card](README.md)

Built to: route-file-template v3

## Table of contents

- [Quick context: where you are](#quick-context-where-you-are)
- [What we know so far](#what-we-know-so-far)
- [What you will achieve](#what-you-will-achieve)
- [The route at a glance](#the-route-at-a-glance)
- [The two lanes](#the-two-lanes)
- [Schedule](#schedule)
- [Tasks](#tasks) (the AI runs these)
- [Instructions](#instructions) (you do these by hand)
- [Quizzes and feedback](#quizzes-and-feedback)
- [Next step](#next-step)

## Quick context: where you are

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-we-know-so-far) ➡️

**Goal:** read CO3 well enough to argue with it in the paper and to wire it into the Loop 12 runs.
**Objective (the whole chain):** every CO3 equation drawn on the part it computes, and every line of the authors'
sampler placed under its equation, with each paper-code difference named by page and line.
**Outcome (the whole chain):** this folder's reading page, five figures, the code walk, and `ladder.json`.
**Outcome board:** none; `ladder.json` records each rung's state.
**Background:** the folder card, [README.md](README.md), says how every file was made.

## What we know so far

Navigation: ⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#what-you-will-achieve) ➡️

| Fact | Status | Shown by | Changes |
|---|---|---|---|
| The release runs co3-hybrid: resample at t = 981, 961, 941 and correct at 921, 901, 881 | found | `co3_numbers.json`, matched to `source/run-log-excerpt-l6-m1-p2-93468.txt`, 2026-11-24 | steps 4 and 5 |
| Passes per step are 10, 10, 5, 5, 5, 5; $\beta$ is 1.0 at t = 981 and 0.9 after | found | `Co3.py` 306 and 307, 2026-11-24 | step 5 |
| The corrector's rescale differs from Algorithm 3 (largest entries in place of lengths, plus a second rescale) | found | `attn_utils.py` 252, 255, 257 against p. 6 | step 5 |
| The Z2 smoke run's folder does not exist yet | open | `ls /datasets/mmolefe/poe_repair_min/loop-12/z2/smoke/`, 2026-11-24 | step 4 |
| The project's loader turns the corrector's logging off | found | `co3_common.py`, `model.latentCorrector.verbose = False` | step 4 |

## What you will achieve

Navigation: ⬅️ [Previous](#what-we-know-so-far) | 📋 [TOC](#table-of-contents) | [Next](#the-route-at-a-glance) ➡️

| Stage | Tier | You will have | You will draw, from a small example | A worked example | Exercises | Quiz |
|---|---|---|---|---|---|---|
| 0 the whole run | read | the overview | the 50-step loop with its six CO3 steps | the eagle and fox render | none this round | none |
| 0.5 the parts | read | four figures | Algorithm 2, Algorithm 3, equation 15, the schedule | equation 15 at gaps 0, 1, 2, 5 | none this round | none |
| 3 the code | read | the code walk | none | none until the Z2 smoke run | none this round | none |
| **Outcome** | | **the ladder in this folder** | | | | **`ladder.json` shows every chosen rung built** |

## The route at a glance

Navigation: ⬅️ [Previous](#what-you-will-achieve) | 📋 [TOC](#table-of-contents) | [Next](#the-two-lanes) ➡️

▶ **Recommended:** `numbers → level 0 → level 0.5 ‖ level 3 → worked case (after Z2) → level 2 → level 1 → close`
(the zoom-ladder order, with the miniature and the illustrations set aside by the owner)

## The two lanes

Navigation: ⬅️ [Previous](#the-route-at-a-glance) | 📋 [TOC](#table-of-contents) | [Next](#schedule) ➡️

| Stage | The AI lane builds | Your lane learns | Quiz | Confidence bar | You rebuild | Compare on |
|---|---|---|---|---|---|---|
| 0 to 3 | the figures and the code walk | reading the page, one level at a time | none this round | you can say what each of the six CO3 steps does | Algorithm 2's step on paper | the resampler figure |

## Schedule

Navigation: ⬅️ [Previous](#the-two-lanes) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

| Block | Run | Checkpoint at the end of the block |
|---|---|---|
| unattended, 2026-11-24 | steps 1 to 3 | built, every figure passed the line check |
| after the Z2 smoke run lands | step 4 | the overview's slot filled from real numbers |
| when the owner asks | steps 5 to 7 | the miniature, the illustrations, the close |

## Tasks

*For the AI to run. Paste each prompt into a session.*

Navigation: ⬅️ [Previous](#schedule) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

### 1. ✅ Level 0: the whole run, annotated

**Outcome:** `co3-overview.png` and `.svg`, drawn by `draw_overview.py`. **Built.**

```text
/zoom-ladder CO3 level 0 only: rerun python3 co3_numbers.py then python3 draw_overview.py in
2-classifier-guidance/artifacts/scenes/co3-sampler/ and confirm the line check passes.
```

### 2. ✅ Level 0.5: one figure per part

**Outcome:** `co3-resampler`, `co3-corrector`, `co3-weights`, `co3-schedule` (PNG and SVG). **Built.**

```text
In 2-classifier-guidance/artifacts/scenes/co3-sampler/, run python3 draw_resampler.py,
draw_corrector.py, draw_weights.py and draw_schedule.py; each prints a line ending "no line crosses
a node, box or label" when right.
```

### 3. ✅ Level 3: the code walk

**Outcome:** `code-walk.md`, written by `gen_codewalk.py`. **Built.**

```text
In 2-classifier-guidance/artifacts/scenes/co3-sampler/, run python3 gen_codewalk.py; it stops if any
cited line in /home/molef/PhD/replications/co3 has moved from commit 35ea972.
```

### 4. 🔧 The worked case with a real latent

**Objective:** fill the overview's marked slot with one real run's numbers. **Starts from:** the Z2 smoke run.

```text
/zoom-ladder CO3: fill the worked-case slot on co3-overview.png and index.md in
2-classifier-guidance/artifacts/scenes/co3-sampler/ from /datasets/mmolefe/poe_repair_min/loop-12/z2/smoke/.
Settled facts: the schedule and weights are in co3_numbers.json; the project's loader sets
latentCorrector.verbose to False, so d_k and w_k must come from the run's own log or capture.
Outputs: co3_numbers.py extended with a "worked_case" block read from that folder, every value
already in co3_numbers.json unchanged; the overview redrawn with the slot showing d_1, d_2, w_1, w_2
per pass and the latent before and after each of the six CO3 steps.
When something is missing: if the run logged no distances, say so in the slot and stop.
Proof: python3 draw_overview.py prints the line-check pass.
```

- [ ] **Gate:** the Z2 smoke folder exists and holds the distances or a capture of them.

### 5. 🔧 Level 2: the miniature (skipped for now)

`ladder.json` marks it `skipped`: owner chose overview, algorithms and code first.

```text
/deep-learning-scene --miniature CO3: a small network with integer weights for one CO3 resampler
pass and one corrector pass, K = 2, every term of Algorithms 2 and 3 and equation 15 on its part,
numbers from an answer key that extends 2-classifier-guidance/artifacts/scenes/co3-sampler/co3_numbers.py,
colours from that folder's ladder.json, symbols from its NOTATION.md.
```

### 6. 🔧 Level 1: the idea illustrated (skipped for now)

`ladder.json` marks it `skipped`: owner chose overview, algorithms and code first.

```text
/prompt-storyboard the CO3 idea from 2-classifier-guidance/artifacts/scenes/co3-sampler/index.md:
steer away from whichever concept the joint prediction sits closest to, drawn on the miniature's
numbers, only eagle and fox pictures; then /render-diagrams on the prompts it writes.
```

### 7. 🔧 The rebuild companion and the closing page

Waits on step 5. The companion is `hld-diagram --paper` step 13; the close is `deep-learning-scene --miniature`
step 12.

## Instructions

*For you to do by hand. Nothing here is delegated.*

Navigation: ⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#quizzes-and-feedback) ➡️

### 8. 📖 Read the ladder top down

- [ ] **Do:** in VS Code open `2-classifier-guidance/artifacts/scenes/co3-sampler/index.md`, press Ctrl+Shift+V for
  the preview, and click the overview thumbnail to open it full size.
- **You should see:** the curve rising from 0.076 at t = 981 to 1 at t = 1, the six CO3 steps with passes 10, 10, 5,
  5, 5, 5, and the eagle and fox picture at the end of the small forward diagram.
- **Done when:** you can say, for each of the six CO3 steps, whether it replaces the latent or edits it, and why.

### 9. 🔍 Check one paper-code difference yourself

- [ ] **Do:** open `/home/molef/PhD/replications/co3/composers/utils/attn_utils.py` at line 252 beside the PDF at
  page 6, Algorithm 3 lines 5 and 6.
- **You should see:** the paper divides two lengths ($\lVert\cdot\rVert_2$); the code takes `.max()` of each.
- **Done when:** you have decided whether our CO3 runs keep the release's rescale or switch to the paper's, and
  written that decision where the Loop 12 plan records its settings.

## Quizzes and feedback

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

No quiz this round.

## Next step

Navigation: ⬅️ [Previous](#quizzes-and-feedback) | 📋 [TOC](#table-of-contents)

**Start with:** instruction 8, reading the page. Step 4 opens when the Z2 smoke run lands. The background is in
[the folder card](README.md).
