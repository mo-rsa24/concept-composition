# 📚 SDXL from the prompt down to one attention head
<!-- template: exempt. Reason: a skill-route runbook built to route-file-template, not a plan -->

> [!NOTE]
> **You will hold:** the SDXL ladder in this folder, read from [the reading page](index.md) · **Time:** levels 0, 0.5,
> 2 and 3 are built; the clean-guess row waits on job 94450 · **Stop point:** every rung in the brief is built or
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

**Goal:** see where CO3's fix and Loop 12's step Z2 readings sit inside SDXL, with the maths and the real sizes at
every level.
**Objective (the whole chain):** one real picture followed from the prompt to the pixels, the U-Net, one transformer
block and the clean guess each drawn with sizes from SDXL's config, one attention head worked by hand beside the real
maps, and the diffusers lines that compute each level.
**Outcome (the whole chain):** this folder's reading page, five figures, the code page, and `ladder.json`.
**Outcome board:** none; `ladder.json` records each rung's state.
**Background:** the folder card, [README.md](README.md), says how every file was made.

## What we know so far

Navigation: ⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#what-you-will-achieve) ➡️

| Fact | Status | Shown by | Changes |
|---|---|---|---|
| SDXL's U-Net holds 17 ResBlocks and 70 transformer blocks, 10 at 64 × 64 and 60 at 32 × 32, none at 128 × 128 | found | `sdxl_numbers.py` from `source/sdxl-config/unet__config.json` and diffusers 0.29.2's constructor, 2026-11-24 | steps 2 and 3 |
| `attention_head_dim` [5, 10, 20] is read as the number of heads, so each head is 64 numbers wide | found | `unet_2d_condition.py` line 241, `unet_2d_blocks.py` line 806 | step 3 |
| Z2's capture records 70 cross-attention layers at 14 steps | found | the smoke run's `seed-6001.json`, field `layers` | steps 2 and 4 |
| The share of picture is 0.6%, 1.7%, 18% and 92% at steps 0, 5, 20 and 45 | found | `sdxl_numbers.json`, both at $t$ and at CO3's $t - 1$ | step 3 |
| Job 94450's clean guesses are not on disk | open | `squeue -j 94450` pending, `/datasets/mmolefe/poe_repair_min/loop-12/co3-inside/` absent, 2026-11-24 | step 6 |

## What you will achieve

Navigation: ⬅️ [Previous](#what-we-know-so-far) | 📋 [TOC](#table-of-contents) | [Next](#the-route-at-a-glance) ➡️

| Stage | Tier | You will have | You will draw, from a small example | A worked example | Exercises | Quiz |
|---|---|---|---|---|---|---|
| 0 the whole run | read | the overview | the prompt, the 50-step loop and the decoder with CO3 and Z2 marked | the eagle and fox render | none this round | none |
| 0.5 the parts | read | three figures | the U-Net, one transformer block, the clean guess | the picture noised at four steps | none this round | none |
| 2 one head | work | the hand-worked head | the score and weight tables for 4 patches and 3 words | the real eagle and fox columns | instruction 9 | none |
| 3 the code | read | the code page | none | none | none this round | none |
| **Outcome** | | **the ladder in this folder** | | | | **`ladder.json` shows every rung in the brief built** |

## The route at a glance

Navigation: ⬅️ [Previous](#what-you-will-achieve) | 📋 [TOC](#table-of-contents) | [Next](#the-two-lanes) ➡️

▶ **Recommended:** `numbers → level 0 → level 0.5 ‖ level 3 → level 2 → clean-guess row (after job 94450)`
(the zoom-ladder order, with the illustrations, the companion and the close left out of the brief)

## The two lanes

Navigation: ⬅️ [Previous](#the-route-at-a-glance) | 📋 [TOC](#table-of-contents) | [Next](#schedule) ➡️

| Stage | The AI lane builds | Your lane learns | Quiz | Confidence bar | You rebuild | Compare on |
|---|---|---|---|---|---|---|
| 0 to 3 | the figures and the code page | reading the page, one level at a time | none this round | you can point to where on the U each of Z2's readings is taken | the hand-worked head's weight table on paper | the level 2 figure |

## Schedule

Navigation: ⬅️ [Previous](#the-two-lanes) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

| Block | Run | Checkpoint at the end of the block |
|---|---|---|
| unattended, 2026-11-24 | steps 1 to 5 | built, every figure passed the line check |
| after job 94450 lands | step 6 | the clean-guess row filled from real decoded latents |
| when the owner asks | step 7 | the illustrations, the companion and the close |

## Tasks

*For the AI to run. Paste each prompt into a session.*

Navigation: ⬅️ [Previous](#schedule) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

### 1. ✅ The numbers

**Outcome:** `sdxl_numbers.json`, written by `sdxl_numbers.py`. **Built.**

```text
In 2-classifier-guidance/artifacts/scenes/sdxl-down-the-layers/, run python3 sdxl_numbers.py; it prints
"resblocks 17, transformer blocks 70 {64: 10, 32: 60}" and stops if any config sum or cited diffusers line has moved.
```

### 2. ✅ Level 0: one picture from the prompt to the pixels

**Outcome:** `l0-sdxl-from-prompt-to-picture.png` and `.svg`, drawn by `draw_l0_overview.py`. **Built.**

```text
In 2-classifier-guidance/artifacts/scenes/sdxl-down-the-layers/, run python3 draw_l0_overview.py; it prints a line
ending "no line crosses a node, box or label" when right.
```

### 3. ✅ Level 0.5: the U-Net, one transformer block, the clean guess

**Outcome:** `l05a-inside-the-unet`, `l05b-one-transformer-block`, `l05c-noise-guess-and-clean-guess` (PNG and SVG). **Built.**

```text
In 2-classifier-guidance/artifacts/scenes/sdxl-down-the-layers/, run python3 draw_l05a_unet.py,
draw_l05b_transformer_block.py and draw_l05c_noise_and_clean_guess.py; each prints the line-check pass.
```

### 4. ✅ Level 2: one attention head by hand

**Outcome:** `l2-one-attention-head-by-hand.png` and `.svg`, drawn by `draw_l2_attention_by_hand.py`. **Built.**

```text
In 2-classifier-guidance/artifacts/scenes/sdxl-down-the-layers/, run python3 draw_l2_attention_by_hand.py; it reads
the smoke run's maps from /home/molef/.claude/jobs/0071c318/tmp/z2-smoke/an_eagle__x__a_fox/seed-6001-maps.npz.
```

### 5. ✅ Level 3: the code page

**Outcome:** `code-walk.md`, written by `gen_codewalk.py`. **Built.**

```text
In 2-classifier-guidance/artifacts/scenes/sdxl-down-the-layers/, run python3 gen_codewalk.py; it stops if any cited
line in source/diffusers-0.29.2/, CO3's checkout at commit 35ea972 or z2_predictors.py has moved.
```

### 6. 🔧 Fill the clean-guess row from job 94450

**Objective:** put the decoded clean guess at each of steps 0 to 5, before and after CO3's passes, in the empty row of
figure 0.5c. **Starts from:** job 94450's outputs.

```text
Fill the empty bottom row of l05c-noise-guess-and-clean-guess.png in
2-classifier-guidance/artifacts/scenes/sdxl-down-the-layers/ from /datasets/mmolefe/poe_repair_min/loop-12/co3-inside/.
Settled facts: the row has six tiles, steps 0 to 5 at t = 981, 961, 941, 921, 901, 881; the schedule is in
sdxl_numbers.json; the picture is "an eagle and a fox", seed 6001.
Outputs: sdxl_numbers.py extended with a "clean_guesses" block naming each decoded file, every value already in
sdxl_numbers.json unchanged; draw_l05c_noise_and_clean_guess.py drawing each tile from those files (two per step if the
job saved before and after); the note in index.md replaced by one sentence on what the row shows.
When something is missing: if the folder holds latents and not pictures, say so in the row and stop; decoding needs a
GPU and the owner's yes.
Proof: python3 draw_l05c_noise_and_clean_guess.py prints the line-check pass; look at the figure.
```

- [ ] **Gate:** `ls /datasets/mmolefe/poe_repair_min/loop-12/co3-inside/` on mscluster lists decoded pictures.

### 7. 🔧 The illustrations, the companion and the close (not in the brief)

`ladder.json` marks level 1 `skipped` and the companion and the close `missing`. Each would start from this folder's
`ladder.json`, `NOTATION.md` and `sdxl_numbers.json`.

## Instructions

*For you to do by hand. Nothing here is delegated.*

Navigation: ⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#quizzes-and-feedback) ➡️

### 8. 📖 Read the ladder top down

- [ ] **Do:** in VS Code open `2-classifier-guidance/artifacts/scenes/sdxl-down-the-layers/index.md`, press
  Ctrl+Shift+V for the preview, and click the level 0 thumbnail to open it full size.
- **You should see:** the prompt on the top left, the orange CO3 box in front of the green U-Net, three gold boxes for
  Z2's readings, and the eagle and fox picture at the bottom right.
- **Done when:** you can say, for each of Z2's three readings, which network it comes from and at which steps.

### 9. ✍️ Work the head by hand

- [ ] **Do:** on paper, take patch p3's query (1, 2) and the three keys (2, 0), (0, 0), (0, 2) from the level 2 figure;
  compute the three scores divided by $\sqrt{2}$, then the softmax.
- **You should see:** scores 1.414, 0.000, 2.828 and weights 0.187, 0.045, 0.768, matching the figure's p3 row.
- **Done when:** you can say which number in that row is the eagle column spilling onto the fox's patch.

## Quizzes and feedback

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

No quiz this round.

## Next step

Navigation: ⬅️ [Previous](#quizzes-and-feedback) | 📋 [TOC](#table-of-contents)

**Start with:** instruction 8, reading the page. Step 6 opens when job 94450 lands. The background is in
[the folder card](README.md).
