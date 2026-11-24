# The digit-map checkers miniature page

Location:   2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/
Ran under:  the digit-map checker zoom ladder (Loop 8, V2), rung 2
Built from: ../numbers/real/count-stages.npz (the saved count checker, job 94116), ../numbers/real/count-final.json (its scores, job 94108), ../2-miniature/miniature.py and its key, ../numbers/numbers.json, ../numbers/examples/, scripts/mnist/pixel_checker.py (PixelHead), scripts/abstract_checkers/v2_learn.py and explain_stages.py
Why:        one page that opens on the real count checker running on a real canvas, stage by stage, then steps and plays the two checkers' forward pass, training backward pass and push backward pass on the ladder's 4-pixel miniature
url:        not published (a private link needs the owner's yes)

The page is [the miniature page](digit-map-checkers-miniature.html), one self-contained HTML file of 2.3 MB that fetches nothing to show its numbers or its mathematics. It is built with `deep-learning-scene --miniature` and reuses [the ladder's miniature](../2-miniature/README.md) unchanged, extended only where a control needed values that key did not hold.

## What the page is made of

```
Model:   the saved count checker (share-0.0 U-Net, PixelHead, 481,665 weights) for frame 1;
         the miniature checker (one conv, SiLU, scale and shift, mean and max, one readout) for frames 4 to 12
Input:   frame 1: canvases apart_1_3, wrong_1_3, apart_3_4, wrong_3_4 at t = 712, 616, 539, 480
         the miniature: 4 pixels; asked pair 1 and 3; canvases apart (1 | 3) and wrong (3 | 2); 16 real timesteps
Frames (14, stepped with the right arrow key or next):
  1  real-system      the real system on a real canvas: the noisy input, then each stage's real output in a row,
                      a red box on the stage the panel explains, that stage drawn large below (12 stages)
  2  two-questions    two real canvases, frozen U-Net, 2 maps, checker, logit
  3  noise            real canvases at the chosen timestep; the miniature's pixels; the real count checker's balanced
                      accuracy at all 16 timesteps, on digits 5 to 9 and on digits 0 to 4
  4  maps             miniature: pixels, frozen expert, the two maps
  5  full-size        PixelHead as CNN volumes, every shape and weight count (481,665)
  6  convolution      miniature conv and SiLU; the kernel's 16 weights on hover and in the panel
  7  scale-and-shift  the time features, gamma and beta, h = SiLU(pre)(1 + gamma) + beta
  8  mean-and-max     the pool, with the max's half marked
  9  logit            the readout to the logit and its chance
  10 labels           count and identity labels, loss, slope sigma(l) - y
  11 training         backward into the kernel, stop at the maps, one step of 0.05
  12 push             backward through the maps and the frozen expert to the pixels; broken run
  13 push-length      the push at length b, 0.4 marked
  14 results          the real scores, the saved count checker's 0.791 first
Untemplated node types: none; the frozen U-Net is one box by design (it has skips, the checker has none)
Equation-bearing nodes: #real-system (every stage), #noise (x_t, balanced accuracy), #maps (map scaling),
  #full-size (conv output size), #convolution, #scale-and-shift (FiLM), #mean-and-max, #logit, #labels (binary
  cross-entropy), #training (chain rule into K), #push (g = grad of l), #push-length (eps_hat), #results
```

Each frame id is also the page's anchor, so `digit-map-checkers-miniature.html#push` opens the push frame.

## Frame 1: the real system, stage by stage

The row across the top holds twelve stages in the order the checker runs them: the noisy canvas, the frozen U-Net's three noise guesses, the two maps, the three stem convolutions, c1, the timestep's scale and shift, c2, the 256 pooled numbers, the logit and the push gradient. Each tile is that stage's real output on the chosen canvas and noise level. The red box marks the stage the side panel explains, and the area below draws that stage large:

| Stage | Drawn large |
|---|---|
| the noisy canvas | clean and noisy canvas side by side, the noising equation |
| U-Net | the three noise guesses on one colour scale |
| maps | both maps, each with its sum (2,048) and peak |
| conv 1, conv 2, conv 3, c1, c2 | 8 channels before SiLU and after it, the same channels in both rows |
| scale, shift | c1's 8 channels before and after, each with its gamma and beta, and strips of all 128 gammas and betas |
| pool | the 128 means and 128 maxes as two strips, each on its own range, and the 16 time features |
| logit | the logit and its chance, and this canvas's logit at the four noise levels |
| push g | the noisy canvas beside the gradient of the logit with respect to it, and its length |

**The rules the pictures follow, printed on the page.** The 8 channels shown are the ones with the largest mean value on that canvas at that timestep. Signed values are blue below 0, white at 0 and red above, and each picture saturates at the 99th percentile of the absolute values drawn with it. Canvases are grey from −1 (black) to +1 (white), and the noisy canvas is clipped there.

**What frame 1 shows that the rest cannot.** The checker gets wrong_1_3 wrong at three of four noise levels: its count label is 1 and its logits are 0.076, −1.115, −1.281 and −1.019 at t = 712, 616, 539 and 480. The logit stage says so.

## What each control does

| Control | What it shows |
|---|---|
| right arrow key, next, back | one idea per frame, in the order the network runs |
| real canvas, noise t (frame 1 only) | the four saved canvases and the four saved noise levels; every tile, picture and number in frame 1 switches |
| step, a click on a stage, Enter on a focused stage (frame 1) | moves the red box one stage on, or to that stage; play walks all twelve |
| canvas: apart or wrong | every miniature number switches canvas |
| question: count or identity | the labels, the loss, the training gradient and the after-one-step logits switch branch |
| how much noise, and sweep noise | the 16 real timesteps from 712 to 480: the real canvases, the miniature's pixels, the time features and every number downstream move together; sweep plays all 16 |
| push length b | the push, the pushed guess and the clean-guess shift for b from 0 to 1 in steps of 0.05, with 0.4 marked |
| play, pause, step, speed | play animates the current frame's pass: the forward pass layer by layer with values riding the edges; the training pass back into the kernel and its stop at the maps; the push pass back through the frozen U-Net into the pixels; or a sweep of t or b. Step moves one layer. Speed is half, one or two times |
| hover or tap a line (frames 4 to 12) | the weight on that line; in the training frame also its gradient and its value after one step |
| broken run: maps built inside no_grad | the push frame with the maps detached, as training builds them: g = (0, 0, 0, 0) |

No animation starts on its own. With reduced motion on, play shows the end state, step moves one layer without travel, and sweep is hidden in favour of step. Frame 1 hides the miniature's controls so its controls fit one row.

## Real and illustrative

**Real**, each from the file named: every picture and number in frame 1 (`../numbers/real/count-stages.npz`, written by `explain_stages.py` in job 94116, whose layer-by-layer logit must match the checker's own to 1e-4 or the job stops); the count checker's balanced accuracy at each of the 16 timesteps, on digits 5 to 9 it never saw (0.632 at t = 712 rising to 0.924 at t = 480, 0.791 over all 16) and on digits 0 to 4 (0.782 over all 16), from `../numbers/real/count-final.json` (V2's `final-count.json`, job 94108, W&B `tyhrpyov`), whose two means `make_page_key.py` asserts round to 0.791 and 0.782 and whose first and last unseen values it checks against the V2 review; the 16 timesteps and their ᾱ (`numbers.json`, from `canvas_unet.py` lines 35 to 38 and `v2_learn.py` line 46), the time features' first frequency, the canvases `apart_1_3` and `wrong_1_3` (data hash f70b0fb3...), every layer shape and weight count, the training counts, and the scores of jobs 94085 and 94080.

**Illustrative**: every miniature weight, the frozen expert's formula, the four-pixel noise draw, the step size, and the display noise on the real canvases (numpy seed 0). The identity checker was never saved, so every identity number on the page is the miniature's. The page tags each panel `real` or `illustrative`.

**Every number on the page is read from `page_key.json`.** For the miniature, the page recomputes the forward pass, the push gradient and the training gradient on the kernel at the chosen timestep and canvas, and prints how far they sit from the key and how far its hand gradients sit from central differences. Over all 64 settings (16 timesteps, 2 questions, 2 canvases), the largest gaps are 5.0e-5 (the key's rounding to 4 places) and 6.8e-10. Frame 1's check line states the checks the job and the key script ran when they wrote the numbers.

## Equations

Every equation is typeset by KaTeX 0.16.11 at build time. `equations.json` holds the TeX of all 58, `render_math.mjs` renders them to HTML with KaTeX's strict mode on, and `build_page.py` inlines that HTML with KaTeX's CSS and its 20 woff2 fonts as data URIs (359 KB). Display equations wider than their box scroll inside it, and every one fits at 1440 px.

## Rebuild

```bash
cd /home/molef/PhD/concept-composition/2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page
npm install --no-save --no-package-lock katex@0.16.11                               # once; node_modules/ is git-ignored
PYTHONDONTWRITEBYTECODE=1 ~/.claude/venvs/figures/bin/python -I make_page_key.py   # 16 timesteps, 21 push lengths, 16 real cells; 94 values equal to the old key at t = 480; mismatch 2.6e-09
PYTHONDONTWRITEBYTECODE=1 ~/.claude/venvs/figures/bin/python -I build_page.py      # 58 equations typeset; writes digit-map-checkers-miniature.html
node ~/.claude/skills/codex-author/scripts/check_scene.mjs --page digit-map-checkers-miniature.html --out /tmp/mp-check   # must_fix 0
python3 -I ~/.claude/skills/hld-diagram/scripts/check_leaks.py digit-map-checkers-miniature.html --db /dev/null --repo /home/molef/PhD/concept-composition
node shoot_stills.mjs                                                                # stills/, light and dark, desktop and phone
```

| File | What it is |
|---|---|
| `make_page_key.py` | extends the miniature to 16 timesteps, 21 push lengths and the broken run, and packs frame 1's real stages and scores; asserts the t = 480 values equal the old key, every gradient against finite differences, each real map's sum, the pooled numbers against c2, the real logits against job 94116's log, and the two per-step means against 0.791 and 0.782 |
| `page_key.json` | every number the page shows (1.7 MB, inlined into the page) |
| `equations.json` | the TeX of every equation and math symbol on the page |
| `render_math.mjs` | typesets `equations.json` with KaTeX |
| `page_template.html` | the page source, with placeholders for the key, the equations and KaTeX's CSS |
| `build_page.py` | inlines the key, the typeset equations and KaTeX's CSS with its fonts |
| `digit-map-checkers-miniature.html` | the built page, 2.3 MB, opens as one file |
| `shoot_stills.mjs` | 20 desktop stills and 4 phone stills, each in light and dark, through the page's `__seek` and `__real` hooks |
| `stills/` | those 48 stills, and `stills/gate/` (git-ignored) with the check gate's 14 desktop-light and 14 phone-dark stills |
| `.gitignore` | keeps `node_modules/` out of git |

## Check results

**The check gate** (`check_scene.mjs`) walked 14 frames at desktop and phone width: 0 must-fix, 0 to check. **A sweep of frame 1** over its 4 canvases, 4 noise levels and 12 stages, at 1440 × 860 and at 390 wide, found no page overflow, every caption at 40 words or fewer, and every check line true (384 states). **Every display equation and every typeset figure label** fits its box at 1440 px on every frame and stage. **The leak check** found 0 email-shaped values. This repo has no people database configured, so the name list it checked against was empty.

**The stills were looked at, light and dark**: frame 1 at the noisy canvas, the maps, conv 1, the scale and shift, the pool, the logit on wrong 3|2 at t = 616 (−1.115, a miss), the push gradient on apart 3|4 at t = 539 and the U-Net's guesses at t = 712; the forward pass mid-convolution with the kernel lines bare; one kernel weight's hover label; the training pass ended with the kernel gradients on the orange lines; the noise frame's two real curves at t = 480 and t = 712; the push at b = 1.00; the results; and frame 1 and the network at phone width. The stills script blocks Google Fonts because headless Chrome stalls on them from WSL, so its stills show the fallback faces. The gate's stills load the real ones.

## Look

The display face is Marhey, chosen here because the scope's diagram-prompts header records none. Colours are the ladder's own key from `../ladder.json`; frame 1's real tensors use a blue-white-red scale, and its red box is the one colour outside that key.

## Still to do, each needing the owner's yes

1. **Publish it as a private link.** Publish `digit-map-checkers-miniature.html` with the Artifact tool, add its URL to the `url:` line above, and file its scene-library rows.
2. **Codex stills and review.** Codex draws three to six stills that walk the page, then queues the read-only review every scene gets before it is published.
3. **Animation as video.** The `animate` skill renders the forward, training and push passes as a video. That is a separate request from this page's in-page animation.
4. **The identity checker's real stages.** Save the identity head and rerun `explain_stages.py` with it; frame 1 would then switch between the two checkers.
