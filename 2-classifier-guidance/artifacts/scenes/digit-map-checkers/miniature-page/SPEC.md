# The digit-map checkers miniature page

Location:   2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/
Ran under:  the digit-map checker zoom ladder (Loop 8, V2), rung 2
Built from: ../numbers/real/trace/ (plain and pushed sampling at every step, job 94161), ../numbers/real/count-stages.npz (the saved count checker, job 94116), readout-mlp.npz (its readout MLP's weights, read from count-head-map.pt of job 94108), ../numbers/real/count-final.json (its scores, job 94108), ../2-miniature/miniature.py and its key, ../numbers/numbers.json, ../numbers/examples/, scripts/mnist/pixel_checker.py (PixelHead), scripts/abstract_checkers/v2_learn.py and explain_stages.py
Why:        one page that opens on the real count checker running on a real canvas, stage by stage, then steps and plays the two checkers' forward pass, training backward pass and push backward pass on the ladder's 4-pixel miniature, and ends on how the real push turns one fused glyph into two digits, step by step
url:        https://claude.ai/artifact/MX2hKBUWBXSzPQSbaxysKC (private)

The page is [the miniature page](digit-map-checkers-miniature.html), one self-contained HTML file of 8.7 MB that fetches nothing to show its numbers or its mathematics. It is built with `deep-learning-scene --miniature` and reuses [the ladder's miniature](../2-miniature/README.md) unchanged, extended only where a control needed values that key did not hold.

## What the page is made of

```
Model:   the saved count checker (share-0.0 U-Net, PixelHead, 481,665 weights) for frame 1;
         the miniature checker (one conv, SiLU, scale and shift, mean and max, one readout) for frames 4 to 12
Input:   frame 1: canvases apart_1_3, wrong_1_3, apart_3_4, wrong_3_4 at t = 712, 616, 539, 480
         frame 14: the 10 unseen pairs at the recorded run's 32 seeds (step V3 of the design), plain and pushed, all 50 sampler steps; pictures for six seeds
         the miniature: 4 pixels; asked pair 1 and 3; canvases apart (1 | 3) and wrong (3 | 2); 16 real timesteps
Frames (15, stepped with the right arrow key or next):
  1  real-system      the real system on a real canvas: the noisy input, then each stage's real output in a row,
                      a red box on the stage the panel explains, that stage drawn large below (12 stages)
  2  two-questions    two real canvases, frozen U-Net, 2 maps, checker, logit
  3  noise            real canvases at the chosen timestep; the miniature's pixels; the real count checker's balanced
                      accuracy at all 16 timesteps, on digits 5 to 9 and on digits 0 to 4
  4  maps             miniature: pixels, frozen expert, the two maps
  5  full-size        PixelHead as CNN volumes, every shape and weight count (481,665)
  6  convolution      miniature conv and SiLU; the kernel's 8 weights, shared by both halves, on its 16 lines
                      on hover and in the panel
  7  scale-and-shift  the time features, gamma and beta, h = SiLU(pre)(1 + gamma) + beta
  8  mean-and-max     the pool, with the max's half marked
  9  logit            the readout to the logit and its chance
  10 labels           count and identity labels, loss, slope sigma(l) - y
  11 training         backward into the kernel, stop at the maps, one step of 0.05
  12 push             backward through the maps and the frozen expert to the pixels; the push read as classifier
                      guidance and as a step added to the score; broken run
  13 push-length      the push at length b, 0.4 marked
  14 two-digits       one seed sampled twice, plain above and pushed below, over the 50 sampler steps: the guess of
                      the clean canvas, the two maps in two colours with their centres marked, g, and the logit;
                      beside it the 320-canvas means of map-centre separation, overlap and logit
  15 results          the real scores, the saved count checker's 0.791 first
Untemplated node types: none; the frozen U-Net is one box by design (it has skips, the checker has none)
Equation-bearing nodes: #real-system (every stage), #noise (x_t, balanced accuracy), #maps (map scaling),
  #full-size (conv output size), #convolution, #scale-and-shift (FiLM), #mean-and-max, #logit, #labels (binary
  cross-entropy), #training (chain rule into K), #push (g = grad of l; grad log p = (1 - p) g;
  the score -eps/sqrt(1 - abar) and the step s_t g the push adds to it), #push-length (eps_hat), #two-digits (map centre, overlap,
  the push), #results
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
| logit | the readout MLP as nodes: the 272 inputs as three strips (128 means, 128 maxes, 16 time features), the 12 hidden units with the largest \|w₂ⱼ hⱼ\| on this canvas at this noise level with their activations, each edge carrying its output weight and coloured by the sign of w₂ⱼ hⱼ, the other 244 units as one dashed edge, the sum that gives the logit, its chance, and the logit at the four noise levels |
| push g | the noisy canvas beside the gradient of the logit with respect to it, and its length |

**The rules the pictures follow, printed on the page.** The 8 channels shown are the ones with the largest mean value on that canvas at that timestep. Signed values are blue below 0, white at 0 and red above, and each picture saturates at the 99th percentile of the absolute values drawn with it. Canvases are grey from −1 (black) to +1 (white), and the noisy canvas is clipped there.

**What frame 1 shows that the rest cannot.** The checker gets wrong_1_3 wrong at three of four noise levels: its count label is 1 and its logits are 0.076, −1.115, −1.281 and −1.019 at t = 712, 616, 539 and 480. The logit stage says so.

## Frame 14: how the push makes two digits

Job 94161 (`trace_push.py` at commit 199210af, on mscluster81) reran the plain and pushed rows of step V3 of the design (does the push compose unseen digits) on the 10 unseen pairs at seeds 1000 to 1031 and recorded every one of the 50 sampler steps. The push acts at steps 15 to 25 (t = 693 to 489) with b = 0.4.

**The two lanes.** One seed is sampled twice with the same noise, plain above and pushed below. Each lane shows, at the chosen step, the model's guess of the clean canvas, the two maps on one image (map a purple, map b green, dark where both are high) with a triangle under each map's centre across the canvas, the gradient g of the checker's logit (on the plain lane too, as the push that would have been applied; an orange border when the push acts), the final canvas, and the logit over all 50 steps with the push window shaded. The two lanes are one run through step 15 and differ from step 16, which `make_page_key.py` asserts. A seed the push separated (plain judged "one blob", pushed judged clean) shows by default; the seed control switches to one it did not separate. The pairs with pictures are 6 | 8 (seeds 1000 and 1001), 5 | 6 (1006 and 1000) and 7 | 9 (1002 and 1000).

**The 320-canvas curves.** Mean map-centre separation in pixels, mean overlap and mean logit at every step, plain (grey) against pushed (orange). `make_page_key.py` asserts they give the review's table at steps 0, 15, 20, 25, 30, 40 and 49 (pushed separation 1.6 at step 15, 6.6 at step 25, 1.1 at step 49; plain 1.6, 1.3, 0.9).

**What the panel says it cannot say**, from `06-does-it-compose/review.md`, "A rerun on another node" and "What the two maps do while the push acts": this rerun gave 41 clean pairs of 320 against V3's recorded 43, because a pushed render is not bit-reproducible across nodes, and the verdict stays V3's; after the window the maps' centres drift back together, so late map centres do not show where the digits ended up.

**Four pictures of this frame.** A card in the panel shows four stills Codex drew from the frame's real pictures and numbers, published with the page as supporting files from `stills-codex/`; [the stills page](digit-map-checkers-miniature-stills.md) walks them in words and records how they were made and checked.

**Picture rules.** Each map and g picture saturates at its own 99th percentile at that step, and the scales are kept in the key. The clean-canvas guesses and final canvases are grey from −1 (black) to +1 (white).

## What each control does

| Control | What it shows |
|---|---|
| right arrow key, next, back | one idea per frame, in the order the network runs |
| pair, seed, sampler step (frame 14 only) | the three example pairs, a won or a lost seed, and the 50 sampler steps; play walks the steps |
| real canvas, noise t (frame 1 only) | the four saved canvases and the four saved noise levels; every tile, picture and number in frame 1 switches |
| step, a click on a stage, Enter on a focused stage (frame 1) | moves the red box one stage on, or to that stage; play walks all twelve |
| canvas: apart or wrong | every miniature number switches canvas |
| question: count or identity | the labels, the loss, the training gradient and the after-one-step logits switch branch |
| noise step t, and sweep noise | the 16 real timesteps from 712 to 480: the real canvases, the miniature's pixels, the time features and every number downstream move together; sweep plays all 16 |
| push length b | the push, the pushed guess and the clean-guess shift for b from 0 to 1 in steps of 0.05, with 0.4 marked |
| play, pause, step, speed | play animates the current frame's pass: the forward pass layer by layer with values riding the edges; the training pass back into the kernel and its stop at the maps; the push pass back through the frozen U-Net into the pixels; or a sweep of t or b. Step moves one layer. Speed is half, one or two times |
| hover or tap a line (frames 4 to 12) | the weight on that line; in the training frame also its gradient and its value after one step |
| broken run: maps built inside no_grad | the push frame with the maps detached, as training builds them: g = (0, 0, 0, 0) |

No animation starts on its own. With reduced motion on, play shows the end state, step moves one layer without travel, and sweep is hidden in favour of step. Frame 1 hides the miniature's controls so its controls fit one row.

## Real and illustrative

**Real**, each from the file named: the readout MLP's weights on frame 1's logit stage (`readout-mlp.npz`, read from the saved `count-head-map.pt` by `read_readout.py`, which walks the file's pickle opcodes without running them; on all 16 saved cells those weights rebuild the stored logit to 2.0e-7 and the FiLM layer rebuilds the stored γ and β to 1.3e-7); every picture and number in frame 1 (`../numbers/real/count-stages.npz`, written by `explain_stages.py` in job 94116, whose layer-by-layer logit must match the checker's own to 1e-4 or the job stops); the count checker's balanced accuracy at each of the 16 timesteps, on digits 5 to 9 it never saw (0.632 at t = 712 rising to 0.924 at t = 480, 0.791 over all 16) and on digits 0 to 4 (0.782 over all 16), from `../numbers/real/count-final.json` (V2's `final-count.json`, job 94108, W&B `tyhrpyov`), whose two means `make_page_key.py` asserts round to 0.791 and 0.782 and whose first and last unseen values it checks against the V2 review; the 16 timesteps and their ᾱ (`numbers.json`, from `canvas_unet.py` lines 35 to 38 and `v2_learn.py` line 46), the time features' first frequency, the canvases `apart_1_3` and `wrong_1_3` (data hash f70b0fb3...), every layer shape and weight count, the training counts, and the scores of jobs 94085 and 94080.

**Illustrative**: every miniature weight, the frozen expert's formula, the four-pixel noise draw, the step size, and the display noise on the real canvases (numpy seed 0). The identity checker was never saved, so every identity number on the page is the miniature's. The page tags each panel `real` or `illustrative`.

**Every number on the page is read from `page_key.json`.** For the miniature, the page recomputes the forward pass, the push gradient and the training gradient on the kernel at the chosen timestep and canvas, and prints how far they sit from the key and how far its hand gradients sit from central differences. Over all 64 settings (16 timesteps, 2 questions, 2 canvases), the largest gaps are 5.0e-5 (the key's rounding to 4 places) and 6.8e-10. Frame 1's check line states the checks the job and the key script ran when they wrote the numbers.

## Equations

Every equation is typeset by KaTeX 0.16.11 at build time. `equations.json` holds the TeX of all 76, `render_math.mjs` renders them to HTML with KaTeX's strict mode on, and `build_page.py` inlines that HTML with KaTeX's CSS and its 20 woff2 fonts as data URIs (359 KB). Display equations wider than their box scroll inside it, and every one fits at 1440 px.

## Rebuild

```bash
cd /home/molef/PhD/concept-composition/2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page
npm install --no-save --no-package-lock katex@0.16.11                               # once; node_modules/ is git-ignored
mkdir -p /tmp/readout-pt-dl && timeout 300 ssh -o BatchMode=yes mscluster107 "cat /datasets/mmolefe/poe_repair_min/loop-8/v2/count-final/count-head-map.pt" > /tmp/readout-pt-dl/count-head-map.pt   # once
PYTHONDONTWRITEBYTECODE=1 ~/.claude/venvs/figures/bin/python -I read_readout.py /tmp/readout-pt-dl/count-head-map.pt   # writes readout-mlp.npz
PYTHONDONTWRITEBYTECODE=1 ~/.claude/venvs/figures/bin/python -I make_page_key.py   # 16 timesteps, 21 push lengths, 16 real cells; 94 values equal to the old key at t = 480; mismatch 2.6e-09
PYTHONDONTWRITEBYTECODE=1 ~/.claude/venvs/figures/bin/python -I build_page.py      # 76 equations typeset; writes digit-map-checkers-miniature.html
node ~/.claude/skills/codex-author/scripts/check_scene.mjs --page digit-map-checkers-miniature.html --out /tmp/mp-check   # must_fix 0
python3 -I ~/.claude/skills/hld-diagram/scripts/check_leaks.py digit-map-checkers-miniature.html --db /dev/null --repo /home/molef/PhD/concept-composition
node shoot_stills.mjs                                                                # stills/, light and dark, desktop and phone
```

| File | What it is |
|---|---|
| `read_readout.py` | reads the readout MLP and the FiLM layer from the saved `.pt` without PyTorch and without unpickling, checks them against job 94116's stored γ, β and logits, and writes `readout-mlp.npz` |
| `readout-mlp.npz` | the readout MLP's weights: W1 256 × 272, b1 256, w2 256, b2 1 (260 KB) |
| `make_page_key.py` | extends the miniature to 16 timesteps, 21 push lengths and the broken run, and packs frame 1's real stages and scores; asserts the t = 480 values equal the old key, every gradient against finite differences, that the 12 drawn hidden units, the other 244 and b2 rebuild each real logit, that (1 − p) drops out of the normalised push and that subtracting along g adds s_t g to the score, each real map's sum, the pooled numbers against c2, the real logits against job 94116's log, and the two per-step means against 0.791 and 0.782 |
| `page_key.json` | every number the page shows (8.4 MB, inlined into the page; 6.6 MB of it is frame 14's pictures) |
| `equations.json` | the TeX of every equation and math symbol on the page |
| `render_math.mjs` | typesets `equations.json` with KaTeX |
| `page_template.html` | the page source, with placeholders for the key, the equations and KaTeX's CSS |
| `build_page.py` | inlines the key, the typeset equations and KaTeX's CSS with its fonts |
| `digit-map-checkers-miniature.html` | the built page, 8.7 MB, opens as one file |
| `shoot_stills.mjs` | 26 desktop stills and 6 phone stills, each in light and dark, through the page's `__seek` and `__real` hooks |
| `stills-codex/` | four stills Codex drew of frame 14, published with the page as supporting files |
| `digit-map-checkers-miniature-stills.md` | the page that walks those four stills, with how they were made and checked |
| `stills/` | those 64 stills, and `stills/gate/` (git-ignored) with the check gate's 14 desktop-light and 14 phone-dark stills |
| `.gitignore` | keeps `node_modules/` out of git |

## Check results

**The check gate** (`check_scene.mjs`) walked 15 frames at desktop and phone width: 0 must-fix, 0 to check (frame 14 first carried 106 words on its figure against the limit of 45, and was cut to fit). **A sweep of what this version changed**, light and dark at 1440 × 860 and 390 × 844, covered frame 1's logit and push stages on all 16 real cells, the push frame on both canvases at all 16 timesteps with the broken run on and off, and frame 14 on its three pairs, both seeds and every seventh step (576 states): no page errors, no page overflow, every caption at 40 words or fewer, every display equation inside its box at 1440 px, and every check line true. Frame 1's check line on the logit stage adds the drawn sum back up (12 units, the other 244 and b₂, each rounded to 3 places) and compares it with the logit. **A noise sweep from a network frame** keeps the network's values on screen and moving with t. **The leak check** found 0 email-shaped values. This repo has no people database configured, so the name list it checked against was empty.

**Play on frame 14** walks the step slider (25 to 30 in 1.5 s at 1×), and step moves it one step.

**The stills were looked at, light and dark**: frame 14 at step 10 before the push (both lanes hold the same state), at step 20 with the push acting (6 and 8, seed 1000: pushed centres 9.1 px apart against 3.1, logit 3.78 against −1.08), at step 49 on 5 and 6, seed 1006 (plain ends as a lone 5, pushed as a 5 and a 6), and on the seed 7 and 9 did not separate, plus frame 14 at phone width; the readout network on apart 3|4 at t = 539 (mixed signs: unit #110 has h = −0.23 and w₂ = −0.054, a positive contribution drawn red), on wrong 3|2 at t = 616 (all 12 drawn units lower the logit, which adds to −1.115), and on apart 1|3 at t = 480 (12 units give +5.068, the other 244 +1.313, b₂ −0.073, ℓ = 6.308); the push frame's classifier-guidance card at t = 480; the readout at phone width, where the figure pans inside its box as the other frame 1 stages do. The stills script blocks Google Fonts because headless Chrome stalls on them from WSL, so its stills show the fallback faces.

## Codex review

Codex reviewed the page as it stood before this version (job 20261124-084855-bd535d, read-only, 21 findings). Each was checked against the code it names.

**Applied**: the mean-and-max caption (both halves get the mean's gradient); the full-size code excerpt shows the stem with its padding as `pixel_checker.py` lines 39 to 41 build it; the real convolution equation carries its spatial indices, stride and padding; the time-feature equation puts the 8 sines first and the 8 cosines after, as `checker.py` lines 72 and 73 do (the page had them interleaved); the training step shows the average over the two canvases; the broken run says the graph has no path from the logit to the pixels and stops quoting the connected run's length; the clean-guess shift is written as ε_product − ε̂; the kernel count is 8 shared weights on 16 lines; the SiLU sentence names the layer's smallest value; the sentence about the logit growing as the noise falls is computed from the four logits; the results card says what the scores show and no more; the two-questions card says the earlier runs trained on 6 pairs and the saved checker on 10; controls that do nothing on a frame are disabled, and the noise slider says which way is less noise; keyboard focus stays on the stage tile after the row is rebuilt; "bar" became "target"; the pool legend says how colour is set; the opening caption gives the two noise numbers as multipliers; a noise sweep from a network frame no longer blanks the network.

**Not applied, with the reason**: the push equation's form (the owner set it); moving numbers out of captions (every caption passes the 40-word check, and the numbers are what each frame is for); on-picture real or illustrative stamps (every panel card carries its tag); backward-pass column labels and renaming the miniature's "frozen U-Net" box (both change the network figure every frame shares, kept for a later pass); moving the check line into a closed box (it is the page's live evidence); a longer opening explanation of every term (frame 2 and the disclosures carry it). Codex could not see the Python files; every line reference it flagged was checked here against them.

A second review (job 20261124-091820-542dff, read-only, 10 findings) covered the readout network, the classifier-guidance card and frame 14. **Applied**: the score card calls the noise guess a score estimate, exact only for an exact noise predictor, and states why the push is a tilt with no exactness guarantee (the checker steers a distribution it was not trained on, and s_t changes with the canvas); the broken run's zeros are finite differences with the maps held fixed; frame 14's captions say the lanes hold the same state before the push (the final thumbnails differ), name step 15 as the first push, and limit the late-steps caveat to map centres; the direction equation states its conditions and splits over two lines; the guidance card's numbers sit in a table; the logit caption names the bias; the disclosure says the faint lines summarise the first layer; "won" and "lost" became "separated" and "not separated", and "V3" is named as the recorded run; the two new cards carry the real tag; the plots name plain and pushed. **Not applied**: a still showing the wrong pair button (it was a misread, and the still shows 5 | 6 selected); step ticks, a "sampler step" axis title and per-plot "plain" and "pushed" words on frame 14, which would take the figure past the gate's 45-word limit (the slider names the step and the "how to read the page" disclosure names the colours).
## Look

The display face is Marhey, chosen here because the scope's diagram-prompts header records none. Colours are the ladder's own key from `../ladder.json`; frame 1's real tensors use a blue-white-red scale, and its red box is the one colour outside that key. Frame 14 adds two map colours, purple (#7570b3) for map a and green (#1b9e77) for map b, which mix to dark where both are high.

## Still to do, each needing the owner's yes

1. **Animation as video.** The `animate` skill renders the forward, training and push passes as a video. That is a separate request from this page's in-page animation.
2. **The identity checker's real stages.** Save the identity head and rerun `explain_stages.py` with it; frame 1 would then switch between the two checkers.
3. **The scene card and its referrer.** The scene grouping's card (`../README.md`) owes a `### Versions` entry for this version and an entry for the Codex stills, and the plan this page serves owes one referrer line in its Figure Catalog. Both sit outside this folder, which this pass was limited to.
