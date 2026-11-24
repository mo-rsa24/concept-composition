# The digit-map checkers miniature page

Location:   2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page/
Ran under:  the digit-map checker zoom ladder (Loop 8, V2), rung 2
Built from: ../2-miniature/miniature.py and its key, ../numbers/numbers.json, ../numbers/examples/, scripts/mnist/pixel_checker.py (PixelHead), scripts/abstract_checkers/v2_learn.py
Why:        one page where the two checkers' forward pass, training backward pass and push backward pass can be stepped and played on the ladder's 4-pixel miniature
url:        not published (a private link needs the owner's yes)

The page is [the miniature page](digit-map-checkers-miniature.html), one self-contained HTML file. It is built with `deep-learning-scene --miniature` and reuses [the ladder's miniature](../2-miniature/README.md) unchanged, extended only where a control needed values that key did not hold.

## What the page is made of

```
Model:   the miniature checker (one conv, SiLU, scale and shift, mean and max, one readout), from ../2-miniature/miniature.py
Input:   4 pixels; asked pair 1 and 3; canvases apart (1 | 3) and wrong (3 | 2); 16 real timesteps
Frames (13, stepped with the right arrow key or next):
  1  two-questions     the real system: two real canvases, frozen U-Net, 2 maps, checker, logit
  2  noise             real canvases at the chosen timestep; the miniature's pixels; count score over 16 timesteps
  3  maps              miniature: pixels, frozen expert, the two maps
  4  full-size         PixelHead as CNN volumes, every shape and weight count (481,665)
  5  convolution       miniature conv and SiLU, the kernel on every edge
  6  scale-and-shift   the time features, gamma and beta, h = SiLU(pre)(1 + gamma) + beta
  7  mean-and-max      the pool, with the max's half marked
  8  logit             the readout to the logit and its chance
  9  labels            count and identity labels, loss, slope sigma(l) - y
  10 training          backward into the kernel, stop at the maps, one step of 0.05
  11 push              backward through the maps and the frozen expert to the pixels; broken run
  12 push-length       the push at length b, 0.4 marked
  13 results           the real scores: count 0.774, identity 0.555, combined 0.704 and 0.511
Untemplated node types: none; the frozen U-Net is one box by design (it has skips, the checker has none)
Equation-bearing nodes: #maps (map scaling), #scale-and-shift (FiLM), #labels (binary cross-entropy), #training (chain rule into K), #push (g = grad of l), #push-length (eps_hat)
```

Each frame id is also the page's anchor, so `digit-map-checkers-miniature.html#push` opens the push frame.

## What each control does

| Control | What it shows |
|---|---|
| right arrow key, next, back | one idea per frame, in the order the network runs |
| canvas: apart or wrong | every miniature number switches canvas |
| question: count or identity | the labels, the loss, the training gradient and the after-one-step logits switch branch |
| how much noise, and sweep noise | the 16 real timesteps from 712 to 480: the real canvases, the miniature's pixels, the time features and every number downstream move together; sweep plays all 16 |
| push length b | the push, the pushed guess and the clean-guess shift for b from 0 to 1 in steps of 0.05, with 0.4 marked |
| play, pause, step, speed | play animates the current frame's pass: the forward pass layer by layer with values riding the edges; the training pass back into the kernel and its stop at the maps; the push pass back through the frozen U-Net into the pixels; or a sweep of t or b. Step moves one layer. Speed is half, one or two times |
| broken run: maps built inside no_grad | the push frame with the maps detached, as training builds them: g = (0, 0, 0, 0) |

No animation starts on its own. With reduced motion on, play shows the end state, step moves one layer without travel, and sweep is hidden in favour of step.

## Real and illustrative

**Real**, each from the file named: the 16 timesteps and their ᾱ (`numbers.json`, from `canvas_unet.py` lines 35 to 38 and `v2_learn.py` line 46), the time features' first frequency, the canvases `apart_1_3` and `wrong_1_3` (data hash f70b0fb3...), every layer shape and weight count, the training counts, and the scores: count 0.774 and 0.781 (job 94085's log lines), identity 0.555 and 0.603 (`artifacts/notes/progress/abstract-checkers/05-does-it-learn/review.md`), combined 0.704 and 0.511 (job 94080).

**Illustrative**: every miniature weight, the frozen expert's formula, the four-pixel noise draw, the step size, and the display noise on the real canvases (numpy seed 0). The page tags each panel `real` or `illustrative`.

**Every number on the page is read from `page_key.json`.** The page also recomputes the forward pass, the push gradient and the training gradient on the kernel at the chosen timestep and canvas. It prints how far they sit from the key and how far its hand gradients sit from central differences. Over all 64 settings (16 timesteps, 2 questions, 2 canvases), the largest gaps are 5.0e-5 (the key's rounding to 4 places) and 6.8e-10.

## Rebuild

```bash
cd /home/molef/PhD/concept-composition/2-classifier-guidance/artifacts/scenes/digit-map-checkers/miniature-page
PYTHONDONTWRITEBYTECODE=1 ~/.claude/venvs/figures/bin/python -I make_page_key.py   # 16 timesteps, 21 push lengths; 94 values equal to the old key at t = 480; mismatch 2.6e-09
PYTHONDONTWRITEBYTECODE=1 ~/.claude/venvs/figures/bin/python -I build_page.py      # writes digit-map-checkers-miniature.html
node ~/.claude/skills/codex-author/scripts/check_scene.mjs --page digit-map-checkers-miniature.html --out /tmp/mp-check   # must_fix 0
python3 -I ~/.claude/skills/hld-diagram/scripts/check_leaks.py digit-map-checkers-miniature.html --db /dev/null --repo /home/molef/PhD/concept-composition
node shoot_stills.mjs                                                                # stills/, light and dark
```

| File | What it is |
|---|---|
| `make_page_key.py` | extends the miniature to 16 timesteps, 21 push lengths and the broken run; asserts the t = 480 values equal the old key and every gradient against finite differences |
| `page_key.json` | every number the page shows (272 KB, inlined into the page) |
| `page_template.html` | the page source, with a placeholder for the key |
| `build_page.py` | inlines the key into the template |
| `digit-map-checkers-miniature.html` | the built page, 338 KB, opens as one file |
| `shoot_stills.mjs` | ten stills in light and dark through the page's `__seek` hook: four frozen mid-pass, two partway through a sweep, four at rest |
| `stills/` | those twenty stills, and `stills/gate/` with the check gate's 13 desktop-light and 13 phone-dark stills |

## Check results

**The check gate** (`check_scene.mjs`) walked 13 frames at desktop and phone width: 0 must-fix, 0 to check. **The leak check** found 0 email-shaped values. This repo has no people database configured, so the name list it checked against was empty.

**The stills were looked at, light and dark**: the forward pass mid-convolution, the training pass running into the kernel and stopping at the maps, the push pass entering the frozen U-Net and reaching the pixels, the broken run with g = 0, the noise sweep at t = 712, the push at b = 1.00, and the identity step (0.772 to 0.565, 0.352 to 0.146, gap 0.4205 to 0.4192). The stills script blocks Google Fonts because headless Chrome stalls on them from WSL, so its stills show the fallback faces. The gate's stills load the real ones.

## Look

The display face is Marhey, chosen here because the scope's diagram-prompts header records none. Colours are the ladder's own key from `../ladder.json`.

## Still to do, each needing the owner's yes

1. **Publish it as a private link.** Publish `digit-map-checkers-miniature.html` with the Artifact tool, add its URL to the `url:` line above, and file its scene-library rows.
2. **Codex stills and review.** Codex draws three to six stills that walk the page, then queues the read-only review every scene gets before it is published.
3. **Animation as video.** The `animate` skill renders the forward, training and push passes as a video. That is a separate request from this page's in-page animation.
4. **Real numbers in place of the miniature's.** One GPU job that saves both trained heads and their maps, logits and gradients on these canvases, as the ladder's closing page describes. The page would then show real values beside the illustrative ones.
