# Verdict: can the score path be a third space beside text and noisy canvases? (step X9)

**On digits, the score path carries which digits are present, below the bar.** At the noise levels that match SDXL's first 16 steps, the digit model's ten predictions on one noisy canvas read digit presence at up to 0.700 (balanced accuracy, 0 to 1, 0.5 is chance), against 0.593 for the same probe on the canvas itself and a bar of 0.75 ([the X5 review](05-signal/review.md)). At the brief's own steps 0 to 15, which are far noisier on the digit model, nothing read the digits. By the brief's rule no shared space was trained, and the owner confirmed skipping steps X6 to X8 on board question 76.

<a href="05-signal/presence-by-timestep.png"><img src="05-signal/presence-by-timestep.png" width="820" alt="Two panels of presence accuracy against timestep. Early window: every input at chance except the error readout at 0.61 to 0.66. SDXL-matched window: the signature rises from 0.50 to 0.70, above the noisy canvas at 0.59 and the path length at 0.62; raw noise stays at 0.50; nothing reaches the 0.75 bar."></a>

## Table of contents

- [Words this uses](#words-this-uses)
- [One row per variation](#one-row-per-variation)
- [What it suggests for Paper 1, the separate-experts paper](#what-it-suggests-for-paper-1-the-separate-experts-paper)
- [What it suggests for Paper 2, the CO3 paper](#what-it-suggests-for-paper-2-the-co3-paper)
- [What was asked and what was done](#what-was-asked-and-what-was-done)
- [Still open](#still-open)

## Words this uses

**Signature.** The ten maps $d_t(c) = \epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing)$, $c = 0, \dots, 9$, on one noisy canvas, signed and unscaled. It is the repaired score path of [the ideas page](03-ideas.md#a-the-score-path-itself): the brief's own form, $d_t$ under the canvas's own labels, hands a reader the answer.

**SDXL-matched window.** The digit model's timesteps 712 down to 480, which carry the same share of signal as SDXL's steps 0 to 15. The brief's steps 0 to 15 on the digit model are timesteps 999 to 693, where the canvas holds 0.6% to 8.7% of its signal.

## One row per variation

| Variation | What was matched | Bar | Result | Verdict |
|---|---|---|---|---|
| Score path carries the digits (X5) | a probe for digit presence on the signature, against the canvas and the raw noise | presence at least 0.75 at some step, and at least 0.1 above the raw noise | 0.700 at t 480, 0.20 above the raw noise; 0.513 in the brief's early window | **null**, W&B `gq51nl74` |
| Canvas with score path | dropped as its own space before any run, because the signature is a fixed function of the canvas ([B2a](03-ideas.md#b-ideas-for-one-shared-space)); its question was read by X5's comparison | none of its own | the signature leads the canvas by 0.107 at t 480 with the same probe | answered inside X5 |
| Text with canvas | W4's model, Loop 9's, reused as the baseline | unseen-pair balanced accuracy at least 0.75, within 0.05 of seen pairs | owned by Loop 9; no result when this loop closed | not this loop's |
| Text with score path | not trained, because X5 missed | the same as W4 | none | not run |
| All three in one space | not trained, because X5 missed | the same as W4 | none | not run |
| Sum lands on the drawn caption (X8, test 1) | not run | top-1 at least 0.75 on composed product paths | none | not run |
| Gap follows the count checker's push (X8, test 2) | not run | mean cosine interval above the shuffled control's | none | not run |

**Two findings came out of X5 beside the verdict.**

**The signature reads identity where Loop 8's maps did not.** On two separate digits, the asked pair against one wrong digit, it reaches 0.655 at t 480. Loop 8's map reader scored 0.511 on the same comparison in the same window. It used the asked pair's two maps, normalised and unsigned ([the V2 review](../abstract-checkers/05-does-it-learn/review.md#the-rehearsal-at-sdxls-noise-level)). The difference is the three things the signature keeps: all ten labels, the sign and the size.

**The diffusion classifier's own readout reads the most and cannot steer.** The per-label error against the true injected noise reads presence at up to 0.746 at t 495, and at 0.61 to 0.66 in the early window where nothing else reads. It needs the noise that made the canvas, so it is a judge of canvases noised from a known picture, never a push (inferred, [the X5 review](05-signal/review.md#what-the-other-inputs-read)).

## What it suggests for Paper 1, the separate-experts paper

Paper 1's checker reads where each expert's label acts on a shared canvas ([the problem statements](../problem-statements.md)). Loop 8's identity checker for that setting missed at 0.555. Its maps were the asked pair's two, made positive and normalised to sum to 1 (the `base` input of `READERS` in `2-classifier-guidance/scripts/abstract_checkers/v2_learn.py`), so they dropped the other labels, the sign and the size. X5 suggests feeding the checker every label's $d_t$, signed and unscaled, at the SDXL-matched noise. On digits that input read the pair against a wrong digit at 0.655, where the normalised maps read 0.511 (inferred). Each expert trained apart has its own null prediction, so each expert's signature subtracts its own (inferred, untested). The suggestion goes to Loop 8 and the classifier-guidance line, which own that checker. It also feeds idea D5 of [the ideas page](03-ideas.md#d-ideas-for-the-two-papers-that-need-no-training), the side vote, which this loop did not run.

The early window says where not to push: on the digit model nothing reads the digits at timesteps 999 to 693, so a checker there has a gradient made of noise. Loop 8 already pushes at 712 to 480 for this reason.

## What it suggests for Paper 2, the CO3 paper

CO3 already places two bodies, and its failures are which body carries which name ([the Loop 6 page](02-the-loops/loop-6-find-the-best-method.md)). X5 suggests the identity signal is in each subject's own signed $d_t$ more than in the noisy latent, at the noise SDXL has at its first 16 steps. That supports idea D3: read each subject's $d_t$ at CO3's corrected steps as a map of which body it acts on. That is Loop 6's to run on SDXL (inferred). The digit result transfers only as a direction to try. SDXL reads text through cross-attention, and a full signature would need one U-Net call per candidate name.

The diffusion classifier's readout suggests a judge for Paper 2's evaluation: score a render's right caption against a swapped one by the model's own error, from unguided predictions (idea D2). It needs no training and reads more than any trained input here. Its cost is many noise samples per caption pair (inferred).

Nothing here tested seed ranking by score-path length (D1), CO3's concept weights from a path average (D4) or the Langevin corrector on CO3 (D6). They pass to Loop 6 as untested ideas.

## What was asked and what was done

| Step | Asked | Done | Where |
|---|---|---|---|
| X1 | read the papers, search for score-path encoders | 12 papers read in full, 301 quotes checked; no paper trains an encoder on a stack of noise predictions | [the reading table](01-reading/reading-table.md) |
| X2 | one page per running loop | five pages in one layout | [the loop pages](02-the-loops/) |
| X3 | ideas and their assumptions | every assumption marked; the score path repaired to all ten labels; no joint score path exists on any digit model | [the ideas page](03-ideas.md) |
| X4 | one design, the owner's check | three run groups, about 4 GPU-hours; yes on board question 71 | [the design](04-design.md) |
| X5 | does the score path carry signal | null by the bar; 0.700 at SDXL-matched noise | [the X5 review](05-signal/review.md) |
| X6 to X8 | train, draw and test the spaces | skipped by the brief's rule, confirmed on board question 76 | this page |
| X9 | the verdict | this page | |

**GPU-hours:** 0.74 of the loop's 6, all on `bigbatch` RTX 3090s, in jobs 94160, 94180 and 94184.

**The bar:** X5, signature presence at least 0.75 at some step and at least 0.1 above the raw noise: **MISS**, 0.700 and 0.20.

## Still open

- Where the signature reaches 0.75, if it does, at timesteps below 480. Reading it needs a new design. Those steps no longer match SDXL's early steps.
- Whether a signature taken along the sampler's own path reads the same as one taken on canvases noised from a known picture.
- Whether the 16 early steps stacked, which read 0.607 where each step alone reads chance, carry the digits or only cancel a noise draw they share.
- W4, the text-with-canvas baseline, belongs to Loop 9 and had no result when this loop closed.
