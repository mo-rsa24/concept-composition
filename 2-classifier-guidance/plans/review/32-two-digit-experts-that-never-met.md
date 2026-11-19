# 🧪 Review: do two digit experts that never met compose into two digits, with and without the checkers' pushes?

**Nothing has run.** This file judges [plan 32](../32-two-digit-experts-that-never-met.md), a
follow-on to [route step 7b](../route-classifier-guided-poe-from-toy-to-animals.md#7b--mnist-digits-on-a-canvas-how-good-must-the-checker-be).
The bars below were written on 2026-11-19, before either expert trained.

- [x] 📝 bars written before the run, in [The bars](#the-bars-written-before-the-run)

## Recommended prompt (when the run lands)

```
/analyze-run @plans/review/32-two-digit-experts-that-never-met.md — <the guided sheets' judge counts and eye calls>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [route step 7b](../route-classifier-guided-poe-from-toy-to-animals.md#7b--mnist-digits-on-a-canvas-how-good-must-the-checker-be) | the step this follows: one shared U-Net, where both pushes gave 285 of 320 |
| [plan 32](../32-two-digit-experts-that-never-met.md) | the tasks that build, train and render, and where they run |
| **this file** | **the bars, and later the verdict** |

## Table of contents

- [Why this is asked](#why-this-is-asked)
- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Axes](#axes)
- [Runs](#runs)
- [The bars, written before the run](#the-bars-written-before-the-run)
- [Controls](#controls)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [Still open](#still-open)
- [Next step](#next-step)

## Why this is asked

Navigation: 📋 [TOC](#table-of-contents) | [Next](#words-this-file-uses) ➡️

Step 7b composed two digits inside one U-Net that had seen every digit. That is the case where a
joint prompt exists, and it is the weakest argument for a product of experts. The case that
justifies one is two experts trained apart, on data that never met, composed at sampling time.

With a blank canvas as the shared base, each expert's product with the other splits into four
arrangements: the two digits on different halves (two of the four, the co-occurrence) or on the
same half (the other two, a fused glyph). The fused arrangements gain weight as the noise falls,
because the digits' ink overlaps. So the plain product should give some co-occurrence but not
reliably, and the pushes' job here is to choose an arrangement the product already contains. If
they do it with experts that never met, the 7b result is a statement about composing separately
trained models, not only prompts inside one model.

## Words this file uses

Navigation: ⬅️ [Why this is asked](#why-this-is-asked) | 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

**A half-canvas digit**

One MNIST training digit on a blank 32 × 64 canvas, in the left half (x from 0 to 4) or the right
half (x from 32 to 36), at a random height from 0 to 4: the layout `Canvases.two()` already uses
for each of its two digits.

**The 0–4 expert and the 5–9 expert**

Two class-conditional canvas U-Nets with step 7b's recipe (64 channels, 40,000 steps of 128
canvases, Adam 0.0002, EMA 0.999). The 0–4 expert trains only on half-canvas digits 0 to 4, the
5–9 expert only on 5 to 9. Each digit's side is 50/50 left or right. Neither sees the other's
digits, and neither sees two digits on one canvas.

**The fixed-side experts**

The same, with the 0–4 digit always on the left and the 5–9 digit always on the right. This is
Bradley et al.'s exact case (factorized conditionals over a shared background), the control that
should compose with no help.

**The base**

The product divides by a base, and a blank canvas taken as an exact point mass is not usable: its
noise prediction (x + √ᾱ_t) / √(1 − ᾱ_t) diverges as the noise falls, 489 against about 1 for a
network at the last step, and the composed canvases blow out to white. Three bases are defined:
the blank canvas given a spread τ (noise prediction √(1 − ᾱ_t)(x + √ᾱ_t) / (ᾱ_t τ² + 1 − ᾱ_t), at
τ = 0.3 and 1.0), and the mean of the two experts' own no-digit predictions. The fixed-side
control picks the base: the one that passes its bar there, or the highest count if several do,
is used for every cell, and the others are reported beside it. Nothing about the side-mixed cells
is read to choose.

**The product**

ε_A(a) + ε_B(b) − ε_base, with digit a asked of the 0–4 expert and digit
b of the 5–9 expert; DDIM in 50 steps with eta 1.0, each seed with its own fixed noise.

**The pushes**

Step 7b's forms, s = 10 each, with the summed pushes capped at 0.1 of the product's size:

| Push | Its question | What it learns from |
|---|---|---|
| pair-aware | log-odds that the canvas becomes a, b on different halves, both clear | pasted pairs (positive), fused overlays, lone digits, wrong pairs (negative), then its own pushed renders labelled by the judge |
| separate-label | ten "is digit d there?" answers; the push raises a and b | pasted canvases with exact digit presence |
| one-half ink | two answers, "is there a digit on the left half?" and "on the right half?"; the push raises both | single half-canvas digits and blank canvases only: no canvas with two digits |
| random | a random direction scaled to the both-pushes size at every step | nothing |

All three checkers read the noisy canvas and the step directly, not either expert's inner maps, so
they belong to neither expert. Every canvas a checker trains on uses MNIST training digits; every
render it trains on comes from seed 1,000 up.

**The judge and the count**

Step 7b's third judge, unchanged. The headline count is **two clear, right digits on different
halves**. Each tile is also called fused, one digit, or wrong identity. Claude calls the headline
column by eye at full resolution; the judge runs about 5% below a careful eye.

**Pairs and seeds**

1 & 9, 3 & 6, 4 & 7, 2 & 8 and 0 & 5: one digit from each expert. Seeds 0 to 63, never trained on.
No knob is tuned in this plan: s, the cap and eta are 7b's.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#axes) ➡️

**Tests the claim** that the 7b pushes compose experts trained apart. Between plain product and
each guided column only the push differs; between the random and both-pushes columns only the
push's direction differs; between both pushes and the one-half ink push only what the checker
learned from differs; between the fixed-side and side-mixed plain products only the experts'
training placement differs.

## Axes

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

| Axis | Values | Carries the science? |
|---|---|---|
| expert placement | side-mixed; fixed-side | fixed-side is the control |
| push | none; random; both (pair-aware + separate-label); one-half ink | yes |
| pair | the five above | coverage |
| seed | 0 to 63 | coverage |

Six cells: side-mixed × the four pushes, fixed-side × no push, and fixed-side × both pushes as a
check that the pushes do no harm where none is needed.

## Runs

Navigation: ⬅️ [Axes](#axes) | 📋 [TOC](#table-of-contents) | [Next](#the-bars-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| Four experts: 0–4 and 5–9, side-mixed and fixed-side | builds the measuring tool | not yet | about 2 h each on one RTX 3090, all four at once | `/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/two-experts/models/` | ⚠️ not started |
| Each expert alone, 64 seeds | builds the measuring tool | not yet | a minute | the same folder, `alone/` | ⚠️ not started |
| Three pixel checkers | builds the push | not yet | about 10 min each | `two-experts/checkers/` | ⚠️ not started |
| The six cells, 5 pairs × 64 seeds | Tests the claim | not yet | minutes | `two-experts/cells/` | ⚠️ not started |

## The bars, written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#controls) ➡️

Counts are two clear, right digits on different halves, out of 320 (5 pairs × 64 seeds).

- [ ] **The experts are usable.** Each expert alone, asked each of its five digits on 64 seeds,
      draws one clear, right digit on at least 58 of 64 per digit by the judge. If not, the
      experts are retrained before anything else is read.

- [ ] **Fixed-side control: plain product composes.** At least 256 of 320 with at least one base. Below that, the
      blank base or the sampler is broken; stop and look before reading any other cell.

- [ ] **Side-mixed: how much co-occurrence the plain product gives.** Descriptive, no pass or
      fail. The κ argument predicts between 0 and 160. **Surprise:** more than 160, since the
      fused arrangements should gain weight as the noise falls.

- [ ] **Random push.** Within 16 of plain product. More than that, and any push's gain is read
      against the random push, not plain product.

- [ ] **Both pushes on the side-mixed experts.** Supports the claim at 224 or more **and** at
      least 64 above the random push. Null within 32 of plain product. Between the two,
      inconclusive.

- [ ] **The one-half ink push, no pair data.** Supports composing with no pair data at 224 or
      more. **Surprise:** within 32 of both pushes, which would mean the pair data is not needed.

- [ ] **Fixed-side with both pushes.** No more than 16 below fixed-side plain product.

- [ ] **Tuned read.** Where the default settings (s = 10, cap 0.1, eta 1.0) leave a guided cell below
      its bar, the loop may tune, as step 7b did. It chooses on the two tuning pairs 2 & 8 and
      0 & 5 at seeds 0 to 63, by the judge's count, one knob per round, at most 8 rounds: s in
      3, 10, 30; the cap in 0.1, 0.3; eta in 0.5, 1.0. The chosen setting is then read once, on
      the three other pairs (1 & 9, 3 & 6, 4 & 7) at fresh seeds 64 to 127, counted by eye, never
      read before. Its bars are the same shares of 192: 134 supports (the 224 of 320 above), 38 above
      the random push, 19 for a null, 10 for the random push and the side effect check.

## Readings and corrections

Navigation: ⬅️ [The bars](#the-bars-written-before-the-run) | 📋 [TOC](#table-of-contents)

- **Experts alone, side-mixed pair (2026-11-19).** The 0–4 expert draws one clear, right digit on 62, 63, 63, 58 and 59 of 64 for digits 0 to 4: the bar holds. The 5–9 expert gives 62, 62, 58, 55 and 62 for digits 5 to 9: it holds except on 8, at 55 against 58. Looked at by eye, the tiles for 8 are mostly clear 8s; the judge's misses are thin or open 8s read under 0.9 sure, and the judge counts about 5% below a careful eye. The written bar is the judge's count, so it is recorded as missed.
- **Correction, written before it runs.** The 5–9 expert is retrained once with the same recipe and a different seed (seed 1) on the idle card, since the fixed-side experts need 70 more minutes anyway. The retrained expert replaces the original only if it passes the bar on every digit. If it also misses, by 3 or fewer and on 8 alone, the original is kept and every cell reports the 8 shortfall beside its counts. Nothing else is changed, and no cell that used the original expert is read until this is settled.
- **A cosmetic fault in the sheets.** The expert-alone sheets framed every tile red because the count returns a numpy boolean; the counts were right and the frames are fixed in the script.

## Controls

Navigation: ⬅️ [The bars](#the-bars-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

| Control | What it does | What comes back |
|---|---|---|
| each expert alone | samples one expert with no partner | that the experts draw their own digits; the floor for "clear" |
| fixed-side plain product | the case Bradley et al. say composes exactly | that the blank base and the two-expert sampler work |
| random push | a push of the same size in a random direction | whether perturbation alone moves the count |
| fixed-side with both pushes | the pushes where no help is needed | whether the pushes damage a product that already composes |

## Could the answer be an artefact

Navigation: ⬅️ [Controls](#controls) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

**The checkers saw both digit ranges.** They are third parties trained on MNIST training digits
of every class, separately; they never see an expert's renders except their own pushed ones. The
experts never see each other's digits. This is the intended design and is stated in the write-up.

**Pasted pairs are pair data of a kind.** They are cut from single digits, not drawn by a model
that saw pairs. The one-half ink push exists to answer whether even that is needed.

**The judge undercounts thick drawn digits.** The headline column is counted by eye as well, and
both counts are reported.

## Still open

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

- Anywhere placement (a digit at any x from 0 to 36), where two independent picks miss each other
  only about 5% of the time; the harder follow-up if this plan supports the claim.
- An averaged base, half of each expert's no-digit prediction, against the blank base.

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Run [plan 32](../32-two-digit-experts-that-never-met.md); read its sheets against the bars above.
