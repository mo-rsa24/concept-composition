# 🧪 Review: can two separately trained digit experts be made to compose at inference, on either side, on 90 of 100 pairs

**Nothing has run.** This file judges [plan 33](../33-experts-that-compose-at-inference.md), the follow-on to
[plan 32](32-two-digit-experts-that-never-met.md). The bars below were written on 2026-11-19, before any experiment of this plan ran.

- [x] 📝 bars written before the run, in [The bars](#the-bars-written-before-the-run)

## Recommended prompt (when the run lands)

```
/analyze-run @plans/review/33-experts-that-compose-at-inference.md — <the tuning-pair counts per experiment and the final read's sheets>
```

## Table of contents

- [Why this is asked](#why-this-is-asked)
- [What the papers say](#what-the-papers-say)
- [Words this file uses](#words-this-file-uses)
- [Experiments](#experiments)
- [The bars, written before the run](#the-bars-written-before-the-run)
- [Controls](#controls)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [Readings and corrections](#readings-and-corrections)
- [Still open](#still-open)

## Why this is asked

Plan 32 found that a 0–4 expert and a 5–9 expert, each drawing its digit on either side, do not compose: both pushes gave 4
of 320 and the canvases were saturated blobs. Each expert says "exactly one digit of my kind is on the canvas", so a canvas
with one digit from each has zero density under both and the product has nothing to sample. A push or a sampler only
reweights or explores what the target contains. The question now is what the experts, or the sampling, must be so that the
product has that mass, and how far it can be taken: two experts that never met composing on 90 of 100 clean pairs, each digit
on either side, on every pair, none of which any expert has seen (an expert never sees the other's digits, so all 25 pairs of a
0–4 digit with a 5–9 digit are unseen).

## What the papers say

Read for this plan, each claim as the source states it:

- **Bradley, Nakkiran, Berthelot, Thornton, Susskind 2025** (arXiv 2502.04549): linear score combination composes exactly when
  the experts' conditionals factorise over a shared background (the factorised-conditionals condition). The project's reading
  register holds it. Its clutter section says objects in clutter make experts approximately factorised.
- **Bradley 2025b, "Local Mechanisms of Compositional Generalization in Conditional Diffusion"** (arXiv 2509.16447, read from
  its html, sections 3, 4 and appendices E.3 and E.4): a conditional model composes over conditions exactly when its score is a
  *local conditional score*, each pixel's score depending on a sparse set of pixels and of conditions (Lemma 1). The paper
  enforces this with an architecture, not a loss: the image is processed as a grid of cells (8 pixels of 128), each denoised from
  an overlapping patch of 5 × 5 cells (40 pixels) conditioned only on the conditions inside it, with the patch's absolute
  position appended; only the centre cell is kept. Trained on one object per image, the local model reaches 5 objects against 1
  for the standard model (table 1; 1 to 3 objects: 9 against 2). With the patch as large as the image (k = 8) the failure
  returns. SDXL is only partly local, spatially and not in its conditions, which the paper calls suggestive. It composes within
  one model; it does not compose separately trained models.
- **Liu et al. 2022** (arXiv 2206.01714) and **Du et al. 2023** (arXiv 2302.11552): conjunction by summing scores; Du et al.
  show reverse diffusion on the summed scores does not sample the product, and that Langevin or HMC correction steps and
  energy-based parameterisation fix some of it. Plan 33 reads these for the sampler experiment and has not re-read their
  experiments for this plan.

## Words this file uses

**Support.** The set of canvases a model gives non-zero density. The product of two experts has support only where both do.

**Either side.** The digit may sit in the left half or the right half (x from 0 to 4 or 32 to 36), at random height, as in
plan 32's `Canvases.one_half`.

**A clean pair.** The third judge's call "two clear, right, apart": exactly two blobs, one in each half, whose top digits over
0 to 9 are the asked pair, each at least 0.9 sure. The headline count on the final read is by eye, because the judge counts about
5% below a careful eye and fails on thick digits.

**The tuning pairs and the final pairs.** All 25 pairs of a 0–4 digit with a 5–9 digit are unseen. Five tune: 1 & 9, 2 & 8,
3 & 6, 4 & 7, 0 & 5, seeds 0 to 63. The other 20 are read once, at seeds 64 to 127, after one setting is chosen.

## Experiments

Each runs on the plan 32 setup (32 × 64 canvases, the same judge, DDIM in 50 steps).

| # | Experiment | Training | What it tests |
|---|---|---|---|
| E0 | **Region composition at inference.** The 0–4 and 5–9 either-side experts from plan 32, unchanged. Each seed gets a random side assignment; each expert denoises its half (full canvas in, or its half-canvas crop in) and the halves are joined: ε = mask·ε_A + (1 − mask)·ε_B | none | whether changing the target to "A's half and B's half" gives the product the mass it lacks |
| E1 | **Union base.** ε_A(a) + ε_B(b) − ε_union, the union being the step 7b share-0 model's no-digit prediction (trained on every digit, one per canvas) | none | the support argument: a union base does not give the product mass |
| E2 | **Experts trained with clutter.** The 0–4 expert and the 5–9 expert each see their digit plus 0 to 2 random non-digit objects (Fashion-MNIST items) at random places, labelled by the digit only; product over each expert's own no-digit prediction | two trainings, 40,000 steps | Bradley 2025's route: clutter as the shared background |
| E2b | **Local experts.** Patch-based experts after Bradley 2025b: each cell denoised from an overlapping patch conditioned on the digit and its side if the digit lies in the patch, centre cell kept, one expert per digit range, trained apart on one digit per canvas | two trainings | that locality, not clutter, is what lets separately trained experts compose |
| E3 | **Samplers.** Langevin or HMC correction steps (Du et al.), and the project's twisted SMC, on the best of E0, E2, E2b | none | whether sampling the product better closes a gap that remains |
| E4 | **The pushes.** The two pixel checkers retrained on the best product's own renders | minutes | reweighting once the product has mass |
| E5 | **A shared latent space.** A VAE over 0 to 9, experts on its latent, trained apart | VAE and two trainings | whether a shared decoder alone helps (predicted: it does not) |

The design of E2b's patch size, side conditioning and per-cell assembly is finalised in task 1.4 of the plan, from Bradley 2025b's
appendix E.3 as read above, and written here before E2b trains.

## The bars, written before the run

Counts are clean pairs out of 320 (five tuning pairs, 64 seeds each) unless a line says otherwise.

- [ ] **The headline.** On the final read, one chosen setting, the other 20 pairs, seeds 64 to 127, counted by eye: **at least 90%
      (1,152 of 1,280) clean pairs, and at least 70% (45 of 64) on every pair.** Anything below is not the goal; it is reported
      with its count.
- [ ] **E1, the union base.** Within 32 of plain product (null). If it gives 160 or more, the support argument is wrong and the
      plan stops to say so.
- [ ] **E0, region composition.** Supports at 288 (90%); promising at 160 (50%) or more, which sends it to E3 and E4; null under 32.
- [ ] **E2, clutter experts.** The same three lines. Each expert must pass 58 of 64 on its own digit first, by the judge, or it is
      retrained once.
- [ ] **E2b, local experts.** The same three lines, same own-digit check. The k = 8 control (patch as large as the canvas) must not
      support: if the control composes as well as the local model, locality is not what did it.
- [ ] **E3 and E4.** Each is read as a gain over the experiment it sits on: supports if it adds at least 64 of 320.
- [ ] **E5.** Reported against E0; predicted null.
- [ ] **No harm.** Whatever is chosen, on the fixed-side experts of plan 32 it must not lose more than 16 of 320 against that setting's
      plain count.

**What would surprise.** E1 giving any real count. E0 reaching the headline with no training. The k = 8 control composing.

## Controls

| Control | What it does | What comes back |
|---|---|---|
| each expert alone | samples one expert asked for its own digit | that the experts draw clear digits; floor for "clear" |
| plain product, no push, same base | the starting point of every experiment | the gain of each addition |
| the zero push or zero correction | a sampler with its extra term at zero | byte-identical canvases to plain product, or the sampler has a bug |
| the k = 8 control for E2b | a patch as large as the canvas | whether locality is the cause |
| the fixed-side experts of plan 32 | the case that partly composes | that nothing chosen makes it worse |

## Could the answer be an artefact

**Pair choice.** All 25 pairs are one 0–4 digit with one 5–9 digit; letters or other classes would be a different test. **The judge.**
Its strictness changes counts by about 5%, so the headline is by eye. **Tuning on the five pairs.** Settings are chosen on them and
read once on the other 20, fresh seeds. **Clutter and Fashion-MNIST.** The distractor objects are not digits, but they are a second
data source the experts share; that is the point of E2 and is stated, not hidden.

## Readings and corrections

Navigation: 📋 [TOC](#table-of-contents)

- **E0 and E1 on plan 32's either-side experts (2026-11-19), five tuning pairs, seeds 0 to 63, no training.** Clean pairs of 320: region composition with whole-canvas inputs **4**; region composition with each expert seeing only its half-canvas crop **124 (39%)**; union base **0**. The zero-push cell equals no push byte for byte. Against the bars: E1 is a null, as predicted, so the union base does not give the product mass; E0 with whole-canvas inputs is a null; E0 with crops is above the null (32) and below the promising line (160), so it is neither supported nor sent on by the bar as written. The crop result is the first non-zero composition of two experts that never met, with no training: each expert denoises only its own half.
- **E2 redesigned, written before it trains (2026-11-19).** The clutter is built from MNIST only, so no second dataset enters an expert. Two variants, trained as separate pairs: **fragments**, in which each expert sees its single digit plus one or two small cut-outs (10 to 16 pixels) of strokes from digits of its own range in the opposite half (half the canvases), and clutter-only canvases (a fragment or two, no digit) as its no-digit case; and **speckle**, in which each expert sees its digit with random bright specks over the canvas (density 0.5 to 2%), identical for both experts, and speckle-only canvases as its no-digit case. Fashion-MNIST is dropped. Product base for both: the mean of the two experts' no-digit predictions (both are clutter-only models). Both variants are trained with the same recipe as plan 32 (40,000 steps of 128, fp16, seed 0) and run alone against the 58-of-64 bar first.
- **Counting for E2 written before it runs.** Fragments appear in the finished canvases, so the strict count (exactly two blobs) would call them damaged. Both are reported: the strict count, and a **clutter-tolerant** count, which requires one blob confidently (at least 0.9) read as the asked digit a and one as b, in different halves, with other blobs allowed. The tolerant count is the one the bars apply to for fragments and is shown beside the strict count in every cell; the sheets are called by eye. Speckle is below the judge's blob size and needs no tolerance.
- **The pushes on the crop composition (2026-11-19), plan 32's either-side experts, five tuning pairs, seeds 0 to 63, default settings (strength 10, cap 0.1, eta 1.0), the pixel checkers of plan 32, no retraining.** Clean pairs of 320: crop composition alone **124** (by pair 1&9 37, 3&6 24, 4&7 28, 2&8 15, 0&5 20); with both pushes **241 (75%)** (57, 46, 41, 48, 49); with the one-half ink push **213** (56, 39, 40, 35, 43). Against the bars, E4's gain is +117 over the crop composition (the bar for a gain is 64): supported. The product is still short of the 288 that would support the composition on the tuning pairs and above the 160 line that sends it on: it goes to the sampler (E3) and to retraining the pair-aware checker on these renders (E4 proper). The headline bar (90% on the 20 final pairs) has not been read. The ink push, which never saw a canvas with two digits, adds 89 on its own.
- **E2c added, written before it runs (2026-11-19): post-hoc clutter on the pretrained experts.** Plan 32's 0–4 and 5–9 either-side experts, which never saw clutter, are each trained 4,000 more steps (about 8 passes of data, under 2% of their first training) on their own digit plus fragments of strokes from their own range, with clutter-only canvases as the no-digit case, starting from their saved weights and the same learning rate. They never see the other range's digits. Composed with the mean of the two experts' no-digit predictions as base, counted strict and clutter-tolerant on the five tuning pairs, and also with the crop composition. The question is whether clutter has to be there from the start or can be added after the experts exist. Bars as for E2; the comparison that matters is against the fragments experts trained from scratch.
- **What the checkers may have seen, a correction written before it runs (2026-11-19).** Plan 32's pixel checkers were trained on pasted pairs of every digit pair except the five tuning pairs, which includes the 20 cross-range pairs (a 0–4 digit with a 5–9 digit) that the final read uses. The 241 and 213 above are valid for the five tuning pairs, which the checkers never saw, but a final read through those checkers would not be a read on unseen pairs. For the final read the checkers are therefore retrained with no cross-range pair in their pasted data: pasted pairs only within a range (0–4 with 0–4, 5–9 with 5–9), with the same recipe, and their on-policy passes use pushed renders of the five tuning pairs only, seeds 1000 up. The one-half ink checker saw only single digits and needs no change. The retrained pair-aware and separate-label checkers are the ones the headline uses. This costs the checkers their best-matched data, so the tuning counts may fall; if they do, that is the honest number.
- **E4 round 1 and a fault in it (2026-11-19).** With the retrained no-cross checkers, the crop composition's pushed renders of the five tuning pairs (48 per pair, seeds 1000 up) were judged two clear, right digits on 170 of 240. The next checker training showed 0 drawn examples: the trainer drops pushed renders of the five tuning pairs by design (they were the held-out pairs of plan 32), so the pair-aware checker had been trained on pasted data only. The first checker, pair-nc1, is discarded. The trainer gets an explicit switch to keep the renders of the tuning pairs for this plan, and the pipeline restarts from the round 1 renders that exist (they do not depend on the fault). Tuning counts from the retrained checkers will carry the fact that they trained on pushed renders of the tuning pairs; the final read does not.
- **Next, before it runs.** The two pixel pushes already trained in plan 32 (both pushes and the one-half ink push) are tried once on the crop composition, default settings (strength 10, cap 0.1, eta 1.0). This is E4's first look and tests whether the 124 can be lifted. The clutter trainings (E2) are on hold until the reader decides what the clutter objects should be (a second dataset, Fashion-MNIST, versus objects built only from MNIST or drawn shapes): no clutter expert has trained.

## Still open

Letters and photos in place of digits, and the SDXL version, where the shared background is a photograph.
