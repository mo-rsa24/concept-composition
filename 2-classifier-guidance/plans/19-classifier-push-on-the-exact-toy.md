# 🧪 A classifier's push on the exact toy: separate-label against pair-aware

## Recommended prompt

```text
/drip-execute-plan 2-classifier-guidance/plans/19-classifier-push-on-the-exact-toy.md
```

## Position in the plan tree

[Root](../../MASTER_PLAN.md) › [the classifier-guided PoE route](route-classifier-guided-poe-from-toy-to-animals.md) ›
this plan, which is the route's step 7 and the board's row A1 in
[the classifier-guided PoE board](../report/classifier-guided-poe/board.md). It follows step 5, whose
SDXL sampler pushes unrescaled and capped, and it gates step 8: a miss here opens the diagnosis
ladder and stops the climb.

## Table of contents

- [Quick context](#quick-context)
- [Considerations](#considerations)
- [The claim under test](#the-claim-under-test)
- [Why this plan exists](#why-this-plan-exists)
- [What happens](#what-happens)
- [Description](#description)
- [Purpose](#purpose)
- [Goal](#goal)
- [Environment Facts This Plan Depends On](#environment-facts-this-plan-depends-on)
- [Success and failure outcomes](#success-and-failure-outcomes)
- [Tasks](#tasks)
- [Instructions](#instructions)
- [Outputs](#outputs)
- [Figure Catalog](#figure-catalog)
- [Error Matrix](#error-matrix)
- [Orchestration](#orchestration)
- [References](#references)
- [Next step](#next-step)

## Quick context

**What this is about**

The exact toy multiplies a circle expert and a square expert, each a mixture of Gaussians over
16 × 32 pictures. At pair weight 0.01 the exact product draws a circle and a square side by side
300 times of 300, and reverse diffusion on the summed noised scores draws them overlaid 300 times of
300 (seed 7). This plan adds a classifier's gradient to that sampler at every step and asks which
classifier fixes it.

**The two classifiers**

A separate-label classifier answers "is there a circle" and "is there a square". A pair-aware
discriminator answers "is this an exact pair draw or one of the sampler's own draws", trained the
way Kim et al. 2023 train theirs.

## Considerations

**The cap**

Step 5 caps the SDXL push at 0.1 of the product's own step. On this toy the true correction is
about half the summed score's size from step 200 to step 50, and capped at 0.1 it composes 4 of 300
against 300 uncapped. The toy runs uncapped, with caps of 0.1, 0.3 and 1 reported beside as a
diagnosis.

**What the discriminator can learn**

Its fake set is the sampler's finished draws noised forward, so its best log-odds gradient is
∇ log q_joint,t − ∇ log q_sampler,t. That differs from the true correction, which is ∇ log of the
true noised product minus the summed scores, so the push is read against both.

## The claim under test

A separate-label classifier's push only re-weights the product, because the summed scores already
are ∇ log p(x | circle) + ∇ log p(x | square). A pair-aware discriminator's push carries the term
the product drops.

## Why this plan exists

The route climbs from the toy to SDXL cat × dog. The toy is the one stage where the true correction
is known in closed form, so a pass here says the method can work and a miss says it cannot, before
any GPU-hour is spent on photos.

## What happens

Five pieces, one per round, each ending on a test run by hand: the push hook, the separate-label
classifier and its push, the pair-aware discriminator, its push and the bar, then the diagnosis.

## Description

Code in `3-understanding-compositionality/artifacts/notes/what-poe-loses-when-it-composes/`:
the hook in `toy_sampler_hybrids.py` (`guided_eps` and `reverse(push=, scale=, rng=, n=)`), and
everything else in `toy_classifier_push.py`, one subcommand per piece. Outputs in
`3-understanding-compositionality/artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/`.

## Purpose

Settle the board's row A1 and give step 8 a working recipe or a diagnosed miss.

## Goal

**The bar, written before any classifier is trained.** At pair weight 0.01, seed 7, 300 draws,
uncapped:

- the pair-aware push at weight w = 1 draws at least 150 of 300 crisp pairs, with overlaid draws no
  higher than this plan's own unguided run on the same seed (300 of 300);
- the separate-label push draws at most 15 of 300 crisp pairs at every scale in 1, 2, 5 and 10.

w = 0.5 and 2 are reported beside the bar and do not decide it.

**Time box:** 4 sittings, about 3 h 45 of the reader's time, 7.5 h budget.

## Environment Facts This Plan Depends On

- torch 2.5.1 with CUDA on the laptop's RTX 4070 is in
  `/home/molef/micromamba/envs/sdxl-local/bin/python`; the base miniconda Python has no torch.
- `artifacts/results/` is gitignored, so figures and counts stay on the laptop; scripts and this plan
  are committed.

## Success and failure outcomes

| Outcome | What it means | What follows |
|---|---|---|
| both halves of the bar pass | a pair-aware push supplies what the product drops, on a toy where that is checkable | step 8, our photos |
| pair-aware under 150 | the push does not carry enough of the correction | the diagnosis ladder: cosine per step, the minimal-pair gap, the cap and w sweeps |
| separate-label over 15 | the maths' claim that it only re-weights is wrong on this toy | stop and re-derive before step 8 |

## Tasks

For Claude to execute. Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** `/verify-plan` on this file, then `/xref-pillar` on it, at the walk's close.

### 1. 🔧 Build the five pieces

▶ Next: [2. Test each piece](#2-test-each-piece)

- [x] **1.1** The push hook: `reverse()` takes `push`, `scale`, `rng` and `n` and enters the push
  through `guided_eps`; `toy_classifier_push.py port` runs unguided, scale 0, the true correction at
  scales 0.5 and 1, and the true-score sampler, on seed 7. ✓ verified 2026-11-17: with the
  condition_score step patched in a throwaway process, scale 0 differs from unguided on 0 of 300
  draws and scale 1 matches the true-score sampler to 7.3e-8; left untyped, the run stops on
  `NotImplementedError`
- [x] **1.2** The separate-label classifier, trained on noised draws from the seven pictures at the
  experts' weights, with the closed-form best classifier beside it; its push at scales 1, 2, 5 and 10.
  `toy_classifier_push.py separate-train`, then `separate-guide`. ✓ verified 2026-11-17: 4,000
  iterations in 12 s on the RTX 4070; accuracy within 0.012 of the closed-form best at steps 10, 50,
  100, 150, 190 and 200 · `separate-label-train.json`, `separate-label-classifier.pt`
- [x] **1.3** The pair-aware discriminator: 3,000 exact pair draws against 3,000 summed-score draws,
  noised forward to a random step, binary cross-entropy weighted as Kim et al. Eq. 4.
  `toy_classifier_push.py disc-train`, with `disc_loss` and `log_odds_grad` left for the reader.
  ✓ verified 2026-11-17: `discriminator-data.npz` holds 3,500 exact draws (3,500 crisp) and 3,500
  sampler draws (3,470 overlaid, 30 faint), seeds 101 and 201 to 205; with both functions patched in a
  throwaway process the trainer separates the held-out draws and the gradient check passes
- [x] **1.4** Its push at w = 0.5, 1 and 2, and the bar's pass or miss line.
  `toy_classifier_push.py pair-guide`, every step, uncapped, the logit clipped at ±11.5 as Kim et al.
  ✓ verified 2026-11-17: runs end to end in 26 s in a throwaway folder and writes the bar line
- [ ] **1.5** The diagnosis: cosine per step against the true correction and against the sampler's own
  ratio, the minimal-pair gap against a shuffled-label floor, the cap sweep; the figures, card,
  finding in `report/`, route rows and the board's A1 row.

## Instructions

For you to follow manually. Do these yourself.

### 2. Test each piece

◀ Needs: [1. Build the five pieces](#1-build-the-five-pieces)

- [x] **2.1** Type `guided_eps` in `toy_sampler_hybrids.py`, then from the toy's notes folder run
  `/home/molef/micromamba/envs/sdxl-local/bin/python -I toy_classifier_push.py port`.
  You should see the guided noise at scale 0 equal to the product's bit for bit on 200 of 200 steps
  in the same loop, a largest difference under 1e-6 between the scale-1 push and the true-score
  sampler, and 300 crisp pairs at scale 1. Scale 0 against a separately run unguided sampler is
  reported too; numpy's summation order can move its last bit between processes. `NotImplementedError`
  means the step is not typed yet; 0 crisp at scale 1 means the sign of the push is flipped.
  ✓ verified 2026-11-17, typed and run by Claude at the reader's word ("you do it"): bit for bit on
  200 of 200 steps; against a separate unguided run 0 of 300 draws differ; scale 1 against the
  true-score sampler 6.7e-16; crisp pairs of 300: unguided 0, scale 0 0, scale 0.5 112, scale 1 300,
  true score 300 · `port.json`
- [x] **2.2** From the toy's notes folder run
  `/home/molef/micromamba/envs/sdxl-local/bin/python -I toy_classifier_push.py separate-guide`
  (about 60 s). You should see nine count lines (unguided, then the trained and the best possible
  classifier at scales 1, 2, 5 and 10) and a bar line naming the trained push's highest crisp count
  with pass or miss. `FileNotFoundError` on `separate-label-classifier.pt` means `separate-train` has
  not run. ✓ verified 2026-11-17, run by Claude at the reader's word ("you do it"): crisp pairs of 300,
  trained at scales 1, 2, 5, 10: 0, 0, 2, 10; best possible: 0, 1, 13, 34; unguided 0 · the trained
  push's highest is 10, inside 15 · `separate-label-guide.json`
- [x] **2.3** Type `disc_loss` and `log_odds_grad` in `toy_classifier_push.py`, then from the toy's
  notes folder run `/home/molef/micromamba/envs/sdxl-local/bin/python -I toy_classifier_push.py disc-train`
  (about 20 s). You should see the loss fall from about 0.69, a gradient check with a largest relative
  error under 1e-2, and held-out accuracy near 1.000 on both kinds at steps 10 to 100, falling toward
  chance by step 200. `NotImplementedError` names the function not yet typed; accuracy below 0.5 on
  both kinds means the labels are flipped; a relative error near 2 means the gradient's sign is flipped. ✓ verified 2026-11-17, typed and run by Claude at the reader's word ("you do it"): gradient
  check 3.3e-4; held-out pairs called real and sampler draws called fake, 1.000 and 1.000 at steps 10,
  50 and 100, 0.956 and 0.980 at 150, 0.646 and 0.662 at 190 · `pair-aware-train.json`,
  `pair-aware-discriminator.pt`
- [x] **2.4** From the toy's notes folder run
  `/home/molef/micromamba/envs/sdxl-local/bin/python -I toy_classifier_push.py pair-guide` (about 30 s).
  You should see four count lines (unguided, then w = 0.5, 1 and 2) and a bar line for w = 1 with
  pass or miss. `FileNotFoundError` on `pair-aware-discriminator.pt` means `disc-train` has not run. ✓ verified 2026-11-17, run by Claude at the reader's word ("you do it"): crisp pairs of 300,
  unguided 0, w = 0.5 16, w = 1 172, w = 2 300; overlaid at w = 1 128 against unguided's 300 · bar line
  `pass` · `pair-aware-guide.json`
- [ ] **2.5** Run `/home/molef/micromamba/envs/sdxl-local/bin/python -I toy_classifier_push.py diagnose`
  from the toy's notes folder (about 2.5 minutes), then open `push-cosine-per-step.png` and
  `minimal-pair-gap-per-step.png` in the card's folder. You should see the push's cosine with the
  sampler's own ratio near 0.85 to 0.96 from step 190 to step 90 while its cosine with the true
  correction stays under 0.25, both dropping to 0 by step 80; and the discriminator's gap rising to
  about 28 log-odds while the shuffled-label line stays at 0. A shuffled-label gap away from 0 means
  the gap is not about the pair.

## Outputs

`port.json`, `port-samples.npz` and `draw-0-unguided-and-pushed.png` from piece 1; the classifier
checkpoints, counts and figures from pieces 2 to 5; the card `README.md`; a finding under
`2-classifier-guidance/report/`.

## Figure Catalog

| Figure | What it shows |
|---|---|
| `the-side-by-side-discriminator.png` | the discriminator as seven stages: the two piles with one real draw each, the noising, the network and its loss, d with held-out accuracy per step, the push, where it enters the sampler, and the crisp counts it gives |
| `push-cosine-per-step.png` | the push's cosine (y, −0.1 to 1) with the sampler's own ratio and with the true correction, against the reverse step (x, 200 to 0), median of 300 draws along the guided run at w = 1 with the 25th to 75th band · `diagnosis.json` |
| `minimal-pair-gap-per-step.png` | log-odds on the side-by-side pair minus on the stacked picture (y) against the noise step (x, 200 to 0), for the discriminator, the separate-label classifier and a discriminator trained on shuffled labels, 256 shared noise draws per step · `diagnosis.json` |
| `draw-0-unguided-and-pushed.png` | draw 0 of seed 7 at its final step: unguided, the true correction at scale 0, and at scale 1, each titled with its label |

## Error Matrix

| Symptom | Cause | Fix |
|---|---|---|
| `NotImplementedError: type the guided step here` | `guided_eps` not typed | type it |
| `ModuleNotFoundError: No module named 'torch'` | run with the base miniconda Python | use the `sdxl-local` Python |

## Orchestration

The walk map is `2-classifier-guidance/artifacts/drips/classifier-push-on-the-exact-toy/EXECUTE_WALK.md`.

## References

- [the route, step 7](route-classifier-guided-poe-from-toy-to-animals.md#7--port-the-push-to-the-exact-toy-separate-label-against-pair-aware)
- [the Kim et al. read map's closing section](../artifacts/drips/read-refining-generative-process-discriminator/READ_MAP.md#what-a1-takes-from-this)
- [the step 4 card, where the push enters their sampler](../artifacts/results/does-classifier-guidance-work-on-the-released-weights/README.md#where-the-push-enters-their-sampler)
- [the toy's co-occurrence card](../../3-understanding-compositionality/artifacts/results/the-exact-toy/how-much-co-occurrence-the-product-needs/README.md)

## Next step

The route's step 8, our photos, once the bar passes.
