# 🧪 Review: on MNIST digits, does a pair-aware checker's push turn fused digits into two, and how good must the checker be?

**Nothing has run.** This file judges step 7b of
[the route from the toy to the animals](../route-classifier-guided-poe-from-toy-to-animals.md#7b--mnist-digits-on-a-canvas-how-good-must-the-checker-be),
which runs beside rung 2 of [the scale-up ladder](../../report/classifier-guided-poe/board.md#the-scale-up-ladder)
and gates nothing; its page is [A2b](../../report/classifier-guided-poe/a2b-mnist-digits-on-a-canvas.md).
The bar below was copied from section 7b on 2026-11-18, before any MNIST model trained, and the
step's walk reads it at the close.

- [x] 📝 bar written before the run: four readings on 7 & 8 and the five held-out pairs, in
      [The question written before the run](#the-question-written-before-the-run)

## Recommended prompt (when the run lands)

```
/analyze-run @plans/review/a2b-mnist-digits-on-a-canvas.md — <the five-version sheet's judge counts and the sweep figure's sidecar>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [the route, step 7b](../route-classifier-guided-poe-from-toy-to-animals.md#7b--mnist-digits-on-a-canvas-how-good-must-the-checker-be) | the design: canvases, U-Nets, the checker's recipe table, the columns, the sweeps |
| [A2b's page](../../report/classifier-guided-poe/a2b-mnist-digits-on-a-canvas.md) | the step's outcomes, for a reader of the report |
| [plan 24](../24-mnist-digits-on-a-canvas.md) | the tasks that build, train and render, and where they run |
| **this file** | **the bar, and later the verdict** |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

**A canvas**

A 32 × 64 one-channel picture with real MNIST training digits pasted onto it. Most hold one digit
at a random place; a share of them hold two different digits side by side, without overlap.

**The share**

The fraction of training canvases that hold two digits. Three U-Nets train, at shares 0, 0.01
and 0.1. At share 0 no training canvas holds two digits.

**Seen and held-out pairs**

The five held-out pairs, 0 & 3, 1 & 9, 3 & 4, 3 & 6 and 4 & 7 (drawn by `random.Random(0)` from
the 45 digit pairs), never appear together on a training canvas. The other 40 pairs fill the
two-digit canvases. 7 & 8 is a seen pair.

**Plain product**

The product of the "7" expert and the "8" expert of one U-Net, sampled with DDIM in 50 steps, with
no push.

**The pair-aware checker, and its push at strength s**

A small head on the frozen U-Net's decoder features that reads the noisy canvas, its step and the
asked pair, and outputs one log-odds that the canvas becomes two clear, right digits. Its push is
ε − s·√(1−ᾱ_t)·∇ log(d / (1−d)), at s = 0, 1, 3, 10 and 30, every step, capped at 0.1 of the
product prediction's norm.

**The separate-label push**

The same form, from two heads on the same features, "7 present" and "8 present". A fused 7 and 8
can satisfy both.

**The random push**

A random direction scaled to the pair-aware push's size at every step, on the same seed and s.

**Best-of-N**

N plain-product draws per seed with the checker picking one. N is matched to a guided draw's
measured cost; N = 16 is reported beside it.

**The judge, and the call ladder**

A digit classifier trained on single MNIST digits and run on each separate blob of a finished
canvas, never on canvases, so no guiding checker shares its training. Each tile is called on the
ladder one digit < fused < two digits, wrong identity < two clear, right digits. The headline count
is **two clear, right digits**. The judge's counts are used only after it agrees with the eye calls
on at least 58 of one 64-tile sheet. Claude makes those calls under the user's delegation, and the
user can veto any of them; until the user has checked them, the gate reads "validated against
Claude's calls".

**Evaluation seeds**

Seeds 0 to 63, never trained on. Every draw the checker trains on comes from seed 1,000 up.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim.** The claim is that a pair-aware checker's push turns a product that fuses two
digits into one that draws them apart, and the sweeps measure how accurate the checker must be for
that. Between plain product and the pair-aware column one mechanism differs (the push at a nonzero
s). Between the pair-aware and random columns only the push's direction differs. Between the
pair-aware and separate-label columns only what the heads were trained to say differs. Between
best-of-N and the pair-aware column the same checker is used two ways at matched compute.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| Timing: 200 training steps on the laptop's RTX 4070 | setup | not yet | about a minute per share | not yet | ⚠️ not started |
| Three U-Nets at shares 0, 0.01, 0.1 | builds the measuring tool | not yet | measured by the timing run | not yet | ⚠️ not started |
| The judge, with its fused test, against 64 eye-called tiles | builds the measuring tool | not yet | minutes | not yet | ⚠️ not started |
| Plain product on 7 & 8 and the five held-out pairs, 64 seeds, three shares (1,152 canvases); the support check at share 0.1 | establishes a baseline | not yet | not measured | not yet | ⚠️ not started |
| The checker and the separate-label heads, on draws from seed 1,000 up | builds the push | not yet | not measured | not yet | ⚠️ not started |
| The five-version sheet, seeds 0 to 63 | Tests the claim | not yet | not measured | not yet | ⚠️ not started |
| Sweep 1 (less data; flipped labels) and sweep 2 (the cap) | Tests the claim | not yet | not measured | not yet | ⚠️ not started |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ **On 7 & 8 and the five held-out pairs, 64 seeds each, does the pair-aware push give two
      clear, right digits where plain product fuses, more than the random push, the separate-label
      push and best-of-N at matched compute?**

      The bar, as section 7b wrote it before anything ran:

      On 7 & 8, 64 seeds, on the model where plain product fuses: plain product gives two clear,
      right digits on at most 8 of 64 (so there is something to fix); the pair-aware push gives at
      least 32 of 64 at some s; the random push and the separate-label push stay within 6 of plain
      product; best-of-N at matched compute stays at least 8 below the pair-aware push. On the five
      held-out pairs, at least 160 of 320. If more than one share fuses, the bar is read on share
      0.01, the lowest share whose training canvases hold two digits. A count out of 64 wobbles by about 4 by chance, so
      smaller gaps are not read.

      **1. Works on seen and held-out pairs.** The checker learned two digits apart.

      **2. Works on seen pairs only.** It learned the pairs, not plurality.

      **3. Too weak to read.** At s = 30 the push changed under half the canvases by eye, or the cap
      fired on more than half the steps.

      **4. No effect.** The pictures changed and no pairs came.

      A share where plain product already gives two clear, right digits on at least 32 of 64 seeds
      for 7 & 8 has nothing to fix and leaves the bar.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] ⚠️ **What does plain product do at each share, read before any checker trains?** Section 7b's
      rule: if plain product fuses at every share, MNIST reproduces SDXL's failure and the checker
      work goes ahead. If it composes at share 0.1 and fuses at 0, the checker work runs at share
      0.01 and on the held-out pairs. If it gives two clear, right digits on at least 32 of 64 for
      7 & 8 at every share, the checker work stops, and the finding goes to A2b as "plain product
      composes on MNIST".

- [ ] ⚠️ **The support check: alone, how often does expert 7 or expert 4 at share 0.1 draw a
      held-out partner beside its digit?** 1,000 seeds each. Share 0 is the control, because the
      product of experts is zero wherever either expert is (Bradley et al. 2025, section 3.2), so
      no push can make two digits there.

- [ ] ⚠️ **How accurate must the checker be? (sweep 1)** Each weaker checker's held-out [balanced
      accuracy](../../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/textbook/06-readout-guidance-on-one-small-network.md#balanced-accuracy-the-hit-rate-on-each-class-averaged) over steps 0 to 19, against its push's strict count of 64 on 7 & 8 at its best s,
      with SDXL's cat × dog checker (0.655) and CLIP on finished pictures (0.771) marked. The rule,
      written before any run: if the count crosses the bar below 0.771, step 10 keeps fetching
      cat × dog data; if it crosses only above 0.771, step 10 switches to a checker that judges
      each animal separately. The weaker checkers change one knob each: less data (1%, 5%, 25%,
      100% of the checker's canvases, clean labels) or wrong labels (10% and 20% flipped, all the
      data; the 0% flip is the 100% run).

- [ ] ⚠️ **Does the cap bind? (sweep 2)** The best checker at its best s with the cap at 0.1, 0.3
      and 1.0, with the share of steps the cap fired. On SDXL the 0.1 cap fired on every early step
      at strength 100.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Nothing yet.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?** At s = 0 every guided column matches plain product byte for
      byte. The random push's norm over the pair-aware push's is 1 at every step. Best-of-N's N
      comes from measured cost on the same card.

- [ ] ⚠️ **Was the measuring tool sound?** The judge agrees with the eye calls (Claude's, under the
      user's veto) on at least 58 of 64 tiles, including fused tiles that a confident single-digit read would miss (a fused
      7 and 8 read as a confident 8, 1.00, on the prototype).

- [ ] ⚠️ **Did the run respect its seeds?** No checker training draw comes from seeds 0 to 63; the
      trainer prints the seed range it read.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| a pair-aware push composes MNIST digits | the share it was read on, plain product's count on the same seeds, the random, separate-label and best-of-N counts, the held-out pairs' count out of 320 |
| checker accuracy decides whether the push composes | the sweep figure with 0.655 and 0.771 marked, one knob per run, and that MNIST has no text encoder or cross-attention |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| The four readings do not name the case where the pair-aware push reaches 32 of 64 but the random push, the separate-label push or best-of-N also come close (inside 6, or inside 8) | the user's call, before the five-version sheet renders | the reading |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Time 200 training steps on one `bigbatch` RTX 3090 beside the laptop's, then train the three U-Nets.
