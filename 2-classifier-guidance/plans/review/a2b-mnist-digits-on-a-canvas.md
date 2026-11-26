# 🧪 Review: on MNIST digits, does a pair-aware checker's push turn fused digits into two, and how good must the checker be?

**Nothing has run.** This file judges step 7b of
[the route from the toy to the animals](../route-classifier-guided-poe-from-toy-to-animals.md#7b--mnist-digits-on-a-canvas-how-good-must-the-checker-be),
which runs beside rung 2 of [the scale-up ladder](../../report/classifier-guided-poe/board.md#the-scale-up-ladder)
and gates nothing; its page is [A2b](../../report/classifier-guided-poe/24-mnist-one-model.md).
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
| [A2b's page](../../report/classifier-guided-poe/24-mnist-one-model.md) | the step's outcomes, for a reader of the report |
| [plan 24](../24-mnist-digits-on-a-canvas.md) | the tasks that build, train and render, and where they run |
| **this file** | **the bar, and later the verdict** |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Follow-on loops, written before they run](#follow-on-loops-written-before-they-run)
- [Controls and improvements on the final setup](#controls-and-improvements-on-the-final-setup-written-before-they-run)
- [Loop 6: the weakest held-out pairs](#loop-6-the-weakest-held-out-pairs-written-before-it-runs)
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
| Timing: 200 training steps, share 0.1 | setup | 2026-11-18, laptop RTX 4070; jobs 89907 and 89908 on `bigbatch` RTX 3090s, commit `8e88f6fe` | 200 steps each | `/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/models/share-0.1/timing-*.json` | done: 486 ms per step on the laptop, 157 fp32 and 112 bf16 on a 3090; 105 min per share at 40,000 steps, fp32 chosen |
| Three U-Nets at shares 0, 0.01, 0.1 | builds the measuring tool | 2026-11-18 07:27 UTC, jobs 89909, 89910, 89911 on `mscluster86`, `87`, `88`, commit `8e88f6fe` | 40,000 steps each, 109 to 120 min | `/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/models/share-<share>/unet.pt` (share 0 is `share-0.0`) | done |
| The judge, with its fused test, against 64 eye-called tiles | builds the measuring tool | 2026-11-18 on the laptop's RTX 4070 | about 5 min | `artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs/judge-gate/` | done: 58 of 64 against Claude's blind eye calls (gate 58), validated against Claude's calls until the user checks them |
| Plain product on 7 & 8 and the five held-out pairs, 64 seeds, three shares (1,152 canvases at g = 1, the same again at g = 3 as a diagnostic); the support check at share 0.1 | establishes a baseline | jobs 89917, 89918, 89964 (share 0 rerun after 89916 asked for the wrong folder name), commit `9aa53551` | 1 to 3 min per share | `/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/plain-product/share-<share>/` | done: fuses at every share |
| The checker and the separate-label heads, on draws from seed 1,000 up | builds the push | draws: job 89936 (3,600 draws, 111 seen and 5 held-out judged two clear, right digits); heads: job 89965 from 09:42 UTC | about 5 min per head | `/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/checkers/` | ⚠️ running |
| The five-version sheet, seeds 0 to 63 | Tests the claim | not yet | not measured | not yet | ⚠️ not started |
| Sweep 1 (less data; flipped labels) and sweep 2 (the cap) | Tests the claim | not yet | not measured | not yet | ⚠️ not started |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [x] ✅ **On 7 & 8 and the five held-out pairs, 64 seeds each, does the pair-aware push give two
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

      **Reading 2, works on seen pairs only, settled by the user on 2026-11-18, with two departures
      named:** the separate-label push composes some (15 of 64, above the bar's 7), and reading 3's
      cap test fires at s = 30 while raising the cap changes nothing at s = 10. Counts by Claude's eye,
      because the judge failed on drawn digits.
      On 7 & 8 at share 0.01, two clear, right digits of 64: plain product 1; pair-aware push 28 at
      s = 3 and 47 at s = 10; random push 1 at s = 10; best-of-2 at matched cost 1 (best-of-16, 16);
      separate-label push 11 at s = 10 and 15 at s = 30. Four of the five 7 & 8 lines hold; the
      separate-label line does not (15 against at most 7). Held-out pairs: 84 of 320 at s = 3 and 82
      at s = 10, against 160 (0 & 3 at 3 and 0, 1 & 9 at 14 and 3, 3 & 4 at 25 and 34, 3 & 6 at 11 and
      14, 4 & 7 at 31 and 31); most other held-out canvases hold two digits with one wrong. The
      closest reading is 2, works on seen pairs only, with two departures: the separate-label push
      composes some, and reading 3's cap test fires at s = 30 (half the samples or more clipped on
      40 of 50 steps) while raising the cap at s = 10 changes nothing (cap 1.0: 49 by eye, 28 by
      the judge against 30 at cap 0.1). `five-versions/eye_counts.json`, `sweep1_eye.json`.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [x] ✅ **What does plain product do at each share, read before any checker trains?** Section 7b's
      rule: if plain product fuses at every share, MNIST reproduces SDXL's failure and the checker
      work goes ahead. If it composes at share 0.1 and fuses at 0, the checker work runs at share
      0.01 and on the held-out pairs. If it gives two clear, right digits on at least 32 of 64 for
      7 & 8 at every share, the checker work stops, and the finding goes to A2b as "plain product
      composes on MNIST".

      **It fuses at every share, so the checker work went ahead.** Two clear, right digits on 7 & 8
      out of 64, at g = 1, judge (Claude's eye in brackets): share 0, 0 (not eye-called); share
      0.01, 0 (1); share 0.1, 1 (4). On the five held-out pairs the judge counts 0 at shares 0 and
      0.01 and 0 to 3 at share 0.1. At share 0.1 the product often draws two digits with one of
      them wrong (by eye 10 of 64 on 7 & 8: a 9 beside an 8, a 3 beside an 8, a 5 beside a 7).
      `artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs/plain-product/`.

- [x] ✅ **The support check: alone, how often does expert 7 or expert 4 at share 0.1 draw a
      held-out partner beside its digit?** 1,000 seeds each. Share 0 is the control, because the
      product of experts is zero wherever either expert is (Bradley et al. 2025, section 3.2), so
      no push can make two digits there.

      **Rarely.** Of 1,000 canvases each, expert 7 alone drew a 4 beside it on 2, and expert 4
      alone drew a 3 on 2 and a 7 on 3; they drew some other digit beside them on 95 and 81.
      `counts-share-0.1.json`.

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

## Follow-on loops, written before they run

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

Started on the user's word after the draft reading. None of them changes the bar above.

- [x] ❌ **Loop 1: can a judge count drawn digits?** The judge still learns from single digits and
      overlays only; its augmentation, thresholds and ladder rule may change. Reference: Claude's
      eye calls, binary (two clear, right digits or not). Tuned on 12 eye-called 7 & 8 sheets
      (plain at shares 0.01 and 0.1; the pair-aware push at s = 3, s = 10 and s = 10 with cap 1.0;
      separate-label at s = 10 and 30; random at s = 10; best-of-2; best-of-16; the 1%-data and
      20%-flip checkers at s = 10). Tested once per try on 13 held-back sheets (the ten held-out
      pair sheets and the 25%-data, 5%-data and 10%-flip checkers at s = 10). **Passes** when the
      held-back agreement is at least 58 of 64 per sheet on average and no held-back sheet's count
      differs from the eye's by more than 4. At most 5 tries; if none passes, counting stays by eye.

      **Three tries, none passes; two left.** Held-back agreement per sheet, mean of 64, and the worst
      count gap: try 1 (the second judge, its rule tuned) 56.9 and 6; try 2 (a third judge trained
      with rotation, slant, scale and three stroke widths; 99.0% on MNIST's test digits) 57.2 and 6;
      try 3 (the three judges averaged) 56.7 and 9. Every try's rule drops the "two digits on one
      spot" class and asks 0.7 to 0.9 confidence per digit. Disagreements split 45 judge-yes against
      43 eye-yes of 832, and several eye calls are wrong on a second look, so the reference is noisy
      by a few tiles per sheet; try 2's total over the 13 sheets is 2 above the eye's 241. The 88
      disagreeing tiles went to the user, shuffled and unlabelled; their calls replace both on those
      tiles and try 2 is rescored once, unchanged. `data/mnist-canvas/24-mnist-one-model/judge-loop/`.

      **Closed without a pass, by its own rule.** The user delegated the referee call; Claude re-read the
      88 shuffled tiles blind to who said what (siding with its first read on 63, with the judge on
      25). Rescored once against that reference, try 2 agrees on 59.2 of 64 per sheet (passes 58)
      but its worst count gap is 6 (1 & 9 at s = 3; needs 4 or less). Counting stays by eye; the
      judge's total runs about 5% low, so in loop 2 it only ranks rounds on the tuning pairs and
      the chosen round is counted by eye on the held-out pairs. `try2-rescored.json`.

- [x] ✅ **Loop 2: does the second digit come out right on new pairs?** One change per round to the
      checker against round 0, the checker retrained without three seen pairs (0 & 6, 1 & 2, 6 & 9,
      drawn by `random.Random(1)`). Each round is chosen on those three pairs (their pair-aware push
      at s = 3 and 10, counted by the loop-1 judge) and only the chosen round is judged on the five
      held-out pairs. **Support:** at least 160 of 320 two clear, right digits on the held-out
      pairs at its best of s = 3 and 10. At most 6 rounds; runs only once loop 1 passes.
      Rounds: (0) the base without the three pairs, drawn canvases relabelled by the loop-1 judge;
      (1) the same canvas asked for the wrong pair as an extra negative; (2) wrong-pair negatives at
      three times their share; (3) the pair fed to every layer of the head; (4) and (5) the two
      best changes combined, then that combination at a second seed.

      **Extended by the user on 2026-11-18 to ten rounds**, after rounds 0 to 3 and before any
      held-out read: (6) the push's strength scheduled by sampler step, strong early and weak late,
      rendering only; (7) the sampler made partly random (DDIM eta above 0) under the push,
      rendering only; (8) the checker reading the digit model's features asked with each of the two
      digits instead of with no digit; (9) the checker retrained on the push's own renders, labelled
      by the third judge, two passes. Every round is chosen on the three tuning pairs; the single
      best configuration across all rounds is judged once on the five held-out pairs, by eye,
      against the same 160 of 320.

      **Rounds 0 to 7 on the tuning pairs** (third judge, of 192 at the better of s = 3 and 10):
      base 39; wrong-pair ask 66; wrong-pair weight 47; the pair fed to every layer of the head 81;
      the last two combined 28, and at a second seed 32; strength by step at best 81 against 85
      constant (turning the push off late drops it to 19 to 28); the partly random sampler (eta 1.0)
      with the push 120 at s = 10, while the same randomness without the push gives 0 to 1. The same
      setting differs by about 4 between runs on different cards. Round 8 (features asked with each digit)
      reaches 97 at s = 10; round 9 (two passes retrained on the push's own renders) 77 then 120,
      lifting 0 & 6 from 4 to 37 of 64. The randomness that lifted round 3 is then tried on rounds
      8 and 9's checkers, still on the tuning pairs, before the single held-out read.

      **Support.** With randomness, round 8's checker reaches 132 and round 9's 156 of 192 on the
      tuning pairs, so the single configuration is round 9's checker (two passes retrained on its own
      push) with DDIM eta 1.0 at s = 10, chosen before any held-out count. On the five held-out pairs
      it gives two clear, right digits on **217 of 320 by Claude's eye** (3 & 4 62, 3 & 6 62, 4 & 7
      35, 1 & 9 33, 0 & 3 25) and 200 by the third judge, against the bar of 160; the same sampler
      without the push gives 0. Misses are mostly a wrong second digit (two 7s, two 9s) or one digit
      alone (0 & 3). The 7 & 8 controls of the original bar (random push, separate-label push,
      best-of-N) were not rerun for this configuration. `loop2/heldout-final.json`, `heldout-final.png`.

- [x] ✅ **Loop 3: is 47 of 64 on 7 & 8 one lucky checker?** The full checker retrained at seeds 1
      and 2, unchanged recipe, its push on 7 & 8 at s = 0, 1, 3, 10, 30. **Holds** if both reach at
      least 32 of 64 at some s.

      **It holds.** Retrained at seeds 1 and 2 (held-out accuracy 0.580 and similar), the push gives
      two clear, right digits on 48 and 57 of 64 at s = 10 by Claude's eye (the first judge: 28 and
      36), against 47 for the original. Jobs 90012 and 90013; `five-versions/sweep1-pair-full-seed{1,2}/`.

## Controls and improvements on the final setup, written before they run

Navigation: ⬅️ [Follow-on loops](#follow-on-loops-written-before-they-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

The final setup is round 9's checker, DDIM eta 1.0, s = 10, cap 0.1, on the share-0.01 model.

- [x] 🟡 **Loop 4: is it the push's direction, on the held-out pairs?** On the five held-out pairs,
      seeds 0 to 63, the same sampler (eta 1.0): a random push matched in size to the final push at
      every step, the separate-label push at s = 10 and 30, and best-of-N plain draws picked by the
      final checker, N matched to a guided draw's measured cost and N = 16 beside it. **Holds** if
      the random and separate-label pushes each stay at or below 30 of 320 and best-of-N at matched
      cost is at least 40 below the final push's 217 (the original bar's per-pair margins of 6 and
      8, times five pairs). Counted by the third judge; any control within 40 of the push's judge
      count of 200 is counted by eye as well.

      **Two of three lines hold; the separate-label line does not.** Third judge, of 320, seeds 0 to
      63, eta 1.0: the final push 199 at s = 10 (206 at s = 30); no push 0; the random push matched
      in size 0 and 1; best-of-2 at matched cost 1, best-of-16 13; the separate-label push 58 and 69.
      No control came within 40 of the push, so none needed eye counts. The direction composes, not
      the push's size or picking among draws; the separate-label push composes some, as it did on
      7 & 8. Job 90111 (its sheet-drawing step failed after the renders saved; counted afterwards).

- [x] ✅ **Loop 5: can the final setup be beaten?** One change per round against the final setup,
      chosen on the three tuning pairs (seeds 0 to 63, third judge): rounds 8 and 9 combined;
      more passes on the push's own pictures with more pictures per pass; "the same digit twice" as
      an explicit no; a finer search over s (6 to 20) and eta (0.8 to 1.2); and further ideas the
      loop finds, each written here before it runs. At most 8 rounds. The best configuration and the
      current one are then rendered once on the held-out pairs at **fresh seeds 64 to 127**, never
      used before, and counted by eye. **Better** if the new one beats the current one there by at
      least 20 of 320 (about 4 per pair, the run-to-run noise); otherwise the current setup stands.

      **So far, on the tuning pairs (third judge, of 192):** the grid keeps eta 1.0 at s = 10 best
      (159; s 6 to 20 and eta 0.8 to 1.2 give 143 to 157). With the plain sampler, rounds 8 and 9
      combined after one pass give 86 and a third pass on round 9 gives 79, against round 9's 120:
      the third pass renders its own training pairs clean on 366 of 444 but the tuning pairs fall,
      which reads as fitting the training pairs. Both are being compared with eta 1.0, the setting
      of the final setup. Queued: the combined round's second pass, and "the same digit twice" as a
      no (rounds 4 and 5 of 8). Rounds 6 to 8, written before they run: (6) the final push plus the separate-label
      push at s = 3 and 10, added; (7) the push averaged over round 9's checker and the combined
      round's second-pass checker; (8) round 9's second pass redone with 48 pushed pictures per
      pair instead of 12. With eta 1.0 at s = 10: rounds 8 and 9 combined after one pass 102 and after two
      149 (6 & 9 at 64 of 64); a third pass on round 9 88; "the same digit twice" 107 (1 & 2 up to
      59, 0 & 6 down to 0). Round 6, the final push plus the separate-label push at s = 10, gives
      **177** (0 & 6 59, 1 & 2 54, 6 & 9 64), the first to beat 159; round 7, two checkers
      averaged, 164; round 8, the second pass with 48 pushed pictures per pair, 136. Loop 5's best
      is round 6 (177 against the current 159); it and the current setup go once on the held-out
      pairs at seeds 64 to 127, with eta 1.0 and no push as the control.

      **Better.** On held-out seeds 64 to 127, counted by eye with the sheets' labels hidden: the
      current setup 215 of 320 (0 & 3 27, 1 & 9 37, 3 & 4 60, 3 & 6 61, 4 & 7 30); round 6, the final
      push plus the separate-label push at s = 10, **285 of 320, 89%** (44, 53, 64, 64, 60); no push
      1 (third judge). The third judge gives 196 and 275. The gain of 70 passes the 20 written for
      it, and the current setup reproduces its 217 from seeds 0 to 63. `loop5-final-seeds64/`,
      `loop2/final-seeds64.png`.

## Loop 6: the weakest held-out pairs, written before it runs

Navigation: ⬅️ [Controls and improvements](#controls-and-improvements-on-the-final-setup-written-before-they-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [x] ❌ **Can the misses seen on 0 & 3 and 1 & 9 be reduced without tuning on them?** Those two
      held-out pairs miss mostly by drawing one digit with no partner (0 & 3) or the same digit twice
      (1 & 9). The current setup is round 6 of loop 5 (pair-aware push s = 10 plus separate-label push
      s = 10, eta 1.0). Rounds are chosen on the three tuning pairs (0 & 6, 1 & 2, 6 & 9; seeds 0 to
      63; third judge), never on a held-out pair, and the judge also counts single-digit canvases so
      the missing-partner failure is visible there. Rounds, all rendering only: (1) the
      separate-label push at s = 20 and 30; (2) the pair-aware push at s = 15 and 20 with the
      separate-label push at 10; (3) the separate-label push weighted toward the digit its head reads
      as absent; (4) both pushes doubled over the first 15 of 50 steps, where the layout forms; (5)
      the best two combined; at most 6. The best round and the current setup then go once on the
      held-out pairs at **fresh seeds 128 to 191**, by eye with labels hidden. **Better** if the
      new one wins there by at least 20 of 320.

      **No round beats the current setup on the tuning pairs, so no held-out read was made.** Of 192
      (third judge): current 179 and 176 in two runs; separate-label push at 20 and 30, 168 and 159
      (one-digit canvases up from 6 to 12 and 23); pair-aware push at 15 and 20, 174 and 178; the
      separate-label push weighted toward the absent digit 172; both pushes doubled over the first
      15 steps 176; the last two combined 174. The tuning pairs sit at about 93% and show only 3 to
      6 one-digit and 1 to 4 same-digit canvases in 192, too few to tune those failures against.
      Seeds 128 to 191 stay unread. `loop6-rounds/`, `loop6-r5/`.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Nothing yet.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?** At s = 0 every guided column matches plain product byte for
      byte. The random push's norm over the pair-aware push's is 1 at every step. Best-of-N's N
      comes from measured cost on the same card.

- [ ] ⚠️ **Was the measuring tool sound?** The judge agrees with the eye calls (Claude's, under the
      user's veto) on at least 58 of 64 tiles, including fused tiles that a confident single-digit
      read would miss (a fused 7 and 8 read as a confident 8, 1.00, on the prototype).

      **Not on the model's own drawings.** It passed on 64 pasted canvases (58 of 64) but agrees with
      Claude's blind eye calls on 31 of 64 drawn canvases (the pair-aware push at s = 10 on 7 & 8),
      where it counts 30 two clear, right digits against the eye's 47: thick drawn digits read as
      "two digits on one spot" or below 0.9. The one fix (crops thickened and blurred at random in
      training) passes the pasted gate at 58 and reaches 36 of 64 on the drawn one. Under the
      brief's rule the counts fall back to eye calls; how far is the user's call.

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

## Cross-references

- The mention of **plain product** in [plan 24's Quick context](../24-mnist-digits-on-a-canvas.md#quick-context).
- The mention of **seen and held-out pairs** in [the classifier-guided outcome board, row A2b](../../report/classifier-guided-poe/board.md).
