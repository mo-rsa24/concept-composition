# What we tried: abstract checkers on digits

This group covers Loop 8's test of abstract checkers: small classifiers trained on digits 0 to 4 that read a property holding across concepts (two digits sitting apart, which digits are present, their left-right order), and whose gradient is added to the plain product of two digit labels while sampling 32 by 64 pixel digit canvases. It holds 17 attempts, in the order they ran, from the test bed on 2026-11-23 to the closing of the SDXL step on 2026-11-25. The board marks every step as serving Paper 2. [The 2026-11-24 reviewer pass](../2026-11-24-review.md#loop-8-test-abstract-checkers) notes that the digit canvas, two labels with no joint label, is Paper 1's shape. Two cautions hold for every entry: the 2,000 "unseen" test canvases of digits 5 to 9 were read by the learning step twice, the push step with a rerun, the removal test twice and the identity step, so they are held out from training but reused across designs; and pushed renders are not bit-reproducible across cluster nodes. The stage folders are under [the abstract-checker notes](../../abstract-checkers/README.md), the verdicts under [the abstract-checker report pages](../../../../../2-classifier-guidance/report/abstract-checkers-on-digits/), and the loop's own one-page account is [the Loop 8 summary](../../abstract-checkers/09-summary/executive-summary.md).

### A repeatable digit test bed

- **Asked:** Does the digit data rebuild byte for byte, and does a push of exactly zero leave the sampler unchanged?
- **Paper:** 2.
- **Tried:** Built pasted-MNIST canvases (20,000 training canvases on the 10 pairs among digits 0 to 4; 2,000 seen-test and 2,000 unseen-test canvases on digits 5 to 9) twice on the laptop and twice on the cluster, then ran the frozen class-conditional digit model (`share-0.0`, DDIM, 50 steps) with no push and with a zero push on all 20 pairs at 8 seeds.
- **Bar, written first:** two rebuilds print the same data hash, and the largest absolute pixel difference between the plain sampler and the zero push is under 1e-5 on all 20 pairs at 8 seeds.
- **Result:** both rebuilds gave hash `f70b0fb3...` on laptop and cluster (the 1% build gave `9b70760a...` twice). Largest pixel difference, zero push against plain: 0.0 (bar under 1e-5); plain against itself: 0.0. A rule that reads layout only and never looks at the digits scores balanced accuracy 0.873 (0-to-1 scale, 0.5 is chance) on clean unseen canvases and 0.5 on right digits against a wrong digit.
- **Verdict:** support (the review says "passed").
- **Why it worked:** verified. The canvases are built from known digits at known places, so labels are exact, and the zero push runs through the same code path as a real push.
- **Kind of failure:** none: it passed its bar.
- **Idea still alive?** alive, it passed. It also showed, with a measurement, that a plurality bar alone can be cleared by a digit-blind layout rule (0.873 balanced accuracy on clean canvases), which shaped every later bar.
- **Rules out:** Do not use a plurality bar that a digit-blind layout rule can clear: that rule's 0.873 on clean canvases is why board question 55 added a right-against-wrong-digits bar to the next step.
- **Leaves open:** the layout rule was never scored on noisy canvases as a baseline; a real (non-zero) push was later found not to be repeatable across nodes.
- **Evidence:** [the test bed review](../../abstract-checkers/04-environment/review.md), [the five kinds of canvas](../../abstract-checkers/04-environment/canvas-kinds.png), [the digit-blind reader](../../abstract-checkers/04-environment/digit-blind-reader.png), [the finding on whether the environment is repeatable](../../../../../2-classifier-guidance/report/abstract-checkers-on-digits/is-the-small-environment-repeatable.md), W&B `xv6mbklb`, jobs 93685 and 93686 (V1).

### One checker for "both asked digits, apart", read at the toy's own early noise

- **Asked:** Can one checker that reads only where each asked digit acts (two maps) tell "both asked digits present and apart" on digits 5 to 9 after training on digits 0 to 4, at the toy's sampler steps 0 to 15?
- **Paper:** 2.
- **Tried:** A map reader and a comparison raw reader (which reads the noisy canvas and is told the asked pair), 4,000 updates of 256, at timesteps 999 to 693. An inner test on the pairs among digits 0 to 3 tried four configurations (base, log totals added, signed differences, 12,000 updates); the frozen one (log totals added) retrained on digits 0 to 4 and read the 2,000 canvases of digits 5 to 9 once, at 16 steps with 4 noise draws each.
- **Bar, written first:** on digits 5 to 9, balanced accuracy at least 0.75 over steps 0 to 15, within 0.05 of digits 0 to 4, and at least 0.75 on two right digits against one wrong digit (the third bar added by the owner's answer to board question 55 before the run). Held in `v2_learn.py` as `MIN_BA`, `MAX_GAP` and `MIN_APART_WRONG`.
- **Result:** balanced accuracy on a 0-to-1 scale where 0.5 is chance. Map reader on digits 5 to 9: 0.540 (bar 0.75, missed); gap to digits 0 to 4: 0.0002 (bar 0.05, met); right against wrong digits: 0.501 (bar 0.75, missed), and 0.504 on the digits it trained on. Inner test: all four configurations read 0.531 to 0.537 on the pairs holding digit 4 and 0.495 to 0.500 on right against wrong. Overfit check: 16 of 16 fixed examples learned. The raw reader did not train: final loss 0.6931 (a constant guess), label 0 on every read.
- **Verdict:** null.
- **Why it failed:** verified that the miss is in learning which digits are present at all, before any carry-over, since right against wrong sits at chance on the trained digits too. Verified (computed from the two schedules) that the toy's steps 0 to 15 keep a signal share of 0.00004 to 0.0076, so the signal-to-noise ratio is 144 times lower than SDXL's at step 0 and 13.7 times lower at step 15. That this noise level caused the miss was inferred at the time and tested by the next three attempts. Why the raw reader failed to train is unknown (no code fault found).
- **Kind of failure:** **not understood**: the design read the checker at the toy's sampler steps 0 to 15 on the belief that step numbers match SDXL's, while the toy's signal-to-noise ratio there is 14 to 144 times lower (verified, computed from the two schedules), so the test sat at noise levels SDXL never uses. **test design**: the raw-reader control never trained (final loss 0.6931, label 0 on every read), so the comparison that would show whether the maps help had no result (verified).
- **Idea still alive?** retry worth it, and retried: the fix (read at the timesteps matched to SDXL's noise, 712 to 480, and train the count and identity questions apart) is the next four entries, where the count half passed. This null says nothing about the idea at a fair noise level.
- **Rules out:** Do not read a digit checker at the toy's sampler steps 0 to 15 (timesteps 999 to 693); use the timesteps matched to SDXL's noise. Do not read a small seen-to-unseen gap as carry-over when both scores are at chance. Do not freeze one of several configurations that all sit at chance (the rule picked 0.4999 out of 0.4946 to 0.4999).
- **Leaves open:** whether the raw reader's failure was a fault; equal signal share in pixels and in SDXL's latents is not shown to be equal difficulty (inferred in the review).
- **Evidence:** [the learning review, its result section](../../abstract-checkers/05-does-it-learn/review.md#the-result) and [inner test](../../abstract-checkers/05-does-it-learn/review.md#the-inner-test), [balanced accuracy by step](../../abstract-checkers/05-does-it-learn/balanced-accuracy-by-step.png), [the finding on whether a checker learns plurality](../../../../../2-classifier-guidance/report/abstract-checkers-on-digits/does-a-checker-learn-plurality-on-unseen-digits.md), W&B `a562r799`, jobs 93691, 93702 and 93905 (V2).

### The same checker rehearsed at SDXL's noise level

- **Asked:** Does the same map reader learn which digits are present when the noise matches SDXL's first 16 steps (board question 64)?
- **Paper:** 2.
- **Tried:** The base map reader and the raw reader trained on the pairs among digits 0 to 3 at the 16 toy timesteps 712 to 480 (signal share 0.0058 to 0.0949), scored on the pairs holding digit 4. Digits 5 to 9 were not read.
- **Bar, written first:** on the pairs holding digit 4, balanced accuracy at least 0.75, within 0.05 of the trained pairs, and at least 0.75 on right against wrong digits.
- **Result:** map reader: 0.704 over all canvas kinds (bar 0.75), gap 0.029, right against wrong 0.511 (bar 0.75), rising from 0.59 at the noisiest step to 0.81 at the least noisy. Raw reader: 0.713 and 0.514. The two readers called 82% and 74% of wrong-digit canvases "both asked digits". Overfit 16 of 16 for both. Read afterwards as a two-digits-apart counter (not planned, so it supports no claim): 0.766 and 0.776.
- **Verdict:** null (a miss by the written rule).
- **Why it failed:** verified that both readers learned where digits sit and not which digits are present, and that the maps are not the block, because the raw reader, told the asked digits, scores the same. Inferred that the mixed label let layout win: wrong-digit canvases are 1 in 8 of the training mix, and the other three label-0 kinds can be told apart by layout alone.
- **Kind of failure:** **wrong data**: wrong-digit canvases were 1 in 8 of the training mix and the other three label-0 kinds separate by layout alone, so the label mostly taught layout (inferred from the mix; the post-hoc count reading of 0.766 fits it). **the idea**, for the identity half at this window: the identity grid later read identity at the early window at only 0.563 with two maps and 0.612 with the ten-label signature, on a pool of 7,500 wrong-digit canvases (verified), so no input tried reads identity at these timesteps on this toy.
- **Idea still alive?** closed for one checker reading both where and which at the early window on this toy; the count half was alive and passed when trained apart (next entry).
- **Rules out:** Do not train one checker on a label that mixes "where" and "which" when the "which" negatives are a small share; layout answers most of the label and the checker learns only that.
- **Leaves open:** whether identity can be read at this noise level by any input (answered by the next attempt and by the identity grid below).
- **Evidence:** [the rehearsal section of the learning review](../../abstract-checkers/05-does-it-learn/review.md#the-rehearsal-at-sdxls-noise-level), [the two noise levels](../../abstract-checkers/05-does-it-learn/two-noise-levels.png), `v2/sdxl-window/rehearse.json`, no W&B run recorded, job 94080 (V2, board question 64).

### A count checker trained apart, rehearsed at SDXL's noise level

- **Asked:** Does a checker trained only on "two digits sitting apart?" (a wrong digit counts as yes) carry over to a digit it never saw, at SDXL's noise (board question 67)?
- **Paper:** 2.
- **Tried:** A map reader and a raw reader on 11,504 training canvases from the pairs among digits 0 to 3 at timesteps 712 to 480, scored on 800 test canvases including the pairs holding digit 4.
- **Bar, written first:** on the pairs holding digit 4, balanced accuracy at least 0.75 and within 0.05 of the trained pairs.
- **Result:** map reader 0.774 on the pairs holding digit 4 against 0.781 on the trained pairs (balanced accuracy, 0.5 is chance), gap 0.007; raw reader 0.775 and 0.793. Rises from 0.64 at the noisiest step to 0.91 at the least noisy. Overfit 16 of 16. By kind, 95% of overlaid canvases called correctly and 65% of single-digit canvases (its weakest kind).
- **Verdict:** support.
- **Why it worked:** verified that the count half of plurality is readable at SDXL-matched noise, as the post-hoc reading of the previous attempt had suggested. Inferred: the label depends on layout only, which the noise leaves visible.
- **Kind of failure:** none: it passed its bar.
- **Idea still alive?** alive, it passed, with the caveat that a digit-blind layout rule was never scored on these noisy canvases, so how much of 0.774 needs the maps is unknown.
- **Rules out:** Do not expect a layout-only checker's pass to say anything about identity.
- **Leaves open:** the layout-only rule that scores 0.873 on clean canvases was never scored on these noisy canvases, so how much of 0.774 a digit-blind rule would reach is unknown.
- **Evidence:** [the two-checker section of the learning review](../../abstract-checkers/05-does-it-learn/review.md#two-checkers-at-sdxls-noise-level), `v2/sdxl-split/split.json`, no W&B run recorded, job 94085 (V2, board question 67).

### An identity checker trained apart, rehearsed at SDXL's noise level

- **Asked:** Can a checker trained only on two right digits (yes) against one right and one wrong digit (no) read which digits are present on a digit it never saw, at SDXL's noise (board question 67)?
- **Paper:** 2.
- **Tried:** Map and raw readers in the same job as the count checker, on 7,004 training canvases (6,000 right-digit, 1,004 wrong-digit, drawn half and half) and 500 test canvases.
- **Bar, written first:** balanced accuracy at least 0.75 on the pairs holding digit 4 (board question 55's bar).
- **Result:** map reader 0.555 on the pairs holding digit 4 (bar 0.75) and 0.603 on the trained pairs, gap 0.048; raw reader 0.556 and 0.605. Rises from 0.48 at the noisiest step to 0.65 at the least noisy. Says no on 63% of reads. Each wrong-digit canvas was drawn about 510 times. Overfit 16 of 16.
- **Verdict:** null (a miss).
- **Why it failed:** verified that the maps are not the limit (the raw reader, told the digits, scores the same). Verified later by the identity grid below that the training pool was not the cause either (7,500 wrong-digit canvases in the same window read 0.563, only 0.008 higher) and that the window was the main cause (moving to the later window added 0.139).
- **Kind of failure:** **the idea**, at this window on this toy: identity is not readable at the timesteps matching SDXL's steps 0 to 15. Fairly tested by the identity grid afterwards: a 7.5 times larger wrong-digit pool added 0.008 and the ten-label signature added 0.049, both still under 0.75 at this window, while the later window added 0.139 (verified). Not **wrong data**: the pool size was measured not to be the cause.
- **Idea still alive?** closed at this window on this toy. Unclear for SDXL: the match between toy timesteps and SDXL steps is inferred, and on SDXL a readout read clean pairs at 0.87 by step 15 (the reviewer pass), so only a read on SDXL's own states would tell.
- **Rules out:** Do not read identity at the timesteps matching SDXL's steps 0 to 15 (712 to 480) on this toy; do not answer an identity miss there with a bigger wrong-digit pool.
- **Leaves open:** nothing further on this recipe; the right-digits checker in the removal test reused it and read 0.556 on digits 5 to 9.
- **Evidence:** [the two-checker section of the learning review](../../abstract-checkers/05-does-it-learn/review.md#two-checkers-at-sdxls-noise-level), `v2/sdxl-split/split.json`, job 94085 (V2, board question 67).

### The count checker read once on digits 5 to 9

- **Asked:** Does the count checker, trained on all 10 pairs among digits 0 to 4 at SDXL-matched noise, keep its skill on digits 5 to 9 (board question 69)?
- **Paper:** 2.
- **Tried:** Map and raw readers trained on the full training split, 4,000 updates, then read `seen_test` and `unseen_test` (2,000 canvases each) once. The saved map reader is the checker the push step uses.
- **Bar, written first:** on digits 5 to 9, balanced accuracy at least 0.75 and within 0.05 of digits 0 to 4. The right-against-wrong-digits bar from board question 55 was moved to the identity checker by board question 67, after the rehearsal above had missed it (the reviewer pass records this as a bar moved off the step).
- **Result:** map reader 0.791 on digits 5 to 9 against 0.782 on digits 0 to 4 (balanced accuracy, 0.5 is chance), gap 0.009; raw reader 0.754 and 0.787, gap 0.033. Rises from 0.63 at the noisiest step to 0.92 at the least noisy; each of the 10 unseen pairs between 0.77 and 0.81. 96% of overlaid canvases called correctly, 65% of single-digit canvases. Overfit 16 of 16.
- **Verdict:** support.
- **Why it worked:** verified on one read that the map reader loses nothing moving to unseen digits while the raw reader loses 0.033. Inferred by the reviewer: the label depends on layout only, so carry-over to new digits is expected rather than a property learned across concepts.
- **Kind of failure:** none: it passed its bar.
- **Idea still alive?** alive, it passed, though it shows less than "a property learned across concepts": its label depends on layout only, so carry-over to new digits is expected (the reviewer's inference), and the right-against-wrong-digits bar was moved off this step before the read.
- **Rules out:** Do not compare this 0.791 with the single checker's 0.540 as one change: the noise window and the task differ (a mixed comparison, flagged on the report page).
- **Leaves open:** whether maps carry over better than pixels in general (one read, no repeat); the digit-blind layout rule on noisy canvases as a baseline.
- **Evidence:** [the count section of the learning review](../../abstract-checkers/05-does-it-learn/review.md#the-count-checker-on-digits-5-to-9), [the finding on whether a checker learns plurality](../../../../../2-classifier-guidance/report/abstract-checkers-on-digits/does-a-checker-learn-plurality-on-unseen-digits.md), W&B `tyhrpyov`, job 94108 (V2, board question 69).

### The count checker's push on digits 5 to 9

- **Asked:** Does adding the count checker's gradient to the plain product while sampling make the digit model draw two separate, right digits on the 10 pairs among digits 5 to 9?
- **Paper:** 2.
- **Tried:** The model that never saw two digits on one canvas (`share-0.0`), pushed at sampler steps 15 to 25 (timesteps 693 to 489) with push length b = 0.4 of the product's norm, chosen on the seen pairs. Four rows of 320 canvases (10 pairs, 32 seeds): plain product, the push, a random push of the same length, and best of 4 starting seeds picked by the checker's logit. Scored by the digit judge.
- **Bar, written first:** the push beats the plain product by more than 1.96 standard errors (bootstrap over the 10 pairs), the random push does not, and the zero push differs from plain by under 1e-5.
- **Result:** clean pairs (two clear, right digits, one per half, judged at 0.9 confidence) of 320: plain 0, push 43, random push 1 (standard error 0.96), best of 4: 0. Lead 43 with standard error 7.58, 5.7 standard errors. Two separate digits of any identity: plain 3, push 161. Push breakdown: 43 clean, 118 two digits not both clear and right, 3 same half, 144 one blob, 12 damaged. Zero push: 0.0. Choosing b on the seen pairs: 0, 8, 20 and 40 clean of 320 at b = 0, 0.1, 0.2 and 0.4. **Records disagree, not resolved here:** 43 of 320 is the recorded run (job 94134 on mscluster79; [the push review](../../abstract-checkers/06-does-it-compose/review.md), line 3; [the report page](../../../../../2-classifier-guidance/report/abstract-checkers-on-digits/does-the-count-checkers-push-compose-unseen-digits.md), line 7; the board's V3 note). 41 of 320 is a rerun on the same pairs and seeds on another node (job 94161 on mscluster81; [the rerun section](../../abstract-checkers/06-does-it-compose/review.md#a-rerun-on-another-node); the report page's disagreement table, line 194), with per-pair counts moving by up to 4. 36 is in the brief handed to Loop 15 ([the session 15 prompt](../../prompts/session-15-directions-and-dials.md), line 102: "lifted composition by 36 pairs where a random push of the same size changed nothing", citing the report index row on the digit method against paste), and Loop 15 flagged it against 43 on [its limits page](../../directions/04-limits-and-what-to-test.md), line 199. The same 36-pair wording also appears in [the method-search review](../../method-search/01-beat-the-baseline/review.md), line 80.
- **Verdict:** support.
- **Why it worked:** verified that the gain comes from the gradient during sampling, since picking the best of 4 plain samples by the checker's own score finds 0. Verified from the rerun's trace: during the push window the two maps' centres move apart from 1.6 to 6.6 pixels on average (plain stays within 1.6), their overlap falls from 0.62 to 0.55, and the checker's mean logit rises to 6.69 where plain falls to -4.14. Verified from the judge's breakdown that the push fixes layout and leaves identity to the two digit labels: about a quarter of the canvases it splits carry both asked digits.
- **Kind of failure:** none: it passed its bar.
- **Idea still alive?** alive, it passed on digits. The record cannot say what it does on SDXL, where CO3 already draws two bodies in 96 of 96 tuning pictures, so a count push has little left to fix there.
- **Rules out:** Do not expect a count push to supply identity. Do not use the checker for selection without the gradient. Do not quote a per-pair count as exact: pushed renders differ across nodes by more than the bootstrap error shows.
- **Leaves open:** the push length was not bracketed (0.4 was the largest of three tried); the reviewer flags row 5-6, third tile, framed clean but possibly two 6s at sheet scale (inferred, low confidence, unchecked at full size); the model that saw two digits in 1% of its training (Test 2) was not run because Test 1 passed; faint correct digits count as not clean.
- **Evidence:** [the push review](../../abstract-checkers/06-does-it-compose/review.md), [plain sheet](../../abstract-checkers/06-does-it-compose/unseen-plain.png) and [pushed sheet](../../abstract-checkers/06-does-it-compose/unseen-push.png), [the evidence figure](../../abstract-checkers/09-summary/evidence-at-a-glance.png), W&B `qhusod91`, jobs 94133, 94134 and 94161 (V3).

### Removing one checker at a time, first run

- **Asked:** With three checkers pushing (count, right digits, left-right order), does removing one lower only its own property?
- **Paper:** 2.
- **Tried:** Trained the right-digits and order checkers on digits 0 to 4 (balanced accuracy on digits 5 to 9: right digits 0.556, order 0.711; count 0.791). Set each push length to 0.229 from the three gradients' mean pairwise cosine of 0.009 (10,560 cosines). Ran eleven rows (plain, all three, each removed, each removed with a random push in its place, each alone) of 320 canvases on digits 5 to 9.
- **Bar, written first:** all three beat plain on fully right canvases by more than 1.96 standard errors; each removal lowers its own column by at least 6.25 percentage points and more than 1.96 standard errors, its random replacement falls by the same rule, and no other column moves by both more than 4.2 points and more than 1.96 standard errors. Inconclusive if a row's total push passes 0.6 of the product's norm on more than 5% of pushed steps, or a cell the verdict reads holds fewer than 40 canvases.
- **Result:** the all-three row passed 0.6 of the product's norm on 5.85% of pushed steps (limit 5%), so it is void; removing count left 18 canvases in the right-digits column and 16 in the order column, and removing order left 28 in its own column (minimum 40). Read without a verdict: removing count lowered count from 41.6% to 14.1% of canvases, removing order lowered order from 88.9% to 45.2%, and removing right digits lowered its column only from 40.6% to 34.2%.
- **Verdict:** inconclusive.
- **Why it failed:** verified. The three unit gradients sum to 0.408 of the product's norm on average and occasionally line up past 0.6; chaining the columns (each scored only on canvases that passed the one before) shrinks the cells below 40.
- **Kind of failure:** **test design**: too few canvases (cells of 16, 18 and 28 against a minimum of 40, after the chained columns cut the row), and a push length rule that let the all-three row pass the void limit on 5.85% of pushed steps (verified).
- **Idea still alive?** retry worth it, and retried: the fix (three times the canvases and a push length set on the seen pairs under the void limit) is the next entry.
- **Rules out:** Do not read unseen pairs before the push length is set on the seen pairs so that the all-checkers row stays under the void limit. Do not size a chained removal test from the full row count; size it from the smallest cell after chaining.
- **Leaves open:** the verdict, which the rerun answers.
- **Evidence:** [the first-run section of the removal review](../../abstract-checkers/08-remove-one-checker/review.md#the-first-run-320-canvases-per-row), `v5/final.json`, no W&B run recorded, jobs 94137, 94153, 94154 and 94172 (V5, board question 73).

### Removing one checker at a time, rerun at three times the canvases

- **Asked:** The same question, under the unchanged bars, with enough canvases and a push length that keeps every row valid (board question 73).
- **Paper:** 2.
- **Tried:** Push length 0.215, chosen on the seen pairs where the all-three row passed 0.6 on 2.2% of pushed steps; eleven rows of 960 canvases (seeds 1000 to 1095) on digits 5 to 9. No row was void (all three: 2.1%).
- **Bar, written first:** the first run's bars, unchanged.
- **Result:** all three beat plain on fully right canvases by 14.0 percentage points (standard error 2.64). Removing count: its own column falls 28.1 points (38.4% to 10.3% of canvases with two digits apart, standard error 2.68), random replacement 25.8 (standard error 2.02), and order rises 6.8 points (standard error 2.75), past the 4.2-point limit. Removing right digits: own column falls 2.6 points (standard error 3.65), random replacement 10.8 (standard error 3.59), order rises 3.0. Removing order: own column falls 40.4 points (88.2% to 47.8% in the asked order, standard error 5.26), random replacement 45.9 (standard error 4.57), right digits falls 6.9 points (standard error 3.46).
- **Verdict:** null (by the written rule, label settled by the owner on board question 75).
- **Why it failed:** verified that the right-digits checker, reading near chance (0.556), controls its property no better than a random push. Inferred by the reviewer: the two spillovers of about 7 points come partly through the chained scoring, since removing count shrinks the canvases reaching the order column from 134 to 38. The null label was in the code before the run (`v5_remove.py`, commit `9242d9e9`), while the design's wording of null was widened afterwards to fit it (board question 75).
- **Kind of failure:** **test design**: the columns were chained, so removing count cut the canvases reaching the order column from 134 to 38 and the about-7-point spillovers can arise through that selection rather than through control (the reviewer's inference, not measured). **done wrong**: the right-digits checker read near chance (0.556 balanced accuracy on digits 5 to 9, verified), so its row could not show control of its property whatever the idea's truth.
- **Idea still alive?** unclear. Count and order each lowered their own property by 28.1 and 40.4 percentage points, with random replacements falling as far, which looks like control; the null rests on spillovers the chained scoring may have produced. What would tell them apart: the unchained rescore of the saved rows (no GPU), and a right-digits checker that reads its property on the sampler's states.
- **Rules out:** Do not score properties chained when the claim is "only its own property moves": score each property on every canvas, registered before the run. Do not put a checker that reads near chance into a removal test and read its row as evidence about control.
- **Leaves open:** the unchained rescore of the saved rows, which needs no GPU, was never done; binding, position and size do not exist on the digit canvas; the SDXL version was never run.
- **Evidence:** [the rerun section of the removal review](../../abstract-checkers/08-remove-one-checker/review.md#the-rerun-960-canvases-per-row), [the removal test diagram](../../abstract-checkers/remove-one-checker.png), [the finding on whether removing one checker removes its property](../../../../../2-classifier-guidance/report/abstract-checkers-on-digits/does-removing-one-checker-remove-its-property.md), W&B `j4anxfgy`, jobs 94185 and 94186 (V5, board questions 73 and 75).

### Reading which digits are present: input against noise level

- **Asked:** Is the identity miss caused by the input (two maps against a ten-label signature of all eleven noise predictions) or by the noise level (early window, timesteps 712 to 480, against later window, 466 to 263)?
- **Paper:** 2.
- **Tried:** A 2 by 2 grid of identity checkers trained on 15,000 pasted canvases (7,500 wrong-digit) from the 6 pairs among digits 0 to 3, batch 128, scored on 500 canvases of the pairs holding digit 4 and 750 of the trained pairs.
- **Bar, written first:** balanced accuracy at least 0.75 on the pairs holding digit 4 (`MIN_BA` in `identity_grid.py`).
- **Result:** balanced accuracy on the pairs holding digit 4 (0.5 is chance): two maps, early 0.563; two maps, later 0.702 (trained pairs 0.827); signature, early 0.612; signature, later 0.888 (trained pairs 0.892, gap 0.004). All four cells learned 16 of 16 fixed examples. The winning cell reads 0.80 at timestep 466, 0.88 by 409 and 0.92 at 263.
- **Verdict:** support.
- **Why it worked:** verified. The later window adds 0.139 with two maps, the signature adds 0.049 in the early window, and together they add 0.325; the bigger wrong-digit pool adds only 0.008 over the earlier 0.555.
- **Kind of failure:** none: it passed its bar.
- **Idea still alive?** alive, it passed on pasted canvases; the chosen cell changed input and window together, and each cell trained once.
- **Rules out:** Do not look for digit identity at the timesteps matching SDXL's steps 0 to 15; do not attribute the identity miss to pool size.
- **Leaves open:** the chosen cell changes input and window together relative to the earlier checker (the grid itself is a clean 2 by 2); each cell trained once with one seed; on SDXL the signature has no fixed list of other labels; job 94337 ran out of memory and was rerun as 94352 and 94353.
- **Evidence:** [the grid section of the identity review](../../abstract-checkers/10-reading-identity/review.md#the-grid), [the identity grid](../../abstract-checkers/10-reading-identity/identity-grid.png), no W&B run recorded, jobs 94335, 94336, 94352 and 94353 (V6).

### The identity checker read on digits 5 to 9, with robustness checks

- **Asked:** Does the winning cell (signature, later window), retrained on digits 0 to 4, read which digits are present on digits 5 to 9, and does that hold up?
- **Paper:** 2.
- **Tried:** Retrained on the 10 pairs among digits 0 to 4, read 1,250 canvases of digits 5 to 9 once (1,000 two-right-digit, 250 one-wrong-digit, 16 timesteps, 4 noise draws). Then a second training seed, a training run with labels shuffled, and per-pair scores with bootstrap standard errors.
- **Bar, written first:** balanced accuracy at least 0.75 on digits 5 to 9. Robustness bars (`MAX_SHUFFLED_BA`, `MIN_PAIR_LOWER`): second seed at least 0.75 on the pairs holding digit 4; shuffled labels at most 0.55; mean over pairs minus 1.96 standard errors at least 0.75. The reviewer infers from commit times that the robustness bars were committed after the 0.888 grid result and before the digits 5 to 9 result.
- **Result:** balanced accuracy 0.861 on digits 5 to 9 and 0.919 on digits 0 to 4 (gap 0.058, no limit set; the count checker's limit was 0.05). Every pair clears 0.75: weakest 6 and 9 at 0.824, strongest 7 and 9 at 0.922; mean 0.861, standard error 0.008, lower edge 0.844. Second seed 0.877 on the pairs holding digit 4. Shuffled labels 0.500 on every pair, calling "the asked pair" on 56.25% of reads (9 of 16 timesteps on every canvas, so its answer depends on the timestep alone).
- **Verdict:** support.
- **Why it worked:** verified that the reading is not one pair or one lucky seed, and that the pipeline does not hand the label to the checker.
- **Kind of failure:** none: it passed its bar.
- **Idea still alive?** alive, it passed on pasted canvases at 0.861. Whether it holds on the sampler's own states is the entry after next, where it did not.
- **Rules out:** Do not quote 0.888 as the identity checker's score: it is the best of four cells picked on the pairs holding digit 4, and 0.861 is the fair number.
- **Leaves open:** these are pasted canvases with noise added, not the sampler's own states (tested two entries below); digits 5 to 9 were read by earlier steps, so they are unread in training only.
- **Evidence:** [the digits 5 to 9 section](../../abstract-checkers/10-reading-identity/review.md#digits-5-to-9) and [the robustness section](../../abstract-checkers/10-reading-identity/review.md#the-robustness-checks), [the robustness ladder](../../abstract-checkers/10-reading-identity/robustness-ladder.png), W&B `cnvjjrh2`, jobs 94383, 94471 and 94472 (V6).

### The identity checker against a rule that needs no training

- **Asked:** Does the trained checker know more than the frozen digit model already does with its own ten noise predictions (the reviewer's fix 2, board question 97)?
- **Paper:** 2.
- **Tried:** A training-free rule calls a canvas "the asked pair" when both asked digits' noise predictions fit the added noise better than the best of the other eight, read on the same 1,250 canvases of digits 5 to 9 at the same timesteps and draws, per read and summed over a canvas's 64 reads.
- **Bar, written first:** the checker beats the per-read rule by more than 1.96 standard errors of the paired difference over the 10 pairs (`MIN_LEAD_SE` in `v6_baselines.py`).
- **Result:** checker 0.861; rule 0.539 per read and 0.603 summed (balanced accuracy, 0.5 is chance); lead 0.322 with standard error 0.009. The rule reads 0.52 to 0.58 on every pair.
- **Verdict:** support (the checker adds to what the model knows).
- **Why it worked:** verified. The frozen model alone barely reads identity on these canvases, so the checker's 0.861 is mostly its own.
- **Kind of failure:** none: it passed its bar.
- **Idea still alive?** alive, it passed: the checker's reading is its own and not the frozen model's (lead 0.322 balanced accuracy, standard error 0.009).
- **Rules out:** Do not present a signature checker's score without this training-free baseline beside it.
- **Leaves open:** nothing on pasted canvases.
- **Evidence:** [the two checks before V7](../../abstract-checkers/10-reading-identity/review.md#two-checks-before-v7), `grid-files/untrained-untrained.json`, run on mscluster107 card 1 under board question 90, no W&B run recorded (V6).

### The identity checker read on the sampler's own states

- **Asked:** Does the identity checker, trained on pasted canvases with noise added, read which digits are present on the states the sampler actually holds at sampler steps 27 to 36, where an identity push would act?
- **Paper:** 2.
- **Tried:** Rendered the count push's row on digits 5 to 9 at seeds 1000 to 1095 (960 canvases), recorded each state at steps 27 to 36, labelled each canvas by the judge's call on its finished picture (123 clean, 49 with two confident digits that are not the asked pair; the rest left out), and read the states with the saved checker.
- **Bar, written first:** balanced accuracy over the 10 steps at least 0.75 (`SAMPLER_MIN_BA`); inconclusive under 40 canvases in either label.
- **Result:** 0.651 over the 10 steps (balanced accuracy, 0.5 is chance): 0.67 at step 27, a peak of 0.737 at step 29, 0.60 to 0.63 by steps 33 to 36. The method review adds the area under the curve (how well it ranks right above wrong, 0.5 is chance): 0.72 at step 27 rising to 0.83 at step 36, and 0.781 with the logit averaged over the 10 steps. It calls 0.35 of wrong-digit canvases "the asked pair" at step 27 and 0.73 to 0.78 from step 33.
- **Verdict:** null (does not read sampler states).
- **Why it failed:** verified that it still ranks the states but its cutoff at logit zero drifts: logits rise for both classes, so by step 33 three quarters of wrong-digit canvases are called right. Inferred: summed scores produce states that are not noised pictures of the product (Reduce, Reuse, Recycle), so pasted-and-noised training does not match them; and a sampler's wrong canvas can hold the same asked digit twice, which pasted training never showed (the judge's per-canvas digits were not saved). Verified later (the visibility check below) that many canvases have not yet drawn their final digits at these steps.
- **Kind of failure:** **wrong data**: trained on pasted canvases with noise added, which are not what a sampler on summed scores holds, and whose wrong canvases never hold the same asked digit twice (inferred; the judge's per-canvas digits were not saved). **could not measure it**: each state was labelled by the finished picture 10 to 20 steps later, while the settling check measured that most canvases have not settled their digits by step 36 (median step 37 plain, 38 pushed; verified), so many labels were not yet true of the state read. **test design**: the gate is balanced accuracy at a logit of zero, while a push along the normalised gradient uses only ranking, and the checker ranked at an area under the curve of 0.781 (verified; whether ranking should gate is board question 96, open).
- **Idea still alive?** unclear. A checker that ranks at 0.72 to 0.83 may still steer, and the zero-point miss may come from labels the states do not yet carry. What would tell them apart: a gate on ranking registered before a run, labels taken where digits have settled, or simply a push test in that window scored on finished pictures.
- **Rules out:** Do not validate a checker meant to push only on forward-noised pasted canvases; read it on the sampler's own states before building on it. Do not swap the gate from balanced accuracy to area under the curve after seeing the result.
- **Leaves open:** whether ranking alone (area under the curve) should gate a push, which is board question 96, still open; labels come from the finished picture 10 to 20 steps after the read.
- **Evidence:** [the two checks before V7](../../abstract-checkers/10-reading-identity/review.md#two-checks-before-v7), [the method review](../../abstract-checkers/10-reading-identity/method-review.md), `grid-files/sampler-sampler.json`, the manager's [table of things that may be wrong](../../manager/status.md#things-that-may-be-wrong) (Loop 8 rows on sampler states, and the Loops 6 and 8 row on ranking against the cutoff), no W&B run recorded (V6, board question 94).

### The identity checker retrained on the sampler's own states

- **Asked:** Does retraining the identity checker on sampler states from digits 0 to 4 let it read fresh digits 5 to 9 on sampler states (board question 94, option A)?
- **Paper:** 2.
- **Tried:** 9,600 pushed renders of digits 0 to 4 (seeds 2000 to 2959) with states at steps 27 to 36, of which 1,486 clean and 550 wrong-digit canvases were labelled (2,036 in all); the signature checker trained 4,000 updates of 128 on the stored states; read once on fresh digits 5 to 9 at seeds 3000 to 3095.
- **Bar, written first:** balanced accuracy over the 10 steps at least 0.75 on fresh digits 5 to 9 (`v6_resample.py`).
- **Result:** 0.552 over the 10 steps (0.49 to 0.61 per step) on 108 clean against 50 wrong-digit canvases (balanced accuracy, 0.5 is chance). On pasted digits 5 to 9: 0.541 (0.750 right on two-right-digit canvases, 0.332 on one-wrong-digit canvases). Overfit 16 of 16; final training loss 0.0014. Area under the curve 0.53 rising to 0.64, 0.596 averaged over steps.
- **Verdict:** null.
- **Why it failed:** verified that it memorised its training set: a loss of 0.0014 with both unseen reads near chance, and it calls about 80% of every canvas "the asked pair" whatever its label. Inferred by the reviewer from the code: each of the 550 wrong-digit canvases was drawn about 465 times, with no fresh noise on stored states, into a head of about 483,000 weights.
- **Kind of failure:** **done wrong**: stored states with no fresh noise, each wrong-digit canvas drawn about 465 times, into a head of about 483,000 weights (inferred by the reviewer from the code), giving a training loss of 0.0014 and near-chance unseen reads (verified memorisation). **wrong data**: 2,036 labelled states, far fewer than the 15,000 pasted canvases that had carried over (verified count). **could not measure it**: the same labels from finished pictures on states whose digits mostly had not settled (verified by the settling check).
- **Idea still alive?** unclear. The named fix (fresh noise on one-step clean estimates at every draw, many more states, held-out seeds of digits 0 to 4 as a positive control) was never run, but the judge itself reads only 0.724 at these steps, so a better recipe may be capped by what the states show. A run with the fix, scored against the judge's 0.724 ceiling, would tell.
- **Rules out:** Do not train a checker on a few thousand stored sampler states without fresh noise on each draw. Hold out seeds of the training digits as a positive control so memorising is caught before the unseen read.
- **Leaves open:** whether it memorised examples or digits 0 to 4 as such (no held-out seeds of digits 0 to 4 were read); mixing pasted canvases and sampler states, first approved under board question 95, was never run, because the question was amended to the visibility check below.
- **Evidence:** [the retraining section](../../abstract-checkers/10-reading-identity/review.md#retraining-on-the-samplers-own-states), [the method review](../../abstract-checkers/10-reading-identity/method-review.md), [the reviewer's second pass on Loop 8](../2026-11-24-review.md#loop-8-again-at-2215-two-misses-on-reading-sampler-states), no W&B run recorded, jobs 94653, 94654 and 94655 (V6, board questions 94 and 95).

### Is identity visible on the sampler's states at all

- **Asked:** Does the digit judge itself, reading the sampler's one-step clean estimate, see which digits are present at sampler steps 27 to 36 (board question 95, amended)?
- **Paper:** 2.
- **Tried:** The judge read the clean estimate of the 2,036 saved digits 0 to 4 states at each of steps 27 to 36, as a diagnostic only, never as a checker. No training.
- **Bar, written first:** balanced accuracy over the 10 steps at least 0.75; under that, identity is not plainly present and the removal rerun closes as not run.
- **Result:** 0.724 over the 10 steps (balanced accuracy, 0.5 is chance), from 0.655 at step 27 to 0.780 at step 36. Share of clean canvases called the asked pair: 0.55 at step 27 rising to 0.84 at step 36. Share of wrong-digit canvases called the asked pair: 0.16 to 0.30.
- **Verdict:** null.
- **Why it failed:** verified. At step 27 the judge finds the asked pair on only 55% of canvases that end clean, and the separate settling check (board questions 97 and 99; 320 canvases per row, seeds 1000 to 1031) measured that the digits are still open through this window: plain product 4% settled by step 15, 16% by step 26, 46% by step 36, median step 37; count push 0%, 15%, 45%, median step 38; 90% of canvases match their finished digits only from step 44 (plain) and 46 (push). So the two checkers' misses are partly a property of the states and not only of the checkers.
- **Kind of failure:** **the idea**, for this window on this digit sampler: the digit judge on the one-step clean estimate is a fair test of whether identity is visible, and it read 0.724 against 0.75, finding the asked pair on only 55% of canvases that end clean at step 27 (verified); the settling check confirms the digits are still open there (verified). Caveat: the clean estimate is one view of a state, so a reader using more of the state was not ruled out.
- **Idea still alive?** closed for reading identity at steps 27 to 36 of this digit sampler at the 0.75 bar. A push in that window was never tested, and the settling check says the window is not too late to change the digits.
- **Rules out:** Do not train a third identity checker for steps 27 to 36 of this digit sampler: the judge itself cannot read identity there at the bar.
- **Leaves open:** whether a push at steps 27 to 36 could still steer identity, since the window is not too late by the settling check.
- **Evidence:** [the visibility section](../../abstract-checkers/10-reading-identity/review.md#is-identity-visible-at-these-steps-at-all) and [the settling section](../../abstract-checkers/10-reading-identity/review.md#when-the-sampler-settles-which-digits-it-draws), `grid-files/clean-check.json` and `grid-files/settle.json`, no W&B run recorded, job 94717 and a shared run on mscluster107 card 1 (V6, board questions 95, 97 and 99).

### The removal test rerun with the identity checker

- **Asked:** Does removing one checker damage only its own property, once the right-digits checker actually reads which digits are present?
- **Paper:** 2.
- **Tried:** Designed only: the removal test's eleven rows of 960 canvases with the signature identity checker pushing at steps 27 to 36, and each property also scored on every canvas (unchained) as the primary reading. Never run.
- **Bar, written first:** the removal test's bars unchanged, plus the unchained columns, in [the identity removal design](../../abstract-checkers/11-remove-one-with-identity/design.md).
- **Result:** none.
- **Verdict:** closed (not run), 2026-11-25.
- **Why it failed:** it did not run. Verified reasons on the board: no checker reads which digits are present on the sampler's states at steps 27 to 36 (0.651, then 0.552, bar 0.75), the judge itself reads only 0.724 there, and since board question 86 there is no SDXL removal test for it to feed. The method review judged it worth less to the papers than its 1.5 GPU-hours (inferred).
- **Kind of failure:** none of the six applies to the idea itself, which never ran; it was closed because its prerequisite, a checker that reads identity on the sampler's states, failed, for the kinds named in the three entries above.
- **Idea still alive?** unclear. What would tell: the unchained rescore of the earlier removal rows (no GPU), then a right-digits checker that passes a read on the sampler's states in the window where it pushes.
- **Rules out:** Do not rerun a removal test with an identity checker that has not first passed a read on the sampler's own states in the window where it would push.
- **Leaves open:** whether one checker controls one property when all three read their property; the unchained rescore of the earlier removal rows.
- **Evidence:** [the settling section of the identity review](../../abstract-checkers/10-reading-identity/review.md#when-the-sampler-settles-which-digits-it-draws), [the method review](../../abstract-checkers/10-reading-identity/method-review.md#is-v7-worth-its-cost), the board's step note (V7, board questions 94 and 95).

### An identity checker pushing on CO3, on SDXL

- **Asked:** Would an identity checker in signature form, pushing on CO3's own run, beat CO3 alone by at least 5 of 96 pictures with two clear subjects and more than 1.96 standard errors?
- **Paper:** 2.
- **Tried:** Designed only. Loop 7's first form pushed the count checker on six object pairs against plain PoE and needed a lead of 3 of 48; the reviewer pass found it aimed at the wrong property and the wrong rival, since CO3 already draws two bodies in 96 of 96 tuning pictures, and the owner had it rewritten on board question 86 to the identity form against CO3. Never run.
- **Bar, written first:** a lead over CO3 alone of at least 5 of 96 two-clear pictures and more than 1.96 standard errors, with a first read of identity on CO3's states at SDXL steps 0 to 15 and 16 to 31 gated at balanced accuracy 0.75.
- **Result:** none.
- **Verdict:** closed (not run), 2026-11-25.
- **Why it failed:** it did not run. The owner closed Paper 2's checker route on board question 111 (option A), because the kept checker's push pointed against the part of the joint prompt that composition drops (median cosine -0.105 at the first step on CO3). The manager's table of things that may be wrong records that question 111's closure partly rests on M13's sign, which was found reversed and corrected, and that M14's verdict may read the wrong way round; re-reading is the owner's call (inferred risk, not re-measured). The same table records that Loop 6's training-free per-place reader was answering the same question for about 0.3 GPU-hours where this test would have spent 5 to 14, and Loop 8 was asked to hold.
- **Kind of failure:** none of the six applies to this test, which never ran. Its closure rests on another loop's direction finding (board question 111), and the manager's table records that this finding partly rests on M13's sign, found reversed and corrected, with M14's verdict possibly read the wrong way round: if so, the closure is **not understood** (inferred risk, not re-measured).
- **Idea still alive?** unclear. What would tell: settling the M13 and M14 sign reading, and a read of identity on CO3's own states at the 0.75 bar (Loop 6's training-free per-place reader was the cheaper route to that read).
- **Rules out:** Do not port the digit signature checker to SDXL as the lead identity reader: the reviewer's ranking of identity readers puts it third, after its two misses on the digit sampler's states.
- **Leaves open:** everything about SDXL; whether the gate should be balanced accuracy or area under the curve (board question 96, open); whether question 111's closure stands once the sign question is settled.
- **Evidence:** [the SDXL design](../../abstract-checkers/07-sdxl/design.md), [the reviewer's first Loop 8 section](../2026-11-24-review.md#loop-8-test-abstract-checkers) and [its ranking of identity readers](../2026-11-24-review.md#choosing-paper-2s-identity-reader-2026-11-25-0250), [the manager's table of things that may be wrong](../../manager/status.md#things-that-may-be-wrong), the board's step note (V4, board questions 86, 96 and 111).
