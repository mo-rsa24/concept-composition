# Walk: the MNIST stage

Walking [the MNIST stage plan](../../../plans/11-do-inclusive-captions-help-poe-compose/plans/02-the-mnist-stage.md)
with `/drip-execute-plan --request`, under a self-paced `/loop`. Branch `exec/classifier-guidance-ladder`
(shared with the classifier-guidance session's working tree, so no branch switch). Nothing is committed
until the user says so.

## Where the walk is

Tasks 1 and 2 are done; the 15 trainings run. The shared judge fails both of this step's checks, so no count is used until the user chooses how to fix it.

Task 1 is done. Task 2's smoke run measured the card's ceiling: 15 trainings as designed take about
14 hours on the laptop, against a schedule of about 3. The 6,000-step pilot drew clean digits, and the user chose
6,000 steps for all 15 on the laptop. `run_all.sh` trains and draws them one after another, dial 0 first.

## The stop point

**Bar.** The verdict table in the plan's "Success and failure outcomes", judged by the user off the
two-panel figure and `score.py`'s printout.

**Time box.** Three sittings of about 1.5 hours of the user's time; about 5 hours of GPU time (`estimate`
until the smoke run).

**Steps at the open:** 9 (tasks 0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 5.2, 6.1; instruction 7 and 8 counted as one
step each: 10 at depth full).

| Sitting | Started | Ended |
|---|---|---|
| 1 | 2026-11-18 07:15 | open |

## Ticked this session

| Step | Evidence |
|---|---|
| 1.1 the canvas maker | `canvases.py` at every dial: one digit 0.9004, 7 & 8 0.0497, other pairs 0.0499, blank 0.0996; "7" draws with a second digit 0.049, 0.105, 0.146, 0.173 against 0.051, 0.103, 0.145, 0.176; no held-out pair reached a canvas (asserted) |
| the 6,000-step pilot (dial 0.3, training seed 1; not one of the 15) | trained in about 14.5 min (144 ms a step with the canvas maker), loss 0.0174 at step 5,000; [its sheet](figures/pilot-6000-steps-dial-0.3-seed-1.png), 32 seeds a row, model outputs: "7", "8" and "4" alone draw a clean single digit on nearly every seed, with a second 7 on seeds 3, 12, 23 of "7 alone"; "7 and 8" draws a 7 beside an 8 on most seeds; the plain 7 × 8 product is mostly a fused 7-in-an-8, with separate pairs on seeds 9, 12, 23, 27. Drawing: a plain row of 256 takes 53 to 69 s with fusion, so a corrected row takes about 24 min and one model's eight rows about 54 min; 15 models about 13.5 h |
| model 1 of 15 (dial 0, training seed 1) | trained in 12.4 min, loss 0.0173; drawn in 36 min (plain rows 24 to 51 s, the corrected 7 × 8 row 875 s, the corrected 4 × 7 row 1,088 s); [all eight rows](figures/dial-0-seed-1-all-rows.png); the dial-0 stop check by eye: 0 of 64 plain 7 × 8 tiles show two separate digits, every one a single fused glyph ([the eye sheet](figures/eye-sheet-dial-0-seed-1-7x8-plain.png)), so the run carries on |
| 5.1 and 5.2 the judge's two checks (run, not ticked: both fail) | the shared judge pinned at `5bff268f`, weights sha256 `b0fb317e…`, on the CPU (`data/mnist/judge/check1.json`). Check 1, floor 475 of 500: pair calls damaged 500, one digit 482, fused 494, two digits with a wrong identity 458, two separate right digits 456; single-label calls damaged 500, wrong digit 484, right digit with a second 461, right digit alone 482. Check 2, dial 0: 55 of 64 against the blind eye calls (floor 58), all nine disagreements eye fused, judge one digit. [The failing tiles](figures/judge-check-failures.png): check 1 fails where one digit of a clean pair reads below the judge's 0.90 confidence line |
| 5.1 this step's own judge, check 1 | PASSED, floor 475 of 500: pair calls damaged 500, one digit 482, fused 494, wrong identity 492, two separate right digits 485; single-label calls damaged 500, wrong digit 484, right digit with a second 487, right digit alone 482 (`data/mnist/judge/check1.json`) |
| 5.2 this step's own judge, check 2, dial 0 | PASSED, 59 of 64 (floor 58), after the user overruled Claude's blind calls on seeds 13, 37, 41, 62 from fused to one digit (both columns kept in `eye-calls/dial-0-seed-1-7x8-plain.csv`); the five left (seeds 10, 42, 47, 48, 54) are eye fused, judge one digit. Dials 0.1, 0.3, 0.6, 1 are still owed, each when its seed-1 model lands |
| 6.1 first scoring, model 1 only (stopped: the judge misreads drawn digits) | this step's judge on dial 0, seed 1: "7 alone" right digit alone 190 of 256, "8 alone" 204, "4 alone" 133 (floors 244 at dial 0); 7 × 8 plain two separate right digits 0, corrected 0. [The tiles called damaged](figures/single-labels-called-damaged.png) are clean single digits: the judge reads 24 of the 66 damaged 7s and 79 of the 123 damaged 4s as its eleventh class, "two digits on one spot", at up to 1.00, and the rest below 0.90. Checks 1 and 2 never showed it a drawn single digit |
| 5.1 to 5.3 the judge, final form (all three checks pass) | single-label rows read each blob with the ten digit classes; product rows keep the eleventh class; the 0.90 confidence line on one-blob tiles only. Check 1: every call 482 to 500 of 500 (floor 475). Check 2, dial 0: 59 of 64 (floor 58, with the user's overrule). Check 3, new, "7 alone" at dial 0, seed 1: 64 of 64 against blind eye calls (`eye-calls/dial-0-seed-1-7-alone.csv`). Reading product rows with ten classes too was tried first: check 3 passed but check 1's fused call fell to 86 of 500 and check 2 to 22 of 64 |
| 6.1 scores, model 1 (dial 0, seed 1) | "7 alone" right digit alone 253, "8 alone" 256, "4 alone" 256 (floors 244: dial 0 counts); 7 × 8 plain two separate right digits 0, corrected 0 (no stop); "7 and 8" as one label 151 |
| models 2 to 4 | dial 0, seeds 2 and 3, and dial 0.1, seed 1, drawn after two more laptop restarts (13:43 and 15:53); scores: dial 0 "7" alone 253, 255, 253 and "8" alone 256 on all three; 7 × 8 plain and corrected two separate right digits 0 on all three; dial 0.1, seed 1: "7" alone 247, "8" alone 252, 7 × 8 plain 1, corrected 1 |
| checks 2 and 3, dial 0.1 (stopped: the judge misses separate pairs) | check 3 ("7 alone") 63 of 64, passed. Check 2 (plain product) 54 of 64, floor 58, failed: Claude's blind calls saw two separate right digits on seeds 23, 31 and 52; the judge calls 23 and 31 fused because it reads each of the two separate clean digits as "two digits on one spot" at 1.00 ([the misread tiles](figures/judge-misses-separate-pairs-dial-0.1.png)). That undercounts the verdict's own call |
| option a tried and reverted: ten classes on two-blob product tiles | check 1 still passed (fused 494 of 500), but check 2 at dial 0.1 fell to 53 of 64 and at dial 0 to 58 of 64: seeds 23 and 31 became "two digits with a wrong identity" (both blobs read 8 at 1.00) and seed 7 at dial 0 the same. At full size ([the ten disagreeing tiles](figures/check2-dial-0.1-disagreements.png)) seeds 23 and 31 each hold a clean 8 beside a 7-8 hybrid, so the judge's "fused" was right and Claude's blind calls from the sheet were wrong. Reverted; check 1 passes again, check 2 at dial 0.1 is 54 of 64: 2 wrong eye calls, 1 eye one digit against judge fused (seed 14, a fused 7-in-8 by eye at full size), 7 eye fused against judge one digit. The verdict's counted call, two separate right digits, agrees between eye (corrected at full size) and judge on all 64 |
| 5.2 check 2 graded on the verdict's call, dials 0 and 0.1 | two separate right digits, yes or no: dial 0 64 of 64 (59 on the full call), dial 0.1 64 of 64 (56 on the full call), floor 58. Claude's blind calls on dial-0.1 seeds 23 and 31 are corrected to fused in a `final_call` column beside the blind `eye_call`, from full-size tiles, after the judge ran |
| model 5 (dial 0.1, seed 2) | trained in 15.5 min, loss 0.0175, drawn in 2,011 s; "7" alone 253, "8" alone 254; 7 × 8 plain and corrected two separate right digits 0 of 256 each; 4 × 7 corrected 0 ([seeds 0 to 15 of four rows](figures/dial-0.1-seed-2-four-rows.png): plain fuses a 7 into an 8, corrected is fused or smeared) |
| model 6 (dial 0.1, seed 3) | trained in 12.7 min, loss 0.0171, drawn in 1,999 s; "7" alone 252, "8" alone 251; 7 × 8 plain and corrected two separate right digits 0 of 256 each; 4 × 7 corrected 0. Dial 0.1 complete: plain 1, 0, 0 and corrected 1, 0, 0 |
| model 7 (dial 0.3, seed 1) | trained in 15.3 min, loss 0.0174, drawn in 2,045 s; "7" alone 230, "8" alone 230 (floors 217 and 214); 7 × 8 plain two separate right digits 24 of 256, corrected 19, 4 × 7 corrected 0. Eye sheets at full size ([plain, seeds 0 to 31](figures/eye-sheet-dial-0.3-seed-1-7x8-plain-1.png), [32 to 63](figures/eye-sheet-dial-0.3-seed-1-7x8-plain-2.png)): Claude's blind calls see 11 separate pairs in the first 64, the judge 10. Check 2 59 of 64 (full call 44), passed; the judge calls three clean pairs fused (seeds 27, 49, 60) and counts two that are not (12, a 7 beside a smeared blob; 23, an 8 beside a 7-8 hybrid) ([the five tiles](figures/check2-dial-0.3-disagreements.png)). Check 3 63 of 64, passed (seed 33, a speck beside the 7) |
| model 8 (dial 0.3, seed 2) | trained in 13.0 min, loss 0.0175, drawn in 1,989 s; "7" alone 234, "8" alone 235; 7 × 8 plain 16 of 256, corrected 16; 4 × 7 corrected 0 ([seeds 0 to 15, both product rows](figures/dial-0.3-seed-2-product-rows.png): the corrected row draws two glyphs more often, one usually a 7-8 hybrid) |
| model 9 (dial 0.3, seed 3) | drawn in 2,001 s; "7" alone 235, "8" alone 241; 7 × 8 plain 12 of 256, corrected 11; 4 × 7 corrected 0. Dial 0.3 complete: plain 24, 16, 12 and corrected 19, 16, 11 (mean 15.3) |
| model 10 (dial 0.6, seed 1) (stopped: check 2 fails) | drawn in 1,974 s; judge counts "7" alone 205 (floor 207), "8" alone 203 (floor 203); 7 × 8 plain 41 of 256, corrected 70; 4 × 7 corrected 0. Full-size eye sheets: Claude's blind calls see 20 separate pairs in the first 64, the judge 10. Check 2 50 of 64 (floor 58), failed: the judge calls 12 clean pairs fused ([the 14 tiles](figures/check2-dial-0.6-disagreements.png)). Check 3 63 of 64, passed. Reading two-blob product tiles with the ten digit classes, tested without editing the judge: dial 0 64, dial 0.1 64, dial 0.3 61, dial 0.6 61 of 64 on the separate-pair call (judge separate 14 against eye 11 at dial 0.3, 23 against 20 at dial 0.6) |
| model 11 (dial 0.6, seed 2) and the corrected-row check (stopped: no judge reading passes on the corrected row) | model 11: "7" alone 216, "8" alone 203; 7 × 8 plain 39 (ten-class reading 73), corrected 48 (ten-class 153). Counts both ways for the five dial-0.3 and dial-0.6 models: corrected 19/49, 16/71, 11/39, 70/162, 48/153 (current/ten-class). Claude's blind calls on the dial-0.6, seed-1 corrected row at full size ([seeds 0 to 31](figures/eye-sheet-dial-0.6-seed-1-7x8-corrected-1.png), [32 to 63](figures/eye-sheet-dial-0.6-seed-1-7x8-corrected-2.png)): 37 of 64 separate pairs. Agreement on the separate-pair call, all five eye sheets (plain 0, 0.1, 0.3, 0.6; corrected 0.6): current 64, 64, 59, 50, 42; ten-class 64, 64, 61, 61, 54; ten-class plus the 0.90 line on every blob 64, 64, 60, 59, 55 (and check 1 fails at 184 of 200 one-digit). check1.json restored to the current judge (500 per call, passed) |
| model 12 (dial 0.6, seed 3) | "7" alone 215, "8" alone 207; 7 × 8 plain 50 (ten-class 89), corrected 39 (ten-class 134). Dial 0.6 complete: corrected 70, 48, 39 current; 162, 153, 134 ten-class; "7" alone 205, 216, 215 against floor 207 |
| model 13 (dial 1, seed 1) | "7" alone 185 (floor 199), "8" alone 183 (floor 196); 7 × 8 plain 56 (ten-class 109), corrected 89 (ten-class 204); 4 × 7 corrected 0 both ways. Both labels under their floors, so dial 1 is not a counting setting whichever judge reads the product |
| model 14 (dial 1, seed 2) | "7" alone 210 (floor 199), "8" alone 186 (floor 196); 7 × 8 plain 59 (ten-class 112), corrected 41 (ten-class 192) |
| model 15 (dial 1, seed 3) and the run's end | "7" alone 198 (floor 199), "8" alone 180 (floor 196); 7 × 8 plain 68 (ten-class 120), corrected 37 (ten-class 154). `run_all.sh` ended 01:31 on 2026-11-19; tasks 3.1 and 4.1 ticked. `score.py` with the current judge prints NULL, 15 of 15; the verdict waits on the user's judge and floor choices |
| corrected 7 × 8 by eye (option `a`) | Claude's blind full-size calls, separate right digits in the first 64 seeds (`eye-calls/dial-*-7x8-corrected.csv`, sheets `figures/eye-sheet-dial-*-7x8-corrected-*.png`): dial 0.3 14, 9, 1 (mean 8.0); dial 0.6 37, 18, 11 (mean 22.0); dial 1 35, 27, 10 (mean 24.0). Ten-class judge on all 256 beside it, as shares: dial 0.3 19%, 28%, 15% against eye 22%, 14%, 2%; it counts a filled blob beside a 7 as an 8. As the bar stands (dials 0.6 and 1 fail the purity floors), the highest counting dial is 0.3 at mean 8.0 of 64, under 9.75: null, pending the user's settling of the calls and of the dial-0.6 floor |
| 2.1 the smoke run (measured, not ticked: no width fits the schedule) | width 48, 6.46 M weights: 249 ms a step as first written; 109 ms after kernel fusion (`torch.compile`), the fused optimizer and a one-call weight average; width 64, 11.48 M weights: 167 ms. Batch 256 takes 227 ms (twice 128), so the card is the limit at about 1,150 canvases a second; batch 512 spills out of card memory (22 canvases a second). 15 trainings × 3.84 M canvases: about 14 h |

## Decisions

| Decision | Chosen | The options there were | Who chose | Why |
|---|---|---|---|---|
| show-points under the loop | show and carry on, except the stops the prompt names (dial 0 composing, the judge failing or uncommitted) | stop at every show-point | Claude, under the user's "run this in a loop" | the loop exists to keep going; the named stops still stop |
| what the network predicts | the noise | the clean picture, as the toy's latent model did | Claude | the usual choice for a pixel U-Net; the sampler is unchanged, since guidance works on predicted noise either way |
| where the plan lives | scope 11, plan 02, beside the SDXL pilot | a new scope | Claude | same question, same verdict family |
| the step's speed | kernel fusion, fused optimizer, one-call weight average; width 48 | width 64 (1.5× slower); CUDA graphs (no faster than fusion here) | Claude | a repair to how the work runs: 249 ms down to 109 ms, nothing the design claims changed |
| training length | 6,000 steps for all 15, on the laptop, each model drawn right after it trains, dial 0 first | 30,000 steps on the laptop (about 31 h); the cluster (about 2 to 3 h, needs a commit and a push) | user, `a` | the pilot drew clean single digits at 6,000 steps; the prompt's fallback cuts steps equally for all 15 |
| the judge | this step's own judge: the shared network and weights (`5bff268f`), with the 0.90 confidence line applied only to one-blob tiles | the shared judge as it is (check 1 fails: 456 of 500 clean pairs); fused only when the top class is "two on one spot" (check 1 passes, check 2 falls to 44 of 64) | user, `a`; the narrower rule by Claude after the first try | the confidence line catches drawn fused glyphs on one blob, and wrongly breaks clean pairs on two |
| check 2's dial-0 eye calls | the user overrules seeds 13, 37, 41, 62 to one digit | keep the calls (fails 55 of 64); read check 2 on the verdict's call only; a fused test for drawn glyphs | user, `overrule` | the four look like a single 8 on the sheet |
| the judge's reading | ten digit classes on single-label rows, all eleven on product rows | ten classes everywhere (check 1 fused 86 of 500, check 2 22 of 64) | user, `a`; the split by row kind by Claude, veto open | the eleventh class answers "two digits on one spot?", which only a product row asks |
| check 3 | added: blind eye calls on 64 drawn single-label tiles per check, floor 58 | checks 1 and 2 only | user, `a` | neither earlier check showed the judge a drawn single digit |
| what check 2 grades | the verdict's call only: two separate right digits, yes or no | tile-by-tile overrule as at dial 0; record dial 0.1 failed and carry on; ten classes on two-blob tiles (tried, reverted) | user, `a` | fused and one digit both count as "did not compose", and the eye and the judge split them differently on 7 of 64 tiles at dial 0.1 without changing any count the verdict reads. Changed after seeing data, so recorded here and in the diff |
| dial 0.3 eye calls | Claude's blind full-size calls accepted as written | overrule any tile | user, `accept` | checks 2 and 3 pass on them (59 and 63 of 64) |
| what reads the corrected 7 × 8 row | blind full-size eye calls on the first 64 seeds of each training at dials 0.3, 0.6 and 1, settled by the user; the bar's shares unchanged (32, 3 and 9.75 of 64, in `bar.py`); the ten-class judge count on all 256 reported beside | ten-class judge on all 256 (54 of 64 on the corrected-row check); the current judge (42 of 64, counts about half) | user, `a` | no judge reading passes on the corrected row. Changed after seeing data, so recorded here and in the diff |
| corrected 7 × 8 eye calls, dials 0.3, 0.6 and 1 | Claude's blind full-size calls on all nine trainings accepted as written: 14, 9, 1; 37, 18, 11; 35, 27, 10 of 64 | overrule any tile | user, `accept` | the borderline tiles (a 7 beside a filled 8, an 8 with a triangular top) were shown and left as called |
| ignoring the data folder | one line `data/mnist/` added to the root `.gitignore` | none | Claude | the prompt says keep it out of git; it was not ignored |

## Out of the session

| What | Id | Serves | State |
|---|---|---|---|
| the 15 trainings and their drawing | `run_all.sh`, launched 10:28, log `data/mnist/run.log` | tasks 3.1 and 4.1 | done 2026-11-19 01:31, about 15 h |

## Interruptions

- The laptop restarted at 09:08 and killed the first smoke run (exit 137).
- From 09:12 the classifier-guidance session's `2-classifier-guidance/scripts/mnist/canvas_unet.py`
  (its own MNIST U-Net, 1,000-step linear schedule, 50-step DDIM) holds about 7 GB of the card at 99%.

## Owed by the closing pass

- The plan audit (2026-11-18, 13:50): the template sections Considerations, The claim, Why this plan exists, What happens, Description, Purpose and goal, What has to pass before this runs; task 0.2 and the close-out group; numbered instruction lines; the pointers on groups 5 and 8; a paired review file `review/02-the-mnist-stage.md` to hold the ✓ lines now in the plan; the figure row and the results card; the master plan's Plans table.
- Check 2 on dials 0.1, 0.3, 0.6 and 1, and check 3 on each, as their seed-1 models land.

## How to approach this next time

**The judge.** Test a judge on the model's own outputs, not only on pasted real digits: a class trained on pasted overlays fired on clean drawn digits at up to 1.00.

**Eye calls.** Draw eye sheets at full tile size before calling a pair separate: at sheet scale a clean 8 beside a 7-8 hybrid read as two separate right digits.

**Piece 1.** Draw canvases on the GPU in one indexed paste per digit slot; the 400,000-draw count takes
10 seconds and checks every share against the design in one printout.
