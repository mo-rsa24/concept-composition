# Walk map: step 7b, MNIST digits on a canvas

Mode: `--request`. Design: [route step 7b](../../../plans/route-classifier-guided-poe-from-toy-to-animals.md#7b--mnist-digits-on-a-canvas-how-good-must-the-checker-be).
Review file: [the bar](../../../plans/review/a2b-mnist-digits-on-a-canvas.md). Plan file: [plan 24](../../../plans/24-mnist-digits-on-a-canvas.md).

## Stop point

- bar: one of section 7b's four readings printed against the review file, and the compose count against checker accuracy plotted with 0.655 and 0.771 marked
- time box: sittings M1 to M3, about 6 hours of the user's time (720 min at 2×); if M3 ends without the five-version sheet, the walk stops and looks at why
- steps at the open: 10 (objectives 0 to 9, depth full)

## Sittings

| Sitting | Start (UTC) | End (UTC) |
|---|---|---|
| M1 | 2026-11-18 05:35 | |

## Position

Reading settled (works on seen pairs only). Loop 3 holds (48 and 57 of 64 by eye). Loop 1 closed without a pass; the judge ranks loop 2's rounds on the tuning pairs only. Step 10's edit written (keep the pair-aware checker, test on cat x dog first). Loops 4 and 5 done. Loop 4: random push and best-of-N hold, separate-label line misses. Loop 5: round 6 (pair-aware + separate-label push) beats the current setup on fresh held-out seeds 64 to 127, 285 against 215 of 320 by blind eye. Next: the closing pass on the user's word.

## Ticks

| Objective | Evidence |
|---|---|
| 6 (code), the checker | `scripts/mnist/checker.py` and `checker.sbatch`, commit `36baa622`; laptop smoke: both kinds train; s = 0 byte-identical to plain product for both pushes, s = 30 changes 8,113 and 6,508 pixel values over 4 seeds |
| loop 5, eta 1.0 at s = 10 | r10a 102, r10b 149, r11 88, r12 twin 107 (0 & 6 at 0); current 159; rounds 6 to 8 queued 90141 to 90145 |
| loop 6 | no round beats the current setup on the tuning pairs (179, 176 against 159 to 178); tuning pairs at 93% with too few of the target failures; held-out seeds 128 to 191 unread |
| loop 5 final read | job 90150, seeds 64 to 127, blind eye: current 215, round 6 285 (89%), no push 1 (judge); judge 196, 275 |
| loop 4 | held-out pairs, eta 1.0, third judge of 320: push 199 (s = 10), 206 (s = 30); no push 0; random 0, 1; best-of-2 1, best-of-16 13; separate-label 58, 69; random and best-of-N lines hold, separate-label line misses |
| loop 5 so far | grid: eta 1.0 s = 10 159 of 192 best; r10a (8+9 combined, pass 1) 86, r11 (third pass) 79 with eta 0; queued 90130 to 90134: eta runs on r10a and r11, combined pass 2 (r10b), twin negatives (r12) |
| loop 2 held-out read | job 90068: 217 of 320 by eye (judge 200), no-push control 0; bar 160: support |
| loop 2 choice | on the tuning pairs with eta: r8 132, r9b 156 (0 & 6 49, 1 & 2 45, 6 & 9 62), r3 120; chosen before any held-out count: r9b's checker, eta 1.0, s = 10; held-out read job 90067 with eta 1.0 and no push as its control |
| loop 2 rounds 8 and 9 | tuning-pair totals of 192 at s = 10, eta 0: r8 expert features 97 (0 & 6 27, 1 & 2 28, 6 & 9 42); r9 pass 1 77; r9 pass 2 120 (0 & 6 37, 1 & 2 28, 6 & 9 55); pushed renders judged clean on the checker's training pairs 214 then 293 of 444; randomness sweeps on r8 and r9b queued (90065, 90066) before the single held-out read |
| loop 2 rounds 6 and 7 | tuning-pair totals of 192, round 3's checker: constant s = 10 85 (81 in round 3's own run: cross-node noise about 4); best schedule 81, push off late 19 to 28; eta 1.0 at s = 10 120 (0 & 6 18, 1 & 2 45, 6 & 9 57), at s = 3 112; control, eta with no push: 0 to 1 (no randomness 3); `five-versions/loop2-r6-schedule`, `loop2-r7-eta`, `loop2-r7-eta-control` |
| loop 3 | checkers at seeds 1 and 2: 7 & 8 at s = 10, 48 and 57 of 64 by eye (judge v1 28 and 36); holds |
| loop 1 | three tries 56.7 to 57.2 of 64 against 58; try 2 rescored against Claude's second blind read: 59.2 of 64, worst gap 6; closed without a pass |
| 7 and 8, eye counts | 7 & 8 of 64: plain 1; pair-aware 28 (s = 3), 47 (s = 10), 49 (s = 10, cap 1.0); random 1; best-of-2 1; best-of-16 16; separate-label 11 (s = 10), 15 (s = 30); held-out 84 of 320 (s = 3), 82 (s = 10); sweep 1 at s = 10: data 25% 27, 5% 26, 1% 6, flip 10% 22, 20% 16; rank correlation of eye count with accuracy at steps 0-9 +0.83, 10-19 +0.60; `five-versions/eye_counts.json`, `sweep1_eye.json` |
| 7, five-version sheet (4,000-step checker, judge v1) | 7 & 8 two clear, right digits of 64: plain 0; pair-aware 1, 15, 30, 7 at s = 1, 3, 10, 30; separate-label 0, 0, 6, 8; random 0, 0, 0, 0; best-of-2 0; best-of-16 4; held-out pairs, pair-aware, best 44 of 320 at s = 3; cap fired on half the samples or more on 0, 0, 15, 40 of 50 steps at s = 1, 3, 10, 30; by Claude's eye the pair-aware s = 10 sheet has 47 of 64; guided 6.28 s and plain 3.69 s per 64 canvases, N = 2; `artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs/five-versions/` |
| 6b, checkers at 20,000 steps | held out over steps 0 to 19: full 0.5845, data 1% 0.521, 5% 0.540, flip 10% 0.585, 20% 0.554 (25% still training); no gain over 4,000 steps |
| 6, checkers at 4,000 steps | held-out balanced accuracy over steps 0 to 19: full 0.585 (0.49 at step 0 rising to 0.74 at step 19), data 1% 0.510, 5% 0.543, flip 10% 0.579; training loss noisy near 0.45 at the end |
| 5, plain product | fuses at every share: 7 & 8 two clear, right digits of 64, judge (eye): share 0, 0; 0.01, 0 (1); 0.1, 1 (4); share 0.1 often draws a wrong partner (eye 10 of 64); support check, share 0.1: expert 7 drew a 4 on 2 of 1,000, expert 4 a 3 on 2 and a 7 on 3; `artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs/plain-product/` |
| 4, the judge | 58 of 64 against Claude's blind eye calls (gate 58); 98.3% of MNIST test digits, 98.8% of test overlays caught, 0.9% of singles called overlays; against how each tile was built, judge 60 of 64, eye 61 of 64; `artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs/judge-gate/`; commit after `8e88f6fe` |
| 2, timing | laptop RTX 4070 fp32 486 ms/step (324 min per share); RTX 3090 fp32 157 ms/step (105 min), bf16 112 ms/step (75 min); 200 steps after 20 warm-up, share 0.1, ch 64, batch 128, 6,141,185 weights; jobs 89907 (bf16, mscluster87) and 89908 (fp32, mscluster86), JSON beside each model folder |
| 3, the three U-Nets launched | jobs 89909 (share 0, mscluster86), 89910 (0.01, mscluster87), 89911 (0.1, mscluster88), fp32, 40,000 steps, commit `8e88f6fe` |
| 1, the short plan | `plans/24-mnist-digits-on-a-canvas.md` written 2026-11-18; 24 was free (highest in plans/, plans/partial/, artifacts/plans/ was 23) |
| 0, the bar in a review file | `plans/review/a2b-mnist-digits-on-a-canvas.md` written 2026-11-18; accepted by the user ("defaults") |

## Decisions

| Decision | Options | Who chose | Why |
|---|---|---|---|
| a: where checkpoints live | `data/mnist-canvas/` (gitignored) · a named path | user, default | environment/ names no laptop output folder; the brief's fallback |
| b: which share the bar reads if several fuse | 0.01 · 0.1 · at the plain-product stop | user, default | lowest share with two-digit canvases; share 0 is the control |
| outputs while /datasets is down | `/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/` · laptop only | user, default | /datasets returned Input/output error at 05:40 UTC; outputs are small |
| commits for the cluster | commit and push on `exec/classifier-guidance-ladder` · laptop only | user, default ("I trust you") | `cluster.sh run` needs a committed, pushed tree; replaces the brief's "commit nothing until done" |
| who calls the sheets | Claude's draft calls with veto after · park at each sheet | user, default | judge gate reads "validated against Claude's calls" until the user checks the 64 tiles |
| orchestration | Slurm `--dependency=afterok` chain plus a `/loop` in this session · `/loop` alone | Claude (how the work runs) | the user asked whether the work survives the laptop losing power |
| precision and steps for the U-Nets | fp32 105 min · bf16 75 min, both 40,000 steps | Claude (how the work runs) | both under the 2-hour rule; fp32 keeps precision out of the question and the three run in parallel |
| bigbatch nodes | mscluster86, 87, 88 pinned | Claude | probe 07:21 UTC: 78, 84, 86 to 89 healthy RTX 3090s; 82 dead card; 49, 66, 81 no answer |
| the product's guidance scale | g = 1 read against the bar; g = 3 drawn beside it as a diagnostic only | Claude, before any plain-product result | g = 1 is the Bayes composition p(x\|a)p(x\|b)/p(x) the share-0 control argument rests on; the design names no scale |
| the judge's fused test | an eleventh class trained on overlaid single digits, plus a split for touching digits | Claude | confidence alone called a fused 7 and 8 a confident 8 |
| drawn canvases' labels for the checker | each drawn canvas from both shares labelled by the judge (two clear, right digits positive, anything else negative) | Claude | keeps the model a canvas came from off the label, so it is no shortcut |
| separate-label training data | pasted canvases only, exact presence (a fused pair counts both digits) | Claude | presence on a drawn canvas is not known exactly |
| which s the controls are read at | each control at its own best s | Claude, before any guided result | the bar names no s for the controls; their best s gives them every chance to compose, the strict reading |
| checker training length | 20,000 steps for every checker, the full sweep rerun at it; the 4,000-step runs kept as a second series | Claude, before any 20,000-step result | the full 4,000-step checker read 0.585 held out with its loss still moving, so the sweep could not reach 0.771; one knob per run holds within each series |
| which checker the bar is read on | the full-data checker with the higher held-out accuracy (4,000 or 20,000 steps), chosen on accuracy alone | Claude, before any 20,000-step compose count | keeps the choice off the compose counts |
| how counts are made once the judge failed on drawn tiles | a: Claude's eye calls for the bar (7 & 8, every column, every s; held-out pairs at the push's best s) and for sweep 1 at s = 3 and 10 · b: eye for the bar, judge v2 with a variant rule for the sweep · c: the user calls one drawn sheet first | user (a) | the brief's fallback; about 2,500 tiles |
| referee for loop 1's 88 disputed tiles | the user · Claude's second blind read | Claude, under the user's "what you decide" | the user delegated; labelled as Claude's second read |
| step 10's edit | written once loop 3 held · held | Claude, under the user's "what you decide" | both retrained checkers reached 32 or more |
| loop 2's counter after loop 1 failed | the third judge ranks rounds on the tuning pairs, eye counts the chosen round on held-out pairs | Claude | the judge's total runs about 5% low, close enough to rank |
| randomness-only control for round 7 | added before reading round 7 against anything held out | Claude | on SDXL randomness alone composed; without it the gain could not be credited to the push |
| loop 2's single configuration | round 9b's checker with eta 1.0 at s = 10 (156 of 192 on the tuning pairs) | Claude, by the written rule (best on the tuning pairs) | no held-out count existed when it was chosen |
| c: the case the four readings leave unnamed | at the five-version stop, before counts are read · a fifth reading now | user, default | not invented by the walk |

## Out of the session (runs, threads, hands)

| Id | Kind | Serves | State |
|---|---|---|---|
| 90111 to 90116 | runs: `COUNT3=1 five_versions.sbatch loop4-controls --pairs "0,3 1,9 3,4 3,6 4,7" --pair-head loop2-r9b-film-onpolicy2 --s-list 10,30 --eta 1.0`; `sweep_sampler.sbatch loop5-grid ...`; `onpolicy.sbatch` then `DRAWS=... loop2_round.sbatch r10a-expert-onpolicy1 --film --expert-feats` and `r11-film-onpolicy3 --film` | loops 4 and 5 | queued |
| 90054, 90055, 90057 to 90063 | runs: r4 `--film --wrong-ask`, r5 the same `--seed 1`; r6 `sweep_sampler.sbatch loop2-r6-schedule loop2-r3-film const:10 sched:...`; r7 `... loop2-r7-eta ... eta:...`; r8 `loop2_round.sbatch r8-film-expert --film --expert-feats`; r9 `onpolicy.sbatch` then `DRAWS=... loop2_round.sbatch r9a/r9b --film` | loop 2 | r4, r5 done; the rest from 14:33 UTC |
| 90036 to 90039 | runs: `loop2_round.sbatch r0-base`, `r1-wrong-ask --wrong-ask`, `r2-wrong-weight --wrong-weight 3`, `r3-film --film`; commit `736d4534` | loop 2 | queued 13:40 UTC |
| 90012, 90013 | loop 3 checkers at seeds 1 and 2 and their 7 & 8 renders | loop 3 | done |
| 89982 to 89985 | runs: `sweep1.sbatch` over the 4,000-step weaker checkers (89982, mscluster86) and the 20,000-step ones (89983, mscluster87, afterany 89972); `five_versions.sbatch cap-0.3-s10 --only-s 10 --cap 0.3` and `cap-1.0-s10 ... --cap 1.0` (89984, 89985, mscluster78) | objective 8 | queued or running from 11:50 UTC; counts by judge v1, canvases saved |
| 89971, 89972, 89973 | runs: `checker.sbatch train` at `--steps 20000`: full + separate (89971, mscluster87); data 0.01/0.05/0.25 (89972, mscluster88, afterany 89966); flip 0.1/0.2 (89973, mscluster78, afterany 89967); commit `8659a427` | objectives 6 and 8 | queued 10:01 UTC |
| 89964 to 89968 | runs: share-0 plain product rerun (89916 asked for `share-0`, the trainer wrote `share-0.0`); `checker.sbatch train` full + separate (89965, mscluster87), data 0.01/0.05/0.25 (89966, mscluster88), flip 0.1/0.2 (89967, mscluster78); `five_versions.sbatch cap-0.1` (89968, mscluster86, afterok 89965) | objectives 5 to 8 | running or queued from 09:42 UTC |
| 89936 | run: `sbatch --nodelist=mscluster78 --dependency=afterok:89910:89911 2-classifier-guidance/scripts/mnist/checker.sbatch draws` | objective 6 | pending on the U-Nets |
| 89916, 89917, 89918 | run: `sbatch --nodelist=<node> --dependency=afterok:<unet job> 2-classifier-guidance/scripts/mnist/plain_product.sbatch <share> [--support]` | objective 5 | pending on the U-Nets |
| 89909, 89910, 89911 | run: `sbatch --time=03:00:00 --nodelist=<node> --job-name=mnist_unet_<share> 2-classifier-guidance/scripts/mnist/canvas_unet.sbatch <share> --steps 40000` | objective 3 | running from 07:27 UTC |

## Rules carried from the brief

Seeds 0 to 63 never trained on. One knob per sweep run. Stop before each launch with the exact command. The judge never trains on canvases and never pushes. Filing on. Commit nothing until the user says done.
