# Walk map: a classifier's push on the exact toy

**Plan:** `2-classifier-guidance/plans/partial/19-classifier-push-on-the-exact-toy.md` (route step 7, board row A1)
**Mode:** `--request` · **Branch:** `exec/classifier-guidance-ladder`
**On:** closed. Next on the route: step 9, rung 2's four readings, in a new session.

## Stop point

- **Bar:** pair-aware push at w = 1 at least 150 of 300 crisp pairs with overlaid no higher than this
  plan's unguided run (300 of 300 on seed 7); separate-label push at most 15 of 300 at every scale in
  1, 2, 5, 10. Pair weight 0.01, seed 7, uncapped. Judged by the counts.
- **Time box:** 4 sittings, about 3 h 45 of the reader's time, 7.5 h budget.
- **Steps at the open:** 5 pieces (depth: every piece needed).

## Sittings

| Sitting | Start | End |
|---|---|---|
| 1 | 2026-11-17 | |

## Toy

Draw 0 of seed 7, followed at step 150 and to its final label. For the discriminator's maths, two
1-D Gaussians N(+1, 1) and N(−1, 1): the best D at x = 0.5 is 0.73, log-odds 2x = 1.

## Decisions

| What | Options | Chose | Who | Why |
|---|---|---|---|---|
| the cap on the toy | none, 0.1 as step 5, 0.3 | none, with 0.1, 0.3, 1 reported beside | reader, defaults | under 0.1 the true correction itself composes 4 of 300 (scratchpad `cap_probe.py`) |
| where the pair-aware bar is read | w = 1 fixed, best of sweep | w = 1, with 0.5 and 2 beside | reader, defaults | fixed before the run |
| where the separate-label bar is read | every scale 1, 2, 5, 10; scale 1 only | every scale | reader, defaults | strictest reading |
| the overlay half | as written, tightened | as written, against this walk's own scale-0 run | reader, defaults | the reader's bar |
| improvements | true correction as a third push; closed-form separate-label beside the trained one | both kept | reader, defaults | |
| which classifier the separate-label bar is read on | the trained one; the trained and the best possible | the trained one, the best possible's 34 at scale 10 reported beside it in the finding | reader (`a`) | the trained classifier is the one the route uses at every later stage |
| who types the guided step in piece 1 | reader, Claude | Claude | reader | "you do it, and show me the result" |
| scale-0 identity judged at | bit for bit, float rounding | bit for bit, inside the same loop (separate runs reported) | reader (`fix`) | one draw differed by 6.2e-8 in the port run; plain unguided run three times, a zero push and the true correction at scale 0 were all bitwise identical in a fresh process, so the gap is numpy's summation order |

## Ticks

| Step | Evidence |
|---|---|
| fix | scale-0 check made bit for bit in the same loop: 200 of 200 steps |
| 1.1 | hook checked with a patched line in a throwaway process, commit `50a1b48b` |
| 1.2 | separate-label classifier trained, accuracy within 0.012 of the closed-form best at every checked step |
| beside the bar | no clip (reader's `improve`): w = 1 172, w = 2 300, identical to the clipped run, so the clip does not cause the 128 failures; they are settled before step 90 while the push is on · inclusive captions on seed 7, no push: pair weight 0.01 0, 0.03 1, 0.1 18, 0.2 53, 0.3 115, 0.5 246 crisp of 300 · `pair-aware-no-clip.json`, `inclusive-captions-comparison.json` |
| 2.4 | Claude ran at the reader's word: w = 0.5 16, w = 1 172, w = 2 300 crisp of 300; overlaid 128 at w = 1; bar line pass, matching the key |
| 1.4 | pair-guide built, runs in 26 s; draw 0 at step 150: push against true correction cosine 0.148, median over 300 draws 0.169 |
| 2.3 | Claude typed the loss and the log-odds gradient at the reader's word and ran the test: matched the key on every row, gradient check 3.3e-4 |
| 1.3 | data built (3,500 crisp against 3,470 overlaid and 30 faint); trainer and gradient check run with patched functions |
| 2.2 | Claude ran at the reader's word: trained 0, 0, 2, 10; best possible 0, 1, 13, 34; bar line pass (highest 10) |
| 2.1 | Claude typed `eps - scale * np.sqrt(1 - a_t) * grad` and ran the test: scale 0 1 of 300 draws off by 6.2e-8, labels 300 of 300 unchanged; scale 1 vs true score 6.7e-16; crisp 0, 0, 112, 300, 300 |

## Answer keys (held until the reader's done)

| Piece | What was predicted | Key | Where |
|---|---|---|---|
| 1 | crisp pairs of 300 with the true correction at scale 0.5 | 112 (overlaid 188), shown: no prediction given | scratchpad `key-port/port.json` |
| 4 | crisp pairs of 300 with the pair-aware push at w = 1 | 172 (overlaid 128); w = 0.5: 16; w = 2: 300 | scratchpad `key-pair/pair-aware-guide.json` |
| 3 | held-out accuracy at step 150, pairs called real and sampler draws called fake | 0.956 and 0.980; gradient check 3.3e-4 | scratchpad `key-disc/pair-aware-train.json` |
| 2 | crisp pairs of 300 with the trained separate-label push at scale 10 | 10, shown: no prediction given, the reader's run matched the key on all nine rows; best possible at scale 10: 34 | scratchpad `key-separate/separate-label-guide.json` |

## Out in runs, threads and handoffs

| Id | Kind | Serves | State |
|---|---|---|---|
| stage-figure | background agent | the reader's `a`: the discriminator as a stage figure with real draws | landed: `the-side-by-side-discriminator.png`, seven stages, README entry written |
| derivation-to-journey | background agent | the reader's ask: the two code lines derived, into the classifier-guidance journey's chapter 3 | landed: section "The two lines of code that train it and take its push" at the end of 3.3, figure 03-two-penalties.png, plan 06 task 1.6; not committed (learning repo has other sessions' edits) |

## How to approach this next time

- A separate-label classifier trained on data that holds the pair with both labels favours the pair when sharpened; check the best possible classifier beside the trained one before reading a pass.
- Port guidance by converting score to noise once (ε = −σ_t · score), then check the push at scale 0 and the perfect push at scale 1 before training anything.

## Closing pass owed

- Done: the card, A1's page, the board redrawn at step 7, the retrospective, 0.1 (verify-plan and xref-pillar), the plan filed to `plans/partial/` (2.5 is its one 🕒 line).
- Written but not committed, beside another session's uncommitted edits: the board's A1 row, `history.md`'s row change, and the route's three new rows and step 7 gate tick.
- Not committed in the learning repo: chapter 3's new section, its figure and plan 06 task 1.6.
- Not dispatched: pillar-cadence and xref-cadence, because another session holds this repo's cadence queue.
