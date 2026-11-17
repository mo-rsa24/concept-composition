# Walk map: a classifier's push on the exact toy

**Plan:** `2-classifier-guidance/plans/19-classifier-push-on-the-exact-toy.md` (route step 7, board row A1)
**Mode:** `--request` · **Branch:** `exec/classifier-guidance-ladder`
**On:** piece 1, the push hook: built, waiting on the reader's test 2.1

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

## Ticks

| Step | Evidence |
|---|---|

## Answer keys (held until the reader's done)

| Piece | What was predicted | Key | Where |
|---|---|---|---|
| 1 | crisp pairs of 300 with the true correction at scale 0.5 | 112 (overlaid 188) | scratchpad `key-port/port.json` |

## Out in runs, threads and handoffs

| Id | Kind | Serves | State |
|---|---|---|---|

## How to approach this next time

## Closing pass owed

All of it.
