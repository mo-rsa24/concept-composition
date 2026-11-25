# Review: does the experts' own vote place the digits, or would any held side do? (step M11)

This file holds what the runs in [the M11 design](design.md) produced. Outputs are on the cluster under
`/datasets/mmolefe/poe_repair_min/loop-6/m11/`, and logs are in `/datasets/mmolefe/poe_repair_min/loop-6/logs/`.

## Runs

| Run | Job | Commits | Output | Outcome |
|---|---|---|---|---|
| Smoke: digits 4 and 7 at 8 seeds, every experiment, with the held-side control | 95225 | `084dfa7b-15d8bdd` | `m11/smoke/` | passed in 44 seconds on mscluster81; the held-side code matched the vote's own prediction with a largest difference of 0.0 |
| Full: five tuning pairs at seeds 192 to 255, five experiments | 95226 | `084dfa7b-15d8bdd` | `m11/full/` | done in 8 minutes 51 seconds on mscluster81, W&B `nu6loaud` |

## The verdict

**The experts' vote does not place the digits: a coin flip does as well. The method ties sample-and-paste.** Against the bars from Loop 10's
design (a difference of at least 64 of 320 is real, within 32 is none), on the five tuning pairs at fresh seeds 192 to 255:

| Experiment | Clean digit pairs of 320 | 1 and 9 | 3 and 6 | 4 and 7 | 2 and 8 | 0 and 5 |
|---|---:|---:|---:|---:|---:|---:|
| Vote, then held, corrector | 312 | 62 | 64 | 63 | 61 | 62 |
| Coin flip, then held, corrector | 313 | 63 | 64 | 63 | 61 | 62 |
| Coin flip, then held, no corrector | 301 | 60 | 62 | 61 | 58 | 60 |
| Shared canvas, no halves, corrector | 0 | 0 | 0 | 0 | 0 | 0 |
| Sample-and-paste | 299 | 61 | 62 | 57 | 58 | 61 |

| Comparison | Lead, of 320 | Reading | What it means for Paper 1 |
|---|---:|---|---|
| Vote over coin | −1 | none | any held side works; the claim reduces to holding a side, and Paper 1 must beat paste where pasting fails (step M10) |
| Coin over coin without the corrector | +12 | none | once a side is held, the corrector adds no real mass on digits |
| Coin over shared canvas | +313 | **not read** | the shared-canvas experiment broke, below |
| Vote over paste | +13 | none | on digits the method ties the simplest rival, as M1 found (295 against 287) |

**The shared-canvas experiment broke, so it says nothing about placement.** Every one of its 320 canvases is 99 to 100% ink (pixels above 128
of 255), against 8 to 15% for every other experiment; the sheet shows white canvases with a few dark specks. The plain product over the
empty-canvas base, with the corrector and both pushes, ran off to full ink rather than drawing two digits on one canvas. The lead of 313
therefore measures a broken sampler as much as missing placement, and the question of what holding a side adds over a working shared canvas
is not answered here. The held-side control passed (0.0), so the vote and coin experiments differ only in who picks the side.

**What the three readable comparisons say together.** On digits, the experts' own vote is not what composes; any held side does, the
corrector adds little once the side is held, and the result matches pasting each digit into its half. Paper 1's case therefore rests on a
setting where pasting cannot apply, which is step M10.

**A rerun of the shared canvas, chosen after seeing the first run.** Loop 13, deciding for the owner on 2026-11-25, approved one rerun of the
shared-canvas experiment over the project's default "floor" base (the empty canvas with a spread of 0.3), at the same pairs, seeds, pushes and
corrector, read against the same bar beside the first run's 0 of 320. It runs after step M14 (`m11_vote.sbatch shared-floor`, output
`m11/shared-floor/`), and its result is compared with the coin experiment's 313 from the first run.
