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

**The rerun ran off to full ink as well, so the shared canvas is still not read.** Over the "floor" base (job 95329, commit `48583260-15d8bdd`,
W&B `l7601xj1`, output `m11/shared-floor/`), the shared canvas again draws 0 clean pairs of 320, and every canvas is 100% ink (pixels above
128 of 255) on all five pairs, as over the first base. The base is therefore not what drives the product to full ink. What remains in that
experiment is the corrector's five Langevin steps at step size 3 times the noise increment, the two checker pushes capped at 0.1 of the
prediction, and two experts each predicting over the whole canvas; finding which of them does it needs runs of its own, and is the owner's
or Loop 13's call. The three readable comparisons above are unaffected.

## Diagnosis of the shared canvas, after two failed runs

Loop 13 approved this diagnosis for the owner on 2026-11-25, after the shared canvas ran off to full ink over both bases. It is a diagnosis,
not a result: the plan was to switch the corrector and the pushes off, and add each back alone only if the bare product drew digits.

**The bare product does not draw digits, so the diagnosis stops at its first run and the bar is not read.** With no corrector and no pushes,
over the "floor" base (job 95421, commit `f8ad1490-15d8bdd`, output `m11/diagnosis/bare/`), the shared canvas draws 0 clean pairs of 320.
It no longer fills the canvas: the share of ink is 0.22 to 0.76 per pair (1 and 9 lowest, 2 and 8 highest), against 0.08 to 0.15 for the
held-side experiments. Every canvas holds one solid white blob on a speckled grey background, with no digit strokes, as the sheets for 1 and
9 and for 2 and 8 show. So the plain product of two experts that each predict over the whole canvas fails on its own; the corrector and the
pushes then drive that failure to full ink, which is what the first two runs showed.

**What it says about M11's question.** The comparison of a held side against a shared canvas cannot be made with these experts, because
without a held side their product draws no digits at all. That is consistent with what holding a side adds, but it is a failure of the
shared-canvas sampler, not a measured lead, so it stays not read.
