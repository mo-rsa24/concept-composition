# Does the experts' own vote place the digits, or would any held side do? (step M11)

## What this asks

Paper 1 composes two digit models that never saw each other's digits, one trained on 0 to 4 and one on 5 to 9. Its best setting lets the two
models vote on which half of the canvas each one draws in, during the first 10 of 50 sampling steps, and then holds that choice. That setting
drew 310 clean pairs of 320 on the tuning pairs. Nothing has compared the vote with a coin flip. If a coin flip does as well, Paper 1 cannot
claim the models choose their own layout, and the claim reduces to "hold a side". This step runs that comparison, and two more that say
whether the sampler or the missing placement limits the plain product. The design is Loop 10's,
[the placement-or-sampler design](../../score-space/10-paper-1-placement-or-sampler.md). The owner's answer on board question 78, decided by
Loop 13, gives it to Loop 6 as one Paper 1 step.

## Words this uses

**Expert.** One of the two digit models, a small U-Net that draws its own digits on a 32 by 64 canvas.

**Held side.** The left or right half of the canvas, given to one expert for the whole run after it is chosen. The other half is hidden from
that expert, replaced by an empty canvas noised to the current level.

**Corrector.** Five Langevin steps at the current noise level before each sampling step, at step size 3 times that step's noise increment.
On these models it raised clean pairs from 257 to 298 of 320 with the halves fixed in advance.

**Clean pair.** A canvas the digit-reading judge calls "two clear, right, apart": exactly two blobs, one per half, read as the asked digits,
each at least 0.9 sure.

## The run

**Group.** Ablation runs: each experiment removes one piece of the method, with everything else held.

**Varied axes.** How the side is chosen (the vote, a coin flip, or no side), and whether the corrector runs. Models, judge, pushes, pairs and
seeds are the same in every experiment.

| Experiment | How the side is set | Corrector | What it isolates |
|---|---|---|---|
| Vote | the experts' vote over steps 0 to 9 of 50, then held, the other half hidden | 5 steps | the method as it stands, with the corrector |
| Coin | a coin flip per seed, then held, the other half hidden | 5 steps | vote against coin: does the vote matter? |
| Coin, no corrector | the same coin flip | none | coin against this: what the corrector adds once the side is given |
| Shared canvas | no halves: the plain product of the two experts over the empty-canvas base | 5 steps | coin against this: what holding a side adds, with the sampler fixed |
| Paste | each expert samples alone, and the two inked halves are joined on the coin's sides | not used | the simplest rival |

Each comparison changes one thing. Vote and coin differ only in who picks the side. Coin and coin without the corrector differ only in the
corrector. Coin and shared canvas differ only in whether a side is held.

**What stays fixed.** The experts `e04-either` and `e59-either`, the judge `judge-v3`, and both checker pushes at strength 10, capped at 0.1 of
the prediction. The pushes come from M1's Paper 1 checkers, which saw only the experts' own samples, as Paper 1 requires. DDIM in 50 steps at
noise 1.0. The five tuning pairs 1 and 9, 3 and 6, 4 and 7, 2 and 8, 0 and 5. Seeds 192 to 255, so 320 canvases per experiment: seeds 0 to 63
tuned the method, 64 to 127 were plan 33's final test, and 128 to 191 its unseen-pair test.

**Where this differs from Loop 10's text.** Loop 10's design names "the 121 setting" for the shared canvas, which used models retrained with
clutter. Here the shared canvas uses the same two experts as every other experiment, so the comparison with the coin changes only the held
side. The vote experiment adds the corrector to plan 33's step 7 setting, which ran without it, so that vote and coin share it.

**The control.** Given the side the vote picks at one call, the held-side code must reproduce the vote's own prediction with a largest
absolute difference under 1e-5. This shows the coin experiment differs from the vote only in who picks the side.

## The bars, taken from Loop 10's design

Loop 10 took these from plans 32 and 33's reviews. For a difference between two experiments, of 320 canvases: at least 64 is a real effect,
within 32 is none, and between is inconclusive.

| Result | What it means for Paper 1 |
|---|---|
| Vote at least 64 above coin | the experts' own vote places the digits, and Paper 1 can claim the models choose their layout |
| Vote within 32 of coin | any held side works; the claim reduces to holding a side, and Paper 1 must beat paste where pasting fails (step M10) |
| Coin at least 64 above coin without the corrector | the sampler loses real mass even when the side is given, as Reduce, Reuse, Recycle predicts |
| Coin at least 64 above shared canvas | placement, not the sampler, is what the plain product lacks |
| Paste within 32 of vote | on digits the method ties the simplest rival, as M1 found (295 against 287) |

**What would surprise.** A shared canvas within 32 of the coin would mean the plain product of these experts already holds two separate digits,
against Reduce, Reuse, Recycle's account that experts trained on disjoint data put almost no mass there.

## Cost and what it buys

**Cost:** about 0.3 GPU-hours on one `bigbatch` card, Loop 10's estimate: five experiments of 320 canvases at 50 steps, four of them with 5
corrector steps per step. The smoke run replaces this with a measured time. **Buys:** whether Paper 1's method rests on the experts' vote, on
holding any side, or on the sampler, which decides what the paper can claim and which rival it must beat.

## The ladder

Each rung writes its numbers and sheets under `/datasets/mmolefe/poe_repair_min/loop-6/m11/<rung>/`. The code is
`2-classifier-guidance/scripts/method_search/m11_vote.py`, launched by `m11_vote.sbatch`.

1. **Smoke.** Digits 4 and 7 at 8 seeds, every experiment, with the control. Records seconds per experiment.
2. **Full.** The five pairs at seeds 192 to 255, logged to W&B, group `loop-6-m11-vote`.

Nothing trains, so there is no overfit or 1% rung.

## What this cannot say

Digits split cleanly into two halves. CLEVR's objects do not, so a result here says what Paper 1 may claim and which test comes next, and
nothing about composing more than two objects.
