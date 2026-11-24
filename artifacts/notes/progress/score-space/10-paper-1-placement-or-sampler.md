# Paper 1 on the digit toy: is the gap the sampler, or the missing placement?

Paper 1 composes two digit models that never saw each other's digits. Its records hold three facts that do not yet add up to a claim. This design is one small run that separates them. It replaces Loop 10's score-path question, which the owner closed on 2026-11-24 ([the Loop 10 verdict](09-verdict.md#why-more-tuning-will-not-lift-this-path)). Nothing here has run.

## The two questions

**1. When does fixing the sampler help?** Adding two experts' scores is the score of their product only at the final step; at every noisy step it is not ([Reduce, Reuse, Recycle, p. 4](https://arxiv.org/pdf/2302.11552#page=4)), so a sampler can lose what the product holds. On the exact toy the product draws a red circle beside a blue square 300 times of 300 while the summed-score sampler stacks them 300 times ([the toy finding](../../../../3-understanding-compositionality/report/the-exact-toy/can-a-classifier-push-fix-the-product.md)).

**2. Does the product hold two separate digits at all?** A product of experts puts its mass where both experts' data overlap, and experts trained on disjoint data fail ([RRR, p. 18](https://arxiv.org/pdf/2302.11552#page=18); [the overlap finding](../../../../3-understanding-compositionality/report/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose.md)). Paper 1's experts never saw two digits together, so their plain product should hold almost no two-digit canvases unless something places each digit.

## What the records already show

All on the five tuning pairs, 320 canvases, counted by the digit-reading judge, from [the two-model finding](../../../../2-classifier-guidance/report/classifier-guided-poe/can-two-separately-trained-digit-models-compose-while-sampling.md):

| Setting | Clean pairs of 320 | Bears on |
|---|---:|---|
| Each expert on its own fixed half, no corrector | 257 | |
| The same, with 5 Langevin corrector steps before each sampling step | 298 | question 1: with placement given, a better sampler recovers about 41 |
| The same, with 20 corrector steps | 306 | question 1 |
| Side chosen early by the experts' own vote, then held, both pushes | 310 | placement by the experts themselves |
| One shared canvas with no halves, both pushes and corrector steps, models retrained with clutter | 121 at best | question 2: without placement, sampler fixes and pushes do not reach the plan's 288 |
| The same held side chosen **at random**, everything else equal | never run | whether the experts' vote does anything |

The vote and the 1,200-of-1,280 headline it produced are on [the void list](../void-list.md), and the finding's own open list asks for the random-side run.

## The run

**Group.** Ablation runs (group 7 of `~/.claude/EXPERIMENT_CONVENTIONS.md`): each arm removes one piece of the method, the vote, the corrector or the held side, with everything else held. A piece that turns out not to matter is a finding that bounds the claim. The models, judge, pairs, sampler and corrector are plan 33's step 7 settings, unchanged, read on fresh seeds 192 to 255 so no tuning seed is reused.

| Arm | Placement | Corrector | What it isolates |
|---|---|---|---|
| A | side chosen by the experts' vote in the first 10 of 50 steps, then held (step 7) | 5 steps | the method as it stands |
| B | side chosen by a coin flip per seed, then held | 5 steps | A against B: does the vote matter? |
| C | side chosen by a coin flip, held | none | B against C: what the sampler fix adds when placement is given |
| D | no halves, one shared canvas (the 121 setting) | 5 steps | B against D: what placement adds, with the sampler held fixed |
| Paste | each expert samples its digit alone on its half, and the halves are joined | not applicable | the exact product when each digit owns a half; the simplest rival |

Each comparison changes one thing: A and B differ only in who picks the side, B and C only in the corrector, B and D only in whether a side is held.

**The bars, taken from plan 32 and plan 33's reviews, not set here.** For a difference between two arms: at least 64 of 320 is a real effect, within 32 is none, and between is inconclusive. For the method itself: 288 of 320 supports composition, 160 is promising, under 32 is no effect.

| Result | What it means for Paper 1 |
|---|---|
| A at least 64 above B | the experts' own vote places digits; Paper 1 can claim the models choose their layout |
| A within 32 of B | any held side works; the claim reduces to "hold a side", and the paper must beat paste on a task where pasting fails (Loop 6's CLEVR test, M7) |
| B at least 64 above C | the sampler loses real mass even when placement is given, as RRR predicts |
| B at least 64 above D | placement, not the sampler, is what the product lacks |
| Paste within 32 of A | on digits the method ties the simplest rival, as Loop 6's M1 already found (295 against 287) |

**Cost:** about 0.3 GPU-hours on one `bigbatch` card: five arms of 320 canvases at 50 steps, with up to 5 corrector steps each, using `2-classifier-guidance/scripts/mnist/two_experts.py` and `canvas_unet.sample(..., eps_fn=..., langevin=(5, c))` as they are. **Buys:** whether Paper 1's method rests on the experts' vote, on holding any side, or on the sampler, which decides what the paper can claim and which rival it must beat.

## What this cannot say

Digits split cleanly into two halves. CLEVR's objects do not, so a result here says what to claim and test next, and nothing about Paper 1's CLEVR target of more than 5 objects. Who runs it, and whether it becomes a board step, is the owner's call.
