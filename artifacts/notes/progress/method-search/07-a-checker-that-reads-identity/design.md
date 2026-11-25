# Can a reader with no training see which animal CO3 drew where? (step M12)

## What this asks

Paper 2 pushes CO3 with a checker, and three trained checkers have now failed to tell CO3's good pictures from its pictures with borrowed
parts: on the 96 tuning pictures the best balanced accuracy is 0.648, under the 0.65 floor ([M5's review](../01b-balance-the-checker-data/review.md)).
By the owner's answer A on board question 100, Paper 2's checker search moves to a reader of identity, and the reviewer's pass of
2026-11-25 ([Choosing Paper 2's identity reader](../../reviewer/2026-11-24-review.md)) names the first one to try: the per-place prediction
difference, which needs no training. On one recorded seed per pair it already marked the borrowing body at steps 4 and 5
([Loop 12's free check](../../leakage/03-one-fix/free-check/README.md)).

## Words this uses

**Per-place difference.** At one sampling step, SDXL predicts the noise four times on the same latent: under the empty prompt, the joint
prompt ("a puma and a condor"), the first animal's prompt and the second's. At each of the latent's 128 by 128 cells, the first animal's
distance to the joint prediction minus the second animal's, each distance taken over the 4 channels. A cell is closer to the first animal
when that difference is negative.

**Score.** The share of cells closer to the less-represented animal: the smaller of the two animals' shares, between 0 and 0.5. A picture
where both animals hold their own cells should score near 0.5, and one where one animal has taken the other's body should score low.

**Two clear.** M1's blind call on CO3's finished picture: one clear subject A and one clear subject B as separate bodies, no borrowed parts,
drawing style ignored. 32 of the 96 tuning pictures are two clear, and 64 have two bodies that are not both clear.

**One-step clean estimate.** The picture the state would become if the sampler jumped straight to the end, from CO3's guided prediction at
its released scale of 0.8. Decoded, it is what any reader could at most see at that step.

## The run

**Group.** Runs that try ideas. The result decides whether Paper 2's checker route has a reader worth pushing on, and makes no claim in
either paper.

**Data.** CO3's saved states for the 96 tuning trajectories (12 tuning pairs at seeds 6001 to 6008), at steps 5, 10, 15, 20, 25 and 30,
under `/datasets/mmolefe/poe_repair_min/loop-6/m5/co3-read-states-shard0/` and `shard1/`. Each half was saved on the card type M1 drew it on,
and all 96 reproduce M1's pictures exactly. No CO3 run is needed.

**Code.** `2-classifier-guidance/scripts/method_search/m5_read_co3.py perplace`, launched by `m5_read.sbatch perplace`. Outputs under
`/datasets/mmolefe/poe_repair_min/loop-6/m12/perplace/`: `reads.jsonl` (the score and the first animal's share per state),
`read-score.json`, and the decoded clean estimates under `clean-estimates/step-NN/<pair>/seed-NNNN.png`.

**The judge, a diagnostic only.** The clean estimates at each step are labelled by `2-classifier-guidance/scripts/rung2/api_label.py`'s
fields with the photograph requirement dropped. Its positive control first: on the 96 finished pictures it must agree with M1's blind
two-clear calls at a balanced accuracy of at least 0.75, or none of its reads count. The judge runs on the laptop, on the owner's Anthropic
API key, so it starts only once that spend is confirmed.

## The bar, written on 2026-11-25 before the run

Taken from the reviewer's pass, as Loop 13 relayed it under question 100.

| Result | What it takes | What follows |
|---|---|---|
| The reader sees the failure | balanced accuracy of at least 0.75 at any one step from 5 to 30, calling "two clear" when the score is at least 0.30, against M1's blind calls (32 against 64) | a push on the score at steps 5 to 15, judged on Paper 2's bar |
| It does not | under 0.65 at every step, and an area under the curve never above 0.714, the best of the three trained checkers | this reader stops |
| Inconclusive | the best step between 0.65 and 0.75 | reported, and the owner decides |
| On ranking only | under 0.75 at every step on the call, but an area under the curve of at least 0.75 | returns to the owner under question 96 |

Reported beside the bar, deciding nothing: the area under the curve at each step; the same balanced accuracy against Loop 9's part labels
(33 borrowed against 40 own parts, the 23 pictures it calls "not two" left out, from `noise-text-critic/02d-binding/part-binding-calls.csv`);
and each pair's share of two-clear pictures beside its mean score, so a reader that only knows which pairs borrow is seen as that.

**The checker route goes to the owner as a stop** if this reader fails, the judge never reaches 0.75 on the clean estimates at steps 5 to 30
(or only from step 25), and Loop 12's test of each animal's own prediction inside its box is null.

**What would surprise.** A score that separates two-clear pictures at step 5, when the judge cannot yet see two bodies in the clean estimate
there. That would mean the per-place difference carries identity before the picture shows it.

## Cost and what it buys

**Cost:** about 0.3 GPU-hours, the reviewer's estimate: 96 trajectories by 6 steps, one 4-prompt SDXL pass and one decode each. No training.
The judge's API tokens are separate: about 672 pictures (6 steps of 96, and the 96 finished pictures for its control), roughly 1.5 million
input tokens at the rate recorded in the checker loop. **Buys:** whether CO3's own predictions carry which animal is where early enough to
push on, before any reader is trained for it.

## What this cannot say

The per-place difference compares each animal's prediction with the joint one, so it can say how much of the picture each animal holds, and
not which body belongs to which. On cheetah and cougar both bodies read "cheetah" in Loop 12's check. A score near 0.5 is what two clear
animals should give, and also what two half-blended ones could give.
