# Review: can a reader with no training see which animal CO3 drew where? (step M12)

This file holds what the runs in [the M12 design](design.md) produced. GPU outputs are on the cluster under
`/datasets/mmolefe/poe_repair_min/loop-6/m12/`; the judge's labels are on the laptop under `outputs/loop-6/m12/judge/`, which git ignores.
The owner's answer to board question 101 allowed the judge's spend and a run off the Slurm queue; the other user's job on mscluster107
ended before it was used, so the run went through Slurm.

## Runs

| Run | Job | Commits | Output | Outcome |
|---|---|---|---|---|
| Per-place score and clean estimates, 96 saved trajectories at steps 5 to 30 | 94807 (bigbatch); a biggpu copy, 94808, cancelled | `4bff38e4-15d8bdd` | `m12/perplace/` | done in 14 minutes 31 seconds on mscluster80 (bigbatch); 94808 cancelled |
| Judge's control: the 96 finished CO3 pictures from M1, against M1's blind two-clear calls | laptop, `m12_judge.py control` | `aebdb268` | `outputs/loop-6/m12/judge/judge-control.json` | passed: balanced accuracy 0.914 against the bar of 0.75 |
| Judge's read of the clean estimates, 576 pictures at steps 5 to 30 | laptop, `m12_judge.py estimates` | `aebdb268` and the guard fix after it | `outputs/loop-6/m12/judge/judge-estimates.json` | done: 1,462,496 input and 61,732 output tokens in all, inside question 15's budget as question 101 allowed |

## The per-place reader

**Inconclusive under the bar as written.** On all 96 tuning trajectories the best balanced accuracy is 0.688, at step 20, between the 0.65
floor and the 0.75 bar, calling "two clear" when the less-represented animal holds at least 0.30 of the cells. The best area under the curve
is 0.742, at step 25, under 0.75, so it is not the ranking-only outcome of question 96. It is above the 0.714 of the best trained checker, so
the reviewer's rule for stopping this reader does not apply. The numbers are in `m12/perplace/read-score.json`.

| Sampler step | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---:|---:|---:|---:|---:|---:|
| Per-place score: balanced accuracy against M1's blind calls, floor 0.65, bar 0.75 | 0.594 | 0.648 | 0.609 | 0.688 | 0.633 | 0.664 |
| Per-place score: area under the curve | 0.574 | 0.688 | 0.678 | 0.700 | 0.742 | 0.720 |
| Per-place score against Loop 9's part labels (40 own parts, 33 borrowed): balanced accuracy | 0.591 | 0.711 | 0.585 | 0.739 | 0.739 | 0.708 |
| The same: area under the curve | 0.578 | 0.709 | 0.638 | 0.741 | 0.789 | 0.780 |
| Judge on the clean estimate: balanced accuracy against M1's blind calls | 0.664 | 0.688 | 0.773 | 0.789 | 0.852 | 0.867 |

**The judge sees identity from step 15.** On the decoded one-step clean estimates it reaches 0.75 at step 15 and rises to 0.867 by step 30.
So which animal is where, and whose parts are on which body, is in CO3's state by step 15, and the reviewer's condition for stopping the
checker route (the judge never reaching 0.75, or only from step 25) is not met. At steps 5 and 10 the judge reads 0.664 and 0.688, and the
per-place score is at or below it at every step.

**Part of the score is the pair, not the picture.** Per pair, the share of two-clear pictures and the mean score at step 20 have a rank
correlation of 0.557 over the 12 pairs (`side-reports.json`). The score is the share of cells closer to the less-represented animal, so it
reads low wherever one animal is much larger than the other, however clearly both are drawn: hippopotamus and swallow, 0 of 8 two clear,
scores 0.081, and eagle and fox, 8 of 8, scores 0.394. Giraffe and tortoise is counted in the correlation and left off the table, as
on every sheet.

| Pair | Two clear of 8 (M1's blind calls) | Mean score at step 20 |
|---|---:|---:|
| eagle × fox | 8 | 0.394 |
| lighthouse × sailboat | 6 | 0.355 |
| zebra × flamingo | 4 | 0.319 |
| bobcat × quail | 3 | 0.220 |
| crow × raven | 3 | 0.382 |
| seal × gull | 2 | 0.385 |
| boar × pheasant | 0 | 0.403 |
| cheetah × cougar | 0 | 0.170 |
| duck × goose | 0 | 0.232 |
| hippopotamus × swallow | 0 | 0.081 |
| puma × condor | 0 | 0.208 |

**What follows.** By the design, an inconclusive result goes to the owner. Any change to the score, such as dividing each animal's share by
its expected size, would be a new reader with its own bar, tested on trajectories this read has not seen.

## The judge's control

**The judge reads two-clear pictures well enough for its reads to count.** It calls a finished picture two clear when it sees two bodies,
a clear first subject, a clear second subject and no hybrid, with the photograph requirement dropped. Against M1's blind calls (32 two
clear, 64 not) its balanced accuracy is 0.914, where 0.5 is chance and the bar is 0.75. It labelled all 96 with no errors, using 208,928
input and 9,472 output tokens, about 2,180 input tokens per picture. At that rate the 576 clean estimates bring the total to about 1.46
million input tokens, inside question 15's budget.
