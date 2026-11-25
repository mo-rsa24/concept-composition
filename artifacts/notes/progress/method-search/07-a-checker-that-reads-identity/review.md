# Review: can a reader with no training see which animal CO3 drew where? (step M12)

This file holds what the runs in [the M12 design](design.md) produced. M12 has no verdict yet. GPU outputs are on the cluster under
`/datasets/mmolefe/poe_repair_min/loop-6/m12/`; the judge's labels are on the laptop under `outputs/loop-6/m12/judge/`, which git ignores.
The owner's answer to board question 101 allowed the judge's spend and a run off the Slurm queue; the other user's job on mscluster107
ended before it was used, so the run goes through Slurm.

## Runs

| Run | Job | Commits | Output | Outcome |
|---|---|---|---|---|
| Per-place score and clean estimates, 96 saved trajectories at steps 5 to 30 | 94807 (bigbatch) and 94808 (biggpu, mscluster107); whichever starts first runs, the other is cancelled | `4bff38e4-15d8bdd` | `m12/perplace/` | queued |
| Judge's control: the 96 finished CO3 pictures from M1, against M1's blind two-clear calls | laptop, `m12_judge.py control` | `aebdb268` | `outputs/loop-6/m12/judge/judge-control.json` | passed: balanced accuracy 0.914 against the bar of 0.75 |

## The judge's control

**The judge reads two-clear pictures well enough for its reads to count.** It calls a finished picture two clear when it sees two bodies,
a clear first subject, a clear second subject and no hybrid, with the photograph requirement dropped. Against M1's blind calls (32 two
clear, 64 not) its balanced accuracy is 0.914, where 0.5 is chance and the bar is 0.75. It labelled all 96 with no errors, using 208,928
input and 9,472 output tokens, about 2,180 input tokens per picture. At that rate the 576 clean estimates bring the total to about 1.46
million input tokens, inside question 15's budget.
