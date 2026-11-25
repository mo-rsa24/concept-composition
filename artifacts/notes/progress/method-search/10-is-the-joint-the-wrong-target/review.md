# Review: is the joint prompt the wrong target where it fails? (step M14)

This file holds what the runs in [the M14 design](design.md) produced. Renders are on the cluster under
`/datasets/mmolefe/poe_repair_min/loop-6/m14/`; the blind calls are on the laptop under `outputs/loop-6/m14/blind/`, which git ignores.

## Runs

| Run | Job | Commits | Output | Outcome |
|---|---|---|---|---|
| Smoke: one joint render | 95234 | `697e06b6-15d8bdd` | `m14-smoke/` | passed in 11 minutes 43 seconds on mscluster81 |
| Joint renders: 12 tuning pairs at seeds 6001 to 6016, CO3's starting noise, guidance 7.5, 50 steps | 95235 | `697e06b6-15d8bdd` | `m14/renders/` | done in 51 minutes 37 seconds on mscluster81, 192 of 192 |
| Blind calls, two reviewers | laptop | none | `blind/calls-first.csv`, `-second.csv`, `-third.csv`, `calls-final.csv` | done: the two reviewers agreed on two clear or not for 168 of 192; a third blind call settled the other 24 (7 two clear, 17 not); 34 of 192 two clear in all |

**The sign of the cosines.** The push changes the noise prediction by minus the checker's gradient, which M13's sign check confirmed
(commit 1b581f08). The bar's signs were corrected before any split (commit 9ad229a0), and the split below reads every cosine as the push's
change, minus the gradient cosine M13 saved.

## The joint prompt's own two-clear rate

**Drawn alone from CO3's starting noise, the joint prompt draws one clear subject A beside one clear subject B in 34 of 192 pictures,
18%.** It is never two clear on three pairs. These counts are read from the settled blind calls and do not depend on the sign question above.

| Pair | Two clear, of 16 |
|---|---:|
| eagle × fox | 7 |
| lighthouse × sailboat | 6 |
| boar × pheasant | 4 |
| crow × raven | 4 |
| bobcat × quail | 3 |
| seal × gull | 2 |
| zebra × flamingo | 2 |
| puma × condor | 1 |
| hippopotamus × swallow | 0 |
| duck × goose | 0 |
| cheetah × cougar | 0 |

Giraffe and tortoise, 5 of 16, is counted in the 34 and left off the table, as on every sheet. For comparison on the same 192 seeds, CO3
drew 73 two-clear pictures (M1's blind calls on seeds 6001 to 6008 and the settled calls on 6009 to 6016), so CO3 doubles the joint
prompt's rate on these pairs.

## The split

**The owner reads M14 as intended: the joint is the wrong target where it fails, at step 0 only.** With the push's true direction, the
design's condition is that the push follows r_t less where the joint fails, by at least 0.05 with the interval below 0. At step 0 the
difference is −0.051 (interval −0.129 to −0.002), which meets it by 0.001; at steps 5 to 15 every interval includes 0, so they are
inconclusive. By question 112's rule, M15 runs. The numbers are in `split.json` in the laptop's `outputs/loop-6/m14/`.

**Beside it, the reading first recorded: the checker is wrong.** The bar's signs were corrected by negating every threshold (commit
9ad229a0), which kept the rule's decisions on the gradient's data and so double-flipped the condition once the cosines were read in the
push's direction. Under that version the condition needs a difference of at least +0.05, which no step meets. The owner chose the intended
reading on 2026-11-25, after the split, through Loop 13.

| Step | Push's cosine with r_t where the joint succeeds (34) | Where the joint fails (158) | Difference, fails minus succeeds | 95% interval |
|---|---:|---:|---:|---|
| 0 | +0.148 | +0.097 | −0.051 | −0.129 to −0.002 |
| 5 | +0.051 | +0.043 | −0.008 | −0.091 to +0.083 |
| 10 | +0.047 | +0.053 | +0.006 | −0.027 to +0.042 |
| 15 | +0.055 | +0.040 | −0.015 | −0.054 to +0.015 |

The interval comes from 10,000 bootstrap resamples of trajectories within each group. The bar for the joint being the wrong target needed a
difference of at least +0.05 with its interval above 0; no step comes near it.

**The surprise the design named happened at step 0.** The push follows r_t more where the joint succeeds, by 0.051, with the interval below 0.
The clean-pair output alone shows the same at step 0 (+0.122 against +0.046, interval −0.125 to −0.019). The checker's push copies the joint
prompt's extra term most where that term draws two clear subjects, which is a weak copy of the teacher, not a correction of its failures.

**The joint render and CO3 disagree on most seeds.** On the 192 seeds, the joint prompt alone is two clear on 34 and CO3 on 73:

| | CO3 two clear | CO3 not |
|---|---:|---:|
| **Joint alone two clear** | 19 | 15 |
| **Joint alone not** | 54 | 104 |

So CO3 reaches two clear on 54 seeds where the joint prompt alone does not, and the joint prompt reaches it on 15 where CO3 does not.

**What it cannot say.** The cosine mixes bodies and background over the whole latent. Only 34 trajectories sit in the joint-succeeds group,
so its medians carry more spread than the other group's.

## The two readings of the condition


| Reading | The joint-wrong-target condition, on the push's cosine with r_t | Step 0 (−0.051, interval −0.129 to −0.002) | Steps 5 to 15 | Verdict |
|---|---|---|---|---|
| The rule as written, its signs corrected (commit 9ad229a0) | fails minus succeeds of at least +0.05, interval above 0 | not met | not met | the checker is wrong |
| The design's meaning, with the push's true direction | fails minus succeeds of −0.05 or below, interval below 0: the push follows r_t less where the joint fails | met, by 0.001 | inconclusive (intervals include 0) | the joint is the wrong target where it fails, at step 0 only |

**Why the two differ.** The design was written when the push's direction was taken to be the checker's gradient, and its condition said the
push opposes r_t mostly where the joint fails. The sign correction negated every threshold so that the rule decided the same way on the same
data as written. Read for what it was meant to test, with the push's true direction (minus the gradient), the same idea becomes "the push
follows r_t less where the joint fails", the opposite inequality. The first keeps the rule's letter; the second keeps its intent. This
choice was made after seeing the result; the owner chose the intended reading.
