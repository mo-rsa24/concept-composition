# Review: is the joint prompt the wrong target where it fails? (step M14)

This file holds what the runs in [the M14 design](design.md) produced. Renders are on the cluster under
`/datasets/mmolefe/poe_repair_min/loop-6/m14/`; the blind calls are on the laptop under `outputs/loop-6/m14/blind/`, which git ignores.

## Runs

| Run | Job | Commits | Output | Outcome |
|---|---|---|---|---|
| Smoke: one joint render | 95234 | `697e06b6-15d8bdd` | `m14-smoke/` | passed in 11 minutes 43 seconds on mscluster81 |
| Joint renders: 12 tuning pairs at seeds 6001 to 6016, CO3's starting noise, guidance 7.5, 50 steps | 95235 | `697e06b6-15d8bdd` | `m14/renders/` | done in 51 minutes 37 seconds on mscluster81, 192 of 192 |
| Blind calls, two reviewers | laptop | none | `blind/calls-first.csv`, `-second.csv`, `-third.csv`, `calls-final.csv` | done: the two reviewers agreed on two clear or not for 168 of 192; a third blind call settled the other 24 (7 two clear, 17 not); 34 of 192 two clear in all |

**Held before the split.** Loop 13's reviewer read the push code as changing the noise prediction by minus the checker's gradient, which
would reverse the sign of M13's cosines as a statement about the push. The split waits for the sign check on M13's states (job 95319).

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
