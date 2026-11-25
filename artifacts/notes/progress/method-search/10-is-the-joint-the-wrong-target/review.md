# Review: is the joint prompt the wrong target where it fails? (step M14)

This file holds what the runs in [the M14 design](design.md) produced. Renders are on the cluster under
`/datasets/mmolefe/poe_repair_min/loop-6/m14/`; the blind calls are on the laptop under `outputs/loop-6/m14/blind/`, which git ignores.

## Runs

| Run | Job | Commits | Output | Outcome |
|---|---|---|---|---|
| Smoke: one joint render | 95234 | `697e06b6-15d8bdd` | `m14-smoke/` | passed in 11 minutes 43 seconds on mscluster81 |
| Joint renders: 12 tuning pairs at seeds 6001 to 6016, CO3's starting noise, guidance 7.5, 50 steps | 95235 | `697e06b6-15d8bdd` | `m14/renders/` | done in 51 minutes 37 seconds on mscluster81, 192 of 192 |
| Blind calls, two reviewers | laptop | none | `blind/calls-first.csv`, `blind/calls-second.csv` | under way; neutral ids, the key kept apart |

**Held before the split.** Loop 13's reviewer read the push code as changing the noise prediction by minus the checker's gradient, which
would reverse the sign of M13's cosines as a statement about the push. The split waits for the sign check on M13's states (job 95319).
