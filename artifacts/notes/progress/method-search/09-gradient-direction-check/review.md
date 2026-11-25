# Review: does the kept checker's push point where the dropped term points? (step M13)

This file holds what the runs in [the M13 design](design.md) produced. Outputs are on the cluster under
`/datasets/mmolefe/poe_repair_min/loop-6/m13/` (CO3) and `m13-poe/` (plain PoE); the per-step summaries are on the laptop under
`outputs/loop-6/m13/`, which git ignores.

## Runs

| Run | Job | Commits | Output | Outcome |
|---|---|---|---|---|
| Smoke, CO3: one fresh trajectory | 95161 | `9b3f1eb4-15d8bdd` | `m13-smoke/` | passed in 2 minutes 50 seconds on mscluster42 |
| CO3: 192 saved trajectories (96 tuning at seeds 6001 to 6008, 96 fresh at 6009 to 6016), steps 0, 5, 10 and 15 | 95162 | `9b3f1eb4-15d8bdd` | `m13/cosines.jsonl` | done in 14 minutes 32 seconds on mscluster42, 768 states |
| Smoke, plain PoE: one trajectory drawn and its states saved | 95165 | `7d552cfc-15d8bdd` | `m13-poe-smoke/` | passed in 4 minutes 27 seconds on mscluster79 |
| Plain PoE: 48 trajectories at Z8's settings (12 tuning pairs, seeds 6001 to 6004), drawn, then the same cosines | 95166 | `7d552cfc-15d8bdd` | `m13-poe/states/`, `m13-poe/cosines/` | done in 21 minutes 30 seconds on mscluster79, 192 states |

## The verdict

**On neither sampler does the kept checker's push point along the dropped term.** The owner's rule for the identity head, that it runs only
if the gradients point the right way, is not met. Under the bar as written: CO3 points against the dropped term (a median of −0.105 at
step 0, past the −0.10 line), and plain PoE is inconclusive, negative at every step but never past −0.10. Neither reaches +0.10 at any step.

| Step | CO3: median cosine of the push with r_t (middle half) | Plain PoE: the same | Random direction, both (spread) |
|---|---|---|---|
| 0 | **−0.105** (−0.204 to −0.027) | −0.075 (−0.217 to −0.003) | 0.000 (0.004 to 0.005) |
| 5 | −0.043 (−0.150 to 0.051) | −0.089 (−0.183 to 0.032) | 0.000 |
| 10 | −0.053 (−0.106 to −0.000) | −0.042 (−0.166 to 0.045) | 0.000 |
| 15 | −0.042 (−0.092 to 0.010) | −0.025 (−0.147 to 0.036) | −0.001 |

The cosine is between the latent gradient of the sum of the kept checker's three outputs (what the push follows) and
r_t = ε(joint) − (ε(A) + ε(B) − ε(empty)); 1 is the same direction, 0 unrelated, −1 opposite. The random row is a seeded Gaussian direction per
state, whose cosine with r_t has a median near 0 and a spread of 0.003 to 0.005, as expected for 65,536 numbers.

**The clean-pair output alone points the same way, less strongly.** Its median cosine with r_t is −0.069, −0.033, −0.038 and −0.034 on CO3,
and −0.029, −0.056, −0.022 and −0.020 on plain PoE, at steps 0 to 15.

**On CO3, two-clear pictures lean further against it at step 0.** Split by the blind calls (M1's on the tuning seeds, the settled calls on
the fresh seeds; 73 two clear of 192), the median at step 0 is −0.139 on two-clear pictures and −0.088 on the rest; at steps 5 to 15 the two
groups sit within 0.04 of each other.

## What it means

**The push tends to cancel the part of the joint prediction that composition drops.** On both samplers the checker's gradient sits on the
opposite side of r_t from the joint prompt, by 5 to 25 times what a random direction gives. That is one reason the push did not lift CO3 in
M1 and why Z8's push drew no two-clear picture on plain PoE: a step along it removes some of what the joint prompt would add.

**What it cannot say.** A cosine over the whole latent mixes bodies and background, so the push could match r_t on the bodies and oppose it
elsewhere. It says nothing about a checker trained for identity, whose gradient could point anywhere. The owner's rule ties that head to
this check, so whether it still runs is the owner's call.
