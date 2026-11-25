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
| Sign check: the actual push code (PushingUNet and Pusher) on 3 fresh CO3 trajectories, steps 0 to 15 | 95253 failed on a timestep-shape bug in the check; 95319 | `48583260-15d8bdd` | `m13-sign/sign-check.json` | done in 3 minutes 46 seconds on mscluster59: the push's change to the noise prediction has a cosine of −1.0 with the gradient on all 12 states |

## The verdict

**The push moves the noise prediction along the dropped term, not against it.** The push code subtracts the scaled gradient from the noise
prediction (`co3_push.py`: the push is strength × σ_t × gradient, returned as ε − push; `poe_repair/methods/_classifier_guided.py` does the
same, ε − push). So the change the push makes to the noise prediction points opposite to the gradient this check measured. The sign check
confirms it on the actual push code: on 12 saved CO3 states, the cosine between (pushed − unpushed) and the gradient is −1.0 every time, and
its cosine with r_t is the gradient's with the sign flipped (for example +0.0407 against −0.0406 at step 0 on puma and condor, seed 6009).

The design defined "the push direction" as the gradient, which is the opposite of what the push does. Read as what the push does, every
median below changes sign. Against the bar as written (a median of at least +0.10 at some step points the right way), CO3 reaches +0.105 at
step 0 and points the right way, and plain PoE is inconclusive, positive at every step from +0.025 to +0.089 but never at +0.10.

| Step | CO3: gradient, as measured | CO3: the push's change | Plain PoE: gradient, as measured | Plain PoE: the push's change | Random direction (spread) |
|---|---:|---:|---:|---:|---|
| 0 | −0.105 | **+0.105** | −0.075 | +0.075 | 0.000 (0.004 to 0.005) |
| 5 | −0.043 | +0.043 | −0.089 | +0.089 | 0.000 |
| 10 | −0.053 | +0.053 | −0.042 | +0.042 | 0.000 |
| 15 | −0.042 | +0.042 | −0.025 | +0.025 | −0.001 |

The middle half of trajectories, for the gradient, runs from −0.204 to −0.027 on CO3 at step 0 and from −0.217 to −0.003 on plain PoE; the
push's change spans the same ranges with the sign flipped. The cosine is with r_t = ε(joint) − (ε(A) + ε(B) − ε(empty)); 1 is the same
direction, 0 unrelated, −1 opposite.

**The clean-pair output alone moves the prediction the same way, less strongly.** Read as the push's change, its median cosine with r_t is
+0.069, +0.033, +0.038 and +0.034 on CO3, and +0.029, +0.056, +0.022 and +0.020 on plain PoE, at steps 0 to 15.

**On CO3, the push follows r_t more on pictures later called two clear, at step 0.** Split by the blind calls (73 two clear of 192), the push's
median cosine at step 0 is +0.139 on two-clear pictures and +0.088 on the rest; at steps 5 to 15 the two groups sit within 0.04 of each other.

## What it means

**The push adds a little of what the joint prompt adds beyond the product.** On both samplers the push's change sits on the same side of r_t
as the joint prompt, by 5 to 25 times what a random direction gives, most at the first steps. So the push did not fail for pointing the
wrong way: in M1 it lifted CO3 by nothing and in Z8 it fixed no plain-PoE failure while moving along r_t. Whatever limits it, its direction
relative to the dropped term is not the reason.

**This reverses the reading the owner's question 111 was answered on.** Question 111 closed the checker route for Paper 2 partly because the
push pointed against r_t. By the owner's rule for the identity head (it runs only if the gradients point the right way), CO3's +0.105 at step
0 now meets the condition. What follows is the owner's call.

**What it cannot say.** A cosine over the whole latent mixes bodies and background. It says nothing about a checker trained for identity,
whose gradient could point anywhere.
