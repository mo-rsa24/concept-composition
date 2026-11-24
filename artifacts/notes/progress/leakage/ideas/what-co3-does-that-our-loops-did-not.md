# What CO3's corrector does that our loops' pushes did not

CO3 is the one method in this project that reliably gives two bodies, and its mechanism differs from every push Loops 6 to 10 tried in five ways. Each difference is a candidate design rule for Paper 2's method, and two of them are issues for the Z5 audit. The mechanism is drawn in the [CO3 zoom ladder](../../../../../2-classifier-guidance/artifacts/scenes/co3-sampler/index.md), and every claim about the code cites `attn_utils.py` and `Co3.py` in the replications hub's `co3` checkout. Statements about what follows for Paper 2 are inferences and are marked so.

## The five differences

**1. Its signal comes from the model itself.** CO3 compares the U-Net's own predictions under the joint prompt and under each subject alone: the distance $d_k = \|\epsilon^{c_k} - \epsilon^{C}\|$ says which subject the joint prediction resembles. No checker, label or picture is involved. The pushes in Loops 6 to 9 came from a trained reader of the latent or the picture, and the kept push lowered CO3's faithful pairs from 11 to 7 of 96 (Loop 6's M1). (inferred) A fix built from the model's own per-subject predictions needs no training and no labels, which fits the owner's preference to leave the diffusion model untrained.

**2. Its weights are one number per subject for the whole canvas, and borrowing is a matter of place.** Equation 15 gives each subject a single weight, so CO3 can change how much of each subject the picture holds and cannot change where each subject's features go. A cougar carrying a cheetah's spots is a local failure. (inferred) Computing the same distance per cell, $d_k(\text{cell}) = |\epsilon^{c_k} - \epsilon^{C}|$ at that cell, would give a weight map per subject, so the corrector could subtract the cheetah's prediction on the cougar's body and leave it on the cheetah's. That is CO3's own combination rule made spatial, close to Loop 10's region-from-$d_t$ candidate, and a candidate in [Paper 2's method](../04-paper-2-method.md).

**3. It acts where its failure is decided.** CO3 corrects only steps 0 to 5, where the two bodies' layout is set, and the failure it targets, one subject missing, is a layout failure. The pushes in other loops chose their windows (Loop 6's kept push at steps 0 to 15, Loop 8's at sampler steps 15 to 25) before anyone checked how much of the picture the latent held there, which is the brief's fifth design issue, and none measured where its target failure is decided. Z2's three windows measure where borrowing is decided. (inferred) If borrowing is set late, a fix belongs late, where CO3 never acts.

**4. Its push weakens as it succeeds.** Each pass recomputes the distances on the new latent, so as the joint prediction stops resembling one subject, the two weights move toward equal and the subtraction stops favouring either. The pushes in Loops 6 to 9 had a fixed strength and cap (strength 30, capped at 0.4 of the prediction) with no signal that the target had been reached. (inferred) A push whose size comes from the remaining gap stops itself.

**5. It puts the latent back where a noisy latent belongs.** After each combination CO3 adds back the empty prompt's noise at the size the step expects ($\sqrt{1-\bar\alpha_t}\,\epsilon^{\phi}$) and rescales the result to the input latent's length (`attn_utils.py` line 257). Gradient pushes add a direction and step on. Loop 6's M9 regions gave 8 clear pairs of 8 with every picture damaged. (inferred) Re-noising and rescaling after any edit may be what keeps an edit from damaging the picture.

## What this asks of Paper 1

Paper 1's two experts never saw a joint prompt, so there is no $\epsilon^{C}$ and CO3's signal does not exist there. (inferred) Its structure still transfers: several passes at one noise level, each edit followed by re-noising and rescaling, with a checker's reading standing in for the distance.

## Where these go

- Difference 2 is a candidate in [Paper 2's method](../04-paper-2-method.md).
- Differences 3, 4 and 5 are audit rows for Z5, beside the five issues in the brief.
