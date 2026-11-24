# Design: what CO3's corrected steps do, on one pair that borrows and one that does not

The owner asked on 2026-11-24 for the CO3 zoom ladder to show the real fix: what CO3 changes at each corrected step, where, with its mathematics, and as pictures. This run records every pass of CO3's resampler and corrector on two pictures, so the ladder can draw them exactly.

## What it is for

**Group.** An illustration for the [CO3 zoom ladder](../../../../../2-classifier-guidance/artifacts/scenes/co3-sampler/index.md). It makes no claim in either paper and has no verdict.

**Varied axis.** The pair: eagle × fox at seed 6001, labelled `own_parts` by W2d, and cheetah × cougar at seed 6001, labelled `borrowed` ("cougar has cheetah spots and tear marks", [the part-binding calls](../../noise-text-critic/02d-binding/part-binding-calls.csv)). Model, settings and seed are M1's CO3 row unchanged. Both are tuning pairs; no held-out pair or seed is used.

## What it records

For each of CO3's 40 passes (10, 10, 5, 5, 5 and 5 at its 6 corrected steps), the latent going in, the four noise predictions (empty prompt, joint prompt, each animal alone), $\bar\alpha_t$ as CO3 computes it, and the weights the code returns from its weight function. Each step's clean-picture guesses are decoded through the VAE at the step's first and last pass. The latent at every one of the 50 steps is kept for CO3 and for the same joint prompt with CO3's correction switched off. The same seed is also rendered with CO3's correction off for the joint prompt, each animal's prompt alone, and with CO3 on.

## How much of the picture the input holds

At the corrected steps the latent is 0.6% to 1.7% picture on SDXL's schedule. The decoded clean-picture guesses are the model's prediction of the finished picture from that latent, so they show what each prompt is heading toward, blurred where the model is unsure.

## The control

**The records reproduce the code.** From each pass's recorded latent, predictions and weights, the runner recomputes the next latent with the authors' formulas (`attn_utils.py`, resampler lines 194 to 198, corrector lines 251 to 257) and compares it with the latent the next pass actually received. A relative difference above 1% on any pass means the records do not describe what ran, and the figures are not drawn.

**The weights the code uses.** The distance list that sets the weights is filled with one distance per subject on each pass and cleared at the end of the pass (`attn_utils.py` lines 187 and 209 in the resampler, 245 and 267 in the corrector), so each pass's weights come from that pass's two distances. The run recorded the weights the code returned on every pass: two per pass on all 80 passes.

## What the run recorded

Slurm job 94450, code snapshot `44b51af4`, outputs under `/datasets/mmolefe/poe_repair_min/loop-12/co3-inside/`. The control passed on both pairs: recomputing each pass from the records reproduces the next latent to a relative difference of at most 0.00073 (eagle × fox) and 0.00077 (cheetah × cougar), against the 0.01 limit. For cheetah × cougar the two distances stay close on every corrected step (for example 0.8 and 1.0 at steps 1 to 3), so the weights stay near −0.55 and −0.45; for eagle × fox they spread further (2.4 and 4.1 at step 5, weights −0.82 and −0.18).

## Cost and what it buys

**Cost:** about 0.3 GPU-hours on one 48 GB card: 2 CO3 renders, 6 renders with the correction off, and about 110 VAE decodes. Outputs about 150 MB under `/datasets/mmolefe/poe_repair_min/loop-12/co3-inside/`. No API tokens.

**Buys:** the real fix drawn exactly: CO3's move at each corrected step on the plane through the joint's and the two animals' clean-picture guesses, with the decoded guesses beside it, on a pair that borrows and one that does not.
