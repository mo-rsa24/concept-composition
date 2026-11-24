# Review: does a linear probe on frozen image vectors read counting and prompt match on unseen pairs? (step W2b)

**State: null for all three encoders.** A logistic regression trained on each encoder's image vectors, one per step from 0 to 15, reads both properties a little better than chance on seen pairs, at best 0.637 and 0.644 for NoisyCLIP. On the 7 unseen pairs it reads 0.527 to 0.586, against the 0.75 bar. The random-number control stays at chance (0.506 and 0.497), so the probe found real but weak information. W2 found sentences could not reach it, and a trained linear layer reaches little more. The [design](design.md) holds the bars, and the numbers are in [the result file](result.json).

## Runs

| Job | Stage | Code | Outcome |
|---|---|---|---|
| 94110 | smoke test | `c6bd3b2b` | Failed at the fit stage: training tiles from saved trajectories hold every state in one file, and the extractor took all of them |
| 94219 | smoke test, 34 pictures | `8cbdaf49` | Ran after the extractor kept only the state each tile names. 0 latents missing, all four readers fitted |
| 94220 | full run: 4,039 pictures, steps 0 to 15, one shared noise draw | `8cbdaf49` | 13 minutes. 3,268 pictures to fit, 349 seen test, 422 from the 7 unseen pairs; 0 latents missing |

## Result

Mean balanced accuracy over steps 0 to 15. The seen test is held-back pictures of training pairs, and the unseen test is the 7 pairs never trained on, read once.

| Encoder | Counting, seen | Counting, unseen | Prompt match, seen | Prompt match, unseen | Verdict |
|---|---|---|---|---|---|
| NoisyCLIP | 0.637 | 0.560 | 0.644 | 0.586 | null |
| frozen CLIP | 0.579 | 0.526 | 0.614 | 0.552 | null |
| Latent CLIP | 0.575 | 0.527 | 0.579 | 0.527 | null |
| random numbers (control) | 0.502 | 0.506 | 0.505 | 0.497 | control passes (at most 0.6) |

The unseen test held 195 two-animal and 227 one-animal or hybrid pictures for counting, and 109 right-pair and 86 wrong-pair pictures for prompt match, so each label clears the minimum of 30.

**What it means.** NoisyCLIP's vectors hold more early information than frozen CLIP's or Latent CLIP's, which fits its noise training. All three fall by 0.05 to 0.08 from seen to unseen pairs, and none comes near the bar. The readout checker on SDXL's own U-Net features read 0.87 by step 15 on real states (W2c, cat and dog only). So the early information sits mostly in the diffusion model's features, not in a CLIP-style image vector. This is inferred from the two readers' different inputs, a mixed comparison.
