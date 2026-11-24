# Review: does a sentence score's gradient improve CO3? (step W3)

**State: not run, because no step was readable.** The brief lets W3 push only on steps that W2 can read. No sentence score read any property over steps 0 to 15:
- **W2.** Every model's mean balanced accuracy fell between 0.46 and 0.56, against the 0.75 bar ([the W2 review](../02-can-it-read/review.md)).
- **W2c.** Prompt ensembling, NoisyCLIP's own training window, sentences that name no species, and the sampler's real states left every sentence row's mean over steps 0 to 15 under 0.6. The best was 0.574 ([the W2c review](../02c-fair-test/review.md)).

The owner chose on 2026-11-24 to stop W3 on these results (board questions 70 and 77). No picture was rendered, and none of W3's 4.5 budgeted GPU-hours was spent.

**What W3's design still offers.** [The design](design.md) and `2-classifier-guidance/scripts/noise_text_critic/w3_steer.py` hold the CO3 push with a fixed push length, a same-length random push and the soft clamp. A reader that does read steps 0 to 15 can use them unchanged. The W2c control, a readout head on SDXL's own features, reached 0.87 by step 15 on the sampler's real states, so it is the reader to try.
