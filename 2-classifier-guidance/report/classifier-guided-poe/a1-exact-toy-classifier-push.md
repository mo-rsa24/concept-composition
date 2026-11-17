# A1: the exact toy, a pair-aware push against a separate-label push

Rung 1 of [the scale-up ladder](board.md#the-scale-up-ladder), run as [plan 19](../../plans/19-classifier-push-on-the-exact-toy.md). On the exact circle-and-square toy at pair weight 0.01, the exact product draws 300 of 300 side-by-side pairs and the sampler that adds the two experts' scores stacks all 300. A small classifier's gradient is added to that sampler at every step, with the experts frozen.

## Outcomes

- [x] 📝 bar written before the run: pair-aware at w = 1 at least 150 of 300 crisp with overlaid no higher than unguided; separate-label at most 15 of 300 at every scale in 1, 2, 5, 10 · `verified` · the plan's Goal, before any classifier was trained
- [x] 💻 code ran: `toy_classifier_push.py`, eight commands, on the laptop's RTX 4070; every count run once as an answer key and once as written, matching on every row · `verified` · commits `50a1b48b` to `5ccb723f`
- [x] 🖼️ figures: [the discriminator as seven stages](../../../3-understanding-compositionality/artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/the-side-by-side-discriminator.png), [the push's cosine per step](../../../3-understanding-compositionality/artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/push-cosine-per-step.png), [the minimal-pair gap per step](../../../3-understanding-compositionality/artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/minimal-pair-gap-per-step.png) · `verified`
- [x] ✅ check passed: both halves of the bar pass · `verified` · `pair-aware-guide.json`, `separate-label-guide.json`

## The verdict

<a href="../../../3-understanding-compositionality/artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/the-side-by-side-discriminator.png"><img src="../../../3-understanding-compositionality/artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/the-side-by-side-discriminator.png" width="600" alt="Seven stages: two piles of real and fake draws, noising, the network, d, the push, where it enters the sampler, and crisp pairs out of 300 at each push weight"></a>

**Pass.** Pair-aware push at w = 1: 172 of 300 crisp, 128 overlaid, against 0 and 300 unguided; 300 of 300 at w = 2. Separate-label push: at most 10 of 300 over scales 1 to 10. Seed 7, 300 draws. The numbers and how they were made are on [the card](../../../3-understanding-compositionality/artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/README.md).

## What it confirms, and what it doesn't

**It confirms that a push trained on the sampler's own mistake turns stacked draws into side-by-side ones, with the experts frozen.** The push points along the sampler's own ratio (cosine 0.85 to 0.96 from step 190 to step 90), it prefers the side-by-side picture to the stacked one by up to 28 log-odds against 0 for shuffled labels, and it never reaches a tenth of the summed score's size, so the SDXL sampler's cap of 0.1 does not limit it.

**It does not show a fix to the product's target.** On this toy the exact product already composes; only the summed-score sampler fails to reach it. The push repairs that sampling failure. Where the product's own target is a hybrid (the toy's competing-captions condition, whose exact product draws 4 or 19 crisp of 300), this result says nothing. On SDXL the gap holds sampler error and model error together, and nothing yet separates them.

**The separate-label half passes on the trained classifier only.** The closed-form best separate-label classifier reaches 34 of 300 at scale 10, because the side-by-side picture is the one training picture carrying both labels. Read on the trained classifier by the reader's choice; the 34 stands beside it.

**Against inclusive captions on the same noises.** With no push, raising the pair weight gives 0, 1, 18, 53, 115 and 246 crisp at 0.01, 0.03, 0.1, 0.2, 0.3 and 0.5. The push at 0.01 beats inclusive captions at 0.3. Both need pictures of the pair (the discriminator saw 3,500).

**Limits.** One seed of 300 draws per count. The labels do not check that the pairs land in the exact product's positions in the right proportions.
