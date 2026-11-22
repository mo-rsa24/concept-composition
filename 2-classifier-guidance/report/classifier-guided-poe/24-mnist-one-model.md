# A2b: MNIST digits on a canvas, how good the checker must be

Beside rung 2 of [the scale-up ladder](board.md#the-scale-up-ladder), gating nothing. Step 7b of
[the route](../../plans/route-classifier-guided-poe-from-toy-to-animals.md#7b--mnist-digits-on-a-canvas-how-good-must-the-checker-be)
holds the full design: a class-conditional pixel U-Net per two-digit share (0, 0.01, 0.1) on 32 × 64
canvases of real MNIST digits, five pairs held out (0 & 3, 1 & 9, 3 & 4, 3 & 6, 4 & 7), a checker
that reads the frozen U-Net's decoder features with the asked pair as an input, and a judge that
reads single digits only. Labels are exact and free here, so the checkers can be made weaker on
purpose, and the compose count against checker accuracy says whether rung 2's cat × dog checker
(0.655 held out, CLIP's ceiling 0.771) could steer at all.

<a href="../../artifacts/drips/route-classifier-guided-poe-from-toy-to-animals/figures/three-networks-in-step-7b.png"><img src="../../artifacts/drips/route-classifier-guided-poe-from-toy-to-animals/figures/three-networks-in-step-7b.png" width="600" alt="Three rows: the pair-aware checker reading a noisy canvas and the asked pair and giving one number whose gradient is the push; the separate-label control asking 7 present and 8 present; the judge cutting a finished canvas into blobs and reading each digit"></a>

## Outcomes

- [x] 📝 the bar copied into a review file in `plans/review/` before any model trains: [the review file](../../plans/review/a2b-mnist-digits-on-a-canvas.md), 2026-11-18 `verified`
- [x] 📝 training time measured before the three models launch: 486 ms per step on the laptop, 157 on a `bigbatch` RTX 3090, so 105 min per share at 40,000 steps (`timing-*.json` beside each model) `verified`
- [x] 🖼️ screenshot: plain product and the support check, as a sheet, per share: share 0.01 (on the cluster at `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs/plain-product/plain-product-share-0.01.png`), [share 0.1](../../artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs/plain-product/plain-product-share-0.1.png); fuses at every share `verified`
- [x] ✅ check passed: the judge agrees with the eye calls on 58 of 64 tiles (the bar is 58), [the gate sheet](../../artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs/judge-gate/judge-against-eye-calls.png); the calls are Claude's, blind, under the user's veto `verified`
- [x] 🖼️ screenshot: the five-version sheet on 7 & 8 and on the five held-out pairs: 7 & 8 (on the cluster at `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs/five-versions/sheet-78.png`); counted by eye after the judge failed on drawn digits `verified`
- [x] 📈 figure: compose count against checker accuracy, with 0.655 and 0.771 marked, and the cap sweep: [the figure](../../artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs/compose-against-checker-accuracy.png); every checker sits below 0.6, so neither mark is reached `verified`
- [x] ✅ check passed: the reading printed as pass or miss against the bar: works on seen pairs only, two departures named, in [the review file](../../plans/review/a2b-mnist-digits-on-a-canvas.md) `verified`

## The bar

On 7 & 8, 64 seeds, on the model where plain product fuses: plain product gives two clear, right
digits on at most 8 of 64; the pair-aware push on at least 32 of 64 at some s; the random push and
the separate-label push within 6 of plain product; best-of-N at matched compute at least 8 below
the pair-aware push. On the five held-out pairs, at least 160 of 320. A share where plain product
already gives at least 32 of 64 for 7 & 8 has nothing to fix and leaves the bar.

## The result

**Works on seen pairs only.** On 7 & 8, a checker that knows the asked pair turns plain product's
fused digit (1 of 64 seeds with a clear 7 beside a clear 8) into two separate digits on 47 of 64 at
strength 10; a random push of the same size gives 1, and best-of-2 at the same cost gives 1. On the
five pairs never seen together the push still makes two digits, but often the wrong second one: 84
of 320. Counted by eye, since the judge misreads the model's thick digits.

`sheet-78.png` is on the cluster at `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs/five-versions/sheet-78.png`
<a href="../../artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs/compose-against-checker-accuracy.png"><img src="../../artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs/compose-against-checker-accuracy.png" width="400" alt="Six checkers between 0.51 and 0.59 accuracy against their push's count of 64, all left of the 0.655 and 0.771 marks; raising the cap from 0.1 to 1.0 moves 47 to 49"></a>

Two departures from the bar: the separate-label push composes some (15 of 64), and the 0.1 cap
clips most steps at s = 30 though raising it changes nothing at s = 10. No digit checker passed 0.59
held-out accuracy, so the 0.771 rule cannot pick step 10's move; two checkers at the same accuracy
composed 47 and 27, so accuracy alone does not decide it.

**A follow-on run reached the new pairs.** A checker that also sees the asked pair at every layer,
retrained twice on its own push's pictures, with a partly random sampler, gives two clear, right
digits on 217 of 320 seeds over the five pairs never seen together (3 & 4 and 3 & 6 at 62 of 64; 0 & 3
at 25), against 0 for the same sampler without the push. Adding the separate-label push to it lifts this
to 285 of 320 (89%) on fresh seeds, where the pair-aware push alone gives 215.

`heldout-final.png` is on the cluster at `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs/loop2/heldout-final.png`

## More pictures from these runs

**The fresh-seed sheet shows the 285 of 320 pair by pair, and two held-out sheets show the lowest and
highest pair from the first run.** The first run's eye counts at strength 3 are 3, 14, 25, 11 and 31
for 0 & 3, 1 & 9, 3 & 4, 3 & 6 and 4 & 7, which add to 84 of 320 `verified`
([the eye counts](../../artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs/five-versions/eye_counts.json)).

| Fresh seeds, both pushes | Lowest held-out pair | Highest held-out pair |
|---|---|---|
| <a href="../../artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs/loop2/final-seeds64.png"><img src="../../artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs/loop2/final-seeds64.png" width="200" alt="Fifteen rows of ten canvases: five held-out pairs, each with no push, the pair-aware push, and both pushes"></a> | <a href="../../artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs/five-versions/sheet-03.png"><img src="../../artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs/five-versions/sheet-03.png" width="200" alt="Five rows of 16 canvases asked for 0 and 3, one row per version"></a> | `sheet-47.png` is on the cluster at `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs/five-versions/sheet-47.png` |
| Ten of each pair's 64 fresh seeds (64 to 73), one row per setup. With both pushes the blind eye counts are 44, 53, 64, 64 and 60 of 64, which add to 285 `verified` (the picture and `data/mnist-canvas/24-mnist-one-model/five-versions/loop5-final-seeds64/eye_counts_blind.json`) | 0 & 3, the first 16 of 64 seeds in five versions. The sheet prints the judge's counts. At strength 3 the pair-aware push gets 2 of 64 from the judge and 3 by eye `verified` | 4 & 7, the same layout. At strength 3 the pair-aware push gets 16 of 64 from the judge and 31 by eye `verified` |

**Each network diagram was tried in four drawing styles.** The styles are your slides, a clean cheat
sheet, a flat schematic and textbook ink, and the card names the cheat-sheet style for the finished
network views `stated` ([the card](../../artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs/README.md#network-view)).

| The guidance sum | The [readout head](which-line-computes-which-part-of-guided-sampling.md#concepts-this-uses) | The U-Net |
|---|---|---|
| `style-samples-guidance.png` is on the cluster at `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs/network-view/styles-guidance/style-samples-guidance.png` | <a href="../../artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs/network-view/styles-readout/style-samples-readout.png"><img src="../../artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs/network-view/styles-readout/style-samples-readout.png" width="200" alt="The readout head diagram drawn in four styles side by side"></a> | `style-samples-unet.png` is on the cluster at `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs/network-view/styles-unet/style-samples-unet.png` |
| The unconditional prediction plus s times the difference, at s = 3 `verified` | Frozen layers, three taps, the head, its map, and the gradient that moves the latent `verified` | Four levels down and up, with skip connections across the U `verified` |

The other 23 versions, judge checks and style sheets in [this result's folder](../../artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs) are held out of the artifact drain because the summaries above contain them.
