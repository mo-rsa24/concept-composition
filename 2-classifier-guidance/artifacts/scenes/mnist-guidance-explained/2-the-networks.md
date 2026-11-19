# 2. The networks: how big, what goes in, what comes out, how each was trained

**The question.** For the canvas U-Net and each of the two checkers: how many weights, what are the input and output shapes, how many layers, how long did it train, and what does it answer on a real canvas?

Every number comes from `architecture/arch_numbers.json`, read off the code, the checkpoints and the training log by `architecture/make_arch_numbers.py` (in [the step 7b results](../../results/how-good-must-the-checker-be-on-mnist/README.md)). The pictures are a real noisy canvas, the U-Net's real noise prediction for it, and four pasted MNIST test canvases noised to t = 489, asked 4 and 7.

## Words on this page, taken apart

Toys are in [the toys](toys/triad_toys.py); every hand answer is computed and asserted there.

**A skip**

💥 **Broken run (without):** squeeze a strip of four pixels [0, 4, 4, 0] down to two by averaging pairs, then spread each back to two: you get [2, 2, 2, 2], and the strip has lost its edges, an error of 8 pixel-units. With the original copied across beside it, the error is 0.

🔤 **The name:** a *skip* jumps over the middle of the network. In the U, the left arm's map at each size is copied straight across to the right arm at the same size and joined on as extra channels, which is why a right-arm bar is part light (the copy) and part dark (what it just computed).

🧸 **Toy:** the four-pixel strip above. 🔨 **Build:** `toys/triad_toys.py`, toy E, gives 8 and 0. 🎞️ **Moving picture:** 🖼️ Diagram wanted: the strip squeezed and spread, with the copy sliding across.

**Scale and shift by the pair (the pair-aware checker's second layer)**

🔤 **The name:** *scale* multiplies and *shift* adds. The asked pair makes two numbers, g and b, and the layer's output h becomes h × (1 + g) + b, so the pair can turn parts of the picture up, down or off before anything else is read.

🧸 **Toy:** h = [2, −1]. Pair A gives g = 0.5, b = 1: the output is [4, −0.5]. Pair B gives g = −1, b = 0: the output is [0, 0], switched off. With no pair at all it stays [2, −1]. 🔨 **Build:** toy F.

💥 **Broken run (without):** give the pair only at the input and the checker reads held-out pairs on 84 of 320 seeds; give it to every layer and retrain on its own pushed pictures with the sampler's fresh noise, and it reads 217 of 320 (three changes at once, step 7b's loop 2, so the gain is not credited to the pair alone). 🎞️ **Moving picture:** 🖼️ Diagram wanted: g and b sliders on the toy.

**Mean and max (the pooling)**

🔤 **The name:** *pool* is a shared pot; *mean* puts everyone's share in and divides, *max* keeps the biggest. The checker pools each 128-channel map over its 128 places to two numbers per channel: 128 means and 128 maxes, 256 in all.

🧸 **Toy:** a 2 × 3 map with one bright cell worth 6 and zeros elsewhere. The mean is 1 and the max is 6. A single bright stroke is washed out by the mean and stands out in the max. 🔨 **Build:** toy G.

**The two taps, and a frozen network the gradient still passes through**

🔤 **The name:** a *tap* is a place a pipe is opened to draw from. Tap A is the decoder's output at 8 × 16 and tap B at 16 × 32; *frozen* means the U-Net's weights are not trained. A frozen layer is still in the path, so the push's gradient goes straight through it to the canvas.

🧸 **Toy:** a frozen layer f(x) = 2x and a head 3 f(x) + 1. At x = 1 the answer is 7 and its slope is 3 × 2 = 6, checked by nudging x by 0.001 (slope 6.000). Cut the frozen layer off, as if it were not in the path, and the slope is 0: the checker would answer but never push. 🔨 **Build:** toy H.

💥 **Broken run (misuse):** treating "frozen" as "no gradient" would give a push of zero at every step. The gradient figures on [page 3](3-the-gradient.md) are the measured case: it enters the frozen U-Net at the taps (0.480 and 0.784) and arrives at the canvas as 3.21.

**EMA (the weights are averaged)**

🔤 **The name:** *exponential moving average*: each new value moves the average by a small fixed share. With decay 0.9 and a target of 1, starting at 0, three steps give 1 − 0.9³ = 0.271. The U-Net uses 0.999, which moves 0.1% a step, so it follows training slowly and smooths its jitter. 🧸 **Toy and build:** toy J.

**Retraining on its own pushes (on-policy)**

💥 **Broken run (without):** a checker taught only on pasted pictures never sees the failures its own push makes. Retrained on 444 of its own pushed pictures per pass, the judge labelled 214 then 293 of them clean, and the tuning pairs read 81, 77 and 120 of 192 over the three checkers (step 7b's loop 2).

🔤 **The name:** *on-policy*: learning from what you yourself do, rather than from a fixed pile someone else made.

## The canvas U-Net: 6,141,185 weights

<a href="2-the-networks/a1-the-canvas-unet.png"><img src="2-the-networks/a1-the-canvas-unet.png" width="480" alt="The canvas U-Net as bars from 32 x 64 down to 8 x 16 and back, skips, the two maps the checkers read, the size and the training recipe"></a>

**In and out.** A 1 x 32 x 64 noisy canvas, the step (0 to 999) and a digit label (0 to 9, or 10 for none) go in. A 1 x 32 x 64 noise prediction comes out; real output at t = 489 has mean −0.033 and spread 1.01.

**Inside.** Bars go from 32 x 64 with 64 channels down to 8 x 16 with 128 channels, through one attention block over 128 places, and back up. There are 17 ResBlocks. The step and label become 256 numbers added inside every ResBlock. The decoder's output at two depths, tap A (128 x 8 x 16) and tap B (128 x 16 x 32), is what the checkers read.

**Size.** Encoder 1,615,040, bottleneck 723,456, decoder 3,717,441, step and label embeddings 85,248.

**Training.** 40,000 steps of 128 canvases built fresh each time, so there are no fixed epochs: 5,120,000 canvases, about 85 passes over MNIST's 60,000 training digits. One canvas in 100 holds two digits (50,931 seen). Adam, learning rate 0.0002, weights averaged with EMA 0.999, squared error on the noise, label dropped to none 10% of the time, 1,000 noise levels with beta from 0.0001 to 0.02. 2 h on one RTX 3090; final loss 0.0077.

## The pair-aware checker: 694,017 weights

<a href="2-the-networks/a2-the-pair-aware-checker.png"><img src="2-the-networks/a2-the-pair-aware-checker.png" width="480" alt="The pair-aware checker: the frozen U-Net with two taps, convolutions with the asked pair scaling and shifting, pooling, an MLP, one log-odds out, real answers on four canvases"></a>

**In and out.** The canvas, the step and the asked pair go in; one number, d's log-odds, comes out (shape B x 1). On the four test canvases it gives 0.82 for a 4 and a 7 apart, 0.63 for two 4s, 0.18 for a lone 4 and 0.03 for the fused pair.

**Inside.** The two taps are joined to 256 channels at 8 x 16, pass through two 3 x 3 convolutions with the pair scaling and shifting between them (16,640 weights), are pooled to 256 numbers, joined with 16 step numbers and the 64-number pair embedding, and go through two linear layers.

**Training.** 4,000 steps of 256 = 1,024,000 canvases, Adam 0.001. Positives: 8,000 pasted pairs and 688 drawn; negatives: 7,998 pasted (fused, lone, wrong pair) and 3,160 drawn, 888 of them its own pushed pictures. The three tuning and five held-out pairs were never shown. 9 min 42 s including renders.

## The separate-label checker: 662,666 weights

<a href="2-the-networks/a3-the-separate-label-checker.png"><img src="2-the-networks/a3-the-separate-label-checker.png" width="480" alt="The separate-label checker: the same taps and layers without the pair input, ten numbers out, real answers on four canvases"></a>

**In and out.** The canvas and the step go in; ten numbers, one per digit, come out (shape B x 10). Each goes through its own sigmoid to a chance that digit is present; [the derivation's step 7](4-the-derivation.md#step-7-the-checkers-outputs-are-logits) says why and why there is no softmax.

**Training.** The same trunk without the pair or its scale and shift, 15,998 pasted canvases with exact digit presence, 4,000 steps of 256, no drawn canvases.

## Quicker views of the same networks

These drawings give less detail than the three sheets above and show where the networks sit in a sampler step. Their numbers come from `network-view/network_view_numbers.json`.

| Figure | What it adds |
|---|---|
| <a href="2-the-networks/nv7b-2-the-canvas-unet.png"><img src="2-the-networks/nv7b-2-the-canvas-unet.png" width="200" alt="The canvas U-Net as bars with skip arcs and attention at 8 x 16"></a> | the U-Net as one picture, 64 channels at 32 x 64 and 128 at 16 x 32 and 8 x 16; the label joins the time vector, with no cross-attention |
| <a href="2-the-networks/nv7b-3-the-pair-aware-checker.png"><img src="2-the-networks/nv7b-3-the-pair-aware-checker.png" width="200" alt="The pair-aware checker with the push's gradient lane back to the canvas"></a> | the pair-aware checker with the push's gradient lane back to the canvas |
| <a href="2-the-networks/nv7b-4-the-separate-label-checker.png"><img src="2-the-networks/nv7b-4-the-separate-label-checker.png" width="200" alt="The separate-label checker with ten digit-present answers"></a> | the separate-label checker; its push raises the two asked digits |
| <a href="2-the-networks/r2-the-separate-label-checker.png"><img src="2-the-networks/r2-the-separate-label-checker.png" width="200" alt="The separate-label checker as circles with 4 and 7 glowing"></a> | the separate-label checker as circles, canvas in, ten answers out, the push flowing back |
| <a href="2-the-networks/r3-the-pair-aware-checker.png"><img src="2-the-networks/r3-the-pair-aware-checker.png" width="200" alt="The pair-aware checker as circles with the pair feeding every layer"></a> | the pair-aware checker as circles, with the asked pair feeding every layer |
| <a href="2-the-networks/nv7b-5-retraining-on-its-own-pushes.png"><img src="2-the-networks/nv7b-5-retraining-on-its-own-pushes.png" width="200" alt="Three checkers in a row, each pushing 444 pictures that the judge labels before the next retrains"></a> | how the pair-aware checker is retrained on its own pushed pictures: 444 per pass, 214 then 293 judged clean |
