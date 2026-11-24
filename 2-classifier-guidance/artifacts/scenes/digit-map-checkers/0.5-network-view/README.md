# The network view

Eight figures that open the checkers from the sampler step down to their weights, in zoom order. The first five are shared by both checkers; dc5 is where the count and identity branches split, and dc6 and dc7 hold for either branch with its own labels and weights.

Every size and count comes from `../numbers/numbers.json`. The canvases are real, rebuilt with the data hash `f70b0fb3...`. The noise on them is one numpy draw for display. The two maps are empty slots, because computing them needs the frozen U-Net on a GPU.

## Shared: the step, the canvas, the maps, the network, the timestep

**Where the checkers sit in one sampler step**

<a href="dc0-where-the-checkers-sit.png"><img src="dc0-where-the-checkers-sit.png" width="560" alt="One sampler step on the digit canvas: three frozen U-Net calls give the product and the two maps, the checker turns the maps into one logit, and the logit's gradient becomes a push of fixed length 0.4 of the product's"></a>

The toy's three U-Net calls make both the product's noise guess and the checker's input. The push is step V3's design; no pushed run exists. The step without a checker is [the 7b sampler step](../../mnist-guidance-explained/1-the-sampler-step/nv7b-1-one-sampler-step.png), whose two checkers are different networks.

**The canvas, and what the noise leaves of it**

<a href="dc1-the-canvas-and-its-noise.png"><img src="dc1-the-canvas-and-its-noise.png" width="560" alt="An apart canvas showing 1 and 3 and a wrong canvas showing 3 and 2, clean and at timesteps 712, 616, 539 and 480; at all four noise levels the digits cannot be seen by eye"></a>

At the 16 timesteps the checkers train on, $\sqrt{\bar\alpha_t}$ runs from 0.076 to 0.308, and on these two canvases the digits cannot be seen by eye at any of the four levels shown.

**Where the two maps come from**

<a href="dc2-where-the-two-maps-come-from.png"><img src="dc2-where-the-two-maps-come-from.png" width="560" alt="The noisy canvas through three frozen U-Net calls, the two differences from the no-digit call, and the scaling that makes each map add to 2,048; the map tiles are marked as owed"></a>

**The checker network**

<a href="dc3-the-checker-network.png"><img src="dc3-the-checker-network.png" width="560" alt="CNN volumes from 2 by 32 by 64 maps through three stem convolutions to 128 by 8 by 16, c1, scale and shift from the time features, c2, mean and max pooling to 256, and an MLP from 272 to one logit; 481,665 weights"></a>

**The timestep as sixteen numbers**

<a href="dc4-the-timestep-as-sixteen-numbers.png"><img src="dc4-the-timestep-as-sixteen-numbers.png" width="560" alt="A 16 by 16 heat map of the real time features, one row per training timestep, sin columns then cos columns, values from minus 1 to plus 1"></a>

## The branch: two label sets

**Two checkers, the same network, two label sets**

<a href="dc5-two-checkers-two-label-sets.png"><img src="dc5-two-checkers-two-label-sets.png" width="560" alt="The five canvas kinds for the asked pair 1 and 3 with the count label and the identity label under each; count trains on 11,504 canvases and identity on 7,004, of which 1,004 are wrong"></a>

## Either branch: the two backward passes

**Training: the error flows back into the checker's weights**

<a href="dc6-training-backprop-into-the-weights.png"><img src="dc6-training-backprop-into-the-weights.png" width="560" alt="The forward pass from the maps through the checker to the logit and the loss, and the backward lane carrying sigma of the logit minus y into each trained layer's weights and stopping before the maps and the U-Net"></a>

**The push: the same gradient carried on into the noisy canvas**

<a href="dc7-the-push-gradient-into-the-canvas.png"><img src="dc7-the-push-gradient-into-the-canvas.png" width="560" alt="The forward pass from the noisy canvas through the three U-Net calls, the maps and the checker to the logit, the backward lane all the way to the canvas, the push formula, and a table comparing the two backward passes"></a>

## How these were made

`make_network_view.py` draws all eight with `../figure_kit.py`, which refuses to write a figure where a line crosses a box or label. Run it with `~/.claude/venvs/figures/bin/python -I make_network_view.py`; each figure prints `0 crossings`.
