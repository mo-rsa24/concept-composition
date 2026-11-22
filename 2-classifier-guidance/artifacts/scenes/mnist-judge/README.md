# The MNIST judge: how its explainer figures were built

**What this is**

This folder holds one file, [the build record](ladder.json), for the pictures that explain the judge of route step 7b. The judge is a 10-way MNIST digit classifier, with a hand-written rule before it that finds the digit blobs and one after it that turns the digit readings into a call `stated`. The record marks the network view as built and the other six levels as missing `verified`.

**Where the pictures are**

The network view and a view of all three networks in the step sit in the figures folder of [the route walk](../../drips/route-classifier-guided-poe-from-toy-to-animals/ROUTE_WALK.md), which lists them.

**How it was made**

[The judge figure's script](../../drips/route-classifier-guided-poe-from-toy-to-animals/figures/make_judge_network_view.py) and [the three-network script](../../drips/route-classifier-guided-poe-from-toy-to-animals/figures/make_three_networks_in_step_7b.py) drew them. The judge they show is a throwaway prototype with 421,642 weights and a test accuracy of 0.9821 on 10,000 digits `verified` ([its numbers file](../../drips/route-classifier-guided-poe-from-toy-to-animals/figures/judge_network_view_numbers.json)). The record says to redraw the figure from whatever weights replace it `stated`.

**Where it is used**

[Step 7b of the route](../../../plans/route-classifier-guided-poe-from-toy-to-animals.md#7b--mnist-digits-on-a-canvas-how-good-must-the-checker-be) designs the judge, and [the report page on the one-model MNIST experiment](../../../report/classifier-guided-poe/24-mnist-one-model.md) shows the three-network view.
