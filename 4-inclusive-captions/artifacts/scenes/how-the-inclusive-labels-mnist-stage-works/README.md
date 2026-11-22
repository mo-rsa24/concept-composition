# How the inclusive-labels MNIST stage works

**What this is**

Figures that take a reader from the guidance equation to the counts of the MNIST stage. The stage asks whether the product of "7" and "8" draws two separate digits when some training canvases filed under "7" also show an 8 `stated`.

**What is here**

[The figures card](figures/README.md) explains seven figures of the stage, r1 to r7, and keeps one earlier version of r6. Two more figures, s1 and s2, show the later share run and are explained on [the share run's results card](../../results/does-the-share-of-two-digit-pictures-help-the-product-compose-on-mnist/README.md). Each of the ten is saved as a PNG and an SVG `verified`.

**How it was made**

[The numbers script](make_mnist_stage_numbers.py) writes [the numbers file](mnist_stage_numbers.json) from the stage's code, checkpoints, samples and result files, and [the chain script](make_r5_trajectories.py) reruns the sampler for the step-by-step figure `stated`. [The stage figure script](make_mnist_stage_figures.py) draws r1 to r7, and [the share-run figure script](make_share_run_figures.py) draws s1 and s2 `stated`.

**Where it is used**

[The report page on inclusive labels](../../../report/do-inclusive-captions-help-the-product-compose/do-inclusive-labels-help-the-product-compose-on-mnist.md), [the report page on the share of two-digit pictures](../../../report/do-inclusive-captions-help-the-product-compose/does-the-share-of-two-digit-pictures-help-the-product-compose-on-mnist.md) and [the MNIST stage plan](../../../plans/11-do-inclusive-captions-help-poe-compose/plans/02-the-mnist-stage.md) all point here.
