# How the product-of-experts toy experiment works

## [How the experiment works](how-the-experiment-works.md)

**What it is.** A note that explains the ideal expert, the exact product and the samplers of the 2D toy experiment, each with its formula and the line of code behind it, plus what changes when the experts are trained ResNets.

**How we got it.** Written from `toy/rebuild/ideal.py`, `samplers.py`, `model.py` and `toy/anneal_samplers.py`, and from the conversation that built the table. The equations for ULA, MALA and UHA follow the repo's classes; the leapfrog line is the standard form, not copied from the code.

**What it lets you say.** What each column of the table computes and which gap in a row is network error and which is sampler error.

**What it cannot say.** It reads the table by eye and gives no total variation numbers. The cause named for the samplers' loss in the joint-point row is untested.

**Where it came from.** [The results folder](../../results/what-the-product-of-two-experts-draws-and-what-samplers-lose/README.md).
