# Classifier guidance on PoE

**[Cat × Dog Guidance Ladder](guidance-ladder.html)** ([published page](https://claude.ai/artifact/7XDRvtF5SnU8VRBf2euPXi))

How a small trained checker pushes SDXL's product-of-experts drawing away from fused cat-dogs, in three zoom levels: one sampling step (three frozen U-Net calls, the PoE sum, the push, the DDIM update), the checker's inside (frozen SDXL features at three scales read by a head of about 0.78 M weights), and how labelled hybrids turn into a push direction. Written on 2026-11-18 during the rung 2 checker loop, to answer why classifier guidance was chosen over Kim et al.'s discriminator guidance. Sizes and formulas are read from `poe_repair/_sdxl/metrics.py`, `poe_repair/methods/_classifier_guided.py` and `scripts/rung2/train_readout_pair.py`; the status numbers are the loop's as of 05:20 UTC and are recorded in [the checker-loop table](../../drips/rung2-pair-aware-discriminator/checker-loop.md).

## The derivation, from assumptions to the sampler step

Moved to [the derivation page](../mnist-guidance-explained/4-the-derivation.md) in [the MNIST guidance explained folder](../mnist-guidance-explained/README.md), which holds the seven figures and the walkthrough. The scripts that draw them, `make_derivation_figures.py`, `make_logit_figure.py` and `make_logit_numbers.py`, stay in `derivation/` here.
