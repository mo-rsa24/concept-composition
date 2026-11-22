# Supervisor meeting: do inclusive captions help the product compose?

**What this is**

Twelve slides that ask one question in two small worlds: if each single caption has seen both things together in some of its training pictures, does the product of the two captions draw both? The worlds are a toy of red circles and blue squares, and MNIST digits `stated`. [The talk-through](talk-through.md) gives each slide, what to say and the questions likely to come up.

**What the last slide shows**

All counts are from training 1 at guidance 3. At a 50% share the toy's product draws a clean pair on 209 of 300 pictures, against 52 of 300 for "a red circle" alone `verified`. At a 40% share the MNIST product draws 137 of 256, while "7" alone already draws a 7 with a second digit on 200 of 256 `verified`, so MNIST shows no margin over its label `stated`. The MNIST judge has not been checked against eye calls on this run `stated`.

**How it was made**

[The slide script](make_simple_figures.py) draws the twelve slides, and [the detailed script](make_supervisor_figures.py) draws the eight fuller versions in [the detailed folder](detailed/). Both read every count from the result files at run time `stated`.

**Where it is used**

[The report page on the toy](../../../report/do-inclusive-captions-help-the-product-compose/does-relabelling-both-shapes-pictures-help-the-toy-product-compose.md) and [the report page on the share of two-digit pictures](../../../report/do-inclusive-captions-help-the-product-compose/does-the-share-of-two-digit-pictures-help-the-product-compose-on-mnist.md) hold the findings, and [the MNIST share-run plan](../../../plans/11-do-inclusive-captions-help-poe-compose/plans/03-the-mnist-share-run.md) holds the run. [The report's list of decks](../../../report/walks-scenes-notes-and-decks.md#decks) names this one.
