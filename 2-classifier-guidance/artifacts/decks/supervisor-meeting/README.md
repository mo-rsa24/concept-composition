# Supervisor meeting: the MNIST digit model and its training canvases

**What this is**

Two slides for a supervisor meeting on [plan 24](../../../plans/24-mnist-digits-on-a-canvas.md), the MNIST experiment of route step 7b. The first shows what the share of two-digit training canvases means, at shares 0, 0.01 and 0.1. The second follows one real call of the share-0.01 digit model, from a noisy canvas to its noise estimate.

**The numbers on the first slide**

Over 40,000 training steps of 128 examples, 0 of 5,120,000 canvases hold two digits at share 0, 50,931 of 5,069,069 at share 0.01, and 464,921 of 4,655,079 at share 0.1 `verified` ([the training logs](sources/)).

**How it was made**

[The slide script](make_slides.py) draws each slide as a PNG and an SVG. It reads every count from the training logs and the saved checkpoint at run time and checks that each split adds up `stated`.

**Where it is used**

[The report's list of decks](../../../report/walks-scenes-notes-and-decks.md#decks) names this one, and [the report page on the one-model MNIST experiment](../../../report/classifier-guided-poe/24-mnist-one-model.md) holds the experiment's results.
