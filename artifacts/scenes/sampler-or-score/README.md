# Sampler or Score

Five frames that separate the two ways a composition of two Gaussian experts can miss its target: the sampler and the score.
The experts are N(0, diag(10, 1)) and N(0, diag(1, 10)) from Catastrophic Compositional Generation. Switches turn the sampler
error, the score error and a 16-particle Feynman-Kac corrector on and off across 8 runs. The last frame shows which inputs a score
reads in three models from Bradley's Local Mechanisms paper, beside the most objects each one reached.

| File | What it is |
|---|---|
| `index.html` | the page source, 5 frames, published as the private artifact below |
| `sim.py` | runs the 8 toy combinations; needs only NumPy and takes about 19 minutes |

**Which reading it serves**

Reading 6, on Catastrophic Compositional Generation and Local Mechanisms. The page is the first picture to open in the
[reading plan's section on what breaks outside the training data](../../notes/progress/targets/reading-plan.md#sitting-6-what-breaks-outside-the-training-data).

**Published at**

[Sampler or Score](https://claude.ai/artifact/GRMF1EVkXeB82XNP5CFK8Q). It is private to its owner until shared from the page's
Share menu. Each frame opens at `#f1` to `#f5`.

**How it was made**

The experts and the base variance of 1 come from the paper. The base variance of 1.25, the trained networks and every distance
are this page's illustrative toy, so none of them reproduces a number in the paper. A NumPy script ran all 8 combinations with 400
Euler-Maruyama steps and 2,000 output samples over 5 sampling seeds. Distance is the sliced Wasserstein-2 distance to fresh target
samples, and lower is closer. With only the sampler wrong, the corrector cuts it from 0.847 to 0.237. With only the score wrong,
the corrector raises it from 0.216 to 0.942. The script is `sim.py` in this folder. `python3 sim.py` needs only NumPy, takes about 19 minutes and writes `toy_results.json` beside it.

**What it does not show**

A second trained network, run on 3 sampling seeds, moves the same way with different sizes, so read the trained-score distances as
directions. Target variance is 10 with sampler error off and 3.33 with it on, so compare the corrector within one setting only.
With exact scores the corrector still leaves variance near 4.2 against the 3.33 target. Frame 5's pixel grid is a schematic, and
only its bars (7, 2 and 3 objects) are the paper's measurements. Those three bars do not rank the models by locality. The page's
two errors are a different pair from the journey diagram `02-two-errors.png`, and it takes no position on the Appendix E sentence
beginning "The only exception".
