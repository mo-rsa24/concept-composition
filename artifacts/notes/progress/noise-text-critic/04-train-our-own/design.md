# Can our own noise-aware text and image model read unseen digits? (step W4)

This step trains a small image encoder on noisy digit canvases against a frozen text encoder, the same recipe NoisyCLIP uses, and tests it on digits it never saw. Digits are the cheapest place to answer the question behind any larger version: does a sentence score learned on some concepts carry to concepts outside its training set?

## Table of contents

- [Words this uses](#words-this-uses)
- [The run](#the-run)
- [The data](#the-data)
- [The sentences](#the-sentences)
- [The model and its training](#the-model-and-its-training)
- [How a canvas is scored](#how-a-canvas-is-scored)
- [The bars](#the-bars)
- [The order of work](#the-order-of-work)
- [The cost of an SDXL version](#the-cost-of-an-sdxl-version)
- [What this cannot say](#what-this-cannot-say)

## Words this uses

**Canvas.** A 32 by 64 pixel greyscale image of pasted MNIST digits, from [step V1's environment](../../abstract-checkers/04-environment/design.md).

**Asked pair.** The two digits a canvas is judged against.

**Seen and unseen.** Seen digits are 0 to 4, the only digits in training. Unseen digits are 5 to 9.

**Caption bank.** Every training sentence, each embedded once by the frozen text encoder.

**Checker score.** The right sentence's cosine with the canvas minus the highest wrong sentence's cosine.

**Balanced accuracy.** The mean of the share of label-1 and the share of label-0 canvases called correctly, from 0 to 1, where 0.5 is chance.

## The run

**Group.** Runs that try a new idea (group 2), following NoisyCLIP's recipe of tuning an image encoder on noisy inputs against a frozen text encoder ([its note](../01-reading/noisyclip.md)). A result here cannot change a claim in Paper 2.

**Varied axis.** Whether the digits are seen or unseen. One model is trained once, and the same model reads both test splits.

**Cost:** about 2 GPU-hours on one `bigbatch` card, covering the smoke test, the overfit test, the 1% run and one full training of 8,000 updates. **Buys:** step W4, and a measured answer to whether a sentence score trained on some concepts reads others.

No diffusion model is trained. The digit diffusion models are not used here, because this step reads noised canvases and draws nothing.

## The data

The step reads V1's `dataset.npz` under `/datasets/mmolefe/poe_repair_min/loop-8/v1/data/` and refuses any file whose hash is outside `DATA_HASHES` in `abstract_checkers/v2_learn.py`. Nothing in that folder is written.

| Split | Digits | Canvases | Used for |
|---|---|---|---|
| `train` | 0 to 4 | 20,000 | training |
| `seen_test` | 0 to 4 | 2,000 | choosing each step's threshold on half, and reading the seen score on the other half |
| `unseen_test` | 5 to 9 | 2,000 | read once, after everything is frozen |

The five kinds of canvas and their labels are V1's: `apart` is label 1, and `overlaid`, `same_half`, `single` and `wrong` are label 0.

## The sentences

Digits are written as words. Each canvas has one true caption, used in training:

| What the canvas holds | True caption |
|---|---|
| Two separate digits, one per half (`apart`, `wrong`) | the digit three and the digit seven, apart |
| Both digits on one spot (`overlaid`) | the digit three on top of the digit seven |
| Both digits in one half (`same_half`) | the digit three and the digit seven on the same side |
| One digit (`single`) | one digit three |

In every two-digit caption the lower digit is named first, so a caption says nothing about left and right.

**Reading an asked pair $(a, b)$.** The right sentence is "the digit $a$ and the digit $b$, apart". The wrong sentences are:

- "one digit $a$" and "one digit $b$"
- "the digit $a$ on top of the digit $b$"
- "the digit $a$ and the digit $b$ on the same side"
- for every other digit $c$ from zero to nine, "the digit $a$ and the digit $c$, apart" and "the digit $b$ and the digit $c$, apart", each with its lower digit named first

## The model and its training

**Text side.** The frozen text encoder and text projection of `openai/clip-vit-large-patch14`, giving one 768-number vector per sentence.

**Image side.** A small convolutional encoder reads the noisy canvas: four convolution blocks with widths 64, 128, 256 and 256, each with group normalisation and a SiLU activation, halving the grid after each block. A learned embedding of the step index shifts and scales each block's features. The final grid is flattened, so left and right stay distinct, and a linear layer gives a 768-number vector.

**Loss.** Cross-entropy over the caption bank, with each canvas's true caption as the target:

$$
\mathcal{L} = -\log \frac{\exp(\cos(f(x_t, t), g(c^\star)) / \tau)}{\sum_{c \in \text{bank}} \exp(\cos(f(x_t, t), g(c)) / \tau)}, \qquad \tau = 0.07
$$

$f$ is the image encoder, $g$ the frozen text encoder and $c^\star$ the canvas's true caption. The bank holds only captions naming digits 0 to 4, so no unseen digit's name reaches the training loss.

**Training.** 8,000 updates at batch 256 with Adam at learning rate $2\times10^{-4}$. Each example draws a canvas and one of the 26 timesteps `ddim_timesteps(50)[:26]`, the digit sampler's steps 0 to 25, then noises the canvas to it. These settings are fixed, and no variation is tried after the unseen split is read.

## How a canvas is scored

**The judged window is the digit sampler's steps 14 to 25** (timesteps 712 to 480). These keep as much of the canvas as SDXL's steps 0 to 15, where the push acts ([the V2 review](../../abstract-checkers/05-does-it-learn/review.md)). The digit sampler's own steps 0 to 15 (timesteps 999 to 693) keep at most 8.7% of the canvas ([Loop 10's ideas page](../../score-space/03-ideas.md), assumption A2), so judging there would set a harder test than SDXL poses. The owner chose this window on 2026-11-24, before any W4 run. Steps 0 to 15 are read as well and reported second.

Each test canvas is read at each of the 26 steps with 4 noise draws, and its checker score is averaged over the draws. At each step, the threshold that maximises balanced accuracy is chosen on the even-indexed half of each `seen_test` pair. The seen score is read on the odd-indexed half. The same thresholds are applied once to all of `unseen_test`.

## The bars

The numbers are the ones step V2 uses, and they are constants in `2-classifier-guidance/scripts/noise_text_critic/w4_train.py`.

| Result | What it takes |
|---|---|
| Support | On `unseen_test` over steps 14 to 25, balanced accuracy is at least 0.75, within 0.05 of the seen score, and at least 0.75 on `apart` against `wrong` |
| Null | The overfit test passes and the frozen model misses any of those three |
| Inconclusive | The overfit test fails, or the model scores below 0.55 on the seen half at every step, because then it has not learned the task |

**Control.** V1's layout-only rule scores 0.5 on `apart` against `wrong`, so a model above 0.75 there is reading which digits are present.

**What would surprise.** A pass on `apart` against `wrong` on unseen digits. The image encoder has never seen a 7 and the loss has never named one, so a pass would mean CLIP's text vectors carry enough about digit shapes for the image side to line up with them.

## The order of work

1. **Smoke test.** 20 updates on the 1% data.
2. **Overfit.** Memorise 16 fixed canvases, with at least 15 given their true caption.
3. **1% of the data.** 1,000 updates, scored on the 1% seen split.
4. **Full training.** 8,000 updates, logged to W&B project `prime_lab/poe-repair-animals-compose`.
5. **Read `unseen_test` once**, and write `review.md` with seen and unseen accuracy by step, canvases beside each number, and the SDXL cost.

## The cost of an SDXL version

`review.md` gives a cost estimate built from this step's measured time per update, NoisyCLIP's printed training settings (10 epochs at batch 128, [its note](../01-reading/noisyclip.md)), and the time to generate its latents on this cluster. If that version looks worth running, this step files an `ask` with the cost, what it buys and a cheaper default, and starts nothing without the owner's yes.

## What this cannot say

**Digits are not photographs.** A 32 by 64 canvas has ten concepts and clean shapes, so a pass here does not show that an SDXL version would transfer, and a miss does not show that it would fail.

**It reads and does not steer.** This step has no sampler in it. A model that reads unseen digits may still have a gradient that does not help a push.

**It tests three of the five properties at most.** The canvas can express count, the right digits and whether they are apart. It cannot express position, size or binding.
