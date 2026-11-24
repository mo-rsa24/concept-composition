# Text-to-Image Diffusion Models are Zero-Shot Classifiers

arXiv 2303.15233, read from the full PDF (17 pages, appendix included). The PDF is marked "Preprint" (p. 1); no venue is printed.

## What it does

The paper turns a text-to-image diffusion model into a classifier: noise the image, denoise it once under each class prompt, and pick the class whose prompt gives the lowest weighted denoising error (p. 2, p. 4). It applies this to Imagen and Stable Diffusion v1.4 and compares them with CLIP ViT-L/14 on 13 classification datasets, a shape-versus-texture test and synthetic attribute binding tasks (p. 2, p. 6 to p. 9). Two tricks cut the cost: reusing the same noise and timestep across classes, and dropping unlikely classes early with a paired t-test (p. 4, p. 5).

## What it reads

The scored quantity is the squared error between the clean input and the model's prediction of it, $\lVert x - \tilde{x}_\theta(x_t, \phi(y_k), t)\rVert_2^2$, where $x_t$ is the input noised by the forward process at a random $t \sim U([0,1])$ and $\phi(y_k)$ is the class name in a prompt template such as "A photo of a $y_k$" (p. 3, p. 4). For SD, $x$ and $\tilde{x}_\theta$ are VAE latents of a 512 × 512 image; for Imagen they are raw pixels of a 64 × 64 image from the base model only (p. 4, p. 6). The super-resolution Imagen models were tried and dropped because they barely respond to the prompt (p. 5, p. 17). Internal features and text embeddings are not read.

## What it trains

Nothing is trained for the reported zero-shot results. An optional learned timestep weighting, used only in the appendix because it needs labels, fits a 20-feature linear softmax model over 20 timestep buckets (p. 4, p. 14):

$$
b_i(x, y_k) = \mathbb{E}_{\epsilon,\ t \sim U[0.05i,\ 0.05(i+1)]} \lVert x - \tilde{x}_\theta(x_t, \phi(y_k), t)\rVert_2^2, \qquad p_v(y = y_k \mid x) = \frac{\exp\big(-\sum_{i=0}^{19} v^i b_i(x, y_k)\big)}{\sum_{y_j} \exp\big(-\sum_{i=0}^{19} v^i b_i(x, y_j)\big)}
$$

It is fit by maximum likelihood, with each bucket estimated from about 100 Monte Carlo samples (p. 14). Calibration (temperature scaling, Platt scaling) is fit on a small held-out set (p. 16).

## How noise predictions are combined or compared

The classifier is (p. 4, Eq. 2):

$$
\tilde{y} = \arg\min_{y_k \in [y_K]} \mathbb{E}_{\epsilon, t}\Big[ w_t \big\lVert x - \tilde{x}_\theta\big(x_t, \phi(y_k), t\big)\big\rVert_2^2 \Big]
$$

$x$ is the clean image or latent, $x_t$ its noised version, $\phi(y_k)$ the prompt for class $k$, $w_t$ a timestep weight. The expectation is a Monte Carlo average: sample $t$, sample $x_t \sim q(x_t \mid x_0)$, score every remaining class on that same $x_t$ and $t$ (p. 4). The weight is $w_t = \exp(-7t)$, hand-picked to look like the learned weighting on CIFAR-100 (p. 4). Each class gets at least 20 and at most 2000 scores, and a class is dropped when a paired t-test against the current best gives $p < 2 \times 10^{-3}$ (p. 5, p. 6). Classes are compared by mean weighted error, so the only thing that differs between two classes' scores on a shared sample is the prompt (p. 4).

## When adding scores or embeddings is valid

This paper does not address it. It treats the weighted denoising error as a proxy for $-\log p_\theta(x \mid y_k)$ and says a monotone weighting keeps it a likelihood-based objective (p. 4). It never adds or subtracts predictions or embeddings across prompts.

## What it says about the noise and the seed

The paper states the diffusion loss "has fairly high variance" (p. 4). Sharing $t$ and $x_t$ across classes improves sample efficiency by roughly 100 times, and pruning adds another 8 to 10 times, on CIFAR-100 with Imagen (p. 5, Fig. 2). Accuracy varies by less than 1% across random seeds (p. 7, Table 1 caption); across random data splits the spread is about ±0.7% for Imagen, ±0.6% for SD and ±0.4% for CLIP, and the diffusion models vary more "due to the inherent randomness in noising images" (p. 15). Four CIFAR-100 prompt templates gave accuracies within 1.5% of each other (p. 6). Raw score softmax probabilities are poorly calibrated (expected calibration error 0.12) because the expectation estimates are noisy; the number of model calls the pruning needed is better calibrated (0.04) (p. 16).

## Which timesteps carry the signal

Measured indirectly through the weighting. With $t \in [0, 1]$ and $t = 0$ the clean image, $w_t = \exp(-7t)$ puts most weight at low noise; at $t = 1$ the weight is about $e^{-7} \approx 0.0009$ of the weight at $t = 0$ (inferred arithmetic from p. 4). On Caltech101, Imagen accuracy is 56.1% with SNR weighting, 62.0% with the VDM weighting, 68.9% with the heuristic and 70.2% learned; SD is 71.4%, 71.9%, 73.0% and 73.1% (p. 15, Table 4). The authors explain Imagen's spread by SNR and VDM weighting putting more weight on the lowest noise levels, where "Imagen tends to be an inaccurate classifier" (p. 15). Learned weights are "pretty similar across datasets" and transfer between them (p. 14). The paper does not plot the learned weights. For binding, the authors hypothesise (unmeasured) that Imagen picks the larger object's colour because colour over a large region helps denoising at high noise levels (p. 9).

## Unseen concepts and composition

This paper does not test held-out concepts. The setting is zero-shot (no training on any task's labels), and the heuristic weighting was tuned on CIFAR-100 only and then applied to the other 12 datasets (p. 4). The compositional test is attribute binding on two-object CLEVR-style renders, 1000 examples per binary task (p. 8). Chance is 50% (inferred from the binary design and the 50% CLIP rows). Results in percent accuracy (p. 9, Table 3):

- Pair tasks, where both objects are described: Imagen reaches 100 (Shape,Color), 99 (Shape,Size) and 86 (Color,Size); SD 85, 63 and 59; CLIP 54, 52 and 48.
- Binding tasks, one object with attributes that could be swapped: Imagen 66 (Shape|Color) and 73 (Color|Shape); SD 65 and 59; CLIP 52 and 53.
- Every task involving position or size binding is near 50% for all three models, except Imagen's pair tasks.

The authors credit part of Imagen's lead to its T5 text encoder and to cross attention between text and image features, which CLIP lacks (p. 9).

## Numbers

| Metric (scale, direction) | Imagen 64×64 (this method) | Best baseline | Dataset | Page, table |
|---|---|---|---|---|
| Top-1 accuracy, % (higher better) | 84.3 | CLIP ViT-L/14 68.6 | CIFAR-100, 4096 examples | p. 7, Table 1 |
| Top-1 accuracy, % | 79.2 (SD v1.4: 19.1) | CLIP 74.3 | MNIST | p. 7, Table 1 |
| Top-1 accuracy, % | 62.7 (SD v1.4: 61.9) | CLIP 63.4 at 64 px, 75.1 at 224 px | ImageNet | p. 7, Table 1 |
| Shape accuracy, % (higher better) | 84.4 (SD v1.4: 72.5) | ViT-22B 68.7; CLIP 51.6 | Cue-Conflict | p. 7, Table 2 |
| Binding accuracy Color\|Shape, % (chance 50, inferred) | 73 (SD v1.4: 59) | CLIP 53 | synthetic CLEVR-style, 1000 examples | p. 9, Table 3 |
| Pair binding Shape,Color, % | 100 (SD v1.4: 85) | CLIP 54 | same | p. 9, Table 3 |
| Caltech101 accuracy by weighting, % | heuristic 68.9 | SNR ("simple") weighting 56.1 | Caltech101 | p. 15, Table 4 |
| Expected calibration error (lower better) | 0.04 (model-call count) | 0.12 (softmax of scores) | CIFAR-100 | p. 16, Fig. 5 |

## What it suggests for this loop

- For a score-path encoder (inferred): the paper's signal is a paired comparison of prompts on one shared noisy input, and pairing alone bought about 100 times fewer samples (p. 5). Score paths for different captions of one canvas and seed should be contrasted against each other as hard negatives, so the encoder cannot win by recognising the seed. This is the same trap the project's same-pair, cross-seed cosine of about 0.002 warns about.
- For a score-path encoder (inferred): the per-class error difference on a shared $x_t$ expands into inner products of prediction differences such as $d_t(a) - d_t(b)$ with the residual. The diffusion classifier is therefore a fixed, untrained readout of the same $d_t$ quantities the encoder will see. On the digit canvases it gives a no-training baseline for caption retrieval that the contrastive encoder must beat.
- For "sum lands on joint" (inferred): the classifier can be run directly on a product-of-experts canvas, scoring the joint caption against the single-digit captions and a swapped caption. This answers the retrieval question without any trained encoder and calibrates what accuracy the encoder should reach.
- For both arithmetic tests (inferred): this paper weights low noise most and found Imagen weakest at the very lowest noise. If the brief's steps 0 to 15 are the high-noise end of the sampler, they are the steps this paper down-weights by up to about 1000 times. The binding hypothesis on p. 9 points the other way for colour: if large-region colour is settled at high noise, an early-step score path could carry colour binding but miss shape.
- For the separate-experts paper (inferred): the same error-based classifier, applied to each expert's prediction, gives a training-free way to check whether a guided sample matches the expected answer, at a cost of tens of model calls per class (p. 5).
- For the CO3 paper (inferred): Table 3 is a ready protocol for testing binding in CO3 outputs, scoring the correct pair caption against a swapped one. Imagen beats chance only on pair and colour/shape binding, and SD loses positional binding entirely (p. 9).
- Relative to the brief (inferred): CLIP ViT-L/14 is at chance on every binding task here (p. 9). That measures the joint image-text match, so it does not contradict the brief's finding that CLIP's text encoder alone keeps word-noun binding. It does show that keeping the binding in the text vector is not enough when images and text are encoded separately.

## Closer work it cites

- Unleashing Text-to-Image Diffusion Models for Visual Perception, Zhao et al. 2023, arXiv 2303.02153 (p. 2, p. 13). Trains task heads on a text-to-image diffusion model's internal representations, aligned with text, for perception tasks.
- Denoising Pretraining for Semantic Segmentation, Brempong et al. 2022, CVPR, no arXiv id printed (p. 1, p. 10). Pretrains a segmentation encoder with a denoising objective.
