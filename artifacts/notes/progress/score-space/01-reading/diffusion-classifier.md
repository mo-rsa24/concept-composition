# Your Diffusion Model is Secretly a Zero-Shot Classifier

arXiv 2303.16203 (v3), read from the full PDF (18 pages, appendix included). No venue is printed on the PDF.

## What it does
The paper turns a conditional diffusion model into a classifier with no training. For each candidate class it noises the image many times and averages the noise-prediction error, then picks the class with the lowest error (p. 3, Algorithm 1). With Stable Diffusion 2.0 it runs zero-shot on eight datasets and on Winoground. With DiT-XL/2 it runs as an ordinary ImageNet classifier (p. 5, p. 6).

## What it reads
The scored thing is the squared error between the true noise and the conditional noise prediction, $\|\epsilon - \epsilon_\theta(x_t, c)\|^2$, at a noisy latent $x_t = \sqrt{\bar\alpha_t}\,x + \sqrt{1-\bar\alpha_t}\,\epsilon$ built from the clean image (p. 3, Algorithm 1; p. 4, Eq. 7). For Stable Diffusion the clean image is resized to 512 by 512 and encoded to a 64 by 64 by 4 latent, and the error is measured in that latent space (p. 16). The condition $c$ is a text embedding of a prompt such as "a photo of a {class name}" for Stable Diffusion, or a class index for DiT (p. 2, Figure 1; p. 3). Timesteps are drawn uniformly from 1 to 1000 (p. 3), and Winoground uses 1000 evenly spaced timesteps (p. 16). Per dataset, the squared $\ell_2$ error may be swapped for an $\ell_1$ error (p. 5, p. 14). No internal features are read by the method itself; only the "SD Features" baseline reads mid-layer U-Net features of size 8 by 8 by 1024 at $t = 100$ with an empty prompt (p. 5, p. 17).

## What it trains
Nothing is trained by the method (p. 4). Two things are trained elsewhere in the paper:

- The baselines. "Synthetic SD Data" trains a ResNet-50 on 10,000 generated images per class. "SD Features" trains a classifier on the mid-layer U-Net features with real labels. The main text calls it a ResNet-50 (p. 5), while the appendix describes a modified ResNet-18 (p. 17, Table 11).
- One small fine-tune in Appendix D. Stable Diffusion was fine-tuned on 175 Ragdoll and Birman cat images. The ordinary noise-prediction loss was kept and a term was added that raises the error $\|\epsilon - \epsilon_\theta(x_t, c_j)\|^2$ for the wrong class $c_j$. Accuracy on those two classes rose from 45% to 85% (p. 16). The weight of the extra term and the training length are not given.

## How noise predictions are combined or compared
The class posterior uses the ELBO in place of $\log p(x \mid c)$ (p. 3, Eq. 6):

$$
p_\theta(c_i \mid x) = \frac{\exp\{-\mathbb{E}_{t,\epsilon}[\|\epsilon - \epsilon_\theta(x_t, c_i)\|^2]\}}{\sum_j \exp\{-\mathbb{E}_{t,\epsilon}[\|\epsilon - \epsilon_\theta(x_t, c_j)\|^2]\}}
$$

Each expectation is a Monte Carlo average over $N$ pairs $(t_i, \epsilon_i)$ with $t_i$ uniform on 1 to 1000 and $\epsilon_i \sim \mathcal{N}(0, I)$ (p. 3 to 4, Eq. 7):

$$
\frac{1}{N}\sum_{i=1}^{N} \left\| \epsilon_i - \epsilon_\theta\!\left(\sqrt{\bar\alpha_{t_i}}\,x + \sqrt{1-\bar\alpha_{t_i}}\,\epsilon_i,\; c_j\right)\right\|^2
$$

The paper rewrites the posterior so that only differences of errors between classes matter (p. 4, Eq. 8):

$$
p_\theta(c_i \mid x) = \frac{1}{\sum_j \exp\{\mathbb{E}_{t,\epsilon}[\|\epsilon - \epsilon_\theta(x_t, c_i)\|^2 - \|\epsilon - \epsilon_\theta(x_t, c_j)\|^2]\}}
$$

The same fixed set $S = \{(t_i, \epsilon_i)\}$ is reused for every class, which the authors compare to a paired difference test (p. 4). The timestep weighting $w_t$ is set to 1 (uniform). Using the variational-bound weighting instead lowers DiT ImageNet accuracy at 256 resolution from 77.5% to 71.6% (p. 3; p. 9, Table 5). The number of evaluations per class is set per dataset by an adaptive schedule that prunes classes in stages. For Pets, each class gets 25 evaluations, the 5 best are kept, and those get 250 in total (p. 5; p. 13, Table 6). Winoground uses 1000 per image-caption pair (p. 16 to 17).

## When adding scores or embeddings is valid
This paper does not address it. Scores for two conditions are never added. The closest test combines a conditional and an unconditional prediction with classifier-free guidance, $\tilde\epsilon = (1+w)\,\epsilon_\theta(x_t, c) - w\,\epsilon_\theta(x_t)$, and scores the error of $\tilde\epsilon$. On Pets, $w = 0$ (no guidance) gives the best accuracy (p. 18, Eq. 10 and Figure 8).

## What it says about the noise and the seed
The paper treats noise variance as the main obstacle. A plain Monte Carlo estimate "even using thousands of samples is not precise enough to distinguish classes reliably" (p. 4). Figure 2 shows, for one Great Pyrenees image and 4 fixed noise draws, that the error for a fixed prompt "varies wildly across the specific ϵi used", while the error difference between the two prompts is much more consistent for each draw (p. 4). No variance number is given; the evidence is the plot.

The fix is to share $(t_i, \epsilon_i)$ across classes and to average many draws (p. 4). Every attempt to narrow the noise distribution failed. On a 10% subset of Pets, accuracy was 41.3% with $\epsilon = 0$, 49.9% with a normal truncated to [−1, 1], 81.5% truncated to [−2.5, 2.5], 86.9% with norm rescaling, and 87.5% with plain $\mathcal{N}(0, I)$ (p. 18, Table 12). Cropping the error map also lowered accuracy (p. 18, Figure 9).

## Which timesteps carry the signal
Measured on Pets with one timestep per class. Accuracy peaks at intermediate noise, and the paper names $t = 500$ (p. 4, Figure 3). Read from the figure: about 46% near $t$ = 400 to 500, about 8% at $t$ near 0, and about 13% at $t$ = 1000, on 37 classes (read from image). As more evaluations are added, uniform timesteps over 1 to 1000 scale best, reaching about 83% on Pets near 500 trials per class. Strategies restricted to $t$ = 0 only or $t$ = 1000 only stay under 30% (p. 5, Figure 4, read from image). The authors attribute the poor scaling of a narrow band such as 475 to 525 to bias in the ELBO estimate (p. 5).

## Unseen concepts and composition
This paper does not test held-out concepts. The zero-shot test sets are largely outside Stable Diffusion's filtered training distribution: after the SD 2.0 filters, only 0 to 3% of CIFAR10, Pets, Flowers, STL10, ImageNet and ObjectNet images would remain (p. 6; p. 7, Table 2). This is a distribution shift, and the classes are not held out from the generator.

Composition is tested on Winoground, where two captions use the same words in a different order (p. 7). The text score averaged over swap types is 38.5 for Diffusion Classifier, 33.0 for OpenCLIP ViT-H/14 and 28.2 for CLIP ViT-L/14, with chance at 25.0 (p. 7, Table 3). Relation swaps (verbs, adjectives, prepositions) score 29.2 against 26.6 for OpenCLIP. Since Stable Diffusion 2.0 uses the OpenCLIP ViT-H/14 text encoder, the authors attribute the gain to better cross-modal binding of concepts to images (p. 7). Appendix D shows one failure: Stable Diffusion inverts and reconstructs Birman and Ragdoll prompts almost identically, so the classifier also confuses them (p. 15).

## Numbers

| Metric (scale, direction) | This method | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| Winoground text score, average over swap types (%, higher better, chance 25.0) | 38.5 | 33.0 (OpenCLIP ViT-H/14) | Winoground | SD 2.0 | p. 7, Table 3 |
| Winoground text score, Relation swaps (%, higher better) | 29.2 | 26.6 (OpenCLIP ViT-H/14) | Winoground | SD 2.0 | p. 7, Table 3 |
| Zero-shot top-1 accuracy (%, higher better) | 87.3 | 94.6 (OpenCLIP ViT-H/14); 75.9 (SD Features, trained) | Oxford-IIIT Pets | SD 2.0 | p. 6, Table 1 |
| Zero-shot top-1 accuracy (%, higher better) | 61.4 | 76.8 (OpenCLIP ViT-H/14); 58.2 (CLIP ResNet-50) | ImageNet, 2000 test images | SD 2.0 | p. 6, Table 1 |
| Supervised top-1 accuracy (%, higher better) | 79.1 | 81.2 (ViT-B/16); 77.9 (ViT-L/32) | ImageNet, 10,000 val images | DiT-XL/2 at 512 | p. 8, Table 4 |
| ImageNet-A top-1 accuracy (%, higher better) | 30.2 | 20.8 (ViT-B/16) | ImageNet-A | DiT-XL/2 at 512 | p. 8, Table 4 |
| Pets accuracy against noise distribution (%, higher better) | 87.5 with $\mathcal{N}(0,I)$ | 41.3 with $\epsilon = 0$ | 10% subset of Pets | SD 2.0 | p. 18, Table 12 |
| Time to classify one image (seconds, lower better) | 18 (Pets) to 1000 (ImageNet) | not compared | per dataset | SD 2.0, RTX 3090 | p. 14, Table 7 |

## What it suggests for this loop
- For a score-path encoder (inferred): a raw error or a raw prediction varies far more with the noise draw than the difference between two conditions under the same draw (p. 4, Figure 2). The brief's $d_t(c)$ is already a difference under a shared $x_t$, which is the quantity this paper found stable. A contrastive encoder should see conditions compared under one shared noise draw, and paths for different conditions should share $(x_t, \epsilon)$ when they are compared.
- For the timesteps (inferred): class information on Pets peaks at middle noise and is weakest at the two ends (p. 4, Figure 3). If steps 0 to 15 of the digit sampler sit at the high-noise end, the encoder may be reading the least informative part of the path. A check is to score each step's $d_t$ alone for digit identity before training on the stack.
- For the two arithmetic tests (inferred): this paper gives a training-free baseline for "sum lands on joint". Under a shared set of $(t, \epsilon)$, compare $\|\epsilon - \epsilon_{\text{PoE}}\|^2$ with $\|\epsilon - \epsilon_\theta(x_t, \text{joint})\|^2$ and with the errors for single-digit captions on a canvas holding both digits. If the product already ranks the joint caption first by this ELBO proxy, a trained encoder has to beat that ranking to add anything.
- For the separate-experts paper (inferred): the ELBO-difference score is a ready, untrained judge of whether a guided product sample matches the expected caption, and it binds better than CLIP on Winoground. Its cost is high: 18 seconds per Pets image at 512 resolution (p. 5).
- For the CO3 paper (inferred): guided predictions made classification worse at every $w > 0$ (p. 18, Figure 8). A likelihood-style score for judging CO3's corrector should be computed from unguided predictions.

## Closer work it cites
- Score-based generative classifiers, Zimmermann, Schott, Song, Dunn and Klindt, arXiv 2110.00473, cited p. 2 (reference [84], p. 12). It builds classifiers from the likelihoods of a class-conditional score model, the same family as this paper's ELBO classifier, on small image datasets.
