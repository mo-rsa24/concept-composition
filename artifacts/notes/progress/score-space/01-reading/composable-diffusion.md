# Compositional Visual Generation with Composable Diffusion Models

arXiv 2206.01714 (v6), read from the full PDF (30 pages, appendix included). No venue is printed on the PDF.

## What it does
The paper treats each conditional diffusion model as an energy-based model and samples from a product of concepts by adding the conditional noise-prediction differences to the unconditional prediction (p. 6, p. 7). It defines two operators, conjunction (AND) and negation (NOT), applied at sampling time with no extra training (p. 3, p. 8). It tests them on CLEVR object positions, Relational CLEVR relations, FFHQ face attributes and pre-trained GLIDE text prompts (p. 8, p. 10).

## What it reads
At every sampling step the same noisy image $x_t$ goes to the model once per concept and once unconditionally, giving the set $\{\epsilon_\theta(x_t,t\mid c_1),\dots,\epsilon_\theta(x_t,t\mid c_n)\}$ plus $\epsilon_\theta(x_t,t)$ (Fig. 2, p. 5; Algorithm 1, p. 8). These are noise predictions in pixel space. The condition enters as an embedding fused with the time embedding: `nn.Embedding(7, d)` for FFHQ attributes and `nn.Linear(2, d)` for a CLEVR $(x, y)$ coordinate (p. 21). For text it uses the small pre-trained GLIDE, one prompt per concept (p. 22). No internal features are read. Every step from $T$ to 1 is composed (Algorithm 1, p. 8).

## What it trains
Each domain gets its own conditional diffusion model trained with the classifier-free recipe: a label is replaced by a null label with 10% probability so the same network also gives the unconditional prediction (p. 21). Data and labels (p. 8, p. 9):

- CLEVR: 30,000 images at 128 × 128, each with 1 to 5 objects and one label, the 2D coordinate of one object.
- Relational CLEVR: 50,000 images at 128 × 128, each with one relation label between two objects.
- FFHQ: 51,067 images labelled by pre-trained classifiers for smile, gender and glasses.

Training uses 1,000 diffusion steps, a cosine noise schedule and AdamW with $\beta_1 = 0.9$, $\beta_2 = 0.999$; CLEVR trains for 7 days (750,000 iterations) and FFHQ for 2 days (250,000 iterations) on one V100 (p. 22, p. 23). The paper does not print its training loss. The code base is improved DDPM and GLIDE (p. 21), so the loss is presumably the standard one (inferred):

$$
L(\theta) = \mathbb{E}_{t,x_0,c,\epsilon}\,\big\|\epsilon - \epsilon_\theta(x_t, t \mid c)\big\|^2, \qquad c \leftarrow \varnothing \text{ with probability } 0.1
$$

For evaluation it also trains binary classifiers on real images: CLEVR position (24,000 train, 6,000 validation, 99.05% validation accuracy), Relational CLEVR (40,000 / 10,000, 99.80%), and three CelebA-HQ attribute classifiers (24,000 / 6,000; 95.01%, 99.20%, 97.49% for smiling, glasses, gender) (p. 19, p. 20).

## How noise predictions are combined or compared
Conjunction (Eq. 11, p. 7):

$$
\hat\epsilon(x_t,t) = \epsilon_\theta(x_t,t) + \sum_{i=1}^{n} w_i\big(\epsilon_\theta(x_t,t\mid c_i) - \epsilon_\theta(x_t,t)\big)
$$

$\epsilon_\theta(x_t,t)$ is the unconditional prediction, $\epsilon_\theta(x_t,t\mid c_i)$ the prediction under concept $c_i$, and $w_i$ a "temperature scaling on concept $c_i$" (p. 7). With one concept this is classifier-free guidance, and the appendix states $w > 1$ there (p. 7, p. 23). With all $w_i = 1$ it is the brief's $\epsilon_{\text{PoE}} = \epsilon_\varnothing + d_t(a) + d_t(b)$ (inferred). Sampling then uses $x_{t-1} \sim \mathcal N(x_t - \hat\epsilon, \sigma_t^2 I)$ (Eq. 12, p. 7).

Negation (Eq. 15, p. 8):

$$
\hat\epsilon(x_t,t) = \epsilon_\theta(x_t,t) + w\big(\epsilon_\theta(x_t,t\mid c_i) - \epsilon_\theta(x_t,t\mid \tilde c_j)\big)
$$

where $\tilde c_j$ is the negated concept and $w$ sets its strength.

The same $x_t$ is fed to every condition at a step, so the noise state is shared across conditions (Algorithm 1, p. 8). One trajectory per image; no averaging over noise draws. There is no weighting over timesteps; every step uses the same rule. The $w_i$ values used in the experiments and the number of sampling steps are not stated.

## When adding scores or embeddings is valid
The conjunction is derived under the assumption that "the concepts are conditionally independent given x" (p. 7, again p. 23). Under that assumption $p(x\mid c_1,\dots,c_n) \propto p(x)\prod_i p(x\mid c_i)/p(x)$ (Eq. 10, p. 7) and the gradient of its log is the sum in Eq. 11. The paper notes that a learned $\epsilon_\theta$ "may not be conservative, and thus does not correspond to a valid probability density", and dismisses this by citing Salimans and Ho (p. 6). It does not address that the sum of noised conditional scores differs from the score of the noised product at $t > 0$. The listed failures are where adding breaks in practice: the base model does not know a concept ("person" in GLIDE), attributes get swapped ("a red bear" for "a bear in a red forest"), and the composition fuses objects into one hybrid ("bird-shape and flower-color object", "dog-fur and sofa-shape object"), which "usually happen[s] when the objects are in the center of the images" (p. 13, Fig. 6 p. 14). "A dog" AND "the sky" gives a dog-shaped cloud where GLIDE given the single sentence gives a dog under the sky (p. 19). Composing diffusion models trained on different datasets had "limited success"; all composed models are instances of the same model (p. 13).

## What it says about the noise and the seed
Not addressed. Each method generates 5,000 images per test setting and FID uses 5,000 images rather than 50,000 (p. 9). No error bars, repeated seeds or seed-variance figures are reported.

## Which timesteps carry the signal
Not measured.

## Unseen concepts and composition
This paper does not test held-out concepts. It holds out combinations: each model is trained on one concept per image (one object position, one relation, one attribute) and tested on 2 and 3 composed concepts (p. 9, p. 12). On CLEVR it composes up to 7 objects (Fig. 4, p. 11) and 8 objects (Fig. 14, p. 29) where training images hold 1 to 5 objects (p. 8). Accuracy falls as components are added: 86.42% for 1 component, 59.20% for 2, 31.36% for 3 on CLEVR (Table 1, p. 11). Relations: two composed relational descriptions per image, 21.84% for 2 and 2.80% for 3 (Table 2, p. 12). Binding: the paper claims composed prompts bind attributes that DALL-E 2 confuses (p. 1) and shows attribute confusion as a failure mode (p. 13). Counting is not tested.

## Numbers

| Metric (scale, direction) | This method | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| Accuracy: % of 5,000 images where the classifier finds every concept, higher is better, 1 component | 86.42 | 70.54 (EBM) | CLEVR positions | diffusion trained from scratch (GLIDE code) | p. 11, Table 1 |
| Same, 2 components | 59.20 | 28.22 (EBM) | CLEVR | same | p. 11, Table 1 |
| Same, 3 components | 31.36 | 7.34 (EBM) | CLEVR | same | p. 11, Table 1 |
| FID on 5,000 images, lower is better, 3 components | 10.51 | 19.01 (StyleGAN2) | CLEVR | same | p. 11, Table 1 |
| Accuracy %, 1 / 2 / 3 components | 60.40 / 21.84 / 2.80 | 78.14 / 24.16 / 4.26 (EBM) | Relational CLEVR | same | p. 12, Table 2 |
| FID, 1 / 2 / 3 components | 29.06 / 29.82 / 26.11 | 17.61 (GLIDE) / 28.56 (GLIDE) / 31.30 (StyleGAN2) | Relational CLEVR | same | p. 12, Table 2 |
| Accuracy %, 1 / 2 / 3 components | 99.26 / 92.68 / 68.86 | 98.74 (EBM) / 95.66 (LACE) / 80.88 (LACE) | FFHQ attributes | same | p. 20, Table 3 |

Text-prompt results with GLIDE and Stable Diffusion are qualitative only (Fig. 3 p. 10, Fig. 1 p. 2).

## What it suggests for this loop
- Score-path encoder (inferred): the paper's $\hat\epsilon$ is exactly the product path the encoder would embed, so its own hybrids ("dog-shaped cloud", "bird-shape and flower-color object") are examples of a product path whose picture is one fused object. An encoder trained on joint paths could be checked first on whether it puts such product paths near the joint caption or near a single concept.
- Sum lands on joint (inferred): the formula is derived only under conditional independence of concepts given $x$. Retrieval of the joint caption from the product path should fail precisely for pairs where the concepts interact, which is where the brief's interaction term is nonzero. Pairs where the paper's composition succeeds (object plus background, "a horse" AND "a yellow flower field") are the natural positive control.
- Gap predicts the term (inferred): the paper offers no measurement of the joint minus product gap. It supplies the baseline arithmetic only.
- Separate-experts paper (inferred): the limitation that diffusion models "trained on different datasets" compose with "limited success" (p. 13) is the risk that paper faces. The paper's suggested remedy is a conservative score field from EBMs (p. 13).
- CO3 paper (inferred): the paper is the baseline sampler; it gives no corrector. Its failure list (center-placed objects fuse) gives a test set where any CO3 improvement should show.

## Closer work it cites
- Salimans and Ho, "Should EBMs model the energy or the score?", ICLR 2021 EBM workshop, no arXiv id printed (p. 6, ref. 44 p. 17). It trains a scalar energy on $x_t$ whose gradient is the score field, which is a learned scalar readout of the score; the paper cites it as evidence that a non-conservative $\epsilon_\theta$ is not a problem.

No cited paper trains an encoder or classifier on noise predictions or trajectories, or aligns diffusion internals to text.
