## Paper 1: composing experts that never met

**The problem**

Two diffusion models are trained on disjoint data, so neither has seen an image holding both concepts, and no joint
prompt or joint dataset exists. Each model's score pulls the whole image toward its own kind of picture, so their plain
product has no reason to place two separate objects. The question is whether the two trained models, plus a small
helper trained without joint images, can draw both concepts as separate objects.

**The setting**

Allowed: the two trained models, samples drawn from them, and a helper trained on those samples. Excluded: any real image
holding both concepts, any prompt naming both, and any model trained on the union of the two datasets.

**The proposed method**

Each model denoises only its own region, and a trained checker guides the combined sample:

$$
\hat\epsilon_t(x_t) = \mathrm{merge}\big(\epsilon^A_t(x_t|_{R_A}),\ \epsilon^B_t(x_t|_{R_B})\big) - \lambda_t\,\sigma_t\,\nabla_{x_t}\log C_t(\text{both concepts present}\mid x_t)
$$

$R_A$ and $R_B$ are the two regions, $\epsilon^A_t$ and $\epsilon^B_t$ are the two models' noise predictions at step $t$,
$C_t$ is the checker, and $\lambda_t$ is the strength of its push.

**The evidence today**

The strongest count is on the void list: 1,200 clean pairs of 1,280 by the digit-reading judge, on 20 MNIST digit pairs
kept out of tuning, with each model on a fixed half of a 32 by 64 pixel canvas. It stays out of any claim until the owner
decides it. Without fixed halves, the best run on one shared canvas draws 121 of 320 (38%) on five tuning pairs, against
a bar of 288. With each model free to draw on either side, two pushes draw 4 of 320. Every result so far is in pixel
space on MNIST or a toy. No shared latent space has been built.

**The closest published work**

Projective Composition composes location-conditioned models that saw only single-object CLEVR images, and 98 of 100
hand-counted images hold all 5 objects. MultiDiffusion reconciles one model across fixed regions. Diffusion Domain
Expansion trains a coordinator on one model's outputs. SuperDiff composes two models trained on disjoint CIFAR-10 halves
and samples their mixture. CoInD trains one model with a conditional-independence penalty.

**What it is measured on**

The five Paper 1 rows in the [benchmark spec](benchmark-spec.md#the-10-benchmarks-to-beat): CLEVR at 5 positions, Shapes3D,
CMNIST AND, CelebA and the disjoint CIFAR-10 halves. Whether the CLEVR target stands is an open question for the owner.

