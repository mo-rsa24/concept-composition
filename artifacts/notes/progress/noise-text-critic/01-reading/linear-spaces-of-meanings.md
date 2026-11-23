# Linear Spaces of Meanings: Compositional Structures in Vision-Language Models

arXiv 2302.14383 (v3), read from the full PDF (18 pages, appendix included).

## What it does

The paper asks when the CLIP embeddings of a factored set of strings, such as colour by object, equal a sum of one vector per factor value, which it calls "ideal words" (p. 1, p. 3). It proves this holds exactly when the factors are conditionally independent given any image under CLIP's own probability model (p. 5). It then uses averaged embeddings for zero-shot classification, debiasing, retrieval and Stable Diffusion prompt edits (p. 6 to 8).

## What the model or score reads

Experiments use the HuggingFace CLIP ViT-L/14 checkpoint, on clean images and on the final text embedding (p. 5). For the Stable Diffusion 2.1 pictures, the text representation is the whole token sequence before the final projection layer, treated as one long vector (p. 6, p. 8). No noisy input is read.

## What it trained on

Nothing is trained. All experiments use a pre-trained CLIP encoder, and ideal words are weighted averages of its embeddings (p. 3, p. 5). A randomly initialized text encoder is used as a control (p. 16).

## How its score or gradient enters

No gradient enters a sampler. The paper changes the prompt embedding that conditions Stable Diffusion, and leaves the diffusion process unchanged (p. 8). The approximation used is

$$
u_{(z_1,\dots,z_k)} \approx u_0 + u_{z_1} + \dots + u_{z_k}, \qquad u_0 = \sum_z \beta_z u_z, \quad u_{z_i} = \frac{1}{\alpha_{z_i}} \sum_{z' : z_i' = z_i} \beta_{z'} u_{z'} - u_0
$$

where $\alpha_{z_i}$ are positive weights summing to 1 within a factor and $\beta_z = \prod_i \alpha_{z_i}$ (Proposition 4, p. 3). Ideal words are not normalized after they are computed (p. 17).

## When adding scores is valid

**The geometric condition (Lemma 2, p. 3).** A set of embeddings is decomposable if and only if the difference $u_z - u_{z'}$ does not depend on the components $z$ and $z'$ share. A decomposable set spans at most $1 + \sum_i (n_i - 1)$ dimensions.

**The probabilistic condition (Proposition 7 and Corollary 8, p. 5; proof p. 12 to 13).** CLIP is read as $\log p(x, y) = u_x^\top v_y + c$ (p. 4). If the image embeddings span $\mathbb R^d$, the text embeddings are decomposable if and only if

$$
p(x(z), y) = q_0(y)\, q_1(z_1, y) \cdots q_k(z_k, y)
$$

which means the factors are conditionally independent given any image.

**Weaker conditions (Proposition 9, p. 5).** If the most likely value of each factor does not depend on the other factors for an image (mode-disentangled), ideal words give the same prediction for that image as the full embeddings. If the full ranking of each factor is independent of the others for images covering the unit sphere (order-disentangled), the embeddings are decomposable.

**Where it weakens.** Idioms raise the residual: the string set {black, green} by {friday, day} has mean squared distance to its best decomposable fit $D$ = 0.225, against 0.053 for {black, green} by {car, bike} (p. 17, Figure 6, read from the image). Harder context factors such as "underwater" and "in a volcano" produce poor pictures (p. 18).

**Syntax alone produces decomposable structure.** A randomly initialized encoder also gives roughly decomposable embeddings when the strings share repeated substrings (p. 16, p. 17). Its residual over mean pair distance is 0.40 on MIT-States and 0.71 on UT Zappos, against 0.29 and 0.28 for trained CLIP, computed from Tables 6 and 7 (p. 16) (inferred).

### Does a difference of two CLIP sentence embeddings isolate binding or left-right order?

**Judgement (inferred): no.** The paper's results do not predict this, and applied to these sentences they predict that the additive part of the embedding cannot carry either property. The paper never tests binding across two subjects, word order or spatial relations. Its factors are attribute by object, object by context and label by spurious attribute (p. 5 to 7).

**The argument for binding (inferred, from Lemma 2).** Take the template "a [c1] [n1] and a [c2] [n2]" with four slot factors. If the embeddings are decomposable over these slots, the noun terms cancel in the swap:

$$
u(\text{red cat, blue dog}) - u(\text{blue cat, red dog}) = (a_{\text{red}} - a_{\text{blue}}) + (a'_{\text{blue}} - a'_{\text{red}}) =: \Delta
$$

$\Delta$ is the same vector for any noun pair, so it holds no information about which noun has which colour. An image of a red cat and a blue dog needs $\Delta^\top v > 0$ for this word order. For the other word order it needs "a blue dog and a red cat" to beat "a red dog and a blue cat", and that difference equals $-\Delta$. Both cannot hold for any image embedding $v$.

**The same argument for left-right order (inferred).** Over the slots (first noun, relation, second noun), the difference "cat left of dog" minus "dog left of cat" equals the negative of "dog right of cat" minus "cat right of dog". An image with the cat on the left needs both to be positive, which is impossible.

**What can carry the property (inferred).** Only the part of the embedding that decomposition discards can carry it. Averaging over both word orders removes every slot term and leaves that part alone:

$$
w_{\text{bind}} = \big[u(\text{a red cat and a blue dog}) + u(\text{a blue dog and a red cat})\big] - \big[u(\text{a blue cat and a red dog}) + u(\text{a red dog and a blue cat})\big]
$$

$$
w_{\text{order}} = \big[u(\text{cat left of dog}) + u(\text{dog right of cat})\big] - \big[u(\text{dog left of cat}) + u(\text{cat right of dog})\big]
$$

Both vectors are exactly zero when the embeddings are decomposable over the slots. How large they are in real CLIP is unmeasured in this paper.

**The limit of this argument (inferred).** With a single template, one sentence pair can still rank one image correctly, because $\Delta^\top v > 0$ is allowed alone. Such a pass says nothing about binding, since $\Delta$ does not depend on the nouns. A test that uses one word order cannot tell the two cases apart.

**The and-score (inferred).** With factors {cat, no cat} by {dog, no dog}, the and-score is $(u_{\text{cat and dog}} - u_{\text{cat}} - u_{\text{dog}} + u_{\text{empty}})^\top v$. When $S$ is CLIP's inner product, the paper's model (p. 4) makes this the log odds ratio of those four captions given the image, up to CLIP's temperature. It is zero for every image if the four embeddings are decomposable, so the and-score reads only the interaction part of the text embedding.

## Which of the five properties it bears on

**Counting.** Not tested.

**Attribute binding.** Not tested across two subjects. MIT-States and UT Zappos score one attribute on one object (p. 5, p. 7).

**Spatial relation.** Not tested.

**Position and size.** Not tested. "big" and "small" appear only as a factor in an embedding plot (p. 6).

**Prompt match.** Partly. Zero-shot classification and retrieval measure which label or caption fits an image, on clean images (p. 6, p. 7).

## Unseen concepts

This paper does not test held-out concepts. CLIP is used zero-shot on attribute and object labels, and the paper drops the seen and unseen pair split as not relevant to that setting (p. 15). One qualitative test mixes ideal words computed from disjoint sets of colours and objects, and the limits of that transfer are left to future work (p. 17, p. 18).

## Numbers

| Metric (scale, direction) | This method | Best baseline | Dataset | Model | Page and table |
|---|---|---|---|---|---|
| Attribute and object pair accuracy, % (higher is better) | 11.5 | 10.0 real words; 7.7 all 28175 pair labels | MIT-States | CLIP ViT-L/14 | p. 6, p. 7, Table 1 |
| Same | 10.8 | 12.4 all pair labels; 8.4 real words | UT Zappos | CLIP ViT-L/14 | p. 7, Table 1 |
| Gap between average and worst-group accuracy, percentage points (lower is better) | 1.6 | 10.1 Orth-Cali | CelebA | CLIP ViT-L/14 | p. 7, Table 2 |
| Same | 23.3 | 15.7 Orth-Cali | Waterbird | CLIP ViT-L/14 | p. 7, Table 2 |
| Mean reciprocal rank, scale as printed in the table (higher is better) | 37.0 ± 1.1 | 28.4 ± 0.7 PALAVRA | DeepFashion2 | CLIP ViT-L/14 | p. 7, Table 3 |
| Decomposition residual $D$, mean squared distance to the best decomposable set (lower is better), trained encoder | 0.23 ± 0.05 MIT-States; 0.16 ± 0.04 UT Zappos | 0.43, 0.51 real words | as named | CLIP ViT-L/14 | p. 16, Table 6 |
| Same, randomly initialized encoder | 0.04 ± 0.02; 0.10 ± 0.02 | 0.16, 0.22 real words | as named | random text encoder | p. 16, Table 7 |

## What to copy, what to avoid

- Before training anything, compute $w_{\text{bind}}$ and $w_{\text{order}}$ on the frozen text encoder for our sentence pairs. This is text-only and needs no GPU. If they are near zero relative to the sentence differences, a noise-aware image tower paired with that frozen text tower has no text direction to follow for these properties (inferred).
- Score each property in both word orders on the same image. A checker that passes one order can be reading the slot term $\Delta$ (inferred).
- Report the and-score as an interaction measure. It is the one score in our set that cancels the additive part by construction (inferred).
- Compare any decomposition result against a random encoder. Repeated-substring sentences such as our swapped pairs decompose even without training (p. 16) (inferred).
- Avoid reading the paper's classification gains as evidence for binding. All of them involve one attribute on one object (p. 5, p. 7) (inferred).

## Closer work it cites

None.
