# Concept Algebra for (Score-Based) Text-Controlled Generative Models

arXiv 2302.03693 (v6), read from the full PDF (22 pages, appendix included).

## What it does

The paper treats the score function of a text-to-image model as a representation of the prompt and proves that, under a condition called causal separability, separate concepts occupy separate subspaces of that score (p. 5). It estimates a concept's subspace from score differences between prompts and edits only that part of the score during sampling (p. 6, p. 7). The examples use Stable Diffusion (p. 1).

## What the model or score reads

The method reads the model's noise prediction $\epsilon_\theta(y_t, t \mid x)$ at the current noisy sample $y_t$, for the empty prompt, the original prompt, the new prompt and the prompts that span the concept (p. 16). The paper treats the noise prediction and the score as equivalent up to a time-varying constant (p. 16). The Stable Diffusion version and the resolution are not printed.

## What it trained on

Nothing is trained for the method itself, because it uses the pretrained model's scores (p. 5). The one exception is the toy example, where a DreamBooth fine-tune (learning rate 5e-6, 800 steps) is deliberately overfitted (p. 19).

## How its score or gradient enters

The edited score replaces the prompt's score at every step:

$$
s_{\text{edit}}(y) \leftarrow (I - \mathrm{proj}_Z(y))\, s[x_{\text{orig}}](y) + \mathrm{proj}_Z(y)\, s[x_{\text{new}}](y)
$$

$\mathrm{proj}_Z(y)$ projects onto the concept subspace at the current $y$ (p. 7). The subspace is estimated from $k+1$ prompts that differ on the concept and agree on everything else:

$$
\hat R_Z(y) := \mathrm{span}\big(\{s[x_i](y) - s[x_0](y) : i = 1, \dots, k\}\big)
$$

(p. 6). In Algorithm 1 the edit is applied to the conditional noise prediction, then classifier-free guidance with scale $w$ is applied, at every DDPM step from $T$ to 1 (p. 16). The projection is $QQ^\top$, where $Q$ holds the top left singular vectors of the prompt-difference matrix up to a variance-explained threshold (p. 16). A second option replaces the projection with a binary pixel mask computed from a score difference, thresholded at 0.1 in the toy example (p. 17, p. 19). The style edit uses the original prompt for the first 20% of timesteps (p. 20). No CLIP or classifier gradient enters.

## When adding scores is valid

**The condition: causal separability (Definition 3.2, p. 5).** The output must split as $Y = g(Y_Z, Y_W, \xi)$ with $g$ invertible and differentiable, and

$$
p(y_Z, y_W, \xi \mid z, w) = p(\xi)\, p(y_Z \mid z)\, p(y_W \mid w)
$$

So the visual part for concept $Z$ can be generated without knowing the value of $W$.

**What it buys (Proposition 3.1, p. 5; proof p. 17).** Under this condition the centred score adds over the two concepts for any product concept distribution $Q_Z Q_W$:

$$
\bar s[Q_Z \times Q_W] = \bar s_Z[Q_Z] + \bar s_W[Q_W]
$$

(p. 6). The uncentred score carries one extra shared term $s_0(y)$ (p. 17, eq. 17).

**Isolating a concept (Proposition 4.1, p. 6).** A difference of scores spans $R_Z(y)$ only if the prompts share the same distribution $Q_W$ over the other concept (p. 6, p. 7). A concept with $L$ categories has a subspace of dimension at most $L-1$ (Proposition 5.1, p. 7). A concept made of independent finer parts has dimension at most $\sum_k (L_k - 1)$ (Proposition 5.2, p. 7).

**Across noise levels (Appendix A, p. 16).** The paper requires causal separability for every noised $Y_t$, $t = 0, \dots, T$. It states this as an assumption and does not show that separability of $Y$ implies it.

**When it fails (p. 5, p. 21, p. 22).** Sex and species fail because the visual signs of sex differ by species, for example antlers on male deer. Figure 8 shows that a sex direction built from "a buck on the grass" and "a doe on the grass" does not transfer sex to a nurse.

## Which of the five properties it bears on

**Counting.** Not tested.

**Attribute binding.** Not tested. Edits change one attribute of one subject, such as dog to cat or sunny to rainy (p. 8).

**Spatial relation.** Not tested.

**Position and size.** Not tested. A score-difference mask locates a subject for the toy edit (p. 10, p. 19), and no position or size is measured.

**Prompt match.** Partly. Ten human raters ranked outputs for adherence to style and content on 49 unusual content and style pairs (p. 8, p. 9). This measures style plus content only.

## Unseen concepts

This paper does not test held-out concepts. Its stress test uses 49 content and style pairs chosen to co-occur rarely in training data, and the training set is not checked for them (p. 8, p. 20).

## Numbers

| Metric (scale, direction) | This method | Best baseline | Dataset | Model | Page and table |
|---|---|---|---|---|---|
| Average proportion from a human ranking of faithfulness to style and content (0 to 1, higher is better; the paper does not define the proportion further), standard error in brackets | 0.476 (0.023) | 0.164 (0.017) concept composition; 0.162 (0.017) direct prompting | 49 content and style pairs, 10 raters | Stable Diffusion, version not printed | p. 8, p. 9, Table 1 |
| Same scale, share of "all bad" | 0.198 (0.018) | not applicable | same | same | p. 9, Table 1 |

## What to copy, what to avoid

- Copy the prompt design rule. Our right and wrong sentences should differ on the one property and agree on everything else, because a difference only isolates a concept when the other concepts share one distribution (p. 6) (inferred).
- Treat binding and left-right order as likely failures of causal separability. Which colour a cat has depends on which subject is being drawn, the same kind of interaction as sex and species (p. 5) (inferred).
- Do not read cat and dog as two separable concepts under this theory. The paper uses dog versus cat as values of one binary concept (p. 8), so the separability result says nothing about drawing both at once (inferred).
- A cheap test on the U-Net side is available: compute $\epsilon(\text{right sentence}) - \epsilon(\text{wrong sentence})$ during sampling. It costs two extra U-Net calls per step and needs no training, and would show whether the frozen SDXL holds a usable direction before training a noise-aware model (inferred).
- Avoid citing the 0.476 as evidence on any of our five properties. It measures style and content preference (p. 9) (inferred).

## Closer work it cites

None.
