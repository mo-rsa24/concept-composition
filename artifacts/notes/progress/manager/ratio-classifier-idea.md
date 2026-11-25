# A classifier trained on the together-against-apart ratio

The owner chose to test this as a third arm of board question 108's first experiment, after the
reviewer reads it. It is an idea, not a result.

## Where it comes from

A product of experts composes two concepts A and B by adding scores. By Bayes' rule the exact score
of the joint is that sum plus one more term:

$$\nabla_x \log p(x \mid A, B) = \underbrace{s_A + s_B - s_\varnothing}_{\text{what the product computes}} + \underbrace{\nabla_x \log \frac{p(A, B \mid x)}{p(A \mid x)\, p(B \mid x)}}_{\text{what the product drops}}$$

The constants $p(A)p(B)/p(A,B)$ vanish under the gradient. CoInD's penalty
$\|[\epsilon(c_A) + \epsilon(c_B) - \epsilon(c_\varnothing)] - \epsilon(c_{A,B})\|^2$ is exactly the
size of the second term, so CoInD and DCFM train it to zero, which this project names the interaction
term $r_t$ (`3-understanding-compositionality/artifacts/notes/coind-conditional-independence-loss/README.md`).

The proposal: supply that term at sampling time, with the experts left frozen, from a classifier on
noisy pictures that estimates the ratio, and add its gradient to the product.

## Why it differs from the parked push

The kept push (Loop 6's readout checker) learned whether a picture is a clean pair. On known
failures it fixed no plain-PoE hybrid and spoiled pictures CO3 had right (Z8, `leakage/08-first-correction-test/review.md`).
The ratio is 1 wherever the two concepts do not affect each other, so its gradient is zero there, and
it is defined on the guided prediction, so it carries to any model with classifier-free guidance.

## The open concern for the reviewer

On combinations no data holder saw, there are no training pictures of A and B together. A joint
classifier trained only on seen combinations has no label for the unseen ones, so whether the ratio is
learnable where it matters is the first question. Candidate estimators:

1. Classify noisy $(x, A, B)$ triples drawn from the data against triples with B shuffled, a standard
   density-ratio estimator, then read it off at the unseen pair.
2. Separate heads for $p(A \mid x)$, $p(B \mid x)$ and a "both present as separate things" head trained
   on seen pairs, relying on the last head to generalise across pairs.
3. A small set of pictures that do show the combinations (the inclusive-captions idea in
   `4-inclusive-captions/`), used only to train the ratio while the experts stay frozen.

DCFM's minimum share of seen combinations (0.19 on Colored MNIST) is read here as the point below which
the interaction cannot be learned from the data alone.

## The experiment it joins

Question 108, experiment 1, from `artifacts/notes/progress/gaps/04-exploiting-dcfm.md` (lead 1):
Colored MNIST built to DCFM's description, two data holders, finished experts with no retraining,
novel-combination FID against DCFM-A's 11.41 and the finished-expert baselines' 25.19 (DCFM Table 1,
p. 7), plus a digit-and-colour classifier's accuracy. Arms: the plain product, the product with the
checker push, and the product with the ratio push.
