# A classifier trained on the together-against-apart ratio

The owner chose to test this as a third arm of board question 108's first experiment (board step Z9,
Loop 12). The reviewer read it on 2026-11-25 and gave "fix"; the fixes are built in below
([the reviewer's section](../reviewer/2026-11-24-review.md), "The ratio classifier, 2026-11-25"). It
is an idea, not a result.

## Where it comes from

For one distribution, with every quantity taken at the same noise level $t$, Bayes' rule gives

$$\nabla_x \log p_t(x \mid A, B) = \underbrace{s_A + s_B - s_\varnothing}_{\text{what the product computes}} + \underbrace{\nabla_x \log \frac{p_t(A, B \mid x)}{p_t(A \mid x)\, p_t(B \mid x)}}_{\text{what the product drops}}$$

The constant $p(A)p(B)/p(A,B)$ vanishes under the gradient wherever $p(A,B) > 0$. CoInD's penalty is
the size of the second term, up to the factor $\sqrt{1-\bar\alpha_t}$ between a noise prediction and a
score, so CoInD and DCFM train it to zero; this project names it the interaction term $r_t$.

**The term itself is published.** CFGComposer writes the same decomposition (it calls the second term
the copula score, Eq. 9, p. 3) and adds it back while sampling, with frozen experts, from a small
jointly trained model, for combinations in its training data. No paper the project has read estimates
it with a classifier on noisy pictures, and none aims it at combinations no data holder saw. That narrow
claim is what is open.

## What the idea does not get for free

- **On a combination no holder saw, the data's ratio is zero.** There $p(A,B) = 0$, so
  $p_t(A, B \mid x) = 0$ at every $x$. A classifier that learns the data faithfully learns a large
  negative log ratio for the pair and pushes the picture away from it. The interaction at an unseen pair
  is not learnable from data at all; it can only come from an assumed structure.
- **The push is not zero on pictures that are already right.** Noising alone makes two independent
  concepts dependent given the noisy picture. The project's exact toy shows a ratio push pointing along
  the sampler's own gap even where the product already composes, with cosine 0.85 to 0.96 from step 190
  to 90 (`2-classifier-guidance/report/classifier-guided-poe/a1-exact-toy-classifier-push.md`). So
  spoiling pictures that are already right has to be counted, as Z8 counted it.
- **Two data holders add a second gap.** The identity needs $s_A$, $s_B$ and $s_\varnothing$ to be
  scores of one distribution. With two holders, each expert's score comes from its own data, so the
  missing piece also holds $\nabla_x \log p_{\text{pooled}}(x \mid A) / p_k(x \mid A)$ per expert, which
  a ratio classifier does not estimate.
- **It carries to other models only as a heuristic.** The classifier reads the noisy picture, and with a
  guidance weight above 1 the sampled score is not a posterior score, so the ratio at weight 1 is not the
  matching term.
- **DCFM's minimum share (0.19 on Colored MNIST) counts main effects only.** It is the number of joint
  observations a model with no interaction needs, not the point where the interaction becomes learnable.

## The estimator to try first

One small network on noisy pictures with three heads: a digit head, a colour head, and an interaction
head whose score for a pair is a rank-$r$ product of a digit embedding and a colour embedding read from
the picture. Its pair softmax runs over seen pairs only, so an unseen pair's score never receives a
gradient of its own and is set by the embeddings alone. The push uses the interaction head's gradient
only. This is the one design in which a pair's absence cannot be learned, so it is the one that can
reach an unseen pair.

Not used: classifying real triples against triples with B shuffled (it counts B's guidance twice and
learns that unseen pairs are fake), and a few pictures showing the combinations (on Colored MNIST,
recolouring a digit makes any pair, so it hands over the answer).

## How it is tested

- Each score in the identity names the network that supplies it, guidance weights are 1, and scores are
  converted to velocity units if the experts are flow models.
- The interaction head must first reach seen combinations it never trained on, before any unseen one
  counts.
- Head accuracy is printed per sampling step.
- Controls beside the ratio arm: a push from the digit and colour heads alone at the same length, a
  random push, a classifier trained on shuffled labels, a single model trained on the pooled data, and a
  count of pictures already right that the push breaks.
- Scores are judged inside the run; DCFM's printed numbers (11.41 novel-combination FID for DCFM-A,
  25.19 for its finished-expert baselines, Table 1, p. 7) are context.

## The experiment it joins

Question 108, experiment 1, from `artifacts/notes/progress/gaps/04-exploiting-dcfm.md` (lead 1):
Colored MNIST built to DCFM's description, two data holders, finished experts with no retraining. Arms:
the plain product, the product with the checker push, and the product with the interaction-head push.
