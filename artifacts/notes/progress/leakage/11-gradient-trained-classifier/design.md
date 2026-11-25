# Design: does a checker trained to push along the experts' own direction reach pairs it never saw? (step Z11)

## What this asks

Arm 2's checker on Colored MNIST lifted the plain product of two finished experts from 46.9% to 76.0% right on the 50 combinations neither holder saw, but a push that only names the digit and the colour did as well, 78.4% ([Z9's review](../09-dcfm-colored-mnist/review.md)). This step trains the same checker with one change: a penalty that aligns the direction its gradient pushes in with a direction worked out from the experts, the technique of Ganz, Kawar and Elad (arXiv 2207.11378, Eq. 4 to 8, pp. 4 and 5). It asks whether that checker reaches pairs it never trained on better than the same checker without the penalty and better than the names-only push. The owner approved it as board step Z11 at about 2.3 GPU-hours (request of 2026-11-25, `requests.md`), built with the reviewer's fixes ("The gradient-trained classifier, 2026-11-25" in `reviewer/2026-11-24-review.md`).

## Words this uses

- **The push**: $-\text{strength}\cdot\sigma_t\,\nabla_x f$, added to the composed noise prediction, so it moves that prediction along $-\nabla_x f$ (strength 10, capped at 0.1 of the prediction's length, Paper 1's kept settings, as arm 2).
- **The target** at a seen pair $(d,c)$: in noise units, $\epsilon_k(x_t, t \mid d, c) - \bar\epsilon_\varnothing(x_t, t)$, the prediction of the one expert $k$ that saw the pair, asked with both labels, minus the mean of both experts with no labels, at guidance weight 1. It is Ganz's class-minus-unconditional score with the pair in place of the class.
- **The penalty**: $1 - \cos(\nabla_x f(x_t; d, c),\ -[\epsilon_k - \bar\epsilon_\varnothing])$. The minus sign makes the push move the noise prediction toward the expert that saw the pair, because the push moves it along $-\nabla_x f$.

## The checker and its training

Arm 2's head (`CheckerHead` in `2-classifier-guidance/scripts/leakage/z9_push.py`): both frozen experts' decoder features, asked with no labels, and the asked digit and colour as two embeddings added. Trained on the **40 seen pairs not held back** (`split.json`, `ratio_held_back` excluded), on the same positives and negatives as arm 2 (real coloured digits of the asked pair against the same digits in a wrong colour or a wrong digit in the asked colour, plus the experts' own drawn pictures judged right or wrong), noised to random steps, for 3,000 steps of batch 128, seed 31.

$$L = L_{\text{BCE}} + \lambda \cdot \frac{1}{|P|} \sum_{(d,c) \in P} w \cdot \big[1 - \cos(\nabla_x f(x_t; d, c),\ -(\epsilon_k - \bar\epsilon_\varnothing))\big]$$

- **Several pairs per picture.** $P$ is the picture's own asked pair and 3 other training pairs drawn at random, each target computed from the expert that saw that pair.
- **Weight by target length.** $w$ is the target's length over the mean target length at that step in the batch, so late steps, where the target is small and mostly noise, count less.
- **$\lambda = 1$, fixed now**, the middle of the range Ganz report (0 to 2, Table 1, p. 7), before any cosine is seen; it is not tuned on any held-back number.

**The twins, same 40 pairs, steps, batch and seed:**

| Twin | What changes | What it rules out |
|---|---|---|
| $\lambda = 0$ | no penalty | the one-change comparison: what the penalty does |
| Shuffled target | each pair's target taken from a fixed other training pair's expert prediction | that any gain is from smoother gradients, not the pair's own direction |

## Printed first, before the gate is judged

The $\lambda = 0$ twin's held-back cosine: on the 10 held-back pairs' product trajectories at seeds 2,000 to 2,015 (arm 4's states, `z9/arm4/traj-held.pt`), the median cosine between the push's change to the noise prediction and the target. It is the number the penalty must beat. The same cosine on training pairs at seeds 2,000 to 2,015 is printed beside it and is never quoted as evidence. The cosine between the push's change and the dropped term $r_t$ (arm 4's target) is reported as a reading that decides nothing. Two random directions in 3,072 numbers have a cosine spread of about 0.018.

## Reports while training, and a rule that can only stop

Added at the owner's request (2026-11-25, through the manager loop), before the first report. Every 500 training steps each checker prints its losses and its held-back median cosine: the push's change to the noise prediction against the target, on a fixed 1,000 of the held-back states at seeds 2,000 to 2,015. The checkers train in the order $\lambda = 0$ twin, penalised checker, shuffled twin.

**The futility rule.** The $\lambda = 0$ twin's held-back median cosine at its last step is the line. If at step 1,500 of 3,000 the penalised checker's held-back median cosine is not above that line, the penalised run stops, the shuffled twin is not trained, and Z11 is reported as stopped for futility, with no gate and no unseen-pair run. The rule can only stop a run: passing it decides nothing, and the gate below stays the judge.

## The gate, on the 10 held-back seen pairs, 1,000 matched pictures

1. The held-back median cosine of the penalised checker is at least 0.10 above the $\lambda = 0$ twin's.
2. Its digit-and-colour accuracy is at least 5 points above the $\lambda = 0$ twin's and at least 5 points above the shuffled twin's, each with an exact one-sided McNemar p < 0.05.

Both must hold, or the unseen-pair run is reported as not reached.

## The bar on the 50 unseen combinations, 5,000 matched pictures

Support when the penalised push's accuracy is at least 5 points above the $\lambda = 0$ twin's and at least 5 points above control a's digit-and-colour push scaled to this arm's length, each with an exact one-sided McNemar p < 0.05; its novel FID is below the $\lambda = 0$ twin's; and the seen pictures the plain product draws right that it breaks are no more than the random push's count plus 50.

| Control | What it does | What comes back |
|---|---|---|
| $\lambda = 0$ twin | arm 2's checker retrained on the same 40 pairs, no penalty | the one-change comparison |
| Shuffled-target twin | the penalty with each pair's target from another training pair | whether the pair's own direction matters, beyond smoother gradients |
| Control a at this arm's length | Z9's separate digit and colour push, scaled per step and picture to the penalised push's length | whether the penalised push carries more than the two attributes |
| Random push at this arm's length | a fresh random direction at that length | whether any change of that size helps |
| Already-right broken | seen pictures the plain product draws right that a push turns wrong | whether the push spoils what the product already draws |

## What it cannot reach, said before it runs

At a combination no holder saw, no expert's two-label prediction exists, so the penalty never touches it; the checker's output there is set by its digit embedding plus its colour embedding. Like arms 3 and 4, it can only carry what it learned on seen pairs through that additive structure. On this split the target on seen pairs is mostly the product averaging over a holder that lacks the picture, a federated effect with no counterpart on SDXL, so a pass shows a supplied direction can be trained into a classifier and survive to unseen pairs, not that SDXL's interaction can.

## Runs and cost

**On RunPod RTX 4090s, if the owner's pods take it** (the owner's request of 2026-11-25, through the manager loop): every comparison row is then drawn on the pods, so all rows share one card type. The checker's training pictures and the frozen experts come from Z9 unchanged, as inputs; the one baseline Z9 drew, the plain product's seen pictures for the already-right count, is drawn again on the pod (`--plain-here`). Nothing from a pod is compared with a picture drawn on a cluster RTX 3090.



`bigbatch`, RTX 3090, excluding `mscluster[44-45,48,50-51,57,61,65,68,71-76,82-83]`; up to two jobs at once beside Z10 (the owner lifted the one-job rule for Z10 and Z11, inside the cap of six). Training the penalised checker and the shuffled twin, about 0.5 GPU-hours each, and the $\lambda = 0$ twin, about 0.2; the gate's pictures about 0.15; about 1.3 to the gate. The unseen run with its controls and the already-right count, about 1.0 more. About 2.3 GPU-hours in all, approved. W&B project `prime_lab/poe-repair-animals-compose`. No API tokens.
