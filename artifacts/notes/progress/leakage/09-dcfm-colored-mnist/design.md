# Design: does a push on finished experts draw combinations no data holder saw? (step Z9)

## What this asks

DCFM (Morshed and Boddeti, arXiv 2606.10153) says experts trained on separate data holders and composed without retraining fail on combinations no holder saw: on Colored MNIST its two finished-expert baselines print novel FID 25.19 and 29.71, against 11.41 for DCFM-A, which retrains its experts after exchanging them (Table 1, p. 7). This step asks whether Paper 1's checker push, added while sampling to two finished experts that are never retrained, draws the unseen combinations better than their plain product does. It is the first experiment of board question 108, which the owner answered on 2026-11-25 ("run experiment 1 first"), with three arms the owner chose; the request is the last s12 line of `requests.md`, and the reading is [the DCFM gap note](../../gaps/04-exploiting-dcfm.md), lead 1.

## Words this uses

- **Data holder**: one silo of training pictures; its expert never sees another holder's pictures.
- **Seen and novel combinations**: a digit-and-colour pair some holder has pictures of, and a pair no holder has.
- **Finished expert**: a model trained once on its holder's pictures and then only sampled from.
- **The plain product**: DCFM's composition rule (p. 3, Eq. 5), the unconditional prediction plus one difference per attribute.
- **The checker push**: a small classifier's gradient added to the composed prediction at each sampling step, Paper 1's correction.

## The data, built to DCFM's description

**Pictures.** Each MNIST digit, 28 by 28, is padded to 32 by 32 and its stroke coloured with one of ten fixed colours on a black background, three channels scaled to $[-1, 1]$. The ten colours are matplotlib's `tab10` palette. DCFM's palette and picture size are not in Loop 4's note of the paper, so this build matches its description (ten digits by ten colours, 100 combinations, half seen; p. 6) and not its exact pictures.

**The split.** 50 of the 100 combinations are seen, drawn once with a fixed seed (20261125) under the condition that every digit and every colour appears in some seen combination, which DCFM's split also requires (p. 3, Definition 3.1). The 50 seen combinations are dealt at random to two holders, 25 each. A holder's training stream draws one of its combinations uniformly and a random MNIST training digit of that class, coloured. The split is saved as `split.json` beside the experts and printed in the review. The draw gives holder 1 every digit but 6 and every colour, and holder 2 every digit and every colour but grey ([the combinations sheet](combinations.png): blue frames holder 1, red frames holder 2, unframed is novel).

## The experts

One per holder, the digit U-Net of Paper 1's MNIST work (`2-classifier-guidance/scripts/mnist/canvas_unet.py`) at 32 by 32 with three channels, conditioned on a digit and a colour through two embeddings that are added. Each label is replaced by "none" on 10% of training examples, independently, so the expert learns $p(x \mid d)$, $p(x \mid c)$ and $p(x)$ as well as $p(x \mid d, c)$, as DCFM's Stage I does with masked labels (p. 5). No independence penalty is added and nothing is exchanged: each expert is trained to completion on its own holder and frozen. Training length is fixed by a timed run before the real one: on an RTX 3090 a step of 128 pictures took 61.9 ms (job 95109), so each expert trains for 20,000 steps, 2.56 million pictures in about 21 minutes, which keeps the step under the 3-hour line with the pooled model of control d trained the same way.

## The three arms

Every arm samples the 50 novel combinations, 100 pictures each from seeds 0 to 99, with DDIM over 50 steps at sampler noise 1.0 (Paper 1's kept setting); seed $k$ gives every arm the same starting noise for the same combination.

| Arm | What is added to the prediction | Settings |
|---|---|---|
| 1. The plain product | nothing: $\bar\epsilon_\varnothing + g[(\bar\epsilon_d-\bar\epsilon_\varnothing)+(\bar\epsilon_c-\bar\epsilon_\varnothing)]$, each $\bar\epsilon_d$ or $\bar\epsilon_c$ the mean, over the holders whose pictures include that digit or colour, of their predictions with that label and the other set to none, and $\bar\epsilon_\varnothing$ the mean of both experts with both labels set to none (in this split holder 1 has no 6 and holder 2 no grey, so those two terms come from one expert) | $g = 1$, the Bayes composition of Paper 1's digit work |
| 2. The checker push | arm 1 plus Paper 1's push from a checker reading "digit $d$ in colour $c$" | strength 10, capped at 0.1 of the prediction's length per picture, every step (Paper 1's kept settings) |
| 3. The interaction push | arm 1 plus the gradient of the interaction head of the classifier below, the rebuilt estimator the reviewer asked for (`reviewer/2026-11-24-review.md`, "The ratio classifier, 2026-11-25"; [the manager's note](../../manager/ratio-classifier-idea.md)) | strength 10 and the cap of 0.1, as arm 2, so the two pushes differ only in direction |

**The checker.** Paper 1's pair-aware checker (`2-classifier-guidance/scripts/mnist/checker.py`): a small head on both frozen experts' decoder features, asked with labels set to none, plus the asked digit and colour, giving one log-odds that the picture becomes that digit in that colour. It trains on seen combinations only, never on a novel one: positives are real coloured training digits of a seen combination and the experts' own seen-combination samples that the digit-and-colour classifier calls right; negatives are the same digits in a wrong colour, a wrong digit in the asked colour, and the experts' samples called wrong. Its accuracy on held-back seen-combination pictures is reported.

## The interaction push, in full

**Why it is rebuilt.** On a combination no holder saw, the data's ratio $p(A,B \mid x)/(p(A \mid x)\,p(B \mid x))$ is zero, so a classifier that learns it faithfully pushes pictures away from the very pair asked for. The estimator below is the one form in which a pair's absence cannot be learned.

**Which network supplies each score.** In $\nabla_x \log p(x \mid d, c) = s_d + s_c - s_\varnothing + \nabla_x \log \frac{p(d,c \mid x)}{p(d \mid x)\,p(c \mid x)}$, the scores $s_d$, $s_c$ and $s_\varnothing$ come from the experts as arm 1 averages them, and the last term from the classifier's interaction head. Guidance weight is 1 throughout. The experts are noise-prediction models, not flow models, so no conversion to velocity units is needed; a score is $-\epsilon/\sigma_t$ and the push is added to the noise prediction as $-\text{strength}\cdot\sigma_t\,\nabla_x(\cdot)$.

**The classifier.** One small network on noisy pictures, given the step, with three heads: a digit head and a colour head (softmax over 10 each), and an interaction head whose score for pair $(d,c)$ is $u_d^\top W(x,t)\, v_c$, a rank-$r$ product ($r = 4$) of a learned digit embedding $u_d$, a learned colour embedding $v_c$ and a picture-dependent $r \times r$ matrix. The interaction head's softmax runs over the training pairs only, so a pair outside them never gets a gradient of its own and is scored through the embeddings alone. It trains on real coloured training digits of its training pairs, noised to random steps. The push uses the interaction head's gradient only, $\nabla_x\, u_d^\top W(x,t)\, v_c$.

**First, seen pairs it never trained on.** 10 of the 50 seen combinations, 5 from each holder, drawn with seed 20261126 and saved in `split.json` as `ratio_held_back`, are kept out of the classifier's training. Before any novel-combination picture counts, the interaction push runs on those 10 pairs, 100 pictures each, against the random push of the same length. **It passes** when its digit-and-colour accuracy is at least 5 points above the random push's with an exact one-sided McNemar p < 0.05. If it fails, the novel-combination run of arm 3 is reported as not reached, and nothing about unseen pairs is claimed. The classifier's accuracy on each head is printed at every sampling step on these pictures.

**Controls beside arm 3** (each on the 50 novel combinations, 100 pictures each, the same seeds):

| Control | What it is | What it rules out |
|---|---|---|
| a. Heads-only push | the gradient of $\log p(d \mid x) + \log p(c \mid x)$ from the same classifier's digit and colour heads, scaled at every step to the interaction push's length | that any gain comes from naming the digit and colour, not from the interaction |
| b. Random push | a fresh random direction at every step at the interaction push's length | that any push of that size helps |
| c. Shuffled-label classifier | the same classifier trained with the pair labels shuffled across pictures, its interaction push | that the push works without learning anything about pairs |
| d. One pooled model | a single U-Net trained on both holders' pictures together, sampled by its own plain product | how far one model with all the data gets, the federated reference |
| e. Already-right pictures broken | on the 50 seen combinations, pictures the plain product draws right that the interaction push (and, beside it, the random push) turns wrong | that the push spoils what already works |

**Arm 3's bar**, in addition to the bar below against arm 1: its digit-and-colour accuracy is at least 5 points above control a's with an exact one-sided McNemar p < 0.05, and its count of already-right pictures broken (control e) is no more than the random push's plus 50 (1% of the 5,000). Failing the first means any gain is from the main-effect heads, not the interaction.

## Arm 4: a model of the term the product drops

The owner added this arm (request of 2026-11-25, `requests.md`), and the reviewer's section "The fourth arm, 2026-11-25" in `reviewer/2026-11-24-review.md` set seven fixes, all below. It is the Colored MNIST counterpart of the ICLR adapter: a small model that predicts the term the product leaves out, with both experts frozen.

**The target.** On a seen pair $(d,c)$, exactly one holder's expert saw it. The target is
$$r_t(x_t, d, c) = \epsilon_k(x_t, t \mid d, c) - \epsilon_{\text{product}}(x_t, t \mid d, c),$$
the prediction of that expert $k$ asked with both labels, minus arm 1's composed prediction, both at guidance weight 1.

**The model.** A small U-Net of the experts' design (`ColourUNet`, 32 channels), given the noisy picture, the step, and the digit and the colour as two separate inputs (two embeddings, added), never one embedding per pair, so a pair it never trained on is reached through its parts. It outputs a 3 by 32 by 32 correction. Sampling with arm 4 uses $\epsilon_{\text{product}} + \hat r_t$, at weight 1.

**Training.** Only on the 40 seen pairs not held back for arm 3, and only along the product's own sampling trajectories: arm 1's sampler run on each of those pairs from seeds 2,000 to 2,015, every step's $x_t$ kept with its target. Nothing is trained on the holders' pictures. 10,000 steps of batch 256, mean squared error.

**The twin.** The same model trained with the combinations shuffled: each trajectory's target is paired with a different training pair's digit and colour, drawn once with a fixed seed.

**The baseline: ask the experts directly.** The two experts asked with both labels at once, their predictions averaged: $\tfrac12[\epsilon_1(x_t,t \mid d,c) + \epsilon_2(x_t,t \mid d,c)]$. It tests whether the experts' own label embeddings already reach a pair, with no added model.

**The gate, on arm 3's 10 held-back seen pairs, 100 pictures each.** First a fit check: on those pairs' product trajectories (seeds 2,000 to 2,015), the model's prediction must explain at least a quarter of the target's variance, $1 - \lVert \hat r - r \rVert^2 / \lVert r \rVert^2 \ge 0.25$, averaged over steps. Then arm 4's digit-and-colour accuracy must be at least 5 points above arm 1's and at least 5 points above the twin's, each with an exact one-sided McNemar p < 0.05. If the gate fails, arm 4's unseen-pair run is reported as not reached.

**The bar on the 50 novel combinations, written before the run.** Arm 4 supports the idea when its accuracy is at least 5 points above arm 1's, above the twin's and above the ask-the-experts baseline's, each with an exact one-sided McNemar p < 0.05, and its novel FID is lower than arm 1's. Accuracy above arm 1 alone, with the twin or the baseline as good, is reported as a gain the model's knowledge of pairs does not explain.

**What it needs from the data holders.** The model is trained in one place on both experts' predictions, so both finished experts must be sent to one site: the same single exchange of models that DCFM-B makes (p. 6), and more than arms 1 to 3 need, which sample at one site from the experts as they are.

**Cost.** About 0.9 GPU-hours: the trajectories and their targets (40 pairs by 16 seeds by 50 steps), the model and its twin, and the sampling for the gate and the novel run. With it Z9 is about 3.3 GPU-hours, which the owner approved (request of 2026-11-25, `requests.md`).

## How pictures are scored

- **Novel FID**: Fréchet Inception distance between each arm's 5,000 pictures and 5,000 real pictures of the same 50 combinations (MNIST test digits, coloured, 100 per combination), with `pytorch-fid`'s Inception weights on 2,048 pooled features. **Seen FID** is computed the same way on the 50 seen combinations for completeness. A real-against-real FID (two disjoint halves of the reference) is reported as the floor.
- **Digit-and-colour accuracy**: a small classifier trained on real coloured MNIST training digits over all 100 combinations, two heads (digit, colour), never trained on a generated picture; a picture is right when both heads name the asked combination. Its accuracy on real coloured test digits of the novel combinations is reported and must be at least 95%, or the measure is invalid.
- **Pictures**: one sheet per arm with one picture per novel combination, sent as each arm lands.

DCFM's printed numbers are reported beside ours as context, marked partly matched: two holders against its ten, our palette and size, and our sample count where DCFM states none.

## The bar

The comparison is arm 2 (and later arm 3) against arm 1 on the 50 novel combinations, 5,000 matched pictures.

| Result | What it takes |
|---|---|
| Support | the push's digit-and-colour accuracy is at least 5 percentage points above the plain product's, an exact one-sided McNemar test on the 5,000 matched pictures gives p < 0.05, and its novel FID is lower than the plain product's |
| Labels only | the accuracy condition holds but novel FID is not lower |
| Null | the accuracy condition fails |
| Inconclusive | a control below fails |

## Controls

1. **A push of zero changes nothing.** With strength 0 the checker arm's sampler must give arm 1's pictures byte for byte, on one seed per combination of a 5-combination subset, in one process on one card.
2. **The judge is good enough.** The digit-and-colour classifier's accuracy on real novel-combination test digits is at least 95%.
3. **The experts are experts.** Each expert's own seen-combination samples (20 per combination) are called right by the classifier at least 80% of the time; otherwise the plain product's failure could be a weak expert rather than composition.

## Cost and what it buys

**Cost** (inferred until the timed run): building the data, seconds on a CPU; two experts, about 0.3 GPU-hours each; the judge, under 0.1; the checker and its sampled training pictures, about 0.3; sampling and FID for arms 1 and 2, about 0.3. About 1.5 GPU-hours before arm 3. Arm 3 adds the classifier and its shuffled twin (about 0.2), the pooled model (about 0.3) and the extra sampling (about 0.3), about 0.8, for about 2.3 GPU-hours in all, inside the request's 2 to 4 and under the 3-hour line; if the timed run puts it over 3, the rest goes to the owner first. Runs go to `bigbatch`, excluding `mscluster[44-45,48,50-51,57,61,65,68,71-76,82-83]`, one job at a time, logged to W&B project `prime_lab/poe-repair-animals-compose`. No API tokens.

**Buys:** whether a push on finished, never-retrained experts reaches unseen combinations, the claim DCFM's retraining rests on. A support sends question 108 to experiment 2 (below DCFM's minimum share of seen combinations); a null says the plain product's failure on Colored MNIST is not one this push repairs.

## The ladder

Outputs go under `/datasets/mmolefe/poe_repair_min/loop-12/z9/<rung>/`.

1. **Data and split**: `split.json` and a sheet of every combination, real pictures, seen ones marked.
2. **Classifier**: accuracy on real test digits, per combination.
3. **Experts**: a timed run, then both trained; control 3.
4. **Arm 1**: the plain product on the novel and seen combinations; FIDs and accuracy.
5. **Checker and arm 2**: control 1, then arm 2; the bar.
6. **Arm 3**: the classifier and its shuffled twin; the held-back seen pairs first; then the novel combinations with controls a to e.
7. **Arm 4**: the trajectories and targets, the model and its twin, the fit check and the held-back gate, then the novel combinations beside the ask-the-experts baseline.

## What this cannot say

Two holders, not ten, so no row matches DCFM's table exactly. One split draw and one push setting. Nothing here bears on separate objects in one picture, Paper 1's own setting.
