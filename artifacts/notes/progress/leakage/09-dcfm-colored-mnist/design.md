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

**The split.** 50 of the 100 combinations are seen, drawn once with a fixed seed (20261125) under the condition that every digit and every colour appears in some seen combination, which DCFM's split also requires (p. 3, Definition 3.1). The 50 seen combinations are dealt at random to two holders, 25 each. A holder's training stream draws one of its combinations uniformly and a random MNIST training digit of that class, coloured. The split is saved as `split.json` beside the experts and printed in the review.

## The experts

One per holder, the digit U-Net of Paper 1's MNIST work (`2-classifier-guidance/scripts/mnist/canvas_unet.py`) at 32 by 32 with three channels, conditioned on a digit and a colour through two embeddings that are added. Each label is replaced by "none" on 10% of training examples, independently, so the expert learns $p(x \mid d)$, $p(x \mid c)$ and $p(x)$ as well as $p(x \mid d, c)$, as DCFM's Stage I does with masked labels (p. 5). No independence penalty is added and nothing is exchanged: each expert is trained to completion on its own holder and frozen. Training length is fixed by a timed run before the real one and stated in the review.

## The three arms

Every arm samples the 50 novel combinations, 100 pictures each from seeds 0 to 99, with DDIM over 50 steps at sampler noise 1.0 (Paper 1's kept setting); seed $k$ gives every arm the same starting noise for the same combination.

| Arm | What is added to the prediction | Settings |
|---|---|---|
| 1. The plain product | nothing: $\bar\epsilon_\varnothing + g[(\bar\epsilon_d-\bar\epsilon_\varnothing)+(\bar\epsilon_c-\bar\epsilon_\varnothing)]$, each $\bar\epsilon$ the mean of the two experts' predictions with that label and the other set to none | $g = 1$, the Bayes composition of Paper 1's digit work |
| 2. The checker push | arm 1 plus Paper 1's push from a checker reading "digit $d$ in colour $c$" | strength 10, capped at 0.1 of the prediction's length per picture, every step (Paper 1's kept settings) |
| 3. The ratio push | arm 1 plus the gradient of a classifier on $p(A,B \mid x)/(p(A \mid x)\,p(B \mid x))$ ([the manager's note](../../manager/ratio-classifier-idea.md)) | built only after the reviewer's section "The ratio classifier, 2026-11-25" in `reviewer/2026-11-24-review.md`, with the estimator and control it names; its settings are written into this file before it runs |

**The checker.** Paper 1's pair-aware checker (`2-classifier-guidance/scripts/mnist/checker.py`): a small head on both frozen experts' decoder features, asked with labels set to none, plus the asked digit and colour, giving one log-odds that the picture becomes that digit in that colour. It trains on seen combinations only, never on a novel one: positives are real coloured training digits of a seen combination and the experts' own seen-combination samples that the digit-and-colour classifier calls right; negatives are the same digits in a wrong colour, a wrong digit in the asked colour, and the experts' samples called wrong. Its accuracy on held-back seen-combination pictures is reported.

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

**Cost** (inferred until the timed run): building the data, seconds on a CPU; two experts, about 0.3 GPU-hours each; the classifier, under 0.1; the checker and its sampled training pictures, about 0.3; sampling 5,000 novel and 5,000 seen pictures per arm plus FID, about 0.3 for arms 1 and 2. About 1.5 GPU-hours before arm 3, inside question 108's 1 to 3. Runs go to `bigbatch`, excluding `mscluster[44-45,48,50-51,57,61,65,68,71-76,82-83]`, one job at a time, logged to W&B project `prime_lab/poe-repair-animals-compose`. No API tokens.

**Buys:** whether a push on finished, never-retrained experts reaches unseen combinations, the claim DCFM's retraining rests on. A support sends question 108 to experiment 2 (below DCFM's minimum share of seen combinations); a null says the plain product's failure on Colored MNIST is not one this push repairs.

## The ladder

Outputs go under `/datasets/mmolefe/poe_repair_min/loop-12/z9/<rung>/`.

1. **Data and split**: `split.json` and a sheet of every combination, real pictures, seen ones marked.
2. **Classifier**: accuracy on real test digits, per combination.
3. **Experts**: a timed run, then both trained; control 3.
4. **Arm 1**: the plain product on the novel and seen combinations; FIDs and accuracy.
5. **Checker and arm 2**: control 1, then arm 2; the bar.
6. **Arm 3**, after the reviewer's verdict.

## What this cannot say

Two holders, not ten, so no row matches DCFM's table exactly. One split draw and one push setting. Nothing here bears on separate objects in one picture, Paper 1's own setting.
