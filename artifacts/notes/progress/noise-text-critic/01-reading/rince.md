# Robust Contrastive Learning against Noisy Views (RINCE)

arXiv 2201.04309v1, read from the full PDF (20 pages, appendix included). The PDF prints no venue.

## What it does

RINCE is a replacement for the InfoNCE contrastive loss that tolerates positive pairs which share no real information, which the paper calls noisy views (p. 1 to 2). One setting, $q \in (0, 1]$, moves the loss between InfoNCE as $q \to 0$ and a fully symmetric loss at $q = 1$ (p. 4). The paper tests it on image, video-audio and graph representation learning (p. 6 to 8).

## What the model or score reads

The loss reads pair scores: $s^+$ for the positive pair and $s_i^-$ for the $K$ negatives, where a score is $f(x)^T g(v)/t$ with temperature $t$ (p. 3). It works with any encoder pair (p. 1). The tested encoders are ResNet-50 for images (p. 6), an 18-layer R(2+1)D video encoder with a 9-layer VGG-like audio encoder (p. 7), and graph isomorphism networks (p. 8). No diffusion latent, U-Net feature or text-image pair appears in any experiment.

## What it trained on

**Noisy CIFAR-10.** SimCLR with ResNet-50, Adam, 500 epochs, learning rate 0.001, temperature 0.5 (p. 17). Label noise flips a class to a related one with probability $\eta/2$, and augmentation noise adds an extra crop to 1/5 of the image size with probability $\eta$ (p. 6).

**ImageNet.** SimCLR at $q = 0.1$ and MoCo v3 at $q = 0.6$, with the original training protocols (p. 6 to 7).

**Video and audio.** Kinetics400 and ACAV100M, 400 epochs on Kinetics at batch size 4,096, learning rate 1e-4, with $q$ rising linearly from 0.01 to 0.4 (p. 7). ACAV100M uses batch size 512 and subsets from 20K to 500K (p. 8, p. 19).

**Graphs.** TUDataset, Adam, 20 epochs, learning rate 0.01, $q = 0.1$, five trials (p. 8).

The weight $\lambda$ is 0.01 for vision and 0.025 for graphs (p. 6).

## How its score or gradient enters

No gradient enters a sampler. RINCE is a training loss (p. 4):

$$
\mathcal{L}^{\lambda,q}_{\text{RINCE}}(s) = \frac{-e^{q \cdot s^+}}{q} + \frac{\left(\lambda \cdot \left(e^{s^+} + \sum_{i=1}^{K} e^{s_i^-}\right)\right)^q}{q}
$$

$s^+$ is the positive score, $s_i^-$ a negative score, $K$ the number of negatives, $q, \lambda \in (0, 1]$ (p. 4). At $q = 1$ this becomes $-(1-\lambda)e^{s^+} + \lambda \sum_i e^{s_i^-}$ (p. 4). As $q \to 0$ it equals InfoNCE plus $\log\lambda$, with matching gradients (Lemma 1, p. 4, proof p. 13 to 14).

Small $q$ puts more gradient on hard positives, the pairs with low $s^+$. Large $q$ puts more on easy positives, so low-scoring pairs are down-weighted (p. 4 to 5). Both ends put exponentially more weight on hard negatives (p. 4).

## When adding scores is valid

This paper does not address it.

## Which of the five properties it bears on

**Counting.** Not tested.

**Attribute binding.** Not tested.

**Spatial relation.** Not tested.

**Position and size.** Not tested.

**Prompt match.** Not tested. No experiment pairs text with an image.

## Unseen concepts

This paper does not test held-out concepts. Its transfer tests move video encoders pretrained on Kinetics400 or ACAV100M to UCF-101 and HMDB-51 (p. 7 to 8), which measures representation quality on other datasets.

## Numbers

All accuracies are top-1 percentages, higher is better.

| Metric | RINCE | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| Accuracy at label noise $\eta = 0.8$ | 91.6 ($q = 1$) | InfoNCE 87.1 | CIFAR-10 | ResNet-50 | p. 19, Table 4 |
| Accuracy at augmentation noise $\eta = 0.4$ | 89.0 ($q = 1$) | InfoNCE 87.3 | CIFAR-10 | ResNet-50 | p. 19, Table 5 |
| Linear evaluation, SimCLR | 70.0 | SimCLR 69.3 | ImageNet | ResNet-50 | p. 6, Table 1 |
| Linear evaluation, MoCo v3 | 74.2 | MoCo v3 73.8 | ImageNet | ResNet-50 | p. 6, Table 1 |
| Kinetics400 pretraining, finetuned | 61.6 | InfoNCE 57.8; GDT 62.3 with stronger augmentation | HMDB-51 | R(2+1)D-18 | p. 7, Table 2 |
| Kinetics400 pretraining, finetuned | 88.8 | InfoNCE 88.6; GDT 90.9 with stronger augmentation | UCF-101 | R(2+1)D-18 | p. 7, Table 2 |
| ACAV100M 200K subset, 200 epochs | 82.368 | InfoNCE 79.94 | UCF-101 | R(2+1)D-18 | p. 19, Table 6 |
| Graph classification | 90.9 ± 0.6 | InfoNCE 89.9 ± 0.4; GraphCL 89.5 ± 0.8 | RDT-B | graph isomorphism network | p. 8, Table 3 |

**Limits of the theory.** The noisy-risk bound needs a noise rate below 0.5 (p. 3, p. 14). The Wasserstein bounds are stated at $q = 1$ with $\lambda > 1/(K+1)$ (p. 5, Theorem 3 on p. 15). The experiments mostly use smaller $q$ (p. 6 to 8), so the guarantee does not cover those runs (inferred).

**Places where the paper is looser than its claims.**

- The paper says it sets $q \in [0.1, 0.5]$ in practice (p. 5). It uses $q = 0.6$ for MoCo v3 (p. 7) and also reports noisy CIFAR-10 runs at $q = 1.0$ (p. 6, p. 19).
- Table 1 prints SimCLR at 69.3 (p. 6). The authors' own corrected SimCLR implementation reached 68.9 (p. 17).
- The text claims new records on all four graph datasets (p. 8). On NCI1, PROTEINS and DD the lead over the best other method is 0.1 to 0.2 points, inside the standard deviations of 0.4 to 0.8 reported for those entries (Table 3, p. 8).

## What to copy, what to avoid

- Use RINCE only as a later variant if this loop retrains a noise-aware checker, with InfoNCE as the baseline. It is a small change to the loss (Fig. 2, p. 4) (inferred).
- Start from $\lambda = 0.01$ and $q$ between 0.1 and 0.5 with a warm-up from 0.01, as in the video runs (p. 5 to 7). With a batch of 128, $K = 127$ and $\lambda = 0.01$ clears the bound's condition $\lambda > 1/128$ (inferred).
- Avoid applying a large $q$ across all noise levels. The noisiest latents will have low positive scores, so RINCE would down-weight them, and steps 0 to 15 are where this project's push must work. Balance the loss per noise level or keep $q$ small (inferred).
- Avoid expecting RINCE to teach binding. It handles positives that share too little information, whereas "a dog to the left of a cat" is a hard negative for "a cat to the left of a dog". That pair helps only if it is placed in the batch (inferred).
- Avoid expecting a robust training loss to fix the earlier CLIP checker failure, where the score rose by 1 to 13.5 logits over a random push and the animals did not move. RINCE says nothing about the gradient with respect to the input (inferred).

## Closer work it cites

None.
