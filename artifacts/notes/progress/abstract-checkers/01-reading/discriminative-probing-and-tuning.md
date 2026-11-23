# Discriminative Probing and Tuning for Text-to-Image Generation

arXiv 2403.04321, read from the full PDF (22 pages, appendix included). No venue is printed on the PDF.

## What it does

DPT trains a small Transformer "discriminative adapter" on Stable Diffusion's U-Net features to do image-text matching and referring-expression grounding (p. 3 to p. 5). It then fine-tunes LoRA layers on the cross-attention with the same discriminative losses (p. 5). An optional self-correction step moves the latent up the gradient of the adapter's image-text match score during sampling (p. 5).

## What the classifier or loss reads

**Input to the adapter.** A noisy latent $z_t$, the timestep $t$ and the prompt $y$ go through the U-Net, and a 2D feature map is taken from one block (Eqn. 1, p. 3):

$$
F_t=\mathrm{UNet}_l(z_t,\mathrm{CLIP}(y),t)
$$

The feature is therefore text-conditioned before the adapter sees it.

**Block and resolution.** The medium block is the stated default (p. 3, footnote 2), with shape 1280 x 8 x 8 (p. 16). The feature gets learnable position embeddings and timestep embeddings by addition, then is flattened (p. 3).

**Adapter.** One Transformer encoder layer and one decoder layer, hidden size 256, feedforward size 2048, 8 attention heads, and $N = 110$ learnable queries (p. 16). Ten queries serve global matching ($M = 10$) and 100 serve local grounding (p. 16).

**Match score.** The $M$ matching queries are projected to CLIP's dimension as $h_i$, and the score is the best cosine against the CLIP text embedding (p. 4):

$$
s(y,z)=\max_{i\in\{1,\dots,M\}}\cos(\mathrm{CLIP}(y),h_i)
$$

So the prompt enters twice: inside the U-Net and again as the comparison vector.

**Grounding head.** The other $N - M$ queries predict an object probability, a 4-number box and a semantic vector, trained with maximum matching, L1, GIoU and a text-to-object contrastive loss (Eqns. 5 to 7, p. 4, p. 5). Self-correction uses only the match score $s$ (Eqn. 9, p. 5).

## What it trained on

**Data.** COCO2014 train for matching: 82,783 images and 414,113 captions (p. 13, p. 14). RefCOCO, RefCOCO+ and RefCOCOg are combined for grounding: 28,158 images and 321,327 expressions (p. 14).

**Labels and negatives.** Each training instance holds an image, a referring expression, a true caption, a hard negative caption, a hard negative image, and a random negative caption and image (p. 14, p. 15). Hard negatives are the top 20 captions or top 4 images retrieved with OpenCLIP ViT-H-14 (p. 14, footnote 4). Real images are noised to a sampled $t$ (p. 3, Eqn. 8, p. 5).

**Loss.** Bidirectional contrastive matching plus the grounding loss, with $\lambda_0$ to $\lambda_3$ set to 1, 5, 2 and 1 (Eqns. 3, 4, 7, 8, p. 4, p. 5, p. 16).

**Stage 1, frozen SD.** 60k steps, batch 220, learning rate 1e-4, AdamW, gradient clip 0.1, about 3 days on one A100 (p. 16).

**Stage 2, LoRA tuning.** 6k steps, batch 8, gradients accumulated over 8 steps, learning rate 1e-4, about 1 day (p. 16). LoRA and the adapter are both trained (p. 5, Table 11, p. 16). LoRA rank 4 scored best on most metrics (Table 7, p. 14).

**Checkpoint choice.** The checkpoint with the highest CLIP score on a COCO validation set is kept (p. 16).

## How its gradient enters

Self-correction changes the latent before the noise prediction (Eqn. 9, p. 5):

$$
\hat z_t=z_t+\eta\,\frac{\partial s(z_t,y)}{\partial z_t}
$$

$\hat z_t$ then goes into the U-Net to predict noise and produce $z_{t-1}$ (p. 5). The default is $\eta = 0.5$ (p. 16), and the useful range reported is 0.05 to 1 (Fig. 4b, p. 8).

The update has no $\sigma_t$ factor. The paper prints no normalization, no cap and no step window. The models are SD-v1.4 and SD-v2.1 (p. 6). Self-correction is only reported on top of a LoRA-tuned model (Table 4, p. 7). No row shows the adapter guiding a frozen SD.

## Unseen concepts

This paper does not test held-out concepts.

Its "out-of-distribution" setting means a different prompt distribution (p. 5). CC-500 is 446 template prompts that join two concepts (p. 15). No object category is removed from adapter training, and overlap between test and training concepts is never measured.

The 16 CC-500 prompts printed in Figs. 10 and 14 use 20 distinct object nouns (p. 18, p. 21). All 20 are COCO object categories (inferred from the standard COCO class list), and COCO is the training set. The OOD result is therefore evidence about prompt format and says nothing about unseen species.

## Word leakage and confounds

**Direct word access.** The adapter reads features computed with $\mathrm{CLIP}(y)$ and compares them to $\mathrm{CLIP}(y)$ (p. 3, p. 4). The paper does not discuss whether the score can read the text from its own conditioning (inferred risk). It also does not say whether $F_t$ is recomputed under each candidate caption when scoring negatives in Eqns. 3 and 4.

**Evaluator overlap.** Hard negatives are mined with OpenCLIP ViT-H-14 (p. 14), and the reported CLIP score uses OpenCLIP ViT-H-14 (p. 5, footnote 3). Checkpoints are also selected on a CLIP score (p. 16). Training and selection may therefore favour the evaluation metric (inferred).

**The tuned model changes.** Most of the gain comes with LoRA weights altering SD, so the result does not describe a frozen generator (Table 4, p. 7).

## Numbers

Scores are as printed. The paper does not state the scale of CLIP, BLIP or GLIP scores. Higher is better for each (p. 13). All rows below use SD-v2.1.

| Metric (scale, direction) | This method | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| BLIP-ITM (printed score, higher better) | DPT+SC 91.33; DPT 90.72 | Attend-and-Excite 90.03; SD 85.45 | CC-500, OOD prompts | SD-v2.1 | p. 6, Table 1 |
| CLIP score (printed, higher better) | DPT+SC 40.25; DPT 40.23 | Attend-and-Excite 39.43; SD 39.24 | CC-500 | SD-v2.1 | p. 6, Table 1 |
| GLIP (printed, higher better) | DPT+SC 53.29; DPT 53.29 | Attend-and-Excite 53.29; SD 52.09 | CC-500 | SD-v2.1 | p. 6, Table 1 |
| TIFA VQA accuracy (higher better) | DPT+SC 84.63 | HN-DiffusionITM 82.02 | TIFA | SD-v2.1 | p. 6, Table 2 |
| T2I-CompBench color (higher better) | DPT+SC 62.59 | Attend-and-Excite 53.72 | T2I-CompBench | SD-v2.1 | p. 6, Table 2 |
| T2I-CompBench spatial (higher better) | DPT+SC 21.04 | SD 16.94 | T2I-CompBench | SD-v2.1 | p. 6, Table 2 |
| Image-text matching accuracy, % (chance 25) | Stage 1 38.52; DPT 38.52 | OpenCLIP ViT-B-32 48.26 | MSCOCO-HN | SD-v2.1 | p. 7, Table 3 |

**What self-correction adds, DPT to DPT+SC on SD-v2.1.** Across the 17 alignment scores in Tables 1 and 2, 10 rise, 2 are unchanged and 5 fall, with a median change of +0.02 points (p. 6, counted from the tables, inferred). The largest rise is T2I-CompBench color, up 2.00 (60.59 to 62.59). CC-500 BLIP-ITM rises 0.61 (90.72 to 91.33), and CC-500 GLIP is unchanged at 53.29. The largest fall is T2I-CompBench texture, down 0.64 (58.24 to 57.60). ABC-6K BLIP-ITM falls 0.56 (86.19 to 85.63), and COCO-NSS1K CLIP falls 0.08 (35.83 to 35.75).

**Discrimination against noise level (p. 16, Table 10).** Image-text matching is accuracy in %. Grounding is precision@1 in %, averaged over the five test sets (p. 14, p. 16). Each row is one DDPM timestep $t$:

| $t$ | Image-text matching, chance 25.00 | Grounding average, chance 17.00 |
|---|---|---|
| 0 | 39.19 | 51.83 |
| 250 | 39.72 | 54.32 |
| 500 | 38.52 | 48.11 |
| 750 | 35.11 | 28.34 |
| 999 | 31.52 | 13.24 |

Grounding at $t = 999$ is 3.76 points below chance. The DPT row in Table 3 matches the $t = 500$ row here, so Table 3 appears to be evaluated at $t = 500$ (inferred).

**Internal inconsistencies.**

- The text gives the medium block as the default (p. 3, p. 16), but the headline DPT numbers (35.83, 78.58, 41.14) equal the `up1` row of Table 6. The `middle` row reads 35.90, 79.25, 41.11 (p. 13).
- The text says self-correction "could consistently improve" alignment (p. 7). The falls listed above contradict this.
- The text says DPT is best "consistently" (p. 6). On SD-v1.4, LayoutGPT scores 35.06 on spatial against 15.45 for DPT+SC (Table 2, p. 6).
- Tuning is reported as 6k steps (p. 16), as saturating at 8k (p. 8), and as best at step 700 (p. 14).
- The text cites Tab. 11 for the matching and timestep results (p. 6, p. 13). Those results are in Tables 3 and 10.

## What to copy, what to avoid

- Copy the held-out test this paper lacks: remove whole species from checker training and report on them. Prompt-format OOD on COCO nouns would not support our claim (inferred).
- Avoid its input. A U-Net feature computed with the prompt, scored against the same prompt, is the leakage our attention-only checkers are meant to remove (inferred).
- Do not cite this as evidence that a guidance-only checker works on a frozen model. Self-correction was only run after LoRA tuning, and its added effect ranges from -0.64 to +2.00 points with a median of +0.02 across 17 alignment scores (inferred).
- Expect weak signal in our guidance window. Steps 0 to 15 of 50 cover roughly $t = 999$ to $t = 700$ (inferred). At the two nearest printed timesteps, 999 and 750, this adapter's grounding precision@1 is 13.24% and 28.34% against chance 17.00%. That fits our CLIP checker's balanced accuracy of 0.50 at step 0. Measure each checker's accuracy per step before guiding with it (inferred).
- Copy the hard-negative construction for a prompt-match checker, using a different model for mining than for evaluation (inferred).
- Copy the learned multi-query readout only if the attention input stays small. It uses 110 queries and took about 3 days on an A100 (p. 16), which is heavy for a small checker (inferred).
- Its Table 4 shows grounding-only tuning beating the combined objective on CC-500 GLIP, 57.03 against 53.29 (p. 7). Keep separate checkers measurable alone before summing them, since objectives can conflict (p. 7, inferred).

## Closer work it cites

None. The zero-shot classifiers it cites [8, 31] use the diffusion model itself to score text, and HN-DiffusionITM [28] fine-tunes the denoiser (p. 3, p. 16, p. 17). None trains a checker for an abstract property.
