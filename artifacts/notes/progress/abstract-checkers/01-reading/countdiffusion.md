# CountDiffusion: Text-to-Image Synthesis with Training-Free Counting-Guidance Diffusion

arXiv 2505.04347 (v1), read from the full PDF (8 pages; the paper has no separate appendix). No venue is printed on the PDF.

## What it does
CountDiffusion generates in two passes from the same starting noise (p. 3, p. 4). The first pass stops at an intermediate step, makes a one-step prediction of the final image, and counts its objects with Grounded SAM. The second pass restarts and edits the object's attention inside a chosen region to add or remove objects (p. 4).

## What the classifier or loss reads
**Counter.** Grounded SAM reads the decoded one-step image estimate $x_{0,mid}$ and the object tags, and returns each object's segmentation mask and the count per class (p. 3, p. 4, Eq. 8 to 10). The tags come from the user or an LLM (p. 4). Its output is compared with the prompt count to choose a correction mask $M$. No gradient passes through Grounded SAM in the printed equations (p. 4, Eq. 12 to 14).

**Loss.** The hand-written loss reads the attention map $A_t$ inside the correction mask and averages the top $P$% of values (p. 4, Eq. 12). The paper calls $A_t$ "the attention map of the object" (p. 1, p. 4). It does not print which layers, what resolution or which tokens are used. Taking the map per object means it is presumably a cross-attention map (inferred).

**Early noisy steps.** The counter does not read them. It reads at $t_{mid}$. Fig. 3 shows the first one-step estimates at $x_{39}$ and $x_{38}$ as blurred, and the caption says the estimate is informative after about 15 of 40 steps (p. 4). The experiments set $T = 40$ and $t_{mid} = 30$, which is after 10 steps (p. 5). The attention correction then acts on the early steps, from $T$ to $t_{mid}$, in the second pass (p. 4).

## What it trained on
Nothing is trained. Grounded SAM is used pretrained with box threshold 0.4, text threshold 0.2 and IoU threshold 0.5 (p. 5). The hand-written loss is the signed top-$P$% mean of masked attention (p. 4, Eq. 12).

## How its gradient enters
The update is on the latent during the second pass, for steps from $T$ to $t_{mid}$ (p. 4, Eq. 12 to 14):

$$loss = \pm \frac{1}{P} \sum \mathbf{topk}(A_t \odot M, P), \quad t \in mid, mid+1, \dots, T$$

$$z_t' \leftarrow z_t \odot \nabla loss \odot \sigma, \quad t \in mid, mid+1, \dots, T$$

$$z_t'' = z_t' \odot M + z_{t,save} \odot (1 - M), \quad t \in mid, mid+1, \dots, T$$

$A_t$ is the attention map, $M$ the correction mask, $\sigma$ a strength whose value is not printed, and $z_{t,save}$ the latent saved from the first pass, which keeps the background fixed (p. 4). The sign is positive when there are too many objects and negative when there are too few (p. 4). The best selection is $P = 50$ (p. 6, Table II). Gaussian smoothing is applied to the attention before the loss (p. 4).

Eq. 13 is printed with element-wise multiplication of the latent and the gradient (p. 4). The method is described as universal guidance, whose update is additive, so this is probably a typo (inferred). No gradient cap or normalisation is printed.

**Two or more classes.** Each class is corrected separately from the same initial noise, and the latents at $t_{mid}$ are averaged (p. 4, Eq. 15):

$$z_{mid} = \frac{1}{n}\sum_{i=1}^{n} z_{mid,i}$$

The models are SDXL and PixArt-Σ, with 40 total steps (p. 5).

## Unseen concepts
This paper does not test held-out concepts. No learned checker is trained here, and Grounded SAM is never split into categories it saw and did not see. The two self-built datasets use the same 25 object categories (p. 4).

## Word leakage and confounds
**Word access.** Grounded SAM receives the object word as a tag, and the attention loss targets that object's map (p. 4). The paper does not discuss word leakage.

**The steering model also grades the result.** Grounded SAM chooses the correction and also produces the counts behind accuracy and MAE (p. 4, p. 5). The accuracy gains are therefore measured by the model that steered them. The paper does not address this.

**Parameter setting.** Fig. 3 describes $t_{mid} = 25$ as the informative point (p. 4), while the experiments use $t_{mid} = 30$ (p. 5). Table III lists no row with 10 guidance steps out of 40 total, which is what $T = 40$ and $t_{mid} = 30$ imply (p. 6) (inferred).

## Numbers

| Metric (scale, direction) | This method | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| Count accuracy, percent of images exactly right by Grounded SAM, higher is better | 59 | 51 CountGen, 34 SDXL | CoCoCount, 200 single-class prompts | SDXL | p. 5, Table I |
| Same | 60 | 40 PixArt-Σ | CoCoCount | PixArt-Σ | p. 5, Table I |
| Count accuracy, higher is better | 45 | 35 CountGen, 21 SDXL | GPTSingleCount, 500 prompts | SDXL | p. 5, Table I |
| Count accuracy with all classes right, higher is better | 31 | 5 SDXL | GPTMultiCount, 100 prompts of 2 or 3 classes with counts 1 to 3 | SDXL | p. 5, Table I |
| Same | 43 | 23 PixArt-Σ | GPTMultiCount | PixArt-Σ | p. 5, Table I |
| MAE in objects, lower is better | 0.90 | 1.28 CountGen, 2.33 SDXL | CoCoCount | SDXL | p. 5, Table I |
| MAE in objects, lower is better | 1.38 | 2.10 SDXL | GPTMultiCount | SDXL | p. 5, Table I |
| **Count 2**, accuracy in percent read from the plot, higher is better | about 76 (SDXL), 71 (PixArt-Σ) | about 67 CountGen, 50 PixArt-Σ, 43 SDXL | GPTSingleCount, 100 prompts at count 2 | SDXL, PixArt-Σ | p. 5, Fig. 4 |

The count 2 result is for one class repeated twice ("two red apples", p. 5). GPTMultiCount is the only test here with different classes in one prompt, and it does not report count 2 or a two-class subset separately (p. 5).

## What to copy, what to avoid
- Copy the two-pass idea if a count must be read: read at a mid step, then restart from the same noise and steer on the early steps. This addresses our finding that decoded early images carry no usable signal for a checker (balanced accuracy 0.50 at step 0) (inferred).
- Compare directly against its multi-class step. It corrects each class in its own pass from the same noise and averages the latents at $t_{mid}$ (p. 4), which makes it a close relative of our PoE of two single-subject prompts (inferred).
- Avoid its evaluation design. A count checker must be scored by a detector or raters separate from the one that steers generation (inferred).
- Avoid reading its 76% at count 2 as evidence for two different subjects, since that test repeats one class. Our task needs a separate measurement on pairs such as a cat and a dog (inferred).
- Its attention push works by raising or lowering attention in a mask chosen from a detected object. Our attention repulsion had no such target and drew two animals on 0 of 8 seeds, so a spatial target may be the missing ingredient (inferred).

## Closer work it cites
None.
