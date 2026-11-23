# Counting Guidance for High Fidelity Text-to-Image Synthesis

arXiv 2306.17567 (v3), read from the full PDF (17 pages, appendix included). No venue is printed on the PDF.

## What it does
The method adds the gradient of a pretrained, class-agnostic counting network to Stable Diffusion's noise prediction at every DDIM step, so the image holds the number of objects the prompt asks for (p. 1, p. 4). For prompts with two object types it first pushes the two words' attention maps apart, then masks the decoded image estimate with each word's map and counts each masked image separately (p. 4 to 6).

## What the classifier or loss reads
**Counter.** The counter is RCC, a reference-less regression network built on features of a pretrained ViT (p. 3). It counts "all salient objects" and receives no word (p. 3).

**Input.** RCC reads the decoded clean-image estimate $\hat{x}_0$, recomputed from the current latent at each step through the VAE decoder (p. 4, Algorithm 1). It never reads noisy latents or UNet features.

**Two object types.** RCC reads the masked decoded image $\hat{x}_{0,i} = \hat{x}_0 \odot M_i^b$, where $M_i^b$ marks the pixels at which word $i$ has the largest attention (p. 6, Eq. 15 and 16). Section 4.2.1 says the masks come from "self-attention maps" (p. 5). Section 4.2.2 then follows Attend-and-Excite, excludes the $\langle sot \rangle$ token and takes one map per object word (p. 5). Excluding a text token only applies to cross-attention, so the maps used are word cross-attention maps (inferred). Layers and resolution are not printed.

**Early noisy steps.** Algorithm 1 applies the loss at every step from $T$ to 1, so it does run at early steps (p. 4). The paper says the early decoded estimate is "of too low quality" for a class-aware counter to identify instances, and gives this as the reason for choosing a class-agnostic one (p. 5). It reports no measurement of RCC's accuracy at early steps.

## What it trained on
Nothing is trained. RCC is used pretrained and is not retrained on noisy images (p. 4). The hand-written losses are the squared relative count error (Eq. 8, p. 4) and the attention overlap loss (Eq. 13, p. 6).

## How its gradient enters
For one object type (p. 4, Eq. 8 and 9):

$$L_{count} = \left| \frac{Count(\hat{x}_0) - N}{N} \right|^2$$

$$\epsilon \leftarrow \epsilon + s_{count}\sqrt{1-\bar{\alpha}_t}\,\nabla_{z_t} L_{count}$$

$N$ is the target count, $z_t$ is the latent and $s_{count}$ is the guidance scale.

For two or more object types (p. 5 to 6, Eq. 11 to 14, 17, 18), with $\hat{M}_{i,j,k}$ the min-max normalised attention of word $i$ at pixel $(j,k)$:

$$L_{attention} = \sum_{j,k}\min_i \hat{M}_{i,j,k} - s_{max}\sum_{j,k}\max_i \hat{M}_{i,j,k}$$

$$\epsilon \leftarrow \epsilon + s_{attention}\sqrt{1-\bar{\alpha}_t}\,\nabla_{z_t} L_{attention}$$

$$\epsilon \leftarrow \epsilon + \sum_i s_{count,i}\sqrt{1-\bar{\alpha}_t}\,\nabla_{z_t} L_{count,i}$$

**Steps and scales.** Guidance runs on all 50 DDIM steps (p. 4, p. 6). $s_{max} = 0.1$ and $s_{attention} = 1$ (p. 6). $s_{count}$ is either the constant 1 or the schedule $\max(0.01,\, 0.2N - 1)$ (p. 6). No gradient normalisation or cap is printed. The models are Stable Diffusion 1.4 and 2.1 (p. 6).

**At count 2.** The schedule gives $s_{count} = \max(0.01, -0.6) = 0.01$. It stays at 0.01 for every target up to 5, so the paper's best setting has counting guidance almost off at count 2 (inferred from the printed formula).

## Unseen concepts
This paper does not test held-out concepts. RCC is class-agnostic by design, but the paper never splits its 34 prompt classes into ones RCC saw and ones it did not (p. 6, p. 11).

## Word leakage and confounds
**Word access.** RCC sees pixels only, so it cannot read the word. The word reaches the count only through the mask in the two-object case (p. 5 to 6). The paper does not discuss word leakage.

**Scoring.** The count is scored by Grounding DINO, a different network from the RCC counter that steers generation, so the method is not graded by its own counter (p. 6, p. 11).

**Two-object evaluation.** The 1122 two-object prompts are scored only by CLIP text-image and text-caption similarity (p. 6, Table 1c on p. 8). The paper prints no count accuracy for any two-object prompt.

## Numbers

| Metric (scale, direction) | This method | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| Normalised MAE: count error divided by target, targets 1 to 20, lower is better | 0.567 (linear schedule), 0.585 (constant) | 0.599 vanilla, 0.601 Attend-and-Excite | 680 single-class prompts | SD 1.4 | p. 8, Table 1a |
| Normalised RMSE, lower is better | 0.692 | 0.709 Attend-and-Excite, 0.746 vanilla | 680 prompts | SD 1.4 | p. 8, Table 1a |
| Normalised MAE, lower is better | 0.461 | 0.473 vanilla | 680 prompts | SD 2.1 | p. 8, Table 1d |
| CLIP text-image similarity on two-object prompts, higher is better | 0.329 | 0.330 Attend-and-Excite, 0.324 vanilla | 1122 two-object prompts | SD 1.4 | p. 8, Table 1c |
| CLIP text-caption similarity on two-object prompts, higher is better | 0.732 | 0.731 Attend-and-Excite, 0.722 vanilla | 1122 two-object prompts | SD 1.4 | p. 8, Table 1c |
| User preference on count, share of 330 comparisons | 23.0% prefer ours | 12.1% prefer vanilla, 64.9% tie | Counting prompts | SD | p. 6, p. 8, Table 1b |
| **Count 2**, normalised MAE read from the plot, lower is better | about 0.50 at $s_{count}$ 0.1, 0.485 at 0.3, 0.545 at 1.0, 0.56 at 3.0 | about 0.47 with no guidance ($s_{count} = 0$) | About 34 images at count 2 (680 prompts over 20 counts, inferred) | SD | p. 13, Fig. 7c |

Tables 1a to 1c are labelled only "Stable Diffusion". They are listed as SD 1.4 here because Table 1d is labelled Stable Diffusion 2 (inferred).

At count 2 every guidance strength in Fig. 7c has a higher error than no guidance (p. 13). An error of 0.47 at target 2 is about 0.94 objects (inferred). The figure prints no error bars. The paper says that "for small N" the performance is better with a smaller $s_{count}$ (p. 11).

Two other papers in this reading set rescored the method. Table 1 of the Detection-Driven paper gives it 9.47% accuracy against 11.15% for its own base model SD 1.4 (`iterative-object-count-optimization`, p. 6). Fig. 9 of Make It Count plots it at about 53% human-rated accuracy at count 2 (`make-it-count`, p. 14).

## What to copy, what to avoid
- Avoid a counter that reads the decoded estimate at early steps. Our CLIP checker on decoded early images read the pair at chance (balanced accuracy 0.50 at step 0), and this paper names the same weakness (p. 5) (inferred).
- Avoid a strong count push at count 2. Fig. 7c shows error rising with guidance strength at this count, and the fitted schedule switches guidance nearly off. Our cap of 0.4 of PoE's prediction norm should be tested at lower values too (inferred).
- Avoid relying on $L_{attention}$ to produce two subjects. It belongs to the same family as our cat and dog map repulsion, which drew two animals on 0 of 8 seeds, and this paper shows only CLIP similarity for two-object prompts (inferred).
- A class-agnostic counter returns 2 for two cats, so a count checker for our task needs a separate signal that each named subject is present (inferred).
- Copy the separation between the steering counter (RCC) and the scoring counter (Grounding DINO) (inferred).

## Closer work it cites
- **Learning to Count Anything: Reference-less Class-agnostic Counting with Weak Supervision** (RCC), arXiv 2205.10203, p. 9. It is a trained counter meant to work across categories, which is the property our count checker needs.
- **Teaching CLIP to Count to Ten** (Paiss et al.), no arXiv id printed, p. 9, described on p. 3. It trains a counting-aware CLIP model.
