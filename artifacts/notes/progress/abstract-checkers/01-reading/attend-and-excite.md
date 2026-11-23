# Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models

arXiv 2301.13826, read from the full PDF (24 pages, appendix included). ACM Transactions on Graphics, Vol. 42, No. 4, Article 1, August 2023 (p. 1).

## What it does

Attend-and-Excite updates the noisy latent during sampling so that every subject word reaches a high peak in its cross-attention map (pp. 3, 4). It targets "catastrophic neglect", where a subject named in the prompt is missing from the image (pp. 1, 2). It runs on a pretrained Stable Diffusion with no added training (p. 2).

## What the classifier or loss reads

**Maps.** The cross-attention maps of the Stable Diffusion v1.4 U-Net, from a forward pass on $z_t$ and the prompt (pp. 4, 10). All 16x16 attention layers and heads are averaged into $A_t \in \mathbb{R}^{16 \times 16 \times N}$ for $N$ tokens (p. 4). The authors choose 16x16 because those units "have been shown to contain the most semantic information" (p. 3).

**Tokens.** The subject tokens, usually the nouns, chosen by a part-of-speech tagger or by the user (pp. 4, 10). If a word is split into several tokens, the dominant one must be identified (p. 10).

**Normalisation across words.** The start-of-text token is dropped and a Softmax is applied over the remaining tokens at each patch (p. 4). Each patch is then a probability distribution over words, so a word gaining share in a patch takes it from the others (inferred).

**Smoothing.** Each subject map goes through a Gaussian filter with kernel size 3 and $\sigma = 0.5$ (pp. 4, 10). Without it, one high patch can satisfy the loss while only part of the subject is drawn (pp. 4, 11).

## What it trained on

Nothing is trained. The hand-written loss asks the weakest subject word for a high attention peak (p. 4, Eq. 2).

## How its gradient enters

The loss and latent update (p. 4, Eqs. 2 and 3):

$$\mathcal{L} = \max_{s \in S} \mathcal{L}_s, \qquad \mathcal{L}_s = 1 - \max(A_t^s)$$

$$z_t' \leftarrow z_t - \alpha_t \cdot \nabla_{z_t} \mathcal{L}$$

**Symbols.** $S$ is the set of subject tokens, $A_t^s$ the smoothed 16x16 map of token $s$ at timestep $t$, and $\alpha_t$ the step size (p. 4).

**Scale.** $\alpha_t$ starts at 20 and decays linearly to a minimum of 10 (p. 10). Classifier-free guidance scale is 7.5 (p. 10). The PDF describes no gradient normalisation, clipping or cap.

**Steps.** The update runs for $t = T, T-1, \ldots, t_{end}$ with $T = 50$ and $t_{end} = 25$, which the appendix describes as the first 25 denoising steps (pp. 4, 11). After the update, the U-Net runs again on $z_t'$ to produce $z_{t-1}$ (p. 4, Algorithm 1, step 16).

**Iterative refinement.** At three points, the latent is updated repeatedly until the weakest token's peak passes a threshold (pp. 4, 5). The points are $t_1 = 0$, $t_2 = 10$, $t_3 = 20$ with thresholds $T_1 = 0.05$, $T_2 = 0.5$, $T_3 = 0.8$ (p. 5). Refinement stops after at most 20 latent updates or when the threshold is reached (p. 10). Timesteps are written as counting down from 50, so these three values only make sense as the 0th, 10th and 20th denoising iterations (inferred).

**Model.** Stable Diffusion v1.4 with the CLIP ViT-L/14 text encoder (p. 10). SDXL is not tested.

## Unseen concepts

This paper does not test held-out concepts. Nothing is trained, so no category can be held out. The prompts draw on 12 animals, 12 objects and 11 colors (pp. 6, 10).

## Word leakage and confounds

**No classifier sees the word.** The loss reads prompt-conditioned attention directly, and the paper does not discuss word leakage in a classifier.

**The paper handles a shortcut in its own loss.** A single patch with high attention can come from a silhouette that resembles part of an animal (p. 4). Gaussian smoothing makes the peak depend on neighbouring patches (p. 4), and the ablation shows the difference on "a rabbit with a crown" and "a dog and squirrel" (p. 11, Fig. 10).

**The paper handles a confound in its metric.** CLIP image-text similarity can score hybrids highly, because a fused object resembles both subjects (pp. 7, 8). Composable Diffusion, which composes separate predictions, "tends to generate images containing a mixture of the subjects" (p. 6) and comes closest to Attend-and-Excite on some image-based scores (p. 7). The authors add a caption-based score to catch this: BLIP captions the image, then CLIP compares caption and prompt (p. 8). Composable Diffusion is the lowest method on that score (pp. 7, 8).

## Numbers

Model is Stable Diffusion v1.4 (p. 10). The benchmark is 66 Animal-Animal, 144 Animal-Object and 66 Object-Object prompts with 64 seeds each (p. 6).

| Metric | Attend-and-Excite | Baselines | Prompts | Page and table |
|---|---|---|---|---|
| CLIP text-text similarity between prompt and BLIP caption, cosine, higher better | 0.806 | Stable Diffusion 0.767, Composable Diffusion 0.692, StructureDiffusion 0.761 | Animal-Animal | p. 7, Table 1 |
| Same | 0.830 | Stable Diffusion 0.793, Composable 0.769, StructureDiffusion 0.781 | Animal-Object | p. 7, Table 1 |
| Same | 0.811 | Stable Diffusion 0.765, Composable 0.759, StructureDiffusion 0.762 | Object-Object | p. 7, Table 1 |
| User study, % of responses choosing the method as best, higher better, 65 respondents | 90.70 | StructureDiffusion 6.98, Stable Diffusion 2.32, Composable 0 | Animal-Animal, 10 prompts, 4 seeds | p. 8, Table 2 |
| Same | 77.64 | Stable Diffusion 13.92, StructureDiffusion 6.75, Composable 1.69 | Animal-Object | p. 8, Table 2 |
| Same | 77.16 | Composable 9.82, StructureDiffusion 7.31, Stable Diffusion 5.71 | Object-Object | p. 8, Table 2 |

The field-table claim checks out exactly: Animal-Animal CLIP text-text similarity is 0.806 for Attend-and-Excite, 0.767 for Stable Diffusion and 0.692 for Composable Diffusion (p. 7, Table 1). The score compares the prompt with a BLIP caption of the image (p. 7).

## What to copy, what to avoid

- Measure the weakest word's attention peak on our 8 failed seeds before changing the loss. Attend-and-Excite says a subject whose attention stays low in early steps is not generated (p. 4). If the dog's peak stays low in our runs, separation had nothing to separate (inferred).
- Do not expect its result to transfer to our separation push. Its loss raises the weakest word's peak and has no term pushing two maps apart (p. 4). Our minus-correlation loss has no term that raises a peak (inferred).
- Check how our two words compete. Its Softmax runs across tokens of one joint prompt at each patch (p. 4). In PoE the two subjects can come from separate forward passes, where no Softmax couples them, so confirm which pass our cat and dog maps come from (inferred).
- Copy the threshold schedule as a stopping rule: require a minimum peak at set steps and update until it is met (pp. 4, 5, 10). Our push has a fixed norm cap and no check that it achieved anything (inferred).
- Copy the Gaussian smoothing before any max over a map (pp. 4, 11) (inferred).
- Compare update strength before concluding a loss fails. It moves $z_t$ with $\alpha_t$ from 20 to 10, up to 20 extra updates at refinement points, then re-runs the U-Net (pp. 4, 10). Our push is capped at 0.4 of PoE's prediction norm on steps 0 to 15 of 50 and may be much weaker (inferred).
- Keep the push in early steps. Latent changes after step 25 had a negligible effect on layout and added artifacts on SD1.4 (p. 11), and our window of steps 0 to 15 sits inside that range (inferred).
- Do not judge fusion with CLIP image-text similarity. Hybrids score high on it (p. 7), which fits our CLIP checker reading the pair at chance on early images. Use a caption-based or per-subject score (inferred).
- Treat Composable Diffusion's reported fusion as the published form of our PoE baseline problem: 0% of user votes on Animal-Animal prompts (p. 8) (inferred).

## Closer work it cites

None.
