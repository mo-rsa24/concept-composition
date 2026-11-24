# Enhancing MMDiT-Based Text-to-Image Models for Similar Subject Generation

arXiv 2411.18301 (v1, 27 Nov 2024), read from the full PDF (18 pages, appendix included). No venue is printed on the PDF.

## What it does

The paper looks at the text-to-image attention maps of Stable Diffusion 3 (an MMDiT, where image and text tokens share one joint attention layer) on prompts naming two to four similar subjects, and names three causes of a subject going missing or mixing (p. 2, p. 4). It then takes gradient steps on the latent during the first half of sampling against three attention losses, one per cause (p. 4, p. 5). At step 5 of 28 it checks whether the subjects' attention regions still overlap; if they do, it restarts from the same starting noise with an extra loss that keeps the worst subject out of the overlap region, and rejects the seed if the overlap persists after the restart (p. 5, p. 6).

## Where it says leakage enters

A combination of three places, all read from the image-to-text part of the joint attention in blocks 5 to 12 of the 24 MMDiT blocks (p. 3):

**Early blocks of cross-attention ("inter-block ambiguity").** Blocks 1 to 4 do not separate the subjects at all; for dissimilar pairs such as "a frog and a bench" blocks 5 onward do; for similar pairs blocks 5 to 8 stay ambiguous while blocks 9 to 12 have separated them, and the wrong information from blocks 5 to 8 "has been injected" into the hidden states (p. 4). Figure 3 shows this for "a bear and a panda", which renders two pandas (p. 4).

**The text encoder ("text encoder ambiguity").** SD3's CLIP tokens and T5 tokens for the same word put their attention in different places; Figure 3 shows CLIP and T5 disagreeing on where "cat" and "dog" go in blocks 9 to 12 (p. 2, p. 4).

**Layout through attention overlap ("semantic ambiguity").** Subjects that are similar in shape and appearance, such as duck and goose, have their attention gathered at the same locations even when the two text encoders agree (p. 2, p. 4).

The evidence is qualitative: one prompt and one seed per cause, with attention maps taken at step 5 of 28 (Figure 3, p. 4). The ablation gives a measurement of the fixes (Figure 7, p. 8): on the three-subject set, scored by the Grounding DINO success rate, the block-alignment loss adds 8.3 points over SD3's 37.9%, the overlap loss adds a further 8.6, the text-encoder loss brings the total to 57.3%, and the restart brings it to 66.7%. Reversing the block-alignment loss, so that blocks 9 to 12 are pulled toward blocks 5 to 8, makes the pictures worse on "a frog and a toad" and "a snake and a lizard" (Figure 8, p. 8; qualitative only). Nothing measures how much each cause contributes on its own, or how often each one occurs.

## What it changes there

Three losses on the cross-attention maps, with $A^{clip}_{i-[a,b]}$ the cross-attention map of subject word $i$ for its CLIP tokens averaged over blocks $a$ to $b$, $A^{t5}$ the same for its T5 tokens, $\hat A$ the normalised map, $N$ the number of subject words, and $\cos$ cosine similarity (p. 4, p. 5):

$$
\mathcal{L}_{ba} = \frac{1}{2N}\sum_{i=1}^{N}\Big[\big(1-\cos(A^{clip}_{i-[5,8]}, \operatorname{sg}A^{clip}_{i-[9,12]})\big) + \big(1-\cos(A^{t5}_{i-[5,8]}, \operatorname{sg}A^{t5}_{i-[9,12]})\big)\Big]
$$

$$
\mathcal{L}_{ta} = \frac{1}{N}\sum_{i=1}^{N}\big(1-\cos(A^{clip}_{i-[9,12]}, A^{t5}_{i-[9,12]})\big)
$$

$$
\mathcal{L}_{ol} = \sum_{1\le i<j\le N} \hat A^{clip}_{i}\cdot\hat A^{clip}_{j} + \hat A^{t5}_{i}\cdot\hat A^{t5}_{j} + \hat A^{clip}_{i}\cdot\hat A^{t5}_{j} + \hat A^{t5}_{i}\cdot\hat A^{clip}_{j} \quad (\text{all over blocks 9 to 12})
$$

$$
\mathcal{L}_{amb} = \lambda_{ba}\mathcal{L}_{ba} + \lambda_{ta}\mathcal{L}_{ta} + \lambda_{ol}\mathcal{L}_{ol}, \qquad z_t' = z_t - \alpha_t \nabla_{z_t}\mathcal{L}_{amb}
$$

- $\operatorname{sg}$ is the stop-gradient (`.detach()` in the paper): blocks 9 to 12 serve as the target for blocks 5 to 8.
- $\lambda_{ba}, \lambda_{ta}, \lambda_{ol} = 1, 0.2, 0.001$ and the step size $\alpha_t = 30$ (p. 5, p. 6).
- $z_t$ is the latent at step $t$.

Overlap check at step 5 (p. 5, p. 6): each subject's map $A_i = \operatorname{Norm}(\tfrac12 A^{clip}_{i-[9,12]} + \tfrac12 A^{t5}_{i-[9,12]})$ is thresholded at 0.2 into a mask $M_i$, and its overlap ratio is

$$
OR_i = \frac{\sum_{x,y} M_i(x,y) \cap \bigcup_{j\ne i} M_j(x,y)}{\sum_{x,y} M_i(x,y)}
$$

the fraction of subject $i$'s region also covered by another subject. If any $OR_i > 0.2$, sampling restarts from the same starting noise and the subject $i^*$ with the largest ratio gets an extra loss that pushes its attention (blocks 5 to 12, both encoders) out of the conflict mask $M_{res} = M_{i^*} \cap \bigcup_{j\ne i^*} M_j$ (Eq. 6, p. 6). If the overlap is still above 0.2 at step 5 of the restart, the seed is rejected (p. 6).

Timesteps: the latent is updated in steps 1 to 14 of 28 (the first half, fraction 0.5), with 1 gradient iteration at steps 1 and 2, 15 iterations at each of steps 3 to 5 (fractions 0.07 to 0.18), and 1 at each of steps 6 to 14 (p. 6, Algorithm 1 on p. 13). That is 56 gradient iterations per image, each a forward and backward pass, on top of the 28 ordinary forward passes (inferred from Algorithm 1, p. 13). Measured time on one A6000: 68.7 s per image without the restart against 20.4 s for plain SD3, and 118.6 s when the restart fires (Table 3, p. 13). The restart fires on 0.4% of two-subject, 29.3% of three-subject and 42.1% of four-subject images (p. 13).

Step range and iteration count were chosen by ablation on the three-subject set (Figure 10, p. 13, values read from the figure): steps 3 to 4, 3 to 5 and 3 to 6 give 54.6%, 57.3% and 57.7% without the restart and 64.4%, 66.7% and 67.3% with it; 5, 10 and 15 iterations give 52.2%, 55.0% and 57.3% without and 61.1%, 65.2% and 66.7% with.

## Does it need training, a layout or extra inputs

Nothing is trained. The method is "training-free ... based only on the input text without using external signals (e.g., layout masks [7])", where [7] is Bounded Attention (p. 3, p. 10). It needs the subject words of the prompt located in both the CLIP and the T5 token sequences, averaging over the tokens of a multi-token word (p. 3). The conflict mask is computed from the model's own attention, not supplied. ChatGPT was used only to write the evaluation prompts (p. 6).

## Which model it runs on

Stable Diffusion 3 Medium (2B), guidance scale 7.0, 28 steps, on one 48 GB A6000; the resolution is not printed (p. 6). The appendix repeats the comparison on SD3.5-Medium (p. 13, p. 14). Everything is tied to MMDiT: block indices 5 to 12 of 24, the joint attention, and the CLIP and T5 token pair (p. 3). No UNet model is tested and no transfer to one is claimed.

## What it measured on similar subjects

The whole benchmark is similar subjects: 90 ChatGPT-made prompts, 30 each with two, three and four similar subjects from animals, plants, fruits, vehicles and balls ("a duck and a goose", "a rose and a tulip", "a whale and a manatee"), 100 seeds per prompt for every method (p. 6). There is no split against dissimilar subjects.

The metric is a success rate, the share of images in which every subject appears (0 to 100%, higher better), scored by Grounding DINO confidence (thresholds 0.5, 0.4 and 0.4 for two, three and four subjects) and by GPT-4o-mini answering yes or no on 50 of the 100 images per prompt (p. 7, p. 12). Prompts where every method averages under 0.2 by Grounding DINO are dropped (p. 12). This success rate counts a mixed subject as a failure only when the detector or GPT-4o-mini no longer finds both subjects; it is not a direct mixing score (inferred from the definitions on p. 12).

On SD3 (Table 1, p. 8), Grounding DINO success rate:

- Two subjects: 68.8% against 61.9% for CONFORM and 53.1% for SD3. GPT-4o-mini: 90.9% against 89.5% and 68.2%.
- Three subjects: 66.7% (67.8% with seed rejection) against 57.1% and 37.9%. GPT-4o-mini: 80.2% (83.9%) against 72.1% and 31.2%.
- Four subjects: 52.7% (60.8%) against 42.4% and 33.1%. GPT-4o-mini: 58.5% (68.7%) against 43.7% and 22.5%.

FID here is measured between each method's images and plain SD3's images, so lower means closer to what SD3 would have drawn: 11.80, 14.26 and 18.17 for this method against 27.40, 35.55 and 30.86 for CONFORM (p. 7, p. 8). A user study with 32 participants picked this method 73.38% of the time against 13.25% for EBAMA (p. 8). On SD3.5-Medium the Grounding DINO success rates are 65.6%, 69.4% and 65.9% (with rejection) against 52.0%, 34.0% and 26.2% for SD3.5 (Table 4, p. 14).

## Which pairs leak and why

Argued from pictures, not measured per pair. The paper names three reasons a pair fails, each shown on one example (Figure 3, p. 4):

- Similar shape and appearance pulls both subjects' attention into the same location: duck and goose (p. 4).
- CLIP and T5 put the same word in different places: cat and dog (p. 4).
- Blocks 5 to 8 fail to separate similar subjects that blocks 9 to 12 do separate: bear and panda (p. 4). Dissimilar subjects ("a frog and a bench") separate from block 5 onward (p. 4).

No text-embedding similarity, per-pair attention overlap or appearance score is reported. The restart rates (0.4%, 29.3%, 42.1% for two, three and four subjects, p. 13) measure how often the step-5 overlap ratio stays above 0.2 after the losses, by number of subjects, not by pair.

## When in sampling leakage is set

Early. The paper reports that the step-5 attention maps (of 28 steps, fraction 0.18) "can reveal the causes of the issues that coincide with the final generated results" (p. 4), and all its diagnostic maps are from step 5 (p. 4). The overlap check at step 5 predicts failure well enough that rejecting the seeds it flags raises the success rate from 52.7% to 60.8% on four subjects (Table 1, p. 8), though the paper does not say how rejected seeds are counted in that rate. Extending the 15-iteration window from steps 3 to 5 to steps 3 to 6 adds 0.4 to 0.6 points (Figure 10a, p. 13), which suggests most of what the losses can change is fixed by step 5 (inferred).

## Numbers

| Metric (scale, direction) | This method | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| Success rate by Grounding DINO, % images with all subjects (higher better) | 68.8 | 61.9 (CONFORM) | 30 two-similar-subject prompts × 100 seeds | SD3 Medium | p. 8, Table 1 |
| Same | 66.7; 67.8 with rejection | 57.1 (CONFORM) | 30 three-subject prompts | SD3 Medium | p. 8, Table 1 |
| Same | 52.7; 60.8 with rejection | 42.4 (CONFORM) | 30 four-subject prompts | SD3 Medium | p. 8, Table 1 |
| Success rate by GPT-4o-mini, % (higher better) | 90.9 / 83.9 / 68.7 | 89.5 / 72.1 / 43.7 (CONFORM) | two / three / four subjects, 50 images per prompt | SD3 Medium | p. 8, Table 1 |
| FID to plain SD3's images (lower is closer to SD3) | 11.80 / 14.26 / 18.17 | 4.15 / 4.25 / 3.89 (A&E, which the paper says barely changes SD3's images, p. 7); 14.51 / 16.40 / 17.69 (EBAMA) | two / three / four subjects | SD3 Medium | p. 8, Table 1 |
| Seconds per image on one A6000 (lower better) | 68.7; 118.6 with restart | 28.6 (EBAMA); 20.4 plain SD3 | | SD3 Medium | p. 13, Table 3 |
| User preference, % of picks | 73.38 | 13.25 (EBAMA) | 25 groups × 32 participants | SD3 Medium | p. 8, Table 2 |
| Success rate by Grounding DINO, % | 65.6 / 69.4 / 65.9 | 62.4 / 53.7 / 36.1 (CONFORM) | two / three / four subjects | SD3.5 Medium | p. 14, Table 4 |

## What it suggests for this loop

- (inferred) For the predictor step: the paper splits leakage between the text side (CLIP and T5 disagreeing on a word) and the attention side (similar subjects' maps gathering in one place). SDXL also concatenates two CLIP text encoders, so the planned CLIP text cosine could be computed per encoder and their disagreement recorded, beside the cross-attention overlap.
- (inferred) The paper's overlap ratio $OR_i$ (fraction of one subject's thresholded region also covered by the other) is a ready definition for the planned cross-attention overlap predictor, and its 0.2 threshold on both the map and the ratio gives a starting point. Its examples line up with this project's labels: duck and goose is the paper's main case of attention gathering in one place, and that pair borrows on 7 of 8 seeds under CO3.
- (inferred) The block result has an SDXL analogue worth checking: if some UNet layers separate the two subjects and others do not at CO3's corrected steps, overlap should be computed per layer group, and averaging over all layers could hide the layers where leakage enters.
- (inferred) For the one-fix step: the paper files Bounded Attention under methods that need "layout masks" (p. 3), which is the cost the one-fix step has to price. This paper's own losses are MMDiT-specific (block 5 to 12 indices, CLIP and T5 pair) and do not port to SDXL as written. The step-5 overlap check and the restart from the same noise with the conflict region blocked do port in principle, as a box-free way to give Bounded Attention or any attention fix a region derived from the model itself.
- (inferred) The cost to expect on SDXL from this style of fix is about 3.4 times plain sampling time (68.7 s against 20.4 s on SD3), and about 5.8 times when the restart fires (p. 13).
- (inferred) For Paper 2's method: this paper's overlap check at step 5 of 28 (fraction 0.18) is a training-free early read of whether the two subjects will separate. It is a baseline the trained checker should beat at the same step, and it sits inside Paper 2's window of steps 0 to 15 of 50.

## Closer work it cites

- Object-conditioned energy-based attention map alignment in text-to-image diffusion models (EBAMA; Zhang, Yu, Wu, ECCV 2024; no arXiv id printed) (p. 3, p. 11). Training-free attention alignment; the second-best user-study pick here and 58.8%, 49.1%, 41.0% Grounding DINO success on two, three, four similar subjects in SD3 (p. 8).
- A-STAR: Test-time attention segregation and retention (Agarwal et al., ICCV 2023; no arXiv id printed) (p. 3, p. 10). Training-free loss reducing the overlap of subjects' cross-attention maps.
- INITNO: Boosting text-to-image diffusion models via initial noise optimization (Guo et al., CVPR 2024; no arXiv id printed) (p. 3, p. 10). Optimises the starting noise against attention losses, training-free.
- Enhancing semantic fidelity in text-to-image synthesis: Attention regulation in diffusion models (Zhang et al., ECCV 2024; no arXiv id printed) (p. 2, p. 11). Training-free attention edit for subject neglect.
- Separate-and-Enhance: Compositional finetuning for text-to-image diffusion models (Bao et al., SIGGRAPH 2024; no arXiv id printed) (p. 3, p. 10). Reduces cross-attention overlap between subjects by fine-tuning.
