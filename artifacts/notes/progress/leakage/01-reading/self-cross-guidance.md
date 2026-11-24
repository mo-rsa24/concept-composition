# Self-Cross Diffusion Guidance for Text-to-Image Synthesis of Similar Subjects

arXiv 2411.18936 (v2, 24 Mar 2025), read from the full PDF (17 pages, appendix included). No venue is printed on the PDF.

## What it does

The paper adds a training-free loss that penalises the overlap between one subject's cross-attention map and the other subject's "aggregated" self-attention map, and takes gradient steps on the latent to lower it during the first half of sampling (p. 4, p. 5). The aggregated self-attention map is a weighted average of the self-attention maps of every patch that the subject's cross-attention map selects by an Otsu threshold, so it covers the whole region the subject attends to rather than only its most responsive patch (p. 4). It also releases a Similar Subjects Dataset (SSD: 31 two-subject and 21 three-subject prompts) and scores mixing with GPT-4o questions (p. 5, p. 12).

## Where it says leakage enters

A combination of self-attention and cross-attention. The claim is that a subject's patches copy patches from the other subject when they attend to the other subject's region through self-attention: "patches from a subject should not show large similarity to patches from other subjects at least for some timesteps and layers to avoid copying patches from other subjects" (p. 1).

The evidence is qualitative. Figure 2 shows one prompt, "a bear and an elephant", at one seed: in the mixed Stable Diffusion image the cross-attention map of one subject overlaps the aggregated self-attention map of the other, and with the guidance the overlap and the mixing go away (p. 2). Figure 3 shows that the aggregated self-attention map of "bear" differs from its cross-attention map, which the paper uses to argue that a loss on cross-attention maps alone is not enough (p. 4, p. 6). Figure 7 compares building the self-attention map from 1 patch, the top 16 patches and all Otsu-masked patches on three prompts: with 1 patch the subjects still mix, with 16 patches mixing remains "in terms of hair, teeth, and feet", and with the masked patches it is gone (p. 6, p. 8). There is no number that ties the size of the self-cross overlap to the amount of mixing, per prompt or per pair. The place is argued from pictures and supported by the improvement the fix brings.

## What it changes there

For subjects $i$ and $j$, the aggregated self-attention map of subject $i$ is (p. 4):

$$
A^s_i = \frac{\sum_{x_m,y_n} A^c_i[x_m,y_n]\, A^s_{x_m,y_n}}{\sum_{x_m,y_n} A^c_i[x_m,y_n]}
$$

- $A^c_i$ is the cross-attention map of subject token $i$ (a $h/P \times w/P$ grid over the latent).
- $(x_m, y_n)$ runs over the patches that Otsu's threshold on $A^c_i$ keeps.
- $A^s_{x_m,y_n}$ is the self-attention map of the patch at $(x_m, y_n)$: how much that patch attends to every other patch.

The overlap between two subjects and the loss are (p. 4):

$$
g(i,j) = \sum_{x,y}\min\big(A^s_i[x,y], A^c_j[x,y]\big) + \sum_{x,y}\min\big(A^c_i[x,y], A^s_j[x,y]\big)
$$

$$
S_{\text{self-cross}} = \sum_{i \neq j,\; i,j \in \Omega} \frac{g(i,j)}{C^2_N}, \qquad
\mathcal{L}_{\text{total}} = S_{\text{self-cross}} + \lambda\, S_{\text{cross-attn}}
$$

- $\Omega$ is the set of subject token indices and $N$ the number of subjects; $C^2_N$ is the number of subject pairs.
- $S_{\text{cross-attn}} = \max_k\big(1 - \max(A^c_k)\big)$ is Attend-and-Excite's term, which pushes each subject to appear at all (p. 3).
- $\lambda$ weights the two terms. Its value is not printed.

The latent is updated by a gradient step on $\mathcal{L}_{\text{total}}$ at each step of the first half of sampling, 25 of 50 steps (a fraction of 0.5), with extra iterative refinement at the 10th and 20th steps until $S_{\text{cross-attn}} < 0.2$ and $S_{\text{self-cross}} < 0.3$ or an iteration cap is reached (p. 5, p. 12). Before sampling, the starting noise is optimised with INITNO against the same loss (p. 5, Algorithm 1). Only subject tokens enter the loss; verbs and adjectives are left out (p. 5).

Layers: attention maps at 16×16 in SD1, at 24×24 in SD2, averaged over heads and layers (p. 12). For SD3 the image-to-text part of the joint attention replaces cross-attention and the image-to-image part replaces self-attention, and the CLIP and T5 maps are combined by an element-wise maximum (p. 12, Eq. 9).

Cost: every gradient step needs a forward and a backward pass through the denoiser, so the first 25 steps cost at least one extra backward pass each, plus the refinement iterations and the INITNO search (inferred from Algorithm 1, p. 5). The paper prints no timings and no count of extra passes.

## Does it need training, a layout or extra inputs

Nothing is trained: "Our method is training-free" (p. 12). It needs no boxes, masks, layout or LLM. It does need the indices of the subject tokens in the prompt, an input to Algorithm 1 (p. 5). Separate-and-Enhance was left out of the comparison because it fine-tunes (p. 5).

## Which model it runs on

SD1.4, SD2.1 and SD3-medium (p. 5, p. 8). Sampling follows Attend-and-Excite: PNDM ("pseudo-numerical methods") with classifier-free guidance, 50 steps, at each model's native resolution (p. 12). SDXL is not tested. The paper says the method "can boost the performance of any Unet-based or transformer-based diffusion models" (p. 1). On SD2.1 the method alone was weaker than CONFORM, and the authors ran CONFORM for the first few steps and Self-Cross afterwards (p. 7, p. 8).

## What it measured on similar subjects

The metric is TIFA-GPT4o "w/o M": the percentage of images, out of 65 seeds per prompt, for which GPT-4o answers False to "Is the generated content a mixture of {class_A} and {class_B}?" (0 to 100, higher means less mixing) (p. 12, p. 13). The paper itself warns that this score is high when a subject is missing: "vanilla SD2.1 usually generates only one subject of the prompt so its 'w/o M' score is high" (p. 7). Existence (Ext) and recognisability (Rec) are read beside it.

SD1.4, Table 4 (p. 7):

- Animal-Animal (66 prompts such as "a bird and a rabbit"): w/o M 92.94 for Self-Cross against 84.22 for CONFORM, 82.26 for INITNO and 72.24 for plain SD1.4. Ext 94.55 against 89.63 (CONFORM) and 39.51 (SD1.4).
- SSD-2 (31 similar-looking pairs such as "a leopard and a tiger", "a beagle and a collie", "a hummingbird and a kingfisher"): w/o M 86.45 against 79.70 for INITNO, 79.50 for CONFORM and 77.47 for SD1.4. Ext 77.67 against 67.54 (CONFORM) and 30.77 (SD1.4).

SD2.1, Table 5 (p. 7):

- Animal-Animal: w/o M 78.79 for Self-Cross alone against 92.49 for CONFORM and 94.66 for CONFORM followed by Self-Cross.
- SSD-2: w/o M 77.08 for Self-Cross alone against 84.932 for plain SD2.1, 89.76 for CONFORM and 93.20 for CONFORM followed by Self-Cross.

SD3-medium, Table 1 (p. 6), SSD-3 (21 three-subject prompts): w/o M 77.15 against 70.08 for plain SD3; Ext 57.92 against 33.54.

On SD1.4, mixing is worse on SSD-2 than on Animal-Animal for every guided method (Self-Cross 86.45 against 92.94; CONFORM 79.50 against 84.22), but plain SD1.4 scores 77.47 on SSD-2 against 72.24 on Animal-Animal (p. 7). Plain SD1.4 also drops a subject in most images (Ext 30.77 on SSD-2), which inflates its w/o M (inferred from p. 7 and the p. 7 footnote on SD2.1).

## Which pairs leak and why

Not measured per pair. The paper argues that mixing "is more prominent when synthesizing multiple similar-looking subjects" (p. 1) and built SSD from pairs that "share similar structures with distinguishable details, e.g., different textures for leopards and tigers" (p. 5). The only evidence at the level of a set of pairs is the SSD-2 against Animal-Animal comparison above, which holds for the guided methods and not for plain SD1.4 (p. 7). No text-embedding similarity, attention overlap or appearance similarity is reported for any pair. The mechanism argued is that the less discriminant patches of a subject (hair, teeth, feet) are where mixing survives when guidance looks only at the most responsive patch (p. 6).

## When in sampling leakage is set

Not measured. The guidance runs only on the first half (25 of 50 steps) because "attention maps from early timestamps are known to be semantically meaningful" (p. 4) and the first half "is more relevant for the semantic structure of synthesized images" (p. 5). Both are stated as known, without a measurement in this paper. The refinement steps are the 10th and 20th of 50 (fractions 0.2 and 0.4) (p. 12).

## Numbers

| Metric (scale, direction) | Self-Cross | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| w/o M, % of images GPT-4o calls not a mixture (0 to 100, higher better) | 92.94 | 84.22 (CONFORM) | Animal-Animal, 66 prompts × 65 seeds | SD1.4 | p. 7, Table 4 |
| w/o M (same scale) | 86.45 | 79.70 (INITNO) | SSD-2, 31 prompts × 65 seeds | SD1.4 | p. 7, Table 4 |
| Ext, % of images with both subjects (higher better) | 77.67 | 67.54 (CONFORM) | SSD-2 | SD1.4 | p. 7, Table 4 |
| w/o M (same scale) | 78.79 alone; 94.66 with CONFORM first | 92.49 (CONFORM) | Animal-Animal | SD2.1 | p. 7, Table 5 |
| w/o M (same scale) | 77.08 alone; 93.20 with CONFORM first | 89.76 (CONFORM); 84.932 (plain SD2.1) | SSD-2 | SD2.1 | p. 7, Table 5 |
| w/o M (same scale) | 77.15 | 70.08 (plain SD3) | SSD-3, 21 prompts | SD3-medium | p. 6, Table 1 |
| BLIP caption to prompt text similarity, % (higher better) | 84.3 | 82.2 (INITNO) | Animal-Animal | SD1.4 | p. 6, Table 2 |
| w/o M (same scale) | 92.94 | 93.80 (Attention Refocusing, needs an LLM layout) | Animal-Animal | SD1.4 | p. 15, Table 10 |

## What it suggests for this loop

- (inferred) For the predictor step: the paper locates mixing in the overlap between one subject's cross-attention map and the other's self-attention region, and argues that cross-attention overlap alone misses it. A fourth predictor per pair is cheap to add beside the three planned ones: $g(i,j)$ from Eq. 6, computed from the same cross-attention maps plus the self-attention maps at CO3's corrected steps. If cross-attention overlap fails to separate crow and raven (0 of 8) from duck and goose (7 of 8) while $g(i,j)$ does, that supports the paper's placement.
- (inferred) The paper's own evidence that mixing survives in "hair, teeth, and feet" when only the peak patch is used matches the project's description of borrowed faces, fur and markings: a predictor built from the peak of the cross-attention map would understate overlap; use the whole thresholded region.
- (inferred) For the one-fix step: Self-Cross needs no boxes, which Bounded Attention in its original form does. It needs a backward pass through the denoiser at each guided step and has only been run on SD1.4, SD2.1 and SD3-medium at 50 PNDM steps. On SD2.1 it lost to CONFORM on mixing unless CONFORM ran first, so a single SDXL run of it alone could understate it. It would be a box-free second candidate if Bounded Attention's boxes are the obstacle.
- (inferred) Composing with CO3: CO3 edits the noise prediction, Self-Cross edits the latent before the prediction, so the two act at different points of a step and could be stacked: take the Self-Cross gradient step on $x_t$, then form CO3's prediction at the updated latent.
- (inferred) For Paper 2's method: the checker gradient on steps 0 to 15 has the same shape as this paper's latent gradient on the first half. $S_{\text{self-cross}}$ is a training-free stand-in for the checker's "two separate objects" term and would be the baseline a trained checker has to beat on similar-looking pairs.

## Closer work it cites

- INITNO: Boosting text-to-image diffusion models via initial noise optimization (Guo et al., CVPR 2024; no arXiv id printed) (p. 3, p. 9). Uses a self-attention conflict score between subjects, the closest prior to this loss, limited to the most discriminant patch, and optimises the starting noise without training.
- A-STAR: Test-time attention segregation and retention (Agarwal et al., ICCV 2023; no arXiv id printed) (p. 3, p. 9). Training-free attention segregation loss between subjects' cross-attention maps.
- Separate-and-Enhance: Compositional finetuning for text2image diffusion models (Bao et al., SIGGRAPH 2024; no arXiv id printed) (p. 3, p. 9). Penalises overlap between cross-attention maps; it fine-tunes, so it explains the same overlap but is not training-free.
- Grounded text-to-image synthesis with attention refocusing (Phung et al., CVPR 2024; no arXiv id printed) (p. 3, p. 15). Fixes mixing through cross- and self-attention, with an LLM-made layout; reaches 93.80 w/o M on Animal-Animal against Self-Cross's 92.94 (p. 15, Table 10).
