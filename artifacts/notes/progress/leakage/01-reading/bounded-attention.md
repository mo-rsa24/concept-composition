# Be Yourself: Bounded Attention for Multi-Subject Text-to-Image Generation

arXiv 2403.16990 (v1, 25 March 2024), read from the full PDF (18 pages, appendix included). No venue is printed on the PDF.

## What it does

The paper argues that leakage between subjects comes from the attention layers, which blend features across the image, so subjects that look or mean alike borrow from each other (p. 1). Its fix masks attention so that each subject's pixels and tokens can only exchange information with themselves and the background, and steers the early latent with a box-localisation loss computed on those masked maps (p. 5, p. 6). It needs one bounding box per subject from the user (p. 5).

## Where it says leakage enters

A combination: cross-attention and self-attention together, with two token-level contributors in the prompt (p. 1, p. 5, p. 14).

**Cross-attention.** Figure 3 projects the cross-attention queries of each subject to 2D with PCA, labelling each query by the subject whose averaged cross-attention map covers it (p. 3, p. 4). When kitten and puppy or hamster and squirrel are drawn together in one SD image, their queries mix, and the pictures show kitten ears on the puppy and hamster-pink feet on the squirrel (p. 4). This is a measurement shown as scatter plots with no statistic attached.

**Self-attention.** Figure 4 shows self-attention maps from a crab's eye or leg attending to the frog's matching part, with crab colour and frog toe pads appearing on both animals (p. 4). Again a picture with no number.

**Both together.** The authors state that the two sources reinforce each other, so fixing only one is insufficient (p. 5). This is an argument; the ablation in Figure 13 (p. 11) is qualitative.

**Which layers.** Semantically similar subjects (kitten and puppy) have mixed queries at all three decoder resolutions (64×64, 32×32, 16×16); visually similar but semantically different subjects (lizard and fruit, similar texture) overlap only at the 64×64 layer (p. 5, Figure 5). Measurement by PCA plot.

**Tokens.** Conjunctions, positional words and numbers "carry semantics related to other subjects around them", and the first padding token [EoT] attends to foreground subjects and can generate subjects on its own, in SD and SDXL (p. 14). Observation, no number.

## What it changes there

The attention map in every masked layer becomes

$$
A_t^{(l)} = \mathrm{softmax}\!\left(Q_t^{(l)} {K_t^{(l)}}^{\top} + M_t\right)
$$

where $Q_t^{(l)}, K_t^{(l)}$ are the queries and keys of attention layer $l$ at timestep $t$, and $M_t$ is a mask of zeros and $-\infty$ that zeroes attention between a subject's pixels and the other subjects' tokens (cross-attention) or the other subjects' box pixels (self-attention) (p. 5, Eq. 3).

**Bounded Guidance** (early steps only) updates the latent by gradient descent:

$$
L_i = 1 - \frac{\sum_{x \in b_i,\, c \in C_i} \hat A[x,c]}{\sum_{x \in b_i,\, c \in C_i} \hat A[x,c] + \alpha \sum_{x \notin b_i,\, c \in C_i} \hat A[x,c]},
\qquad
z_t^{\text{opt}} = z_t - \beta \nabla_{z_t} \sum_i L_i^2
$$

where $b_i$ is subject $i$'s box, $C_i$ is the set of keys belonging to subject $i$ (its box pixels in self-attention; its tokens plus the first [EoT] token in cross-attention), $\hat A$ is the masked attention map averaged over heads and layers, $\alpha$ weights attention that falls outside the box (set to the number of subjects), and $\beta$ is the step size (p. 6, Eqs. 4 and 5; p. 15).

**Bounded Denoising** (every step) runs the ordinary forward pass with the masks applied. After the guidance phase each box is replaced by a segmentation mask made by KMeans clustering of self-attention maps from the UNet's middle block and first up-block, each cluster labelled to a subject by its overlap with that subject's thresholded cross-attention map (p. 7, p. 14, Eqs. 6 and 7).

**Masking rules.** In Bounded Denoising each subject token sees only its own box pixels; in Bounded Guidance all tokens may also see the background (p. 14). Self-attention lets every subject see the background and the background see every subject (p. 14). Bounded self-attention is applied to both the conditional and the unconditional noise prediction, because masking only the conditional one created artifacts under classifier-free guidance (p. 14). Conjunctions, positional words and numbers are removed after text encoding by an automatic part-of-speech tagger, and the first [EoT] token is confined to the union of the boxes (p. 14).

**Layers.** The guidance loss uses attention layers 12 to 19 in SD and 70 to 81 in SDXL; mask refinement uses layers 14 to 19 in SD and 70 to 81 in SDXL (p. 15). The paper does not say how the layers are numbered. The denoising masks are applied "in the attention layers" with no subset named (p. 5), so all attention layers is the likely reading (inferred).

**Timesteps.** 50 denoising steps. Guidance runs for $t \in [T, T_{\text{guidance}}]$ with $T_{\text{guidance}} = 0.7$ (p. 5, p. 15). Read as a fraction of the schedule this is the first 30% of steps, about steps 0 to 15 of 50 (inferred). Bounded Denoising runs at every step (p. 7). The segmentation masks are refreshed every 5 steps (p. 15).

**Extra passes.** Each guidance step runs 5 gradient-descent iterations, each backpropagating through the UNet (p. 6, p. 15), with early stopping once the mean loss reaches 0.2 or lower (p. 15). Step size starts at 10 to 25 and decays linearly to 5 to 10 in SDXL (8 to 15 decaying to 2 to 5 in SD) (p. 15). Over about 15 guidance steps that is up to about 75 extra forward-and-backward UNet passes per image, and none during the remaining steps (inferred). KMeans runs on the GPU every 5 steps (p. 15).

**Run time.** Not printed. The only statement is that KMeans runs on the GPU "to maintain real-time performance" (p. 15).

## Does it need training, a layout or extra inputs

Nothing is trained (p. 1, p. 5).

It needs one bounding box per subject and the list of subject token spans (p. 5). The paper offers no automatic layout: the boxes are a user input throughout. The only automatic step that turns boxes into something else is the post-guidance mask refinement, which converts each box into a segmentation mask from self-attention clusters (p. 7, p. 14). For the user study the layouts were produced by asking ChatGPT for five animal pairs, a background prompt and four layouts (p. 9). The conclusion names generating a seed that matches a given layout as future work (p. 11). It also needs a part-of-speech tagger to drop conjunctions and numbers (p. 14).

## Which model it runs on

Stable Diffusion (version not printed; cited as Rombach et al. 2022) and SDXL (p. 2, p. 3). All quantitative comparisons use Stable Diffusion "for fairness" (p. 7); SDXL appears only in qualitative figures (Figures 1, 9, 10, 13, 16, 17). Resolution and sampler are not printed; 50 denoising steps (p. 15). The layer numbers for both SD and SDXL are given, so the method is stated for both (p. 15). No code link is printed.

## What it measured on similar subjects

**User study on visually similar animals.** ChatGPT supplied five pairs of visually similar animals with four layouts each; six non-curated images per layout per method; 32 users gave 330 responses, asked to pick realistic images that match prompt and layout without leakage (p. 9). Table 2 reports the estimated probability that a picked image came from Bounded Attention against each competitor: 0.85 against Layout-guidance, 0.72 against BoxDiff, 0.95 against MultiDiffusion, on a 0 to 1 scale where 0.5 would mean no preference (p. 9, Table 2; the 0.5 reading is inferred). The model used for the study is not named; the comparison methods are SD methods, so SD is likely (inferred). The pair names are not printed.

**No number on SDXL.** The SDXL similar-subject results (five dog breeds, three kittens and two dogs) are pictures only (p. 8, p. 9).

## Which pairs leak and why

**Query proximity.** Semantic similarity shows up as cross-attention query proximity, and mixed queries when the subjects are generated together lead to leakage (p. 4). Kitten and puppy queries overlap a little when drawn separately and mix fully when drawn together; hamster and squirrel queries are mixed even when drawn separately (p. 4).

**Layer level.** Semantic similarity mixes queries at all decoder resolutions; texture similarity mixes them at the highest resolution only (p. 5).

**Tokens.** Conjunctions, numbers and the first padding token [EoT] spread one subject's semantics onto the other (p. 14).

All of this is shown with PCA scatter plots and pictures. No number relates a pair's similarity to how much it leaks, and text-embedding similarity is not measured.

## When in sampling leakage is set

Not measured. The paper states, citing Patashnik et al. 2023, that "subject outlines are roughly determined in the initial time steps and evolve gradually thereafter" (p. 7), and that fine-detail leaks "emerge only in later stages" (p. 7). On p. 2 it cites the same source for "shape and visual details, which are determined only in later stages" (p. 2). It chose to guide only during the first part of the schedule to preserve quality (p. 7).

## Numbers

| Metric (scale, direction) | Bounded Attention | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| Counting precision (0 to 1, higher better) | 0.83 | 0.81 BoxDiff | DrawBench | SD | p. 9, Table 1 |
| Counting recall (0 to 1, higher better) | 0.88 | 0.78 SD, LG, BoxDiff | DrawBench | SD | p. 9, Table 1 |
| Counting F1 (0 to 1, higher better) | 0.82 | 0.76 BoxDiff | DrawBench | SD | p. 9, Table 1 |
| Spatial accuracy (0 to 1, higher better) | 0.36 | 0.35 Layout-guidance | DrawBench | SD | p. 9, Table 1 |
| User preference for BA over competitor (0 to 1, 0.5 = parity) | 0.85 / 0.72 / 0.95 | against LG / BoxDiff / MultiDiffusion | 5 similar-animal pairs × 4 layouts | not named (SD likely, inferred) | p. 9, Table 2 |

The authors state that the Table 1 metrics "have limitations in capturing semantic leakage" because the detector was trained on real images (p. 9). No error bars, seeds or significance tests are printed.

## What it suggests for this loop

- For the predictor step (inferred): the paper locates leakage in the attention queries, so the cross-attention overlap predictor is the one it supports. Its Figure 5 suggests computing that overlap separately at the low-resolution (semantic) and highest-resolution (texture) layers, because a texture-sharing pair can overlap only at the 64×64 layer. It also suggests checking the first [EoT] padding token's map, which the paper says attends to both subjects. It says nothing about text-embedding cosine.
- For the predictor step (inferred): the kitten and puppy versus hamster and squirrel contrast says query mixing measured when the subjects are drawn *separately* already predicts mixing when drawn together. This project could test that on crow and raven (which do not borrow) against cheetah and cougar (which do).
- For the one-fix step (inferred): Bounded Attention as published needs boxes. CO3 already produces two separate bodies, so the boxes could come from CO3's own early layout: run CO3 to the end of its corrected steps, cluster the self-attention maps from the middle block and first up-block as the paper does after its guidance phase, and apply Bounded Denoising only. That drops Bounded Guidance and its roughly 75 extra forward-and-backward passes. It changes the method, so the result would be Bounded Denoising seeded by CO3's layout, and should be named that way.
- For the one-fix step (inferred): CO3 combines several noise predictions per step. The paper applies bounded self-attention to both the conditional and unconditional passes and warns that masking only one creates artifacts (p. 14). The same consistency question applies to CO3's single-subject passes and needs a decision before running.
- For the one-fix step (inferred): the paper's guidance window (first 30% of 50 steps) coincides with CO3's corrected steps 0 to 15. Running Bounded Guidance on top of CO3 in the same window would stack two edits to the early latent.
- For Paper 2's method (inferred): the paper's own analysis says pushing two subjects' queries apart by latent optimization harms images (p. 4), and its fix works by masking the information flow. A method that changes $d_t$ by subtraction or projection should check whether it pushes queries apart in the same harmful way.

## Closer work it cites

- **Cross-image attention for zero-shot appearance transfer** (Alaluf et al., arXiv 2311.03335), p. 4: cited for self-attention features corresponding across semantically similar objects, which is the mechanism behind self-attention leakage.
- **LoCo: Locally constrained training-free layout-to-image synthesis** (Zhao et al., arXiv 2311.12342), p. 6 and p. 14: training-free attention guidance; source of the observation that the first [EoT] token attends to foreground subjects and can generate subjects alone.
- **Multi-Concept T2I-Zero: tweaking only the text embeddings and nothing else** (Tunanyan et al., arXiv 2310.07419), p. 2 and p. 14: training-free text-embedding change for multiple concepts; cited for padding tokens mattering for the last-named subject.
- **Localizing object-level shape variations with text-to-image diffusion models** (Patashnik et al., arXiv 2303.11306), p. 2, p. 7, p. 14: source of the self-segmentation used for mask refinement and of the claim about when subject outlines are set.
- **Training-free layout control with cross-attention guidance** (Chen et al., arXiv 2304.03373), p. 3 and p. 4: the Layout-guidance baseline whose query separation is analysed in Figure 3.
- **Localized text-to-image generation for free via cross attention control** (He et al., arXiv 2306.14636), p. 3: masks attention maps to input boxes.
- **Dense text-to-image generation with attention modulation** (Kim et al., ICCV 2023, no arXiv id printed), p. 2 and p. 3: training-free attention attenuation by segment.
- **Grounded text-to-image synthesis with attention refocusing** (Phung et al., arXiv 2306.05427), p. 2 and p. 4: losses on cross- and self-attention maps; the paper treats it as trained because it runs on GLIGEN.
- **BoxDiff** (Xie et al., ICCV 2023, no arXiv id printed), p. 7: training-free box-constrained cross-attention guidance, best baseline on counting precision.
