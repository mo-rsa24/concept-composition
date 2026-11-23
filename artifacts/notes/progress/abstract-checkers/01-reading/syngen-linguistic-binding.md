# Linguistic Binding in Diffusion Models: Enhancing Attribute Correspondence through Attention Map Alignment

arXiv 2306.08877, read from the full PDF (24 pages, appendix included). NeurIPS 2023 (p. 1).

## What it does

SynGen parses the prompt to find each noun and the words modifying it, then updates the noisy latent during sampling so each modifier's attention map overlaps its noun and separates from unrelated words (pp. 3, 4). It targets wrong attribute binding, such as a yellow sunflower and a pink flamingo for "a pink sunflower and a yellow flamingo" (pp. 1, 2). No model is retrained (p. 1).

## What the classifier or loss reads

**The maps.** The loss reads the cross-attention maps of the Stable Diffusion 1.4 U-Net (pp. 5, 13). The maps are taken "as in" Prompt-to-Prompt (p. 5).

**Resolution is not printed.** Appendix A defines a map over $D^2$ patches, where $D$ is "the dimensionality of the latent feature map in an intermediate denoising step" (p. 13). No value of $D$ appears on any of the 24 pages, and the PDF does not say which layers are used or how they are averaged. Settling this needs the released code, which was outside this reading.

**Normalisation.** Each token's map is divided by its sum, giving a distribution over patches (p. 13). A map's overall strength is therefore removed before the loss sees it (inferred).

**Which tokens.** A spaCy dependency parse finds nouns and their modifiers (relations amod, nmod, compound, npadvmod, acomp, conj) (p. 3). For a word split into pieces, the distance is computed per piece and only the piece with the largest distance enters the loss (p. 13). The unrelated set $U(S_i)$ is every word outside $S_i$ (p. 4). The PDF does not say whether "a", "and" or special tokens are removed from it.

## What it trained on

Nothing is trained. The hand-written loss is $\mathcal{L} = \mathcal{L}_{pos} + \mathcal{L}_{neg}$ over attention maps (p. 4). The parser is the out-of-the-box spaCy `en_core_web_trf` model (p. 13).

## How its gradient enters

The pull between each modifier and its noun (p. 4, Eq. 1):

$$\mathcal{L}_{pos}(A,S) = \sum_{i=1}^{k} \sum_{(m,n)\in P(S_i)} dist(A_m, A_n)$$

The push between those pairs and every unrelated word (p. 4, Eq. 2):

$$\mathcal{L}_{neg} = -\sum_{i=1}^{k} \frac{1}{|U(S_i)|} \sum_{(m,n)\in P(S_i)} \sum_{u\in U(S_i)} \frac{1}{2}\Big(dist(A_m, A_u) + dist(A_u, A_n)\Big)$$

The total loss and the distance, a symmetric Kullback-Leibler divergence (p. 4, Eq. 3):

$$\mathcal{L} = \mathcal{L}_{pos} + \mathcal{L}_{neg}, \qquad dist(A_i, A_j) = \tfrac{1}{2} D_{KL}(A_i \| A_j) + \tfrac{1}{2} D_{KL}(A_j \| A_i)$$

$$D_{KL}(A_i \| A_j) = \sum_{pixels} A_i \log(A_i / A_j)$$

The latent update (p. 5):

$$z_t' = z_t - \alpha \cdot \nabla_{z_t} \mathcal{L}$$

**Symbols.** $S_i$ is the $i$-th noun-modifier set, $P(S_i)$ its (modifier, noun) pairs, $U(S_i)$ the words outside it, and $A_w$ the attention map of word $w$ (p. 4). $\alpha$ is the scale factor (p. 14).

**Steps and scale.** The update runs in the first 25 of 50 denoising steps with $\alpha = 20$ and guidance scale 7.5 (pp. 5, 13). Each of those steps takes one gradient update, and the U-Net then denoises the updated latent (p. 5). The PDF describes no gradient normalisation, clipping or cap.

**How the settings were chosen.** On 100 DVMP prompts with a fixed seed, the number of update steps was varied from 5 to 50 in fives (p. 14). Few steps left leakage or unformed images, and all 50 steps raised the rate of blurred images (p. 14). Larger $\alpha$ gave stronger concept separation until images became blurred (p. 15).

**Model.** Stable Diffusion 1.4 only (p. 5). SDXL is not tested.

## Unseen concepts

This paper does not test held-out concepts. Nothing is trained, so no category can be held out. The DVMP prompt set contains unusual combinations such as "a pink spotted panda" (p. 5), which tests prompts and no learned classifier.

## Word leakage and confounds

The loss has no classifier that could cheat from the word, and the paper does not discuss this kind of leakage. It does address a confound in evaluation: CLIP acts like a bag of words and cannot tell which attribute belongs to which object (p. 23). Its phrase-level CLIP metric agrees with the human majority only 43.5% of the time, where random choice among four methods gives 25% (p. 23). The main results therefore use human raters (p. 6).

## Numbers

The model is Stable Diffusion 1.4 (p. 5). "Majority share" is the percent of prompts where one method won the majority of three raters, with the four methods and "no majority winner" summing to 100, and higher is better for a method (p. 7). "No majority winner" also absorbs the raters' "equally good" and "equally bad" answers (p. 23, Fig. 23).

| Metric | SynGen | Best baseline | Dataset | Page and table |
|---|---|---|---|---|
| Concept separation, majority share | 28.00 | 11.17 (A&E); 50.17 no majority winner | ABC-6K, 600 prompts | p. 7, Table 1 |
| Concept separation, majority share | 24.84 | 13.33 (A&E); 53.67 no majority winner | DVMP, 600 prompts | p. 7, Table 1 |
| Concept separation, majority share | 38.42 | 18.08 (A&E); 37.29 no majority winner | A&E set, 177 prompts, no animal-animal | p. 7, Table 1 |
| Proper binding, 0 to 100, higher better | 74.90 | 52.47 (A&E) | DVMP, 200 prompts | p. 7, Table 2 |
| Improper binding, 0 to 100, lower better | 19.49 | 30.44 (Stable Diffusion) | DVMP | p. 7, Table 2 |
| Entity neglect, 0 to 100, lower better | 16.26 | 10.77 (A&E) | DVMP | p. 7, Table 2 |
| Entity neglect, 0 to 100, lower better | 02.82 | 01.41 (A&E); 20.56 for Stable Diffusion | A&E set | p. 7, Table 2 |

**Which loss term does the work** (p. 9, Table 3). 100 DVMP prompts, values are percent preferred by human raters among four variants.

| Variant | Concept separation | Visual appeal |
|---|---|---|
| Both losses | 27 | 22 |
| Positive only | 0 | 11 |
| Negative only (push apart) | 3 | 35 |
| Stable Diffusion | 4 | 28 |
| No majority winner | 66 | 4 |

The field-table claim checks out exactly: ABC-6K concept separation is 28.00 for SynGen, 11.17 for A&E and 50.17 with no majority winner (p. 7, Table 1).

## What to copy, what to avoid

- Do not cite SynGen as evidence that separating attention maps draws two animals. For "a cat and a dog" there is no modifier, so $P(S_i)$ is empty, both sums in Eqs. 1 and 2 are zero, and the update does nothing (inferred). The authors drop the animal-animal prompts "as it does not contain modifiers" (p. 5).
- Read the negative-only row as the closest published test of our push. Pushing apart alone won 3% of concept-separation votes against 4% for plain Stable Diffusion and 27% for both losses (p. 9). This agrees with our 0 of 8 seeds. The prompts differ (DVMP has modifiers) and the distance differs (KL against our correlation) (inferred).
- Add a strength term before trusting a separation term. SynGen's maps are normalised to sum to 1 (p. 13) and Pearson correlation ignores a map's scale and offset, so neither loss can ask a faint word to become stronger. This is a candidate explanation for our 0 of 8, untested (inferred).
- Check the update form. SynGen moves $z_t$ and re-runs the U-Net on the moved latent (p. 5), while our push is added to the noise prediction with a cap of 0.4 of PoE's norm. A comparison of step sizes needs the gradient norms, which the paper does not report (inferred).
- Check the window. SynGen uses 25 of 50 steps and reports that fewer steps corrected binding less well (pp. 5, 14). Our push runs on steps 0 to 15 of 50 (inferred).
- Avoid a large push term. The negative loss can grow until the latent leaves the distribution and images blur or become incoherent (p. 20).
- Copy human or per-subject judging for separation, because whole-prompt CLIP scores cannot see binding (p. 23) (inferred).

## Closer work it cites

- Harnessing the Spatial-Temporal Attention of Diffusion Models for High-Fidelity Text-to-Image Synthesis, arXiv 2304.03869 (pp. 10, 12, ref. 24). It trains a layout predictor to find the pixels for each noun and optimises latents with it (p. 10). It is a trained spatial model, which makes it the nearest match here, though it is a predictor of layout and is not described as a classifier for an abstract property.
