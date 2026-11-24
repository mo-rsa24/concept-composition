# A Cat Is A Cat (Not A Dog!): Unraveling Information Mix-ups in Text-to-Image Encoders through Causal Analysis and Embedding Optimization

arXiv 2410.00321 (v5, 8 November 2024), read from the full PDF (20 pages, appendix included). Printed venue: 38th Conference on Neural Information Processing Systems (NeurIPS 2024) (p. 1).

## What it does

The paper argues that the CLIP text encoder's causal attention mask lets each later token absorb the earlier tokens, so in "a cat and a dog" the dog token carries cat information (p. 2, p. 3). This makes the first-named animal appear alone more often and makes fused creatures more likely (p. 4). The fix, TEBOpt, optimizes the text embedding before sampling so each subject token stays close to its embedding from a prompt naming that subject alone and moves away from the other tokens (p. 6).

## Where it says leakage enters

A combination: the text encoder's causal self-attention, which then shows up as overlapping cross-attention maps in the UNet (p. 2, p. 9).

**Order swap (measurement).** 400 prompts "a/an <obj1> and a/an <obj2>" over 17 animals, then the same with the order swapped, same seeds, SD1.4 (p. 3, p. 4). The first-named animal appears alone in 46.00% of images when named first and 21.75% when named second; the other animal goes from 20.00% to 47.00% (p. 4, Table 1). Both animals appear in 12.25% and 11.75% of images (p. 4, Table 1).

**Masking tokens (measurement).** Masking both animal tokens in the UNet's cross-attention still leaves the animals in the image, carried by the start, end and padding tokens; mixture falls 2.5 percentage points and missing objects rise 11.75 points (p. 4, p. 5, Table 2). Masking one animal's token drops that animal sharply but not to zero (p. 5, Table 2).

**Pure embedding swap (measurement).** Replacing the second animal's token embedding with its embedding from "a photo of a <obj2>" balances which animal appears (information bias 2.30 to 1.45, where 1 is balanced) but cuts both-animals-present from 12.25% to 7.00% (p. 5, Table 3). Removing all accumulation across tokens gives 0.00% both-present (p. 18, Table 9). The paper reads this as: accumulated context is needed for two objects to coexist, and too much of it causes bias and mixing (p. 18).

**Link to cross-attention (measurement plus argument).** TEBOpt lowers the average cosine between the two animal tokens from 0.454 to 0.286 and raises the symmetric KL distance between their cross-attention maps from 2.321 to 2.637 (p. 8, Table 5). The mechanism from similar token embeddings to overlapping maps to a fused creature or a missing one is argued in words (p. 2, p. 5).

## What it changes there

The text embedding of each object token, once per prompt, before sampling (p. 6). The loss is

$$
\mathcal{L}^{\text{pos}}_{\text{TEB}}(\varepsilon, \varepsilon^{p}) = \min_{i \in O} \operatorname{sim}(\varepsilon_i, \varepsilon^{p}_{(i)}),
\qquad
\mathcal{L}^{\text{neg}}_{\text{TEB}}(\varepsilon) = \frac{1}{k(m-1)} \sum_{i \in O} \sum_{\substack{j=1 \\ j \neq i}}^{m-1} \operatorname{sim}(\varepsilon_i, \varepsilon_j),
$$

$$
\mathcal{L}_{\text{TEB}} = -\mathcal{L}^{\text{pos}}_{\text{TEB}} + \mathcal{L}^{\text{neg}}_{\text{TEB}},
\qquad
\operatorname{sim}(u, v) = \frac{u \cdot v}{\lVert u \rVert \lVert v \rVert}
$$

where $\varepsilon_i$ is the text encoder's output embedding for token $i$ in the full prompt, $\varepsilon^{p}_{(i)}$ is the same object's token embedding from the prompt "a photo of a <object>", $O$ is the set of object token positions, $k$ is the number of objects, and $m$ is the number of tokens counted, excluding the start and end tokens (p. 6, Eqs. 4 to 6). The negative term pushes each object token away from every other counted token, articles and "and" included (inferred from the sum's range).

**Iterations.** At most 20 optimization iterations, stopping early when the loss reaches −0.7 (positive term 0.95, negative term 0.25) (p. 12). Optimizer and learning rate are not printed.

**Layers and timesteps.** None inside the UNet. The optimized embedding replaces the prompt embedding for all 50 steps and every cross-attention layer (inferred from p. 6 and p. 12). TEBOpt can be stacked with latent-optimization methods such as Attend-and-Excite and SynGen (p. 6, p. 8).

**Extra passes.** No extra UNet passes; the optimization touches only the text embedding (inferred). Measured run time on one RTX 3090: 8.21 seconds per image for SD1.4 and 9.14 seconds per image with TEBOpt, averaged over 400 images (p. 12).

## Does it need training, a layout or extra inputs

Nothing is trained (p. 1, p. 9). No boxes, masks, layout or reference image. It needs the positions of the object tokens in the prompt and one extra encoder call per object for the "a photo of a <object>" prompt (p. 6). The paper does not say how object tokens are found in a free prompt; its prompts are templates (inferred).

## Which model it runs on

All analyses and main results use SD1.4 with CLIP ViT-L/14 (77 tokens, 768 dimensions, 12 heads), PNDM scheduler, guidance scale 7.5, 50 steps, on an RTX 3090 (p. 12). Resolution is not printed. The supplement adds SD1.5 with ELLA, SDXL-Turbo and SD3 on the T2I-CompBench colour set (p. 15, p. 16, Table 7) and FID for SDXL-Turbo and SD3 (p. 16). The paper does not say how TEBOpt handles SDXL-Turbo's two text encoders or SD3's three. Code is at github.com/basiclab/Unraveling-Information-Mix-ups (p. 1).

## What it measured on similar subjects

The whole main benchmark is similar subjects: 400 prompts "a/an <obj1> and a/an <obj2>" with both objects drawn from 17 animals (cat, dog, bird, bear, lion, horse, elephant, monkey, frog, turtle, rabbit, mouse, panda, zebra, gorilla, penguin, chicken) (p. 6).

**Metric.** OWL-ViT detects each named animal; a "mixture" is two boxes for different animals overlapping by more than 90% (p. 7). The metric agrees with human labels on 81% of 400 images (p. 7). This counts fused creatures and missing animals; it cannot see part-borrowing between two separate, correctly detected bodies (inferred from the category definitions).

**Results on SD1.4 (p. 8, Table 4).** Percent of images, higher better for "both present", lower better for "only mixture":

| | Both present | Only a fused creature | Info bias (1 = balanced) |
|---|---|---|---|
| SD1.4 | 14.5% | 9.5% | 2.65 |
| SD1.4 + TEBOpt | 15.0% | 8.5% | 1.39 |
| Attend-and-Excite | 34.0% | 11.5% | 2.17 |
| Attend-and-Excite + TEBOpt | 35.0% | 7.5% | 1.38 |
| SynGen | 52.0% | 10.5% | 0.62 |
| SynGen + TEBOpt | 56.5% | 9.5% | 0.72 |
| Structure Diffusion | 7.0% | 11.5% | 1.13 |

Information bias is the count of images with only the first animal divided by the count with only the second (p. 4). The headline "125.42% improvement" is the difference between two distances from balance, 164.71% for SD1.4 and 39.29% with TEBOpt (p. 8, footnote 1). It is a difference in percentage points of distance. It measures which animal goes missing, which is a separate quantity from how often a fused creature appears.

**Mixed-category comparison.** On the T2I-CompBench spatial set (people, animals and objects, 1,000 prompts) SD1.4 produces almost no fused creatures: 0.2% only-mixture (p. 16, Table 8). The paper says nouns from different categories have larger text-embedding distance, which "conceals" the mixture problem (p. 16). This compares two different prompt sets, so more than category differs (inferred).

## Which pairs leak and why

**Single-word text similarity.** Figure 6(a) gives CLIP cosine similarity between single animal words for 13 animals, from 0.24 (panda and mouse) to 0.60 (frog and turtle); penguin and panda is 0.58 (p. 9, read from the figure). The paper says animals with similar colours or sizes have higher similarity (p. 8).

**Map distance against text similarity.** Figure 6(b) gives the symmetric KL distance between two animals' cross-attention maps at early steps for 17 pairs ordered by text similarity (p. 9). Read from the bars, turtle and frog (most similar) sit near 1.15 and panda and bird (least similar) near 2.45, with dog and bird an outlier near 0.6 (p. 9, read from the figure). No correlation coefficient is given. The paper concludes that similar text embeddings activate overlapping areas and contribute to mixture (p. 9).

**Not measured.** No per-pair mixture rate is set against text similarity or map distance. The link from similarity to leakage goes through map distance, and the step from map distance to fused images is argued, partly by citing SynGen (p. 9).

**Token position.** The later-named object's token carries the earlier one's information, so the first-named object dominates (p. 3, p. 4). With three objects the bias grows with distance from the first-named object (p. 15, Table 6).

**Padding tokens.** The start, end and padding tokens accumulate general prompt information and can draw both animals when the animal tokens are masked (p. 4, p. 5).

## When in sampling leakage is set

Not measured. Figure 6(b) uses cross-attention maps "in the early denoising steps" without saying which (p. 9), and the distance is taken from a single step rather than accumulated (p. 12).

## Numbers

| Metric (scale, direction) | TEBOpt | Baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| Info bias (only-first ÷ only-second; 1 balanced) | 1.39 | 2.65 SD1.4 | 400 two-animal prompts | SD1.4 | p. 8, Table 4 |
| Both animals present (%, higher better) | 15.0% | 14.5% SD1.4 | same | SD1.4 | p. 8, Table 4 |
| Only a fused creature (%, lower better) | 8.5% | 9.5% SD1.4 | same | SD1.4 | p. 8, Table 4 |
| Only a fused creature, on top of Attend-and-Excite (%, lower better) | 7.5% | 11.5% A&E | same | SD1.4 | p. 8, Table 4 |
| Cosine between the two animal tokens (0 to 1, lower better) | 0.286 | 0.454 SD1.4 | same | SD1.4 | p. 8, Table 5 |
| Symmetric KL between the two cross-attention maps (higher better) | 2.637 | 2.321 SD1.4 | same | SD1.4 | p. 8, Table 5 |
| Both objects present, colour set (%, higher better) | 60.5% | 57.1% | T2I-CompBench colour, 1,000 prompts | SDXL-Turbo | p. 16, Table 7 |
| Info bias, colour set (1 balanced) | 1.14 | 1.46 | same | SDXL-Turbo | p. 16, Table 7 |
| Both objects present, colour set | 76.3% | 68.4% | same | SD3 | p. 16, Table 7 |
| FID with CLIP-L/14 features, 5,000 images (lower better) | 200.71 | 202.50 | COCO 2017 val | SDXL-Turbo | p. 16 |
| Seconds per image (lower better) | 9.14 | 8.21 | 400 images | SD1.4, RTX 3090 | p. 12 |

No seeds beyond one per prompt, error bars or significance tests are printed.

## What it suggests for this loop

- For the predictor step (inferred): the paper supports text cosine as a predictor through cross-attention overlap. Its Table 5 measures cosine between the two tokens *inside* the joint prompt (0.454 on average, range 0.291 to 0.669), which is larger than single-word cosines because the causal mask copies the first subject into the second. The text-cosine predictor should be computed both ways: the two names alone, and the two subject tokens inside CO3's joint prompt, for each of SDXL's two text encoders.
- For the predictor step (inferred): crow and raven borrow on 0 of 8 seeds while cheetah and cougar borrow on 7. If crow and raven have a text cosine as high as cheetah and cougar, text similarity alone fails here, which matches the brief's reading. The paper never tests per-pair leakage against its predictors, so it cannot settle this.
- For the predictor step (inferred): the paper finds a direction (the first-named subject dominates). Swapping the subject order in CO3's joint prompt on one leaking pair would show whether borrowing runs from the first-named subject into the second.
- For the one-fix step (inferred): TEBOpt needs no boxes and adds about 1 second per image on SD1.4, and changes only the prompt embedding, so it could run in front of CO3 unchanged. Two limits: its metric counts fused creatures and missing animals, which CO3 already mostly avoids, and its SDXL evidence is SDXL-Turbo on the colour set only, with no statement on how the two encoders are handled.
- For the one-fix step (inferred): CO3's single-subject branches already use the "pure" embeddings that Table 3 swapped in. Table 3 shows that pure embeddings balance which animal appears but lower co-existence, which suggests the remaining borrowing in CO3 comes through the joint-prompt branch.
- For Paper 2's method (inferred): $d_t(c)$ for the joint prompt inherits the second subject's absorbed copy of the first. Comparing $d_t$ for the joint prompt computed from the original against the TEBOpt embedding would show how much of the $d_t$ map overlap starts in the text encoder.

## Closer work it cites

- **Predicated Diffusion: predicate logic-based attention guidance for text-to-image diffusion models** (Sueyoshi and Matsubara, CVPR 2024, no arXiv id printed), p. 9: training-free attention guidance that splits a prompt into independent objects to fix missing objects.
- **Grounded text-to-image synthesis with attention refocusing** (Phung et al., CVPR 2024, no arXiv id printed), p. 9: GPT-4 boxes and two attention losses that keep each object inside its region.
