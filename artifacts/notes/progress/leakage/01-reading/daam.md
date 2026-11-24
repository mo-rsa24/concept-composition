# What the DAAM: Interpreting Stable Diffusion Using Cross Attention

arXiv 2210.04885 (v5, 8 Dec 2022), read from the full PDF (13 pages, appendix included). No venue is printed on the PDF.

## What it does

DAAM builds one heat map per prompt word by upscaling every cross-attention map in the U-Net to image size and summing over heads, layers and all sampling steps (p. 3). The paper checks the maps against hand-drawn segmentation masks and human ratings (pp. 4 to 5), then uses them to study syntax (p. 6) and two kinds of "feature entanglement" (pp. 7 to 8). The part that matters here is Section 5.1: prompts "a(n) <noun> and a(n) <noun>" with two nouns from the same WordNet category (cohyponyms) are drawn correctly less often, and their two DAAM maps overlap more (p. 7).

## Where it says leakage enters

Cross-attention, by a correlational measurement (p. 7). The paper does not intervene to test cause.

**What it measured.** 1,000 prompts from the template "a(n) <noun> and a(n) <noun>", half with two cohyponyms (e.g. "a giraffe and a zebra") and half with two unrelated COCO objects (e.g. "a cake and a bus") (p. 7). One image per prompt. Three crowd annotators per image picked which objects were present; a generation counts as correct if both nouns are present (p. 7). Overlap is the IoU of the two nouns' binarized DAAM maps (threshold τ = 0.4 of each map's maximum), averaged over images as mIoU (p. 7, threshold from p. 4, IoU definition Eqn. 8 on p. 12).

**What it found.**
- Accuracy (share of images with both objects present): 61% for non-cohyponym pairs against 52% for cohyponym pairs, significant at the 99% level by an exact test (p. 7).
- DAAM map mIoU (0 to 100 scale, higher means more shared pixels): 46.7 for cohyponym pairs against 22.9 for non-cohyponym pairs (p. 7).
- Split by both factors (Figure 6 top, read from the page image, p. 7): non-cohyponym correct 5.43, non-cohyponym incorrect 29.5, cohyponym correct 43.5, cohyponym incorrect 60.9. The paper reads this as "incorrectness and cohyponymy independently increase the overlap" (p. 7).

**What kind of failure.** The failure scored is omission: "only one of the words is generated in the image, not all" (p. 7). Attribute borrowing between two drawn bodies is not scored. A picture with two bodies, one wearing the other's face, would count as correct under this protocol (inferred).

**Limit on cause.** Overlap is read from the same generation whose correctness is scored, summed over all 30 steps. When only one object is drawn, both words' attention plausibly lands on that one object, so high overlap could be a consequence of the omission as well as its cause (inferred). The paper words the link as "suggesting entangled attention and composition" (p. 7).

## What it changes there

Nothing. DAAM is a read-out method; it adds no guidance and changes no sampling step. The cross-attention it reads is

$$
F^{(i)}_t(\hat h_{i,t}, X) = \operatorname{softmax}\!\left( \frac{(W_q^{(i)} \hat h_{i,t})(W_k^{(i)} X)^\top}{\sqrt d} \right)
$$

where $\hat h_{i,t}$ is the image feature at U-Net block $i$ and step $t$, $X$ the word embeddings, and $W_q, W_k$ the query and key projections (p. 3). The per-word map is

$$
D^R_k[x,y] = \sum_{i,j,\ell} \tilde F^{(i)\downarrow}_{t_j,k,\ell}[x,y] + \tilde F^{(i)\uparrow}_{t_j,k,\ell}[x,y]
$$

where $k$ is the word, $i$ the down or up block, $j$ the sampling step, $\ell$ the attention head, and the tilde means bicubically upscaled to image size (p. 3, Eqn. 6). The binary map is

$$
D^{I_\tau}_k[x,y] = \mathbb{I}\!\left( D^R_k[x,y] \ge \tau \max_{i,j} D^R_k[i,j] \right)
$$

with $\tau \in [0,1]$ (p. 3, Eqn. 7). Layers: all cross-attention layers at 16x16, 32x32 and 64x64 (p. 11, Figure 11). Steps: all 30 of 30 (fraction 1.0) (p. 11). Extra passes: none are added; the maps come from the ordinary forward pass (inferred; the paper rejects gradient attribution because it would need "a backpropagation pass for every pixel for all T time steps", p. 3).

For the adjective study (Section 5.2), the paper fixes all cross-attention maps to those of a seed prompt, following Prompt-to-Prompt, to hold layout fixed while swapping one adjective (p. 8). This is a probe and is not offered as a fix.

## Does it need training, a layout or extra inputs

Nothing is trained (p. 4: the maps segment nouns "despite not having been trained to perform this task"). DAAM needs only the prompt and the index of the word of interest. No boxes, masks, LLM or reference image (pp. 3 to 4). The syntax study uses a dependency parser (CoreNLP) to pick word pairs, and the cohyponym study uses a hand-built WordNet ontology (pp. 6 to 7, 12 to 13); neither is part of DAAM itself.

## Which model it runs on

Stable Diffusion 2.0 base, 512 by 512 pixels, 30 inference steps, DPM solver, classifier-free guidance scale 7.5, through HuggingFace Diffusers (p. 2, p. 11). The paper does not test other models or claim transfer. The method applies to any U-Net with cross-attention (inferred); SDXL has the same kind of layers.

## What it measured on similar subjects

The whole of Section 5.1 is a similar-subject split (p. 7).

**Pairs.** "Similar" means WordNet cohyponyms: two COCO objects under the same hypernym in a hand-built ontology (p. 7, ontology pp. 12 to 13). The animal groups in that ontology are farm animal (cow, horse, sheep), pets (cat, dog) and wild animal (bear, elephant, giraffe, zebra) (p. 13). No per-pair or per-category numbers are reported; results are pooled over all cohyponym pairs and over all non-cohyponym pairs. The only pairs named are giraffe and zebra (cohyponyms), cake and bus, zebra and fridge, giraffe and fridge (p. 7).

**Metric and result.**

| Group | Both objects present (%) | DAAM map mIoU (0 to 100) |
|---|---|---|
| Non-cohyponym pairs | 61 | 22.9 |
| Cohyponym pairs | 52 | 46.7 |

(p. 7). No method is compared; the baseline is the non-cohyponym group.

**Accuracy by overlap** (Figure 6 bottom, read from the page image, p. 7). Overlap bins are per-image IoU at most 0.4 (low), 0.4 to 0.6 (mid), at least 0.6 (high):

| Group | High overlap | Mid overlap | Low overlap |
|---|---|---|---|
| Non-cohyponym, % both present | 36 | 56.1 | 77.5 |
| Cohyponym, % both present | 9.75 | 19.3 | 71.7 |

The text summarises this as 71.7 to 77.5% accuracy at low overlap against 9.8 to 36% at high overlap (p. 7). The number of images in each cell is not printed.

## Which pairs leak and why

The paper's account has two parts, both measured only as pooled correlations (p. 7).

1. **Category.** Pairs from the same WordNet category fail more (52% against 61% both present) and overlap more (46.7 against 22.9 mIoU). Category raises overlap even when generation succeeds: 43.5 mIoU for correct cohyponym images against 5.43 for correct non-cohyponym images (Figure 6, p. 7, read from image).
2. **Attention overlap.** Overlap separates success from failure inside each category. Cohyponym pairs with low overlap are drawn correctly 71.7% of the time, close to the 77.5% of non-cohyponym pairs with low overlap; cohyponym pairs with high overlap succeed 9.75% of the time (Figure 6, p. 7, read from image).

The paper does not measure text-embedding similarity, visual similarity, token position or padding tokens. "Semantically similar" is defined only by WordNet membership (p. 7). In the syntax study, nouns joined by "and" have the lowest content-word overlap, 38.2 mIoU in Table 2 against 47.5 for unrelated word pairs, which the paper attributes "partially to feature entanglement" (p. 6).

## When in sampling leakage is set

Not measured for the cohyponym study; its maps sum all 30 steps (p. 7, p. 11).

The appendix ablation (Figure 11, p. 11, read from the page image, values approximate to about 0.5 points) measures segmentation quality, mIoU80 on COCO-Gen, from DAAM maps built over the first n or last n of the 30 steps. With the full sum at about 60.7:
- first 1 step only: about 46.5; first 5: about 53.4; first 10: about 56.7; first 15: about 60.3.
- last 30 down to the last 1 step only: from about 60.7 down to about 50.6; last steps from 15 onward give about 53.

So the first half of the 30 steps already gives almost the full localization of each noun in the final image. The paper does not say which end of the schedule "first" is; reading it as the noisiest steps is (inferred). The paper's own reading is only that "all time steps and layers contribute positively" (p. 11).

## Numbers

| Metric (scale, direction) | This paper's value | Comparison | Dataset | Model | Page, source |
|---|---|---|---|---|---|
| Both objects present, % (higher is better) | 52 (cohyponym pairs) | 61 (non-cohyponym pairs) | 1,000 "a(n) X and a(n) Y" prompts over COCO objects | SD 2.0 base | p. 7, text |
| DAAM map IoU between the two nouns, mIoU 0 to 100 (higher is more overlap) | 46.7 (cohyponym) | 22.9 (non-cohyponym) | same | SD 2.0 base | p. 7, text |
| Both present %, low per-image IoU (at most 0.4) | 71.7 (cohyponym) | 77.5 (non-cohyponym) | same | SD 2.0 base | p. 7, Figure 6 (image) |
| Both present %, high per-image IoU (at least 0.6) | 9.75 (cohyponym) | 36 (non-cohyponym) | same | SD 2.0 base | p. 7, Figure 6 (image) |
| Noun segmentation mIoU80, 0 to 100 (higher is better), DAAM at τ = 0.4 | 64.8 | 84.0 (Mask2Former, supervised); 35.8 (STEGO, unsupervised) | COCO-Gen, 100 generated images | SD 2.0 base | p. 4, Table 1 |
| Head and dependent map mIoU for "and"-joined nouns, 0 to 100 | 38.2 | 47.5 (unrelated word pairs) | 1,000 COCO captions | SD 2.0 base | p. 6, Table 2 |

## What it suggests for this loop

- For the predictor step: (inferred) cross-attention overlap is the predictor DAAM supports, and within one category it separates success (71.7%) from failure (9.75%) better than category membership does (52% against 61%). This fits the brief's crow and raven case: two cohyponyms that do not borrow would be the low-overlap cohyponym cell. DAAM gives no data on text cosine or $d_t$ map cosine, so it cannot rank them against overlap.
- For the predictor step: (inferred) DAAM's overlap is summed over all steps of the same image whose failure is scored. To use overlap as a predictor of borrowing, compute it at CO3's corrected early steps only, so that the overlap is read before the layout is set; otherwise it partly measures the outcome. DAAM's Figure 11 suggests early-step attention already localizes nouns (about 46.5 mIoU80 from one step, about 60 from the first half).
- For the predictor step: (inferred) DAAM's recipe is a ready implementation choice: bicubic upscale, sum heads and layers at all resolutions, threshold at 0.4 of the max, IoU. Its overlap bins (0.4 and 0.6) give a starting cut point for "high overlap".
- For the one-fix step: (inferred) DAAM offers no fix. Its result implies a fix should lower the overlap of the two subjects' cross-attention maps, which is what Bounded Attention does by masking; DAAM's correct cohyponym images still sit at 43.5 mIoU, so a fix may need to push overlap well below what successful SD generations show.
- For Paper 2's method: (inferred) DAAM's failure measure is omission. CO3 already solves omission on these pairs, so DAAM's accuracy numbers do not carry over to borrowing; only its overlap measurement does.

## Closer work it cites

None. The nearest citation is Prompt-to-Prompt (Hertz et al., 2022, arXiv 2208.01626, p. 9), which DAAM uses to fix cross-attention maps and hold layout constant (p. 8); it is an editing method and does not measure or fix leakage between subjects.

## Internal inconsistencies

- The abstract and introduction give DAAM's segmentation as "58.9–64.8 mIoU" (p. 1); the results text gives "57.9–64.8 mIoU80" (p. 4). Table 1's DAAM mIoU80 values run 57.9 to 64.8 (p. 4).
- The text gives "38.7 mIoU" for "and"-joined nouns (p. 6); Table 2 on the same page gives 38.2.
- The text says the setup gives "28" objects with a cohyponym "across 16 distinct hypernyms" (p. 7); the printed ontology lists 42 leaf nouns under 15 lowest-level categories (pp. 12 to 13).
- Figure 11's full-sum line sits at about 60.7 on an axis labelled mIoU80 (COCO-Gen) (p. 11); Table 1 gives DAAM-0.4 mIoU80 on COCO-Gen as 64.8, and 60.7 as its mIoU∞ (p. 4).
- The text calls DAAM "rows 5–7" of Table 1 (p. 4); DAAM is rows 8 to 10.
