# DeLeaker: Dynamic Inference-Time Reweighting for Semantic Leakage Mitigation in Text-to-Image Models

arXiv 2510.15015, read from the full PDF (41 pages, appendix included). The PDF is printed as "Under review", with no venue.

## What it does

DeLeaker edits the pre-softmax attention scores of FLUX.1-dev during sampling, with no training, no optimisation and no boxes (p. 1, p. 3). It finds each subject's image tokens from early image-to-text attention, then blocks each subject's pixels from attending to the other subject's words, cuts only the strongest pixel-to-pixel attention between the two subjects, and boosts each subject's pixels' attention to its own words (p. 3, p. 4). The paper also builds SLIM, a set of 1,130 FLUX images that human annotators confirmed show leakage, and a three-step VLM pipeline that scores whether a corrected image leaks less than the original (p. 4, p. 6).

## Where it says leakage enters

Cross-attention between image and text, which in FLUX is the image-to-text part of one joint attention. Within-modality attention (text to text, image to image) matters much less.

The evidence is a measurement: an ablation on the 840 SLIM pair images scored by the automatic pipeline (Table 2, p. 8; absolute values Table 15, p. 29). Removing the boost of each subject's own words drops the share of "major improvement" to 0.54 of the full method's (46.07% to 25.00% of images). Removing the block on the other subject's words is reported as a 29% reduction in improvement (p. 9). That figure adds the relative drops of the major column (6.72%) and the minor column (21.95%) (p. 29); the share of images improved moves from 55.83% to 50.60%, a relative drop of 9.4% (p. 29, arithmetic inferred). Adding a suppression of text-to-text attention makes results worse by 9% to 20% across the columns, and image-to-image suppression alone gives 11.90% major improvement against 46.07% for the full method (p. 9, p. 29). From this the paper concludes that leakage comes from "the faulty alignment between" the modalities rather than from weaknesses inside the text side or the image side (p. 9).

A second, qualitative piece: in three leaking examples (cat and cheetah, bat and eagle, butterfly and moth) the attention masks for each subject are cleanly separate at step 1 and have blended by step 12 of 20 (Figure 5, p. 15).

The paper also states, as an argument without its own measurement, that leakage is "primarily driven by the visual similarity of the entities" (p. 14).

## What it changes there

All edits act on the pre-softmax score $\mathrm{Attn}_{qk}$ between query token $q$ and key token $k$ in FLUX's joint attention (p. 4, p. 20).

**Subject masks.** For subject $i$ with text tokens $\mathcal{E}_i^{txt}$, an image token $q$ joins subject $i$'s mask when its head-averaged score to that subject's words is high (p. 3, p. 20):

$$
\mathcal{E}_i^{img} = \{\, q \in \mathcal{I} \mid \mathrm{Attn}_{qk} > \mu_i + \beta_1 \sigma_i,\ k \in \mathcal{E}_i^{txt} \,\}
$$

- $\mathcal{I}$ is the set of all image tokens; $\mu_i$ and $\sigma_i$ are the mean and standard deviation of the image-to-word scores for subject $i$; $\beta_1 = 0.9$ (p. 19, p. 20).
- Masks are averaged over a window of blocks and steps early in sampling (temporal smoothing) and cleaned by a morphological closing then opening with a $3\times3$ ellipse (spatial smoothing) (p. 19).

**The three edits.** With masks fixed, for subjects $i \neq j$ (written here as the prose describes them, p. 4):

$$
\mathrm{Attn}'_{qk} =
\begin{cases}
-\infty & q \in \mathcal{E}_i^{img},\ k \in \mathcal{E}_j^{img},\ (q,k) \in H_{ij}^{img\text{-}img} \quad \text{(cut strong pixel-to-pixel links between subjects)}\\
-\infty & q \in \mathcal{E}_i^{img},\ k \in \mathcal{E}_j^{txt} \quad \text{(block the other subject's words)}\\
\alpha \cdot \mathrm{Attn}_{qk} & q \in \mathcal{E}_i^{img},\ k \in \mathcal{E}_i^{txt} \quad \text{(boost the subject's own words)}\\
\mathrm{Attn}_{qk} & \text{otherwise}
\end{cases}
$$

$$
H_{ij}^{img\text{-}img} = \{ (q,k) \mid \mathrm{Attn}_{qk} > \mu_{ij} + \beta_2 \sigma_{ij},\ q,k \in \mathcal{I} \}
$$

- $\mu_{ij}$, $\sigma_{ij}$ are the mean and standard deviation of scores between subject $i$'s and subject $j$'s image tokens; $\beta_2 = 2$; $\alpha = 1.2$ (p. 4, p. 19). Larger $\alpha$ such as 2.0 produces artifacts (Figure 7, p. 16).
- The printed Eq. 3 (p. 4) and Eq. 11 (p. 20) have the subscripts swapped against the prose: they set the subject's own words to $-\infty$ and boost the other subject's words, and the first case names the same subject's image tokens. The prose and the ablation names match the version above (inferred).

**Layers and steps.** FLUX.1-dev is run for 20 steps, each step passing through 57 transformer blocks, 1,140 blocks in all (p. 19). Interventions run from block 57 to block 741, about steps 1 to 13 of 20, or 5% to 65% of sampling, in every block in that range (p. 19; step conversion inferred). Mask aggregation starts at block 12 and ends at block 171 according to the text (about the first 3 steps, 15%) or block 456 according to Table 6 (about the first 8 steps, 40%) (p. 19). On SANA the image-to-image edit is dropped because it gave inconsistent results and sometimes artifacts (p. 19).

**Extra passes.** None. The masks are taken from the attention scores of the same forward pass, and the edits replace scores in place (p. 3; "optimization-free", p. 1). No runtime is printed.

## Does it need training, a layout or extra inputs

Nothing is trained (p. 1, p. 9). No boxes, masks, LLM or reference image are needed at sampling time (p. 9). The method needs to know which prompt tokens belong to each subject, $\mathcal{E}_i^{txt}$ (p. 3); the paper does not say how they are found, so a parsed or hand-given subject list is required (inferred). The coefficients were chosen "based on a qualitative review of a few samples external to the SLIM dataset" (p. 4).

## Which model it runs on

FLUX.1-dev, 20 steps, guidance scale 3.5 (p. 7, p. 19), and SANA as a second model (p. 7, p. 25). Resolution and sampler are not printed. Both are diffusion transformers with joint image-text attention. The paper contrasts them with UNet models like Stable Diffusion, where text enters through separate cross-attention layers (p. 7), and does not run DeLeaker on any UNet model. On SDXL the image-to-text edits would go in the cross-attention layers and the image-to-image edit in the self-attention layers (inferred).

## The SLIM dataset and how leakage is scored

**What is in it.** 1,130 (prompt, seed, image) triples from FLUX.1-dev, each showing leakage after human checking (p. 4, p. 36). GPT-4o grew a 90-animal list to 200 animals and paired visually similar animals ("For example: (Cow, Horse), (Goat, Sheep)") into 200 prompts, each rendered at 5 seeds (p. 4, p. 5, p. 36). Five subsets, with images kept out of those generated (Table 17, p. 36):

| Subset | Example prompt | Generated | Kept after human check | Share of generated that leaked |
|---|---|---|---|---|
| Animal pairs | "A cow and a horse in a farm" | 1000 | 328 | 33% |
| Animal pairs, interaction | "A raccoon is hugging an opossum" | 1000 | 265 | 27% |
| Animal pairs, interaction and style | "A watercolor painting of a raccoon dancing with an opossum" | 1000 | 247 | 25% |
| Animal triplets | "A raccoon, an opossum, and a panda playing together" | 500 | 116 | 23% |
| Fruit and vegetable triplets | "A bowl containing a strawberry, a tomato, and a cherry" | 500 | 175 | 35% |

The weighted average is 23% of tested images leaking (p. 36). A SANA set holds 368 human-checked images (91, 213 and 64 in the three pair subsets) (p. 36), and a further 222 FLUX images with missing or extra subjects support a counting analysis (p. 27).

**How images got in.** First an automatic filter: GroundingDINO boxes refined by SAM into masks, matched to subject names by the Hungarian algorithm on CLIP similarity, then Gemini asked per masked subject "Does the image contain features unique to a <raccoon>? Answer with yes/no only." This filter had "a non-negligible false-positive rate" (p. 37). Then two authors labelled each survivor by a written protocol (p. 5, p. 30): leakage is present when the subjects look like two of the same kind, when one carries a feature unique to the other (a dalmatian's spots on a golden retriever), or when a hybrid cannot be attributed to either. Blur, background colour bleed and occlusion do not count (p. 30).

**How a fix is scored.** Comparatively, original image against corrected image, by Gemini 1.5 in three steps (p. 6, Figure 3 p. 5, prompts p. 39 to 41): (1) list the visual differences between the two subjects from general knowledge and from reference images of each subject rendered alone by the same model; (2) rate how typical each subject looks in each image given those differences; (3) compare the two images, shown in random order, and output one of five labels: major or minor improvement, no change, minor or major degradation (p. 6). Results are the share of images in each label. Image-text alignment is checked with VQAScore, quality with KID, and closeness to the original with LPIPS (p. 6).

**How well the scorer agrees with people.** On Amazon Mechanical Turk, 60 pair prompts, three annotators each, 980 responses; annotators answer "In which image does the [entity] look more typical?" on a five-point scale for each subject (p. 6, p. 32). Agreement between annotators is 0.52 (quadratic weighted Fleiss' κ), and the Spearman correlation between human and automatic labels is 0.432 with p below 0.001 (p. 6, p. 7, p. 25). People and the VLM usually agree on the direction of change and differ on its size (p. 7).

**CLIP and BLIP do not see leakage.** On 821 pair images, CLIP image-text similarity and BLIP yes-probability, on each subject's crop against its own name or the other subject's name, have Spearman correlation between -0.027 and 0.027 with the automatic labels, all with p above 0.4 (Table 5, p. 18).

## What it measured on similar subjects

The whole of SLIM is pairs or triplets of visually similar subjects, so the main result is the similar-subject result. On the 840 animal-pair images (automatic labels, Table 1, p. 7): DeLeaker gives major improvement on 46.07% and minor on 9.76%, with degradation on 18.81% (5.83% minor plus 12.98% major). The best baseline, an Entity Description Prompt that appends LLM-written descriptions of each animal, gives 35.60% and 11.07%, with 27.62% degradation. The layout-based baselines degrade most images: RAG-Diffusion 73.25%, RPF 53.64%, 3DIS 55.18% (p. 7). In the human study (60 images), 67.8% of DeLeaker's images were judged improved (13.56% major plus 54.24% minor) against 61.29% for the Entity Description Prompt, and none was judged a major degradation (p. 8, p. 25).

By subset (Table 14, p. 29), total improvement is 42.4% on plain animal pairs, 62.6% on interacting pairs and 66.4% on interacting pairs in a style (p. 9). On triplets DeLeaker's total improvement is 51.73% for animals and 66.10% for fruit and vegetables, against 41.38% and 59.77% for the Entity Description Prompt (Table 11, p. 27). On SANA it improves 64.01% of 368 images and degrades 18.69%, while the Entity Description Prompt improves 64.14% and degrades 15.86%, so the prompt baseline is slightly ahead on SANA; DeLeaker keeps the picture much closer to the original (LPIPS 0.35 against 0.70) (Table 10, p. 26). The text describes DeLeaker as ahead with 15% degradation (p. 26).

There is no breakdown by individual pair and no per-pair leakage rate.

## Which pairs leak and why

Argued, not measured per pair. The paper says leakage is primarily driven by visual similarity, "more likely to occur between a horse and a donkey than between a cow and a parrot" (p. 14), and that "the effect is particularly acute within fine-grained categories (e.g., dog breeds)" (p. 4); no number is attached to either. For fruit and vegetables "leakage is rare in pairs but emerges in triplets" (p. 5). The paper hypothesises that interaction and style make leakage stronger (p. 5), and reads the rising mitigation rates in Table 3 as "clear evidence" (p. 9). The human-checked leakage rates in Table 17 go the other way: 33% of plain pair images leaked, 27% of interacting pairs, 25% of interacting pairs in a style (p. 36).

The ablation also bears on which predictor matters: suppressing text-to-text attention hurts (p. 9), which the paper reads as leakage not coming from the text side alone (p. 9). It cites Kaplan et al. 2025 for a separate finding that one cause of leakage is lexical leakage inside the text encoder (p. 9).

## When in sampling leakage is set

Qualitative only. Subject masks are accurate at step 1 (block 50 of 1,140) and in leaking cases begin to blend by step 12 of 20, block 660, about 60% of sampling (Figure 5, p. 15). Mask quality is better early than late, as earlier UNet work also found (p. 19). Figure 8 plots the relative change in cross-subject attention between DeLeaker and the original run against block index: the image-to-text change dips to about -0.25 once per step between roughly blocks 80 and 650 and shrinks toward block 1,140; the image-to-image change falls to around -0.3 from about block 400 onward (p. 17, values read from the plot image). No experiment varies the window to find when leakage becomes fixed.

## Numbers

| Metric (scale, direction) | This method | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| Major improvement, share of images (0 to 100%, higher better) | 46.07% | 35.60% Entity Description Prompt | SLIM pairs, 840 images | FLUX.1-dev | p. 7, Table 1 |
| Total degradation, share of images (lower better) | 18.81% | 27.62% Entity Description Prompt | SLIM pairs, 840 | FLUX.1-dev | p. 7, Table 1 |
| Human-judged total improvement (higher better) | 67.80% | 61.29% Entity Description Prompt | 60 pair images, 3 annotators | FLUX.1-dev | p. 8; p. 25, Table 7 |
| VQAScore, text-image alignment (0 to 1, higher better) | 0.68 | 0.64 Instruction Prompt | SLIM pairs | FLUX.1-dev | p. 7, Table 1 |
| LPIPS to the original image (lower is closer) | 0.22 | 0.33 Instruction Prompt | SLIM pairs | FLUX.1-dev | p. 7, Table 1 |
| Total improvement on animal triplets | 51.73% | 41.38% Entity Description Prompt | 116 images | FLUX.1-dev | p. 27, Table 11 |
| Total improvement (higher better) | 64.01% | 64.14% Entity Description Prompt (ahead) | SANA set, 368 | SANA | p. 26, Table 10 |
| Correct subject count kept when the original was correct | 98.81% | 97.26% Instruction Prompt | SLIM pairs | FLUX.1-dev | p. 28, Figure 15 |
| Spearman correlation of CLIP or BLIP scores with leakage labels | between -0.027 and 0.027 (p above 0.4) | | 821 pair images | | p. 18, Table 5 |

## What it suggests for this loop

- For the predictor step (inferred): the ablation favours the image-to-text place over the text encoder. Suppressing text-to-text attention made results worse (p. 9). That fits the brief's crow and raven result, where name similarity does not predict borrowing. It predicts that cross-attention overlap at CO3's corrected steps will separate leaking pairs better than CLIP text cosine. The evidence comes from FLUX, where text and image share one attention, and the text-side test was a suppression, so it does not rule out the CLIP text cosine as a weak predictor on SDXL.
- For the predictor step (inferred): Figure 5 shows masks clean at step 1 and blended by step 12 of 20 in leaking FLUX images. If SDXL behaves the same, the overlap of the two subjects' cross-attention maps may separate leaking from clean pairs better late in CO3's window (around steps 10 to 15 of 50) than at step 0. Computing the overlap per step, not only averaged, would show this.
- For scoring borrowing (inferred): CLIP and BLIP crop scores did not correlate with leakage labels at all (p. 18). The project's hand labels should stay the ground truth. If an automatic scorer is needed, SLIM's comparative three-step VLM pipeline is the tested design, with a human agreement of 0.432 Spearman.
- For the one-fix step (inferred): DeLeaker needs no boxes, no training and no extra passes, so it is the cheapest alternative to Bounded Attention on CO3's leaking pairs. On SDXL it becomes: masks from cross-attention thresholded at mean plus 0.9 standard deviations, set the other subject's token logits to $-\infty$ for a subject's pixels, multiply the own-token logits by 1.2, and drop self-attention logits above mean plus 2 standard deviations between the two subjects' pixels. It applies only to CO3's joint-prompt pass, since the single-subject passes have one subject. The paper never ran it on a UNet, and the image-to-image edit already failed to transfer to SANA (p. 19).
- For the one-fix step (inferred): DeLeaker cannot add a subject whose mask did not form early (p. 26). CO3 already draws two bodies, so this limit should not bite there.
- For Paper 2's method (inferred): the largest single gain came from boosting each subject's pixels' attention to its own words ($\alpha = 1.2$, p. 9). A method that corrects the noise prediction could test the same idea as a per-subject reweighting of $d_t(c)$ inside each subject's mask.

## Closer work it cites

| Title | arXiv | Page | Why |
|---|---|---|---|
| Follow the flow: On information flow across textual tokens in text-to-image models (Kaplan et al. 2025) | 2504.01137 | p. 9, p. 11 | Cited as finding leakage between lexical items inside the text encoder; the direct test of the CLIP text cosine predictor. |
| DALLE-2 is seeing double: Flaws in word-to-concept mapping in Text2Image models (Rassin et al. 2022) | not printed | p. 9, p. 12 | First identification of semantic leakage in text-to-image models. |
| Training-free regional prompting for diffusion transformers, RPF (Chen et al. 2024a) | 2411.02395 | p. 7, p. 10 | Training-free removal of attention across bounding boxes; needs boxes from an LLM. |
| Region-aware text-to-image generation via hard binding and soft refinement, RAG-Diffusion (Chen et al. 2024b) | 2411.06558 | p. 7, p. 10 | Training-free restriction of attention to each box's local description in early steps; needs boxes. |
| Dynamic prompt learning: Addressing cross-attention leakage for text-based image editing (Yang et al. 2023) | not printed | p. 13, p. 14 | Names cross-attention leakage directly; fixes it by optimising prompt tokens at inference, in image editing. |
