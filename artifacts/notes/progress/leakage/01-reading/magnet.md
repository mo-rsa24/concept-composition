# Magnet: We Never Know How Text-to-Image Diffusion Models Work, Until We Learn How Vision-Language Models Function

arXiv 2409.19967, read from the full PDF (27 pages, appendix included). 38th Conference on Neural Information Processing Systems (NeurIPS 2024), printed on p. 1.

## What it does

Magnet edits only the text embedding. For each object word in the prompt it adds a "positive binding vector" that pulls toward the object's own adjective and subtracts a "negative binding vector" that pushes away from the other objects' adjectives (p. 3 to p. 4). The vectors are differences of CLIP word embeddings taken from short prompts built around the object and its five nearest neighbour nouns, and their strengths come from how far the prompt's padding embedding has drifted from its first end-of-text embedding (p. 4 to p. 5). The UNet and sampler are untouched.

## Where it says leakage enters

The text encoder, specifically the end-of-text token and the padding tokens after it, carried into the image through cross-attention.

**The claim.** CLIP pads every prompt to 77 tokens with copies of the end-of-text token. Those padding embeddings carry a blend of every concept in the prompt, and in cross-attention their maps cover both objects even where the two object words' own maps do not overlap. The paper says this entanglement "explains color leakage and object sticking" (p. 3).

**Evidence 1: a measurement on the text encoder alone.** Over 60 object nouns and 7 colours, the paper compares each object's embedding with and without a colour in front of it, by Euclidean distance and cosine (Fig. 1, p. 2). The object's own word embedding moves a lot for some colour and object combinations ("black" against "white" sheep) and the end-of-text embedding moves less (p. 2). Appendix A.1 repeats this with 16 attributes, 960 embeddings and a PCA, on CLIP ViT-L/14 and ViT-H/14 (p. 14). This shows where the colour information sits in the text encoding. It does not measure leakage between two objects.

**Evidence 2: swapping embeddings.** For "red chair", "black sheep" and "blue apple", the paper regenerates with the coloured prompt's word embedding or its end-of-text and padding embeddings replaced by the uncoloured ones (Fig. 2a, p. 3; Fig. 14, p. 15). Removing the colour from the end-of-text and padding embeddings loses the colour for unnatural concepts. Splitting the padding into three groups shows that the colour lives mostly in the end-of-text token and the first 23 padding tokens (p. 15). These are a handful of images.

**Evidence 3: padding drift and attention overlap.** The cosine between the first end-of-text embedding and each later padding embedding falls with token index, from 1.0 to about 0.7 by the last padding token, and falls further for "blue apple" than for "red chair" (Fig. 2b, p. 3; values read from the plot). For two-object prompts ("a blue bird and a brown car"), the cross-attention maps of padding tokens cover both objects (Fig. 2c, p. 3). Appendix A.3 shows six failure types this way, including colour leakage, a horse head on a bag body, and a sheep lost to a car (Fig. 15, p. 16). This is qualitative: one prompt per failure type, no count of how often padding overlap goes with a failure.

So the evidence for the text-encoder location is one measurement on embeddings plus image demonstrations; the link from padding overlap to leakage is argued from examples.

## What it changes there

It replaces each object's word embedding before sampling starts (Eq. 4, p. 4):

$$
\hat c_{E_i} = c_{E_i} + \alpha_i \, v_i^{pos} - \beta_i \, v_i^{neg}
$$

$$
v_i^{pos} = \frac{1}{K} \sum_{k=1}^{K} \Big[ F\big(B_k^{(i)}, \tilde P_k^{pos}\big) - F\big(B_k^{(i)}, \tilde P_k^{uc}\big) \Big], \qquad
v_i^{neg} = \frac{1}{K} \sum_{k=1}^{K} \Big[ F\big(B_k^{(i)}, \tilde P_k^{neg}\big) - F\big(B_k^{(i)}, \tilde P_k^{uc}\big) \Big]
$$

$$
\alpha_i = e^{\lambda - \omega_i}, \qquad \beta_i = 1 - \omega_i^2, \qquad \omega_i = \cos\!\big(G(\tilde P_i^{pos}), H(\tilde P_i^{pos})\big)
$$

- $c_{E_i}$ is the CLIP word embedding of object $i$ inside the full prompt; $\hat c_{E_i}$ replaces it (p. 4).
- $B_k^{(i)}$ are the $K = 5$ nouns from a precomputed list of 614 whose word embeddings have the highest cosine with object $i$'s own word embedding (p. 5, p. 18).
- $F(B, \tilde P)$ is the word embedding of noun $B$ inside a short prompt: $\tilde P^{uc}$ is the noun alone, $\tilde P^{pos}$ is the noun with object $i$'s adjective $A_i$, $\tilde P^{neg}$ is the noun with another object's adjective $A_j$ (p. 4 to p. 5). With $M$ objects there are $M-1$ negative vectors per object (p. 4); how they are combined into one $v_i^{neg}$ in Eq. 4 is not printed.
- $G$ and $H$ pick the first end-of-text embedding and the last padding embedding of the positive prompt; $\omega_i$ is their cosine, so a lower $\omega_i$ (more drift) gives a stronger pull $\alpha_i$; $\lambda = 0.6$ (p. 4, p. 17).

**Layers and steps.** The edited embedding conditions the UNet at every step $t = T, \dots, 1$ (p. 5), so a fraction 1.0 of the 50 steps (p. 18), through every cross-attention layer. Whether the unconditional branch of classifier-free guidance is touched is not printed.

**Extra passes.** No extra UNet passes. The extra work is text-encoder passes for the short prompts: per object, $K$ neighbours times one unconditional, one positive and $M-1$ negative prompts (inferred from Eqs. 5 and 6). Measured cost: 6.81 s per image against 6.62 s for SD1.4, +2.9% (Table 2, p. 6), and the neighbour step adds 0.25 s per image at $K = 5$ (p. 20).

## Does it need training, a layout or extra inputs

Nothing is trained (p. 1). It needs the Stanza dependency parser to pair each object noun with its adjective, and the parser fails on phrasings like "a streetlight that is green" (p. 3, p. 17). It needs a list of 614 candidate nouns, generated with ChatGPT and checked by hand, encoded once per text encoder (p. 18); an LLM or BERT can be used instead to pick neighbours (p. 18). No boxes, masks or reference images (p. 1).

## Which model it runs on

SD1.4 with the CLIP ViT-L/14 text encoder, 50 steps, guidance scale 7.5, one RTX 3090 (p. 18). Resolution and sampler are not printed. FID is also reported on SD2.1 (p. 6). SD2.1, SDXL and PixArt (T5 encoder) are shown with one qualitative prompt each; the CLIP models use the same hyperparameters and equations, and PixArt uses a fixed strength (Fig. 10, p. 8 to p. 10). How SDXL's two text encoders and its pooled embedding are handled is not described. The paper shows it running alongside layout guidance with boxes, ControlNet with a depth map, and Attend-and-Excite (p. 8 to p. 9). Code is released (p. 1).

## What it measured on similar subjects

No result is reported per subject type. The CC-500 subset is built with a split into two living objects (animals and plants), one living and one non-living, and two non-living, 80 prompts each, 240 prompts and 10 images per prompt (p. 5), but Tables 1 and 2 pool all three (p. 6). Two-animal prompts appear only as single images: "a pink cat and a brown sheep" where human raters marked neither animal present for SD and both present with the right colours for Magnet (Fig. 17b, p. 19), and "a yellow cat and a brown sheep" where Magnet and SD gave similar images (Fig. 22, p. 22).

## Which pairs leak and why

**Noun-to-noun CLIP text similarity.** The paper computes it but never relates it to a binding failure. The only cosine between two nouns' CLIP text embeddings is the neighbour search: a target noun's word embedding against 614 candidate nouns' word embeddings, top 5 kept (p. 5, p. 18). It is used to pick neighbours that share the target's "attribute bias" ("wolf" and "lion" for "bear", p. 18), and the paper reports no number linking it to which object pairs leak. Fig. 22 says Magnet changes little for "two unrelated concepts" (p. 22), which is a qualitative remark without a similarity value. So whether noun-noun text cosine predicts binding failure is not measured.

**What the paper does tie to failure.** Attribute bias: objects with a strongly preferred colour (banana yellow, broccoli green, tiger yellow) bind unnatural colours badly and need a stronger push (p. 2, p. 14, p. 21). Drift between the first end-of-text and the last padding embedding, $\omega$, is the per-concept strength signal. Over 19,648 adjective-noun prompts (614 objects times 32 attributes) $\omega$ peaks at 0.66 and 0.71 and becomes rare below 0.6 and above 0.82 (Fig. 16, p. 17). The paper argues a low $\omega$ marks a concept whose padding has "forgotten" the attribute (p. 4), but it does not count failures against $\omega$.

**Padding-token attention overlap** is shown on single examples for colour leakage and object sticking (p. 3, p. 16) and is never measured across pairs.

## When in sampling leakage is set

Not measured. In the limitations the paper suspects "the color layout has been determined in the early stage", to explain images where Magnet puts the right colours on the wrong objects' positions (p. 21). This is a guess with no step numbers.

## Numbers

| Metric (scale, direction) | Magnet | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| Attribute alignment, human count of objects with the right attribute (%, higher better) | 74.0 | 66.2 (Attend-and-Excite); 63.9 (StructureDiffusion); 59.1 (SD1.4) | CC-500, 240 prompts × 10 images | SD1.4 | p. 6, Table 2 |
| Object existence, human count (%, higher better) | 68.6 | 84.6 (Attend-and-Excite); 65.8 (SD1.4) | CC-500 | SD1.4 | p. 6, Table 2 |
| Object detected by GroundingDINO (%, higher better) | 76.5 | 84.3 (Attend-and-Excite); 72.1 (StructureDiffusion) | CC-500 | SD1.4 | p. 6, Table 2 |
| Seconds per image (lower better) | 6.81 (+2.9%) | 6.62 (SD1.4); 13.4 (Attend-and-Excite) | 200 images | SD1.4, RTX 3090 | p. 6, Table 2 |
| Share of human votes for clearer attribute separation (votes normalised to 100, higher better) | 29.43 | 18.57 (Attend-and-Excite); "no winner" 27.44 | CC-500 | SD1.4 | p. 6, Table 1 |
| Share of human votes for clearer object separation (votes normalised to 100) | 24.86 | 26.29 (Attend-and-Excite); "no winner" 34.28 | CC-500 | SD1.4 | p. 6, Table 1 |
| Share of human votes for clearer attribute separation (votes normalised to 100) | 27.14 | 19.71 (Attend-and-Excite); "no winner" 34.29 | ABC-6K, 600 prompts | SD1.4 | p. 6, Table 1 |
| FID on 10k COCO captions (lower better) | 18.92 / 19.20 | 19.04 (SD1.4) / 19.76 (SD2.1) | MS-COCO | SD1.4 / SD2.1 | p. 6, text |
| Votes for clearer separation with and without neighbours (normalised to 100): object, attribute | 27.1, 28.6 with; 9.1, 6.4 without | "no winner" 61.5, 62.9 | ablation | SD1.4 | p. 8, Table 3 |

In the coarse comparison "no winner" is the largest single share in five of six columns; the exception is CC-500 attribute separation, where Magnet has 29.43 (p. 6, Table 1).

## What it suggests for this loop

- For the predictor step (inferred): the paper's account says the name-cosine predictor looks at the wrong tokens. It finds the object word embeddings "naturally separated" (p. 3) and the entanglement in the padding tokens. A text-side predictor in its spirit is the overlap, at CO3's corrected steps, of the padding tokens' cross-attention with each subject's region, or the drift $\omega$ of the joint prompt. Both can be computed from the same attention hooks the overlap predictor needs.
- For the predictor step (inferred): crow and raven not borrowing while cheetah and cougar do is compatible with this account, since the paper's failure signal is padding entanglement rather than how close the two names are. The paper gives no data on name similarity either way.
- For the one-fix step (inferred): Magnet is built around adjective and noun pairs. If the joint prompt names two animals with no adjectives, there is no $A_i$ for $\tilde P^{pos}$ and no other adjective for $\tilde P^{neg}$, so both vectors are zero and Magnet changes nothing. Using the other subject's name as the "negative attribute" would be a new method the paper does not test.
- For the one-fix step (inferred): it would run inside CO3's SDXL sampler without boxes and at almost no cost, since it only edits the conditioning. SDXL is shown on one image with no numbers, and the handling of SDXL's second encoder and pooled embedding is unstated.
- For Paper 2's method (inferred): the paper's evidence that the padding tokens carry both subjects suggests a cheap control: in CO3's joint-prompt pass, swap the padding and end-of-text embeddings for those of the single-subject prompts and see whether part borrowing drops on the leaking pairs.

## Closer work it cites

- "Get what you want, not what you don't: Image content suppression for text-to-image diffusion models", Li et al. 2024, arXiv 2402.05375 (p. 12). It edits the padded end-of-text embeddings without training to remove content, and Magnet leans on it for the claim that padding embeddings carry the prompt's semantics (p. 3, p. 16).
- "Uncovering the text embedding in text-to-image diffusion models", Yu et al. 2024, arXiv 2404.01154 (p. 13). Cited as a text-embedding study for controlling objects (p. 9); the title suggests a training-free analysis of the same embeddings (inferred).
- "When are lemons purple? The concept association bias of vision-language models", Tang et al., EMNLP 2023, no arXiv id printed (p. 11). Magnet cites it for vision-language models lacking compositional understanding (p. 1); from its title it studies CLIP attaching an attribute to the wrong concept, the text-side half of leakage (inferred).
- "Dense text-to-image generation with attention modulation", Kim et al., ICCV 2023, no arXiv id printed (p. 13). A training-free attention change against leakage; it needs a layout (p. 9).
