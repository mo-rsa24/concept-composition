# Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis

arXiv 2212.05032, read from the full PDF (21 pages, appendix included). Published as a conference paper at ICLR 2023 (printed on every page).

## What it does

The method (called StructureDiffusion) parses the prompt into noun phrases, encodes each noun phrase separately with the frozen CLIP text encoder, and pastes each encoding back into the full prompt's token positions (p. 4). Inside every cross-attention layer it keeps the attention map computed from the full prompt's keys and averages the outputs obtained with the full prompt's values and with each noun phrase's values (p. 4). A second variant also builds one attention map per noun phrase, to recover objects that go missing (p. 5).

## Where it says leakage enters

A combination: the text encoder first, cross-attention maps second.

**Text encoder (the paper's main claim).** CLIP's text encoder uses a causal mask, so a later token carries the meaning of the tokens before it, and "red" in "a green bag and a red apple" carries "green" (p. 3). The evidence is one swap experiment: the tokens of "a red apple" are replaced by the same tokens taken from the encoding of "a green bag and a red apple", the seed is kept, and the apple comes out with a solid green region (Fig. 11 left, p. 16). This is a single qualitative demonstration with no count over prompts or seeds.

**Cross-attention maps (a second cause).** For "a furry bear watching a bird", the attention map of the token "bird" lights up the bear's mouth in five layers, and the bear gets a beak-like mouth (Fig. 11 right, p. 16). This case is the closest in the paper to the project's leakage, since one animal borrows a part of the other. It is again one image.

**The paper contradicts itself on the second cause.** On p. 14 it argues that attention maps are not the critical reason for colour leakage and that "inaccurate attention distributions contribute little to the final image content"; on p. 16 it says inaccurate attention maps lead to the beak on the bear.

**Padding tokens.** An ablation that drops or keeps the 77-token padding shows the padding embeddings are "fully contextualized" with the whole prompt's meaning: keeping only the nearest padding embedding still preserves the layout and semantics (Fig. 7, p. 8; p. 9). This is an argument from four example images and an IS/FID comparison whose numbers are not printed for each pattern.

## What it changes there

It changes the value path of every cross-attention layer, keeping the keys and the attention map from the full prompt (p. 4):

$$
M_t = \mathrm{Softmax}\!\left(\frac{Q_t K_p^{\top}}{\sqrt{d}}\right), \qquad
O_t = \frac{1}{k+1} \sum_{i \in \{p, 1, \dots, k\}} M_t \, V_i, \qquad V_i = f_V(\overline{W}_i)
$$

- $Q_t = f_Q(X_t)$ is the query from the image feature map $X_t$ at step $t$; $K_p = f_K(W_p)$ is the key from the full prompt's CLIP encoding $W_p$; $d$ is the per-head feature dimension (p. 3).
- $M_t$ is the attention map of shape heads × pixels × 77 tokens (p. 3).
- $\overline{W}_i$ is noun phrase $c_i$'s own CLIP encoding $W_i = \mathrm{CLIP}_{text}(c_i)$, with its tokens between `<bos>` and `<pad>` inserted into the full prompt's sequence at their positions (p. 4). $\overline{W}_p = W_p$.
- $k$ is the number of noun phrases found by the parser; $O_t$ is the layer output fed onward (p. 4).

The variant for missing objects uses one key per noun phrase, $M_t^i = f_M(Q_t, f_K(W_i))$, and averages $M_t^i V_i$ (Eqs. 5 and 6, p. 5; the printed Eq. 6 writes $V_k$, which reads as a typo for $V_i$ (inferred)).

**Layers and steps.** Every cross-attention layer in the UNet, at every one of the 50 sampling steps, so a fraction 1.0 of the steps (Algorithm 1, p. 5; 50 PLMS steps, p. 13).

**Extra passes.** No extra UNet forward or backward pass. The text encoder runs $k$ more times once per prompt, and each cross-attention layer does $k$ more value products; the paper says cross-attention cost grows with the number of noun phrases (p. 9) and calls the extra cost "trivial" (footnote, p. 4). No seconds per image are printed in this paper. Magnet's paper measured it at 7.94 s against 6.62 s for plain SD1.4, a 20.0% increase (Magnet, p. 6).

## Does it need training, a layout or extra inputs

Nothing is trained (p. 2, p. 5). It needs a parser: the Stanza constituency parser for noun phrases by default, or a scene-graph parser (p. 13, p. 8). It needs no boxes, masks, LLM or reference image (p. 2).

## Which model it runs on

Stable Diffusion v1.4, 50 PLMS steps, guidance scale 7.5, fixed seed shared with the baseline, no "a photo of" prefix (p. 13). Resolution is not printed. The code is released as supplementary material (p. 10). The paper makes no claim about other models.

## What it measured on similar subjects

No similar-subject split. CC-500 prompts have the form "a [colorA] [objectA] and a [colorB] [objectB]" and pool every object pair in one table (p. 7). Some figure prompts pair two animals ("a brown dog and a blue horse", "a red bird and a green banana"; Fig. 13, p. 18), and one goat counting example appears (Fig. 4, p. 6), all qualitative.

## Which pairs leak and why

Token position: a later token absorbs the meaning of earlier tokens through the causal mask (p. 3), shown on one prompt (Fig. 11, p. 16). Padding tokens carry the whole prompt (p. 9). Nothing is measured per pair, and text-embedding similarity between the two nouns is not measured.

## When in sampling leakage is set

Not measured. Attention maps are shown at equal intervals from step 50 to step 1 for two prompts, and for "a tall brown building with a tall green clock tower behind" the method makes the attention on the clock region form earlier (Fig. 9, p. 15). No step is named as the one where leakage or layout is fixed.

## Numbers

| Metric (scale, direction) | StructureDiffusion | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| Images with both objects, human label (% of images, higher better) | 38.0 | 34.5 (SD1.4) | CC-500 | SD1.4 | p. 7, Table 2 |
| Images with both objects in the right colours, human label (%, higher better) | 22.7 | 20.6 (Composable Diffusion); 19.2 (SD1.4) | CC-500 | SD1.4 | p. 7, Table 2 |
| Images with zero or one object, human label (%, lower better) | 62.0 | 65.5 (SD1.4) | CC-500 | SD1.4 | p. 7, Table 2 |
| Images with both objects, GLIP detector (%, higher better) | 31.2 | 31.0 (SD1.4) | CC-500 | SD1.4 | p. 7, Table 2 |
| Images where human labels agree with GLIP (%) | 47.6 | 46.4 (SD1.4), 48.9 (Composable) | CC-500 | SD1.4 | p. 7, Table 2 |
| Head-to-head alignment vote: win / lose / tie (% of pairs) | 42.2 / 35.6 / 22.2 | against SD1.4 | ABC-6K, 1500 pairs after removing the 20% most similar pairs | SD1.4 | p. 6, Table 1 |
| Head-to-head alignment vote: win / lose / tie (%) | 31.8 / 27.7 / 38.9 | against SD1.4 | CC-500 | SD1.4 | p. 6, Table 1 |
| Head-to-head alignment vote: win / lose / tie (%) | 46.5 / 30.1 / 22.8 | against Composable Diffusion | CC-500, 600 images | SD1.4 | p. 6, Table 1 |
| FID on 10,000 COCO captions (lower better) | 17.9 | 18.0 (SD1.4) | MS-COCO | SD1.4 | p. 8, text |

The head-to-head numbers on ABC-6K exclude the 20% of image pairs most similar to SD's output, and the paper says the gain "could be compromised" on the full set (p. 17).

## What it suggests for this loop

- For the predictor step (inferred): the paper places colour leakage in the text encoder through token order, and the name-cosine predictor encodes each name alone, so it cannot see this order effect. Encoding each name inside the joint prompt and measuring how much the second name's token moves would test the paper's claim directly.
- For the predictor step (inferred): the paper separates layout (carried by keys and attention maps) from content (carried by values) (p. 4, p. 14). The cross-attention overlap predictor then measures layout; part borrowing that travels through values would show in the $d_t$ map cosine and could be missed by the attention overlap.
- For the one-fix step (inferred): the fix needs no boxes and no extra UNet passes, and CO3 already has single-subject prompts, so the two subject names are the noun phrases without any parser. On SDXL the noun phrases would have to be encoded with both text encoders, and the pooled text embedding SDXL also uses would stay from the full prompt; neither is tested in the paper.
- For the one-fix step (inferred): the measured gain is small (both objects present in 38.0% against 34.5% of CC-500 images) and an independent human study rated it close to plain SD (Magnet, p. 6 Table 1, p. 7), so it is a weak candidate beside Bounded Attention.
- For Paper 2's method (inferred): the paper's split between attention maps (where things go) and values (what they look like) is a way to state the project's finding that CO3 fixes the two bodies but leaves part borrowing: CO3's correction may fix the layout while the joint prompt's values still mix the subjects.

## Closer work it cites

None that fits the brief's filter. The paper compares against Composable Diffusion (Liu et al. 2022, arXiv 2206.01714, p. 6 and p. 11), which sums the noise predictions of separate prompts. On CC-500 it left an object out more often than plain SD1.4: zero or one object in 69.7% of images against 65.5% (p. 7, Table 2). It is listed in the row file for that number, since CO3 also combines noise predictions.
