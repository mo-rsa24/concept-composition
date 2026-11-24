# CONFORM: Contrast is All You Need For High-Fidelity Text-to-Image Diffusion Models

arXiv 2312.06059, read from the full PDF (27 pages, appendix included). No venue is printed on the PDF.

## What it does

CONFORM nudges the noisy latent during the first half of sampling so that each subject's cross-attention map moves away from the other subject's map and toward the maps of its own attribute words (p. 2, p. 4). The push comes from a contrastive (InfoNCE) loss over cosine similarities between 16×16 cross-attention maps, followed by one gradient step on the latent before the denoising step (p. 3, p. 4). Nothing is trained, and the same loss runs on Stable Diffusion 1.5 and on the pixel-space Imagen model (p. 1, p. 6, p. 7).

## Where it says leakage enters

Cross-attention. For "a bear and an elephant" the paper shows the two subjects' cross-attention maps overlapping and says the overlap "makes it difficult to differentiate the two subjects and leads the model to produce only elephants" (Fig. 2a, p. 2). For "a purple crown and a yellow suitcase" it shows the two colour maps misaligned and the colours mixing (Fig. 2b, p. 2). It also shows the attribute maps scattering as the backward process runs (Fig. 3, p. 2). It cites DAAM for the finding that adjectival modifiers and co-hyponyms (words from the same category) give entangled features in cross-attention maps (p. 3).

All of this is argument from visualised attention maps on single examples. The paper never measures overlap across many prompts and relates it to outcomes. Its miscounting failures are blamed on the CLIP text encoder, again by argument with a citation (p. 2).

The failures CONFORM targets are a missing subject, colour on the wrong object and wrong counts (p. 1). The closest of these to this loop's leakage is colour binding; part borrowing between two bodies that are both present is not studied (inferred).

## What it changes there

The loss acts on cross-attention maps of the subject and attribute tokens, for one positive pair $(j, j^+)$:

$$
\mathcal{L} = -\log \frac{\exp\!\big(\mathrm{sim}(A^{j}, A^{j^+})/\tau\big)}{\sum_{n \in \{j^+, j^-_1, \dots, j^-_N\}} \exp\!\big(\mathrm{sim}(A^{j}, A^{n})/\tau\big)},
\qquad
\mathrm{sim}(u, v) = \frac{u^\top v}{\lVert u \rVert\,\lVert v \rVert}
$$

$$
z_t' = z_t - \alpha_t \nabla_{z_t} \mathcal{L}
$$

- $A^{j}$ is the 16×16 cross-attention map of token $j$, compared as a vector by cosine similarity (p. 3, p. 4).
- $j^+$ is a token in the same group: an object and its attribute words form one group, and each token's own map from the previous step $t+1$ also counts as positive (p. 4).
- $j^-_1 \dots j^-_N$ are all tokens of the other groups, at step $t$ and at step $t+1$ (p. 4).
- $\tau$ is the temperature, set to 0.5; the loss is averaged over all positive pairs (p. 4, p. 6).
- $z_t$ is the noisy latent and $\alpha_t$ the step size, set to 20 (p. 4, p. 6).

For a prompt with two animals and no attribute words, the only positives are each animal's map at $t$ and $t+1$, and the negatives are the other animal's maps, so the loss repels the two subject maps from each other (inferred from p. 4).

Layers: the 16×16 cross-attention maps, chosen because Prompt-to-Prompt found them most semantically meaningful (p. 3). The paper does not say which blocks or how heads are combined.

Timesteps: optimisation runs from iteration 0 and stops after iteration 25 of 50, so the first half of sampling (fraction 0 to 0.5), because optimising at every step gave artifacts (p. 6, p. 12). At iterations 0, 10 and 20 it repeats the optimisation "multiple times" before the denoising step (p. 6); the number of repeats is not printed.

Passes per optimised step: one UNet forward to read $A_t$, one backward through the UNet for $\nabla_{z_t}\mathcal{L}$, then one forward on $z_t'$ to denoise (Algorithm 1, p. 11). The step $t+1$ maps are kept from the previous step, so they cost no pass (p. 11). Runtime is about 20 seconds "per prompt" on an NVIDIA L4 with SD 1.5 and 50 iterations (p. 6, p. 11); with 64 seeds per prompt this reads as per image (inferred). No plain SD runtime is printed to compare against.

## Does it need training, a layout or extra inputs

Nothing is trained (p. 1). It needs a dictionary of subject token indices and, for each subject, its attribute token indices (Algorithm 1 input, p. 11); the paper does not say how this grouping is produced. It needs no boxes, masks, LLM or reference image. On Imagen, where T5 splits a word into pieces ("zebra" into "ze", "bra"), the pieces are treated as positives (p. 7).

## Which model it runs on

Stable Diffusion v1.5 (p. 6, p. 11) and Imagen with its T5-XL text encoder (p. 3, p. 7). Resolution, sampler and guidance scale are not printed; sampling uses 50 iterations (p. 6). The paper calls the method model-agnostic (p. 3, p. 8). It has no SDXL result.

SDXL at 1024×1024 has cross-attention at 64×64 and 32×32 latent resolution only, so the 16×16 maps the paper uses do not exist there; the 32×32 level is the closest analogue (inferred).

## What it measured on similar subjects

The Animal-Animal set uses the template "a [animalA] and a [animalB]", 66 prompts, 64 seeds per prompt (p. 6, p. 11). Results with SD 1.5:

- TIFA, the share of GPT-3.5-written questions about the prompt that UnifiedQA-v2 answers correctly on the image (0 to 1, higher is better): CONFORM 0.95, Divide & Bind 0.93, Attend-and-Excite 0.92, SD 0.68 (Table 2, p. 7; metric p. 8).
- Text-text similarity, CLIP similarity between the prompt and a BLIP caption of the image (higher is better): CONFORM 0.82, Divide & Bind 0.81, Attend-and-Excite 0.80, SD 0.76 (Table 1, p. 7).
- User study, 25 respondents picking the best image per prompt and seed: CONFORM 72% of votes, Attend-and-Excite 14.75%, Divide & Bind 8.25%, SD 5% (Table 3, p. 8).
- CLIP image-text similarity (Fig. 7, p. 7, read from the bar chart): full prompt about 0.34 for CONFORM against about 0.31 for SD; minimum over the two subjects about 0.25 against about 0.22.

With Imagen, Animal-Animal TIFA is 0.84 for Imagen, 0.84 for Attend-and-Excite and 0.84 for CONFORM, so no gain (Table 2, p. 7).

Every one of these scores asks whether both named animals appear. None asks whether one animal carries the other's parts (inferred). The qualitative examples "a cat and a dog on the bridge" and "a bird and a horse" show SD drawing two cats or two horses and CONFORM turning one of them into the missing animal (Fig. 5, p. 5; p. 6, p. 7). That is subject replacement, a different failure from part borrowing.

## Which pairs leak and why

Not measured per pair. The paper argues that overlapping subject maps let one subject take over (bear and elephant, p. 2). It cites DAAM for co-hyponyms entangling cross-attention features (p. 3), which predicts more leakage for two words of the same category. It also reports that when the initial attention map largely excludes an object, CONFORM struggles to recover it (Fig. 8a, p. 8), and that on Imagen the refinement can split one object into separated parts ("a bird and an elephant", Fig. 8b, p. 8).

## When in sampling leakage is set

Not measured as a number. Fig. 3 (p. 2) shows attribute attention dissolving over the backward steps in SD. Optimising after iteration 25 of 50 caused artifacts, and refinement at iterations 0, 10 and 20 "allows the model to make necessary adjustments to the attention maps" (p. 12). The limitation in Fig. 8a (p. 8) says the starting attention map limits what the optimisation can fix, which points to the earliest steps (inferred).

## Numbers

| Metric (scale, direction) | CONFORM | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| TIFA (0 to 1, higher better) | 0.95 | D&B 0.93 (SD 0.68) | Animal-Animal, 66 prompts × 64 seeds | SD 1.5 | p. 7, Table 2 |
| TIFA | 0.94 | A&E and D&B 0.91 | Animal-Object | SD 1.5 | p. 7, Table 2 |
| TIFA | 0.88 | D&B 0.83 | Object-Object | SD 1.5 | p. 7, Table 2 |
| TIFA | 0.74 | D&B 0.73 | Multi-Object | SD 1.5 | p. 7, Table 2 |
| TIFA | 0.84 | Imagen 0.84 | Animal-Animal | Imagen | p. 7, Table 2 |
| Text-text CLIP similarity (higher better) | 0.82 | D&B 0.81 (SD 0.76) | Animal-Animal | SD 1.5 | p. 7, Table 1 |
| User study votes (% of 25 respondents' picks) | 72% | A&E 14.75% | Animal-Animal, 10 prompts × 4 seeds overall | SD 1.5 | p. 8, Table 3 |
| CLIP-full / CLIP-min, ablation without the step $t+1$ maps | 0.36 / 0.26 | without $t+1$: 0.35 / 0.25 (SD 0.33 / 0.24) | small ablation prompt set | SD 1.5 | p. 12, Table 5 |

## What it suggests for this loop

- Predictor step: CONFORM's similarity is the cosine between two subjects' flattened cross-attention maps (Eq. 4, p. 4). Computing the cross-attention overlap predictor as this same cosine at CO3's corrected steps would make the predictor and a candidate fix the same quantity (inferred).
- Predictor step: the paper also ties each map to its own map at the previous step. Recording the overlap at every step from 0 to 25, rather than at one step, would show whether leaking pairs separate late or never (inferred).
- One-fix step: CONFORM needs no boxes, only the token indices of the two subject names, so it could run on CO3's leaking pairs as a no-layout alternative to Bounded Attention (inferred). Its cost is one extra UNet backward and forward per step over the first half of sampling, plus repeats at three steps, and its memory cost on SDXL at 1024×1024 is untested (inferred).
- One-fix step: the attention maps would have to come from the joint-prompt pass, the only pass in CO3 where both subject tokens appear together (inferred). Its benchmarks reward both animals being present, which CO3 already achieves, so the paper gives no evidence that it reduces part borrowing between two present bodies (inferred).
- Paper 2's method: the same contrastive loss written on the two subjects' $d_t$ maps instead of their attention maps would be the idea in the project's own quantity (inferred).

## Closer work it cites

- A-Star: Test-time attention segregation and retention for text-to-image synthesis, arXiv 2306.14544 (p. 3, p. 9). It minimises overlap between concepts' attention maps and their change across steps, a training-free attention change aimed at the same overlap.
- What the DAAM: Interpreting Stable Diffusion using cross attention, arXiv 2210.04885 (p. 3, p. 10). It is cited for co-hyponyms and adjectival modifiers producing entangled cross-attention features, an explanation of why same-category subjects leak.
- DenseDiffusion: Dense text-to-image generation with attention modulation, Kim et al., ICCV 2023, no arXiv id printed (p. 3, p. 9). It is a training-free attention change, but it needs a user layout.
