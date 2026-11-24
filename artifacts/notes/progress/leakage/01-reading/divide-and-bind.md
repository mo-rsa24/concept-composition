# Divide & Bind Your Attention for Improved Generative Semantic Nursing

arXiv 2307.10864 (v3), read from the full PDF (15 pages, supplementary material included). Published at the 34th British Machine Vision Conference 2023, as printed on every page.

## What it does

Divide & Bind changes the noisy latent at each of the first 25 of 50 sampling steps with a gradient step on a loss computed from Stable Diffusion's 16×16 cross-attention maps (p. 4, p. 6). The attendance loss raises the total variation of each object word's map, so the map grows several high-activation regions instead of one peak (p. 2, p. 5). The binding loss pulls each attribute word's map onto its object's map by minimising their Jensen–Shannon divergence (p. 5, p. 6). Nothing is trained.

## Where it says leakage enters

Cross-attention. The paper argues that as prompts get more complex, object tokens compete in the attention maps, and "the single excitation of one object token may overlap with others", which gives either a missing object or "hybrid objects, exhibiting features of both semantic classes", with a mixed dog-turtle as the example (p. 2). It supports this with Fig. 3 (p. 4), which shows the "dog" and "turtle" maps over the timesteps for "A dog and a turtle on the street, snowy scene": under Attend-and-Excite the dog's single peak on the right is later taken over by the turtle (p. 5). It also argues that cross-attention layers are "the only interaction between the text prompt and the diffusion model" (p. 2).

The evidence is an argument backed by visualised maps on single prompts. No overlap is measured across prompts. Colour leaking onto the wrong object is shown qualitatively (Fig. 4, p. 4; Fig. S.3, p. 14), and the binding loss moves the metrics by at most 0.006 on a 0 to 1 scale (Table S.3, p. 14). Miscounting is blamed on the CLIP text encoder, by argument (p. 9).

## What it changes there

$$
\min_{z_t} \mathcal{L}_{D\&B} = \mathcal{L}_{attend} + \lambda\, \mathcal{L}_{bind}, \qquad z_t' \leftarrow z_t - \alpha_t \nabla_{z_t} \mathcal{L}_{D\&B}
$$

$$
\mathcal{L}_{attend} = -\min_{s \in S} TV(A^s_t), \qquad
TV(A^s_t) = \sum_{i,j} \big|A^s_t[i+1,j] - A^s_t[i,j]\big| + \big|A^s_t[i,j+1] - A^s_t[i,j]\big|
$$

$$
\mathcal{L}_{bind} = \mathrm{JSD}\big(\tilde{A}^r_t \,\Vert\, \tilde{A}^s_t\big)
$$

- $z_t$ is the noisy latent at step $t$ and $\alpha_t$ the update rate (p. 4, p. 5).
- $S$ is the set of object tokens; $A^s_t[i,j]$ is the cross-attention value of object token $s$ at position $(i,j)$ (p. 3, p. 5).
- $TV$ is the total variation of the map: the summed absolute differences between neighbouring positions. Taking the minimum over objects targets the worst object, and the minus sign means the loss raises it (p. 5).
- $\tilde{A}^r_t$ and $\tilde{A}^s_t$ are the attribute map and its object's map after rescaling the attribute map to the object map's range and a spatial softmax, read as probability distributions over the $h \times w$ positions (p. 5, p. 6).
- $\lambda = 1$ when the prompt carries attribute words and $\lambda = 0$ otherwise (p. 6).

Layers: the 16×16 cross-attention maps (p. 6). The paper does not say which blocks or how heads are combined.

Timesteps: from $t = T = 50$ down to $t_{end} = 25$, the first half of a 50-step run (fraction 0 to 0.5) (p. 6). The paper chooses this because "the image semantics are determined by the initial denoising steps", citing other work (p. 6). The step size and any repeated refinement follow Attend-and-Excite's settings, which this paper does not restate (p. 6).

Passes per optimised step: one forward to read the attention maps, one backward through the UNet for the gradient, one forward on $z_t'$ to denoise (Algorithm 1, p. 15). On a V100 with 50 steps, SD takes about 13 s per image and Divide & Bind about 17 s, about 1.3 times as long (p. 15).

## Does it need training, a layout or extra inputs

Nothing is trained (p. 5). It needs the indices of the object tokens and of their attribute tokens, chosen by hand or extracted by GPT-3 with in-context examples (p. 6, p. 15); for the COCO caption sets GPT-3 chose them (p. 15). It needs no boxes, masks or reference image.

## Which model it runs on

Stable Diffusion; the version, resolution, sampler and guidance scale are not printed (p. 3, p. 15). Sampling uses 50 steps (p. 6, p. 15). The paper makes no claim about other models. CONFORM later says it is unclear how the binding term would apply when a text encoder splits one word into several tokens (CONFORM, arXiv 2312.06059, p. 7).

SDXL at 1024×1024 has no 16×16 cross-attention level, only 64×64 and 32×32, so the method would need its resolution choice redone (inferred).

## What it measured on similar subjects

Animal-Animal ("a [animalA] and a [animalB]", 66 prompts) and Animal-Scene (the same with a scene, 56 prompts) are the two-animal sets (Table 1, p. 6). The number of seeds per prompt is not printed. Because these prompts carry no attribute words, only the attendance loss runs on them (inferred from p. 6).

- TIFA, the accuracy of the mPLUG question-answering model on questions about the prompt (0 to 1, higher is better), Animal-Animal: Divide & Bind is 32% above SD and Attend-and-Excite 31% above SD, with SD at about 0.70 (Fig. 6, p. 8, read from the bar chart).
- Text-text similarity, CLIP ViT-B/16 similarity between the prompt and BLIP captions (higher is better), Animal-Animal: Attend-and-Excite 5.6% above SD and Divide & Bind 5.5% above SD (Fig. 6, p. 8, read from the bar chart). The supplementary table gives SD 0.77, Attend-and-Excite 0.80, Divide & Bind 0.81 (Table S.2, p. 14).
- CLIP image-text similarity, Animal-Animal: full prompt SD 0.312, Attend-and-Excite 0.333, Divide & Bind 0.331; minimum over the two animals SD 0.220, Attend-and-Excite 0.249, Divide & Bind 0.246 (Table S.1, p. 13).

The paper's own reading is that Divide & Bind "is on-par with A&E on Animal-Animal" (p. 7). Its gains are on longer prompts and on Multi-Object, where entities may share a category, as in "one cat and two dogs" (p. 6): TIFA 0.785 against Attend-and-Excite 0.755 and SD 0.647 (Table 2, p. 8).

These metrics ask whether each named animal is present. None scores a hybrid or a borrowed part (inferred).

## Which pairs leak and why

Not measured per pair. The paper argues that token competition grows with prompt complexity (p. 2) and that a single peak "can be easily taken over by the other competitor token" (p. 5). It cites DAAM (Tang et al. 2023) for the same phenomenon (p. 2). It also reports that rare combinations such as "a gray apple" stay hard because the model inherits data bias (p. 8, p. 9).

## When in sampling leakage is set

Not measured. The paper cites Liew et al. (2022) and Kwon et al. (2023) for semantics being set in the first denoising steps and optimises only the first 25 of 50 steps for that reason (p. 6). Its Fig. 3 description says the dog token explores both left and right regions "in the initial phase" before the turtle takes the left (p. 5); this is a qualitative reading of one prompt.

## Numbers

| Metric (scale, direction) | Divide & Bind | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| TIFA (0 to 1, higher better), gain over SD | +32% | A&E +31% (SD about 0.70) | Animal-Animal, 66 prompts | SD | p. 8, Fig. 6 (read from image) |
| Text-text CLIP similarity, gain over SD | +5.5% | A&E +5.6% | Animal-Animal | SD | p. 8, Fig. 6 (read from image) |
| Text-text CLIP similarity | 0.81 | A&E 0.80 (SD 0.77) | Animal-Animal | SD | p. 14, Table S.2 |
| CLIP full prompt / minimum object | 0.331 / 0.246 | A&E 0.333 / 0.249 (SD 0.312 / 0.220) | Animal-Animal | SD | p. 13, Table S.1 |
| TIFA | 0.785 | A&E 0.755 (SD 0.647) | Multi-Object, 30 prompts | SD | p. 8, Table 2 |
| TIFA | 0.840 | A&E 0.824 (SD 0.791) | COCO-Subject, 30 captions | SD | p. 8, Table 2 |
| Runtime per image (seconds, lower better) | 17 | SD 13 | 50 steps, V100 | SD | p. 15 |

## What it suggests for this loop

- Predictor step: the paper's account is that hybrids form where the two subjects' attention peaks overlap and one token takes over the other's region. That predicts the cross-attention overlap predictor should separate CO3's leaking pairs from the rest (inferred).
- Predictor step: total variation of each subject's map counts how many separate high-activation regions it has. It is cheap to log beside the overlap and would show whether a leaking subject's map has collapsed to one shared peak (inferred).
- One-fix step: Divide & Bind needs no boxes, only the subject token indices, at about 1.3 times the runtime on SD (p. 15). On two-animal prompts only its attendance term runs, and that term aims to make each animal present, which CO3 already achieves, so nothing in the paper says it would stop part borrowing between two present bodies (inferred).
- Paper 2's method: the binding term (JSD between a word's map and its subject's map) is a template for tying part words such as "beak" or "fur" to their own subject, if prompts named the parts (inferred).

## Closer work it cites

- What the DAAM: Interpreting Stable Diffusion using cross attention, Tang et al., ACL 2023 (p. 2, p. 11). No arXiv id is printed in this paper's reference list; CONFORM's list prints it as 2210.04885. It is cited for the same token-competition and hybrid phenomenon.
