# Diffusion Self-Guidance for Controllable Image Generation

arXiv 2306.00986, read from the full PDF (17 pages, appendix included). The PDF prints "Preprint. Under review." (p. 1).

## What it does

Self-guidance adds the gradient of a hand-written energy to the noise prediction during sampling (p. 3). The energy is built from the model's own cross-attention maps and activations, so no extra model is trained (p. 1, p. 4). Position, size, shape and appearance of a named word can then be moved, held fixed or copied from another image (p. 4 to p. 8).

## What the classifier or loss reads

There is no classifier. The loss reads these quantities from the ordinary forward pass $\epsilon_\theta(z_t; t, y)$ (p. 4).

**Attention maps.** $\mathcal{A}_{i,t} \in \mathbb{R}^{H_i \times W_i \times K}$ are softmax-normalized cross-attention maps at layer $i$ and step $t$, with $K$ text tokens. An object at token indices $k$ uses channels $\mathcal{A}_{i,t,\cdot,\cdot,k}$ (p. 4). The softmax axis is not printed. Softmax over tokens at each pixel is the usual form (inferred).

**Layers.** Centroid, size and shape use all 36 cross-attention layers in Imagen's encoder, bottleneck and decoder, at 8 x 8, 16 x 16 and 32 x 32 (p. 13). The per-layer losses are averaged over layers (Eqn. 9, p. 5).

**Position (Eqn. 5, p. 4).** The centre of mass of the raw attention channel:

$$
\mathrm{centroid}(k)=\frac{1}{\sum_{h,w}\mathcal{A}_{h,w,k}}
\begin{bmatrix}\sum_{h,w} w\,\mathcal{A}_{h,w,k}\\[2pt] \sum_{h,w} h\,\mathcal{A}_{h,w,k}\end{bmatrix}
$$

The first entry is horizontal and the second is vertical. The worked targets such as $(0.3, 0.5)$ imply $h$ and $w$ are scaled to the 0 to 1 range, though the paper never defines the grid (p. 4, inferred). Centroids skip the threshold because binarizing is "empirically less important" for them (p. 13).

**Size (Eqn. 6, p. 4, redefined by Eqns. 14 to 16, p. 13).** The fraction of the canvas covered by the soft-thresholded map:

$$
\mathrm{normalize}(X)=\frac{X-\min_{h,w}(X)}{\max_{h,w}(X)-\min_{h,w}(X)}
$$

$$
\mathcal{A}^{\mathrm{thresh}}=\mathrm{normalize}\big(\mathrm{sigmoid}(s\cdot(\mathrm{normalize}(\mathcal{A})-0.5))\big),\qquad s=10
$$

$$
\mathrm{size}(k)=\frac{1}{HW}\sum_{h,w}\mathcal{A}^{\mathrm{thresh}}_{h,w,k}
$$

The minimum and maximum are taken per channel (p. 13).

**Shape (Eqn. 7, p. 4).** The thresholded map itself, compared with a target by an L1 distance that is averaged spatially (p. 4, p. 13):

$$
\mathrm{shape}(k)=\mathcal{A}^{\mathrm{thresh}}_{k}
$$

**Appearance (Eqn. 8, p. 5).** Activations $\Psi$ averaged inside the object's soft mask:

$$
\mathrm{appearance}(k)=\frac{\sum_{h,w}\mathrm{shape}(k)\odot\Psi}{\sum_{h,w}\mathrm{shape}(k)}
$$

$\Psi$ comes from the penultimate decoder layer and the final cross-attention operation. A stop-gradient on the attention means only the activations receive gradient in this term (p. 13).

## What it trained on

Nothing is trained (p. 1). The hand-written losses are L1 distances between each property and a target. A target is a user value, a transformed copy of the original property, or the property from another image (p. 4 to p. 6).

## How its gradient enters

The update as printed in Eqn. 4 (p. 3):

$$
\hat\epsilon_t=(1+s)\,\epsilon_\theta(z_t;t,y)-s\,\epsilon_\theta(z_t;t,\varnothing)+v\,\sigma_t\,\nabla_{z_t} g(z_t;t,y)
$$

Here $s$ is the classifier-free guidance strength, $v$ is the self-guidance weight, $\sigma_t$ converts a score to a noise prediction, and $g$ is the energy (p. 3).

**Model and steps.** All experiments use Imagen at 1024 x 1024 (p. 5) with $N = 1024$ DDPM steps (p. 13). The first $3N/16$ steps use self-guidance, the last $N/32$ do not, and the remaining $25N/32$ alternate (p. 13). That guides about 592 of 1024 steps, or 57.8% (inferred, 192 plus half of 800).

**Scale.** $v = 7500$ because per-pixel gradients are often $10^{-7}$ to $10^{-6}$ (p. 13). Term weights are tuned by edit, with total weight kept below about 5 to avoid artifacts (p. 17). Layout weights sit 1 to 2 orders of magnitude above appearance weights (p. 17). The paper prints no cap or normalization on the gradient norm.

## Unseen concepts

This paper does not test held-out concepts. No learned classifier exists to hold categories out from.

## Word leakage and confounds

The loss selects maps by token index, so it depends on the word being in the prompt. The paper states that this precludes control of any object absent from the text (p. 9).

**Attention can carry appearance.** Guiding the shapes of many tokens "occasionally betrays the appearance of an object", which the authors attribute to hidden high-frequency patterns across attention channels (p. 9). Corrupting target shapes with Gaussian noise helps reduce this (p. 14).

**Appearance features carry layout.** High appearance weights leak object position (p. 9).

**Interacting objects stay entangled.** Correlated attention maps hinder independent control of objects that interact (p. 9).

**Function words.** Words such as "a photo of" are often used for message passing and have layout-correlated attention, so the authors avoid guiding their layouts (p. 14).

**Prompt selection.** The protocol first verifies that the model already makes compelling samples for the prompt without self-guidance (p. 17). The results therefore mostly edit objects that already exist (inferred).

## Numbers

The paper reports no quantitative metric and no baseline score. Comparisons with ControlNet, Prompt-to-Prompt and DreamBooth are figures only (p. 6, p. 8).

| Metric (scale, direction) | This method | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| None reported | Qualitative figures only | None scored | Hand-chosen prompts | Imagen, 1024 x 1024 | pp. 5 to 8, no table |

Settings a reimplementation would need:

| Setting | Value and meaning | Page |
|---|---|---|
| $v$ | 7500, overall multiplier on the energy gradient | p. 13 |
| $N$ | 1024 DDPM steps; 256 and 512 also work with occasional weight changes | p. 13 |
| Threshold sharpness $s$ | 10, inside the sigmoid of Eqn. 15 | p. 13 |
| Attention layers | 36, at 8 x 8, 16 x 16 and 32 x 32 | p. 13 |
| Move weights, Eqn. 9 | $w_0 \in [0.5, 2]$ other shapes, $w_1 \in [0.03, 0.3]$ appearance, $w_2 \in [0.5, 5]$ moved shape | p. 14 |
| Move weights, Eqn. 17 | $w_2 \in [0.5, 2]$ size, $w_3 \in [1, 3]$ centroid | p. 15 |

## What to copy, what to avoid

- Copy Eqn. 5 as the differentiable position property, on the raw map (inferred).
- Copy Eqns. 14 to 16 for size, and add an absolute signal such as the raw peak or raw mass. Min-max normalization stretches even a faint, flat channel to the range 0 to 1, so $\mathcal{A}^{\mathrm{thresh}}$ alone can report a sizeable blob for a subject that was never drawn (inferred). A presence checker built only on it could miss exactly our failure.
- Guard the denominator of Eqn. 14, which is zero for a constant map (inferred).
- Test whether attention-only checkers still identify species. The p. 9 finding that attention patterns can encode appearance weakens the assumption that word maps are species-free. Blur the maps or add Gaussian noise during training, following the p. 14 mitigation, and measure species decoding from the maps (inferred).
- Avoid relying on a centroid or separation term to create a missing subject. Our attention-separation push drew two animals on 0 of 8 SDXL seeds, and this paper's edits start from prompts that already work (p. 17) (inferred).
- Copy the collage idea of Eqn. 12 (p. 6) as a candidate: take each subject's target shape from its own single-subject pass, which PoE already computes. The paper warns that positions from separate images can be incompatible (p. 6, p. 7), so the targets would need placing apart (inferred).
- Check the softmax before reusing these formulas under PoE. In one joint prompt the tokens compete at each pixel. In two single-subject passes the cat and dog maps are normalized separately and do not compete, so overlap and size mean something different (inferred).
- Keep our cap of 0.4 of PoE's norm in place of a fixed $v$. The value 7500 is tied to Imagen's gradient magnitudes (p. 13) and has no reason to transfer to SDXL (inferred).
- Compute properties at more than one resolution. The paper averages 8 x 8, 16 x 16 and 32 x 32, while our failed push used 32 x 32 only (inferred).

## Closer work it cites

None.
