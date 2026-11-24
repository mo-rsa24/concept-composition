# Be Decisive: Noise-Induced Layouts for Multi-Subject Generation

arXiv 2505.21488, read from the full PDF (14 pages, appendix included). No venue is printed on the PDF.

## What it does

It runs Bounded Attention on SDXL without any boxes from the user or an LLM. A small trained network reads the layout the initial noise already implies from the UNet's attention features, clusters it into one region per subject plus background, and Bounded Attention then masks attention between those regions (p. 3, p. 4). During the first 15 of 50 steps, gradient steps on the latent keep each region stable from one step to the next, so a subject that has been given a region keeps it (p. 5, p. 12).

## Where it says leakage enters

The paper places leakage in the attention layers, cross-attention and self-attention both, and adds that the attention mask only prevents it when the layout behind the mask stays stable across steps. So the place is a combination: attention, gated by layout.

The attention claim is an argument carried over from Bounded Attention: "Prior work [Dahary et al. 2025] identified the model's attention layers as the primary source of this leakage", where "features of semantically similar subjects are indiscriminately blended" (p. 3). This paper does not measure it again.

The layout claim rests on one qualitative example (Figure 5, p. 5). Without the stabilising guidance, the middle dove of "a parrot and two doves" sits where three clusters meet and changes cluster from step to step. Its lower body is first assigned to the parrot cluster, and the finished dove has a parrot-like tail (p. 5). The figure is an observation on one seed. No rate is given.

## What it changes there

**Bounded Attention masking at every denoising step.** In cross-attention each subject's pixels attend only to that subject's tokens. In self-attention each subject's pixels attend only to their own region and the background (p. 3). The mask $M^t$ comes from the predicted layout described below, so no external layout is needed (p. 3).

**Soft layout.** A readout network (lightweight convolutional heads after Readout Guidance) takes the cross-attention queries and self-attention keys of SDXL's decoder layers plus the time embedding, and outputs a $64 \times 64 \times 10$ feature map $S^t$ (p. 4, p. 12). Pixels whose 10-dimensional features are similar are predicted to belong to the same subject (p. 4).

**Hard layout.** K-Means splits $S^t$ into $k+1$ clusters, $k$ subjects and one background. The background is the cluster that overlaps the image border most. Each subject cluster is split again while its variance is above $\sigma^2_{cluster} = 0.025$ (p. 5, p. 12). After the first step ($t=T$) each cluster is given a subject name by the Hungarian algorithm on the average cross-attention map of each subject noun. At later steps the soft layouts of the previous $w = 30$ steps are stacked before clustering, and clusters are matched to the previous step's clusters by IoU (p. 5, p. 12).

**Decisive guidance.** After each denoising step the latent $z_{t-1}$ is updated by gradient descent so that its soft layout $S^{t-1}$ agrees with the previous hard layout $M^t$ (p. 5). The update follows Bounded Attention's form (p. 3), with the loss swapped:

$$
z_{t-1} \leftarrow z_{t-1} - \beta \, \nabla_{z_{t-1}} \mathcal{L}_{decisive}, \qquad
\mathcal{L}_{decisive} = \alpha_{cross}\mathcal{L}_{cross} + \alpha_{var}\mathcal{L}_{var} + \alpha_{dice}\mathcal{L}_{dice}
$$

$$
\mathcal{L}_{var} = \frac{1}{k+1}\sum_{j=0}^{k} \frac{1}{|M^t_j|} \sum_{x_i \in M^t_j} \mathrm{sim}^2\!\left(S^{t-1}[x_i], \mu^{t-1}_j\right), \qquad
\mu^{t-1}_j = \frac{1}{|M^t_j|}\sum_{x_i \in M^t_j} S^{t-1}[x_i]
$$

$$
P^{t-1}[x_i,\cdot] = \mathrm{softmax}\!\left(\left\{\mathrm{sim}\!\left(S^{t-1}[x_i], \mu^{t-1}_j\right)/\tau\right\}_{j=0}^{k}\right)
$$

- $z_{t-1}$ is the noisy latent after the denoising step, $\beta$ the step size of the latent update (p. 3).
- $\mathcal{L}_{cross}$ is Bounded Attention's cross-attention loss, which pulls each subject's cross-attention inside its region (p. 3, p. 5). Bounded Attention's self-attention loss $\mathcal{L}_{self}$ is dropped, which the paper says cuts runtime (p. 5).
- $\mathcal{L}_{var}$ is meant to keep each cluster's features close to their mean $\mu_j^{t-1}$ (p. 5). $M^t_j$ is cluster $j$ of the previous hard layout, $x_i$ a pixel, sim the cosine similarity (p. 4, p. 5). As printed, minimising a sum of $\mathrm{sim}^2$ would push features away from the mean, so the printed sign or form is probably a typo (inferred).
- $\mathcal{L}_{dice}$ is the Dice loss between $M^t$ and the soft assignment $P^{t-1}$; $\tau = 15$ is its temperature (p. 5, p. 12).
- Weights: $\alpha_{cross} = 0.3$, $\alpha_{var} = 0.21$, $\alpha_{dice} = 0.49$ (p. 12).

**Layers and steps.** The readout network reads the decoder attention layers (p. 4). Guidance runs during the first 15 of 50 denoising steps, which is the first 30% of sampling, with 5 gradient-descent iterations per step (p. 12). The masking runs inside the denoising steps; the paper does not say it stops after step 15 (inferred from Figure 3, p. 3).

**Extra passes.** The paper states 5 gradient iterations per guided step (p. 12). Each one needs the UNet's attention features of $z_{t-1}$ and a gradient back to $z_{t-1}$, so about 5 extra forward and backward UNet passes at each of 15 steps, 75 in all (inferred). Measured cost: about 77 seconds and 36 GB of VRAM per image on an A100, against 7 seconds and 8 GB for plain SDXL (p. 12).

## Does it need training, a layout or extra inputs

A readout network is trained. The data are about 1,500 SDXL images from random prompts that name 1 to 3 of the 20 MSCOCO classes used by Make It Count, with 1 to 10 instances each, segmented by GroundedSAM. Overlapping masks are filtered out, and there is no filtering on whether the image matches its prompt (p. 4, p. 12). Training uses a triplet loss on 50 pixel triplets per image, margin 0.5, for 5,000 steps at learning rate $10^{-4}$ (p. 4, p. 12). The paper says the network generalises to unseen classes, backgrounds and prompt structures (p. 12); no number backs this.

No boxes, masks, LLM or reference image are needed at sampling time (p. 2). The method does need the subject nouns and their count $k$ from the prompt, since it clusters into $k+1$ parts and labels clusters by each noun's cross-attention (p. 5). The paper does not say how the nouns are found; a parsed or hand-given subject list is required (inferred).

## Which model it runs on

SDXL, 50 steps, DDPM scheduler, classifier-free guidance scale 7.5 (p. 12). The resolution is not printed. The readout network is trained on SDXL's own features, so moving to another model means retraining it (inferred). The paper makes no transfer claim to other models. It shows one integration with a personalization method that injects features through cross-attention (Figure 7, p. 6).

## What it measured on similar subjects

**User study on visually similar animals.** ChatGPT wrote 25 prompts, each listing three to four visually similar but distinct animals with a background. Participants saw 5 images from this method and 5 from one competitor and chose the images with realistic compositions that match the prompt. There were 192 responses from 32 participants (p. 7). Table 1 reports the probability that a chosen image came from this method, on a 0 to 1 scale where 0.5 means no preference: 0.74 against plain SDXL, 0.87 against LLM+BA (Bounded Attention given an LLM's boxes), 0.96 against RPG, 0.89 against Ranni, 0.9 against LMD+ (p. 7). The 0.53 against Make It Count was measured on 5 extra prompts with one class in quantity (p. 7).

**Ablation on the same study.** On 10 of the prompts with 19 participants, the fraction of each configuration's images that users selected was 0.832 for the full method, 0.016 with no guidance, 0.442 without $\mathcal{L}_{cross}$, 0.447 without $\mathcal{L}_{var}$, and 0.105 without $\mathcal{L}_{dice}$ (Table 3, p. 12).

**Qualitative.** A polar, grizzly, panda and koala bear (Figure 8, p. 7), and a ferret, squirrel and crow across seeds 0 to 6 (Figure 11, p. 10), where LLM+BA gets 4 of 7 seeds right and this method 7 of 7 by the authors' reading (p. 6).

The paper says the automatic metrics it reports "fail to detect semantic leakage" because they come from models trained on real images (p. 7). There is no two-animal split and no per-pair breakdown.

## Which pairs leak and why

Not measured. The only statement is the argument carried from Bounded Attention that features of semantically similar subjects blend in attention (p. 3). Figure 5 points to a second cause for one image: a subject sitting on an unstable boundary between clusters inherits a neighbour's part (p. 5).

## When in sampling leakage is set

Not measured with numbers. The paper argues that low frequencies are set early, so the initial noise largely fixes the final layout (p. 1), and shows qualitatively that the layout read from the initial noise matches the final composition (Figure 2, p. 2). The soft layout is coarse at the first steps and finer later (p. 4; Figure 4 shows $t = 50, 45, 40, 35$). The design choices follow from this: subjects are named after the first step, and guidance runs only in the first 15 of 50 steps (p. 5, p. 12). Figure 5b shows a boundary that oscillates during early steps producing a hybrid (p. 5).

## Numbers

| Metric (scale, direction) | This method | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| User preference, probability a chosen image is this method's (0 to 1, 0.5 is parity) | 0.74 vs SDXL; 0.87 vs LLM+BA | lowest margin: 0.74 vs plain SDXL | 25 ChatGPT prompts, 3 to 4 similar animals; 32 participants | SDXL | p. 7, Table 1 |
| Color binding, BLIP-VQA (0 to 1, higher better) | 0.704 | 0.685 LLM+BA | T2I-CompBench, 200 prompts | SDXL | p. 12, Table 2 |
| Texture binding, BLIP-VQA (0 to 1, higher better) | 0.686 | 0.665 LLM+BA | T2I-CompBench, 200 prompts | SDXL | p. 12, Table 2 |
| Single-class count F1 by GroundedSAM (0 to 1, higher better) | 0.837 | 0.885 LMD+ | T2I-CompBench, 200 prompts | SDXL | p. 12, Table 2; p. 7 |
| Multi-class count F1 (0 to 1, higher better) | 0.723 | 0.898 LMD+ | T2I-CompBench, 200 prompts | SDXL | p. 12, Table 2 |
| Layout diversity, mean 1 minus IoU between seeds (0 to 1, higher more diverse) | 0.718 | 0.679 Ranni | 20 single-class prompts, 5 seeds | SDXL | p. 12, Table 2; p. 7 |
| Ablation, fraction of images selected (0 to 1, higher better) | 0.832 full | 0.447 without $\mathcal{L}_{var}$; 0.016 without guidance | 10 prompts, 19 participants | SDXL | p. 12, Table 3 |
| Time and memory per image, A100 | 77 s, 36 GB | 7 s, 8 GB plain SDXL | | SDXL | p. 12 |

## What it suggests for this loop

- For the predictor step (inferred): the paper labels each layout cluster after the first SDXL step by the cross-attention map of each noun (p. 5), so SDXL's cross-attention already separates two subjects at step 0 well enough to name regions. Cross-attention overlap at CO3's corrected steps is therefore measurable early. Figure 5b suggests one more predictor worth logging beside the three planned ones: how much the boundary between the two subjects' attention regions moves from step to step in the early window.
- For the predictor step (inferred): the paper gives nothing that separates crow and raven from duck and goose. Its only pair-level claim is that similar subjects blend, which the brief's crow and raven result already contradicts as a full explanation.
- For the one-fix step (inferred): this is Bounded Attention without boxes on the same model and step count (SDXL, 50 steps), which matches CO3's setting. It needs a readout network trained on SDXL decoder features, about 1,500 segmented images and 5,000 steps, and roughly 75 extra forward and backward passes per image (11 times the runtime). If Bounded Attention is run with fixed boxes first and the boxes fight CO3's own layout, this is the paper's stated reason to switch: an imposed layout "can conflict with the layout implied by the sampled initial noise" (p. 1).
- For the one-fix step (inferred): CO3 combines several noise predictions per step. The masks here act inside one UNet pass under one prompt. Only the joint-prompt pass has two subjects to bound, so the masking would apply there; the single-subject passes have nothing to mask.
- For Paper 2's method (inferred): the cheaper path the authors name for future work, replacing the latent optimisation by "feature injection from a control map representing the target clusters" (p. 8), avoids the gradient passes and could fit a correction applied to the noise prediction.

## Closer work it cites

| Title | arXiv | Page | Why |
|---|---|---|---|
| Multi-Concept T2I-Zero: Tweaking Only The Text Embeddings and Nothing Else (Tunanyan et al. 2023) | 2310.07419 | p. 2, p. 11 | A training-free change to text embeddings for multi-subject prompts; bears on whether a text-side fix can stop borrowing. |
| A-star: Test-time attention segregation and retention for text-to-image synthesis (Agarwal et al. 2023) | not printed | p. 2, p. 8 | Training-free guidance that pushes different subjects' cross-attention maps apart, a direct handle on the cross-attention overlap predictor. |
