# Data-Driven Loss Functions for Inference-Time Optimization in Text-to-Image (Learn-to-Steer)

arXiv 2509.02295v2, read from the full PDF (26 pages, appendix included).

## What it does

Learn-to-Steer trains a small classifier to read the spatial relation between two objects from a diffusion model's word cross-attention maps (p. 2, p. 3). During generation, the classifier's cross-entropy against the prompted relation is differentiated and the latent is updated with that gradient (p. 4). The method is tested on FLUX.1-schnell, FLUX.1-dev, SD 2.1 and SD 1.4 (p. 5), and SDXL is not among them.

## What the classifier or loss reads

**Input.** The classifier is $\hat p = C(A_s, A_o, t)$: the attention maps of the subject word, the attention maps of the object word, and the denoising timestep (p. 4). The relation word's own attention map is not an input (p. 4, p. 13).

**Shape of the maps.** Each object contributes all $L \times H$ cross-attention maps of size $h \times w$, where $L$ is the number of cross-attention layers and $H$ the number of heads (p. 15). The paper does not print the values of $h$ and $w$. It does not say how a multi-token name is reduced to one map, or which classifier-free guidance branch supplies the maps.

**Architecture (p. 15).**

1. A shared-weight aggregator treats each of the $L \cdot H$ maps as one token, projects it to $d = 256$, and adds positional encodings and a timestep embedding $\phi(t)$.
2. A four-layer, four-head transformer exchanges information across maps. A four-layer, single-head transformer with a learned query then pools them. SD models use two-layer, two-head transformers here.
3. A linear layer outputs one aggregated $h \times w$ map per object.
4. The two aggregated maps are stacked as a two-channel image, flattened, projected to $d = 256$ and passed through a four-layer, eight-head transformer. SD models use two layers and four heads.
5. The output is concatenated with a 16-dimensional timestep embedding and an MLP returns logits over the relation classes.

**Classes.** Above, below, left of, right of, plus a fifth "neutral" class used in training and unused during generation (p. 5, p. 16). There is no class for fused objects.

**Aggregation choice.** Keeping all maps scored 0.70 on GenEval position (fraction of images with the correct relation, 0 to 1, higher is better) on the FLUX.1-dev validation set. A single global mean map scored 0.78, and the authors rejected it because it gave "dull" or neutral backgrounds (p. 13, Table 9).

## What it trained on

**Images.** Real images from GQA plus synthetic images from Image-Generation-CoT (p. 4). Flux models use 20K real and 4K synthetic images. SD models use 4K real and 4K synthetic, because SD has 7 times fewer maps (p. 5). GQA images with several instances of one category are filtered out, and hypernym relations are removed using WordNet (p. 5).

**Map extraction.** Maps come from inverting each image, using RF-Inversion for Flux and Fixed-Point Inversion for SD, both at guidance scale 0 (p. 5). Maps are taken roughly every 5 denoising steps plus the last step, for example {5, 10, 15, 20, 25, 49}, and at {1, 3} for FLUX.1-schnell (p. 5).

**The inversion fix (p. 4).** Each training image is inverted twice. One inversion uses the prompt that correctly states the relation ("A dog to the left of a cat"). The other uses a randomly sampled incorrect relation ("A dog above a cat"). Both sets of maps get the label of the relation actually in the image, here "left of". The same spatial layout therefore appears under different relation words. The text calls the correct prompt "mismatching", which reads as a wording slip (inferred).

**Training setup (p. 16).** Batch size 64, AdamW with weight decay 0.05, learning rate $5 \cdot 10^{-5}$ for Flux and $10^{-4}$ for SD, ReduceOnPlateau with factor 0.5 and patience 5. Training runs at most 200 epochs for Flux and 150 for SD, with early stopping after 10 epochs. The loss is cross-entropy (p. 3, Fig. 2). Training takes 7 hours on an NVIDIA A100 80GB (p. 6).

**Tuning.** Design choices and hyperparameters were selected on a GenEval-style validation set with no overlap with the test prompts (p. 4, p. 5).

## How its gradient enters

The latent is updated directly, and the noise prediction is left unchanged (p. 4):

$$
\mathcal{L}_t = -\log \hat p[r], \qquad z_t \leftarrow z_t - \alpha \nabla_{z_t} \mathcal{L}_t
$$

$\hat p$ is the classifier's distribution over relations, $r$ is the relation parsed from the prompt with SpaCy (p. 4, p. 16), and $\alpha$ is the step size.

- **Steps:** the first 50% of denoising steps. FLUX.1-schnell optimises only the initial noise (p. 4, p. 16).
- **Scale:** $\alpha = 5$ for SD 2.1 and Flux, $\alpha = 7.5$ for SD 1.4 (p. 16).
- **Iterations:** 15 optimisation iterations per step (p. 16).
- **Normalisation or cap:** none is stated.
- **Cost:** SD 2.1 goes from 4.5 s to 90 s per image and FLUX.1-dev from 11 s to 6 minutes (p. 7). On SD 2.1, using 15 iterations only in the first 10% of steps and one iteration afterwards cuts 90 s to 25 s, with validation accuracy moving from 0.65 to 0.6 (p. 7).
- **Several relations:** the optimisation target alternates between relations at each timestep (p. 4).

## Unseen concepts

**What was held out.** 16 object categories, tested on 200 prompts: teapot, corgi, furby, robot, turtle, rabbit, butterfly, key, pig, bee, tiger, hammer, monkey, razor, snake, stapler (p. 7, p. 14). Six came from the T2I-CompBench vocabulary as words outside both Learn-to-Steer's and COMPASS's training data. ChatGPT proposed the other ten (p. 7, p. 14).

**How the split was built.** The authors picked words they report as absent from the training data (p. 7, p. 14). The paper describes no planned split that removed categories from a training set, and it does not print the training vocabulary or the overlap check. Word-level absence leaves close relatives possible: corgi is a kind of dog and tiger a kind of cat, and "cat" is listed as in-distribution (p. 14). Whether dog images were trained on cannot be confirmed from the paper (inferred).

**Results (p. 5, Table 2).** GenEval position on the unseen categories, 0 to 1, higher is better:

| Model | Vanilla | COMPASS, run from its code | Learn-to-Steer |
|---|---|---|---|
| FLUX.1-dev | 0.18 | 0.58 | 0.60 |
| SD 2.1 | 0.15 | 0.52 | 0.60 |
| SD 1.4 | 0.07 | 0.40 | 0.47 |

**How to read them.** GenEval's detector knows only the 80 COCO categories, so the authors replaced it with CLIPSeg for open-set categories (p. 7, p. 14). Table 2 therefore most likely uses a different scorer and different prompts from Table 1, so 0.60 against 0.61 on FLUX.1-dev is not a clean seen-versus-unseen gap (inferred). No per-category scores are given, and the classifier's own accuracy on held-out categories is not reported. All relation classes were seen in training, and only object categories were held out.

**Multi-relation prompts.** With 3 objects and 2 relations on FLUX.1-dev, accuracy is 0.36 on in-distribution categories and 0.28 on unseen ones (p. 6, Table 3).

## Word leakage and confounds

**The problem as reported.** A classifier trained on maps from correctly captioned inversions reached high training accuracy and failed during generation (p. 4). When images were inverted with a wrong relation word, it predicted the wrong word ("above" for a dog actually left of a cat), so it was reading the prompted relation from the maps (p. 4). The relation word's own map is not an input, so the trace must sit in the object words' maps (inferred).

**Evidence for the fix.** On the FLUX.1-schnell validation set, GenEval position is 0.425 without the fix and 0.57 with it (p. 8, Table 4). The paper says the augmentation "eliminates" leakage (p. 8), but it reports no direct leakage measure after the fix, such as how often the trained classifier still follows a wrong prompt word.

**Stated limits of the fix.** It assumes inversion succeeds and gives well-behaved maps even under the incorrect prompt. Complex prompts were not tested, and images with several instances of one object are less stable (p. 15).

**Confounds left open (inferred).**

- Leakage of the object words is not treated, so the classifier could still use object-specific map patterns.
- Training maps come from inversion at guidance 0, while SD generation uses guidance 7.5 (p. 5, p. 16). The paper does not discuss this shift.
- The diagonal-relations section says GenEval's verifier was needed to generate synthetic training data (p. 7). If the four main relations were filtered the same way, the training labels and the headline benchmark share a scorer.

## Numbers

Scales: G. Pos. is the fraction of 400 images (100 prompts, 4 images each) with the correct relation, 0 to 1, higher is better (p. 5). T. Spatial is the T2I-CompBench spatial score on 216 prompts, 10 images each, higher is better (p. 6).

| Metric | This method | Vanilla | Best baseline | Data | Model | Page, table |
|---|---|---|---|---|---|---|
| G. Pos. | 0.54 | 0.07 | COMPASS 0.51 | GenEval | SD 2.1 | p. 5, Table 1 |
| G. Pos. | 0.61 | 0.20 | COMPASS 0.60 | GenEval | FLUX.1-dev | p. 5, Table 1 |
| G. Pos. | 0.52 | 0.26 | FOR-Spatial 0.35 | GenEval | FLUX.1-schnell | p. 5, Table 1 |
| G. Pos. | 0.36 | 0.05 | COMPASS 0.46 (beats this method) | GenEval | SD 1.4 | p. 5, Table 1 |
| T. Spatial | 0.365 | 0.089 | COMPASS, run from its code, 0.257 | T2I-CompBench | SD 2.1 | p. 5, Table 1 |
| T. Spatial | 0.392 | 0.177 | FOR-Spatial 0.357 | T2I-CompBench | FLUX.1-dev | p. 5, Table 1 |
| G. Pos., 16 unseen categories | 0.60 | 0.18 | COMPASS 0.58 | 200 prompts | FLUX.1-dev | p. 5, Table 2 |
| G. Pos., 16 unseen categories | 0.60 | 0.15 | COMPASS 0.52 | 200 prompts | SD 2.1 | p. 5, Table 2 |
| G. Pos., validation, leakage fix on or off | 0.57 on, 0.425 off | | | GenEval validation | FLUX.1-schnell | p. 8, Table 4 |

## What to copy, what to avoid

- Copy the wrong-word check from p. 4: invert an image under a wrong property word and count how often the checker follows the word. Report it as a rate, which this paper does not do (inferred).
- Copy the two-inversion augmentation for any checker trained on prompts that contain the property word (inferred). For PoE, also try inverting under the two single-subject prompts, so the property word never enters the maps. The paper does not test this (inferred).
- Hold out whole families, such as all dogs or all felines, because a word-level hold-out like corgi or tiger may leave near neighbours in training (inferred).
- Use one scorer for seen and unseen categories, so the gap between them can be read as a generalisation gap (inferred).
- Add a "fused" class or label. Learn-to-Steer has none, and fusion is the failure this project targets (inferred).
- Do not read our 0-of-8 result for pushing the cat and dog maps apart as evidence against a learned attention checker. That was one hand-written loss, and this paper's learned loss reaches 0.54 where the hand-written STORM loss reaches 0.19 on SD 2.1 (p. 5, Table 1), though on a joint prompt and a different model (inferred).
- Test the dose before calling a checker a failure. Learn-to-Steer takes 15 iterations per step at $\alpha = 5$ across the first 50% of steps with no stated cap, far more than our one push per step capped at 0.4 of PoE's prediction norm on steps 0 to 15 of 50 (inferred).
- Check image quality when using one mean map per word. Our 32x32 single-map input is closest to the global-mean variant, which scored highest on position and damaged backgrounds (p. 13, Table 9) (inferred).
- Expect a transfer gap to PoE. Its maps come from one joint prompt where both object words compete for each pixel, while PoE has two prompts that each hold one subject (inferred).

## Closer work it cites

None. The cited steering methods use hand-written losses, external reward models or fine-tuning, and none trains a classifier for an abstract property or tests one on unseen concepts (p. 2, p. 3).

## Cross-references

- The mention of **Learn-to-Steer** in [the context page on the word-map checker, the idea in six pictures, section "A small judge reads only the two maps"](../../../../../context/methods/06-the-word-map-checker/03-the-illustrations.md#a-small-judge-reads-only-the-two-maps).
- The mention of **Learn-to-Steer** in [the context page on the word-map checker, worked on an 8-number latent](../../../../../context/methods/06-the-word-map-checker/04-the-miniature.md).
