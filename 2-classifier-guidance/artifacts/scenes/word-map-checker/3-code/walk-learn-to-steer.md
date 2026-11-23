# The Learn-to-Steer code, line by line

This page follows one word's attention maps through the authors' code, from the moment SD 2.1's U-Net computes them to the gradient that moves the latent, then goes back to how the head was trained. Read it beside [Learn-to-Steer's paper](https://arxiv.org/pdf/2509.02295v2) when you want to know what a sentence of the paper became in code.

**What was checked.** The code is [`sapir1779/learn-to-steer` at commit `a2ae427`](https://github.com/sapir1779/learn-to-steer/tree/a2ae42746ec93c5936410964be01c85c0ca2266e), cloned read-only. It was read and never run, so no block below reports a seeded run. Worked values come from the small network in [the miniature's answer key](../2-miniature/miniature_answer_key.json), and full-scale sizes come from [the ladder's numbers file](../numbers/numbers.json). Symbols follow [the ladder's notation](../NOTATION.md).

**SDXL is our proposal.** The authors ran FLUX.1-schnell, FLUX.1-dev, SD 2.1 and SD 1.4 ([page 5](https://arxiv.org/pdf/2509.02295v2#page=5)). Every SDXL size on this page says what their head would receive if it were moved there, and nobody has run that.

## Table of contents

1. [Catching each attention map](#1-catching-each-attention-map)
2. [Keeping the prompt's branch](#2-keeping-the-prompts-branch)
3. [Picking the word's tokens](#3-picking-the-words-tokens)
4. [Bringing every map to 16 by 16](#4-bringing-every-map-to-16-by-16)
5. [The aggregator: many maps into one](#5-the-aggregator-many-maps-into-one)
6. [The classifier: two maps and the step into five logits](#6-the-classifier-two-maps-and-the-step-into-five-logits)
7. [The loss](#7-the-loss)
8. [The refine loop](#8-the-refine-loop)
9. [Where the loop sits in the sampler](#9-where-the-loop-sits-in-the-sampler)
10. [Training data: two inversions, one label](#10-training-data-two-inversions-one-label)
11. [Training the head](#11-training-the-head)
- [Where the code and the paper differ](#where-the-code-and-the-paper-differ)
- [What the code settles that the paper left open](#what-the-code-settles-that-the-paper-left-open)
- [What could not be checked](#what-could-not-be-checked)
- [Cross-references](#cross-references)

## How many maps one word hands the head

| Model | Layers × heads | Maps per word | Source |
|---|---|---|---|
| SD 1.4 | 16 layers × 8 heads | 128 | its U-Net config |
| SD 2.1 | 16 layers, heads 5, 10, 20 or 20 by depth | 195 (inferred) | head list assumed because its config returned HTTP 401; 1368 / 195 = 7.0 agrees with the paper's "7× less maps" on [page 5](https://arxiv.org/pdf/2509.02295v2#page=5) |
| FLUX.1-dev | 57 × 24 | 1368 | `model_utils.py` line 37 |
| SDXL (our proposal) | 10 layers × 10 heads on the 64 grid, 60 layers × 20 heads on the 32 grid | 1300 | its U-Net config |

The shapes below use SD 2.1's 195 and give SDXL's 1300 beside it.

## 1. Catching each attention map

**Role: the paper's own method.** The U-Net computes this softmax anyway. The method's addition is the one line that keeps a copy.

**Code.** `pipelines/sd/sd2_attention_processor.py`, lines 34 to 37 ([read them at the commit](https://github.com/sapir1779/learn-to-steer/blob/a2ae42746ec93c5936410964be01c85c0ca2266e/pipelines/sd/sd2_attention_processor.py#L34-L37)).

```python
attention_probs = attn.get_attention_scores(query, key, attention_mask)
self.attention_cache.store_attention_map(self.block_type, attention_probs, is_cross, batch_size)

hidden_states = torch.bmm(attention_probs, value)
```

**The equation** ([page 3](https://arxiv.org/pdf/2509.02295v2#page=3), where $A_t \in \mathbb{R}^{h \times w \times N}$):

$$
A^{(\ell)}[p, i] \;=\; \frac{\exp\big(q_p^\top k_i \,/\, \sqrt{d_{\text{head}}}\big)}{\sum_{j=1}^{77} \exp\big(q_p^\top k_j \,/\, \sqrt{d_{\text{head}}}\big)}
$$

**Symbols.** $p$ is one patch of the grid, $i$ one of the 77 text tokens, $q_p$ the patch's query, $k_i$ the token's key, $d_{\text{head}}$ the width of one head and $\ell$ the layer-and-head number. The sum runs over all 77 tokens, the start token included.

**Shape.** In: query (heads × batch, $h \cdot w$, head width), key (heads × batch, 77, head width). Out: (heads × batch, $h \cdot w$, 77).

**Worked value.** In the miniature a patch with cat-ness 2 and key 2 scores $e^{4}$ against a background token's $e^{0}$, so the cat word gets 0.982 of that patch.

## 2. Keeping the prompt's branch

**Role: the paper's own method.**

**Code.** `pipelines/sd/sd2_attention_cache.py`, lines 55 to 65 ([read them at the commit](https://github.com/sapir1779/learn-to-steer/blob/a2ae42746ec93c5936410964be01c85c0ca2266e/pipelines/sd/sd2_attention_cache.py#L55-L65)).

```python
def store_attention_map(self, block_type, attn_probs, is_cross, batch_size):
    if not is_cross or not self.is_initialized():
        return

    if batch_size > 1:
        prompt_attn_start_idx = attn_probs.shape[0] // 2
        attn_probs = attn_probs[prompt_attn_start_idx:]

    for cache_type in self.cache_types:
        value = self._get_probs_value(cache_type, attn_probs)
        self.add(cache_type, block_type, value)
```

**What it computes.** Self-attention is dropped. In a batch of two (empty prompt first, prompt second) only the second half is kept, and each map is reshaped to (heads, 77, $h$, $w$). During the refine loop the batch is 1, because the optimiser runs the U-Net on the prompt's embedding alone (block 8).

**Shape.** In: (heads × batch, $h \cdot w$, 77). Out: one entry of (heads, 77, $h$, $w$) per layer, grouped by block type.

## 3. Picking the word's tokens

**Role: the paper's own method.**

**Code.** `utils.py`, lines 114 to 132 ([read them at the commit](https://github.com/sapir1779/learn-to-steer/blob/a2ae42746ec93c5936410964be01c85c0ca2266e/utils.py#L114-L132)), with the token search in `find_word_token_inds` at lines 77 to 100.

```python
word_token_inds = instances_token_inds[instance_idx]
vis_text_attn_cache = attention_cache.get_cross_attn_visual_text_cache()[timestep]
# (57, 24, N, 16, 16), where N = len(word_token_inds)
all_layers_word_cross_attn_maps = vis_text_attn_cache.get_attn_maps_from_indices(word_token_inds)
```

```python
if len(word_token_inds) > 1:
    # (57, 24, N, 16, 16) --> (57, 24, 16, 16)
    word_cross_attn_maps = torch.mean(word_cross_attn_maps, dim=2)
```

**The equation.** The paper does not print one. For a name split into tokens $i \in I_s$:

$$
A^{(\ell)}_s[p] \;=\; \frac{1}{|I_s|} \sum_{i \in I_s} A^{(\ell)}[p, i]
$$

**Symbols.** $s$ is the subject word, $I_s$ the positions of its tokens in the prompt. When a word appears twice the first occurrence is used (`instance_idx=0`).

**What stays in the softmax.** `remove_start_token=True` only changes how the word is looked up. The start token still holds its share of every patch's weight, and the map is never renormalised across words.

**Shape.** In: (layers, heads, $|I_s|$, $h$, $w$) per grid side. Out: (layers × heads, $h$, $w$) per grid side.

## 4. Bringing every map to 16 by 16

**Role: the paper's own method.**

**Code.** `utils.py`, lines 60 to 74 ([read them at the commit](https://github.com/sapir1779/learn-to-steer/blob/a2ae42746ec93c5936410964be01c85c0ca2266e/utils.py#L60-L74)), called from lines 152 to 154.

```python
for res in sorted(attn_maps_dict.keys()):
    attn = attn_maps_dict[res]  # (L, H, H_in, W_in)
    LH, H_in, W_in = attn.shape

    attn = attn.view(LH, 1, H_in, W_in)
    resized = F.interpolate(attn, size=(target_size, target_size), mode='bilinear', align_corners=False)

    if renormalize:
        resized = resized / (resized.sum(dim=(-2, -1), keepdim=True) + 1e-8)

    all_maps.append(resized.squeeze(1))  # (L*H, target_size, target_size)
```

**The equation.** The paper does not print the size or this step:

$$
\hat A^{(\ell)}_s \;=\; \frac{\operatorname{bilinear}_{16}\big(A^{(\ell)}_s\big)}{\sum_{p} \operatorname{bilinear}_{16}\big(A^{(\ell)}_s\big)[p] + 10^{-8}}
$$

**Symbols.** $\hat A^{(\ell)}_s$ is the resized map and adds to 1 over the 16 × 16 grid. The division is over patches, so a word that takes little of the picture and a word that takes much both enter the head with the same total.

**Worked value.** The miniature's layer 1 cat map (0.982, 0.982, 0.1192, 0.1192) sums to 2.2024 and becomes (0.4459, 0.4459, 0.0541, 0.0541).

**What shrinking by four reads.** `F.interpolate` without antialiasing samples at each output cell's centre. Moving one input cell at a time through it on CPU shows that a 32 to 16 cell averages all 4 cells of its block, each at weight 0.25, and a 64 to 16 cell reads only the middle 4 of its 16 cells (rows 1 and 2, columns 1 and 2 of the block), each at weight 0.25. The other 12 cells of a 64 grid block have no influence. This affects SD 2.1's 64 grid maps and would affect SDXL's.

**A condition worth knowing.** Line 152 skips the whole call when a model has one grid side and it is already 16. Those maps are not renormalised.

**Shape.** In: (layers × heads, $h$, $w$) for each grid side. Out: (195, 16, 16) on SD 2.1, (1300, 16, 16) on SDXL (our proposal).

## 5. The aggregator: many maps into one

**Role: the paper's own method.** One network with shared weights runs once for the subject and once for the object.

**Code.** `train_relation_classifier/model.py`, lines 152 to 182 ([read them at the commit](https://github.com/sapir1779/learn-to-steer/blob/a2ae42746ec93c5936410964be01c85c0ca2266e/train_relation_classifier/model.py#L152-L182)).

```python
x = cross_attention_map.view(b, c, -1)  # shape = (B, 57 * 24, 256)
x = self.input_projection(x)  # shape = (B, 57 * 24, d_model)
x = self.positional_encoding(x)  # shape = (B, 57 * 24, d_model)
```

```python
for layer in self.transformer.layers:
    x = layer(x + timestep_emb)
```

```python
query = self.query_vector.repeat(x.shape[0], 1, 1)  # shape = (B, 1, d_model)
attn_output, _ = self.layer_attention(query, x, x)  # shape = (B, 1, d_model)
attn_output = attn_output.squeeze(1)  # shape = (B, d_model)
```

**The equations** ([page 15](https://arxiv.org/pdf/2509.02295v2#page=15)):

$$
\begin{aligned}
T_\ell &= W_{\text{in}}\operatorname{vec}\big(\hat A^{(\ell)}_s\big) + \operatorname{PE}_\ell && \text{one token per map}\\
T &\leftarrow \operatorname{TransformerLayer}\big(T + \phi(b)\big) && \text{repeated for each layer}\\
z &= \operatorname{MultiheadAttention}(q,\ T,\ T) && \text{a learned query pools the tokens}\\
\bar A_s &= \operatorname{reshape}_{16 \times 16}\big(W_{\text{out}}\, z\big)
\end{aligned}
$$

**Symbols.** $W_{\text{in}}$ projects a flattened map from 256 numbers to $d = 256$. $\operatorname{PE}_\ell$ is the sinusoidal code for the token's position in the list of maps. $b$ is the step bucket from block 8 and $\phi(b)$ its row in a table of 50 learned vectors. $q$ is the learned query, $z$ the pooled vector and $W_{\text{out}}$ maps it back to 256 numbers, so $\bar A_s$ is a free real-valued map with no softmax and no renormalisation.

**Shape.** (195, 16, 16) to (195, 256), then (1, 256), then (16, 16). On SDXL (our proposal) the first size is 1300.

**Size.** The SD 2.1 configuration has 2 transformer layers with 2 heads and a 2-head pooling layer, 1,068,032 trained weights. The FLUX.1-dev configuration has 4 layers with 4 heads and a 4-head pooling layer, 2,516,224. Both counts come from rebuilding these layers in [the numbers script](../numbers/make_numbers.py).

**Worked value.** The miniature fixes the pooling weights at 3/4 and 1/4, so 0.75 × 0.4459 + 0.25 × 0.383 = 0.4302. A real query weighs each map by its content.

## 6. The classifier: two maps and the step into five logits

**Role: the paper's own method.**

**Code.** `train_relation_classifier/model.py`, lines 75 to 88 ([read them at the commit](https://github.com/sapir1779/learn-to-steer/blob/a2ae42746ec93c5936410964be01c85c0ca2266e/train_relation_classifier/model.py#L75-L88)).

```python
x = maps
x = x.view(x.size(0), -1)  # shape = (batch_size, 2 * 16 * 16)
x = self.projection(x)  # shape = (batch_size, d_model)
x = self.positional_encoding(x.unsqueeze(1))  # shape = (batch_size, 1, d_model)
x = self.transformer(x)  # shape = (batch_size, 1, d_model)
x = x.squeeze(1)  # shape = (batch_size, d_model)
```

**The equations** ([page 15](https://arxiv.org/pdf/2509.02295v2#page=15)):

$$
\begin{aligned}
y &= W_{\text{pair}}\operatorname{vec}\big[\bar A_s,\ \bar A_o\big] + \operatorname{PE}_0\\
y' &= \operatorname{Transformer}(y)\\
\text{logits} &= \operatorname{MLP}\big(\big[\,y',\ \phi_{16}(b)\,\big]\big)
\end{aligned}
$$

**Symbols.** $W_{\text{pair}}$ projects the 512 numbers of the two stacked maps to 256. $\phi_{16}(b)$ is a second learned table of 16-number step vectors. The MLP has hidden sizes 256 and 128 with LayerNorm, LeakyReLU and dropout.

**One token.** The whole flattened pair enters the transformer as a single token, so each attention weight it computes is exactly 1. Every spatial comparison between the two maps has to be made by the linear layer $W_{\text{pair}}$ and the feed-forward parts. The paper's description reads as if the transformer compares positions.

**Shape.** (2, 16, 16) to 512 to 256, then (1, 256), then 256 + 16 = 272, then 5 logits. The class count comes from the paper's four relations plus "neutral" ([page 16](https://arxiv.org/pdf/2509.02295v2#page=16)); in code it is the size of a mapping file not in the repository.

**Size.** 896,549 trained weights in the SD 2.1 configuration, for a head total of 1,964,581. The FLUX.1-dev configuration has 2,344,741, for a total of 4,860,965.

## 7. The loss

**Role: steering.**

**Code.** `pipelines/test_time_optimization.py`, lines 144 to 148 ([read them at the commit](https://github.com/sapir1779/learn-to-steer/blob/a2ae42746ec93c5936410964be01c85c0ca2266e/pipelines/test_time_optimization.py#L144-L148)), with the criterion in `train_relation_classifier/loss_utils.py`, line 16.

```python
def compute_loss(self, latents, timestep, iter_idx, store_results=False):
    latents, _ = self.denoise(latents, timestep)  ## To store cross-attention maps
    maps, timestep_batch, target_inds = self.prepare_relation_classifier_inputs(iter_idx)
    outputs = self.relation_classifier(maps, timestep_batch)
    loss = self.loss_criterion(outputs, target_inds)
```

**The equation** ([page 4](https://arxiv.org/pdf/2509.02295v2#page=4)):

$$
\mathcal L_t \;=\; -\log \hat p[r], \qquad \hat p = \operatorname{softmax}(\text{logits}), \quad \text{logits} = C(A_s, A_o, b)
$$

**Symbols.** $r$ is the relation parsed from the prompt, $C$ the aggregator and classifier together, and $\hat p[r]$ the probability the head gives that relation.

**Worked value.** On the miniature's fused body the two maps are identical, all five logits are 0, $\hat p[r] = 0.2$ and $\mathcal L_t = \ln 5 = 1.6094$.

## 8. The refine loop

**Role: steering.** This is the part of the method that moves the picture.

**Code.** `pipelines/test_time_optimization.py`, lines 175 to 187 for the loop and 137 to 141 for the update ([the loop](https://github.com/sapir1779/learn-to-steer/blob/a2ae42746ec93c5936410964be01c85c0ca2266e/pipelines/test_time_optimization.py#L175-L187), [the update](https://github.com/sapir1779/learn-to-steer/blob/a2ae42746ec93c5936410964be01c85c0ca2266e/pipelines/test_time_optimization.py#L137-L141)).

```python
for i in range(max_refinement_steps + 1):
    self.loss_idx = i % len(self.relation_triplets)  ## cycle through relation triplets if multiple are given
    store_results = (self.config.test_classification and i == 0)
    loss, latents = self.compute_loss(latents, timestep, iter_idx, store_results)

    if i == 0 and loss < self.config.min_loss:
        success = False
        break

    if loss != 0 and loss > loss_target:
        latents = self.update_latents(latents, loss, iter_idx)
    else:
        break
```

```python
grad = torch.autograd.grad(loss, latents, create_graph=False, retain_graph=False)[0]
latents = latents - step_size * grad
```

**The equation** ([page 4](https://arxiv.org/pdf/2509.02295v2#page=4), which writes $z_t$ and $\alpha$):

$$
x_t \;\leftarrow\; x_t - \eta\, \nabla_{x_t} \mathcal L_t
$$

**Symbols.** $\eta$ is the step size: 5 on SD 2.1 and Flux, 7.5 on SD 1.4. It multiplies the raw gradient with no normalisation and no cap.

**How many times.** `max_refinement_steps` is 15, so the loop makes 16 passes and each pass can update, for up to 16 updates per sampler step. The paper prints 15. The loop stops early once the loss falls to its target, which is $10^{-5}$ at sampler step 0, $3 \times 10^{-5}$ at step 1, $5 \times 10^{-5}$ at step 2 and $7 \times 10^{-5}$ from step 3 (lines 46 to 55). The `min_loss` check at the top is inactive, because `min_loss` stays at its default of 0 and a cross-entropy cannot go below it.

**What the head is told about time.** The sampler step number is snapped up to one of the step numbers the head trained on, in `prepapre_inference_indices` (lines 59 to 74) and `get_inference_idx` (lines 86 to 91):

| Sampler steps | Head's step bucket $b$ |
|---|---|
| 0 to 5 | 5 |
| 6 to 10 | 10 |
| 11 to 15 | 15 |
| 16 to 20 | 20 |
| 21 to 25 | 25 |
| 26 to 49 | 49 |

The optimised steps are 0 to 24, so the buckets in use are 5, 10, 15, 20 and 25. Steps 0 to 4 are read with a bucket from a later step, because the head never trained on them.

**Worked value.** On the miniature at $\eta = 5$ the loss goes 1.6094, 0.2280, 0.1884, 0.1573 over the first passes, all 16 updates fire, and $\hat p[r]$ ends at 0.9936.

## 9. Where the loop sits in the sampler

**Role: ordinary sampler, with the steering hooked in before it.**

**Code.** `pipelines/sd/pipeline_sd2_debug.py`, lines 347 to 350 and 365 to 394 ([the prompt branch](https://github.com/sapir1779/learn-to-steer/blob/a2ae42746ec93c5936410964be01c85c0ca2266e/pipelines/sd/pipeline_sd2_debug.py#L347-L350), [the loop](https://github.com/sapir1779/learn-to-steer/blob/a2ae42746ec93c5936410964be01c85c0ca2266e/pipelines/sd/pipeline_sd2_debug.py#L365-L394)).

```python
if test_time_opt is not None and test_time_opt.should_run(i):
    latent_model_input = self.scheduler.scale_model_input(latents, t)
    latents = test_time_opt.run(latent_model_input, t, i)

# expand the latents if we are doing classifier free guidance
latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
```

**What it computes.** The latent is optimised first, using only the prompt's branch. The ordinary classifier-free guidance step at scale 7.5 then runs from the moved latent:

$$
\varepsilon = \varepsilon_\varnothing + 7.5\,\big(\varepsilon_{\text{prompt}} - \varepsilon_\varnothing\big)
$$

The noise prediction is never edited, which is the main difference from our sampler's push ([our sampler's walk](walk-our-sampler.md)). `run.py` lines 217 to 221 load a DDIM scheduler for SD 2.1, whose `scale_model_input` returns its input unchanged.

**Which steps.** `should_run` needs $0 \le i < 25$ of 50 steps, set by `end_idx` in `run.py` line 56 for SD 2.1.

**Cost.** SD 2.1 goes from 4.5 s to 90 s per image and FLUX.1-dev from 11 s to 6 minutes ([page 7](https://arxiv.org/pdf/2509.02295v2#page=7)).

## 10. Training data: two inversions, one label

**Role: the paper's own method.** This is the fix for relation leakage.

**Code.** `train_relation_classifier/data_creation/attention_maps_generation/prompt_generator.py`, lines 54 to 66 ([read them at the commit](https://github.com/sapir1779/learn-to-steer/blob/a2ae42746ec93c5936410964be01c85c0ca2266e/train_relation_classifier/data_creation/attention_maps_generation/prompt_generator.py#L54-L66)), and `generate_relations_data.py` in the same folder, lines 291 and 305 to 338 ([the data loop](https://github.com/sapir1779/learn-to-steer/blob/a2ae42746ec93c5936410964be01c85c0ca2266e/train_relation_classifier/data_creation/attention_maps_generation/generate_relations_data.py#L305-L338)).

```python
pos_prompt = self.generate_prompt(obj1, obj2, rel_name, rel_name)
src_prompts.append((pos_prompt, POS_PROMPT_TYPE, rel_name, [obj1, obj2]))

# Generate negative prompts
neg_rel_names = self.sample_negative_relations(rel_name, self.num_prompts)
for neg_rel_name in neg_rel_names:
    neg_prompt = self.generate_prompt(obj1, obj2, rel_name, neg_rel_name)
    src_prompts.append((neg_prompt, NEG_PROMPT_TYPE, rel_name, [obj1, obj2]))
```

**What it computes** ([page 4](https://arxiv.org/pdf/2509.02295v2#page=4)). Each image is inverted under its true relation and under one random wrong relation (`num_prompts=1`). Both sets of maps carry the true relation as the label, in the third field of each tuple:

$$
\big(A_s^{\text{true word}},\, A_o^{\text{true word}},\, b\big) \mapsto r_{\text{image}}, \qquad \big(A_s^{\text{wrong word}},\, A_o^{\text{wrong word}},\, b\big) \mapsto r_{\text{image}}
$$

**Where the maps come from.** After inversion, the pipeline reconstructs the image and caches maps at steps 5, 10, 15, 20, 25 and 49 (`generate_relations_data.py` line 160). SD uses Fixed-Point Inversion with 5 fixed-point iterations (line 185); Flux uses RF-Inversion with guidance 0 (line 119).

**A gap in the balance.** A reconstruction below 20 dB PSNR (`min_psnr=20`, line 275) triggers a `break` at line 312. The true-word prompt comes first in the list, so an image whose wrong-word inversion fails keeps its true-word samples and has no wrong-word partner. The code does not count how often this happened. Whether it matters for leakage is inferred and untested.

**Worked value.** In the miniature's leakage example, a head trained on correct captions puts weight 5.5886 on the word's trace and 1.6336 on the map's spatial reading. Given an image truly left whose prompt says "right of", it answers right with probability 0.9929. Trained on both prompts, its weight on the trace is 0.0 and on the spatial reading 11.9015, and the same case gets 0.0085.

**Evidence for the fix.** GenEval position on FLUX.1-schnell's validation set is 0.425 without it and 0.57 with it ([page 8](https://arxiv.org/pdf/2509.02295v2#page=8), Table 4; 0 to 1, higher is better).

## 11. Training the head

**Role: the paper's own method.**

**Code.** `train_relation_classifier/train.py`, lines 163 to 176 and 219 ([read them at the commit](https://github.com/sapir1779/learn-to-steer/blob/a2ae42746ec93c5936410964be01c85c0ca2266e/train_relation_classifier/train.py#L163-L176)), with the values in `train_relation_classifier/configs/sd2_config.yaml`.

```python
self.optimizer = optim.AdamW(self.model.parameters(), lr=lr,
                              weight_decay=self.trainer_config.weight_decay)
```

**The objective:**

$$
\min_{C}\ \frac{1}{B} \sum_{n=1}^{B} -\log \operatorname{softmax}\big(C(A_{s,n}, A_{o,n}, b_n)\big)[r_n]
$$

**Symbols.** $B = 64$ is the batch size and $n$ one training sample.

| Setting | SD 2.1 config | FLUX.1-dev config |
|---|---|---|
| Learning rate | $10^{-4}$ | $5 \times 10^{-5}$ |
| Weight decay | 0.05 | 0.05 |
| Epochs at most | 100 | 200 |
| Early stopping patience | 10 | 10 |
| Plateau scheduler | factor 0.5, patience 5 | factor 0.5, patience 5 |
| Dropout | 0.3 | 0.3 |

## Where the code and the paper differ

Each row was checked against the file and line at commit `a2ae427`.

| What | The paper | The code | Where |
|---|---|---|---|
| Pooling the maps into one | a four-layer, single-head transformer with a learned query ([page 15](https://arxiv.org/pdf/2509.02295v2#page=15)) | one `MultiheadAttention` layer with a learned query: 4 heads on FLUX.1-dev, 2 on SD 2.1 | `train_relation_classifier/model.py:133-135,176-177`, configs' `layer_attn_nheads` |
| The classifier's transformer | two-channel image flattened, projected and passed through a transformer ([page 15](https://arxiv.org/pdf/2509.02295v2#page=15)) | the whole flattened pair is one token, so its self-attention attends only to itself | `train_relation_classifier/model.py:76-80` |
| Iterations per step | 15 ([page 16](https://arxiv.org/pdf/2509.02295v2#page=16)) | `range(15 + 1)` passes, each able to update: up to 16 updates | `pipelines/test_time_optimization.py:175-187`; `run.py:49` for FLUX.1-dev, `run.py:55` for SD 2.1, `run.py:61` for SD 1.4 |
| What "timestep" means to the head | the denoising timestep $t$ ([page 4](https://arxiv.org/pdf/2509.02295v2#page=4)) | the step index 0 to 49, snapped up to one of 5, 10, 15, 20, 25, 49 | `pipelines/test_time_optimization.py:59-74,86-91,115` |
| Training epochs on SD 2.1 | 150 for SD models ([page 16](https://arxiv.org/pdf/2509.02295v2#page=16)) | `num_epochs` 100 for SD 2.1, 150 for SD 1.4 | `train_relation_classifier/configs/sd2_config.yaml:30`, `sd14_config.yaml:30` |
| Inversion guidance on SD | guidance scale 0 ([page 5](https://arxiv.org/pdf/2509.02295v2#page=5)) | `guidance_scale` 1.0, which this pipeline treats as no classifier-free guidance | `train_relation_classifier/data_creation/attention_maps_generation/generate_relations_data.py:163`, `pipelines/sd/pipeline_sd2_inversion_debug.py:113` |

The last row changes nothing in practice if the paper's 0 means "guidance off". Flux's inversion does pass 0, at `generate_relations_data.py:119`.

## What the code settles that the paper left open

| Question | Answer in the code | Where |
|---|---|---|
| How a multi-token name becomes one map | the mean over the name's tokens | `utils.py:127-129` |
| Which guidance branch supplies the maps | the prompt's own branch: the optimiser runs the U-Net on the text embedding alone | `pipelines/sd/pipeline_sd2_debug.py:347-350`, `pipelines/sd/sd2_test_time_optimization.py:33-41` |
| The size of the maps | resized to 16 × 16, each renormalised to sum to 1 over the grid | `utils.py:51-74,152-154`, configs' `map_size` |

## What could not be checked

**No run.** The authors' code was not executed, so none of its printed shapes or losses were observed. Shapes come from reading it and from rebuilding the head's layers in [the numbers script](../numbers/make_numbers.py).

**SD 2.1's map count.** Its U-Net config returned HTTP 401. The 195 uses the head list 5, 10, 20, 20 and is marked inferred.

**The checkpoints.** The README links trained weights on Google Drive. They were not downloaded, so the class count of 5 rests on the paper and the weight counts rest on the configs.

## Cross-references

- [The walk through our own sampler](walk-our-sampler.md) shows how the same maps are read today and what a port to SDXL would change.
- [The rebuild companion](../rebuild-from-fundamentals.md) takes each equation on this page down to first-course maths.
- [The miniature's script](../2-miniature/miniature.py) computes every worked value quoted here.
- [The checked reading note on the paper](../../../../../artifacts/notes/progress/abstract-checkers/01-reading/learn-to-steer.md) holds the benchmark tables.
