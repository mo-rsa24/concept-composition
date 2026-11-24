# 🌍 Context: the word-map checker, the code that would run it

⬅️ Previous: [the miniature](04-the-miniature.md) | 📋 [The pages of this method](01-overview.md#the-pages-of-this-method) | Next: [the methods index](../00-INDEX.md) ➡️

This page reads two codebases block by block and ties each block to a figure and to its twin in the miniature. The first is Learn-to-Steer's released code ([`sapir1779/learn-to-steer` at commit `a2ae427`](https://github.com/sapir1779/learn-to-steer/tree/a2ae42746ec93c5936410964be01c85c0ca2266e)), which the ladder cloned read-only, read and never ran; its excerpts are quoted as the ladder's [Learn-to-Steer code walk](../../../2-classifier-guidance/artifacts/scenes/word-map-checker/3-code/walk-learn-to-steer.md) records them, because the checkout is not in this repository. The second is our own sampler, which would carry a word-map head on SDXL, and the one attention push we have run; those excerpts were checked this session against the files at commit `cbff8c0e`, where `poe_repair/methods/_classifier_guided.py` is unchanged since `3867b3f1` and `2-classifier-guidance/scripts/rung2/attention_push.py` since `ff028dd5`, both clean. The ladder's two walks are merged here into one page. No word-map head exists in our code yet.

<a href="../../../2-classifier-guidance/artifacts/scenes/word-map-checker/0.5-network-view/wm5-two-ways-to-apply-the-push.png"><img src="../../../2-classifier-guidance/artifacts/scenes/word-map-checker/0.5-network-view/wm5-two-ways-to-apply-the-push.png" width="720" alt="Top: Learn-to-Steer moves the latent itself many times, the latent through the frozen model on the prompt branch, maps then head, the loss minus log p hat of r, and the gradient back to the latent, step size 5, up to 16 updates per step on the first 25 of 50 steps. Bottom: our sampler changes the noise prediction once with a cap, the latent through the frozen U-Net with 3 prompts, the product, push then one DDIM step, the checker's gradient shortened to at most c of the prediction's norm, default 0.1, proposed 0.4"></a>

*The two codebases on this page, one per row. Top, Learn-to-Steer: latent x_t, the frozen model on the prompt's branch only, maps then head, the loss −log p̂[r], and the orange gradient back into the latent: x_t ← x_t − η ∇ L_t, step size 5 (SD 2.1, Flux), no cap stated, up to 16 updates per step in the code (15 printed), on the first 25 of 50 steps. Bottom, our sampler: latent x_t, the frozen U-Net with 3 prompts, the product ε_PoE, then the push and one DDIM step; ε ← ε_PoE − λ σ_t ∇ log p̂[r], shortened to at most c of its norm, one push per sampler step; the code's default c is 0.1 and 0.4 is proposed. Footer: our hand-written attention-separation push drew two animals on 0 of 8 SDXL seeds (seeds 9 to 16, steps 0 to 19, draft calls, no verdict recorded).* ✅ *Read on the figure at full size.*

## Table of contents

- [Words this page uses](#words-this-page-uses)
- [The blocks and where they live](#the-blocks-and-where-they-live)
- [Learn-to-Steer: catching and preparing the maps](#learn-to-steer-catching-and-preparing-the-maps)
- [Learn-to-Steer: the head](#learn-to-steer-the-head)
- [Learn-to-Steer: the loss and the refine loop](#learn-to-steer-the-loss-and-the-refine-loop)
- [Learn-to-Steer: training data and training](#learn-to-steer-training-data-and-training)
- [Our sampler: the product and the head call](#our-sampler-the-product-and-the-head-call)
- [Our sampler: the attention push already run](#our-sampler-the-attention-push-already-run)
- [Our sampler: the push and the cap](#our-sampler-the-push-and-the-cap)
- [What a head on SDXL would change](#what-a-head-on-sdxl-would-change)
- [Where the code and the paper differ](#where-the-code-and-the-paper-differ)
- [Where this came from](#where-this-came-from)

## Words this page uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#the-blocks-and-where-they-live) ➡️

**Attention processor**: the small object diffusers calls inside each attention layer to compute the weights; replacing it is how both codebases get a copy of the maps.

**Prompt branch**: the U-Net pass on the prompt's text, as against the pass on the empty prompt that classifier-free guidance also runs.

**`log_prob_both(z, t)`**: the one method our sampler asks any head for: a number standing for the log chance that the picture is right. Any head that provides it plugs in.

**`torch.autograd.grad(y, x)`**: asks PyTorch for the slope of y with respect to x, through every recorded step between them.

**Strength λ and cap c**: the code's `strength` and `push_cap`: how much of the gradient is used, and the largest share of the product's length a push may have.

## The blocks and where they live

Navigation: ⬅️ [Words this page uses](#words-this-page-uses) | 📋 [TOC](#table-of-contents) | [Next](#learn-to-steer-catching-and-preparing-the-maps) ➡️

**Learn-to-Steer's blocks are quoted as the ladder read them at `a2ae427`; ours were read in the files this session.**

| Block | Where it lives | What it computes | The miniature's twin | Mark |
|---|---|---|---|---|
| catch each map | authors' `pipelines/sd/sd2_attention_processor.py`, lines 34 to 37 | a copy of every cross-attention softmax | `word_maps`, `miniature.py` line 53 | ✍️ |
| pick the word and resize | authors' `utils.py`, lines 60 to 74 and 114 to 132 | the word's column, mean over its tokens, 16 × 16, divided by its sum | `pooled`, line 58 | ✍️ |
| aggregator and classifier | authors' `train_relation_classifier/model.py`, lines 75 to 88 and 152 to 182 | 1,300 maps per word to one map, the pair to 5 logits | `forward`, line 63 | ✍️ |
| loss and refine loop | authors' `pipelines/test_time_optimization.py`, lines 137 to 148 and 175 to 187 | −log p̂[r]; x_t ← x_t − η ∇ L_t up to 16 times | `lts_loop`, line 110 | ✍️ |
| the product | our `_classifier_guided.py`, lines 149 to 158 | ε_PoE = ε^g_cat + ε^g_dog − ε_∅ | the key's `product_step` | ✅ |
| the head call | our `_classifier_guided.py`, lines 200 to 212 | ∇_{x_t} log p for any head | `gradient`, line 87 | ✅ |
| the attention push | our `attention_push.py`, lines 36 to 45 and 77 to 87 | one mean map per word at 32 × 32; minus their correlation | `correlation`, line 129 | ✅ |
| push, cap and DDIM step | our `_classifier_guided.py`, lines 46 to 61, 213, 228 to 232 and 246 | raw push, capped, subtracted, then the step | `ddim`, line 124 | ✅ |

The miniature's file is [the ladder's miniature script](../../../2-classifier-guidance/artifacts/scenes/word-map-checker/2-miniature/miniature.py). ✅

## Learn-to-Steer: catching and preparing the maps

Navigation: ⬅️ [The blocks and where they live](#the-blocks-and-where-they-live) | 📋 [TOC](#table-of-contents) | [Next](#learn-to-steer-the-head) ➡️

**The U-Net computes the softmax anyway; the method's addition is one line that keeps a copy, then the word's column is cut out, brought to 16 × 16 and divided by its sum.** ✍️ The ladder's code walk, blocks 1 to 4.

```python
attention_probs = attn.get_attention_scores(query, key, attention_mask)
self.attention_cache.store_attention_map(self.block_type, attention_probs, is_cross, batch_size)
```

`sd2_attention_processor.py`, lines 34 to 35 of the quoted 34 to 37. The cache keeps cross-attention only, and from a batch of two (empty prompt first) only the prompt's half. ✍️ Block 2.

```python
resized = F.interpolate(attn, size=(target_size, target_size), mode='bilinear', align_corners=False)
if renormalize:
    resized = resized / (resized.sum(dim=(-2, -1), keepdim=True) + 1e-8)
```

`utils.py`, from lines 60 to 74. A word split into several tokens becomes the mean over its tokens (lines 127 to 129), and the start token keeps its share of every patch, because the map is never renormalised across words. ✍️ Blocks 3 and 4. The equations, worked on the miniature, are on [the network view](02-the-network-view.md#from-one-layer-to-a-words-stack).

> [!NOTE]
> Line 152 skips the whole resize when a model has one grid size and it is already 16, and those maps are then not renormalised. ✍️ The ladder's code walk, block 4. It does not apply to SDXL, whose grids are 64 and 32.

## Learn-to-Steer: the head

Navigation: ⬅️ [Learn-to-Steer: catching and preparing the maps](#learn-to-steer-catching-and-preparing-the-maps) | 📋 [TOC](#table-of-contents) | [Next](#learn-to-steer-the-loss-and-the-refine-loop) ➡️

**One aggregator, shared by both words, turns each map into a token, mixes the tokens, and pools them with a learned query; the classifier flattens the pair into one token and reads five logits.** ✍️ The ladder's code walk, blocks 5 and 6, at the commit's `model.py`.

```python
query = self.query_vector.repeat(x.shape[0], 1, 1)  # shape = (B, 1, d_model)
attn_output, _ = self.layer_attention(query, x, x)  # shape = (B, 1, d_model)
```

```python
x = x.view(x.size(0), -1)  # shape = (batch_size, 2 * 16 * 16)
x = self.projection(x)  # shape = (batch_size, d_model)
x = self.positional_encoding(x.unsqueeze(1))  # shape = (batch_size, 1, d_model)
x = self.transformer(x)  # shape = (batch_size, 1, d_model)
```

The first is the pooling (lines 176 to 177), the second the classifier's start (lines 76 to 80). The `unsqueeze(1)` makes the whole pair one token, so the transformer's attention has nothing to compare it with. ✍️ Figure: [the head reads the two stacks](02-the-network-view.md#the-head-reads-the-two-stacks). Shapes on SDXL, our proposal: (1,300, 16, 16) per word → (1,300, 256) → (1, 256) → (16, 16); the pair (2, 16, 16) → 512 → 256 → with a 16-number step vector, 272 → 5 logits. ✍️ The numbers file.

## Learn-to-Steer: the loss and the refine loop

Navigation: ⬅️ [Learn-to-Steer: the head](#learn-to-steer-the-head) | 📋 [TOC](#table-of-contents) | [Next](#learn-to-steer-training-data-and-training) ➡️

**At each of the first 25 of 50 steps the loop recomputes the maps, takes the cross-entropy against the asked relation, and subtracts 5 times its raw gradient from the latent, up to 16 times, before ordinary guidance runs from the moved latent.** ✍️ The ladder's code walk, blocks 7 to 9.

```python
for i in range(max_refinement_steps + 1):
    ...
    loss, latents = self.compute_loss(latents, timestep, iter_idx, store_results)
    ...
    if loss != 0 and loss > loss_target:
        latents = self.update_latents(latents, loss, iter_idx)
    else:
        break
```

```python
grad = torch.autograd.grad(loss, latents, create_graph=False, retain_graph=False)[0]
latents = latents - step_size * grad
```

`test_time_optimization.py`, lines 175 to 187 (the dots stand for its lines setting the relation index and the first-pass checks) and 137 to 141. `max_refinement_steps` is 15, so 16 passes. The loop stops once the loss reaches its target, 10⁻⁵ at sampler step 0 rising to 7 × 10⁻⁵ from step 3. ✍️ Block 8. On the miniature all 16 updates fire and "left of" ends at 0.9936. ✅ [The miniature page](04-the-miniature.md#stage-4-two-ways-to-apply-it).

🔤 **The name:** refine loop
- **refine**: improve by small repeated changes. **loop**: the same steps run again.
- **On the picture:** the top row's orange arrow from the loss back to the latent, travelled up to 16 times per sampler step.
- **Formula, in those words:** while the loss is above its target and fewer than 16 passes are done, x_t ← x_t − η ∇_{x_t} L_t.
- **Consequence:** each pass runs the U-Net and the head forward and backward, which is why the paper reports SD 2.1 going from 4.5 s to 90 s per image. ✅ The paper's text, p. 7. On the miniature the gradient's length falls from 0.7664 at pass 0 to 0.0098 at pass 15. ✅ The answer key.

The noise prediction is never edited; after the loop the ordinary step runs with guidance scale 7.5 on SD 2.1. ✍️ Block 9.

## Learn-to-Steer: training data and training

Navigation: ⬅️ [Learn-to-Steer: the loss and the refine loop](#learn-to-steer-the-loss-and-the-refine-loop) | 📋 [TOC](#table-of-contents) | [Next](#our-sampler-the-product-and-the-head-call) ➡️

**Each training image is inverted under its true relation and one random wrong relation, both carry the true relation as the label, and the head trains with AdamW on cross-entropy.** ✍️ The ladder's code walk, blocks 10 and 11.

```python
pos_prompt = self.generate_prompt(obj1, obj2, rel_name, rel_name)
src_prompts.append((pos_prompt, POS_PROMPT_TYPE, rel_name, [obj1, obj2]))
neg_rel_names = self.sample_negative_relations(rel_name, self.num_prompts)
for neg_rel_name in neg_rel_names:
    neg_prompt = self.generate_prompt(obj1, obj2, rel_name, neg_rel_name)
    src_prompts.append((neg_prompt, NEG_PROMPT_TYPE, rel_name, [obj1, obj2]))
```

`prompt_generator.py`, from lines 54 to 66 (one comment line dropped). The third field of each tuple, `rel_name`, is the true relation, so the wrong-word sample is labelled with the truth. Maps are cached at sampler steps 5, 10, 15, 20, 25 and 49. ✍️ Block 10. Training on the SD 2.1 settings: batch 64, learning rate 10⁻⁴, weight decay 0.05, at most 100 epochs with early stopping after 10. ✍️ Block 11. Why the wrong word is there is on [the network view](02-the-network-view.md#training-on-the-right-and-the-wrong-word).

**The training sentence, one setting at a time.** The settings above stack six training words, so here they are one per sentence. ✅ The arithmetic computed this session.

- **Batch 64**: the head sees 64 training examples, averages their losses, and makes one update from that average.
- **Learning rate 10⁻⁴**: the size of each update's step on the weights. Textbook pages are on [the learning rate card](../../../../../Obsidian/Artificial%20Intelligence/Machine%20Learning/Learning%20Rate.md#where-to-learn-this).
- **AdamW**: the rule that makes each update. Adam scales each weight's step by how large and how steady that weight's recent gradients have been; the W is weight decay kept as a separate step.
- **Weight decay 0.05**: at every update each weight is also shrunk toward 0 by the learning rate × 0.05, a factor of 1 − 10⁻⁴ × 0.05 = 0.999995. With no gradient pushing back, a weight of 1 would be 0.9512 after 10,000 updates, so only weights the loss keeps needing stay large.
- **Epoch**: one full pass over the training set; "at most 100" caps training at 100 passes. Textbook pages are on [the epoch card](../../../../../Obsidian/Artificial%20Intelligence/Machine%20Learning/Epoch.md#where-to-learn-this).
- **Early stopping after 10**: training ends early once 10 epochs in a row bring no improvement on the held-out check. 🔍 Read from the setting's usual meaning; the code walk's block 11 holds the rule.

Read whole again: batch 64, learning rate 10⁻⁴, weight decay 0.05, at most 100 epochs with early stopping after 10.

## Our sampler: the product and the head call

Navigation: ⬅️ [Learn-to-Steer: training data and training](#learn-to-steer-training-data-and-training) | 📋 [TOC](#table-of-contents) | [Next](#our-sampler-the-attention-push-already-run) ➡️

**Our sampler runs the U-Net once on three prompts, guides each subject's prediction, combines them, and asks any head for one number whose gradient it takes with respect to the latent.** ✅ Read in `_classifier_guided.py` this session.

```python
eps_a_raw, eps_b_raw, eps_uncond = noise.chunk(3)
eps_a = guided_eps(eps_a_raw, eps_uncond, guidance_scale)
eps_b = guided_eps(eps_b_raw, eps_uncond, guidance_scale)
return poe_eps(eps_a, eps_b, eps_uncond)
```

Lines 155 to 158 of `product_eps` (lines 149 to 158). Each subject's guided prediction is ε_∅ + g (ε_subject − ε_∅), and the product is their sum minus ε_∅ once. ✅ What the product is and why the empty prompt is taken off once is on [PoE composition](../../world/poe-composition.md#what-poe-composition-is).

🔤 **The name:** classifier-free guidance, the guidance scale g
- **guidance**: steering the sampler toward what the prompt asks. **classifier-free**: it needs no separate classifier, because the gap between the prediction with the prompt and the prediction with the empty prompt already points toward the prompt.
- **On the code:** `guided_eps(eps_a_raw, eps_uncond, guidance_scale)`, once per subject.

$$
\textcolor{#c0262d}{\underbrace{\varepsilon^{g}_{\text{cat}}}_{\text{the guided prediction}}}
= \textcolor{#6b7a88}{\underbrace{\varepsilon_\varnothing}_{\text{the empty prompt}}}
+ \underbrace{g}_{\text{guidance scale}}\,\big(\textcolor{#1f5fd0}{\underbrace{\varepsilon_{\text{cat}}}_{\text{told "a cat"}}} - \textcolor{#6b7a88}{\varepsilon_\varnothing}\big)
$$

- **Consequence, on one latent number** (toy values): ε_∅ = 0.2 and ε_cat = 0.5. At g = 0 the guided prediction is 0.2, the empty prompt alone; at g = 1 it is 0.5, the prompt's own prediction; at g = 7.5 it is 0.2 + 7.5 × 0.3 = 2.45, the gap stretched 7.5 times. With ε_dog = 0.1 the guided dog is 0.2 + 7.5 × (−0.1) = −0.55, and the product on that number is 2.45 + (−0.55) − 0.2 = 1.70. ✅ Computed this session. Textbook pages and videos are on [the classifier-free guidance card](../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Classifier-Free%20Guidance.md#where-to-learn-this).

```python
elif head is not None and strength != 0.0 and in_window:
    with torch.enable_grad():
        z = latents.detach().float().requires_grad_(True)
        ...
        log_p = head.log_prob_both(z, t_batch)
        grad = torch.autograd.grad(log_p, z)[0]
```

Lines 201 to 212; the dots stand for the timestep and signal-share lines and the branch for heads that read the decoded estimate instead. `in_window` is true when the step index lies in `guidance_window` (lines 188 to 189). ✅ A word-map head would be a class with `reads_estimate = False` and a `log_prob_both` that returns log p̂[r]; this file would then need no change. ✍️ The ladder's sampler walk, item 1.

## Our sampler: the attention push already run

Navigation: ⬅️ [Our sampler: the product and the head call](#our-sampler-the-product-and-the-head-call) | 📋 [TOC](#table-of-contents) | [Next](#our-sampler-the-push-and-the-cap) ➡️

**Our one push on SDXL's word maps keeps only the 32 × 32 layers, averages their heads and the 60 layers into one map per word, divides per patch over the prompt's words, and scores minus the correlation of the two maps.** ✅ Read in `attention_push.py` this session.

```python
if seq == 32 * 32:
    self.kept.append(probs.view(batch, attn.heads, seq, -1).mean(1).float())
```

```python
m = torch.stack(kept).mean(0)                                   # (2, 1024, 77)
wa = m[0, :, 1:1 + self.n_a]
wb = m[1, :, 1:1 + self.n_b]
a = (wa / wa.sum(-1, keepdim=True))[:, self.i_a - 1]
b = (wb / wb.sum(-1, keepdim=True))[:, self.i_b - 1]
```

```python
def correlation(self, x_t, timestep):
    a, b = self.maps(x_t, timestep)
    a, b = a - a.mean(), b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-8)
```

Lines 42 to 43, 77 to 81 and 84 to 87; `log_prob_both` returns minus the correlation (lines 92 to 93). ✅

| Question | Our attention push | Learn-to-Steer |
|---|---|---|
| how a map is normalised | per patch, over the prompt's words, start token dropped | per map, over the grid, start token left in |
| maps per word | 1 mean map from the 60 layers at 32 × 32 | all of them: 1,300 if moved to SDXL |
| where the two words come from | two prompts, "a cat" and "a dog" | one joint prompt |
| what is scored | a hand-written correlation | a trained classifier |
| steps | 0 to 19 in the 0-of-8 run | the first 25 of 50 |

✅ Our column read in the file; ✍️ Learn-to-Steer's from the ladder's sampler walk. This push drew two animals on 0 of 8 seeds (9 to 16), strong pushes zooming into one face; draft calls, no verdict recorded, and the report's row does not print the cap used. ✍️ [The early-push report](../../../2-classifier-guidance/report/classifier-guided-poe/does-an-early-push-separate-the-cat-and-dog.md#which-kind-of-push-moves-the-animals).

## Our sampler: the push and the cap

Navigation: ⬅️ [Our sampler: the attention push already run](#our-sampler-the-attention-push-already-run) | 📋 [TOC](#table-of-contents) | [Next](#what-a-head-on-sdxl-would-change) ➡️

**The gradient is scaled by the strength and the noise level, shortened to at most c times the product's length, subtracted from the product, and reaches the latent only through the DDIM step.** ✅ Read in `_classifier_guided.py` this session.

```python
sigma_t = torch.sqrt(1.0 - alpha_bar)
...
raw = float(strength) * sigma_t * grad
push, push_before_cap, capped = capped_push(
    raw, eps_p,
    push_cap_late if (push_cap_late is not None and step_index >= push_cap_late_from) else push_cap)
eps_p = (eps_p.float() - push).to(dtype)
```

Lines 213 and 228 to 232; the dots stand for the backward-guidance branch (lines 214 to 227), which only heads reading the decoded estimate use. ✅

```python
def capped_push(push: torch.Tensor, eps: torch.Tensor, cap: float | None):
    raw_norm = push.norm()
    if cap is None:
        return push, float(raw_norm), False
    limit = float(cap) * eps.float().norm()
    if raw_norm > limit:
        return push * (limit / raw_norm), float(raw_norm), True
    return push, float(raw_norm), False
```

Lines 49 to 61 without the docstring; `PUSH_CAP_FRACTION = 0.1` at line 46 is the default cap, and 0.4 is the proposed value for the abstract checkers. ✅ The file; ✍️ the numbers file for the proposal. On the miniature, the raw push of length 1.2262 is over the limit 0.4 × 1.4142 = 0.5657, so it is scaled by 0.4613. ✅ [The overview's worked push](01-overview.md#the-push-by-what-moves).

After the subtraction, line 246 forms the clean-latent estimate from the pushed prediction, `x0 = tweedie_mean(latents, alpha_bar_t, eps_p)`, and the DDIM step follows. The latent itself is never edited, which is the difference from Learn-to-Steer's loop. ✅

🔤 **The name:** Tweedie's estimate, `tweedie_mean`
- **Tweedie**: the statistician Maurice Tweedie, after whom the formula for the average clean value behind a noisy one is named. 🔍 Not rechecked against the original paper this session.
- **mean**: the average. Many clean latents could have been noised into this one, and the estimate is their average, the model's best single guess of where the step is heading.
- **Formula, in those words:** take the predicted noise, scaled by the noise level σ_t, off the latent, then divide by √ᾱ_t to undo the shrinking of the signal.

$$
\textcolor{#1d7a3e}{\underbrace{\hat x_0}_{\text{the clean guess}}} = \frac{\textcolor{#1f5fd0}{\underbrace{x_t}_{\text{the latent}}} - \sigma_t\,\textcolor{#c0262d}{\underbrace{\varepsilon_p}_{\text{the pushed prediction}}}}{\sqrt{\bar\alpha_t}}
$$

- **Consequence, on one latent number** (toy values, at the miniature's ᾱ_t = 0.36, σ_t = 0.8): x_t = 2 and ε_p = 0.5 give (2 − 0.8 × 0.5) / 0.6 = 2.6667. A push that lowers ε_p by 0.1 to 0.4 raises the guess to 2.8, so taking the push off the prediction moves the clean guess the opposite way, by σ_t / √ᾱ_t = 1.3333 times the push. ε_p = 0 gives 3.3333, the latent only unshrunk. ✅ Computed this session. What the estimate is used for across the project is on [where the data comes from](../../data/01-where-it-comes-from.md#the-journey).

🔤 **The name:** the canary
- **canary**: the bird miners carried underground, whose distress warned of bad air before the people noticed. Here, a check that fails loudly if a head changes what it should not.
- **On the code:** the comment at line 200, "The canary lives here: at strength 0 nothing below runs and eps_p is untouched", and the `strength != 0.0` test on line 201.
- **Formula, in those words:** at strength 0 the head is never called, so the rollout is byte-identical to the plain product sampler `run_cfg_poe`.
- **Consequence:** a head that moved the baseline would make every comparison column meaningless; any word-map head that goes through this branch keeps the guarantee. ✅ The module docstring, lines 20 to 22.

## What a head on SDXL would change

Navigation: ⬅️ [Our sampler: the push and the cap](#our-sampler-the-push-and-the-cap) | 📋 [TOC](#table-of-contents) | [Next](#where-the-code-and-the-paper-differ) ➡️

**This is our proposal, as the ladder's sampler walk lists it; nothing here is built or tested.** ✍️ [The ladder's sampler walk](../../../2-classifier-guidance/artifacts/scenes/word-map-checker/3-code/walk-our-sampler.md#what-a-learn-to-steer-head-on-sdxl-would-change).

1. **A head class with the sampler's interface.** `reads_estimate = False` and `log_prob_both(z, t)` returning log p̂[r]; the canary, the cap and the size-matched random push then all apply.
2. **Keep every map.** The attention push keeps only the 32 × 32 layers and averages their heads; the head needs both grids, 100 + 1,200 = 1,300 maps per word. The memory cost of holding 70 layers' maps with their gradients has not been measured.
3. **Copy the authors' map preparation.** Take the word's column of the raw 77-token softmax, resize to 16 × 16, divide by the sum over the grid. An antialiased resize is worth testing on the 100 maps from the 64 grid, because the plain one reads 4 of every 16 cells.
4. **Translate t into the head's step bucket.** On our 50-step schedule the step is (981 − t) / 20, and steps 0 to 15 fall in buckets 5, 10 and 15.
5. **Train on SDXL's own maps.** An authors' checkpoint would load onto 1,300 maps but read SDXL's layers in positions trained on another model's; training needs SDXL inversions under the true and a wrong relation word.
6. **Decide which prompts the training maps come from.** Inverting under the two single-subject prompts would keep the relation word out of the maps (inferred, untested in the paper).
7. **Add a class for "fused".**
8. **Test the dose before calling the head a failure.** Matching Learn-to-Steer needs an inner loop that updates `latents` before `product_eps` runs, or a much larger cap; an inner loop adds up to 16 U-Net passes with gradients per pushed step.
9. **Expect the earliest steps to be weak.** A word-map detector on SD 2.1 scores 18.0 of 100 at t = 800 against 58.4 at t = 200, and 10 of our 16 pushed steps sit above t = 800. Whether that carries to SDXL is untested.

Loop 7's design proposes a different input and push for items 2, 3 and 4; both are listed under [where the sources disagree](01-overview.md#where-the-sources-disagree).

## Where the code and the paper differ

Navigation: ⬅️ [What a head on SDXL would change](#what-a-head-on-sdxl-would-change) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**Six places where the released code and the paper say different things, each checked by the ladder against the file and line at `a2ae427`.** ✍️ The ladder's code walk; the paper's side ✅ read in its saved text where quoted.

| What | The paper | The code | Where in the code |
|---|---|---|---|
| pooling the maps into one | "another four-layer, single-head transformer", with a learned query (p. 15) | one attention layer with a learned query: 4 heads on FLUX.1-dev, 2 on SD 2.1 | `model.py:133-135,176-177` |
| the classifier's transformer | the two-channel image flattened, projected and passed through a transformer (p. 15) | the whole pair is one token, so its attention attends only to itself | `model.py:76-80` |
| iterations per step | "number of optimization iterations per step are 15" (p. 16) | `range(15 + 1)` passes, each able to update: up to 16 | `test_time_optimization.py:175-187` |
| what "timestep" means to the head | the denoising timestep t (p. 4) | the step index 0 to 49, snapped up to 5, 10, 15, 20, 25 or 49 | `test_time_optimization.py:59-74,86-91,115` |
| training epochs on SD 2.1 | 150 for SD models (p. 16) | 100 in `sd2_config.yaml`, 150 in `sd14_config.yaml` | `sd2_config.yaml:30` |
| inversion guidance on SD | guidance scale 0 (p. 5) | `guidance_scale` 1.0, which this pipeline treats as no guidance | `generate_relations_data.py:163` |

The last row changes nothing if the paper's 0 means "guidance off". ✍️ The ladder.

## Where this came from

Navigation: ⬅️ [Where the code and the paper differ](#where-the-code-and-the-paper-differ) | 📋 [TOC](#table-of-contents)

| What | How it was established | When |
|---|---|---|
| Every excerpt from our two files, with its line numbers | Read in `_classifier_guided.py` and `attention_push.py` at `cbff8c0e`; `git log` shows each unchanged since `3867b3f1` and `ff028dd5`, `git status` shows both clean | 2026-11-24 |
| Every Learn-to-Steer excerpt and line number | Stated by the ladder's code walk at `a2ae427`; the checkout is not in this repository, so not rechecked | 2026-11-20 |
| Paper quotes on pp. 7, 15 and 16 | Read in the ladder's saved text of the paper | 2026-11-24 |
| `--attention` in `render_results.py` turning the attention push on | Stated by the ladder's sampler walk; not checked here | 2026-11-20 |
| The miniature twins and their line numbers | Read in the ladder's `miniature.py` | 2026-11-24 |
| The 0-of-8 row | Read in the early-push report, line 56 | 2026-11-24 |
| The proposal's nine items | Stated by the ladder's sampler walk, lines 261 to 277 | 2026-11-24 |

⬅️ Previous: [the miniature](04-the-miniature.md) | 📋 [The pages of this method](01-overview.md#the-pages-of-this-method) | Next: [the methods index](../00-INDEX.md) ➡️

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
