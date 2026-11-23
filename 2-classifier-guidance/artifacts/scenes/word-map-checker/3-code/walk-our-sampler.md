# Our sampler's push, line by line

This page walks the two files that would carry a word-map checker on SDXL: [the classifier-guided product sampler](../../../../../poe_repair/methods/_classifier_guided.py), which adds a push to the product of experts, and [the attention push](../../../../scripts/rung2/attention_push.py), the one push we have already run on word maps. It ends with the list of changes a Learn-to-Steer head would need.

**What was checked.** `_classifier_guided.py` last changed at commit `3867b3f1` and `attention_push.py` at `ff028dd5`, both on 2026-11-18, and neither has uncommitted edits. Line numbers below are from those versions. Worked values come from [the miniature's answer key](../2-miniature/miniature_answer_key.json), full-scale values from [the ladder's numbers file](../numbers/numbers.json), and symbols from [the ladder's notation](../NOTATION.md).

**Nothing on this page was run for the ladder.** Learn-to-Steer has never run on SDXL. The last section is our proposal, and none of it is built.

## Table of contents

1. [The product's noise prediction](#1-the-products-noise-prediction)
2. [The window and the head call](#2-the-window-and-the-head-call)
3. [How the attention push reads the maps](#3-how-the-attention-push-reads-the-maps)
4. [The attention push's score, and what it drew](#4-the-attention-pushs-score-and-what-it-drew)
5. [From gradient to push](#5-from-gradient-to-push)
6. [The cap](#6-the-cap)
7. [The push enters the DDIM step](#7-the-push-enters-the-ddim-step)
8. [The canary](#8-the-canary)
- [How this differs from Learn-to-Steer](#how-this-differs-from-learn-to-steer)
- [What a Learn-to-Steer head on SDXL would change](#what-a-learn-to-steer-head-on-sdxl-would-change)
- [Cross-references](#cross-references)

## 1. The product's noise prediction

**Role: ordinary sampler.**

**Code.** `poe_repair/methods/_classifier_guided.py`, lines 149 to 158.

```python
def product_eps(timestep) -> torch.Tensor:
    latent_input = scheduler.scale_model_input(latents.repeat(3, 1, 1, 1), timestep)
    noise = unet(
        latent_input, timestep, encoder_hidden_states=pe,
        added_cond_kwargs=cond, timestep_cond=None,
    ).sample
    eps_a_raw, eps_b_raw, eps_uncond = noise.chunk(3)
    eps_a = guided_eps(eps_a_raw, eps_uncond, guidance_scale)
    eps_b = guided_eps(eps_b_raw, eps_uncond, guidance_scale)
    return poe_eps(eps_a, eps_b, eps_uncond)
```

**The equation:**

$$
\varepsilon_{\text{PoE}} = \varepsilon^{g}_{\text{cat}} + \varepsilon^{g}_{\text{dog}} - \varepsilon_\varnothing, \qquad \varepsilon^{g}_{c} = \varepsilon_\varnothing + g\,(\varepsilon_{c} - \varepsilon_\varnothing)
$$

**Symbols.** $\varepsilon_c$ is the U-Net's prediction for the prompt "a cat" or "a dog", $\varepsilon_\varnothing$ for the empty prompt, and $g$ the guidance scale. The two subjects run in two separate prompts through one three-branch forward pass.

**Shape.** $3 \times 4 \times 128 \times 128$ in and out of the U-Net, $1 \times 4 \times 128 \times 128$ after the combine.

**Worked value.** The miniature uses $g = 1$, $\varepsilon_{\text{cat}} = \varepsilon_{\text{dog}}$ with entries of $\pm 0.25$ and $\varepsilon_\varnothing = 0$, so $\varepsilon_{\text{PoE}}$ has entries of $\pm 0.5$ and norm 1.4142.

## 2. The window and the head call

**Role: steering.**

**Code.** `poe_repair/methods/_classifier_guided.py`, lines 188 to 189 and 201 to 212.

```python
elif head is not None and strength != 0.0 and in_window:
    with torch.enable_grad():
        z = latents.detach().float().requires_grad_(True)
        t_batch = timestep.reshape(1).to(device=device).expand(z.shape[0])
        alpha_bar = scheduler.alphas_cumprod[int(timestep.item())].to(device=device).float()
        if getattr(head, "reads_estimate", False):
            # a picture checker: it decodes the estimate of the finished picture from x_t and
            # this step's product prediction, held fixed (rung2/clip_push.py)
            log_p = head.log_prob_both(z, t_batch, eps=eps_p.detach().float(), alpha_bar=alpha_bar)
        else:
            log_p = head.log_prob_both(z, t_batch)
        grad = torch.autograd.grad(log_p, z)[0]
```

**The equation:**

$$
g_t = \nabla_{x_t} \log p(y \mid x_t)
$$

**Symbols.** $y$ is whatever the head scores. The sampler only asks a head for one number, `log_prob_both(z, t)`, which is why a new checker can plug in with no change to this file. $g_t$ has the latent's shape.

**The window.** `guidance_window=(first, stop)` pushes on steps `first <= step < stop`. The proposed window for the abstract checkers is steps 0 to 15 of 50. SDXL's shipped scheduler gives those steps timesteps 981 down to 681, and 10 of the 16 sit above $t = 800$.

## 3. How the attention push reads the maps

**Role: steering.**

**Code.** `2-classifier-guidance/scripts/rung2/attention_push.py`, lines 36 to 45 and 65 to 82.

```python
probs = attn.get_attention_scores(query, key, attention_mask)
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

**The equations.** Heads are averaged within a layer, the 60 layers at $32 \times 32$ are averaged into one table, and the start token is dropped:

$$
\bar m[p, i] = \frac{1}{60}\sum_{l=1}^{60} \frac{1}{H_l}\sum_{h=1}^{H_l} A^{(l,h)}[p, i], \qquad
a[p] = \frac{\bar m[p, i_{\text{cat}}]}{\sum_{j \in \text{words of the prompt}} \bar m[p, j]}
$$

**Symbols.** $p$ is one of the 1024 patches, $i_{\text{cat}}$ the position of "cat" in its own prompt and $j$ runs over that prompt's word tokens. For the prompt "a cat" the word tokens are "a" and "cat". The dog map $b$ is the same computation on the prompt "a dog".

**What the division does.** It is per patch and over words, so $a[p]$ is the share of patch $p$'s word attention that "cat" holds. Learn-to-Steer divides per map and over patches ([the table below](#how-this-differs-from-learn-to-steer)).

**Shape.** One map per word: $(1024,)$, a $32 \times 32$ grid written flat. The 64 grid's 10 layers are not read. Across all layers and heads SDXL offers 1300 maps per word, and this reader collapses them to 1.

## 4. The attention push's score, and what it drew

**Role: steering. This is the push we already tried on word maps.**

**Code.** `2-classifier-guidance/scripts/rung2/attention_push.py`, lines 84 to 93. It is turned on by `--attention` in `2-classifier-guidance/scripts/rung2/render_results.py`, lines 272 to 276.

```python
def correlation(self, x_t, timestep):
    a, b = self.maps(x_t, timestep)
    a, b = a - a.mean(), b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-8)
```

**The equation:**

$$
\log p \;\longleftarrow\; -\operatorname{corr}(a, b) = -\frac{(a-\bar a)^\top (b-\bar b)}{\lVert a-\bar a\rVert\,\lVert b-\bar b\rVert}
$$

The left side is a score standing in for a log probability. A correlation of 1 means both words light up the same patches.

**Why it was tried.** The overlap of the top 10% of the "cat" and "dog" maps, as intersection over union (1 means the same places), has a median over 8 seeds of 0.506 to 0.814 across steps 2 to 40 for the product and 0.225 to 0.394 for the joint prompt "a cat and a dog" ([the report on the early push](../../../../report/classifier-guided-poe/does-an-early-push-separate-the-cat-and-dog.md)).

**What it drew.** **This push drew two animals on 0 of 8 SDXL seeds**, on seeds 9 to 16 with the push on steps 0 to 19. These are draft calls and no verdict is recorded. Strong pushes zoomed into one face. The report's row does not print the cap used.

**How far that result reaches.** It is one hand-written loss. The Learn-to-Steer note warns against reading it as evidence against a learned attention checker: on SD 2.1 the learned loss reaches 0.54 GenEval position (0 to 1, higher is better) where the hand-written STORM loss reaches 0.19 ([Table 1](https://arxiv.org/pdf/2509.02295v2#page=5)), though on a joint prompt and a different model (inferred).

**What the miniature adds.** When two maps are identical their correlation is at its maximum, so its gradient is zero. On the miniature's fused body the gradient norm is 0.0, and after nudging one latent value by 0.1 it is 0.000382. The spatial head's gradient on the same fused body has norm 0.7664. SDXL's maps are never exactly identical, so this shows the shape of the loss near a perfect overlap and does not explain the 0 of 8.

## 5. From gradient to push

**Role: steering.**

**Code.** `poe_repair/methods/_classifier_guided.py`, lines 213 and 228.

```python
sigma_t = torch.sqrt(1.0 - alpha_bar)
```

```python
raw = float(strength) * sigma_t * grad
```

**The equation** (Dhariwal and Nichol's Algorithm 2, with nothing rescaled):

$$
\text{raw} = \lambda\, \sigma_t\, \nabla_{x_t} \log p(y \mid x_t)
$$

**Symbols.** $\lambda$ is the push's strength (the code's `strength`) and $\sigma_t = \sqrt{1-\bar\alpha_t}$ the noise scale, which shrinks the push as the picture clears.

**Worked value.** With $\lambda = 2$ and $\sigma_t = 0.8$, the miniature's raw push is $(0.15, -0.15, 0.5945, -0.5945)$ on the cat channel and the negative of that on the dog channel, with norm 1.2262.

## 6. The cap

**Role: steering. Our addition, absent from both Dhariwal and Nichol and Learn-to-Steer.**

**Code.** `poe_repair/methods/_classifier_guided.py`, lines 46, 49 to 61 and 229 to 231.

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

**The equation:**

$$
\text{push} = \text{raw} \cdot \min\!\left(1,\ \frac{c\,\lVert\varepsilon_{\text{PoE}}\rVert}{\lVert\text{raw}\rVert}\right)
$$

**Symbols.** $c$ is the cap. The default `PUSH_CAP_FRACTION` is 0.1 (line 46), and the proposed value for the abstract checkers is 0.4. Scaling keeps the direction and changes only the length.

**Worked value.** The miniature's limit is $0.4 \times 1.4142 = 0.5657$. The raw norm 1.2262 is above it, so the push is scaled by 0.4613 to norm 0.5657.

## 7. The push enters the DDIM step

**Role: steering, then ordinary sampler.**

**Code.** `poe_repair/methods/_classifier_guided.py`, lines 232 and 243 to 262.

```python
eps_p = (eps_p.float() - push).to(dtype)
```

```python
x0 = tweedie_mean(latents, alpha_bar_t, eps_p)
```

**The equations:**

$$
\begin{aligned}
\varepsilon' &= \varepsilon_{\text{PoE}} - \text{push}\\
\hat x_0 &= \frac{x_t - \sigma_t\, \varepsilon'}{\sqrt{\bar\alpha_t}}\\
x_{t-1} &= \sqrt{\bar\alpha_{t-1}}\,\hat x_0 + \sigma_{t-1}\,\varepsilon'
\end{aligned}
$$

**Symbols.** $\hat x_0$ is the estimate of the finished latent and $t-1$ the next timestep on the sampler's schedule. The latent itself is never edited, so the push reaches it only through this step.

**How much of the push reaches the latent.** Substituting gives $\partial x_{t-1}/\partial \varepsilon' = \sigma_{t-1} - \sigma_t \sqrt{\bar\alpha_{t-1}/\bar\alpha_t}$. With the miniature's $\bar\alpha_t = 0.36$ and $\bar\alpha_{t-1} = 0.64$ that is $0.6 - 0.8 \times 0.8/0.6 = -0.4667$, so a push of norm 0.5657 moves the latent by $0.4667 \times 0.5657 = 0.264$.

**Worked value.** After the capped push and one DDIM step the head's probability for "left of" goes from 0.2 to 0.2367. Without the push it stays at 0.2.

## 8. The canary

**Role: a check on the code.**

**Code.** `poe_repair/methods/_classifier_guided.py`, lines 20 to 22 and 200 to 201.

```python
# The canary lives here: at strength 0 nothing below runs and eps_p is untouched.
elif head is not None and strength != 0.0 and in_window:
```

**What it guarantees.** At `strength == 0` the head is never called, so the rollout is byte-identical to the plain product sampler `run_cfg_poe`. A head that moved the baseline would make every comparison column meaningless, so any new head keeps this property by going through this branch.

## How this differs from Learn-to-Steer

| Question | Our attention push | Learn-to-Steer | Where |
|---|---|---|---|
| How a map is normalised | per position over the prompt's words, start token dropped | per map over the grid, start token left in the softmax | `attention_push.py:78-81`; `utils.py:68-69` in the authors' code |
| How many maps per word | one mean map from the 60 layers at 32 × 32 | all maps, each resized to 16 × 16: 195 on SD 2.1 (inferred), 1368 on FLUX.1-dev, 1300 if moved to SDXL | `attention_push.py:42-43,77`; `utils.py:51-74` |
| Where the two words come from | two prompts, "a cat" and "a dog", so the words never compete in one softmax | one joint prompt, so the words compete at every patch | `attention_push.py:70-79`; `pipeline_sd2_debug.py:347-350` |
| What is scored | a hand-written correlation | a trained classifier | `attention_push.py:84-87`; `model.py:187-223` |
| Where the gradient goes | a push on the noise prediction | an update on the latent, before the prediction | `_classifier_guided.py:232`; `test_time_optimization.py:140` |
| Size of the step | one push per step, capped at $c\lVert\varepsilon_{\text{PoE}}\rVert$ | up to 16 updates per step at $\eta = 5$, no cap | `_classifier_guided.py:49-61`; `test_time_optimization.py:175-187` |
| Which steps | steps 0 to 19 in the 0-of-8 run | the first 25 of 50 | the report's table row; `run.py:56` |

**Competition, on the miniature.** At a fused patch with cat-ness and dog-ness both 2 and key 2, each word in its own prompt takes 0.982 of the patch. In one joint prompt the same two words split it and "cat" takes 0.4955.

**The dose, on the miniature.** From the same fused start, one capped push and one DDIM step reach $\hat p[\text{left of}] = 0.2367$ and move the latent by 0.264. Learn-to-Steer's 16 updates at $\eta = 5$ reach 0.9936 and move it by 6.0072. The weights are set by hand, so the ratio shows the size of the gap between the two doses and predicts nothing on SDXL.

## What a Learn-to-Steer head on SDXL would change

This is our proposal. Nothing here is built or tested, and Learn-to-Steer has never run on SDXL.

1. **Add a head class with the sampler's interface.** A new class exposes `reads_estimate = False` and `log_prob_both(z, t_batch)` returning $\log \hat p[r]$. Then `_classifier_guided.py` needs no change for a one-push-per-step version, and the canary, the cap and the size-matched random push all apply.

2. **Keep every map.** `_KeepCrossAttention` keeps only the 32 × 32 layers and averages their heads (`attention_push.py:42-43`). The head needs both grids per head: 100 maps from the 64 grid and 1200 from the 32 grid, 1300 per word. The memory cost of holding 70 layers' maps with their gradients has not been measured.

3. **Copy the authors' map preparation.** Take the word's column of the raw 77-token softmax, resize to 16 × 16 and divide by the sum over the grid, as `utils.py:51-74` does. Bilinear resizing from 64 to 16 reads only the middle 4 of each block's 16 cells, so an antialiased resize is a real alternative to test on the 100 maps from the 64 grid.

4. **Translate the timestep into the head's step bucket.** The sampler hands the head $t$, and the head's time tables want a step number. On our 50-step schedule the step is $(981 - t)/20$, and steps 0 to 15 fall in buckets 5, 10 and 15. Steps 0 to 4 would be read with bucket 5.

5. **Train on SDXL's own maps.** The head's weight count does not depend on the number of maps (1,964,581 in the SD 2.1 configuration, 4,860,965 in the FLUX.1-dev one), so an authors' checkpoint would load onto 1300 maps. It would then read SDXL's layers in positions trained on another model's, so its output would carry no meaning. Training needs SDXL inversions with the true and a wrong relation word.

6. **Decide which prompts the training maps come from.** The authors' maps come from one joint prompt. Ours would come from two single-subject prompts, where the words do not compete. The note suggests also inverting under the two single-subject prompts so the relation word never enters the maps (inferred, untested in the paper).

7. **Add a class for "fused".** The head has four relations plus "neutral", and on the miniature's identical maps it gives 0.2 to all five. Fusion is the failure this project targets, so the note proposes a fused class (inferred).

8. **Test the dose before calling the head a failure.** The sampler's single capped push is far smaller than Learn-to-Steer's loop. Matching it needs either an inner loop that updates `latents` before `product_eps` is called, or a much larger cap. An inner loop adds up to 16 U-Net passes with gradients per pushed step.

9. **Expect the earliest steps to be weak.** A word-map detector in Detector Guidance scores 18.0 mAP at IoU 0.5 (0 to 100, higher is better) at $t = 800$ against 58.4 at $t = 200$ on SD 2.1, and 10 of our 16 pushed steps sit above $t = 800$. Whether this carries to SDXL is untested.

## Cross-references

- [The walk through the authors' code](walk-learn-to-steer.md) covers every Learn-to-Steer line named in the table.
- [The rebuild companion](../rebuild-from-fundamentals.md) takes each equation here down to first-course maths.
- [The report on the early push](../../../../report/classifier-guided-poe/does-an-early-push-separate-the-cat-and-dog.md) holds the attention push's result row.
- [The checked reading note on Learn-to-Steer](../../../../../artifacts/notes/progress/abstract-checkers/01-reading/learn-to-steer.md) holds the design advice cited above.
