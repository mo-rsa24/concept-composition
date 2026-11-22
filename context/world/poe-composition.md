# 🌍 PoE composition (and Mono, the target it is compared against)

Product-of-Experts (PoE) composition is the method under study: a way to make SDXL draw two
named concepts at once by [combining two separate single-concept predictions](../../../../goal-setting/learning/deep-learning/diffusion-models/05-poe-composition-diffusion/plans/07-poe-factorisation-and-normaliser.md) at every denoising
step, instead of retraining the model or hand-writing a joint prompt. It is the thing that fails
(see [chimera.md](chimera.md)) and the thing the project's fix (see
[lora-corrector.md](lora-corrector.md)) is trying to repair. Sibling file
[interaction-term.md](interaction-term.md) explains the specific gap between PoE and its target.

**Product, on three candidate pictures.** Each expert scores every candidate, and the scores are multiplied, so a picture needs both experts' approval. Scores invented for the toy (see the [Product of Experts card](../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Product%20Of%20Experts.md#where-to-learn-this)):

| Candidate | Cat expert | Dog expert | Product | Share after rescaling |
|---|---|---|---|---|
| a cat alone | 0.5 | 0.1 | 0.05 | 0.19 |
| a dog alone | 0.1 | 0.5 | 0.05 | 0.19 |
| both | 0.4 | 0.4 | 0.16 | 0.62 |

Lower the cat expert's score for "both" from 0.4 to 0.1 and the product falls from 0.16 to 0.04, so the share falls from 0.62 to 0.29. The logs of the three products are −3.00, −3.00 and −1.83, the sum of the two experts' logs, which is why the combination below is described as adding predictions.

![A PoE render of "a cat" and "a dog" composed by Product-of-Experts, seed 42, no correction applied](../images/world/01-poe-chimera-example.png)
*This is the failure mode by default: PoE has fused the two animals into one creature rather than
placing a cat and a dog side by side.*

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [What PoE composition is](#what-poe-composition-is)
- [What it looks like](#what-it-looks-like)
- [Why the project cares](#why-the-project-cares)
- [How it shows up in the data](#how-it-shows-up-in-the-data)
- [What people get wrong](#what-people-get-wrong)
- [Where this came from](#where-this-came-from)
- [🖼️ In pictures, code and further reading](#in-pictures-code-and-further-reading)
- [Cross-references](#cross-references)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-poe-composition-is) ➡️

- **Expert**: one of the two single-concept predictions PoE combines, e.g. the model's prediction
  for "a cat" alone, and separately for "a dog" alone.
- **Denoising step**: diffusion generation runs a fixed schedule of steps (here, 50) that turn
  pure noise into an image; each step's prediction can be combined differently.
- **Mono**: the alternative to PoE, literally typing the joint prompt ("a cat and a dog") and
  letting the model handle composition itself. Used only as the target the fix is measured
  against, never shipped as the method, because typing every joint prompt by hand is the thing
  PoE composition exists to avoid.

## What PoE composition is

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#what-it-looks-like) ➡️

**At every denoising step, PoE runs the model once per concept and combines the two predictions
into one, instead of running the model once on a joint prompt.** ✅

Read directly from the paper's own description (`writing/iclr/iclr2027_conference.tex`, abstract):
composing pretrained diffusion models "at inference time by sampling from a product of experts,
one expert per concept." The combination is a form of linear score composition: each expert's
prediction is added (with classifier-free guidance weighting) rather than the model being asked to
reason about both concepts together in one forward pass. ✅

**Mono is the same model, given the literal joint prompt instead.** ✅

`monolithic.png` in a pilot run is the Mono render: one forward pass of SDXL on "a cat and a
dog" together. It usually succeeds at showing both concepts, because the model was trained on
image-caption pairs that already describe multiple things together. It is the ceiling PoE is
compared against, not a proposed method, per `MASTER_PLAN.md`'s glossary: "Mono / the ceiling: the
cheat that works... it composes fine, but it defeats the point, so we only use it as the target." ✅

## What it looks like

Navigation: ⬅️ [What PoE composition is](#what-poe-composition-is) | 📋 [TOC](#table-of-contents) | [Next](#why-the-project-cares) ➡️

![The Mono render of the same pair and seed: the literal joint prompt "a cat and a dog", one forward pass, no PoE combination](../images/world/02-mono-ceiling-example.png)
*Both animals are recognisable and separated, unlike the PoE render above. This is the target the
correction tries to reach without ever seeing this joint prompt at inference.*

Both images are existing pipeline output, copied from `data/pilot/seed_42/a_cat__x__a_dog/` on
2026-08-24, not newly captured.

## Why the project cares

Navigation: ⬅️ [What it looks like](#what-it-looks-like) | 📋 [TOC](#table-of-contents) | [Next](#how-it-shows-up-in-the-data) ➡️

**The gap between the PoE render and the Mono render, step by step, is exactly what the project
tries to measure and then close.** ✅

That gap is named and defined in [interaction-term.md](interaction-term.md). Whether the project's
LoRA composes "like Mono" is the whole test, per
[purpose/03-what-working-means.md](../purpose/03-what-working-means.md#the-test).

## How it shows up in the data

Navigation: ⬅️ [Why the project cares](#why-the-project-cares) | 📋 [TOC](#table-of-contents) | [Next](#what-people-get-wrong) ➡️

| Column | Stands for | Example | Entry |
|---|---|---|---|
| `arm` | which render method produced a given run: PoE, Mono, or a corrected variant | `poe` (example) | [Dictionary § arm](../data/02-dictionary.md#arm) |
| `model_id` | the exact pretrained model both PoE and Mono run on | `stabilityai/stable-diffusion-xl-base-1.0` | [Dictionary § model_id](../data/02-dictionary.md#model_id) |

## What people get wrong

Navigation: ⬅️ [How it shows up in the data](#how-it-shows-up-in-the-data) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**Mono is not a candidate fix.** ✍️ It is only ever the reference the correction is scored
against. A result that quietly lets the sampler fall back to the Mono prediction at full
correction strength would make every method look equally good for the wrong reason; this is
recorded as a real risk the project checked for (`context/research-guidelines.md`: "a full-strength
shortcut would have made every control row reproduce the real one... switching it off during
control runs is a decision recorded in the design plan").

## Where this came from

Navigation: ⬅️ [What people get wrong](#what-people-get-wrong) | 📋 [TOC](#table-of-contents) | [Next](#in-pictures-code-and-further-reading) ➡️

| What | How it was established | When |
|---|---|---|
| PoE combines one expert prediction per concept per step | Read in `writing/iclr/iclr2027_conference.tex`, abstract | 2026-08-24 |
| Mono is the joint-prompt cheat, used only as the target | Read in `MASTER_PLAN.md`, Glossary | 2026-08-24 |
| The two example images | Read directly, `data/pilot/seed_42/a_cat__x__a_dog/{poe,monolithic}.png` | 2026-08-24 |
| The full-strength shortcut risk and its fix | Read in `context/research-guidelines.md` | 2026-08-24 |

<!-- related:start (drain) -->
<a id="in-pictures-code-and-further-reading"></a>

## 🖼️ In pictures, code and further reading

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents) | [Next](#cross-references) ➡️

The page's two pictures are one pair and one seed. The pictures below put the plain product beside the joint prompt on several seeds of two models, SDXL and PixArt, then beside the joint prompt of four published models at one seed, and the code shows the two sums.

### How it fits together

<a href="../../3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/poe-against-the-joint/what-the-fusion-is-not-caused-by/summary-grid.png"><img src="../../3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/poe-against-the-joint/what-the-fusion-is-not-caused-by/summary-grid.png" width="200" alt="Three setups, each with a joint prompt column and a summed prompt column"></a>

Three setups side by side, SDXL at its usual settings, SDXL at PixArt's settings and PixArt, each with a column for the joint prompt ("a cat and a dog") and a column for the sum of the two single prompts, which is the product. The sheet shows 4 seeds per setup in two columns, 24 tiles in all. Every summed tile shows one animal and every joint tile shows two animals `verified` (opened and read by eye). The footer says 0 of 8 seeds give two clear animals in the summed column in each setup and that the full sheets are in "the two results folders", so the 4 per setup on this sheet are a subset `verified` (read from the picture's footer text).

### The code behind it

Two methods, one function each. The product sums two guided predictions and takes the empty prompt off once. Excerpts from `poe_repair/_sdxl/metrics.py` (lines 12 to 17) and `poe_repair/methods/_sampling.py` (lines 168 to 171):

```python
def guided_eps(eps_cond: torch.Tensor, eps_uncond: torch.Tensor, guidance_scale: float) -> torch.Tensor:
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)


def poe_eps(eps_a: torch.Tensor, eps_b: torch.Tensor, eps_uncond: torch.Tensor) -> torch.Tensor:
    return eps_a + eps_b - eps_uncond
```

```python
        eps_a_raw, eps_b_raw, eps_uncond = noise.chunk(3)
        eps_a = guided_eps(eps_a_raw, eps_uncond, guidance_scale)
        eps_b = guided_eps(eps_b_raw, eps_uncond, guidance_scale)
        eps_p = poe_eps(eps_a, eps_b, eps_uncond)
```

- The model is run once on a batch of three prompts (concept A, concept B, empty) and the output is split into three.
- `guided_eps` is the guidance weighting the page mentions: the empty prediction plus `guidance_scale` times the distance from the concept's prediction to it.
- `poe_eps` adds the two guided predictions and subtracts the empty one once, because each guided prediction already contains it.

A small invented example, not from a run, for one number in the prediction: empty 0.2, concept A 0.5, concept B 0.4, guidance scale 2. Guided A is 0.2 + 2 × (0.5 − 0.2) = 0.8 and guided B is 0.2 + 2 × (0.4 − 0.2) = 0.6, so the product is 0.8 + 0.6 − 0.2 = 1.2.

The joint prompt method is the same single-prompt sampler with one prompt. Excerpt from `poe_repair/run.py` (lines 128 to 137):

```python
    elif name == "poe":
        seq_a, pool_a = _encode(cell.prompt_a, ctx)
        seq_b, pool_b = _encode(cell.prompt_b, ctx)
        out = run_cfg_poe(
            seq_a=seq_a, pool_a=pool_a, seq_b=seq_b, pool_b=pool_b, **common,
        )
    elif name == "mono":
        joint = joint_prompt(cell.prompt_a, cell.prompt_b, template=ctx.joint_template)
        seq_j, pool_j = _encode(joint, ctx)
        out = run_cfg(seq_cond=seq_j, pool_cond=pool_j, **common)
```

- `poe` encodes the two prompts separately and calls the product sampler.
- `mono` builds one prompt from the two with a template and calls the plain single-prompt sampler `run_cfg`, whose noise prediction is `guided_eps` alone (lines 227 and 228 of `_sampling.py`).

### Further illustrations

**Evidence**

<a href="../../1-lora-correction/artifacts/results/measuring-composition/the-grids/what-the-main-study-grids-show/published-models-then-ours.png"><img src="../../1-lora-correction/artifacts/results/measuring-composition/the-grids/what-the-main-study-grids-show/published-models-then-ours.png" width="200" alt="Cat and dog seed 42 across SD 1.4, SD 2.1, SD 3.5, SDXL and CO3"></a>

Cat and dog at seed 42, the page's own pair and seed. From the left: SD 1.4, SD 2.1, SD 3.5 and SDXL on the joint prompt, then SDXL's plain product, then CO3 (product with its own corrector), then an empty panel for this project's adapter marked "not rendered" `verified` (opened). The SDXL joint tile shows a cat and a dog and the SDXL plain product tile shows one animal `verified`. The CO3 tile shows a tabby and a black animal, both cat-shaped by eye, and no dog `verified`; the card says it starts from its own noise, so it is not a like-for-like tile `stated`.

**The card and the picture differ.** The card lists three rows (cat and dog, butterfly and flower meadow, camel and forest). The picture holds the cat and dog row only. Neither is changed.

### Go deeper

| What | Link |
|---|---|
| The failure this method produces | [chimera](chimera.md) |
| The gap between the two methods | [interaction term](interaction-term.md) |
| The fix | [LoRA corrector](lora-corrector.md) |
| The three-setup sheet's card | [what the fusion is not caused by](../../3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/poe-against-the-joint/what-the-fusion-is-not-caused-by/README.md) |
| The grids card | [what the main study's grids show](../../1-lora-correction/artifacts/results/measuring-composition/the-grids/what-the-main-study-grids-show/README.md) |
| The linked learning walk on the product's factorisation | [PoE factorisation and normaliser](../../../../goal-setting/learning/deep-learning/diffusion-models/05-poe-composition-diffusion/plans/07-poe-factorisation-and-normaliser.md) |

### Still open

- The page's two pictures are byte-identical to `data/pilot/seed_42/a_cat__x__a_dog/poe.png` and `monolithic.png` `verified` (compared with `cmp`).
<!-- related:end -->

## Cross-references

Navigation: ⬅️ [Previous](#in-pictures-code-and-further-reading) | 📋 [TOC](#table-of-contents)

- The mention of **a product of experts** (composing SDXL as an additive combination of two expert predictions) in [the scene map for the Gaussian score's base case](../../../../goal-setting/learning/deep-learning/diffusion-models/05-poe-composition-diffusion/artifacts/drips/gaussian-score-base-case/SCENE_MAP.md) (relevance match).
- The mention of **product-of-experts** in [the finding "does a Langevin corrector remove part of the correction"](../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-a-langevin-corrector-remove-part-of-the-correction.md).
- The mention of **product-of-experts** in [the finding "does a corrector alone produce two animals"](../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md).
- The mention of **product-of-experts composition** in [the figure plan "the same rule on four models"](../../1-lora-correction/plans/03-does-the-correction-cause-composition/plans/figures/12-the-same-rule-on-four-models.md).
- The mention of **PoE** in [the review of "the same rule on four models"](../../1-lora-correction/plans/03-does-the-correction-cause-composition/review/12-the-same-rule-on-four-models.md).
- The mention of **product-of-experts composition** in [the finding "does the adapter return two when both prompts name the same concept"](../../1-lora-correction/report/is-the-adapter-just-copying-the-joint-prompt/does-the-adapter-return-two-when-both-prompts-name-the-same-concept.md).
- The mention of **PoE**, as the outputs the scorer was trained on, in [the failure catalog entry poe-score-002](../../environment/known-failures.md#entry-id-poe-score-002).
- The mention of **denoising step**, where the second UNet forward of the first step runs out of memory without `@torch.no_grad()`, in [the failure catalog entry poe-lora-004](../../environment/known-failures.md#entry-id-poe-lora-004).
- The mention of **denoising step**, the unit every dependent variable is recorded per, in [the whole corrector design](../../1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/source/the-whole-corrector-design.md#the-claim).
- The mention of **product-of-experts composition** in [the plan "the cheapest test: does RoentGen-v2 draw two findings"](../../3-understanding-compositionality/plans/10-does-roentgen-draw-two-findings/plans/01-the-cheapest-test.md).
- The mention of **PoE** in the route "did SDXL see a wet camel, and does PoE only draw what it saw".
- The mention of **plain product**, as the column the adapter is compared against, in [the draft captions for the appendix render grids, Figure A's pointing sentence](../../1-lora-correction/report/paper-review/captions-showcase-grids.md#where-our-method-does-best-two-figures).
- The mention of **PoE** in [the causal-composition route](../../3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/route-causal-does-an-effect-need-support.md) and [the write-up it tests](../../3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/what-poe-loses-when-it-composes.md#31-product-of-experts-and-the-and-operator).
- The mention of **PoE** in [the caption walk on wet camels and wet dogs](../../3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/today-camel-and-dog-in-the-training-data.md), [the six-checkpoint toy walk](../../3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/today-the-toy-failure-in-six-checkpoints.md#why-this-exists) and [the third-concept walk](../../3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/today-the-third-concept.md).
- The mention of **product-of-experts** in [the finding "can a text-space third expert restore composition"](../../5-text-embedding-third-expert/report/can-a-text-space-third-expert-restore-composition.md#the-claim-and-where-it-came-from), which adds a third condition to the same two-expert sum.
- The mention of **plain PoE** as the comparison render in [the plan that puts the LoRA on the text encoders](../../3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/today-a-lora-on-the-text-encoders.md), which adapts the two text encoders rather than the denoiser.
- The mention of **product of experts**, the sampler a classifier's push is added to, in [the route for classifier-guided PoE](../../2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md#quick-context-where-you-are).

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
