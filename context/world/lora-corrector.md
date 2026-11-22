# 🌍 The LoRA corrector

A LoRA (low-rank adapter) here is a small set of extra trained weights bolted onto SDXL's
cross-attention layers, trained to predict the interaction term (see
[interaction-term.md](interaction-term.md)) at every denoising step and add it back into the PoE
run, without ever seeing the literal joint prompt at inference. It is the project's one shippable
fix, as distinct from the interaction term itself, which is only ever computed from the joint
prompt during training and is not something a deployed system has access to.

> Low-rank means the extra weights are kept as two thin matrices multiplied together rather than
> one full-sized matrix, so the correction carries far fewer trainable numbers than the layer it
> attaches to.

Where to learn the two borrowed words: the [LoRA card](../../../../Obsidian/Artificial%20Intelligence/Large%20Language%20Models/LoRA.md#where-to-learn-this) and the [cross-attention card](../../../../Obsidian/Artificial%20Intelligence/Deep%20Learning/Cross-Attention.md#where-to-learn-this). Rank is the width of the two thin matrices. On a toy layer of 100 by 100 weights (invented; this page does not state SDXL's layer sizes):

| What is trained | Shape | Numbers |
|---|---|---|
| the full layer | 100 by 100 | 10,000 |
| rank 8 | 100 by 8, times 8 by 100 | 1,600 (16%) |
| rank 4 | 100 by 4, times 4 by 100 | 800 (8%) |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [What the LoRA corrector is](#what-the-lora-corrector-is)
- [What it looks like](#what-it-looks-like)
- [Why the project cares](#why-the-project-cares)
- [How it shows up in the data](#how-it-shows-up-in-the-data)
- [What people get wrong](#what-people-get-wrong)
- [Where this came from](#where-this-came-from)
- [🖼️ In pictures, code and further reading](#in-pictures-code-and-further-reading)
- [Cross-references](#cross-references)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-the-lora-corrector-is) ➡️

- **Rank-8**: the size of the low-rank adapter; a smaller number of trained parameters than
  retraining the attention weights outright.
- **Cross-attention (`attn2`)**: the layer inside SDXL's UNet where the text prompt's embedding
  enters the image-generation computation; this is where the LoRA is attached.
- **Mono-free**: at inference the LoRA never receives or encodes the joint prompt. Only its own
  trained weights and the PoE run's own state are used. This is the property that matters most
  about the fix.
- **lora**: written out in full in prose, per the word list in
  `plans/standing/retrofit-poe-repair-min.md`. The word "adapter" appears only inside code
  identifiers and in older documents quoted verbatim.

## What the LoRA corrector is

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#what-it-looks-like) ➡️

**A rank-8 LoRA on SDXL's cross-attention layers, trained on the cached interaction term, added at
inference without seeing the joint prompt.** ✅

Quoted from `MASTER_PLAN.md`'s Mission: "we train a rank-8 cross-attention LoRA on the cached
guided residual r_t = ε̃_J − ε̃_PoE so that, at inference and without ever encoding the joint
prompt (Mono-free), the corrected PoE prediction moves toward the Mono ceiling." ✅

**It is trained per configuration, not once for the whole project.** ✍️

Different training runs pool over different sets of pairs and seeds, matching the project's
five-step ladder (see [purpose/03-what-working-means.md](../purpose/03-what-working-means.md#the-test)):
one pair and one seed, one pair pooled over several seeds, one group of pairs pooled together, or
the whole studied set at once. Which pooling unit is deployable is the project's open research
question, not a settled fact this file can assert.

## What it looks like

Navigation: ⬅️ [What the LoRA corrector is](#what-the-lora-corrector-is) | 📋 [TOC](#table-of-contents) | [Next](#why-the-project-cares) ➡️

A LoRA is a set of trained numeric weights, not a directly picturable object. What it produces
(a corrected render, ideally closer to the Mono image than plain PoE) is picturable, and an
example of that comparison is owed:

> 📷 **Picture wanted**: one run's PoE render, LoRA-corrected render, and Mono render side by
> side, so a reader can see what the correction visibly moved. Not yet selected; a strong
> candidate is any run behind `1-lora-correction/artifacts/results/measuring-composition/the-scorers/can-we-trust-the-compose-score/compose-scorer-validation/scorer_validated.json`'s
> `passing_spaces: instance_count` result. Save as
> `images/world/05-lora-corrected-vs-poe-vs-mono.png`.

## Why the project cares

Navigation: ⬅️ [What it looks like](#what-it-looks-like) | 📋 [TOC](#table-of-contents) | [Next](#how-it-shows-up-in-the-data) ➡️

**It is the one artefact the project can ship.** ✅

The interaction term itself needs the joint prompt to compute, which defeats the purpose of PoE
composition. `MASTER_PLAN.md`'s Expected Outcome names the LoRA (or a catalogue of LoRAs, one per
group) as "A deployable, Mono-free PoE corrector whose reach is characterised", which is the
project's actual deliverable.

## How it shows up in the data

Navigation: ⬅️ [Why the project cares](#why-the-project-cares) | 📋 [TOC](#table-of-contents) | [Next](#what-people-get-wrong) ➡️

| Column | Stands for | Example | Entry |
|---|---|---|---|
| `arm` | one of the values this column takes is the LoRA-corrected render | `lora_residual_inject` (example, a method name in `poe_repair/methods/_sampling.py`) | [Dictionary § arm](../data/02-dictionary.md#arm) |
| `pair_slug` | which pair a given LoRA checkpoint was trained on or evaluated against | `a_cat__x__a_dog` (example) | [Dictionary § pair_slug](../data/02-dictionary.md#pair_slug) |
| `group` | the pooling unit a LoRA checkpoint was trained over, when pooled by group | `group6_coherent_collision` (example) | [Dictionary § group](../data/02-dictionary.md#group--group_label) |

## What people get wrong

Navigation: ⬅️ [How it shows up in the data](#how-it-shows-up-in-the-data) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**A LoRA that composes on its training pair is not evidence it generalises.** ✍️

`context/research-guidelines.md` records this as a live worry, not a resolved one: whether a
single-pair-trained LoRA's apparent transfer to a sibling pair is a real hit or "a memorised
correction that happens to fit" is exactly why the project's own step 3 (`does-the-fix-reach-unseen-pairs`)
was downgraded from something the paper must clear to an optional short check, and step 4
(group-wise pooling on concept-disjoint pairs) was named the reviewer-credible version instead
(`plans/standing/retrofit-poe-repair-min.md`'s reference to `1-lora-correction/report/experiments-log.md` EXP-03).

## Where this came from

Navigation: ⬅️ [What people get wrong](#what-people-get-wrong) | 📋 [TOC](#table-of-contents) | [Next](#in-pictures-code-and-further-reading) ➡️

| What | How it was established | When |
|---|---|---|
| The LoRA's rank, attachment point and Mono-free property | Read in `MASTER_PLAN.md`, Mission | 2026-08-24 |
| The deployable-artefact framing | Read in `MASTER_PLAN.md`, Expected Outcome | 2026-08-24 |
| The single-pair-transfer confound and its downgrade | Read in `context/research-guidelines.md` and `1-lora-correction/report/experiments-log.md` | 2026-08-24 |
| `arm` value `lora_residual_inject` | Read in `README.md`'s repo layout, `poe_repair/methods/_sampling.py` listing | 2026-08-24 |

<!-- related:start (drain) -->
<a id="in-pictures-code-and-further-reading"></a>

## 🖼️ In pictures, code and further reading

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents) | [Next](#cross-references) ➡️

The page's "Picture wanted" asks for one run's PoE render, corrected render and Mono render side by side. The pictures below hold that comparison for six pairs, and the code shows where the correction is added.

### How it fits together

*Picture held back.* `1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-transfer/does-the-fix-reach-unseen-pairs/hard-vs-easy-transfer/hard_vs_easy_transfer.png` holds a frog and toad row, which the [pair ledger](../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md) keeps off every sheet, so it is named here and not shown.

One row per pair, three columns: Mono (the joint prompt, the target), PoE and LoRA. In each of the six rows the Mono tile and the LoRA tile show two animals and the PoE tile shows one animal in the foreground `verified` (opened and read by eye). The card says these are the existing triptychs from the pooled rank-8 adapter's first training phase at step 60,000, one seed per pair, no new render `stated` (the folder's query file). The card adds that in the seal and walrus row the second LoRA animal lacks tusks, so the row reads as two seals `stated`; the picture shows no tusks on either LoRA animal `verified`.

| Row | Set | Compose rate the card gives |
|---|---|---|
| Frog and toad | held out | 0.9375 |
| Eagle and hawk | held out | 0.9375 |
| Seal and walrus | held out | 0.9375 |
| Leopard and jaguar | held out | 1.0 |
| Wolf and husky | training pair | none, never held out |
| Lion and tiger | training pair | none, never held out |

`stated` (the folder's query file).

### The code behind it

The correction is the difference between the product's prediction with the adapter on and with it off. Excerpt from `poe_repair/methods/_sampling.py` (lines 858 to 869):

```python
            eps_a_l = guided_eps(_eps_a_raw_l, _eps_u_l, guidance_scale)
            eps_b_l = guided_eps(_eps_b_raw_l, _eps_u_l, guidance_scale)
            eps_poe_lora = poe_eps(eps_a_l, eps_b_l, _eps_u_l)
            delta_hat = eps_poe_lora - eps_poe_frozen
            delta_capped, applied = _maybe_cap_correction(
                delta_hat, eps_poe_frozen, correction_max_rel_norm,
            )
            delta_norm_per_step.append(float(delta_hat.float().norm().item()))
            cap_scale_per_step.append(float(applied))
            _lam_t = _lambda_value(lambda_schedule, step_index, num_inference_steps,
                                   float(lambda_value))
            eps_t = eps_poe_frozen + _lam_t * delta_capped
```

- Lines 858 to 860 combine the adapter-on predictions into a second product (`eps_poe_lora`), and `eps_poe_frozen` is the same product with the adapter off, computed earlier at line 831.
- `delta_hat` is the learned stand-in for the interaction term, and `_maybe_cap_correction` limits its size relative to the product's prediction.
- The last line is the same shape as the true-term line in [interaction term](interaction-term.md#the-code-behind-it): the product plus λ times a correction, with the learned one in place of the joint minus product.

The joint prompt is not read at inference. Lines 683 to 686 of the same file:

```python
    The joint prompt ``(seq_j, pool_j)`` is no longer consumed at
    inference — the LoRA is the only thing carrying the correction. It is
    kept in the signature for backwards-compat with the probe wiring;
    pass anything (e.g. the unconditional embedding) without effect.
```

`verified` (read in the code). This is the Mono-free property the page names.

### Further illustrations

**Evidence**

*Picture held back.* `1-lora-correction/artifacts/results/measuring-composition/the-grids/what-the-main-study-grids-show/the-same-concept-twice.png` has a tile with a shape that could not be identified at full size, so it is named here and not shown.

Four rows, each naming one concept twice (dog and dog at seeds 2 and 9, cat and cat at seed 9, tiger and tiger at seed 9), and four columns: the joint prompt, the joint prompt through the adapter, the plain product and the adapter on the product ("Ours"). The plain product column shows one animal in every row, and the adapter column shows two or more animals in every row (three dogs at dog seed 9) `verified` (opened and counted by eye). The card says the plain product returns one animal in all five cells and the adapter two in four of five `stated`.

**The picture and the card differ in two places.** The card lists five cells under three columns. The picture has four rows, with dog seed 1 missing, and a fourth column for the joint prompt through the adapter. Both are given and neither is changed. The picture's footer, "Adapter trained for 1,000 epochs", agrees with the card's `v58-03-self` at step 50,000, because an epoch is 50 steps `stated` (`scripts/shared/main_study_grids.py`, line 25).

### Go deeper

| What | Link |
|---|---|
| The gap the adapter imitates | [interaction term](interaction-term.md) |
| The method it corrects | [PoE composition](poe-composition.md) |
| The failure it repairs | [chimera](chimera.md) |
| How its failures look | [how a render fails](how-a-render-fails.md) |
| The same-concept finding | [does the adapter return two when both prompts name the same concept](../../1-lora-correction/report/is-the-adapter-just-copying-the-joint-prompt/does-the-adapter-return-two-when-both-prompts-name-the-same-concept.md) |
| The six-pair figure's query and method | [hard against easy transfer](../../1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-transfer/does-the-fix-reach-unseen-pairs/hard-vs-easy-transfer/QUERY.md) |
| The grids card | [what the main study's grids show](../../1-lora-correction/artifacts/results/measuring-composition/the-grids/what-the-main-study-grids-show/README.md) |

### Still open

- The page's own slot, `images/world/05-lora-corrected-vs-poe-vs-mono.png`, is not filled; the six-pair figure above is a candidate and the page itself is unchanged.
- The page's candidate source, `scorer_validated.json`, exists only in the cluster checkout (checked 2026-11-21).
- The figure has one seed per pair, so no picture here shows how typical the shown seed is, as the card also says.
<!-- related:end -->

## Cross-references

Navigation: ⬅️ [Previous](#in-pictures-code-and-further-reading) | 📋 [TOC](#table-of-contents)

- The mention of **the softness of the corrected render** (what the adapter costs in fidelity, and two attempts to remove it) in [can a corrector or a clean tail sharpen the adapter's renders](../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/can-a-corrector-or-a-clean-tail-sharpen-the-adapters-renders.md) (relevance match).
- The mention of **rank-8** (the adapter the paper argues can carry the correction) in [the whole corrector design](../../1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/source/the-whole-corrector-design.md#why-this-plan-exists).
- The mention of **adapter** in [the write-up on what PoE loses when it composes](../../3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/what-poe-loses-when-it-composes.md#35-the-interaction-term).
- The mention of **pooled LoRA**, the rank-32 checkpoint the third-expert runs are given the recipe of, in [the plan that fits a learned "and" token](../../3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/today-a-learned-and-token.md).
- The mention of **the trained LoRA**, the correction a classifier's gradient would stand in for, in [the route for classifier-guided PoE](../../2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md#quick-context-where-you-are) (relevance match).

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
