# 🌍 Chimera

A chimera is what PoE composition produces when it fails: one creature with parts of both
requested concepts fused together, instead of two separate concepts in one scene. It is the
project's name for the failure that [interaction-term.md](interaction-term.md) tries to explain
and [lora-corrector.md](lora-corrector.md) tries to fix.

<a href="../images/world/01-poe-chimera-example.png"><img src="../images/world/01-poe-chimera-example.png" width="200" alt="One fused white creature with a cat's pointed ears and a dog's muzzle"></a>

**Why the name.** In Greek myth a chimera is one creature built from parts of a lion, a goat and a snake `stated` (the word's origin). Here it is one animal with a cat's pointed ears on a dog's face, as in the picture (plain PoE, a cat and a dog, seed 42) `verified` (opened and read by eye). Pull it apart into a cat beside a dog and the detector's count goes from 1 to 2, which flips the label from blend to compose `verified` (1 is the count on `catdog_poe_blend` in the data table below, and 2 is what the rule at [what a compose rate is](compose-rate.md#what-a-compose-rate-is) asks for).

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [What a chimera is](#what-a-chimera-is)
- [What it looks like](#what-it-looks-like)
- [Why the project cares](#why-the-project-cares)
- [How it shows up in the data](#how-it-shows-up-in-the-data)
- [What people get wrong](#what-people-get-wrong)
- [Where this came from](#where-this-came-from)
- [🖼️ In pictures, code and further reading](#in-pictures-code-and-further-reading)
- [Cross-references](#cross-references)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-a-chimera-is) ➡️

- **Blend**: this project's word for a chimera result when discussing the scorer's binary label
  (`compose` vs `blend`), see [compose-rate.md](compose-rate.md).
- **Single concept / noise**: the two other named failure modes alongside chimera, per
  `MASTER_PLAN.md`'s Mission line: "PoE usually fails (chimera / single concept / noise)".

## What a chimera is

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#what-it-looks-like) ➡️

**One blended creature carrying features of both requested concepts, instead of two distinct
concepts in the frame.** ✅

`MASTER_PLAN.md`'s Glossary, verbatim: "Chimera: that failure. One animal with parts of both,
instead of two animals." It is defined specifically against the animal-pair pool this project
currently studies (see [animal-pair.md](animal-pair.md)); the same underlying failure on
non-animal pairs is described more generally in `MASTER_PLAN.md`'s Mission as "single concept /
noise" alongside it. ✅

## What it looks like

Navigation: ⬅️ [What a chimera is](#what-a-chimera-is) | 📋 [TOC](#table-of-contents) | [Next](#why-the-project-cares) ➡️

The plain PoE render in [poe-composition.md](poe-composition.md#what-it-looks-like) (copied from
`data/pilot/seed_42/a_cat__x__a_dog/poe.png`) is this project's own chimera example: one fused
creature rather than a cat beside a dog.

## Why the project cares

Navigation: ⬅️ [What it looks like](#what-it-looks-like) | 📋 [TOC](#table-of-contents) | [Next](#how-it-shows-up-in-the-data) ➡️

**A chimera is the visible symptom; the interaction term is the measured cause the project
targets.** ✅

The paper's abstract states the causal claim directly: PoE "can fail catastrophically, producing a
single blended chimera instead of a scene containing both concepts, and we trace this failure to a
specific, correctable gap" (`writing/iclr/iclr2027_conference.tex`). Whether a given render is a
chimera or a real composition is exactly what the scorer in
[compose-rate.md](compose-rate.md) is built to call.

## How it shows up in the data

Navigation: ⬅️ [Why the project cares](#why-the-project-cares) | 📋 [TOC](#table-of-contents) | [Next](#what-people-get-wrong) ➡️

| Column | Stands for | Example | Entry |
|---|---|---|---|
| `n_instances` | how many distinct animal instances the scorer's detector found; 1 usually means a chimera or a dropped concept | `1` (example, from `catdog_poe_blend` in `scorer_validated.json`) | [Dictionary § n_instances](../data/02-dictionary.md#n_instances) |

## What people get wrong

Navigation: ⬅️ [How it shows up in the data](#how-it-shows-up-in-the-data) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**A chimera is not the only failure the instance-count scorer catches, and "2 instances" is not
proof of a real chimera-free success.** ✍️

`1-lora-correction/artifacts/results/measuring-composition/the-scorers/can-we-trust-the-compose-score/do-the-successful-cells-contain-both-animals/two-of-the-same-animal/` holds one confirmed case of two instances of the *same*
animal (two dogs, no cat) scored as a compose success, because the detector cannot tell which two
animals it found, only how many. See
[compose-rate.md § What people get wrong](compose-rate.md#what-people-get-wrong).

## Where this came from

Navigation: ⬅️ [What people get wrong](#what-people-get-wrong) | 📋 [TOC](#table-of-contents) | [Next](#in-pictures-code-and-further-reading) ➡️

| What | How it was established | When |
|---|---|---|
| The definition of a chimera | Read in `MASTER_PLAN.md`, Glossary | 2026-08-24 |
| The chimera example image | Read directly, `data/pilot/seed_42/a_cat__x__a_dog/poe.png` | 2026-08-24 |
| The causal claim linking chimera to the interaction term | Read in `writing/iclr/iclr2027_conference.tex`, abstract | 2026-08-24 |
| The "two of the same animal" scorer miss | Read in `1-lora-correction/artifacts/results/measuring-composition/the-scorers/can-we-trust-the-compose-score/do-the-successful-cells-contain-both-animals/README.md` | 2026-08-24 |

<!-- related:start (drain) -->
<a id="in-pictures-code-and-further-reading"></a>

## 🖼️ In pictures, code and further reading

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents) | [Next](#cross-references) ➡️

The same cat and dog, asked two ways, give two animals as one sentence and one fused animal as a sum of two prompts. The pictures below show that in three setups across two models, SDXL and PixArt.

### How it fits together

<a href="../../3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/poe-against-the-joint/what-the-fusion-is-not-caused-by/summary-grid.png"><img src="../../3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/poe-against-the-joint/what-the-fusion-is-not-caused-by/summary-grid.png" width="200" alt="Three setups, each with a joint prompt column and a summed prompt column"></a>

Three setups side by side: SDXL at its usual settings, SDXL at PixArt's settings, and PixArt. In each, the left column ("a cat and a dog" as one sentence) shows two animals and the right column (the sum of "a cat" and "a dog") shows one fused animal. The figure's own footer says 0 of 8 seeds give two clear animals in the summed column in each of the three setups `verified` (read from the picture's footer text). Its card says only "candidate causes ruled out" and that the runs behind each panel are not recorded beside it `stated`. The joint column can still draw the wrong animals: at SDXL's usual settings, seed 13 looks like two dogs `verified` (read by eye).

### The code behind it

The label `blend` on this page is produced by counting animals. `label` is `compose` when the detector finds two or more, otherwise `blend`. Excerpt from `poe_repair/experiments/compose_scorer_validation/detection_scorer.py` (lines 110 to 116):

```python
    return InstanceScore(
        label="compose" if n_animal >= 2 else "blend",
        n_instances=n_animal,
        n_instances_head=n_head,
        conf_a=_best_conf(dets, query_a),
        conf_b=_best_conf(dets, query_b),
    )
```

- `n_animal >= 2` is the whole rule: it counts animals and does not name them, which is why two dogs score as `compose`.
- `n_instances` is the column the table above describes; `n_instances_head` is a second count from the query "animal head", and the two confidences are diagnostic only, as the file's own comment says.

### What comes out

<a href="../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/does-a-clip-free-model-hybridise/pixart-cat-dog-sheet.png"><img src="../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/does-a-clip-free-model-hybridise/pixart-cat-dog-sheet.png" width="200" alt="Eight seeds by four columns of cat, dog, summed prompts and joint prompt"></a>

PixArt-alpha (no CLIP in it), eight seeds, columns cat alone, dog alone, the summed prompts, the joint prompt. Every summed tile shows one animal, and the joint column shows two `verified` (opened and read by eye). The card separates the summed tiles into six hybrids and two single cats (seeds 2 and 3), and counts seven joint tiles as a clear cat and dog, calls made by eye from the full-resolution tiles `stated` (card).

| Column | Two clear animals | One hybrid animal | One single animal |
|---|---|---|---|
| Summed "a cat" and "a dog" | 0 of 8 | 6 of 8 | 2 of 8 (both cats) |
| Joint "a cat and a dog" | 7 of 8 | 0 of 8 | 0 of 8 |

`stated` (the sheet's card; the eighth joint tile, seed 2, holds two cats and counts as neither).

### Further illustrations

**Evidence**

`how-a-render-fails-gallery.png` is on the cluster at `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/which-adapter-composes-cat-and-dog/how-a-render-fails-gallery.png`

Six panels of failed renders. The top-left panel is labelled "One fused animal", from plain product-of-experts composition with no correction on the cat and dog, seed 9 `verified` (the panel shows a cat's ear above a dog's muzzle, the enlarged patch holds the face only, and the panel's label gives the same setup). The other five panels are other failures, described in [how a render fails](how-a-render-fails.md).

<a href="../../data/pilot/seed_42/a_cat__x__a_dog/poe.png"><img src="../../data/pilot/seed_42/a_cat__x__a_dog/poe.png" width="200" alt="One white animal with pointed ears and a short muzzle"></a>

The project's own chimera example, plain PoE on the cat and dog at seed 42: one white animal with pointed ears and a short muzzle, in place of a cat beside a dog `verified` (opened and read by eye).

### Go deeper

| What | Link |
|---|---|
| How a render fails, the other looks | [how a render fails](how-a-render-fails.md) |
| The scorer that calls compose or blend | [compose rate](compose-rate.md) |
| The measured cause | [interaction term](interaction-term.md) |
| The plain PoE render this page uses | [PoE composition](poe-composition.md#what-it-looks-like) |
| The PixArt result card | [does a model with no CLIP still fuse a cat and a dog](../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/does-a-clip-free-model-hybridise/README.md) |
| The ruled-out causes card | [what the fusion is not caused by](../../3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/poe-against-the-joint/what-the-fusion-is-not-caused-by/README.md) |

### Still open

- This block shows no pair that is not two animals, and the page's definition covers animals only. `the-fix-is-not-about-animals.png` in the main-study grids folder holds plain product renders of non-animal pairs, and its tent and snowy mountain tile at seed 9 reads as one tent-shaped peak `verified`. Its card says no finding leans on that grid yet, so whether the tile counts as a chimera is not recorded `stated`.
<!-- related:end -->

## Cross-references

Navigation: ⬅️ [Previous](#in-pictures-code-and-further-reading) | 📋 [TOC](#table-of-contents)

- The mention of **a hybrid** in [the figure plan "the same rule on four models"](../../1-lora-correction/plans/03-does-the-correction-cause-composition/plans/figures/12-the-same-rule-on-four-models.md) (relevance match).
- The mention of **the same hybrid** in [the review of "the same rule on four models"](../../1-lora-correction/plans/03-does-the-correction-cause-composition/review/12-the-same-rule-on-four-models.md) (relevance match).
- The mention of **the fused animal** a species classifier turned into a cat in [the route for classifier-guided PoE](../../2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md#what-we-know-so-far) (relevance match).
- The mention of **the fused animal** the product draws, which a pair-aware classifier is trained to tell from two separate animals, in [the route for classifier-guided PoE](../../2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md#quick-context-where-you-are) (relevance match).
- The mention of **the fused cat-dog, this project's chimera** in [the plan "The product of experts as two classifiers"](../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/plans/03-the-product-as-two-classifiers.md#tasks).
- The mention of **the fused animal, this project's chimera** in [the plan "The separate-label push built"](../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/plans/04-the-separate-label-push-built.md#description).
- The mention of **the fused animal, this project's chimera** in [the plan "The best discriminator, and its log-odds slope"](../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/plans/05-the-best-discriminator.md#tasks).
- The mention of **the fused animal, this project's chimera** in [the plan "The dropped term, on four rows"](../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/plans/07-the-dropped-term-table.md#tasks).
- The mention of **the fused animal, this project's chimera** in [the plan "The pair-aware push built"](../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/plans/08-the-pair-aware-push-built.md#tasks).
- The mention of **the fused animal, this project's chimera** in [the plan "The account you could defend"](../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/plans/09-the-account-you-could-defend.md#tasks).
- The mention of **the fused animal, this project's chimera** in [the master plan of the classifier-guidance journey](../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/MASTER_PLAN.md#goals).
- The mention of **PoE's one fused animal**, which the readout checker's push turns into two bodies, in [plan 26, two pushes, bodies early and species late, quick context](../../2-classifier-guidance/plans/26-two-pushes-bodies-early-species-late.md#quick-context) (relevance match).

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
