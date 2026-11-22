# 🌍 What we produce

For a chosen pair of concepts, a seed, and a render method, the pipeline produces a generated
image and a small set of numbers describing how that image was made and whether it succeeded.
This file says what those outputs are, in the words a reader of the paper or the plan tree would
use, and links to the dictionary for the exact fields.

## Table of contents

- [The output, concretely](#the-output-concretely)
- [Who reads it](#who-reads-it)
- [What they do differently](#what-they-do-differently)
- [Where this came from](#where-this-came-from)
- [🖼️ In pictures, code and further reading](#in-pictures-code-and-further-reading)

## The output, concretely

Navigation: 📋 [TOC](#table-of-contents) | [Next](#who-reads-it) ➡️

**One generated image per (pair, seed, render method, step schedule), plus the per-step latent
trajectory it was decoded from.** ✅

A pilot run lives in `data/pilot/seed_42/a_cat__x__a_dog/`. Read this session directly. ✅

| File | What it is |
|---|---|
| `poe.png` | plain PoE |
| `monolithic.png` | Mono, the joint-prompt cheat |
| `solo_a.png`, `solo_b.png` | each concept alone, the two "experts" |
| `trajectory_manifold.png` | a 2-D projection of the denoising path |
| `summary.json` | the per-step and final size of the interaction term, `d_t_poe_vs_mono` and `d_T_poe_vs_mono` (see [data/02-dictionary.md § d_T](../data/02-dictionary.md#d_t--d_t)) |

**One compose/blend label and a compose rate, per pair, from the scorer.** ✅

`1-lora-correction/artifacts/results/measuring-composition/the-scorers/can-we-trust-the-compose-score/compose-scorer-validation/scorer_validated.json` and `1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-transfer/does-the-fix-reach-unseen-pairs/fail_rate.md`
carry the validated verdict: an image counts as "compose" when a GroundingDINO detector finds at
least two distinct "animal" instances after non-max suppression. See
[world/compose-rate.md](../world/compose-rate.md) for what that scorer can and cannot tell a
reader. ✅

> Non-max suppression merges detector boxes that cover the same animal, so one animal is not
> counted twice. The scorer's own page spells it out in
> [world/compose-rate.md](../world/compose-rate.md) `stated` (its NMS entry).

**One trained LoRA checkpoint per training configuration**, a set of weights on SDXL's
cross-attention layers, used at inference to reconstruct the interaction term without seeing the
joint prompt. See [world/lora-corrector.md](../world/lora-corrector.md). ✍️

## Who reads it

Navigation: ⬅️ [The output, concretely](#the-output-concretely) | 📋 [TOC](#table-of-contents) | [Next](#what-they-do-differently) ➡️

**The paper's evidence register.** ✅

The images, compose rates, and trajectory figures feed a set of named figures tracked in
`MASTER_PLAN.md`'s paper table (steps 13 and 14) and written up into the ICLR 2027 submission at
`writing/iclr/`. A figure's own reasoning lives beside the plan that owns it, never copied into this
folder (see [Cross-linking](../00-INDEX.md) in the format this folder follows).

**The plan tree's verdicts.** ✅

Every run is judged against a threshold written down before the run, per this project's `CLAUDE.md`
and `~/.claude/EXPERIMENT_CONVENTIONS.md`. W&B (project `prime_lab/poe-repair-animals-compose`)
holds the curves; the plan tree holds only the verdict, the run id, and the threshold it was judged
against.

## What they do differently

Navigation: ⬅️ [Who reads it](#who-reads-it) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**A verdict on transfer reach changes what the paper claims to deliver.** ✍️

Per `MASTER_PLAN.md`'s Expected Outcome: if a single LoRA composes across the whole studied set of
pairs and seeds, the paper claims one deployable corrector. If it does not, the paper claims a
catalogue: one trained corrector per group of similar pairs, each backed by its own held-out-seed
and held-out-pair evidence. Both are documented outcomes; neither is a failure of the project, per
`MASTER_PLAN.md`'s Definition of Done.

> Held-out means the seeds or pairs were kept aside and never used during training, so a result on
> them says how the corrector does on cases it has not seen.

## Where this came from

Navigation: ⬅️ [What they do differently](#what-they-do-differently) | 📋 [TOC](#table-of-contents) | [Next](#in-pictures-code-and-further-reading) ➡️

| What | How it was established | When |
|---|---|---|
| A pilot run's five images and its summary.json fields | Read directly, `data/pilot/seed_42/a_cat__x__a_dog/` | 2026-08-24 |
| The compose/blend rule and its scorer | Read in `1-lora-correction/artifacts/results/measuring-composition/the-scorers/can-we-trust-the-compose-score/compose-scorer-validation/scorer_validated.json` | 2026-08-24 |
| W&B owns the numbers, the plan tree owns the verdict | Read in project `CLAUDE.md` | 2026-08-24 |
| The two possible shapes of the deliverable (single LoRA vs catalogue) | Read in `MASTER_PLAN.md`, Expected Outcome | 2026-08-24 |

<!-- related:start (drain) -->
<a id="in-pictures-code-and-further-reading"></a>

## 🖼️ In pictures, code and further reading

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

One pilot run, the cat and dog pair at seed 42, is the smallest complete example of what the pipeline writes out. The pictures below are that run's own files.

### What comes out

<a href="../../data/pilot/seed_42/a_cat__x__a_dog/solo_a.png"><img src="../../data/pilot/seed_42/a_cat__x__a_dog/solo_a.png" width="200" alt="A white cat facing the camera, grey tones"></a>
<a href="../../data/pilot/seed_42/a_cat__x__a_dog/solo_b.png"><img src="../../data/pilot/seed_42/a_cat__x__a_dog/solo_b.png" width="200" alt="A white dog facing the camera, grey tones"></a>
<a href="../../data/pilot/seed_42/a_cat__x__a_dog/monolithic.png"><img src="../../data/pilot/seed_42/a_cat__x__a_dog/monolithic.png" width="200" alt="A white cat beside a white dog on a sofa"></a>
<a href="../../data/pilot/seed_42/a_cat__x__a_dog/poe.png"><img src="../../data/pilot/seed_42/a_cat__x__a_dog/poe.png" width="200" alt="One white furry face with cat ears and a dog muzzle"></a>
<a href="../../data/pilot/seed_42/a_cat__x__a_dog/trajectory_manifold.png"><img src="../../data/pilot/seed_42/a_cat__x__a_dog/trajectory_manifold.png" width="200" alt="Four curves from one start point to four end stars"></a>

Left to right: `solo_a.png` (a cat), `solo_b.png` (a dog), `monolithic.png` (a cat beside a dog), `poe.png` (one fused animal with a cat's ears and a dog-like muzzle) and `trajectory_manifold.png` `verified` (all five opened and read). The trajectory plot is titled "group6_coherent_collision / a_cat__x__a_dog / seed 42", axes PC1/MDS1 across and PC2/MDS2 up, four curves (Monolithic, Poe, Solo A, Solo B) leaving one shared start point and ending at four different stars `verified`. The chimera page uses `poe.png` as the project's own chimera example `stated` ([chimera](../world/chimera.md#what-it-looks-like)).

The numbers in the same folder, from `summary.json`:

| Field | Value in this run |
|---|---|
| `pair` | `a cat`, `a dog` |
| `seed` | `42` |
| `model_id` | `stabilityai/stable-diffusion-xl-base-1.0` |
| `d_T_poe_vs_mono` | `0.24645473062992096` |
| `d_t_poe_vs_mono`, values at index 0, 10, 25 and 50 (the last) | `0.0`, `0.004811020102351904`, `0.04612419754266739`, `0.24645473062992096` |

`verified` (read from the file). The `d_t_poe_vs_mono` array holds 51 values here, and the last equals `d_T_poe_vs_mono`.

### The code behind it

The compose or blend label is one comparison on the detector's animal count. Excerpt from `poe_repair/experiments/compose_scorer_validation/detection_scorer.py` (lines 110 to 116):

```python
    return InstanceScore(
        label="compose" if n_animal >= 2 else "blend",
        n_instances=n_animal,
        n_instances_head=n_head,
        conf_a=_best_conf(dets, query_a),
        conf_b=_best_conf(dets, query_b),
    )
```

- `n_animal >= 2` is the whole rule for `compose`; the file's own comment says the two confidences are diagnostic and not used for the label.

### Further illustrations

**Evidence**

*Picture held back.* `1-lora-correction/artifacts/results/measuring-composition/the-scorers/can-we-trust-the-compose-score/do-the-successful-cells-contain-both-animals/contact-sheet.png` holds a frog and toad row, which the [pair ledger](../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md) keeps off every sheet, so it is named here and not shown.

What the scorer sees across a set: 32 renders titled with the detector's count, 30 reading 2 or more and 2 reading `n=1` `verified` (counted on the picture; matches the card).

### Go deeper

| What | Link |
|---|---|
| What each field means | [the dictionary](../data/02-dictionary.md) |
| How a run becomes these files | [where the data comes from](../data/01-where-it-comes-from.md) |
| What a compose rate is | [compose rate](../world/compose-rate.md) |
| The trained corrector | [LoRA corrector](../world/lora-corrector.md) |
| The audit card behind the contact sheet | do the successful cells contain both animals, at `1-lora-correction/artifacts/results/measuring-composition/the-scorers/can-we-trust-the-compose-score/do-the-successful-cells-contain-both-animals/README.md` (not linked, because the folder holds the held-back pair) |

### Still open

- The page names `scorer_validated.json` and `fail_rate.md`. Both exist only in the cluster checkout (checked 2026-11-21), because their `results/` folders are git-ignored. No sample of either is quoted here.
- No listing of a trained LoRA checkpoint is placed here; this drain read none.
- The pilot folder also holds `decoded_images.png`, `grid_assets.json` and a `grid_assets/` folder that the page's file list does not name; they are not placed here.
<!-- related:end -->

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
