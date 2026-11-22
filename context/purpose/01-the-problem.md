# 🌍 The problem this project solves

Text-to-image diffusion models like Stable Diffusion XL (SDXL) are usually asked for one prompt
at a time. One cheap way to make a pretrained model draw two named concepts at once, without
retraining it, is to run two "expert" predictions side by side, one per concept, and combine them
at every denoising step. This is Product-of-Experts (PoE) composition, and it frequently fails: it
paints a blend of the two concepts (a chimera) or drops one of them, rather than a scene showing
both. This project asks why, and tests one specific fix. What a chimera and PoE composition
actually are, with an example image, is in [world/poe-composition.md](../world/poe-composition.md)
and [world/chimera.md](../world/chimera.md); this file is only the why.

## Table of contents

- [Who has the problem](#who-has-the-problem)
- [What it costs them today](#what-it-costs-them-today)
- [What this project does about it](#what-this-project-does-about-it)
- [Where this came from](#where-this-came-from)
- [🖼️ In pictures, code and further reading](#in-pictures-code-and-further-reading)
- [Cross-references](#cross-references)

## Who has the problem

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-it-costs-them-today) ➡️

**Anyone who wants a pretrained diffusion model to render two named concepts in one image without
retraining the whole model or hand-writing a joint prompt for every possible pair of concepts.** ✍️

The alternative to PoE composition is to type the joint prompt directly ("a cat and a dog") and
let the model's own training handle it. That works (this project calls it Mono, see
[world/poe-composition.md § What PoE composition is](../world/poe-composition.md#what-poe-composition-is)), but it
defeats the purpose of a compositional method: at scale, the whole point is to combine pretrained
single-concept experts instead of retraining or hand-prompting for every combination.

## What it costs them today

Navigation: ⬅️ [Who has the problem](#who-has-the-problem) | 📋 [TOC](#table-of-contents) | [Next](#what-this-project-does-about-it) ➡️

**PoE composition on SDXL blends or drops concepts far more often than it renders both.** ✅

Measured directly: `1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-transfer/does-the-fix-reach-unseen-pairs/fail_rate.md`, 8 seeds per pair under the
validated instance-count scorer (see
[world/compose-rate.md](../world/compose-rate.md#what-a-compose-rate-is)). 15 of 17 pairs in that
table fail on 8 of 8 seeds (fail-rate 1.00), including the reference pair `a_cat__x__a_dog`. Two
pairs fail less often: `a_donkey__x__a_pony` at 6 of 8 and `a_crocodile__x__an_alligator` at 5 of
8. A dissimilar control pair, `an_elephant__x__a_penguin`, also fails 8 of 8, so "the two concepts
are visually different" does not predict success on its own; `1-lora-correction/report/experiments-log.md`'s EXP-03 registers
this as a pre-registered null rather than the hypothesis. ✍️

> Registering it as a null means the expectation written down before the run was no effect: how
> different the two animals look does not change how often the pair fails.

**When PoE does render both concepts, the render still is not trustworthy on its own.** ✍️

The card at `1-lora-correction/artifacts/results/measuring-composition/the-scorers/can-we-trust-the-compose-score/do-the-successful-cells-contain-both-animals/README.md` opened all 32 of the strongest-correction runs behind a 94% headline compose rate by eye. What it found:

| Finding | Count or value |
|---|---|
| Runs opened by eye | 32 |
| Real scorer error (two of the same animal, called a success) | 1 of 32 |
| More that could not be called confidently either way | 5 more, most because the pool deliberately contains look-alike pairs (leopard vs jaguar, cow vs buffalo) |
| The card's conclusion for the true rate | somewhere around 87% to 94% |

A printed rate should say so rather than stand alone. This is why
[world/compose-rate.md § What people get wrong](../world/compose-rate.md#what-people-get-wrong)
exists.

## What this project does about it

Navigation: ⬅️ [What it costs them today](#what-it-costs-them-today) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**It traces the PoE failure to one specific, measurable gap and tests whether closing that gap
fixes composition.** ✅

The gap is the interaction term: the per-step difference between the prediction the model would
make if it saw the literal joint prompt (Mono) and the prediction PoE actually makes by combining
two separate experts. The paper's own framing (`writing/iclr/iclr2027_conference.tex`, abstract):
"[PoE] can fail catastrophically, producing a single blended chimera instead of a scene containing
both concepts, and we trace this failure to a specific, correctable gap. This gap is the
interaction term... the signal omitted by linear score composition." What the interaction term is,
concretely, is in [world/interaction-term.md](../world/interaction-term.md). ✅

**The fix under test is a small trained corrector, not the joint prompt itself.** ✅

A rank-8 low-rank adapter (LoRA) on SDXL's cross-attention layers is trained to predict the
interaction term at every denoising step and add its prediction back into the PoE run, without the
LoRA ever seeing the joint prompt at inference (Mono-free). What a LoRA does here is in
[world/lora-corrector.md](../world/lora-corrector.md).

> Low-rank means the extra weights are stored as two thin matrices multiplied together instead of
> one full-sized one, so the correction has far fewer trainable numbers than the layer it sits on.
> Rank-8 is how thin: eight columns wide.

> Mono-free means the joint prompt is never given to the model at inference. It is used only while
> training, to work out what the correction should have been.

This already closes the gap on one memorised pair, per `MASTER_PLAN.md` Objective 1. The open
research question is how far the fix reaches: one trained pair, a pooled set of seeds, a group of
similar pairs, or a single corrector spanning the whole studied set of pairs. That reach is what
[purpose/03-what-working-means.md](03-what-working-means.md) defines as the test.

## Where this came from

Navigation: ⬅️ [What this project does about it](#what-this-project-does-about-it) | 📋 [TOC](#table-of-contents) | [Next](#in-pictures-code-and-further-reading) ➡️

| What | How it was established | When |
|---|---|---|
| PoE composition fails on 15 of 17 pool pairs at fail-rate 1.00 | Read in `1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-transfer/does-the-fix-reach-unseen-pairs/fail_rate.md` | 2026-08-24 |
| The headline compose rate needs a stated bound, not a bare number | Read in `1-lora-correction/artifacts/results/measuring-composition/the-scorers/can-we-trust-the-compose-score/do-the-successful-cells-contain-both-animals/README.md` | 2026-08-24 |
| The interaction term is the gap the fix targets | Read in `writing/iclr/iclr2027_conference.tex`, abstract | 2026-08-24 |
| The reach question (one pair vs group vs whole taxonomy) is the open question | Read in `MASTER_PLAN.md`, Mission and Objectives | 2026-08-24 |

<!-- related:start (drain) -->
<a id="in-pictures-code-and-further-reading"></a>

## 🖼️ In pictures, code and further reading

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents) | [Next](#cross-references) ➡️

The problem in one picture is a cat and a dog asked for two ways: one sentence draws two animals, a sum of two prompts draws one. The pictures and code below show the failure, the sum that causes it, and the dial that the fix turns.

### How it fits together

<a href="../../3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/poe-against-the-joint/what-the-fusion-is-not-caused-by/summary-grid.png"><img src="../../3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/poe-against-the-joint/what-the-fusion-is-not-caused-by/summary-grid.png" width="200" alt="Three setups, each with a joint prompt column and a summed prompt column"></a>

Three setups, SDXL at its usual settings, SDXL at PixArt's settings and PixArt, each with the joint prompt "a cat and a dog" on the left and the sum of "a cat" and "a dog" on the right. The figure's footer reads "Two clear animals in the summed column: 0 of 8 seeds in each of the three setups" `verified` (read from the picture). Which runs sit behind each panel is not recorded beside the figure `stated` (its card).

### The code behind it

The sum is one line of arithmetic. Excerpt from `poe_repair/methods/_sampling.py` (lines 168 to 171), inside the loop that denoises one step at a time:

```python
        eps_a_raw, eps_b_raw, eps_uncond = noise.chunk(3)
        eps_a = guided_eps(eps_a_raw, eps_uncond, guidance_scale)
        eps_b = guided_eps(eps_b_raw, eps_uncond, guidance_scale)
        eps_p = poe_eps(eps_a, eps_b, eps_uncond)
```

- One model call returns three predictions: for concept A, for concept B and for the empty prompt (`eps_uncond`).
- `poe_eps` in `poe_repair/_sdxl/metrics.py` returns `eps_a + eps_b - eps_uncond`, and the docstring of the enclosing sampler `run_cfg_poe` in `_sampling.py` writes the same sum `verified` (both files read).

### What comes out

*Picture held back.* `1-lora-correction/artifacts/results/measuring-composition/the-scorers/can-we-trust-the-compose-score/do-the-successful-cells-contain-both-animals/contact-sheet.png` holds a frog and toad row, which the [pair ledger](../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md) keeps off every sheet, so it is named here and not shown.

All 32 strongest-correction renders behind the 94% headline, each titled with its pair, seed and the detector's animal count. Counting the titles, 2 read `n=1` and 30 read 2 or more, which is the 30 of 32 the card calls the 94% `verified` (counted on the picture; matches the card). The card sorts them by eye into 12 with both animals clearly there, 1 with two of the same animal, 5 it cannot decide, 12 renders of three look-alike pairs (cow and buffalo, frog and toad, leopard and jaguar) and 2 scored as a failure `stated` (card and `calls.json`). The card says both failures were scored correctly `stated`, while the seal and walrus image at seed 9, counted `n=1`, shows two separate animals `verified` (opened and read).

### Further illustrations

**Evidence**

*Picture held back.* `1-lora-correction/artifacts/results/measuring-composition/the-grids/how-big-the-strength-grid-should-be/strength-grid-six-columns.png` has a tile with a shape that could not be identified at full size, so it is named here and not shown.
*Picture held back.* `1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-transfer/does-the-fix-reach-unseen-pairs/hard-vs-easy-transfer/hard_vs_easy_transfer.png` holds a frog and toad row, which the [pair ledger](../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md) keeps off every sheet, so it is named here and not shown.

The fix under test, left: cat and dog, seed 9, six strengths of the correction (λ = 0, 0.5, 0.6, 0.65, 0.75, 1) across the top. In the top row (the real correction) λ=0 is one fused face and λ=1 is a cat beside a dog `verified` (opened and read). The three lower rows are controls (wrong pair, wrong seed, wrong order) and stay one fused animal in every column `verified` (opened and read). The card reads the grid the same way: at λ 0.6 and 0.65 the real-correction row already shows a cat and a dog, while the other three rows stay single fused animals `stated` (card).

Reach, right: six pairs, columns Mono (the joint-prompt target), PoE and LoRA, one seed per pair. In the PoE column every row is one animal `verified`; in the LoRA column every row shows two animals `verified` (opened and read). The card adds that for seal and walrus the second animal has no tusks, so the pair reads as two seals, and that each row is a single seed and not an aggregate `stated` (card). Only the frog and toad, eagle and hawk, seal and walrus, and leopard and jaguar rows were held out. Wolf and husky and lion and tiger are training pairs with no held-out score `stated` (card).

### Go deeper

| What | Link |
|---|---|
| What a chimera is | [chimera](../world/chimera.md) |
| What PoE composition is | [PoE composition](../world/poe-composition.md) |
| The scorer and its limits | [compose rate](../world/compose-rate.md) |
| The measured gap | [interaction term](../world/interaction-term.md) |
| The 32-run audit card | do the successful cells contain both animals, at `1-lora-correction/artifacts/results/measuring-composition/the-scorers/can-we-trust-the-compose-score/do-the-successful-cells-contain-both-animals/README.md` (not linked, because the folder holds the held-back pair) |
| The strength grid card | [how big the strength grid should be](../../1-lora-correction/artifacts/results/measuring-composition/the-grids/how-big-the-strength-grid-should-be/README.md) |
| The test this page leads to | [what working means](03-what-working-means.md) |

### Still open

- The page cites `fail_rate.md` for the 15 of 17 failing pairs. That file exists only in the cluster checkout (checked 2026-11-21), and its 17 rows agree with the page: 15 at 8 of 8, `a_donkey__x__a_pony` at 6 of 8 and `a_crocodile__x__an_alligator` at 5 of 8. No picture or table of the 17 rows is placed here.
- The audit card says "One definite error in 30" (the 30 scored compose), while the page above says "1 of 32". Both are left as written.
<!-- related:end -->

## Cross-references

Navigation: ⬅️ [Previous](#in-pictures-code-and-further-reading) | 📋 [TOC](#table-of-contents)

- The mention of **what this journey composes later** (the log-space sum-of-scores identity) in [the scene map for the Gaussian score's base case](../../../../goal-setting/learning/deep-learning/diffusion-models/05-poe-composition-diffusion/artifacts/drips/gaussian-score-base-case/SCENE_MAP.md) (relevance match).

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
