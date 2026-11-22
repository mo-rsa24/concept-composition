# Can a classifier push fix the product?

The exact toy's product draws a red circle and a blue square side by side, and the sampler that adds the two experts' scores draws them stacked. This folder holds the classifiers trained to push that sampler toward side by side, and what they say.

## The answer

**Yes, when the classifier is trained on what the sampler gets wrong.** At pair weight 0.01, seed 7, 300 draws, unguided reverse diffusion draws 0 crisp pairs and stacks all 300. A discriminator trained on exact side-by-side draws against the sampler's own stacked draws gives 172 crisp pairs at push weight 1 and 300 at weight 2. A classifier trained only on "circle present" and "square present" gives at most 10 at any scale from 1 to 10. Both halves of the bar, written before any classifier was trained, pass. `verified`

| what pushes the sampler | crisp pairs of 300 | overlaid | file |
|---|---|---|---|
| nothing | 0 | 300 | `port.json` |
| the true correction, scale 0.5 | 112 | 188 | `port.json` |
| the true correction, scale 1 | 300 | 0 | `port.json` |
| separate-label classifier, scales 1, 2, 5, 10 | 0, 0, 2, 10 | 298, 297, 295, 290 | `separate-label-guide.json` |
| closed-form best separate-label classifier, same scales | 0, 1, 13, 34 | 294, 298, 287, 266 | `separate-label-guide.json` |
| pair-aware discriminator, w = 0.5, 1, 2 | 16, **172**, 300 | 284, **128**, 0 | `pair-aware-guide.json` |

**The bar.** Pair-aware at w = 1: at least 150 crisp and no more overlaid than unguided. 172 and 128 against 300: pass. Separate-label (trained) at most 15 at every scale: highest 10: pass. The closed-form best separate-label classifier would miss at scale 10 (34), because the side-by-side picture is the one training picture carrying both labels, so sharpening both labels pulls toward it. The bar was read on the trained classifier by the reader's choice, and the 34 is kept here so the pass is not read as more than it is.

**[The push against the true correction, one draw](draw-0-push-against-true-correction-step-150.png)**

<a href="draw-0-push-against-true-correction-step-150.png"><img src="draw-0-push-against-true-correction-step-150.png" width="500" alt="Draw 0 at step 150 as noise, the true correction summed over colour looking like noise, and the discriminator's push showing a circle-shaped and a square-shaped region side by side"></a>

At step 150, while draw 0 still looks like noise, the discriminator's push already shows the pair's layout; the true correction there does not. Their cosine is 0.15 (median 0.17 over 300 draws), and the push is 1.2 long against the true correction's 24.9. `verified`; made by a scratch script on the unguided run of seed 7, whose numbers `diagnosis.json` repeats along the guided run.

**[Which target the push points at, step by step](push-cosine-per-step.png)**

<a href="push-cosine-per-step.png"><img src="push-cosine-per-step.png" width="500" alt="Cosine of the push against reverse step, an orange line near 0.85 to 0.96 against the sampler's own ratio and a blue line under 0.21 against the true correction, both falling to 0 by step 80"></a>

Cosine of the push (y) against the reverse step (x, 200 at pure noise to 0), median of 300 draws along the guided run at w = 1, band 25th to 75th. Against the sampler's own ratio (the true noised product's score minus the score of the 3,500 finished sampler draws noised forward): 0.96 at step 190, 0.87 at 150, 0.85 at 100. Against the true correction: at most 0.21. The true correction is half the summed score's size at every step (0.50 at 190, 0.50 at 100); the push is 0.02 to 0.07 of it. Kim et al.'s theory predicts the first target, and the toy confirms it. The size gap says most of the true correction is the summed scores pulling twice as hard toward the centre, which the push does not carry. `verified` for the numbers, `inferred` for the reading. `diagnosis.json`

**[How much each network prefers the pair](minimal-pair-gap-per-step.png)**

<a href="minimal-pair-gap-per-step.png"><img src="minimal-pair-gap-per-step.png" width="500" alt="Log-odds on the side-by-side picture minus on the stacked picture against noise step: the discriminator rising to about 28, the separate-label classifier near 4, shuffled labels flat at 0"></a>

Log-odds on the side-by-side picture minus on the circle stacked on the square (y), against the noise step (x), 256 shared noise draws per step. Discriminator: 2.2 at step 180, 20.2 at 100, 27.9 at 10. Separate-label classifier: 0.1, 4.0, 3.6. A discriminator trained the same way on shuffled labels: within 0.003 of 0 at every step. The discriminator composes because it prefers the pair, not by chance. `verified`, `diagnosis.json`

**The cap and the clip.** Capped at 0.1, 0.3 or 1 times the summed score's size, the pair-aware push still gives 172 at w = 1, because it never reaches 0.1. The true correction under the same caps gives 4, 229 and 300. Kim et al.'s clip (no push once d is within 10⁻⁵ of 0 or 1) is set for every stacked draw by step 60, yet with the clip removed w = 1 still gives 172 and w = 2 gives 300 (`pair-aware-no-clip.json`): the 128 failures are settled before step 90, while the push is on. `verified`, `diagnosis.json`

**Against inclusive captions, the same 300 noises.** Inclusive captions caption a both-shapes picture with each single name, which on this toy is the pair weight inside each expert. No push, seed 7: pair weight 0.01 gives 0 crisp, 0.03 gives 1, 0.1 18, 0.2 53, 0.3 115, 0.5 246 (`inclusive-captions-comparison.json`). The push at pair weight 0.01 (172) beats inclusive captions at 0.3, thirty times the pairs in training. The two methods differ in what they change: inclusive captions retrain the experts, the push leaves them frozen and adds a classifier at sampling. Both need pictures of the pair: the discriminator saw 3,500. On this toy retraining is free because the experts are closed form; on a real model it is not. `verified` for the counts.

**How it was made.** Every count comes from `3-understanding-compositionality/artifacts/notes/what-poe-loses-when-it-composes/toy_classifier_push.py`, one command per piece (`port`, `separate-train`, `separate-guide`, `disc-train`, `pair-guide`, `diagnose`, `no-clip`, `inclusive`), run with `/home/molef/micromamba/envs/sdxl-local/bin/python -I` on the laptop's RTX 4070 on 2026-11-17. The sampler is `reverse()` in `toy_sampler_hybrids.py` with a push hook entering through `guided_eps`, the condition_score form of guided-diffusion: ε minus the weight times √(1 − ᾱ_t) times the gradient. Training data: `discriminator-data.npz`, 3,500 exact draws (seed 101) and 3,500 sampler draws (seeds 201 to 205), never seed 7. Checkpoints: `separate-label-classifier.pt`, `pair-aware-discriminator.pt`. Every guided count was first run as an answer key in a throwaway folder and then rerun as written; the two matched on every row. The plan is `2-classifier-guidance/plans/partial/19-classifier-push-on-the-exact-toy.md`.

**What it cannot say.** Every count is one seed of 300 draws. The labels say crisp, overlaid, faint or single, not whether the pairs land in the exact product's positions in the right proportions. The sampler's own ratio at small steps is a mixture with one sharp spike per draw, so the late-step cosines against it are read loosely, and from step 80 the push is 0 anyway.

**[The side-by-side discriminator](the-side-by-side-discriminator.png)**

<a href="the-side-by-side-discriminator.png"><img src="the-side-by-side-discriminator.png" width="600" alt="Seven stages left to right: a real held-out draw with a red circle and a blue square side by side and a fake held-out draw with the two stacked into one purple blob, with each pile's counts; the noising equation; the network and its weighted loss; d with bars of held-out accuracy at steps 10 to 200; the log-odds gradient; the guided noise prediction; and what the sampler draws with the push, four pictures of draw 0 at no push and w 0.5, 1 and 2 above bars of crisp pairs out of 300"></a>

**What it shows.** The pair-aware discriminator of plan 19 as seven stages: the two piles it learns from (one real draw and one fake draw, both held-out row 3,000), noising to a random step, the network and its loss, d with held-out accuracy at every checked step, the gradient used as the push, the line where that push enters the sampler, and what the sampler draws once the push is in. Every symbol is named on its stage.

**How we got it.** `the-side-by-side-discriminator-inputs.py` reads `discriminator-data.npz` and `pair-aware-train.json` (both written by `disc_data` and `disc_train` in `artifacts/notes/what-poe-loses-when-it-composes/toy_classifier_push.py`) and writes `the-side-by-side-discriminator-draws.csv`, `the-side-by-side-discriminator-accuracy.csv` and the two draw pictures, each 16 × 32 padded white to a square. `the-side-by-side-discriminator-result-inputs.py` reads `pair-aware-guide-samples.npz`, `pair-aware-guide.json` and `separate-label-guide.json`, labels every guided draw with the toy's own `label` (and stops unless those labels reproduce the saved counts), and writes `the-side-by-side-discriminator-result.csv`, `the-side-by-side-discriminator-guided-draws.csv` and the four pictures of draw 0. `~/.claude/skills/hld-diagram/scripts/draw_stages.py the-side-by-side-discriminator.json --id 3000` draws the figure from those files, so the pile counts, the accuracies, the gradient check and the crisp counts are read at draw time. The equations are typed into the spec with the function each comes from beside it. Drawn 2026-11-17.

**What it lets you say.** The discriminator separates the two piles perfectly on held-out draws at steps 10, 50 and 100, still at 0.956 and 0.980 at step 150, and falls toward chance at steps 190 and 200, where the fake pile is called fake only 0.216 of the time. Its gradient matches a finite difference to a relative error of 3.3 × 10⁻⁴. `verified`, from `pair-aware-train.json`.

**What the push does to the sampler.** With the push added at every step (seed 7, 300 draws per run), crisp side-by-side pairs go from 0 with no push to 16 at w = 0.5, 172 at w = 1 and 300 at w = 2; the bar fixed before the run (at w = 1, at least 150 crisp and no more overlaid than unguided) passes with 172 crisp and 128 overlaid. The trained separate-label classifier, which only knows "circle present" and "square present", reaches at most 10 of 300 at any scale. Draw 0, from the same starting noise in every run, stays overlaid until w = 2. `verified`, from `pair-aware-guide.json` and `separate-label-guide.json`.

**d on the two draws.** `the-side-by-side-discriminator-d.py` loads `pair-aware-discriminator.pt` into the toy's `Net`, noises each row-3,000 draw to every checked step 200 times (seed 0) and writes `the-side-by-side-discriminator-d.json`; the inputs script copies the step-100 values onto the figure. At step 100, d is 1.00 on the real draw (0.993 to 1.000) and 0.00 on the fake one (0.000 to 0.000). At step 150 the noise draw starts to matter: 0.938 on average for the real draw, as low as 0.192. Run on the laptop on CPU, 2026-11-17. `verified`.

**What it cannot say.** Whether the crisp pairs at w = 2 are spread over the exact product's positions in the right proportions: the labelling only says crisp, overlaid, faint or single, and every guided count is one seed. The separate-label comparison on the figure is the trained classifier; the closed-form best separate-label classifier reaches 34 of 300 at scale 10 (`separate-label-guide.json`), still far below the pair-aware push. One value on the figure is typed, not read: the λ range (0.36 at step 100 to 9.80 at step 200), recomputed from `LAMBDA`'s formula on the toy's schedule. `inferred`.

**Versions.** `the-side-by-side-discriminator__2026-11-17.png` (with its `.json`) is the six-stage version drawn before guiding ran, with the crisp count marked not yet measured. `the-side-by-side-discriminator__2026-11-17-2.png` is the current seven-stage version, which adds the result stage and names the push strength w, as the guide run does.

**Where it came from.** Plan 19 in `2-classifier-guidance/plans/partial/19-classifier-push-on-the-exact-toy.md`; the recipe from Kim et al. 2023, section 3, as the read map's "What A1 takes from this" records it.

<!-- cluster:start -->

## On the cluster

The other 2 files of this result (0.0 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/3-understanding-compositionality/artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `(files at this level)` | 2 | 0.0 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/3-understanding-compositionality/artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/3-understanding-compositionality/artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/<run>/ <run>/
```

<!-- cluster:end -->
