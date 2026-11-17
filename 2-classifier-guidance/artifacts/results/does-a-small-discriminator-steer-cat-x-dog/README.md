# Does a small pair-aware discriminator steer SDXL's cat × dog product?

Rung 2 of the classifier-guided PoE ladder ([A3q's page](../../../report/classifier-guided-poe/a3q-cat-x-dog-quick-discriminator-test.md),
[plan 21](../../../plans/21-rung2-a-small-discriminator-steers-cat-x-dog.md),
[plan 22](../../../plans/22-rung2-a-discriminator-trained-on-the-benchmark-pairs.md)). Everything here
was made on 2026-11-17.

## training-set/

**What it is.** The 73 renders rung 2's first discriminator trained on: seeds 1 to 8 × six prompts,
plus 25 fresh-noise draws that replaced damaged lone-animal and product tiles and added pairs.
`calls.csv` holds every call (17 positives, 24 intact negatives); `problem-seeds.csv` lists the 23
pair-prompt tiles that did not give one cat and one dog apart.

**How it was made.** `scripts/rung2/render_training_set.py` on `mscluster107` GPU 0; the reader
called the first 48 tiles, and delegated the rest to Claude with veto after.

<a href="training-set/strips/kept-positives.png"><img src="training-set/strips/kept-positives.png" width="200" alt="The 17 positive tiles: one cat and one dog, separate, intact photos"></a>
<a href="training-set/strips/kept-negatives.png"><img src="training-set/strips/kept-negatives.png" width="200" alt="The 24 negative tiles: a lone cat, a lone dog and a plain product per seed 1 to 8"></a>

## result-sheet/

**What it is.** Seeds 1 to 4 and 9 to 12 by plain product, best of 2, guided at s = 10, 30, 100 and
a same-sized random push at each, rendered with the first discriminator. `calls-draft.csv` holds
Claude's draft calls (the reader has not confirmed them): 0 of 4 composed on seeds 9 to 12 in every
column. `result.json` holds the per-step push sizes, cap firing and checker scores.

<a href="result-sheet/result-sheet.png"><img src="result-sheet/result-sheet.png" width="200" alt="The 8 by 8 result sheet"></a>

## diagnostics/

**What it is.** What the first discriminator reads: filmstrips of the predicted picture with its
attention per step (seeds 1, 3, 9, 10), its score through all 50 steps per run, and every training
tile kind scored per step. Made by `scripts/rung2/diagnose.py` on `mscluster107` GPU 1.

<a href="diagnostics/kinds-through-time.png"><img src="diagnostics/kinds-through-time.png" width="200" alt="Logit against step per tile kind: hybrids and wrong pairs from the pair prompts score with the positives"></a>

## discriminator/

**What it is.** Held-out balanced accuracy per sampler step for every discriminator trained:
v1 (17 positives, easy negatives), v2 (pair-prompt failures added as negatives), v3 head only and
v3 with adapters (81 positives over 38 benchmark pairs). Only v1 passes, on a check that held no
hard negatives.

<a href="discriminator/held-out-accuracy-all-checkers.png"><img src="discriminator/held-out-accuracy-all-checkers.png" width="200" alt="Held-out accuracy against step for four discriminators; only v1 clears 0.75"></a>

## dataset-v3/

**What it is.** Plan 22's training set from about 1,000 local 1024 px pair pictures: 543 called by
two independent passes of helper agents (91% agreement), the 37 disagreements in the 87 two-things
pairs settled by Claude at full size. `calls-final.csv` records how each call was decided;
`spot-check-sheet.png` is the reader's sample. The latents and the pictures for training live in
`/home-mscluster/mmolefe/poe_repair_min_scratch/outputs/classifier_guidance_ladder/rung2/` while
`/datasets` is down.

<a href="dataset-v3/spot-check-sheet.png"><img src="dataset-v3/spot-check-sheet.png" width="200" alt="Spot-check sheet: 12 positives and 12 negatives with their calls"></a>
