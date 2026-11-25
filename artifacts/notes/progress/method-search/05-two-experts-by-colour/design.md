# Can two experts trained on separate colours compose scenes neither saw? (CLEVR Stage 2)

## What this asks

Paper 1 composes diffusion models trained on separate data. On CLEVR with one object per expert at a fixed place, sample-and-paste is
nearly perfect: each of M7's four experts places its object in 256 of 256 samples. This test removes what pasting needs. One expert sees
only red objects and one only blue, 1 to 3 at a time, anywhere in the frame, and neither is told where to draw. A scene asks for x red and
y blue objects, all separate, and the count rises from 2 until the composition breaks. The owner chose this test on 2026-11-24, in chat, as
option A, "two experts, told what and not where, pushed from 2 objects up to where they break". Its source is Stage 2 of
[the CLEVR plan](../../targets/clevr-scaling-plan.md).

## Words this uses

**Expert.** A small diffusion model trained only on scenes of one colour. It takes a count label k, from 1 to 3, and draws k objects of its
colour anywhere.

**Cell.** One request, x red and y blue objects, with x and y each 1, 2 or 3. The 9 cells ask for 2 to 6 objects.

**Exact share.** The share of a cell's images holding exactly x red and y blue objects, all separate.

## The run

**Group.** Comparison runs. **Varied axis.** How the product of the two experts is sampled. The models, cells, seeds and scorer stay fixed.

| Row | What it does | What it separates |
|---|---|---|
| Plain product | red expert at count x, plus blue expert at count y, minus the empty-scene model, with EDM2's 32-step Heun sampler | the floor |
| Product with Loop 8's count checker | the same, plus the count checker's push, retrained on these experts' maps at the noise levels of the sampler's early steps and pushing only on those steps, as in [its digit test](../../abstract-checkers/06-does-it-compose/review.md) | Paper 1's method |
| Product with Du et al.'s MCMC corrector | the plain product with the corrector's steps between sampling steps | a better sampler, apart from a new signal |
| Mask paste | for each image, the red expert's sample at count x, with the blue expert's foreground laid over it: the pixels of the blue sample that differ from the mean empty scene by more than 0.1 on a 0-to-1 scale. CPU only, from the same samples the other rows draw | the rival a composition must beat: objects that land apart need no composition |

Mask paste needs no positions, because each expert's own sample says where its objects are. It fails where a blue object lands on or
touches a red one, so its exact share falls as the count rises.

**The count checker's training set.** Loop 8's count checker is retrained on mask pastes of the two experts' own samples, labelled with the
counts the experts were asked for, at the noise levels of the sampler's early steps. No real scene holding both colours is used, as in
[its digit test](../../abstract-checkers/04-environment/review.md), where the checker trained on pasted canvases.

## Data and models

**Renders.** The CLEVR generator and Blender setup from M7, at 320 by 240, squashed to 64 by 64, with default shapes, sizes, materials,
camera jitter and light jitter, and objects placed anywhere by the generator's own rule. Only the colour is restricted.

| Set | Pictures | Used by |
|---|---:|---|
| Red only, 1, 2 or 3 objects | 5,000 per count, 15,000 in all | the red expert |
| Blue only, 1, 2 or 3 objects | 5,000 per count, 15,000 in all | the blue expert |
| 0 to 3 red with 0 to 3 blue, every combination | 250 per combination, 4,000 in all | the scorer only, 3,200 to train and 800 held out |

**Models.** Each expert is EDM2's `edm2-img64-xs` with its count as a class label, trained for 2,000,896 images at batch 256 with the
learning-rate ramp scaled to the run, the recipe M7's overfit and 1% rungs passed. The empty-scene model is M7's, trained on the same
renderer. No model sees a scene holding both colours, and only the scorer does.

## The scorer

A small convolutional network reads a 64 by 64 picture and counts red and blue objects, each from 0 to 3. It must count both colours exactly
on at least 95% of the 800 held-out renders before any sample is scored. Merged objects lower the count, so a merge counts as a failure.
Thirty-two images per cell and row are also called by eye at full size under neutral names, and compared with the scorer.

## The bars, fixed before the first run

**Each expert alone.** At each count k, at least 90% of 256 samples hold exactly k objects of the expert's colour and none of the other.
An expert that misses makes the stage inconclusive.

**Where a row breaks.** The largest total N for which every cell with x + y = N has an exact share of at least 50%, Bradley's half-of-images
rule applied per cell.

| Result | What it takes |
|---|---|
| Support | the count checker row leads the plain product, the corrector row and the mask paste, each by at least 5 points of exact share averaged over the 9 cells and by more than 1.96 standard errors over cells |
| Paste explains it | the mask paste matches or beats the checker row; Paper 1 then needs scenes whose objects overlap or shade each other |
| Corrector explains it | the corrector row matches the checker row within 1.96 standard errors |
| Null | no lead over the plain product |
| Inconclusive | an expert misses 90%, the scorer misses 95%, or the eye calls disagree with the scorer on more than 1 image in 10 |

**Past 6 objects.** If a row holds every cell up to 6 objects, a third expert sees only green, and cells up to 9 objects run on the same bars.
It is costed below and starts only with the owner's yes.

## Paper 1's evaluation measures, beside the exact share

The owner accepted Paper 1's evaluation plan for this test on board question 109, on 2026-11-25:
[the evaluation plan](../../targets/paper-1-evaluation-plan.md) sets five measures where pasting fails (which object hides which, one light
for every object, a shadow falling on the other object, seams and realism, colour staying with its object), each with its own reader check
and the plan's margin. They score this test's samples and change none of the bars above. Three additions from that answer apply here.

**A fifth row: mask paste with a shadow generator.** The mask paste, then a published multi-object shadow generator such as MultiShadow
(arXiv 2603.02743), scored on measures 2 to 4 only. It runs only if the generator works on 64-pixel CLEVR, checked first on 16 mask
pastes from the experts' samples, enlarged to the generator's input size and shrunk back. It works when two blind reviewers both see a shadow
added beside every object, all falling the same way, with the objects themselves unchanged, on at least 12 of the 16. If no released
generator with weights exists, or it fails that check, the row is reported as not run and why.

**DepthArb's box-level occlusion score beside measure 1.** For each overlapping red and blue pair, the share of the two objects' box overlap
covered by the nearer object's box (arXiv 2603.23924), reported beside the plan's pixel reading and deciding nothing.

**The count checker's training images beside every result.** A list of each image the count checker trained on: its file, the two expert
samples it was pasted from with their seeds, and the counts it was labelled with. It shows that no image holding both colours, and no real
scene, reached the checker.

## Cost and what it buys

The renders run on CPU. Blender 2.78c, which the CLEVR generator needs, ships no GPU code for the cluster's RTX 3090 and Blackwell cards and
tries to build it with NVIDIA's compiler, which the cluster does not have, so every GPU frame fails (job 94448).

Rates are M7's measurements: 5.46 seconds per single-object render on one CPU node, with more objects budgeted at 7 seconds, and 2.25
seconds per thousand images seen on one RTX PRO 6000 Blackwell.

| Part | CPU node-hours | GPU-hours |
|---|---:|---:|
| Renders, 30,000 for the experts and 4,000 for the scorer | about 66 | none |
| Two experts at 2,000,896 images each, with start-up | none | about 2.6 |
| Count checker retrained on the experts' samples | none | about 1 |
| Sampling 3 rows by 9 cells by 1,024 images, three models per step, with the corrector's extra steps | none | about 4 |
| Scorer, expert checks and pastes | about 1 | none |
| **Stage 2 in all** | **about 67** | **about 7.6** |
| The evaluation plan's measures on these samples: reference renders (1,024 for each of the 9 cells and 500 crowded), readers, realism distance and the classifier test | about 21 | under 2 |
| The shadow generator's check on 16 mask pastes, and the fifth row on the measures 2 to 4 images if it passes | none | under 1, estimate |
| **With the evaluation measures** | **about 88** | **about 10.5** |
| A green expert and cells to 9 objects, only with the owner's yes | about 33 | about 4 |

**Buys.** Paper 1's test where pasting cannot apply, with a measured point where two separately trained experts stop composing, and
whether a checker moves that point.

## The ladder

1. **Smoke.** 8 renders per set checked by eye for colour and count, one expert trained for 65,536 images, and one batch sampled.
2. **Overfit.** One expert fits 16 pictures, and its samples hold the right count of the right colour in at least 15 of 16.
3. **1% of the data.** 50 pictures per count train each expert, and the logged loss falls.
4. **Controls.** The scorer's 95% check and each expert's 90% check.
5. **Comparison.** The four rows on the 9 cells, and the evaluation plan's five measures on the same samples, with the fifth row on
   measures 2 to 4 if the shadow generator passes its check.

## Watching the full runs, and when to stop one

The two experts train on RunPod RTX 4090 pods with the same arguments as the cluster job (`m10_pod/run_ladder.sh`), 128 images per pass
and batch 256. Each full run saves a checkpoint every quarter
(500,224 images). The owner asked to see each quarter and to stop a run that is failing; this rule was written on 2026-11-25, before the
first quarter.

**What each quarter shows.** `m10_pod/quarter_watch.py` samples the quarter's 0.100 EMA snapshot on the CPU, 16 seeds (0 to 15) per count
label in 32 steps, and reports the right-count share: how many of the 48 samples a colour-blob counter (`m10_pod/blob_count.py`, threshold 30
on the colour channel's lead, pieces of at least 4 pixels) reads as holding the asked count. The counter is cheap and imperfect: on real
training pictures it reads 0.82 of red and 0.86 of blue right (1.00 at one object, 0.78 and 0.96 at two, 0.68 and 0.62 at three, where
objects hide each other), so a sample share is read against that ceiling. With 48 samples a share moves by about 0.07 by chance. The
quarter's mean training loss comes from `stats.jsonl`.

**The stopping rule.** It only stops a failing run and never declares success, which stays with the bars above.

| Check | Stop the run when |
|---|---|
| any quarter | the loss is not a number, or the samples are blank or noise by eye |
| 75% (1,500,672 images), the first check after half the run | the right-count share is not above its value at 50% and is below 0.75 times the counter's ceiling on real pictures of that colour (0.62 for red, 0.65 for blue), or the quarter's mean loss is more than 0.05 above the 50% quarter's |

## Running the comparison

These settings were written on 2026-11-25, after the experts finished training and before any comparison row ran. The owner chose, through
Loop 13, to run every GPU row on one RunPod RTX 4090 (the red expert's pod), so every row shares a card type, within about 7 GPU-hours.
The scorer trains on the cluster's CPUs.

**Models.** Each expert's final 0.100 EMA snapshot (2,000,896 images) and M7's empty-scene model, its final 0.100 snapshot.

**The product.** At each sampler step the three denoised estimates combine as D = D_red(x; σ, x) + D_blue(x; σ, y) − D_empty(x; σ), the
product of experts written in denoised form. EDM2's sampler: 32 Heun steps, σ from 80 to 0.002 with ρ = 7, no added noise. Each cell runs
seeds 0 to 1,023, and a seed gives the same starting noise in every row.

**The count checker.** It reads the two maps M_red = D_red − D_empty and M_blue = D_blue − D_empty at the noisy picture (6 channels), the log
of σ and the asked counts (x and y, each one of three), and gives one logit: the picture will hold exactly x red and y blue objects, all
separate. The network is the scorer's four convolution blocks with those inputs.

- **Its training set.** Paste sources are each expert's samples at seeds 100,000 to 100,511, 512 per count, never seeds 0 to 1,023. A
  training picture lays a blue sample's foreground over a red sample, as the mask paste does. It is a positive when no blue foreground pixel
  lies within 1 pixel of a red foreground pixel and it is asked the counts its sources were asked for. Negatives come in equal number from
  two kinds: the foregrounds touch or overlap, asked their sources' counts; or they are apart, asked one count that differs by 1. Each picture
  is noised at a σ drawn from the push window. Seeds 100,410 to 100,511 are held out.
- **Its bar, before its row runs.** A balanced accuracy of at least 0.75 on the held-out pastes, the digit test's bar.
- **The push.** On sampler steps 10 to 14 (σ 10.5 to 3.8), which cover the digit test's window by noise level (σ 13.1 to 3.1), each
  denoised estimate moves to D + b · ‖x − D‖ · g / ‖g‖, where g is the gradient of the checker's logit with respect to x and b = 0.4, the
  digit test's frozen value, not tuned here. A zero push (b = 0) must match the plain product to within 1e-5 per pixel.

**The corrector.** After every sampler step to a σ above 0, two unadjusted Langevin steps on the product at that σ: x ← x + ε·s + √(2ε)·z,
with s = (D − x)/σ², ε = 0.05·σ² and z drawn from the seed's own generator. Du et al.'s sampler family; the step count and size are set here
and not tuned.

**The mask paste.** The red expert's sample at count x and the blue expert's at count y, both at the cell's seed, with the mean empty scene
taken over 1,024 empty-scene samples at seeds 0 to 1,023.

**Reporting as it goes.** Each row runs in four blocks of seeds (0 to 255, 256 to 511, 512 to 767, 768 to 1,023) over all 9 cells. After each
block the owner gets a grid (the first 8 seeds of each cell) and the exact share so far, beside the plain product's on the same seeds.

**When to stop early.** These only stop failing work and never declare a result.

| What | Stop when |
|---|---|
| any row | the measured rate, after its first block, puts the whole comparison past 8 GPU-hours: stop and ask the owner |
| the count checker's training | its held-out balanced accuracy at half its updates is 0.55 or lower |
| the count checker's row | after its second block, its exact share over the seeds done is at or below the plain product's on the same seeds; the row is then reported as a null on those 4,608 images |

## What this cannot say

Two colours make each expert's objects easy to tell apart, so this tests composition of counts and places, and leaves identity mixing to
Paper 2. CLEVR objects at 64 pixels are small and seldom touch, so occlusion is rare here. The empty-scene model and the experts share one
renderer, which a real deployment of separately trained models would not guarantee.
