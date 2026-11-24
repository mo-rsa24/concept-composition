# Does sample-and-paste already solve small CLEVR? (step M7)

## What this asks

Paper 1 tied sample-and-paste on MNIST, so it needs a test where pasting is not enough. This stage trains small CLEVR models that each
saw one object, pastes their samples together, and measures how often each object lands in place. If that rate is high enough to predict
7 objects, counting objects cannot separate Paper 1 from paste, and the owner gets a replacement test before any tuning on CLEVR.

## Words this uses

**Expert.** A small diffusion model trained only on pictures of one object at one position.

**Background model.** A model of the same size trained only on empty scenes.

**Sample and paste.** Sample each expert alone, cut a soft-edged window around its position, and lay the windows on one background sample.

**In-place rate.** The share of asked objects the scorer finds at their asked positions, from 0 to 1.

**Bradley's rule.** A count K succeeds when all K objects appear in at least 50% of images and at least K-2 appear in at least 90%.

## The run

**Group.** Baselines (group 5). The paste control is frozen by this design and is not tuned after a number lands.

**Varied axis.** How many experts are pasted: 2 or 3. Renderer, model size, training length, sampler and scorer stay the same.

**Data rule.** No model sees a real picture holding two objects. Only the scorer does, and it trains no checker. Bradley's 12 printed
positions stay held out.

## Positions

The tuning positions, as (x, y) fractions of the image width and height, are:

| Expert | Position | Used with 2 experts | Used with 3 experts |
|---|---|---|---|
| 1 | (0.30, 0.30) | yes | yes |
| 2 | (0.60, 0.30) | yes | yes |
| 3 | (0.40, 0.60) | no | yes |

None is one of Bradley's 12. Each pair is at least 19 pixels apart along one axis at 64 pixels. All three lie inside the range real CLEVR
objects occupy: over 97,358 validation objects, the 1st to 99th percentile is 0.13 to 0.89 in x and 0.23 to 0.79 in y.

## Data

The renderer is the CLEVR generator at `f0ce2c81` with Blender 2.78c, with its default objects, sizes, colours, materials, camera jitter
and light jitter, rendered at 320 by 240 with 32-pixel tiles and squashed to 64 by 64. A wrapper in
`2-classifier-guidance/scripts/method_search/` places each object on the ground point that projects to its target position under that
image's jittered camera, with a uniform offset of up to 0.02 of the image in x and y.

| Set | Pictures | Used by |
|---|---:|---|
| One object at position 1, 2 or 3 | 5,000 each | that expert |
| Empty scenes | 5,000 | the background model |
| Every subset of the three positions, 0 to 3 objects, 250 per subset | 2,000 | the scorer, split 1,600 for training and 400 held out |

## Models

Each of the 4 models is EDM2's `edm2-img64-xs`, unconditional, trained once for 2 million images seen, a quarter of the earlier 8 million
image length. The preset ramps its learning rate over 10 million of its 1,074 million images, 0.93% of training, and each model keeps that
fraction, about 19,000 images for a run of 2 million. An unscaled ramp would hold such a run at or below 20% of the full rate. Each model
trains with a batch of 256, against the preset's 2,048, so a run of 2,000,896 images makes 7,816 weight updates. Sampling uses
EDM2's 32-step Heun sampler with 65 model calls and no guidance. The one-network control belongs to step M6 and trains only if this stage
does not stop.

## The scorer

A small convolutional classifier reads a 64 by 64 picture and answers, for each of the three positions, whether an object is there. It
must reach 99% accuracy per position on the 400 held-out renders before any sample is scored. Its threshold is 0.5.

## The paste, frozen before any sample exists

Each expert and the background draw from their own seeds. Around each asked position, a window of 18 by 18 pixels is taken from that
expert's sample, with its outer 2 pixels faded linearly, and laid on the background sample. Windows do not overlap at these positions.

**Control on the window.** On 256 real single-object renders per position, at least 95% of each object's own footprint, the pixels that differ
from the mean empty scene by more than 0.1 on a 0-to-1 scale and connect to the position, must fall inside the window. Position 3's window is 18
wide and 24 tall, starting on the row where the other two end, because its objects sit nearer the camera. The owner set this check and window on
board question 66, after the first check, which counted every changed pixel in the frame, missed at all three positions.

## What gets measured

1. **Each expert alone.** 256 samples. It must draw its object at its position in at least 90%, per the [scaling plan](../../targets/clevr-scaling-plan.md).
2. **Paste of experts 1 and 2.** 1,024 images: the rate of each object in place and the share with both.
3. **Paste of experts 1, 2 and 3.** 1,024 images: the rate of each object in place and the share with all three.

Thirty-two images per paste count are also called by eye, enlarged, under neutral names, and compared with the scorer.

## The bar, fixed before the first run

The **paste rate** is the mean of the three objects' in-place rates on the 1,024 pasted images with 3 experts. Bradley's rule, with objects
assumed independent, predicts 7 objects once this rate reaches 0.906, because $0.906^7 \approx 0.50$.

| Result | What it takes | What happens |
|---|---|---|
| Paste predicted to reach 7 | paste rate at least 0.906 | M7 stops, and the owner gets a replacement test where pasting fails, such as objects close enough to overlap or shade each other |
| Paste falls short | paste rate below 0.906 | step M6 runs on these models |
| Inconclusive | an expert misses 90%, the scorer misses 99%, or the window control fails | the failure is reported, and no verdict is read |

**What would surprise.** Each expert must pass 90% to count, and the stop rule needs 90.6%, so an expert that passes will usually trigger
the stop. A paste rate well below an expert's own rate would mean the paste step is damaging objects, and that would be checked first.

## Cost and what it buys

**Cost:** about 32 CPU node-hours to render (15,000 single-object pictures at 5.46 seconds, 5,000 empty scenes at 4.50 and 2,000 scorer
pictures at a budgeted 5.2) and 8 to 10 GPU-hours for 4 models at the plan's 1.8 to 2.4 each, plus sampling and the scorer. No API tokens.
The smoke rung replaces these budgets with measured times.

**Buys:** step M7, and the answer to whether object count on CLEVR can tell Paper 1 apart from pasting before any time goes into tuning it.

## The ladder

Each rung saves its config, numbers, sheet and commit under `/datasets/mmolefe/poe_repair_min/loop-6/m7/<rung>/`.

1. **Smoke test.** Render 8 pictures per set, check by eye that each object sits at its position, train one model for a few thousand
   images, and sample one batch. Records seconds per picture and per thousand images.
2. **Overfit.** One model trains on 16 pictures of position 1 for 256,000 images seen. Its 16 samples must look like those scenes by eye, and
   the scorer must find the object at position 1 in at least 15 of them. The scorer separately fits 16 real pictures and gets at least 15 right.
3. **1% of the data.** 50 pictures per set train, and the logged loss falls.
4. **Controls.** The window control and the scorer's 99% check.
5. **Comparison.** The full renders, four trainings, and the three measurements above.

## What this cannot say

Three experts cannot show 7 objects, so the prediction assumes objects succeed independently. The models train on 5,000 images for a
quarter of the earlier length, and Bradley trains far longer on far more. A paste rate on three positions says nothing about crowded
scenes, where objects overlap and shade each other. That is the case a replacement test would target.
