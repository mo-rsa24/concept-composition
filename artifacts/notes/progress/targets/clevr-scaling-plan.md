# CLEVR scaling plan: how many objects survive as experts are added

This experiment tests how far composition holds as more objects are asked for, for both papers. Paper 1 adds experts that were
trained apart. Paper 2 uses one network composed with itself, plus our checker. Session 5 writes each run's design from this page,
and the [fairness audit](fairness-audit.md) records why each rule below exists.

![One CLEVR experiment for both papers](one-clevr-experiment-for-both-papers.png)

## What each method does

Every method runs on models trained for the same length, at the same positions and seeds.

| Method | Models trained | How it makes N objects | What it is for |
|---|---|---|---|
| One network | One location network and one empty-background network | Bradley's rule: add the scores, asking the same network about each position | The rival's rule on our models |
| One network with checker | The same two networks | That composition plus our checker | Paper 2 |
| Experts | One expert per position and the same background network | Add the N experts' scores | Shows what training apart does alone |
| Experts with checker | The same experts | That composition plus our checker, once step D3 has chosen it | Paper 1 |
| Sample and paste | The same experts | Sample each expert alone, then paste its object onto one background | The control Paper 1 has to beat |

Bradley notes that with an empty background the sampler acts as if each model
["pastes" an object](https://arxiv.org/pdf/2502.04549#page=2), so experts at fixed positions could score well with no real
composition. That is why the paste control is required.

## The order of the runs

**Data rungs.** The first rung is single-object CLEVR at 64 pixels from the public
[CLEVR generator](https://github.com/facebookresearch/clevr-dataset-gen). The second is 128 pixels, Bradley's image size. Blender is
not on the cluster's path, so the first step is to install it and time one batch of renders.

**Stage 1: the expert ladder.** Train the background, the one-network control and experts 1 and 2, then test 2 objects. Add one
expert per rung and test one more object. Earlier experts are never retrained.

**Stage 2: two experts holding several objects.** Model A sees only red objects and model B only blue, each 1, 2 or 3 at a time and
anywhere in the frame. The 6 models pair into 9 scenes of 2 to 6 objects. Success is x red and y blue objects, all separate. These
models see real multi-object images, so this is its own table and does not count toward the one-object target. Giving each model
its own half of the image is not allowed, because that is two pictures pasted side by side.

**Inclusive labels do not run here.** Training a diffusion model on inclusive captions is too heavy for the main results, so they get
one appendix paragraph, written from the existing toy, MNIST and SDXL results.

## The rules, fixed before the first run

1. Each new expert must draw its one object at its position in at least 90% of 256 samples, read by the scorer, so a weak expert is
   not read as a scaling limit. It gets one training, and a failure is reported. The 90% is our own mark. Bradley's one-object model
   places its object in [100 of 100 hand-counted images](https://arxiv.org/pdf/2502.04549#page=17).
2. Positions are Bradley's [12 printed locations](https://arxiv.org/pdf/2509.16447#page=23), used in his printed order.
3. Models use his (x, y) location conditioner and the [32-step Heun sampler with 65 model calls](https://arxiv.org/pdf/2502.04549#page=15).
4. The one-network control should reach about 4 objects, the number Bradley prints for it. If it falls short at our training length,
   the shortfall is printed beside every comparison with it.
5. The ladder stops after two failed object counts in a row, or at 10 objects.
6. At the first failed count, the N groups that each leave one expert out are tested, to separate the number of experts from one bad
   position. At 10 experts that is 10,240 images, where testing every smaller group would take 1,036,288.
7. In Stage 1, no model and no checker sees a real image holding two objects. The checker trains only on samples from the models
   and composites pasted from them at random positions, and the position sets used for evaluation are held out. A checker trained
   without composites is reported beside it. A checker that sees real multi-object images does not count toward the 7-object target.
8. The scorer trains on real rendered images of 1 to 12 objects, reads objects at the requested positions as well as the total, must
   reach 99% on held-out renders, and is never used to train a checker.
9. The picture-quality bar is a threshold in the scorer's code, written before the first run.

## How every method is scored

Every method uses 1,024 images per object count and reports three things.

**Most objects that succeed.** A count K succeeds when K objects appear in at least 50% of images and at least K-2 appear in at
least 90%. This is Bradley's rule.

**Images with all N objects in place.** This is the count behind Projective Composition's 98 of 100 at 5 objects and 97 of 100 at
6. A person also checks 100 images per count at full size under neutral file names, so those numbers sit directly beside ours.

**Picture quality.** A count with the objects in place but a degraded image does not pass.

## What counts as a win

Printed numbers are context, because Bradley trains on 134 million images and our full length is 8 million. A win needs all of these
at equal training length:

- more objects that succeed than the one-network control;
- at least 7 objects, because Bradley's rule already places all 6 in 97 of 100 images and first clearly fails at 7;
- for Paper 1, more objects than the sample-and-paste control.

| Table | Printed context | What we need |
|---|---|---|
| One object seen, one network | 6 in place in 97 of 100 images, first clear failure at 7 | **7 objects, and more than the control** |
| Experts trained apart | No rival prints this | **7 objects, and more than the control and the paste** |
| Two experts holding several objects | No direct rival. Bradley's colour model reaches 4 after training on 1 to 3 objects, on a different task | A new table |

## What it costs

All costs are estimates. Our last CLEVR training at 8 million images took about 13h 50m for two models sharing one card, and a lone
probe ran at 4.05 seconds per thousand images, so a model costs 7 to 9.4 GPU-hours. The first rung measures the real cost. Rendering
is unmeasured.

| Stage | Models | GPU-hours at a quarter of that length | GPU-hours at the full length |
|---|---|---|---|
| Ladder, up to 2 experts | 4 | 7 to 9.4 | 28 to 38 |
| Ladder, up to 6 experts | 8 | 14 to 19 | 56 to 75 |
| Ladder, up to 10 experts | 12 | 21 to 28 | 84 to 113 |
| Two experts holding several objects | 6 | 11 to 14 | 42 to 56 |
