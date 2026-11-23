# The CLEVR scaling experiment

## What this asks

How many objects survive when diffusion models trained on one object at a time are composed? A count K succeeds when K objects appear in at least 50%
of images and at least K-2 appear in at least 90%, and the largest K where every count up to it succeeds is called Kmax. This design turns the agreed
[CLEVR scaling plan](../../targets/clevr-scaling-plan.md) into runs. The [fairness audit](../../targets/fairness-audit.md) records why each rule exists.

## Status

The renderer is installed and timed, and no training data has been rendered. On a 28-thread i9-10940X with 32-pixel render tiles, one single-object image takes
5.46 seconds and one empty scene takes 4.50 seconds (job 93314). The generator's default 256-pixel tiles took 51.63 and 46.96 seconds (job 93287), because few
threads have work at that tile size. The generator's help text says tile size does not change the picture, and that has not been checked here. Any run past 24 GPU-hours goes to the owner first.

## The rows this serves

| Row | Method | Printed Kmax | Source | How it is judged |
|---|---|---|---|---|
| `Bradley-Local-T1-004` | Local patch model, trained on one object | 5 | Local Mechanisms Table 1, p3 | printed context; our rebuild is labelled a reimplementation |
| `Bradley-Projective-T2-003` | Explicit composition with an empty background | 4 | Projective Composition Table 2, p39 | printed context; its rule runs on our models |

Printed numbers stay context, because Bradley trains on 134,217,728 images per model and our full length is 8 million. No rerun here is labelled
`reproduced` below Bradley's training length. A win needs at least 7 objects at equal training length, more than the one-network control, and for Paper 1
more than sample-and-paste.

## The methods

**Group.** The one-network control and sample-and-paste are baselines (group 5). Experts without a checker are the first comparison. The checker runs
join after step D3 has chosen one.

**Varied axis.** The method, one at a time, on the same training length, positions and seeds.

| Method | Models | How it makes N objects |
|---|---|---|
| One network | one location network, one empty-background network | Bradley's rule: add the scores, asking the same network about each position |
| One network with checker | the same two | that composition plus the checker (Paper 2) |
| Experts | one expert per position, the same background network | add the N experts' scores |
| Experts with checker | the same experts | that composition plus the checker (Paper 1) |
| Sample and paste | the same experts | sample each expert alone once per seed and paste its object onto one background |
| Local patch model | one patch network | rebuilt from Local Mechanisms p29, after a costed question |

The local patch model has no released code: the paper links none, and a GitHub search by title and by arXiv id returned nothing. Its rebuild waits for its
own question.

## The protocol

| Setting | Value | Source |
|---|---|---|
| Renderer | CLEVR generator at `f0ce2c81` with Blender 2.78c, default objects, sizes and colours | Projective Composition p15 |
| Image size | rendered at 320 by 240 and downsampled, to 64 pixels first and then 128 | p15 and the plan's two data rungs |
| Positions | (0.65, 0.65), (0.65, 0.25), (0.25, 0.65), (0.35, 0.35), (0.45, 0.65), (0.45, 0.25), (0.65, 0.45), (0.25, 0.45), (0.45, 0.45), (0.55, 0.55), (0.35, 0.55), (0.55, 0.35), in this order | Local Mechanisms p23 |
| Model | EDM2 `edm2-img64-xs`, with the class embedding replaced by concatenated Fourier embeddings of x and y | Projective Composition p15 |
| Guidance | none: raw conditional scores | p15 |
| Sampler | EDM2's 32-step Heun, 65 model calls | p15 |
| Final weights | the average over the last 4,096 updates | p15 |
| Images scored | 1,024 per object count | Local Mechanisms p23 |

## The rules, fixed before the first run

1. Each expert must draw its one object at its position in at least 90% of 256 samples, read by the scorer. It gets one training, and a failure is reported.
2. The ladder trains the background, the control and experts 1 and 2, tests 2 objects, then adds one expert and one count per rung. Earlier experts are
   never retrained.
3. The control must reach Kmax 4 at the same training length before any method is compared with it. If it does not, the shortfall is reported.
4. The ladder stops after two failed counts in a row, or at 10 objects.
5. At the first failed count, the N leave-one-out groups are tested, which separates the number of experts from one bad position.
6. In Stage 1 no model and no checker sees a real image holding two objects. The checker trains on model samples and composites pasted from them, at random
   positions, and the evaluated position sets are held out. A no-composite checker is reported beside it.
7. The scorer trains on real renders of 1 to 12 objects, reads objects at the requested positions as well as the total count, and is never used to train a
   checker. It must reach 99% on held-out renders.
8. The picture-quality bar is a constant in the scorer's code, committed before any composed image is scored.
9. A person checks 100 images per count at full size, under neutral file names.

## The ladder of runs

1. **Install and time.** ~~Install Blender 2.78c and render one batch of 8 single-object images and 8 empty scenes on one node, recording seconds per image.~~
   Done: 5.46 seconds per single-object image with 32-pixel tiles.
2. **Set the data size.** Write the images per model and the rendering cost here, from the timing.
3. **Smoke test.** A short training of one location network and one batch of samples, recording seconds per thousand images.
4. **Overfit.** The model fits 16 images, and the scorer fits a small set.
5. **Stage 1 at a quarter of the 8 million image length.** The plan estimates 7 to 9.4 GPU-hours up to 2 experts and 21 to 28 up to 10. A run past 24 GPU-hours
   goes to the owner first.
6. **Stage 2 and Stage 3**, each with its own table, after Stage 1.

## What is still open

- How many images each expert sees, which the timing sets.
- The picture-quality measure and its threshold, which must be committed before any composed image is scored.
- Whether to rebuild the local patch model, which gets a costed question.
