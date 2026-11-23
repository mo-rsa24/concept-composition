# CLEVR scaling plan: how many objects survive as experts are added

This experiment tests how far composition holds as more objects are asked for, for both papers. Paper 1 adds experts that were
trained apart. Paper 2 uses one network composed with itself, plus our checker. Session 5 writes each run's design from this page
and launches nothing for CLEVR until [the fairness audit](fairness-audit.md) has covered it.

![One CLEVR experiment for both papers](one-clevr-experiment-for-both-papers.png)

## What each method does

| Method | Models trained | How it makes N objects | Paper |
|---|---|---|---|
| Control | One location network and one empty-background network | Bradley's rule: add the scores, asking the same network about each position | Rival |
| One network with checker | The same two networks | The control's composition plus our checker | Paper 2 |
| Experts trained apart | One expert per position and the same background network | Compose N experts, with our checker once step D3 has chosen it | Paper 1 |

## The order of the runs

**Data rungs.** Each stage climbs from the cheapest data that can answer it. The first rung is Du's public 64-pixel cube dataset
(6.2 GB, not yet opened, so its contents must be checked). The second is single-object CLEVR at 64 pixels from the public
[CLEVR generator](https://github.com/facebookresearch/clevr-dataset-gen). The third is 128 pixels, Bradley's size, and only a result
there can beat a printed number.

**Stage 1: the expert ladder.** Train the background, the one-network control and experts 1 and 2, then test 2 objects. Add one
expert per rung and test one more object. Earlier experts are never retrained.

**Stage 2: two experts holding several objects.** Model A sees only red objects and model B only blue, each 1, 2 or 3 at a time and
anywhere in the frame. The 6 models pair into 9 scenes of 2 to 6 objects. Success is x red and y blue objects, all separate. Giving
each model its own half of the image is not allowed, because that is two pictures pasted side by side.

**Stage 3: inclusive labels.** A separate table on models trained with 1 to 3 objects, with every object labelled.

## The rules, fixed before the first run

1. Each new expert must pass alone, drawing its one object in the right place, so a weak expert is not read as a scaling limit.
2. The order of positions is written down before the first run.
3. The ladder stops after two failed object counts in a row, or at 10 objects.
4. At the first failed count, every smaller group of the trained experts is tested, to separate the number of experts from one bad
   position.
5. In the blue tables, no model and no checker sees a real image holding two objects. The checker may use single-object images and
   composites pasted from them. A checker that sees real multi-object images is judged in the Stage 3 table.

## How every method is scored

Every method runs on the same positions and seeds, with 1,024 images per object count, and reports three things.

**Most objects that succeed.** A count K succeeds when K objects appear in at least 50% of images and at least K-2 appear in at
least 90%. This is Bradley's rule.

**Images with all N objects in place.** This is the count behind Projective Composition's 98 of 100 at 5 objects. A person also
checks 100 images per count by eye, so that number sits directly beside ours.

**Picture quality.** A count with the objects in place but a degraded image does not pass.

## The numbers to compare with

| Table | Printed result | What we need |
|---|---|---|
| One object seen, one network | 4 for Bradley's rule, 5 for the local patch model | **6 objects at 128 pixels** |
| Experts trained apart | No rival prints this | A new table, compared with the control on the same data |
| Two experts holding several objects | No direct rival. Bradley's colour model reaches 4 after training on 1 to 3 objects, on a different task | A new table |
| Trained on 1 to 3 objects | 7 for labelling every object, 9 for the local patch model | 10 objects, which is also where Bradley's counting classifier stops |

The two Bradley papers disagree on the 1-to-3 result. Projective Composition credits the learned model with 9, while the Local
Mechanisms paper prints 7 for that model and 9 only for its local patch models. This page uses the Local Mechanisms numbers.

## What it costs

All costs are estimates from our last CLEVR training of about 7.5 GPU-hours per model at 8 million images seen, and the first rung
measures the real ones. Rendering cost is unmeasured.

| Stage | Models | GPU-hours at a quarter or the full training length |
|---|---|---|
| Ladder, up to 2 experts | 4 | 7.5 or 30 |
| Ladder, up to 6 experts | 8 | 15 or 60 |
| Ladder, up to 10 experts | 12 | 22.5 or 90 |
| Two experts holding several objects | 6 | 11 or 45 |
| Inclusive labels | 1 per training set | about 2 or 8 each |
