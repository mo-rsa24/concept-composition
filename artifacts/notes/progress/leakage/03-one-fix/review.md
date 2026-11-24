# Review: does correcting each animal's region later in sampling stop the borrowing? (step Z3)

**No.** With one box per animal drawn by eye on CO3's finished pictures, the drawn-box correction at steps 15 to 30, which inside each animal's box takes twice the joint prompt's guess minus the other animal's guess and gives the animal's own prediction weight 0, gives 5 pictures of 48 with two clear subjects that keep their own parts, the same as CO3 alone (5) and as the whole-picture correction in the same window (5). The bar in [the design](design.md) needed a lead of at least 4 with an exact one-sided McNemar p below 0.05; the lead is 0 (3 seeds gained, 3 lost, p = 0.66). **Null.** This was the positive control: with the regions known, this correction does not remove borrowing as the blind calls count it, so step 3 (a reader that finds the regions) does not run. Each animal's own prediction never entered its box, so the test says nothing about a rule that uses it.

## Runs

| Rung | Slurm job | W&B | Pictures | Outcome |
|---|---|---|---|---|
| [Smoke](smoke/README.md) and both controls | 94635 | `p3ydtazf` | 2 | empty boxes give the whole-picture row's picture pixel for pixel; the latent entering step 15 is unchanged (0.0) |
| [Small](small/README.md) | 94647 | `x1gpc3pz` | 32 | looked at, not counted |
| Full | 94678 | `xql3iwpp` | 64 more | 96 corrected pictures in all |

Code snapshot `f4bb39be`. CO3 alone is Z2's one-card full run (Slurm job 94591). Median 49 to 56 seconds per picture; about 1.6 GPU-hours. The boxes are in [the boxes file](boxes.json) with a contact sheet per pair, drawn before any fix render; 38 of 48 pictures hold one of each animal and 10 do not.

## The count

Two blind reviewers called all 144 pictures on [Loop 6's rubric](../../method-search/01-beat-the-baseline/blind-count-rubric.md) with drawing style ignored, under neutral ids, with the key held apart (the measure decided on board question 92). They gave the same rung on 128 of 144 pictures (89%) and split on `two_clear` for 16; a third blind call settled the 16 (14 `two_bodies`, 2 `two_clear`). The calls, the key and the scored result are in [the blind folder](blind/), scored by `2-classifier-guidance/scripts/leakage/z3_blind.py`.

| Row | Two clear subjects, of 48 | Broken anatomy, of 48 |
|---|---|---|
| CO3 alone | 5 | 1 |
| Whole-picture correction, steps 15 to 30 | 5 | 2 |
| Drawn-box correction, steps 15 to 30 | 5 | 4 |

Two clear subjects per pair, of 8:

| Pair | CO3 alone | Whole-picture | Drawn-box |
|---|---|---|---|
| cheetah × cougar | 0 | 1 | 1 |
| duck × goose | 0 | 0 | 0 |
| puma × condor | 1 | 1 | 0 |
| boar × pheasant | 0 | 0 | 1 |
| bobcat × quail | 2 | 1 | 1 |
| seal × gull | 2 | 2 | 2 |

Matched seeds: the drawn-box row against CO3 alone gained 3 and lost 3 (p = 0.66); the whole-picture row against CO3 alone gained 2 and lost 2 (p = 0.69); the drawn-box row against the whole-picture row gained 1 and lost 1 (p = 0.75). The broken-anatomy guard is not reached (4 against 1 plus 3).

## What the notes show, without a bar

**On cheetah × cougar the borrowing moves from the body to the face.** The reviewers' notes on that pair's 8 pictures per row mention cheetah spots on the cougar 6 and 7 times for CO3 alone, 3 and 4 for the whole-picture row, and 2 and 2 for the drawn-box row (reviewer 1 and reviewer 2). They mention cheetah tear lines on the cougar's face 2 and 3, 3 and 3, and 6 and 5 times. The drawn-box correction takes most borrowed spots off the cougar's body, which the [small read's sheet](small/README.md) shows by eye, and the pictures still count as borrowed because the face keeps the cheetah's tear lines. This tally counts words in notes written for other purposes, so it carries no bar.

**On duck × goose nothing changes.** All 24 pictures keep mallard heads on goose-shaped bodies; none is `two_clear` in any row.

## What this means for Paper 2

The positive control failed on the measure that counts: a correction that knows exactly where each animal is, applied at steps 15 to 30, does not raise the number of pictures with two subjects that keep their own parts. Three readings are open, and the owner and Z4 choose among them (inferred):

- The borrowed parts that survive (tear lines on a face, a mallard head on a goose) are set before step 15 or inside the box the correction leaves to its own animal; a window that reaches earlier, or a rule that also corrects the face, would test that.
- One pass per step is too weak; on cheetah × cougar it moved the spots and not the face.
- The joint prompt's prediction carries the cheetah's features on the cougar's face as well as its body, and subtracting the cheetah's prediction removes only part of them. A rule that puts each animal's own prediction inside its box, which this test never used, would test whether the cougar's own prediction draws a clean cougar there.

## What this cannot say

Six pairs and eight seeds are tuning data. The boxes are axis-aligned and include parts of the other animal where the two overlap. The count is binary per picture, so a picture that loses most borrowed parts and keeps one still counts as borrowed. The rule acts with one pass per step at steps 15 to 30 only.
