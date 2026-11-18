# Review: the readout checker trained on PoE's own early states

The questions and their pass marks for [plan 25](../25-the-readout-checker-on-poes-own-states.md), written on
2026-11-18 before any run was labelled, trained or pushed.

## Words this file uses

**Run.** One cat × dog drawing from one seed's starting noise, either by plain PoE ("a cat" × "a dog") or by the
joint prompt "a cat and a dog", with its states at sampler steps 0 to 20 and every fifth step after saved.

**Label.** How a run finished, called by the CLIP ViT-L/14 checker fine-tuned on finished pictures (0.900
held-out [balanced accuracy](../../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/textbook/06-readout-guidance-on-one-small-network.md#balanced-accuracy-the-hit-rate-on-each-class-averaged)): clean above a logit of +2, reject below −2, left out in between. Every saved state
of the run carries it.

**Reading check.** Balanced accuracy, clean against reject, on the saved states of the held-out joint runs
(seeds 3001 to 3200), at steps 0, 2, 4, 6, 8 and 10, averaged. Joint runs only, so a checker that learnt
"joint prompt against PoE" instead of the picture's content scores 0.50 here.

## Questions

**Q1. Does training on PoE's own states teach the readout checker to read the early steps?**

- Support: the reading check reaches 0.75 or more and beats iteration 4's reading of the same states
  (`readout-catdog-more3-long/head_step_002000.pt`, scored with `--score --traj-labels`) by at least 0.05.
- Null: within 0.05 of iteration 4.
- Inconclusive: fewer than 20 clean or 20 reject held-out joint runs survive the labelling margin.

**Q2. Does its push make more clean cat-and-dog pictures?**

On held-out seeds 9 to 16 at strengths 10, 30 and 100, called from full-size tiles on the ladder single <
hybrid < two bodies or heads < two clear (a clean cat and a clean dog):

- Support: "two clear" on at least 2 more seeds than plain PoE and than the [size-matched random push](a3q-cat-x-dog-quick-discriminator-test.md#words-this-file-uses), at some
  strength, and at least as many as the identity push (iteration 6) at that strength.
- Null: no more "two clear" seeds than iteration 6.
- Style check, read beside the count: a seed that turns from a drawing into a photograph is noted on its call;
  a support that rests only on such seeds is called inconclusive.

**Q3. Do 1,000 more joint-prompt runs change the reading?**

Written before the extra runs were drawn. The same training (`readout-catdog-traj`'s settings) with 1,000 more joint-prompt
runs (seeds 3401 to 4400, labelled the same way) added, read with the same check on the same held-out joint runs:

- Support for scaling further: the mean over steps 0 to 10 rises by 0.02 or more over `readout-catdog-traj` at step 2,000.
- Null: within 0.02; more runs of this kind are not what the reading lacks.

## Runs

| Run | What | Status |
| --- | --- | --- |
| trajectories | 1,200 seeds × 2 runs on mscluster111 and 112 | done: 2,400 runs, all 48 blocks |
| labels | `label_trajectories.py`, counts per split and mode | done: training joint 508 clean, 464 reject, 28 left out; training PoE 0 clean, 1,000 reject; held-out joint 105 clean, 90 reject, 5 left out; held-out PoE 0 clean, 200 reject. Spot check: 16 of 16 clean calls right, at most 1 doubtful reject in 16 (seed 2749) |
| baseline reading | iteration 4 scored on the held-out runs | done: held-out joint runs (105 clean, 90 reject) at steps 0, 2, 4, 6, 8, 10: 0.510, 0.548, 0.644, 0.747, 0.794, 0.817, mean 0.677; steps 15 and 20 read 0.843 and 0.848. Q1 support needs 0.75 or more over steps 0 to 10 |
| readout-catdog-traj | iteration 4's settings plus `--traj-frac 0.5` | done: mean over steps 0 to 10 0.730 at step 2,000 (0.671, 0.740, 0.725 at steps 500, 1,000, 1,500); the older noised-picture check 1 of 20 steps at 0.75 |
| push sheet | `render_results.py --readout`, seeds 9 to 16 | queued after training |
| plus 1,000 | 1,000 joint runs, seeds 3401 to 4400, then `readout-catdog-traj-plus1000` | drawing |

## Cross-references

- The mention of **the reading check** in [plan 25, the readout checker on PoE's own early states, quick context](../25-the-readout-checker-on-poes-own-states.md#quick-context).

## Answers

**Q1: in between.** The reading check on held-out joint runs at step 2,000: 0.591, 0.647, 0.732, 0.774, 0.803, 0.835 at steps 0
to 10, mean 0.730, against 0.677 for iteration 4. It clears the 0.05 margin (+0.053) and misses the 0.75 bar. Training on
PoE's own states traded reading those states for reading finished pictures with noise added: the older check passes 1 of
20 steps.
