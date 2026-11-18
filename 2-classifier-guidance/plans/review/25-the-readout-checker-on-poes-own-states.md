# Review: the readout checker trained on PoE's own early states

The questions and their pass marks for [plan 25](../25-the-readout-checker-on-poes-own-states.md), written on
2026-11-18 before any run was labelled, trained or pushed.

## Words this file uses

**Run.** One cat × dog drawing from one seed's starting noise, either by plain PoE ("a cat" × "a dog") or by the
joint prompt "a cat and a dog", with its states at sampler steps 0 to 20 and every fifth step after saved.

**Label.** How a run finished, called by the CLIP ViT-L/14 checker fine-tuned on finished pictures (0.900
held-out balanced accuracy): clean above a logit of +2, reject below −2, left out in between. Every saved state
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

- Support: "two clear" on at least 2 more seeds than plain PoE and than the size-matched random push, at some
  strength, and at least as many as the identity push (iteration 6) at that strength.
- Null: no more "two clear" seeds than iteration 6.
- Style check, read beside the count: a seed that turns from a drawing into a photograph is noted on its call;
  a support that rests only on such seeds is called inconclusive.

## Runs

| Run | What | Status |
| --- | --- | --- |
| trajectories | 1,200 seeds × 2 runs on mscluster111 and 112 | drawing |
| labels | `label_trajectories.py`, counts per split and mode | not started |
| baseline reading | iteration 4 scored on the held-out runs | not started |
| readout-catdog-traj | iteration 4's settings plus `--traj-frac 0.5` | not started |
| push sheet | `render_results.py --readout`, seeds 9 to 16 | not started |
