# The gradient-trained classifier on CO3

The owner chose to take the classifier trained so its push points the right way from coloured digits to
CO3 on SDXL, after the reviewer reads this. It is a proposal, not a result.

## Why

On coloured digits, a checker trained with a penalty that pulls its push toward a target direction got
88.8% of the 50 unseen digit-colour combinations right, against 73.8% for a push that only names the
digit and colour at the same length, 63.1% for the same checker trained without the penalty and 47.0%
for a random push; its push lined up with the target at 0.514 on held-back pairs against 0.064 untrained
(`leakage/11-gradient-trained-classifier/`, review and gate). On SDXL, the kept checker's push lined up
with the joint-minus-product term at only +0.105 on CO3 and could not split hybrids or fix borrowed parts
reliably (`method-search/09-gradient-direction-check/review.md`, corrected; `leakage/08-first-correction-test/review.md`).

## The proposal, as the reviewer named it

Train a checker on CO3's noisy states whose push is pulled toward a target direction taken from each
animal's own prediction inside its box (the near miss: 14 two-clear pictures of 48 against CO3's 8,
`leakage/07-own-prediction/review.md`), so the trained push supplies "which animal is where" without
needing boxes at sampling time. CO3 borrows parts on the same pairs again and again (cheetah × cougar on
7 of 8 seeds), and its two words already attend to separate regions, so the failure sits in what each
region draws (`method-search/11-a-score-that-names-no-subject/review.md`).

## Open questions for the reviewer

1. The target: is "each animal's own prediction in its box minus CO3's prediction" the right direction,
   given the boxes came from the finished picture (a best case), and how should it be computed at the
   noise levels the push acts on?
2. Training data and labels: which CO3 trajectories train the checker (the 192 blind-labelled ones, the
   12 tuning pairs at seeds 6001 to 6016), how pairs are held out, and how to avoid the pair-only
   shortcut that scored 0.757.
3. The gate and bar, written first: held-back pairs, the comparisons (CO3 alone, the same checker with no
   penalty, a random push at the same length, the per-animal box correction itself as the ceiling), and
   the count of already-right pictures broken.
4. Card type and cost: the CO3 rows the project already has were drawn on cluster RTX 3090s; RunPod has
   no 3090s, so either every row is redrawn on RunPod 4090s or the test stays on the cluster.
