# Verdict: where CO3's subjects borrow each other's parts, and what stops it (step Z6)

**CO3 draws two bodies and one borrows the other's parts, and this loop did not find where that starts or a fix that stops it.** Three readings inside SDXL, taken across the whole sampling path, do not predict which pairs borrow; a correction late in sampling that subtracts the other animal's prediction inside each animal's hand-drawn region leaves the count of clean pictures where CO3 alone has it. Paper 2's next test puts each animal's own prediction inside its own region, which Z3 never did.

## The mechanism

**Unknown, and three places are ruled out as predictors.** Over CO3's 12 tuning pairs, the similarity of the two nouns in the text encoder, the overlap of their attention maps and the similarity of their pushes on the picture, each read early (steps 0 to 5), in the middle (10 to 20) and late (25 to 45), give a largest Spearman $\rho$ of 0.489 against the borrowed counts, under the bar of 0.706 ([Z2's review](02-what-predicts-borrowing/review.md), W&B `llct9xp0`). Crow × raven reads highest on every reading and never borrows; puma × condor reads low and borrows on 7 of 8 seeds. The readings follow how alike two animals are, and borrowing does not.

**What the recorded cases show (one seed each, inferred).** On cheetah × cougar at seed 6001, plain SDXL draws two cheetahs; CO3 pushes away from the cheetah at steps 4 and 5 and finishes a cougar-shaped animal with cheetah spots ([the capture note](co3-inside/design.md)). CO3's own signal, read cell by cell, marks both bodies as "cheetah" at step 5 ([the free check](03-one-fix/free-check/README.md)). The borrowing there looks like an unfinished change of animal.

## The fix against its bar

**Null.** On the six borrowing pairs (48 pictures per row), CO3 with one corrector pass per step at steps 15 to 30, which inside each animal's drawn box takes twice the joint prompt's guess minus the other animal's guess (the animal's own prediction gets weight 0), gives 5 pictures with two clear subjects that keep their own parts; CO3's whole-picture weights in the same window give 5, and CO3 alone 5. The bar needed a lead of at least 4 with an exact McNemar p below 0.05; the lead is 0 (p = 0.66). Two blind reviewers agreed on 128 of 144 rungs, and a third settled 16 ([Z3's review](03-one-fix/review.md), W&B `xql3iwpp`). On cheetah × cougar the correction takes the borrowed spots off the cougar's body and the cheetah's tear lines stay on its face, so the pictures still count as borrowed.

## The method chosen

**Each animal's own prediction inside its own region, tested next; the text-side fix held behind a free check; Bounded Attention the rival** ([Paper 2's method](04-paper-2-method.md), [the method review](method-review.md)). Z3's correction gave each animal's own prediction weight 0 inside its box, so whether the cougar's own prediction draws a clean cougar where the cougar is has not been tested; that test costs about 0.8 GPU-hours on the 48 pictures. The text-side swap cannot reach borrowing by the first-named animal (boar × pheasant) and runs only if a free text-only check supports it. Per-cell CO3, regions read from $d_t$ and the checker choosing CO3's settings are closed by Z2, Z3 and board question 80. The identity reward LoRA is held, because it trains the diffusion model.

## The reruns proposed for Loops 8, 9 and 10

[The design audit](05-design-audit.md) finds 30 design issues: 3 could change a recorded verdict, 16 possibly could and 11 could not. Board question 97, decided by the manager loop for the owner, runs the first rerun in Loop 8 (where the digit sampler settles its digits, about 0.04 GPU-hours), holds the second, and reads the third from Loop 12's saved CO3 records at no GPU cost: [the push-region read](05-audit-rerun-3/README.md) finds each animal's push only partly on its own body at CO3's corrected steps.

## What it suggests for Paper 1

Paper 1 composes two experts trained apart, each drawing its own region, so neither expert's prediction can carry the other subject's features. This loop cannot yet say whether that protects Paper 1 from borrowing. Z3's correction never put an animal's own prediction inside its box (inside each box it used twice the joint prompt's guess minus the other animal's), so it does not show whether a single model's prediction for "cougar" carries cheetah features. A test that uses each animal's own prediction inside its box would answer that, and with it whether Paper 1's experts are protected by construction (inferred).

## Compute

About 3.8 GPU-hours: 1.8 for Z2's four rungs, 0.3 for the CO3 capture run, 1.7 for Z3's three rungs. No API tokens.
