# Where Loop 12 ends: what it found about borrowing, and what the two papers take from it

Loop 12 asked where, with CO3 on SDXL, one subject borrows the other's parts, and what stops it. On SDXL it found no reading that predicts which pairs borrow and no fix that passes its bar; the nearest is each animal's own prediction inside its own box (Z7, a lead of 6 two-clear pictures in 48, p = 0.073). Its two later steps turned the same question on the two papers' own methods: the LoRA from Paper 1 separates bodies on new pairs but does not make them the right animals (Z10), and on Colored MNIST a checker trained to push along the experts' own direction reaches combinations no expert saw, where naming the digit and the colour separately does not reach as far (Z11).

<a href="loop-12-progress.png"><img src="loop-12-progress.png" width="620" alt="Loop 12's steps Z1 to Z11 with their verdicts, and the binding plan as a detour into the LoRA project"></a>

## Every step and its verdict

| Step | Question | Verdict against the bar written before it ran |
|---|---|---|
| Z1 | which papers say where leakage enters | 13 papers in [the mechanism table](01-reading/mechanism-table.md); none measures which pairs leak and why |
| Z2 | does any reading inside SDXL predict which pairs borrow | null on all seven tests: largest Spearman ρ 0.489 against a bar of 0.706 ([Z2's review](02-what-predicts-borrowing/review.md)) |
| Z3 | does subtracting the other animal's prediction in each box stop borrowing | null: 5 of 48 two clear, the same as CO3 alone ([Z3's review](03-one-fix/review.md)) |
| Z4 | Paper 2's method | [the method choice](04-paper-2-method.md) |
| Z5 | design issues in Loops 8 to 10 | [the design audit](05-design-audit.md), with [rerun 3](05-audit-rerun-3/) read from saved records |
| Z6 | the loop's verdict on mechanism and fix | [the verdict](06-verdict.md): mechanism unknown, three places ruled out as predictors |
| Z7 | does each animal's own prediction in its box stop borrowing | null by the bar, the nearest: 14 of 48 against 8, p = 0.073, mostly duck × goose ([Z7's review](07-own-prediction/review.md)) |
| Z8 | does the kept early push fix the failures of three samplers | parked: it passed on CO3 only, and broke good CO3 draws ([Z8's review](08-first-correction-test/review.md)) |
| Z9 | does Paper 1's checker push compose unseen digit-colour pairs | yes, 46.9% to 76.0%, but a push naming digit and colour separately does as well (78.4%), so the lift is attribute guidance ([Z9's review](09-dcfm-colored-mnist/review.md)) |
| Z10 | is the LoRA's correction a dial or a switch on new pairs | no effect on two clear (0 to 2 of 32); it separates bodies (hybrids 29 to 4) without naming them ([Z10's review](10-co-occurrence-dial/review.md)) |
| Z11 | does a checker trained to push along the experts' direction reach unseen pairs | **support**: 88.8% against 63.1% (no penalty) and 73.8% (digit and colour named separately), FID 10.8 against 17.0 ([Z11's review](11-gradient-trained-classifier/review.md)) |

## What Paper 1 takes from it (the LoRA correction)

The LoRA does one of the two jobs composition needs. Raising its weight separates a hybrid into two bodies on pairs it never trained on, but the bodies are rarely the two animals asked for: a fox-llama hybrid becomes two llamas. Its training target, the joint prompt's prediction minus the plain product's, is mostly "two things instead of one", so binding each body to its word is a weak signal inside it. [The binding plan](../../../../1-lora-correction/binding-plan.md) tests first whether each animal's own prediction in its box completes the job (option A, board question 113, running as Slurm job 95479), and only then trains a binding loss on regions from Grounded-SAM, the recipe TokenCompose (CVPR 2024) used.

Z10 also found that with this checkpoint attached and switched off, SDXL does not draw the plain product (a mean of 10.8 grey levels away), so any earlier weighted run that took "off" from the same U-Net compared against about a third of the adapter. Which recorded results did so is not checked.

## What Paper 2 takes from it (classifier guidance)

Z9 showed that a checker told the pair does no better than a push told the digit and the colour separately; Z11 showed that training the checker so its push points along the experts' own direction does better than both, on combinations no expert saw. Where Z11's checker fails, it draws a seen pair that shares the asked colour (a red 6 for a red 0), the same kind of failure as the LoRA's two llamas. The target Z11 trained toward exists only because two holders each saw half the pairs; whether the same penalty can be built on SDXL, where no expert has seen the pair, needs its own design.

## Still open

- Whether option A passes: the LoRA's separation plus each animal's own prediction in its box, on Z10's eight pairs. Its boxes are drawn by eye, so it decides only whether to go on; a version with detector boxes is owed before anything from it is cited.
- Whether Z7's lead holds on more seeds (seeds 6009 to 6016, about 1.7 GPU-hours, the owner's call).
- Which earlier results used the LoRA's leaky off switch.
- Whether Z11's penalty has an SDXL form.

## Compute

Z3 about 1.6 GPU-hours, Z9 3.70 (0.4 over its approval), Z10 2.94 (0.11 over), on `bigbatch` RTX 3090 cards; Z11 1 h 30 min on a RunPod RTX 4090, plus about 2 hours idle after its first launch stopped. Z2, Z7 and Z8's compute is in their reviews.
