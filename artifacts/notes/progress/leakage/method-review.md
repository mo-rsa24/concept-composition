# Method review after two nulls (Loop 12)

The manager loop asked for this review on 2026-11-24 after Z2 and Z3 both came back null, with new runs paused until it is written. It follows the reviewer's second pass on Loop 12 (`reviewer/2026-11-24-review.md`, "Loop 12 again, at 23:00"). Any change of direction it suggests is the owner's call.

## What the two nulls share

**Neither step ever used each animal's own prediction where that animal's body is.** Z2 read three things built from the pair (the two nouns' text similarity, the overlap of their attention maps, the cosine of their pushes) and set them against the borrowed counts across pairs ([Z2's review](02-what-predicts-borrowing/review.md)). Z3's correction, inside each animal's drawn box, took twice the joint prompt's guess minus the other animal's guess, and gave the animal's own prediction weight 0 (`2-classifier-guidance/scripts/leakage/z3_regions.py` lines 66 to 72; [Z3's review](03-one-fix/review.md)). Both nulls therefore leave open the test the reviewer names: whether the cougar's own prediction, placed where the cougar is, draws a clean cougar.

**Z3's window was chosen from advice the recorded seed already contradicts.** The correction acted at steps 15 to 30. On cheetah × cougar at seed 6001 the joint prompt's prediction reads "cheetah" on both bodies by steps 4 and 5 ([the free check](03-one-fix/free-check/README.md)), and the reviewer now says its own advice to wait until step 15 was partly wrong.

**The measure is all-or-nothing.** A picture counts only when both subjects are clear with their own parts and no animal is extra or missing. On cheetah × cougar the drawn-box row took most borrowed spots off the cougar's body and left the cheetah's tear lines on its face, and every such picture still counts as borrowed.

## What the papers already read by Loop 9 say

**Remove only the other animal's component.** Concept Algebra treats concepts as subspaces of the score and edits a prediction by projecting onto a concept's subspace, estimated from prompts that differ only on that concept (p. 7; [Loop 9's note](../noise-text-critic/01-reading/concept-algebra.md)). Inside the cougar's box, subtracting only the projection of the joint prediction onto the cheetah's direction would remove cheetah features and keep the rest of the joint prediction, where Z3 subtracted the cheetah's whole prediction (inferred).

**Each region's own condition is what local composition needs.** Local Mechanisms of Compositional Generalization proves that a conditional distribution factorizes over disjoint sets of pixels exactly when the score at each pixel depends only on its own set and that set's condition (p. 6; [Loop 9's note](../noise-text-critic/01-reading/local-compositional-generalization.md)). Putting each animal's own prediction inside its own box is that condition made literal; Z3 never imposed it.

**The text side cannot be fixed from the second word alone.** Linear Spaces of Meanings proves embeddings add up exactly when the factors are conditionally independent given the image (Lemma 2, p. 3; [Loop 9's note](../noise-text-critic/01-reading/linear-spaces-of-meanings.md)). In this project's labels the first-named animal borrows on boar × pheasant (the boar gets bird claws and a feathered tail, W2d's notes at seeds 6005 to 6008), which a causal text encoder cannot reach by editing the second word; and the end-of-text, padding and pooled vectors still carry both animals after a swap.

**A partial fix needs a per-subject measure beside the bar.** Mechanisms of Projective Composition defines a correct composition as one that matches each expert when viewed through that expert's own features (Definition 4.1, p. 5; [Loop 9's note](../noise-text-critic/01-reading/projective-composition.md)). Scoring each subject's parts separately, from the per-part notes the blind calls already hold, would show a partial fix that the all-or-nothing count hides. This is a report beside the bar; the bar stays as the owner set it on board question 92.

## What other loops' results already explain

**Loop 6's M9 put each subject's own prediction in its own region and drew no pair.** On the plain product, from step 0 with the right boxes, every region row drew one animal or a fused one; with bootstrapping it drew 8 clear pairs of 8, every picture damaged; its automatic box reader failed 11 of 24 reads at step 10 ([M9's review](../method-search/04-bind-each-body/review.md)). That was from step 0, before any layout. On CO3, the layout of two bodies is already set by step 5, so own predictions inside drawn boxes after step 5 meet a different situation, and that has not run (inferred).

**The recorded seed shows each animal's push only partly on its own body at CO3's corrected steps.** At step 5 on cheetah × cougar, 62% of the cheetah's push mask lies inside the cheetah's box and 42% of the cougar's inside the cougar's ([the audit's rerun 3](05-audit-rerun-3/README.md)). Prompted alone from noise, "cougar" draws a cougar with no spots (the capture run's `plain-b.png`).

## Whether each step is still worth its cost

| Test | What it would show | GPU-hours | Worth it? |
|---|---|---|---|
| A free text-only check | how far each noun's vector inside the joint prompt moves from its encoding alone, per pair, set against the borrowed counts; no picture | 0 | yes, first |
| Each animal's own prediction inside its drawn box | the positive control Z3 meant to be: inside each box the animal's own guess replaces twice the joint minus the other, at steps 5 to 30 (after CO3's own correction), beside CO3 alone on the 48 pictures | about 0.8 | yes: it is the direct test of whether borrowing lives in the joint prompt's prediction |
| The same with only the other animal's component removed | Concept Algebra's projection inside each box | about 0.8 | after the test above, if that test damages pictures |
| The text-side swap, with a positive control | 8 plain-SDXL seeds with the swapped embedding first, then CO3 with the swap on the 48 pictures, beside a word-order check | about 1.5 | only after the free check; it cannot reach first-named borrowing |
| Bounded Attention as the rival | the rival's upper bound with the drawn boxes | about 2 | when a candidate clears the bar |

## What I would change in the design

1. **Restate Z3 as what it tested**, done in [Z3's review](03-one-fix/review.md), [Z3's design](03-one-fix/design.md) and [the verdict](06-verdict.md).
2. **Run the own-prediction test next**, at steps 5 to 30, with the same boxes, blind calls and bar, and a per-subject part score reported beside the bar.
3. **Hold the text-side test** until the free text-only check says the second noun's vector moves more for the borrowing pairs, and add its plain-SDXL positive control.

The text-side test at about 3.4 GPU-hours is over the 3-hour line, so it goes to the owner after this review; the own-prediction test at about 0.8 is inside it.
