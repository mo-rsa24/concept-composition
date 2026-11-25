# Review: pair-aware or separate labels for the checker? (step M2)

M2 did not run.

## Closed: the checker route for Paper 2

The owner closed the checker route for Paper 2 on 2026-11-25, answering board question 111 with A. Three findings close it.

- **No reader of this kind sees the picture beyond its pair.** A lookup that knows only which animal pair a picture belongs to, and nothing
  about the picture, scores 0.757 balanced accuracy on the fresh seeds ([the reviewer's pass of 2026-11-25](../../reviewer/2026-11-24-review.md)).
  The first per-place reader read 0.688 on the tuning seeds and 0.632 on the fresh ones, and the size-corrected reader 0.635, both below that
  lookup; scored only between pictures of the same pair, the per-place reader is at chance (0.39 to 0.51).
- **The kept checker's push points against the term composition drops.** Its median cosine with r_t = ε(joint) − (ε(A) + ε(B) − ε(empty))
  is −0.105 at step 0 on CO3 and negative at every step from 0 to 15 on plain PoE ([M13's review](../09-gradient-direction-check/review.md)).
- **The trained checkers do not see CO3's failure.** The kept checker and M5's two retrained ones reach at best 0.602, 0.617 and 0.648, under
  the 0.65 floor ([M5's review](../01b-balance-the-checker-data/review.md)).

The 192 blind-labelled pictures (96 tuning, 96 fresh) stay where they are for later use.
