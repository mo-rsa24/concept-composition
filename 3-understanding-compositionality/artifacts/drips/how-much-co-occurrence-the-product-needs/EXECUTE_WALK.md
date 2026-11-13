# Walk: how much co-occurrence does the product need?

Mode `--request`. Plan: [the walk's plan](../../../plans/10-what-poe-loses-when-it-composes/today-how-much-co-occurrence-the-product-needs.md).
Branch `exec/how-much-co-occurrence`.

## Position

Piece 1 of 5 ticked. Opening piece 2, the sweep and the closed-form crossing.

## The stop point

**Bar.** The card states the threshold weight as a number and the competing-caption counts out of 300,
judged by the user from the sweep figure.

**Time box.** One sitting, about 2 hours. Sitting 1 opened 2026-11-04 07:02.

## The toy

The three circle positions at x = 6, 16, 26 and the three squares at the same, 16 by 32 RGB, σ = 0.1;
the pair image with its circle at x = 10 and its square at x = 22.

## Decisions

| # | Decision | Chosen | Options there were | Who | Why |
|---|---|---|---|---|---|
| 1 | the figure's condition-B mark | at 0.1, B's real pair weight, with 0.5 labelled as the sweep's top | at 0.5 as the request wrote it; both marked | default accepted | the code gives the pair 0.1, so a mark at 0.5 labelled B would be false on the figure's face |
| 2 | the competing-caption forms | build both and show the two count sets | derive both, build only the picked one | default accepted | it is the stop the request asked for, and the second form is cheap once the first runs |
| 3 | the closed-form crossing | computed beside the grid threshold | grid only | default accepted | no compute cost, and it saves the answer from being a grid floor |
| 4 | where the plan lives | `3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/` | a new scope | default accepted | the note already belongs to that scope |
| 5 | sample images every piece | yes, as a clickable link and a path in chat | only at the figure stop | user asked | the user corroborates the counts against the pictures rather than taking them |

## Ticked

| Piece | Evidence | Commit |
|---|---|---|
| 1, condition D with w as a flag | D at w = 0 is byte-identical to condition A (exact 300 hybrids, reverse diffusion 2 faint + 298 hybrids, MALA 300 hybrids, acceptance 0.37659). D at w = 0.1 is B's density exactly, verified on the means and the weights | `10025b16` |

## What the walk learned

**The two sampled curves carry real draw noise.** D at w = 0.1 and condition B are the same density, and
the exact sampler agrees at 300 of 300, but reverse diffusion returned 10 crisp against B's 14 and
annealed MALA returned 30 against B's 14. So a sampled crisp count of 14 and one of 30 can come from one
density at one weight, and the sweep's two sampled curves must be read as noisy to about that width. The
exact curve is the one the threshold is read off.

## Runs, threads and handoffs

None out.

## Closing pass still owed

Everything: the card, the idea-map line, the note's README table, `/verify-plan`, `/xref-pillar`,
`/sync-plan-tree`, the cadence dispatch, the retrospective.
