# Experts trained apart: do the step 7b pushes compose them

A follow-on to [A2b](24-mnist-one-model.md), judged by [plan 32's review file](../../plans/review/32-two-digit-experts-that-never-met.md).
The card with the counts and sheets is [the results card](../../artifacts/results/32-mnist-two-experts/README.md).

## Outcomes

- [x] 📝 the bars written before the run, in the review file, 2026-11-19 `verified`
- [x] ✅ the experts alone: 9 of 10 digits at or above 58 of 64; the 5–9 expert's 8 (55) and, fixed-side, its 7 (57) miss `verified`
- [x] ❌ check failed: the fixed-side control, where the argument says the experts compose exactly: 139 of 320 at the best base, bar 256 `verified`
- [x] ❌ check failed: both pushes on the side-mixed experts: 4 of 320, bar 224, within 32 of plain product (0) `verified`
- [x] ✅ check passed: the pushes do no harm on the fixed-side experts: 139 of 320 without, 168 with `verified`

## The result

**They do not compose.** Two digit experts trained apart, a 0–4 expert and a 5–9 expert, give two clear, right digits on
0 of 320 seeds when each draws its digit on either side. Both pushes give 4, a random push 2, the one-half ink push 0,
and four settings of strength, cap and randomness gave 0 on the two tuning pairs. The canvases are single saturated
blobs: the product is destroyed before a push acts, and a push can only reweight what the product still has. When
each expert always draws on its own fixed side, the product keeps some mass: 139 of 320 compose with no push and
both pushes lift it to 168, still short of the 224 that would support the claim. What was not tried is a base the
two experts share.
