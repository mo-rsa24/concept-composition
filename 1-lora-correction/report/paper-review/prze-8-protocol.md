# PRZE-8: the comparison protocol

Written before any run so no comparison can be called unfair afterwards.

## Outcomes

- [x] 📝 markdown created: one protocol page stating (1) identical starting noise, steps, guidance weight and resolution per model, (2) each baseline at its published defaults plus the same small tuning grid run on a tuning set that excludes every evaluation pair, (3) the rule that picks the images shown, fixed before any result, (4) network evaluations per image as a column, (5) every comparison where more than one thing differs marked mixed ✓ verified 2026-11-10: [the protocol page](protocol.md), rules 2, 3, 5, 6, 7
- [x] 📝 markdown created: one adapter checkpoint named and frozen before any held-out result is read, chosen on a tuning set that excludes every evaluation pair, and used for every adapter number and image in the paper (the current Figure 5 mixes three adapters) Open: the procedure that names it is fixed in [rule 1](protocol.md#what-each-rule-means-in-practice); the name lands after the 144 showcase renders ✓ verified 2026-11-10: v58-03-self step 30,000, shared seed 10, picked by the showcase rule from 144 calls ([the calls](showcase-calls.csv)); named in [the frozen set](held-out-set.md), commit `089abd6e` before any held-out result
- [x] 📝 markdown created: the rule that every quantitative result ships with a qualitative grid from the same run, rows pairs and seeds picked by the fixed list, columns the compared methods, so no number appears without the images it was counted on ✓ verified 2026-11-10: [the protocol page](protocol.md), rules 5 and 8
- [x] ✅ check passed: you have read and agreed the page before the first evaluation job starts ✓ confirmed by hand 2026-11-10: "Yes", after choosing option B (adapter picked on the two showcase pairs); status line on [the protocol page](protocol.md) reads agreed
- [ ] ✅ check passed: every result row carries the axis values that produced it (model, sampler, pair, seed, group, setting), so a number can be attributed afterwards
