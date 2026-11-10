# PRZE-3b: a held-out set big enough for rates

Four held-out pairs cannot carry a success rate. Answers PrzE's question 1 and UHz6's question about rates over all held-out pairs and seeds, and about pairs that do not compete.

## Outcomes

- [x] 📝 markdown created: the held-out list in three groups (joint prompting fails; joint works but the pair is outside the 48; foreground and background pairs that do not compete), chosen and saved before any result, about 10 pairs per group (estimate) ✓ verified 2026-11-10: [the frozen set](held-out-set.md), commit `089abd6e`: hard 3, easy but unseen 29, not competing 21, from a census of joint prompt and plain product at seeds 9 to 16; the hard group is under 10 because the joint prompt rarely fails on these candidates
- [x] ✅ check passed: none of the held-out pairs appears in the training pool, checked against the 48 ✓ verified 2026-11-10: every candidate was drawn from the pair files minus the 48 of `pair_pool_v58.yaml`; the frozen set lists the 48 apart
- [ ] ✅ check passed: the cat and dog joint prompt is judged per seed, since it is wrong on 4 of 8 held-out seeds per earlier notes, and unjudgeable seeds are excluded and counted Open: cat and dog is now a showcase pair and counts in no rate; the per-seed judging applies to the showcase calls, which judged it per seed
- [ ] 💻 code ran: every method sampled on every held-out pair at seeds 9 to 16, with compose rate and an interval per group
