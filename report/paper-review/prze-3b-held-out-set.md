# PRZE-3b: a held-out set big enough for rates

Four held-out pairs cannot carry a success rate. Answers PrzE's question 1 and UHz6's question about rates over all held-out pairs and seeds, and about pairs that do not compete.

## Outcomes

- [ ] 📝 markdown created: the held-out list in three groups (joint prompting fails; joint works but the pair is outside the 48; foreground and background pairs that do not compete), chosen and saved before any result, about 10 pairs per group (estimate)
- [ ] ✅ check passed: none of the held-out pairs appears in the training pool, checked against the 48
- [ ] ✅ check passed: the cat and dog joint prompt is judged per seed, since it is wrong on 4 of 8 held-out seeds per earlier notes, and unjudgeable seeds are excluded and counted
- [ ] 💻 code ran: every method sampled on every held-out pair at seeds 9 to 16, with compose rate and an interval per group
