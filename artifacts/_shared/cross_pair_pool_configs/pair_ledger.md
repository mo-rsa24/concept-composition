# Pair ledger: decisions on subjects the guard holds back

`scripts/showcase/disallowed_subjects.py` keeps two lists. `BLOCKED` is never rendered. `ASK_FIRST`
may be rendered only under a decision recorded here, and a caller passes `allow_ask_first=True` only
for the prompts a row below covers. A row never lifts `BLOCKED`.

| Date | Subjects | Prompts covered | Decision | Who | Why |
|---|---|---|---|---|---|
| 2026-11-10 | frog, mouse | the 45 prompts of the Attend-and-Excite sets that name them (21 of 66 animals, 24 of 144 animals with objects), in `data/prompt-sets/attend-and-excite/a.e_prompts.json`, for the review response's R1 and R5 only | allowed | the user | keeps the three sets at the 66, 144 and 66 prompts the CO3 paper scores, so the two tables can be set side by side |
| 2026-11-10 | frog, toad | frog and toad, the look-alike census pair | not rendered | held | outside the decision above |
