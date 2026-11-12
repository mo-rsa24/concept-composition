# Pair ledger: decisions on subjects the guard holds back

`scripts/showcase/disallowed_subjects.py` keeps two lists. `BLOCKED` is never rendered. `ASK_FIRST`
may be rendered only under a decision recorded here, and a caller passes `allow_ask_first=True` only
for the prompts a row below covers. A row never lifts `BLOCKED`.

| Date | Subjects | Prompts covered | Decision | Who | Why |
|---|---|---|---|---|---|
| 2026-11-10 | frog, mouse | the 45 prompts of the Attend-and-Excite sets that name them (21 of 66 animals, 24 of 144 animals with objects), in `data/prompt-sets/attend-and-excite/a.e_prompts.json`, for the review response's R1 and R5 only | allowed | the user | keeps the three sets at the 66, 144 and 66 prompts the CO3 paper scores, so the two tables can be set side by side |
| 2026-11-10 | frog, toad | frog and toad, the look-alike census pair | not rendered | held | outside the decision above |
| 2026-11-12 | frog | the 21 T2I-CompBench prompts that name a frog (20 of 300 spatial, 1 of 300 complex), in `data/prompt-sets/t2i-compbench/`, for the review response's R14 only; mouse stays dropped there | allowed | the user | keeps the spatial set at 284 of 300 prompts rather than 264 |
| 2026-11-12 | computer mouse, baseball bat; mouse | "computer mouse" and "baseball bat(s)" as whole phrases in the GeneVal two-object and counting sets and T2I-CompBench spatial (objects, set aside before the subject check, so bat stays blocked on its own); mouse in the 16 T2I-CompBench spatial prompts that name it; for R14 only | allowed | the user | GeneVal two-object and counting and T2I-CompBench spatial run complete (99, 80, 300 prompts) |
| 2026-11-12 | spider, web | the 2 T2I-CompBench complex prompts about a spider's web | never rendered | the user | spider stays permanently blocked |
