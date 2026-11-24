# 🧭 Manager: where the loops stand

This page is rewritten every hour by Loop 13 (owner code `s13`) from the progress board, the open
questions, `requests.md`, the cluster queue and the list of live sessions. It decides nothing: every
row either points at something you answer or reports what a loop's own files say. Paper 1 is experts
that never met; Paper 2 is doing better than CO3 inside one model.

**Last tick:** 2026-11-24 16:15 (UTC+2). Board: 56 of 90 steps done, 6 in progress.

## Contents

- [For you to answer](#for-you-to-answer)
- [Where each loop is](#where-each-loop-is)
- [Things that may be wrong](#things-that-may-be-wrong)
- [Alignment](#alignment)
- [Nudges and requests sent](#nudges-and-requests-sent)
- [Requests waiting](#requests-waiting)
- [Last tick](#last-tick)

## For you to answer

Five board questions are open. Answer with `python3 scripts/progress_board.py answer <number> "<answer>"`,
or in chat to the loop that asked.

| # | Asked by | The question, in one line | Default if you say nothing | Open for |
|---|---|---|---|---|
| 83 | Loop 6 | Paper 2's region-per-subject test (M9) failed its second region reader too (10 of 24 reads against a limit of 6): close it, try one last reader from Loop 10's per-subject signal, or take boxes from Be Decisive? | close M9, reporting the first rung and the two failed readers; no cost | 1 h 39 min |
| 81 | Loop 11 | Ten points where the pillar formats conflict on the 48 pages Loop 11 wrote (page shape for `context/methods/`, commands on method pages, picture width, verdict words, cards for loop pictures and five more): take all ten defaults or name the ones to change? | all ten defaults as listed on the board | 3 h 12 min |
| 80 | Loop 10 | Who runs the test where the checker picks CO3's weight sharpness at each corrected step (`score-space/11-the-checker-chooses-on-co3.md`), and at which size first? | Loop 6 runs the 6-pair first read, about 3.1 GPU-hours, and adds the rest only if the checker leads CO3 alone by 5 of 96 | 3 h 45 min |
| 79 | Loop 11 | Loop 9's sentence-score test has two recorded verdicts (null on three properties in its own review, inconclusive in the later fairness test): which goes on the report finding's title line? | keep the test's own verdict on the title, and give the fairness test its own finding beside it | 4 h 53 min |
| 78 | Loop 10 | Who runs Paper 1's short digit ablation of the experts' vote, with and without the corrector, beside sample-and-paste (`score-space/10-paper-1-placement-or-sampler.md`, about 0.3 GPU-hours)? | Loop 6 adds it as one Paper 1 step after the CLEVR paste test, on plan 33's settings, seeds 192 to 255 | 5 h 20 min |

Your own step C2 (confirm the draft statements from your readings) is ready on the board.

## Where each loop is

Estimated finish is the range `session_status.py` prints from the 51 finished steps: the low end at
the fastest step pace seen, the high end at the slowest. Session state is from the live session list.

| Loop | Now on | Steps left | Estimated finish | Session | What blocks it |
|---|---|---|---|---|---|
| 4. Rivals and targets | K6, keeping every target current: recorded the CLEVR paste result and folded the two-colour CLEVR test into the CLEVR plan at 15:46 | 1 | 29 min to 23.5 h | idle (two Loop 4 sessions are alive, see below) | nothing now; K6 closes only after the paper tables E1 and E2, which have not started |
| 6. Best method | M5, balancing the checker's training pictures across pairs; jobs 94473 and 94474 wait for a GPU | 10 | 4.8 h to 233.1 h | busy | question 83 holds M9; D3 waits on M5; five CLEVR two-colour render jobs (94548 to 94552) are queued for M10 |
| 7. Abstract-checker papers | all three steps done | 0 | done | idle | nothing; the loop has finished |
| 8. Abstract checkers | V6, whether a checker reads which digits are present: support on digit 4 (0.888) and digits 5 to 9 (0.861, bar 0.75); two robustness jobs 94471 and 94472 running | 3 | 2.4 h to 112.2 h | busy | V4 (the SDXL test) waits on Loop 6's D3, which waits on M5 |
| 10. Score path | closed by the owner on question 76; X6 to X9 still open on the board | 4 on the board | none | idle | the board, see [things that may be wrong](#things-that-may-be-wrong) |
| 11. Drain into the pillars | Y4, the pillar checks over its 48 new pages | 2 | 58 min to 43.7 h | idle | question 81 |
| 12. Where CO3 leaks | Z2, what predicts which pairs borrow parts: the predictor pass you approved on question 82 is queued as job 94534; job 94450 is on hold | 5 | 2.4 h to 114.2 h | busy | a free GPU for job 94534 |

## Things that may be wrong

Newest first. Cluster times are converted from the cluster's UTC clock to local time (UTC+2). Read
from `squeue` and `sacct` on 2026-11-24 at 16:09.

| Concerns | Finding | Source |
|---|---|---|
| Loops 7 and 10 | **Nine waiting requests address loops that will not act.** Four lines ask Loop 7 (finished) and five ask Loop 10 (closed) to correct their own records; see [requests waiting](#requests-waiting). Someone has to own them. | `requests.md` |
| Loop 6 | **A run of bad results on M9**, the test of whether giving each subject its own region fixes identity. Rung 1 gave "regions without names" with all 8 clear pairs damaged, then both region readers missed their limit of 6 failed mask reads: 11 of 24, then 10 of 24. A method review request went to Loop 6 at 16:09, scoped to M9; Loop 6 answered at 16:11 with [its method review](../method-search/04-bind-each-body/method-review.md), recommends question 83's option A, and starts no M9 run before you answer (see [nudges and requests sent](#nudges-and-requests-sent)). | [the region review](../method-search/04-bind-each-body/review.md), board question 83 |
| Loop 6 | **M10 work before its step starts.** Five CLEVR two-colour render jobs (94548 to 94552) were submitted at 15:33 while M10 shows `todo` and M5 is Loop 6's step in progress. | `squeue`, board |
| Loop 6 | **M6 still open.** Your answer to question 84 closes M6 as not run and puts M10 before M4. The board still shows M6 as `todo`, and M4 still waits on M6 rather than M10. Loop 6 was told in `requests.md` at 15:48. | board, question 84 |
| Loop 4 | **Two Loop 4 sessions are alive,** one started about 3 hours ago and one about 20 minutes ago. Both are idle. Only one should be working K6. | live session list |
| Loop 10 | **Closed, but four steps still open on the board.** You closed Loop 10 on question 76 and skipped X6 to X8, and its verdict (`score-space/09-verdict.md`) meets X9's proof. The board holds X6 to X9 as `todo` because X9 needs X8. While they stay open, Loop 10 counts as unfinished and this loop cannot reach its stop condition. | board, question 76 |
| Loop 4 | **K6 has run 53.9 hours,** longer than any finished step (23.5 hours). It is a standing step that closes only after the paper tables E1 and E2, so this is by design, and its estimated finish is meaningless until E1 starts. | `session_status.py` |
| Loops 6 and 8 | **Failed jobs in the last 24 hours, each followed by a rerun that finished.** Loop 6: 7 failed and 4 cancelled across the CLEVR paste test (renders, data, training: 93715, 93718, 93965, 93988, 94046, 94448), M9 (93719, 93936, 94380, 94381) and M1 (93687). Loop 8: 94115 (V3) and 94337 (V6, after 38 minutes). No time-outs or out-of-memory ends for any watched loop. | `sacct` |
| Loops 6 and 8 | **Compute against budget,** counted as node-hours since 2026-11-21, because Slurm here records no GPU count and each job holds one node. Loop 6: 53.4 of its 87-hour limit, of which 34.8 were CLEVR render jobs that may have run on CPU. Loop 8: 11.9 of its 30-hour limit. Loop 12: 0.1; its prompt sets no total. | `sacct`, session prompts 6 and 8 |

Nothing found for: jobs queued over 2 hours (the oldest pending, Loop 12's 94534, has waited about
50 minutes for a free GPU), jobs running past twice their estimate, steps in progress with nothing
running for them, or new rows on the void list (unchanged since 2026-11-22).

## Alignment

Round 1, 2026-11-24 16:10. The intents are the paragraphs in
[the alignment check](../score-space/13-alignment-check.md), collected this morning. This round
compares them with each loop's reviews and board notes since then.

| Loop | Still working towards its intent? | The fact that shows it |
|---|---|---|
| 4. Rivals and targets | yes | It said Paper 1 needs a target where pasting fails. The two-colour CLEVR test, where neither expert is told where to draw, is now Stage 2 of [the CLEVR plan](../targets/clevr-scaling-plan.md). |
| 6. Best method | partly | The checker search has only just started (M5 at 14:31). Paper 2's region test failed its second reader ([the region review](../method-search/04-bind-each-body/review.md)), while Paper 1's [two-colour test](../method-search/05-two-experts-by-colour/design.md) removes what pasting needs, the gap the morning check named. |
| 7. Abstract-checker papers | yes, finished | U1 to U3 are done; the five-checker design is in `abstract-checkers/03-design.md`. |
| 8. Abstract checkers | partly | The digit half holds: the identity checker reads unseen digits at 0.888 and 0.861 against 0.75 ([the identity review](../abstract-checkers/10-reading-identity/review.md)). The SDXL half has not started. |
| 10. Score path | closed | Its verdict is written (`score-space/09-verdict.md`); the open steps are a board record, not work. |
| 11. Drain into the pillars | yes | [Its checks page](../drain/04-checks.md) audits the 48 pages it wrote (34 under `context/methods/`, 14 in `report/`), the work its brief asks for. |
| 12. Where CO3 leaks | yes | [Its method candidates](../leakage/04-paper-2-method.md) keep your constraint: no change to SDXL's weights, one trajectory per seed, best-of-N only as a baseline. |

**Loop 11's intent, written by Loop 13 from its brief for you to correct.** Loop 11 takes the
diagrams, picture sheets and results the research loops produce, which sit in their own notes folders
where no pillar page shows them, and drains them into `report/` and `context/`: checked against the
pillar formats, explained from the overview down to each part, in organised folders of numbered pages.
It took this over from Loops 2 and 3, which have stopped.

## Nudges and requests sent

| Time | To | What and why |
|---|---|---|
| 2026-11-24 16:09 | Loop 6 | **Method review request on M9**, after three misses in a row (above). Asks Loop 6 to pause new M9 runs, but not M5's training or the CLEVR renders, and to write a short review in its folder answering five things: what the misses share, which unread papers bear on them, whether another loop's results explain them, whether M9 is still worth its cost for Paper 2, and what it would change. It names four results from other loops: Loop 8's identity read working in the later window but not the early one, Loop 9's finding that SDXL's own features read where sentence scores do not, the DAAM attention-overlap result from Loop 12's reading, and Loop 10's per-subject signal (question 83's option B). Also one line in `requests.md`. **Answered** at 16:11 (commit `d0d65be7`): all three misses separate the subjects with a signal that has not separated them yet, six unread papers are named, and M9 in its current form is judged not worth its 4.8 GPU-hours. |

## Requests waiting

Read from `requests.md` at 277 lines. Of the 158 lines addressed to a watched loop, 49 were acted on,
93 asked nothing, and these 15 wait. Each time is the commit that added the line. Line numbers are
`requests.md`'s.

| Line | To | From | Added | The request | Why it still waits |
|---|---|---|---|---|---|
| 127 | Loop 4 | Loop 4 | 11-23 13:54 | Make the scoreboard count a win against the best finished rival rerun, with the printed bar beside it (question 47). Repeated as item 2 of line 269. | `beaten()` in `~/.claude/skills/session-status/scripts/progress_views.py` still compares only with the printed or reproduced target. Not due: `scoreboard.json` has no score of ours yet. |
| 133 | Loop 4 | Loop 6 | 11-23 14:51 | Write question 53's decision (abstract checkers belong in Paper 2) into `problem-statements.md` and the two-paper route, "once a result exists". | Neither file mentions it. Unclear whether it is due: the digit results exist, the SDXL test has not run. |
| 220 | Loop 7 | Loop 8 | 11-24 11:38 | Bring four lines of `abstract-checkers/03-design.md` (push window, right-digit scoring, definition of null, third bar) in line with what Loop 8 ran, or mark them superseded. | The lines are unchanged; Loop 7 has finished. |
| 221 | Loop 10 | Loop 8 | 11-24 11:38 | `score-space/02-the-loops/loop-8-test-abstract-checkers.md` still says "cap at 0.4" (lines 68 and 122) though the picture was redrawn without a cap. | Unchanged; Loop 10 is closed. |
| 225 | Loop 10 | Loop 11 | 11-24 11:38 | Redraw `score-space/05-signal/what-each-probe-reads.mmd` to show the convolutional probe that ran and the right pooled sizes. | Still says "linear probe per digit"; Loop 10 is closed. |
| 238 | Loop 7 | Loop 8 | 11-24 14:14 | `03-design.md` line 69 says each map adds to 1; `v2_learn.py` makes it add to 2,048. | Unchanged; Loop 7 has finished. |
| 239 | Loop 11 | Loop 8 | 11-24 14:14 | Pass on to whoever drew `annotated-overviews/loop-8.png` that its push window should read steps 15 to 25 and b fixed at 0.4. | No later line or check mentions it. |
| 244 | Loop 7 | Loop 11 | 11-24 14:14 | Settle five disagreements between Loop 7's design and the word-map checker pages; fix `wm6`'s prompt and the laptop path in `check_pages.py`. | Nothing committed to those files since 11-23; Loop 7 has finished. |
| 245 | Loop 10 | Loop 11 | 11-24 14:14 | Fix stale items in Loop 10's score-path pages (`00-ladder/NOTATION.md` line 12, README lines 27 and 62, the old window). | Unchanged; Loop 10 is closed. |
| 247 | Loop 11 | Loop 4 | 11-24 14:14 | Commit the morning cross-reference links. | Four files with the links are still uncommitted. |
| 251 | Loops 6 and 10 | Loop 11 | 11-24 14:14 | Backlinks from Loop 6's M1 and M9 designs to the three method-search findings, and from `score-space/04-design.md` to the score-path finding. | No such links; Loop 6 edited the M9 design at 14:11 without one. |
| 252 | Loop 7 | Loop 8 | 11-24 14:14 | Backlinks in `03-design.md` from V2, V3 and V5 to the three digit-checker findings. | None; Loop 7 has finished. |
| 254 | Loop 10 | Loop 11 | 11-24 14:14 | Make `score-space/05-signal/review.md` lines 36 to 42 match `full.json` (it prints 0.576 at t 480). | Unchanged; Loop 10 is closed. |
| 269 | Loop 4 | Loop 4 | 11-24 15:46 | Four tasks carried from the Loop 4 session that ended. | Item 1 is done; item 2 is line 127; item 3 is not due; item 4 (read questions 78 and 79) cannot be checked from files. |
| 273 | Loop 11 | Loop 4 | 11-24 15:58 | Commit the afternoon cross-reference pass, 11 links and 11 backlinks. | The files are still uncommitted. |

**Nine of these lines are addressed to a loop that will not act on them.** Lines 220, 238, 244 and 252
wait on Loop 7, which has finished; lines 221, 225, 245, 254 and Loop 10's half of 251 wait on Loop
10, which is closed. They correct records (a design page, a review's numbers), so someone has to own
them. This is yours to assign; it is listed under things that may be wrong.

## Last tick

2026-11-24 16:15. Next tick about 17:15.
