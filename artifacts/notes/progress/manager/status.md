# 🧭 Manager: where the loops stand

Loop 13 rewrites this page every hour. It reads the progress board, the open questions, the requests
between loops, the cluster and the live sessions, decides the questions it is allowed to decide for you,
and leaves the rest here. Paper 1 is experts that never met; Paper 2 is doing better than CO3 inside one
model.

**Last tick:** 2026-11-24 17:03 (UTC+2). The board stands at 57 of 90 steps done and 6 in progress.

## Contents

- [Decided for you, and still for you](#decided-for-you-and-still-for-you)
- [Where each loop is](#where-each-loop-is)
- [Resources](#resources)
- [Things that may be wrong](#things-that-may-be-wrong)
- [Alignment](#alignment)
- [Nudges, requests and decisions sent](#nudges-requests-and-decisions-sent)
- [Requests waiting](#requests-waiting)
- [Last tick](#last-tick)

## Decided for you, and still for you

### Decided for you since you last looked

Each was reversible, inside the asking loop's budget, and changed no recorded bar or verdict. To
overrule one, answer it again: `python3 scripts/progress_board.py answer <number> "<your answer>"`.

| # | The question | Decided | Why |
|---|---|---|---|
| 83 | Should Loop 6 close its test of whether drawing each subject in its own region stops borrowed parts, after three tries missed? | **Close it**, reporting what it found | Loop 6's own [method review](../method-search/04-bind-each-body/method-review.md) and the [reviewer](../reviewer/2026-11-24-review.md#loop-6-find-the-best-method) agree: every try split the picture using a signal that had not yet separated the two subjects, and a fourth would cost about 4.9 GPU-hours without fixing that. Costs nothing. |
| 78 | Who runs Loop 10's small digit test of whether the experts' vote beats simply pasting their samples, for Paper 1? | **Loop 6**, after its current step | About 0.3 GPU-hours inside Loop 6's budget, and it adds the paste comparison the reviewer finds Paper 1's tests lack. |
| 81 | Loop 11's ten format conflicts on the 48 method and finding pages it wrote | **All ten of its defaults** | None changes a recorded number or verdict; where two verdicts exist both stay side by side. The reviewer found Loop 11's work sound. No GPU. |
| 85 | The reviewer's fixes for Loop 6's search for the best checker | **Yes to two of three**: first check whether the checker can tell CO3's clean pictures from its borrowed-part ones at each step (about 1.2 GPU-hours), and give the two-colour CLEVR test a paste comparison (CPU only) | Both are reversible and inside budget. The third fix changes how Loop 6 is judged, so it is yours, as question 87. |
| none | Eight requests were addressed to Loops 7 and 10, which have stopped | **Re-routed**: four to Loop 8, four to Loop 11 | Each corrects a page that disagreed with what ran; no result changes. A ninth, on numbers in Loop 10's digit-signal review, Loop 10 fixed itself (commit `77cf7bb7`: five copying slips corrected to match its result file, verdict unchanged). |

**Acted on.** Loop 6 carried out 83, 78 and 85 by 16:31 (commits `6f8f65e5` and `58cee28a`): the region test is closed on the board, the digit test is its new step M11, and the check of whether its checker can see CO3's failure is designed with its bar written first and runs as jobs 94593 and 94594 before any comparison renders.

### Answered by you

At 16:55, in chat, recorded on the board and sent to each loop:

| # | The question | Your answer |
|---|---|---|
| 88 | Should Loop 11 stop after its last two steps until a research loop has something new? | yes, the default |
| 87 | Should Loop 6's checker search be judged on two clear subjects, drawing style ignored? | yes; damage reported beside it, two reviewers per picture |
| 79 | Which of Loop 9's two recorded verdicts leads its report title? | the default: the test's own verdict on the title, and the fairness test as its own finding beside it (answered at 16:59) |
| 86 | Should Loop 8's SDXL test push identity on CO3's own run and be judged against CO3 alone? | yes; it starts from Loop 6's training list, about 6 GPU-hours. Loop 8 is asked to use the same two-clear-subjects measure as Loop 6 so the two compare |

### Still for you

Newest first. A default happens only if you answer with it; nothing changes while a question waits.

| # | The question, in one line | Why it is yours | Default | Waiting |
|---|---|---|---|---|
| 89 | The board script cannot change a step once it exists, so three records disagree with your answers: Loop 8's SDXL test still points at the old prerequisite and proof, the closed small CLEVR test still holds the go or no-go, and Loop 10's skipped steps keep a closed loop unfinished. May Loop 13 add an edit command to the board script and fix exactly these three? | changes a shared script and other loops' records | the records stay, and this page keeps listing them | just filed |
| 80 | Should a checker choose CO3's weight-sharpness setting at each of its corrected steps ([design](../score-space/11-the-checker-chooses-on-co3.md))? First read on 6 of 12 pairs | over 3 GPU-hours on one run (about 3.1) | **you chose to wait** until Loop 6's per-step read of its checker on CO3 lands (jobs 94593 and 94594); Loop 13 brings it back with that result | since 12:25 |

Your own reading step (confirm the draft statements from your readings) is ready on the board.

## Where each loop is

<a href="loops-and-blockers.png"><img src="loops-and-blockers.png" width="200" alt="Each watched loop as a box, with arrows from your open questions to the work they hold, cluster jobs in blue, and stranded requests dashed grey."></a>

The finish range is `session_status.py`'s, from the 52 finished steps: fastest pace seen to slowest.

| Loop | Doing now, and why | Steps left | Estimated finish | Session | Held by |
|---|---|---|---|---|---|
| 4. Rivals and targets | Keeping every target current. It recorded the CLEVR paste result and placed the two-colour CLEVR test in the CLEVR plan. | 1 | none useful: it ends after the paper tables, which have not started | idle | nothing |
| 6. Best method | Training two checkers on differently balanced lists of training pictures, to learn which list Paper 2's checker should use (step M5). The training failed twice within 3 minutes and is running again. Two-colour CLEVR renders for Paper 1 run beside it. | 10 | 4.7 h to 232.7 h | busy | its own training job; question 87 decides how the result is judged |
| 7. Abstract-checker papers | Finished; its session is closed. | 0 | done | closed | nothing |
| 8. Abstract checkers | Checking that its digit checker really reads which digits are present. Its robustness bars hold: a second seed scores 0.877 balanced accuracy on unseen digit 4 (bar 0.75; 0.5 is chance), shuffled labels score 0.500, and the weakest pair 0.844. Now running, with bars fixed first: the same reading by a rule that needs no training, and a read on real sampler states (jobs 94595 to 94597). | 3 | 2.4 h to 111.5 h | busy | the SDXL test waits on Loop 6's training list and on question 86 |
| 11. Drain into the pillars | Auditing the 48 pages it wrote against the pillar formats. | 2 | 57 min to 43.4 h | idle | nothing since question 81 was decided |
| 12. Where CO3 leaks | Measuring what predicts which animal pairs borrow each other's parts under CO3. Its small first read finished in 16 minutes; the main pass waits for a node. | 5 | 2.4 h to 113.9 h | busy | a free node for job 94450 |

## Resources

Read at 16:28 from `squeue`, `sacct`, `sinfo` and `sinfo -R`.

| What | Now |
|---|---|
| Our jobs against the caps | 2 running on `bigbatch` of 6 allowed (Loop 6's checker training, Loop 8's identity check); 0 on `biggpu` of 1 allowed; 8 queued |
| Longest wait | Loop 12's main predictor pass (job 94450, 4-hour limit): about 1 h 50 min, waiting for resources |
| Nodes out of service | `bigbatch`: 12 down and 5 drained of 48. `biggpu`: 2 down and 2 drained of 7, including mscluster108 and mscluster111, drained at 09:27 and 09:10 today after a failed job kill |
| Idle capacity | none on either partition; every working node is allocated |
| Compute against budget since 2026-11-21 | Loop 6: 53.4 node-hours of 87 (34.8 were CLEVR renders that may have used no GPU). Loop 8: 11.9 of 30. Loop 12: about 0.4, no total set. Slurm here records no GPU count, so these are node-hours, an upper bound |

**What would make the queue move faster:** nothing within our control right now. There is no idle
`biggpu` card to move to, and the drained nodes need the administrator.

## Things that may be wrong

Newest first.

| Concerns | Finding | Source |
|---|---|---|
| Loop 6 | **Design note, not a problem.** The new check scores whether the checker's score lands on the right side of zero, a fixed cut. If its area under the curve, reported beside, is high while that accuracy is low, the checker separates CO3's clean pictures from its borrowed-part ones and only its cut is off. Read the two numbers together when the check lands. | `method-search/01b-balance-the-checker-data/design.md`, the check section |
| Loop 6 | **The checker training failed twice in 3 minutes** (jobs 94473 and 94474) before the current run (94579) started. | `sacct` |
| Loop 12 | **The main predictor pass has waited about 1 h 50 min** for a node; it becomes slow compute at 2 hours. | `squeue` |
| Loops 6 and 8 | **A run of bad results, now reviewed.** Loop 6's region test missed three times in a row and Loop 8's misses shared a cause. Both answered their reviews within minutes and corrected their records and pictures (Loop 6 commit `f29cf613`, Loop 8 commit `8c310b3a`). | [the review](../reviewer/2026-11-24-review.md) |
| Loop 6 | **A closed step is still open on the board.** Your answer to question 84 closed the small CLEVR test (step M6) as not run, but the board still shows it to do, and the go or no-go still waits on it. | board, question 84 |
| Loop 10 | **Closed, but four steps open on the board** (listed under Still for you). | board, question 76 |
| Loop 4 | **The standing step has run 54 hours**, longer than any finished step (23.5 hours). It ends only after the paper tables, by design. | `session_status.py` |
| Loops 6 and 8 | **Failed jobs in the last 24 hours, each followed by a rerun that finished**: Loop 6, 9 failed and 4 cancelled; Loop 8, 2 failed. No time-outs or out-of-memory ends. | `sacct` |

Nothing found for: jobs running past twice their estimate, steps in progress with nothing running for
them, or new rows on the void list (unchanged since 2026-11-22).

## Alignment

Round 1, 2026-11-24 16:10, against the intents in [the alignment check](../score-space/13-alignment-check.md).

| Loop | Still working towards its intent? | Reviewer | The fact that shows it |
|---|---|---|---|
| 4. Rivals and targets | yes | next pass | It said Paper 1 needs a test where pasting fails; the two-colour CLEVR test is now in [the CLEVR plan](../targets/clevr-scaling-plan.md). |
| 6. Best method | partly, now re-aimed | **fix** | The checker search still aims at drawing two separate bodies, which CO3 already does in 96 of 96 pictures, while CO3's real failure is borrowed parts ([review](../reviewer/2026-11-24-review.md#loop-6-find-the-best-method)). |
| 7. Abstract-checker papers | yes, finished | not reviewed | Its three steps are done; the five-checker design is in `abstract-checkers/03-design.md`. |
| 8. Abstract checkers | partly, now re-aimed | **fix** | The digit half holds (0.861 on unseen digits 5 to 9 against 0.75), but its SDXL test counts bodies against the plain product, not identity against CO3 ([review](../reviewer/2026-11-24-review.md#loop-8-test-abstract-checkers)). |
| 10. Score path | closed | not reviewed | Its verdict is written (`score-space/09-verdict.md`). |
| 11. Drain into the pillars | yes | **continue** | [Its checks page](../drain/04-checks.md) covers all 48 pages it wrote, with 15 of 28 findings fixed. |
| 12. Where CO3 leaks | yes | next pass | [Its method candidates](../leakage/04-paper-2-method.md) keep your constraints: SDXL's weights untouched, one run per seed, best-of-N only as a baseline. |

**Loop 11's intent, written by Loop 13 from its brief for you to correct.** Loop 11 takes the
diagrams, picture sheets and results the research loops produce, which sit in their own notes folders
where no pillar page shows them, and drains them into `report/` and `context/`: checked against the
pillar formats, explained from the overview down to each part, in organised folders of numbered pages.
It took this over from Loops 2 and 3, which have stopped.

## Nudges, requests and decisions sent

Newest first.

| Time | To | What and why |
|---|---|---|
| 17:02 | Loop 8 | Its rewritten SDXL test passes the design check (stop rule, random and zero controls, comparison prompts fixed in advance, the same two-clear measure as Loop 6; about 7 GPU-hours, inside budget). Its board record goes to you as question 89. |
| 16:30 | Loop 10 | Its question 78 is decided; its open requests are re-routed. |
| 16:39 | Loop 8 | The four requests re-routed from Loop 7 take the form Loop 8 proposed: the design text stays as written, and one added section lists what ran differently, line by line with commits. Loop 8 also put this to you in its own session; your answer there governs. |
| 16:30 | Loop 8 | Four requests re-routed from Loop 7, which has finished. |
| 16:30 | Loop 11 | Question 81 decided (all ten defaults); four requests re-routed from Loop 10. |
| 16:30 | Loop 6 | Questions 83, 78 and 85 decided; question 87 left with you, so its bar stands until then. |
| 16:25 | Loop 8 | **Reviewer verdict: fix**, with the review and question 86. |
| 16:25 | Loop 6 | **Reviewer verdict: fix**, with the review and question 85. |
| 16:09 | Loop 6 | **Method review request** on the region test after three misses; answered at 16:11 with [its method review](../method-search/04-bind-each-body/method-review.md). |

### Sessions closed

Under the closing rule: idle, and the loop finished, closed, or a second copy of one loop. Open requests were carried over first.

| Time | Session | Loop | Why |
|---|---|---|---|
| 16:46 | `70624adb` | 10, the score path | closed by you on question 76; its requests went to Loop 11 and question 80 stays with you |
| 16:46 | `38bc00f5` | 4, rivals and targets | the older of two sessions; `f3676c33` keeps running and the board now names it |
| 16:44 | `09f14d5d` | 7, abstract-checker papers | every step done |
| 16:44 | `bed4762f` | 9, concepts into scores | closed by you |
| 16:44 | `b0fc9e51` | 5, reproduce the rivals | closed by you |

Four more entries in the session list (an old Loop 4 copy, Loops 2 and 3, and a Loop 7 copy in the learning folder) have no running process, so there was nothing to close.

## Requests waiting

Read from `requests.md` at 16:15 (277 lines then). Of the 158 lines addressed to a watched loop, 49 had
been acted on, 93 asked nothing, and 15 waited. Nine of those were re-routed or fixed this tick
(above). Still waiting:

| Line | To | From | Added | The request | Why it still waits |
|---|---|---|---|---|---|
| 127 and 269 | Loop 4 | Loop 4 | 11-23 13:54 | Make the scoreboard count a win against the best finished rival rerun (your question 47). | Not due: no score of ours exists yet. |
| 133 | Loop 4 | Loop 6 | 11-23 14:51 | Write your decision that abstract checkers belong in Paper 2 into the problem statements and the two-paper route, once a result exists. | Unclear whether due: digit results exist, the SDXL test has not run. |
| 239 | Loop 11 | Loop 8 | 11-24 14:14 | Pass on to whoever drew `annotated-overviews/loop-8.png` that its push window and push length are wrong. | No later line mentions it. |
| 247 and 273 | Loop 11 | Loop 4 | 11-24 14:14 and 15:58 | Commit the cross-reference links written this morning and afternoon. | The files are still uncommitted. |
| 251 | Loop 6 | Loop 11 | 11-24 14:14 | Backlinks from Loop 6's first-test and region-test designs to the three method-search findings. | No such links yet. |

## Last tick

2026-11-24 17:03. Next tick about 17:06. The briefing for you is [the briefing page](briefing.md).
