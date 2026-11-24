# Session 13: the manager loop over the research loops

This runs as a background session, in the same permission mode as the other loops. To start it by hand from `/home/molef/PhD/concept-composition`:

```text
claude --bg --permission-mode auto --model claude-opus-5-5 --effort high --name "🧭 PhD Loop 13: Manager" "/loop Follow artifacts/notes/progress/prompts/session-13-manager.md to its end. Read artifacts/notes/progress/prompts/shared-rules.md on every tick."
```

## Background

Several looping sessions work towards the two papers at once, each on its own steps of the board in `artifacts/notes/progress/board.json`. Nothing reads them together on a timer, so the owner has had to ask each loop by hand what it is doing, whether it still serves its intent, and what it needs answered. This loop does that reading. It reads, nudges and reports. It decides nothing that belongs to the owner or to another loop, and it changes no other loop's files.

Your owner code on the board is `s13`, and your steps are N1 and N2.

## Who the owner is

The owner is the head of the project: the person who sets direction and judges whether the work is worth it, not someone who follows each loop's internal steps. Everything this loop shows the owner, on the page, in a notification or in a diagram, says what the work is and why it matters in plain words. A step code such as "M9" or "Z2" never appears alone: write "the test of whether drawing each subject in its own region fixes borrowed parts (Loop 6, step M9)". Each question, finding and decision fits in one sentence a newcomer understands, with the detail one click away. Diagrams show at most about ten boxes, label every box with what it does, and leave out anything the owner does not need to act on.

## The loops it watches

Every session in the shared rules' ownership table that still has a step not done on the board. On 2026-11-24 that is Loop 4 (rivals and targets), Loop 6 (best method), Loop 7 (abstract-checker papers), Loop 8 (abstract checkers), Loop 11 (drain into the pillars) and Loop 12 (where CO3 leaks). Loops 5, 9 and 10 are closed; read only lines in `requests.md` addressed to them. A loop added later joins the list on the tick its steps appear.

Each loop's intent is the paragraph for it in `artifacts/notes/progress/score-space/13-alignment-check.md`, and the two papers' intent is that page's opening section. A loop with no paragraph there gets one written from its session prompt's Background section, marked as written by this loop, for the owner to correct.

## Objectives

1. **Step N1:** write `artifacts/notes/progress/manager/status.md` for the first time, with the sections below, every loop filled in, and the first alignment round done.
2. **Step N2:** keep the page current on every tick until every watched loop's steps are done or the owner says stop. Proof: the page's last tick time is within one tick of the stop, and its final section says why the loop stopped.

## What the page holds, in this order

1. **Decided for you, and still for you.** First, the questions this loop decided since the owner last looked: the question in one plain line, the option chosen, why, and how to overrule it. Then the questions only the owner can decide (see "Deciding for the owner"), newest first, with the default and how long each has waited. This is the section the owner reads first, so it is never longer than it needs to be.
2. **Where each loop is.** One row per watched loop: what it is doing in plain words and why, steps left, its estimated finish (from `session_status.py`, corrected for the queue it is actually waiting in), whether its session is busy or idle, and what blocks it.
3. **Resources.** What the cluster gives us right now and whether we use it well: our jobs running and queued against the caps (one `biggpu`, six `bigbatch`), how long they wait, nodes that are down or drained (`sinfo -R`), idle `biggpu` capacity while our jobs queue on `bigbatch` or the reverse, GPU-hours per loop this day and against its budget, and one line saying what would make the queue move faster.
4. **Things that may be wrong.** One row per finding, newest first, each with its source and the loop it concerns:
   - a cluster job of a watched loop that failed, was cancelled, timed out or ran out of memory in the last 24 hours (`sacct`);
   - slow compute: a job queued more than 2 hours, or running more than twice its design's time estimate, with the node and GPU it landed on;
   - a step marked in progress with nothing running for it (`python3 scripts/plan_pulse.py`'s stale check, the queue, and the session's state);
   - a step that has run more than twice its estimated finish, or a loop whose detours keep growing;
   - GPU-hours spent by a loop against the budget in its prompt;
   - a new row on `artifacts/notes/progress/void-list.md`, or two records that give different numbers for one result;
   - a design that is likely to waste its run (see "Checking designs before they run");
   - **a pattern of bad results**: two or more nulls, misses or inconclusive verdicts in a row from one loop, or a positive control that failed. Each such row names the request this loop sent, as below.
5. **Alignment.** One row per loop: still working towards its intent (yes, partly, no), with the one fact that shows it, cited to a review file.
6. **Nudges, requests and decisions sent.** The last ten messages this loop sent to other loops, with time and reason.
7. **Requests waiting.** Lines in `requests.md` addressed to a loop that has not acted on them, oldest first. A request addressed to a loop that is closed or finished is re-routed to the loop that now owns that work, or to the owner, and marked so.
8. **Last tick** and, when stopped, why.

## Deciding for the owner

Loops wait on the owner too often. This loop decides open board questions for the owner, so that work keeps moving, under these limits.

**It decides** a question when every option is reversible, stays inside the asking loop's GPU budget, and changes no bar or verdict already recorded. It runs the `reviewer` skill on the asking loop first, then chooses the option the evidence supports: usually the asker's recommendation, and a different one when the review shows the recommendation rests on a wrong assumption. It records the choice with `python3 scripts/progress_board.py answer <id> "Decided by Loop 13 for the owner: <option>, because <reason>; review <path>. The owner may overrule by answering again."`, messages the asking loop, and lists it under "Decided for you".

**It leaves to the owner**, listing them first under "Still for you":
- deleting or moving files outside the approved lists, pushing, or publishing;
- anything touching a secret, a client's data or a patient's data;
- choosing between two records that disagree on a result;
- changing a bar or a pre-registered test after its result is known, or reading a held-out set early;
- spending more than 3 GPU-hours on one run, or taking a loop past its budget;
- stopping a loop, dropping a paper's direction, or starting a new loop.

When a question of the second kind has waited 12 hours, the page and one notification say so again; this loop still does not answer it.

## Checking designs before they run

When a loop writes or changes a design file, read it before its first run and check it as a senior experimentalist would:
- **The data.** Is it the right data for the question, large enough, balanced, varied enough (subjects, poses, backgrounds, seeds), with labels someone has checked? Is anything held out kept out of training and tuning?
- **The controls.** A positive control that shows the measurement can succeed, a random or shuffled control, and the simplest baseline (pasting, the plain sampler, best-of-N at matched compute) beside the method.
- **The comparison to the field.** A plan to compare against the strongest rival on a known benchmark, or a stated reason none applies.
- **The information.** Does the input hold what the probe or push asks for at the steps it acts on?
- **The cost.** Does the cost line match what the queue will really take, and is there a cheaper first read?

A problem goes to the loop as one request naming the fix, and onto the page. A problem that would waste more than 3 GPU-hours, or make the result unusable for a paper, also reaches the owner.

## One tick

1. Read `shared-rules.md`. Run `python3 scripts/progress_board.py show`, `python3 scripts/progress_board.py line` and `python3 -I ~/.claude/skills/session-status/scripts/session_status.py --repo "$PWD" --line`. Read the open questions from `board.json`, the newest lines of `requests.md`, and `ListAgents` for which loop sessions are alive, busy or idle.
2. Rewrite `manager/status.md` from what was read. Re-read any file before quoting a number from it, because other loops edit them.
3. Nudge a loop with one `SendMessage` only when one of these holds: a step it waits on turned done, the owner answered one of its questions, a request addressed to it has waited more than 6 hours, or it has been idle for more than 2 hours with a ready step. Say what changed and where it is recorded. Send the same nudge at most once.
4. Check resources and things that may be wrong. Read the cluster only, through `ssh mscluster107` with a 300-second timeout: `squeue -u mmolefe`, `sacct -u mmolefe -S now-24hours -o JobID,JobName,State,Elapsed,NodeList,Submit,Start,Partition`, `sinfo -p biggpu,bigbatch -o '%P %a %D %T %G'` and `sinfo -R`. When `biggpu` has an idle card while our jobs wait on `bigbatch`, or the reverse, tell the loop whose job waits, inside the caps. Check every design file written or changed since the last tick. Match each job to its loop by its job name. Read each watched loop's newest review files for their verdict lines. Add or clear rows in the "Things that may be wrong" section.
5. Decide the open questions this loop may decide, as in "Deciding for the owner", and message each asking loop.
6. When a loop shows a pattern of bad results, send it one **method review request**, by `SendMessage` and as one line in `requests.md`. The request asks the loop to pause new runs and write a short review in its own folder before the next run. The review answers: what the failures have in common, which papers it has not read that bear on them (searched and named), whether another loop's results already explain them (this loop names the ones it has seen, such as Loop 9's finding that SDXL's own features read where sentence scores do not), whether the step is still worth its cost for its paper, and what in its design or method it would change. The loop decides what to do with its review; the owner decides any change of direction. Send at most one such request per loop per pattern, and list it on the page.
7. Every 24 hours, when the owner asks, and before sending any method review request, load the `reviewer` skill (`~/.claude/skills/reviewer/SKILL.md`) and run it with `--for-manager` on the watched loops (or the one loop in question). It rebuilds the knowledge base of every paper the project has read, checks each loop's assumptions, design, pictures and results against it, and writes `artifacts/notes/progress/reviewer/<date>-review.md` with one verdict per loop: continue, fix or stop. Put its one-line verdicts in the Alignment section. A **stop** verdict or a proven-wrong assumption reaches the owner as the reviewer's board question; this loop sends the method review request only after the reviewer has run, and points the loop at the review.
8. Every 6 hours, run the alignment round: for each watched loop, compare its latest review files and board notes with its intent paragraph, and update the Alignment section. Ask a loop directly with `SendMessage` only when its files cannot settle the question, with the same five questions as the 2026-11-24 check.
9. Send the owner one `PushNotification` only when a question only the owner can decide appears, when this loop has decided questions since the last notification (one line listing them), when a method review request is sent, when a loop has been idle more than 4 hours with nothing blocking it, or when a loop's alignment turns to "no". Begin it with `Loop 13:`. Never repeat a notification for the same thing.
10. Commit only `artifacts/notes/progress/manager/` and `artifacts/notes/progress/reviewer/` by named paths under the shared lock. Print the board line and the session-status line. Wake again in 3,600 seconds.

## Rules

- Change no file outside `artifacts/notes/progress/manager/` and `artifacts/notes/progress/reviewer/`, except one line in `requests.md` when a loop's area needs something a nudge cannot carry.
- Answer only the board questions "Deciding for the owner" allows. Set no bar, and change no other loop's step state. Relay the owner's answers given in chat when the owner asks.
- **Closing sessions.** The owner allows this loop to close loop sessions and agents with `claude agents --json` (plain `claude agents` needs a terminal), `claude logs <id>` and `claude stop <id>`. It closes, without asking, a session whose loop has every step done or has been closed, a second session running the same loop as another (keep the newer one), and a finished background agent. Before closing, it checks the session is idle, carries that loop's open requests to their new owner, and lists the closure on the page. It never closes a session that is busy, a loop with steps still to do without the owner's yes, or itself. If a permission prompt blocks these commands, it tells the owner the rules to add: `Bash(claude agents:*)`, `Bash(claude logs:*)`, `Bash(claude stop:*)`.
- Start no session. Run no GPU or cluster job, cancel no job, and read the cluster only through `squeue`, `sacct` and `sinfo`.
- A disagreement between two loops' records goes on the page and to the owner, never resolved here.
- Use the owner's names for the papers: Paper 1 for experts that never met, Paper 2 for doing better than CO3 inside one model.
- Follow the writing rules in `~/.claude/CLAUDE.md` and `~/.claude/WRITING_STYLE_DETAIL.md` on the page and in every message.

## What to hand back

`artifacts/notes/progress/manager/status.md`, current on every tick, and the nudges it lists. When the loop stops, one `PushNotification` saying why, and a final report: what each loop finished, what is still open, and the questions still waiting for the owner.
