# Session 13: the manager loop over the research loops

This runs as a background session, in the same permission mode as the other loops. To start it by hand from `/home/molef/PhD/concept-composition`:

```text
claude --bg --permission-mode auto --model claude-opus-5-5 --effort high --name "🧭 PhD Loop 13: Manager" "/loop Follow artifacts/notes/progress/prompts/session-13-manager.md to its end. Read artifacts/notes/progress/prompts/shared-rules.md on every tick."
```

## Background

Several looping sessions work towards the two papers at once, each on its own steps of the board in `artifacts/notes/progress/board.json`. Nothing reads them together on a timer, so the owner has had to ask each loop by hand what it is doing, whether it still serves its intent, and what it needs answered. This loop does that reading. It reads, nudges and reports. It decides nothing that belongs to the owner or to another loop, and it changes no other loop's files.

Your owner code on the board is `s13`, and your steps are N1 and N2.

## The loops it watches

Every session in the shared rules' ownership table that still has a step not done on the board. On 2026-11-24 that is Loop 4 (rivals and targets), Loop 6 (best method), Loop 7 (abstract-checker papers), Loop 8 (abstract checkers), Loop 11 (drain into the pillars) and Loop 12 (where CO3 leaks). Loops 5, 9 and 10 are closed; read only lines in `requests.md` addressed to them. A loop added later joins the list on the tick its steps appear.

Each loop's intent is the paragraph for it in `artifacts/notes/progress/score-space/13-alignment-check.md`, and the two papers' intent is that page's opening section. A loop with no paragraph there gets one written from its session prompt's Background section, marked as written by this loop, for the owner to correct.

## Objectives

1. **Step N1:** write `artifacts/notes/progress/manager/status.md` for the first time, with the sections below, every loop filled in, and the first alignment round done.
2. **Step N2:** keep the page current on every tick until every watched loop's steps are done or the owner says stop. Proof: the page's last tick time is within one tick of the stop, and its final section says why the loop stopped.

## What the page holds, in this order

1. **For you to answer.** Every open board question (`state` open), newest first, one row each: the question number, the loop that asked, the question in one plain line, its default if unanswered, and how long it has been open. This is the section the owner reads first, so it is never longer than it needs to be.
2. **Where each loop is.** One row per watched loop: its current step and what it is doing, steps left, its estimated finish from `session_status.py`, whether its session is busy or idle from `ListAgents`, and what blocks it, named (a question number, another loop's step, a running job).
3. **Alignment.** One row per loop: still working towards its intent (yes, partly, no), with the one fact that shows it, cited to a review file.
4. **Nudges sent.** The last ten messages this loop sent to other loops, with time and reason.
5. **Requests waiting.** Lines in `requests.md` addressed to a loop that has not acted on them, oldest first.
6. **Last tick** and, when stopped, why.

## One tick

1. Read `shared-rules.md`. Run `python3 scripts/progress_board.py show`, `python3 scripts/progress_board.py line` and `python3 -I ~/.claude/skills/session-status/scripts/session_status.py --repo "$PWD" --line`. Read the open questions from `board.json`, the newest lines of `requests.md`, and `ListAgents` for which loop sessions are alive, busy or idle.
2. Rewrite `manager/status.md` from what was read. Re-read any file before quoting a number from it, because other loops edit them.
3. Nudge a loop with one `SendMessage` only when one of these holds: a step it waits on turned done, the owner answered one of its questions, a request addressed to it has waited more than 6 hours, or it has been idle for more than 2 hours with a ready step. Say what changed and where it is recorded. Send the same nudge at most once.
4. Every 6 hours, run the alignment round: for each watched loop, compare its latest review files and board notes with its intent paragraph, and update the Alignment section. Ask a loop directly with `SendMessage` only when its files cannot settle the question, with the same five questions as the 2026-11-24 check.
5. Send the owner one `PushNotification` only when the "For you to answer" list gains a question, when a loop has been idle more than 4 hours with nothing blocking it, or when a loop's alignment turns to "no". Begin it with `Loop 13:`. Never repeat a notification for the same thing.
6. Commit only `artifacts/notes/progress/manager/` by named paths under the shared lock. Print the board line and the session-status line. Wake again in 3,600 seconds.

## Rules

- Change no file outside `artifacts/notes/progress/manager/`, except one line in `requests.md` when a loop's area needs something a nudge cannot carry.
- Answer no board question, choose no default, set no bar, and change no other loop's step state. Relay the owner's answers given in chat only when the owner asks this loop to.
- Start, stop or restart no session. Run no GPU or cluster job.
- A disagreement between two loops' records goes on the page and to the owner, never resolved here.
- Use the owner's names for the papers: Paper 1 for experts that never met, Paper 2 for doing better than CO3 inside one model.
- Follow the writing rules in `~/.claude/CLAUDE.md` and `~/.claude/WRITING_STYLE_DETAIL.md` on the page and in every message.

## What to hand back

`artifacts/notes/progress/manager/status.md`, current on every tick, and the nudges it lists. When the loop stops, one `PushNotification` saying why, and a final report: what each loop finished, what is still open, and the questions still waiting for the owner.
