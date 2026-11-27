# Session 18: supervise the two setup loops

Launch from the repository root, after Loops 16 and 17:

```text
claude --bg --permission-mode auto --model claude-opus-5-5 --effort high --name "🧭 PhD Loop 18: Setup supervisor" "/loop Follow artifacts/notes/progress/prompts/session-18-setup-supervisor.md to its end."
```

## Background

Two loops write setup code on the laptop at the same time. Loop 16 ("🧪 PhD Loop 16: Paper 1 setup
code") writes `2-classifier-guidance/scripts/paper1/` from `plans/paper-1-clevr-touching/plans/01-setup.md`.
Loop 17 ("🔬 PhD Loop 17: Paper 2 setup code") writes `2-classifier-guidance/scripts/paper2/` from
`plans/paper-2-dials/plans/01-setup.md`. After each task, each loop messages this session with its
files and its check's output. This session reviews the code, answers the questions the owner allows it
to answer, and keeps one page the owner can read. The owner said yes on 2026-11-26 to writing the code;
nothing runs on the cluster.

## Objectives

1. On every tick, read the messages from Loops 16 and 17 and review each task they delivered.
2. For each task, read the code, run `python3 -m py_compile` and the task's laptop check yourself,
   and compare the code with the setting it must serve in the paper's first-result plan.
3. Reply to the loop with "accepted" or with a numbered list of fixes, each naming the file, the line
   and what must change.
4. Answer the loops' open questions on the progress board when the rules below allow it.
5. Keep the page `artifacts/notes/progress/setup/supervisor.md` current.
6. When both loops have finished, run the final review and write the owner's briefing.

## What to read

- Both papers' `README.md`, `plans/01-setup.md` and first-result plan
  (`plans/paper-1-clevr-touching/plans/02-first-result.md`,
  `plans/paper-2-dials/plans/03-identity-dial-first-result.md`): what the code must do.
- `plans/paper-2-dials/plans/02-species-judge.md`: the six kinds, the seven rules and the judges.
- `environment/overview.md` and `environment/hpc/execution-protocol.md`: the environments and the
  checks every job script opens with.
- `artifacts/notes/progress/prompts/session-16-paper-1-setup.md` and
  `artifacts/notes/progress/prompts/session-17-paper-2-setup.md`: the two loops' rules.
- `artifacts/notes/progress/reviewer/consensus.md`: the rules every new design follows.

## What is already known

- The Blackwell nodes mscluster110 and 112 need `co3_bw`; `co3` silently runs on the CPU there.
- Large outputs belong on `/datasets`; `/home-mscluster` once filled and killed checkpointing.
- A disabled adapter still acts, so "off" is an untouched U-Net in another process.
- The earlier reward fine-tune's fidelity term was silently off until commit `a72f748`.
- The empty-scene model for Paper 1 has no recorded cluster path.

## When something is missing

- **A question the rules let you decide:** decide it, record it on the board with
  `python3 scripts/progress_board.py answer <id> "Decided by Loop 18 for the owner: <choice>, because <reason>. The owner may overrule by answering again."`,
  and message the asking loop.
- **A question that is the owner's:** leave it on the board, list it on the page under "Waiting on
  you", and tell the asking loop to carry on with work that does not depend on it.
- **A loop silent for more than 2 hours with tasks left:** send it one message naming its next task.

## Rules

- **You decide** code-level choices that are reversible and change no recorded bar, verdict or run
  cost: argument names, file layout, test stand-ins, which of two plan wordings the code follows when
  the first-result plan is clear.
- **The owner decides** anything that runs on the cluster or RunPod (weight fetches, picture moves,
  CPU or GPU jobs), the three Paper 2 step 0 questions, where the empty-scene model lives, any change
  to a bar, a sample size or a stage, commits, pushes and deletions.
- Write only `artifacts/notes/progress/setup/supervisor.md`, board answers, and messages. Never edit
  the loops' code yourself; ask the loop to.
- Check on every tick with `git status --short 2-classifier-guidance/scripts/ tests/ plans/` that
  each loop wrote only in its own area, and stop a loop's task with a message if it crossed.
- Run nothing on the cluster. Commit, push and delete nothing.
- Every line on the page names work by what it does, never by a step code alone.

## What to hand back

- `artifacts/notes/progress/setup/supervisor.md`, holding, in order: one line per loop on where it
  stands; each task with its review result; the decisions made for the owner; "Waiting on you", the
  owner's open questions; and, at the end, the suggested commit (the file list and a one-line message).
- The final review on the same page: every file compiles, every laptop test passes, and each
  first-result setting appears in the code where it should, as a table of setting, file and line,
  and yes or no.

## What you will see

Along the way, after each review, a short note in chat names the task, the result, and any decision
made.

At the end, a closing report in chat:
1. what was asked;
2. what was done, with the page's path;
3. the outcome, as tasks accepted out of 12 (8 for Paper 1, 4 for Paper 2) and the final review's
   table;
4. what is waiting on the owner;
5. at most one question for the owner.

The owner judges from the page. This loop is done when both loops have stopped, the final review is
on the page and the suggested commit is listed; then it stops its own loop.
