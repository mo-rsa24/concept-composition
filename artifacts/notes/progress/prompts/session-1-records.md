# Session 1: records

Paste this one line into a new Claude Code session opened at `/home/molef/PhD/concept-composition`:

```text
/loop Follow artifacts/notes/progress/prompts/session-1-records.md to its end. Read artifacts/notes/progress/prompts/shared-rules.md on every tick.
```

## Background

The repository `/home/molef/PhD/concept-composition` holds five research projects and 84 Claude sessions
of work. Many changed files are uncommitted. This session saves them, closes the sessions that matter,
indexes the rest and archives the ones that are safe. Your owner code on the board is `s1`.

## Objectives

1. Step A2: commit the changed paths in named batches, one project or purpose per batch.
2. Step B1: make sure each of the 11 chosen sessions has a closing note.
3. Step T1: write a one-line index of the other 73 sessions.
4. Step T2: archive the sessions that are safe to close.
5. Step B2: run `/absorb-sessions` and clear the stale records it finds.

## What to read

- `artifacts/notes/progress/prompts/shared-rules.md`: the rules, read on every tick.
- `artifacts/notes/progress/prompts/closing-prompts.md`: the 11 sessions and what each note must hold.
- `~/.claude/skills/absorb-sessions/SKILL.md`: the census and the record sweep.
- `CLAUDE.md` and `~/.claude/CLAUDE.md`: the repository's folders and the writing rules.
- `~/.claude/projects/-home-molef-PhD-concept-composition*/`: the session transcripts.

## What is already known

- The 11 sessions are `68c9e75c`, `1c41950d`, `712164c1`, `d40b9299`, `276dcd9e`, `862f382f`, `078babda`,
  `bcb64c30`, `49d5b78b`, `3975729c` and `8959e932`.
- On 2026-11-21, 63 of the other 73 sessions had no unsaved file, 8 had unsaved files and 2 were active.
  That check cannot see git-ignored outputs.
- Session `078babda` wrote to `/home/molef/PhD/medical-imaging/compositional_diffusion`, a separate repository.
- Session `276dcd9e` wrote one file that no longer exists. Trace it and report it.
- `.gitignore` already keeps figures, builds and PDFs of the Overleaf export out of git.

## When something is missing

- **A closing note is missing.** Write it from the session's transcript and mark it `written from the
  transcript`.
- **A session wrote to its transcript in the last 30 minutes.** Treat it as live. Skip it and return later.
- **A file over 50 MB would be committed.** Leave it out and list it for the user.
- **A changed file's purpose is unclear.** Put it in a batch named `needs a look`, leave it uncommitted and
  ask at the end of A2.

## Rules

- Show the proposed A2 batches once as a table, with counts per batch, and wait for the user's yes.
- Archive one session first by moving its `.jsonl` file and its folder to
  `~/.claude/projects-archive/<project folder>/`. Ask the user to confirm the session picker still works.
  Archive the others only after that yes.
- A session is safe only when it is not live, every file it wrote exists, and none of them is uncommitted.
- Write the restore command for each archived session in `artifacts/notes/session-closes/session-index.md`.

## What to hand back

- The commits, listed by hash and batch.
- `artifacts/notes/session-closes/<session id>.md`: 11 notes saying what the session found, where its
  files are, what is unfinished and which paper it feeds.
- `artifacts/notes/session-closes/session-index.md`: 73 rows with title, branch, files written and verdict.
- One dated `DRIFT_LOG.md` entry from `/absorb-sessions`.

## What you will see

Along the way, the session shows the commit batches and asks for one yes, then archives one session and
asks you to check the picker. At the end it reports what was asked, what was done with paths, the outcome
against each step's proof, what is open, and at most one question.

You judge it from `git status`, the 11 notes and the index; it is done when A2, B1, T1, T2 and B2 are green.
