# Closing prompts for the 11 sessions

Each session knows things its transcript only hints at, so it writes its own closing note. Open the
session, paste its block and let it finish. A session you skip still gets a note: Session 1 writes it from
the transcript and marks it so.

## The prompt

Replace `<id>` and `<owed>` with the row below.

```text
Please close this session so its work is not lost. Do not start new work or launch anything.

Background: the project is moving to a two-paper plan in plans/route-two-papers.md. This session's work
must be findable after it closes.

Objectives:
1. Write artifacts/notes/session-closes/<id>.md under /home/molef/PhD/concept-composition.
2. Say in it what this session set out to do, what it found with the numbers, and where each result lives.
3. List every file it made, and mark each one committed, uncommitted or outside this repository.
4. Say what is unfinished, the exact next step, and which paper it feeds: Paper 1, Paper 2, the ICLR paper,
   or none.
5. Finish this owed item: <owed>.

Rules: delete nothing, commit nothing, and change no verdict. If something you remember disagrees with a
file, write down both. Follow the writing rules in ~/.claude/CLAUDE.md.

When done, print the note's path and one line saying what is unfinished. I will judge it by whether a stranger
could pick up this work from that note alone.
```

## The sessions

| Session | What it is | `<owed>` |
|---|---|---|
| `1c41950d` | SDXL cat × dog checker | State plan 31's open round and the CO3 ideas it proposed |
| `712164c1` | Route for binding each body to its concept | Record the settled rounds and the seven proposed board rows |
| `862f382f` | Readout Guidance paper read | Record where the read stopped |
| `276dcd9e` | Guidance on our cat and dog photos | Record the outage, both runs' last steps, the resume steps and the missing file |
| `d40b9299` | The MNIST route | None |
| `078babda` | Chest X-ray finding pairs | Record the chosen pairs, the open test-set question and the medical repository's uncommitted files |
| `bcb64c30` | MNIST inclusive-captions pilot | State the verdict still owed |
| `49d5b78b` | Inclusive-captions toy results and slides | List the results that have no card yet |
| `3975729c` | Reframing MNIST around inclusive captions | None |
| `68c9e75c` | Idea walk, beyond the product of experts | Record where the walk stopped |
| `8959e932` | Experts at inference, E0e and this planning | The E0e by-eye count on pairs 3-8 and 4-9, and the paused idea walk's claims |
