# Prompts: route step 15, explain it back

Two prompts, pasted in order into fresh sessions in `/home/molef/PhD/poe_repair_min`, after steps 5 to 14 have run. They replace the prompt blocks under step 15 of `plans/route-answer-the-iclr-reviews.md`.

## 1. Merge the sessions into one retro

```text
/merge-sessions --retro <the session ids of steps 1 to 14, one per line, listed by the user>
Model: Sonnet, medium effort. Box: 10 minutes of your time. Filing: tied.
Controls: none; the merge runs and leaves one file.
Side routes: none.

Background
This is the retrospective of the work that answered Reviewer PrzE's review of submission 56173: the
ground check, protocol, held-out set, compute plan, corrector proof, scores, SD 1.5 set,
three-concept
test, gated trainings, paper revision and replies. The retro file feeds /drip-retro. The repo is
/home/molef/PhD/poe_repair_min.

Objectives
1. Merge the listed sessions into one retro file in dependency order, checked against the
   transcripts
   and the disk.
2. For each chunk, state the question it answered, the bar written first, and the result.

What to read
- The session transcripts named by the user, and artifacts/drips/answer-iclr-reviews/ROUTE_WALK.md.
- plans/route-answer-the-iclr-reviews.md: the "What we know so far" table, as the spine.
- report/paper-review/board.md and every row page's outcome lines.

What is already known
- Six things must be explainable afterward: the one frozen checkpoint (v58-03-self at step
  30,000); the corrector proof; what each score measures and its blind spot; the held-out groups and
  tuning set, including that the 53 held-out pairs are unseen as pairs only (30 share a concept with
  a training pair, 6 share both); how training targets were chosen (SDXL's own joint-prompt
  prediction, cells kept by eye over rounds v54 to v58, no target from CO3, a few targets showing
  only one animal; evidence: the run's config.json and appendix A); and where a baseline beats the
  adapter if one does.

When something is missing
- If a session id has no transcript, list it and carry on with the rest.
- If a chunk has no recorded bar, say so in the file instead of inventing one.

Rules
- Check every claim against the disk. Write nothing the transcripts and files do not show.

What to hand back
- artifacts/notes/session-merges/2026-11-iclr-reviews-prze-retro.md.

What you will see
The retro file's path and its chunk list, and at most one question.
```

## 2. Walk the retro and explain it back

```text
/drip-retro artifacts/notes/session-merges/2026-11-iclr-reviews-prze-retro.md
Model: Sonnet, medium effort. Box: 30 minutes of your time. Filing: tied.
Controls: each chunk example (the table or image behind it); the last chunk prove (explain the six
things aloud, then check each against its file).
Side routes: after the retro, /quiz-me <the retro file> (a short live quiz on the six things, with
the misses recorded for next time).

Background
The retro file merges the sessions that answered Reviewer PrzE. The reader will meet reviewers and
supervisors who press on exactly the six things listed in the retro. Retention matters more than
speed here: the reader must be able to defend the work without notes.

Objectives
1. Walk each chunk, asking the reader to predict the result before it opens.
2. Record each prediction and whether it held.
3. End by having the reader explain the six things, and check each against its evidence file.

What to read
- artifacts/notes/session-merges/2026-11-iclr-reviews-prze-retro.md, and the board and tables it
  cites.

What is already known
- The reader has seen every result already; the aim now is explaining, not discovering.

When something is missing
- If a result file named in a chunk is missing, say so and skip that prediction.

Rules
- One chunk per round. Prediction first, result second.

What to hand back
- The retro file updated with each prediction and its outcome, and the list of things the reader
  could not yet explain.

What you will see
Along the way: each chunk's question, then the result after your prediction. At the end: which of
the six things you explained correctly, which missed, and at most one question.
```

- `added:` a sixth thing to explain, how training targets were chosen (`artifacts/drips/answer-iclr-reviews/2026-11-11-presentation-retro.md` and the v58-03-self config) · `added:` that the held-out pairs are unseen as pairs only · `added:` the adapter of record by name
- `dropped:` nothing
