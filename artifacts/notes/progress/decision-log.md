# 🧭 Progress: decision log for actions that are hard to undo

Session 2 may delete a file, change a recorded number, touch one of the six fixed folders or
overwrite a picture when the decision is justified. An action counts as justified only when it meets
its row below and has an entry in the log at the end of this page, written before the action runs.

## When each action may run

| Action | It may run when | The way back |
|---|---|---|
| Delete a file on the laptop | A byte-identical copy is verified by checksum on `/datasets` or in the laptop archive, and no page, card, plan, paper or script names the file | the verified copy |
| Change a recorded number | The number was copied wrongly from a data file that plainly holds another value, and both are quoted in the entry | the old value in the entry, and git where the file is tracked |
| Touch a fixed folder | `poe_repair/paths.py` changes in the same commit and every import of it still works | `git revert`, plus the restore manifest |
| Overwrite a picture | The original is first copied to the laptop archive and its checksum matches | the archived original |

## What stays with the owner

- Any row of the [void list](void-list.md), and any case where two sources disagree about what a
  result was.
- Anything in `writing/overleaf-iclr/`.
- Any removal on the cluster.
- Patient, client or secret material.
- A picture held for a subject check, unless the owner clears it.

## What each entry records

Date, action, the exact path, why, the evidence with its source file, the way back, and the check run
afterwards with its result.

## Log

| Date | Action | Path | Why | Evidence | Way back | Check after |
|---|---|---|---|---|---|---|
