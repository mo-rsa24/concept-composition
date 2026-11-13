# Walk: does one prompt already say "one animal"?

Mode: `--request`. Plan:
[the plan](../../../../3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/today-does-one-prompt-already-say-one-animal.md)

## Where the walk is

Piece 1 of 5, build done and committed, encoder pass not yet run. The worktree went clean and the
launch still failed: the cluster checkout sits on `af849a2e` of another branch and holds 11 untracked
`artifacts/drips/` folders whose paths are tracked on this branch, so `git checkout` there refuses to
overwrite them. That is a repo-level divergence between the two checkouts, not a condition of this
walk, and `/reconcile-machines` owns it.

Branch `exec/does-one-prompt-already-say-one-animal`, first commit `46026ce4`.

## The stop point

**Bar.** Both claims carry a verdict against the criteria written into the plan before anything ran,
in a finding at `report/what-poe-loses-when-it-composes/does-one-prompt-already-say-one-animal.md`,
with the axis figure and the orthogonal-share numbers in their sidecar files.

**Time box.** 2 sittings, about 2.5 hours.

| Sitting | Started | Ended |
|---|---|---|
| 1 | 2026-11-04 | open |

## The toy

"a cat", "a dog", "a cat and a dog" and "" through SDXL's two text encoders: four embeddings, every
later number quoted back against them.

## Decisions

| # | Decision | Chosen | The options there were | Who chose | Why |
|---|---|---|---|---|---|
| 1 | where the plan lives | `3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/` | scope 10, or scope 06 where the CLIP-free finding sits | default accepted | scope 10 owns the language-space claims |
| 2 | how the one-vs-many axis is built | matched prompt pairs over the pool's 81 nouns | matched pairs, count words alone, a trained linear probe | default accepted | a displacement over many nouns averages out what is specific to one noun |
| 3 | the predictive half of claim two | quote the two existing nulls, never re-measure | quote them, or re-measure with a fresh pre-registered bar | default accepted | the bar already failed; re-running it in a new geometry after seeing the answer is what `EXPERIMENT_CONVENTIONS.md` forbids |
| 4 | which claim is load-bearing | claim one | stated in the request as a question for the plan to answer | Claude, open to veto | claim one is the first objection a reviewer raises and nothing in the repo answers it; claim two's measurement half is already null, so its remainder is framing |
| 5 | both suggested improvements | kept | droppable | default accepted | the pool-wide read stops the answer being a cat-by-dog anecdote; the third encoder family separates "CLIP" from "any encoder" |
| 6 | how to reach the cluster with another session's files dirty | wait for that session to commit | wait, send the bytes over unsynced, or commit their files | user, answered `e` | keeps the run on the repo's pinned launch path, so the run's provenance holds |

## Ticked this session

Nothing yet. Task 1.1 is written and committed but not ticked: its verification is the held-out
ratio, which has not run.

## Repairs made to the plan

- Instruction 6's command pointed at `/datasets/.../outputs/one_animal_axis/`. The script writes to
  `5-text-embedding-third-expert/artifacts/results/what-the-text-already-says/does-one-prompt-already-say-one-animal/`, matching `and_vector_norms.py`, so the
  command now reads that path through `cluster.sh sh`.

## Out of the session

| Kind | Id | What it serves | State |
|---|---|---|---|
| background watch | `b9g4vfwzd` | waited for the worktree to go clean | returned clean on the first check |

## Closing pass still owed

Every part of it: the diagrams drain, the artifact filing with its card, the report finding, the plan
move by how finished it is, `sync-plan-tree`, the playground mark, and the cadence dispatch.

## How to approach this next time

- An axis built from a difference of embeddings needs a same-shape random floor before it means
  anything; the raw projection is what the arithmetic produces by chance in a wide space.
