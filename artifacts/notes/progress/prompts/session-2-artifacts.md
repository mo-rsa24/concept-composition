# Session 2: artifacts

Paste this one line into a new Claude Code session opened at `/home/molef/PhD/concept-composition`:

```text
/loop Follow artifacts/notes/progress/prompts/session-2-artifacts.md to its end. Read artifacts/notes/progress/prompts/shared-rules.md on every tick.
```

## Background

The five projects hold 33,356 artifact files (20.4 GB), and only 4,880 are in git. None has ever been
drained, so most results can only be found by opening folders. This session turns them into report pages
built to the current standards, names the results that are false, and lists what is safe to retire. Your
owner code on the board is `s2`. Your first step waits until step A2 is green.

## Objectives

1. Step B3: drain each project's `artifacts/` into report pages to the current page standard.
2. Step B4: write the void list, naming every result known to be false, with its reason and source.
3. Step T3: write the prune list of files nothing refers to, with size and reason.
4. Step T4: after the user approves the lists, move those files to `_archive/` with a restore manifest.

## What to read

- `artifacts/notes/progress/prompts/shared-rules.md`: the rules, read on every tick.
- `~/.claude/skills/artifacts-drain/SKILL.md` and its `references/dispatch.md`, `enrich-page.md` and
  `pillar-rules.md`: how a drain and an enrichment run.
- `~/.claude/PAGE_LOOK_FORMAT.md` and `~/.claude/REPORT_FORMAT.md`: how a report page must look.
- `~/.claude/ARTIFACT_TREE_FORMAT.md`: the artifact folders, cards and verdicts.
- `~/.claude/agents/drain-cadence.md`: the agent that writes the pages.
- `environment/known-failures.md`, each project's `report/` and each plan's review file: known faults.
- `artifacts/notes/session-closes/`: the closing notes, as Session 1 writes them.

## What is already known

- The projects to drain are `1-lora-correction` (7,873 result files), `2-classifier-guidance` (13,749),
  `3-understanding-compositionality` (2,393), `4-inclusive-captions` (210),
  `5-text-embedding-third-expert` (76) and the repository root (123).
- `4-inclusive-captions` has no `report/` folder, and the drain refuses a project without one.
- The artifacts hold 25,120 images, so looking at each one is not affordable.
- The drain may create and edit cards and written notes under `artifacts/`. It moves a file only from a list the user approved, with a restore manifest.
- This session may delete a file, change a recorded number, touch a fixed folder or overwrite a picture only under the conditions in `artifacts/notes/progress/decision-log.md`, with an entry written first.
- Step O1 follows `artifacts/notes/progress/organise-plan.md`, which organises `2-classifier-guidance/artifacts` and `4-inclusive-captions/artifacts`.
- Six folders must stay where they are; `shared-rules.md` lists them.
- Known faults include a batch-size timing comparison marked void in `environment/hpc/throughput.md`, and
  the fidelity term that was silently off in reward fine-tuning before commit `a72f748`.

## When something is missing

- **A project has no `report/`.** Run `/report-pulse --build` on it. If that cannot finish, skip the project
  and report it.
- **A folder holds thousands of near-identical renders.** Place one representative sheet per grouping, and
  use `sweep.py --hold` on the raw renders, with the reason on the grouping's card.
- **It is unclear whether a result is false.** Mark it `needs your call` with both sides, and never void it
  alone.

## Rules

- Drain one project's smallest grouping first. Send the page to the user and wait for one yes on its look.
- Then work one question or ticket at a time, with agents of ten pages or fewer.
- Enrich `report/` pages only. Session 3 enriches `context/`, `environment/` and `runbook/`.
- Run `names-cadence` after enrichment on the same folder, never at the same time.
- Commit report pages in named batches under the lock.
- A file a pillar page references is never pruned.

## What to hand back

- Report pages under each project's `report/`, each passing `check_enrich.py`.
- `artifacts/notes/progress/void-list.md`: result, why it is false, source, and who decides.
- `artifacts/notes/progress/prune-list.md`: path, size, why nothing needs it, and the proposed move.
- `artifacts/notes/progress/restore-manifest.tsv`: old and new path for each file moved.

## What you will see

Along the way, the session shows one pilot page and asks for a yes, then sends the void and prune lists
for your approval. At the end it reports what was asked, what was done with paths, the outcome against
each step's proof, what is open, and at most one question.

You judge it from the report pages and the two lists; it is done when `sweep.py` prints `new 0 changed 0`
for every project and T4 is green.
