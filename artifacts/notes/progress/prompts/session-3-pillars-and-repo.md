# Session 3: pillars, then the join

Paste this one line into a new Claude Code session opened at `/home/molef/PhD/concept-composition`:

```text
/loop Follow artifacts/notes/progress/prompts/session-3-pillars-and-repo.md to its end. Read artifacts/notes/progress/prompts/shared-rules.md on every tick.
```

## Background

The repository's pillars (`context/`, `environment/`, `runbook/`, `report/`) have detailed standards and
have drifted behind the work. This session brings them current, maps the cluster's storage and tidies the
repository. It then runs the join automatically: as soon as steps B2, B4 and T5 are green, it writes what
we know, scouts the field, fixes the benchmark spec, drafts the two problem statements and builds the paper
repository. Your owner code on the board is `s3`. Your first step waits until step A2 is green.

## Objectives

1. Step B5: bring `context/`, `environment/` and `runbook/` to their current formats and map cluster storage.
2. Step T5: run `/tidy-repo` so the repository root holds only known folders.
3. Step B6: write one page of what we know, with each finding, its evidence and its status.
4. Step C1: scout the field twice, with Claude and with GPT-6 Astra separately, and check each row in the paper.
5. Step C3: write the benchmark spec from `artifacts/notes/progress/targets/numbers.csv`, which Session 4 reads from the rivals' own papers.
6. Step C4: draft the two problem statements and run `/pressure-test` on each.
7. Step D1: build the paper repository so one command reprints the MNIST table.

## What to read

- `artifacts/notes/progress/prompts/shared-rules.md`: the rules, read on every tick.
- `plans/route-two-papers.md`: the two papers, their bars and the datasets.
- `artifacts/notes/progress/rival-datasets-first-pass.md`: what each rival benchmarks on, to verify in C3.
- `~/.claude/CONTEXT_FORMAT.md`, `~/.claude/ENVIRONMENT_CONTEXT_FORMAT.md`, `~/.claude/RUNBOOK_FORMAT.md`,
  `~/.claude/PAGE_LOOK_FORMAT.md` and `~/.claude/STANDARDS.md`: the pillar standards.
- `~/.claude/skills/artifacts-drain/references/pillar-rules.md`: the enrichment rules for each pillar.
- `~/.claude/skills/tidy-repo/SKILL.md`, `~/.claude/skills/pressure-test/SKILL.md` and
  `~/.claude/skills/codex-author/SKILL.md`.
- `artifacts/notes/session-closes/`, `report/`, each project's `report/`, and Session 2's void list.

## What is already known

- Paper 1 composes separately trained experts. E0e read 1,200 of 1,280 clean pairs by the judge on 20 unseen
  MNIST pairs, with fixed halves, in pixel space. No shared latent space has been built.
- Paper 2 restores plurality in one model. On 128 cat × dog seeds, by eye, plain PoE got 0, the checker push
  54, the joint prompt 69 and CO3 101.
- "Without the joint prompt" is not a contribution for Paper 2, because the prompt is available.
- `Diffusion Domain Expansion` (arXiv 2605.23275) is a direct rival for Paper 1 and is read first.
- The checker's 4,400-picture pool came from Attend-and-Excite renders, so a checker trained on it cannot be
  scored fairly on those prompts.
- GPT-6 Astra is reached through the `codex` command, already configured on this machine.
- Bradley's CLEVR training reportedly took about 16 hours on 32 A100 cards, so smaller CLEVR versions come first.

## When something is missing

- **Step B6 is waiting on another session.** Run `show`, then schedule the next wake for 1,800 seconds.
- **The cluster is unreachable.** Write the storage map from `environment/` and mark it unverified. Finish it
  when the cluster returns.
- **Codex is unavailable.** Run the second scout with a separate Claude agent, and say so in the table.
- **A paper's PDF will not load.** Try its HTML page, then mark that row `could not read`.

## Rules

- Enrich `context/`, `environment/` and `runbook/` by the steps in the drain skill's `dispatch.md`.
- In `context/`, leave out any inferred claim or write it as a gap line.
- Keep every `Entry ID` heading and anchor in `environment/`.
- Give the two scouts the same brief, and let neither see the other's answer before the merge.
- Leave results on Session 2's void list out of the findings in `what-we-know.md`, and list them apart.
- Show `what-we-know.md` to the user when B6 lands.
- Stop after C4 and show the two statements with their verdicts before starting D1.
- Create the paper repository at `/home/molef/PhD/papers/composing-experts-that-never-met/`, with local git
  only. Copy the code it needs, and leave the originals in place.

## What to hand back

- Updated pillars, each passing its audit, and the storage map in `environment/storage.md`.
- `artifacts/notes/progress/what-we-know.md`, `field-table.md` and `benchmark-spec.md`.
- `artifacts/notes/progress/problem-statements.md`, with each `/pressure-test` verdict.
- The new paper repository, whose one command reprints 1,200 of 1,280 from the saved counts.

## What you will see

Along the way, the session shows the page of what we know, then the two problem statements for your yes. At
the end it reports what was asked, what was done with paths, the outcome against each step's proof, what is
open, and at most one question.

You judge it from the pillars, the statements and the one command; it is done when B5, T5, B6, C1, C3, C4 and
D1 are green.
