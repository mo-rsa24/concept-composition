# Session 11: drain the loops' figures and results into the pillars

This runs as a background session, in the same permission mode as the other loops. To start it by hand from `/home/molef/PhD/concept-composition`:

```text
claude --bg --permission-mode auto --model claude-opus-5-5 --effort high --name "🗂️ PhD Loop 11: Drain The Loops Into The Pillars" "/loop Follow artifacts/notes/progress/prompts/session-11-drain-the-loops-into-the-pillars.md to its end. Read artifacts/notes/progress/prompts/shared-rules.md on every tick."
```

## Background

Loops 5 to 10 keep producing diagrams, picture sheets and results. These sit in each loop's own notes folder, under `artifacts/`, or in `2-classifier-guidance/artifacts/scenes/`, and none of them is on a pillar page. A reader opening `report/` or `context/` today finds none of this work.

On 2026-11-24 the owner asked for every one of these diagrams and good results to be drained into the pillars, checked against the pillar formats, and explained from the overview down to each part with the teaching triad, in organised folders of numbered pages. That request went to Loops 2 and 3, whose board steps are all done and whose sessions no longer run, so this loop takes it over. Your owner code on the board is `s11`, and your steps are Y1 to Y5.

## Objectives

1. **Step Y1, the census.** Run `/artifacts-drain /home/molef/PhD/concept-composition --dry-run`, and list every figure, sheet and result file the loops have made that is on no pillar page, together with the folders the drain does not look in. Write the list to `artifacts/notes/progress/drain/01-census.md`, with each item's loop, what it shows, and the page it should land on.
2. **Step Y2, results into `report/`.** Place each loop's results on the `report/` finding that owns its question: verdicts, the numbers against their bars, the run ids and the picture sheets. A loop's results go into its project's `report/` (`1-lora-correction/report/` or `2-classifier-guidance/report/`), created per `~/.claude/REPORT_FORMAT.md` where it does not exist yet. Use the `drain-cadence` agent, dispatched by `/artifacts-drain`. Only a review file holding a verdict counts as a result, and a step that is still running gets no finding yet.
3. **Step Y3, method pages in `context/`.** Build `context/methods/` as numbered folders, one per method the loops use: for example `01-the-sentence-score-push/`, `02-latent-clip/`, `03-the-readout-checker/` and `04-the-score-path/`. Each folder has numbered pages that go from the overview down to each part: `01-overview.md`, then one page per part. Each page puts the loop's annotated-explainer overview first, then the `clip-contrastive` or network figure that opens that part. Explain each name and equation with the teaching triad (`~/.claude/TEACHING_TRIAD_FORMAT.md`): the picture first, each word of the name taken apart, and each equation shown by what moves when each symbol moves. Use the `names-cadence` agent for the names pass. The order comes from the ladders the loops have already built, such as `2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/README.md`.
4. **Step Y4, check against the pillars.** Run `/context-pulse`, `/report-pulse` and `/xref-pillar` over everything Y2 and Y3 wrote, and fix every finding in this loop's own files. Write the outcome to `artifacts/notes/progress/drain/04-checks.md`.
5. **Step Y5, keep draining.** On every later tick, rerun the census, drain anything new that has a verdict, and extend the method pages when a loop adds a figure. Y5 is done when Loops 5 to 10 have all stopped and a final census finds nothing undrained.

## What to read

- `artifacts/notes/progress/prompts/shared-rules.md`: the rules, read on every tick.
- `artifacts/notes/progress/requests.md`: the request of 2026-11-24 to Loops 2 and 3 (search for "method figures from Loops 7, 8 and 9"), which this loop now answers.
- `~/.claude/ARTIFACT_TREE_FORMAT.md`, `~/.claude/REPORT_FORMAT.md`, `~/.claude/CONTEXT_FORMAT.md`, `~/.claude/PAGE_LOOK_FORMAT.md`, `~/.claude/TEACHING_TRIAD_FORMAT.md`, `~/.claude/CROSS_REFERENCE_FORMAT.md` and `~/.claude/PILLAR_CADENCE_FORMAT.md`: what each page must look like.
- `report/00-INDEX.md` and `context/00-INDEX.md`: what already exists, so nothing is written twice.
- `artifacts/notes/progress/held-pictures.md` and `cleared-pictures.txt`: pictures that failed or passed the subject check.
- `artifacts/notes/progress/void-list.md`: results that may be false. Nothing on it goes onto a page as a finding.
- **The loops' folders:** `artifacts/notes/progress/reproductions/` (Loop 5), `method-search/` (Loop 6), `abstract-checkers/` (Loops 7 and 8), `noise-text-critic/` (Loop 9), `score-space/` (Loop 10) and `targets/` (Loop 4), plus `2-classifier-guidance/artifacts/scenes/` and each project's `artifacts/results/`.

## Rules

- **Read, never rewrite, a loop's folder.** This loop writes only to the pillars and to `artifacts/notes/progress/drain/`. When a loop's file needs a change, such as a wrong caption or a broken link, write one line to `requests.md` addressed to that loop.
- **Never rewrite a result.** A verdict, a number or a bar goes onto a page exactly as its review file records it, with a link to that review. Where two sources disagree, record both, report the disagreement in `requests.md` and in chat, and resolve nothing alone.
- **Check every picture at full size against `scripts/shared/disallowed_subjects.py` and `held-pictures.md` before placing it.** Giraffe × tortoise stays off every page. A picture that fails is listed in `held-pictures.md` and left off.
- **Each picture goes onto a page as a thumbnail linking to the original,** `<a href="<path>"><img src="<path>" width="200" alt="<what it shows>"></a>`, with a caption that names its axes, its data and the meaning of every number.
- **Move a file only from an owner-approved list, with a restore manifest.** Never delete anything.
- **No GPU and no cluster jobs.** This loop reads files and writes pages.
- **Spend no Claude API tokens from the owner's key.**
- **Commit only named paths, with the lock,** and write commit messages with no attribution line.
- **Follow the writing rules in `~/.claude/CLAUDE.md`** on every page.

## What to hand back

- `artifacts/notes/progress/drain/01-census.md`, `04-checks.md` and a final census showing nothing left undrained.
- Each loop's results on its project's `report/` findings, with index rows in `report/00-INDEX.md`.
- `context/methods/`, one numbered folder per method with numbered pages from the overview down, listed in `context/00-INDEX.md`.
- One line per finished step in `requests.md`, addressed to every loop whose work it placed.

## What you will see

Y1 lands as a census table in chat. Y2 and Y3 land one method or one finding at a time, each with its thumbnail. At the end the session reports what was placed and where, what was held back and why, and at most one question. The loop is done when Y1 to Y5 are green.
