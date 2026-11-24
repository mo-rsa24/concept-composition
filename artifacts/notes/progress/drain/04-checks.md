# 🗂️ Drain checks: the pillar audit of the pages Loop 11 wrote

This page records step Y4: the pillar checks run over the 48 pages steps Y2 and Y3 wrote, listed in
[the page list](y2-y3-pages.txt), and what became of every finding. The 48 are 34 pages under
`context/methods/` (the methods index and eight method folders) and 14 report pages (13 findings and
`report/00-INDEX.md`).

## Words this page uses

**Finding** here means one thing an audit flagged, not a report finding. Each has an id: C for the
context pillar, R for the report pillar, X for cross-references.

**Owner's call** means the formats disagree with each other, or a fix would change how a recorded
result reads, so the owner decides. Those were asked as board question 81, which the manager session (Loop 13) answered for the owner under the shared rules: all ten defaults. Items 1, 2 and 4 edit this repo's `CLAUDE.md`, and the owner gave that yes directly on 2026-11-24.

## How the checks ran

| Check | How | When |
|---|---|---|
| `/context-pulse`, default audit | the `pillar-cadence` agent, read only, over `context/methods/` and its rows in `context/00-INDEX.md` | 2026-11-24 |
| `/report-pulse`, default audit | the same agent, over the 13 findings, their three group folders and their rows in `report/00-INDEX.md` | 2026-11-24 |
| `/xref-pillar` | the `xref-cadence` agent over all 48 pages, writing links inside the pillars only | 2026-11-24 |

**What passed outright.** Every method page has a contents list, breadcrumbs, the built-to stamp, a
provenance table and marks. Every relative link and anchor on the 47 pages in scope resolves. No page
uses a dash as a clause separator. The discovery routes to both pillars are in place.

## The findings

| Id | What is wrong | Where | Who fixes it | State |
|---|---|---|---|---|
| C1 | `context/methods/` uses an overview-then-parts page shape that CONTEXT_FORMAT does not have | the folder | question 81, with the owner's yes for `CLAUDE.md` | fixed, `7f256f18` |
| C2 | seven method pages carry runnable commands, which CONTEXT_FORMAT bans and the worked-example rule asks for | `01/03`, `02/04`, `04/03`, `06/04`, `07/03`, `07/04`, `08/02` | question 81, with the owner's yes for `CLAUDE.md` | fixed, `7f256f18` |
| C3 | run status (never ran, missed its bar) sits on method pages and in the methods index | `00-INDEX.md` rows 31 to 38, `04/01`, `04/04`, `05/01`, `07/01` | question 81 | fixed, `1f349d62` |
| C4 | the stated mark is used for numbers copied from a named review without reopening it | 325 marks across the folder | question 81, with the owner's yes for `CLAUDE.md` | fixed, `7f256f18` |
| C5 | four diagram slots and the load-bearing inferred claims are not listed on `context/00-INDEX.md` | Pictures still missing, Still open | this loop | fixed, `128e1a30` |
| C6 | the methods row on `context/00-INDEX.md` still says one method is written and seven planned | `context/00-INDEX.md` line 50 | this loop | fixed, `128e1a30` |
| C7 | opening pictures are embedded at 720 px wide, against the 200 px house thumbnail | all 33 method pages | question 81 | fixed, `1f349d62` |
| C8 | 49 chunks hold more than one bold phrase, mostly name blocks bolding each word on one bullet | 26 method pages | this loop | fixed, `128e1a30` |
| C9 | step labels (V1 to V3, W1b to W4, X4 to X8) used without saying what the step did | about 200 uses in 15 files | this loop | fixed, `128e1a30` |
| C10 | a file name used as link text | `04/02` line 56 | this loop | fixed, `128e1a30` |
| C11 | one lookup row covers all eight methods on `context/00-INDEX.md` | line 32 | this loop | fixed, `128e1a30` |
| C12 | `MASTER_PLAN.md` has no Learning Coverage section | `MASTER_PLAN.md` | outside this loop's files; `/learning-pulse` | reported |
| R1 | three finding titles use the reviews' own verdict words instead of support, null or inconclusive | the rivals, environment and regions findings | owner's call, item 6 | settled by question 81 item 6: the reviews' own verdict words stay on the title lines |
| R2 | five findings carry two verdicts | rivals, checker learning, five properties, regions, fairness | owner's call, item 7 | settled by question 81 item 7: each finding keeps its review's verdicts side by side |
| R3 | 26 embedded pictures sit in the loops' progress folders, which have no cards | 12 findings | owner's call, item 8 | question 81 item 8: each loop is asked in `requests.md` to card its step folders |
| R4 | four rungs have a number and neither a figure nor a slot | removal rungs 3 to 5, push rung 5 | this loop | fixed, `eda9d9a4` |
| R5 | the report index's Figures still missing list carries none of the 20 slots in these findings | `report/00-INDEX.md` | this loop | fixed, `eda9d9a4` |
| R6 | the question-group rows use folder paths as link text | `report/00-INDEX.md`, the three new rows | this loop | fixed, `eda9d9a4` |
| R7 | the Cross-references section on one finding has no breadcrumb or contents entry | environment finding | this loop | fixed, `eda9d9a4` |
| R8 | "(board question 75)" sits in a finding title | removal finding and its index row | this loop | fixed, `eda9d9a4` |
| R9 | claims run past four sentences | most of the 13 findings | this loop | fixed, `eda9d9a4` |
| R10 | seven findings cite the board step and design instead of a plan file | seven findings | owner's call, item 9 | settled by question 81 item 9: the board step and its design count as the plan |
| R11 | most numbers are marked stated because the cluster result files were not opened | 184 stated against 93 verified | this loop, read only on the cluster | fixed, `8f2dea1b`: verified 93 to 198, stated 184 to 93; nine page-against-file mismatches recorded on their pages and sent to the loops |
| R12 | no finding has a Depends on section, and few link their method pages | 13 findings | the cross-reference pass | fixed, `f3cee98a` |
| R13 | the repeatability check is filed as a finding rather than an environment fact | environment finding | owner's call, item 10 | settled by question 81 item 10: the repeatability check stays a finding |
| C13 | a second audit (Loop 4's, line 259 of `requests.md`) found three Cross-references sections on method overviews with no breadcrumb or contents entry | `02/01`, `05/01`, `08/01` | this loop | fixed, `97bbb1c1` |
| C14 | the same audit names about 15 inferred claims with no Still open row and three diagram slots with no save path | `05/01`, `05/02`, `05/03`, `04/04`, `08/02`, `02/02`, `02/03`, `06/05` and others; slots on `02/01`, `06/02`, `08/03` | owner's call: the claims were judged not load-bearing on 2026-11-24 and are listed in this loop's report; a slot's save path follows question 81 item 8 | save paths fixed, `1f349d62`; the claims stay off Still open as judged, listed for the owner |
| R14 | as C12 | `MASTER_PLAN.md` | outside this loop's files | reported |

## Seen in passing, outside this loop's files

`report/00-INDEX.md` as a whole has no contents list, breadcrumbs or built-to stamp, and four links
that were broken before this loop touched it: line 37 (a blend-gallery README), line 134 (the
energy-penalty folder), and lines 292 and 294 (two figure explainers).

## Outcome

Every one of the 28 findings is fixed or settled. 20 were fixed on Loop 11's pages and C14 in part (its slots name save paths; its 15 inferred claims stay off Still open as judged), 4 were settled by question 81 as kept (R1, R2, R10, R13), R3's cards were requested from the loops that own the folders, and C12 and R14 (a Learning Coverage section in `MASTER_PLAN.md`) sit outside this loop's files and are reported to `/learning-pulse`.
