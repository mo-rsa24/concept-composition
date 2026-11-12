# 🧑‍🏫 Present the ICLR 56173 revision to the supervisor

**Position in the plan tree:** a standalone plan beside [the review-answering route](route-answer-the-iclr-reviews.md), which owns the paper edits and the replies. This plan only shows them.

## Table of contents

- [Quick context: where you are](#quick-context-where-you-are)
- [Considerations](#considerations)
- [Why this plan exists](#why-this-plan-exists)
- [What happens](#what-happens)
- [Tasks](#tasks)
- [Instructions](#instructions)
- [What has to pass before this runs](#what-has-to-pass-before-this-runs)
- [Outputs](#outputs)
- [Environment Facts This Plan Depends On](#environment-facts-this-plan-depends-on)
- [Next step](#next-step)

## Quick context: where you are

**What this is**

Three reviewers (k4nd, UHz6, PrzE) commented on submission 56173. The revision in `paper/iclr-56173/iclr2027_conference.tex` and the replies in `replies/` answer them. This plan turns that into something the supervisor can read claim by claim, and moves the revision into the supervisor's Overleaf project by drag and drop.

**What it produces**

One annotated figure per group of claims, built from real crops, showing the reviewer's words, the submitted paper's weak passage, the revised answer, and the reply. Each figure is saved as its own PNG and PDF, and all of them are combined into one multi-page PDF that opens with the claim table. Alongside that, an upload bundle that compiles in Overleaf, and the manual steps for uploading it.

## Considerations

**Review sources**

The review crops come from the five OpenReview screenshots in `artifacts/drips/present-revision-to-supervisor/review-screenshots/`. Page numbers come from `Reviews.pdf`, which has no text layer, so its passages are located by tesseract OCR on the 200 dpi renders.

**Pending results**

Table 2 (slot R14, GenEval and T2I-CompBench) and ToMe's row of Table 1 are still rendering. Their figures carry a marked pending panel with the slot name, and no number.

**Overleaf layout**

The bundle goes into a `revision/` folder inside the supervisor's project, so no file of the original is replaced. The new main file is `revision/iclr2027_conference_revision.tex`.

**What the figures may and may not do**

Boxes and labels mark text that is already there. No reply or paper sentence is changed for a figure; new sentences go to Molefe first.

## Why this plan exists

The supervisor has to agree to the replies before they are sent. Reading three reviews, a 13-page submission, a 20-page revision and two reply files side by side is what stops that from happening. One page per claim group, showing all four parts together, makes it a reading of minutes.

## What happens

```
review crop ──▶ submitted weak passage ──▶ revised passage/figure
      └──────────────── reply panel below ────────────────┘
```

## Tasks

For Claude to execute. Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **Run `/verify-plan plans/present-revision-to-supervisor.md` and `/xref-pillar` on it** (at the close of this walk)

### 1. 📋 The claim table

- [ ] **Split the three reviews into single claims, group them, and map each one to the submitted page and the revised tex line**
  - Output: `artifacts/drips/present-revision-to-supervisor/claims.md`
  - Check: every bullet and numbered point of the three reviews appears as one row, each with its Reviews.pdf page.
  - ▶ Next: 4.1

### 2. 🖼️ The figures

- [ ] **2.1 Build the figure for the toy group (the training loss and what the adapter returns) end to end**
  - Renders at 200 dpi with pdftoppm, crops in `artifacts/drips/present-revision-to-supervisor/crops/`, drawing script and figure in `artifacts/scenes/iclr-review-response/`
  - Check: the PNG and PDF exist, and every panel prints its source file and page under it.
  - ◀ Needs: 4.1 · ▶ Next: 4.2
- [ ] **2.2 Build the remaining group figures, a README card per figure, and a contact sheet**
  - ◀ Needs: 4.2

### 3. 📦 The Overleaf bundle and the supervisor document

- [ ] **3.1 Build `overleaf-bundle/revision/` with the renamed tex, every included file and the bibliography, plus a zip**
  - Check: a Tectonic build of the bundle on the cluster gives 20 pages, the main text ends on page 10, 0 undefined references.
  - ▶ Next: 4.3
- [ ] **3.2 Write `overleaf-steps.md`**
  - ▶ Next: 4.4
- [ ] **3.3 Combine the claim table and the figures into `artifacts/scenes/iclr-review-response/supervisor-review-response.pdf`**

## Instructions

For you to follow manually. Do these yourself.

### 4. ✅ Read and approve each stop

- [ ] **4.1 Read the claim table** and say yes, or name the rows to change
  - ◀ Needs: 1 · ▶ Next: 2.1
- [ ] **4.2 Look at the first figure** and say yes, or name what to change
  - ◀ Needs: 2.1 · ▶ Next: 2.2
- [ ] **4.3 Read the bundle's file list** before uploading
  - ◀ Needs: 3.1
- [ ] **4.4 Follow the Overleaf steps**: drag the `revision/` folder into the project, set the main file, and compile
  - What you should see: a 20-page PDF whose main text ends on page 10, with every figure and table showing
  - ◀ Needs: 3.2

## What has to pass before this runs

The revised tex builds today (20 pages, main text ends on page 10, 0 undefined references, per the last commit on `exec/correction-loss-01`).

## Outputs

| What | Where |
|---|---|
| claim table | `artifacts/drips/present-revision-to-supervisor/claims.md` |
| raw crops | `artifacts/drips/present-revision-to-supervisor/crops/` |
| figures, script, cards, supervisor PDF, contact sheet | `artifacts/scenes/iclr-review-response/` |
| upload bundle and zip | `artifacts/drips/present-revision-to-supervisor/overleaf-bundle/` |
| Overleaf steps | `artifacts/drips/present-revision-to-supervisor/overleaf-steps.md` |

## Environment Facts This Plan Depends On

- No system LaTeX on the laptop; builds run with Tectonic on the cluster (`environment/paper.md`).
- Cluster actions go through `scripts/cluster.sh`.

## Next step

The supervisor reads the PDF, and the agreed replies are posted on OpenReview.
