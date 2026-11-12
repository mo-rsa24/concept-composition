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

- [x] ✅ **Split the three reviews into single claims, group them, and map each one to the submitted page and the revised tex line**
  - Output: `artifacts/drips/present-revision-to-supervisor/claims.md`
  - Check: every bullet and numbered point of the three reviews appears as one row, each with its Reviews.pdf page.
  - ✓ verified 2026-11-12: 35 claims in 12 groups, PrzE 5, UHz6 13, k4nd 17
  - ▶ Next: 4.1

### 2. 🖼️ The figures

- [x] ✅ **2.1 Build the figure for the toy group (the training loss and what the adapter returns) end to end**
  - Renders at 200 dpi with pdftoppm, crops in `artifacts/drips/present-revision-to-supervisor/crops/`, drawing script and figure in `artifacts/scenes/iclr-review-response/`
  - Check: the PNG and PDF exist, and every panel prints its source file and page under it.
  - ✓ verified 2026-11-12: g07 PNG 5890×3166 and PDF written, seven panels each with source and page (commit b2f7abde)
  - ◀ Needs: 4.1 · ▶ Next: 4.2
- [x] ✅ **2.2 Build the remaining group figures, a README card per figure, and a contact sheet**
  - ✓ verified 2026-11-12: 12 figures as PNG and PDF, 12 cards, contact sheet; every panel prints its source and page
  - ◀ Needs: 4.2

- [ ] **2.3 Rebuild Figure 10 (slot R9) from the scorers Tables 1 and 3 use, in place of the by-eye failure labels**
  - Images: the 848 window renders (adapter on steps 0 to 10 and 0 to 25) scored by `scripts/review_score_blipvqa.py`, `review_score_clip_ir.py` and `review_score_comp.py` from `rows_windows.csv` (`scripts/review_window_rows.py`); adapter off and all 50 are Table 3's PoE and ours rows, already scored.
  - The figure: x axis the four settings (off, steps 0 to 10, 0 to 25, all 50 of 50), 424 images each (53 held-out pairs, seeds 9 to 16, the same starting latent per seed); BLIP-VQA mean per held-out group with a 95% interval from resampling pairs then seeds; Comp. on the animal pairs; CLIP score and ImageReward.
  - Reading, written before any window score exists: steps 0 to 25 and all 50 overlap on BLIP-VQA in every group, so the UHz6 Q3 reply says the first 25 steps give the same result; steps 0 to 10 below both in a group, so the reply says ten steps are too short for the learned correction there; any other outcome is reported as it lands and the reply changes to match.
  - Caption, reply wording and figure go to Molefe before the paper or a reply changes.

### 3. 📦 The Overleaf bundle and the supervisor document

- [x] ✅ **3.1 Build `overleaf-bundle/revision/` with the renamed tex, every included file and the bibliography, plus a zip**
  - Check: a Tectonic build of the bundle on the cluster gives the same pages as the repo build, the main text ends on page 10, 0 undefined references.
  - Builder: `python3 artifacts/drips/present-revision-to-supervisor/make_overleaf_bundle.py`; rerun once Table 2 (slot R14) lands.
  - ▶ Next: 4.3
- [x] ✅ **3.2 Write `overleaf-steps.md`**
  - ▶ Next: 4.4
- [ ] **3.3 Write `review.tex`: the claim groups in order, each quoting the reviewers' weaknesses and questions, then what the revision did, then the reply we plan to send, with the group's figure**
  - Output: `artifacts/scenes/iclr-review-response/review.tex`, built with Tectonic on mscluster
  - Check: every one of the 35 claims in the claim table is quoted once
- [ ] **3.4 Combine the claim table and the figures into `artifacts/scenes/iclr-review-response/supervisor-review-response.pdf`**

## Instructions

For you to follow manually. Do these yourself.

### 4. ✅ Read and approve each stop

- [x] ✅ **4.1 Read the claim table** and say yes, or name the rows to change
  - ✓ confirmed by hand 2026-11-12: "done. Happy with the claim table"
  - ◀ Needs: 1 · ▶ Next: 2.1
- [x] ✅ **4.2 Look at the first figure** and say yes, or name what to change
  - ✓ confirmed by hand 2026-11-12: "done and happy"
  - ◀ Needs: 2.1 · ▶ Next: 2.2
- [x] ✅ **4.3 Read the bundle's file list** before uploading
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
