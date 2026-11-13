# Retrospective: the paper's presentation pass, between route steps 4 and 5

Walk: [the step 4 walk map](EXECUTE_WALK-step-4.md), continued in the same session. Working copy: `paper/iclr-56173/`, built with Tectonic on the cluster. Commits `cb39179d` to `d2e9c333`.

## What was asked

The paper is judged on presentation, so every figure has to read as part of one finished paper: the same face and size of lettering, full width with no side gaps, a main text that runs text, figure, text, and an appendix in order.

## What changed

**One lettering rule, measured.** Figures 2, 3, 4, 6 (the limitations, stacked) and 12 (the rank and steps sweep) are drawn at 5.5 in in Liberation Serif, labels 8 pt, titles 9 pt, and `1-lora-correction/scripts/figure_text_audit.py` passes each one. Figure 2 is redrawn from the 1024 px renders (it had been saved at 100 dpi, 92 px per tile) with seven strength columns. The audit now measures rotated labels across their width.

**Tools.** `1-lora-correction/scripts/paper_font.py` sets the face for every figure script; `1-lora-correction/scripts/paper_tile_figure.py` gained two-line column labels, a note under each tile, a footnote line and a panel letter; `1-lora-correction/scripts/figure7_limitations.py` draws the limitations figure side by side or `--stacked`.

**Main-text layout.** Figures at the top of a page, at most two per page, at least a fifth of every page text; each section's figures defined at the start of the section; Tables 1 and 2 in one float; a barrier before the Discussion so no main-text figure prints after it. The rank and steps ablation moved to appendix C.5 and is still cited from Section 4.1. Empty space at page feet in the main text went from 10.6 in to 1.3 in, and 0.7 in of that is the old gap under Figure 1.

**Appendix.** Section C is six subsections by reviewer question (benchmark prompts, held-out pairs, SD 1.5, samplers, how the adapter behaves, three concepts), each with a lead-in paragraph under its heading and then its own figures.

## Before and after

| | Before | After |
|---|---|---|
| figures passing the 8 pt audit | 3 of 8 | 6 of 8 |
| main-text empty page feet | 10.6 in | 1.3 in |
| main text ends | page 11, with a page of only figures | page 11, text on every page |
| appendix order | figures ahead of their text, no grouping | heading, lead-in, its figures |

## Decisions

| Decision | Chosen | Who chose |
|---|---|---|
| Figure 2 width | seven columns, 3.20 in | you, `seven` |
| Figures 6 and 7 | kept separate (a sweep and a limitations figure discuss different things) | you |
| Limitations figure | original layout, stacked, real type | you |
| Rank and steps ablation | appendix C.5, cited from Section 4.1 | you |
| Appendix lead-ins | six paragraphs, setup only, no results | you, `yes` |

## Still open

- Figure 1 (3.1 pt) and the training diagram (3.6 pt) have text baked into their pictures; the re-lettering recipe in `runbook/making-the-papers-figures/` covers them.
- The limitations figure's pictures are crops from the step-50,000 checkpoint; they are rendered again from the adapter of record (v58-03-self, step 30,000) in route step 13.
- The appendix has 1.4 to 3.8 in empty at the foot of most pages, beside placeholders whose heights are guesses. Each placeholder height is a ceiling, so a filled figure takes its own height; one tidy pass after the results land closes the rest. A figure drawn taller than its box is shrunk with its lettering, so each result is drawn within its box.
- Body-text edits for step 13: "Figure 12 shows" could name Appendix C.5; the paper text still names step 50,000 as the adapter's checkpoint.
