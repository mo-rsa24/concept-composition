# Evidence: what is here and what the paper does with it

One row per item. A folder whose name starts with a figure id (F2, F6, F7, F8) feeds that row of
the figure register, `paper/iclr/figures.md`. A folder without a prefix is supporting or
superseded and is visibly not load-bearing. **Caption owed** means the result is good and its
caption or writeup has to be rewritten fresh and plain before the paper may use it.

## Feeds the paper

| Item | Figure it feeds | State |
|---|---|---|
| `paper/iclr/figures/compose-rate-as-correction-rises.png` | F2, the headline figure: compose rate as more correction is added | result stands (the true correction takes compose rate from 7% to 93%, the controls stay flat); the exact percentages wait on the re-score; **caption owed** |
| `1-lora-correction/artifacts/results/what-the-correction-is/what-it-looks-like/which-way-the-correction-points/does-the-subspace-test-predict-transfer/` (`QUERY.md`, `geometry_vs_transfer.png`, `result.json`) | F6, why it is learnable | the held-out projection licenses nothing about transfer. `QUERY.md` is the argument the caption must not exceed |
| `1-lora-correction/artifacts/results/what-the-correction-is/what-it-looks-like/which-way-the-correction-points/what-the-spectrum-measures/` (`QUERY.md`, `result.json`) | F6, why it is learnable | **what the spectrum would give by luck does not control for ‖r_t‖ spread, and against one that does the pooled stack is 1.4x at k=8 rather than 10.7x.** What direction structure survives is within single runs (4.8x at k=8), not across pairs (1.2x). F6's shared-structure argument does not stand and that figure needs a decision |
| `1-lora-correction/artifacts/results/what-the-correction-is/what-it-looks-like/residual-dynamics/content-change-relative-to-attention-change/` (`measure-fairness.md`, `RUN_ON_THIS_NODE.md`) | F7, the mechanism panel | replicated, median 1.52x over 64 runs. `measure-fairness.md` is why the obvious measure gives the opposite answer, which the appendix needs; **caption owed** |
| `1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-transfer/does-the-fix-reach-unseen-pairs/hard-vs-easy-transfer/` (`demo.py`, `hard_vs_easy_transfer.png`) | F8, transfer | supporting: the hard-vs-easy split behind the transfer read; **caption owed, and the figure regenerates from `demo.py`** |
| `1-lora-correction/artifacts/results/what-the-correction-is/does-it-cause-composition/how-much-correction-is-needed/plausibility_climb.png` | corroborates F3/F5 | keep; the corrected sign reading lives in the cache-analyses review |
| `1-lora-correction/artifacts/results/what-the-correction-is/when-it-acts/when-the-correction-must-arrive/commitment-step/` (`QUERY.md`, `result.json`, `commitment-step-per-pair.png`) | no figure yet, EXP-01 of `1-lora-correction/report/experiments-log.md` | the step where a run's estimate settles varies by pair (medians 18 to 36 over 162 runs), but every pair settles after the correction has stopped working. EXP-04 then showed it predicts nothing about the window, so this measure is not tracking the decision |
| `1-lora-correction/artifacts/results/what-the-correction-is/when-it-acts/when-the-correction-must-arrive/window-vs-commitment/` (`QUERY.md`, `result.json`, `window-vs-commitment.png`) | no figure yet, EXP-04 of `1-lora-correction/report/experiments-log.md` | **all 8 pairs peak at the same window, steps 0 to 10, while their settling steps span 13.** The window does not move with the pair, so the adapter's schedule cannot be misplaced per pair and composition is decided inside the first fifth of the run. Left-censored: the best window is the earliest the grid holds |

> Held-out means kept out of training and used only for testing, so a number measured on a held-out
> pair says something about pairs the method never saw.

## Supporting, not drawn from

| Item | Why it stays |
|---|---|
| `artifacts/_quarantine/results-archive/group-a-failure.md`, `internal-force-failure.md` | the negative controls behind the paper's "external correctors and internal forces fail" sentence (DoD 6) |
| `artifacts/_quarantine/results-archive/{lora-success,residual-diagnostics,conditioning-window}.md` | May-era writeups, superseded by the review files; history only |
| `1-lora-correction/report/instrument-smoke-test.md`, `1-lora-correction/report/normalization-preregistration.md` | how the measuring tools were built and checked, and what was committed before any result was read |
| `1-lora-correction/report/decision-timeline.md`, `1-lora-correction/report/results-summary.md` | the running record; RESULTS_SUMMARY defers to the review files where they disagree |
| the regeneration code for F6/F8, now merged beside each figure's evidence folder under `1-lora-correction/artifacts/results/what-the-correction-is/what-it-looks-like/which-way-the-correction-points/` | the throwaway code that produced `F6`'s and `F8`'s images; regeneration path, never cited |
| `artifacts/plans/parked/artifact-reconciliation/inventory/` | the artifact-reconciliation scope's four output documents |
| `1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/` | the limitations figure (figures 6 and 7: the adapter's settings on cat and dog, zoomed hybrid faces); the paper's limitations appendix is commented out, so it feeds nothing until that appendix returns |
| `1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/how-far-one-adapter-reaches/does-each-adapter-hold-across-ten-seeds/` | every adapter and method across ten seeds, one grid per pair for seven pairs; the answer to a reviewer asking whether a result holds beyond one seed |

## Not paper material

`dl-scene/`, `learning-captures/`, `recap/` (already archived), `artifacts/notes/batch-shape-nondeterminism/`
(formerly `show-me/`), `todoist-staging/`: learning and tooling artifacts. They serve sessions, not
the manuscript.
