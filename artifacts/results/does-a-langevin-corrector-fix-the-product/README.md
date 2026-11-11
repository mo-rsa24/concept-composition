# Does a Langevin corrector fix the product?

Appendix figure R6 of the ICLR 56173 revision and what leads to it, for reviewer PrzE's request for an MCMC corrector baseline. The plan is [PRZE-2](../../../plans/prze-2-langevin-corrector.md); the slot is R6 in [the layout sheet](../../../report/paper-review/layout-sheet.md).

## five-columns-layout-draft.png

<a href="five-columns-layout-draft.png"><img src="five-columns-layout-draft.png" width="200" alt="R6 layout draft: four pair rows by five sampler columns, grey placeholder tiles"></a>

**What it is.** The layout R6 will fill, with no image rendered yet: rows are the four Figure 5 pairs at seed 10, columns are joint, joint + Langevin, PoE, PoE + Langevin and CO3, all from the same starting latent at seed 10. `stated`

**How it was made.** Drawn with matplotlib from the slot's row and column lists on 2026-11-11, before any corrector render, so the layout can be agreed first. Replaced by `five_columns.png` once piece 5 renders.

